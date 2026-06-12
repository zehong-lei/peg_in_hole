"""Global pre-insertion end-effector OCP.

Solves a finite-horizon trajectory optimisation problem (SLSQP) once when
entering MOVE_TO_PREINSERT.  The resulting p_traj is then tracked open-loop
by a TrajectoryTracker.

State    p_k ∈ R³  EE Cartesian position
Control  v_k ∈ R³  EE Cartesian velocity
Dynamics p_{k+1} = p_k + dt·v_k

Decision variable: z = [v_0; …; v_{N-1}] ∈ R^{3N}

Cost (all analytic gradients):
  J_terminal  = Q_T   · ‖p_N − p_goal‖²
  J_vel       = Q_vel · Σ ‖v_k‖²
  J_smooth    = Q_sm  · Σ_{k≥1} ‖v_k − v_{k-1}‖²   (acceleration proxy)
  J_clearance = w_cl  · Σ_k w_lat(k) · max(0, z_clear − p_k[2])²
  J_workspace = w_ws  · Σ_{k,d} [max(0, lo_d − p_k[d])² + max(0, p_k[d] − hi_d)²]

Clearance weight w_lat(k) = min(1, ‖p_k[:2] − hole[:2]‖ / lat_threshold)
is precomputed from the straight-line initial guess so the gradient remains
analytic.

Hard constraints: velocity box bounds only (workspace+clearance are soft).
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class OCPResult:
    p_traj: np.ndarray        # (N+1, 3) EE position trajectory
    v_traj: np.ndarray        # (N,   3) EE velocity trajectory
    solve_time_ms: float      # wall-clock solve time [ms]
    success: bool             # SLSQP converged AND terminal error acceptable
    cost: float               # final objective value
    terminal_error: float     # ‖p_N − p_goal‖ [m]


class PreInsertionOCP:
    """One-shot SLSQP OCP for EE pre-insertion trajectory."""

    def __init__(self, task_cfg: dict):
        cfg = task_cfg.get("preinsert_ocp", {}).get("ocp", {})
        self.N  = int(cfg.get("horizon", 30))
        self.dt = float(cfg.get("dt", 0.10))
        self.v_max = np.array(cfg.get("v_max", [0.15, 0.15, 0.10]), dtype=float)

        self.Q_terminal  = float(cfg.get("Q_terminal",  2000.0))
        self.Q_vel       = float(cfg.get("Q_vel",          0.5))
        self.Q_smooth    = float(cfg.get("Q_smooth",        5.0))
        self.w_clearance = float(cfg.get("w_clearance",   500.0))
        self.w_workspace = float(cfg.get("w_workspace",   100.0))
        self.eps_terminal = float(cfg.get("eps_terminal",  0.005))
        self.lateral_threshold = float(
            cfg.get("clearance_lateral_threshold", 0.05))

        board_c  = np.array(task_cfg["board"]["center"], dtype=float)
        board_hs = np.array(task_cfg["board"]["half_size"], dtype=float)
        self.board_top_z = float(board_c[2] + board_hs[2])
        peg_hl = float(task_cfg["peg"]["half_length"])
        # EE must be above board_top + 2·peg_hl to keep peg tip above board
        self.board_clearance_z = self.board_top_z + 2.0 * peg_hl

        ws = cfg.get("workspace", {})
        self._ws_bounds = [
            (float(ws.get("x_min", -0.8)), float(ws.get("x_max",  0.8))),
            (float(ws.get("y_min", -0.5)), float(ws.get("y_max",  0.8))),
            (float(ws.get("z_min",  0.35)), float(ws.get("z_max", 0.80))),
        ]

        self._z_warm: Optional[np.ndarray] = None   # warm-start cache

    # ── public API ────────────────────────────────────────────────────────────

    def solve(self, p_start: np.ndarray, p_goal: np.ndarray,
              hole_pos: np.ndarray) -> OCPResult:
        """Solve OCP from p_start to p_goal, with hole position for clearance
        weighting.

        Returns OCPResult; on failure success=False and p_traj is the best
        found solution (can still be used as fallback).
        """
        t0 = time.perf_counter()
        N, dt = self.N, self.dt
        p0 = p_start.copy()
        pg = p_goal.copy()

        # ── precompute lateral-distance clearance weights from straight-line ──
        # (fixed during optimisation so gradient stays analytic)
        v_init = (pg - p0) / (N * dt)
        # CS_init[k] = (k+1)*v_init, P_init[k] = p0 + dt*(k+1)*v_init for k=0..N-1
        ks = np.arange(1, N + 1)[:, None]           # (N, 1)
        P_init = p0 + dt * ks * v_init              # (N, 3) — points P[1]..P[N]
        lat_dists = np.linalg.norm(
            P_init[:, :2] - hole_pos[:2], axis=1)   # (N,)
        lat_thr = max(self.lateral_threshold, 1e-6)
        cl_weights = np.minimum(1.0, lat_dists / lat_thr)  # (N,) ∈ [0,1]

        # ── objective and analytic gradient ───────────────────────────────────
        def obj_and_grad(z: np.ndarray):
            V  = z.reshape(N, 3)
            CS = np.cumsum(V, axis=0)     # CS[k] = Σ_{j=0}^{k} V[j]  (N, 3)
            P  = np.empty((N + 1, 3))
            P[0]  = p0
            P[1:] = p0 + dt * CS         # P[k] = p0 + dt·CS[k-1]   k=1..N

            grad_V = np.zeros_like(V)
            cost   = 0.0

            # 1. Terminal cost  Q_T·‖P[N]−p_goal‖²
            err_T  = P[N] - pg
            cost  += self.Q_terminal * float(err_T @ err_T)
            # ∂J_T/∂V[j] = 2·Q_T·dt·err_T  (∀ j, since P[N] = p0+dt·Σ V[j])
            grad_V += (2.0 * self.Q_terminal * dt) * err_T

            # 2. Velocity cost  Q_vel·Σ‖V[k]‖²
            cost   += self.Q_vel * np.sum(V ** 2)
            grad_V += 2.0 * self.Q_vel * V

            # 3. Smoothness cost  Q_sm·Σ_{k≥1} ‖V[k]−V[k-1]‖²
            if N > 1:
                dV = np.diff(V, axis=0)          # (N-1, 3)
                cost += self.Q_smooth * np.sum(dV ** 2)
                sm_g = np.zeros_like(V)
                # j contributes via dV[j-1] (if j>0) and −dV[j] (if j<N-1)
                sm_g[:-1] -= 2.0 * self.Q_smooth * dV
                sm_g[1:]  += 2.0 * self.Q_smooth * dV
                grad_V += sm_g

            # 4. Board clearance penalty  w_cl·Σ_k w_lat(k)·max(0, z_cl−P_k[2])²
            viol_z = np.maximum(0.0, self.board_clearance_z - P[1:, 2])  # (N,)
            wviol  = cl_weights * viol_z                                   # (N,)
            cost  += self.w_clearance * float(wviol @ viol_z)
            # ∂J_cl/∂V[j,2] = −2·w_cl·dt·Σ_{k≥j} w_lat(k)·viol_z(k)
            suf_wviol = np.cumsum(wviol[::-1])[::-1]   # (N,) suffix sums
            grad_V[:, 2] -= 2.0 * self.w_clearance * dt * suf_wviol

            # 5. Workspace penalty  w_ws·Σ_{k,d}[max(0,lo−P_k[d])²+max(0,P_k[d]−hi)²]
            for dim, (lo, hi) in enumerate(self._ws_bounds):
                lo_v = np.maximum(0.0, lo - P[1:, dim])   # (N,)
                hi_v = np.maximum(0.0, P[1:, dim] - hi)   # (N,)
                cost += self.w_workspace * (float(lo_v @ lo_v) + float(hi_v @ hi_v))
                suf_lo = np.cumsum(lo_v[::-1])[::-1]
                suf_hi = np.cumsum(hi_v[::-1])[::-1]
                grad_V[:, dim] += 2.0 * self.w_workspace * dt * (suf_hi - suf_lo)

            return cost, grad_V.ravel()

        # ── warm start: prior solution or straight-line ───────────────────────
        z0 = (self._z_warm.copy()
              if self._z_warm is not None and len(self._z_warm) == 3 * N
              else np.tile(v_init, N))

        lb = np.tile(-self.v_max, N)
        ub = np.tile( self.v_max, N)

        from scipy.optimize import minimize, Bounds
        res = minimize(
            obj_and_grad, z0,
            method='SLSQP',
            jac=True,
            bounds=Bounds(lb, ub, keep_feasible=True),
            options={'maxiter': 300, 'ftol': 1e-7, 'disp': False},
        )
        self._z_warm = res.x.copy()

        # ── reconstruct full trajectory ───────────────────────────────────────
        V  = res.x.reshape(N, 3)
        CS = np.cumsum(V, axis=0)
        P  = np.empty((N + 1, 3))
        P[0]  = p_start
        P[1:] = p_start + dt * CS

        term_err = float(np.linalg.norm(P[N] - p_goal))
        success  = bool(
            (res.success or res.status in (0, 1, 2))
            and term_err < self.eps_terminal * 5.0
        )

        return OCPResult(
            p_traj=P,
            v_traj=V,
            solve_time_ms=(time.perf_counter() - t0) * 1e3,
            success=success,
            cost=float(res.fun),
            terminal_error=term_err,
        )
