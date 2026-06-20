"""Contact-implicit MPC for peg-hole insertion using the reduced LCS model.

Solver: finite-horizon LQR via backward Riccati recursion.  λ=0 in planning;
analytical, < 1 ms.  Cost: lateral centering + insertion progress + velocity.

Public interface
----------------
  mpc.solve(x0, u_prev=None) → (u0, X_plan, success)
  mpc.last_info              → SolveInfo (always set after solve)
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ── Reduced-order LCS model ───────────────────────────────────────────────────
# State  x = [ex, ey, z, vx, vy, vz]
#   ex, ey : EE lateral error w.r.t. hole centre [m]
#   z      : EE descent from INSERT start (positive = deeper) [m]
#   vx, vy : lateral velocity [m/s]
#   vz     : descent velocity (positive = inserting deeper) [m/s]
# Control  u = [Fx, Fy, Fz_insert]  (Fz_insert>0 maps to world -z)
# Contacts (4-wall box approximation of circular hole)
# LCS form: x_{k+1} = A x_k + B u_k + D λ_k
#           0 ≤ λ_k ⊥ φ(x_k) = E x_k + c ≥ 0

N_CONTACTS = 4


class ReducedLCSModel:
    """Discrete-time LCS model matrices for peg-hole insertion."""

    def __init__(self, cfg: dict):
        m    = cfg.get("eff_mass",   0.5)
        b    = cfg.get("damping",    8.0)
        bz   = cfg.get("damping_z", 12.0)
        dt   = cfg.get("dt",         0.01)
        self.clearance = cfg.get("clearance", 0.004)
        self.dt = dt

        ax = 1.0 - b  * dt / m
        az = 1.0 - bz * dt / m

        self.A = np.array([
            [1., 0., 0., dt,  0., 0.],
            [0., 1., 0., 0.,  dt, 0.],
            [0., 0., 1., 0.,  0., dt],
            [0., 0., 0., ax,  0., 0.],
            [0., 0., 0., 0.,  ax, 0.],
            [0., 0., 0., 0.,  0., az],
        ])
        self.B = np.zeros((6, 3))
        self.B[3, 0] = dt / m
        self.B[4, 1] = dt / m
        self.B[5, 2] = dt / m

        self.D = np.zeros((6, N_CONTACTS))
        self.D[3, 0] = -dt / m
        self.D[3, 1] =  dt / m
        self.D[4, 2] = -dt / m
        self.D[4, 3] =  dt / m

        self.E = np.array([
            [-1., 0., 0., 0., 0., 0.],
            [ 1., 0., 0., 0., 0., 0.],
            [0., -1., 0., 0., 0., 0.],
            [0.,  1., 0., 0., 0., 0.],
        ])
        self.c = np.full(N_CONTACTS, self.clearance)

    def gap(self, x: np.ndarray) -> np.ndarray:
        return self.E @ x + self.c

    def step(self, x: np.ndarray, u: np.ndarray, lam: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u + self.D @ lam


@dataclass
class SolveInfo:
    solver: str
    success: bool
    solve_time_ms: float
    F_des: np.ndarray        # (3,) first MPC control  [Fx, Fy, Fz_insert]
    phi_0: np.ndarray        # (4,) gap function at current state


class LCSMPC:
    """Finite-horizon MPC for peg-hole local contact dynamics."""

    def __init__(self, model: ReducedLCSModel, cfg: dict):
        self.model    = model
        self.N        = cfg.get("horizon",      8)
        self.z_goal   = cfg.get("z_goal",    0.075)
        self.Q_lat    = cfg.get("Q_lat",     500.0)
        self.Q_z      = cfg.get("Q_z",       200.0)
        self.Q_vel    = cfg.get("Q_vel",       1.0)
        self.R_u      = cfg.get("R_u",         0.01)
        self.Q_N      = cfg.get("Q_N_scale",   5.0)
        self.u_max    = np.array(cfg.get("u_max",  [ 5.0,  5.0, 10.0]))
        self.u_min    = np.array(cfg.get("u_min",  [-5.0, -5.0,  0.5]))

        # Reference state and cost matrices
        self.x_des       = np.array([0., 0., self.z_goal, 0., 0., 0.])
        self.Q           = np.diag([self.Q_lat, self.Q_lat, self.Q_z,
                                    self.Q_vel, self.Q_vel, self.Q_vel])
        self.Q_terminal  = self.Q_N * self.Q
        self.R           = self.R_u * np.eye(3)

        # Pre-compute LQR gains once
        self._Ks: list = []
        self._precompute_lqr()

        # Diagnostics (populated after every solve)
        self.last_info: Optional[SolveInfo] = None

    # ── LQR pre-computation  O(N·n³) ─────────────────────────────────────────

    def _precompute_lqr(self) -> None:
        A, B = self.model.A, self.model.B
        P = self.Q_terminal.copy()
        Ks = []
        for _ in range(self.N):
            BtP = B.T @ P
            M   = self.R + BtP @ B
            K   = np.linalg.solve(M, BtP @ A)
            Ks.append(K)
            P   = self.Q + A.T @ P @ A - A.T @ P @ B @ K
        self._Ks = list(reversed(Ks))

    # ── analytical LQR solve  < 1 ms ─────────────────────────────────────────

    def _solve_lqr(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        A, B = self.model.A, self.model.B
        X = np.empty((self.N + 1, 6))
        X[0] = x0
        U = np.empty((self.N, 3))
        for k in range(self.N):
            u = np.clip(-self._Ks[k] @ (X[k] - self.x_des), self.u_min, self.u_max)
            U[k] = u
            X[k + 1] = A @ X[k] + B @ u
        return U[0].copy(), X, True

    # ── public interface ──────────────────────────────────────────────────────

    def solve(self, x0: np.ndarray,
              u_prev: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Solve MPC from current LCS state.

        Parameters
        ----------
        x0     : (6,) current state [ex, ey, z_descent, vx, vy, vz]
        u_prev : (3,) previous control (unused by LQR; kept for API stability)

        Returns
        -------
        u0     : (3,) first control  [Fx, Fy, Fz_insert]
        X_plan : (N+1, 6) planned trajectory
        success: bool
        Side effect: self.last_info updated
        """
        t0 = time.perf_counter()
        u0, X, success = self._solve_lqr(x0)
        dt_ms = (time.perf_counter() - t0) * 1e3

        phi0 = self.model.gap(x0)
        self.last_info = SolveInfo(
            solver="lqr",
            success=success,
            solve_time_ms=dt_ms,
            F_des=u0.copy(),
            phi_0=phi0.copy(),
        )
        return u0, X, success
