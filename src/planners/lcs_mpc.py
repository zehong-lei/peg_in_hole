"""Contact-implicit MPC for peg-hole insertion using the reduced LCS model.

Two solvers
-----------
  "lqr"   (default)
      Finite-horizon LQR via backward Riccati recursion.  λ=0 in planning;
      analytical, < 1 ms.  Cost: lateral + insertion progress + velocity.

  "slsqp"
      SLSQP NLP with box constraints (u_min≤u≤u_max, λ≥0) and soft cost:
        - lateral centering + insertion progress + velocity  (quadratic in x)
        - force effort                                        (R_u ||u||²)
        - force smoothness                                    (R_smooth ||Δu||²)
        - soft complementarity                               (ρ  λᵢ φᵢ)
        - penetration penalty                                (w_pen Σ max(0,−φᵢ)²)
      Analytical gradient via adjoint method → ~1–5 ms per solve.
      Recommended: mpc_freq_ratio ≥ 25 (~20 Hz).

Public interface
----------------
  mpc.solve(x0, u_prev=None) → (u0, X_plan, success)
  mpc.last_info              → SolveInfo (always set after solve)
"""

import time
import numpy as np
from dataclasses import dataclass, field
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
    obj_val: float
    n_iter: int              # SLSQP QP iterations (0 for LQR)
    F_des: np.ndarray        # (3,) first MPC control  [Fx, Fy, Fz_insert]
    lambda_0: np.ndarray     # (4,) contact forces at step 0  (0 for LQR)
    phi_0: np.ndarray        # (4,) gap function at current state
    comp_penalty: float      # ρ Σ_i λ_i max(0, φ_i)  — complementarity residual


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
        self.R_smooth = cfg.get("R_smooth",    0.05)   # force smoothness weight
        self.rho      = cfg.get("rho_comp",  100.0)    # soft complementarity
        self.w_pen    = cfg.get("w_pen",    1000.0)    # state penetration penalty
        self.Q_N      = cfg.get("Q_N_scale",   5.0)
        self.u_max    = np.array(cfg.get("u_max",  [ 5.0,  5.0, 10.0]))
        self.u_min    = np.array(cfg.get("u_min",  [-5.0, -5.0,  0.5]))
        self.lam_max  = cfg.get("lam_max",    50.0)
        self.solver   = cfg.get("solver",    "lqr")

        # Reference state and cost matrices
        self.x_des       = np.array([0., 0., self.z_goal, 0., 0., 0.])
        self.Q           = np.diag([self.Q_lat, self.Q_lat, self.Q_z,
                                    self.Q_vel, self.Q_vel, self.Q_vel])
        self.Q_terminal  = self.Q_N * self.Q
        self.R           = self.R_u * np.eye(3)

        # Pre-compute LQR gains once
        self._Ks: list = []
        self._precompute_lqr()

        # SLSQP state
        self._z_warm: Optional[np.ndarray] = None
        self._last_lambda: np.ndarray = np.zeros(N_CONTACTS)

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

    # ── adjoint-based objective + gradient  O(N·n²) ──────────────────────────

    def _obj_and_grad(self, z: np.ndarray, x0: np.ndarray,
                      u_prev: np.ndarray) -> Tuple[float, np.ndarray]:
        N, nu, nc = self.N, 3, N_CONTACTS
        A, B, D   = self.model.A, self.model.B, self.model.D
        E, c      = self.model.E, self.model.c

        U = z[:N * nu].reshape(N, nu)
        L = z[N * nu:].reshape(N, nc)

        # ── forward rollout ──────────────────────────────────────────────────
        X   = np.empty((N + 1, 6))
        PHI = np.empty((N + 1, nc))
        X[0]   = x0
        PHI[0] = E @ x0 + c
        for k in range(N):
            X[k + 1]   = A @ X[k] + B @ U[k] + D @ L[k]
            PHI[k + 1] = E @ X[k + 1] + c

        # ── stage cost partial derivatives ───────────────────────────────────
        dl_dx = np.zeros((N + 1, 6))
        dl_du = np.zeros((N, nu))
        dl_dl = np.zeros((N, nc))
        cost  = 0.0

        for k in range(N):
            xk   = X[k];    uk = U[k];    lk = L[k];    phi_k = PHI[k]
            e_k  = xk - self.x_des
            Qe_k = self.Q @ e_k

            # Quadratic state cost
            cost     += float(e_k @ Qe_k)
            dl_dx[k] += 2.0 * Qe_k

            # Force effort
            cost     += self.R_u * float(uk @ uk)
            dl_du[k] += 2.0 * self.R_u * uk

            # Force smoothness  ‖u_k − u_{k-1}‖²
            u_km1     = u_prev if k == 0 else U[k - 1]
            du        = uk - u_km1
            cost     += self.R_smooth * float(du @ du)
            dl_du[k] += 2.0 * self.R_smooth * du
            if k > 0:
                dl_du[k - 1] -= 2.0 * self.R_smooth * du   # cross-term

            # Soft complementarity  ρ λᵢ φᵢ
            cost     += self.rho * float(lk @ phi_k)
            dl_dl[k] += self.rho * phi_k                    # ∂(ρ λᵀ φ)/∂λ
            dl_dx[k] += self.rho * (E.T @ lk)              # ∂(ρ λᵀ (Ex+c))/∂x

            # Penetration penalty  w Σ max(0,−φ)²
            pen       = np.maximum(0.0, -phi_k)
            cost     += self.w_pen * float(pen @ pen)
            dl_dx[k] -= 2.0 * self.w_pen * (E.T @ pen)     # ∂(w pen²)/∂x = −2w Eᵀ pen

        # Terminal cost
        eN = X[N] - self.x_des
        cost     += float(eN @ self.Q_terminal @ eN)
        dl_dx[N] += 2.0 * (self.Q_terminal @ eN)

        # ── backward pass (adjoint) ──────────────────────────────────────────
        p     = dl_dx[N].copy()
        grad_u = np.empty_like(dl_du)
        grad_l = np.empty_like(dl_dl)
        for k in range(N - 1, -1, -1):
            grad_u[k] = dl_du[k] + B.T @ p
            grad_l[k] = dl_dl[k] + D.T @ p
            p          = dl_dx[k] + A.T @ p

        return cost, np.concatenate([grad_u.ravel(), grad_l.ravel()])

    # ── SLSQP NLP with analytical gradient  ~1–5 ms ──────────────────────────

    def _solve_slsqp(self, x0: np.ndarray,
                     u_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                    bool, float, int]:
        from scipy.optimize import minimize, Bounds
        N, nu, nc = self.N, 3, N_CONTACTS

        def _combined(z):
            return self._obj_and_grad(z, x0, u_prev)

        n = N * (nu + nc)

        # Warm-start: shift last solution or use LQR solution
        if self._z_warm is not None and len(self._z_warm) == n:
            z0 = self._z_warm.copy()
        else:
            # LQR warm-start (λ=0)
            _, X_lqr, _ = self._solve_lqr(x0)
            z0 = np.zeros(n)
            for k in range(N):
                e = X_lqr[k] - self.x_des
                u_k = np.clip(-self._Ks[k] @ e, self.u_min, self.u_max)
                z0[k * nu:(k + 1) * nu] = u_k

        lb = np.concatenate([np.tile(self.u_min, N), np.zeros(N * nc)])
        ub = np.concatenate([np.tile(self.u_max, N), np.full(N * nc, self.lam_max)])

        res = minimize(
            _combined, z0,
            method='SLSQP',
            jac=True,
            bounds=Bounds(lb, ub, keep_feasible=True),
            options={'maxiter': 100, 'ftol': 1e-5, 'disp': False},
        )

        self._z_warm = res.x.copy()

        # Unpack solution
        U = res.x[:N * nu].reshape(N, nu)
        L = res.x[N * nu:].reshape(N, nc)
        self._last_lambda = L[0].copy()

        # Roll out trajectory
        A, B, D = self.model.A, self.model.B, self.model.D
        X = np.empty((N + 1, 6))
        X[0] = x0
        for k in range(N):
            X[k + 1] = A @ X[k] + B @ U[k] + D @ L[k]

        success = res.success or res.status in (0, 1, 2, 9)
        return U[0].copy(), X, success, float(res.fun), int(res.nit)

    # ── public interface ──────────────────────────────────────────────────────

    def solve(self, x0: np.ndarray,
              u_prev: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Solve MPC from current LCS state.

        Parameters
        ----------
        x0     : (6,) current state [ex, ey, z_descent, vx, vy, vz]
        u_prev : (3,) previous control for smoothness cost (LQR ignores this)

        Returns
        -------
        u0     : (3,) first control  [Fx, Fy, Fz_insert]
        X_plan : (N+1, 6) planned trajectory
        success: bool
        Side effect: self.last_info updated
        """
        if u_prev is None:
            u_prev = np.array([0., 0., 1.])

        t0 = time.perf_counter()

        if self.solver == "lqr":
            u0, X, success = self._solve_lqr(x0)
            obj_val, n_iter = 0.0, 0
            lam0 = np.zeros(N_CONTACTS)
        else:
            u0, X, success, obj_val, n_iter = self._solve_slsqp(x0, u_prev)
            lam0 = self._last_lambda

        dt_ms = (time.perf_counter() - t0) * 1e3

        phi0 = self.model.gap(x0)
        comp = float(np.sum(lam0 * np.maximum(0.0, phi0)))

        self.last_info = SolveInfo(
            solver=self.solver,
            success=success,
            solve_time_ms=dt_ms,
            obj_val=obj_val,
            n_iter=n_iter,
            F_des=u0.copy(),
            lambda_0=lam0.copy(),
            phi_0=phi0.copy(),
            comp_penalty=comp,
        )
        return u0, X, success
