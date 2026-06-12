"""Utility functions and tracker for pre-insertion OCP trajectories."""

import numpy as np
from typing import Optional


# ── scalar metrics ────────────────────────────────────────────────────────────

def compute_path_length(p_traj: np.ndarray) -> float:
    """Sum of Euclidean segment lengths along (N+1, 3) trajectory [m]."""
    if len(p_traj) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p_traj, axis=0), axis=1)))


def compute_smoothness(v_traj: np.ndarray, dt: float) -> float:
    """RMS acceleration magnitude derived from velocity trajectory (N, 3) [m/s²]."""
    if len(v_traj) < 2 or dt <= 0.0:
        return 0.0
    acc = np.diff(v_traj, axis=0) / dt        # (N-1, 3) accelerations
    return float(np.sqrt(np.mean(np.sum(acc ** 2, axis=1))))


def compute_min_board_clearance(p_traj: np.ndarray, board_top_z: float) -> float:
    """Minimum EE height above the board surface along (N+1, 3) trajectory [m]."""
    return float(np.min(p_traj[:, 2]) - board_top_z)


# ── trajectory tracker ────────────────────────────────────────────────────────

class TrajectoryTracker:
    """Time-based tracker for a precomputed EE position trajectory.

    At each control cycle the caller passes the current simulation time and
    receives the next waypoint on the OCP trajectory.  The index advances
    by one every dt_ocp seconds (pure time-based scheduling).

    Parameters
    ----------
    p_traj  : (N+1, 3) position trajectory from PreInsertionOCP
    dt_ocp  : OCP timestep [s] — spacing between successive waypoints
    v_traj  : (N, 3) optional velocity trajectory; if omitted, get_velocity()
              uses finite differences from p_traj.
    """

    def __init__(self, p_traj: np.ndarray, dt_ocp: float,
                 v_traj: Optional[np.ndarray] = None):
        self.p_traj  = p_traj.copy()   # (N+1, 3)
        self.v_traj  = v_traj.copy() if v_traj is not None else None  # (N, 3)
        self.dt_ocp  = dt_ocp
        self._N      = len(p_traj) - 1
        self._t_start: Optional[float] = None

    def reset(self, t_now: float) -> None:
        """Call once when stage begins tracking."""
        self._t_start = t_now

    def _frac(self, t_now: float) -> tuple[int, float]:
        """Return (k, alpha) where k is interval index and alpha ∈ [0,1)."""
        elapsed = max(0.0, t_now - self._t_start)
        f = elapsed / self.dt_ocp
        k = int(f)
        return k, f - k

    def get_target(self, t_now: float) -> np.ndarray:
        """Return interpolated target position (one step ahead, linearly smoothed)."""
        if self._t_start is None:
            return self.p_traj[-1].copy()
        k, alpha = self._frac(t_now)
        # Maintain one-step-ahead lead; lerp between p[k+1] and p[k+2]
        k1 = min(k + 1, self._N)
        k2 = min(k + 2, self._N)
        return (1.0 - alpha) * self.p_traj[k1] + alpha * self.p_traj[k2]

    def get_velocity(self, t_now: float) -> np.ndarray:
        """Return interpolated desired EE velocity (3,) [m/s]."""
        if self._t_start is None or self.is_done(t_now):
            return np.zeros(3)
        k, alpha = self._frac(t_now)
        if self.v_traj is not None:
            k0 = min(k,     len(self.v_traj) - 1)
            k1 = min(k + 1, len(self.v_traj) - 1)
            return (1.0 - alpha) * self.v_traj[k0] + alpha * self.v_traj[k1]
        k0 = min(k, self._N - 1)
        k1 = min(k + 1, self._N - 1)
        v0 = (self.p_traj[k0 + 1] - self.p_traj[k0]) / self.dt_ocp
        v1 = (self.p_traj[k1 + 1] - self.p_traj[k1]) / self.dt_ocp
        return (1.0 - alpha) * v0 + alpha * v1

    def get_acceleration(self, t_now: float) -> np.ndarray:
        """Return desired EE acceleration (3,) [m/s²].

        Computed as the finite difference of interpolated velocities —
        piecewise constant per OCP interval, continuous across the horizon.
        Returns zero at or after the trajectory horizon.
        """
        if self._t_start is None or self.is_done(t_now):
            return np.zeros(3)
        k, _ = self._frac(t_now)
        if self.v_traj is not None and len(self.v_traj) >= 2:
            k0 = min(k,     len(self.v_traj) - 2)
            return (self.v_traj[k0 + 1] - self.v_traj[k0]) / self.dt_ocp
        k0 = min(k, self._N - 2)
        return ((self.p_traj[k0 + 2] - 2.0 * self.p_traj[k0 + 1] + self.p_traj[k0])
                / self.dt_ocp ** 2)

    def is_done(self, t_now: float) -> bool:
        """True when the full trajectory horizon has elapsed."""
        if self._t_start is None:
            return False
        return (t_now - self._t_start) >= self._N * self.dt_ocp

    def is_terminal(self, ee_pos: np.ndarray, tol: float) -> bool:
        """True when EE is within tol of the final trajectory waypoint."""
        return bool(np.linalg.norm(ee_pos - self.p_traj[-1]) < tol)

    @property
    def final_pos(self) -> np.ndarray:
        return self.p_traj[-1].copy()
