"""Operational-space controller with optional inertia shaping.

Impedance-only (dynamics_aware=False):
    F_task = Kp @ e + Kd @ (v_des − v_ee) + F_des
    tau    = J(q)^T @ F_task + qfrc_bias

Dynamics-aware (dynamics_aware=True, M provided):
    Translational: a_cmd = a_des + Kp_pos @ e_pos + Kd_pos @ (v_des − v_ee_pos)
                   F_motion = Λ @ a_cmd,   Λ = (J3 M^{-1} J3^T + ε I)^{-1}
    Rotational:    F_rot = Kp_rot @ e_rot + Kd_rot @ (−v_ee_rot)   (impedance)
    F_task = [F_motion; F_rot] + F_des
    tau    = J^T @ F_task + qfrc_bias

When v_des/a_des are omitted the controller degenerates to pure impedance.

Three gain modes:
  free_space  — stiff tracking; OCP trajectory execution
  insertion   — compliant x/y + LCS-MPC force feedforward
  recovery    — gentle gains for retract/search manoeuvres
"""

import numpy as np
from typing import Optional, Tuple

from .position_controller import _rotation_error

_TAU_MAX  = np.array([87., 87., 87., 87., 12., 12., 12.])
_KP_IK    = 20.0
_KR_IK    = 5.0
_LAM_IK   = 0.01
_VEL_MAX  = 0.25
_JVEL_MAX = 2.0
_LOOKAHEAD = 0.30


class OperationalSpaceController:
    """Task-space controller with optional inertia shaping.

    Parameters
    ----------
    cfg : dict   controller.yaml → operational_space_controller section
    dt  : float  simulation timestep [s]
    nv  : int    number of arm DOFs (7 for Panda)
    """

    def __init__(self, cfg: dict, dt: float, nv: int = 7):
        self.dt  = dt
        self.nv  = nv

        def _diag6(pos_key, pos_def, rot_key, rot_def) -> np.ndarray:
            kp = cfg.get(pos_key, pos_def)
            kr = cfg.get(rot_key, rot_def)
            return np.diag(np.asarray(kp + kr, dtype=float))

        self._Kp: dict = {
            'free_space': _diag6("kp_pos_free",     [300., 300., 300.],
                                  "kp_rot_free",     [30.,  30.,  30.]),
            'insertion':  _diag6("kp_pos_insert",   [80.,  80.,  50.],
                                  "kp_rot_insert",   [10.,  10.,  20.]),
            'recovery':   _diag6("kp_pos_recovery", [150., 150., 150.],
                                  "kp_rot_recovery", [20.,  20.,  20.]),
        }
        self._Kd: dict = {
            'free_space': _diag6("kd_pos_free",     [30.,  30.,  30.],
                                  "kd_rot_free",     [3.,   3.,   3.]),
            'insertion':  _diag6("kd_pos_insert",   [15.,  15.,  10.],
                                  "kd_rot_insert",   [1.,   1.,   2.]),
            'recovery':   _diag6("kd_pos_recovery", [20.,  20.,  20.],
                                  "kd_rot_recovery", [2.,   2.,   2.]),
        }

        self._insert_fz     = float(cfg.get("insert_force_z", 8.0))
        self._max_tau       = float(cfg.get("max_torque", 87.0))
        self._dynamics_aware = bool(cfg.get("dynamics_aware", False))
        self._lambda_eps    = float(cfg.get("lambda_eps", 1e-3))

        self._kp_ik    = float(cfg.get("kp_ik",    _KP_IK))
        self._kr_ik    = float(cfg.get("kr_ik",    _KR_IK))
        self._lam_ik   = float(cfg.get("lam_ik",   _LAM_IK))
        self._lookahead = float(cfg.get("lookahead", _LOOKAHEAD))

        self._tracking_errs: list = []
        self._tau_norms:     list = []
        self._f_task_norms:  list = []
        self._motion_forces: list = []
        self._f_des_norms:   list = []

    # ── public API ────────────────────────────────────────────────────────────

    def reset_metrics(self) -> None:
        self._tracking_errs = []
        self._tau_norms     = []
        self._f_task_norms  = []
        self._motion_forces = []
        self._f_des_norms   = []

    def get_metrics(self) -> dict:
        te = self._tracking_errs
        tn = self._tau_norms
        ft = self._f_task_norms
        mf = self._motion_forces
        fd = self._f_des_norms

        def _rms(lst):
            return float(np.sqrt(np.mean(np.array(lst) ** 2))) if lst else 0.0

        return {
            # impedance-mode metric names
            "os_mean_tracking_err": float(np.mean(te)) if te else 0.0,
            "os_peak_tracking_err": float(np.max(te))  if te else 0.0,
            "os_mean_ctrl_effort":  float(np.mean(tn)) if tn else 0.0,
            "os_peak_tau":          float(np.max(tn))  if tn else 0.0,
            # dynamics-aware metrics
            "os_max_tracking_error": float(np.max(te))  if te else 0.0,
            "os_peak_force_cmd":     float(np.max(ft))  if ft else 0.0,
            "os_tau_rms":            _rms(tn),
            "os_motion_force_rms":   _rms(mf),
            "os_fdes_rms":           _rms(fd),
        }

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        x_des: np.ndarray,
        rot_des: np.ndarray,
        J: np.ndarray,                       # (6, nv) full EE Jacobian
        qfrc_bias: np.ndarray,               # (7,) gravity + Coriolis
        v_des: Optional[np.ndarray] = None,  # (3,) or (6,) desired EE velocity
        a_des: Optional[np.ndarray] = None,  # (3,) desired EE acceleration
        F_des: Optional[np.ndarray] = None,  # (6,) LCS-MPC force feedforward
        mode: str = 'free_space',
        M: Optional[np.ndarray] = None,      # (7,7) joint mass matrix for inertia shaping
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute torque command and servo joint target.

        Returns
        -------
        tau   : (7,) torques → qfrc_applied
        q_des : (7,) joint target → ctrl[:7]
        """
        J7   = J[:, :self.nv]
        v_ee = J7 @ qdot

        e_pos = x_des - ee_pos
        e_rot = _rotation_error(rot_des, ee_rot)
        err6  = np.concatenate([e_pos, e_rot])

        # Velocity error for impedance fallback / rotation
        if v_des is not None:
            vd = np.asarray(v_des, dtype=float)
            if vd.shape[0] == 3:
                vd = np.concatenate([vd, np.zeros(3)])
            v_err = vd - v_ee
        else:
            v_err = -v_ee

        Kp = self._Kp.get(mode, self._Kp['free_space'])
        Kd = self._Kd.get(mode, self._Kd['free_space'])

        if self._dynamics_aware and M is not None:
            # ── Inertia-shaped translational + impedance rotational ───────────
            J3 = J7[:3, :]   # (3, 7) translational Jacobian
            Lambda = self._compute_lambda(J3, M)

            v_des3 = (np.asarray(v_des, dtype=float)[:3]
                      if v_des is not None else np.zeros(3))
            a_des3 = (np.asarray(a_des, dtype=float)[:3]
                      if a_des is not None else np.zeros(3))

            a_cmd   = a_des3 + Kp[:3, :3] @ e_pos + Kd[:3, :3] @ (v_des3 - v_ee[:3])
            F_motion = Lambda @ a_cmd                          # (3,)
            F_rot    = Kp[3:, 3:] @ e_rot + Kd[3:, 3:] @ v_err[3:]
            F_task   = np.concatenate([F_motion, F_rot])
        else:
            # ── Pure impedance ────────────────────────────────────────────────
            F_task   = Kp @ err6 + Kd @ v_err
            F_motion = None

        if F_des is not None:
            F_task = F_task + np.asarray(F_des, dtype=float)
        elif mode == 'insertion':
            F_task[2] -= self._insert_fz

        tau = J7.T @ F_task + qfrc_bias[:self.nv]
        tau = np.clip(tau, -self._max_tau, self._max_tau)

        q_des = self._ik_target(q, ee_pos, ee_rot, x_des, rot_des, J7)

        # Accumulate metrics
        self._tracking_errs.append(float(np.linalg.norm(e_pos)))
        self._tau_norms.append(float(np.linalg.norm(tau)))
        self._f_task_norms.append(float(np.linalg.norm(F_task)))
        f_motion_norm = (float(np.linalg.norm(F_motion))
                         if F_motion is not None
                         else float(np.linalg.norm(F_task[:3])))
        self._motion_forces.append(f_motion_norm)
        self._f_des_norms.append(
            float(np.linalg.norm(F_des[:3])) if F_des is not None else 0.0)

        return tau, q_des

    # ── private ───────────────────────────────────────────────────────────────

    def _compute_lambda(self, J3: np.ndarray, M7: np.ndarray) -> np.ndarray:
        """Compute 3×3 operational-space inertia matrix.

        Λ = (J3 M^{-1} J3^T + ε I)^{-1}
        Solved via linear system to avoid explicit matrix inversion.
        """
        try:
            X = np.linalg.solve(M7, J3.T)   # (7, 3): M7 @ X = J3^T
        except np.linalg.LinAlgError:
            X = np.linalg.lstsq(M7, J3.T, rcond=None)[0]
        Lambda_inv = J3 @ X + self._lambda_eps * np.eye(3)
        return np.linalg.inv(Lambda_inv)

    def _ik_target(
        self,
        q: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        x_des: np.ndarray,
        rot_des: np.ndarray,
        J7: np.ndarray,
    ) -> np.ndarray:
        e_pos = x_des - ee_pos
        e_rot = _rotation_error(rot_des, ee_rot)
        v_cmd = np.concatenate([self._kp_ik * e_pos,
                                 self._kr_ik * e_rot])
        vn = np.linalg.norm(v_cmd[:3])
        if vn > _VEL_MAX:
            v_cmd[:3] *= _VEL_MAX / vn
        A  = J7 @ J7.T + self._lam_ik * np.eye(6)
        dq = J7.T @ np.linalg.solve(A, v_cmd)
        dq = np.clip(dq, -_JVEL_MAX, _JVEL_MAX)
        return q + dq * self._lookahead
