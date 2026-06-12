"""Cartesian impedance controller for contact-rich insertion.

Uses torque commands (d.qfrc_applied) on top of position-hold servos.
During insertion:
  - Low x/y stiffness → peg can slide laterally to find hole
  - Constant downward z force → controlled insertion
  - Roll/pitch compliance → peg can tilt into hole chamfer

Reference: Hogan, N. (1985). Impedance Control: An Approach to Manipulation.
"""

import numpy as np
from .position_controller import _rotation_error


class ImpedanceController:
    """Cartesian impedance for free-space and insertion.

    Parameters
    ----------
    cfg : dict   controller.yaml → impedance_controller section
    dt  : float  simulation timestep
    nv  : int    arm DOFs
    """

    def __init__(self, cfg: dict, dt: float, nv: int = 7):
        self.dt = dt
        self.nv = nv
        self._cfg = cfg

        # Free-space gains
        self.Kp_free = np.diag(cfg["kp_pos_free"] + cfg["kp_rot_free"])    # (6,6)
        self.Kd_free = np.diag(cfg["kd_pos_free"] + cfg["kd_rot_free"])

        # Insertion gains
        self.Kp_ins = np.diag(cfg["kp_pos_insert"] + cfg["kp_rot_insert"])
        self.Kd_ins = np.diag(cfg["kd_pos_insert"] + cfg["kd_rot_insert"])

        self.insert_fz = cfg["insert_force_z"]   # constant downward force (N)
        self.max_tau = cfg["max_torque"]

    def compute_free(
        self,
        q: np.ndarray,         # (7,) current joint positions
        qdot: np.ndarray,      # (7,)
        ee_pos: np.ndarray,    # (3,) current EE position
        ee_rot: np.ndarray,    # (3,3)
        pos_des: np.ndarray,   # (3,)
        rot_des: np.ndarray,   # (3,3)
        J: np.ndarray,         # (6, nv)
        qfrc_bias: np.ndarray = None,  # (7,) gravity+Coriolis for comp
    ) -> np.ndarray:
        """Return joint torques for free-space position tracking (7,)."""
        tau = self._impedance(q, qdot, ee_pos, ee_rot, pos_des, rot_des,
                              J, self.Kp_free, self.Kd_free, f_extra=None)
        if qfrc_bias is not None:
            tau = tau + qfrc_bias[:self.nv]
        return tau

    def compute_insert(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        pos_des: np.ndarray,   # XY target; Z is ignored (use constant force)
        rot_des: np.ndarray,
        J: np.ndarray,
        qfrc_bias: np.ndarray = None,
    ) -> np.ndarray:
        """Return joint torques for insertion (7,).

        Applies constant downward force + low XY/tilt stiffness so the peg
        can self-align into the hole.
        """
        f_extra = np.zeros(6)
        f_extra[2] = -self.insert_fz   # -z = downward in world frame
        tau = self._impedance(q, qdot, ee_pos, ee_rot, pos_des, rot_des,
                              J, self.Kp_ins, self.Kd_ins, f_extra=f_extra)
        if qfrc_bias is not None:
            tau = tau + qfrc_bias[:self.nv]
        return tau

    def compute_insert_mpc(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        pos_des: np.ndarray,
        rot_des: np.ndarray,
        J: np.ndarray,
        qfrc_bias: np.ndarray,
        F_des: np.ndarray,     # (6,) MPC-planned EE wrench [fx,fy,fz,tx,ty,tz]
    ) -> np.ndarray:
        """Insertion torques with LCS-MPC force feedforward.

        Uses insertion impedance gains for position tracking and adds F_des
        as the feedforward wrench instead of a hardcoded constant force.
        """
        tau = self._impedance(q, qdot, ee_pos, ee_rot, pos_des, rot_des,
                              J, self.Kp_ins, self.Kd_ins, f_extra=F_des)
        tau = tau + qfrc_bias[:self.nv]
        return tau

    def _impedance(
        self,
        q, qdot, ee_pos, ee_rot, pos_des, rot_des,
        J, Kp, Kd, f_extra,
    ) -> np.ndarray:
        J7 = J[:, :self.nv]
        v_ee = J7 @ qdot          # (6,) EE velocity

        e_pos = pos_des - ee_pos
        e_rot = _rotation_error(rot_des, ee_rot)
        err = np.concatenate([e_pos, e_rot])  # (6,)

        F_imp = Kp @ err - Kd @ v_ee
        if f_extra is not None:
            F_imp += f_extra

        tau = J7.T @ F_imp
        return np.clip(tau, -self.max_tau, self.max_tau)
