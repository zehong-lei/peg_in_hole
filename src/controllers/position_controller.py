"""Cartesian resolved-motion-rate position controller.

Uses damped pseudo-inverse Jacobian to compute desired joint velocities,
integrates with a lookahead window to get a joint position target,
then applies high-gain joint PD torques via qfrc_applied.

The calling code should:
    env.d.qfrc_applied[:7] = controller.compute(...)
    env.d.ctrl[:7] = q_curr   # servo at neutral → provides damping only
"""

import numpy as np


# Panda joint torque limits [Nm] (hardware spec)
_TAU_MAX = np.array([87., 87., 87., 87., 12., 12., 12.])

# Custom high-gain joint PD (above servo gains; not limited by servo forcerange)
_KP_JOINT = np.array([600., 600., 400., 400., 200., 80., 80.])
_KD_JOINT = np.array([60.,  60.,  40.,  40.,  20.,  8.,  8.])


class PositionController:
    """Cartesian position / orientation tracking via Jacobian IK + torque control.

    Parameters
    ----------
    cfg : dict   controller.yaml → position_controller section
    dt  : float  simulation timestep
    nv  : int    number of arm DOFs (7 for Panda)
    """

    def __init__(self, cfg: dict, dt: float, nv: int = 7):
        self.kp = cfg["kp_pos"]
        self.kr = cfg["kp_rot"]
        self.lam = cfg["damping"]
        self.max_cart_vel = cfg["max_cart_vel"]
        self.max_jvel = cfg["max_joint_vel"]
        self.lookahead = cfg.get("lookahead", 0.30)
        self.dt = dt
        self.nv = nv
        self.kp_joint = _KP_JOINT.copy()
        self.kd_joint = _KD_JOINT.copy()

    def compute(
        self,
        q_curr: np.ndarray,       # (7,) current joint positions
        qdot_curr: np.ndarray,    # (7,) current joint velocities
        ee_pos: np.ndarray,       # (3,)
        ee_rot: np.ndarray,       # (3,3)
        pos_des: np.ndarray,      # (3,)
        rot_des: np.ndarray,      # (3,3)
        J: np.ndarray,            # (6, nv) full Jacobian at EE
        qfrc_bias: np.ndarray,    # (7,) gravity + Coriolis forces
    ) -> tuple:
        """Return (tau, q_des): joint torques (7,) via qfrc_applied and
        joint position target (7,) to be written into ctrl[:7].

        Setting ctrl[:7] = q_des lets the high-gain servo drive toward the
        IK target (saturating at ±87 / ±12 Nm), while qfrc_applied adds
        gravity compensation + our softer PD.  The combined torque gives
        terminal joint velocity ~0.2 rad/s instead of ~0.03 rad/s.
        """
        e_pos = pos_des - ee_pos
        e_rot = _rotation_error(rot_des, ee_rot)
        err = np.concatenate([e_pos, e_rot])

        # Cartesian velocity command
        gains = np.array([self.kp] * 3 + [self.kr] * 3)
        v_cart = gains * err
        v_cart = _clamp_vec(v_cart, self.max_cart_vel)

        # Damped pseudo-inverse
        J7 = J[:, :self.nv]
        A = J7 @ J7.T + self.lam * np.eye(6)
        J_pinv = J7.T @ np.linalg.solve(A, np.eye(6))

        dq_vel = J_pinv @ v_cart
        dq_vel = np.clip(dq_vel, -self.max_jvel, self.max_jvel)

        # Joint target: lookahead integration
        q_des = q_curr + dq_vel * self.lookahead

        # Gravity-compensated PD (softer than the servo; servo does the heavy lifting)
        tau = (qfrc_bias[:self.nv]
               + self.kp_joint * (q_des - q_curr)
               + self.kd_joint * (-qdot_curr))

        return np.clip(tau, -_TAU_MAX * 3, _TAU_MAX * 3), q_des


def _rotation_error(R_des: np.ndarray, R_curr: np.ndarray) -> np.ndarray:
    """Rotation error as angular velocity vector (3,) in world frame."""
    R_err = R_des @ R_curr.T
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    if abs(angle) < 1e-7:
        return np.zeros(3)
    skew = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    return skew * angle / (2.0 * np.sin(angle))


def _clamp_vec(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v[:3])
    if n > max_norm:
        v = v.copy()
        v[:3] *= max_norm / n
    return v
