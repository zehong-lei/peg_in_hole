"""Low-pass filter state estimator.

Smooths noisy observations before passing to controllers.
Can be replaced with an EKF or particle filter if needed.
"""

import numpy as np
from dataclasses import dataclass, field
from src.sensors.sensor_wrapper import Observation


@dataclass
class BeliefState:
    """Filtered state estimate available to controllers."""
    q: np.ndarray
    qdot: np.ndarray
    gripper_width: float
    ee_pos: np.ndarray
    ee_rot: np.ndarray
    peg_pos: np.ndarray
    peg_rot: np.ndarray
    hole_pos: np.ndarray
    hole_rot: np.ndarray
    contact_detected: bool
    external_force: np.ndarray
    stage: int
    time: float
    # Pose provenance — placed after all required fields
    peg_pos_source: str = field(default="noisy_ground_truth")
    hole_pos_source: str = field(default="noisy_ground_truth")
    # Grasp state (set externally by MultiTaskAssemblyTask after weld activation)
    grasp_transform: object = field(default=None)        # np.ndarray (3,) or None
    grasp_stability_status: str = field(default="not_grasped")
    visual_grasp_residual: object = field(default=None)  # reserved: float or None
    peg_in_hand_confidence: object = field(default=None) # reserved: float or None


class StateEstimator:
    """First-order low-pass filter over observations.

    alpha = 1.0  → no filtering (pass-through)
    alpha = 0.0  → frozen (never update)
    """

    def __init__(self, noise_cfg: dict):
        alpha_pose = noise_cfg["state_estimator"]["pose_alpha"]
        alpha_ft = noise_cfg["state_estimator"]["force_alpha"]
        self._ap = alpha_pose
        self._af = alpha_ft
        self._initialized = False
        self._state: BeliefState = None  # type: ignore

    def update(self, obs: Observation) -> BeliefState:
        if not self._initialized:
            self._state = BeliefState(
                q=obs.q.copy(),
                qdot=obs.qdot.copy(),
                gripper_width=obs.gripper_width,
                ee_pos=obs.ee_pos.copy(),
                ee_rot=obs.ee_rot.copy(),
                peg_pos=obs.peg_pos.copy(),
                peg_rot=obs.peg_rot.copy(),
                hole_pos=obs.hole_pos.copy(),
                hole_rot=obs.hole_rot.copy(),
                contact_detected=obs.contact_detected,
                external_force=obs.external_force.copy(),
                stage=obs.stage,
                time=obs.time,
                peg_pos_source=obs.peg_pos_source,
                hole_pos_source=obs.hole_pos_source,
            )
            self._initialized = True
            return self._state

        s = self._state
        # Joint state: trust encoder directly
        s.q = obs.q.copy()
        s.qdot = obs.qdot.copy()
        s.gripper_width = obs.gripper_width
        s.ee_pos = obs.ee_pos.copy()
        s.ee_rot = obs.ee_rot.copy()

        # Peg pose: bypass low-pass for perception/propagation sources;
        # filter only when source is noisy GT (alpha=0.3 smooths jitter).
        s.peg_pos_source = obs.peg_pos_source
        if obs.peg_pos_source == "noisy_ground_truth":
            s.peg_pos = self._ap * obs.peg_pos + (1 - self._ap) * s.peg_pos
            s.peg_rot = _slerp_mat(s.peg_rot, obs.peg_rot, self._ap)
        else:
            s.peg_pos = obs.peg_pos.copy()
            s.peg_rot = obs.peg_rot.copy()

        # Hole pose: same logic
        s.hole_pos_source = obs.hole_pos_source
        if obs.hole_pos_source == "noisy_ground_truth":
            s.hole_pos = self._ap * obs.hole_pos + (1 - self._ap) * s.hole_pos
            s.hole_rot = _slerp_mat(s.hole_rot, obs.hole_rot, self._ap)
        else:
            s.hole_pos = obs.hole_pos.copy()
            s.hole_rot = obs.hole_rot.copy()

        # Force: low-pass filter
        s.external_force = self._af * obs.external_force + (1 - self._af) * s.external_force
        f_norm = np.linalg.norm(s.external_force[:3])
        s.contact_detected = f_norm > 1.5   # slightly lower than sensor threshold

        s.stage = obs.stage
        s.time = obs.time
        return s


def _slerp_mat(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two rotation matrices by fraction t."""
    dR = R1 @ R0.T
    # Use axis-angle of dR
    trace = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    angle = np.arccos(trace)
    if abs(angle) < 1e-7:
        return R1.copy()
    K = (dR - dR.T) / (2 * np.sin(angle))
    axis = np.array([K[2, 1], K[0, 2], K[1, 0]])
    dR_t = (np.eye(3) + np.sin(t * angle) * K
            + (1 - np.cos(t * angle)) * (K @ K))
    return dR_t @ R0
