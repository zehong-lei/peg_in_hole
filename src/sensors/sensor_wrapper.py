"""SensorWrapper: converts ground truth to noisy observations.

The controller is ONLY allowed to call get_observation().
It must never call env.get_true_state() or access env._d directly.

Peg pose source lifecycle
--------------------------
  "noisy_ground_truth"  : default; GT xpos + Gaussian noise each step.
  "rgbd_pointcloud"     : perception estimate injected via set_peg_pos_estimate().
                          Held constant until grasp or cleared.
  "grasp_propagation"   : post-grasp; peg position derived from EE pose via stored
                          T_grasp offset.  Activated by activate_grasp_propagation().
"""

from dataclasses import dataclass, field
import numpy as np

from .noise_model import GaussianNoise, DelayedNoise, orientation_noise


@dataclass
class Observation:
    """Partial, noisy observation available to the controller."""
    q: np.ndarray             # (7,) joint positions
    qdot: np.ndarray          # (7,) joint velocities
    gripper_width: float      # m

    ee_pos: np.ndarray        # (3,) end-effector position estimate
    ee_rot: np.ndarray        # (3,3) end-effector rotation estimate

    peg_pos: np.ndarray       # (3,) peg position estimate
    peg_rot: np.ndarray       # (3,3) peg rotation estimate
    peg_pos_cov: np.ndarray   # (3,3) covariance of peg position estimate

    hole_pos: np.ndarray      # (3,) hole entrance position estimate
    hole_rot: np.ndarray      # (3,3) hole axis estimate
    hole_pos_cov: np.ndarray  # (3,3) covariance of hole position estimate

    contact_detected: bool
    external_force: np.ndarray   # (6,) [fx,fy,fz, tx,ty,tz] estimate

    stage: int                # current planner stage index
    time: float

    # Pose source tags — added after all required fields to preserve backward compat
    peg_pos_source: str = field(default="noisy_ground_truth")
    hole_pos_source: str = field(default="noisy_ground_truth")


class SensorWrapper:
    """Wraps PegInHoleEnv to produce noisy observations.

    Parameters
    ----------
    env : PegInHoleEnv
    noise_cfg : dict  (from noise.yaml)
    noise_level : str  "easy" | "medium" | "hard"
    seed : int
    """

    def __init__(self, env, noise_cfg: dict,
                 noise_level: str = "easy", seed: int = 0,
                 hole_pos_bias: np.ndarray = None,
                 hole_pos_sigma: float = None):
        self._env = env
        self._cfg = noise_cfg
        self._level = noise_level
        self._rng = np.random.default_rng(seed)

        jcfg = noise_cfg["joint"]
        ftcfg = noise_cfg["force_torque"]
        pcfg = noise_cfg["fallback_pose_noise"][noise_level]

        self._q_noise = GaussianNoise(jcfg["position_sigma"], rng=self._rng)
        self._qdot_noise = GaussianNoise(jcfg["velocity_sigma"], rng=self._rng)

        self._ft_noise = DelayedNoise(
            sigma=ftcfg["force_sigma"] + ftcfg["torque_sigma"],
            delay_steps=ftcfg["delay_steps"],
            bias=ftcfg["bias"],
            rng=self._rng,
        )

        self._pos_sigma = pcfg["pos_sigma"]
        self._rot_sigma = pcfg["rot_sigma"]
        self._contact_threshold = noise_cfg["contact_threshold"]

        # Per-sensor sigmas (hole_pos_sigma=0 models a fixed calibrated fixture)
        self._peg_pos_sigma_m  = pcfg["pos_sigma"]
        self._hole_pos_sigma_m = pcfg["pos_sigma"] if hole_pos_sigma is None else float(hole_pos_sigma)

        self._current_stage: int = 0
        # Intentional fixed bias on hole position (stress testing only)
        self._hole_pos_bias = np.array(hole_pos_bias, dtype=float) if hole_pos_bias is not None else np.zeros(3)
        # Hole override (set by PerceptionModule)
        self._hole_pos_override: np.ndarray | None = None
        self._hole_pos_source: str = "noisy_ground_truth"

        # Peg pose injection — three-state pipeline:
        #   rgbd_pointcloud: static estimate set at sub-task start
        #   grasp_propagation: offset in EE frame, computed at grasp moment
        self._peg_pos_override: np.ndarray | None = None
        self._peg_grasp_offset_ee: np.ndarray | None = None   # T_peg_in_ee (3,)
        self._peg_rot_in_ee: np.ndarray | None = None          # R_peg_in_ee (3,3)
        self._peg_pos_source: str = "noisy_ground_truth"

    # ── stage ─────────────────────────────────────────────────────────────────

    def set_stage(self, stage: int) -> None:
        self._current_stage = stage

    # ── hole pose injection ───────────────────────────────────────────────────

    def set_hole_pos_estimate(self, pos: np.ndarray | None) -> None:
        """Override hole position used by get_observation().

        Pass an ndarray from PerceptionModule to inject an estimated hole pose
        (replaces per-step Gaussian noise + bias).  Pass None to revert to the
        built-in noise model.
        """
        self._hole_pos_override = (np.array(pos, dtype=float)
                                   if pos is not None else None)
        self._hole_pos_source = "perception_estimate" if pos is not None else "noisy_ground_truth"

    # ── peg pose injection ────────────────────────────────────────────────────

    def set_peg_pos_estimate(self, pos: np.ndarray | None) -> None:
        """Inject a perception-estimated peg position (pre-grasp).

        Called once per sub-task from MultiTaskAssemblyTask after observe().
        Clears any active grasp-propagation state.
        """
        self._peg_pos_override = np.array(pos, dtype=float) if pos is not None else None
        self._peg_pos_source = "rgbd_pointcloud" if pos is not None else "noisy_ground_truth"
        self._peg_grasp_offset_ee = None
        self._peg_rot_in_ee = None

    def activate_grasp_propagation(self, ee_pos: np.ndarray, ee_rot: np.ndarray) -> None:
        """Record T_grasp at the moment of weld activation.

        T_peg_in_ee = R_ee^T @ (p_peg_est − p_ee_est)

        After this call every get_observation() derives peg_pos from EE pose,
        making peg tracking immune to arm occlusion of the scene camera.

        Falls back to GT peg_pos when no perception estimate is available.
        """
        ts = self._env.get_true_state()
        p_peg = self._peg_pos_override if self._peg_pos_override is not None else ts.peg_pos
        self._peg_grasp_offset_ee = ee_rot.T @ (p_peg - ee_pos)
        self._peg_rot_in_ee = ee_rot.T @ ts.peg_rot   # GT rotation (not estimated visually)
        self._peg_pos_source = "grasp_propagation"

    def clear_peg_pose_estimate(self) -> None:
        """Reset all peg pose injection — call at the start of each sub-task."""
        self._peg_pos_override = None
        self._peg_grasp_offset_ee = None
        self._peg_rot_in_ee = None
        self._peg_pos_source = "noisy_ground_truth"

    @property
    def grasp_transform(self) -> np.ndarray | None:
        """Current T_peg_in_ee offset (3,), or None if no grasp active."""
        return self._peg_grasp_offset_ee.copy() if self._peg_grasp_offset_ee is not None else None

    def get_observation(self) -> Observation:
        ts = self._env.get_true_state()   # only SensorWrapper reads true state

        # Joint noise
        q_meas = self._q_noise.sample(ts.q)
        qdot_meas = self._qdot_noise.sample(ts.qdot)

        # EE from FK using noisy q — here we just add small noise to true EE
        # (in real system, EE is computed from noisy joint angles)
        ee_pos = self._q_noise.sample(ts.ee_pos)  # tiny noise via joint encoder
        ee_rot = ts.ee_rot.copy()  # rotation changes negligibly

        # Peg pose: propagation > perception estimate > GT+noise
        if self._peg_grasp_offset_ee is not None:
            # Post-grasp: T_peg_world = p_ee + R_ee @ T_peg_in_ee
            peg_pos = ee_pos + ee_rot @ self._peg_grasp_offset_ee
            peg_rot = ee_rot @ self._peg_rot_in_ee
            peg_pos_source = "grasp_propagation"
        elif self._peg_pos_override is not None:
            # Pre-grasp: use perception estimate (constant, no per-step noise)
            peg_pos = self._peg_pos_override.copy()
            peg_rot = orientation_noise(ts.peg_rot, self._rot_sigma, self._rng)
            peg_pos_source = "rgbd_pointcloud"
        else:
            # Fallback: GT + Gaussian noise
            peg_pos = ts.peg_pos + self._rng.normal(0.0, self._pos_sigma, size=3)
            peg_rot = orientation_noise(ts.peg_rot, self._rot_sigma, self._rng)
            peg_pos_source = "noisy_ground_truth"

        # Hole pose: perception estimate (fixed) or GT+noise
        if self._hole_pos_override is not None:
            hole_pos = self._hole_pos_override.copy()
            hole_pos_source = self._hole_pos_source
        else:
            hole_sigma = self._hole_pos_sigma_m
            hole_pos = (ts.hole_pos
                        + (self._rng.normal(0.0, hole_sigma, size=3) if hole_sigma > 0.0 else 0.0)
                        + self._hole_pos_bias)
            hole_pos_source = "noisy_ground_truth"
        hole_rot = orientation_noise(np.eye(3), self._rot_sigma * 0.3, self._rng)

        cov = (self._pos_sigma ** 2) * np.eye(3)

        # Force/torque noise + delay
        ft_true = ts.contact_force  # [fx,fy,fz,tx,ty,tz]
        ft_meas = self._ft_noise.sample(ft_true)
        f_norm = np.linalg.norm(ft_meas[:3])
        contact_detected = f_norm > self._contact_threshold

        return Observation(
            q=q_meas,
            qdot=qdot_meas,
            gripper_width=ts.gripper_width,
            ee_pos=ee_pos,
            ee_rot=ee_rot,
            peg_pos=peg_pos,
            peg_rot=peg_rot,
            peg_pos_cov=cov,
            hole_pos=hole_pos,
            hole_rot=hole_rot,
            hole_pos_cov=cov,
            contact_detected=contact_detected,
            external_force=ft_meas,
            stage=self._current_stage,
            time=ts.time,
            peg_pos_source=peg_pos_source,
            hole_pos_source=hole_pos_source,
        )
