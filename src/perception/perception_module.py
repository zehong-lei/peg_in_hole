"""Perception data types, noise model, and PerceptionModule backends.

PerceptionModule reads true poses from AssemblyEnv, adds configurable Gaussian
noise, and produces a SceneObservation with estimated poses and a derived task
sequence.  Supports noisy-ground-truth and RGB-D point-cloud backends.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from src.sensors.noise_model import orientation_noise
from .types import PoseEstimate, SceneObservation
from .camera_module import CameraModule
from .pointcloud_pose_estimator import PointCloudPoseEstimator
from .board_pose_estimator import BoardPoseEstimator


# ── Pose noise model ──────────────────────────────────────────────────────────

class PoseNoiseModel:
    """Additive Gaussian noise on position; axis-angle noise on rotation."""

    def __init__(self,
                 pos_sigma: float,
                 rot_sigma: float = 0.0,
                 bias: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None):
        self._pos_sigma = float(pos_sigma)
        self._rot_sigma = float(rot_sigma)
        self._bias = np.array(bias, dtype=float) if bias is not None else np.zeros(3)
        self._rng  = rng if rng is not None else np.random.default_rng()

    def sample_pos(self, true_pos: np.ndarray) -> np.ndarray:
        pos = np.array(true_pos, dtype=float)
        if self._pos_sigma > 0.0:
            pos = pos + self._rng.normal(0.0, self._pos_sigma, size=3)
        return pos + self._bias

    def sample_rot(self, true_rot: np.ndarray) -> np.ndarray:
        if self._rot_sigma > 0.0:
            return orientation_noise(true_rot, self._rot_sigma, self._rng)
        return np.array(true_rot, dtype=float)

    def pos_covariance(self) -> np.ndarray:
        return (self._pos_sigma ** 2) * np.eye(3)

    @classmethod
    def from_cfg(cls,
                 cfg_level: dict,
                 rng: Optional[np.random.Generator] = None,
                 is_hole: bool = True) -> "PoseNoiseModel":
        """Construct from one noise-level sub-dict in perception.yaml."""
        pos_sigma = float(cfg_level.get("pos_sigma", 0.0))
        if is_hole:
            bias = np.array(cfg_level.get("bias", [0.0, 0.0, 0.0]), dtype=float)
            return cls(pos_sigma=pos_sigma, rot_sigma=0.0, bias=bias, rng=rng)
        else:
            rot_deg = float(cfg_level.get("rot_sigma_deg", 0.0))
            return cls(pos_sigma=pos_sigma, rot_sigma=np.deg2rad(rot_deg), rng=rng)

# MuJoCo site names for each hole entrance
_HOLE_SITE_MAP = {
    "round_hole":  "round_hole_entrance",
    "square_hole": "square_hole_entrance",
    "rect_slot":   "rect_slot_entrance",
}

# Canonical peg body names (order defines task sequence)
_FULL_PEG_ORDER = ["peg", "peg_square", "peg_rect"]


class PerceptionModule:
    """Noisy-ground-truth perception module.

    Parameters
    ----------
    env            : AssemblyEnv
    perception_cfg : dict  (from perception.yaml)
    noise_level    : "easy" | "medium" | "hard" | "custom"
    seed           : int
    custom_hole_sigma : float | None
        If noise_level == "custom", override hole pos_sigma with this value.
    """

    def __init__(self,
                 env,
                 perception_cfg: dict,
                 noise_level: str = "easy",
                 seed: int = 0,
                 custom_hole_sigma: Optional[float] = None,
                 enable_cameras: bool = False,
                 backend: str = "noisy_ground_truth",
                 scene_cfg: Optional[dict] = None):
        self._env     = env
        self._cfg     = perception_cfg
        self._level   = noise_level
        self._backend = backend

        rng = np.random.default_rng(seed)

        hole_noise_cfg = perception_cfg["noise"]["hole"][noise_level]
        peg_noise_cfg  = perception_cfg["noise"]["peg"][noise_level]

        if noise_level == "custom" and custom_hole_sigma is not None:
            hole_noise_cfg = dict(hole_noise_cfg)
            hole_noise_cfg["pos_sigma"] = float(custom_hole_sigma)

        self._hole_noise = PoseNoiseModel.from_cfg(hole_noise_cfg, rng=rng, is_hole=True)
        self._peg_noise  = PoseNoiseModel.from_cfg(peg_noise_cfg,  rng=rng, is_hole=False)

        self._shape_matching: dict[str, list[str]] = perception_cfg["shape_matching"]
        self._object_names: list[str] = perception_cfg["objects"]
        self._hole_names:   list[str] = perception_cfg["holes"]

        # rgbd_pointcloud backend forces cameras on
        if backend == "rgbd_pointcloud":
            enable_cameras = True

        cam_cfg = perception_cfg.get("cameras")
        self._camera: Optional[CameraModule] = (
            CameraModule(env.m, cam_cfg) if enable_cameras and cam_cfg else None
        )

        # Point-cloud + board estimators (rgbd_pointcloud backend only)
        self._pc_estimator: Optional[PointCloudPoseEstimator] = None
        self._board_estimator: Optional[BoardPoseEstimator] = None
        if backend == "rgbd_pointcloud":
            pc_cfg = perception_cfg.get("rgbd_pointcloud", {})
            # Hole noise for this backend (default: zero-noise / custom)
            _hole_lvl = pc_cfg.get("hole_noise_level", "custom")
            _hole_nc  = perception_cfg["noise"]["hole"].get(_hole_lvl,
                         perception_cfg["noise"]["hole"]["custom"])
            self._pc_hole_noise = PoseNoiseModel.from_cfg(_hole_nc, rng=rng, is_hole=True)
            # Peg half-lengths from scene_cfg
            peg_hl = self._extract_peg_half_lengths(scene_cfg or {})
            self._pc_estimator = PointCloudPoseEstimator(
                env.m, self._camera, pc_cfg, peg_hl)
            # Board pose estimator (optional; needs "board:" section in pc_cfg)
            board_cfg = pc_cfg.get("board")
            if board_cfg:
                self._board_estimator = BoardPoseEstimator(env.m, self._camera, board_cfg)

    # ── public API ─────────────────────────────────────────────────────────────

    def observe(self, env=None) -> SceneObservation:
        """Capture a scene observation.

        Dispatches to the configured backend:
          - "noisy_ground_truth" : Gaussian-noise fallback (adds noise to GT poses)
          - "rgbd_pointcloud"    : colour+depth vision backend (camera-based)

        Parameters
        ----------
        env : AssemblyEnv | None
            If None, uses the env passed at construction.
        """
        _env = env if env is not None else self._env
        if self._backend == "rgbd_pointcloud":
            return self._observe_rgbd(_env)
        return self._observe_noisy_gt(_env)

    # ── backend implementations ────────────────────────────────────────────────

    def _observe_noisy_gt(self, _env) -> SceneObservation:
        """Noisy-ground-truth backend: adds Gaussian noise to simulator GT poses."""
        t_now = float(_env.d.time)

        hole_estimates: dict[str, PoseEstimate] = {}
        for hole_name in self._hole_names:
            true_pos = self._get_true_hole_pos(_env, hole_name)
            est_pos  = self._hole_noise.sample_pos(true_pos)
            hole_estimates[hole_name] = PoseEstimate(
                name=hole_name,
                pos=est_pos,
                rot=np.eye(3),
                pos_cov=self._hole_noise.pos_covariance(),
                observed_at_time=t_now,
            )

        peg_estimates: dict[str, PoseEstimate] = {}
        for peg_name in self._object_names:
            true_pos, true_rot = self._get_true_peg_pose(_env, peg_name)
            est_pos = self._peg_noise.sample_pos(true_pos)
            est_rot = self._peg_noise.sample_rot(true_rot)
            peg_estimates[peg_name] = PoseEstimate(
                name=peg_name,
                pos=est_pos,
                rot=est_rot,
                pos_cov=self._peg_noise.pos_covariance(),
                observed_at_time=t_now,
            )

        task_sequence = self._derive_task_sequence(peg_estimates, hole_estimates)
        return SceneObservation(
            peg_estimates=peg_estimates,
            hole_estimates=hole_estimates,
            task_sequence=task_sequence,
            captured_at_time=t_now,
        )

    def _observe_rgbd(self, _env) -> SceneObservation:
        """RGB-D point-cloud backend.

        Holes: board depth-band detection → CAD-offset hole inference.
               Falls back to GT + hole_noise if board not detected.
        Pegs:  HSV colour segmentation + point-cloud centroid/OBB.
               Falls back to noisy GT per peg if not detected.
        """
        t_now = float(_env.d.time)

        # ── Board pose → hole position inference ─────────────────────────────
        board_est = None
        if self._board_estimator is not None:
            board_est = self._board_estimator.estimate(_env.d, t_now=t_now)

        hole_estimates: dict[str, PoseEstimate] = {}
        for hole_name in self._hole_names:
            if board_est is not None:
                off = self._board_estimator.hole_offsets.get(hole_name)
                if off is not None:
                    hole_pos = self._board_estimator.infer_hole_pos(
                        board_est, off["x"], off["y"])
                    est_pos = self._pc_hole_noise.sample_pos(hole_pos)
                else:
                    true_pos = self._get_true_hole_pos(_env, hole_name)
                    est_pos  = self._pc_hole_noise.sample_pos(true_pos)
            else:
                # Fallback: ground truth + noise
                true_pos = self._get_true_hole_pos(_env, hole_name)
                est_pos  = self._pc_hole_noise.sample_pos(true_pos)

            hole_estimates[hole_name] = PoseEstimate(
                name=hole_name,
                pos=est_pos,
                rot=np.eye(3),
                pos_cov=self._pc_hole_noise.pos_covariance(),
                observed_at_time=t_now,
            )

        # ── Pegs — point-cloud estimation with noisy-GT fallback ─────────────
        pc_results = self._pc_estimator.estimate(_env.d, t_now=t_now)

        peg_estimates: dict[str, PoseEstimate] = {}
        for peg_name in self._object_names:
            if peg_name in pc_results:
                peg_estimates[peg_name] = pc_results[peg_name]
            else:
                true_pos, true_rot = self._get_true_peg_pose(_env, peg_name)
                est_pos = self._peg_noise.sample_pos(true_pos)
                est_rot = self._peg_noise.sample_rot(true_rot)
                peg_estimates[peg_name] = PoseEstimate(
                    name=peg_name,
                    pos=est_pos,
                    rot=est_rot,
                    pos_cov=self._peg_noise.pos_covariance(),
                    observed_at_time=t_now,
                )

        task_sequence = self._derive_task_sequence(peg_estimates, hole_estimates)
        return SceneObservation(
            peg_estimates=peg_estimates,
            hole_estimates=hole_estimates,
            task_sequence=task_sequence,
            captured_at_time=t_now,
        )

    def get_perception_errors(self,
                              scene_obs: SceneObservation,
                              env=None) -> dict:
        """Compute true-vs-estimated errors for diagnostics.

        Returns
        -------
        dict with keys:
          hole_pos_errors : {hole_name → float}  Euclidean error in metres
          peg_pos_errors  : {peg_name  → float}
          mean_hole_pos_error : float
          mean_peg_pos_error  : float
          mean_object_pose_error : float  (same as mean_peg_pos_error for pos)
        """
        _env = env if env is not None else self._env

        hole_errors: dict[str, float] = {}
        for hole_name, est in scene_obs.hole_estimates.items():
            true_pos = self._get_true_hole_pos(_env, hole_name)
            hole_errors[hole_name] = float(np.linalg.norm(est.pos - true_pos))

        peg_errors: dict[str, float] = {}
        for peg_name, est in scene_obs.peg_estimates.items():
            true_pos, _ = self._get_true_peg_pose(_env, peg_name)
            peg_errors[peg_name] = float(np.linalg.norm(est.pos - true_pos))

        mean_hole = float(np.mean(list(hole_errors.values()))) if hole_errors else 0.0
        mean_peg  = float(np.mean(list(peg_errors.values())))  if peg_errors  else 0.0

        return {
            "hole_pos_errors":       hole_errors,
            "peg_pos_errors":        peg_errors,
            "mean_hole_pos_error":   mean_hole,
            "mean_peg_pos_error":    mean_peg,
            "mean_object_pose_error": mean_peg,   # alias used in metrics
        }

    # ── camera API ────────────────────────────────────────────────────────────

    def get_rgb(self, name: str) -> np.ndarray:
        """Render an RGB image from the named camera.

        Parameters
        ----------
        name : "top" or "oblique" (short names) or full MuJoCo camera name

        Returns
        -------
        np.ndarray  shape (H, W, 3) uint8
        """
        if self._camera is None:
            raise RuntimeError(
                "Camera rendering is not enabled. "
                "Pass enable_cameras=True when constructing PerceptionModule."
            )
        return self._camera.get_rgb(name, self._env.d)

    def get_depth(self, name: str) -> np.ndarray:
        """Render a depth image from the named camera.

        Parameters
        ----------
        name : "top" or "oblique" (short names) or full MuJoCo camera name

        Returns
        -------
        np.ndarray  shape (H, W) float32, values in metres
        """
        if self._camera is None:
            raise RuntimeError(
                "Camera rendering is not enabled. "
                "Pass enable_cameras=True when constructing PerceptionModule."
            )
        return self._camera.get_depth(name, self._env.d)

    # ── private helpers ────────────────────────────────────────────────────────

    def _get_true_hole_pos(self, env, hole_name: str) -> np.ndarray:
        """Read hole entrance site position directly from MuJoCo data.

        Uses env._all_site_ids (NOT env.site_ids) so the active-task alias
        does not interfere when observing non-active holes.
        """
        site_name = _HOLE_SITE_MAP[hole_name]
        sid = env._all_site_ids[site_name]
        return env.d.site_xpos[sid].copy()

    def _get_true_peg_pose(self, env, peg_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read peg body position and rotation matrix."""
        bid = env._body_ids_raw[peg_name]
        pos = env.d.xpos[bid].copy()
        rot = env.d.xmat[bid].reshape(3, 3).copy()
        return pos, rot

    @staticmethod
    def _extract_peg_half_lengths(scene_cfg: dict) -> dict:
        """Read peg z-half-extents from scene.yaml structure."""
        pegs = scene_cfg.get("pegs", {})
        hl: dict[str, float] = {}
        if "round" in pegs:
            hl["peg"] = float(pegs["round"].get("half_length", 0.070))
        if "square" in pegs:
            hl["peg_square"] = float(pegs["square"]["half_size"][2])
        if "rect" in pegs:
            hl["peg_rect"] = float(pegs["rect"]["half_size"][2])
        return hl

    def _derive_task_sequence(self,
                               peg_estimates: dict,
                               hole_estimates: dict) -> list:
        """Derive (peg, hole) task sequence from shape-matching config.

        Order follows the canonical peg order defined in _FULL_PEG_ORDER so
        that the sequence is deterministic regardless of dict iteration order.
        """
        sequence = []
        for peg_name in _FULL_PEG_ORDER:
            if peg_name not in peg_estimates:
                continue
            matched_holes = self._shape_matching.get(peg_name, [])
            for hole_name in matched_holes:
                if hole_name in hole_estimates:
                    sequence.append((peg_name, hole_name))
                    break   # first match wins
        return sequence
