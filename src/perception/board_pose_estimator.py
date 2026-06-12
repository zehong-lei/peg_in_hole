"""BoardPoseEstimator: estimate the assembly board pose from top-camera depth.

Algorithm
---------
1. Render depth from camera_top.
2. Backproject all pixels to world-frame 3D points.
3. Z-band filter: keep points at z ∈ [top_z - z_lo, top_z + z_hi]
   — isolates the board top surface (table is 60mm below, arm is far above).
4. Coarse XY spatial filter: discard points outside the known board region,
   eliminating staging-area pegs and robot fingers.
5. PCA-oriented bounding-box (OBB) center → board XY position.
6. PCA yaw, normalized to (-π/2, π/2] to resolve the ±sign eigenvector ambiguity.
7. Return BoardPoseEstimate or None if < min_points survive.

Hole inference
--------------
Given a BoardPoseEstimate, each hole's world position is:
    hole_pos = R(yaw) @ local_offset_xy + board_center_xy   (XY)
    hole_z   = board_top_z                                  (Z = entrance height)

where local_offset_xy comes from the scene CAD geometry stored in the config.

Note on yaw ambiguity
---------------------
PCA eigenvectors have an arbitrary ±sign, so the returned angle can be θ or θ+π.
Normalizing to (-π/2, π/2] maps both to the same value. This is correct for any
board tilt in (-π/2, π/2] but silently flips the X direction for boards tilted
beyond ±90°. In this scene the board is axis-aligned (yaw ≈ 0°) so this is fine.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

from .pointcloud_utils import depth_to_pointcloud, pca_obb_center, pca_yaw


@dataclass
class BoardPoseEstimate:
    """Result of one board pose estimation call."""
    center_xy: np.ndarray   # (2,) world XY of board centre (at top surface level)
    yaw: float              # board rotation about Z, radians, in (-π/2, π/2]
    top_z: float            # board top surface world Z
    n_pts: int              # number of depth points used for the estimate
    observed_at_time: float # simulation time stamp


class BoardPoseEstimator:
    """Estimate assembly board pose from the top-down depth camera.

    Parameters
    ----------
    model         : mujoco.MjModel
    camera_module : CameraModule
    board_cfg     : dict from perception.yaml["rgbd_pointcloud"]["board"]
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 camera_module,
                 board_cfg: dict):
        self._model   = model
        self._cam_mod = camera_module
        self._cfg     = board_cfg

        cam_name = board_cfg.get("camera", "camera_top")
        full_cam = camera_module._resolve(cam_name)
        self._cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, full_cam)
        if self._cam_id < 0:
            raise ValueError(f"Camera '{full_cam}' not found in model.")
        self._cam_short = cam_name

        self._top_z    = float(board_cfg.get("top_z", 0.310))
        self._z_lo     = float(board_cfg.get("z_lo_margin", 0.005))
        self._z_hi     = float(board_cfg.get("z_hi_margin", 0.005))
        self._min_pts  = int(board_cfg.get("min_points", 200))

        x_range = board_cfg.get("x_range", [0.44, 0.67])
        y_range = board_cfg.get("y_range", [0.03, 0.18])
        self._x_min, self._x_max = float(x_range[0]), float(x_range[1])
        self._y_min, self._y_max = float(y_range[0]), float(y_range[1])

        # CAD hole offsets in board-local XY frame {hole_name: {x, y}}
        self._hole_offsets: dict[str, dict] = board_cfg.get("hole_offsets", {})

        self._last_debug: dict = {}
        self._cached_estimate: Optional[BoardPoseEstimate] = None

    # ── public ───────────────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Discard any cached board estimate so the next call re-detects."""
        self._cached_estimate = None

    def estimate(self,
                 data: mujoco.MjData,
                 t_now: float = 0.0,
                 use_cache: bool = True) -> Optional[BoardPoseEstimate]:
        """Estimate the board pose from the current depth frame.

        Returns None if fewer than min_points are found after filtering.
        """
        if use_cache and self._cached_estimate is not None:
            return self._cached_estimate

        cam_pos = data.cam_xpos[self._cam_id].copy()
        cam_mat = data.cam_xmat[self._cam_id].reshape(3, 3).copy()
        fovy    = float(self._model.cam_fovy[self._cam_id])

        depth = self._cam_mod.get_depth(self._cam_short, data)
        H, W  = depth.shape

        pts, _ = depth_to_pointcloud(depth, cam_pos, cam_mat, fovy, W, H)

        # Z-band: board top surface
        z_ok = ((pts[:, 2] >= self._top_z - self._z_lo) &
                (pts[:, 2] <= self._top_z + self._z_hi))
        pts = pts[z_ok]

        # Coarse XY spatial filter
        xy_ok = ((pts[:, 0] >= self._x_min) & (pts[:, 0] <= self._x_max) &
                 (pts[:, 1] >= self._y_min) & (pts[:, 1] <= self._y_max))
        pts = pts[xy_ok]

        self._last_debug = {
            "n_pts": len(pts),
            "status": "ok" if len(pts) >= self._min_pts else "too_few",
        }

        if len(pts) < self._min_pts:
            return None

        xy_pts = pts[:, :2]

        center = pca_obb_center(xy_pts)
        yaw    = pca_yaw(xy_pts)

        # Normalize yaw to (-π/2, π/2] — resolves PCA ±sign ambiguity
        if yaw > np.pi / 2:
            yaw -= np.pi
        elif yaw <= -np.pi / 2:
            yaw += np.pi

        self._last_debug["center_xy"] = center.tolist()
        self._last_debug["yaw_deg"]   = float(np.rad2deg(yaw))

        result = BoardPoseEstimate(
            center_xy=center,
            yaw=yaw,
            top_z=self._top_z,
            n_pts=len(pts),
            observed_at_time=t_now,
        )

        if use_cache:
            self._cached_estimate = result

        return result

    def infer_hole_pos(self,
                       board_est: BoardPoseEstimate,
                       local_x: float,
                       local_y: float) -> np.ndarray:
        """Compute world-frame hole entrance position from board pose + CAD offset.

        Parameters
        ----------
        board_est        : result of estimate()
        local_x, local_y : hole centre in board-local XY frame (metres)

        Returns
        -------
        (3,) world position  [x, y, board_top_z]
        """
        cy, sy = np.cos(board_est.yaw), np.sin(board_est.yaw)
        R2 = np.array([[cy, -sy], [sy, cy]])
        xy = board_est.center_xy + R2 @ np.array([local_x, local_y])
        return np.array([xy[0], xy[1], board_est.top_z])

    def infer_all_holes(self, board_est: BoardPoseEstimate) -> dict[str, np.ndarray]:
        """Infer all configured hole entrance positions.

        Returns
        -------
        {hole_name: (3,) world position}  — only holes in hole_offsets config
        """
        return {
            name: self.infer_hole_pos(board_est, off["x"], off["y"])
            for name, off in self._hole_offsets.items()
        }

    @property
    def hole_offsets(self) -> dict:
        return self._hole_offsets

    @property
    def last_debug(self) -> dict:
        return self._last_debug
