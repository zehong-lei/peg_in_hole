"""PointCloudPoseEstimator: peg pose estimation from top-camera RGB-D.

Algorithm
---------
1. Render RGB + depth from camera_top.
2. Segment each peg's color mask (HSV thresholds).
3. Morphological cleaning per peg (open/close/fill_holes/largest_component).
4. Backproject depth at cleaned mask pixels to world-frame 3D points.
5. Filter by world Z (above table surface, below robot arm).
6. Remove statistical outliers.
7. Compute three center estimates:
     centroid   — mean XY of surviving points
     bbox2d     — midpoint of axis-aligned bounding box in XY
     pca_obb    — center of PCA-oriented bounding box in XY
8. Select center by peg's configured center_method.
9. Z set from geometry: table_z + peg_half_length (body-centre Z).
10. PCA on XY cloud → yaw estimate (meaningful only for rect peg).
11. Return {peg_name → PoseEstimate}.

Detection is optional: if < min_points survive for a peg, the entry is
absent and the caller should fall back to ground-truth + noise.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import mujoco

from .types import PoseEstimate
from .color_segmenter import ColorSegmenter, apply_morph
from .pointcloud_utils import (
    depth_to_pointcloud, remove_outliers, pca_yaw, pca_obb_center
)


def _yaw_to_rotmat(yaw: float) -> np.ndarray:
    """2-D yaw angle → 3×3 rotation matrix (rotation around z-axis)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0., 0., 1.0]])


class PointCloudPoseEstimator:
    """Estimate peg poses from the top-down RGB-D camera.

    Parameters
    ----------
    model      : mujoco.MjModel
    camera_module : CameraModule  (for RGB + depth rendering)
    pc_cfg     : dict  (perception.yaml["rgbd_pointcloud"])
    peg_half_lengths : {peg_name → half-z in metres}
                Used to recover body-centre Z from surface point cloud.
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 camera_module,
                 pc_cfg: dict,
                 peg_half_lengths: dict):
        self._model   = model
        self._cam_mod = camera_module
        self._cfg     = pc_cfg
        self._hl      = peg_half_lengths

        cam_name = pc_cfg.get("camera", "camera_top")
        full_cam = camera_module._resolve(cam_name)
        self._cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, full_cam)
        if self._cam_id < 0:
            raise ValueError(f"Camera '{full_cam}' not found in model.")

        self._cam_short = cam_name

        color_specs = pc_cfg["peg_colors"]
        self._segmenter = ColorSegmenter(color_specs)

        self._table_z    = float(pc_cfg.get("table_z", 0.250))
        self._z_margin   = float(pc_cfg.get("peg_z_min_margin", 0.005))
        self._z_max      = float(pc_cfg.get("peg_z_max", 0.420))
        self._min_pts    = int(pc_cfg.get("min_points", 8))
        self._outlier_k  = int(pc_cfg.get("outlier_k", 5))
        self._outlier_std = float(pc_cfg.get("outlier_std", 2.0))

        self._last_debug: dict = {}

    # ── public ───────────────────────────────────────────────────────────────

    def estimate(self,
                 data: mujoco.MjData,
                 t_now: float = 0.0,
                 save_debug_dir: Optional[Path] = None,
                 ) -> dict[str, Optional[PoseEstimate]]:
        """Estimate peg poses from the current simulation state.

        Parameters
        ----------
        data          : current MjData
        t_now         : simulation time stamp to embed in PoseEstimate
        save_debug_dir: if given, save per-peg debug images to this directory

        Returns
        -------
        {peg_name → PoseEstimate}  — missing key means detection failed
        """
        cam_pos = data.cam_xpos[self._cam_id].copy()
        cam_mat = data.cam_xmat[self._cam_id].reshape(3, 3).copy()
        fovy    = float(self._model.cam_fovy[self._cam_id])

        rgb   = self._cam_mod.get_rgb(self._cam_short, data)
        depth = self._cam_mod.get_depth(self._cam_short, data)
        H, W  = depth.shape

        seg_stats = self._segmenter.segment_with_stats(rgb)

        estimates: dict[str, Optional[PoseEstimate]] = {}
        debug: dict = {}

        for peg_name, stats in seg_stats.items():
            spec  = self._cfg["peg_colors"][peg_name]
            mask_raw = stats["mask"]
            count_raw = stats["count"]

            dbg = {
                "seg_count_raw": count_raw,
                "mean_h": stats["mean_h"],
                "mean_s": stats["mean_s"],
                "mean_v": stats["mean_v"],
            }

            # Morphological cleaning
            morph_cfg = spec.get("morph", {})
            mask = apply_morph(mask_raw, morph_cfg)
            count = int(mask.sum())
            dbg["seg_count"] = count

            if count < self._min_pts:
                dbg["status"] = "seg_too_few"
                debug[peg_name] = dbg
                if save_debug_dir is not None:
                    self._save_debug(save_debug_dir, peg_name, rgb, mask_raw, mask,
                                     None, None, None, None)
                continue

            # Backproject masked pixels
            pts, pix = depth_to_pointcloud(
                depth, cam_pos, cam_mat, fovy, W, H,
                mask=mask, max_depth=5.0)

            # Filter by world Z
            z_min = self._table_z + self._z_margin
            z_ok  = (pts[:, 2] > z_min) & (pts[:, 2] < self._z_max)
            pts   = pts[z_ok]
            pix   = pix[z_ok]

            dbg["pts_after_z_filter"] = len(pts)

            if len(pts) < self._min_pts:
                dbg["status"] = "z_filter_too_few"
                debug[peg_name] = dbg
                if save_debug_dir is not None:
                    self._save_debug(save_debug_dir, peg_name, rgb, mask_raw, mask,
                                     None, None, None, None)
                continue

            # Outlier removal
            pts = remove_outliers(pts, k=self._outlier_k, std_thresh=self._outlier_std)

            dbg["pts_after_outlier"] = len(pts)

            if len(pts) < self._min_pts:
                dbg["status"] = "outlier_too_few"
                debug[peg_name] = dbg
                if save_debug_dir is not None:
                    self._save_debug(save_debug_dir, peg_name, rgb, mask_raw, mask,
                                     None, None, None, None)
                continue

            # Three center estimates
            xy_pts = pts[:, :2]

            centroid_xy  = xy_pts.mean(axis=0)
            bbox2d_xy    = (xy_pts.min(axis=0) + xy_pts.max(axis=0)) / 2.0
            pca_obb_xy   = pca_obb_center(xy_pts)

            dbg["centroid_xy"]  = centroid_xy.tolist()
            dbg["bbox2d_xy"]    = bbox2d_xy.tolist()
            dbg["pca_obb_xy"]   = pca_obb_xy.tolist()

            # Select center by configured method
            center_method = spec.get("center_method", "centroid")
            if center_method == "bbox2d":
                xy_est = bbox2d_xy
            elif center_method == "pca_obb":
                xy_est = pca_obb_xy
            else:
                xy_est = centroid_xy

            dbg["center_method"] = center_method

            # Z from geometry
            half_z = self._hl.get(peg_name, 0.060)
            est_z  = self._table_z + half_z
            est_pos = np.array([xy_est[0], xy_est[1], est_z])

            # Rotation via PCA yaw
            yaw     = pca_yaw(xy_pts)
            est_rot = _yaw_to_rotmat(yaw)

            # Covariance
            centered = xy_pts - centroid_xy
            cov_2d   = (centered.T @ centered) / max(len(pts) - 1, 1)
            pos_cov  = np.diag([cov_2d[0, 0], cov_2d[1, 1], 1e-4])

            dbg["status"]  = "detected"
            dbg["est_pos"] = est_pos.tolist()
            dbg["yaw_deg"] = float(np.rad2deg(yaw))
            dbg["n_pts"]   = len(pts)
            debug[peg_name] = dbg

            if save_debug_dir is not None:
                self._save_debug(save_debug_dir, peg_name, rgb, mask_raw, mask,
                                 pts, centroid_xy, bbox2d_xy, pca_obb_xy)

            estimates[peg_name] = PoseEstimate(
                name=peg_name,
                pos=est_pos,
                rot=est_rot,
                pos_cov=pos_cov,
                observed_at_time=t_now,
            )

        self._last_debug = debug
        return estimates

    @property
    def last_debug(self) -> dict:
        """Debug info from the most recent estimate() call."""
        return self._last_debug

    # ── debug image saving ────────────────────────────────────────────────────

    def _save_debug(self,
                    out_dir: Path,
                    peg_name: str,
                    rgb: np.ndarray,
                    mask_raw: np.ndarray,
                    mask_clean: np.ndarray,
                    pts_world,
                    centroid_xy,
                    bbox2d_xy,
                    pca_obb_xy):
        """Save per-peg debug images: raw RGB, HSV mask, cleaned mask, overlay."""
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image as PILImage
            _pil = True
        except ImportError:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _pil = False

        def _save_array(arr_uint8, path):
            if _pil:
                PILImage.fromarray(arr_uint8).save(str(path))
            else:
                plt.imsave(str(path), arr_uint8)

        # Raw RGB
        _save_array(rgb, out_dir / f"{peg_name}_rgb.png")

        # Raw HSV mask (green overlay on grey background)
        vis_raw = np.stack([mask_raw * 128] * 3, axis=-1).astype(np.uint8)
        vis_raw[mask_raw, 1] = 255
        _save_array(vis_raw, out_dir / f"{peg_name}_mask_raw.png")

        # Cleaned mask
        vis_clean = np.stack([mask_clean * 128] * 3, axis=-1).astype(np.uint8)
        vis_clean[mask_clean, 1] = 255
        _save_array(vis_clean, out_dir / f"{peg_name}_mask_clean.png")

        # Overlay (only when detection succeeded)
        if pts_world is not None and centroid_xy is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyArrowPatch

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(rgb)

            # Project 3-D world points back to pixel coords for overlay
            # (approximate: use cleaned mask pixel centroids for simplicity)
            rows, cols = np.where(mask_clean)
            if len(rows):
                ax.scatter(cols, rows, s=1, c="lime", alpha=0.5, label="clean mask")

            # Mark the three center estimates in image space if camera intrinsics available
            ax.set_title(f"{peg_name}  centroid/bbox2d/pca_obb  (world XY)")
            ax.axis("off")

            handles = []
            for label, xy, color in [
                ("centroid",  centroid_xy, "red"),
                ("bbox2d",    bbox2d_xy,   "blue"),
                ("pca_obb",   pca_obb_xy,  "yellow"),
            ]:
                handles.append(mpatches.Patch(color=color, label=f"{label} ({xy[0]:.3f},{xy[1]:.3f})"))

            ax.legend(handles=handles, loc="upper right", fontsize=6)
            plt.tight_layout()
            fig.savefig(str(out_dir / f"{peg_name}_overlay.png"), dpi=100)
            plt.close(fig)
