"""Point-cloud utility functions for peg pose estimation.

All operations use plain numpy — no open3d or pcl required.
"""

from __future__ import annotations
import numpy as np


def depth_to_pointcloud(
    depth: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
    mask: np.ndarray | None = None,
    max_depth: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject a depth image to world-frame 3D points.

    MuJoCo camera convention: camera looks along -z_cam, +y_cam is image-up.
    Image convention: u (column) increases right, v (row) increases downward.

    Parameters
    ----------
    depth    : (H, W) float32 — metres along optical axis
    cam_pos  : (3,) camera world position  (d.cam_xpos[cam_id])
    cam_mat  : (3, 3) camera-frame → world-frame rotation  (d.cam_xmat[cam_id].reshape(3,3))
    fovy_deg : vertical field of view in degrees  (m.cam_fovy[cam_id])
    width, height : image dimensions
    mask     : (H, W) bool  — if given, only backproject True pixels
    max_depth : skip pixels with depth value > this

    Returns
    -------
    points  : (N, 3) float64 world-frame points
    pixels  : (N, 2) int    corresponding (col, row) pixel coordinates
    """
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg / 2.0))
    fx = fy
    cx = width  / 2.0
    cy = height / 2.0

    if mask is not None:
        rows, cols = np.where(mask)
    else:
        rows, cols = np.mgrid[0:height, 0:width]
        rows = rows.ravel()
        cols = cols.ravel()

    d = depth[rows, cols]
    valid = (d > 1e-3) & (d < max_depth)
    rows, cols, d = rows[valid], cols[valid], d[valid]

    # Camera-frame coordinates (camera looks along -z)
    x_c = (cols - cx) / fx * d
    y_c = -(rows - cy) / fy * d   # image-v down → camera-y up (flip)
    z_c = -d                       # depth is positive; camera z points backward

    p_cam = np.stack([x_c, y_c, z_c], axis=1)           # (N, 3)
    points = (cam_mat @ p_cam.T).T + cam_pos             # (N, 3) world frame

    pixels = np.stack([cols, rows], axis=1)
    return points, pixels


def remove_outliers(
    points: np.ndarray,
    k: int = 5,
    std_thresh: float = 2.0,
) -> np.ndarray:
    """Remove statistical outliers via mean k-NN distance threshold.

    Parameters
    ----------
    points     : (N, 3)
    k          : number of neighbours to average
    std_thresh : keep points whose mean-knn-dist < global_mean + std_thresh * global_std

    Returns
    -------
    (M, 3) inlier points
    """
    n = len(points)
    if n <= k + 1:
        return points

    # Pairwise squared distances (N × N) — works well for N < 2000
    diff = points[:, None, :] - points[None, :, :]      # (N, N, 3)
    sq   = (diff ** 2).sum(axis=-1)                      # (N, N)
    sq_sorted = np.sort(sq, axis=1)[:, 1:k + 1]         # exclude self (0)
    mean_k = np.sqrt(sq_sorted).mean(axis=1)             # (N,)

    mu  = mean_k.mean()
    sig = mean_k.std()
    return points[mean_k < mu + std_thresh * sig]


def pca_obb_center(points_xy: np.ndarray) -> np.ndarray:
    """Compute the center of the PCA-oriented bounding box (OBB) of a 2-D point set.

    Projects points onto the two PCA axes, finds min/max extents on each axis,
    and returns the midpoint in world (XY) coordinates.

    Parameters
    ----------
    points_xy : (N, 2)

    Returns
    -------
    (2,) center of the OBB
    """
    if len(points_xy) < 3:
        return points_xy.mean(axis=0)

    centered = points_xy - points_xy.mean(axis=0)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)          # columns are eigenvectors, ascending
    # Principal axes: eigvecs[:, -1] (major), eigvecs[:, -2] (minor)
    ax0 = eigvecs[:, -1]
    ax1 = eigvecs[:, -2]

    proj0 = points_xy @ ax0
    proj1 = points_xy @ ax1

    mid0 = (proj0.min() + proj0.max()) / 2.0
    mid1 = (proj1.min() + proj1.max()) / 2.0

    return mid0 * ax0 + mid1 * ax1


def pca_yaw(points_xy: np.ndarray) -> float:
    """Estimate yaw from 2D point cloud via PCA.

    Returns the angle of the principal axis in radians, in (-π/2, π/2].
    Useful for box pegs; round pegs have no meaningful yaw.

    Parameters
    ----------
    points_xy : (N, 2) x-y columns

    Returns
    -------
    yaw in radians
    """
    if len(points_xy) < 3:
        return 0.0
    centered = points_xy - points_xy.mean(axis=0)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue
    principal = eigenvectors[:, -1]
    return float(np.arctan2(principal[1], principal[0]))
