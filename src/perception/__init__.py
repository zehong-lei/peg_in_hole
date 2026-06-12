"""Perception interface.

Provides a PerceptionModule that produces estimated peg/hole poses from
noisy-ground-truth, RGB-D point-cloud, or board-pose backends without
changing any downstream code.
"""

from .types import PoseEstimate, SceneObservation
from .perception_module import PoseNoiseModel, PerceptionModule
from .camera_module import CameraModule
from .color_segmenter import ColorSegmenter, rgb_to_hsv, apply_morph
from .pointcloud_utils import depth_to_pointcloud, remove_outliers, pca_yaw, pca_obb_center
from .pointcloud_pose_estimator import PointCloudPoseEstimator
from .board_pose_estimator import BoardPoseEstimator, BoardPoseEstimate

__all__ = [
    "PoseEstimate",
    "SceneObservation",
    "PoseNoiseModel",
    "PerceptionModule",
    "CameraModule",
    "ColorSegmenter",
    "rgb_to_hsv",
    "apply_morph",
    "depth_to_pointcloud",
    "remove_outliers",
    "pca_yaw",
    "pca_obb_center",
    "PointCloudPoseEstimator",
    "BoardPoseEstimator",
    "BoardPoseEstimate",
]
