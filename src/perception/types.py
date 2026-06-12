"""Perception data classes shared across perception submodules."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PoseEstimate:
    """Estimated pose of one object or hole entrance.

    Intentionally contains NO true_pos field — downstream code must treat
    this as an opaque noisy measurement.
    """
    name: str
    pos: np.ndarray       # (3,) estimated position
    rot: np.ndarray       # (3,3) estimated rotation matrix
    pos_cov: np.ndarray   # (3,3) position covariance
    observed_at_time: float = 0.0


@dataclass
class SceneObservation:
    """Snapshot of all perceived objects and derived task sequence."""
    peg_estimates:    dict = field(default_factory=dict)   # {peg_name  → PoseEstimate}
    hole_estimates:   dict = field(default_factory=dict)   # {hole_name → PoseEstimate}
    task_sequence:    list = field(default_factory=list)   # ordered [(peg, hole), ...]
    captured_at_time: float = 0.0
