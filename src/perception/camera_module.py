"""CameraModule: RGB-D rendering from named MuJoCo cameras.

Two fixed cameras:
  camera_top     — top-down view covering staging area and assembly board
  camera_oblique — oblique front/side view of the assembly board

Public API uses short names ("top", "oblique") that are mapped to the
full MuJoCo camera names ("camera_top", "camera_oblique").
"""

from __future__ import annotations
import numpy as np
import mujoco


class CameraModule:
    """Renders RGB and depth images from named cameras in a MuJoCo scene.

    Parameters
    ----------
    model        : mujoco.MjModel
    camera_cfg   : dict  (from perception.yaml["cameras"])
                   Must contain:
                     names      : list[str]  MuJoCo camera names
                     resolution : [width, height]
    """

    def __init__(self, model: mujoco.MjModel, camera_cfg: dict):
        self._model = model
        w, h = camera_cfg["resolution"]
        self._renderer = mujoco.Renderer(model, height=h, width=w)
        # Map short name → full MuJoCo camera name
        self._name_map: dict[str, str] = {}
        for full_name in camera_cfg["names"]:
            # "camera_top" → short key "top"; pass-through if no prefix
            short = full_name[len("camera_"):] if full_name.startswith("camera_") else full_name
            self._name_map[short] = full_name
            self._name_map[full_name] = full_name  # full name also accepted

    # ── public API ─────────────────────────────────────────────────────────

    def get_rgb(self, name: str, data: mujoco.MjData) -> np.ndarray:
        """Render an RGB image from camera ``name``.

        Parameters
        ----------
        name : short ("top", "oblique") or full ("camera_top") camera name
        data : current MjData

        Returns
        -------
        np.ndarray  shape (H, W, 3) dtype uint8
        """
        cam = self._resolve(name)
        self._renderer.update_scene(data, camera=cam)
        return self._renderer.render().copy()

    def get_depth(self, name: str, data: mujoco.MjData) -> np.ndarray:
        """Render a depth image from camera ``name``.

        Parameters
        ----------
        name : short or full camera name
        data : current MjData

        Returns
        -------
        np.ndarray  shape (H, W) dtype float32, values in metres
        """
        cam = self._resolve(name)
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=cam)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return depth

    @property
    def camera_names(self) -> list[str]:
        """Full MuJoCo camera names managed by this module."""
        return list({v for v in self._name_map.values()})

    # ── private ─────────────────────────────────────────────────────────────

    def _resolve(self, name: str) -> str:
        if name not in self._name_map:
            raise KeyError(
                f"Unknown camera '{name}'. "
                f"Available: {sorted(self._name_map.keys())}"
            )
        return self._name_map[name]
