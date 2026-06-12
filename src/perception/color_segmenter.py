"""Color segmentation for peg detection.

Uses HSV color space (pure numpy, no OpenCV dependency) to produce
per-peg binary masks from a rendered RGB image.  Optional morphological
cleaning (open, close, fill_holes, largest_component) via scipy.ndimage.

HSV thresholds were calibrated against MuJoCo-rendered images from
camera_top with the scene's standard headlight (diffuse=0.6, ambient=0.3).
"""

from __future__ import annotations
import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert H×W×3 uint8 RGB to H×W×3 float32 HSV.

    Returns H ∈ [0, 360], S ∈ [0, 1], V ∈ [0, 1].
    """
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]

    maxc  = np.maximum(np.maximum(r, g), b)
    minc  = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(maxc > 1e-6, delta / maxc, 0.0)

    # Hue — compute all three branches unconditionally, then select
    eps = 1e-10
    hr = (60.0 * ((g - b) / (delta + eps))) % 360.0
    hg =  60.0 * ((b - r) / (delta + eps) + 2.0)
    hb =  60.0 * ((r - g) / (delta + eps) + 4.0)

    h = np.where(maxc == r, hr,
        np.where(maxc == g, hg, hb)) % 360.0
    h = np.where(delta < 1e-6, 0.0, h)

    return np.stack([h, s, v], axis=-1)


def apply_morph(mask: np.ndarray, morph_cfg: dict) -> np.ndarray:
    """Apply morphological operations to a 2-D boolean mask.

    Supported operations (applied in order when truthy):
      open            : erosion then dilation (removes small blobs)
      close           : dilation then erosion (fills small holes)
      fill_holes      : flood-fill from border then invert
      largest_component : keep only the largest connected component

    Parameters
    ----------
    mask     : (H, W) bool
    morph_cfg: dict with optional keys open, close (int — iterations),
               fill_holes (bool), largest_component (bool)

    Returns
    -------
    (H, W) bool — cleaned mask (copy, does not modify input)
    """
    if not morph_cfg:
        return mask

    from scipy import ndimage as ndi

    m = mask.astype(bool)

    open_iters = int(morph_cfg.get("open", 0))
    if open_iters > 0:
        m = ndi.binary_erosion(m,  iterations=open_iters)
        m = ndi.binary_dilation(m, iterations=open_iters)

    close_iters = int(morph_cfg.get("close", 0))
    if close_iters > 0:
        m = ndi.binary_dilation(m, iterations=close_iters)
        m = ndi.binary_erosion(m,  iterations=close_iters)

    if morph_cfg.get("fill_holes", False):
        m = ndi.binary_fill_holes(m)

    if morph_cfg.get("largest_component", False):
        labeled, n = ndi.label(m)
        if n > 1:
            sizes = ndi.sum(m, labeled, range(1, n + 1))
            best  = int(np.argmax(sizes)) + 1
            m     = labeled == best

    return m


class ColorSegmenter:
    """Generate per-peg color masks from a rendered RGB image.

    Each peg has a colour specification defined by (h_min, h_max, s_min, v_min).
    Hue is in degrees [0, 360]; saturation and value in [0, 1].

    Parameters
    ----------
    color_specs : dict  {peg_name → {h_min, h_max, s_min, v_min}}
                  Loaded from perception.yaml["rgbd_pointcloud"]["peg_colors"].
    """

    def __init__(self, color_specs: dict):
        self._specs = color_specs

    def segment(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Segment ``rgb`` into one boolean mask per peg.

        Parameters
        ----------
        rgb : (H, W, 3) uint8

        Returns
        -------
        {peg_name: (H, W) bool mask}
        """
        hsv = rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        masks: dict[str, np.ndarray] = {}
        for peg_name, spec in self._specs.items():
            h_min = float(spec["h_min"])
            h_max = float(spec["h_max"])
            s_min = float(spec["s_min"])
            v_min = float(spec["v_min"])

            h_ok = (h >= h_min) & (h <= h_max)
            sv_ok = (s >= s_min) & (v >= v_min)
            masks[peg_name] = h_ok & sv_ok

        return masks

    def segment_with_stats(
        self, rgb: np.ndarray
    ) -> dict[str, dict]:
        """Segment and return masks with diagnostic statistics.

        Returns
        -------
        {peg_name: {"mask": (H,W) bool, "count": int,
                    "mean_h": float, "mean_s": float, "mean_v": float}}
        """
        hsv   = rgb_to_hsv(rgb)
        masks = self.segment(rgb)
        result = {}
        for peg_name, mask in masks.items():
            count = int(mask.sum())
            if count > 0:
                mean_h = float(hsv[mask, 0].mean())
                mean_s = float(hsv[mask, 1].mean())
                mean_v = float(hsv[mask, 2].mean())
            else:
                mean_h = mean_s = mean_v = 0.0
            result[peg_name] = {
                "mask":   mask,
                "count":  count,
                "mean_h": mean_h,
                "mean_s": mean_s,
                "mean_v": mean_v,
            }
        return result
