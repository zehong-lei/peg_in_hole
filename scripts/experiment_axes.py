"""Orthogonal experimental axes for the peg-in-hole benchmark.

Each axis maps one CLI choice onto exactly ONE config subtree.  The axes are
independent by construction: applying one axis never implicitly enables or
disables a layer owned by another axis.

  controller  → ctrl_cfg["operational_space_controller"]   (+ controller_kind)
  planner     → task_cfg["preinsert_ocp"] / task_cfg["joint_ocp"]
  contact     → task_cfg["lcs_mpc"] / task_cfg["contact_recovery"]
  perception  → sensor / PerceptionModule wiring (returned as a spec)

Values that are not yet implemented raise NotImplementedError rather than
silently aliasing onto an existing layer.
"""
from __future__ import annotations

CONTROLLERS = ["jointpos", "impedance", "osc", "osc-lambda"]
PLANNERS    = ["waypoint", "ee-ocp", "joint-ocp"]
CONTACTS    = ["none", "spiral", "force-guided", "lcs-mpc"]
PERCEPTIONS = ["gt", "gt-noise", "ema", "rgbd"]


# ── controller axis ─────────────────────────────────────────────────────────
def apply_controller(ctrl_cfg: dict, controller: str) -> tuple[str, bool]:
    """Configure the controller axis.

    Mutates only ctrl_cfg["operational_space_controller"].
    Returns (controller_kind, needs_os_ctrl).
    """
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller {controller!r}; choose from {CONTROLLERS}")
    osc = ctrl_cfg["operational_space_controller"]
    if controller == "osc":
        osc["dynamics_aware"] = False
        return "osc", True
    if controller == "osc-lambda":
        osc["dynamics_aware"] = True
        return "osc-lambda", True
    # jointpos / impedance need no operational-space controller instance
    return controller, False


# ── planner axis ────────────────────────────────────────────────────────────
def apply_planner(task_cfg: dict, planner: str) -> None:
    """Configure the planner axis.  Mutates only preinsert_ocp / joint_ocp."""
    if planner == "waypoint":
        task_cfg["preinsert_ocp"]["enabled"] = False
    elif planner == "ee-ocp":
        task_cfg["preinsert_ocp"]["enabled"] = True
    elif planner == "joint-ocp":
        raise NotImplementedError(
            "joint-ocp planner is not wired into ScriptedPlanner yet")
    else:
        raise ValueError(f"unknown planner {planner!r}; choose from {PLANNERS}")


# ── contact axis ────────────────────────────────────────────────────────────
def apply_contact(task_cfg: dict, contact: str,
                  slsqp_freq: int | None = None,
                  spiral_radius_mm: float | None = None) -> None:
    """Configure the contact axis.  Mutates only lcs_mpc / contact_recovery.

    spiral_radius_mm scales the spiral search pattern's max radius (the recovery
    search envelope); applies to both 'spiral' and 'lcs-mpc' (which uses the
    spiral as its safety fallback).  None leaves the config default (6 mm).
    """
    rec = task_cfg["contact_recovery"]
    lcs = task_cfg["lcs_mpc"]
    if spiral_radius_mm is not None:
        rec["spiral_max_radius"] = spiral_radius_mm * 1e-3
    if contact == "none":
        lcs["enabled"] = False
        rec["enabled"] = False
    elif contact == "spiral":
        lcs["enabled"] = False
        rec["enabled"] = True
    elif contact == "lcs-mpc":
        lcs["enabled"] = True
        rec["enabled"] = True   # safety fallback when MPC cannot escape a jam
        if slsqp_freq is not None:
            lcs["mpc"]["solver"] = "slsqp"
            lcs["mpc"]["mpc_freq_ratio"] = slsqp_freq
    elif contact == "force-guided":
        raise NotImplementedError(
            "force-guided contact policy is not implemented yet")
    else:
        raise ValueError(f"unknown contact {contact!r}; choose from {CONTACTS}")


# ── perception axis ─────────────────────────────────────────────────────────
def perception_spec(perception: str, noise_sigma_mm: float | None = None) -> dict:
    """Return a wiring spec for the perception axis (no config mutation).

    Keys
    ----
    use_perception     : bool   — instantiate a PerceptionModule
    backend            : str|None
    noise_level        : str    — SensorWrapper fallback-pose / FT noise level
    hole_pos_sigma     : float  — per-step sensor hole noise std (0 = calibrated)
    perception_level   : str|None — PerceptionModule noise level ("custom" here)
    custom_hole_sigma_m: float|None — hole pos noise std for the custom level [m]

    The base lateral offset is applied as a deterministic hole *bias* by the
    caller (in the perception custom level when perception is on; via the sensor
    when it is `gt`).  noise_sigma_mm sets the stochastic hole error for
    `gt-noise` (default 1 mm if unset, matching the legacy easy level).
    """
    if perception == "gt":
        # Ground-truth poses: no hole noise; lateral offset injected as a fixed
        # sensor hole bias by the caller.  Peg falls back to sensor noise floor.
        return dict(use_perception=False, backend=None,
                    noise_level="easy", hole_pos_sigma=0.0,
                    perception_level=None, custom_hole_sigma_m=None)
    if perception == "gt-noise":
        sig = (1.0 if noise_sigma_mm is None else noise_sigma_mm) * 1e-3
        return dict(use_perception=True, backend="noisy_ground_truth",
                    noise_level="easy", hole_pos_sigma=0.0,
                    perception_level="custom", custom_hole_sigma_m=sig)
    if perception == "rgbd":
        # rgbd hole error comes from the vision pipeline itself → add no extra
        # Gaussian (custom sigma 0); the base offset is still applied as a bias.
        return dict(use_perception=True, backend="rgbd_pointcloud",
                    noise_level="easy", hole_pos_sigma=0.0,
                    perception_level="custom", custom_hole_sigma_m=0.0)
    if perception == "ema":
        raise NotImplementedError(
            "ema perception axis has no distinct toggle (the StateEstimator "
            "EMA filter is always active); not implemented as a separate mode")
    raise ValueError(f"unknown perception {perception!r}; choose from {PERCEPTIONS}")
