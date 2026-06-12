#!/usr/bin/env python3
"""Point-cloud peg pose estimation tests.

Tests
-----
T1  Top camera renders non-trivial RGB and depth
T2  Color segmentation detects gold (round peg) mask
T3  Color segmentation detects blue (square peg) mask
T4  Color segmentation detects red/orange (rect peg) mask
T5  Backprojection places table-surface points at correct world Z
T6  Round peg detected; XY position error < 5 mm
T7  Square peg detected; XY position error < 5 mm
T8  Rect peg detected; XY position error < 8 mm (target < 5 mm); yaw finite
T9  PerceptionModule(backend=rgbd_pointcloud).observe() returns complete SceneObservation
T10 Multi-task episode runs to completion with rgbd_pointcloud backend

Usage
-----
    python3 scripts/test_peg_perception.py
    python3 scripts/test_peg_perception.py --save-debug   # save debug PNGs
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import yaml
import mujoco

from src.envs.assembly_env import AssemblyEnv
from src.envs.scene_builder import build_scene
from src.perception.camera_module import CameraModule
from src.perception.color_segmenter import ColorSegmenter, rgb_to_hsv, apply_morph
from src.perception.pointcloud_utils import (
    depth_to_pointcloud, pca_obb_center
)
from src.perception.pointcloud_pose_estimator import PointCloudPoseEstimator
from src.perception.perception_module import PerceptionModule

_REPO  = Path(__file__).parents[1]
_SCENE = _REPO / "configs" / "scene.yaml"
_TASK  = _REPO / "configs" / "task.yaml"
_PERC  = _REPO / "configs" / "perception.yaml"
_NOISE = _REPO / "configs" / "noise.yaml"
_OUT   = _REPO / "outputs" / "cameras"

# Point-cloud peg detection thresholds
_XY_ERR_ROUND  = 0.005   # 5 mm
_XY_ERR_SQUARE = 0.005   # 5 mm
_XY_ERR_RECT   = 0.008   # 8 mm (target < 5 mm)

_PASS = "PASS"
_FAIL = "FAIL"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_cfgs():
    with open(_SCENE) as f: scene_cfg = yaml.safe_load(f)
    with open(_TASK)  as f: task_cfg  = yaml.safe_load(f)
    with open(_PERC)  as f: perc_cfg  = yaml.safe_load(f)
    with open(_NOISE) as f: noise_cfg = yaml.safe_load(f)
    return scene_cfg, task_cfg, perc_cfg, noise_cfg


def _build_env(scene_cfg, task_cfg):
    env = AssemblyEnv(scene_cfg, task_cfg, seed=0)
    env.reset()
    return env


def _build_camera(m, perc_cfg):
    return CameraModule(m, perc_cfg["cameras"])


def _true_peg_xy(env, peg_name):
    bid = env._body_ids_raw[peg_name]
    return env.d.xpos[bid][:2].copy()


# ── tests ─────────────────────────────────────────────────────────────────────

def run_tests(save_debug: bool = False) -> list[tuple[str, str, str]]:
    """Run all tests, return list of (test_id, status, message) tuples."""
    results = []

    def record(tid, ok, msg=""):
        status = _PASS if ok else _FAIL
        results.append((tid, status, msg))

    scene_cfg, task_cfg, perc_cfg, noise_cfg = _load_cfgs()
    env   = _build_env(scene_cfg, task_cfg)
    cam   = _build_camera(env.m, perc_cfg)
    pc_cfg = perc_cfg["rgbd_pointcloud"]

    rgb   = cam.get_rgb("top", env.d)
    depth = cam.get_depth("top", env.d)

    # ── T1: camera renders valid images ──────────────────────────────────────
    rgb_ok   = rgb.shape == (480, 640, 3) and rgb.std() > 1
    depth_ok = depth.shape == (480, 640) and depth.min() > 0.5 and depth.max() < 2.0
    record("T1", rgb_ok and depth_ok,
           f"rgb={rgb.shape} std={rgb.std():.1f}  depth=[{depth.min():.3f},{depth.max():.3f}]")

    # ── T2–T4: color segmentation (raw mask, before morph) ───────────────────
    segmenter = ColorSegmenter(pc_cfg["peg_colors"])
    seg = segmenter.segment_with_stats(rgb)

    peg_labels = {"peg": "T2 gold", "peg_square": "T3 blue", "peg_rect": "T4 red"}
    for peg_name, label in peg_labels.items():
        info = seg[peg_name]
        ok   = info["count"] >= pc_cfg["min_points"]
        record(label[:2], ok,
               f"{peg_name}: {info['count']} px  H={info['mean_h']:.0f}°  "
               f"S={info['mean_s']:.2f}  V={info['mean_v']:.2f}")

    # ── T5: backprojection world-Z ────────────────────────────────────────────
    cam_id  = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, "camera_top")
    cam_pos = env.d.cam_xpos[cam_id]
    cam_mat = env.d.cam_xmat[cam_id].reshape(3, 3)
    fovy    = float(env.m.cam_fovy[cam_id])
    H, W    = depth.shape

    pts, _ = depth_to_pointcloud(depth, cam_pos, cam_mat, fovy, W, H)
    table_pts = pts[(pts[:, 2] > 0.248) & (pts[:, 2] < 0.252)]
    table_ok  = len(table_pts) > 500
    z_err     = abs(table_pts[:, 2].mean() - 0.250) if table_ok else float("nan")
    record("T5", table_ok and (z_err < 0.005),
           f"table pts={len(table_pts)}  mean_z={table_pts[:,2].mean():.4f}  err={z_err*1000:.1f}mm")

    # ── T6–T8: peg detection with per-method error comparison ────────────────
    peg_hl = PerceptionModule._extract_peg_half_lengths(scene_cfg)
    debug_dir = _OUT if save_debug else None
    estimator = PointCloudPoseEstimator(env.m, cam, pc_cfg, peg_hl)
    estimates = estimator.estimate(env.d, t_now=float(env.d.time),
                                   save_debug_dir=debug_dir)

    thresholds = {"peg": (_XY_ERR_ROUND, "T6"),
                  "peg_square": (_XY_ERR_SQUARE, "T7"),
                  "peg_rect":   (_XY_ERR_RECT,   "T8")}

    for peg_name, (thresh, tid) in thresholds.items():
        true_xy = _true_peg_xy(env, peg_name)

        if peg_name not in estimates:
            dbg = estimator.last_debug.get(peg_name, {})
            record(tid, False,
                   f"{peg_name}: NOT detected  status={dbg.get('status','?')} "
                   f"seg={dbg.get('seg_count',0)}px")
            continue

        dbg = estimator.last_debug[peg_name]

        # Per-method errors
        method_errors = {}
        for method, key in [("centroid", "centroid_xy"),
                             ("bbox2d",   "bbox2d_xy"),
                             ("pca_obb",  "pca_obb_xy")]:
            if key in dbg:
                xy_m = np.array(dbg[key])
                method_errors[method] = float(np.linalg.norm(xy_m - true_xy))

        used_method = dbg.get("center_method", "centroid")
        est = estimates[peg_name]
        xy_err = float(np.linalg.norm(est.pos[:2] - true_xy))

        method_str = "  ".join(
            f"{m}={e*1000:.1f}mm" for m, e in method_errors.items()
        )

        extra = ""
        if tid == "T8":
            yaw_deg = dbg.get("yaw_deg", 0.0)
            extra   = f"  yaw={yaw_deg:.1f}°"

        ok = xy_err < thresh
        record(tid, ok,
               f"{peg_name}: [{used_method}] err={xy_err*1000:.1f}mm  "
               f"thresh={thresh*1000:.0f}mm  "
               f"methods: {method_str}"
               f"  pts={dbg['n_pts']}{extra}")

    if save_debug:
        print(f"  [debug] images saved to {_OUT}/")

    # ── T9: PerceptionModule interface ────────────────────────────────────────
    try:
        pm = PerceptionModule(
            env, perc_cfg,
            noise_level="easy",
            seed=0,
            backend="rgbd_pointcloud",
            scene_cfg=scene_cfg,
        )
        obs = pm.observe()
        interface_ok = (
            len(obs.peg_estimates) == 3
            and len(obs.hole_estimates) == 3
            and len(obs.task_sequence) == 3
            and all(isinstance(v.pos, np.ndarray) for v in obs.peg_estimates.values())
        )
        record("T9", interface_ok,
               f"pegs={list(obs.peg_estimates.keys())}  "
               f"holes={list(obs.hole_estimates.keys())}  "
               f"seq_len={len(obs.task_sequence)}")
    except Exception as e:
        record("T9", False, f"exception: {e}")

    # ── T10: multi-task episode with rgbd_pointcloud backend ─────────────────
    try:
        from src.sensors.sensor_wrapper import SensorWrapper
        from src.estimators.state_estimator import StateEstimator
        from src.controllers.position_controller import PositionController
        from src.controllers.impedance_controller import ImpedanceController
        from src.planners.scripted_planner import ScriptedPlanner
        from src.tasks.multi_task_assembly import MultiTaskAssemblyTask

        with open(_REPO / "configs" / "controller.yaml") as f:
            ctrl_cfg = yaml.safe_load(f)

        env2 = AssemblyEnv(scene_cfg, task_cfg, seed=42)
        pm2  = PerceptionModule(
            env2, perc_cfg,
            noise_level="easy",
            seed=42,
            backend="rgbd_pointcloud",
            scene_cfg=scene_cfg,
        )
        sensor    = SensorWrapper(env2, noise_cfg)
        estimator = StateEstimator(noise_cfg)
        planner   = ScriptedPlanner(task_cfg, ctrl_cfg)
        pos_ctrl  = PositionController(ctrl_cfg["position_controller"],
                                       env2.dt)
        imp_ctrl  = ImpedanceController(ctrl_cfg["impedance_controller"],
                                        env2.dt)

        task = MultiTaskAssemblyTask(
            env2, sensor, estimator, planner,
            pos_ctrl, imp_ctrl,
            task_cfg, ctrl_cfg,
            perception=pm2,
        )
        result = task.run_episode(max_steps_per_task=15000)
        n_ok  = result["num_tasks_completed"]
        n_tot = len(task._task_sequence)
        ep_ok = result.get("total_success", False)
        record("T10", True,
               f"completed={n_ok}/{n_tot}  total_success={ep_ok}  "
               f"rec={result['total_recovery_attempts']}")
    except Exception as e:
        record("T10", False, f"exception: {e}")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-debug", action="store_true",
                        help="Save colour-mask and overlay debug images to outputs/cameras/")
    args = parser.parse_args()

    print("Point-Cloud Peg Pose Estimation Tests  (10 tests)\n")
    results = run_tests(save_debug=args.save_debug)

    passed = failed = 0
    for tid, status, msg in results:
        marker = "  " if status == _PASS else "! "
        print(f"  {status}  {tid}  {msg}")
        if status == _PASS:
            passed += 1
        else:
            failed += 1

    print("\n" + "─" * 60)
    print(f"Results: {passed}/{passed+failed} passed")
    if failed:
        print(f"  {failed} test(s) failed — see messages above")


if __name__ == "__main__":
    main()
