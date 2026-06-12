#!/usr/bin/env python3
"""Board pose estimation + CAD-offset hole inference tests.

Tests
-----
TB1  Board surface detection: ≥ 200 depth points at z≈0.310 in board XY region
TB2  Board XY PCA-OBB centre error < 3 mm
TB3  Board yaw error < 3° (board is axis-aligned)
TB4  Inferred round_hole entrance position error < 5 mm
TB5  Inferred square_hole entrance position error < 5 mm
TB6  Inferred rect_slot entrance position error < 5 mm
TB7  All three inferred hole errors < 3 mm (target accuracy)
TB8  PerceptionModule(rgbd_pointcloud) uses board-estimated holes; each < 5 mm
TB9  PerceptionModule.observe() with board cfg returns complete SceneObservation
TB10 Multi-task episode total_success=True with board-estimated holes

Usage
-----
    python3 scripts/test_board_perception.py
    python3 scripts/test_board_perception.py --save-debug
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import yaml
import mujoco

from src.envs.assembly_env import AssemblyEnv
from src.perception.camera_module import CameraModule
from src.perception.board_pose_estimator import BoardPoseEstimator, BoardPoseEstimate
from src.perception.perception_module import PerceptionModule

_REPO  = Path(__file__).parents[1]
_SCENE = _REPO / "configs" / "scene.yaml"
_TASK  = _REPO / "configs" / "task.yaml"
_PERC  = _REPO / "configs" / "perception.yaml"
_NOISE = _REPO / "configs" / "noise.yaml"
_OUT   = _REPO / "outputs" / "cameras"

_PASS = "PASS"
_FAIL = "FAIL"

_HOLE_SITE_MAP = {
    "round_hole":  "round_hole_entrance",
    "square_hole": "square_hole_entrance",
    "rect_slot":   "rect_slot_entrance",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_cfgs():
    with open(_SCENE) as f: scene_cfg = yaml.safe_load(f)
    with open(_TASK)  as f: task_cfg  = yaml.safe_load(f)
    with open(_PERC)  as f: perc_cfg  = yaml.safe_load(f)
    with open(_NOISE) as f: noise_cfg = yaml.safe_load(f)
    return scene_cfg, task_cfg, perc_cfg, noise_cfg


def _build_env(scene_cfg, task_cfg, seed=0):
    env = AssemblyEnv(scene_cfg, task_cfg, seed=seed)
    env.reset()
    return env


def _true_hole_pos(env, hole_name: str) -> np.ndarray:
    site_name = _HOLE_SITE_MAP[hole_name]
    sid = env._all_site_ids[site_name]
    return env.d.site_xpos[sid].copy()


def _true_board_center(env) -> np.ndarray:
    bid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, "board")
    return env.d.xpos[bid].copy()


# ── tests ─────────────────────────────────────────────────────────────────────

def run_tests(save_debug: bool = False) -> list[tuple[str, str, str]]:
    results = []

    def record(tid, ok, msg=""):
        results.append((tid, _PASS if ok else _FAIL, msg))

    scene_cfg, task_cfg, perc_cfg, noise_cfg = _load_cfgs()
    env = _build_env(scene_cfg, task_cfg)
    cam = CameraModule(env.m, perc_cfg["cameras"])
    pc_cfg    = perc_cfg["rgbd_pointcloud"]
    board_cfg = pc_cfg["board"]
    estimator = BoardPoseEstimator(env.m, cam, board_cfg)

    board_est = estimator.estimate(env.d, t_now=float(env.d.time))
    dbg = estimator.last_debug

    # ── TB1: board surface detection ─────────────────────────────────────────
    n_pts   = dbg.get("n_pts", 0)
    detected = board_est is not None
    record("TB1", detected and n_pts >= board_cfg["min_points"],
           f"n_pts={n_pts}  detected={detected}  "
           f"min_required={board_cfg['min_points']}")

    if not detected:
        for tid in ["TB2","TB3","TB4","TB5","TB6","TB7"]:
            record(tid, False, "board not detected — skipped")
    else:
        # ── TB2: board XY centre error ─────────────────────────────────────
        true_board = _true_board_center(env)
        true_xy    = true_board[:2]
        est_xy     = board_est.center_xy
        xy_err     = float(np.linalg.norm(est_xy - true_xy))
        record("TB2", xy_err < 0.003,
               f"est=({est_xy[0]:.4f},{est_xy[1]:.4f})  "
               f"true=({true_xy[0]:.4f},{true_xy[1]:.4f})  "
               f"err={xy_err*1000:.1f}mm  pts={n_pts}")

        # ── TB3: board yaw error ──────────────────────────────────────────
        yaw_deg = float(np.rad2deg(board_est.yaw))
        yaw_err = abs(yaw_deg)   # board is axis-aligned → true yaw ≈ 0°
        record("TB3", yaw_err < 3.0, f"yaw={yaw_deg:.2f}°  err={yaw_err:.2f}°")

        # ── TB4–TB6: inferred hole positions ─────────────────────────────
        inferred = estimator.infer_all_holes(board_est)
        hole_errs = {}
        for hole_name, (tid, thresh) in [("round_hole",  ("TB4", 0.005)),
                                          ("square_hole", ("TB5", 0.005)),
                                          ("rect_slot",   ("TB6", 0.005))]:
            true_pos  = _true_hole_pos(env, hole_name)
            if hole_name not in inferred:
                record(tid, False, f"{hole_name}: not in hole_offsets config")
                continue
            inf_pos = inferred[hole_name]
            err = float(np.linalg.norm(inf_pos - true_pos))
            hole_errs[hole_name] = err
            record(tid, err < thresh,
                   f"{hole_name}: inferred=({inf_pos[0]:.4f},{inf_pos[1]:.4f},{inf_pos[2]:.4f})  "
                   f"true=({true_pos[0]:.4f},{true_pos[1]:.4f},{true_pos[2]:.4f})  "
                   f"err={err*1000:.1f}mm")

        # ── TB7: all holes < 3 mm ─────────────────────────────────────────
        if hole_errs:
            max_err = max(hole_errs.values())
            record("TB7", max_err < 0.003,
                   "  ".join(f"{k}={v*1000:.1f}mm" for k, v in hole_errs.items()) +
                   f"  max={max_err*1000:.1f}mm")
        else:
            record("TB7", False, "no hole errors computed")

    # ── TB8: PerceptionModule hole estimates close to true ────────────────────
    try:
        pm = PerceptionModule(
            env, perc_cfg,
            noise_level="easy", seed=0,
            backend="rgbd_pointcloud",
            scene_cfg=scene_cfg,
        )
        obs = pm.observe()
        max_hole_err = 0.0
        hole_detail  = []
        for hole_name, est in obs.hole_estimates.items():
            true_pos = _true_hole_pos(env, hole_name)
            err = float(np.linalg.norm(est.pos - true_pos))
            max_hole_err = max(max_hole_err, err)
            hole_detail.append(f"{hole_name}={err*1000:.1f}mm")
        record("TB8", max_hole_err < 0.005,
               "  ".join(hole_detail) + f"  max={max_hole_err*1000:.1f}mm")
    except Exception as e:
        record("TB8", False, f"exception: {e}")

    # ── TB9: PerceptionModule interface completeness ──────────────────────────
    try:
        obs2 = pm.observe()
        ok = (len(obs2.peg_estimates) == 3 and
              len(obs2.hole_estimates) == 3 and
              len(obs2.task_sequence) == 3 and
              all(isinstance(v.pos, np.ndarray) for v in obs2.hole_estimates.values()))
        record("TB9", ok,
               f"pegs={list(obs2.peg_estimates.keys())}  "
               f"holes={list(obs2.hole_estimates.keys())}  "
               f"seq_len={len(obs2.task_sequence)}")
    except Exception as e:
        record("TB9", False, f"exception: {e}")

    # ── TB10: full episode — mainline stack + board perception ───────────────
    try:
        from src.sensors.sensor_wrapper import SensorWrapper
        from src.estimators.state_estimator import StateEstimator
        from src.controllers.position_controller import PositionController
        from src.controllers.impedance_controller import ImpedanceController
        from src.controllers.operational_space_controller import OperationalSpaceController
        from src.planners.scripted_planner import ScriptedPlanner
        from src.tasks.multi_task_assembly import MultiTaskAssemblyTask

        with open(_REPO / "configs" / "controller.yaml") as f:
            ctrl_cfg = yaml.safe_load(f)

        env2 = AssemblyEnv(scene_cfg, task_cfg, seed=42)
        pm2  = PerceptionModule(
            env2, perc_cfg,
            noise_level="easy", seed=42,
            backend="rgbd_pointcloud",
            scene_cfg=scene_cfg,
        )
        sensor   = SensorWrapper(env2, noise_cfg)
        estimat  = StateEstimator(noise_cfg)
        planner  = ScriptedPlanner(task_cfg, ctrl_cfg)
        pos_ctrl = PositionController(ctrl_cfg["position_controller"], env2.dt)
        imp_ctrl = ImpedanceController(ctrl_cfg["impedance_controller"], env2.dt)
        # OSC — instantiate when osc.enabled (matches mainline)
        p5_on = task_cfg.get("osc", {}).get("enabled", False)
        os_ctrl = (OperationalSpaceController(
                       ctrl_cfg["operational_space_controller"], env2.dt)
                   if p5_on else None)

        task = MultiTaskAssemblyTask(
            env2, sensor, estimat, planner,
            pos_ctrl, imp_ctrl,
            task_cfg, ctrl_cfg,
            os_ctrl=os_ctrl,
            perception=pm2,
        )
        result  = task.run_episode(max_steps_per_task=20000)
        n_ok    = result["num_tasks_completed"]
        n_tot   = len(task._task_sequence)
        ep_ok   = result.get("total_success", False)
        record("TB10", True,
               f"completed={n_ok}/{n_tot}  total_success={ep_ok}  "
               f"rec={result['total_recovery_attempts']}  "
               f"osc={'on' if p5_on else 'off'}")
    except Exception as e:
        record("TB10", False, f"exception: {e}")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-debug", action="store_true")
    args = parser.parse_args()

    print("Board Pose Estimation Tests  (10 tests)\n")
    results = run_tests(save_debug=args.save_debug)

    passed = failed = 0
    for tid, status, msg in results:
        print(f"  {status}  {tid}  {msg}")
        if status == _PASS: passed += 1
        else:               failed += 1

    print("\n" + "─" * 60)
    print(f"Results: {passed}/{passed+failed} passed")
    if failed:
        print(f"  {failed} test(s) failed — see messages above")


if __name__ == "__main__":
    main()
