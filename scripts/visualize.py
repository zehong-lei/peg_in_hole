#!/usr/bin/env python3
"""Interactive MuJoCo viewer for the multi-task assembly episode.

Usage: python3 scripts/visualize.py [--tasks round square rect]
       [--seed N] [--noise easy|medium|hard] [--speed F] [--no-osc]
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.envs.assembly_env import AssemblyEnv
from src.sensors.sensor_wrapper import SensorWrapper
from src.estimators.state_estimator import StateEstimator
from src.controllers.position_controller import PositionController
from src.controllers.impedance_controller import ImpedanceController
from src.controllers.operational_space_controller import OperationalSpaceController
from src.planners.scripted_planner import ScriptedPlanner, Stage
from src.tasks.multi_task_assembly import MultiTaskAssemblyTask
from src.perception import PerceptionModule

_REPO = Path(__file__).parents[1]

_FULL_SEQUENCE = [
    ("peg",        "round_hole"),
    ("peg_square", "square_hole"),
    ("peg_rect",   "rect_slot"),
]
_TASK_ALIAS = {
    "round":  ("peg",        "round_hole"),
    "square": ("peg_square", "square_hole"),
    "rect":   ("peg_rect",   "rect_slot"),
}
_STAGE_LABEL = {
    Stage.MOVE_TO_PREGRASP:  "pregrasp",
    Stage.GRASP:             "grasp",
    Stage.LIFT:              "lift",
    Stage.MOVE_TO_PREINSERT: "preinsert",
    Stage.ALIGN:             "align",
    Stage.INSERT:            "insert",
    Stage.RELEASE:           "release",
    Stage.RETREAT:           "retreat",
    Stage.DONE:              "DONE",
    Stage.FAILED:            "FAILED",
}


def load_cfg(name: str) -> dict:
    return yaml.safe_load((_REPO / "configs" / f"{name}.yaml").read_text())


def resolve_sequence(task_names: list) -> list:
    wanted = {_TASK_ALIAS[n] for n in task_names}
    return [p for p in _FULL_SEQUENCE if p in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assembly episode live viewer")
    parser.add_argument("--tasks", nargs="+", default=["round", "square", "rect"],
                        choices=["round", "square", "rect"])
    parser.add_argument("--seed",   type=int,   default=0)
    parser.add_argument("--noise",  default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--speed",  type=float, default=1.0,
                        help="Playback multiplier (2 = 2× real-time)")
    parser.add_argument("--no-osc", action="store_true",
                        help="Disable OSC (use impedance controller only)")
    args = parser.parse_args()

    task_sequence = resolve_sequence(args.tasks)

    scene_cfg = load_cfg("scene")
    task_cfg  = load_cfg("task")
    ctrl_cfg  = load_cfg("controller")
    noise_cfg = load_cfg("noise")
    perc_cfg  = load_cfg("perception")

    # Extend timeouts for viewer (more time to watch each stage)
    to = list(task_cfg["stages"]["stage_timeout"])
    to[3] = 15.0
    to[5] = 150.0
    task_cfg["stages"]["stage_timeout"] = to
    task_cfg["contact_recovery"]["max_attempts"] = 10

    if args.no_osc:
        task_cfg["osc"]["enabled"] = False

    p5_enabled = task_cfg.get("osc", {}).get("enabled", False)

    env      = AssemblyEnv(scene_cfg, task_cfg, seed=args.seed)
    sensor   = SensorWrapper(env, noise_cfg, noise_level=args.noise,
                             seed=args.seed, hole_pos_sigma=0.0)
    estimator = StateEstimator(noise_cfg)
    planner  = ScriptedPlanner(task_cfg, ctrl_cfg)
    pos_ctrl = PositionController(ctrl_cfg["position_controller"],
                                  dt=task_cfg["sim"]["dt"])
    imp_ctrl = ImpedanceController(ctrl_cfg["impedance_controller"],
                                   dt=task_cfg["sim"]["dt"])
    os_ctrl  = (OperationalSpaceController(
                    ctrl_cfg["operational_space_controller"],
                    dt=task_cfg["sim"]["dt"])
                if p5_enabled else None)
    perception = PerceptionModule(
        env, perc_cfg,
        noise_level=args.noise,
        seed=args.seed,
        backend="rgbd_pointcloud",
        scene_cfg=scene_cfg,
    )

    task = MultiTaskAssemblyTask(
        env, sensor, estimator, planner,
        pos_ctrl, imp_ctrl, task_cfg, ctrl_cfg,
        os_ctrl=os_ctrl,
        task_sequence=task_sequence,
        perception=perception,
    )

    dt        = task_cfg["sim"]["dt"]
    step_wall = dt / args.speed

    print(f"Assembly viewer  |  tasks={' → '.join(args.tasks)}  "
          f"seed={args.seed}  noise={args.noise}  speed={args.speed}×  "
          f"osc={'on' if p5_enabled else 'off'}")
    print("Opening viewer …  (close window to exit)\n")

    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        v.cam.azimuth   = 145.0
        v.cam.elevation = -20.0
        v.cam.distance  = 1.4
        v.cam.lookat    = [0.50, 0.05, 0.35]
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True

        # ── per-step callbacks ────────────────────────────────────────────────
        t_step     = [time.perf_counter()]
        prev_task  = [-1]
        prev_stage = [None]

        def log_cb(task_idx, step, belief, cmd, ctrl):
            if task_idx != prev_task[0]:
                prev_task[0]  = task_idx
                prev_stage[0] = None
                peg_name, hole_name = task_sequence[task_idx]
                print(f"─── Task {task_idx+1}/{len(task_sequence)}: "
                      f"{peg_name} → {hole_name} ───")

            if cmd.stage != prev_stage[0]:
                ts    = env.get_true_state()
                hz    = env.d.site_xpos[env.site_ids["hole_entrance"]][2]
                depth = (hz - ts.peg_tip_pos[2]) * 1000
                f_mag = float(np.linalg.norm(belief.external_force[:3]))
                label = _STAGE_LABEL.get(cmd.stage, str(cmd.stage))
                src   = getattr(belief, "peg_pos_source", "?")
                print(f"  step={step:5d}  {label:<12}  "
                      f"ee_z={ts.ee_pos[2]:.3f}  "
                      f"depth={depth:+.1f}mm  F={f_mag:.1f}N  "
                      f"peg_src={src}")
                if cmd.stage in (Stage.DONE, Stage.FAILED):
                    status = "✓ OK" if cmd.stage == Stage.DONE else "✗ FAILED"
                    depth_val = max(0.0, depth)
                    print(f"  → {status}  depth={depth_val:.1f}mm  "
                          f"rec={planner.recovery_attempts}  "
                          f"t={env.d.time:.1f}s\n")
                prev_stage[0] = cmd.stage

        def step_cb():
            v.sync()
            elapsed   = time.perf_counter() - t_step[0]
            remaining = step_wall - elapsed
            if remaining > 0:
                time.sleep(remaining)
            t_step[0] = time.perf_counter()
            return None if v.is_running() else False

        t_step[0] = time.perf_counter()
        task.run_episode(max_steps_per_task=20000, log_cb=log_cb, step_cb=step_cb)

        if v.is_running():
            print("All tasks done — viewer remains open (close window to exit).")
            while v.is_running():
                mujoco.mj_step(env.m, env.d)
                v.sync()
                time.sleep(dt)


if __name__ == "__main__":
    main()
