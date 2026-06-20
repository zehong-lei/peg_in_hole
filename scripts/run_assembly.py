#!/usr/bin/env python3
"""Multi-task scripted assembly episode runner (mainline pipeline).

Task sequence:  round_peg → square_peg → rect_peg

Mainline stack (single fixed configuration):
  planner    : EE-space pre-insertion OCP            (task.preinsert_ocp)
  contact    : reduced LCS-MPC force feedforward      (task.lcs_mpc)
               + spiral search recovery fallback      (task.contact_recovery)
  controller : operational-space control with inertia shaping (osc-lambda)
  perception : ground-truth poses (+sensor noise) or the RGB-D vision pipeline

Usage examples
--------------
  python scripts/run_assembly.py
  python scripts/run_assembly.py --tasks round
  python scripts/run_assembly.py --perception rgbd --seeds 5
  python scripts/run_assembly.py --offset-mm 6 --verbose
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from src.envs.assembly_env import AssemblyEnv
from src.sensors.sensor_wrapper import SensorWrapper
from src.estimators.state_estimator import StateEstimator
from src.controllers.position_controller import PositionController
from src.controllers.operational_space_controller import OperationalSpaceController
from src.planners.scripted_planner import ScriptedPlanner
from src.tasks.multi_task_assembly import MultiTaskAssemblyTask
from src.perception import PerceptionModule


_REPO_ROOT = Path(__file__).parents[1]

# Full ordered task sequence
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


def load_cfg(name: str) -> dict:
    return yaml.safe_load((_REPO_ROOT / "configs" / f"{name}.yaml").read_text())


def resolve_task_sequence(task_names: list[str]) -> list[tuple[str, str]]:
    """Convert shorthand names to (peg, hole) pairs preserving full-sequence order."""
    wanted = [_TASK_ALIAS[n] for n in task_names]
    return [p for p in _FULL_SEQUENCE if p in wanted]


def build_episode(task_sequence: list, *, perception: str, offset_mm: float,
                  seed: int):
    """Instantiate all mainline modules for one episode.

    perception : 'gt'   — sensor reports noisy ground-truth poses.
                 'rgbd' — poses come from the RGB-D point-cloud vision pipeline.
    offset_mm  : lateral hole misalignment (mm, X axis), injected as a
                 deterministic hole bias.
    """
    scene_cfg = load_cfg("scene")
    task_cfg  = load_cfg("task")
    ctrl_cfg  = load_cfg("controller")
    noise_cfg = load_cfg("noise")

    # The config defaults already encode the mainline (OCP + LCS-MPC + OSC-λ);
    # here we only set runner-level multi-task timeouts.
    to = list(task_cfg["stages"]["stage_timeout"])
    to[3] = 15.0     # MOVE_TO_PREINSERT — OCP may need longer
    to[5] = 150.0    # INSERT
    task_cfg["stages"]["stage_timeout"] = to
    task_cfg["contact_recovery"]["max_attempts"] = 10

    use_perc  = (perception == "rgbd")
    # Lateral offset → deterministic hole bias.  With the vision pipeline on, the
    # bias is injected into the PerceptionModule; otherwise onto the sensor.
    hole_bias = ([offset_mm * 1e-3, 0.0, 0.0]
                 if (offset_mm and not use_perc) else None)

    env       = AssemblyEnv(scene_cfg, task_cfg, seed=seed)
    sensor    = SensorWrapper(env, noise_cfg, noise_level="easy",
                              seed=seed, hole_pos_bias=hole_bias,
                              hole_pos_sigma=0.0)
    estimator = StateEstimator(noise_cfg)
    planner_o = ScriptedPlanner(task_cfg, ctrl_cfg)
    pos_ctrl  = PositionController(ctrl_cfg["position_controller"],
                                   dt=task_cfg["sim"]["dt"])
    os_ctrl   = OperationalSpaceController(
                    ctrl_cfg["operational_space_controller"],
                    dt=task_cfg["sim"]["dt"])

    perception_mod = None
    if use_perc:
        perception_cfg = load_cfg("perception")
        # Inject base offset (bias) into the custom hole level; vision supplies
        # its own error so no extra Gaussian sigma is added.
        custom = dict(perception_cfg["noise"]["hole"]["custom"])
        custom["bias"] = [offset_mm * 1e-3, 0.0, 0.0]
        custom["pos_sigma"] = 0.0
        perception_cfg["noise"] = dict(perception_cfg["noise"])
        perception_cfg["noise"]["hole"] = dict(perception_cfg["noise"]["hole"])
        perception_cfg["noise"]["hole"]["custom"] = custom
        perception_mod = PerceptionModule(
            env, perception_cfg,
            noise_level="custom", seed=seed,
            backend="rgbd_pointcloud", scene_cfg=scene_cfg)

    task = MultiTaskAssemblyTask(
        env, sensor, estimator, planner_o,
        pos_ctrl, task_cfg, ctrl_cfg,
        os_ctrl=os_ctrl,
        task_sequence=task_sequence if not use_perc else None,
        perception=perception_mod,
    )
    return task


def run_episode(task_sequence: list, **kw) -> dict:
    task = build_episode(task_sequence, **kw)
    return task.run_episode(max_steps_per_task=20000)


def print_per_task(result: dict) -> None:
    tr = result["task_results"]
    header = (f"\n  {'Task':<14} {'Succ':>5} {'Depth':>7} {'PkF':>7} "
              f"{'Rec':>5} {'Jam':>4} {'PIerr':>8} {'FinErr':>8} {'Time':>7}  Reason")
    print(header)
    print("  " + "-" * 90)
    for r in tr:
        label = f"{r['peg_name']}→{r['hole_name']}"
        line = (f"  {label:<14} {str(r['success']):>5} "
                f"{r['insertion_depth']*1000:>6.1f}mm "
                f"{r['peak_contact_force']:>6.1f}N "
                f"{r['recovery_attempts']:>5d} "
                f"{r.get('jam_events', 0):>4d} "
                f"{r['preinsert_error_m']*1000:>6.2f}mm "
                f"{r.get('final_pose_error_m', 0.0)*1000:>6.2f}mm "
                f"{r['task_time']:>6.1f}s  "
                f"{r.get('failure_reason', ''):<}")
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-task assembly episode runner")
    parser.add_argument("--tasks", nargs="+",
                        default=["round", "square", "rect"],
                        choices=["round", "square", "rect"],
                        help="Sub-tasks to run (in sequence order)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--perception", default="gt", choices=["gt", "rgbd"],
                        help="Pose source: noisy ground truth or RGB-D vision")
    parser.add_argument("--offset-mm", type=float, default=0.0,
                        help="Lateral hole misalignment (mm, X axis)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-task breakdown for each seed")
    args = parser.parse_args()

    task_sequence = resolve_task_sequence(args.tasks)
    n_tasks = len(task_sequence)

    print(f"\nMulti-Task Assembly  (mainline: OCP + LCS-MPC + OSC-λ)")
    print(f"  Tasks      : {' → '.join(args.tasks)}")
    print(f"  Perception : {args.perception}")
    print(f"  Offset     : {args.offset_mm:.1f} mm")
    print(f"  Seeds      : {args.seeds}")

    all_results = []
    wall0 = time.time()

    for seed in range(args.seeds):
        t0 = time.time()
        result = run_episode(
            task_sequence,
            perception=args.perception,
            offset_mm=args.offset_mm,
            seed=seed)
        elapsed = time.time() - t0

        completed = result["num_tasks_completed"]
        total_ok  = result["total_success"]
        rec_total = result["total_recovery_attempts"]
        pkf       = result["max_peak_force"]
        ep_t      = result["total_time"]

        status = "ALL OK" if total_ok else f"{completed}/{n_tasks}"
        print(f"  seed={seed:2d}  {status:<8}  completed={completed}/{n_tasks}"
              f"  rec={rec_total}  pkF={pkf:.1f}N  sim={ep_t:.1f}s  wall={elapsed:.1f}s")

        if args.verbose:
            print_per_task(result)

        all_results.append(result)

    # ── aggregate summary ──────────────────────────────────────────────────
    wall = time.time() - wall0
    ep_success_rate = np.mean([r["total_success"] for r in all_results])
    task_success_rates = []
    for ti in range(n_tasks):
        rates = []
        for r in all_results:
            if ti < len(r["task_results"]):
                rates.append(r["task_results"][ti]["success"])
            else:
                rates.append(False)
        task_success_rates.append(np.mean(rates))

    mean_completed = np.mean([r["num_tasks_completed"] for r in all_results])
    mean_rec       = np.mean([r["total_recovery_attempts"] for r in all_results])
    mean_pkf       = np.mean([r["max_peak_force"] for r in all_results])
    mean_time      = np.mean([r["total_time"] for r in all_results])

    print(f"\n{'='*60}")
    print(f"Episode success rate : {ep_success_rate:.1%}  ({args.seeds} seeds)")
    print(f"Mean tasks completed : {mean_completed:.2f} / {n_tasks}")
    for ti, (peg, hole) in enumerate(task_sequence):
        label = f"  {peg}→{hole}"
        print(f"  {label:<28}  {task_success_rates[ti]:.1%}")
    print(f"Mean recovery/ep     : {mean_rec:.2f}")
    print(f"Mean peak force      : {mean_pkf:.1f} N")
    print(f"Mean sim time/ep     : {mean_time:.1f} s")
    print(f"Total wall time      : {wall:.0f} s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
