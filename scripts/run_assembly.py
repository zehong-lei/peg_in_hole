#!/usr/bin/env python3
"""Multi-task scripted assembly episode runner.

Task sequence:  round_peg → square_peg → rect_peg

The experiment is configured through four ORTHOGONAL axes; each one maps to a
single config subtree and never implicitly toggles another layer:

  --controller {jointpos, impedance, osc, osc-lambda}
  --planner    {waypoint, ee-ocp, joint-ocp}
  --contact    {none, spiral, force-guided, lcs-mpc}
  --perception {gt, gt-noise, ema, rgbd}

Usage examples
--------------
  python scripts/run_assembly.py
  python scripts/run_assembly.py --tasks round
  python scripts/run_assembly.py --controller impedance --contact spiral
  python scripts/run_assembly.py --controller osc-lambda --planner ee-ocp \
                                 --contact lcs-mpc --seeds 5
  python scripts/run_assembly.py --verbose
"""

import argparse
import copy
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
from src.controllers.impedance_controller import ImpedanceController
from src.controllers.operational_space_controller import OperationalSpaceController
from src.planners.scripted_planner import ScriptedPlanner
from src.tasks.multi_task_assembly import MultiTaskAssemblyTask
from src.perception import PerceptionModule

import experiment_axes as ax


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


def build_episode(task_sequence: list, *, controller: str, planner: str,
                  contact: str, perception: str, offset_mm: float,
                  noise_level: str, seed: int, slsqp_freq: int,
                  spiral_radius_mm: float | None = None,
                  noise_sigma_mm: float | None = None):
    """Instantiate all modules for one episode using the four orthogonal axes."""
    scene_cfg = load_cfg("scene")
    task_cfg  = load_cfg("task")
    ctrl_cfg  = load_cfg("controller")
    noise_cfg = load_cfg("noise")

    # ── apply the four axes (each mutates only its own subtree) ───────────────
    controller_kind, needs_os = ax.apply_controller(ctrl_cfg, controller)
    ax.apply_planner(task_cfg, planner)
    ax.apply_contact(task_cfg, contact, slsqp_freq=slsqp_freq,
                     spiral_radius_mm=spiral_radius_mm)
    pspec = ax.perception_spec(perception, noise_sigma_mm=noise_sigma_mm)

    # ── multi-task episode timeouts (runner tuning, not an axis) ──────────────
    to = list(task_cfg["stages"]["stage_timeout"])
    to[3] = 15.0     # MOVE_TO_PREINSERT — OCP may need longer
    to[5] = 150.0    # INSERT
    task_cfg["stages"]["stage_timeout"] = to
    task_cfg["contact_recovery"]["max_attempts"] = 10

    # Lateral offset → deterministic hole bias (the controlled misalignment).
    # When perception is on, the bias goes into the perception custom level
    # (the sensor's hole estimate is overridden by PerceptionModule each task);
    # when it is `gt`, it goes onto the sensor directly.
    use_perc  = pspec["use_perception"]
    hole_bias = ([offset_mm * 1e-3, 0.0, 0.0]
                 if (offset_mm and not use_perc) else None)

    env       = AssemblyEnv(scene_cfg, task_cfg, seed=seed)
    sensor    = SensorWrapper(env, noise_cfg, noise_level=pspec["noise_level"],
                              seed=seed, hole_pos_bias=hole_bias,
                              hole_pos_sigma=pspec["hole_pos_sigma"])
    estimator = StateEstimator(noise_cfg)
    planner_o = ScriptedPlanner(task_cfg, ctrl_cfg)
    pos_ctrl  = PositionController(ctrl_cfg["position_controller"],
                                   dt=task_cfg["sim"]["dt"])
    imp_ctrl  = ImpedanceController(ctrl_cfg["impedance_controller"],
                                    dt=task_cfg["sim"]["dt"])
    os_ctrl   = (OperationalSpaceController(
                     ctrl_cfg["operational_space_controller"],
                     dt=task_cfg["sim"]["dt"])
                 if needs_os else None)

    perception_mod = None
    if use_perc:
        perception_cfg = load_cfg("perception")
        # Inject base offset (bias) + controlled hole sigma into the custom level.
        custom = dict(perception_cfg["noise"]["hole"]["custom"])
        custom["bias"] = [offset_mm * 1e-3, 0.0, 0.0]
        custom["pos_sigma"] = float(pspec["custom_hole_sigma_m"])
        perception_cfg["noise"] = dict(perception_cfg["noise"])
        perception_cfg["noise"]["hole"] = dict(perception_cfg["noise"]["hole"])
        perception_cfg["noise"]["hole"]["custom"] = custom
        perception_mod = PerceptionModule(
            env, perception_cfg,
            noise_level=pspec["perception_level"], seed=seed,
            backend=pspec["backend"], scene_cfg=scene_cfg)

    task = MultiTaskAssemblyTask(
        env, sensor, estimator, planner_o,
        pos_ctrl, imp_ctrl, task_cfg, ctrl_cfg,
        os_ctrl=os_ctrl,
        task_sequence=task_sequence if not pspec["use_perception"] else None,
        perception=perception_mod,
        controller_kind=controller_kind,
    )
    return task


def run_episode(task_sequence: list, **kw) -> dict:
    task = build_episode(task_sequence, **kw)
    return task.run_episode(max_steps_per_task=20000)


def print_per_task(result: dict) -> None:
    tr = result["task_results"]
    has_perception = any("hole_pos_error_m" in r for r in tr)
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
    parser.add_argument("--controller", default="osc-lambda", choices=ax.CONTROLLERS)
    parser.add_argument("--planner",    default="ee-ocp",     choices=ax.PLANNERS)
    parser.add_argument("--contact",    default="lcs-mpc",    choices=ax.CONTACTS)
    parser.add_argument("--perception", default="gt",         choices=ax.PERCEPTIONS)
    parser.add_argument("--offset-mm", type=float, default=0.0,
                        help="Lateral hole misalignment (mm, X axis)")
    parser.add_argument("--spiral-radius-mm", type=float, default=None,
                        help="Spiral search max radius (mm); default = config (6mm)")
    parser.add_argument("--noise-sigma-mm", type=float, default=None,
                        help="Hole perception noise sigma (mm) for gt-noise")
    parser.add_argument("--slsqp-freq", type=int, default=25)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-task breakdown for each seed")
    args = parser.parse_args()

    task_sequence = resolve_task_sequence(args.tasks)
    n_tasks = len(task_sequence)

    print(f"\nMulti-Task Assembly")
    print(f"  Tasks      : {' → '.join(args.tasks)}")
    print(f"  Controller : {args.controller}")
    print(f"  Planner    : {args.planner}")
    print(f"  Contact    : {args.contact}")
    print(f"  Perception : {args.perception}")
    print(f"  Offset     : {args.offset_mm:.1f} mm")
    print(f"  Seeds      : {args.seeds}")

    all_results = []
    wall0 = time.time()

    for seed in range(args.seeds):
        t0 = time.time()
        result = run_episode(
            task_sequence,
            controller=args.controller, planner=args.planner,
            contact=args.contact, perception=args.perception,
            offset_mm=args.offset_mm, noise_level="easy",
            seed=seed, slsqp_freq=args.slsqp_freq,
            spiral_radius_mm=args.spiral_radius_mm,
            noise_sigma_mm=args.noise_sigma_mm)
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
