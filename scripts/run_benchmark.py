#!/usr/bin/env python3
"""Orthogonal-axis sweep harness for the peg-in-hole benchmark.

Sweeps one or more experimental axes over lateral offsets and seeds and writes a
single tidy CSV (one row per trial).  Holding three axes fixed while sweeping the
fourth isolates that layer's effect — e.g. Round 1 sweeps the controller axis
with planner=waypoint, contact=none, perception=gt.

Round 1 (default) — controller comparison
-----------------------------------------
  python scripts/run_benchmark.py

  → controller ∈ {jointpos, impedance, osc, osc-lambda}
    planner=waypoint, contact=none, perception=gt
    offsets ∈ {0,3,6,9} mm, seeds 0..9

Other sweeps
------------
  python scripts/run_benchmark.py --controllers osc-lambda \
         --contacts none spiral lcs-mpc --offsets 0 5 10 --seeds 20
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import experiment_axes as ax
from run_assembly import run_episode, resolve_task_sequence


CSV_FIELDS = [
    "trial_id", "seed", "controller", "planner", "contact", "perception",
    "offset_mm", "spiral_radius_mm", "noise_sigma_mm", "success", "failure_reason",
    "insertion_time", "peak_force", "mean_force", "jamming_count",
    "retry_count", "final_pose_error", "compute_time",
    # Planner-axis (approach-phase) diagnostics
    "preinsert_pose_error", "path_length", "smoothness",
    "max_joint_vel", "max_joint_accel", "solve_time", "initial_contact_force",
    # Perception-axis diagnostics
    "pose_estimation_error", "effective_offset",
]


def run_trial(trial_id, *, controller, planner, contact, perception,
              offset_mm, seed, slsqp_freq, task_sequence,
              spiral_radius_mm=None, noise_sigma_mm=None):
    """Run a single trial and return one CSV row dict."""
    t0 = time.time()
    result = run_episode(
        task_sequence,
        controller=controller, planner=planner, contact=contact,
        perception=perception, offset_mm=offset_mm, noise_level="easy",
        seed=seed, slsqp_freq=slsqp_freq, spiral_radius_mm=spiral_radius_mm,
        noise_sigma_mm=noise_sigma_mm)
    compute_time = time.time() - t0

    # Single-task sweep → read the first sub-task result.
    r = result["task_results"][0] if result["task_results"] else {}
    # Perception diagnostics: pose_estimation_error = realized hole estimate
    # error (perception on); effective_offset = displacement of the planner's
    # aim point from the true hole (= that error, or the commanded bias for gt).
    pose_est_err = float(r.get("hole_pos_error_m", 0.0))
    eff_offset = pose_est_err if perception != "gt" else offset_mm * 1e-3
    return {
        "trial_id":         trial_id,
        "seed":             seed,
        "controller":       controller,
        "planner":          planner,
        "contact":          contact,
        "perception":       perception,
        "offset_mm":        offset_mm,
        "spiral_radius_mm": ("" if spiral_radius_mm is None
                             else spiral_radius_mm),
        "noise_sigma_mm":   ("" if noise_sigma_mm is None
                             else noise_sigma_mm),
        "success":          int(bool(r.get("success", False))),
        "failure_reason":   r.get("failure_reason", "incomplete"),
        "insertion_time":   round(float(r.get("task_time", 0.0)), 4),
        "peak_force":       round(float(r.get("peak_contact_force", 0.0)), 4),
        "mean_force":       round(float(r.get("avg_contact_force", 0.0)), 4),
        "jamming_count":    int(r.get("jam_events", 0)),
        "retry_count":      int(r.get("recovery_attempts", 0)),
        "final_pose_error": round(float(r.get("final_pose_error_m", 0.0)), 6),
        "compute_time":     round(compute_time, 4),
        "preinsert_pose_error": round(float(r.get("preinsert_error_m", 0.0)), 6),
        "path_length":          round(float(r.get("approach_path_len_m", 0.0)), 6),
        "smoothness":           round(float(r.get("approach_smoothness", 0.0)), 4),
        "max_joint_vel":        round(float(r.get("max_joint_vel", 0.0)), 4),
        "max_joint_accel":      round(float(r.get("max_joint_accel", 0.0)), 4),
        "solve_time":           round(float(r.get("ocp_solve_ms", 0.0)), 4),
        "initial_contact_force": round(float(r.get("initial_contact_force", 0.0)), 4),
        "pose_estimation_error": round(pose_est_err, 6),
        "effective_offset":      round(eff_offset, 6),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Orthogonal-axis benchmark sweep")
    p.add_argument("--controllers", nargs="+", default=ax.CONTROLLERS,
                   choices=ax.CONTROLLERS)
    p.add_argument("--planners", nargs="+", default=["waypoint"],
                   choices=ax.PLANNERS)
    p.add_argument("--contacts", nargs="+", default=["none"],
                   choices=ax.CONTACTS)
    p.add_argument("--perceptions", nargs="+", default=["gt"],
                   choices=ax.PERCEPTIONS)
    p.add_argument("--offsets", nargs="+", type=float, default=[0, 3, 6, 9],
                   help="Lateral offsets in mm")
    p.add_argument("--spiral-radii", nargs="+", type=float, default=[None],
                   help="Spiral search max radii in mm to sweep "
                        "(default: config 6mm). Only affects spiral/lcs-mpc.")
    p.add_argument("--noise-sigmas", nargs="+", type=float, default=[None],
                   help="Hole perception noise sigmas in mm to sweep "
                        "(default: gt-noise's 1mm). Only affects gt-noise.")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--task", default="round", choices=["round", "square", "rect"],
                   help="Single sub-task to benchmark")
    p.add_argument("--slsqp-freq", type=int, default=25)
    p.add_argument("--out", default="results/round1_controllers.csv",
                   help="Output CSV path")
    args = p.parse_args()

    task_sequence = resolve_task_sequence([args.task])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).parents[1] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Noise sigma only varies gt-noise; other perceptions run once (sigma=None).
    def sigmas_for(perception):
        return args.noise_sigmas if perception == "gt-noise" else [None]

    shared = (len(args.controllers) * len(args.planners) * len(args.contacts)
              * len(args.offsets) * len(args.spiral_radii) * args.seeds)
    n_total = shared * sum(len(sigmas_for(p)) for p in args.perceptions)
    print(f"Benchmark sweep — {n_total} trials → {out_path}")
    print(f"  controllers  : {args.controllers}")
    print(f"  planners     : {args.planners}")
    print(f"  contacts     : {args.contacts}")
    print(f"  perceptions  : {args.perceptions}")
    print(f"  offsets(mm)  : {args.offsets}")
    print(f"  spiral_r(mm) : {args.spiral_radii}")
    print(f"  noise_σ(mm)  : {args.noise_sigmas}")
    print(f"  seeds        : {args.seeds}  (task={args.task})")
    print()

    rows = []
    trial_id = 0
    wall0 = time.time()
    # Stream rows to disk as we go so a long sweep survives interruption.
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for controller in args.controllers:
            for planner in args.planners:
                for contact in args.contacts:
                    for perception in args.perceptions:
                      for offset_mm in args.offsets:
                        for spiral_r in args.spiral_radii:
                          for noise_sigma in sigmas_for(perception):
                            for seed in range(args.seeds):
                                row = run_trial(
                                    trial_id, controller=controller,
                                    planner=planner, contact=contact,
                                    perception=perception, offset_mm=offset_mm,
                                    seed=seed, slsqp_freq=args.slsqp_freq,
                                    task_sequence=task_sequence,
                                    spiral_radius_mm=spiral_r,
                                    noise_sigma_mm=noise_sigma)
                                writer.writerow(row)
                                fh.flush()
                                rows.append(row)
                                trial_id += 1
                                ok = "OK " if row["success"] else "FAIL"
                                stag = "" if noise_sigma is None else f" σ={noise_sigma:.0f}mm"
                                print(f"  [{trial_id:>4}/{n_total}] {perception:<9}"
                                      f" off={offset_mm:>4.1f}mm{stag} seed={seed:<2}"
                                      f" {ok} pkF={row['peak_force']:>5.1f}N"
                                      f" pe={row['pose_estimation_error']*1000:>5.2f}mm"
                                      f" fin={row['final_pose_error']*1000:>5.2f}mm"
                                      f" [{row['failure_reason']}]")

    wall = time.time() - wall0
    print(f"\nWrote {len(rows)} rows to {out_path}  ({wall:.0f}s)")
    _summary(rows, args)


def _summary(rows, args) -> None:
    """Print success-rate and mean-peak-force tables (controller × offset)."""
    offsets = sorted({r["offset_mm"] for r in rows})
    controllers = [c for c in ax.CONTROLLERS if c in {r["controller"] for r in rows}]

    def cell(ctrl, off, key, agg):
        vals = [r[key] for r in rows
                if r["controller"] == ctrl and r["offset_mm"] == off]
        return agg(vals) if vals else float("nan")

    print("\nSuccess rate (controller × offset_mm)")
    hdr = "  " + f"{'controller':<12}" + "".join(f"{o:>8.0f}" for o in offsets)
    print(hdr); print("  " + "-" * (12 + 8 * len(offsets)))
    for ctrl in controllers:
        line = f"  {ctrl:<12}"
        for off in offsets:
            line += f"{cell(ctrl, off, 'success', np.mean):>8.0%}"
        print(line)

    print("\nMean peak force [N] (controller × offset_mm)")
    print(hdr); print("  " + "-" * (12 + 8 * len(offsets)))
    for ctrl in controllers:
        line = f"  {ctrl:<12}"
        for off in offsets:
            line += f"{cell(ctrl, off, 'peak_force', np.mean):>8.1f}"
        print(line)


if __name__ == "__main__":
    main()
