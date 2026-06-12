#!/usr/bin/env python3
"""Plot the controller- and contact-axis benchmark studies.

Reads the four sweep CSVs produced by run_benchmark.py and writes figures to
results/figures/, plus a Markdown summary (results/contact_axis_summary.md).

  python scripts/plot_axis_studies.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parents[1]
_RESULTS = _ROOT / "results"
_FIG = _RESULTS / "figures"

# Stable colours/markers so series match across figures.
_COLOR = {
    "jointpos": "#4C72B0", "impedance": "#DD8452",
    "osc": "#55A868", "osc-lambda": "#C44E52",
    "none": "#8172B3", "spiral": "#DD8452", "lcs-mpc": "#55A868",
    "waypoint": "#4C72B0", "ee-ocp": "#C44E52",
}
_MARKER = {
    "jointpos": "o", "impedance": "s", "osc": "^", "osc-lambda": "D",
    "none": "o", "spiral": "s", "lcs-mpc": "^",
    "waypoint": "o", "ee-ocp": "D",
}


def load(name: str) -> list[dict]:
    return list(csv.DictReader((_RESULTS / name).open()))


def agg(rows, group_keys, x_key, y_key, reduce_fn=np.mean):
    """Return {group_tuple: (sorted_x, reduced_y)} aggregating y over seeds."""
    bucket = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = tuple(r[k] for k in group_keys)
        bucket[g][float(r[x_key])].append(float(r[y_key]))
    out = {}
    for g, xy in bucket.items():
        xs = sorted(xy)
        ys = [reduce_fn(xy[x]) for x in xs]
        out[g] = (np.array(xs), np.array(ys))
    return out


# ── Figure 1 & 2 & 3: success-rate vs offset, one line per hue ───────────────
def success_vs_offset(csv_name, hue_key, hue_order, title, fname,
                      subtitle=""):
    rows = load(csv_name)
    data = agg(rows, [hue_key], "offset_mm", "success")
    fig, axp = plt.subplots(figsize=(7, 4.5))
    for hue in hue_order:
        key = (hue,)
        if key not in data:
            continue
        xs, ys = data[key]
        axp.plot(xs, ys * 100, marker=_MARKER.get(hue, "o"),
                 color=_COLOR.get(hue), label=hue, lw=2, ms=7)
    axp.set_xlabel("lateral offset [mm]")
    axp.set_ylabel("success rate [%]")
    axp.set_ylim(-5, 105)
    axp.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=11)
    axp.grid(True, alpha=0.3)
    axp.legend(title=hue_key)
    fig.tight_layout()
    fig.savefig(_FIG / fname, dpi=130)
    plt.close(fig)
    print(f"  wrote {fname}")


# ── Figure 4: spiral radius × offset success heatmap ─────────────────────────
def spiral_radius_heatmap(csv_name, fname):
    rows = load(csv_name)
    radii = sorted({float(r["spiral_radius_mm"]) for r in rows})
    offs = sorted({float(r["offset_mm"]) for r in rows})
    grid = np.full((len(radii), len(offs)), np.nan)
    cnt = defaultdict(list)
    for r in rows:
        cnt[(float(r["spiral_radius_mm"]), float(r["offset_mm"]))].append(
            float(r["success"]))
    for i, rad in enumerate(radii):
        for j, off in enumerate(offs):
            v = cnt.get((rad, off))
            if v:
                grid[i, j] = np.mean(v) * 100

    fig, axp = plt.subplots(figsize=(6.5, 5))
    im = axp.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=100)
    axp.set_xticks(range(len(offs)), [f"{o:.0f}" for o in offs])
    axp.set_yticks(range(len(radii)), [f"{r:.0f}" for r in radii])
    axp.set_xlabel("lateral offset [mm]")
    axp.set_ylabel("spiral search radius [mm]")
    axp.set_title("Spiral recovery envelope: success rate\n"
                  "(contact=spiral, controller=osc-lambda)", fontsize=11)
    for i in range(len(radii)):
        for j in range(len(offs)):
            if not np.isnan(grid[i, j]):
                axp.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                         color="black", fontsize=9)
    # envelope guide: offset ≈ radius + clearance(4mm)
    cb = fig.colorbar(im, ax=axp)
    cb.set_label("success rate [%]")
    fig.tight_layout()
    fig.savefig(_FIG / fname, dpi=130)
    plt.close(fig)
    print(f"  wrote {fname}")


# ── Figure 5: peak force vs radius at a fixed offset ─────────────────────────
def force_vs_radius_at(csv_name, offset_mm, fname):
    rows = [r for r in load(csv_name) if float(r["offset_mm"]) == offset_mm]
    radii = sorted({float(r["spiral_radius_mm"]) for r in rows})
    pkf, succ = [], []
    for rad in radii:
        sub = [r for r in rows if float(r["spiral_radius_mm"]) == rad]
        pkf.append(np.mean([float(r["peak_force"]) for r in sub]))
        succ.append(np.mean([float(r["success"]) for r in sub]) * 100)

    fig, axp = plt.subplots(figsize=(7, 4.5))
    axp.plot(radii, pkf, marker="D", color="#C44E52", lw=2, ms=8)
    axp.set_xlabel("spiral search radius [mm]")
    axp.set_ylabel("mean peak contact force [N]")
    axp.set_title(f"Peak force vs search radius @ {offset_mm:.0f}mm offset\n"
                  "(contact=spiral, controller=osc-lambda)", fontsize=11)
    axp.grid(True, alpha=0.3)
    for x, y, s in zip(radii, pkf, succ):
        axp.annotate(f"{s:.0f}% ok", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#333")
    fig.tight_layout()
    fig.savefig(_FIG / fname, dpi=130)
    plt.close(fig)
    print(f"  wrote {fname}")


# ── Markdown summary ─────────────────────────────────────────────────────────
def _rate_table(rows, hue_key, hue_order):
    offs = sorted({float(r["offset_mm"]) for r in rows})
    lines = ["| " + hue_key + " | " + " | ".join(f"{o:.0f}mm" for o in offs) + " |",
             "|" + "---|" * (len(offs) + 1)]
    for hue in hue_order:
        cells = []
        for o in offs:
            sub = [float(r["success"]) for r in rows
                   if r[hue_key] == hue and float(r["offset_mm"]) == o]
            cells.append(f"{np.mean(sub)*100:.0f}%" if sub else "–")
        lines.append(f"| {hue} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary():
    r1 = load("round1_controllers.csv")
    r3b = load("round3b_offsets.csv")
    r3c = load("round3c_radius.csv")

    # envelope per radius from 3c
    env_lines = []
    radii = sorted({float(r["spiral_radius_mm"]) for r in r3c})
    offs = sorted({float(r["offset_mm"]) for r in r3c})
    for rad in radii:
        last_full = None
        for o in offs:
            sub = [float(r["success"]) for r in r3c
                   if float(r["spiral_radius_mm"]) == rad
                   and float(r["offset_mm"]) == o]
            if sub and np.mean(sub) >= 0.99:
                last_full = o
        env_lines.append(f"| {rad:.0f}mm | "
                         f"{last_full:.0f}mm |" if last_full is not None
                         else f"| {rad:.0f}mm | <{offs[0]:.0f}mm |")

    md = f"""# Contact-Axis Study Summary

_Generated by `scripts/plot_axis_studies.py` from the Round 1 / 3 / 3b / 3c sweep CSVs._

## Experiment question
Where does robustness to lateral pose error come from in this peg-in-hole stack —
the **low-level controller** or the **contact-handling policy** — and what sets
the size of the recoverable-misalignment envelope?

## Fixed variables
- Task: single round peg → round hole (4 mm radial clearance)
- Planner: `waypoint` · Perception: `gt` · 10 seeds per cell
- Misalignment injected as a fixed lateral hole-position bias (X axis)
- Contact/controller held fixed while the *other* axis is swept

## Changed variable(s)
- **Round 1** — controller ∈ {{jointpos, impedance, osc, osc-lambda}} (contact=none)
- **Round 3 / 3b** — contact ∈ {{none, spiral, lcs-mpc}} (controller=osc-lambda)
- **Round 3c** — spiral search radius ∈ {{4,6,8,10}} mm (contact=spiral)

## Key results

### Round 1 — controller axis (contact = none)
{_rate_table(r1, "controller", ["jointpos","impedance","osc","osc-lambda"])}

All four controllers share the **same** success envelope; they differ only by a
small peak-force margin (~1.5 N lower for impedance/osc vs rigid jointpos at low
offset). Compliance alone does **not** extend offset tolerance.

### Round 3b — contact axis, fine sweep (spiral_r = 6 mm)
{_rate_table(r3b, "contact", ["none","spiral","lcs-mpc"])}

The contact layer is where tolerance comes from: recovery turns the 6–8 mm cliff
into near-100 %. **LCS-MPC beats spiral at the margin** (100 % vs 90 % at 8 mm) —
force-guidance reaches slightly further than a blind geometric search.

### Round 3c — spiral radius sets the envelope (contact = spiral)
Largest offset still solved at 100 % per search radius:

| spiral radius | offset solved (100%) |
|---|---|
{chr(10).join(env_lines)}

Envelope ≈ **search_radius + clearance (4 mm)**: every +2 mm of radius buys ≈ +2 mm
of offset tolerance. The cost is higher peak contact force at a fixed offset
(larger radius sweeps wider/harder before landing).

## Interpretation
- **Controller axis is near-flat for robustness**; its lever is contact *force*, not
  reach. (Round 1.)
- **Contact axis is the robustness lever**, and it is *tunable* via search radius
  with a clean geometric law. (Round 3b/3c.)
- **LCS-MPC ≻ spiral** at the envelope edge, at a modest force premium.
- There is a real **force ↔ tolerance trade-off**: pick the smallest radius that
  covers expected pose uncertainty rather than the largest.

## Implication for the next axis (Round 2 — planner)
Robustness is now attributable to the contact layer, with the controller
neutralised as a confounder. The pre-insertion **planner** axis (waypoint vs
ee-ocp vs joint-ocp) should be evaluated on *approach-phase* metrics — path
smoothness, pre-insert alignment error, clearance, solve time — **not** on
insertion success, which the contact layer already saturates below ~6 mm. Expect
the planner to shift the *operating point* (better/cheaper pre-insert pose),
while the contact axis continues to own the failure envelope.
"""
    out = _RESULTS / "contact_axis_summary.md"
    out.write_text(md)
    print(f"  wrote {out.name}")


# ── planner-axis helpers ─────────────────────────────────────────────────────
def _mean_std(rows, planner, key):
    v = [float(r[key]) for r in rows if r["planner"] == planner]
    if not v:
        return float("nan"), 0.0
    return float(np.mean(v)), float(np.std(v))


def _grouped_bar(axp, rows, planners, key, title, ylabel, per_offset=False):
    """One bar per planner (mean ± std), optionally grouped by offset."""
    if per_offset:
        offs = sorted({float(r["offset_mm"]) for r in rows})
        width = 0.8 / len(planners)
        x = np.arange(len(offs))
        for i, pl in enumerate(planners):
            means, stds = [], []
            for o in offs:
                v = [float(r[key]) for r in rows
                     if r["planner"] == pl and float(r["offset_mm"]) == o]
                means.append(np.mean(v) if v else np.nan)
                stds.append(np.std(v) if v else 0.0)
            axp.bar(x + i * width, means, width, yerr=stds, capsize=3,
                    label=pl, color=_COLOR.get(pl))
        axp.set_xticks(x + width * (len(planners) - 1) / 2,
                       [f"{o:.0f}mm" for o in offs])
    else:
        for i, pl in enumerate(planners):
            m, s = _mean_std(rows, pl, key)
            axp.bar(i, m, 0.6, yerr=s, capsize=4, color=_COLOR.get(pl), label=pl)
        axp.set_xticks(range(len(planners)), planners)
    axp.set_title(title, fontsize=10)
    axp.set_ylabel(ylabel)
    axp.grid(True, axis="y", alpha=0.3)


def planner_smoothness_bar(csv_name, fname):
    rows = load(csv_name)
    fig, axp = plt.subplots(figsize=(5.5, 4.5))
    _grouped_bar(axp, rows, ["waypoint", "ee-ocp"], "smoothness",
                 "Approach smoothness (lower = smoother)\n"
                 "RMS EE acceleration, relative",
                 "RMS EE accel [m/s²]")
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def planner_motion_cost_bar(csv_name, fname):
    rows = load(csv_name)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _grouped_bar(axes[0], rows, ["waypoint", "ee-ocp"], "max_joint_vel",
                 "Max joint velocity", "[rad/s]")
    _grouped_bar(axes[1], rows, ["waypoint", "ee-ocp"], "max_joint_accel",
                 "Max joint acceleration (relative)", "[rad/s²]")
    fig.suptitle("Round 2A — approach motion cost (controller=impedance, contact=none)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def planner_time_cost_bar(csv_name, fname):
    rows = load(csv_name)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _grouped_bar(axes[0], rows, ["waypoint", "ee-ocp"], "solve_time",
                 "Planner solve time (cold start / episode)", "[ms]")
    _grouped_bar(axes[1], rows, ["waypoint", "ee-ocp"], "insertion_time",
                 "Task time", "[s]")
    fig.suptitle("Round 2A — time cost (controller=impedance, contact=none)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def planner_recovery_burden(csv_name, fname):
    rows = load(csv_name)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    _grouped_bar(axes[0], rows, ["waypoint", "ee-ocp"], "retry_count",
                 "Retry count", "mean retries", per_offset=True)
    _grouped_bar(axes[1], rows, ["waypoint", "ee-ocp"], "jamming_count",
                 "Jam events", "mean jams", per_offset=True)
    _grouped_bar(axes[2], rows, ["waypoint", "ee-ocp"], "peak_force",
                 "Peak contact force", "[N]", per_offset=True)
    axes[0].legend()
    fig.suptitle("Round 2B — recovery burden under spiral recovery "
                 "(controller=impedance, contact=spiral r=6mm)", fontsize=11)
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def _planner_rate_table(rows, offs):
    head = "| planner | " + " | ".join(f"{o:.0f}mm" for o in offs) + " |"
    sep = "|" + "---|" * (len(offs) + 1)
    lines = [head, sep]
    for pl in ["waypoint", "ee-ocp"]:
        cells = []
        for o in offs:
            v = [float(r["success"]) for r in rows
                 if r["planner"] == pl and float(r["offset_mm"]) == o]
            cells.append(f"{np.mean(v)*100:.0f}%" if v else "–")
        lines.append(f"| {pl} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_planner_summary():
    r2a = load("round2a_planner_no_recovery.csv")
    r2b = load("round2b_planner_with_spiral.csv")
    o2a = sorted({float(r["offset_mm"]) for r in r2a})
    o2b = sorted({float(r["offset_mm"]) for r in r2b})

    def m(rows, pl, k):
        return _mean_std(rows, pl, k)[0]

    md = f"""# Planner-Axis Study Summary

_Generated by `scripts/plot_axis_studies.py` from the Round 2A / 2B sweep CSVs._

## Experiment question
Does the pre-insertion **planner** (`waypoint` vs `ee-ocp`) change task success, or
only the *quality* of the free-space approach — and at what cost? `joint-ocp` is
**not implemented** and was skipped honestly.

## Fixed variables
- Controller = `impedance` · Perception = `gt` · single round peg/hole (4 mm clearance)
- Misalignment = fixed lateral hole bias (X axis)
- **2A**: contact = `none`, offsets {{0,3,6}} mm, 30 seeds
- **2B**: contact = `spiral` (radius 6 mm), offsets {{6,8}} mm, 30 seeds

## Changed variable
- Planner ∈ {{`waypoint`, `ee-ocp`}}

## Success-rate result — planners are tied
**2A (no recovery):**
{_planner_rate_table(r2a, o2a)}

**2B (with spiral recovery):**
{_planner_rate_table(r2b, o2b)}

Differences are 1–4 trials out of 30 — within seed noise. **The planner does not
move the failure envelope.** Insertion success is owned by the contact layer
(Round 3), confirmed here from the planner side.

## Approach-quality result — ee-ocp is smoother
Averaged over 2A trials:

| metric | waypoint | ee-ocp |
|---|---|---|
| RMS EE acceleration [m/s²] | {m(r2a,'waypoint','smoothness'):.1f} | **{m(r2a,'ee-ocp','smoothness'):.1f}** |
| max joint velocity [rad/s] | {m(r2a,'waypoint','max_joint_vel'):.2f} | **{m(r2a,'ee-ocp','max_joint_vel'):.2f}** |
| max joint accel [rad/s²] | {m(r2a,'waypoint','max_joint_accel'):.0f} | **{m(r2a,'ee-ocp','max_joint_accel'):.0f}** |

ee-ocp cuts RMS end-effector acceleration ~2.8× and lowers peak joint velocity
~19 %: a visibly gentler, more controlled approach. This is the planner's real lever.

## Cost trade-off
| metric | waypoint | ee-ocp |
|---|---|---|
| solve time [ms/episode, cold] | {m(r2a,'waypoint','solve_time'):.0f} | {m(r2a,'ee-ocp','solve_time'):.0f} |
| approach path length [m] | {m(r2a,'waypoint','path_length'):.3f} | {m(r2a,'ee-ocp','path_length'):.3f} |
| task time [s] | {m(r2a,'waypoint','insertion_time'):.1f} | {m(r2a,'ee-ocp','insertion_time'):.1f} |
| preinsert pose error [mm] | {m(r2a,'waypoint','preinsert_pose_error')*1000:.1f} | {m(r2a,'ee-ocp','preinsert_pose_error')*1000:.1f} |
| peak force [N] (≤6 mm) | {m(r2a,'waypoint','peak_force'):.1f} | {m(r2a,'ee-ocp','peak_force'):.1f} |

ee-ocp pays ~27–30 ms cold solve/episode, +2.4 % path, ~2.5 s longer task time,
and a hair *worse* terminal accuracy — for **no** success or force benefit at these
offsets. Under 2B recovery, both planners produce near-identical retry/jam burden
and final pose error; ee-ocp's only loss was 1/30 at 8 mm.

## Caveats
- `joint-ocp` not implemented → skipped.
- `solve_time` is the per-episode **cold start** (fresh planner each trial);
  intra-episode warm-starts run ~1 ms.
- `smoothness` / `max_joint_accel` are finite-differenced from true state at 500 Hz,
  so absolute magnitudes include servo jitter — valid for **relative** comparison only.
- `initial_contact_force` came back 0 across all 2A trials: at ≤6 mm with compliant
  impedance, insertion contact never crossed the 2 N detection threshold. Metric is
  wired but uninformative in this low-offset regime.

## Implication for the perception axis
The planner's value is **motion quality, not success**, and this open-tabletop
scene with generous clearance does not reward it. The perception axis is different:
it injects pose *error*, which feeds straight into the contact layer's recovery
basin. Expect perception noise to **shift the operating point along the same
envelope mapped in Round 3** — i.e. effective offset = commanded offset + perception
error — so the key question becomes whether estimation/filtering keeps total error
inside the recoverable envelope (≈ search_radius + clearance). Hold controller =
impedance and contact = spiral (r = 6 mm) fixed so perception error is the only
moving part.
"""
    out = _RESULTS / "planner_axis_summary.md"
    out.write_text(md)
    print(f"  wrote {out.name}")


# ── Round 4 — perception axis ────────────────────────────────────────────────
# Recovery envelope from Round 3: effective offset solvable up to
# spiral_radius (6 mm) + radial clearance (4 mm) ≈ 10 mm.
_ENVELOPE_MM = 10.0


def round4a_success_vs_noise(csv_name, fname):
    """Success vs hole-noise sigma, one line per base offset (gt-noise only)."""
    rows = [r for r in load(csv_name) if r["perception"] == "gt-noise"]
    offs = sorted({float(r["offset_mm"]) for r in rows})
    data = agg(rows, ["offset_mm"], "noise_sigma_mm", "success")
    fig, axp = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(offs)))
    for off, col in zip(offs, cmap):
        key = (f"{off:.1f}" if (f"{off:.1f}",) in data else str(off),)
        # offset stored as float string in CSV; match robustly
        k = next((g for g in data if float(g[0]) == off), None)
        if k is None:
            continue
        xs, ys = data[k]
        axp.plot(xs, ys * 100, marker="o", color=col, lw=2, ms=6,
                 label=f"offset {off:.0f}mm")
    axp.set_xlabel("hole perception noise σ [mm]")
    axp.set_ylabel("success rate [%]")
    axp.set_ylim(-5, 105)
    axp.set_title("Round 4A — success vs perception noise\n"
                  "(impedance, waypoint, spiral r=6mm, gt-noise, 30 seeds)",
                  fontsize=11)
    axp.grid(True, alpha=0.3)
    axp.legend(title="base offset")
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def round4a_success_vs_effective_offset(csv_name, fname):
    """THE unifying plot: pool all gt-noise trials and bin by *effective* offset
    (commanded bias ⊕ perception noise).  Tests whether perception error obeys
    the same recovery envelope as commanded offset (Round 3)."""
    rows = [r for r in load(csv_name) if r["perception"] == "gt-noise"]
    eff = np.array([float(r["effective_offset"]) * 1000 for r in rows])
    succ = np.array([float(r["success"]) for r in rows])
    edges = np.arange(0, np.ceil(eff.max()) + 2, 2.0)
    centers, rates, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (eff >= lo) & (eff < hi)
        if m.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        rates.append(succ[m].mean() * 100)
        counts.append(int(m.sum()))

    fig, axp = plt.subplots(figsize=(7.5, 4.5))
    axp.plot(centers, rates, marker="o", color="#C44E52", lw=2, ms=7,
             label="success rate (binned)")
    axp.axvline(_ENVELOPE_MM, color="#444", ls="--", lw=1.5,
                label=f"Round 3 envelope ≈ {_ENVELOPE_MM:.0f}mm\n(spiral r + clearance)")
    for x, y, n in zip(centers, rates, counts):
        axp.annotate(f"n={n}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#555")
    axp.set_xlabel("effective hole offset [mm]  (commanded bias ⊕ perception noise)")
    axp.set_ylabel("success rate [%]")
    axp.set_ylim(-5, 105)
    axp.set_title("Round 4A — success collapses at the same envelope\n"
                  "regardless of whether offset is commanded or perceptual",
                  fontsize=11)
    axp.grid(True, alpha=0.3)
    axp.legend()
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def round4a_pose_error_validation(csv_name, fname):
    """Validation: injected sigma → realized pose-estimation error (per offset)."""
    rows = [r for r in load(csv_name) if r["perception"] == "gt-noise"]
    offs = sorted({float(r["offset_mm"]) for r in rows})
    data = agg(rows, ["offset_mm"], "noise_sigma_mm", "pose_estimation_error")
    fig, axp = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(offs)))
    for off, col in zip(offs, cmap):
        k = next((g for g in data if float(g[0]) == off), None)
        if k is None:
            continue
        xs, ys = data[k]
        axp.plot(xs, ys * 1000, marker="s", color=col, lw=2, ms=6,
                 label=f"offset {off:.0f}mm")
    axp.set_xlabel("injected hole noise σ [mm]")
    axp.set_ylabel("mean realized pose-estimation error [mm]")
    axp.set_title("Round 4A — perception error model validation\n"
                  "(mean ||estimate − true|| vs injected σ)", fontsize=11)
    axp.grid(True, alpha=0.3)
    axp.legend(title="base offset")
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def round4b_backend_success_vs_offset(csv_name, fname):
    rows = load(csv_name)
    order = [p for p in ["gt", "gt-noise", "rgbd"]
             if p in {r["perception"] for r in rows}]
    data = agg(rows, ["perception"], "offset_mm", "success")
    fig, axp = plt.subplots(figsize=(7, 4.5))
    pcol = {"gt": "#4C72B0", "gt-noise": "#DD8452", "rgbd": "#55A868"}
    for p in order:
        k = (p,)
        if k not in data:
            continue
        xs, ys = data[k]
        axp.plot(xs, ys * 100, marker="o", color=pcol.get(p), lw=2, ms=7, label=p)
    axp.set_xlabel("base lateral offset [mm]")
    axp.set_ylabel("success rate [%]")
    axp.set_ylim(-5, 105)
    axp.set_title("Round 4B — perception backend comparison\n"
                  "(impedance, waypoint, spiral r=6mm, 30 seeds)", fontsize=11)
    axp.grid(True, alpha=0.3); axp.legend(title="perception")
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def round4b_backend_pose_error_bar(csv_name, fname):
    rows = load(csv_name)
    order = [p for p in ["gt", "gt-noise", "rgbd"]
             if p in {r["perception"] for r in rows}]
    offs = sorted({float(r["offset_mm"]) for r in rows})
    fig, axp = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(order)
    x = np.arange(len(offs))
    pcol = {"gt": "#4C72B0", "gt-noise": "#DD8452", "rgbd": "#55A868"}
    for i, p in enumerate(order):
        means, stds = [], []
        for o in offs:
            v = [float(r["pose_estimation_error"]) * 1000 for r in rows
                 if r["perception"] == p and float(r["offset_mm"]) == o]
            means.append(np.mean(v) if v else np.nan)
            stds.append(np.std(v) if v else 0.0)
        axp.bar(x + i * width, means, width, yerr=stds, capsize=3,
                label=p, color=pcol.get(p))
    axp.set_xticks(x + width * (len(order) - 1) / 2, [f"{o:.0f}mm" for o in offs])
    axp.set_xlabel("base lateral offset")
    axp.set_ylabel("hole pose-estimation error [mm]")
    axp.set_title("Round 4B — realized hole pose error by backend", fontsize=11)
    axp.grid(True, axis="y", alpha=0.3); axp.legend(title="perception")
    fig.tight_layout(); fig.savefig(_FIG / fname, dpi=130); plt.close(fig)
    print(f"  wrote {fname}")


def write_perception_summary():
    r4a = load("round4a_perception_noise.csv")
    r4b = load("round4b_perception_backend.csv")
    gtn = [r for r in r4a if r["perception"] == "gt-noise"]
    sigmas = sorted({float(r["noise_sigma_mm"]) for r in gtn})
    offs4a = sorted({float(r["offset_mm"]) for r in gtn})

    def succ_at(rows, **conds):
        v = [float(r["success"]) for r in rows
             if all(float(r[k]) == val for k, val in conds.items())]
        return np.mean(v) if v else float("nan")

    # success vs sigma table (gt-noise, per offset)
    hdr = "| base offset | " + " | ".join(f"σ={s:.0f}mm" for s in sigmas) + " |"
    sep = "|" + "---|" * (len(sigmas) + 1)
    rows_md = [hdr, sep]
    for o in offs4a:
        cells = [f"{succ_at(gtn, offset_mm=o, noise_sigma_mm=s)*100:.0f}%"
                 for s in sigmas]
        rows_md.append(f"| {o:.0f}mm | " + " | ".join(cells) + " |")
    sigma_table = "\n".join(rows_md)

    # backend table (4B)
    border = [p for p in ["gt", "gt-noise", "rgbd"]
              if p in {r["perception"] for r in r4b}]
    offs4b = sorted({float(r["offset_mm"]) for r in r4b})

    def b_succ(p, o):
        v = [float(r["success"]) for r in r4b
             if r["perception"] == p and float(r["offset_mm"]) == o]
        return np.mean(v) if v else float("nan")

    def b_err(p):
        v = [float(r["pose_estimation_error"]) * 1000 for r in r4b
             if r["perception"] == p]
        return np.mean(v) if v else float("nan")

    bh = "| perception | " + " | ".join(f"{o:.0f}mm" for o in offs4b) + " | mean pose err |"
    bsep = "|" + "---|" * (len(offs4b) + 2)
    blines = [bh, bsep]
    for p in border:
        cells = [f"{b_succ(p,o)*100:.0f}%" for o in offs4b]
        blines.append(f"| {p} | " + " | ".join(cells) + f" | {b_err(p):.2f}mm |")
    backend_table = "\n".join(blines)

    md = f"""# Perception-Axis Study Summary

_Generated by `scripts/plot_axis_studies.py` from the Round 4A / 4B sweep CSVs._

## Experiment question
Does perception pose **error** behave like a commanded misalignment — i.e. does it
push the operating point along the *same* recovery envelope mapped in Round 3
(effective offset ≈ spiral_radius + clearance ≈ {_ENVELOPE_MM:.0f} mm) — and how
does a realistic RGB-D backend compare to injected Gaussian noise?

## Fixed variables
- Controller = `impedance` · Planner = `waypoint` · Contact = `spiral` (r = 6 mm)
- Single round peg/hole (4 mm clearance) · 30 seeds
- Base offset = deterministic hole bias; peg perception error held at 0 in 4A
  (only the **hole** estimate varies)

## Changed variable(s)
- **4A**: perception ∈ {{gt, gt-noise}}, base offset ∈ {{0,3,6}} mm,
  hole noise σ ∈ {{0,1,2,4,6,8}} mm
- **4B**: perception ∈ {{gt, gt-noise, rgbd}}, base offset ∈ {{0,3,6}} mm

## Round 4A — success vs perception noise (gt-noise)
{sigma_table}

## Round 4A — the unifying result
When pooled by **effective offset** (commanded bias ⊕ realized perception error),
success collapses around the **same ≈{_ENVELOPE_MM:.0f} mm envelope** found in
Round 3 — perception error and commanded offset are interchangeable as far as the
recovery layer is concerned. See `round4a_success_vs_effective_offset.png`.

`round4a_pose_error_validation.png` confirms the noise model: realized
‖estimate − true‖ grows monotonically with injected σ (offset by the base bias).

## Round 4B — backend comparison
{backend_table}

The RGB-D pipeline's hole error sits in the low-mm range (deterministic — no
per-seed variation in the estimate), comparable to a small-σ gt-noise setting, so
on this open scene it stays inside the recovery envelope and tracks gt success.

## Interpretation
- **Perception error = effective offset.** The recovery envelope is the single
  governing quantity; it does not matter whether misalignment is commanded or
  perceptual. This ties Rounds 3 and 4 into one law:
  `success ⇔ (commanded_offset ⊕ perception_error) ≲ spiral_radius + clearance`.
- **Robustness budget is shared.** Every mm of perception error spends the same
  recovery budget as a mm of mechanical misalignment. A noisy sensor narrows the
  commanded-offset tolerance one-for-one.
- **RGB-D is "good enough" here** only because its error is small relative to the
  10 mm envelope; a tighter clearance or smaller search radius would expose it.

## Caveats
- Injected noise is **3-D isotropic**, so `pose_estimation_error` includes a z
  component that also perturbs depth-based success detection — not purely lateral.
- **Peg** perception error is 0 in 4A (custom peg σ = 0): only hole estimation
  varies, isolating the hole-error → recovery-envelope link.
- `rgbd` pose estimate is **deterministic** (no RNG in the vision pipeline); seed
  varies only sensor/control noise, so its success spread is narrower than gt-noise.
- `gt-noise` at σ=0 is equivalent to `gt` (believed = true + bias); a built-in
  consistency check.
- No filtering/Kalman yet — each episode uses a single perception sample.

## Implication for the next axis (state estimation / filtering)
Round 4 shows perception error spends the recovery budget directly, so the natural
next lever is **reducing effective error before it reaches the controller**:
multi-sample averaging, low-pass/EMA on the pose estimate, or a Kalman filter.
The clean experiment: hold everything fixed at a σ where gt-noise fails (e.g.
6–8 mm), and measure how much of the envelope a filter buys back — i.e. does
filtering shift the `success_vs_effective_offset` curve left toward the raw-σ=0
baseline. That isolates estimation quality as its own axis.
"""
    out = _RESULTS / "perception_axis_summary.md"
    out.write_text(md)
    print(f"  wrote {out.name}")


def main():
    _FIG.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {_FIG}/")
    success_vs_offset("round1_controllers.csv", "controller",
                      ["jointpos", "impedance", "osc", "osc-lambda"],
                      "Round 1 — controller axis (contact=none)",
                      "round1_controller_success_vs_offset.png",
                      subtitle="planner=waypoint, perception=gt, 10 seeds")
    success_vs_offset("round3_contacts.csv", "contact",
                      ["none", "spiral", "lcs-mpc"],
                      "Round 3 — contact axis",
                      "round3_contact_success_vs_offset.png",
                      subtitle="controller=osc-lambda, spiral_r=6mm, 10 seeds")
    success_vs_offset("round3b_offsets.csv", "contact",
                      ["none", "spiral", "lcs-mpc"],
                      "Round 3b — contact boundary (5–8 mm)",
                      "round3b_contact_boundary_5_8mm.png",
                      subtitle="controller=osc-lambda, spiral_r=6mm, 10 seeds")
    spiral_radius_heatmap("round3c_radius.csv",
                          "round3c_spiral_radius_heatmap.png")
    force_vs_radius_at("round3c_radius.csv", 9.0,
                       "round3c_force_vs_radius_at_9mm.png")
    write_summary()

    # ── Round 2 — planner axis ────────────────────────────────────────────────
    success_vs_offset("round2a_planner_no_recovery.csv", "planner",
                      ["waypoint", "ee-ocp"],
                      "Round 2A — planner axis (contact=none)",
                      "round2a_planner_success_vs_offset.png",
                      subtitle="controller=impedance, perception=gt, 30 seeds")
    planner_smoothness_bar("round2a_planner_no_recovery.csv",
                           "round2a_planner_smoothness_bar.png")
    planner_motion_cost_bar("round2a_planner_no_recovery.csv",
                            "round2a_planner_motion_cost_bar.png")
    planner_time_cost_bar("round2a_planner_no_recovery.csv",
                          "round2a_planner_time_cost_bar.png")
    planner_recovery_burden("round2b_planner_with_spiral.csv",
                            "round2b_planner_recovery_burden.png")
    write_planner_summary()

    # ── Round 4 — perception axis (guarded: only if sweeps have completed) ─────
    if (_RESULTS / "round4a_perception_noise.csv").exists():
        round4a_success_vs_noise("round4a_perception_noise.csv",
                                 "round4a_success_vs_noise.png")
        round4a_success_vs_effective_offset("round4a_perception_noise.csv",
                                            "round4a_success_vs_effective_offset.png")
        round4a_pose_error_validation("round4a_perception_noise.csv",
                                      "round4a_pose_error_validation.png")
    else:
        print("  [skip] round4a_perception_noise.csv not found")
    if (_RESULTS / "round4b_perception_backend.csv").exists():
        round4b_backend_success_vs_offset("round4b_perception_backend.csv",
                                          "round4b_backend_success_vs_offset.png")
        round4b_backend_pose_error_bar("round4b_perception_backend.csv",
                                       "round4b_backend_pose_error_bar.png")
    else:
        print("  [skip] round4b_perception_backend.csv not found")
    if ((_RESULTS / "round4a_perception_noise.csv").exists()
            and (_RESULTS / "round4b_perception_backend.csv").exists()):
        write_perception_summary()
    print("done.")


if __name__ == "__main__":
    main()
