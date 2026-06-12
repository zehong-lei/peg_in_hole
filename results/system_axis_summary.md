# System Axis Study — Project Control Summary

_Top-level integration of the controller, planner, contact, and perception axis
studies. Generated as a hand-authored synthesis over the Round 1–4 sweep CSVs and
their per-axis summaries (`contact_axis_summary.md`, `planner_axis_summary.md`,
`perception_axis_summary.md`)._

---

## 1. Project goal

Build a **modular peg-in-hole benchmark** in which the robot manipulation stack is
decomposed into independent, swappable layers, so that each layer's contribution to
**robustness under lateral pose/perception uncertainty** can be measured in
isolation. The scientific aim is not "make insertion work" (it already does) but to
attribute *where robustness comes from* and *what each layer costs*, building a
defensible mental model of how control/optimization/perception choices trade off in
a contact-rich task.

Concrete task: Franka-style arm inserts a round peg (radius 10 mm) into a round hole
(radius 14 mm → **4 mm radial clearance**) on a fixed board, MuJoCo simulation.

---

## 2. System architecture

Per-step closed loop (`MultiTaskAssemblyTask._run_single_task`):

```
observe → estimate → plan → control → step
  │          │         │       │
SensorWrapper │   ScriptedPlanner │
+PerceptionModule │  (+preinsert_ocp,  ControllerKind:
              StateEstimator   lcs_mpc, spiral)  position / impedance / OSC
              (EMA low-pass)                     └→ τ, q_des → MuJoCo
```

Layer modules (each independently selectable):

| Layer | Module(s) | Config subtree |
|---|---|---|
| Perception / estimation | `PerceptionModule` (noisy-GT, RGB-D), `StateEstimator` (EMA) | `perception.yaml`, sensor wiring |
| Task / reference | `ScriptedPlanner` (8-stage state machine) | `task.stages` |
| Motion planning | `preinsert_ocp` (EE-space SLSQP) | `task.preinsert_ocp` |
| Contact handling | spiral search, `lcs_mpc` (force feedforward) | `task.contact_recovery`, `task.lcs_mpc` |
| Low-level control | position / impedance / operational-space (±Λ) | `controller.yaml` |

The experiment driver (`scripts/run_benchmark.py` + `scripts/experiment_axes.py`)
exposes four **orthogonal axes**; each maps to exactly one config subtree and never
implicitly toggles another layer. Misalignment is injected as a deterministic hole
**bias** (commanded offset); perception noise as a hole-estimate sigma.

---

## 3. Experimental axes — fixed vs changed variables

| Round | Axis (changed) | Fixed | Sweep | Seeds |
|---|---|---|---|---|
| 1 | **controller** {jointpos, impedance, osc, osc-λ} | planner=waypoint, contact=none, perception=gt | offset 0/3/6/9 mm | 10 |
| 2A | **planner** {waypoint, ee-ocp} | controller=impedance, contact=none, perception=gt | offset 0/3/6 mm | 30 |
| 2B | **planner** {waypoint, ee-ocp} | controller=impedance, contact=spiral(6 mm), perception=gt | offset 6/8 mm | 30 |
| 3 / 3b | **contact** {none, spiral, lcs-mpc} | controller=osc-λ, planner=waypoint, perception=gt | offset 0–9 / 5–8 mm | 10 |
| 3c | **spiral radius** {4,6,8,10 mm} | controller=osc-λ, contact=spiral | offset 5/7/9/11 mm | 10 |
| 4A | **perception** {gt, gt-noise} | controller=impedance, planner=waypoint, contact=spiral(6 mm) | offset 0/3/6 mm × σ 0/1/2/4/6/8 mm | 30 |
| 4B | **perception** {gt, gt-noise, rgbd} | controller=impedance, planner=waypoint, contact=spiral(6 mm) | offset 0/3/6 mm | 30 |

Not implemented (skipped honestly): `joint-ocp` planner, `force-guided` contact,
`ema` perception axis.

---

## 4. One key conclusion per axis

- **Controller axis — flat for success; lever is contact *force*.**
  All four controllers share the identical success envelope (100/100/~80/0 % at
  0/3/6/9 mm). They differ only by a small peak-force margin (~1.5 N lower for
  impedance/OSC vs rigid jointpos at low offset). Compliance alone does **not**
  extend offset tolerance. _(Round 1)_

- **Planner axis — flat for success; lever is motion *quality*.**
  waypoint and ee-ocp are statistically tied on success (with and without recovery).
  ee-ocp cuts RMS end-effector acceleration **2.8×** (11.3 → 4.0) and peak joint
  velocity ~19 % (1.27 → 1.03 rad/s) — at a cost of ~28 ms cold solve/episode,
  +2.4 % path, +~2.5 s task time, and a hair worse terminal accuracy. On this
  open, generously-cleared scene the planner buys quality, not success. _(Round 2)_

- **Contact axis — *owns* the success envelope, and it is tunable.**
  Recovery converts the 6–8 mm cliff that no controller could fix into ~100 %.
  LCS-MPC edges spiral at the margin (100 % vs 90 % at 8 mm). The solvable envelope
  scales cleanly: **offset_max ≈ spiral_radius + clearance**; +2 mm radius ≈ +2 mm
  tolerance — at the cost of higher peak force at fixed offset. _(Round 3/3b/3c)_

- **Perception axis — error = effective offset; one envelope governs both.**
  Pooled by effective offset (commanded bias ⊕ realized perception error), success
  collapses at the **same ≈10 mm envelope** as commanded offset. Higher base offset
  shifts the success-vs-σ curve down one-for-one. RGB-D (≈2.8 mm hole error) stays
  inside the envelope and tracks gt here. _(Round 4)_

---

## 5. Unified interpretation

The four axes reduce to a **single governing law**:

```
effective_offset = commanded_offset  ⊕  perception_error
success  ⇔  effective_offset  ≲  contact_recovery_envelope
                                  ( ≈ spiral_radius + radial_clearance )
```

- The **contact layer** sets the envelope size.
- The **controller** and **planner** do not move the envelope; they set the
  *operating quality inside it* (contact force, motion smoothness, time).
- **Perception error and mechanical misalignment are interchangeable** inputs to
  the same envelope — every mm of either spends the same recovery budget.

This is why Rounds 3 and 4 produce the same ~10 mm collapse from two different
sources, and why Rounds 1 and 2 are flat on success: they touch quality, not the
envelope.

---

## 6. Current design principle

> **Choose the smallest contact-recovery envelope that still covers the expected
> pose + perception uncertainty.**

Rationale: the envelope is free to enlarge geometrically (bigger search radius), but
**larger envelopes raise force and time cost** — at a fixed offset, peak contact
force and retry/jam counts grow with search radius (Round 3c), and recovery episodes
take longer. Oversizing the envelope "to be safe" pays a continuous force/time tax on
*every* insertion, including the easy ones. The right sizing is:

```
spiral_radius  ≈  expected_total_error_budget − radial_clearance
where expected_total_error_budget = commanded_offset_3σ ⊕ perception_error_3σ
```

Corollary: investing in the controller/planner improves *how gently/cleanly* an
insertion happens but cannot substitute for an adequately-sized recovery envelope;
investing in perception accuracy directly shrinks the required envelope (and thus the
force/time cost), which is often the cheaper lever.

---

## 7. What remains open

- **Estimation / filtering not yet studied.** Each episode uses a single perception
  sample; no EMA/Kalman/multi-sample averaging on the pose estimate. This is the
  direct lever on `perception_error` and is the recommended next axis (§8).
- **Unimplemented axis values:** `joint-ocp` planner, `force-guided` contact,
  `ema` perception — defined in the axis enum but not wired.
- **Single geometry only.** All sweeps use the round peg / round hole (4 mm
  clearance). Square/rect pegs, tighter clearances, and the multi-task assembly
  sequence are built but not swept — clearance is a first-class term in the envelope
  law and deserves its own sweep.
- **Lateral-only misalignment.** Orientation error (roll/pitch/yaw) and the
  square/rect orientation-alignment regime are untested; the law is currently a
  position-only statement.
- **Friction / dynamics uncertainty** not swept.
- **Noise is 3-D isotropic**, so perception-error results include a z-component that
  also perturbs depth-based success detection — not a purely lateral test.
- **RGB-D estimate is deterministic** (no per-seed pose variation); its robustness
  spread is narrower than gt-noise and conclusions about it are scene-specific.
- **Planner value under-stressed.** ee-ocp's motion-quality advantage was measured
  on an open scene with no obstacles/joint-limit pressure, where it cannot show a
  success benefit. A constrained-approach scenario is needed to value it fairly.

---

## 8. Next recommended axis: estimation / filtering

Round 4 established that perception error spends the recovery budget one-for-one, so
the highest-leverage next move is **reducing effective error before it reaches the
controller**.

- **Hypothesis:** a pose filter (multi-sample averaging → EMA → Kalman) shifts the
  `success_vs_effective_offset` curve **left**, back toward the σ=0 baseline, by
  shrinking realized `perception_error`.
- **Clean experiment:** fix controller=impedance, planner=waypoint,
  contact=spiral(6 mm); hold gt-noise at a σ where it fails (6–8 mm); sweep the
  estimator {none, EMA(α), N-sample average, Kalman} and measure recovered envelope
  (success-rate lift) vs added latency/observations. Isolate estimation quality as
  its own axis with everything else frozen.
- **Expected payoff framing:** filtering buys envelope cheaply (compute/latency)
  relative to enlarging the spiral radius (force/time on every insertion), so it
  should be the preferred robustness lever when uncertainty is stochastic rather
  than systematic.

---

### Artifact index
- Data: `results/round{1,2a,2b,3,3b,3c,4a,4b}_*.csv` (shared CSV schema)
- Figures: `results/figures/` (15 figures across 4 axes)
- Per-axis write-ups: `results/contact_axis_summary.md`,
  `results/planner_axis_summary.md`, `results/perception_axis_summary.md`
- Drivers: `scripts/run_benchmark.py`, `scripts/experiment_axes.py`,
  `scripts/plot_axis_studies.py`
