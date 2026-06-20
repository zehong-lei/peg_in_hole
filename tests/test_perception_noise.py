#!/usr/bin/env python3
"""Perception module noise verification tests.

T1  SceneObservation has all expected pegs and holes
T2  Noisy estimates differ from true poses (noise is applied)
T3  Zero-noise passthrough: estimates ≈ true poses
T4  task_sequence order matches canonical round→square→rect
T5  get_perception_errors() returns all expected keys
T6  set_hole_pos_estimate() overrides SensorWrapper hole noise
T7  Backward compat: no perception → old behaviour unchanged
T8  PerceptionModule derives task_sequence without hardcoded list
T9  Episode result dict contains hole_pos_error_m / peg_pos_error_m

Usage
-----
  python3 scripts/test_perception_noise.py
"""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

_REPO_ROOT = Path(__file__).parents[1]


def load_cfg(name: str) -> dict:
    return yaml.safe_load((_REPO_ROOT / "configs" / f"{name}.yaml").read_text())


def build_env():
    from src.envs.assembly_env import AssemblyEnv
    scene_cfg = load_cfg("scene")
    task_cfg  = load_cfg("task")
    env = AssemblyEnv(scene_cfg, task_cfg, seed=0)
    env.reset()
    return env, scene_cfg, task_cfg


def _ok(name: str):
    print(f"  PASS  {name}")


def _fail(name: str, msg: str):
    print(f"  FAIL  {name}: {msg}")
    return False


# ─── T1: SceneObservation completeness ────────────────────────────────────────

def test_t1_scene_obs_completeness():
    from src.perception import PerceptionModule
    env, _, _ = build_env()
    pcfg = load_cfg("perception")

    perc = PerceptionModule(env, pcfg, noise_level="easy", seed=0)
    obs = perc.observe()

    expected_pegs  = {"peg", "peg_square", "peg_rect"}
    expected_holes = {"round_hole", "square_hole", "rect_slot"}

    assert set(obs.peg_estimates.keys())  == expected_pegs,  \
        f"peg keys mismatch: {set(obs.peg_estimates.keys())}"
    assert set(obs.hole_estimates.keys()) == expected_holes, \
        f"hole keys mismatch: {set(obs.hole_estimates.keys())}"
    assert len(obs.task_sequence) == 3, \
        f"expected 3 tasks, got {len(obs.task_sequence)}"
    assert obs.captured_at_time >= 0.0
    _ok("T1 SceneObservation completeness")
    return True


# ─── T2: Noise is applied ─────────────────────────────────────────────────────

def test_t2_noise_applied():
    from src.perception import PerceptionModule
    env, _, _ = build_env()
    pcfg = load_cfg("perception")

    perc = PerceptionModule(env, pcfg, noise_level="hard", seed=42)
    obs  = perc.observe()
    errs = perc.get_perception_errors(obs)

    any_hole_nonzero = any(v > 0.0 for v in errs["hole_pos_errors"].values())
    any_peg_nonzero  = any(v > 0.0 for v in errs["peg_pos_errors"].values())
    assert any_hole_nonzero, "all hole errors are exactly 0.0 — noise not applied"
    assert any_peg_nonzero,  "all peg  errors are exactly 0.0 — noise not applied"
    _ok("T2 Noise is applied")
    return True


# ─── T3: Zero-noise passthrough ───────────────────────────────────────────────

def test_t3_zero_noise_passthrough():
    from src.perception import PerceptionModule
    env, _, _ = build_env()
    pcfg = load_cfg("perception")

    perc = PerceptionModule(env, pcfg, noise_level="custom", seed=0,
                            custom_hole_sigma=0.0)
    obs  = perc.observe()
    errs = perc.get_perception_errors(obs)

    for hole_name, err in errs["hole_pos_errors"].items():
        assert err < 1e-9, f"hole {hole_name} error={err} with zero noise"
    _ok("T3 Zero-noise passthrough (holes)")
    return True


# ─── T4: Task sequence order ──────────────────────────────────────────────────

def test_t4_task_sequence_order():
    from src.perception import PerceptionModule
    env, _, _ = build_env()
    pcfg = load_cfg("perception")

    perc = PerceptionModule(env, pcfg, noise_level="easy", seed=0)
    obs  = perc.observe()

    expected_order = [
        ("peg",        "round_hole"),
        ("peg_square", "square_hole"),
        ("peg_rect",   "rect_slot"),
    ]
    assert obs.task_sequence == expected_order, \
        f"order mismatch: {obs.task_sequence}"
    _ok("T4 Task sequence order")
    return True


# ─── T5: Error dict structure ─────────────────────────────────────────────────

def test_t5_error_dict_structure():
    from src.perception import PerceptionModule
    env, _, _ = build_env()
    pcfg = load_cfg("perception")

    perc = PerceptionModule(env, pcfg, noise_level="easy", seed=0)
    obs  = perc.observe()
    errs = perc.get_perception_errors(obs)

    required_keys = {
        "hole_pos_errors", "peg_pos_errors",
        "mean_hole_pos_error", "mean_peg_pos_error",
        "mean_object_pose_error",
    }
    missing = required_keys - set(errs.keys())
    assert not missing, f"missing keys: {missing}"
    assert isinstance(errs["mean_hole_pos_error"], float)
    assert isinstance(errs["mean_peg_pos_error"],  float)
    _ok("T5 Error dict structure")
    return True


# ─── T6: SensorWrapper hole override ─────────────────────────────────────────

def test_t6_sensor_hole_override():
    from src.sensors.sensor_wrapper import SensorWrapper
    env, _, task_cfg = build_env()
    noise_cfg = load_cfg("noise")

    sensor = SensorWrapper(env, noise_cfg, noise_level="easy", seed=0,
                           hole_pos_sigma=0.0)

    fake_hole = np.array([1.0, 2.0, 3.0])
    sensor.set_hole_pos_estimate(fake_hole)

    env.set_active_task("peg", "round_hole")
    obs = sensor.get_observation()
    assert np.allclose(obs.hole_pos, fake_hole, atol=1e-9), \
        f"expected override {fake_hole}, got {obs.hole_pos}"

    # revert
    sensor.set_hole_pos_estimate(None)
    obs2 = sensor.get_observation()
    assert not np.allclose(obs2.hole_pos, fake_hole, atol=1e-9), \
        "override not cleared after set_hole_pos_estimate(None)"

    _ok("T6 SensorWrapper hole override")
    return True


# ─── T7: Backward compat — no perception ─────────────────────────────────────

def test_t7_backward_compat():
    from src.envs.assembly_env import AssemblyEnv
    from src.sensors.sensor_wrapper import SensorWrapper
    from src.estimators.state_estimator import StateEstimator
    from src.controllers.position_controller import PositionController
    from src.planners.scripted_planner import ScriptedPlanner
    from src.tasks.multi_task_assembly import MultiTaskAssemblyTask

    scene_cfg = load_cfg("scene")
    task_cfg  = load_cfg("task")
    ctrl_cfg  = load_cfg("controller")
    noise_cfg = load_cfg("noise")

    # Extend timeouts
    to = list(task_cfg["stages"]["stage_timeout"])
    to[3] = 15.0
    to[5] = 150.0
    task_cfg["stages"]["stage_timeout"] = to
    task_cfg["contact_recovery"]["max_attempts"] = 10

    env       = AssemblyEnv(scene_cfg, task_cfg, seed=0)
    sensor    = SensorWrapper(env, noise_cfg, noise_level="easy", seed=0,
                              hole_pos_sigma=0.0)
    estimator = StateEstimator(noise_cfg)
    planner   = ScriptedPlanner(task_cfg, ctrl_cfg)
    pos_ctrl  = PositionController(ctrl_cfg["position_controller"],
                                   dt=task_cfg["sim"]["dt"])

    task = MultiTaskAssemblyTask(
        env, sensor, estimator, planner, pos_ctrl,
        task_cfg, ctrl_cfg,
        # No perception kwarg — old signature still works
        task_sequence=[("peg", "round_hole")],
    )

    assert task._perception is None, "perception should default to None"
    assert task._task_sequence == [("peg", "round_hole")]
    _ok("T7 Backward compat (no perception)")
    return True


# ─── T8: Perception derives task sequence ─────────────────────────────────────

def test_t8_task_derivation_from_perception():
    from src.envs.assembly_env import AssemblyEnv
    from src.sensors.sensor_wrapper import SensorWrapper
    from src.estimators.state_estimator import StateEstimator
    from src.controllers.position_controller import PositionController
    from src.planners.scripted_planner import ScriptedPlanner
    from src.tasks.multi_task_assembly import MultiTaskAssemblyTask
    from src.perception import PerceptionModule

    scene_cfg = load_cfg("scene")
    task_cfg  = load_cfg("task")
    ctrl_cfg  = load_cfg("controller")
    noise_cfg = load_cfg("noise")
    pcfg      = load_cfg("perception")

    to = list(task_cfg["stages"]["stage_timeout"])
    to[3] = 15.0
    to[5] = 150.0
    task_cfg["stages"]["stage_timeout"] = to
    task_cfg["contact_recovery"]["max_attempts"] = 10

    env       = AssemblyEnv(scene_cfg, task_cfg, seed=0)
    sensor    = SensorWrapper(env, noise_cfg, noise_level="easy", seed=0,
                              hole_pos_sigma=0.0)
    estimator = StateEstimator(noise_cfg)
    planner   = ScriptedPlanner(task_cfg, ctrl_cfg)
    pos_ctrl  = PositionController(ctrl_cfg["position_controller"],
                                   dt=task_cfg["sim"]["dt"])

    env.reset()
    perception = PerceptionModule(env, pcfg, noise_level="easy", seed=0)

    task = MultiTaskAssemblyTask(
        env, sensor, estimator, planner, pos_ctrl,
        task_cfg, ctrl_cfg,
        task_sequence=None,    # must be derived from perception
        perception=perception,
    )

    expected = [("peg", "round_hole"), ("peg_square", "square_hole"),
                ("peg_rect", "rect_slot")]
    assert task._task_sequence == expected, \
        f"derived sequence mismatch: {task._task_sequence}"
    _ok("T8 Perception derives task sequence")
    return True


# ─── T9: Episode result has perception keys ───────────────────────────────────

def test_t9_episode_result_keys():
    from src.envs.assembly_env import AssemblyEnv
    from src.sensors.sensor_wrapper import SensorWrapper
    from src.estimators.state_estimator import StateEstimator
    from src.controllers.position_controller import PositionController
    from src.planners.scripted_planner import ScriptedPlanner
    from src.tasks.multi_task_assembly import MultiTaskAssemblyTask
    from src.perception import PerceptionModule

    scene_cfg = load_cfg("scene")
    task_cfg  = load_cfg("task")
    ctrl_cfg  = load_cfg("controller")
    noise_cfg = load_cfg("noise")
    pcfg      = load_cfg("perception")

    to = list(task_cfg["stages"]["stage_timeout"])
    to[3] = 15.0
    to[5] = 150.0
    task_cfg["stages"]["stage_timeout"] = to
    task_cfg["contact_recovery"]["max_attempts"] = 10

    env       = AssemblyEnv(scene_cfg, task_cfg, seed=0)
    sensor    = SensorWrapper(env, noise_cfg, noise_level="easy", seed=0,
                              hole_pos_sigma=0.0)
    estimator = StateEstimator(noise_cfg)
    planner   = ScriptedPlanner(task_cfg, ctrl_cfg)
    pos_ctrl  = PositionController(ctrl_cfg["position_controller"],
                                   dt=task_cfg["sim"]["dt"])

    env.reset()
    perception = PerceptionModule(env, pcfg, noise_level="easy", seed=0)

    task = MultiTaskAssemblyTask(
        env, sensor, estimator, planner, pos_ctrl,
        task_cfg, ctrl_cfg,
        task_sequence=None,
        perception=perception,
    )

    # Run only the round-peg sub-task to keep the test fast
    task._task_sequence = [("peg", "round_hole")]
    result = task.run_episode(max_steps_per_task=20000)

    tr = result["task_results"][0]
    assert "hole_pos_error_m" in tr, "hole_pos_error_m missing from task result"
    assert "peg_pos_error_m"  in tr, "peg_pos_error_m missing from task result"
    assert isinstance(tr["hole_pos_error_m"], float)
    assert isinstance(tr["peg_pos_error_m"],  float)
    _ok("T9 Episode result has perception keys")
    return True


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    tests = [
        ("T1 SceneObservation completeness",   test_t1_scene_obs_completeness),
        ("T2 Noise is applied",                test_t2_noise_applied),
        ("T3 Zero-noise passthrough",          test_t3_zero_noise_passthrough),
        ("T4 Task sequence order",             test_t4_task_sequence_order),
        ("T5 Error dict structure",            test_t5_error_dict_structure),
        ("T6 SensorWrapper hole override",     test_t6_sensor_hole_override),
        ("T7 Backward compat",                 test_t7_backward_compat),
        ("T8 Perception derives task seq",     test_t8_task_derivation_from_perception),
        ("T9 Episode result keys",             test_t9_episode_result_keys),
    ]

    print(f"\nPerception Module Tests  ({len(tests)} tests)\n")

    passed = failed = 0
    for name, fn in tests:
        try:
            ok = fn()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    print(f"\n{'─'*40}")
    print(f"Results: {passed}/{len(tests)} passed"
          + (f"  ({failed} failed)" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
