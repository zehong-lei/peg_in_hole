"""MultiTaskAssemblyTask — multi-peg assembly episode orchestrator.

Iterates over a fixed task_sequence for each (peg, hole) pair.
Between sub-tasks the simulation continues running (no full reset);
only the active peg/hole pointer and planner state change.

On first task failure the episode terminates (v1 behaviour).

Per-task metrics
----------------
  success, insertion_depth, peak_contact_force, recovery_attempts,
  preinsert_error_m, task_time, steps

Episode metrics
---------------
  total_success, num_tasks_completed, num_tasks_attempted,
  total_time, total_recovery_attempts, max_peak_force, task_results
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from src.envs.assembly_env import AssemblyEnv
from src.sensors.sensor_wrapper import SensorWrapper
from src.estimators.state_estimator import StateEstimator
from src.controllers.position_controller import PositionController
from src.controllers.operational_space_controller import OperationalSpaceController
from src.planners.scripted_planner import ScriptedPlanner, Stage
from src.planners.trajectory_utils import compute_path_length, compute_smoothness, compute_min_board_clearance


class MultiTaskAssemblyTask:
    """Orchestrates a sequence of peg-in-hole sub-tasks in one episode.

    Parameters
    ----------
    env       : AssemblyEnv
    sensor    : SensorWrapper  (wrapping env)
    estimator : StateEstimator
    planner   : ScriptedPlanner
    pos_ctrl  : PositionController
    task_cfg  : dict  (task.yaml)
    ctrl_cfg  : dict  (controller.yaml)
    os_ctrl   : OperationalSpaceController | None
    task_sequence : list[(peg_name, hole_name)] | None
        If None, reads from scene_cfg["task_sequence"] via env._scene_cfg.
    """

    def __init__(self,
                 env: AssemblyEnv,
                 sensor: SensorWrapper,
                 estimator: StateEstimator,
                 planner: ScriptedPlanner,
                 pos_ctrl: PositionController,
                 task_cfg: dict,
                 ctrl_cfg: dict,
                 os_ctrl: Optional[OperationalSpaceController] = None,
                 task_sequence: Optional[list] = None,
                 perception=None):
        self._env      = env
        self._sensor   = sensor
        self._estimator = estimator
        self._planner  = planner
        self._pos_ctrl = pos_ctrl
        self._os_ctrl  = os_ctrl
        self._tc       = task_cfg
        self._cc       = ctrl_cfg
        self._perception = perception   # PerceptionModule | None

        # Mainline low-level control: operational-space controller (with inertia
        # shaping) realises compliant/feedforward-bearing commands; free-space
        # tracking always uses the Cartesian position controller.
        self._use_os = os_ctrl is not None

        if task_sequence is not None:
            self._task_sequence = [tuple(p) for p in task_sequence]
        elif perception is not None:
            scene_obs = perception.observe()
            self._task_sequence = scene_obs.task_sequence
        else:
            self._task_sequence = [
                tuple(p) for p in env._scene_cfg.get("task_sequence", [])
            ]

    # ── episode entry point ────────────────────────────────────────────────────

    def run_episode(self, max_steps_per_task: int = 15000,
                    log_cb=None, step_cb=None) -> dict:
        """Run one full multi-task episode.

        Returns episode-level and per-task metrics.
        log_cb  : optional callable(task_idx, step, belief, cmd, ctrl)
        step_cb : optional callable() → bool | None; called after each env.step().
                  Return False to abort the current task early (e.g. viewer closed).
        """
        self._env.reset()
        episode_t0 = self._env.d.time

        task_results: list[dict] = []
        total_recovery = 0
        max_peak_force = 0.0

        for task_idx, (peg_name, hole_name) in enumerate(self._task_sequence):
            # ── configure sub-task ────────────────────────────────────────
            self._env.set_active_task(peg_name, hole_name)
            hl = AssemblyEnv.peg_half_length(peg_name)
            self._planner.set_peg_half_length(hl)

            # Clear peg grasp state from previous sub-task before any perception call
            self._sensor.clear_peg_pose_estimate()

            t_now = self._env.d.time
            self._planner.reset(t=t_now)
            self._estimator._initialized = False
            if self._os_ctrl is not None:
                self._os_ctrl.reset_metrics()

            # Per-task perception: observe scene and inject hole + peg estimates
            _perception_meta: dict = {}
            if self._perception is not None:
                _task_scene_obs = self._perception.observe()
                hole_est = _task_scene_obs.hole_estimates.get(hole_name)
                self._sensor.set_hole_pos_estimate(
                    hole_est.pos if hole_est is not None else None)
                peg_est = _task_scene_obs.peg_estimates.get(peg_name)
                self._sensor.set_peg_pos_estimate(
                    peg_est.pos if peg_est is not None else None)
                perr = self._perception.get_perception_errors(_task_scene_obs)
                _perception_meta = {
                    "hole_pos_error_m": perr["hole_pos_errors"].get(hole_name, 0.0),
                    "peg_pos_error_m":  perr["peg_pos_errors"].get(peg_name, 0.0),
                }
            else:
                self._sensor.set_hole_pos_estimate(None)
                # peg estimate already cleared above — falls back to GT+noise

            task_t0 = self._env.d.time

            cb = None
            if log_cb is not None:
                _tidx = task_idx
                def cb(step, belief, cmd, ctrl, _ti=_tidx):
                    log_cb(_ti, step, belief, cmd, ctrl)

            result = self._run_single_task(max_steps_per_task, cb, step_cb=step_cb)
            result["task_idx"]  = task_idx
            result["peg_name"]  = peg_name
            result["hole_name"] = hole_name
            result["task_time"] = float(self._env.d.time - task_t0)
            result.update(_perception_meta)

            task_results.append(result)
            total_recovery += result.get("recovery_attempts", 0)
            max_peak_force = max(max_peak_force, result.get("peak_contact_force", 0.0))

            if not result["success"]:
                break   # v1: terminate episode on first failure

        num_completed = sum(1 for r in task_results if r["success"])
        total_time = float(self._env.d.time - episode_t0)

        return {
            "total_success":          num_completed == len(self._task_sequence),
            "num_tasks_completed":    num_completed,
            "num_tasks_attempted":    len(task_results),
            "total_time":             total_time,
            "total_recovery_attempts": total_recovery,
            "max_peak_force":         max_peak_force,
            "task_results":           task_results,
        }

    # ── single sub-task loop ───────────────────────────────────────────────────

    def _run_single_task(self, max_steps: int, log_cb=None, step_cb=None) -> dict:
        """Execute one peg-in-hole sub-task.  Does NOT reset the environment."""
        ctrl = np.zeros(8)
        ctrl[7] = self._tc["robot"]["gripper_open_ctrl"]

        peak_force      = 0.0
        force_sum       = 0.0
        contact_steps   = 0
        pre_release_depth = 0.0
        weld_deactivated  = False
        step = 0

        # ── approach-phase (MOVE_TO_PREINSERT) diagnostics for the planner axis ──
        # These isolate the planner's effect: only MOVE_TO_PREINSERT differs
        # between waypoint and ee-ocp; every other stage is identical.
        dt_sim         = self._tc["sim"]["dt"]
        approach_path  = 0.0          # executed EE path length [m]
        approach_acc2  = 0.0          # Σ ||ee_accel||² for RMS smoothness
        approach_accn  = 0
        max_qvel       = 0.0          # max |qvel| over joints [rad/s]
        max_qaccel     = 0.0          # max |qaccel| over joints [rad/s²]
        init_contact_f = 0.0          # force at first INSERT contact [N]
        _prev_ee       = None
        _prev_ee_vel   = None
        _prev_qvel     = None

        for step in range(max_steps):
            # 1. Observe
            obs = self._sensor.get_observation()
            self._sensor.set_stage(int(self._planner.stage))

            # 2. Estimate
            belief = self._estimator.update(obs)

            # 3. Plan
            cmd = self._planner.update(belief)

            # 4. Control
            J          = self._env.get_ee_jacobian()
            q_curr     = obs.q
            qdot_curr  = obs.qdot
            ee_pos, ee_rot = self._env.get_ee_pose()
            qfrc_bias  = self._env.d.qfrc_bias[:7].copy()
            M7         = self._env.get_mass_matrix() if self._use_os else None

            # Approach-phase kinematics (true state, noise-free for clean
            # planner attribution).  Reset trackers when not in the stage.
            if self._planner.stage == Stage.MOVE_TO_PREINSERT:
                qvel_true = self._env.d.qvel[:7]
                max_qvel = max(max_qvel, float(np.max(np.abs(qvel_true))))
                if _prev_qvel is not None:
                    max_qaccel = max(max_qaccel, float(
                        np.max(np.abs((qvel_true - _prev_qvel) / dt_sim))))
                if _prev_ee is not None:
                    approach_path += float(np.linalg.norm(ee_pos - _prev_ee))
                    ee_vel = (ee_pos - _prev_ee) / dt_sim
                    if _prev_ee_vel is not None:
                        ee_acc = (ee_vel - _prev_ee_vel) / dt_sim
                        approach_acc2 += float(ee_acc @ ee_acc)
                        approach_accn += 1
                    _prev_ee_vel = ee_vel
                _prev_ee   = ee_pos.copy()
                _prev_qvel = qvel_true.copy()
            else:
                _prev_ee = _prev_ee_vel = _prev_qvel = None

            _, q_des = self._pos_ctrl.compute(
                q_curr, qdot_curr, ee_pos, ee_rot,
                cmd.target_pos, cmd.target_rot, J, qfrc_bias)
            ctrl[:7] = q_des

            # A "compliant phase" is any insertion/recovery or feedforward-bearing
            # command; it is realised by the operational-space controller (with
            # inertia shaping + LCS-MPC force feedforward).  Free-space tracking
            # always uses the Cartesian position controller.
            _compliant_phase = (cmd.use_impedance or cmd.use_lcs_mpc
                                or cmd.v_des is not None
                                or cmd.ctrl_mode in ('insertion', 'recovery'))

            if self._use_os and _compliant_phase:
                tau, q_des_os = self._os_ctrl.compute(
                    q_curr, qdot_curr, ee_pos, ee_rot,
                    cmd.target_pos, cmd.target_rot, J, qfrc_bias,
                    v_des=cmd.v_des,
                    a_des=cmd.a_des,
                    F_des=cmd.F_des,
                    mode=cmd.ctrl_mode,
                    M=M7,
                )
                ctrl[:7] = q_des_os
            else:
                tau, _ = self._pos_ctrl.compute(
                    q_curr, qdot_curr, ee_pos, ee_rot,
                    cmd.target_pos, cmd.target_rot, J, qfrc_bias)

            self._env.d.qfrc_applied[:7] = tau
            ctrl[7] = cmd.gripper_ctrl

            # 5. Grasp weld management
            if self._planner.grasp_should_activate:
                self._env.activate_grasp_weld()
                # Record T_grasp and switch peg_pos_source to "grasp_propagation"
                self._sensor.activate_grasp_propagation(belief.ee_pos, belief.ee_rot)
                belief.grasp_transform = self._sensor.grasp_transform
                belief.grasp_stability_status = "stable"
                self._planner.clear_grasp_flag()

            if self._planner.stage == Stage.RELEASE:
                if not weld_deactivated:
                    ts = self._env.get_true_state()
                    sids = self._env.site_ids
                    hole_z = self._env.d.site_xpos[sids["hole_entrance"]][2]
                    pre_release_depth = max(0.0, hole_z - ts.peg_tip_pos[2])
                    weld_deactivated = True
                self._env.deactivate_grasp_weld()

            # 6. Step simulation
            self._env.step(ctrl)

            if step_cb is not None:
                if step_cb() is False:
                    break

            # 7. Track metrics
            f_mag = float(np.linalg.norm(belief.external_force[:3]))
            peak_force = max(peak_force, f_mag)
            if belief.contact_detected:
                force_sum  += f_mag
                contact_steps += 1
            if (init_contact_f == 0.0 and belief.contact_detected
                    and self._planner.stage == Stage.INSERT):
                init_contact_f = f_mag

            if log_cb:
                log_cb(step, belief, cmd, ctrl)

            # 8. Terminal check
            if self._planner.stage in (Stage.DONE, Stage.FAILED):
                break

        # ── post-task metrics ────────────────────────────────────────────────
        ts = self._env.get_true_state()
        sids = self._env.site_ids
        hole_xpos = self._env.d.site_xpos[sids["hole_entrance"]]
        hole_z = hole_xpos[2]
        post_fall_depth = max(0.0, hole_z - ts.peg_tip_pos[2])
        # Final lateral pose error: peg-tip XY vs true hole-entrance XY
        final_pose_error = float(np.linalg.norm(ts.peg_tip_pos[:2] - hole_xpos[:2]))

        success = bool(
            self._planner.stage == Stage.DONE
            and pre_release_depth >= self._tc["stages"]["insertion_depth_goal"] * 0.8
        )
        avg_force = force_sum / max(contact_steps, 1)

        pi_ee   = self._planner._preinsert_final_ee_pos
        pi_goal = self._planner._preinsert_goal_pos
        preinsert_error = (float(np.linalg.norm(pi_ee - pi_goal))
                           if pi_ee is not None and pi_goal is not None else 0.0)

        # OCP metrics
        board_top_z = (self._tc["board"]["center"][2]
                       + self._tc["board"]["half_size"][2])
        ocp_res = self._planner._ocp_result
        if ocp_res is not None:
            ocp_solve_ms  = float(ocp_res.solve_time_ms)
            ocp_path_len  = float(compute_path_length(ocp_res.p_traj))
            ocp_clearance = float(compute_min_board_clearance(ocp_res.p_traj, board_top_z))
        else:
            ocp_solve_ms = ocp_path_len = ocp_clearance = 0.0

        # OSC metrics
        os_m = self._os_ctrl.get_metrics() if self._os_ctrl is not None else {}

        return {
            "success":             success,
            "stage":               int(self._planner.stage),
            "pre_release_depth":   float(pre_release_depth),
            "insertion_depth":     float(post_fall_depth),
            "peak_contact_force":  float(peak_force),
            "avg_contact_force":   float(avg_force),
            "recovery_attempts":   int(self._planner.recovery_attempts),
            "jam_events":          int(self._planner.jam_events),
            "failure_reason":      ("" if success else
                                    (self._planner.failure_reason or "incomplete")),
            "preinsert_error_m":   preinsert_error,
            "final_pose_error_m":  final_pose_error,
            "approach_path_len_m": float(approach_path),
            "approach_smoothness": float(np.sqrt(approach_acc2 / max(approach_accn, 1))),
            "max_joint_vel":       float(max_qvel),
            "max_joint_accel":     float(max_qaccel),
            "initial_contact_force": float(init_contact_f),
            "steps":               step + 1,
            "ocp_solve_ms":        ocp_solve_ms,
            "ocp_path_length_m":   ocp_path_len,
            "ocp_min_clearance_m": ocp_clearance,
            "os_mean_tracking_err": os_m.get("os_mean_tracking_err", 0.0),
            "os_peak_tracking_err": os_m.get("os_peak_tracking_err", 0.0),
            # Peg pose provenance at task end (for audit / acceptance criteria)
            "peg_pos_source_final": getattr(belief, "peg_pos_source", "unknown"),
            "grasp_stability":      getattr(belief, "grasp_stability_status", "unknown"),
        }
