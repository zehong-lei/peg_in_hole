"""Scripted stage-machine planner for peg-in-hole task.

Stages:
  0  MOVE_TO_PREGRASP   Move EE above peg
  1  GRASP              Descend and close gripper
  2  LIFT               Raise to lift height
  3  MOVE_TO_PREINSERT  Move to above hole
  4  ALIGN              Fine XY alignment over hole
  5  INSERT             Impedance-controlled insertion
  6  RELEASE            Open gripper
  7  RETREAT            Move EE up and back

The planner outputs a Command dict consumed by the task controller.
It uses only Observation / BeliefState — no ground truth.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np

from src.estimators.state_estimator import BeliefState
from .lcs_mpc import LCSMPC, ReducedLCSModel, N_CONTACTS
from .preinsert_ocp import PreInsertionOCP, OCPResult
from .trajectory_utils import TrajectoryTracker

# Outward spiral search offsets (dx, dy) in metres, relative to estimated hole centre.
# Centre (0,0) is tried first in normal descent; offsets 1..N are tried after each jam.
# This is the BASE pattern (6 mm max cardinal radius); the planner scales it to
# contact_recovery.spiral_max_radius at construction time.
_SPIRAL_BASE_RADIUS = 0.006
_SPIRAL_OFFSETS = np.array([
    [ 0.000,  0.000],   # 0 centre — tried in initial descent
    [ 0.002,  0.000],   # ring 1 (2 mm), cardinal
    [ 0.000,  0.002],
    [-0.002,  0.000],
    [ 0.000, -0.002],
    [ 0.002,  0.002],   # ring 1 diagonal (≈ 2.8 mm)
    [-0.002,  0.002],
    [-0.002, -0.002],
    [ 0.002, -0.002],
    [ 0.004,  0.000],   # ring 2 (4 mm), cardinal
    [ 0.000,  0.004],
    [-0.004,  0.000],
    [ 0.000, -0.004],
    [ 0.004,  0.004],   # ring 2 diagonal (≈ 5.6 mm)
    [-0.004,  0.004],
    [-0.004, -0.004],
    [ 0.004, -0.004],
    [ 0.006,  0.000],   # ring 3 (6 mm), cardinal
    [ 0.000,  0.006],
    [-0.006,  0.000],
    [ 0.000, -0.006],
], dtype=np.float64)


class Stage(IntEnum):
    MOVE_TO_PREGRASP = 0
    GRASP = 1
    LIFT = 2
    MOVE_TO_PREINSERT = 3
    ALIGN = 4
    INSERT = 5
    RELEASE = 6
    RETREAT = 7
    DONE = 8
    FAILED = 9


@dataclass
class Command:
    """Output of planner consumed by the controller."""
    target_pos: np.ndarray         # (3,) desired EE position
    target_rot: np.ndarray         # (3,3) desired EE rotation
    gripper_ctrl: float            # 0=closed, 255=open
    use_impedance: bool = False    # True → impedance; False → position control
    use_lcs_mpc: bool = False      # True → LCS-MPC force feedforward via compute_insert_mpc
    F_des: Optional[np.ndarray] = None  # (6,) desired EE wrench [N]
    v_des: Optional[np.ndarray] = None  # (3,) desired EE velocity for OS feedforward
    a_des: Optional[np.ndarray] = None  # (3,) desired EE acceleration for inertia-shaping feedforward
    ctrl_mode: str = 'free_space'       # 'free_space' | 'insertion' | 'recovery'
    stage: Stage = Stage.MOVE_TO_PREGRASP


# Desired EE orientation: Z axis pointing down (gripper down configuration)
_ROT_DOWN = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])   # local Z = world -Z, local Y = world -Y


class ScriptedPlanner:
    """Stage machine.  Call update() once per control cycle.

    Parameters
    ----------
    task_cfg  : dict   (task.yaml)
    ctrl_cfg  : dict   (controller.yaml gripper sub-section)
    """

    def __init__(self, task_cfg: dict, ctrl_cfg: dict):
        self._tc = task_cfg
        self._cc = ctrl_cfg

        self._stage = Stage.MOVE_TO_PREGRASP
        self._stage_start_time: float = 0.0
        self._timeouts = task_cfg["stages"]["stage_timeout"]

        # Peg half-length — dynamically overridable per sub-task
        self._peg_half_length: float = task_cfg.get("peg", {}).get("half_length", 0.070)

        # Cached goals (set at stage start)
        self._goal_pos: np.ndarray = np.zeros(3)
        self._goal_rot: np.ndarray = _ROT_DOWN.copy()
        self._gripper_ctrl: float = task_cfg["robot"]["gripper_open_ctrl"]

        # Insertion tracking
        self._insert_start_z: Optional[float] = None
        self._grasp_activated: bool = False

        # Tolerance shortcuts
        self._pos_tol = task_cfg["stages"]["pos_tol"]
        self._rot_tol = task_cfg["stages"]["rot_tol"]
        self._xy_tol = task_cfg["stages"]["align_xy_tol"]

        # Contact-recovery state machine
        self._rec_state: str = "descend"      # "descend" | "retract" | "search"
        self._rec_contact_steps: int = 0      # consecutive steps satisfying jam condition
        self._rec_attempts: int = 0           # retract-search-retry cycles used
        self._rec_search_idx: int = 1         # next _SPIRAL_OFFSETS index to try
        self._rec_retract_z: float = 0.0      # target z during retract
        self._rec_current_offset: np.ndarray = np.zeros(2)  # current XY offset from hole
        self._rec_jam_start_z: float = 0.0    # EE z when lateral force first exceeded threshold

        # Spiral search pattern, scaled from the base template to the configured
        # max radius (contact_recovery.spiral_max_radius, default = base 6 mm).
        _rec_cfg0 = task_cfg.get("contact_recovery", {})
        _spiral_r = float(_rec_cfg0.get("spiral_max_radius", _SPIRAL_BASE_RADIUS))
        self._spiral_offsets = _SPIRAL_OFFSETS * (_spiral_r / _SPIRAL_BASE_RADIUS)

        # Benchmark diagnostics
        self._fail_reason: str = ""           # set whenever stage → FAILED
        self._jam_events: int = 0             # count of jam / force-abort onsets

        # EE-space OCP for MOVE_TO_PREINSERT
        p4cfg = task_cfg.get("preinsert_ocp", {})
        self._p4_enabled: bool = p4cfg.get("enabled", False)
        self._ocp: Optional[PreInsertionOCP] = None
        self._ocp_tracker: Optional[TrajectoryTracker] = None
        self._ocp_result: Optional[OCPResult] = None
        self._preinsert_final_ee_pos: Optional[np.ndarray] = None
        self._preinsert_goal_pos: Optional[np.ndarray] = None
        if self._p4_enabled:
            self._ocp = PreInsertionOCP(task_cfg)

        # LCS-MPC state
        p3cfg = task_cfg.get("lcs_mpc", {})
        self._p3_enabled: bool = p3cfg.get("enabled", False)
        self._mpc: Optional[LCSMPC] = None
        self._mpc_u0: np.ndarray = np.array([0., 0., 1.])   # cached MPC control
        self._mpc_step: int = 0
        self._mpc_freq: int = p3cfg.get("mpc", {}).get("mpc_freq_ratio", 10)
        self._mpc_failures: int = 0
        self._prev_ee_pos: Optional[np.ndarray] = None
        self._mpc_solve_times: list = []   # ms per solve (for benchmarking)
        if self._p3_enabled:
            lcs_cfg = dict(p3cfg.get("lcs_model", {}))
            lcs_cfg["clearance"] = task_cfg["board"]["hole_radius"] - task_cfg["peg"]["radius"]
            mpc_cfg = dict(p3cfg.get("mpc", {}))
            # z_goal: EE descent from insert_start until success
            # = preinsert_z_offset + insertion_depth_goal - peg_half_length
            tc = task_cfg["stages"]
            mpc_cfg["z_goal"] = (tc["preinsert_z_offset"]
                                 + tc["insertion_depth_goal"]
                                 - task_cfg["peg"]["half_length"])
            self._mpc = LCSMPC(ReducedLCSModel(lcs_cfg), mpc_cfg)

    def set_peg_half_length(self, hl: float) -> None:
        """Override peg half-length for the current sub-task."""
        self._peg_half_length = hl

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def recovery_attempts(self) -> int:
        return self._rec_attempts

    @property
    def failure_reason(self) -> str:
        return self._fail_reason

    @property
    def jam_events(self) -> int:
        return self._jam_events

    def _set_failed(self, reason: str) -> None:
        """Transition to FAILED and record the cause (first cause wins)."""
        self._stage = Stage.FAILED
        if not self._fail_reason:
            self._fail_reason = reason

    def reset(self, t: float = 0.0) -> None:
        """Reset planner state.  Pass current sim time for correct timeout tracking."""
        self._stage = Stage.MOVE_TO_PREGRASP
        self._stage_start_time = t
        self._insert_start_z = None
        self._grasp_activated = False
        self._gripper_ctrl = self._tc["robot"]["gripper_open_ctrl"]
        self._rec_state = "descend"
        self._rec_contact_steps = 0
        self._rec_attempts = 0
        self._rec_search_idx = 1
        self._rec_retract_z = 0.0
        self._rec_current_offset = np.zeros(2)
        self._rec_jam_start_z = 0.0
        self._fail_reason = ""
        self._jam_events = 0
        # LCS-MPC reset
        self._mpc_u0 = np.array([0., 0., 1.])
        self._mpc_step = 0
        self._mpc_failures = 0
        self._prev_ee_pos = None
        self._mpc_solve_times = []
        if self._mpc is not None:
            self._mpc._z_warm = None
            self._mpc._last_lambda = np.zeros(N_CONTACTS)
        # EE-space OCP reset
        self._ocp_tracker = None
        self._ocp_result  = None
        self._preinsert_final_ee_pos = None
        self._preinsert_goal_pos     = None
        if self._ocp is not None:
            self._ocp._z_warm = None

    def update(self, belief: BeliefState) -> Command:
        """Compute planner command from current belief state."""
        tc = self._tc["stages"]
        t = belief.time

        # Timeout check
        if self._stage not in (Stage.DONE, Stage.FAILED):
            idx = int(self._stage)
            if idx < len(self._timeouts):
                if t - self._stage_start_time > self._timeouts[idx]:
                    self._set_failed(f"timeout_stage{idx}")

        stage = self._stage

        if stage == Stage.MOVE_TO_PREGRASP:
            peg_pos = belief.peg_pos
            target = peg_pos + np.array([0.0, 0.0,
                self._peg_half_length + tc["pregrasp_z_offset"]])
            self._goal_pos = target
            cmd = self._make_cmd(target, gripper=self._tc["robot"]["gripper_open_ctrl"])
            if self._reached_pos(belief.ee_pos, target, self._pos_tol * 2):
                self._advance_stage(t)
            return cmd

        elif stage == Stage.GRASP:
            # Descend to peg centre height, close gripper.
            # ee_site is 0.103 m BELOW hand, so target[2] = peg centre z
            # places the fingertips directly at the peg midpoint.
            peg_pos = belief.peg_pos
            target = peg_pos.copy()
            target[2] = peg_pos[2]  # EE site at peg centre
            self._goal_pos = target
            # Close gripper progressively
            close_target = self._tc["robot"]["gripper_close_ctrl"]
            self._gripper_ctrl = max(self._gripper_ctrl
                                     - self._cc["gripper"]["close_speed_ctrl_per_step"],
                                     close_target)
            cmd = self._make_cmd(target, gripper=self._gripper_ctrl)
            # Advance when gripper closed and EE near peg
            gripper_done = abs(self._gripper_ctrl - close_target) < 1.0
            ee_near = self._reached_pos(belief.ee_pos, target, self._pos_tol * 1.5)
            if gripper_done and ee_near:
                self._grasp_activated = True   # signal env to activate weld
                self._advance_stage(t)
            return cmd

        elif stage == Stage.LIFT:
            target = belief.ee_pos.copy()
            target[2] = tc["lift_z"]
            self._goal_pos = target
            cmd = self._make_cmd(target, gripper=self._tc["robot"]["gripper_close_ctrl"])
            if self._reached_pos(belief.ee_pos, target, self._pos_tol):
                self._advance_stage(t)
            return cmd

        elif stage == Stage.MOVE_TO_PREINSERT:
            hole = belief.hole_pos
            preinsert_goal = hole.copy()
            preinsert_goal[2] = hole[2] + tc["preinsert_z_offset"]
            gripper = self._tc["robot"]["gripper_close_ctrl"]

            if self._p4_enabled and self._ocp is not None:
                # ── EE-space OCP ──────────────────────────────────────────────
                if self._ocp_tracker is None:
                    result = self._ocp.solve(belief.ee_pos, preinsert_goal, hole)
                    self._ocp_result = result
                    if result.success:
                        self._ocp_tracker = TrajectoryTracker(
                            result.p_traj, self._ocp.dt,
                            v_traj=result.v_traj)   # store velocities for feedforward
                        self._ocp_tracker.reset(t)

                if self._ocp_tracker is not None:
                    if self._ocp_tracker.is_done(t):
                        ocp_target = preinsert_goal
                        v_des_cmd  = None
                        a_des_cmd  = None
                    else:
                        ocp_target = self._ocp_tracker.get_target(t)
                        v_des_cmd  = self._ocp_tracker.get_velocity(t)
                        a_des_cmd  = self._ocp_tracker.get_acceleration(t)
                    self._goal_pos = ocp_target
                    cmd = self._make_cmd(ocp_target, gripper=gripper,
                                         v_des=v_des_cmd,
                                         a_des=a_des_cmd,
                                         ctrl_mode='free_space')
                    if self._reached_pos(belief.ee_pos, preinsert_goal,
                                         self._pos_tol):
                        self._preinsert_final_ee_pos = belief.ee_pos.copy()
                        self._preinsert_goal_pos = preinsert_goal.copy()
                        self._advance_stage(t)
                    return cmd
                # EE OCP failed → fall through to scripted behaviour

            # ── Scripted fallback ─────────────────────────────────────────────
            self._goal_pos = preinsert_goal
            cmd = self._make_cmd(preinsert_goal, gripper=gripper)
            if self._reached_pos(belief.ee_pos, preinsert_goal, self._pos_tol):
                self._preinsert_final_ee_pos = belief.ee_pos.copy()
                self._preinsert_goal_pos = preinsert_goal.copy()
                self._advance_stage(t)
            return cmd

        elif stage == Stage.ALIGN:
            hole = belief.hole_pos
            target = belief.ee_pos.copy()
            target[0] = hole[0]
            target[1] = hole[1]
            target[2] = hole[2] + tc["preinsert_z_offset"]
            self._goal_pos = target
            cmd = self._make_cmd(target, gripper=self._tc["robot"]["gripper_close_ctrl"])
            xy_err = np.linalg.norm(belief.ee_pos[:2] - hole[:2])
            if xy_err < self._xy_tol:
                self._insert_start_z = belief.ee_pos[2]
                self._advance_stage(t)
            return cmd

        elif stage == Stage.INSERT:
            hole = belief.hole_pos
            gripper = self._tc["robot"]["gripper_close_ctrl"]
            rec_cfg = self._tc.get("contact_recovery", {})
            rec_enabled = rec_cfg.get("enabled", False)

            peg_hl = self._peg_half_length
            ee_goal = tc["preinsert_z_offset"] + tc["insertion_depth_goal"] - peg_hl
            descent_per_step = tc["insert_descend_speed"] * self._tc["sim"]["dt"]
            depth_goal = tc["insertion_depth_goal"]

            def _success_reached() -> bool:
                # Primary: check actual peg-tip depth from belief peg position.
                # This is robust to grasp offset errors between EE and peg CoM.
                peg_tip_z = belief.peg_pos[2] - peg_hl
                hole_top_z = belief.hole_pos[2]
                if (hole_top_z - peg_tip_z) >= depth_goal:
                    return True
                # Fallback: EE-descent-based check (for compatibility)
                if self._insert_start_z is None:
                    return False
                return (self._insert_start_z - belief.ee_pos[2]) > ee_goal

            if not rec_enabled:
                # Simple descent, no recovery
                target = belief.ee_pos.copy()
                target[0] = hole[0]
                target[1] = hole[1]
                target[2] = self._goal_pos[2] - descent_per_step
                self._goal_pos = target
                cmd = self._make_cmd(target, gripper=gripper,
                                     ctrl_mode='insertion')
                if np.linalg.norm(belief.external_force[:3]) > tc["max_force_abort"]:
                    self._jam_events += 1
                    self._set_failed("force_abort")
                elif _success_reached():
                    self._advance_stage(t)
                return cmd

            # ── LCS-MPC + force feedforward ───────────────────────────────────
            # When rec_state is retract/search, delegate to recovery code below.
            if (self._p3_enabled and self._mpc is not None
                    and self._rec_state == "descend"):
                return self._insert_lcs_mpc(belief, hole, gripper,
                                            descent_per_step, tc,
                                            _success_reached)

            # ── Contact-recovery state machine ────────────────────────────────
            if self._rec_state == "descend":
                target = np.array([
                    hole[0] + self._rec_current_offset[0],
                    hole[1] + self._rec_current_offset[1],
                    self._goal_pos[2] - descent_per_step,
                ])
                self._goal_pos = target
                cmd = self._make_cmd(target, gripper=gripper,
                                     ctrl_mode='insertion')

                # Force threshold: immediately retract when contact force is excessive.
                # With recovery enabled, redirect to retract instead of hard failure.
                if np.linalg.norm(belief.external_force[:3]) > tc["max_force_abort"]:
                    self._jam_events += 1
                    if self._rec_attempts >= rec_cfg["max_attempts"]:
                        self._set_failed("force_abort_max_recovery")
                    else:
                        self._rec_retract_z = belief.ee_pos[2] + rec_cfg["retract_height"]
                        self._rec_state = "retract"
                        self._rec_contact_steps = 0
                    return cmd

                # Jamming detection: sustained lateral force AND depth stagnation.
                # Peg rubbing the rim while sliding in is NOT a jam — only trigger
                # recovery when the EE has stopped making downward progress.
                lat_f = np.linalg.norm(belief.external_force[:2])
                if lat_f > rec_cfg["lateral_force_threshold"] and belief.contact_detected:
                    if self._rec_contact_steps == 0:
                        self._rec_jam_start_z = belief.ee_pos[2]  # save z at onset
                    self._rec_contact_steps += 1
                else:
                    self._rec_contact_steps = 0

                if self._rec_contact_steps >= rec_cfg["jam_window_steps"]:
                    # Check whether EE actually descended during the jam window
                    depth_dropped = self._rec_jam_start_z - belief.ee_pos[2]
                    min_progress = descent_per_step * rec_cfg["jam_window_steps"] * 0.3
                    if depth_dropped < min_progress:
                        # Truly stuck — trigger recovery
                        self._jam_events += 1
                        if self._rec_attempts >= rec_cfg["max_attempts"]:
                            self._set_failed("jam_max_recovery")
                            return cmd
                        self._rec_retract_z = belief.ee_pos[2] + rec_cfg["retract_height"]
                        self._rec_state = "retract"
                        self._rec_contact_steps = 0
                    else:
                        # Still progressing (rim contact while sliding in) — reset counter
                        self._rec_contact_steps = 0

                elif _success_reached():
                    self._advance_stage(t)

                return cmd

            elif self._rec_state == "retract":
                target = np.array([
                    hole[0] + self._rec_current_offset[0],
                    hole[1] + self._rec_current_offset[1],
                    self._rec_retract_z,
                ])
                self._goal_pos = target
                cmd = self._make_cmd(target, gripper=gripper,
                                     ctrl_mode='recovery')
                if belief.ee_pos[2] >= self._rec_retract_z - 0.001:
                    # Reached retract height; advance to next spiral offset
                    self._rec_attempts += 1
                    if self._rec_search_idx >= len(self._spiral_offsets):
                        self._set_failed("search_exhausted")
                        return cmd
                    self._rec_state = "search"
                return cmd

            else:  # "search"
                new_off = self._spiral_offsets[self._rec_search_idx]
                target = np.array([
                    hole[0] + new_off[0],
                    hole[1] + new_off[1],
                    self._rec_retract_z,
                ])
                self._goal_pos = target
                cmd = self._make_cmd(target, gripper=gripper,
                                     ctrl_mode='recovery')
                xy_err = np.linalg.norm(belief.ee_pos[:2] - target[:2])
                if xy_err < self._xy_tol * 1.5:
                    # EE settled at new XY — begin descending from here
                    self._rec_current_offset = new_off.copy()
                    self._rec_search_idx += 1
                    self._goal_pos[2] = self._rec_retract_z
                    self._rec_state = "descend"
                    self._rec_contact_steps = 0
                    self._rec_jam_start_z = 0.0
                return cmd

        elif stage == Stage.RELEASE:
            target = belief.ee_pos.copy()
            self._goal_pos = target
            self._gripper_ctrl = min(self._gripper_ctrl
                                     + self._cc["gripper"]["close_speed_ctrl_per_step"],
                                     self._tc["robot"]["gripper_open_ctrl"])
            cmd = self._make_cmd(target, gripper=self._gripper_ctrl)
            if abs(self._gripper_ctrl - self._tc["robot"]["gripper_open_ctrl"]) < 1.0:
                self._grasp_activated = False
                self._advance_stage(t)
            return cmd

        elif stage == Stage.RETREAT:
            target = belief.ee_pos.copy()
            target[2] = tc["lift_z"]
            self._goal_pos = target
            cmd = self._make_cmd(target, gripper=self._tc["robot"]["gripper_open_ctrl"])
            if self._reached_pos(belief.ee_pos, target, self._pos_tol):
                self._stage = Stage.DONE
            return cmd

        else:   # DONE or FAILED
            cmd = self._make_cmd(belief.ee_pos.copy(),
                                 gripper=self._tc["robot"]["gripper_open_ctrl"])
            return cmd

    # ── LCS-MPC helpers ──────────────────────────────────────────────────────

    def _get_lcs_state(self, belief: BeliefState) -> np.ndarray:
        """Extract 6-dim LCS state from belief.

        x = [ex, ey, z_descent, vx, vy, vz_descent]
        z_descent  = insert_start_z - ee_z  (increases as EE descends)
        vz_descent > 0 means EE going deeper
        """
        ee = belief.ee_pos
        hole = belief.hole_pos
        ex = ee[0] - hole[0]
        ey = ee[1] - hole[1]
        z_desc = (self._insert_start_z - ee[2]) if self._insert_start_z else 0.0

        dt_sim = self._tc["sim"]["dt"]
        if self._prev_ee_pos is not None:
            dv = (ee - self._prev_ee_pos) / dt_sim
            vx = float(dv[0])
            vy = float(dv[1])
            vz = float(-dv[2])  # world -z = descent direction
        else:
            vx = vy = vz = 0.0

        return np.array([ex, ey, z_desc, vx, vy, vz])

    def _insert_lcs_mpc(self, belief: BeliefState, hole: np.ndarray,
                        gripper: float, descent_per_step: float,
                        tc: dict, success_fn) -> Command:
        """INSERT with LCS-MPC force feedforward; falls back to contact-recovery on jam."""
        rec_cfg = self._tc.get("contact_recovery", {})

        # Force-abort → retract
        force_mag = np.linalg.norm(belief.external_force[:3])
        if force_mag > tc["max_force_abort"]:
            self._jam_events += 1
            max_att = rec_cfg.get("max_attempts", 8)
            if self._rec_attempts >= max_att:
                self._set_failed("force_abort_max_recovery")
                return self._make_cmd(belief.ee_pos.copy(), gripper=gripper,
                                      ctrl_mode='recovery')
            self._rec_retract_z = (belief.ee_pos[2]
                                   + rec_cfg.get("retract_height", 0.003))
            self._rec_state = "retract"
            self._rec_contact_steps = 0
            self._mpc._z_warm = None   # reset warm-start after recovery
            return self._make_cmd(belief.ee_pos.copy(), gripper=gripper,
                                  ctrl_mode='recovery')

        # ── LCS state ──────────────────────────────────────────────────────
        x_lcs = self._get_lcs_state(belief)
        self._prev_ee_pos = belief.ee_pos.copy()

        # ── run MPC at reduced frequency ──────────────────────────────────
        if self._mpc_step % self._mpc_freq == 0:
            u0, _X, ok = self._mpc.solve(x_lcs, u_prev=self._mpc_u0)
            if self._mpc.last_info is not None:
                self._mpc_solve_times.append(self._mpc.last_info.solve_time_ms)
            if ok:
                self._mpc_u0 = u0
                self._mpc_failures = 0
            else:
                self._mpc_failures += 1
        self._mpc_step += 1

        # Disable LCS-MPC and fall back to contact-recovery if solver keeps failing
        p3cfg = self._tc.get("lcs_mpc", {})
        max_fail = p3cfg.get("mpc_max_failures", 10)
        if self._mpc_failures >= max_fail:
            self._p3_enabled = False   # disable for this episode
            return self._make_cmd(belief.ee_pos.copy(), gripper=gripper,
                                  impedance=True, ctrl_mode='insertion')

        # ── descent target ────────────────────────────────────────────────
        target = np.array([
            hole[0] + self._rec_current_offset[0],
            hole[1] + self._rec_current_offset[1],
            self._goal_pos[2] - descent_per_step,
        ])
        self._goal_pos = target

        # ── MPC force feedforward (world frame) ────────────────────────────
        # u0 = [Fx_world, Fy_world, Fz_insert]
        # Fz_insert > 0 → world -z (downward)
        F_des = np.array([
            self._mpc_u0[0],
            self._mpc_u0[1],
            -self._mpc_u0[2],   # convert LCS insertion direction to world -z
            0., 0., 0.,
        ])

        # ── Jamming detection safety fallback ─────────────────────────────
        lat_f = np.linalg.norm(belief.external_force[:2])
        lat_thresh = rec_cfg.get("lateral_force_threshold", 1.5)
        jam_window = rec_cfg.get("jam_window_steps", 50)

        if lat_f > lat_thresh and belief.contact_detected:
            if self._rec_contact_steps == 0:
                self._rec_jam_start_z = belief.ee_pos[2]
            self._rec_contact_steps += 1
        else:
            self._rec_contact_steps = 0

        if self._rec_contact_steps >= jam_window:
            depth_dropped = self._rec_jam_start_z - belief.ee_pos[2]
            min_prog = descent_per_step * jam_window * 0.3
            if depth_dropped < min_prog:
                self._jam_events += 1
                max_att = rec_cfg.get("max_attempts", 8)
                if self._rec_attempts >= max_att:
                    self._set_failed("jam_max_recovery")
                    return self._make_cmd(target, gripper=gripper,
                                         use_lcs_mpc=True, F_des=F_des,
                                         ctrl_mode='insertion')
                # Trigger retract
                self._rec_retract_z = belief.ee_pos[2] + rec_cfg.get("retract_height", 0.003)
                self._rec_state = "retract"
                self._rec_contact_steps = 0
                self._mpc._z_warm = None   # reset MPC warm-start after retract
            else:
                self._rec_contact_steps = 0

        if success_fn():
            self._advance_stage(belief.time)

        return self._make_cmd(target, gripper=gripper,
                              use_lcs_mpc=True, F_des=F_des,
                              ctrl_mode='insertion')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_cmd(self, pos: np.ndarray, gripper: float,
                  impedance: bool = False,
                  use_lcs_mpc: bool = False,
                  F_des: Optional[np.ndarray] = None,
                  v_des: Optional[np.ndarray] = None,
                  a_des: Optional[np.ndarray] = None,
                  ctrl_mode: str = 'free_space') -> Command:
        return Command(
            target_pos=pos.copy(),
            target_rot=_ROT_DOWN.copy(),
            gripper_ctrl=float(gripper),
            use_impedance=impedance,
            use_lcs_mpc=use_lcs_mpc,
            F_des=F_des.copy() if F_des is not None else None,
            v_des=v_des.copy() if v_des is not None else None,
            a_des=a_des.copy() if a_des is not None else None,
            ctrl_mode=ctrl_mode,
            stage=self._stage,
        )

    def _advance_stage(self, t: float) -> None:
        self._stage = Stage(int(self._stage) + 1)
        self._stage_start_time = t

    @staticmethod
    def _reached_pos(curr: np.ndarray, target: np.ndarray, tol: float) -> bool:
        return np.linalg.norm(curr - target) < tol

    @property
    def grasp_should_activate(self) -> bool:
        """True when planner just completed grasp stage; env activates weld."""
        return self._grasp_activated

    def clear_grasp_flag(self) -> None:
        self._grasp_activated = False
