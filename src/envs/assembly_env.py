"""Multi-task assembly environment.

Wraps the assembly scene and exposes a unified interface for multi-task
episodes with dynamic peg/hole selection.

Active peg and hole are switched via set_active_task(); all sensor, estimator,
and planner code then works against the currently active pair without changes.
"""

from __future__ import annotations
import numpy as np
import mujoco

from dataclasses import dataclass

from src.envs.scene_builder import (
    build_scene, get_body_ids, get_site_ids,
    ARM_JOINT_NAMES, EE_SITE_NAME,
    GRASP_EQ_NAME, GRASP_EQ_SQ_NAME, GRASP_EQ_RECT_NAME,
)


@dataclass
class TrueState:
    """Ground-truth state — only SensorWrapper may use this."""
    q: np.ndarray
    qdot: np.ndarray
    gripper_width: float
    ee_pos: np.ndarray
    ee_rot: np.ndarray
    peg_pos: np.ndarray
    peg_rot: np.ndarray
    peg_tip_pos: np.ndarray
    hole_pos: np.ndarray
    contact_force: np.ndarray   # (6,) wrench [fx,fy,fz,tx,ty,tz]
    time: float


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → MuJoCo quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _get_body_contact_wrench(m: mujoco.MjModel, d: mujoco.MjData,
                              body_id: int,
                              external_body_ids: set = None) -> np.ndarray:
    """Sum contact wrenches on body_id in world frame → (6,) [fx,fy,fz,tx,ty,tz].

    external_body_ids: if set, only count contacts where the other body is in
    this set (e.g. board only, excludes gripper contacts).
    """
    total_f = np.zeros(3)
    total_t = np.zeros(3)
    body_pos = d.xpos[body_id]
    cf_buf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        if b1 == body_id:
            other = b2
        elif b2 == body_id:
            other = b1
        else:
            continue
        if external_body_ids is not None and other not in external_body_ids:
            continue
        mujoco.mj_contactForce(m, d, i, cf_buf)
        normal = c.frame[:3]
        t1     = c.frame[3:6]
        t2     = c.frame[6:9]
        f_world = cf_buf[0] * normal + cf_buf[1] * t1 + cf_buf[2] * t2
        if b2 == body_id:
            f_world = -f_world
        total_f += f_world
        total_t += np.cross(c.pos - body_pos, f_world)
    return np.concatenate([total_f, total_t])


# Half-length (z half-extent) per peg body name
_PEG_HALF_LENGTH = {
    "peg":        0.070,
    "peg_square": 0.060,
    "peg_rect":   0.060,
}

# Grasp weld equality name per peg
_PEG_WELD = {
    "peg":        GRASP_EQ_NAME,
    "peg_square": GRASP_EQ_SQ_NAME,
    "peg_rect":   GRASP_EQ_RECT_NAME,
}

# Tip site name per peg
_PEG_TIP_SITE = {
    "peg":        "peg_tip",
    "peg_square": "peg_square_tip",
    "peg_rect":   "peg_rect_tip",
}

# Entrance site name per hole
_HOLE_ENTRANCE_SITE = {
    "round_hole":  "round_hole_entrance",
    "square_hole": "square_hole_entrance",
    "rect_slot":   "rect_slot_entrance",
}

# MuJoCo joint name per peg body name
_PEG_JOINT = {
    "peg":        "peg_joint",
    "peg_square": "peg_square_joint",
    "peg_rect":   "peg_rect_joint",
}

# Scene-config key per peg body name (for initial_pos lookup)
_PEG_CFG_KEY = {
    "peg":        "round",
    "peg_square": "square",
    "peg_rect":   "rect",
}


class AssemblyEnv:
    """Multi-shape assembly environment with dynamic task switching.

    Call set_active_task() before each sub-task; all sensor/estimator/controller
    code then works against the currently active peg/hole pair without changes.

    Usage
    -----
    env = AssemblyEnv(scene_cfg, task_cfg)
    env.reset()                                     # full reset (all pegs)
    env.set_active_task("peg",        "round_hole")
    # ... run sub-task loop ...
    env.set_active_task("peg_square", "square_hole")
    env.set_active_task("peg_rect",   "rect_slot")
    """

    def __init__(self, scene_cfg: dict, task_cfg: dict, seed: int = 0):
        self._scene_cfg = scene_cfg
        self._task_cfg  = task_cfg
        self.rng = np.random.default_rng(seed)

        self._m, self._d = build_scene(scene_cfg)
        self._body_ids_raw  = get_body_ids(self._m)
        self._all_site_ids  = get_site_ids(self._m)

        # Pre-resolve weld equality IDs
        self._weld_ids: dict[str, int] = {}
        for bname, wname in _PEG_WELD.items():
            eid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_EQUALITY, wname)
            self._weld_ids[bname] = eid

        # Pre-resolve qpos / dof addresses per peg
        self._peg_qpos_adr: dict[str, int] = {}
        self._peg_dof_adr:  dict[str, int] = {}
        for bname, jname in _PEG_JOINT.items():
            jid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._peg_qpos_adr[bname] = int(self._m.jnt_qposadr[jid])
            self._peg_dof_adr[bname]  = int(self._m.jnt_dofadr[jid])

        # Initial peg world positions (from scene config)
        self._initial_peg_pos: dict[str, np.ndarray] = {}
        for bname, ck in _PEG_CFG_KEY.items():
            self._initial_peg_pos[bname] = np.array(
                scene_cfg["pegs"][ck]["initial_pos"], dtype=float)

        # Active task (default: round peg + round hole)
        self._active_peg  = "peg"
        self._active_hole = "round_hole"

    # ── task switching ────────────────────────────────────────────────────────

    def set_active_task(self, peg_name: str, hole_name: str) -> None:
        """Switch the active peg/hole pair.  No physics changes."""
        self._active_peg  = peg_name
        self._active_hole = hole_name

    @staticmethod
    def peg_half_length(peg_name: str) -> float:
        return _PEG_HALF_LENGTH[peg_name]

    # ── properties mirroring PegInHoleEnv ────────────────────────────────────

    @property
    def m(self) -> mujoco.MjModel:
        return self._m

    @property
    def d(self) -> mujoco.MjData:
        return self._d

    @property
    def dt(self) -> float:
        return self._m.opt.timestep

    @property
    def body_ids(self) -> dict:
        return self._body_ids_raw

    @property
    def site_ids(self) -> dict:
        """Return site-id dict with 'peg_tip' and 'hole_entrance' aliased to active task."""
        ids = dict(self._all_site_ids)
        ids["peg_tip"]       = self._all_site_ids[_PEG_TIP_SITE[self._active_peg]]
        ids["hole_entrance"] = self._all_site_ids[_HOLE_ENTRANCE_SITE[self._active_hole]]
        return ids

    # ── simulation control ────────────────────────────────────────────────────

    def step(self, ctrl: np.ndarray) -> None:
        np.copyto(self._d.ctrl, ctrl)
        mujoco.mj_step(self._m, self._d)

    def reset(self) -> None:
        """Full reset: robot home, all pegs to initial positions, welds off."""
        mujoco.mj_resetData(self._m, self._d)

        # Robot home
        q_home = self._task_cfg["robot"]["home_qpos"]
        for i, jname in enumerate(ARM_JOINT_NAMES):
            jid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._d.qpos[self._m.jnt_qposadr[jid]] = q_home[i]

        self._d.ctrl[7] = self._task_cfg["robot"]["gripper_open_ctrl"]

        # Deactivate all welds
        for eid in self._weld_ids.values():
            if eid >= 0:
                self._d.eq_active[eid] = 0

        # Reset all pegs to initial positions
        for bname, ipos in self._initial_peg_pos.items():
            qadr = self._peg_qpos_adr[bname]
            dadr = self._peg_dof_adr[bname]
            self._d.qpos[qadr:qadr + 3] = ipos
            self._d.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            self._d.qvel[dadr:dadr + 6] = 0.0

        mujoco.mj_forward(self._m, self._d)

    # ── grasp weld (active peg only) ─────────────────────────────────────────

    def activate_grasp_weld(self) -> None:
        eid = self._weld_ids.get(self._active_peg, -1)
        if eid < 0:
            return
        hand_id = self._body_ids_raw["hand"]
        peg_id  = self._body_ids_raw[self._active_peg]

        pos_hand = self._d.xpos[hand_id].copy()
        rot_hand = self._d.xmat[hand_id].reshape(3, 3).copy()
        pos_peg  = self._d.xpos[peg_id].copy()
        rot_peg  = self._d.xmat[peg_id].reshape(3, 3).copy()

        rel_pos  = rot_hand.T @ (pos_peg - pos_hand)
        rel_quat = _mat_to_quat(rot_hand.T @ rot_peg)

        self._m.eq_data[eid, 0:3]  = [0.0, 0.0, 0.0]
        self._m.eq_data[eid, 3:6]  = rel_pos
        self._m.eq_data[eid, 6:10] = rel_quat
        self._d.eq_active[eid] = 1

    def deactivate_grasp_weld(self) -> None:
        eid = self._weld_ids.get(self._active_peg, -1)
        if eid >= 0:
            self._d.eq_active[eid] = 0

    # ── true state ───────────────────────────────────────────────────────────

    def get_true_state(self) -> TrueState:
        m, d = self._m, self._d
        bid  = self._body_ids_raw
        sids = self.site_ids   # dynamic aliases

        # Arm joints
        q = np.array([
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]]
            for jn in ARM_JOINT_NAMES
        ])
        qdot = np.array([
            d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]]
            for jn in ARM_JOINT_NAMES
        ])

        # Gripper
        fj1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        gw = float(d.qpos[m.jnt_qposadr[fj1_id]])

        # EE
        ee_sid = self._all_site_ids[EE_SITE_NAME]
        ee_pos = d.site_xpos[ee_sid].copy()
        ee_rot = d.site_xmat[ee_sid].reshape(3, 3).copy()

        # Active peg body
        peg_bid    = bid[self._active_peg]
        peg_pos    = d.xpos[peg_bid].copy()
        peg_rot    = d.xmat[peg_bid].reshape(3, 3).copy()
        peg_tip_pos = d.site_xpos[sids["peg_tip"]].copy()

        # Active hole entrance
        hole_pos = d.site_xpos[sids["hole_entrance"]].copy()

        # Contact force: active peg ↔ board (excludes gripper contacts)
        board_id = bid.get("board", -1)
        external_ids = {board_id} if board_id >= 0 else None
        contact_force = _get_body_contact_wrench(m, d, peg_bid, external_ids)

        return TrueState(
            q=q, qdot=qdot, gripper_width=gw,
            ee_pos=ee_pos, ee_rot=ee_rot,
            peg_pos=peg_pos, peg_rot=peg_rot, peg_tip_pos=peg_tip_pos,
            hole_pos=hole_pos,
            contact_force=contact_force,
            time=float(d.time),
        )

    # ── Jacobian / mass matrix ────────────────────────────────────────────────

    def get_ee_jacobian(self) -> np.ndarray:
        m, d = self._m, self._d
        sid  = self._all_site_ids[EE_SITE_NAME]
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        return np.vstack([jacp, jacr])

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        sid = self._all_site_ids[EE_SITE_NAME]
        pos = self._d.site_xpos[sid].copy()
        rot = self._d.site_xmat[sid].reshape(3, 3).copy()
        return pos, rot

    def get_mass_matrix(self) -> np.ndarray:
        m, d = self._m, self._d
        M_full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, M_full, d.qM)
        return M_full[:7, :7]
