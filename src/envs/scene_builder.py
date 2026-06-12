"""Tabletop multi-shape assembly scene builder.

Scene layout
------------
  Left (x≈0.40, y≈[-0.20, 0.00]):  parts staging area
    · round_peg   (cylinder, body name "peg")
    · square_peg  (box body "peg_square")
    · rect_peg    (box body "peg_rect")

  Right (x≈0.55, y≈0.10):  assembly board (fixed, sits on table)
    · round_hole  pocket (4-wall frame + base stopper)
    · square_hole pocket
    · rect_slot   pocket

Canonical names
---------------
  Body "peg", joint "peg_joint", sites "peg_tip"/"peg_top", equality
  "grasp_weld", and site "hole_entrance" (= round_hole_entrance) are the
  names the mainline pipeline drives the round peg through.
"""

from pathlib import Path
import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).parents[2]
_PANDA_XML = str(_REPO_ROOT / "assets" / "mujoco_menagerie"
                 / "franka_emika_panda" / "panda.xml")

ARM_JOINT_NAMES    = [f"joint{i}" for i in range(1, 8)]
EE_SITE_NAME       = "ee_site"
GRASP_EQ_NAME      = "grasp_weld"
GRASP_EQ_SQ_NAME   = "grasp_weld_sq"
GRASP_EQ_RECT_NAME = "grasp_weld_rect"

# Board colour (slate-blue), land colour (lighter slate)
_BOARD_RGBA = [0.40, 0.50, 0.62, 1.0]


def _cam_quat_lookat(pos, target, up=None):
    """Compute MuJoCo camera quaternion (w,x,y,z) for a lookat camera.

    MuJoCo camera convention: camera looks along -z_cam, +y_cam is image up.
    If up is None, world-y is used for near-vertical cameras (|forward_z|>0.9)
    and world-z is used otherwise.
    """
    pos    = np.asarray(pos,    dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - pos
    forward /= np.linalg.norm(forward)

    if up is None:
        up = np.array([0.0, 1.0, 0.0]) if abs(forward[2]) > 0.9 \
             else np.array([0.0, 0.0, 1.0])
    else:
        up = np.asarray(up, dtype=float)

    # Camera axes in world frame
    z_cam = -forward
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    x_cam = right
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    # Build rotation matrix and convert to quaternion
    R = np.column_stack([x_cam, y_cam, z_cam])
    s = float(np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]))
    w = 0.5 * s
    x = (R[2, 1] - R[1, 2]) / (2.0 * s)
    y = (R[0, 2] - R[2, 0]) / (2.0 * s)
    z = (R[1, 0] - R[0, 1]) / (2.0 * s)
    return [w, x, y, z]


_HARD_CONTACT = dict(
    solref=[0.002, 1.0],
    solimp=[0.999, 0.9999, 0.0001, 0.5, 2.0],
    friction=[0.8, 0.005, 0.0001],
)


def _set_hard(g) -> None:
    g.solref  = [0.002, 1.0]
    g.solimp  = [0.999, 0.9999, 0.0001, 0.5, 2.0]
    g.friction = [0.8, 0.005, 0.0001]


def _box(body, name, pos, size, rgba) -> None:
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos  = list(pos)
    g.size = list(size)
    g.rgba = list(rgba)
    _set_hard(g)


def _add_hole_pocket(board, prefix, hcx, hcy,
                     hhx, hhy, ohx, ohy,
                     wall_cz, wall_hz, rgba) -> None:
    """Add 4 box walls forming a rectangular hole pocket at (hcx, hcy).

    Parameters
    ----------
    hcx/hcy   : hole centre x/y in board-local frame
    hhx/hhy   : hole half-extents (inner clear region)
    ohx/ohy   : outer half-extents of the wall cell
    wall_cz   : z centre of the wall geoms (board-local)
    wall_hz   : half-height of the wall geoms
    """
    # Left wall  (covers full ohy in y, from -ohx to -hhx in x)
    _box(board, f"{prefix}_wall_L",
         [hcx - (ohx + hhx) / 2,  hcy,               wall_cz],
         [(ohx - hhx) / 2,         ohy,               wall_hz], rgba)
    # Right wall
    _box(board, f"{prefix}_wall_R",
         [hcx + (ohx + hhx) / 2,  hcy,               wall_cz],
         [(ohx - hhx) / 2,         ohy,               wall_hz], rgba)
    # Front wall (covers hhx in x, from +hhy to +ohy in y)
    _box(board, f"{prefix}_wall_F",
         [hcx,                     hcy + (ohy + hhy) / 2, wall_cz],
         [hhx,                     (ohy - hhy) / 2,        wall_hz], rgba)
    # Back wall
    _box(board, f"{prefix}_wall_B",
         [hcx,                     hcy - (ohy + hhy) / 2, wall_cz],
         [hhx,                     (ohy - hhy) / 2,        wall_hz], rgba)


def _add_assembly_board(world: mujoco.MjSpec, cfg: dict) -> None:
    """Build the assembly board with round, square, and rect hole pockets."""
    bc = cfg["assembly_board"]
    bcx, bcy, bcz = bc["center"]
    bhx, bhy, bhz = bc["half_size"]    # bhz = 0.025
    hole_depth     = bc["hole_depth"]  # 0.040

    # Board-local z coordinates
    floor_z  = bhz - hole_depth          # z of hole floors  = -0.015
    base_cz  = (-bhz + floor_z) / 2     # base slab centre   = -0.020
    base_hz  = (floor_z - (-bhz)) / 2   # base slab half-z   = 0.005  (10 mm thick)
    wall_cz  = (floor_z + bhz) / 2      # wall centre z      = +0.005
    wall_hz  = (bhz - floor_z) / 2      # wall half-z        = 0.020  (40 mm tall)

    board = world.add_body()
    board.name = "board"
    board.pos  = [bcx, bcy, bcz]

    # ── Base slab (hole floors + structural base) ─────────────────────────
    _box(board, "board_base",
         [0.0, 0.0, base_cz], [bhx, bhy, base_hz], _BOARD_RGBA)

    # ── Land between and around hole cells ────────────────────────────────
    # Cell outer half (uniform for all holes in x): ohx = ohy = 0.022
    ohx = ohy = 0.022
    # Hole x-offsets
    hcx_r, hcx_s, hcx_rs = (cfg["holes"]["round"]["local_x_offset"],
                              cfg["holes"]["square"]["local_x_offset"],
                              cfg["holes"]["rect"]["local_x_offset"])

    # Left edge land: x ∈ [-bhx, hcx_r - ohx] = [-0.100, -0.082]
    # centre = (-0.100 + -0.082)/2 = -0.091,  half = 0.009
    _box(board, "land_left",
         [(-bhx + hcx_r - ohx) / 2,   0.0, wall_cz],
         [(hcx_r - ohx + bhx) / 2,    bhy, wall_hz], _BOARD_RGBA)

    # Gap between round and square cells: x ∈ [-0.038, -0.022], width=0.016
    x_r_right = hcx_r + ohx   # -0.038
    x_s_left  = hcx_s - ohx   # -0.022
    _box(board, "land_mid1",
         [(x_r_right + x_s_left) / 2, 0.0, wall_cz],
         [(x_s_left - x_r_right) / 2, bhy, wall_hz], _BOARD_RGBA)

    # Gap between square and rect cells: x ∈ [+0.022, +0.038]
    x_s_right = hcx_s  + ohx   # +0.022
    x_rs_left = hcx_rs - ohx   # +0.038
    _box(board, "land_mid2",
         [(x_s_right + x_rs_left) / 2, 0.0, wall_cz],
         [(x_rs_left - x_s_right) / 2, bhy, wall_hz], _BOARD_RGBA)

    # Right edge land: x ∈ [+0.082, +0.100]
    _box(board, "land_right",
         [(hcx_rs + ohx + bhx) / 2, 0.0, wall_cz],
         [(bhx - hcx_rs - ohx) / 2, bhy, wall_hz], _BOARD_RGBA)

    # Front land (y ∈ [+ohy, +bhy]): spans full board width
    _box(board, "land_front",
         [0.0, (ohy + bhy) / 2, wall_cz],
         [bhx, (bhy - ohy) / 2, wall_hz], _BOARD_RGBA)

    # Back land
    _box(board, "land_back",
         [0.0, -(ohy + bhy) / 2, wall_cz],
         [bhx, (bhy - ohy) / 2,  wall_hz], _BOARD_RGBA)

    # ── Round hole pocket ─────────────────────────────────────────────────
    rh   = cfg["holes"]["round"]
    hr   = rh["radius"]   # 0.011
    _add_hole_pocket(board, "rh", hcx_r, 0.0, hr, hr, ohx, ohy,
                     wall_cz, wall_hz, _BOARD_RGBA)
    # Visual round marker (no collision)
    cyl = board.add_geom()
    cyl.name = "rh_visual"
    cyl.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    cyl.size = [hr * 0.99, wall_hz, 0.0]
    cyl.pos  = [hcx_r, 0.0, wall_cz]
    cyl.rgba = [0.15, 0.15, 0.15, 1.0]
    cyl.contype = 0
    cyl.conaffinity = 0
    # Sites
    for sname in ("hole_entrance", "round_hole_entrance"):
        s = board.add_site()
        s.name = sname
        s.pos  = [hcx_r, 0.0, bhz]
        s.size = [0.001, 0.001, 0.001]
        s.rgba = [1.0, 0.0, 0.0, 0.0]

    # ── Square hole pocket ────────────────────────────────────────────────
    sh              = cfg["holes"]["square"]
    shhx, shhy      = sh["half_size"]
    _add_hole_pocket(board, "sh", hcx_s, 0.0, shhx, shhy, ohx, ohy,
                     wall_cz, wall_hz, _BOARD_RGBA)
    s2 = board.add_site()
    s2.name = "square_hole_entrance"
    s2.pos  = [hcx_s, 0.0, bhz]
    s2.size = [0.001, 0.001, 0.001]
    s2.rgba = [0.0, 0.0, 1.0, 0.0]

    # ── Rect slot pocket ──────────────────────────────────────────────────
    rs              = cfg["holes"]["rect"]
    rshhx, rshhy    = rs["half_size"]
    _add_hole_pocket(board, "rs", hcx_rs, 0.0, rshhx, rshhy, ohx, ohy,
                     wall_cz, wall_hz, _BOARD_RGBA)
    s3 = board.add_site()
    s3.name = "rect_slot_entrance"
    s3.pos  = [hcx_rs, 0.0, bhz]
    s3.size = [0.001, 0.001, 0.001]
    s3.rgba = [0.0, 1.0, 0.0, 0.0]


def _add_round_peg(world: mujoco.MjSpec, cfg: dict) -> None:
    """Round peg — body name 'peg', driven by the mainline pipeline."""
    pc   = cfg["pegs"]["round"]
    px, py, pz = pc["initial_pos"]
    r    = pc["radius"]
    hl   = pc["half_length"]
    mass = pc["mass"]
    rgba = pc["color"]

    body = world.add_body()
    body.name = "peg"
    body.pos  = [px, py, pz]

    fj = body.add_freejoint()
    fj.name = "peg_joint"

    g = body.add_geom()
    g.name = "peg_geom"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [r, hl, 0.0]
    g.mass = mass
    g.rgba = rgba
    g.friction = [2.0, 0.05, 0.01]

    for sname, z_off in (("peg_tip", -hl), ("peg_top", hl)):
        s = body.add_site()
        s.name = sname
        s.pos  = [0.0, 0.0, z_off]
        s.size = [0.001, 0.001, 0.001]
        s.rgba = [0.0, 1.0, 0.0, 0.0]


def _add_box_peg(world: mujoco.MjSpec, cfg_peg: dict) -> None:
    """Add a box-shaped peg (square or rect)."""
    name   = cfg_peg["name"]
    px, py, pz = cfg_peg["initial_pos"]
    hx, hy, hz = cfg_peg["half_size"]
    mass   = cfg_peg["mass"]
    rgba   = cfg_peg["color"]

    body = world.add_body()
    body.name = name
    body.pos  = [px, py, pz]

    fj = body.add_freejoint()
    fj.name = f"{name}_joint"

    g = body.add_geom()
    g.name = f"{name}_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [hx, hy, hz]
    g.mass = mass
    g.rgba = rgba
    g.friction = [2.0, 0.05, 0.01]

    # Tip and top sites (z-axis: up when lying flat, tip is bottom)
    for sname, z_off in ((f"{name}_tip", -hz), (f"{name}_top", hz)):
        s = body.add_site()
        s.name = sname
        s.pos  = [0.0, 0.0, z_off]
        s.size = [0.001, 0.001, 0.001]
        s.rgba = [0.0, 1.0, 0.0, 0.0]


def build_scene(cfg: dict) -> tuple:
    """Build the multi-shape assembly scene.

    Returns (MjModel, MjData).  Defines the canonical names used by the
    mainline pipeline: body 'peg', joint 'peg_joint', sites
    'peg_tip'/'peg_top', equality 'grasp_weld', and site 'hole_entrance'.
    """
    spec = mujoco.MjSpec.from_file(_PANDA_XML)
    spec.option.timestep  = cfg["sim"]["dt"]
    spec.option.gravity   = [0.0, 0.0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    spec.stat.center  = [0.45, 0.0, 0.35]
    spec.stat.extent  = 1.1
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.global_.azimuth   = 145.0
    spec.visual.global_.elevation = -20.0
    spec.visual.global_.offwidth  = 1920
    spec.visual.global_.offheight = 1080

    world = spec.worldbody

    # Lighting
    light = world.add_light()
    light.name = "scene_light"
    light.pos  = [0.5, 0.0, 1.5]
    light.dir  = [0.0, 0.0, -1.0]

    # Floor
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.rgba = [0.28, 0.28, 0.28, 1.0]

    # Table (raised so top is at z=0.250)
    tc = cfg["table"]
    tcx, tcy, tcz = tc["center"]
    thx, thy, thz = tc["half_size"]
    table = world.add_body()
    table.name = "table"
    table.pos  = [tcx, tcy, tcz]
    tg = table.add_geom()
    tg.name = "table_geom"
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = [thx, thy, thz]
    tg.rgba = [0.75, 0.70, 0.60, 1.0]
    tg.friction = [0.6, 0.005, 0.0001]

    # Assembly board (3 hole pockets)
    _add_assembly_board(world, cfg)

    # Pegs
    _add_round_peg(world, cfg)
    _add_box_peg(world, cfg["pegs"]["square"])
    _add_box_peg(world, cfg["pegs"]["rect"])

    # EE site on Panda hand
    hand_body = spec.body("hand")
    ee_site = hand_body.add_site()
    ee_site.name = EE_SITE_NAME
    ee_site.pos  = [0.0, 0.0, 0.103]
    ee_site.size = [0.005, 0.005, 0.005]
    ee_site.rgba = [0.0, 0.8, 0.8, 0.0]

    # Cameras — positions and orientations from scene config
    for cam_name, cam_cfg in cfg.get("cameras", {}).items():
        cam = world.add_camera()
        cam.name = cam_name
        cam.pos  = cam_cfg["pos"]
        cam.quat = _cam_quat_lookat(cam_cfg["pos"], cam_cfg["lookat"])
        cam.fovy = float(cam_cfg.get("fov", 60))

    # Grasp weld for round peg
    eq = spec.add_equality()
    eq.type     = mujoco.mjtEq.mjEQ_WELD
    eq.name     = GRASP_EQ_NAME
    eq.objtype  = mujoco.mjtObj.mjOBJ_BODY
    eq.name1    = "hand"
    eq.name2    = "peg"
    eq.active   = False
    eq.solref   = [0.004, 1.0]
    eq.solimp   = [0.999, 0.9999, 0.0001, 0.5, 2.0]

    # Grasp weld for square peg
    eq_sq = spec.add_equality()
    eq_sq.type    = mujoco.mjtEq.mjEQ_WELD
    eq_sq.name    = GRASP_EQ_SQ_NAME
    eq_sq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq_sq.name1   = "hand"
    eq_sq.name2   = "peg_square"
    eq_sq.active  = False
    eq_sq.solref  = [0.004, 1.0]
    eq_sq.solimp  = [0.999, 0.9999, 0.0001, 0.5, 2.0]

    # Grasp weld for rect peg
    eq_re = spec.add_equality()
    eq_re.type    = mujoco.mjtEq.mjEQ_WELD
    eq_re.name    = GRASP_EQ_RECT_NAME
    eq_re.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq_re.name1   = "hand"
    eq_re.name2   = "peg_rect"
    eq_re.active  = False
    eq_re.solref  = [0.004, 1.0]
    eq_re.solimp  = [0.999, 0.9999, 0.0001, 0.5, 2.0]

    m = spec.compile()
    d = mujoco.MjData(m)

    q_home = cfg["robot"]["home_qpos"]
    for i, jname in enumerate(ARM_JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        d.qpos[m.jnt_qposadr[jid]] = q_home[i]

    d.ctrl[7] = cfg["robot"]["gripper_open_ctrl"]
    mujoco.mj_forward(m, d)
    return m, d


# ── Body / site ID registries ─────────────────────────────────────────────

def get_body_ids(m: mujoco.MjModel) -> dict:
    names = ["peg", "peg_square", "peg_rect",
             "board", "table", "hand", "left_finger", "right_finger"]
    return {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}


def get_camera_ids(m: mujoco.MjModel) -> dict:
    """Return {camera_name: camera_id} for all named cameras in the model."""
    return {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i): i
        for i in range(m.ncam)
    }


def get_site_ids(m: mujoco.MjModel) -> dict:
    names = [EE_SITE_NAME,
             "peg_tip", "peg_top",
             "peg_square_tip", "peg_square_top",
             "peg_rect_tip",   "peg_rect_top",
             "hole_entrance",
             "round_hole_entrance",
             "square_hole_entrance",
             "rect_slot_entrance"]
    return {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in names}
