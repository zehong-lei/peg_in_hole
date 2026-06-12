#!/usr/bin/env python3
"""Assembly scene geometry verification script.

Checks:
  1. Board bottom == table top (no floating)
  2. Hole entrance positions
  3. Peg initial bottom == table top
  4. Hole depth > insertion_depth_goal
  5. Peg half_length > hole_depth  (gripper clears board during insertion)
  6. Hole spacing >= minimum (40 mm)
  7. Hole half-sizes > peg half-sizes (positive clearance)

Usage:
    python3 scripts/check_geometry.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import yaml

from src.tasks.assembly_scene import AssemblyScene

_REPO_ROOT = Path(__file__).parents[1]
_INSERTION_DEPTH_GOAL = 0.045   # m (from task.yaml)
_MIN_HOLE_SPACING     = 0.040   # m


def check(label: str, value: bool, detail: str = "") -> bool:
    status = "✓ PASS" if value else "✗ FAIL"
    msg = f"  [{status}]  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return value


def main():
    print("Loading assembly scene …")
    scene = AssemblyScene.from_config()
    cfg   = scene._cfg

    ab      = cfg["assembly_board"]
    table   = cfg["table"]
    holes   = cfg["holes"]
    pegs    = cfg["pegs"]

    table_top  = table["center"][2]  + table["half_size"][2]
    board_bot  = ab["center"][2]     - ab["half_size"][2]
    board_top  = ab["center"][2]     + ab["half_size"][2]
    hole_depth = ab["hole_depth"]
    hole_floor = board_top - hole_depth

    print("\n" + "─" * 55)
    print("1. Board sits on table")
    print("─" * 55)
    check("board_bottom == table_top",
          abs(board_bot - table_top) < 1e-4,
          f"board_bot={board_bot:.4f}  table_top={table_top:.4f}")

    print("\n" + "─" * 55)
    print("2. Hole positions")
    print("─" * 55)
    hole_names = ["round_hole", "square_hole", "rect_slot"]
    hole_entrances = {}
    for hname in hole_names:
        pos, _ = scene.get_hole_pose(hname)
        hole_entrances[hname] = pos
        check(f"{hname} entrance_z == board_top",
              abs(pos[2] - board_top) < 1e-4,
              f"entrance_z={pos[2]:.4f}  board_top={board_top:.4f}")

    print("\n" + "─" * 55)
    print("3. Hole depth vs insertion goal")
    print("─" * 55)
    check("hole_depth > insertion_depth_goal",
          hole_depth > _INSERTION_DEPTH_GOAL,
          f"hole_depth={hole_depth:.3f}  goal={_INSERTION_DEPTH_GOAL:.3f}")
    check("hole_floor > board_bottom (floor doesn't breach board)",
          hole_floor > board_bot,
          f"hole_floor={hole_floor:.4f}  board_bot={board_bot:.4f}")

    print("\n" + "─" * 55)
    print("4. Peg initial positions on table")
    print("─" * 55)
    peg_keys = [("peg", "round"), ("peg_square", "square"), ("peg_rect", "rect")]
    for bname, ck in peg_keys:
        pc   = pegs[ck]
        ipos = pc["initial_pos"]
        hl   = pc.get("half_length", pc.get("half_size", [0, 0, 0])[2])
        bot  = ipos[2] - hl
        check(f"{bname} bottom == table_top",
              abs(bot - table_top) < 1e-3,
              f"peg_bot={bot:.4f}  table_top={table_top:.4f}")

    print("\n" + "─" * 55)
    print("5. Peg half_length > hole_depth  (gripper clears board)")
    print("─" * 55)
    round_hl = pegs["round"]["half_length"]
    check("round_peg half_length > hole_depth",
          round_hl > hole_depth,
          f"half_length={round_hl:.3f}  hole_depth={hole_depth:.3f}")

    print("\n" + "─" * 55)
    print("6. Hole spacing >= minimum")
    print("─" * 55)
    xs = [hole_entrances[h][0] for h in hole_names]
    xs_sorted = sorted(xs)
    for i in range(len(xs_sorted) - 1):
        spacing = xs_sorted[i + 1] - xs_sorted[i]
        check(f"spacing holes {i}→{i+1} >= {_MIN_HOLE_SPACING*1000:.0f} mm",
              spacing >= _MIN_HOLE_SPACING,
              f"spacing={spacing*1000:.1f} mm")

    print("\n" + "─" * 55)
    print("7. Hole clearance > 0  (hole larger than peg)")
    print("─" * 55)
    # Round
    rh_r = holes["round"]["radius"]
    rp_r = pegs["round"]["radius"]
    check("round hole radius > round peg radius",
          rh_r > rp_r,
          f"hole_r={rh_r:.4f}  peg_r={rp_r:.4f}  gap={rh_r-rp_r:.4f}")

    # Square
    sh_hx, sh_hy = holes["square"]["half_size"]
    sp_hx = sp_hy = pegs["square"]["half_size"][0]
    check("square hole hx > square peg hx",
          sh_hx > sp_hx,
          f"hole={sh_hx:.4f}  peg={sp_hx:.4f}  gap={(sh_hx-sp_hx)*2000:.1f}mm/side")

    # Rect
    rs_hx, rs_hy = holes["rect"]["half_size"]
    rp_hx, rp_hy = pegs["rect"]["half_size"][:2]
    check("rect slot hx > rect peg hx",
          rs_hx > rp_hx,
          f"hole={rs_hx:.4f}  peg={rp_hx:.4f}")
    check("rect slot hy > rect peg hy",
          rs_hy > rp_hy,
          f"hole={rs_hy:.4f}  peg={rp_hy:.4f}")

    print()
    scene.print_scene_summary()


if __name__ == "__main__":
    main()
