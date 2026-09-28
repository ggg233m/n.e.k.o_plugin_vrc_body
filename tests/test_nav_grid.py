from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.nav_grid import FREE, OCC, UNK, GridMeta, NavGrid

S = 0.755
RES = 0.05


def room(h: int = 80, w: int = 120) -> np.ndarray:
    """左右两个 free 房间，中间一堵墙，墙上有一个门；右上角一块 unknown。"""
    g = np.full((h, w), FREE, np.uint8)
    g[:, 58:62] = OCC
    g[35:50, 58:62] = FREE          # 门：15 格 × 0.03775 ≈ 0.57 m 宽
    g[:20, 90:] = UNK
    g[0, :], g[-1, :], g[:, 0], g[:, -1] = OCC, OCC, OCC, OCC
    return g


def make(g: np.ndarray) -> NavGrid:
    ng = NavGrid(g, GridMeta(RES, (0.0, 0.0), S))
    ng.build(radius_m=0.25)
    return ng


class NavGridTests(unittest.TestCase):
    def test_cell_world_roundtrip(self):
        ng = make(room())
        for rc in [(0, 0), (10, 40), (79, 119)]:
            self.assertEqual(ng.to_cell(ng.to_world(rc)), rc)
        # 第 0 行是 y 最大
        self.assertGreater(ng.to_world((0, 0))[1], ng.to_world((79, 0))[1])

    def test_plan_through_door(self):
        ng = make(room())
        a, b = ng.to_world((42, 20)), ng.to_world((42, 100))
        c = ng.plan(a, b)
        self.assertTrue(c.accepted, c.reason)
        self.assertGreater(c.length_m, 2.9)
        self.assertGreaterEqual(c.corridor_half_width_m, 0.25)
        for p, q in zip(c.waypoints_xy_m, c.waypoints_xy_m[1:]):
            self.assertTrue(ng.segment_clear(p, q))

    def test_goal_unknown_rejected_not_snapped(self):
        ng = make(room())
        c = ng.plan(ng.to_world((42, 20)), ng.to_world((5, 110)))
        self.assertFalse(c.accepted)
        self.assertEqual(c.reason, "goal_unknown")

    def test_goal_next_to_wall_snaps_to_center(self):
        ng = make(room())
        goal = ng.to_world((25, 56))          # 贴着墙，已观测但装不下
        self.assertEqual(ng.classify(goal), "too_narrow")
        c = ng.plan(ng.to_world((42, 20)), goal)
        self.assertTrue(c.accepted, c.reason)
        end = c.waypoints_xy_m[-1]
        self.assertEqual(ng.classify(end), "free_center")
        self.assertLessEqual(math.hypot(end[0] - goal[0], end[1] - goal[1]), 0.5 + 1e-6)

    def test_closed_door_disconnects(self):
        g = room()
        g[35:50, 58:62] = OCC
        ng = make(g)
        c = ng.plan(ng.to_world((42, 20)), ng.to_world((42, 100)))
        self.assertEqual(c.reason, "goal_disconnected")

    def test_narrow_door_rejected(self):
        g = room()
        g[:, 58:62] = OCC
        g[40:44, 58:62] = FREE      # 4 格 ≈ 0.15 m，装不下 0.25 m 半径
        ng = make(g)
        self.assertEqual(ng.plan(ng.to_world((42, 20)), ng.to_world((42, 100))).reason,
                         "goal_disconnected")

    def test_unknown_is_not_walkable_even_next_to_free(self):
        g = np.full((60, 60), UNK, np.uint8)
        g[20:40, 20:40] = FREE      # 0.75 m 见方，四周全是 unknown
        ng = make(g)
        # 中心区只剩离 unknown ≥ 0.25 m 的一小块
        self.assertEqual(ng.classify(ng.to_world((21, 21))), "too_narrow")
        self.assertEqual(ng.classify(ng.to_world((30, 30))), "free_center")

    def test_walked_corridor_counts_but_not_through_obstacle(self):
        g = np.full((40, 100), UNK, np.uint8)
        ng = NavGrid(g, GridMeta(RES, (0.0, 0.0), S))
        path = np.array([ng.to_world((20, 5)), ng.to_world((20, 95))])
        ng.build(radius_m=0.25, walked=[path])
        c = ng.plan(path[0], path[1])
        self.assertTrue(c.accepted, c.reason)
        g2 = g.copy()
        g2[:, 50] = OCC
        ng2 = NavGrid(g2, GridMeta(RES, (0.0, 0.0), S))
        ng2.build(radius_m=0.25, walked=[path])
        self.assertFalse(ng2.plan(path[0], path[1]).accepted)

    def test_start_snap_limited(self):
        ng = make(room())
        near_wall = ng.to_world((20, 57))        # 贴墙（避开门），离中心区不到 0.5 m
        self.assertEqual(ng.classify(near_wall), "too_narrow")
        self.assertTrue(ng.plan(near_wall, ng.to_world((42, 20))).accepted)
        c = ng.plan(near_wall, ng.to_world((42, 20)), start_snap_m=0.0)
        self.assertFalse(c.accepted)
        self.assertTrue(c.reason.startswith("start_"))

    def test_non_tristate_values_become_unknown(self):
        g = room()
        g[30:50, 10:30] = 200
        ng = make(g)
        self.assertEqual(ng.classify(ng.to_world((40, 20))), "unknown")

    def test_load_requires_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            pgm = Path(d) / "x_map.pgm"
            cv2.imwrite(str(pgm), room())
            with self.assertRaises(FileNotFoundError):
                NavGrid.load(pgm)
            pgm.with_suffix(".json").write_text(json.dumps({
                "resolution_m": RES, "origin_xy_m": [1.0, -2.0], "rows": 80, "cols": 120,
                "row0": "max_y", "world_scale": S}), encoding="utf-8")
            ng = NavGrid.load(pgm)
            x, y = ng.to_world((79, 0))
            self.assertAlmostEqual(x, (1.0 + 0.5 * RES) * S, places=6)
            self.assertAlmostEqual(y, (-2.0 + 0.5 * RES) * S, places=6)
            pgm.with_suffix(".json").write_text(json.dumps({
                "resolution_m": RES, "origin_xy_m": [0, 0], "rows": 81, "cols": 120,
                "row0": "max_y"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                NavGrid.load(pgm)


if __name__ == "__main__":
    unittest.main()
