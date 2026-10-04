from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import nav_grid
from neko_anyadance_body.backend.nav_grid import (
    FREE, OCC, UNK, GridMeta, NavGrid, TerrainClass,
)

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

    def test_line_ok_visits_every_cell_the_segment_crosses(self):
        """近轴向的斜线必须采到它真正压过的每一格。

        两条反例各自针对一种漏法：切比雪夫步数（漏 (44,20)），以及只补 L1 步数
        却仍用 round 取整（漏 (8,5)）。两格都是线段真正穿过的格。
        """
        ng = make(room())
        ng.center = np.ones(ng.grid.shape, bool)
        ng.center[44, 20] = False
        self.assertFalse(ng._line_ok((42, 20), (45, 21)), "漏掉了真正压过的 (44,20)")
        ng.center[44, 20] = True
        ng.center[8, 5] = False
        self.assertFalse(ng._line_ok((5, 5), (9, 6)), "漏掉了真正压过的 (8,5)")

    def test_line_ok_stays_true_on_a_clean_run(self):
        ng = make(room())
        ng.center = np.ones(ng.grid.shape, bool)
        self.assertTrue(ng._line_ok((10, 10), (13, 13)))   # 纯对角，不该被误伤
        self.assertTrue(ng._line_ok((10, 10), (40, 10)))   # 纯直行，行为不变
        self.assertTrue(ng._line_ok((20, 30), (20, 30)))   # 零长度

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
            pgm.with_suffix(".json").write_text(json.dumps({     # 尺寸对、只缺 world_scale：不许回落默认值
                "resolution_m": RES, "origin_xy_m": [0, 0], "rows": 80, "cols": 120,
                "row0": "max_y"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "world_scale"):
                NavGrid.load(pgm)


class TerrainChannelTests(unittest.TestCase):
    """S0 地形代价通道。

    这里测的**不是**"地形判别准不准"（那要等有生产者之后），而是通道本身的契约：
    默认惰性、非法输入响亮地报错、UNKNOWN 不被当成 free、水不是障碍。
    每条都对应 Docs 里的一条硬约束，不是风格偏好。
    """

    def test_no_terrain_channel_is_inert(self):
        """没挂地形层时，_cost() 必须与"加地形之前"逐位相同。

        这是 tools/mapping_gate 之外的第二道保证：门要跑真实录制（慢），
        这里只要 1 ms。
        """
        ng = make(room())
        self.assertFalse(ng.has_terrain)
        self.assertIsNone(ng.terrain_cost_map())
        expect = np.where(ng.center,
                          1.0 + 2.0 * np.clip(0.6 - ng.clearance, 0, None) / 0.6, np.inf)
        self.assertTrue(np.array_equal(ng._cost(), expect, equal_nan=True))

    def test_attach_rejects_shape_and_bad_values(self):
        ng = make(room())
        with self.assertRaisesRegex(ValueError, "形状"):
            ng.attach_terrain(np.zeros((3, 3), np.uint8))
        bad = np.zeros(ng.grid.shape, np.uint8)
        bad[0, 0] = 99
        # 越界类别必须抛错，不能 clamp 成最近的合法值：clamp 会把"上游算错了"
        # 伪装成"上游判成了某个合法类别"，那正是约束 1 禁止的。
        with self.assertRaisesRegex(ValueError, "非法取值"):
            ng.attach_terrain(bad)
        with self.assertRaisesRegex(ValueError, "NaN"):
            ng.attach_terrain(np.zeros(ng.grid.shape, np.uint8),
                              np.full(ng.grid.shape, np.nan, np.float32))
        with self.assertRaisesRegex(ValueError, "missing"):
            ng.attach_terrain(np.zeros(ng.grid.shape, np.uint8), missing="乐观")
        self.assertFalse(ng.has_terrain)          # 上面每次抛错后都还是"没有地形"

    def test_water_is_expensive_not_lethal(self):
        """水必须是**有限**代价。

        这是整个 S0 里最关键的一条：若水被判成障碍，正确地图与错误地图
        都给出"过不去"，下游任何以能否通行为判据的信号就没有梯度。
        """
        ng = make(room())
        k = np.full(ng.grid.shape, int(TerrainClass.FLAT), np.uint8)
        k[40:44, 30:34] = int(TerrainClass.WATER)
        ng.attach_terrain(k)
        t = ng.terrain_cost_map()
        self.assertTrue(np.isfinite(t[41, 31]))
        self.assertGreater(t[41, 31], t[41, 20])          # 比平地贵
        self.assertLess(t[41, 31], nav_grid.TERRAIN_COST_MAX)  # 但不是障碍

    def test_pit_and_obstacle_are_capped_not_inf(self):
        ng = make(room())
        k = np.full(ng.grid.shape, int(TerrainClass.FLAT), np.uint8)
        k[40:44, 30:34] = int(TerrainClass.PIT)
        k[50:54, 30:34] = int(TerrainClass.OBSTACLE)
        ng.attach_terrain(k)
        t = ng.terrain_cost_map()
        self.assertTrue(np.isfinite(t).all())   # inf 会在序列化/比较里变成 nan
        self.assertEqual(t[41, 31], nav_grid.TERRAIN_COST_MAX)
        self.assertEqual(t[51, 31], nav_grid.TERRAIN_COST_MAX)

    def test_unknown_is_not_free(self):
        """核心红线：UNKNOWN 的代价必须远高于 FLAT，默认 lethal 下顶到 C_max。"""
        ng = make(room())
        k = np.full(ng.grid.shape, int(TerrainClass.FLAT), np.uint8)
        k[40:44, 30:34] = int(TerrainClass.UNKNOWN)
        ng.attach_terrain(k)
        t = ng.terrain_cost_map()
        self.assertGreater(t[41, 31], t[41, 20] * 5)
        self.assertEqual(t[41, 31], nav_grid.TERRAIN_COST_MAX)

    def test_low_confidence_moves_cost_toward_conservative(self):
        """自信地判错必须比"不确定"贵：低置信 → 代价往保守值插值。"""
        ng = make(room())
        k = np.full(ng.grid.shape, int(TerrainClass.FLAT), np.uint8)
        hi = np.ones(ng.grid.shape, np.float32)
        lo = np.full(ng.grid.shape, 0.1, np.float32)
        ng.attach_terrain(k, hi)
        t_hi = ng.terrain_cost_map()
        ng.attach_terrain(k, lo)
        t_lo = ng.terrain_cost_map()
        self.assertAlmostEqual(float(t_hi[41, 31]), 1.0, places=5)
        self.assertGreater(float(t_lo[41, 31]), float(t_hi[41, 31]) * 5)

    def test_missing_policy_changes_only_unknown_cells(self):
        """lethal 与 inherit 只在 UNKNOWN 格上分道；FLAT 格两者必须完全一致。

        否则 inherit 就成了"偷偷把整个地形层关掉"的别名，A/B 会失去意义。
        """
        g = np.full((80, 120), FREE, np.uint8)
        g[0, :], g[-1, :], g[:, 0], g[:, -1] = OCC, OCC, OCC, OCC
        k = np.full(g.shape, int(TerrainClass.FLAT), np.uint8)
        k[40:44, 30:34] = int(TerrainClass.UNKNOWN)
        ng_l, ng_i = make(g), make(g)
        ng_l.attach_terrain(k, missing="lethal")
        ng_i.attach_terrain(k, missing="inherit")
        c_l, c_i = ng_l._cost(), ng_i._cost()
        # lethal：UNKNOWN 被顶到 C_max（有限，不是 inf——见 nav_grid 里的说明）
        self.assertEqual(c_l[41, 31], nav_grid.TERRAIN_COST_MAX)
        # inherit：UNKNOWN 退回纯几何代价
        self.assertLess(c_i[41, 31], nav_grid.TERRAIN_COST_MAX)
        # FLAT 格两种策略必须一致
        self.assertEqual(c_l[41, 20], c_i[41, 20])

    def test_terrain_cost_reorders_routes(self):
        """端到端：一条几何上等价的近路，被地形判成坑之后必须绕开。

        两个容易写错的断言，这里都踩过：

        * **不能**断言"路径被拒绝"。S0 刻意不做硬阻断（见 ``nav_grid._cost``
          的说明：硬阻断会让对错地图给出同一结果，信号就没梯度了）。
          不走要靠 S1 的路径代价预算。
        * **不能**断言"长度变长 1.5 倍"。绕行代价取决于坑相对房间的几何比例，
          这里只有 0.7%，写死倍数会得到一个测不出东西的脆弱断言。

        真正该断言的是**路径不穿过被判成坑的格**——它与几何无关，
        而且恰好是 ``_string_pull`` 会悄悄违反的那一条。
        """
        g = np.full((80, 120), FREE, np.uint8)
        g[0, :], g[-1, :], g[:, 0], g[:, -1] = OCC, OCC, OCC, OCC
        ng = make(g)
        s, t = ng.to_world((40, 8)), ng.to_world((40, 112))
        before = ng.plan(s, t)
        self.assertTrue(before.accepted, before.reason)

        pit = (38, 43, 52, 68)          # r0, r1, c0, c1
        k = np.full(g.shape, int(TerrainClass.FLAT), np.uint8)
        k[pit[0]:pit[1], pit[2]:pit[3]] = int(TerrainClass.PIT)
        ng.attach_terrain(k)

        self.assertEqual(ng._cost()[40, 60], nav_grid.TERRAIN_COST_MAX)
        self.assertFalse(ng._passable()[40, 60])
        rc_s = ng.snap_to_center(s, 0.5)[0]
        rc_t = ng.snap_to_center(t, 0.5)[0]
        self.assertFalse(ng._line_ok(rc_s, rc_t),
                         "直线仍被判为通畅 ⇒ _string_pull 会把绕行拉直穿坑")

        after = ng.plan(s, t)
        self.assertTrue(after.accepted, f"不该硬阻断：{after.reason}")
        self.assertGreater(len(after.waypoints_xy_m), len(before.waypoints_xy_m),
                           "加了地形层却仍是 2 个折点的直线，说明地形代价没进规划")
        for p, q in zip(after.waypoints_xy_m, after.waypoints_xy_m[1:]):
            a, b = ng.to_cell(p), ng.to_cell(q)
            n = int(abs(b[0] - a[0]) + abs(b[1] - a[1])) + 1
            rc = np.floor(np.linspace(a[0], b[0], n) + 1e-9).astype(int), \
                 np.floor(np.linspace(a[1], b[1], n) + 1e-9).astype(int)
            hit = (((rc[0] >= pit[0]) & (rc[0] < pit[1])
                    & (rc[1] >= pit[2]) & (rc[1] < pit[3])).any())
            self.assertFalse(hit, "折点之间仍横穿坑")

    def test_water_does_not_force_detour_as_hard_as_pit(self):
        """水和坑都是"贵"，但不该等价——这是 TERRAIN_COST 有限分档的全部意义。

        若把水也设成 C_max，"这片水多深"这个判断就不再产生任何代价差异，
        而那恰恰是材质支路唯一要学的东西。
        """
        g = np.full((80, 120), FREE, np.uint8)
        g[0, :], g[-1, :], g[:, 0], g[:, -1] = OCC, OCC, OCC, OCC
        kw = np.full(g.shape, int(TerrainClass.FLAT), np.uint8)
        kw[38:43, 52:68] = int(TerrainClass.WATER)
        nw = make(g)
        nw.attach_terrain(kw)
        w_cost = float(nw.terrain_cost_map()[40, 60])
        kp = np.full(g.shape, int(TerrainClass.FLAT), np.uint8)
        kp[38:43, 52:68] = int(TerrainClass.PIT)
        npg = make(g)
        npg.attach_terrain(kp)
        p_cost = float(npg.terrain_cost_map()[40, 60])
        self.assertLess(w_cost, p_cost)
        self.assertLess(w_cost, nav_grid.TERRAIN_COST_MAX)


if __name__ == "__main__":
    unittest.main()
