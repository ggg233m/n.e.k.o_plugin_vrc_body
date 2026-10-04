from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.nav_grid import FREE, OCC, UNK
from neko_anyadance_body.backend.nav_mapping import (KeyframeGridMapper, MapperConfig, NavSession, frontiers,
                                                     stereo_points)

CAM_H = 1.73


def pose(x: float, y: float, yaw: float = 0.0, pitch: float = 0.0, z: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])   # 正 pitch = 低头（x 前转向 −z）
    T = np.eye(4)
    T[:3, :3] = rz @ ry
    T[:3, 3] = (x, y, z)
    return T


def world_scene() -> np.ndarray:
    """地图系点：x∈[−1,6] y∈[−1.5,1.5] 的地面（z=−CAM_H），x=3.5 处一堵 1 m 高的墙（y∈[−1.5,0.3]）。"""
    gx, gy = np.meshgrid(np.arange(-1.0, 6.0, 0.03), np.arange(-1.5, 1.5, 0.03))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -CAM_H)])
    wy, wz = np.meshgrid(np.arange(-1.5, 0.3, 0.03), np.arange(-CAM_H + 0.4, -CAM_H + 1.2, 0.05))
    wall = np.column_stack([np.full(wy.size, 3.5), wy.ravel(), wz.ravel()])
    return np.vstack([ground, wall])


def observe(scene: np.ndarray, T: np.ndarray) -> np.ndarray:
    """模拟一个关键帧：只看前方 3 m 以内的点，存成 base 系。"""
    local = (scene - T[:3, 3]) @ T[:3, :3]
    keep = (local[:, 0] > 0.3) & (np.hypot(local[:, 0], local[:, 1]) < 3.0)
    return local[keep]


def value_at(ng, xy_track: tuple[float, float]) -> int:
    s = ng.meta.world_scale
    rc = ng.to_cell((xy_track[0] * s, xy_track[1] * s))
    return int(ng.grid[rc]) if ng.inside(rc) else UNK     # 图外 = 没看见过


class MapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = world_scene()

    def mapper_with(self, poses):
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        for i, T in enumerate(poses):
            m.add_keyframe(i, observe(self.scene, T), T)
        return m

    def test_duplicate_keyframe_id_rejected(self) -> None:
        # 与 LoopCloser.add_keyframe 同口径的廉价防御：重复 id 会静默覆盖点云与位姿。
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        m.add_keyframe(0, observe(self.scene, pose(0, 0)), pose(0, 0))
        with self.assertRaises(ValueError):
            m.add_keyframe(0, observe(self.scene, pose(1.0, 0)), pose(1.0, 0))
        # 摘掉点云之后重新入同一个 id 不算重复：原地补帧走的就是这条路。
        m.drop_points(0)
        self.assertEqual(len(m), 0)
        m.add_keyframe(0, observe(self.scene, pose(1.0, 0)), pose(1.0, 0))
        self.assertEqual(len(m), 1)

    def test_ground_free_wall_obstacle_behind_unknown(self) -> None:
        ng = self.mapper_with([pose(0, 0), pose(1.0, 0)]).rasterize()
        self.assertEqual(value_at(ng, (2.0, 0.0)), FREE)
        self.assertEqual(value_at(ng, (3.52, -0.8)), OCC)
        # 离所有关键帧都 > 3 m 的地方从没被看见：必须是 unknown，不能因为"周围是地面"就填 free。
        self.assertEqual(value_at(ng, (5.5, 1.0)), UNK)
        self.assertEqual(value_at(ng, (-0.9, 0.0)), UNK)

    def test_head_pitch_does_not_turn_ground_into_obstacle(self) -> None:
        # 低头 12°：base 系里 3 m 外的地面会"抬高"约 0.6 m；按关键帧地图系判高就不受影响。
        ng = self.mapper_with([pose(0, 0, pitch=math.radians(12)), pose(0.5, 0, pitch=math.radians(-8))]).rasterize()
        occ_near = ng.grid == OCC
        s = ng.meta.world_scale
        r, c = ng.to_cell((3.0 * s, 0.9 * s))      # 墙以外（y>0.3）的远处地面
        self.assertEqual(int(ng.grid[r, c]), FREE)
        self.assertLess(occ_near.sum() * 0.01, 0.3 * 2)   # 障碍只有那堵墙那么点面积（追踪 m²）

    def test_unmodelled_pitch_ground_tilt_is_fitted_out(self) -> None:
        # 位姿里没有的 14° 俯仰（HMD 与相机外参误差）：远处地面在关键帧里一路抬高，常数偏移修不掉，
        # 地面上冒出一片假障碍。斜面拟合后假障碍消失，墙不受影响。
        def occ_cells(cap: float, **kw):
            m = KeyframeGridMapper(MapperConfig(res_m=0.10, ground_plane_max_deg=cap, **kw))
            m.add_keyframe(0, observe(self.scene, pose(1.0, 0, pitch=math.radians(14))), pose(1.0, 0))
            ng = m.rasterize()
            s, occ = ng.meta.world_scale, ng.grid == OCC

            def box(x0, x1, y0, y1):
                r0, c0 = ng.to_cell((x0 * s, y1 * s))
                r1, c1 = ng.to_cell((x1 * s, y0 * s))
                return int(occ[r0:r1 + 1, c0:c1 + 1].sum())
            return box(1.3, 3.3, -1.4, 1.4) + box(3.3, 4.0, 0.4, 1.4), box(3.3, 3.7, -1.4, 0.2)

        # 扁平计数口径（q_tiers=False）：斜面拟合是唯一防线，不拟合就冒一片假障碍。
        ground_off, wall_off = occ_cells(0.0, q_tiers=False)
        ground_on, wall_on = occ_cells(15.0, q_tiers=False)
        self.assertGreater(ground_off, 20)          # 证明这条用例确实需要斜面
        self.assertLessEqual(ground_on, 3)
        # 差 1 格是 BORDER_CONSTANT 带来的：3×3 多数滤波在图外沿不再镜像回填自己，
        # 边缘的障碍格少一票支撑。跟"斜面拟合有没有吃掉墙"无关，放 2 格的容差。
        self.assertAlmostEqual(wall_on, wall_off, delta=2)
        self.assertGreater(wall_on, 20)
        # 默认口径：质量分层已经把远场票否掉了，斜面拟合退成第二道防线（两道都留着）。
        # 要保证的不是"更干净"，而是 ① 墙不会因为分层而被当成量化倾斜清掉，② 光靠分层
        # 也不会在斜面上冒出成片假障碍（6 格量级，而不是扁平口径下那 20+ 格）。
        g_tier_off, w_tier_off = occ_cells(0.0)
        g_tier_on, w_tier_on = occ_cells(15.0)
        self.assertGreater(w_tier_on, 20)
        self.assertLessEqual(g_tier_off, 10)
        self.assertLess(g_tier_on, g_tier_off)

    def test_trajectory_z_drift_ignored(self) -> None:
        # 轨迹 z 漂了 0.4 m 且点跟着一起漂（同一个关键帧内一致）：高度相对该关键帧，结果不变。
        T = pose(1.0, 0, z=0.4)
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        m.add_keyframe(0, observe(self.scene, pose(1.0, 0)), T)
        ng = m.rasterize()
        self.assertEqual(value_at(ng, (2.5, 0.0)), FREE)

    def test_pose_update_moves_cells_no_residue(self) -> None:
        m = self.mapper_with([pose(0, 0), pose(1.0, 0)])
        before = m.rasterize()
        self.assertEqual(value_at(before, (3.52, -0.8)), OCC)
        # 回环把两个关键帧整体往 −x 挪 0.5 m：墙应该跟着到 x≈3.0，原位置不留残影。
        moved = m.update_poses({0: pose(-0.5, 0), 1: pose(0.5, 0)})
        self.assertAlmostEqual(moved, 0.5, places=6)
        after = m.rasterize()
        self.assertEqual(value_at(after, (3.02, -0.8)), OCC)
        self.assertNotEqual(value_at(after, (3.52, -0.8)), OCC)

    def test_origin_aligned_to_resolution(self) -> None:
        ng = self.mapper_with([pose(0.03, 0.07)]).rasterize()
        ox, oy = ng.meta.origin_xy_m
        self.assertAlmostEqual(ox / 0.10, round(ox / 0.10), places=6)
        self.assertAlmostEqual(oy / 0.10, round(oy / 0.10), places=6)

    def test_frontier_goal_is_walkable_and_connected(self) -> None:
        ng = self.mapper_with([pose(0, 0), pose(1.0, 0)]).rasterize()
        ng.build(radius_m=0.25)
        s = ng.meta.world_scale
        start = (0.5 * s, 0.0)
        fs = frontiers(ng, start)
        self.assertTrue(fs)
        for f in fs:
            self.assertEqual(ng.classify(f.goal_xy_m), "free_center")
            self.assertTrue(ng.plan(start, f.goal_xy_m).accepted)
        # 从起点看得到的 frontier 至少有一个在前方（墙上方 y>0.3 的通道方向）。
        self.assertTrue(any(f.goal_xy_m[0] / s > 2.0 for f in fs))

    def test_walked_breaks_on_pose_jump_not_backed_by_osc(self) -> None:
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        s = m.cfg.world_scale
        # 0→1 OSC 走了 0.8 m，SLAM 也 ≈0.75 m：连上；1→2 SLAM 跳 2 m 但 OSC 只走 0.3 m：断开。
        for i, (x, d) in enumerate([(0.0, 0.0), (1.0, 0.8), (3.0, 1.1), (3.5, 1.5)]):
            m.add_keyframe(i, np.zeros((0, 3)), pose(x, 0), d)
        polys = m.walked()
        self.assertEqual(len(polys), 2)
        self.assertAlmostEqual(float(polys[0][-1][0]), 1.0 * s, places=6)
        self.assertAlmostEqual(float(polys[1][0][0]), 3.0 * s, places=6)

    def test_trail_follows_keyframe_and_bridges_long_hops(self) -> None:
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        s = m.cfg.world_scale
        # 关键帧相距 4 m，OSC 只报 2.9 m（尺度误差）：光靠关键帧连线会断；中间逐帧轨迹每步都对得上。
        m.add_keyframe(0, np.zeros((0, 3)), pose(0, 0), 0.0)
        for i in range(1, 8):
            m.add_trail(0, pose(0.5 * i, 0), 0.5 * i * s)
        m.add_keyframe(1, np.zeros((0, 3)), pose(4.0, 0), 2.9)
        self.assertEqual(len(m.walked()), 1)
        # 回环把关键帧 0 往 +y 挪 1 m：挂在它上面的轨迹跟着走。
        m.update_poses({0: pose(0, 1.0)})
        mid = m.walked()[0][3]
        self.assertAlmostEqual(float(mid[1]), 1.0 * s, places=6)

    def test_frontier_empty_when_start_not_walkable(self) -> None:
        ng = self.mapper_with([pose(0, 0)]).rasterize()
        ng.build(radius_m=0.25)
        self.assertEqual(frontiers(ng, (50.0, 50.0)), [])


def ghost_block(x0: float, x1: float, y0: float, y1: float, h0: float, h1: float) -> np.ndarray:
    """地图系一块实心障碍点（离地 h0..h1），用来模拟某一帧的鬼影。"""
    gx, gy, gz = np.meshgrid(np.arange(x0, x1, 0.03), np.arange(y0, y1, 0.03), np.arange(h0, h1, 0.05))
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel() - CAM_H])


class DropPointsTests(unittest.TestCase):
    """原地补帧：去掉旧帧的点云 = 从没加过它；位姿、轨迹、锚点照旧。"""

    def test_drop_matches_build_without_it_and_keeps_pose(self) -> None:
        scene = world_scene()
        poses = [pose(0, 0), pose(0.02, 0.01), pose(1.0, 0)]
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        for i, T in enumerate(poses):
            m.add_keyframe(i, observe(scene, T), T, osc_dist_m=float(i))
        m.add_trail(1, pose(0.5, 0), osc_dist_m=1.5)
        m.rasterize()
        anc = m.anchor((0.03, 0.0))
        walked = m.walked()
        m.drop_points(1)
        m.drop_points(1)                                    # 重复调用无副作用
        ref = KeyframeGridMapper(MapperConfig(res_m=0.10))
        for i in (0, 2):
            ref.add_keyframe(i, observe(scene, poses[i]), poses[i])
        a, b = m.rasterize(), ref.rasterize()
        self.assertEqual(len(m), 2)
        np.testing.assert_array_equal(a.grid, b.grid)
        self.assertEqual(anc[0], 1)
        self.assertIsNotNone(m.resolve(anc))
        self.assertEqual(len(m.walked()), len(walked))
        np.testing.assert_allclose(np.vstack(m.walked()), np.vstack(walked))

    def test_drop_points_clamps_the_cam_h_throttle_counter(self) -> None:
        """``_cam_h_at`` 必须跟着 len(_pts) 一起缩，否则 cam_h 会被冻结。

        它记录的是「上次改 cam_h 时的关键帧数」，而 len(_pts) 可能下降（离线回放
        tools/q_tier_ab.py:87-92 是「先摘 k-1、可能再摘 k、最后加 k」）。一旦
        _cam_h_at 越过 len(_pts)，_update_cam_h 的 d 变负 ⇒ 恒 < 阈值 ⇒ 走 return，
        cam_h **不再更新**（注意：是冻结，不是"更频繁重算"），且要多等
        (_cam_h_at - len) 帧才恢复。夹回之后等待帧数就是设计值。
        """
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        scene = world_scene()
        for i, T in enumerate([pose(0, 0), pose(0.5, 0), pose(1.0, 0)]):
            m.add_keyframe(i, observe(scene, T), T)
        m._cam_h_at = len(m)                       # 假装刚在满规模时改过 cam_h
        self.assertEqual(m._cam_h_at, 3)
        m.drop_points(0)
        m.drop_points(1)
        self.assertEqual(m._cam_h_at, 1)           # 跟着缩到 1，而不是停在 3
        self.assertLessEqual(m._cam_h_at, len(m))


class RayClearTests(unittest.TestCase):
    """射线清除：墙留着、只在一帧里出现又被后来视线看穿的鬼影清掉、走过的中心线清掉；回环不重追线。"""

    def setUp(self) -> None:
        self.scene = world_scene()
        # 第 0 帧从 (−0.5, 0) 看到 x∈[2, 2.3] 的一块假障碍，之后几帧都没有它，而且视线从它中间穿过去看地面。
        self.ghost = ghost_block(2.0, 2.3, 0.5, 1.0, 0.35, 0.6)
        self.poses = [pose(-0.5, 0), pose(0, 0), pose(0.5, 0), pose(1.0, 0), pose(1.5, 0)]

    def build(self, ray_clear: bool) -> KeyframeGridMapper:
        m = KeyframeGridMapper(MapperConfig(res_m=0.10, ray_clear=ray_clear))
        for i, T in enumerate(self.poses):
            scene = np.vstack([self.scene, self.ghost]) if i == 0 else self.scene
            m.add_keyframe(i, observe(scene, T), T)
        return m

    def test_ghost_seen_through_is_cleared_wall_is_kept(self) -> None:
        base = self.build(False).rasterize()
        self.assertEqual(value_at(base, (2.15, 0.75)), OCC)       # 旧规则：一帧的鬼影压过了地面
        m = self.build(True)
        ng = m.rasterize()
        self.assertEqual(value_at(ng, (2.15, 0.75)), FREE)
        self.assertGreater(m.ray_cleared_cells, 0)
        # 每帧都打中的墙不受影响（视线止于墙面前 ray_stop_m，不会自己看穿自己）。
        for y in (-1.2, -0.8, -0.3, 0.1):
            self.assertEqual(value_at(ng, (3.52, y)), OCC)
            self.assertEqual(value_at(base, (3.52, y)), OCC)

    def test_ray_clear_off_changes_nothing(self) -> None:
        m = self.build(False)
        ng = m.rasterize()
        self.assertEqual(m.ray_cleared_cells, 0)
        self.assertIsNone(m._ray_hit)
        self.assertEqual(value_at(ng, (3.52, -0.8)), OCC)

    def test_ray_near_exempt_off_changes_nothing(self) -> None:
        # 默认关：veto 掩码必须与改动前逐格相同。跑两遍比栅格不够——把 _ray_veto 的返回值比掉。
        got: list[np.ndarray] = []

        class Spy(KeyframeGridMapper):
            def _ray_veto(self, walked):
                v = super()._ray_veto(walked)
                got.append(v.copy())
                return v

        for _ in (0, 1):
            m = Spy(MapperConfig(res_m=0.10))
            for i, T in enumerate(self.poses):
                scene = np.vstack([self.scene, self.ghost]) if i == 0 else self.scene
                m.add_keyframe(i, observe(scene, T), T)
            m.rasterize()
        self.assertEqual(len(got), 2)
        self.assertTrue(np.array_equal(got[0], got[1]))
        self.assertGreater(int(got[0].sum()), 0)          # 确实有格被清，不是空掩码糊弄过去

    def _plane_grid(self, mean_h: np.ndarray, on_ratio: float = 0.8):
        """摆一张累加器网格，让 ``_near_plane`` 的门只由 mean_h / on_ratio 决定。

        ``mean_h`` 给每格的**平均**高度——平面这一项看的就是它（见 ``_near_plane_parts``
        里三个变体的对照：换成 ``_ray_hit`` 的众数层在 045615 上是净亏）。
        """
        c = MapperConfig(ray_clear=True)
        m = KeyframeGridMapper(c)
        m._acc_lo = (0, 0)
        m._acc_g = np.zeros(mean_h.shape)
        m._acc_o = np.full(mean_h.shape, 30.0)                       # ≥ ray_plane_pts
        m._acc_on = np.full(mean_h.shape, 30.0 * on_ratio)           # 近距成分
        m._acc_ob_h = m._acc_o * mean_h
        m._acc_ob_h2 = m._acc_o * mean_h * mean_h
        nlay = int(np.ceil((c.obst_top_m - c.ground_tol_m) / c.ray_z_m))
        hit = np.zeros((nlay, *mean_h.shape), np.int32)
        mis = np.zeros((nlay, *mean_h.shape), np.int32)
        hit[5] = 1
        mis[5] = 10                                                   # 看穿远多于打中 ⇒ 默认会被清
        m._ray_hit, m._ray_mis = hit, mis
        return m

    def test_ray_near_exempt_gate_is_geometry_not_vote_count(self) -> None:
        """门必须是"有没有一张连贯的面"，不是"票多不多"。

        平的那张：默认被看穿清掉，豁免后保回来。
        高度乱跳的那张（不是一张面）：豁免也不该动它——纯远场假障碍照清。
        """
        flat = np.full((5, 5), 0.85)
        m = self._plane_grid(flat)
        # 只断言中心区：5×5 网格的角点在 3×3 窗口里只有 4 个有效邻居，够不到 cnt≥6，
        # 那是有意的（图沿的"高度一致"是采样边界凑出来的，不作数）。
        self.assertTrue(m._near_plane()[1:4, 1:4].all(), "等高近距网格应当判为有面")
        self.assertTrue(m._ray_veto(None)[2, 2], "默认应被判成看穿而清掉")
        m.cfg.ray_near_exempt = True
        self.assertFalse(m._ray_veto(None)[2, 2], "豁免后应当保它")

        noisy = np.tile(np.array([0.35, 0.85, 1.60, 0.85, 1.60]), (5, 1))   # 邻域极差 1.25 m
        m2 = self._plane_grid(noisy)
        self.assertFalse(m2._near_plane().any(), "高度乱跳的网格不是一张面")
        m2.cfg.ray_near_exempt = True
        self.assertTrue(m2._ray_veto(None)[2, 2], "不是面 ⇒ 豁免不动它")

    def test_ray_near_exempt_gate_needs_near_votes(self) -> None:
        """同一张面，但票几乎全是远距离打的 ⇒ 不得豁免。

        δz=z²/25.5，3 m 外已经 35 cm，"远处的面"和"近处的面"不是一回事，
        不能只看形状就把远场票的假障碍也保回来。
        """
        m = self._plane_grid(np.full((5, 5), 0.85), on_ratio=0.05)
        self.assertFalse(m._near_plane().any(), "远距票不构成近距平面证据")
        m.cfg.ray_near_exempt = True
        self.assertTrue(m._ray_veto(None)[2, 2], "纯远场仍应被看穿清掉")

    def test_walked_centerline_clears_obstacle_nobody_saw_through(self) -> None:
        # 障碍旁边只有一条侧面的地面（定地面高度用），没有视线穿过它，只能靠"身体从这里走过去了"。
        gx, gy = np.meshgrid(np.arange(0.3, 3.0, 0.03), np.arange(0.8, 1.5, 0.03))
        side = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -CAM_H)])
        blob = np.vstack([ghost_block(1.4, 1.7, -0.3, 0.3, 0.5, 1.0), side])

        def occ_on_path(osc_backed: bool) -> int:
            m = KeyframeGridMapper(MapperConfig(res_m=0.10))
            s = m.cfg.world_scale
            m.add_keyframe(0, observe(blob, pose(0, 0)), pose(0, 0), 0.0)
            # OSC 路程对得上 → 连成走廊；没有 OSC 时 3 m 一跳超过硬门限 → 断开，没有走廊。
            m.add_keyframe(1, np.zeros((0, 3)), pose(3.0, 0), 3.0 * s if osc_backed else None)
            return int(value_at(m.rasterize(), (1.55, 0.0)) == OCC)

        self.assertEqual(occ_on_path(False), 1)
        self.assertEqual(occ_on_path(True), 0)

    def test_loop_shift_reuses_rays_and_matches_fresh_build(self) -> None:
        from neko_anyadance_body.backend import nav_mapping as NM

        calls = []
        real = NM._ray_voxels

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        with mock.patch.object(NM, "_ray_voxels", counting):
            m = self.build(True)
            m.rasterize()
            self.assertEqual(len(calls), 5)
            m.rasterize()                                          # 什么都没变：不追线
            self.assertEqual(len(calls), 5)
            # 回环整体平移 0.5 m（整格）：只挪下标，不重追线。
            shifted = {i: pose(-1.0 + 0.5 * i, 0) for i in range(5)}
            m.update_poses(shifted)
            inc = m.rasterize()
            self.assertEqual(len(calls), 5)
            m.add_keyframe(5, observe(self.scene, pose(2.0, 0)), pose(1.5, 0))
            m.rasterize()                                          # 新关键帧只追它自己
            self.assertEqual(len(calls), 6)
        # 增量结果与直接按终态位姿建图一致（平移是整格，没有取整差）。
        fresh = KeyframeGridMapper(MapperConfig(res_m=0.10))
        for i, T in enumerate(self.poses):
            scene = np.vstack([self.scene, self.ghost]) if i == 0 else self.scene
            fresh.add_keyframe(i, observe(scene, T), shifted[i])
        ref = fresh.rasterize()
        self.assertEqual(inc.grid.shape, ref.grid.shape)
        self.assertTrue((inc.grid == ref.grid).all())
        self.assertEqual(value_at(ref, (1.65, 0.75)), FREE)       # 鬼影跟着挪了 0.5 m，仍被清掉
        self.assertGreaterEqual(int(m._ray_hit.min()), 0)
        self.assertGreaterEqual(int(m._ray_mis.min()), 0)


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = world_scene()
        self.m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        self.s = self.m.cfg.world_scale

    def add(self, i: int, T: np.ndarray) -> None:
        self.m.add_keyframe(i, observe(self.scene, T), T)

    def est(self, x: float, y: float, th: float = 0.0) -> dict:
        return {"state": "localized", "xy": (x * self.s, y * self.s), "theta": th, "sigma_m": 0.2}

    def test_anchor_follows_loop_closure(self) -> None:
        self.add(0, pose(0, 0))
        self.add(1, pose(1.0, 0, yaw=0.3))
        anc = self.m.anchor((2.0, 0.5))
        self.assertEqual(anc[0], 1)
        # 回环：节点 1 平移 (−0.4, 0.2) 并多转 0.1 rad，目标应随节点刚体变换。
        self.m.update_poses({1: pose(0.6, 0.2, yaw=0.4)})
        got = self.m.resolve(anc)
        d = np.array([1.0, 0.5])
        c, s = math.cos(0.1), math.sin(0.1)
        want = np.array([0.6, 0.2]) + np.array([[c, -s], [s, c]]) @ d
        self.assertAlmostEqual(got[0], want[0], places=6)
        self.assertAlmostEqual(got[1], want[1], places=6)

    def test_goal_in_unknown_is_blocked_until_seen(self) -> None:
        self.add(0, pose(0, 0))
        sess = NavSession(self.m)
        sess.goto((2.8 * self.s, 0.9 * self.s))          # 离关键帧 3 m 外：还没看见
        info = sess.on_map_update(self.est(0.5, 0.0))
        self.assertFalse(info["accepted"])
        self.assertTrue(info["reason"].startswith("goal_"))
        self.assertEqual(sess.step(self.est(0.5, 0.0))["state"], "blocked")
        # 往前走一个关键帧后看见了，同一个目标就能规划。
        self.add(1, pose(1.0, 0))
        info = sess.on_map_update(self.est(0.5, 0.0))
        self.assertTrue(info["accepted"], info)
        self.assertIn(sess.step(self.est(0.5, 0.0))["state"], ("following", "turning"))

    def test_explore_picks_reachable_frontier(self) -> None:
        self.add(0, pose(0, 0))
        self.add(1, pose(1.0, 0))
        sess = NavSession(self.m)
        sess.explore()
        info = sess.on_map_update(self.est(0.5, 0.0))
        self.assertTrue(info["accepted"], info)
        g = sess.goal_xy()
        self.assertEqual(sess.ng.classify(g), "free_center")

    def test_replans_on_loop_closure(self) -> None:
        self.add(0, pose(0, 0))
        self.add(1, pose(1.0, 0))
        sess = NavSession(self.m)
        sess.goto((2.0 * self.s, 0.5 * self.s))
        self.assertTrue(sess.on_map_update(self.est(0.5, 0.0))["accepted"])
        g0 = sess.goal_xy()
        self.m.update_poses({0: pose(-0.3, 0), 1: pose(0.7, 0)})
        info = sess.on_map_update(self.est(0.2, 0.0))
        self.assertTrue(info["accepted"], info)
        g1 = sess.goal_xy()
        self.assertAlmostEqual((g0[0] - g1[0]) / self.s, 0.3, places=5)
        self.assertAlmostEqual(info["waypoints_xy_m"][-1][0], g1[0], places=3)


    def test_stale_plan_does_not_override_newer_intent(self) -> None:
        self.add(0, pose(0, 0))
        self.add(1, pose(1.0, 0))
        sess = NavSession(self.m)
        sess.goto((2.0 * self.s, 0.5 * self.s))
        snap = sess.snapshot()
        res = sess.compute(self.est(0.5, 0.0), snap)     # 建图线程锁外在算……
        sess.cancel()                                     # ……这时用户取消了
        info = sess.apply(res)
        self.assertEqual(info["reason"], "superseded")
        self.assertEqual(sess.mode, "idle")
        self.assertIsNone(sess.follower)
        self.assertIs(sess.ng, res["ng"])                 # 栅格照样换新

    def test_async_replan_waits_instead_of_blocking(self) -> None:
        self.add(0, pose(0, 0))
        self.add(1, pose(1.0, 0))
        asked = []
        sess = NavSession(self.m, request_update=lambda: asked.append(1))
        sess.goto((2.0 * self.s, 0.5 * self.s))
        self.assertEqual(sess.step(self.est(0.5, 0.0))["state"], "wait_map")
        self.assertTrue(sess.on_map_update(self.est(0.5, 0.0))["accepted"])
        with mock.patch.object(sess.follower, "step", return_value={"state": "replan", "forward": 0.3}):
            out = sess.step(self.est(0.5, 0.0))
        self.assertEqual((out["forward"], asked), (0.0, [1]))
        self.assertEqual(sess.step(self.est(0.5, 0.0))["reason"], "replanning")


class StereoPointsTests(unittest.TestCase):
    def test_flat_disparity_gives_points_ahead_in_base_frame(self) -> None:
        rng = np.random.default_rng(0)
        left = (rng.random((120, 160)) * 255).astype(np.uint8)
        d = 8
        right = np.roll(left, -d, axis=1)
        pts = stereo_points(left, right, fx=100.0, cx=80.0, cy=60.0, baseline_m=0.1, max_range_m=5.0)
        self.assertGreater(len(pts), 500)
        z = np.median(pts[:, 0])
        self.assertAlmostEqual(float(z), 100.0 * 0.1 / d, delta=0.05)
        # 光学 x 右 → base −y：像素右半边的点 y 为负。
        self.assertLess(float(np.median(pts[pts[:, 1] < 0, 1])), 0)


class CoverageCountsTests(unittest.TestCase):
    """覆盖伴生网格（_acc_near/_acc_far）：分桶正确、与 seen 严格一致、可逆、随回环平移、cam_h 重建一致。"""

    def setUp(self) -> None:
        self.scene = world_scene()

    def mapper_with(self, poses, **kw):
        m = KeyframeGridMapper(MapperConfig(res_m=0.10, **kw))
        for i, T in enumerate(poses):
            m.add_keyframe(i, observe(self.scene, T), T)
        return m

    @staticmethod
    def _seen_cells(arr) -> set:
        rows, cols = np.nonzero(arr)
        return {(int(r), int(c)) for r, c in zip(rows, cols)}

    def _cov(self, m):
        m.rasterize()                         # 触发增量同步（幂等：无 dirty 时为 no-op）
        cc = m.coverage_counts()
        return cc, self._seen_cells((cc["near"] + cc["far"]) > 0.5)

    def test_coverage_counts_accumulate(self) -> None:
        m = self.mapper_with([pose(0, 0), pose(1.0, 0)], cov_near_m=1.5)
        cc, cells = self._cov(m)
        self.assertTrue(cells)
        # 不变量：near+far 的格集合与 rasterize 用的 seen（acc_g|acc_o）严格一致（同一批选中点）。
        self.assertEqual(cells, self._seen_cells((m._acc_g > 0.5) | (m._acc_o > 0.5)))
        self.assertTrue(np.array_equal(cc["near"] + cc["far"], m._acc_g + m._acc_o))
        # 两个桶都有内容：observe() 只留 3 m 内的点，cov_near_m=1.5 时近/远各应存在。
        self.assertGreater(float(cc["near"].sum()), 0.0)
        self.assertGreater(float(cc["far"].sum()), 0.0)

    def test_coverage_counts_reversible(self) -> None:
        m = self.mapper_with([pose(0, 0), pose(1.0, 0)], cov_near_m=1.5)
        base_cc, base_cells = self._cov(m)
        m0 = self.mapper_with([pose(0, 0)], cov_near_m=1.5)
        cc0, cells0 = self._cov(m0)
        m.drop_points(1)                      # 原地补帧路径：撤掉 kf1 的点
        _, cells = self._cov(m)
        # 两个 mapper 的累加器形状/原点可以不同，比格集合与总量。
        self.assertEqual(cells, cells0)
        cc, _ = self._cov(m)
        self.assertAlmostEqual(float(cc["near"].sum()), float(cc0["near"].sum()), places=3)
        self.assertAlmostEqual(float(cc["far"].sum()), float(cc0["far"].sum()), places=3)
        m.add_keyframe(1, observe(self.scene, pose(1.0, 0)), pose(1.0, 0))
        cc2, cells2 = self._cov(m)
        self.assertEqual(cells2, base_cells)  # 加减同一批数，精确还原
        self.assertAlmostEqual(float(cc2["near"].sum()), float(base_cc["near"].sum()), places=3)
        self.assertAlmostEqual(float(cc2["far"].sum()), float(base_cc["far"].sum()), places=3)

    def test_coverage_counts_loop_shift(self) -> None:
        m = self.mapper_with([pose(0, 0), pose(1.0, 0)], cov_near_m=1.5)
        m.rasterize()
        cc0, cells0 = self._cov(m)
        total0 = float(cc0["near"].sum() + cc0["far"].sum())
        m.update_poses({0: pose(0.5, 0), 1: pose(1.5, 0)})    # 回环整体挪 0.5 m = 5 格
        m.rasterize()
        cc1, cells1 = self._cov(m)
        self.assertAlmostEqual(total0, float(cc1["near"].sum() + cc1["far"].sum()), places=3)
        self.assertEqual(cells1, {(r, c + 5) for r, c in cells0})   # 跟着平移，无残影

    def test_coverage_cam_h_rebuild(self) -> None:
        # 地面高度变了 → 直方图中位越滞回 → cam_h 换 → 累加器连 near/far 一起清空重建，不变量仍成立。
        # 两个 hi 关键帧让新高度簇占多数（中位估计才会跨过滞回带）。
        hi = self.scene.copy()
        hi[:, 2] -= 0.15                      # 相机离地比 CAM_H 高 15 cm
        m = KeyframeGridMapper(MapperConfig(res_m=0.10, cov_near_m=1.5))
        m.add_keyframe(0, observe(self.scene, pose(0, 0)), pose(0, 0))
        self._cov(m)
        h_before = m.cam_h
        m.add_keyframe(1, observe(hi, pose(2.0, 0)), pose(2.0, 0))
        m.add_keyframe(2, observe(hi, pose(3.0, 0)), pose(3.0, 0))
        self._cov(m)
        self.assertNotEqual(m.cam_h, h_before)
        cc, cells = self._cov(m)
        self.assertEqual(cells, self._seen_cells((m._acc_g > 0.5) | (m._acc_o > 0.5)))
        self.assertTrue(np.array_equal(cc["near"] + cc["far"], m._acc_g + m._acc_o))

    def test_cov_near_boundary(self) -> None:
        # 体素中心恰在阈值两侧：2.975 → 近，3.025 → 远（体素中心 = (floor(p/0.05)+0.5)·0.05）。
        pts = np.array([[2.97, 0.0, -CAM_H], [3.03, 0.0, -CAM_H]], np.float32)
        m = KeyframeGridMapper(MapperConfig(res_m=0.10, cov_near_m=3.0))
        m.add_keyframe(0, pts, pose(0, 0))
        m.rasterize()
        cc, _ = self._cov(m)
        self.assertAlmostEqual(float(cc["near"].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(cc["far"].sum()), 1.0, places=6)


class QualityTierTests(unittest.TestCase):
    """观测质量分层：按"票是几米外打的"复核扁平计数的障碍判定。

    物理依据：视差深度误差 δz = z²/(fx·b)·δd，fx 202.5 / 基线 0.126 下 1 像素 ≈ z²/25.5 m
    （3 m→35 cm、5 m→98 cm），而障碍高度带 (0.3, 2.0) 有 1.7 m 宽。1.5 m 以外真地面被抖进
    障碍带是必然的，min_pts / 3×3 多数 / occ_ground_ratio 只压密度补不回信息。
    """

    NEAR, MID = 1.5, 3.0            # 与 MapperConfig 默认 q_near_m / q_mid_m 对齐

    def one_shot(self, at: tuple[float, float], *blocks: np.ndarray, **kw) -> KeyframeGridMapper:
        """单关键帧：站在 at 看一次 ``blocks``（点已是 base 系，机位在原点）。

        场景**必须带真地板**：``_ground_correction`` 与 cam_h 直方图都是拿地面当参照的，
        没有地板时它们会把墙本身当地面拟合（实测 cam_h 漂到 1.26，墙被判成空地），
        测的就不是分层规则而是地面估计的边角行为。
        """
        m = KeyframeGridMapper(MapperConfig(res_m=0.10, **kw))
        T = pose(*at)
        m.add_keyframe(0, np.vstack([world_scene(), *blocks]).astype(np.float32), T)
        return m

    def test_far_only_phantom_demoted_to_unknown_not_free(self) -> None:
        # 4 m 外的孤岛：票全在远带。旧规则判障碍；分层后不再是障碍——
        # 但那块地板也只在 4 m 外看过（远带地面同样不可信），所以降级成 unknown 而不是 free。
        blob = ghost_block(4.0, 4.6, -0.3, 0.3, 0.5, 1.0)
        cell = (4.3, 0.0)
        self.assertEqual(value_at(self.one_shot((0, 0), blob, q_tiers=False).rasterize(), cell), OCC)
        self.assertEqual(value_at(self.one_shot((0, 0), blob).rasterize(), cell), UNK)

    def test_far_phantom_over_near_seen_floor_becomes_free(self) -> None:
        # 真实的假障碍长法：远机位（3 m 外）把地板抖成了障碍票，后来走近了、1 m 内看清是地板，
        # 但远机位的票还在累加器里。没有近距地面证据时它只能降级成 unknown；走近看清了就判回空地。
        # 这就是"走过的地方变障碍、点了不动"的正解。
        blob = ghost_block(2.9, 3.3, -0.3, 0.3, 0.5, 1.0)
        cell = (3.1, 0.0)

        def walk(**kw) -> object:
            m = KeyframeGridMapper(MapperConfig(res_m=0.10, **kw))
            m.add_keyframe(0, np.vstack([world_scene(), blob]).astype(np.float32), pose(0, 0))
            m.add_keyframe(1, world_scene().astype(np.float32), pose(2.5, 0))   # 1 m 内看清是地板
            return m.rasterize()

        self.assertEqual(value_at(walk(q_tiers=False), cell), OCC)   # 旧规则：走近了也还是障碍
        self.assertEqual(value_at(walk(), cell), FREE)

    def test_near_confirmed_obstacle_survives(self) -> None:
        # 1 m 内看的墙：近距票即决定性，任何分层参数都该留住它。
        wall = ghost_block(0.8, 1.0, -0.6, 0.6, 0.4, 1.2)
        self.assertEqual(value_at(self.one_shot((0, 0), wall).rasterize(), (0.9, 0.0)), OCC)

    def test_single_keyframe_dense_mid_wall_survives(self) -> None:
        # 只被一个关键帧看到的中距墙（刚走近就看见了）：票全在中距、够密 ⇒ 单帧例外定案。
        # 这是"分层别把真墙误杀"的保证。关掉单帧例外后它降级——但**降级成 unknown 而不是 free**，
        # 墙仍然不可走（nav_grid：unknown 不进可走区），所以不会因为分层就穿墙。
        wall = ghost_block(2.2, 2.8, -0.5, 0.5, 0.4, 1.2)
        cell = (2.5, 0.0)
        self.assertEqual(value_at(self.one_shot((0, 0), wall).rasterize(), cell), OCC)
        strict = self.one_shot((0, 0), wall, q_solo_pts=10 ** 6).rasterize()
        self.assertEqual(value_at(strict, cell), UNK)
        self.assertNotEqual(value_at(strict, cell), FREE)
        strict.build(radius_m=0.25, walked=[])
        self.assertFalse(strict.center[strict.to_cell((cell[0] * strict.meta.world_scale,
                                                       cell[1] * strict.meta.world_scale))])

    def test_mid_needs_multi_frame_confirmation(self) -> None:
        # 中距多帧确认这一条本身：中距鬼影被两个机位看到才定案，只有一个看到就不定案。
        # 两条都关掉单帧例外（q_solo_pts 抬到天上）测纯多帧逻辑；ray_clear 也关掉，
        # 否则第二个机位看穿它的那条视线会先把票清掉，测的就不是分层了。
        ghost = ghost_block(2.2, 2.4, -0.05, 0.05, 0.5, 0.7)
        cell = (2.3, 0.0)
        kw = dict(q_solo_pts=10 ** 6, ray_clear=False)

        def seen_by(*ats):
            m = KeyframeGridMapper(MapperConfig(res_m=0.10, **kw))
            for i, at in enumerate(ats):
                m.add_keyframe(i, np.vstack([world_scene(), ghost]).astype(np.float32), pose(*at))
            return m.rasterize()

        self.assertEqual(value_at(self.one_shot((0, 0), ghost, q_tiers=False, ray_clear=False).rasterize(),
                                  cell), OCC)                    # 旧规则：单帧就判障碍
        self.assertNotEqual(value_at(seen_by((0, 0)), cell), OCC)  # 只看过一次 ⇒ 不定案
        # 第二个机位错开半格（0.05 m = 一个体素）：两帧的体素正好对进同一批格，才谈得上"多帧一致"。
        self.assertEqual(value_at(seen_by((0, 0), (0.05, 0.0)), cell), OCC)
        m3 = KeyframeGridMapper(MapperConfig(res_m=0.10, ray_clear=False,
                                             q_solo_pts=10 ** 6, q_mid_kf=3))
        for i, at in enumerate(((0, 0), (0.05, 0.0))):
            m3.add_keyframe(i, np.vstack([world_scene(), ghost]).astype(np.float32), pose(*at))
        # q_mid_kf 抬到 3 又变回不定案：门槛是跟着配的，不是写死"两个"。
        self.assertNotEqual(value_at(m3.rasterize(), cell), OCC)
    def test_q_tiers_off_reproduces_flat_count_rule(self) -> None:
        # 开关关掉必须**逐格**等于历史规则：别让新参数在关闭时还悄悄改别的东西。
        scene = np.vstack([world_scene(), ghost_block(2.9, 3.3, -0.3, 0.3, 0.5, 1.0)])
        poses = [pose(0, 0), pose(1.0, 0.4), pose(2.0, -0.3)]
        grids = []
        for tiers in (False, True):
            m = KeyframeGridMapper(MapperConfig(res_m=0.10, q_tiers=tiers))
            for i, T in enumerate(poses):
                m.add_keyframe(i, observe(scene, T), T)
            grids.append(m.rasterize().grid)
        self.assertEqual(grids[0].shape, grids[1].shape)
        self.assertGreater(int((grids[0] == OCC).sum() - (grids[1] == OCC).sum()), 0)  # 确实降了

    def test_q_near_pts_off_matches_legacy_formula(self) -> None:
        """``q_near_pts=0``（默认）必须与改动前的公式**逐格**一致。

        不靠"跑两遍一样"那种自证——把 ``_by_quality`` 的入参截下来，按**旧公式**重算一遍
        再比 occ/restored/rejected 三个返回值。``near_only`` 恒 False 时新公式退化成
        ``occ = (base_occ>0) & ok``、``rejected = (base_occ>0) & ~ok``，这才是真正的 no-op。
        """
        box: dict[str, object] = {}

        class Spy(KeyframeGridMapper):
            def _by_quality(self, base_occ, n_g, n_o, o_n, o_m, g_n, o_mk):
                out = super()._by_quality(base_occ, n_g, n_o, o_n, o_m, g_n, o_mk)
                box["in"] = (base_occ.copy(), n_g.copy(), n_o.copy(),
                             o_n.copy(), o_m.copy(), g_n.copy(), o_mk.copy())
                box["out"] = tuple(a.copy() for a in out)
                return out

        scene = np.vstack([world_scene(), ghost_block(1.9, 2.4, -0.3, 0.3, 0.4, 1.2)])
        m = Spy(MapperConfig(res_m=0.10))
        for i, T in enumerate([pose(0, 0), pose(0.6, 0.3), pose(1.2, -0.2)]):
            m.add_keyframe(i, observe(scene, T), T)
        m.rasterize()

        base_occ, n_g, n_o, o_n, o_m, g_n, o_mk = box["in"]
        occ, restored, rejected = box["out"]
        c = MapperConfig()
        hi = c.min_pts - 0.5
        near = o_n > hi
        mid = (o_m > hi) & (o_mk >= c.q_mid_kf - 0.5) & (n_o >= c.occ_ground_ratio * n_g)
        far = np.maximum(n_o - o_n - o_m, 0.0)
        solo = (o_m > hi) & (n_o >= c.q_solo_pts) & (far <= c.q_far_tol * np.maximum(n_o, 1.0))
        ok = near | mid | solo
        self.assertTrue(np.array_equal(occ, (base_occ > 0) & ok))
        self.assertTrue(np.array_equal(rejected, (base_occ > 0) & ~ok))
        self.assertTrue(np.array_equal(restored, (base_occ > 0) & ~ok & (g_n > 0.5)))

    def test_q_near_pts_low_rescues_near_thin_obstacle(self) -> None:
        """近距、票少、但背后地面清楚的**细**障碍：关着不是障碍，调低门槛后应当是障碍。

        直接调 ``_by_quality`` 并喂手工数组，不走整条管线——要复现的是
        ``tools/near_band_check.py`` 在 044153 上量到的那种形状（n_g 中位 130 / n_o 中位 26）：
        ``n_o`` 远大于 min_pts，却因为 ``n_o < occ_ground_ratio * n_g`` 被 base 规则判 0。
        用真实场景去凑这个比值要靠点的疏密碰运气，测的就不是这条规则了。
        """
        m = KeyframeGridMapper(MapperConfig())
        c = MapperConfig()
        shape = (1, 6)
        n_g = np.full(shape, 130.0)
        n_o = np.array([[26.0, 26.0, 0.0, 40.0, 3.0, 3.0]])
        # o_n 近距票：前两格全是近距（thin obstacle 的近距观测），后几格分别对应
        # 远距、地面压制不够、票不足、票不足
        o_n = np.array([[26.0, 26.0, 0.0, 1.0, 0.0, 0.0]])
        base = np.array([[0, 0, 0, 1, 0, 0]], np.uint8)   # 只有第 3 格过了扁平规则

        def call(q_near_pts: int) -> tuple:
            m.cfg.q_near_pts = q_near_pts
            occ, restored, rejected = m._by_quality(
                base, n_g, n_o, o_n, np.zeros(shape), np.zeros(shape), np.zeros(shape))
            return occ

        off = call(0)
        on1 = call(1)
        self.assertFalse(off[0, 0], "默认关闭时近距细障碍不该被这条规则救回")
        self.assertTrue(on1[0, 0], "q_near_pts=1 应当救回近距票够但被地面压制挡掉的格")
        self.assertFalse(on1[0, 2], "纯远距票不能被救——判据是**近距**票，不是单纯降门槛")
        self.assertTrue(on1[0, 3], "已经过了扁平规则的格不受影响")

    def test_bands_follow_configured_ranges(self) -> None:
        # 分带边界跟着 q_near_m/q_mid_m 走，不是写死的常数。
        wall = ghost_block(2.0, 2.6, -0.3, 0.3, 0.4, 1.2)
        cell = (2.3, 0.0)
        # 把 q_near_m 抬到 3.0 ⇒ 2.3 m 处变成"近距" ⇒ 单帧即定案。
        self.assertEqual(value_at(self.one_shot((0, 0), wall, q_near_m=3.0).rasterize(), cell), OCC)
        # 把 q_mid_m 压到 1.0 ⇒ 同一块只剩远场票 ⇒ 降级。
        all_far = self.one_shot((0, 0), wall, q_near_m=0.5, q_mid_m=1.0).rasterize()
        self.assertNotEqual(value_at(all_far, cell), OCC)


class HeightBandTests(unittest.TestCase):
    """高度分带占用：把 ``obst_top_m`` 之上（和地面之下）被整段丢弃的高度捡回来。

    这层是**旁路产物**，不喂导航。所以本类最要紧的不是"分带对不对"，而是
    **地面层三态栅格必须逐格不变**——分带只多记几段高度，没资格改动任何既有判定。
    """

    @staticmethod
    def stack() -> np.ndarray:
        """地面 + 一堵 0.4~1.2 m 的墙（world_scene 自带）+ 头顶 2.6 m 的一块板（模拟二楼/天桥）。"""
        return np.vstack([world_scene(), ghost_block(2.2, 2.8, -0.6, 0.6, 2.6, 2.8)]).astype(np.float32)

    def build(self, **kw) -> KeyframeGridMapper:
        m = KeyframeGridMapper(MapperConfig(res_m=0.10, **kw))
        m.add_keyframe(0, self.stack(), pose(0, 0))
        return m

    def band_at(self, m: KeyframeGridMapper, xy: tuple[float, float]) -> list[float]:
        """某格四条带的累计点数（取 band_grid 的 NavGrid 序，行 0 = 最大 y）。"""
        bg = m.band_grid()
        self.assertIsNotNone(bg)
        r, c = self._cell(m, xy)
        return [float(bg[b][r, c]) for b in range(4)]

    @staticmethod
    def _cell(m: KeyframeGridMapper, xy: tuple[float, float]) -> tuple[int, int]:
        """地图系(追踪米) → (NavGrid 行, 列)。按 rasterize 的映射手算，不依赖内部缓存。"""
        bg = m.band_grid()
        h, w = bg.shape[1], bg.shape[2]
        (ax, ay) = m._acc_lo
        lx, ly = m._grid_lo
        res = m.cfg.res_m
        acc_r = int(np.floor(xy[1] / res)) - ay        # 累加器行
        acc_c = int(np.floor(xy[0] / res)) - ax        # 累加器列
        unflipped = acc_r - (ly - ay)                   # n_g 行 = 累加器行 + (ly - ay)
        return h - 1 - unflipped, acc_c - (lx - ax)

    def test_ground_layer_is_bit_identical_with_and_without_bands(self) -> None:
        # 这是本改动唯一不可让步的性质：分带开关不许动地面层一个格。
        a = self.build(hi_bands=False).rasterize().grid
        b = self.build(hi_bands=True).rasterize().grid
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)

    def test_above_obst_top_lands_in_a_band_not_in_obstacle(self) -> None:
        # 2.6 m 的板在 obst_top_m=2.0 之上：地面层判不了它，带里必须有票。
        m = self.build()
        ng = m.rasterize()
        self.assertGreater(self.band_at(m, (2.5, 0.0))[1], 0)      # lo 带 = (2.0, 3.5]
        self.assertEqual(value_at(ng, (2.5, 0.0)), FREE)           # 地面层没把它当障碍

    def test_band_edges_follow_config(self) -> None:
        # 带边界跟着 hi_band_m / hi_band_top_m 走，不是写死的常数。
        m = self.build(hi_band_m=2.2, hi_band_top_m=2.4)
        m.rasterize()
        self.assertEqual(m.band_counts()["edges_m"], [-0.3, 2.0, 2.2, 2.4])
        v = self.band_at(m, (2.5, 0.0))
        self.assertEqual(v[1], 0.0)           # 2.6 m 不再落 lo 带
        self.assertGreater(v[3], 0.0)         # 落到最高带

    def test_bands_off_gives_empty(self) -> None:
        m = self.build(hi_bands=False)
        m.rasterize()
        self.assertIsNone(m.band_grid())
        self.assertIsNone(m.band_counts())

    def test_bands_are_reversible_on_drop(self) -> None:
        # 增删记账：加一帧再撤掉，累加器要精确回到原点（refresh 走的就是这条路）。
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        m.add_keyframe(0, self.stack(), pose(0, 0))
        m.rasterize()
        base = [b.copy() for b in (m._acc_b0, m._acc_b1, m._acc_b2, m._acc_b3)]
        m.add_keyframe(1, self.stack(), pose(0.5, 0.0))
        m.rasterize()
        self.assertTrue(any(float(b.sum()) > float(o.sum()) for b, o in
                            zip((m._acc_b0, m._acc_b1, m._acc_b2, m._acc_b3), base)))
        m.drop_points(1)
        m.rasterize()
        for b, o in zip((m._acc_b0, m._acc_b1, m._acc_b2, m._acc_b3), base):
            np.testing.assert_allclose(b, o, rtol=0, atol=1e-9)

    def test_bands_match_a_fresh_build_after_loop_closure_shift(self) -> None:
        # 回环整体平移 0.5 m（整格）：分带要像地面层一样"只挪下标"，增量结果与
        # 按终态位姿重建**逐格相同**。注意整格平移后栅格窗口也跟着挪（这是 _shift 的本意），
        # 所以不能断言数组整体右移——那才是错的。
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        poses = {0: pose(0, 0), 1: pose(1.0, 0.4), 2: pose(2.0, -0.3)}
        for i, T in poses.items():
            m.add_keyframe(i, self.stack(), T)
        m.rasterize()
        shifted = {0: pose(0.5, 0.0), 1: pose(1.5, 0.4), 2: pose(2.5, -0.3)}
        m.update_poses(shifted)
        inc = m.rasterize()
        inc_b = m.band_grid()
        fresh = KeyframeGridMapper(MapperConfig(res_m=0.10))
        for i, T in shifted.items():
            fresh.add_keyframe(i, self.stack(), T)
        ref = fresh.rasterize()
        ref_b = fresh.band_grid()
        self.assertEqual(inc.grid.shape, ref.grid.shape)
        np.testing.assert_array_equal(inc.grid, ref.grid)
        self.assertEqual(inc_b.shape, ref_b.shape)
        np.testing.assert_allclose(inc_b, ref_b, rtol=0, atol=1e-9)
        self.assertTrue((inc_b[1] > 0).any())         # 确实有票，不是两边都空蒙对了


    def test_surface_moments_recover_a_known_slab_height(self) -> None:
        # 一块已知高度的水平板：矩算出的 mean_h 必须落在板高附近（验符号与量纲，不是验精度）。
        # 场景**必须带地面**：`rasterize` 的图幅是按 `_acc_g`/`_acc_o` 定的，只有头顶点、
        # 没有地面点时图幅会缩到机位附近，头顶数据被裁掉（``band_grid``/``surface_grid`` 只覆盖
        # NavGrid 图幅）。真实场景总有地板，所以这不是缺陷，但单测里不能这么造场景。
        for target in (2.6, 3.2, 4.0):
            m = KeyframeGridMapper(MapperConfig(res_m=0.10))
            scene = np.vstack([world_scene(), ghost_block(2.0, 3.0, -0.6, 0.6, target, target + 0.05)])
            m.add_keyframe(0, scene.astype(np.float32), pose(0, 0))
            m.rasterize()
            sg = m.surface_grid()
            vals = [v for v in sg["mean_h"].ravel() if np.isfinite(v) and 0 < v < 12]
            self.assertTrue(vals, f"target={target}: 没抓到任何头顶点")
            self.assertAlmostEqual(float(np.median(vals)), target, delta=0.12)
            self.assertLess(float(np.nanmedian(sg["sd_h"])), 0.12)   # 单层板，格内应当很薄

    def test_surface_moments_empty_when_bands_off(self) -> None:
        m = self.build(hi_bands=False)
        m.rasterize()
        self.assertIsNone(m.surface_counts())
        self.assertIsNone(m.surface_grid())

    def test_surface_moments_reversible_on_drop(self) -> None:
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        m.add_keyframe(0, self.stack(), pose(0, 0))
        m.rasterize()
        base = [a.copy() for a in (m._acc_sb_n, m._acc_sb_h, m._acc_sb_h2)]
        m.add_keyframe(1, self.stack(), pose(0.5, 0.0))
        m.rasterize()
        self.assertGreater(float(m._acc_sb_n.sum()), float(base[0].sum()))
        m.drop_points(1)
        m.rasterize()
        for a, b in zip((m._acc_sb_n, m._acc_sb_h, m._acc_sb_h2), base):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_surface_moments_match_fresh_build_after_loop_shift(self) -> None:
        m = KeyframeGridMapper(MapperConfig(res_m=0.10))
        for i, T in enumerate((pose(0, 0), pose(1.0, 0.4), pose(2.0, -0.3))):
            m.add_keyframe(i, self.stack(), T)
        m.rasterize()
        shifted = (pose(0.5, 0.0), pose(1.5, 0.4), pose(2.5, -0.3))
        m.update_poses({i: T for i, T in enumerate(shifted)})
        inc = m.surface_grid()
        self.assertIsNotNone(inc)
        fresh = KeyframeGridMapper(MapperConfig(res_m=0.10))
        for i, T in enumerate(shifted):
            fresh.add_keyframe(i, self.stack(), T)
        fresh.rasterize()
        ref = fresh.surface_grid()
        for k in ("n", "sum_h", "sum_h2", "mean_h"):
            np.testing.assert_allclose(inc[k], ref[k], rtol=0, atol=1e-9)

    def test_surface_moments_independent_of_band_edges(self) -> None:
        # 矩存的是绝对离地高，不该随 hi_band_m 变——改带边界不用迁移历史。
        def run(**kw):
            m = KeyframeGridMapper(MapperConfig(res_m=0.10, **kw))
            m.add_keyframe(0, self.stack(), pose(0, 0))
            m.rasterize()
            return m.surface_grid()
        a, b = run(hi_band_m=3.0, hi_band_top_m=4.5), run(hi_band_m=2.5, hi_band_top_m=8.0)
        np.testing.assert_allclose(a["mean_h"], b["mean_h"], rtol=0, atol=1e-9)

    def test_ground_layer_unchanged_by_moments(self) -> None:
        # 加矩不能动地面层（与分带同一个不可让步的性质）。
        a = self.build().rasterize().grid
        m = self.build()
        m.rasterize()
        m._acc_sb_h[:] = 12345.0                    # 人为破坏矩
        m.rasterize()
        np.testing.assert_array_equal(a, m.rasterize().grid)


if __name__ == "__main__":
    unittest.main()
