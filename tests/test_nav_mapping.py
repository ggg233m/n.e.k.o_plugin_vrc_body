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
        def occ_cells(cap: float):
            m = KeyframeGridMapper(MapperConfig(res_m=0.10, ground_plane_max_deg=cap))
            m.add_keyframe(0, observe(self.scene, pose(1.0, 0, pitch=math.radians(14))), pose(1.0, 0))
            ng = m.rasterize()
            s, occ = ng.meta.world_scale, ng.grid == OCC

            def box(x0, x1, y0, y1):
                r0, c0 = ng.to_cell((x0 * s, y1 * s))
                r1, c1 = ng.to_cell((x1 * s, y0 * s))
                return int(occ[r0:r1 + 1, c0:c1 + 1].sum())
            return box(1.3, 3.3, -1.4, 1.4) + box(3.3, 4.0, 0.4, 1.4), box(3.3, 3.7, -1.4, 0.2)

        ground_off, wall_off = occ_cells(0.0)
        ground_on, wall_on = occ_cells(15.0)
        self.assertGreater(ground_off, 20)          # 证明这条用例确实需要斜面
        self.assertLessEqual(ground_on, 3)
        self.assertEqual(wall_on, wall_off)
        self.assertGreater(wall_on, 20)

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


if __name__ == "__main__":
    unittest.main()
