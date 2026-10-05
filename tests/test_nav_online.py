from __future__ import annotations

import json
import math
import tempfile
import threading
import time
import tomllib
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import nav_online
from neko_anyadance_body import config
from neko_anyadance_body.backend.nav_loop import LoopConfig
from neko_anyadance_body.backend.nav_mapping import MapperConfig
from neko_anyadance_body.backend.nav_online import (DeadReckoner, KeyframePolicy, OnlineNavConfig, OnlineNavigator,
                                                    SessionRecorder, check_baseline, hmd_to_base_rotation,
                                                    near_obstacle)

S = 0.755
CAM_H = 1.73


def hmd_yaw(deg: float) -> np.ndarray:
    """SteamVR 站立系（x 右 y 上 z 后）绕 +y 转 deg：从上往下看是逆时针（左转）。"""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def osc(ts: float, vx: float, vz: float) -> dict:
    return {"timestamp": ts, "velocity_x": vx, "velocity_z": vz}


class DeadReckonerTest(unittest.TestCase):
    def test_hmd_yaw_left_is_ccw_in_base(self) -> None:
        r = hmd_to_base_rotation(hmd_yaw(90.0))
        np.testing.assert_allclose(r[:2, 0], [0.0, 1.0], atol=1e-9)   # 前方转到 +y（左）

    def test_forward_walk_output_leads_by_velocity_times_lag(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.13)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        T = dr.update(1.0, r, [osc(0.0, 0.0, 1.0)])
        self.assertAlmostEqual(T[0, 3] * S, 1.13, places=6)   # 积分 1.0 + 1 m/s × 0.13 s
        self.assertAlmostEqual(T[1, 3], 0.0, places=9)
        self.assertAlmostEqual(dr.dist_m, 1.0, places=6)      # 路程不含超前量
        self.assertEqual(dr.samples, 1)                     # 重复样本不重复计

    def test_lag_compensation_survives_late_samples(self) -> None:
        # 旧实现把积分终点推到 now+lag，迟到样本又从 now+lag 起生效，两者抵消：换 lag 输出不变。
        r = hmd_to_base_rotation(np.eye(3))
        out = []
        for lag in (0.0, 0.2):
            dr = DeadReckoner(S, osc_lag_s=lag)
            dr.update(0.0, r, [osc(0.0, 0.0, 0.0)])
            dr.update(1.0, r, [osc(0.95, 0.0, 2.0)])       # 每个样本都晚 50 ms 才到
            out.append(dr.update(2.0, r, [osc(1.95, 0.0, 2.0)])[0, 3] * S)
        self.assertAlmostEqual(out[1] - out[0], 2.0 * 0.2, places=6)

    def test_stop_removes_lead(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.2)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        T = dr.update(1.0, r, [osc(1.0, 0.0, 0.0)])
        self.assertAlmostEqual(T[0, 3] * S, 1.0, places=6)

    def test_strafe_right_goes_to_negative_y(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 1.0, 0.0)])
        T = dr.update(0.5, r, [])
        self.assertAlmostEqual(T[1, 3] * S, -0.5, places=6)

    def test_heading_rotates_velocity(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(hmd_yaw(90.0))
        dr.update(0.0, r, [osc(0.0, 0.0, 2.0)])
        T = dr.update(1.0, r, [])
        np.testing.assert_allclose(T[:2, 3] * S, [0.0, 2.0], atol=1e-9)

    def test_velocity_change_mid_interval_and_late_sample_ignored(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        dr.update(1.0, r, [osc(0.5, 0.0, 0.0)])            # 0.5 s 时停下
        T = dr.update(2.0, r, [osc(0.2, 0.0, 5.0)])        # 迟到样本：不回改、不生效
        self.assertAlmostEqual(T[0, 3] * S, 0.5, places=6)
        self.assertEqual(dr.samples, 2)

    def test_non_finite_velocity_skipped(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(np.eye(3))
        T = dr.update(1.0, r, [osc(0.1, float("nan"), 1.0)])
        self.assertEqual(float(T[0, 3]), 0.0)
        self.assertEqual(dr.samples, 0)

    # ---- 静默封顶：人站着不动，路程不能一直涨（2026-10 现场）----
    def test_silence_stops_extrapolating_instead_of_drifting_forever(self) -> None:
        # 复现现场：最后一个速度包丢了 0 之后 OSC 彻底静默，旧实现会 0.4 m/s 永远积分下去。
        dr = DeadReckoner(S, osc_lag_s=0.0, zoh_max_s=2.5, zoh_fade_s=1.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 0.4)])
        dr.update(3600.0, r, [])                       # 站了一小时，一个新包都没有
        self.assertLessEqual(dr.dist_m, 0.4 * 3.5 + 1e-9)   # 满速 2.5 s + 淡出 1 s 到顶
        self.assertLess(dr.dist_m, 1.5)
        self.assertAlmostEqual(float(dr.xy[0] * S), dr.dist_m, places=9)   # 位姿不再偷偷跑
        # 丢掉的那段照实记账，不假装本来就该是 0
        self.assertAlmostEqual(dr.holdout_m + dr.dist_m, 0.4 * 3600.0, places=3)
        self.assertAlmostEqual(dr.holdout_s, 3600.0 - 3.0, places=6)

    def test_short_silence_keeps_zoh_protocol_semantics(self) -> None:
        # 真实的匀速长走每 ~0.09 s 一个包；短空档必须照旧满速外推（协议语义：静默 = 速度没变）。
        dr = DeadReckoner(S, osc_lag_s=0.0, zoh_max_s=2.5, zoh_fade_s=1.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        dr.update(0.9, r, [])
        self.assertAlmostEqual(dr.dist_m, 0.9, places=9)
        self.assertAlmostEqual(dr.holdout_m, 0.0, places=9)

    def test_jump_gap_up_to_observed_max_is_not_truncated(self) -> None:
        # 20260929 录制里最长的"真位移"空档是 2.03 s @ 4 m/s（滞空），默认上限必须放过它。
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 4.0)])
        dr.update(2.03, r, [])
        self.assertAlmostEqual(dr.dist_m, 4.0 * 2.03, places=6)
        self.assertAlmostEqual(dr.holdout_m, 0.0, places=9)

    def test_new_sample_after_silence_resumes_full_speed(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0, zoh_max_s=2.5, zoh_fade_s=1.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 0.4)])
        dr.update(600.0, r, [])                       # 静默很久：已经停下积分
        before = dr.dist_m
        dr.update(600.1, r, [osc(600.1, 0.0, 0.4)])   # 又走起来了，新速度从 600.1 起生效
        self.assertAlmostEqual(dr.dist_m, before, places=9)
        dr.update(600.2, r, [])                       # 之后照旧满速积分
        self.assertAlmostEqual(dr.dist_m - before, 0.4 * 0.1, places=6)
        self.assertAlmostEqual(dr.holdout_m, 0.4 * (600.2 - 3.0 - 0.1), places=6)

    def test_zero_speed_silence_is_never_counted_as_holdout(self) -> None:
        # 停住之后的静默是真的 0×Δt，不该被算成"被丢掉的位移"（否则现场噪音淹没真信号）。
        dr = DeadReckoner(S, osc_lag_s=0.0, zoh_max_s=2.5, zoh_fade_s=1.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 0.0)])
        dr.update(3600.0, r, [])
        self.assertEqual(dr.holdout_s, 0.0)
        self.assertEqual(dr.holdout_m, 0.0)
        self.assertEqual(dr.dist_m, 0.0)

    def test_legit_gap_is_measured_so_the_cap_can_be_checked_in_the_field(self) -> None:
        # "恒速不发包"这个前提要能被现场读数证伪：两端都非零的静默段最长多长，直接报出来。
        dr = DeadReckoner(S, osc_lag_s=0.0, zoh_max_s=2.5, zoh_fade_s=1.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        dr.update(0.016, r, [osc(0.016, 0.0, 1.0)])
        self.assertAlmostEqual(dr.max_legit_gap_s, 0.016, places=6)
        dr.update(2.0, r, [osc(2.0, 0.0, 0.0)])       # 停住：两端不都非零，不计入
        self.assertAlmostEqual(dr.max_legit_gap_s, 0.016, places=6)
        dr.update(5.0, r, [osc(5.0, 0.0, 1.0)])       # 3 s 的"两端非零"空档被记下
        self.assertAlmostEqual(dr.max_legit_gap_s, 3.0, places=6)


def ground_and_wall(wall_x_track: float | None) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(0.3, 3.0, 0.05), np.arange(-1.5, 1.5, 0.05))
    pts = [np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -CAM_H)])]
    if wall_x_track is not None:
        wy, wz = np.meshgrid(np.arange(-0.5, 0.5, 0.03), np.arange(-CAM_H + 0.5, -CAM_H + 1.2, 0.05))
        pts.append(np.column_stack([np.full(wy.size, wall_x_track), wy.ravel(), wz.ravel()]))
    return np.vstack(pts).astype(np.float32)


class KeyframePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p = KeyframePolicy(OnlineNavConfig())
        self.assertEqual(self.p.decide(0.0, np.zeros(2), 0.0, 0.0), "new")

    def test_still_age_frame_is_refresh(self) -> None:
        self.assertIsNone(self.p.decide(1.0, np.zeros(2), 0.0, 0.0))
        self.assertEqual(self.p.decide(3.0, np.array([0.05, 0.0]), math.radians(5), 0.0), "refresh")
        # 慢慢挪了 0.2 m（没到 0.4 m 触发距离）：到龄补的帧是新视角，追加不替换。
        self.assertEqual(self.p.decide(6.0, np.array([0.05 + 0.2 / S, 0.0]), math.radians(5), 0.0), "new")

    def test_distance_triggers_new(self) -> None:
        self.assertEqual(self.p.decide(0.5, np.array([0.45 / S, 0.0]), 0.0, 0.0), "new")

    def test_fast_turn_defers_then_falls_back(self) -> None:
        yaw = math.radians(40)
        self.assertIsNone(self.p.decide(0.5, np.zeros(2), yaw, 90.0))
        self.assertIsNone(self.p.decide(1.0, np.zeros(2), yaw, 90.0))
        self.assertEqual(self.p.deferred, 2)
        # 转了 1 s 还没停：兜底照取。
        self.assertEqual(self.p.decide(1.6, np.zeros(2), yaw, 90.0), "new")
        # 转慢下来立刻取，推迟计时从头算。
        self.assertIsNone(self.p.decide(1.7, np.zeros(2), yaw * 2, 90.0))
        self.assertEqual(self.p.decide(1.8, np.zeros(2), yaw * 2, 10.0), "new")

    def test_defer_timeout_never_yields_refresh(self) -> None:
        """原地快摆头耗光推迟预算后必须按 new 取帧，不能判成 refresh。

        refresh 会替换上一关键帧的点云（nav_online:925-927），并连带删掉那一帧的
        回环词袋（nav_xsession.on_keyframe）与磁盘上的 kf npz / thumb jpg
        （nav_memory._write_kf）——删掉的东西找不回来。而 new 只是多一份噪声。

        触发形态：净位移 < kf_still_m、净转角 < kf_still_deg，但角速度持续超过
        kf_defer_dps（原地摆头/抖动就是这种：净变化很小、瞬时角速度很大）。
        注意 defer 分支只有到龄（kf_max_age_s=3.0）之后才进得去，所以时间点
        必须跨过 3 s。
        """
        self.assertIsNone(self.p.decide(1.0, np.zeros(2), 0.0, 0.0))
        self.assertIsNone(self.p.decide(3.0, np.zeros(2), math.radians(2.0), 90.0))
        self.assertIsNone(self.p.decide(3.5, np.zeros(2), math.radians(-2.0), 90.0))
        self.assertEqual(self.p.deferred, 2)
        # 净位移 0、净转角 1° —— 按旧逻辑这里恰好满足 refresh 的三个条件。
        self.assertEqual(self.p.decide(4.2, np.zeros(2), math.radians(1.0), 90.0), "new")

    def test_fast_turn_while_walking_falls_back_on_distance(self) -> None:
        self.assertIsNone(self.p.decide(0.2, np.array([0.5 / S, 0.0]), 0.0, 60.0))
        self.assertEqual(self.p.decide(0.4, np.array([0.85 / S, 0.0]), 0.0, 60.0), "new")


class NearObstacleTest(unittest.TestCase):
    def kw(self) -> dict:
        return dict(cam_h=CAM_H, scale=S, ahead_m=0.7, half_width_m=0.25, ground_tol_m=0.3,
                    top_m=2.0, min_pts=40)

    def test_wall_ahead_triggers_ground_does_not(self) -> None:
        r = np.eye(3)
        self.assertFalse(near_obstacle(ground_and_wall(None), r, **self.kw())[0])
        hit, n = near_obstacle(ground_and_wall(0.5 / S), r, **self.kw())
        self.assertTrue(hit)
        self.assertGreaterEqual(n, 40)

    def test_detection_follows_head_heading(self) -> None:
        # 点在 base 系（跟头走），头左转 90° 后正前方仍是这些点；近的触发、远的不触发。
        r = hmd_to_base_rotation(hmd_yaw(90.0))
        self.assertTrue(near_obstacle(ground_and_wall(0.5 / S), r, **self.kw())[0])
        far = near_obstacle(ground_and_wall(1.5 / S), r, **self.kw())
        self.assertFalse(far[0])


class FakeSensors:
    fx, cx, cy, baseline_m = 202.5, 360.0, 202.5, 0.126

    def __init__(self) -> None:
        self.closed = False

    def open(self) -> dict:
        return {"size": [720, 405]}

    def hmd_rotation(self) -> np.ndarray:
        return np.eye(3)

    def read_stereo(self):
        z = np.zeros((4, 4), np.uint8)
        return z, z

    def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(self, armed: bool = True, vz: float = 0.0, record_root: Path | None = None,
                 frozen_osc: bool = False) -> None:
        self.armed = armed
        self.vz = vz
        self.frozen_osc = frozen_osc
        self.moves: list[tuple[float, int]] = []
        self.turns: list[float] = []
        self.stops = 0
        self.sensors = FakeSensors()
        self._osc_ts: float | None = None
        cfg = OnlineNavConfig(map_min_interval_s=0.1, stereo_period_s=0.05, kf_max_age_s=0.3)
        self.nav = OnlineNavigator(
            motion_history=self._motion,
            send_move=lambda f, ms: self.moves.append((f, ms)),
            send_turn=lambda d: self.turns.append(d),
            stop_motion=self._stop,
            drive_block_reason=lambda: None if self.armed else "autonomy_not_armed",
            sensors_factory=lambda: self.sensors, cfg=cfg, record_root=record_root)

    def _motion(self) -> list[dict]:
        # 时间戳必须和 navigator 同一个 monotonic 时钟且**新鲜**：真实回传是变化驱动的，
        # 拿一个固定的 0.0 当时间戳等于宣称"最后一条速度报文是开机以来的"，航位推算
        # 会按静默封顶把它丢掉（见 DeadReckoner._hold_dt），测的就不是同一件事了。
        if self.frozen_osc:                      # 录制去重那条用：每拍都是同一个样本
            self._osc_ts = time.monotonic() if self._osc_ts is None else self._osc_ts
        else:
            now = time.monotonic()
            self._osc_ts = now if self._osc_ts is None else max(self._osc_ts + 1e-6, now)
        return [osc(self._osc_ts, 0.0, self.vz)]

    def _stop(self) -> None:
        self.stops += 1


class ControlTickTest(unittest.TestCase):
    def prime(self, h: Harness, stereo_age: float = 0.0) -> None:
        now = time.monotonic()
        h.nav._pose, h.nav._pose_at = np.eye(4), now
        h.nav._stereo_at = now - stereo_age

    def test_idle_sends_nothing(self) -> None:
        h = Harness()
        self.prime(h)
        h.nav._control_tick(time.monotonic())
        self.assertEqual((h.moves, h.turns, h.stops), ([], [], 0))

    def test_not_armed_blocks_and_stops_once(self) -> None:
        h = Harness(armed=False)
        self.prime(h)
        h.nav.session.explore()
        h.nav._moving = True
        h.nav._control_tick(time.monotonic())
        h.nav._control_tick(time.monotonic())
        self.assertEqual(h.nav._drive_block, "autonomy_not_armed")
        self.assertEqual(h.stops, 1)
        self.assertEqual(h.moves, [])

    def test_stale_stereo_blocks(self) -> None:
        h = Harness()
        self.prime(h, stereo_age=5.0)
        h.nav.session.explore()
        h.nav._control_tick(time.monotonic())
        self.assertEqual(h.nav._drive_block, "stereo_stale")
        self.assertEqual(h.moves, [])

    def test_stale_pose_blocks(self) -> None:
        h = Harness()
        self.prime(h)
        h.nav._pose_at -= 5.0
        h.nav.session.explore()
        h.nav._control_tick(time.monotonic())
        self.assertEqual(h.nav._drive_block, "pose_stale")

    def test_follower_output_mapped_to_commands(self) -> None:
        h = Harness()
        self.prime(h)
        h.nav.session.explore()
        with mock.patch.object(h.nav.session, "step",
                               return_value={"state": "following", "forward": 0.6, "turn_rate": math.radians(30)}):
            h.nav._control_tick(time.monotonic())
        self.assertEqual(h.moves, [(0.6, 400)])
        self.assertAlmostEqual(h.turns[0], 3.0)          # 30°/s ÷ 10 Hz，正数 = 左转

    def test_goto_requires_running(self) -> None:
        self.assertEqual(Harness().nav.goto(1.0, 0.0)["reason"], "navmesh_not_running")
        self.assertFalse(Harness().nav.goto(float("nan"), 0.0)["ok"])


class ThreadsTest(unittest.TestCase):
    def test_slow_map_rebuild_does_not_block_status_or_control(self) -> None:
        # 实机：853 帧回环后一次重建 >1 s，旧实现握着 _nav_lock，status 请求 30 s 超时、控制线程停拍。
        h = Harness(armed=False, vz=0.5)
        real = nav_online.NavSession.compute
        gate = threading.Event()

        def slow(self, est, snap):
            gate.set()
            time.sleep(1.5)
            return real(self, est, snap)

        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)),                 mock.patch.object(nav_online.NavSession, "compute", slow):
            h.nav.start()
            try:
                self.assertTrue(gate.wait(5.0))
                t0 = time.perf_counter()
                h.nav.status()
                h.nav._control_tick(time.monotonic())
                self.assertTrue(h.nav.cancel()["ok"])
                self.assertLess(time.perf_counter() - t0, 0.3)
            finally:
                h.nav.stop()

    def test_end_to_end_keyframes_map_and_shutdown(self) -> None:
        h = Harness(armed=False, vz=0.5)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and h.nav.status()["map_updates"] < 2:
                    time.sleep(0.05)
                self.assertTrue(h.nav.explore()["ok"])
                time.sleep(0.3)
                st = h.nav.status()
            finally:
                h.nav.stop()
        self.assertGreaterEqual(st["keyframes"], 2)
        self.assertGreaterEqual(st["map_updates"], 2)
        self.assertEqual(st["pose_source"], "dead_reckoning+loop_closure")
        self.assertEqual(st["loop_closure"]["keyframes"], st["keyframes"])
        self.assertEqual(st["pose_state"], "localized")
        self.assertGreater(st["odometry_distance_m"], 0.0)
        # 静默封顶的现场读数：有新报文在流 → 静默时长接近 0、丢弃量必须是 0。
        ho = st["odometry_holdout"]
        self.assertEqual((ho["zoh_max_s"], ho["zoh_fade_s"]), (2.5, 1.0))
        self.assertLess(ho["last_sample_age_s"], 0.5)
        self.assertEqual((ho["dropped_s"], ho["dropped_m"]), (0.0, 0.0))
        self.assertEqual(st["drive_block"], "autonomy_not_armed")
        self.assertEqual(h.moves, [])
        self.assertEqual(st["errors"], [])
        self.assertIsNotNone(h.nav.grid_png())
        view, ng = h.nav.grid_view(), h.nav.session.ng
        self.assertEqual((view["rows"], view["cols"]), ng.grid.shape)
        # UI 用这套元数据把点击像素换成导航系世界米，必须与 NavGrid.to_world 一致。
        r, c = view["rows"] // 3, view["cols"] // 2
        o, res, s = view["origin_xy_track_m"], view["resolution_track_m"], view["world_scale"]
        ui_xy = ((o[0] + (c + 0.5) * res) * s, (o[1] + (view["rows"] - 1 - r + 0.5) * res) * s)
        self.assertEqual(tuple(round(v, 9) for v in ui_xy), tuple(round(v, 9) for v in ng.to_world((r, c))))
        self.assertEqual(ng.to_cell(ui_xy), (r, c))
    def test_grid_view_layers(self) -> None:
        """栅格底图图层。

        最要紧的一条是 **tristate 必须与加图层之前逐位一致** —— 这套东西是纯显示，
        改它绝不能动到既有那张图。golden 门管的是建图，本条管的是渲染。
        """
        import base64
        import cv2
        h = Harness(armed=False)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)), \
             mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(0.4)
            finally:
                h.nav.stop()

        def decode(view):
            return cv2.imdecode(np.frombuffer(base64.b64decode(view["png_base64"]), np.uint8),
                                cv2.IMREAD_COLOR)

        ng = h.nav.session.ng
        base = h.nav.grid_view()
        self.assertEqual(base["layer"], "tristate")
        # 与显式指定同名层必须一致
        self.assertTrue(np.array_equal(decode(base), decode(h.nav.grid_view("tristate"))))
        # 与手写的旧算法逐位一致 —— 只允许**覆盖层**那几个像素不同（位姿红点/目标绿点/
        # 路径蓝线是加图层之前就有的，画在底图之上）。逐格断言"只有这些像素不同"，
        # 而不是"完全相同"，否则这条测试在有位姿时必然失败。
        g = ng.grid
        expect = np.full(g.shape + (3,), 90, np.uint8)
        expect[g == nav_online.FREE] = (170, 170, 170)
        expect[g == nav_online.OCC] = (0, 0, 0)
        expect[ng.center] = (255, 255, 255)
        got = decode(base)
        diff = (got != expect).any(axis=2)
        overlay_bgr = {(0, 0, 255), (0, 200, 0), (255, 120, 0)}   # 位姿 / 目标 / 路径
        bad = [tuple(int(v) for v in got[r, c])
               for r, c in zip(*np.nonzero(diff)) if tuple(int(v) for v in got[r, c]) not in overlay_bgr]
        self.assertEqual(bad, [], f"tristate 底图被改了，共 {int(diff.sum())} 格不是覆盖层色：{bad[:6]}")

        for layer in ("surface", "bands", "clearance"):
            v = h.nav.grid_view(layer)
            self.assertIsNotNone(v, layer)
            self.assertEqual(v["layer"], layer)
            img = decode(v)
            self.assertEqual(img.shape, g.shape + (3,), layer)
            self.assertIn("ramp", v)
            # 不同图层必须画出不同的东西，否则说明图层接错了累加器
            self.assertFalse(np.array_equal(img, decode(base)), f"{layer} 与三态图相同")
            self.assertGreater(len(np.unique(img.reshape(-1, 3), axis=0)), 1, layer)

        with self.assertRaisesRegex(ValueError, "layer"):
            h.nav.grid_view("nope")

    def test_grid_view_layers_survive_nan(self) -> None:
        """NaN 必须在转 uint8 **之前**清掉：``(NaN*255).astype(uint8)`` 是未定义行为，
        会打出 ``invalid value encountered in cast`` 并写出垃圾像素。"""
        import warnings
        f = np.array([[0.0, np.nan, 1.0], [np.inf, 0.5, np.nan]], np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            img, ramp = nav_online._ramp_layer(
                f, 1.0, lambda v: np.isfinite(v), 0.0, 1.0, "m")
        self.assertEqual(img.shape, (2, 3, 3))
        # 有限值是 0.0 / 1.0 / 0.5 三格（nan 与 inf 都不算）
        self.assertEqual(ramp["valid_cells"], 3)
        self.assertEqual(tuple(img[0, 1]), (70, 70, 70))  # NaN → 灰，不是垃圾色
        self.assertEqual(tuple(img[1, 0]), (70, 70, 70))  # Inf → 灰

    def test_grid_view_layers_without_hmd(self) -> None:
        """未知图层必须**抛错**而不是静默回落到三态 —— 否则调用方以为自己看的是
        新图层，实际拿到旧图。回落白名单是 HTTP 层（process.py）的事，两者职责不同。"""
        h = Harness(armed=False)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)), \
             mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(0.3)
            finally:
                h.nav.stop()
        if h.nav.session.ng is None:
            self.skipTest("没有栅格，跳过")
        with self.assertRaises(ValueError):
            h.nav.grid_view("__definitely_not_a_layer__")

    def test_grid_view_layers_exposed_to_service(self) -> None:
        from neko_anyadance_body.backend.nav_online import _GRID_LAYERS
        self.assertIn("tristate", _GRID_LAYERS)
        self.assertIn("surface", _GRID_LAYERS)
        # 图层名是白名单，HTTP 层按它回落；写成集合会丢顺序，这里只查内容
        self.assertEqual(set(_GRID_LAYERS) & {"tristate", "surface", "bands", "clearance"},
                         set(_GRID_LAYERS))

    def test_session_is_persisted_to_world_memory(self) -> None:
        from neko_anyadance_body.backend.nav_memory import MemoryConfig, NavMemoryStore, world_dir_name
        with tempfile.TemporaryDirectory() as tmp:
            store = NavMemoryStore(Path(tmp), MemoryConfig(min_session_keyframes=1))
            h = Harness(armed=False, vz=0.5)
            h.nav.memory = store
            h.nav._world_identity = lambda: {"world_key": "wrld_test", "world_name": "t", "world_source": "manual"}
            with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                     mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
                h.nav.start()
                try:
                    self.assertEqual(h.nav.status()["memory"]["status"], "recording")
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline and h.nav.status()["keyframes"] < 3:
                        time.sleep(0.05)
                    # ⚠️ 这里必须**当场**断言，不能等到下面 assertGreaterEqual。
                    # 原来的写法是：等 5s 拿不到 3 帧就静默退出循环，然后错误延后
                    # 两三个断言才爆出来，消息还与真因无关（曾经只剩 "+ complete"，
                    # 被 _run_tests.py 的 [:160] 截断 + GBK 控制台双重毁掉，
                    # 真实断言完全看不见）。一条 flaky 测试每复发一次就要重新
                    # 取证一次，这行断言就是把那笔账一次性付掉。
                    #
                    # 余量参考：隔离跑整条用例约 0.8s，预算 5.0s（6 倍）。
                    # **所以这条一旦触发就不是"机器慢"**，是建图线程真的卡住了。
                    n_kf = h.nav.status()["keyframes"]
                    self.assertGreaterEqual(
                        n_kf, 3,
                        f"等 5.0s 只产出 {n_kf} 个关键帧（隔离跑约 0.8s / 预算 5.0s）。"
                        "若确认非机器慢，用 research/tools/_flaky_nav_online.py 取证。")
                finally:
                    h.nav.stop()
            wid = world_dir_name("wrld_test")
            sessions = store.list_sessions(wid)
            self.assertEqual(len(sessions), 1)
            info = store.session(wid, sessions[0]["session_id"])
            self.assertEqual(info["status"], "complete")
            self.assertGreaterEqual(info["keyframes"], 3)
            self.assertEqual(info["baseline_m"], 0.126)
            self.assertEqual(info["world_scale"], S)
            poses = store.load_poses(wid, sessions[0]["session_id"])
            self.assertEqual(len(poses["ids"]), info["keyframes"])
            self.assertIsNone(store.active_status())

    def test_stop_hands_sid_to_align_thread_body(self) -> None:
        """stop() 顺序回归：``_end_memory`` 先把 ``_mem`` 清掉，采纳线程体必须收到 stop 捕获的 sid。

        2026-10-05 live 实测指纹：``xsession_align == {"ok": false, "reason": "memory_gone"}``
        ——采纳**从未真正跑过**（旧测试直接调线程体，绕开了真实 stop 顺序）。
        这里用 spy 替线程体、断言实参（无时序依赖）：sid 必须非空、world_dir 必须是 tracker 的目录。
        """
        import types as _types
        seen: dict = {}
        wd = Path("wrld_probe_dir")                  # 只作实参比对，不碰盘
        h = Harness(armed=False)
        # 模拟"内存会话还活着"：不接 memory store（少一层集成面，测的就是 stop 的顺序约定：
        # stop 必须在 _end_memory 清掉 _mem 之前把 sid 抓下来）。
        h.nav._mem = _types.SimpleNamespace(session_id="SID_PROBE")
        h.nav._xs = _types.SimpleNamespace(
            world_dir=wd, write_back=lambda *a, **k: {"written": True},
            status=lambda: {"active": True, "reason": "ok"})
        with mock.patch.object(h.nav, "_align_xsession",
                               side_effect=lambda sid, world_dir: seen.update(sid=sid, wd=world_dir)):
            h.nav.stop()
        if h.nav._align_thread is not None:
            h.nav._align_thread.join(timeout=5.0)
        self.assertEqual(seen.get("sid"), "SID_PROBE", f"线程体没拿到 sid（memory_gone 的根因）：{seen}")
        self.assertEqual(seen.get("wd"), wd)

    def test_unknown_world_runs_without_memory(self) -> None:
        from neko_anyadance_body.backend.nav_memory import NavMemoryStore
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(armed=False)
            h.nav.memory = NavMemoryStore(Path(tmp))
            h.nav._world_identity = lambda: {"world_key": None}
            h.nav.start()
            try:
                self.assertEqual(h.nav.status()["memory"], {"active": False, "reason": "world_unknown"})
            finally:
                h.nav.stop()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_external_hmd_yaw_snap_reported(self) -> None:
        h = Harness(armed=False)
        yaw = {"deg": 0.0}
        h.sensors.hmd_rotation = lambda: hmd_yaw(yaw["deg"])
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(0.3)
                self.assertEqual(h.nav.status()["hmd_yaw_jumps"], [])
                yaw["deg"] = 90.0
                time.sleep(0.3)
                jumps = h.nav.status()["hmd_yaw_jumps"]
            finally:
                h.nav.stop()
        self.assertEqual(len(jumps), 1)
        self.assertAlmostEqual(jumps[0]["delta_deg"], 90.0, delta=0.5)

    def test_sensor_open_failure_reported(self) -> None:
        h = Harness()

        class Broken(FakeSensors):
            def open(self) -> dict:
                raise RuntimeError("SteamVR not running")

        h.nav._sensors_factory = Broken
        h.nav.start()
        time.sleep(0.2)
        st = h.nav.status()
        h.nav.stop()
        self.assertTrue(any("SteamVR not running" in e for e in st["errors"]))
        self.assertEqual(st["pose_state"], "unknown")

    def test_baseline_mismatch_reported_but_keeps_running(self) -> None:
        h = Harness(armed=False)
        h.sensors.baseline_m = 0.063          # 发行版驱动
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(0.2)
                st = h.nav.status()
            finally:
                h.nav.stop()
        self.assertIs(st["baseline_check"]["ok"], False)
        self.assertTrue(any(e.startswith("baseline_mismatch") for e in st["errors"]))
        self.assertEqual(st["pose_state"], "localized")

    def test_baseline_ok_has_no_error(self) -> None:
        h = Harness(armed=False)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(0.2)
                st = h.nav.status()
            finally:
                h.nav.stop()
        self.assertIs(st["baseline_check"]["ok"], True)
        self.assertFalse(any(e.startswith("baseline_mismatch") for e in st["errors"]))


class ConfigSurfaceTest(unittest.TestCase):
    def test_world_scale_single_source(self) -> None:
        c = OnlineNavConfig(world_scale=0.9)
        self.assertEqual((c.mapper.world_scale, c.loop.world_scale), (0.9, 0.9))
        c = OnlineNavConfig(world_scale=0.8, mapper=MapperConfig(world_scale=0.5))
        self.assertEqual(c.mapper.world_scale, 0.8)
        self.assertEqual(KeyframePolicy(c).scale, 0.8)

    def test_check_baseline(self) -> None:
        c = OnlineNavConfig()
        self.assertTrue(check_baseline(0.1262, c)["ok"])
        self.assertFalse(check_baseline(0.063, c)["ok"])
        self.assertFalse(check_baseline(float("nan"), c)["ok"])
        self.assertIsNone(check_baseline(0.063, OnlineNavConfig(expected_baseline_m=0.0))["ok"])

    def test_config_keys_match_backend_fields(self) -> None:
        for spec, cls in ((config.NAVMESH_ONLINE_KEYS, OnlineNavConfig), (config.NAVMESH_MAPPER_KEYS, MapperConfig),
                          (config.NAVMESH_LOOP_KEYS, LoopConfig)):
            names = {f.name: f for f in fields(cls)}
            for key, (kind, _lo, _hi) in spec.items():
                self.assertIn(key, names, f"{cls.__name__}.{key}")
                self.assertIsInstance(getattr(cls(), key), kind, f"{cls.__name__}.{key}")
            self.assertNotIn("world_scale", spec)

    def test_defaults_equal_backend_defaults(self) -> None:
        c = OnlineNavConfig.from_plugin(config.PluginConfig.from_mapping({}).navmesh)
        self.assertEqual(c, OnlineNavConfig())

    def test_plugin_toml_round_trip(self) -> None:
        with (Path(__file__).resolve().parents[1] / "plugin.toml").open("rb") as fh:
            nav = config.PluginConfig.from_mapping(tomllib.load(fh)).navmesh
        c = OnlineNavConfig.from_plugin(nav)
        self.assertEqual((c.world_scale, c.expected_baseline_m, c.mapper.range_m), (0.755, 0.126, 5.0))

    def test_overrides_and_validation(self) -> None:
        nav = config.PluginConfig.from_mapping({"navmesh": {
            "world_scale": 0.8, "kf_dist_m": 0.5, "stop_min_pts": 20, "loop_closure": False,
            "mapper": {"range_m": 4.0, "ray_clear": False}, "loop": {"min_inliers": 40}}}).navmesh
        c = OnlineNavConfig.from_plugin(nav)
        self.assertEqual((c.world_scale, c.kf_dist_m, c.stop_min_pts, c.loop_closure), (0.8, 0.5, 20, False))
        self.assertEqual((c.mapper.range_m, c.mapper.ray_clear, c.mapper.world_scale), (4.0, False, 0.8))
        self.assertEqual((c.loop.min_inliers, c.loop.world_scale), (40, 0.8))
        for bad in ({"world_scale": 0}, {"kf_dst_m": 0.5}, {"mapper": {"world_scale": 0.9}},
                    {"mapper": {"range_m": "far"}}, {"loop": {"min_inliers": 1.5}}, {"mapper": 3}):
            with self.assertRaises(ValueError, msg=str(bad)):
                config.PluginConfig.from_mapping({"navmesh": bad})


class RecorderTest(unittest.TestCase):
    def run_session(self, root: Path | None, record: bool, frozen_osc: bool = False) -> dict:
        h = Harness(armed=False, vz=0.5, record_root=root, frozen_osc=frozen_osc)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start(record=record)
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and h.nav.status()["keyframes"] < 3:
                    time.sleep(0.05)
            finally:
                st = h.nav.stop()
        return st

    def test_default_off_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            st = self.run_session(Path(d), record=False)
            self.assertIsNone(st["recording"])
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_recording_has_everything_needed_to_replay(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            st = self.run_session(Path(d), record=True, frozen_osc=True)
            rec = st["recording"]
            self.assertEqual(rec["stopped"], "closed")
            out = Path(rec["dir"])
            self.assertEqual(out.parent, Path(d))
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("mapper", meta["config"])
            ev = [json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            kfs = [e for e in ev if e["kind"] == "kf"]
            self.assertEqual(ev[0]["kind"], "sensors")
            self.assertAlmostEqual(ev[0]["baseline_m"], FakeSensors.baseline_m)
            # 录到的关键帧数与状态一致、id 连续，每个都有点云文件。
            self.assertEqual([e["k"] for e in kfs], list(range(st["keyframes"])))
            self.assertTrue(all(isinstance(e["refresh"], bool) for e in kfs))
            self.assertGreaterEqual(len(kfs), 3)
            for e in kfs:
                with np.load(out / "kf" / f"{e['k']:06d}.npz") as z:
                    self.assertEqual(z["pts"].shape, ground_and_wall(None).shape)
                    self.assertEqual(z["T_dr"].shape, (4, 4))
            hmd = (out / "hmd.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(hmd), len(kfs))
            self.assertEqual(np.array(json.loads(hmd[0])["R"]).shape, (3, 3))
            # 假 OSC 历史每拍都一样：同一时间戳只记一次。
            osc_lines = (out / "osc.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(osc_lines), 1)
            with np.load(out / "final_poses.npz") as final:
                self.assertEqual(list(final["ids"]), [e["k"] for e in kfs])

    def test_size_limit_stops_recording_not_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r = SessionRecorder(Path(d) / "s", max_bytes=2000, meta={"x": 1})
            for i in range(3):
                r.keyframe(i, np.zeros((500, 3), np.float32), np.eye(4), np.eye(4), 0.0, None)
            self.assertEqual(r.status()["stopped"], "size_limit")
            self.assertEqual(r.status()["keyframes"], 1)
            r.line("hmd", {"t": 0.0})
            r.close(None)
            self.assertFalse((Path(d) / "s" / "hmd.jsonl").exists())
            self.assertEqual(r.status()["stopped"], "size_limit")

    def test_record_without_root_reports_error(self) -> None:
        st = self.run_session(None, record=True)
        self.assertIsNone(st["recording"])
        self.assertTrue(any("record_root_not_configured" in e for e in st["errors"]))


class CoverageViewTest(unittest.TestCase):
    """覆盖可视化快照：available 语义、指标口径、meta 同构、status()/grid_view() 契约不动。"""

    def run_nav(self, h, until, timeout=5.0):
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),             mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and not until(h.nav):
                    time.sleep(0.05)
                return until(h.nav)
            finally:
                h.nav.stop()

    def test_coverage_view_none_before_map(self) -> None:
        h = Harness()
        self.assertIsNone(h.nav.coverage_view())
        self.assertIsNone(h.nav.grid_view())          # 既有行为不受影响

    def test_coverage_metrics(self) -> None:
        h = Harness(armed=False, vz=0.5)
        self.assertTrue(self.run_nav(h, lambda n: n.status()["map_updates"] >= 2))
        view = h.nav.coverage_view()
        self.assertTrue(view["available"])
        m = view["metrics"]
        self.assertGreater(m["hull_cells"], 0)        # 走过 → hull 非空
        self.assertGreater(m["observed_cells"], 0)    # 走过的走廊有观测
        self.assertTrue(0.0 <= m["coverage_ratio"] <= 1.0)
        self.assertGreaterEqual(m["hull_m2"], m["observed_m2"])
        self.assertEqual(m["hole_cells"], m["hull_cells"] - m["observed_cells"])
        g = view["grid"]                              # 与 grid_view 同构的 meta（前端坐标换算复用）
        for k in ("rows", "cols", "origin_xy_track_m", "resolution_track_m", "world_scale", "frame"):
            self.assertIn(k, g)
        self.assertEqual(g["resolution_track_m"], 0.30)
        self.assertTrue(g["png_base64"])
        self.assertTrue(view["trend"])
        self.assertAlmostEqual(view["trend"][-1]["coverage_ratio"], m["coverage_ratio"], places=3)

    def test_coverage_status_and_grid_untouched(self) -> None:
        h = Harness(armed=False, vz=0.5)
        self.assertTrue(self.run_nav(h, lambda n: n.status()["map_updates"] >= 1))
        keys = set(h.nav.status())
        self.assertFalse([k for k in keys if "coverage" in k or k.startswith("cov")])
        view = h.nav.grid_view()                      # PNG 输出契约不变：rows/cols 与主栅格一致
        self.assertEqual((view["rows"], view["cols"]), h.nav.session.ng.grid.shape)


class XSessionAlignTest(unittest.TestCase):
    """P0.3b：会话末自动采纳线程体（同步调用验证接线，不 spawn 真线程）。"""

    def test_align_thread_body_writes_result(self) -> None:
        import types
        from tests.test_nav_xsession import build_align_fixture
        from neko_anyadance_body.backend.nav_xsession import XSessionConfig

        with tempfile.TemporaryDirectory() as tmp:
            wdir = Path(tmp) / "wrld_x-abc"
            build_align_fixture(wdir)
            nav = OnlineNavigator.__new__(OnlineNavigator)
            nav._state_lock = threading.Lock()
            nav._errors = []
            nav._align_out = None
            nav.cfg = OnlineNavConfig(xsession=XSessionConfig(align_min_constraints=8))
            nav._align_xsession("NEW", wdir)          # 直接调线程体（sid/world_dir 由 stop 捕获后传入）
            out = nav._align_out
            self.assertIsNotNone(out)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["old_sid"], "OLD")
            self.assertTrue(Path(out["merged"]).is_file())


if __name__ == "__main__":
    unittest.main()
