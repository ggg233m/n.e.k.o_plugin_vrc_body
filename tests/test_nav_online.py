from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import nav_online
from neko_anyadance_body import config
from neko_anyadance_body.backend.nav_loop import LoopConfig
from neko_anyadance_body.backend.nav_mapping import MapperConfig
from neko_anyadance_body.backend.nav_prior import PriorConfig
from neko_anyadance_body.backend.nav_online import (DeadReckoner, HeadingTrack, KeyframePolicy,
                                                    OnlineNavConfig, OnlineNavigator, RateMeter,
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


class RateMeterTest(unittest.TestCase):
    """实测频率必须能看出「配置 10 Hz、跑出来 7.94 Hz」这种缺口。

    2026-10-18：`stereo_period_s = 0.1` 的门写在 `tick >= next_stereo` 上，而 tick 是位姿
    循环的实际节拍。位姿跑 15.87 Hz（tick 0.063 s）时，0.1 s 只能等到第 2 个 tick ⇒
    实际步长 0.126 s = 7.94 Hz，比配置低 21%。此前没有任何字段报过这件事。
    """

    def test_hz_and_gap_from_regular_marks(self) -> None:
        m = RateMeter(16)
        for i in range(16):
            m.mark(i * 0.063)                 # 15.87 Hz 的位姿 tick
        self.assertAlmostEqual(m.hz(), 1.0 / 0.063, places=4)
        self.assertAlmostEqual(m.gap_p50_s(), 0.063, places=6)
        self.assertEqual(m.samples(), 16)

    def test_empty_and_single_mark_are_none_not_zero(self) -> None:
        """没测到要给 None —— 报 0 会被读成「卡死」，那是另一回事。"""
        m = RateMeter(8)
        self.assertIsNone(m.hz())
        self.assertIsNone(m.gap_p50_s())
        m.mark(1.0)
        self.assertIsNone(m.hz())
        self.assertIsNone(m.gap_p50_s())        # <3 个点算不出中位间隔

    def test_window_forgets_old_marks(self) -> None:
        """只报最近 N 次：换速度档后读数要跟着变，不能被全局平均糊住。"""
        m = RateMeter(4)
        for i in range(10):
            m.mark(i * 1.0)                     # 1 Hz
        self.assertAlmostEqual(m.hz(), 1.0, places=6)
        m.mark(10.0)
        m.mark(10.5)
        m.mark(11.0)
        m.mark(11.5)                            # 后四个是 2 Hz
        self.assertAlmostEqual(m.hz(), 2.0, places=6)

    def test_quantization_step_is_visible_as_integer_ticks(self) -> None:
        """核心判据：`stereo_gap / pose_gap` ≈ 整数 ⇒ 撞在量化台阶上，不是算不过来。"""
        tick = 0.063
        pose, stereo = RateMeter(64), RateMeter(64)
        for i in range(64):
            pose.mark(i * tick)
        for j in range(32):
            stereo.mark(j * 2 * tick)           # 每 2 个 tick 才等到门
            # 注意：stereo 的 mark 时刻必须落在 pose 的 tick 网格上（它们同源）
        ticks = stereo.gap_p50_s() / pose.gap_p50_s()
        self.assertAlmostEqual(ticks, 2.0, places=6)
        self.assertLess(stereo.hz(), 1.0 / 0.1)  # 7.94 < 10（配置的目标）

    def test_non_integer_ticks_means_really_too_slow(self) -> None:
        """反例：算不过来时步长不是整数个 tick —— 判据要能区分这两种病。"""
        tick = 0.063
        pose, stereo = RateMeter(64), RateMeter(64)
        for i in range(64):
            pose.mark(i * tick)
        for j in range(20):
            stereo.mark(j * 0.235)              # 1440 那种 235 ms，非 tick 整数倍
        ticks = stereo.gap_p50_s() / pose.gap_p50_s()
        self.assertGreater(abs(ticks - round(ticks)), 0.2)


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

    def test_speed_scale_scales_position_lead_and_distance(self) -> None:
        r = hmd_to_base_rotation(np.eye(3))
        dr = DeadReckoner(S, osc_lag_s=0.1, speed_scale=0.8)
        dr.update(0.0, r, [osc(0.0, 0.0, 2.0)])
        T = dr.update(1.0, r, [])
        self.assertAlmostEqual(T[0, 3] * S, 0.8 * 2.0 * 1.1, places=6)   # 积分和超前都按 0.8 缩
        self.assertAlmostEqual(dr.dist_m, 1.6, places=6)

    def test_speed_scale_default_is_1(self) -> None:
        self.assertEqual(OnlineNavConfig().osc_speed_scale, 1.0)

    def test_speed_scale_1_is_identity(self) -> None:
        r = hmd_to_base_rotation(np.eye(3))
        a, b = DeadReckoner(S, osc_lag_s=0.1), DeadReckoner(S, osc_lag_s=0.1, speed_scale=1.0)
        for dr in (a, b):
            dr.update(0.0, r, [osc(0.0, 0.5, 1.0)])
        np.testing.assert_array_equal(a.update(1.0, r, []), b.update(1.0, r, []))

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

    # ---- 朝向取样：局部系速度必须乘「被测量那一刻」的朝向（见 _basis_at）----

    def test_midpoint_heading_uses_the_segment_heading_not_the_current_one(self) -> None:
        # yaw 从 0 线性转到 90°，速度恒为前进 1 m/s，积分区间 [0,1]：
        #   旧行为：整段共用 t=1 的朝向 ⇒ 位移指向 90°
        #   中点：  用 t=0.5 的朝向 ⇒ 位移指向 45°
        def psi(t: float) -> float:
            return math.radians(90.0 * t)

        r0, r1 = hmd_to_base_rotation(hmd_yaw(0.0)), hmd_to_base_rotation(hmd_yaw(90.0))
        old = DeadReckoner(S, osc_lag_s=0.0)
        old.update(0.0, r0, [osc(0.0, 0.0, 1.0)])
        T_old = old.update(1.0, r1, [])
        new = DeadReckoner(S, osc_lag_s=0.0, heading_at=psi)
        new.update(0.0, r0, [osc(0.0, 0.0, 1.0)])
        T_new = new.update(1.0, r1, [])
        self.assertAlmostEqual(math.degrees(math.atan2(T_old[1, 3], T_old[0, 3])), 90.0, places=4)
        self.assertAlmostEqual(math.degrees(math.atan2(T_new[1, 3], T_new[0, 3])), 45.0, places=4)
        self.assertAlmostEqual(float(np.hypot(*T_new[:2, 3])) * S, 1.0, places=6)   # 中点保长度
        self.assertEqual(new.heading_segments, 1)
        self.assertEqual(new.heading_fallbacks, 0)

    def test_without_heading_provider_behaviour_is_bit_identical(self) -> None:
        # 关掉取样器（dr_heading_midpoint=False / 旧调用方）必须逐位等于旧实现：
        # 不能因为新增通路悄悄改了没打开它的路径。
        r = hmd_to_base_rotation(hmd_yaw(37.0))
        a = DeadReckoner(S, osc_lag_s=0.13)
        b = DeadReckoner(S, osc_lag_s=0.13, heading_at=None)
        for dr in (a, b):
            dr.update(0.0, r, [osc(0.0, 0.4, 1.2)])
            dr.update(0.5, r, [osc(0.4, 0.0, 2.0)])
        np.testing.assert_array_equal(a.update(1.0, r, []), b.update(1.0, r, []))
        self.assertEqual((b.heading_segments, b.heading_fallbacks), (0, 0))

    def test_heading_provider_returning_none_falls_back_and_is_counted(self) -> None:
        # 位姿历史还没喂上（启动头几帧）时按当前朝向算，并如实计数 —— 接线掉了
        # 表现为 heading_segments 恒 0，而不是静默换成另一套行为。
        r = hmd_to_base_rotation(hmd_yaw(20.0))
        plain = DeadReckoner(S, osc_lag_s=0.0)
        fallback = DeadReckoner(S, osc_lag_s=0.0, heading_at=lambda t: None)
        for dr in (plain, fallback):
            dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        np.testing.assert_array_equal(plain.update(1.0, r, []), fallback.update(1.0, r, []))
        self.assertEqual(fallback.heading_segments, 0)
        self.assertEqual(fallback.heading_fallbacks, 1)

    def test_output_lead_still_uses_the_current_heading(self) -> None:
        # 超前项 v·lag 描述的是**现在**的位置，必须用当前朝向；只有积分段用中点。
        def psi(t: float) -> float:
            return math.radians(90.0 * t)

        r0, r1 = hmd_to_base_rotation(hmd_yaw(0.0)), hmd_to_base_rotation(hmd_yaw(90.0))
        base = DeadReckoner(S, osc_lag_s=0.0, heading_at=psi)
        lead = DeadReckoner(S, osc_lag_s=0.2, heading_at=psi)
        for dr in (base, lead):
            dr.update(0.0, r0, [osc(0.0, 0.0, 1.0)])
        d = (lead.update(1.0, r1, [])[:2, 3] - base.update(1.0, r1, [])[:2, 3]) * S
        self.assertAlmostEqual(math.degrees(math.atan2(d[1], d[0])), 90.0, places=4)
        self.assertAlmostEqual(float(np.hypot(*d)), 1.0 * 0.2, places=6)


class HeadingTrackTest(unittest.TestCase):
    """朝向历史：给积分器按区间中点插值用。"""

    def test_interpolates_and_reports_span(self) -> None:
        h = HeadingTrack()
        for i, yaw in enumerate((0.0, 10.0, 20.0)):
            h.push(float(i) * 0.5, math.radians(yaw))
        self.assertAlmostEqual(math.degrees(h.at(0.25)), 5.0, places=6)
        self.assertAlmostEqual(math.degrees(h.at(0.75)), 15.0, places=6)
        self.assertAlmostEqual(h.span_s(), 1.0, places=6)

    def test_outside_the_track_returns_none(self) -> None:
        h = HeadingTrack()
        self.assertIsNone(h.at(0.0))                       # 一个点都没有
        h.push(0.0, 0.0)
        self.assertIsNone(h.at(0.0))                       # 只有一个点：插不了，别猜
        h.push(1.0, 0.1)
        self.assertIsNone(h.at(-0.5))                      # 早于起点
        self.assertIsNone(h.at(1.5))                       # 晚于终点

    def test_unwraps_across_pi_so_interpolation_does_not_jump(self) -> None:
        # 展开存储：跨 ±π 的两点插出来必须是连续的中值，不是绕远的那条。
        h = HeadingTrack()
        h.push(0.0, math.radians(170.0))
        h.push(1.0, math.radians(-170.0))
        self.assertAlmostEqual(abs(math.degrees(h.at(0.5))), 180.0, places=6)

    def test_window_is_bounded_and_clock_rollback_overwrites(self) -> None:
        h = HeadingTrack(span_s=1.0)
        for i in range(40):                                # 20 s 的样本，窗口只留 1 s
            h.push(i * 0.5, math.radians(i))
        self.assertLessEqual(h.span_s(), 1.0 + 0.5)
        t_last, y_last = h._t[-1], h._y[-1]
        h.push(t_last - 0.1, math.radians(-90.0))          # 时钟回退：改最后一点，不新增
        self.assertEqual(len(h._t), len(h._y))
        self.assertAlmostEqual(h._t[-1], t_last - 0.1, places=9)
        self.assertNotAlmostEqual(h._y[-1], y_last, places=6)


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


class _FakeMatcher:
    """替身视差后端（``make_stereo_backend`` 的返回值）：恒返回 4×4 零视差，不跑 SGBM / LAS2。"""

    def __init__(self, kind: str = "fake") -> None:
        self.kind = kind
        self.last_left = None
        self.last_right = None

    def compute(self, left, right):
        self.last_left = left
        self.last_right = right
        return np.zeros((4, 4), np.float32)

    def describe(self):
        return {"kind": self.kind}


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

    def snapshot_devices(self) -> list:
        return [{"i": 0, "c": 1, "r": 0, "v": True, "T": [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, 0]]}]

    def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(self, armed: bool = True, vz: float = 0.0, record_root: Path | None = None,
                 frozen_osc: bool = False, avatar: dict | None = None, decoupled: bool = True) -> None:
        self.armed = armed
        self.vz = vz
        self.frozen_osc = frozen_osc
        self.moves: list[tuple[float, int]] = []
        self.turns: list[float] = []
        self.stops = 0
        self.sensors = FakeSensors()
        self._osc_ts: float | None = None
        # 默认后端是 las2：不钉 sgbm 的话没 mock make_stereo_backend 的用例会真去载 LAS2
        # （.venv 里退回 + 记错误；.venv_las 里首帧编译几分钟）。要测 las2 的用例自己改 cfg。
        cfg = OnlineNavConfig(map_min_interval_s=0.1, stereo_period_s=0.05, kf_max_age_s=0.3,
                              decoupled=decoupled, stereo_backend="sgbm")
        self.nav = OnlineNavigator(
            motion_history=self._motion,
            send_move=lambda f, ms: self.moves.append((f, ms)),
            send_turn=lambda d: self.turns.append(d),
            stop_motion=self._stop,
            drive_block_reason=lambda: None if self.armed else "autonomy_not_armed",
            sensors_factory=lambda: self.sensors, cfg=cfg, record_root=record_root,
            vertical_history=lambda: [{"timestamp": 1.0, "source": "Grounded", "velocity_y": None,
                                       "grounded": True},
                                      {"timestamp": 2.0, "source": "VelocityY", "velocity_y": 0.4,
                                       "grounded": False}],
            # 同一 bundle 两条共用时间戳：必须按 seq 去重，不能按 t。
            param_history=lambda: [{"seq": 1, "timestamp": 1.0, "name": "ScaleFactor", "value": 1.1131},
                                   {"seq": 2, "timestamp": 1.0, "name": "/avatar/eyeheight", "value": 1.4},
                                   {"seq": 3, "timestamp": 1.5, "name": "Upright", "value": 0.98}],
            # 默认不带眼高：world_scale 不随头像缩放（= S），其余用例的世界米断言才成立。
            avatar_snapshot=lambda: dict(avatar or {"avatar_id": "avtr_x"}))

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


class SteamVROpenGuardTest(unittest.TestCase):
    """打开 SteamVR 的**前置自检**与**原生崩翻译**。

    2026-10-06 实况：`sensors_open: OSError: exception: access violation reading
    0x0000000000000000` —— `openvr.init()` 的错误码是 OK，但拿到的 IVRSystem 是空指针，
    直到第一次调用才在原生层崩。这里钉住三件事：① SteamVR 没在运行/没装时**不许去 init**
    （那正是崩的来源）；② init 成功过才允许 shutdown；③ 原生崩要翻成"该怎么办"，且**保留原始文本**。
    """

    class _FakeOpenVR:
        VRApplication_Background = 3
        Eye_Left, Eye_Right = 0, 1

        def __init__(self, *, installed=True, hmd=True, system="stub", init_exc=None,
                     call_exc=None):
            self.installed, self.hmd = installed, hmd
            self._system, self._init_exc, self._call_exc = system, init_exc, call_exc
            self.init_calls = self.shutdown_calls = 0

        def isRuntimeInstalled(self):
            return self.installed

        def isHmdPresent(self):
            return self.hmd

        def init(self, app_type, startup_info=None):
            self.init_calls += 1
            if self._init_exc is not None:
                raise self._init_exc
            return self._system

        def shutdown(self):
            self.shutdown_calls += 1

    def _open(self, fake):
        with mock.patch.dict(sys.modules, {"openvr": fake}):
            sensors = nav_online.OpenVRSensors()
            try:
                sensors.open()
            finally:
                sensors.close()

    def test_refuses_before_init_when_steamvr_is_not_running(self) -> None:
        fake = self._FakeOpenVR(hmd=False)
        with self.assertRaises(RuntimeError) as raised:
            self._open(fake)
        self.assertIn("SteamVR 没在运行", str(raised.exception))
        self.assertEqual(fake.init_calls, 0, "SteamVR 没开时不许去 init —— 那正是读空指针的来源")
        self.assertEqual(fake.shutdown_calls, 0)

    def test_refuses_when_runtime_is_missing(self) -> None:
        fake = self._FakeOpenVR(installed=False)
        with self.assertRaises(RuntimeError) as raised:
            self._open(fake)
        self.assertIn("未安装", str(raised.exception))
        self.assertEqual(fake.init_calls, 0)

    def test_null_system_is_reported_not_dereferenced(self) -> None:
        fake = self._FakeOpenVR(system=None)
        with self.assertRaises(RuntimeError) as raised:
            self._open(fake)
        self.assertIn("空对象", str(raised.exception))
        self.assertEqual(fake.init_calls, 1)
        self.assertEqual(fake.shutdown_calls, 0, "没 init 成功就不许 shutdown")

    def test_native_crash_becomes_actionable_and_keeps_raw_text(self) -> None:
        raw = "exception: access violation reading 0x0000000000000000"

        class DeadSystem:
            def getEyeToHeadTransform(self, eye):
                raise OSError(raw)

        fake = self._FakeOpenVR(system=DeadSystem())
        with self.assertRaises(RuntimeError) as raised:
            self._open(fake)
        msg = str(raised.exception)
        self.assertIn("access violation", msg, "原始文本是现场证据，不许吞")
        self.assertIn("先启动 SteamVR", msg)
        self.assertEqual(fake.shutdown_calls, 1, "init 成功过才允许 shutdown")

    def test_error_code_failure_is_translated(self) -> None:
        fake = self._FakeOpenVR(init_exc=RuntimeError("VRInitError_Init_HmdNotFound"))
        with self.assertRaises(RuntimeError) as raised:
            self._open(fake)
        self.assertIn("HmdNotFound", str(raised.exception))
        self.assertEqual(fake.shutdown_calls, 0, "init 抛错时不许 shutdown")


class SnapshotDevicesTest(unittest.TestCase):
    """devices.jsonl 的来源：读本拍 hmd_rotation() 填好的位姿数组，不再另取一次。"""

    class _Pose:
        def __init__(self, x: float, valid: bool = True) -> None:
            self.mDeviceToAbsoluteTracking = [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 1.0, -0.3]]
            self.bPoseIsValid = valid

    class _System:
        # 槽位 → (类别, 角色, 序列号)；没列的槽位视为未连接。
        devs = {0: (1, 0, "HMD-SN"), 1: (2, 1, "LH-SN"), 2: (2, 2, ""), 5: (3, 0, "TRK-SN")}

        def __init__(self) -> None:
            self.pose_calls = 0
            self.prop_calls = 0

        def getDeviceToAbsoluteTrackingPose(self, origin, predict, poses):
            self.pose_calls += 1

        def isTrackedDeviceConnected(self, i):
            return i in self.devs

        def getTrackedDeviceClass(self, i):
            return self.devs[i][0]

        def getControllerRoleForTrackedDeviceIndex(self, i):
            return self.devs[i][1]

        def getStringTrackedDeviceProperty(self, i, prop):
            self.prop_calls += 1
            if prop == 7000:
                raise RuntimeError("TrackedProp_UnknownProperty")
            return self.devs[i][2] if prop == 1002 else ""

    def _sensors(self):
        fake_vr = mock.Mock(Prop_SerialNumber_String=1002, Prop_ModelNumber_String=1001,
                            Prop_TrackingSystemName_String=1000, Prop_ControllerType_String=7000)
        s = nav_online.OpenVRSensors()
        s._vr, s._system = fake_vr, self._System()
        s._poses = [self._Pose(0.1 * i, valid=i != 5) for i in range(8)]
        return s

    def test_lists_connected_devices_with_pose_and_role(self) -> None:
        s = self._sensors()
        d = s.snapshot_devices()
        self.assertEqual(s._system.pose_calls, 0, "必须复用本拍位姿，不许另取一次（时刻会错开）")
        self.assertEqual([x["i"] for x in d], [0, 1, 2, 5])
        self.assertEqual([(x["c"], x["r"]) for x in d], [(1, 0), (2, 1), (2, 2), (3, 0)])
        self.assertEqual(d[1]["serial"], "LH-SN")
        self.assertNotIn("serial", d[2], "空属性不记")
        self.assertNotIn("ctype", d[0], "取不到的属性跳过，不抛")
        self.assertFalse(d[3]["v"])
        self.assertEqual(np.array(d[1]["T"]).shape, (3, 4))
        self.assertAlmostEqual(d[1]["T"][0][3], 0.1)
        json.dumps(d)                                     # 能直接落 jsonl

    def test_static_props_are_cached_role_is_not(self) -> None:
        s = self._sensors()
        s.snapshot_devices()
        n = s._system.prop_calls
        s._system.devs = {**s._system.devs, 1: (2, 2, "LH-SN")}     # 两只手柄互换左右
        d = s.snapshot_devices()
        self.assertEqual(s._system.prop_calls, n, "型号/序列号每个设备只问一次")
        self.assertEqual(d[1]["r"], 2)


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

        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()),                 mock.patch.object(nav_online.NavSession, "compute", slow):
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
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
            h.nav.start()
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and h.nav.status()["map_updates"] < 2:
                    time.sleep(0.05)
                self.assertTrue(h.nav.explore()["ok"])
                time.sleep(0.3)
                st = h.nav.status()
                cfg_used = h.nav.cfg
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
        # 实测频率必须与「配置频率」**并排**出现在 status 里（声明值 != 生效值的第二例）。
        rt = st["rates"]
        # 目标值必须**跟着配置走**：测试台自己把 stereo_period_s 调成 0.05（20 Hz），
        # 这里写死 10.0 就会红 —— 而红得对，说明字段没被写死。
        self.assertEqual(rt["pose_hz_target"], cfg_used.pose_hz)
        self.assertEqual(rt["stereo_hz_target"], round(1.0 / cfg_used.stereo_period_s, 2))
        for k in ("pose_hz", "stereo_hz", "keyframe_hz", "pose_gap_p50_s",
                  "stereo_gap_p50_s", "stereo_ticks", "samples"):
            self.assertIn(k, rt, "status()[\"rates\"] 少了 %s" % k)
        # 跑起来了就三个表都该有样本；没样本说明打点位置挪了。
        self.assertGreaterEqual(rt["samples"]["pose"], 2)
        self.assertGreaterEqual(rt["samples"]["stereo"], 1)
        self.assertGreaterEqual(rt["samples"]["keyframe"], 1)
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
             mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
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
             mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
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
        self.assertIn("prior", _GRID_LAYERS)
        # 图层名是白名单，HTTP 层按它回落；写成集合会丢顺序，这里只查内容
        self.assertEqual(set(_GRID_LAYERS) & {"tristate", "prior", "surface", "bands", "clearance"},
                         set(_GRID_LAYERS))

    def test_prior_layer_marks_what_the_prior_actually_filled(self) -> None:
        """``prior`` 图层：亮色 = 先验**真补进去**的格，暗色 = 先验有意见但 live 已有证据。

        这一层是"先验注入到底生效了没有"的唯一肉眼入口 —— 没有它，``applied_mask``
        只有 status 里的计数，看不出补在哪、跟 live 冲突在哪。
        """
        import base64 as b64
        import cv2
        from neko_anyadance_body.backend.nav_prior import PriorConsumer, PriorConfig, project_prior
        from tests.test_nav_prior import make_grid, sample_prior, cell_of, SAMPLE_XY
        import tempfile as _tf

        with _tf.TemporaryDirectory() as tmp:
            wdir = Path(tmp) / "wrld_x"
            h = Harness(armed=False)
            h.nav._prior = PriorConsumer(
                PriorConfig(), wdir, world_scale=S,
                pose_of=lambda k: None,
                pose_of_old=lambda sid, kf: None, table_base=lambda sid: None)
            ovl = project_prior(sample_prior(), np.eye(2), np.zeros(2))
            h.nav._prior._overlay = ovl
            h.nav.session.ng = make_grid(world_scale=S)      # 全 UNK 的会话帧栅格
            # live 先看了一格：先把先验唯一的 OCC 格改成 live FREE（先验不许动它）
            r, c = cell_of(h.nav.session.ng, *SAMPLE_XY[4])
            h.nav.session.ng.grid[r, c] = nav_online.FREE
            ovl.apply_into(h.nav.session.ng)
            view = h.nav.grid_view("prior")
            img = cv2.imdecode(np.frombuffer(b64.b64decode(view["png_base64"]), np.uint8),
                               cv2.IMREAD_COLOR)                       # BGR，与后端 _PRIOR_COLORS 同序
            self.assertTrue(view["ramp"]["available"])
            self.assertEqual(view["ramp"]["applied_free"], 4)
            self.assertEqual(view["ramp"]["applied_occ"], 0)           # 被 live free 挡住了
            self.assertEqual(view["ramp"]["not_applied"], 1)
            r, c = cell_of(h.nav.session.ng, *SAMPLE_XY[0])
            self.assertEqual(tuple(int(v) for v in img[r, c]), (150, 225, 150), "先验补 free 应是亮绿")
            r, c = cell_of(h.nav.session.ng, *SAMPLE_XY[4])
            self.assertEqual(tuple(int(v) for v in img[r, c]), (50, 50, 130), "未采纳的先验障碍应是暗红")
            r, c = cell_of(h.nav.session.ng, -1.5, -1.5)               # 先验范围外
            self.assertEqual(tuple(int(v) for v in img[r, c]), (90, 90, 90), "先验不知道的格保持 unknown 灰")

    def test_prior_layer_occ_color_is_red_not_blue(self) -> None:
        """BGR 陷阱回归：``cv2.imencode`` 认 BGR，写 RGB 会让"先验障碍"显示成蓝色。

        triatate / 灰阶那几层看不出来（对称色），彩色层才暴露 —— 用一组已知格把它钉住。
        """
        import base64 as b64
        import cv2
        from neko_anyadance_body.backend.nav_prior import (L_OCC, PriorMap, PriorConsumer,
                                                           PriorConfig, project_prior)
        from tests.test_nav_prior import make_grid, cell_of
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(armed=False)
            h.nav._prior = PriorConsumer(PriorConfig(), Path(tmp) / "w", world_scale=S,
                                         pose_of=lambda k: None)
            lab = np.zeros((40, 40), np.uint8)
            lab[15, 20] = L_OCC
            ovl = project_prior(PriorMap(labels=lab, origin_xy=np.array([-2.0, -2.0]), res_m=0.1,
                                         world_scale=S, base_sid="R", meta={}, path=Path("x")),
                                np.eye(2), np.zeros(2))
            h.nav._prior._overlay = ovl
            h.nav.session.ng = make_grid(world_scale=S)
            ovl.apply_into(h.nav.session.ng)
            img = cv2.imdecode(np.frombuffer(b64.b64decode(
                h.nav.grid_view("prior")["png_base64"]), np.uint8), cv2.IMREAD_COLOR)
            r, c = cell_of(h.nav.session.ng, 0.05, -0.45)
            b, g, rr = (int(v) for v in img[r, c])
            self.assertEqual((rr, g, b), (225, 60, 60), f"先验障碍应显示为红，实际 RGB=({rr},{g},{b})")

    def test_prior_layer_without_prior_is_honest(self) -> None:
        import base64 as b64
        import cv2
        h = Harness(armed=False)
        h.nav.session.ng = None                      # 未建图 → grid_view 直接 None
        h.nav._prior = None
        self.assertIsNone(h.nav.grid_view("prior"))
        from neko_anyadance_body.backend.nav_prior import PriorConsumer, PriorConfig
        from tests.test_nav_prior import make_grid
        h.nav._prior = PriorConsumer(PriorConfig(), Path("nowhere"), world_scale=S,
                                     pose_of=lambda k: None)
        h.nav.session.ng = make_grid(world_scale=S)
        # 还没跑过第一次栅格化 ⇒ 状态是 idle（建图线程一调 overlay() 就变 no_prior）
        self.assertEqual(h.nav.grid_view("prior")["ramp"]["reason"], "idle")
        self.assertIsNone(h.nav._prior.overlay())        # 没有先验文件 ⇒ 一格都不注入
        view = h.nav.grid_view("prior")
        self.assertFalse(view["ramp"]["available"])
        self.assertEqual(view["ramp"]["reason"], "no_prior")
        decoded = cv2.imdecode(np.frombuffer(b64.b64decode(view["png_base64"]), np.uint8),
                              cv2.IMREAD_COLOR)
        self.assertEqual(tuple(int(v) for v in decoded[0, 0]), (90, 90, 90))

    def test_session_is_persisted_to_world_memory(self) -> None:
        from neko_anyadance_body.backend.nav_memory import MemoryConfig, NavMemoryStore, world_dir_name
        with tempfile.TemporaryDirectory() as tmp:
            store = NavMemoryStore(Path(tmp), MemoryConfig(min_session_keyframes=1))
            h = Harness(armed=False, vz=0.5)
            h.nav.memory = store
            h.nav._world_identity = lambda: {"world_key": "wrld_test", "world_name": "t", "world_source": "manual"}
            with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                     mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
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
            # 把 error 带进断言消息：这条曾经只报 "'error' != 'complete'"，
            # 看不到写盘线程到底抛了什么（`SessionWriter._run` 把异常吞进 _meta["error"]）。
            self.assertEqual(info["status"], "complete",
                             f"会话状态异常，error={info.get('error')!r}")
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

    def test_confirmed_constraints_feed_prior_consumer(self) -> None:
        """P0.3b 先验消费接线：建图线程把 ``on_keyframe`` 的确认约束喂进 PriorConsumer。

        先验（世界系）能不能注入，全靠这批约束估出的 gauge；旧实现把返回值丢了，
        这一步断了在线消费就永远是 no gauge（空转不报错，只有 status 能看出来）。
        """
        from neko_anyadance_body.backend.nav_memory import MemoryConfig, NavMemoryStore
        from neko_anyadance_body.backend.nav_prior import PriorConsumer

        class FakeTracker:
            def __init__(self, cfg, world_dir, sid, session_dir=None, world_scale=None):
                self.world_dir = Path(world_dir)
                self.seen: list[int] = []

            def status(self):
                return {"active": True, "reason": "ok"}

            def write_back(self, table):
                return {"written": False}

            def on_keyframe(self, k, feat, dist, R, refresh):
                self.seen.append(int(k))
                return [{"new_sid": "S", "new_kf": int(k), "old_sid": "O", "old_kf": 0,
                         "inliers": 40, "t_ab": [0.0, 0.0, 0.0], "R_ab": np.eye(3).tolist()}]

            def old_pose(self, sid, kf):
                return np.eye(3), np.zeros(2)

            def table_base(self, sid):
                return "O"

        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(armed=False, vz=0.5)
            h.nav.memory = NavMemoryStore(Path(tmp), MemoryConfig(min_session_keyframes=1))
            h.nav._world_identity = lambda: {"world_key": "wrld_test", "world_name": "t",
                                             "world_source": "manual"}
            with mock.patch.object(nav_online, "XSessionTracker", FakeTracker),                     mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                     mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
                h.nav.start()
                try:
                    self.assertIsInstance(h.nav._prior, PriorConsumer)
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline and h.nav.status()["prior"]["n_constraints"] < 1:
                        time.sleep(0.05)
                    st = h.nav.status()["prior"]
                finally:
                    h.nav.stop()
        self.assertGreaterEqual(st["n_constraints"], 1,
                                f"确认约束没进先验消费（n={st['n_constraints']}）：{st}")
        self.assertEqual(st["state"], "no_prior", "临时世界没有 prior/ ⇒ 如实报 no_prior")

    def test_external_hmd_yaw_snap_reported(self) -> None:
        h = Harness(armed=False)
        yaw = {"deg": 0.0}
        h.sensors.hmd_rotation = lambda: hmd_yaw(yaw["deg"])
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
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
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
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
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
            h.nav.start()
            try:
                time.sleep(0.2)
                st = h.nav.status()
            finally:
                h.nav.stop()
        self.assertIs(st["baseline_check"]["ok"], True)
        self.assertFalse(any(e.startswith("baseline_mismatch") for e in st["errors"]))


class ConfigSurfaceTest(unittest.TestCase):
    def test_avatar_scale_factor(self) -> None:
        ref = 1.2577
        self.assertEqual(nav_online.avatar_scale_factor({}, ref), (1.0, "none"))
        self.assertEqual(nav_online.avatar_scale_factor({"/avatar/eyeheight": 1.4}, 0.0), (1.0, "disabled"))
        f, src = nav_online.avatar_scale_factor({"EyeHeightAsMeters": 1.4, "/avatar/eyeheight": 2.0,
                                                 "ScaleFactor": 3.0}, ref)
        self.assertEqual(src, "EyeHeightAsMeters")
        self.assertAlmostEqual(f, 1.4 / ref)
        self.assertEqual(nav_online.avatar_scale_factor({"ScaleFactor": 1.1131}, ref), (1.1131, "ScaleFactor"))
        # 坏值（非数、0、离谱倍数）跳过，落到下一个来源或不缩放。
        self.assertEqual(nav_online.avatar_scale_factor({"EyeHeightAsMeters": 0.0, "ScaleFactor": "x"}, ref),
                         (1.0, "none"))
        self.assertEqual(nav_online.avatar_scale_factor({"/avatar/eyeheight": float("nan"), "ScaleFactor": 2.0},
                                                        ref), (2.0, "ScaleFactor"))

    def test_avatar_scale_applied_once_per_start(self) -> None:
        snap = {"/avatar/eyeheight": 1.4}
        nav = Harness(armed=False, avatar=snap).nav
        nav._avatar_snapshot = lambda: dict(snap)
        base = nav.cfg.world_scale
        f = 1.4 / 1.2577
        for _ in range(2):                       # 重开会话不叠乘
            nav._apply_avatar_scale(dict(snap))
            self.assertAlmostEqual(nav.cfg.world_scale, base * f)
            self.assertAlmostEqual(nav.cfg.mapper.world_scale, base * f)
            self.assertAlmostEqual(nav.cfg.loop.world_scale, base * f)
        self.assertEqual(nav.cfg.osc_speed_scale, OnlineNavConfig().osc_speed_scale)   # DR 的 k 不动
        self.assertFalse(nav._read_avatar_scale()["changed"])
        snap["/avatar/eyeheight"] = 1.2577       # 会话中改回默认：报 changed，读数本身不改已用的尺度
        self.assertTrue(nav._read_avatar_scale()["changed"])
        snap["/avatar/eyeheight"] = 1.4 * 1.008  # 1% 以内不算变
        self.assertFalse(nav._read_avatar_scale()["changed"])
        self.assertAlmostEqual(nav.cfg.world_scale, base * f)
        nav._apply_avatar_scale({})              # 收不到眼高 ⇒ 回到配置原值
        self.assertEqual(nav.cfg.world_scale, base)
        nav.cfg = OnlineNavConfig(world_scale=0.9, avatar_ref_eye_height_m=0.0)   # 外部换 cfg 以新值为准；0 = 关
        nav._apply_avatar_scale(dict(snap))
        self.assertEqual(nav.cfg.world_scale, 0.9)

    def test_rescale_waits_for_settle_then_restarts(self) -> None:
        """会话中缩放偏离 >1%：读数稳住 2 s 才重开（拖动手势是连续变的），重开后 world_scale 跟新眼高。"""
        snap = {"/avatar/eyeheight": 1.4}
        nav = Harness(armed=False, avatar=snap).nav
        nav._avatar_snapshot = lambda: dict(snap)
        nav._apply_avatar_scale(dict(snap))
        calls: list[dict] = []
        with mock.patch.object(nav_online.threading, "Thread") as T:
            T.side_effect = lambda target, args, **kw: mock.Mock(start=lambda: calls.append(args[0]))
            nav._watch_avatar_scale(0.0)
            snap["/avatar/eyeheight"] = 1.30         # 拖动中
            nav._watch_avatar_scale(1.0)
            snap["/avatar/eyeheight"] = 1.2577
            nav._watch_avatar_scale(2.0)
            nav._watch_avatar_scale(3.0)
            self.assertEqual(calls, [])            # 1.2577 才看到 1 s
            self.assertTrue(nav.status()["avatar_scale"]["changed"])
            nav._watch_avatar_scale(4.0)
            self.assertEqual(len(calls), 1)
            nav._watch_avatar_scale(5.0)           # 重开进行中不再触发
            self.assertEqual(len(calls), 1)
        nav._rescaling = False
        # 真跑一次重开：stop+start，记录一条，world_scale 回到 S × 1.0。
        with mock.patch.object(type(nav), "running", new_callable=mock.PropertyMock, return_value=True),                 mock.patch.object(nav, "stop") as stop,                 mock.patch.object(nav, "start", side_effect=lambda **kw: nav._apply_avatar_scale(dict(snap))) as start:
            nav._start_args = {"record": True, "record_images": False}
            nav._rescale_restart(calls[0])
        stop.assert_called_once()
        start.assert_called_once_with(record=True, record_images=False)
        r = nav._rescales[-1]
        self.assertAlmostEqual(r["from_factor"], 1.4 / 1.2577, places=4)
        self.assertEqual((r["to_factor"], r["world_scale"]), (1.0, S))
        self.assertFalse(nav._rescaling)

    def test_rescale_skipped_if_user_stopped(self) -> None:
        nav = Harness(armed=False).nav
        nav._stop.set()
        with mock.patch.object(type(nav), "running", new_callable=mock.PropertyMock, return_value=True),                 mock.patch.object(nav, "start") as start:
            nav._rescale_restart({"factor": 1.0})
        start.assert_not_called()
        self.assertEqual(nav._rescales, [])

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
                          (config.NAVMESH_LOOP_KEYS, LoopConfig), (config.NAVMESH_PRIOR_KEYS, PriorConfig)):
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
            "mapper": {"range_m": 4.0, "ray_clear": False}, "loop": {"min_inliers": 40},
            "prior": {"min_constraints": 12, "max_pos_med_m": 0.25}}}).navmesh
        c = OnlineNavConfig.from_plugin(nav)
        self.assertEqual((c.world_scale, c.kf_dist_m, c.stop_min_pts, c.loop_closure), (0.8, 0.5, 20, False))
        self.assertEqual((c.mapper.range_m, c.mapper.ray_clear, c.mapper.world_scale), (4.0, False, 0.8))
        self.assertEqual((c.loop.min_inliers, c.loop.world_scale), (40, 0.8))
        self.assertEqual((c.prior.min_constraints, c.prior.max_pos_med_m), (12, 0.25))
        for bad in ({"world_scale": 0}, {"kf_dst_m": 0.5}, {"mapper": {"world_scale": 0.9}},
                    {"mapper": {"range_m": "far"}}, {"loop": {"min_inliers": 1.5}}, {"mapper": 3},
                    {"prior": {"max_pos_med_m": "tight"}}, {"prior": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                config.PluginConfig.from_mapping({"navmesh": bad})


class RecorderTest(unittest.TestCase):
    def run_session(self, root: Path | None, record: bool, frozen_osc: bool = False,
                    record_images: bool = False, avatar: dict | None = None, decoupled: bool = True) -> dict:
        h = Harness(armed=False, vz=0.5, record_root=root, frozen_osc=frozen_osc, avatar=avatar,
                    decoupled=decoupled)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
            h.nav.start(record=record, record_images=record_images)
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
            st = self.run_session(Path(d), record=True, frozen_osc=True,
                                  avatar={"avatar_id": "avtr_x", "/avatar/eyeheight": 1.4})
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
            # 每个关键帧都能按 t 逐位配上一行设备位姿（解耦路径拿不到 k，只靠 t）。
            dev = [json.loads(x) for x in (out / "devices.jsonl").read_text(encoding="utf-8").splitlines()]
            dev_t = {x["t"] for x in dev}
            self.assertTrue(all(e["t"] in dev_t for e in kfs), "有关键帧配不上设备位姿")
            self.assertEqual(dev[0]["d"], FakeSensors().snapshot_devices())
            hmd = (out / "hmd.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(hmd), len(kfs))
            self.assertEqual(np.array(json.loads(hmd[0])["R"]).shape, (3, 3))
            # 假 OSC 历史每拍都一样：同一时间戳只记一次。
            osc_lines = (out / "osc.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(osc_lines), 1)
            # VelocityY / Grounded 也落盘（多层/竖直漂移评估用），同样按时间戳去重。
            vert = [json.loads(x) for x in (out / "vert.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(vert, [{"t": 1.0, "src": "Grounded", "vy": None, "g": True},
                                    {"t": 2.0, "src": "VelocityY", "vy": 0.4, "g": False}])
            params = [json.loads(x) for x in (out / "params.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(params, [{"t": 1.0, "n": "ScaleFactor", "v": 1.1131},
                                      {"t": 1.0, "n": "/avatar/eyeheight", "v": 1.4},
                                      {"t": 1.5, "n": "Upright", "v": 0.98}])
            self.assertEqual(meta["avatar"], {"avatar_id": "avtr_x", "/avatar/eyeheight": 1.4})
            # 眼高 1.4 ⇒ world_scale 按 1.4/1.2577 放大，配置里的 S 原样留作基准。
            f = 1.4 / 1.2577
            self.assertEqual(meta["avatar_scale"]["source"], "/avatar/eyeheight")
            self.assertAlmostEqual(meta["avatar_scale"]["factor"], f, places=4)
            self.assertAlmostEqual(meta["avatar_scale"]["base_world_scale"], S)
            self.assertAlmostEqual(meta["config"]["world_scale"], S * f)
            self.assertAlmostEqual(meta["config"]["mapper"]["world_scale"], S * f)
            self.assertAlmostEqual(st["world_scale"], S * f)
            self.assertFalse(st["avatar_scale"]["changed"])
            with np.load(out / "final_poses.npz") as final:
                self.assertEqual(list(final["ids"]), [e["k"] for e in kfs])

    def test_recording_writes_devices_in_inline_stereo_path(self) -> None:
        """解耦路径 devices 行用 ``t`` 配对；非解耦路径关键帧号同步，也能配上。"""
        with tempfile.TemporaryDirectory() as d:
            st = self.run_session(Path(d), record=True, decoupled=False)
            out = Path(st["recording"]["dir"])
            ev = [json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            kfs = [e for e in ev if e["kind"] == "kf"]
            dev = [json.loads(x) for x in (out / "devices.jsonl").read_text(encoding="utf-8").splitlines()]
            dev_t = {x["t"] for x in dev}
            self.assertTrue(all(e["t"] in dev_t for e in kfs), "有关键帧配不上设备位姿")

    def test_device_snapshot_failure_stops_only_that_stream(self) -> None:
        calls = []

        def boom(self):
            calls.append(1)
            raise OSError("access violation")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(FakeSensors, "snapshot_devices", boom):
            st = self.run_session(Path(d), record=True)
            out = Path(st["recording"]["dir"])
            self.assertGreaterEqual(st["keyframes"], 3, "设备读失败不许拖垮建图")
            self.assertEqual(len(calls), 1, "失败一次就停这一路，不每拍重试")
            self.assertFalse((out / "devices.jsonl").exists())
            self.assertIn("record_devices", json.dumps(st["errors"], ensure_ascii=False))

    def test_record_images_writes_keyframe_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            st = self.run_session(Path(d), record=True, record_images=True)
            out = Path(st["recording"]["dir"])
            kfs = sorted(int(p.stem) for p in (out / "kf").glob("*.npz"))
            self.assertGreaterEqual(len(kfs), 3)
            for k in kfs:
                for eye in "LR":
                    img = cv2.imread(str(out / "kf_img" / f"{k:06d}_{eye}.png"), cv2.IMREAD_UNCHANGED)
                    self.assertIsNotNone(img, f"{k}_{eye}")
                    self.assertEqual(img.shape, (4, 4))
            self.assertTrue(st["recording"]["images"])
            self.assertTrue(json.loads((out / "meta.json").read_text(encoding="utf-8"))["record_images"])

    def test_record_images_keep_the_sensor_colour_space(self) -> None:
        """存档原图必须跟采集端同一个颜色空间，彩色 PNG 的颜色要对（RGB→BGR 后 imwrite）。

        回归点：曾经上游递进来的是 read_stereo() 的灰度
        （3 通道 RGB 就在手边却被丢掉），落盘的永远是 L 型 PNG
        ⇒ 下游给点云上色时只能去猜颜色。
        同时钉住另一件同等重要的事：灰度输入仍然原样落灰度 ——
        所有既有读法（cv2.imread(..., IMREAD_GRAYSCALE)）在改动前后必须一致。
        """
        for name, has_rgb in (("rgb", True), ("gray", False)):
            with self.subTest(sensor=name), tempfile.TemporaryDirectory() as d:
                h = Harness(armed=False, vz=0.5, record_root=Path(d))
                if has_rgb:
                    def read_rgb():
                        # 三通道**各不相同**：全填同一个值就等于自己造了一张灰图，
                        # "三通道不全等"这条断言会永远为假。
                        left = np.zeros((4, 4, 3), np.uint8)
                        left[..., 0], left[..., 1], left[..., 2] = 7, 11, 13
                        right = left + 5
                        return left, right
                else:
                    read_rgb = None
                h.sensors.read_stereo_rgb = read_rgb
                with mock.patch.object(nav_online, "stereo_points",
                                       return_value=ground_and_wall(None)), \
                     mock.patch.object(nav_online, "make_stereo_backend",
                                       return_value=_FakeMatcher()):
                    h.nav.start(record=True, record_images=True)
                    try:
                        deadline = time.monotonic() + 5.0
                        while time.monotonic() < deadline and h.nav.status()["keyframes"] < 3:
                            time.sleep(0.05)
                    finally:
                        st = h.nav.stop()
                out = Path(st["recording"]["dir"])
                kfs = sorted(int(p.stem) for p in (out / "kf").glob("*.npz"))
                self.assertGreaterEqual(len(kfs), 3, f"{name}: 关键帧一个都没落")
                for k in kfs:
                    for eye in "LR":
                        p = out / "kf_img" / f"{k:06d}_{eye}.png"
                        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                        self.assertIsNotNone(raw, f"{name} {k}_{eye}")
                        self.assertEqual(raw.shape[:2], (4, 4))
                        self.assertEqual(raw.ndim, 3 if has_rgb else 2,
                                         f"{name} {k}_{eye} 颜色空间被改掉了")
                        if has_rgb:
                            # 通道顺序也要钉：read_stereo_rgb 给 RGB，imwrite 把数组当 BGR ⇒
                            # 不转就红蓝互换（20261025_202630 就是这样落盘的）。正确落盘后
                            # imread（BGR）的像素应是 (13, 11, 7)，转回 RGB 才是传进去的 (7, 11, 13)。
                            px = raw.reshape(-1, raw.shape[-1])[0].tolist()
                            want = [13, 11, 7] if eye == "L" else [18, 16, 12]
                            self.assertEqual(px, want, f"{name} {k}_{eye} 红蓝反了（RGB 没转 BGR 就 imwrite）")
                        else:
                            # 灰度路径不能被我这次改动弄坏。
                            self.assertEqual(set(np.unique(raw).tolist()), {0})

    def test_las2_gets_rgb_sgbm_gets_gray(self) -> None:
        """LAS2 喂真彩（RGB 比 gray3 EPE 低 5.5–9.5%），SGBM 照旧吃灰度。

        回归点：选 las2 但没有记忆、没开录图时，``need_rgb`` 曾经为假 ⇒ 根本不读 RGB。
        """
        for kind in ("las2", "sgbm"):
            with self.subTest(backend=kind):
                h = Harness(armed=False, vz=0.5)
                h.nav.cfg.stereo_backend = kind

                def read_rgb():
                    left = np.zeros((4, 4, 3), np.uint8)
                    left[..., 0], left[..., 1], left[..., 2] = 7, 11, 13
                    return left, left + 5
                h.sensors.read_stereo_rgb = read_rgb
                spy = _FakeMatcher(kind)
                with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)), \
                     mock.patch.object(nav_online, "make_stereo_backend", return_value=spy):
                    h.nav.start()
                    try:
                        deadline = time.monotonic() + 5.0
                        while time.monotonic() < deadline and spy.last_left is None:
                            time.sleep(0.02)
                    finally:
                        h.nav.stop()
                self.assertIsNotNone(spy.last_left, f"{kind}: 视差后端一帧都没收到")
                if kind == "las2":
                    self.assertEqual(spy.last_left.shape, (4, 4, 3))
                    self.assertEqual(spy.last_left[0, 0].tolist(), [7, 11, 13], "通道序必须是 RGB")
                    self.assertEqual(spy.last_right[0, 0].tolist(), [12, 16, 18])
                else:
                    self.assertEqual(spy.last_left.shape, (4, 4))

    def test_record_images_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            st = self.run_session(Path(d), record=True)
            self.assertFalse((Path(st["recording"]["dir"]) / "kf_img").exists())
            self.assertFalse(st["recording"]["images"])

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
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),             mock.patch.object(nav_online, "make_stereo_backend", return_value=_FakeMatcher()):
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
            # 逐对采纳之后必须再跑一次世界树：它才看得见"别的会话之间的桥"
            tree = out.get("world_tree")
            self.assertIsNotNone(tree, out)
            self.assertIn("aligned", tree)

    def test_world_tree_pass_runs_after_pairwise_align(self) -> None:
        seen: dict = {}

        nav = OnlineNavigator.__new__(OnlineNavigator)
        nav._state_lock = threading.Lock()
        nav._errors = []
        nav._align_out = None
        nav.cfg = OnlineNavConfig()
        wd = Path("wrld_probe")                       # 只作实参比对，不碰盘
        with mock.patch.object(nav_online, "align_into_auto",
                               return_value={"ok": True, "old_sid": "OLD"}), \
             mock.patch.object(nav_online, "align_world_tree",
                               side_effect=lambda d, c: seen.update(dir=d) or
                               {"ok": True, "aligned": 1, "skipped": 3, "refused": 0,
                                "components": [1], "report": "r"}):
            nav._align_xsession("NEW", wd)
        self.assertEqual(seen.get("dir"), wd, "并树必须作用在同一个世界目录上")
        out = nav._align_out
        self.assertTrue(out["ok"], "并树不影响逐对采纳的结果字段")
        self.assertEqual(out["world_tree"]["aligned"], 1)
        self.assertEqual(out["world_tree"]["components"], 1)

    def test_world_tree_can_be_switched_off(self) -> None:
        from neko_anyadance_body.backend.nav_xsession import XSessionConfig
        nav = OnlineNavigator.__new__(OnlineNavigator)
        nav._state_lock = threading.Lock()
        nav._errors = []
        nav._align_out = None
        nav.cfg = OnlineNavConfig(xsession_world_tree=False)
        with mock.patch.object(nav_online, "align_into_auto", return_value={"ok": True}), \
             mock.patch.object(nav_online, "align_world_tree") as tree:
            nav._align_xsession("NEW", Path("wrld_probe"))
        tree.assert_not_called()
        self.assertNotIn("world_tree", nav._align_out)
        self.assertIsInstance(XSessionConfig(), XSessionConfig)

    def test_world_tree_failure_keeps_align_result_and_records_error(self) -> None:
        nav = OnlineNavigator.__new__(OnlineNavigator)
        nav._state_lock = threading.Lock()
        nav._errors = []
        nav._align_out = None
        nav.cfg = OnlineNavConfig()
        with mock.patch.object(nav_online, "align_into_auto", return_value={"ok": True, "old_sid": "OLD"}), \
             mock.patch.object(nav_online, "align_world_tree", side_effect=RuntimeError("boom")):
            nav._align_xsession("NEW", Path("wrld_probe"))
        self.assertTrue(nav._align_out["ok"], "并树炸了不能把采纳结果也带下去")
        self.assertFalse(nav._align_out["world_tree"]["ok"])
        self.assertIn("boom", nav._align_out["world_tree"]["reason"])
        self.assertTrue(any("xsession_tree" in e for e in nav._errors), nav._errors)


class KeyframeDedupStatusTest(unittest.TestCase):
    """位置去重要**当场可读**：`kf_max_age_s` 让原地不动也每 3 s 补一帧，长停顿的会话
    会把同一地点反复计成独立关键帧。它同时污染离线拟合的权重与位姿图的 σ，
    所以不能再靠事后翻盘发现（实测 045615 有 41.5%，其余 15 场 ≤2%）。"""

    def _status(self, keyframes: int, unique: int, refreshed: int) -> dict:
        nav = OnlineNavigator(motion_history=lambda: [], send_move=lambda f, ms: None,
                              send_turn=lambda d: None, stop_motion=lambda: None,
                              drive_block_reason=lambda: None,
                              sensors_factory=lambda: FakeSensors(),
                              cfg=OnlineNavConfig(stereo_period_s=0.05),
                              avatar_snapshot=lambda: {"avatar_id": "avtr_x"})
        nav._keyframes = keyframes        # noqa: SLF001 - 直接摆状态，不跑线程
        nav._kf_unique = unique           # noqa: SLF001
        nav._kf_refreshed = refreshed     # noqa: SLF001
        return nav.status()

    def test_clean_session_reports_ok(self) -> None:
        st = self._status(10, 10, 0)
        self.assertEqual(st["keyframes_unique"], 10)
        self.assertEqual(st["keyframes_duplicate_frac"], 0.0)
        self.assertEqual(st["keyframe_dedup_reason"], "ok")

    def test_standing_still_is_named_as_the_cause(self) -> None:
        st = self._status(100, 55, 45)
        self.assertEqual(st["keyframes_unique"], 55)
        self.assertAlmostEqual(st["keyframes_duplicate_frac"], 0.45, places=4)
        self.assertEqual(st["keyframe_dedup_reason"], "standing_still_refresh",
                         "重复率高且几乎全来自补帧时必须点名补帧")

    def test_repeats_without_refresh_are_named_differently(self) -> None:
        st = self._status(100, 55, 0)
        self.assertEqual(st["keyframe_dedup_reason"], "refresh_and_repeat",
                         "补帧解释不了全部重复时不能把锅全给它")


if __name__ == "__main__":
    unittest.main()
