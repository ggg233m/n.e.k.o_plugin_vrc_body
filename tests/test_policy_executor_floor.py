# -*- coding: utf-8 -*-
"""「不许比跟随器差」的速度下限 + obs_t 相位口径。

两条都是 2026-11-02 的在线诊断结论，都用**可证伪的算术**钉住，不靠感觉：

1. **下限**（SafetyGate.floor_frac / executor.floor_provider）
   VLA 接管后 A* 摇杆被静音，跟随器的 forward 仍在同一拍算出来但没发。实测策略稳态
   只有 0.25-0.5 m/s，而跟随器是同一条路径的满轴 4.0 m/s —— 降级在结构上没有兜底。
   下限只抬**前进分量**，且跟随器自己不走时必须为 0（该停的地方必须能停）。

2. **obs_t**（PolicyRunner.captured_at_source）
   obs_t 在推理前打戳会让执行器永远从 k=L/0.1 起手，块的前几步从不执行；而生产配置
   下那一窗正好是模型的减速谷底。用真实采集时刻后 age 才如实。
"""
from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

from policy_executor import (K_STEPS, PolicyExecutor, SafetyGate, omega_to_delta_deg,
                             omega_to_turn)  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402


def _block(vz: float, vx: float = 0.0):
    return [[vx, vz] for _ in range(K_STEPS)]


class SafetyGateFloorTests(unittest.TestCase):
    def test_omega_positive_is_normalized_as_right_turn(self) -> None:
        self.assertAlmostEqual(omega_to_turn(1.57, 1.57), 1.0)
        self.assertAlmostEqual(omega_to_turn(-1.57, 1.57), -1.0)
        # scheduler 的 correction_deg 正值是左转，因此右正标签必须发负角度。
        self.assertAlmostEqual(omega_to_delta_deg(1.57, 1.57, 100), -9.0, places=2)
        self.assertAlmostEqual(omega_to_delta_deg(-1.57, 1.57, 100), 9.0, places=2)

    def test_floor_zero_is_bitwise_old_behaviour(self) -> None:
        block = [[0.1 * i, 0.05 * i] for i in range(K_STEPS)]
        gate = SafetyGate(v_max=99.0, dv_max=99.0)
        out = gate.apply(block, False, False, 0.0, floor_mps=0.0)
        np.testing.assert_allclose(out, np.asarray(block), rtol=0, atol=0)

    def test_floor_raises_forward_component(self) -> None:
        gate = SafetyGate(v_max=99.0, dv_max=99.0)
        out = gate.apply(_block(0.05), False, False, 0.0, floor_mps=0.6)
        self.assertTrue(np.allclose(out[:, 1], 0.6))

    def test_floor_never_lowers_a_faster_policy(self) -> None:
        """策略比下限快时必须原样保留 —— 下限只能抬，不能压。"""
        gate = SafetyGate(v_max=99.0, dv_max=99.0)
        out = gate.apply(_block(1.5), False, False, 0.0, floor_mps=0.6)
        self.assertTrue(np.allclose(out[:, 1], 1.5))

    def test_floor_does_not_touch_lateral(self) -> None:
        """侧向是 VLA 的修正自由度，抬前进分量不能顺带把它抬走。"""
        gate = SafetyGate(v_max=99.0, dv_max=99.0)
        out = gate.apply(_block(0.0, vx=0.2), False, False, 0.0, floor_mps=0.6)
        self.assertTrue(np.allclose(out[:, 0], 0.2))

    def test_gated_block_ignores_floor(self) -> None:
        """gated（到点）必须能停车 —— 下限不许把停车命令抬起来。"""
        gate = SafetyGate(v_max=99.0, dv_max=99.0)
        out = gate.apply(_block(0.0), True, False, 0.0, floor_mps=0.6)
        self.assertTrue(np.allclose(out, 0.0))

    def test_stale_block_ignores_floor(self) -> None:
        gate = SafetyGate(v_max=99.0, dv_max=99.0)
        out = gate.apply(_block(0.0), False, True, 0.0, floor_mps=0.6)
        self.assertTrue(np.allclose(out, 0.0))

    def test_floor_still_respects_v_max(self) -> None:
        """下限与限速同时成立时，v_max 必须赢（安全门永远是上界）。"""
        gate = SafetyGate(v_max=1.2, dv_max=99.0)
        out = gate.apply(_block(0.0), False, False, 0.0, floor_mps=3.0)
        self.assertTrue(np.all(np.linalg.norm(out, axis=1) <= 1.2 + 1e-9))


class ExecutorFloorProviderTests(unittest.TestCase):
    def _executor(self, floor):
        ex = PolicyExecutor(lambda x, y, d: True, stale_s=99.0, dry_run=True,
                            gate=SafetyGate(v_max=99.0, dv_max=99.0),
                            floor_provider=floor)
        return ex

    def test_provider_scaled_by_floor_frac(self) -> None:
        ex = self._executor(lambda: 1.0)
        ex.submit(_block(0.0), time.perf_counter())
        self.assertTrue(np.allclose(ex._block[:, 1], 0.6))

    def test_floor_frac_zero_disables(self) -> None:
        ex = PolicyExecutor(lambda x, y, d: True, stale_s=99.0, dry_run=True,
                            gate=SafetyGate(v_max=99.0, dv_max=99.0, floor_frac=0.0),
                            floor_provider=lambda: 1.0)
        ex.submit(_block(0.0), time.perf_counter())
        self.assertTrue(np.allclose(ex._block[:, 1], 0.0))

    def test_provider_exception_means_no_floor(self) -> None:
        """读不到跟随器就不能设下限，更不能因此停车。"""
        def boom():
            raise RuntimeError("no follower")
        ex = self._executor(boom)
        ex.submit(_block(0.0), time.perf_counter())
        self.assertTrue(np.allclose(ex._block, 0.0))

    def test_follower_at_zero_means_stop_allowed(self) -> None:
        """跟随器自己不走（到点/未定位/原地转）时下限为 0。"""
        ex = self._executor(lambda: 0.0)
        ex.submit(_block(0.0), time.perf_counter())
        self.assertTrue(np.allclose(ex._block, 0.0))


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def act(self, cur, prev, wp, body, preprocessed=False):
        self.calls.append({"wp": wp, "body": body})
        return dict(block=[[0.0, 0.0]] * 12, gated=False, obs_t=9e9, latency_ms=1.0)


def _frame(v: int) -> np.ndarray:
    return np.full((4, 4, 3), v, np.uint8)


class RunnerObsTTests(unittest.TestCase):
    def _run(self, captured_at):
        got = {}
        rt = _FakeRuntime()

        def on_block(block, obs_t, meta):
            got["obs_t"] = obs_t
            got["meta"] = meta

        runner = PolicyRunner(rt, lambda: _frame(1), lambda: (0.0, 0.8), lambda: (0.0, 0.0),
                              on_block=on_block, dry_run=False, hz=200.0,
                              stale_s=99.0, captured_at_source=captured_at)
        runner.start()
        deadline = time.time() + 2.0
        while not got and time.time() < deadline:
            time.sleep(0.005)
        runner.stop()
        return got

    def test_captured_at_becomes_obs_t(self) -> None:
        """obs_t 应该是**图像采集时刻**，而不是推理开始时刻（后者晚于它）。"""
        cap = time.perf_counter() - 0.35
        got = self._run(lambda: cap)
        self.assertTrue(got, "runner 没有产出")
        self.assertAlmostEqual(got["obs_t"], cap, places=6)
        self.assertGreater(got["meta"]["frame_age_ms"], 340.0)

    def test_missing_provider_falls_back_to_t0(self) -> None:
        got = self._run(None)
        self.assertTrue(got)
        # 退路：obs_t ≈ 本拍开始时刻，frame_age 接近 0
        self.assertLess(got["meta"]["frame_age_ms"], 50.0)

    def test_provider_exception_falls_back(self) -> None:
        def boom():
            raise RuntimeError("no clock")
        got = self._run(boom)
        self.assertTrue(got)
        self.assertLess(got["meta"]["frame_age_ms"], 50.0)

    def test_future_capture_is_clamped(self) -> None:
        """采集时刻晚于本拍开始是不可能的；夹住防止跨域误用把 age 算成负数。"""
        got = self._run(lambda: time.perf_counter() + 100.0)
        self.assertTrue(got)
        self.assertGreaterEqual(got["meta"]["frame_age_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
