# -*- coding: utf-8 -*-
"""PolicyRunner 的历史帧选择：numpy 帧不能被当真值判断。

2026-11-01 真机 dry-run 抓到的 bug：``prev = self._prev(t0) or cur`` 对 numpy
数组求布尔，抛 `ValueError: The truth value of an array with more than one
element is ambiguous`，178/180 拍全废。这里锁死「不传 prev_source 时用当前帧
兜底」和「传了就用传进来的」两条路径。
"""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'tools'))

from policy_runner import PolicyRunner  # noqa: E402


class _FakeRuntime:
    """记录每次 act 收到的 prev 帧，其余照 PolicyRuntime 的形状返回。"""

    def __init__(self):
        self.calls = []

    def act(self, cur, prev, wp, body, preprocessed=False):
        self.calls.append({'cur': cur, 'prev': prev, 'wp': wp, 'body': body})
        return dict(block=[[0.0, 0.0]] * 12, gated=False, obs_t=0.0, latency_ms=1.0)


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, np.uint8)


class PolicyRunnerPrevFrameTests(unittest.TestCase):
    def _run_once(self, **kwargs):
        rt = _FakeRuntime()
        cur = _frame(1)
        runner = PolicyRunner(rt, lambda: cur, lambda: (0.0, 0.8), lambda: (0.0, 0.0),
                              dry_run=True, hz=50.0, **kwargs)
        runner.start()
        deadline = time.time() + 2.0
        while not rt.calls and time.time() < deadline:
            time.sleep(0.01)
        runner.stop()
        return rt

    def test_numpy_frame_does_not_raise_truth_value_error(self) -> None:
        """核心回归：没有 prev_source 时，numpy 主循环不能抛 ValueError。"""
        rt = self._run_once()
        self.assertTrue(rt.calls, 'runner 没有产出任何一拍')
        self.assertIs(rt.calls[0]['prev'], rt.calls[0]['cur'])

    def test_prev_source_is_used_when_provided(self) -> None:
        prev = _frame(9)
        rt = self._run_once(prev_source=lambda: prev)
        self.assertTrue(rt.calls)
        self.assertIs(rt.calls[0]['prev'], prev)

    def test_prev_source_exception_falls_back_to_current(self) -> None:
        def boom():
            raise RuntimeError('no history')
        rt = self._run_once(prev_source=boom)
        self.assertTrue(rt.calls)
        self.assertIs(rt.calls[0]['prev'], rt.calls[0]['cur'])

    def test_buffer_history_is_preferred_over_current_frame(self) -> None:
        """缓冲里已有距今 0.4 s 的帧时，应该用它而不是复制当前帧。"""
        rt = _FakeRuntime()
        old = _frame(7)
        runner = PolicyRunner(rt, lambda: _frame(1), lambda: (0.0, 0.8), lambda: (0.0, 0.0),
                              dry_run=True, hz=50.0)
        # 手动塞一个「距今 0.4 s」的历史帧
        runner._push(time.perf_counter() - 0.4, old)
        runner.start()
        deadline = time.time() + 2.0
        while not rt.calls and time.time() < deadline:
            time.sleep(0.01)
        runner.stop()
        self.assertTrue(rt.calls)
        self.assertIs(rt.calls[0]['prev'], old)


if __name__ == '__main__':
    unittest.main()
