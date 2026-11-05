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

from policy_histframe import HISTORY_FRAME_S  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402


class _FakeRuntime:
    """记录每次 act 收到的 prev 帧，其余照 PolicyRuntime 的形状返回。"""

    def __init__(self):
        self.calls = []
        self.act_dim = 2

    def act(self, cur, prev, wp, body, preprocessed=False):
        self.calls.append({'cur': cur, 'prev': prev, 'wp': wp, 'body': body})
        return dict(block=[[0.0, 0.0]] * 12, gated=False, obs_t=0.0, latency_ms=1.0)


class _SlowThreeChannelRuntime(_FakeRuntime):
    def __init__(self):
        super().__init__()
        self.act_dim = 3

    def act(self, cur, prev, wp, body, preprocessed=False):
        self.calls.append({'cur': cur, 'prev': prev, 'wp': wp, 'body': body})
        return dict(block=[[0.1, 0.2, 0.3]] * 12, gated=False, obs_t=0.0,
                    latency_ms=1000.0)


class _GuidanceRuntime(_FakeRuntime):
    def act(self, cur, prev, wp, body, mode=0, path=None, **_kwargs):
        self.calls.append({"cur": cur, "prev": prev, "wp": wp, "body": body,
                           "mode": mode, "path": path})
        return dict(block=[[0.0, 0.0]] * 12, gated=False, obs_t=0.0, latency_ms=1.0)


class _PromptRuntime(_GuidanceRuntime):
    def act(self, cur, prev, wp, body, mode=0, path=None, task_prompt=None, **kwargs):
        self.calls.append({"task_prompt": task_prompt})
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
        """缓冲里已有距今 HISTORY_FRAME_S 的帧时，应该用它而不是复制当前帧。

        2026-11-05 更新：历史帧间隔由 0.4 s 改为 0.2 s（policy_histframe）。
        本用例原来硬编码 0.4，改口径后 0.4 s 的帧落在 0.15 s 容差之外，
        会走到「回退当前帧」分支 —— 那是**旧口径的正确行为**，不是回归。
        现在改为从共享常量取，随口径自动跟随。
        """
        rt = _FakeRuntime()
        old = _frame(7)
        runner = PolicyRunner(rt, lambda: _frame(1), lambda: (0.0, 0.8), lambda: (0.0, 0.0),
                              dry_run=True, hz=50.0)
        # 手动塞一个「距今 HISTORY_FRAME_S」的历史帧（不写字面量，避免再次漂移）
        runner._push(time.perf_counter() - HISTORY_FRAME_S, old)
        runner.start()
        deadline = time.time() + 2.0
        while not rt.calls and time.time() < deadline:
            time.sleep(0.01)
        runner.stop()
        self.assertTrue(rt.calls)
        self.assertIs(rt.calls[0]['prev'], old)

    def test_stale_three_channel_block_keeps_action_dimension(self) -> None:
        """三通道 B3 过期时也必须生成 [vx, vz, omega] 零块。"""
        rt = _SlowThreeChannelRuntime()
        blocks = []
        runner = PolicyRunner(
            rt, lambda: _frame(1), lambda: (0.0, 0.0), lambda: (0.0, 0.0),
            on_block=lambda block, _t, _meta: blocks.append(block),
            dry_run=False, hz=50.0, stale_s=0.01,
        )
        runner.start()
        deadline = time.time() + 2.0
        while not blocks and time.time() < deadline:
            time.sleep(0.01)
        runner.stop()
        self.assertTrue(blocks, 'runner 没有产出过期动作块')
        self.assertTrue(all(len(row) == 3 for row in blocks[0]))
        self.assertTrue(all(value == 0.0 for row in blocks[0] for value in row))

    def test_guidance_provider_is_soft_input_and_is_logged(self) -> None:
        rt = _GuidanceRuntime()
        frames = []
        runner = PolicyRunner(
            rt, lambda: _frame(1), lambda: (9.0, 9.0), lambda: (0.0, 0.0),
            guidance_provider=lambda: {
                "valid": True,
                "waypoint": (0.2, 0.8),
                "path": [(0.2, 0.8), (0.4, 1.4)],
                "mode": 0,
                "confidence": 0.7,
                "pose_age_s": 0.1,
                "sigma_m": 0.3,
                "reason": "ok",
                "source": "astar_navmesh",
            },
            on_block=lambda _block, _t, meta: frames.append(meta),
            dry_run=False, hz=50.0,
        )
        runner.start()
        deadline = time.time() + 2.0
        while not rt.calls and time.time() < deadline:
            time.sleep(0.01)
        runner.stop()
        self.assertTrue(rt.calls)
        self.assertEqual(rt.calls[0]["wp"], (0.2, 0.8))
        self.assertEqual(rt.calls[0]["path"], [(0.2, 0.8), (0.4, 1.4)])
        self.assertTrue(frames)
        self.assertEqual(frames[0]["guide_confidence"], 0.7)

    def test_task_prompt_is_forwarded_as_scene_level_input(self) -> None:
        rt = _PromptRuntime()
        runner = PolicyRunner(
            rt, lambda: _frame(1), lambda: (0.0, 0.8), lambda: (0.0, 0.0),
            prompt_provider=lambda: "自由探索这个房间，看到障碍物就绕开",
            dry_run=True, hz=50.0,
        )
        runner.start()
        deadline = time.time() + 2.0
        while not rt.calls and time.time() < deadline:
            time.sleep(0.01)
        runner.stop()
        self.assertTrue(rt.calls)
        self.assertEqual(rt.calls[0]["task_prompt"], "自由探索这个房间，看到障碍物就绕开")


if __name__ == '__main__':
    unittest.main()
