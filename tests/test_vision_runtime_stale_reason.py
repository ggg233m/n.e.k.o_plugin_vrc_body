"""测试 VisionRuntime.latest_frame 的 stale_reason 分支。"""
from __future__ import annotations

import unittest
from typing import Any

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.vision import VisionRuntime
from neko_anyadance_body.backend.world_state import WorldStateStore


class _FakeFrameSource:
    """模拟采集源，可控制 window_minimized 状态。"""

    name = "test_source"

    def __init__(self, *, window_minimized: bool = False) -> None:
        self.closed = False
        self.window_minimized = window_minimized
        self.window_obscured = window_minimized

    def read(self):
        return b"fake_frame_data"

    def status(self) -> dict[str, Any]:
        return {
            "available": not self.closed,
            "name": self.name,
            "window_minimized": self.window_minimized,
            "window_obscured": self.window_obscured,
        }

    def close(self) -> None:
        self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class VisionRuntimeStaleReasonTests(unittest.TestCase):
    """测试 latest_frame 在窗口被遮挡/最小化时返回不同的 reason。"""

    def test_window_minimized_reason_when_source_reports_minimized(self) -> None:
        """源报 window_minimized=True 时,latest_frame 应返回 window_minimized reason。"""
        store = WorldStateStore()
        clock = _Clock()
        source = _FakeFrameSource(window_minimized=True)
        runtime = VisionRuntime(store=store, clock=clock)
        runtime.source = source

        # 模拟有缓存帧但已过期
        runtime._frame_cache = {
            "mime": "image/jpeg",
            "width": 640,
            "height": 480,
            "bytes": 1024,
            "data": b"fake_jpeg_data",
        }
        runtime._frame_cache_at = clock.now
        runtime._last_obscured_at = clock.now

        # 推进时间使帧过期
        clock.now += 5.0

        # 触发 latest_frame,max_age_ms=100 使其因过期而进入 stale 分支
        result = runtime.latest_frame(max_age_ms=100)

        self.assertFalse(result.get("available"))
        self.assertEqual(result["reason"], "window_minimized")

    def test_window_obscured_reason_when_source_reports_only_obscured(self) -> None:
        """源报 window_obscured=True 但 window_minimized=False 时,应返回 window_obscured。"""
        store = WorldStateStore()
        clock = _Clock()
        source = _FakeFrameSource(window_minimized=False)
        source.window_obscured = True
        runtime = VisionRuntime(store=store, clock=clock)
        runtime.source = source

        runtime._frame_cache = {
            "mime": "image/jpeg",
            "width": 640,
            "height": 480,
            "bytes": 1024,
            "data": b"fake_jpeg_data",
        }
        runtime._frame_cache_at = clock.now
        runtime._last_obscured_at = clock.now

        clock.now += 5.0

        result = runtime.latest_frame(max_age_ms=100)

        self.assertFalse(result.get("available"))
        self.assertEqual(result["reason"], "window_obscured")

    def test_fallback_to_window_obscured_when_source_status_fails(self) -> None:
        """源的 status() 抛异常时,应 fallback 到 window_obscured。"""
        store = WorldStateStore()
        clock = _Clock()

        class _FailingSource:
            name = "failing"

            def status(self):
                raise RuntimeError("status unavailable")

            def read(self):
                return b"data"

            def close(self):
                pass

        runtime = VisionRuntime(store=store, clock=clock)
        runtime.source = _FailingSource()

        runtime._frame_cache = {
            "mime": "image/jpeg",
            "width": 640,
            "height": 480,
            "bytes": 1024,
            "data": b"fake_jpeg_data",
        }
        runtime._frame_cache_at = clock.now
        runtime._last_obscured_at = clock.now

        clock.now += 5.0

        result = runtime.latest_frame(max_age_ms=100)

        self.assertFalse(result.get("available"))
        self.assertEqual(result["reason"], "window_obscured")


if __name__ == "__main__":
    unittest.main()
