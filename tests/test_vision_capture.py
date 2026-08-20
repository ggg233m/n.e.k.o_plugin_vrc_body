from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import vision
from neko_anyadance_body.backend.vision import (
    DxcamFrameSource,
    WindowTrackedFrameSource,
    _normalize_region,
    optional_dependency_status,
)


class _FakeSource:
    """记录自己是用哪块区域构造的，并允许断言 close 被调用过。"""

    name = "fake"

    def __init__(self, region) -> None:
        self.region = dict(region) if region else None
        self.closed = False
        self.reads = 0

    def read(self):
        self.reads += 1
        return f"frame@{self.region}"

    def status(self):
        return {"available": not self.closed, "name": self.name}

    def close(self) -> None:
        self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeDxcam:
    def enum_dxgi_adapters(self):
        return [object()]

    def output_info(self):
        return "Device[0] Output[0]"


class _LegacyDxcam:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if "backend" in kwargs:
            raise TypeError("backend is not supported by this old DXcam")
        return "legacy-camera"


class VisionCaptureTests(unittest.TestCase):
    def test_optional_dependency_status_reports_winrt_capability(self) -> None:
        status = optional_dependency_status()
        for key in ("winrt", "winrt_graphics_capture", "dxcam_winrt"):
            self.assertIn(key, status)
            self.assertIsInstance(status[key], bool)
        self.assertEqual(
            status["dxcam_winrt"],
            status["dxcam"] and status["winrt_graphics_capture"],
        )

    def test_auto_candidates_include_winrt_only_when_projection_is_available(self) -> None:
        source = object.__new__(DxcamFrameSource)
        source._requested_device_idx = -1
        source._requested_output_idx = -1
        source._requested_backend = "auto"
        source._winrt_available = True
        candidates = source._build_candidates(_FakeDxcam())
        self.assertIn((0, None, "dxgi"), candidates)
        self.assertIn((0, None, "winrt"), candidates)

        source._winrt_available = False
        candidates_without_winrt = source._build_candidates(_FakeDxcam())
        self.assertTrue(candidates_without_winrt)
        self.assertTrue(all(item[2] == "dxgi" for item in candidates_without_winrt))

    def test_explicit_winrt_does_not_silently_fallback_to_legacy_dxgi(self) -> None:
        source = object.__new__(DxcamFrameSource)
        source._dxcam = _LegacyDxcam()
        with self.assertRaises(TypeError):
            source._create_camera((0, None, "winrt"))
        self.assertEqual(len(source._dxcam.calls), 1)
        self.assertEqual(source._dxcam.calls[0]["backend"], "winrt")

    def test_legacy_dxcam_can_still_use_dxgi_fallback(self) -> None:
        source = object.__new__(DxcamFrameSource)
        source._dxcam = _LegacyDxcam()
        self.assertEqual(source._create_camera((0, None, "dxgi")), "legacy-camera")
        self.assertEqual(len(source._dxcam.calls), 2)
        self.assertNotIn("backend", source._dxcam.calls[-1])


class RegionNormalizationTests(unittest.TestCase):
    def test_window_rect_form_gains_width_and_height(self) -> None:
        """``find_window_region`` 返回 left/top/right/bottom，MSS 只认 width/height。

        MSS 逐键覆盖监视器字典，缺 width/height 时尺寸就留在整块显示器上：窗口
        裁剪只挪了原点，画面仍是全屏。补齐两种写法后两个采集器才真的裁同一块。
        """
        region = _normalize_region({"left": 100, "top": 50, "right": 1380, "bottom": 770})
        self.assertEqual(region["width"], 1280)
        self.assertEqual(region["height"], 720)
        self.assertEqual((region["left"], region["top"]), (100, 50))

    def test_size_form_gains_right_and_bottom(self) -> None:
        # 反方向同理：DXcam 读 right/bottom，缺键时旧代码补 0 直接抛 ValueError。
        region = _normalize_region({"left": 100, "top": 50, "width": 1280, "height": 720})
        self.assertEqual((region["right"], region["bottom"]), (1380, 770))

    def test_degenerate_and_missing_bounds_are_rejected_loudly(self) -> None:
        for bad in (
            {"left": 10, "top": 10, "right": 10, "bottom": 100},
            {"left": 10, "top": 10, "right": 100, "bottom": 10},
            {"left": 10, "top": 10},
            {"right": 100, "bottom": 100},
        ):
            with self.subTest(region=bad):
                with self.assertRaises(ValueError):
                    _normalize_region(bad)

    def test_none_stays_none(self) -> None:
        self.assertIsNone(_normalize_region(None))


class WindowTrackedFrameSourceTests(unittest.TestCase):
    def _build(self, rects, *, interval_s=5.0):
        """``rects`` 是每次解析依次返回的窗口矩形。"""
        clock = _Clock()
        pending = list(rects)
        built: list[_FakeSource] = []

        def resolver(_title):
            return pending.pop(0) if pending else None

        def factory(region):
            source = _FakeSource(region)
            built.append(source)
            return source

        tracked = WindowTrackedFrameSource(
            title="VRChat",
            factory=factory,
            interval_s=interval_s,
            clock=clock,
            resolver=resolver,
        )
        return tracked, clock, built

    def test_moved_window_is_re_resolved_after_the_interval(self) -> None:
        """窗口矩形只在启动时解析一次，窗口一被拖动就永远抓错位置。

        DXcam/MSS 都在构造时把区域固定下来，没有改区域的接口，所以过期的矩形
        不会自愈：采集会一直送回旧坐标那块画面，检测器看到的是桌面而不是游戏。
        """
        first = {"left": 0, "top": 0, "right": 1280, "bottom": 720}
        second = {"left": 400, "top": 200, "right": 1680, "bottom": 920}
        tracked, clock, built = self._build([first, second])
        self.assertEqual(built[0].region["left"], 0)

        clock.now += 10.0
        tracked.read()
        self.assertEqual(len(built), 2)
        self.assertEqual(built[1].region["left"], 400)
        self.assertTrue(built[0].closed, "旧采集源必须先关：DXGI 不允许同输出并存两个复制会话")
        self.assertEqual(tracked.status()["window_rebuilds"], 1)

    def test_unchanged_rect_does_not_rebuild_the_source(self) -> None:
        # 重建 DXGI 复制会话是重操作，TTL 到点只该解析坐标，不该无条件重建。
        rect = {"left": 0, "top": 0, "right": 1280, "bottom": 720}
        tracked, clock, built = self._build([rect, dict(rect), dict(rect)])
        for _ in range(2):
            clock.now += 10.0
            tracked.read()
        self.assertEqual(len(built), 1)
        self.assertEqual(tracked.status()["window_rebuilds"], 0)

    def test_reads_within_the_interval_do_not_re_resolve(self) -> None:
        rect = {"left": 0, "top": 0, "right": 1280, "bottom": 720}
        moved = {"left": 9, "top": 9, "right": 1289, "bottom": 729}
        tracked, clock, built = self._build([rect, moved], interval_s=5.0)
        clock.now += 1.0
        tracked.read()
        self.assertEqual(len(built), 1, "TTL 未到就重新解析会让 10 Hz 采集每帧调一次 FindWindow")

    def test_vanished_window_keeps_the_last_known_rect(self) -> None:
        """窗口暂时消失时保留旧矩形，而不是回退全屏。

        最小化、切桌面、Alt-Tab 都会让 FindWindow 短暂失败。此时改抓全屏会把
        桌面内容喂进检测器，比暂时抓一块过期区域危险得多。
        """
        rect = {"left": 400, "top": 200, "right": 1680, "bottom": 920}
        tracked, clock, built = self._build([rect])  # 之后的解析都返回 None
        clock.now += 10.0
        tracked.read()
        self.assertEqual(len(built), 1)
        status = tracked.status()
        self.assertFalse(status["window_found"])
        self.assertEqual(status["window_region"]["left"], 400)

    def test_zero_interval_pins_the_startup_rect(self) -> None:
        rect = {"left": 0, "top": 0, "right": 1280, "bottom": 720}
        moved = {"left": 400, "top": 200, "right": 1680, "bottom": 920}
        tracked, clock, built = self._build([rect, moved], interval_s=0.0)
        clock.now += 600.0
        tracked.read()
        self.assertEqual(len(built), 1)

    def test_close_releases_the_inner_source_and_stops_reading(self) -> None:
        rect = {"left": 0, "top": 0, "right": 1280, "bottom": 720}
        tracked, _clock, built = self._build([rect])
        tracked.close()
        self.assertTrue(built[0].closed)
        self.assertIsNone(tracked.read(), "关闭后不能再返回帧，否则生命周期门会被重新打开")


if __name__ == "__main__":
    unittest.main()
