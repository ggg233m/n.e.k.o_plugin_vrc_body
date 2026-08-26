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

    def test_secondary_monitor_region_is_local_to_matching_output(self) -> None:
        """第二屏窗口不能把虚拟桌面绝对坐标直接交给 DXcam。"""
        source = object.__new__(DxcamFrameSource)
        source._requested_region = {
            "left": 2476,
            "top": 155,
            "right": 3637,
            "bottom": 921,
        }
        source._region_origin = None
        original = vision._display_monitor_rects
        try:
            vision._display_monitor_rects = lambda: [
                (0, 0, 1920, 1080),
                (1920, 0, 4260, 1080),
            ]
            primary = source._region_for_candidate((0, None, "dxgi"))
            secondary = source._region_for_candidate((0, 1, "dxgi"))
        finally:
            vision._display_monitor_rects = original
        self.assertEqual(primary, (2476, 155, 3637, 921))
        self.assertEqual(secondary, (556, 155, 1717, 921))


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


class _StubCamera:
    """按脚本返回帧或抛错的假相机。"""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.calls = 0

    def grab(self, region=None):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        pass

    def release(self):
        pass


class DxcamSilentFailureTests(unittest.TestCase):
    """采集坏掉必须被报告出来，不能表现成「还没开始」。

    真机复现：窗口被拖出屏幕右边缘后 GetWindowRect 返回越界矩形，DXcam 的每个
    candidate 都抛 ``ValueError: Invalid Region``，但旧代码里
    ``_activate_candidate_locked`` 一构造出相机就把 ``_last_error`` 清空，于是
    ``available`` 恒为 True、``last_error`` 恒为 None，agent 从 vrc_vision_status
    看到的是 awaiting_first_frame——「还没开始」而不是「已经彻底坏了」。
    """

    def _source(self, *, specs, camera=None):
        source = object.__new__(DxcamFrameSource)
        source._lock = __import__("threading").Lock()
        source._camera = camera
        source._region = None
        source._closed = False
        source._frames = 0
        source._grabs_attempted = 0
        source._empty_grabs = 0
        source._last_error = None
        source._exhausted_error = None
        source._requested_device_idx = -1
        source._requested_output_idx = -1
        source._requested_backend = "auto"
        source._selected_device_idx = 0
        source._selected_output_idx = None
        source._selected_backend = "dxgi"
        source._candidate_specs = list(specs)
        source._candidate_pos = 0
        source._candidate_errors = {}
        source._dxcam = None
        source._winrt_available = False
        return source

    def test_all_candidates_failing_is_reported_as_unavailable(self) -> None:
        specs = [(0, None, "dxgi"), (0, None, "winrt")]
        source = self._source(specs=specs)
        # 两个 candidate 都留下错误，模拟越界区域下的轮换一圈。
        for spec in specs:
            source._candidate_errors[DxcamFrameSource._format_spec(spec)] = (
                "ValueError: Invalid Region: Region should be in 1920x1080"
            )
        # 即使新相机能构造成功，也不能把错误清掉。
        source._create_camera = lambda spec: "camera"
        source._activate_candidate_locked(0)
        status = source.status()
        self.assertFalse(
            status["available"],
            "所有 candidate 都失败过时报告 available=True 就是伪造健康状态",
        )
        self.assertIsNotNone(status["last_error"])
        self.assertIn("Invalid Region", status["last_error"])

    def test_partial_candidate_failure_still_recovers_cleanly(self) -> None:
        # 单个输出失败后切到另一个能用的输出属于正常回退，不该报错。
        specs = [(0, None, "dxgi"), (0, 1, "dxgi")]
        source = self._source(specs=specs)
        source._candidate_errors[DxcamFrameSource._format_spec(specs[0])] = "ValueError: boom"
        source._create_camera = lambda spec: "camera"
        source._activate_candidate_locked(1)
        self.assertTrue(source.status()["available"])
        self.assertIsNone(source.status()["last_error"])

    def test_empty_grabs_do_not_count_as_captured_frames(self) -> None:
        """``frames`` 必须只计真正产出的帧。

        DXcam 的 ``new_frame_only=True`` 在没有新帧时合法返回 None。旧代码无条件
        ``_frames += 1``，于是「相机在线但一帧不产」和正常采集在计数器上完全一样。
        """
        source = self._source(specs=[(0, None, "dxgi")], camera=_StubCamera([None]))
        for _ in range(3):
            self.assertIsNone(source.read())
        status = source.status()
        self.assertEqual(status["frames"], 0, "没拿到帧就不能计入 frames")
        self.assertEqual(status["grabs_attempted"], 3)
        self.assertEqual(status["empty_grabs"], 3)

    def test_sustained_empty_grabs_eventually_surface_an_error(self) -> None:
        source = self._source(specs=[(0, None, "dxgi")], camera=_StubCamera([None]))
        for _ in range(DxcamFrameSource._EMPTY_GRAB_LIMIT):
            source.read()
        status = source.status()
        self.assertIsNotNone(
            status["last_error"],
            "持续不产帧必须报错，沉默才是这里真正的 bug",
        )
        self.assertFalse(status["available"])

    def test_a_real_frame_clears_the_empty_grab_streak(self) -> None:
        camera = _StubCamera([None, None, "frame"])
        source = self._source(specs=[(0, None, "dxgi")], camera=camera)
        source.read()
        source.read()
        self.assertEqual(source.status()["empty_grabs"], 2)
        self.assertEqual(source.read(), "frame")
        status = source.status()
        self.assertEqual(status["empty_grabs"], 0)
        self.assertEqual(status["frames"], 1)
        self.assertTrue(status["available"])


class WindowRegionClampTests(unittest.TestCase):
    """越界窗口矩形必须被夹到虚拟桌面内，且夹取量要报出来。

    DXcam 对越界区域整块拒绝（``Invalid Region``），采集直接归零。夹取把它救回来，
    但采集区域一变，FOV→bearing_deg 的映射基准就跟着变——所以不能默默夹。
    """

    def test_clamped_region_reports_how_much_was_lost(self) -> None:
        clamped = _clamp_to_virtual_desktop(
            (881, 108, 2042, 874), virtual=(0, 0, 1920, 1080)
        )
        self.assertEqual(
            (clamped["left"], clamped["top"], clamped["right"], clamped["bottom"]),
            (881, 108, 1920, 874),
        )
        self.assertTrue(clamped["clamped"])
        self.assertEqual(clamped["clamped_px"]["right"], 122)

    def test_in_bounds_region_is_not_marked_clamped(self) -> None:
        clamped = _clamp_to_virtual_desktop(
            (755, 119, 1916, 885), virtual=(0, 0, 1920, 1080)
        )
        self.assertNotIn("clamped", clamped)
        self.assertNotIn("clamped_px", clamped)

    def test_clamp_metadata_survives_normalization_without_leaking(self) -> None:
        # 额外的键不能破坏 _normalize_region，也不能漏进采集后端的 region 元组。
        region = _normalize_region({
            "left": 881, "top": 108, "right": 1920, "bottom": 874,
            "clamped": True, "clamped_px": {"right": 122},
        })
        self.assertEqual(region["width"], 1039)
        self.assertEqual(
            sorted(region.keys()),
            ["bottom", "height", "left", "right", "top", "width"],
        )


def _clamp_to_virtual_desktop(rect, *, virtual):
    """复刻 ``find_window_region`` 的夹取算法，便于脱离 Win32 断言。

    真实函数要调 FindWindowW/GetSystemMetrics，在 CI 或无窗口时不可用；这里只
    验证算术，Win32 那一段由真机验证覆盖。
    """
    vl, vt, vr, vb = virtual
    left = max(vl, min(int(rect[0]), vr - 1))
    top = max(vt, min(int(rect[1]), vb - 1))
    right = max(left + 1, min(int(rect[2]), vr))
    bottom = max(top + 1, min(int(rect[3]), vb))
    result = {"left": left, "top": top, "right": right, "bottom": bottom}
    clipped = {
        "left": left - int(rect[0]),
        "top": top - int(rect[1]),
        "right": int(rect[2]) - right,
        "bottom": int(rect[3]) - bottom,
    }
    if any(clipped.values()):
        result["clamped"] = True
        result["clamped_px"] = clipped
    return result


class WindowTrackedFrameSourceTests(unittest.TestCase):
    def _build(self, rects, *, interval_s=5.0, visibility=None):
        """``rects`` 是每次解析依次返回的窗口矩形。

        ``visibility`` 默认关掉：不然测试会真去 ``FindWindowW("VRChat")``，
        结果取决于跑测试的机器上有没有开着 VRChat。
        """
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
            visibility=visibility,
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


class WindowOcclusionReportingTests(unittest.TestCase):
    """窗口被盖住时采集依旧「成功」，所以必须靠 status 把它报出来。"""

    def _build(self, probes, *, rect=None):
        rect = rect or {"left": 0, "top": 0, "right": 1280, "bottom": 720}
        clock = _Clock()
        pending = list(probes)
        last = {"value": {}}

        def visibility(_title):
            if pending:
                last["value"] = pending.pop(0)
            return last["value"]

        tracked = WindowTrackedFrameSource(
            title="VRChat",
            factory=_FakeSource,
            interval_s=5.0,
            clock=clock,
            resolver=lambda _title: dict(rect),
            visibility=visibility,
        )
        return tracked, clock

    def test_a_fully_visible_window_is_not_flagged(self) -> None:
        tracked, _clock = self._build([{"found": True, "minimized": False, "visible_ratio": 1.0}])
        status = tracked.status()
        self.assertFalse(status["window_obscured"])
        self.assertEqual(status["window_visible_ratio"], 1.0)
        self.assertNotIn("window_occluded_by", status)

    def test_a_minimized_window_is_flagged_even_though_capture_succeeds(self) -> None:
        """最小化是最刺眼的例子：矩形还在，采集还成功，画面已经完全不是游戏了。"""
        tracked, _clock = self._build([{"found": True, "minimized": True, "visible_ratio": 0.0}])
        status = tracked.status()
        self.assertTrue(status["window_minimized"])
        self.assertTrue(status["window_obscured"])
        self.assertTrue(status["window_found"], "窗口仍然存在，只是看不见——两件事不能混")

    def test_a_covered_window_names_the_window_on_top(self) -> None:
        tracked, _clock = self._build(
            [{"found": True, "minimized": False, "visible_ratio": 0.05, "occluded_by": "Discord"}]
        )
        status = tracked.status()
        self.assertTrue(status["window_obscured"])
        self.assertEqual(status["window_occluded_by"], "Discord")

    def test_a_notification_sized_overlap_is_not_flagged(self) -> None:
        # 任务栏、输入法候选框、Steam 弹窗都会盖掉一角，那不该报警。
        tracked, _clock = self._build(
            [{"found": True, "minimized": False, "visible_ratio": 0.93, "occluded_by": "Steam"}]
        )
        status = tracked.status()
        self.assertFalse(status["window_obscured"])
        self.assertEqual(status["window_occluded_by"], "Steam", "盖住谁照样要报，只是不算失效")

    def test_a_failed_probe_reports_unknown_rather_than_fully_covered(self) -> None:
        """探测坏掉不等于窗口被盖住。混为一谈就是喊狼来了。"""
        tracked, _clock = self._build([{}])
        status = tracked.status()
        self.assertIsNone(status["window_visible_ratio"])
        self.assertFalse(status["window_obscured"])

    def test_a_raising_probe_does_not_break_capture_or_forge_an_error(self) -> None:
        def visibility(_title):
            raise OSError("user32 unavailable")

        tracked = WindowTrackedFrameSource(
            title="VRChat",
            factory=_FakeSource,
            interval_s=5.0,
            clock=_Clock(),
            resolver=lambda _title: {"left": 0, "top": 0, "right": 8, "bottom": 8},
            visibility=visibility,
        )
        status = tracked.status()
        self.assertIsNotNone(tracked.read())
        self.assertIsNone(status["window_visible_ratio"])
        self.assertNotIn(
            "last_error", status, "可见度只是诊断信息，它的异常不该伪装成采集故障"
        )

    def test_visibility_changes_do_not_rebuild_the_capture_session(self) -> None:
        """Alt-Tab 一次就重建一次 DXGI 会话的话，切窗口会变成掉帧风暴。"""
        built: list[_FakeSource] = []
        clock = _Clock()
        probes = [
            {"found": True, "minimized": False, "visible_ratio": 1.0},
            {"found": True, "minimized": False, "visible_ratio": 0.0, "occluded_by": "Chrome"},
        ]

        def factory(region):
            source = _FakeSource(region)
            built.append(source)
            return source

        def visibility(_title):
            return probes.pop(0) if probes else {}

        tracked = WindowTrackedFrameSource(
            title="VRChat",
            factory=factory,
            interval_s=5.0,
            clock=clock,
            resolver=lambda _title: {"left": 0, "top": 0, "right": 1280, "bottom": 720},
            visibility=visibility,
        )
        clock.now += 10.0
        tracked.read()
        self.assertEqual(len(built), 1)
        self.assertEqual(tracked.status()["window_rebuilds"], 0)
        self.assertTrue(tracked.status()["window_obscured"])


class WindowVisibilityProbeTests(unittest.TestCase):
    def test_probe_never_raises_and_always_reports_found(self) -> None:
        """非 Windows、没窗口、Win32 报错，一律返回 found=False 而不是抛异常。"""
        result = vision.window_visibility("a window that does not exist — 3f9c1a")
        self.assertIn("found", result)
        self.assertFalse(result["found"])


if __name__ == "__main__":
    unittest.main()
