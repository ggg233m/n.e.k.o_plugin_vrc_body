from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import vision
from neko_anyadance_body.backend.vision import (
    DxcamFrameSource,
    WgcWindowFrameSource,
    WindowTrackedFrameSource,
    _normalize_region,
    optional_dependency_status,
)


class _FakeSource:
    """记录自己是用哪块区域构造的，并允许断言 close 被调用过。"""

    name = "fake"

    def __init__(self, region, *, window_minimized: bool = False) -> None:
        self.region = dict(region) if region else None
        self.closed = False
        self.reads = 0
        self.window_minimized = window_minimized

    def read(self):
        self.reads += 1
        return f"frame@{self.region}"

    def status(self):
        return {
            "available": not self.closed,
            "name": self.name,
            "window_obscured": self.window_minimized,
            "window_minimized": self.window_minimized,
        }

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
    ``available`` 恒为 True、``last_error`` 恒为 None，agent 从
    ``body_status(include=["vision"])`` 看到的是 awaiting_first_frame
    ——「还没开始」而不是「已经彻底坏了」。
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

        def factory(region, **kwargs):
            source = _FakeSource(region, **kwargs)
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


class _FakeWgcSession:
    """替身 WgcSession：不碰 D3D11，只按脚本交还帧。"""

    def __init__(self, hwnd, *, capture_cursor=False, frames=None, fail_on_read=None) -> None:
        self.hwnd = hwnd
        self.capture_cursor = capture_cursor
        self.size = (1922, 1041)
        self.closed = 0
        self._frames = list(frames if frames is not None else ["frame"])
        self._fail_on_read = fail_on_read
        self.reads = 0

    def read(self):
        self.reads += 1
        if self._fail_on_read is not None and self.reads >= self._fail_on_read:
            raise RuntimeError("device removed")
        if not self._frames:
            return None
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed += 1


class WgcWindowFrameSourceTests(unittest.TestCase):
    """按窗口捕获：遮挡不再是失效条件，但窗口缺失/最小化/会话作废仍然是。"""

    def _patch(self, *, supported=True, session_factory=None):
        """把 ``backend.wgc_capture`` 换成替身模块。

        采集源是在 ``_open`` 里 ``from .wgc_capture import ...`` 的，所以只能在
        ``sys.modules`` 这一层拦——真模块会去建 D3D11 设备，CI 上没有。
        """
        import sys
        import types

        created: list[_FakeWgcSession] = []

        def build(hwnd, *, capture_cursor=False):
            session = (session_factory or _FakeWgcSession)(hwnd, capture_cursor=capture_cursor)
            created.append(session)
            return session

        module = types.ModuleType("neko_anyadance_body.backend.wgc_capture")
        module.WgcSession = build
        module.wgc_supported = lambda: supported
        key = "neko_anyadance_body.backend.wgc_capture"
        previous = sys.modules.get(key)
        sys.modules[key] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__(key, previous)
            if previous is not None
            else sys.modules.pop(key, None)
        )
        return created

    def test_it_captures_without_a_window_rect(self) -> None:
        """这条路径不需要屏幕坐标：捕获项自己就是那个窗口。"""
        created = self._patch()
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        self.assertEqual(source.read(), "frame")
        self.assertEqual(created[0].hwnd, 0x1234)
        status = source.status()
        self.assertTrue(status["available"])
        self.assertEqual(status["name"], "wgc_window")
        self.assertEqual(status["capture_size"], {"width": 1922, "height": 1041})
        self.assertEqual(status["frames"], 1)
        self.assertIsNone(status["last_error"])

    def test_occlusion_is_not_a_failure_mode_on_this_path(self) -> None:
        """整个采集源里没有可见比例的概念——遮挡在 DWM 按窗口合成下不成立。

        这是本采集源存在的全部理由，所以值得单独钉住：``window_obscured`` 只跟
        最小化走，不会因为别的窗口压在上面而变 True。
        """
        self._patch()
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234, minimized=lambda _h: False)
        self.addCleanup(source.close)
        source.read()
        status = source.status()
        self.assertFalse(status["window_obscured"])
        self.assertNotIn("window_visible_ratio", status)
        self.assertNotIn("window_occluded_by", status)

    def test_a_missing_window_is_reported_not_raised(self) -> None:
        """VRChat 还没启动不该让后端起不来。"""
        self._patch()
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: None)
        self.addCleanup(source.close)
        status = source.status()
        self.assertFalse(status["available"])
        self.assertFalse(status["window_found"])
        self.assertIn("window not found", status["last_error"])
        self.assertIsNone(source.read())

    def test_a_window_that_appears_later_is_picked_up_without_a_restart(self) -> None:
        """先起后端后起游戏是常态，不能要求用户重启插件。"""
        self._patch()
        handles = [None, None, 0x99]

        def resolver(_title):
            return handles.pop(0) if handles else 0x99

        source = WgcWindowFrameSource(title="VRChat", resolver=resolver)
        self.addCleanup(source.close)
        self.assertIsNone(source.read())
        self.assertEqual(source.read(), "frame")
        self.assertTrue(source.status()["available"])

    def test_a_failed_session_still_reports_the_window_as_found(self) -> None:
        """窗口在、但 WGC 建不起来，是和「游戏没开」完全不同的故障。"""

        def explode(hwnd, *, capture_cursor=False):
            raise RuntimeError("D3D11CreateDevice failed")

        self._patch(session_factory=explode)
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        status = source.status()
        self.assertFalse(status["available"])
        self.assertTrue(status["window_found"], "窗口找到了，问题出在捕获会话上")
        self.assertIn("D3D11CreateDevice failed", status["last_error"])

    def test_an_unsupported_platform_degrades_instead_of_crashing(self) -> None:
        self._patch(supported=False)
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        self.assertIsNone(source.read())
        self.assertIn("unavailable", source.status()["last_error"])

    def test_a_minimized_window_yields_no_frame_rather_than_a_frozen_one(self) -> None:
        """最小化时 DWM 不再合成，帧池只会交还旧帧。冻帧会被下游当成「现在」。"""
        created = self._patch()
        minimized = {"value": False}
        source = WgcWindowFrameSource(
            title="VRChat",
            resolver=lambda _t: 0x1234,
            minimized=lambda _h: minimized["value"],
        )
        self.addCleanup(source.close)
        source.read()
        minimized["value"] = True
        self.assertIsNone(source.read())
        self.assertEqual(created[0].reads, 1, "最小化时根本不该去读帧池")
        status = source.status()
        self.assertTrue(status["window_minimized"])
        self.assertTrue(status["window_obscured"], "最小化是这条路径上唯一剩下的失效模式")

    def test_a_probe_that_explodes_is_treated_as_not_minimized(self) -> None:
        """探测失败宁可错判成没最小化：丢掉本可用的画面代价更大。"""
        self._patch()

        def explode(_hwnd):
            raise OSError("user32 exploded")

        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234, minimized=explode)
        self.addCleanup(source.close)
        self.assertEqual(source.read(), "frame")
        self.assertFalse(source.status()["window_minimized"])

    def test_no_new_frame_is_not_an_error(self) -> None:
        """帧池没新帧时返回 None，不该被记成故障。"""
        self._patch(
            session_factory=lambda hwnd, **kw: _FakeWgcSession(hwnd, frames=[], **kw)
        )
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        self.assertIsNone(source.read())
        status = source.status()
        self.assertTrue(status["available"], "空帧不代表会话作废")
        self.assertEqual(status["empty_reads"], 1)
        self.assertIsNone(status["last_error"])

    def test_a_window_resize_does_not_freeze_capture_forever(self) -> None:
        """改分辨率后 WgcSession 因尺寸不符一直返回 None 且不抛异常。

        光靠异常路径永远等不到重建：画面会永久卡死，而 status 还显示
        available=True、last_error=None——最难查的那种故障。所以连续空帧到达
        上限必须主动重建。
        """
        created = self._patch(
            session_factory=lambda hwnd, **kw: _FakeWgcSession(hwnd, frames=[], **kw)
        )
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        for _ in range(200):
            source.read()
        self.assertGreater(len(created), 1, "持续空帧必须触发重建，否则画面永久卡死")
        status = source.status()
        self.assertGreater(status["rebuilds"], 0)

    def test_a_brief_gap_in_frames_does_not_rebuild(self) -> None:
        """空帧本身是正常的，重建实测 226 ms 比帧预算还贵，不能抖一下就重建。"""
        created = self._patch(
            session_factory=lambda hwnd, **kw: _FakeWgcSession(
                hwnd, frames=[None, None, "frame"], **kw
            )
        )
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        for _ in range(3):
            source.read()
        self.assertEqual(len(created), 1, "短暂空帧不该重建会话")
        self.assertEqual(source.status()["rebuilds"], 0)

    def test_a_failing_session_build_backs_off_instead_of_retrying_every_frame(self) -> None:
        """建会话失败实测 5.9 ms、成功 226 ms，都够得上 100 ms 的帧预算。

        VRChat 没开着的整段时间里每帧都重试，会把采集线程一直卡在建 D3D 设备上。
        """
        attempts = []

        def explode(hwnd, *, capture_cursor=False):
            attempts.append(1)
            raise RuntimeError("D3D11CreateDevice failed")

        self._patch(session_factory=explode)
        clock = _Clock()
        source = WgcWindowFrameSource(
            title="VRChat",
            resolver=lambda _t: 0x1234,
            clock=clock,
        )
        self.addCleanup(source.close)
        attempts.clear()
        for _ in range(10):
            source.read()
        self.assertEqual(attempts, [], "退避期内不该反复建会话")
        self.assertTrue(source.status()["reopen_backoff"])
        clock.now += 2.0
        source.read()
        self.assertEqual(len(attempts), 1, "退避到期后必须再试一次")

    def test_a_missing_window_retries_every_read_without_backoff(self) -> None:
        """找不到窗口只花 0.2 ms 的 FindWindow，退避反而会拖慢游戏启动后的恢复。"""
        self._patch()
        looked_up = []

        def resolver(_title):
            looked_up.append(1)
            return None

        source = WgcWindowFrameSource(
            title="VRChat", resolver=resolver, clock=_Clock()
        )
        self.addCleanup(source.close)
        looked_up.clear()
        for _ in range(5):
            source.read()
        self.assertEqual(len(looked_up), 5, "窗口不存在时应每次都便宜地重试")
        self.assertFalse(source.status()["reopen_backoff"])

    def test_a_dead_session_is_rebuilt_on_the_next_read(self) -> None:
        """窗口改分辨率或设备丢失后会话就作废了，必须自己重建。"""
        created = self._patch(
            session_factory=lambda hwnd, **kw: _FakeWgcSession(hwnd, fail_on_read=1, **kw)
        )
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        self.assertIsNone(source.read())
        self.assertEqual(len(created), 2, "读失败后应重建会话")
        self.assertEqual(created[0].closed, 1, "旧会话必须关掉，否则句柄泄漏")
        status = source.status()
        self.assertEqual(status["rebuilds"], 1)
        # 重建成功后 last_error 归零：它描述的是当下能不能用，不是历史。抖动的
        # 痕迹由 rebuilds 计数保留，那才是长期可观测的信号。
        self.assertTrue(status["available"])
        self.assertIsNone(status["last_error"])

    def test_a_rebuild_that_also_fails_keeps_the_error_visible(self) -> None:
        """窗口真没了的时候，重建会再失败一次，那条错误必须留在 status 上。"""
        states = {"fail_open": False}

        def factory(hwnd, **kw):
            if states["fail_open"]:
                raise RuntimeError("window vanished")
            states["fail_open"] = True
            return _FakeWgcSession(hwnd, fail_on_read=1, **kw)

        self._patch(session_factory=factory)
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        self.addCleanup(source.close)
        self.assertIsNone(source.read())
        status = source.status()
        self.assertFalse(status["available"])
        self.assertEqual(status["rebuilds"], 1)
        self.assertIn("window vanished", status["last_error"])

    def test_close_is_idempotent_and_stops_serving_frames(self) -> None:
        created = self._patch()
        source = WgcWindowFrameSource(title="VRChat", resolver=lambda _t: 0x1234)
        source.read()
        source.close()
        source.close()
        self.assertEqual(created[0].closed, 1, "重复 close 不该重复释放 COM 对象")
        self.assertIsNone(source.read(), "关闭后不能再返回帧")
        self.assertFalse(source.status()["available"])


class WindowVisibilityProbeTests(unittest.TestCase):
    def test_probe_never_raises_and_always_reports_found(self) -> None:
        """非 Windows、没窗口、Win32 报错，一律返回 found=False 而不是抛异常。"""
        result = vision.window_visibility("a window that does not exist — 3f9c1a")
        self.assertIn("found", result)
        self.assertFalse(result["found"])


if __name__ == "__main__":
    unittest.main()
