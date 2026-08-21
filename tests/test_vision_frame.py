"""给 agent 看画面的那条路径：编码、单槽缓存、过期拒绝。

刻意不依赖 Pillow / numpy——这套测试要能在没有任何可选视觉依赖的机器上跑。
``encode_frame_jpeg`` 对「已经实现了 save/size 的对象」是鸭子类型的，所以这里
用一个最小替身就能验到降采样、模式转换和质量钳位的判断逻辑本身。
"""

from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.vision import VisionRuntime, encode_frame_jpeg
from neko_anyadance_body.backend.world_state import WorldStateStore


class _FakeImage:
    """只实现 ``encode_frame_jpeg`` 用到的那几个成员的 PIL 替身。

    ``resize`` / ``convert`` 返回新实例（和 Pillow 一致），所以调用痕迹记在共享
    的 ``record`` 上，否则链式调用之后就查不到原对象上了。
    """

    def __init__(self, width: int, height: int, mode: str = "RGB", record: dict | None = None):
        self.size = (width, height)
        self.mode = mode
        self.record = record if record is not None else {"resize": [], "convert": [], "save": []}

    def resize(self, target, _resample=None):
        self.record["resize"].append(tuple(target))
        return _FakeImage(int(target[0]), int(target[1]), self.mode, self.record)

    def convert(self, mode: str):
        self.record["convert"].append(mode)
        return _FakeImage(self.size[0], self.size[1], mode, self.record)

    def save(self, buf, *, format: str, quality: int) -> None:
        self.record["save"].append({
            "format": format,
            "quality": quality,
            "size": self.size,
            "mode": self.mode,
        })
        buf.write(b"\xff\xd8" + bytes([min(255, max(0, quality))]) + b"\xff\xd9")


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class EncodeFrameJpegTests(unittest.TestCase):
    def test_bytes_pass_through_and_report_unknown_size(self) -> None:
        """已经编码好的帧不重编码；尺寸未知就说未知，不猜。"""
        payload = b"\xff\xd8already-encoded\xff\xd9"
        data, width, height = encode_frame_jpeg(payload, max_width=320)
        self.assertIs(data, payload)
        self.assertEqual((width, height), (0, 0))

    def test_wide_frames_are_downscaled_to_max_width(self) -> None:
        image = _FakeImage(1920, 1080)
        data, width, height = encode_frame_jpeg(image, max_width=960)
        self.assertEqual(image.record["resize"], [(960, 540)])
        # 返回的尺寸是降采样之后的，不是原始分辨率。
        self.assertEqual((width, height), (960, 540))
        self.assertEqual(image.record["save"][0]["size"], (960, 540))
        self.assertTrue(data.startswith(b"\xff\xd8"))

    def test_narrow_frames_are_never_upscaled(self) -> None:
        """放大只会凭空造像素，还让 token 更贵。"""
        image = _FakeImage(640, 360)
        _, width, height = encode_frame_jpeg(image, max_width=960)
        self.assertEqual(image.record["resize"], [])
        self.assertEqual((width, height), (640, 360))

    def test_max_width_zero_disables_downscaling(self) -> None:
        image = _FakeImage(1920, 1080)
        _, width, _ = encode_frame_jpeg(image, max_width=0)
        self.assertEqual(image.record["resize"], [])
        self.assertEqual(width, 1920)

    def test_unsupported_modes_are_converted_but_rgb_and_l_are_left_alone(self) -> None:
        rgba = _FakeImage(320, 240, mode="RGBA")
        encode_frame_jpeg(rgba, max_width=0)
        self.assertEqual(rgba.record["convert"], ["RGB"])
        self.assertEqual(rgba.record["save"][0]["mode"], "RGB")
        for mode in ("RGB", "L"):
            image = _FakeImage(320, 240, mode=mode)
            encode_frame_jpeg(image, max_width=0)
            self.assertEqual(image.record["convert"], [], mode)

    def test_quality_is_clamped_to_the_supported_band(self) -> None:
        low = _FakeImage(320, 240)
        encode_frame_jpeg(low, max_width=0, quality=1)
        self.assertEqual(low.record["save"][0]["quality"], 30)
        high = _FakeImage(320, 240)
        encode_frame_jpeg(high, max_width=0, quality=500)
        self.assertEqual(high.record["save"][0]["quality"], 95)

    def test_unencodable_frames_raise_valueerror(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be encoded"):
            encode_frame_jpeg(object(), max_width=0)

    def test_save_failures_are_reported_as_valueerror(self) -> None:
        class _Broken(_FakeImage):
            def save(self, buf, *, format: str, quality: int) -> None:
                raise OSError("no JPEG encoder")

        with self.assertRaisesRegex(ValueError, "cannot be encoded"):
            encode_frame_jpeg(_Broken(320, 240), max_width=0)


class FrameCacheTests(unittest.TestCase):
    def _runtime(self, clock: _Clock, **kwargs) -> VisionRuntime:
        return VisionRuntime(
            WorldStateStore(clock=clock),
            clock=clock,
            frame_cache_max_width=0,
            **kwargs,
        )

    def test_no_frame_yet_is_stated_explicitly(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock)
        result = runtime.latest_frame()
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "no_frame_cached")
        self.assertTrue(result["capture_active"])
        self.assertNotIn("data", result)

    def test_cached_frame_is_returned_within_the_age_limit(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(0.5)
        result = runtime.latest_frame(max_age_ms=3000)
        self.assertTrue(result["available"])
        self.assertEqual(result["mime"], "image/jpeg")
        self.assertEqual((result["width"], result["height"]), (640, 360))
        self.assertEqual(result["bytes"], len(result["data"]))
        self.assertAlmostEqual(result["age_ms"], 500.0, places=1)

    def test_stale_frames_are_refused_rather_than_returned(self) -> None:
        """过期的画面比没有画面更危险：agent 会拿它当现在。"""
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(4.0)
        result = runtime.latest_frame(max_age_ms=3000)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "frame_stale")
        self.assertNotIn("data", result)
        # 年龄仍要如实回报，调用方才知道差了多少。
        self.assertAlmostEqual(result["age_ms"], 4000.0, places=1)

    def test_capture_stopped_hides_and_clears_the_cache(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        runtime.set_capture_state(False, "manual_stop")
        stopped = runtime.latest_frame()
        self.assertFalse(stopped["available"])
        self.assertFalse(stopped["capture_active"])
        self.assertEqual(stopped["reason"], "manual_stop")
        # 重新开采集也不能把停机前的画面当成现在的。
        runtime.set_capture_state(True, "active")
        self.assertEqual(runtime.latest_frame()["reason"], "no_frame_cached")

    def test_caching_is_throttled_by_interval(self) -> None:
        """编码是给 agent 看的额外开销，不能每帧都做。"""
        clock = _Clock()
        runtime = self._runtime(clock, frame_cache_interval_s=1.0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(0.2)
        runtime._cache_frame(_FakeImage(1280, 720), clock.now)
        held = runtime.latest_frame()
        self.assertEqual(held["width"], 640)
        clock.advance(1.0)
        runtime._cache_frame(_FakeImage(1280, 720), clock.now)
        self.assertEqual(runtime.latest_frame()["width"], 1280)

    def test_encode_failures_degrade_instead_of_breaking_capture(self) -> None:
        """看不到图是降级，掉帧才是故障——编码异常绝不能冒到采集线程。"""
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(object(), clock.now)
        result = runtime.latest_frame()
        self.assertFalse(result["available"])
        self.assertIn("cannot be encoded", result["reason"])
        self.assertIn("cannot be encoded", runtime.status()["frame_cache"]["last_error"] or "")

    def test_a_good_frame_clears_a_previous_encode_error(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(object(), clock.now)
        clock.advance(2.0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        self.assertTrue(runtime.latest_frame()["available"])
        self.assertIsNone(runtime.status()["frame_cache"]["last_error"])

    def test_status_reports_cache_age_without_exposing_pixels(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(0.25)
        cache_status = runtime.status()["frame_cache"]
        self.assertTrue(cache_status["cached"])
        self.assertAlmostEqual(cache_status["age_ms"], 250.0, places=1)
        self.assertNotIn("data", cache_status)

    def test_frames_never_leak_into_world_state(self) -> None:
        """约束二：帧只喂理解，不产生实体也不产生事件。

        看修订号而不只是看列表空不空——空列表在这个场景下本来就会成立，
        推进了修订号才说明缓存那一步碰了世界状态。
        """
        clock = _Clock()
        runtime = self._runtime(clock)
        before = runtime.snapshot()["status"]["revision"]
        for _ in range(3):
            clock.advance(2.0)
            runtime._cache_frame(_FakeImage(640, 360), clock.now)
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["status"]["revision"], before)
        self.assertEqual(snapshot["entities"], [])
        self.assertEqual(snapshot["events"], [])
        # 但帧确实缓存下来了，否则上面三条断言只是在测「什么都没发生」。
        self.assertTrue(runtime.latest_frame()["available"])


class VisionFrameServiceTests(unittest.TestCase):
    """``BackendService.vision_frame`` 的行为，用一个替身运行时隔离出来测。"""

    class _StubService:
        """只借 ``BackendService.vision_frame`` 这一个方法，不启动真后端。"""

        def __init__(self, runtime: VisionRuntime) -> None:
            self.vision = runtime

    def _service(self, runtime: VisionRuntime):
        from neko_anyadance_body.backend.service import BackendService

        stub = self._StubService(runtime)
        stub.vision_frame = BackendService.vision_frame.__get__(stub, type(stub))
        return stub

    def test_frame_is_returned_as_base64_not_raw_bytes(self) -> None:
        import base64

        clock = _Clock()
        runtime = VisionRuntime(WorldStateStore(clock=clock), clock=clock, frame_cache_max_width=0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        result = self._service(runtime).vision_frame()
        self.assertTrue(result["available"])
        # JSON 走不了 bytes；原始字段必须换掉而不是并存。
        self.assertNotIn("data", result)
        self.assertEqual(len(base64.b64decode(result["data_base64"])), result["bytes"])

    def test_unavailable_frames_carry_no_payload(self) -> None:
        clock = _Clock()
        runtime = VisionRuntime(WorldStateStore(clock=clock), clock=clock)
        result = self._service(runtime).vision_frame()
        self.assertFalse(result["available"])
        self.assertNotIn("data", result)
        self.assertNotIn("data_base64", result)

    def test_zero_max_age_is_floored_instead_of_meaning_unbounded(self) -> None:
        """运行时里 0 表示不限龄；模型那侧写 0 只会是「我要最新的」。"""
        clock = _Clock()
        runtime = VisionRuntime(WorldStateStore(clock=clock), clock=clock, frame_cache_max_width=0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(10.0)
        service = self._service(runtime)
        self.assertEqual(service.vision_frame(max_age_ms=0)["reason"], "frame_stale")
        # 逃生口仍在运行时那一层，只是模型够不到。
        self.assertTrue(runtime.latest_frame(max_age_ms=0)["available"])

    def test_malformed_max_age_falls_back_to_the_default(self) -> None:
        clock = _Clock()
        runtime = VisionRuntime(WorldStateStore(clock=clock), clock=clock, frame_cache_max_width=0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(2.0)
        service = self._service(runtime)
        for bogus in (None, "soon", float("nan"), object()):
            self.assertTrue(service.vision_frame(max_age_ms=bogus)["available"], bogus)

    def test_absurd_max_age_is_capped(self) -> None:
        clock = _Clock()
        runtime = VisionRuntime(WorldStateStore(clock=clock), clock=clock, frame_cache_max_width=0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(45.0)
        service = self._service(runtime)
        self.assertEqual(service.vision_frame(max_age_ms=10**9)["reason"], "frame_stale")


if __name__ == "__main__":
    unittest.main()
