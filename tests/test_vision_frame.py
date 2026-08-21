"""给 agent 看画面的那条路径：编码、单槽缓存、过期拒绝。

刻意不依赖 Pillow / numpy——这套测试要能在没有任何可选视觉依赖的机器上跑。
``encode_frame_jpeg`` 对「已经实现了 save/size 的对象」是鸭子类型的，所以这里
用一个最小替身就能验到降采样、模式转换和质量钳位的判断逻辑本身。
"""

from __future__ import annotations

import unittest
from unittest import mock

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.vision import (
    VisionRuntime,
    draw_detection_overlay,
    encode_frame_jpeg,
    overlay_boxes_geometry,
)
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


class OverlayGeometryTests(unittest.TestCase):
    """归一化 bbox → 像素矩形。纯算术，刻意不碰 Pillow。"""

    @staticmethod
    def _box(bbox, **extra):
        item = {"id": "e1", "label": "person", "confidence": 0.5, "bbox": bbox}
        item.update(extra)
        return item

    def test_normalized_bbox_maps_to_pixels(self) -> None:
        [box] = overlay_boxes_geometry([self._box([0.25, 0.5, 0.75, 1.0])], width=960, height=640)
        self.assertEqual(box["rect"], (240, 320, 720, 640))
        self.assertEqual(box["label"], "person")
        self.assertEqual(box["confidence"], 0.5)

    def test_edge_touching_boxes_survive_and_fill_the_canvas(self) -> None:
        """贴边不是越界；0..1 整幅框必须画得出来，否则最近的目标反而没框。"""
        [box] = overlay_boxes_geometry([self._box([0.0, 0.0, 1.0, 1.0])], width=100, height=80)
        self.assertEqual(box["rect"], (0, 0, 100, 80))

    def test_out_of_range_bbox_is_clamped_not_dropped(self) -> None:
        [box] = overlay_boxes_geometry([self._box([-0.4, -1.0, 1.9, 2.5])], width=100, height=80)
        self.assertEqual(box["rect"], (0, 0, 100, 80))

    def test_boxes_entirely_outside_the_canvas_are_dropped(self) -> None:
        """钳完塌成一条边的框不画：一条线会被读成「这里检测到了东西」。"""
        self.assertEqual(overlay_boxes_geometry([self._box([1.2, 1.2, 1.8, 1.9])], width=100, height=80), [])
        self.assertEqual(overlay_boxes_geometry([self._box([-0.9, -0.8, -0.2, -0.1])], width=100, height=80), [])

    def test_degenerate_and_inverted_boxes_are_dropped(self) -> None:
        for bbox in ([0.5, 0.5, 0.5, 0.5], [0.5, 0.2, 0.5, 0.9], [0.8, 0.2, 0.3, 0.9]):
            self.assertEqual(overlay_boxes_geometry([self._box(bbox)], width=100, height=80), [], bbox)

    def test_sub_pixel_boxes_are_dropped_rather_than_rounded_up(self) -> None:
        self.assertEqual(overlay_boxes_geometry([self._box([0.5, 0.5, 0.5001, 0.5001])], width=100, height=80), [])

    def test_non_finite_and_malformed_entries_are_skipped(self) -> None:
        nan = float("nan")
        boxes = [
            self._box([nan, 0.1, 0.9, 0.9]),
            self._box([0.1, 0.1, float("inf"), 0.9]),
            self._box("not-a-bbox"),
            self._box([0.1, 0.2]),
            self._box(["a", "b", "c", "d"]),
            "not-a-mapping",
            self._box([0.1, 0.1, 0.9, 0.9]),
        ]
        self.assertEqual(len(overlay_boxes_geometry(boxes, width=100, height=80)), 1)

    def test_empty_and_absent_inputs_are_not_errors(self) -> None:
        for boxes in ([], None, ()):
            self.assertEqual(overlay_boxes_geometry(boxes, width=100, height=80), [])

    def test_zero_or_bogus_canvas_yields_nothing(self) -> None:
        box = [self._box([0.1, 0.1, 0.9, 0.9])]
        for width, height in ((0, 80), (100, 0), (-5, 80), ("wide", 80)):
            self.assertEqual(overlay_boxes_geometry(box, width=width, height=height), [], (width, height))

    def test_clipped_and_bearing_are_carried_through_for_labelling(self) -> None:
        """贴边的表观高度已饱和，不能再当距离用——这个事实必须一路带到图上。"""
        [box] = overlay_boxes_geometry(
            [self._box([0.1, 0.0, 0.9, 0.999],
                       attributes={"apparent_height_clipped": True, "bearing_deg": -28.8})],
            width=100, height=80,
        )
        self.assertTrue(box["clipped"])
        self.assertEqual(box["bearing_deg"], -28.8)

    def test_missing_attributes_do_not_claim_clipping(self) -> None:
        for attributes in (None, {}, "nope"):
            [box] = overlay_boxes_geometry(
                [self._box([0.1, 0.1, 0.9, 0.9], attributes=attributes)], width=100, height=80
            )
            self.assertFalse(box["clipped"], attributes)
            self.assertIsNone(box["bearing_deg"], attributes)

    def test_unusable_confidence_becomes_zero_rather_than_raising(self) -> None:
        [box] = overlay_boxes_geometry(
            [self._box([0.1, 0.1, 0.9, 0.9], confidence="high")], width=100, height=80
        )
        self.assertEqual(box["confidence"], 0.0)


class OverlayFrameTests(unittest.TestCase):
    """``latest_frame(overlay=True)`` 的行为：报错位、不污染缓存、缺依赖时降级。"""

    def _runtime(self, clock: _Clock, **kwargs) -> VisionRuntime:
        return VisionRuntime(
            WorldStateStore(clock=clock),
            clock=clock,
            frame_cache_max_width=0,
            **kwargs,
        )

    @staticmethod
    def _entity(entity_id: str, bbox, **attributes):
        return {
            "id": entity_id,
            "label": "person",
            "confidence": 0.8,
            "bbox": list(bbox),
            "attributes": dict(attributes),
        }

    def test_overlay_is_opt_in_and_absent_by_default(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        self.assertNotIn("overlay", runtime.latest_frame())

    def test_empty_world_returns_the_frame_unchanged_not_an_error(self) -> None:
        """没有实体不是故障：给原图、说 0 个框，别让看图这条路整体失败。"""
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        plain = runtime.latest_frame()["data"]
        result = runtime.latest_frame(overlay=True)
        self.assertTrue(result["available"])
        self.assertEqual(result["overlay"]["entities_available"], 0)
        self.assertEqual(result["overlay"].get("boxes_drawn", 0), 0)
        if result["overlay"]["drawn"]:
            # 真有 Pillow 时会重编码一遍，字节可以不同，但尺寸必须还是那一张。
            self.assertEqual(result["bytes"], len(result["data"]))
        else:
            self.assertEqual(result["data"], plain)

    def test_overlay_never_mutates_the_frame_cache(self) -> None:
        """缓存里必须留原始像素：唤醒推送用的是同一份，烧了框就再也还原不回来。"""
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        runtime.store.ingest([self._entity("track:1", [0.2, 0.2, 0.6, 0.9])], observed_at=clock.now)
        before = runtime.latest_frame()["data"]
        runtime.latest_frame(overlay=True)
        self.assertEqual(runtime.latest_frame()["data"], before)

    def test_skew_is_computed_from_frame_age_and_world_age(self) -> None:
        """一秒前的像素配现在的框，等于伪造位置。错位量必须报出来。"""
        clock = _Clock()
        runtime = self._runtime(clock, frame_cache_interval_s=1.0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(0.9)
        runtime.store.ingest([self._entity("track:1", [0.2, 0.2, 0.6, 0.9])], observed_at=clock.now)
        clock.advance(0.1)
        overlay = runtime.latest_frame(overlay=True)["overlay"]
        self.assertAlmostEqual(overlay["frame_age_ms"], 1000.0, places=0)
        self.assertAlmostEqual(overlay["world_age_ms"], 100.0, places=0)
        self.assertAlmostEqual(overlay["skew_ms"], 900.0, places=0)

    def test_large_skew_is_flagged_so_it_cannot_be_read_as_simultaneous(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock, frame_cache_interval_s=1.0)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        clock.advance(2.0)
        runtime.store.ingest([self._entity("track:1", [0.2, 0.2, 0.6, 0.9])], observed_at=clock.now)
        overlay = runtime.latest_frame(max_age_ms=5000, overlay=True)["overlay"]
        self.assertTrue(overlay["skew_warning"])

    def test_fresh_frame_and_fresh_world_are_not_flagged(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock, frame_cache_interval_s=1.0)
        runtime.store.ingest([self._entity("track:1", [0.2, 0.2, 0.6, 0.9])], observed_at=clock.now)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        overlay = runtime.latest_frame(overlay=True)["overlay"]
        self.assertNotIn("skew_warning", overlay)

    def test_unavailable_frames_are_never_overlaid(self) -> None:
        """没有画面时叠框无从谈起，也不能因此多出一个 overlay 字段。"""
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime.store.ingest([self._entity("track:1", [0.2, 0.2, 0.6, 0.9])], observed_at=clock.now)
        result = runtime.latest_frame(overlay=True)
        self.assertFalse(result["available"])
        self.assertNotIn("overlay", result)

    def test_missing_pillow_degrades_to_the_original_frame(self) -> None:
        """看不到框是降级，掉帧才是故障：给原图 + 说明原因，而不是报 unavailable。"""
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        runtime.store.ingest([self._entity("track:1", [0.2, 0.2, 0.6, 0.9])], observed_at=clock.now)
        plain = runtime.latest_frame()["data"]
        with mock.patch(
            "neko_anyadance_body.backend.vision.draw_detection_overlay",
            side_effect=ValueError("overlay requires Pillow: no module named PIL"),
        ):
            result = runtime.latest_frame(overlay=True)
        self.assertTrue(result["available"])
        self.assertEqual(result["data"], plain)
        self.assertFalse(result["overlay"]["drawn"])
        self.assertIn("Pillow", result["overlay"]["reason"])
        self.assertEqual(result["overlay"]["entities_available"], 1)

    def test_undecodable_cached_frame_still_returns_the_frame(self) -> None:
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(_FakeImage(640, 360), clock.now)
        runtime.store.ingest([self._entity("track:1", [0.2, 0.2, 0.6, 0.9])], observed_at=clock.now)
        result = runtime.latest_frame(overlay=True)
        # _FakeImage 存的是伪 JPEG，真 Pillow 解不开；两种环境下都不能变成 unavailable。
        self.assertTrue(result["available"])
        self.assertIn("drawn", result["overlay"])

    def test_skipped_boxes_are_counted_so_silence_is_not_mistaken_for_zero(self) -> None:
        """世界里有 3 个实体、图上只画了 1 个，这个差必须报出来。

        这里必须缓存**真** JPEG：``_FakeImage`` 造的伪 JPEG 连 Pillow 都解不开，
        于是绘制走降级分支，``boxes_drawn`` 根本不会被算出来——断言就成了空转。
        """
        try:
            from io import BytesIO

            from PIL import Image
        except Exception:  # pragma: no cover - 取决于机器上有没有可选依赖
            self.skipTest("Pillow is not installed")
        buf = BytesIO()
        Image.new("RGB", (120, 90), (12, 12, 12)).save(buf, format="JPEG", quality=80)
        clock = _Clock()
        runtime = self._runtime(clock)
        runtime._cache_frame(buf.getvalue(), clock.now)
        runtime.store.ingest(
            [
                self._entity("track:1", [0.2, 0.2, 0.6, 0.9]),
                self._entity("track:2", [1.4, 1.4, 1.8, 1.9]),
                self._entity("track:3", [0.5, 0.5, 0.5, 0.5]),
            ],
            observed_at=clock.now,
        )
        overlay = runtime.latest_frame(overlay=True)["overlay"]
        self.assertTrue(overlay["drawn"], overlay.get("reason"))
        self.assertEqual(overlay["entities_available"], 3)
        self.assertEqual(overlay["boxes_drawn"], 1)
        self.assertEqual(overlay["boxes_skipped"], 2)


class DrawDetectionOverlayTests(unittest.TestCase):
    """绘制那一层需要真 Pillow，所以整类按依赖 skip。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from PIL import Image, ImageDraw  # noqa: F401
        except Exception:  # pragma: no cover - 取决于机器上有没有可选依赖
            raise unittest.SkipTest("Pillow is not installed")

    @staticmethod
    def _jpeg(width: int = 120, height: int = 90) -> bytes:
        from io import BytesIO

        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", (width, height), (12, 12, 12)).save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def test_drawing_reports_how_many_boxes_landed_on_the_image(self) -> None:
        data, drawn = draw_detection_overlay(
            self._jpeg(),
            [
                {"id": "a", "label": "person", "confidence": 0.83, "bbox": [0.1, 0.1, 0.5, 0.9]},
                {"id": "b", "label": "person", "confidence": 0.4, "bbox": [1.4, 1.4, 1.9, 1.9]},
            ],
        )
        self.assertEqual(drawn, 1)
        self.assertTrue(data.startswith(b"\xff\xd8"))

    def test_drawing_changes_pixels_and_leaves_the_input_bytes_alone(self) -> None:
        original = self._jpeg()
        data, drawn = draw_detection_overlay(
            original, [{"id": "a", "label": "person", "confidence": 0.9, "bbox": [0.1, 0.1, 0.9, 0.9]}]
        )
        self.assertEqual(drawn, 1)
        self.assertNotEqual(data, original)
        self.assertEqual(original, self._jpeg())

    def test_geometry_is_preserved_through_the_round_trip(self) -> None:
        from io import BytesIO

        from PIL import Image

        data, _ = draw_detection_overlay(self._jpeg(120, 90), [])
        with Image.open(BytesIO(data)) as image:
            self.assertEqual(image.size, (120, 90))

    def test_warning_is_burned_into_the_image_not_just_reported(self) -> None:
        """JSON 字段会被忽略，画面顶部的红条不会。"""
        plain, _ = draw_detection_overlay(self._jpeg(), [])
        warned, _ = draw_detection_overlay(self._jpeg(), [], warning="SKEW 900ms")
        self.assertNotEqual(plain, warned)

    def test_undecodable_input_raises_so_the_caller_can_degrade(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be decoded"):
            draw_detection_overlay(b"not-a-jpeg", [])


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
