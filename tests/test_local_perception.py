from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional perception dependency
    np = None  # type: ignore[assignment]

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import local_perception as _lp
from neko_anyadance_body.backend.local_perception import (
    OpenVinoLocalDetector,
    _LabelLoadError,
    _load_labels,
    cap_openmp_threads,
)
from neko_anyadance_body.config import PluginConfig


class LocalPerceptionTests(unittest.TestCase):
    def test_missing_model_is_explicitly_unavailable_and_does_not_fabricate(self) -> None:
        detector = OpenVinoLocalDetector()
        status = detector.status()
        self.assertFalse(status["available"])
        self.assertFalse(status["degraded"])
        self.assertIn("model_path", str(status["last_error"]))
        with self.assertRaises(RuntimeError):
            detector.observe(object(), now=1.0)

    def test_injected_yolo_output_becomes_tracked_entities_and_relations(self) -> None:
        frames = [
            [
                # cx、cy、宽度、高度、objectness，以及 person 的类别分数。
                [0.25, 0.50, 0.20, 0.30, 0.95, 0.90],
                [0.75, 0.50, 0.20, 0.30, 0.90, 0.80],
            ],
            [
                [0.26, 0.50, 0.20, 0.30, 0.95, 0.90],
                [0.74, 0.50, 0.20, 0.30, 0.90, 0.80],
            ],
        ]

        def infer(_frame):
            return frames.pop(0)

        detector = OpenVinoLocalDetector(infer=infer, confidence_threshold=0.4)
        first = detector.observe(object(), now=1.0)
        second = detector.observe(object(), now=1.1)
        self.assertEqual(first.source, "openvino")
        self.assertEqual(len(first.entities), 2)
        self.assertEqual(len(second.entities), 2)
        first_ids = {item["id"] for item in first.entities}
        second_ids = {item["id"] for item in second.entities}
        self.assertEqual(first_ids, second_ids)
        left = min(first.entities, key=lambda item: item["bbox"][0])
        self.assertLess(left["attributes"]["bearing_deg"], 0.0)
        self.assertTrue(any(item["type"] == "left_of" for item in left["relations"]))
        # 深度/OCR 缺失是检测器的永久能力边界，属于 status()，不应混进逐帧
        # 不确定性——否则每一帧观测都带着同样两条噪声，导航器会永久停摆。
        self.assertEqual(first.uncertainties, ())
        capabilities = detector.status()["capabilities"]
        self.assertTrue(capabilities["object_detection"])
        self.assertFalse(capabilities["depth"])
        self.assertFalse(capabilities["ocr"])

    def test_small_post_nms_batch_is_not_transposed_as_channel_first(self) -> None:
        # 这个测试的核心断言是矩阵方向正确：(4,6) 不被误判为 (6,4)。
        # (6,4) 转置后的行长度为 4，会被解码器丢弃（六列检查失败），得到 0 个实体。
        # (4,6) 保持不动，4 行全部过阈值；4 个相同位置的框是同一目标的重复——NMS
        # 正确地合并成 1 个，而不是 0 个（转置错误）。
        rows = [[0.1, 0.1, 0.2, 0.2, 0.9, 0.0] for _ in range(4)]
        detector = OpenVinoLocalDetector(infer=lambda _frame: rows, confidence_threshold=0.5)
        observation = detector.observe(object(), now=1.2)
        # 转置错误 -> 0；NMS 正确合并重复框 -> 1。
        self.assertEqual(len(observation.entities), 1)

    @unittest.skipIf(np is None, "numpy is optional")
    def test_tensor_layout_is_keyed_on_row_width_not_detection_count(self) -> None:
        """检测数恰好等于某个列宽时不得被误判为通道优先布局。

        旧启发式按 ``shape[0]`` 的魔数区间猜测，导致 7、8、84、85 个检测的
        NMS 后输出被转置成 6 行；正确的 YOLOX ``(N,85)`` 在 N=16..84 时同样
        被破坏。列宽判定对两者都稳定。
        """
        detector = OpenVinoLocalDetector(infer=lambda _frame: None)
        cases = (
            # (shape, expected_rows, expected_cols)
            ((4, 6), 4, 6),      # NMS 后小批量
            ((7, 6), 7, 6),      # 曾被转置成 (6,7)
            ((8, 6), 8, 6),      # 曾被转置成 (6,8)
            ((84, 6), 84, 6),    # 曾被转置成 (6,84)
            ((85, 6), 85, 6),
            ((16, 85), 16, 85),  # YOLOX anchors-first，曾被转置成 (85,16)
            ((80, 85), 80, 85),
            ((8400, 85), 8400, 85),
            ((84, 8400), 8400, 84),   # YOLOv8 channels-first，必须转置
            ((6, 7), 6, 7),      # SSD 六个检测
            ((1, 4, 6), 4, 6),   # 三维分支曾把它转成 (6,4)
            ((1, 5, 6), 5, 6),
            ((1, 6, 7), 6, 7),
            ((1, 80, 85), 80, 85),
            ((1, 200, 7), 200, 7),
            ((1, 84, 8400), 8400, 84),
            ((1, 8400, 85), 8400, 85),
        )
        for shape, want_rows, want_cols in cases:
            with self.subTest(shape=shape):
                rows = detector._as_rows([np.zeros(shape, dtype=np.float32)])
                self.assertEqual(len(rows), want_rows)
                self.assertEqual(len(rows[0]), want_cols)

    @unittest.skipIf(np is None, "numpy is optional")
    def test_bare_tensor_decodes_identically_to_wrapped_and_named_outputs(self) -> None:
        # 裸张量交给 list() 会按第一轴拆成一维切片，绕过布局判定：同一个
        # ``(84, 8400)`` 曾经裸传得到 84 行、包成列表却得到 8400 行。
        detector = OpenVinoLocalDetector(infer=lambda _frame: None)
        tensor = np.zeros((84, 8400), dtype=np.float32)
        bare = detector._as_rows(tensor)
        wrapped = detector._as_rows([tensor])
        named = detector._as_rows({"output": tensor})
        self.assertEqual(len(bare), 8400)
        self.assertEqual(len(bare[0]), 84)
        self.assertEqual((len(bare), len(bare[0])), (len(wrapped), len(wrapped[0])))
        self.assertEqual((len(bare), len(bare[0])), (len(named), len(named[0])))

    def test_apparent_height_is_published_for_approach_control(self) -> None:
        # 二维检测器无法推断深度，但检测框高度是实测量；导航器用它闭环。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [{
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": [0.4, 0.25, 0.6, 0.75],
                }],
            },
        )
        attributes = detector.observe(object(), now=7.0).entities[0]["attributes"]
        self.assertAlmostEqual(attributes["apparent_height"], 0.5, places=5)
        self.assertFalse(attributes["apparent_height_clipped"])
        # 仍然不伪造米制距离。
        self.assertEqual(attributes["depth"], "unknown")
        self.assertNotIn("distance_m", attributes)

    def test_edge_clipped_target_is_flagged_as_unmeasurable(self) -> None:
        # 贴到画面上下边说明目标超出视野，表观高度饱和，不能再当作距离使用。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [{
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": [0.3, 0.0, 0.7, 1.0],
                }],
            },
        )
        attributes = detector.observe(object(), now=7.5).entities[0]["attributes"]
        self.assertTrue(attributes["apparent_height_clipped"])

    @unittest.skipIf(np is None, "numpy is optional")
    def test_raw_yolo_anchor_cluster_collapses_to_one_entity_per_object(self) -> None:
        """裸 YOLO 输出的 anchor 簇必须被抑制成每个目标一个实体。

        仅按置信度排序截断时，留下的是同一个目标的重复框——实测两个人会变成
        48 个 person 实体、2256 条关系，等于向 LLM 谎报房间人数。
        """
        rng = np.random.default_rng(7)
        anchors = []
        for center_x, center_y in ((200.0, 320.0), (450.0, 330.0)):
            for _ in range(24):
                row = [0.0] * 84
                row[0] = center_x + rng.normal(0, 6)
                row[1] = center_y + rng.normal(0, 8)
                row[2] = 90 + rng.normal(0, 7)
                row[3] = 260 + rng.normal(0, 18)
                row[4] = 0.55 + rng.random() * 0.4
                anchors.append(row)
        # 其余 anchor 低于阈值，应当在解码阶段就被丢弃。
        for _ in range(300):
            anchors.append([320.0, 320.0, 10.0, 10.0, 0.01] + [0.0] * 79)

        channels_first = np.asarray(anchors, dtype=np.float32).T[None, ...]
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: channels_first,
            confidence_threshold=0.35,
            input_width=640,
            input_height=640,
        )
        observation = detector.observe(object(), now=1.0)
        self.assertEqual(len(observation.entities), 2)
        self.assertEqual({item["label"] for item in observation.entities}, {"person"})
        # 两个实体互为左右关系，不应出现重复框带来的关系爆炸。
        self.assertLessEqual(sum(len(item["relations"]) for item in observation.entities), 4)

    def test_overlapping_distinct_classes_are_not_suppressed(self) -> None:
        # 只在同类之间抑制：人拿着杯子时两个框本就高度重叠。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [
                    {"label": "person", "confidence": 0.9, "bbox": [0.30, 0.20, 0.70, 0.90]},
                    {"label": "cup", "confidence": 0.8, "bbox": [0.31, 0.21, 0.69, 0.89]},
                ],
            },
        )
        observation = detector.observe(object(), now=2.0)
        self.assertEqual({item["label"] for item in observation.entities}, {"person", "cup"})

    def test_already_suppressed_output_is_unchanged(self) -> None:
        # 后处理格式（或 nms=True 导出）已经去重，抑制必须是幂等的。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: [
                [0.05, 0.10, 0.25, 0.90, 0.90, 0.0],
                [0.40, 0.10, 0.60, 0.90, 0.85, 0.0],
                [0.70, 0.10, 0.95, 0.90, 0.80, 0.0],
            ],
            confidence_threshold=0.5,
        )
        observation = detector.observe(object(), now=2.5)
        self.assertEqual(len(observation.entities), 3)

    def test_suppression_keeps_the_highest_scoring_box_of_a_cluster(self) -> None:
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [
                    {"label": "person", "confidence": 0.55, "bbox": [0.30, 0.20, 0.70, 0.90]},
                    {"label": "person", "confidence": 0.95, "bbox": [0.32, 0.22, 0.72, 0.92]},
                ],
            },
        )
        observation = detector.observe(object(), now=3.0)
        self.assertEqual(len(observation.entities), 1)
        self.assertAlmostEqual(observation.entities[0]["confidence"], 0.95, places=5)

    def test_named_tensor_mapping_is_flattened_without_silent_drop(self) -> None:
        # OpenVINO/旁路适配器常用 {output_name: tensor} 形式返回结果。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {"output": [[0.5, 0.5, 0.4, 0.4, 0.9, 0.9]]},
            confidence_threshold=0.5,
        )
        observation = detector.observe(object(), now=1.3)
        self.assertEqual(len(observation.entities), 1)

    def test_ssd_pixel_coordinates_are_normalized_and_low_confidence_filtered(self) -> None:
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: [
                [0, 1, 0.9, 64, 128, 320, 512],
                [0, 2, 0.1, 10, 10, 100, 100],
            ],
            input_width=640,
            input_height=640,
            confidence_threshold=0.5,
        )
        observation = detector.observe(object(), now=3.0)
        self.assertEqual(len(observation.entities), 1)
        self.assertEqual(observation.entities[0]["label"], "bicycle")
        self.assertEqual(observation.entities[0]["bbox"], [0.1, 0.2, 0.5, 0.8])

    @unittest.skipIf(np is None, "numpy is optional")
    def test_yolov8_channel_first_output_uses_class_score_without_objectness(self) -> None:
        # 84 = 4 个框坐标 + 80 个 COCO 类别分数；一个 anchor 即可测试解码器形状。
        row = [0.5, 0.5, 0.4, 0.4] + [0.0] * 80
        row[4 + 0] = 0.9  # person 类别
        channels_first = np.asarray(row, dtype=np.float32).reshape(1, 84, 1)
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: channels_first,
            confidence_threshold=0.5,
        )
        observation = detector.observe(object(), now=3.5)
        self.assertEqual(len(observation.entities), 1)
        self.assertEqual(observation.entities[0]["label"], "person")

    @unittest.skipIf(np is None, "numpy is optional")
    def test_single_class_yolov8_five_column_output_is_decoded(self) -> None:
        """单类 YOLOv8 导出省略类别向量，行宽为 5：``[cx,cy,w,h,score]``。

        deepghs/anime_person_detection 只有 ``person`` 一类，ONNX 输出形状是
        ``[1,5,8400]``。解码器早先按 ``len(row) < 6`` 丢弃整批行，整个模型族
        静默地产出零检测——可用性完全消失，却不报任何错误。
        """
        anchors = np.zeros((5, 8400), dtype=np.float32)
        # 模型输入空间（640x640）的像素坐标，这正是 ultralytics ONNX 的约定。
        anchors[:, 0] = [320.0, 320.0, 100.0, 260.0, 0.91]
        raw = anchors[None, ...]
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: raw,
            confidence_threshold=0.35,
            input_width=640,
            input_height=640,
        )
        observation = detector.observe(object(), now=1.0)
        self.assertEqual(len(observation.entities), 1)
        entity = observation.entities[0]
        self.assertEqual(entity["label"], "person")
        self.assertAlmostEqual(entity["confidence"], 0.91, places=5)
        # (320±50, 320±130) / 640 —— 像素坐标必须归一化，而不是被钳到 1.0。
        self.assertEqual(
            [round(value, 6) for value in entity["bbox"]],
            [0.421875, 0.296875, 0.578125, 0.703125],
        )

    @unittest.skipIf(np is None, "numpy is optional")
    def test_five_column_geometry_matches_the_wide_row_branch(self) -> None:
        # 同一个框经 5 列分支和 84 列分支解码必须得到同一个 bbox；两条分支
        # 对 cx/cy/w/h 的解释和归一化都相同，只有类别向量的有无不同。
        box = [320.0, 320.0, 100.0, 260.0]
        narrow = np.asarray([box + [0.91]], dtype=np.float32)
        wide_row = box + [0.0] * 80
        wide_row[4 + 0] = 0.91  # person
        wide = np.asarray([wide_row], dtype=np.float32)
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: None,
            confidence_threshold=0.35,
            input_width=640,
            input_height=640,
        )
        (from_narrow,) = detector._decode_rows(narrow)
        (from_wide,) = detector._decode_rows(wide)
        self.assertEqual(from_narrow.label, from_wide.label)
        self.assertEqual(
            [round(v, 6) for v in from_narrow.bbox],
            [round(v, 6) for v in from_wide.bbox],
        )

    @unittest.skipIf(np is None, "numpy is optional")
    def test_five_column_anchor_grid_is_prefiltered_like_wider_rows(self) -> None:
        """预筛选必须覆盖 5 列，否则 8400 个 anchor 会逐行进 Python。

        逐行转换 8400 行要上百毫秒，足以把 10 Hz 采集压到 3 Hz 并触发导航器的
        observation_stale——这正是预筛选存在的理由，单类模型不该被排除在外。
        """
        anchors = np.zeros((5, 8400), dtype=np.float32)
        anchors[:, 0] = [320.0, 320.0, 100.0, 260.0, 0.91]
        raw = anchors[None, ...]
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: None,
            confidence_threshold=0.35,
            input_width=640,
            input_height=640,
        )
        self.assertEqual(len(detector._as_rows(raw)), 8400)
        pruned = detector._as_rows(raw, prefilter=True)
        self.assertLess(len(pruned), 100)
        # 预筛选是纯优化：唯一过阈值的 anchor 必须还在。
        self.assertTrue(any(abs(row[4] - 0.91) < 1e-5 for row in pruned))

    @unittest.skipIf(np is None, "numpy is optional")
    def test_tiny_high_confidence_boxes_are_dropped_as_noise(self) -> None:
        """几十像素的高分框是假阳性，不是远处的人。

        实测真实帧里出现过 27×27 px 的 0.9 分框。跟踪器会给它分配实体 ID，
        导航器再用 ``apparent_height`` 反推距离，于是把噪点当成一个站在很远处
        的人——比漏检更糟，因为它会主动驱动动作。
        """
        # 640 输入下 27 px ≈ 4.2%… 取更小的 12 px（1.9%）作为噪点，
        # 同时给一个正常大小的框，确认过滤只吃掉小的那个。
        rows = np.asarray([
            [320.0, 320.0, 12.0, 12.0, 0.93],
            [200.0, 300.0, 100.0, 260.0, 0.88],
        ], dtype=np.float32)
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: None,
            confidence_threshold=0.35,
            input_width=640,
            input_height=640,
            min_box_ratio=0.02,
        )
        decoded = detector._decode_rows(rows)
        self.assertEqual([round(item.confidence, 2) for item in decoded], [0.88])

    @unittest.skipIf(np is None, "numpy is optional")
    def test_min_box_ratio_zero_disables_the_filter(self) -> None:
        # 关掉过滤必须真的关掉：调低阈值是排查漏检的第一步，这条路径不能失效。
        rows = np.asarray([[320.0, 320.0, 12.0, 12.0, 0.93]], dtype=np.float32)
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: None,
            confidence_threshold=0.35,
            input_width=640,
            input_height=640,
            min_box_ratio=0.0,
        )
        self.assertEqual(len(detector._decode_rows(rows)), 1)

    def test_min_box_ratio_also_applies_to_record_form_adapters(self) -> None:
        # OpenCV HOG 的 detectMultiScale 同样会吐小框，记录式适配器不能绕过过滤。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [
                    {"label": "person", "confidence": 0.9, "bbox": [0.50, 0.50, 0.51, 0.51]},
                    {"label": "person", "confidence": 0.8, "bbox": [0.10, 0.20, 0.30, 0.80]},
                ],
            },
            min_box_ratio=0.02,
        )
        observation = detector.observe(object(), now=4.0)
        self.assertEqual(len(observation.entities), 1)
        self.assertAlmostEqual(observation.entities[0]["confidence"], 0.8, places=5)

    def test_distant_person_survives_the_width_axis(self) -> None:
        """远处的人是「窄」而不是「小」，不能和噪点用同一个阈值裁掉。

        1920 宽的采集区里，共用 2% 阈值要求最小宽 38 px；一个 2.6:1 的站立
        avatar 到那个宽度时高度已经占了画面 5% 以上，而更远一点的人（宽 24 px、
        高 62 px）会被整条丢掉——导航器的距离闭环整个挂在 ``apparent_height``
        上，看不见就等于房间对面没人。
        """
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [
                    # 远处的人：宽 1.25%（< 2%），高 3.2%。
                    {"label": "person", "confidence": 0.7, "bbox": [0.40, 0.44, 0.4125, 0.472]},
                    # 27×27 px 级别的方形噪点：两条边都不到 1.5%。
                    {"label": "person", "confidence": 0.95, "bbox": [0.70, 0.70, 0.714, 0.714]},
                ],
            },
            min_box_width_ratio=0.008,
            min_box_height_ratio=0.02,
        )
        observation = detector.observe(object(), now=4.0)
        self.assertEqual(len(observation.entities), 1)
        # 留下的必须是远处的人，不是高分噪点。
        self.assertAlmostEqual(observation.entities[0]["confidence"], 0.7, places=5)

    def test_wall_seam_shaped_boxes_are_still_rejected(self) -> None:
        """放松宽阈值不能把墙缝和 UI 边框放进来。

        这是原本「两条边都要过关」规则守住的东西：细长误检面积不小，但没有
        一条边像人。高度轴仍然是硬门槛。
        """
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [
                    # 横向的 UI 边框：很宽，但高度只有 0.6%。
                    {"label": "person", "confidence": 0.9, "bbox": [0.10, 0.50, 0.90, 0.506]},
                ],
            },
            min_box_width_ratio=0.008,
            min_box_height_ratio=0.02,
        )
        observation = detector.observe(object(), now=4.0)
        self.assertEqual(observation.entities, ())

    def test_legacy_shared_min_box_ratio_still_applies_to_both_axes(self) -> None:
        # 只传共用值的旧调用方不能因为默认值变成非对称而被静默改掉判定。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [
                    # 宽 1.25% < 共用的 2%，旧行为下应当被丢弃。
                    {"label": "person", "confidence": 0.7, "bbox": [0.40, 0.40, 0.4125, 0.60]},
                ],
            },
            min_box_ratio=0.02,
        )
        self.assertEqual(detector.observe(object(), now=4.0).entities, ())
        status = detector.status()
        self.assertAlmostEqual(status["min_box_width_ratio"], 0.02, places=6)
        self.assertAlmostEqual(status["min_box_height_ratio"], 0.02, places=6)

    def test_record_form_is_supported_for_explicit_fallback_adapters(self) -> None:
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [{
                    "label": "person",
                    "confidence": 0.8,
                    "bbox": [0.1, 0.2, 0.3, 0.8],
                }],
            },
        )
        observation = detector.observe(object(), now=4.0)
        self.assertEqual(len(observation.entities), 1)
        self.assertEqual(observation.entities[0]["label"], "person")

    def test_tracker_keeps_id_when_label_flickers_on_same_box(self) -> None:
        outputs = [
            {"detections": [{"label": "person", "confidence": 0.9, "bbox": [0.2, 0.2, 0.4, 0.8]}]},
            {"detections": [{"label": "avatar", "confidence": 0.9, "bbox": [0.2, 0.2, 0.4, 0.8]}]},
        ]
        detector = OpenVinoLocalDetector(infer=lambda _frame: outputs.pop(0))
        first = detector.observe(object(), now=1.0)
        second = detector.observe(object(), now=1.1)
        self.assertEqual(first.entities[0]["id"], second.entities[0]["id"])

    @unittest.skipIf(np is None, "numpy is optional")
    def test_openvino_core_is_loaded_lazily_without_importing_real_runtime(self) -> None:
        class _Input:
            any_name = "images"
            partial_shape = (1, 3, 8, 8)

        class _Model:
            inputs = [_Input()]

        class _Compiled:
            def __call__(self, _inputs):
                return np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]])

        class _Core:
            def read_model(self, **_kwargs):
                return _Model()

            def compile_model(self, _model, _device):
                return _Compiled()

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.xml"
            model.write_text("<fake/>", encoding="utf-8")
            detector = OpenVinoLocalDetector(
                model_path=str(model),
                openvino_core=_Core(),
                confidence_threshold=0.5,
            )
            self.assertTrue(detector.status()["available"])
            observation = detector.observe(np.zeros((8, 8, 3), dtype=np.uint8), now=5.0)
            self.assertEqual(len(observation.entities), 1)

    @unittest.skipIf(np is None, "numpy is optional")
    def test_onnx_can_fallback_to_opencv_dnn_when_openvino_is_unavailable(self) -> None:
        class _Net:
            def setInput(self, _tensor):
                return None

            def forward(self):
                return np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]])

        fake_cv2 = types.SimpleNamespace(
            dnn=types.SimpleNamespace(readNet=lambda _path: _Net()),
        )
        previous = sys.modules.get("cv2")
        sys.modules["cv2"] = fake_cv2  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / "model.onnx"
                model.write_bytes(b"fake")
                detector = OpenVinoLocalDetector(model_path=str(model))
                self.assertTrue(detector.status()["available"])
                self.assertEqual(detector.status()["runtime"], "opencv_dnn")
                observation = detector.observe(np.zeros((8, 8, 3), dtype=np.uint8), now=6.0)
                self.assertEqual(len(observation.entities), 1)
        finally:
            if previous is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = previous

    @unittest.skipIf(np is None, "numpy is optional")
    def test_prefilter_never_changes_which_detections_are_decoded(self) -> None:
        """预筛选是纯优化：过阈值的行必须与全量解码逐位一致。

        裸 YOLO 的第 2 列是像素宽度，不是分数。曾经用无门限的 ``row[2]``
        当上界，结果每一行都"过阈值"，截断退化成按框宽排序——实测真实模型
        的 5 个检测被砍成 1 个。
        """
        rng = np.random.default_rng(11)
        anchors = []
        for center_x, center_y, class_index in ((160.0, 300.0, 0), (470.0, 330.0, 5)):
            for _ in range(12):
                row = [0.0] * 84
                row[0] = center_x + rng.normal(0, 5)
                row[1] = center_y + rng.normal(0, 6)
                row[2] = 95 + rng.normal(0, 6)
                row[3] = 250 + rng.normal(0, 15)
                row[4 + class_index] = 0.6 + rng.random() * 0.35
                anchors.append(row)
        for _ in range(900):
            row = [320.0, 320.0, 88.0, 240.0] + [0.0] * 80
            row[4 + 0] = 0.02
            anchors.append(row)
        raw = np.asarray(anchors, dtype=np.float32).T[None, ...]
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: raw,
            confidence_threshold=0.35,
            input_width=640,
            input_height=640,
        )
        full = detector._as_rows(raw)
        pruned = detector._as_rows(raw, prefilter=True)
        self.assertEqual(len(full), 924)
        # 只有过阈值的 anchor 需要逐行解释，其余在 numpy 侧就被丢掉。
        self.assertLess(len(pruned), 100)

        def fingerprint(items):
            return sorted(
                (item.label, round(item.confidence, 6), tuple(round(v, 6) for v in item.bbox))
                for item in items
            )

        with_prefilter = detector._decode_rows(raw)
        original = type(detector)._as_rows
        try:
            type(detector)._as_rows = lambda self, outputs, *, prefilter=False: original(
                self, outputs, prefilter=False
            )
            without_prefilter = detector._decode_rows(raw)
        finally:
            type(detector)._as_rows = original
        self.assertEqual(fingerprint(with_prefilter), fingerprint(without_prefilter))
        self.assertEqual({item.label for item in with_prefilter}, {"person", "bus"})

    @unittest.skipIf(np is None, "numpy is optional")
    def test_prefilter_keeps_ssd_rows_whose_score_column_is_a_real_score(self) -> None:
        # 预筛选是保守上界：高分行的上界必须 ≥ 其真实分数（永不丢失）。
        # 低分行的上界可以更高，只要高分行不被丢弃就够了。
        # 只验证 "高分行不被预筛选丢弃" 这一正确性保证；
        # 低分行能否被丢弃是优化，不是正确性要求。
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: None,
            input_width=640,
            input_height=640,
            confidence_threshold=0.5,
        )
        rows = np.asarray(
            [
                [0, 1, 0.90, 64, 128, 320, 512],   # SSD 高分行，真实分数 0.90
                [0, 2, 0.10, 10, 10, 100, 100],     # SSD 低分行，真实分数 0.10
            ],
            dtype=np.float32,
        )
        bound = detector._score_upper_bound(np, rows)
        # 保守性保证：高分行的上界不得低于真实分数（否则会被误丢）。
        self.assertGreaterEqual(float(bound[0]), 0.90)

    @unittest.skipIf(np is None, "numpy is optional")
    def test_onnxruntime_is_preferred_for_onnx_models(self) -> None:
        # 宿主捆绑包带 onnxruntime 但没有 openvino/cv2；ONNX 必须走这条路。
        class _Input:
            name = "images"
            shape = ["batch", 3, "height", "width"]

        class _Session:
            def __init__(self, path, providers=None):
                self.providers = providers

            def get_inputs(self):
                return [_Input()]

            def run(self, _outputs, _feed):
                return [np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]], dtype=np.float32)]

        fake_ort = types.SimpleNamespace(InferenceSession=_Session)
        previous = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = fake_ort  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / "model.onnx"
                model.write_bytes(b"fake")
                detector = OpenVinoLocalDetector(model_path=str(model), confidence_threshold=0.5)
                status = detector.status()
                self.assertTrue(status["available"])
                self.assertEqual(status["runtime"], "onnxruntime")
                # 动态维度不能覆盖配置的输入尺寸。
                self.assertEqual(status["input_size"], [640, 640])
                observation = detector.observe(np.zeros((8, 8, 3), dtype=np.uint8), now=8.0)
                self.assertEqual(len(observation.entities), 1)
        finally:
            if previous is None:
                sys.modules.pop("onnxruntime", None)
            else:
                sys.modules["onnxruntime"] = previous

    @unittest.skipIf(np is None, "numpy is optional")
    def test_optional_cuda_provider_is_detected_before_cpu_and_falls_back_safely(self) -> None:
        """CUDA 可用才抢在 CPU 前；缺 provider/动态库或显式关闭都回落 CPU。"""
        class _Input:
            any_name = "images"
            partial_shape = (1, 3, 8, 8)
            name = "images"
            shape = [1, 3, 8, 8]

        class _Model:
            inputs = [_Input()]

        class _Compiled:
            def __call__(self, _inputs):
                return np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]])

        class _Core:
            def __init__(self, available_devices: list[str]) -> None:
                self.available_devices = list(available_devices)
                self.compile_calls: list[str] = []

            def read_model(self, **_kwargs):
                return _Model()

            def compile_model(self, _model, device, config=None):
                self.compile_calls.append(device)
                return _Compiled()

        def construct(
            available_providers: list[str],
            *,
            cuda_error: str | None = None,
            cuda_mode: str = "auto",
            cuda_device_id: int = 0,
            openvino_devices: list[str] | None = None,
        ):
            captured: dict[str, object] = {"provider_queries": 0}

            def get_available_providers():
                captured["provider_queries"] = int(captured["provider_queries"]) + 1
                return list(available_providers)

            class _Session:
                def __init__(self, _path, providers=None):
                    captured["providers"] = providers
                    names = [
                        item[0] if isinstance(item, tuple) else item
                        for item in (providers or [])
                    ]
                    if names and names[0] == "CUDAExecutionProvider" and cuda_error:
                        raise OSError(cuda_error)
                    self._providers = names

                def get_providers(self):
                    return list(self._providers)

                def get_inputs(self):
                    return [_Input()]

                def run(self, _outputs, _feed):
                    return [np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]])]

            fake_ort = types.SimpleNamespace(
                get_available_providers=get_available_providers,
                InferenceSession=_Session,
            )
            core = _Core(openvino_devices or ["CPU"])
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                detector = OpenVinoLocalDetector(
                    model_path=str(model),
                    openvino_core=core,
                    onnxruntime_cuda=cuda_mode,
                    onnxruntime_cuda_device_id=cuda_device_id,
                    intra_op_threads=0,
                )
            return detector, core, captured

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"fake")

            cuda, cuda_core, cuda_capture = construct(
                ["CUDAExecutionProvider", "CPUExecutionProvider"], cuda_device_id=2
            )
            cuda_status = cuda.status()
            self.assertEqual(cuda_status["runtime"], "onnxruntime_cuda")
            self.assertEqual(cuda_status["resolved_device"], "CUDA.2")
            self.assertEqual(cuda_core.compile_calls, [])
            self.assertEqual(
                cuda_capture["providers"][0],
                ("CUDAExecutionProvider", {"device_id": 2}),
            )
            self.assertIsNone(cuda_status["onnxruntime"]["cuda_error"])

            auto_intel, auto_intel_core, auto_intel_capture = construct(
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                openvino_devices=["GPU", "CPU"],
            )
            self.assertEqual(auto_intel.status()["resolved_device"], "GPU")
            self.assertEqual(auto_intel_core.compile_calls, ["GPU"])
            self.assertEqual(auto_intel_capture["provider_queries"], 0)

            preferred, preferred_core, _ = construct(
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                cuda_mode="prefer",
                openvino_devices=["GPU", "CPU"],
            )
            self.assertEqual(preferred.status()["runtime"], "onnxruntime_cuda")
            self.assertEqual(preferred_core.compile_calls, [])

            missing, missing_core, _ = construct(["CPUExecutionProvider"])
            missing_status = missing.status()
            self.assertEqual(missing_status["runtime"], "openvino")
            self.assertEqual(missing_status["resolved_device"], "CPU")
            self.assertEqual(missing_core.compile_calls, ["CPU"])
            self.assertIn("unavailable", missing_status["onnxruntime"]["cuda_error"])

            broken, broken_core, _ = construct(
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                cuda_error="cudnn64_9.dll is missing",
            )
            broken_status = broken.status()
            self.assertEqual(broken_status["runtime"], "openvino")
            self.assertEqual(broken_core.compile_calls, ["CPU"])
            self.assertIn("cudnn64_9.dll", broken_status["onnxruntime"]["cuda_error"])

            disabled, disabled_core, disabled_capture = construct(
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                cuda_mode="disabled",
            )
            disabled_status = disabled.status()
            self.assertEqual(disabled_status["runtime"], "openvino")
            self.assertEqual(disabled_core.compile_calls, ["CPU"])
            self.assertEqual(disabled_capture["provider_queries"], 0)
            self.assertFalse(disabled_status["onnxruntime"]["cuda_probed"])

    @unittest.skipIf(np is None, "numpy is optional")
    def test_onnxruntime_session_is_capped_on_both_thread_pools(self) -> None:
        """限 intra-op 还不够：留着 inter-op 池会在上限之外再开一组线程。

        实测默认配置（10 线程 + 背靠背推理）在 20 核机器上稳定吃掉 9.13 核，
        VRChat 自己开始掉帧。所以这里断言的不只是"设了个数"，而是两个池都被
        收窄、且执行模式是顺序的——只设 intra_op_num_threads 会让上限漏掉。
        """
        class _Input:
            name = "images"
            shape = ["batch", 3, "height", "width"]

        class _Options:
            def __init__(self) -> None:
                self.intra_op_num_threads = 0
                self.inter_op_num_threads = 0
                self.execution_mode = None

        captured = {}

        class _Session:
            def __init__(self, path, sess_options=None, providers=None):
                captured["options"] = sess_options

            def get_inputs(self):
                return [_Input()]

            def run(self, _outputs, _feed):
                return [np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]], dtype=np.float32)]

        fake_ort = types.SimpleNamespace(
            InferenceSession=_Session,
            SessionOptions=_Options,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        )
        previous = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = fake_ort  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / "model.onnx"
                model.write_bytes(b"fake")
                detector = OpenVinoLocalDetector(model_path=str(model), intra_op_threads=3)
                status = detector.status()
                self.assertTrue(status["available"])
                self.assertEqual(status["intra_op_threads"], 3)
                self.assertTrue(status["thread_cap_applied"])
                self.assertIsNone(status["thread_cap_error"])
                options = captured["options"]
                self.assertEqual(options.intra_op_num_threads, 3)
                self.assertEqual(options.inter_op_num_threads, 1)
                self.assertEqual(options.execution_mode, "sequential")
        finally:
            if previous is None:
                sys.modules.pop("onnxruntime", None)
            else:
                sys.modules["onnxruntime"] = previous

    @unittest.skipIf(np is None, "numpy is optional")
    def test_missing_thread_knob_degrades_the_cap_not_the_detector(self) -> None:
        """省 CPU 的设置不能变成关掉感知的理由。

        上限是优化，不是正确性前提。运行时不认这个旋钮时正确的行为是满线程
        跑（降级），而不是让 ``available`` 变 false（故障）——后者是拿故障换
        降级。但降级必须留痕：``thread_cap_applied`` 得能把"限住了"和"没限
        住"区分开，否则排查吃满 CPU 时只看得见一个撒谎的配置值。
        """
        class _Input:
            name = "images"
            shape = [1, 3, 8, 8]

        class _Session:
            def __init__(self, path, providers=None):
                self.providers = providers

            def get_inputs(self):
                return [_Input()]

            def run(self, _outputs, _feed):
                return [np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]], dtype=np.float32)]

        # 没有 SessionOptions 的运行时：构造上限的第一步就会失败。
        fake_ort = types.SimpleNamespace(InferenceSession=_Session)
        previous = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = fake_ort  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / "model.onnx"
                model.write_bytes(b"fake")
                detector = OpenVinoLocalDetector(model_path=str(model), intra_op_threads=2)
                status = detector.status()
                self.assertTrue(status["available"])
                self.assertEqual(status["runtime"], "onnxruntime")
                self.assertFalse(status["thread_cap_applied"])
                self.assertIsNotNone(status["thread_cap_error"])
                observation = detector.observe(np.zeros((8, 8, 3), dtype=np.uint8), now=9.0)
                self.assertEqual(len(observation.entities), 1)
        finally:
            if previous is None:
                sys.modules.pop("onnxruntime", None)
            else:
                sys.modules["onnxruntime"] = previous

    @unittest.skipIf(np is None, "numpy is optional")
    def test_openvino_thread_cap_survives_a_two_argument_compile_model(self) -> None:
        """OpenVINO 的上限走 compile_model 的第三个参数，但不能强求它存在。

        自带运行时/测试替身的 ``compile_model`` 可能只接受两个位置参数；多传
        一个不该把它们的检测器整体打掉。收到上限的实现要真的收到，没收到的
        要照常可用。
        """
        class _Input:
            any_name = "images"
            partial_shape = (1, 3, 8, 8)

        class _Model:
            inputs = [_Input()]

        class _Compiled:
            def __call__(self, _inputs):
                return np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]])

        class _ModernCore:
            def __init__(self) -> None:
                self.config = None

            def read_model(self, **_kwargs):
                return _Model()

            def compile_model(self, _model, _device, config=None):
                self.config = config
                return _Compiled()

        class _LegacyCore:
            def read_model(self, **_kwargs):
                return _Model()

            def compile_model(self, _model, _device):
                return _Compiled()

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.xml"
            model.write_text("<fake/>", encoding="utf-8")

            modern = _ModernCore()
            detector = OpenVinoLocalDetector(
                model_path=str(model), openvino_core=modern, intra_op_threads=4
            )
            self.assertTrue(detector.status()["available"])
            self.assertEqual(modern.config, {"INFERENCE_NUM_THREADS": 4})
            self.assertTrue(detector.status()["thread_cap_applied"])

            legacy = OpenVinoLocalDetector(
                model_path=str(model), openvino_core=_LegacyCore(), intra_op_threads=4
            )
            status = legacy.status()
            self.assertTrue(status["available"])
            self.assertFalse(status["thread_cap_applied"])
            self.assertIsNotNone(status["thread_cap_error"])
            observation = legacy.observe(np.zeros((8, 8, 3), dtype=np.uint8), now=10.0)
            self.assertEqual(len(observation.entities), 1)

    def test_auto_device_falls_through_each_qualified_gpu(self) -> None:
        """GPU.0 失败后必须尝试 GPU.1，而不是把两者折叠成 GPU。"""
        class _Input:
            any_name = "images"
            partial_shape = (1, 3, 8, 8)

        class _Model:
            inputs = [_Input()]

        class _Compiled:
            def __call__(self, _inputs):
                return np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]])

        class _Core:
            available_devices = ["CPU", "GPU.0", "GPU.1"]

            def __init__(self) -> None:
                self.calls: list[str] = []

            def read_model(self, **_kwargs):
                return _Model()

            def compile_model(self, _model, device, config=None):
                self.calls.append(device)
                if device == "GPU.0":
                    raise RuntimeError("GPU.0 cannot compile this model")
                return _Compiled()

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.xml"
            model.write_text("<fake/>", encoding="utf-8")
            core = _Core()
            detector = OpenVinoLocalDetector(
                model_path=str(model), openvino_core=core, intra_op_threads=0
            )
            status = detector.status()
            self.assertEqual(status["runtime"], "openvino")
            self.assertEqual(status["resolved_device"], "GPU.1")
            self.assertEqual(core.calls, ["GPU.0", "GPU.1"])
            self.assertIn("GPU.0: RuntimeError", status["device_fallbacks"][0])

    def test_config_exposes_explicit_model_and_degraded_fallback_options(self) -> None:
        config = PluginConfig.from_mapping({
            "vision": {
                "enabled": True,
                "local_backend": "openvino",
                "model_path": "models/yolox.xml",
                "labels_path": "models/labels.txt",
                "device": "GPU",
                "fallback_backend": "opencv_hog",
                "confidence_threshold": 0.5,
                "input_width": 320,
                "input_height": 320,
                "horizontal_fov_deg": 100,
                "max_detections": 12,
                "identity_reid_enabled": True,
                "identity_reid_similarity": 0.92,
                "identity_reid_margin": 0.06,
                "identity_reid_retention_s": 900,
                "identity_reid_max_identities": 48,
            },
        })
        self.assertEqual(config.vision.model_path, "models/yolox.xml")
        self.assertEqual(config.vision.fallback_backend, "opencv_hog")
        self.assertEqual(config.vision.input_width, 320)
        self.assertEqual(config.vision.max_detections, 12)
        self.assertTrue(config.vision.identity_reid_enabled)
        self.assertEqual(config.vision.identity_reid_similarity, 0.92)
        self.assertEqual(config.vision.identity_reid_margin, 0.06)
        self.assertEqual(config.vision.identity_reid_retention_s, 900)
        self.assertEqual(config.vision.identity_reid_max_identities, 48)


class OpenMpCapTests(unittest.TestCase):
    """进程级 OpenMP 收口。

    实测这才是吃满 CPU 的主因：numpy/BLAS 的 OpenMP 池按逻辑核数开线程并自旋，
    仅采集路径（检测器换成空实现、零次推理）就空转掉 7.23 核，收口后 0.11 核，
    吞吐一模一样。所以这里测的是「有没有真的去收」和「收的结果有没有被谎报」，
    不测具体核数——那要真机。
    """

    def setUp(self) -> None:
        # 环境变量与首次成功记录都是进程级状态，必须存档还原，否则测试之间会
        # 互相污染（先跑的那个把 setdefault 占掉，后跑的就永远看不到自己的值）。
        self._saved_env = {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY")
        }
        self._saved_state = dict(_lp._OPENMP_STATE)
        # numpy 在不在 sys.modules 里决定走「赶上了」还是「太晚了」分支，而它取决
        # 于测试顺序（别的用例会导入 numpy）。这里显式摘掉，让默认情形确定为
        # 「赶上了」；需要另一条分支的用例自己塞回去。
        self._saved_numpy = sys.modules.pop("numpy", None)
        for key in self._saved_env:
            os.environ.pop(key, None)
        _lp._OPENMP_STATE.clear()

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._saved_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = self._saved_numpy
        _lp._OPENMP_STATE.clear()
        _lp._OPENMP_STATE.update(self._saved_state)

    def test_openmp_is_capped_even_when_openvino_wins_over_onnxruntime(self) -> None:
        """OpenMP 收口不能挂在 ONNX Runtime 那条路上。"""
        class _Input:
            any_name = "images"
            partial_shape = (1, 3, 8, 8)

        class _Model:
            inputs = [_Input()]

        class _Compiled:
            def __call__(self, _inputs):
                return np.asarray([[0.5, 0.5, 0.5, 0.5, 0.9, 0.9]])

        class _Core:
            available_devices = ["CPU"]

            def read_model(self, **_kwargs):
                return _Model()

            def compile_model(self, _model, _device, config=None):
                return _Compiled()

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_text("<fake/>", encoding="utf-8")
            detector = OpenVinoLocalDetector(
                model_path=str(model), openvino_core=_Core(), intra_op_threads=3
            )
            status = detector.status()
            self.assertEqual(status["runtime"], "openvino")
            self.assertTrue(status["available"])
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "3")
            self.assertEqual(os.environ.get("OMP_WAIT_POLICY"), "PASSIVE")

    def test_cap_sets_both_thread_count_and_wait_policy(self) -> None:
        # 只设线程数只能到 0.87 核；WAIT_POLICY 才是把 7.23 核压到 0.11 核的那一
        # 个。少设一个就等于把大头留在门后，所以两个都必须写。
        result = cap_openmp_threads(2)

        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "2")
        self.assertEqual(os.environ.get("OMP_WAIT_POLICY"), "PASSIVE")
        self.assertEqual(result["requested"], 2)

    def test_explicit_operator_environment_wins(self) -> None:
        # 这是进程级全局状态，运维显式设了就该说了算——我们只补默认值。
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

        cap_openmp_threads(2)

        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "8")
        self.assertEqual(os.environ.get("OMP_WAIT_POLICY"), "ACTIVE")

    def test_zero_threads_leaves_the_environment_untouched(self) -> None:
        # 0 表示「不设上限」，留给需要裸速度的基准测量。它不该偷偷改环境。
        result = cap_openmp_threads(0)

        self.assertNotIn("OMP_NUM_THREADS", os.environ)
        self.assertNotIn("OMP_WAIT_POLICY", os.environ)
        self.assertFalse(result["env_applied"])

    def test_a_later_call_cannot_downgrade_a_successful_cap(self) -> None:
        # 检测器会在 BackendService 之后再调一次。那时 numpy 早就导入了，单看那
        # 次必然 too_late=True——如果让它覆盖记录，status() 就会在一次成功的收口
        # 上报「没限住」，把排查引到完全错误的方向。
        first = cap_openmp_threads(2)
        self.assertTrue(first["env_applied"])
        self.assertFalse(first["too_late"])

        sys.modules["numpy"] = types.ModuleType("numpy")
        second = cap_openmp_threads(2)

        self.assertTrue(second["env_applied"])
        self.assertFalse(second["too_late"])

    def test_missing_openmp_runtime_is_not_an_error(self) -> None:
        # 没有 OpenMP 运行时可限，只意味着少一项优化，不该让感知失败——和线程
        # 上限一样，降级优于故障。
        original = _lp._OPENMP_RUNTIME_DLLS
        _lp._OPENMP_RUNTIME_DLLS = ("definitely_not_a_real_openmp_dll",)
        sys.modules["numpy"] = types.ModuleType("numpy")
        try:
            result = cap_openmp_threads(2)
        finally:
            _lp._OPENMP_RUNTIME_DLLS = original

        self.assertIsNone(result["runtime_dll"])
        self.assertIsNone(result["error"])


class LabelLoadingTests(unittest.TestCase):
    """标签文件解析。

    错误的标签维度会让 ``_decode_rows`` 命中错误的分支（YOLOv8 vs YOLOX），
    产出满置信度的垃圾类名——所以无法识别的格式必须报错，而不是静默回落。
    """

    def _write(self, directory: str, name: str, text: str) -> str:
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_booru_meta_json_labels_key_is_recognised(self) -> None:
        # deepghs/booru_yolo 的 meta.json 用 "labels" 而不是 "names"；早先只认
        # "names"，26 类被静默换成 80 个 COCO 标签，解码出 class_3354 之类的垃圾。
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "meta.json",
                '{"labels": ["head", "bust", "hcat"], "name": "booru_yolov8s_aa11"}',
            )
            self.assertEqual(_load_labels(path), ("head", "bust", "hcat"))

    def test_ultralytics_names_forms_are_recognised(self) -> None:
        cases = {
            "list.json": ('{"names": ["person", "cat"]}', ("person", "cat")),
            "map.json": ('{"names": {"1": "cat", "0": "person"}}', ("person", "cat")),
            "bare_map.json": ('{"1": "cat", "0": "person"}', ("person", "cat")),
            "array.json": ('["person", "cat"]', ("person", "cat")),
            "plain.txt": ("person\n# comment\ncat\n", ("person", "cat")),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, (text, expected) in cases.items():
                with self.subTest(name=name):
                    self.assertEqual(_load_labels(self._write(directory, name, text)), expected)

    def test_missing_path_falls_back_to_builtin_coco(self) -> None:
        self.assertEqual(_load_labels(None)[0], "person")
        self.assertEqual(len(_load_labels(None)), 80)

    def test_unrecognisable_label_files_raise_instead_of_falling_back(self) -> None:
        cases = {
            "scalar.json": "42",
            "broken.json": "{not json",
            "empty.txt": "\n# only comments\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, text in cases.items():
                with self.subTest(name=name):
                    path = self._write(directory, name, text)
                    with self.assertRaises(_LabelLoadError):
                        _load_labels(path)
            with self.assertRaises(_LabelLoadError):
                _load_labels(str(Path(directory) / "does_not_exist.json"))

    def test_bad_label_file_is_surfaced_in_status_not_swallowed(self) -> None:
        # 回落 COCO 仍然发生（检测器可用），但原因必须出现在 status()，
        # 否则运维只会看到一堆置信度 1.0 的错误类名而无从追查。
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "labels.json", "{not json")
            detector = OpenVinoLocalDetector(labels_path=path)
            status = detector.status()
            self.assertIsNotNone(status["label_load_error"])
            self.assertIn("labels.json", str(status["label_load_error"]))

    def test_good_label_file_reports_no_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "labels.json", '{"labels": ["person"]}')
            detector = OpenVinoLocalDetector(labels_path=path)
            self.assertIsNone(detector.status()["label_load_error"])


if __name__ == "__main__":
    unittest.main()
