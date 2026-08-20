from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional perception dependency
    np = None  # type: ignore[assignment]

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.local_perception import OpenVinoLocalDetector
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
            },
        })
        self.assertEqual(config.vision.model_path, "models/yolox.xml")
        self.assertEqual(config.vision.fallback_backend, "opencv_hog")
        self.assertEqual(config.vision.input_width, 320)
        self.assertEqual(config.vision.max_detections, 12)


if __name__ == "__main__":
    unittest.main()
