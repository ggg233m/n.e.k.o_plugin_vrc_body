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
        self.assertIn("depth_unavailable", first.uncertainties)

    def test_small_post_nms_batch_is_not_transposed_as_channel_first(self) -> None:
        rows = [[0.1, 0.1, 0.2, 0.2, 0.9, 0.0] for _ in range(4)]
        detector = OpenVinoLocalDetector(infer=lambda _frame: rows, confidence_threshold=0.5)
        observation = detector.observe(object(), now=1.2)
        self.assertEqual(len(observation.entities), 4)

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
