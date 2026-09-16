from __future__ import annotations

import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy 是本地视觉的可选依赖
    np = None  # type: ignore[assignment]

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.avatar_identity import AvatarIdentityRegistry, _similarity
from neko_anyadance_body.backend.local_perception import OpenVinoLocalDetector, _Detection
from neko_anyadance_body.backend.reid_embedder import OsnetReidEmbedder

_MODEL = (
    Path(__file__).resolve().parents[1]
    / "models/osnet_x0_25_msmt17/osnet_x0_25_msmt17_dynamic.onnx"
)


def _onnxruntime_available() -> bool:
    try:
        from neko_anyadance_body.backend.local_perception import import_onnxruntime

        import_onnxruntime()
        return True
    except Exception:
        return False


class _FakeSession:
    """把裁剪的均值颜色映射为固定向量的替身，覆盖不依赖 ORT 的逻辑路径。"""

    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self):
        class _Input:
            name = "input"
            shape = [None, 3, 256, 128]

        return [_Input()]

    def run(self, _outputs, feeds):
        self.calls += 1
        tensor = feeds["input"]
        mean_rgb = tensor.mean(axis=(0, 2, 3))
        return [np.concatenate([mean_rgb, np.ones(1, dtype=np.float32)])[None, :]]


@unittest.skipIf(np is None, "numpy 是本地视觉的可选依赖")
class OsnetReidEmbedderTests(unittest.TestCase):
    def test_missing_model_is_unavailable_without_raising(self) -> None:
        embedder = OsnetReidEmbedder(model_path="does/not/exist.onnx")
        self.assertFalse(embedder.available)
        self.assertIsNotNone(embedder.status()["error"])
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.assertIsNone(embedder.embed(frame, (0.1, 0.1, 0.9, 0.9)))

    def test_injected_session_produces_normalized_vector(self) -> None:
        embedder = OsnetReidEmbedder(model_path="injected", session=_FakeSession())
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)
        vector = embedder.embed(frame, (0.1, 0.1, 0.9, 0.9))
        self.assertIsNotNone(vector)
        assert vector is not None
        self.assertAlmostEqual(sum(item * item for item in vector), 1.0, places=5)
        self.assertEqual(embedder.status()["embed_count"], 1)

    def test_invalid_bbox_and_non_image_frame_return_none(self) -> None:
        embedder = OsnetReidEmbedder(model_path="injected", session=_FakeSession())
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.assertIsNone(embedder.embed(frame, (0.9, 0.1, 0.1, 0.9)))
        self.assertIsNone(embedder.embed(object(), (0.1, 0.1, 0.9, 0.9)))

    def test_registry_never_matches_across_descriptor_dimensions(self) -> None:
        # 直方图向量与嵌入向量长度不同；_similarity 必须返回 0 而不是异常，
        # 否则换描述子后残存的旧身份会被误匹配。
        self.assertEqual(_similarity((1.0, 0.0), (1.0, 0.0, 0.0)), 0.0)


@unittest.skipIf(np is None, "numpy 是本地视觉的可选依赖")
@unittest.skipUnless(_MODEL.exists(), "OSNet 模型未随仓库检出")
@unittest.skipUnless(_onnxruntime_available(), "onnxruntime 不可用")
class OsnetRealModelTests(unittest.TestCase):
    """用真实 ONNX 验证端到端语义：同外观相近、不同外观拉开。"""

    @staticmethod
    def _frame(top_color, bottom_color):
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        frame[30:80, 90:150] = top_color
        frame[80:140, 90:150] = bottom_color
        return frame

    def test_same_appearance_scores_higher_than_different(self) -> None:
        embedder = OsnetReidEmbedder(model_path=_MODEL)
        self.assertTrue(embedder.available, embedder.status()["error"])
        bbox = (90 / 240, 30 / 160, 150 / 240, 140 / 160)
        red = embedder.embed(self._frame((200, 40, 55), (40, 60, 200)), bbox)
        red_darker = embedder.embed(
            (self._frame((200, 40, 55), (40, 60, 200)) * 0.75).astype(np.uint8), bbox
        )
        green = embedder.embed(self._frame((30, 190, 80), (230, 230, 230)), bbox)
        assert red is not None and red_darker is not None and green is not None
        same = _similarity(red, red_darker)
        different = _similarity(red, green)
        self.assertGreater(same, different)

    def test_detector_uses_osnet_and_reports_status(self) -> None:
        bbox = (0.1, 0.1, 0.6, 0.9)
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [{"label": "person", "confidence": 0.95, "bbox": bbox}],
            },
            identity_reid_model_path=str(_MODEL),
        )
        status = detector.status()["identity_reid"]
        self.assertEqual(status["descriptor"], "osnet_onnx")
        self.assertEqual(status["similarity_threshold"], 0.75)
        self.assertTrue(status["embedder"]["available"])

        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        frame[20:140, 30:140] = (200, 40, 55)
        entity = detector.observe(frame, now=1.0).entities[0]
        self.assertTrue(entity["id"].startswith("avatar:session:"))

    def test_missing_model_falls_back_to_histogram_threshold(self) -> None:
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {"detections": []},
            identity_reid_model_path="does/not/exist.onnx",
            identity_reid_similarity=0.9,
        )
        status = detector.status()["identity_reid"]
        self.assertEqual(status["descriptor"], "color_histogram")
        self.assertEqual(status["similarity_threshold"], 0.9)
        self.assertFalse(status["embedder"]["available"])


@unittest.skipIf(np is None, "numpy 是本地视觉的可选依赖")
class RegistryDescriptorInjectionTests(unittest.TestCase):
    def test_registry_uses_injected_descriptor(self) -> None:
        calls: list[tuple] = []

        def descriptor(_frame, bbox):
            calls.append(tuple(bbox))
            return (1.0, 0.0, 0.0)

        registry = AvatarIdentityRegistry(
            session_token="test",
            descriptor_fn=descriptor,
            descriptor_name="injected",
        )
        first = registry.assign(
            [_Detection("person", 0.95, (0.1, 0.1, 0.5, 0.9), track_id=1)],
            object(),
            now=1.0,
            source_name="openvino",
        )[1]
        second = registry.assign(
            [_Detection("person", 0.95, (0.5, 0.1, 0.9, 0.9), track_id=2)],
            object(),
            now=2.0,
            source_name="openvino",
        )[2]
        self.assertEqual(first.identity_id, second.identity_id)
        self.assertEqual(second.method, "appearance_reid")
        self.assertEqual(registry.status()["descriptor"], "injected")
        self.assertGreaterEqual(len(calls), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
