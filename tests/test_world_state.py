from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.vision import VisionObservation, VisionRuntime
from neko_anyadance_body.backend.world_state import WorldStateStore


class WorldStateStoreTests(unittest.TestCase):
    def test_entity_is_bounded_and_expires(self) -> None:
        now = [10.0]
        store = WorldStateStore(clock=lambda: now[0], default_ttl_s=1.0)
        snapshot = store.ingest(
            entities=[
                {
                    "id": "button",
                    "label": "button",
                    "confidence": 1.5,
                    "bbox": [-1.0, 0.25, 1.5, 0.75],
                    "state": "pressed",
                    "attributes": {"long": "x" * 1000},
                    "source": [f"source_{index}" for index in range(20)],
                    "relations": [{"target": index} for index in range(20)],
                }
            ],
            source="yolo",
        )
        entity = snapshot["entities"][0]
        self.assertEqual(entity["confidence"], 1.0)
        self.assertEqual(entity["bbox"], [0.0, 0.25, 1.0, 0.75])
        self.assertLessEqual(len(entity["attributes"]["long"]), 512)
        self.assertLessEqual(len(entity["source"]), 8)
        self.assertLessEqual(len(entity["relations"]), 16)
        self.assertTrue(snapshot["available"])

        now[0] = 11.1
        expired = store.snapshot()
        self.assertFalse(expired["available"])
        self.assertEqual(expired["entities"], [])
        self.assertIn("no_recent_visual_observation", expired["uncertainties"])

    def test_invalid_entities_are_ignored_and_entity_count_is_bounded(self) -> None:
        store = WorldStateStore(max_entities=2, clock=lambda: 1.0)
        snapshot = store.ingest(
            entities=[
                {"id": "a", "label": "a", "confidence": 0.4},
                {"id": "b", "label": "b", "confidence": 0.9},
                {"id": "c", "label": "c", "confidence": 0.8},
                {"label": "missing_id"},
            ]
        )
        self.assertEqual(len(snapshot["entities"]), 2)
        self.assertEqual({item["id"] for item in snapshot["entities"]}, {"b", "c"})


class _Detector:
    name = "fake_yolo"

    def __init__(self) -> None:
        self.calls = 0

    def status(self):
        return {"available": True, "backend": "test"}

    def observe(self, frame, *, now):
        self.calls += 1
        return VisionObservation(
            entities=({"id": "avatar_1", "label": "avatar", "confidence": 0.9},),
            source=self.name,
            observed_at=now,
        )


class _Semantic:
    name = "fake_vlm"

    def __init__(self) -> None:
        self.calls = 0

    def status(self):
        return {"available": True, "backend": "test"}

    def observe(self, frame, *, world, now):
        self.calls += 1
        return VisionObservation(
            events=({"type": "avatar_seen", "target_id": "avatar_1", "confidence": 0.8},),
            source=self.name,
            observed_at=now,
        )


class VisionRuntimeTests(unittest.TestCase):
    def test_detector_runs_every_frame_and_semantic_backend_is_rate_limited(self) -> None:
        now = [0.0]
        detector = _Detector()
        semantic = _Semantic()
        runtime = VisionRuntime(
            WorldStateStore(clock=lambda: now[0]),
            detector=detector,
            semantic=semantic,
            semantic_cooldown_s=1.0,
            clock=lambda: now[0],
        )

        runtime.process_frame(object())
        now[0] = 0.25
        runtime.process_frame(object())
        now[0] = 1.1
        runtime.process_frame(object())

        self.assertEqual(detector.calls, 3)
        self.assertEqual(semantic.calls, 2)
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["status"]["entity_count"], 1)
        self.assertEqual(snapshot["status"]["event_count"], 2)
        self.assertTrue(snapshot["vision"]["detector"]["available"])
        self.assertTrue(snapshot["vision"]["semantic"]["available"])

    def test_backend_error_is_reported_without_raising_to_control_loop(self) -> None:
        class BrokenDetector(_Detector):
            def observe(self, frame, *, now):
                raise RuntimeError("detector offline")

        runtime = VisionRuntime(detector=BrokenDetector())
        snapshot = runtime.process_frame(object())
        self.assertIn("detector offline", snapshot["vision"]["last_error"])


if __name__ == "__main__":
    unittest.main()
