from __future__ import annotations

import threading
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.vision import (
    MssFrameSource,
    VisionObservation,
    VisionRuntime,
    VisionWorker,
    optional_dependency_status,
)
from neko_anyadance_body.backend.world_state import (
    WorldEntity,
    WorldEvent,
    WorldStateStore,
    stable_entity_id,
)


class WorldStateStoreTests(unittest.TestCase):
    def test_stable_entity_id_and_track_id_fallback(self) -> None:
        self.assertEqual(stable_entity_id("yolo", "cup", 7), "yolo:cup:7")
        self.assertEqual(stable_entity_id("yolo", "cup:large", 7), "yolo:cup_large:7")
        with self.assertRaises(ValueError):
            stable_entity_id("yolo", "cup", "")

        store = WorldStateStore(clock=lambda: 10.0)
        snapshot = store.ingest(
            entities=[{
                "track_id": 7,
                "label": "cup",
                "confidence": 0.9,
            }],
            source="yolo",
        )
        self.assertEqual(snapshot["entities"][0]["id"], "yolo:cup:7")

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

    def test_future_timestamps_are_clamped_and_older_tracks_do_not_replace_newer(self) -> None:
        now = [10.0]
        store = WorldStateStore(clock=lambda: now[0], default_ttl_s=1.0)
        first = store.ingest(
            entities=[{
                "id": "yolo:cup:7",
                "label": "cup",
                "confidence": 0.9,
                "state": "new",
            }],
            source="yolo",
            observed_at=999.0,
        )
        self.assertEqual(first["entities"][0]["age_ms"], 0.0)

        now[0] = 10.5
        reordered = store.ingest(
            entities=[{
                "id": "yolo:cup:7",
                "label": "cup",
                "confidence": 0.95,
                "state": "old",
            }],
            source="yolo",
            observed_at=9.5,
        )
        self.assertEqual(reordered["entities"][0]["state"], "new")
        self.assertEqual(reordered["status"]["last_observation_age_ms"], 500.0)

        now[0] = 11.1
        self.assertFalse(store.snapshot()["available"])

    def test_dataclass_observations_cannot_bypass_normalization(self) -> None:
        now = [20.0]
        store = WorldStateStore(clock=lambda: now[0])
        snapshot = store.ingest(
            entities=[WorldEntity(
                id="yolo:cup:8",
                label="cup",
                confidence=4.0,
                source=("yolo",),
                observed_at=999_999.0,
                ttl_s=0.1,
            )],
            events=[WorldEvent(
                kind="cup_seen",
                target_id="yolo:cup:8",
                confidence=3.0,
                source=("yolo",),
                observed_at=999_999.0,
            )],
        )
        self.assertEqual(snapshot["entities"][0]["confidence"], 1.0)
        self.assertEqual(snapshot["entities"][0]["age_ms"], 0.0)
        self.assertEqual(snapshot["events"][0]["confidence"], 1.0)
        self.assertEqual(snapshot["events"][0]["age_ms"], 0.0)

        now[0] = 20.2
        expired = store.snapshot()
        self.assertEqual(expired["entities"], [])
        self.assertEqual(expired["events"][0]["age_ms"], 200.0)


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


class _FrameSource:
    name = "fake_source"

    def __init__(self) -> None:
        self.closed = False
        self.frames = 0

    def status(self):
        return {"available": not self.closed, "frames": self.frames}

    def read(self):
        if self.closed:
            return None
        self.frames += 1
        return {"frame": self.frames}

    def close(self):
        self.closed = True


class _WorkerDetector:
    name = "fake_detector"

    def __init__(self) -> None:
        self.calls = 0
        self.first_result = threading.Event()

    def status(self):
        return {"available": True, "backend": "test"}

    def observe(self, frame, *, now):
        self.calls += 1
        self.first_result.set()
        return VisionObservation(
            entities=({
                "track_id": 7,
                "label": "cup",
                "confidence": 0.9,
                "ttl_s": 0.5,
            },),
            source=self.name,
            observed_at=now,
            frame_id=str(frame["frame"]),
        )


class VisionRuntimeTests(unittest.TestCase):
    def test_optional_mss_source_degrades_without_model_dependencies(self) -> None:
        source = MssFrameSource()
        status = source.status()
        self.assertIn("available", status)
        if not optional_dependency_status()["mss"]:
            self.assertFalse(status["available"])
        source.close()

    def test_callback_and_worker_keep_capture_outside_control_path(self) -> None:
        observed = []
        store = WorldStateStore()
        runtime = VisionRuntime(
            store,
            detector=_WorkerDetector(),
            observation_callback=lambda item, result: observed.append((item, result)),
        )
        source = _FrameSource()
        detector = runtime.detector
        worker = VisionWorker(runtime, source, interval_s=0.01, queue_size=1)
        self.assertTrue(worker.start())
        self.assertTrue(detector.first_result.wait(1.0))
        worker.stop()

        status = worker.status()
        self.assertFalse(status["running"])
        self.assertGreaterEqual(status["frames_captured"], 1)
        self.assertGreaterEqual(status["frames_processed"], 1)
        self.assertTrue(source.closed)
        self.assertTrue(observed)
        snapshot = store.snapshot()
        self.assertEqual(snapshot["entities"][0]["id"], "fake_detector:cup:7")
        self.assertEqual(snapshot["entities"][0]["source"], ["fake_detector"])

    def test_worker_without_detector_does_not_start(self) -> None:
        source = _FrameSource()
        runtime = VisionRuntime(WorldStateStore())
        worker = VisionWorker(runtime, source, interval_s=0.01)
        self.assertFalse(worker.start())
        self.assertIn("no detector", worker.status()["last_error"])
        worker.stop()

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
