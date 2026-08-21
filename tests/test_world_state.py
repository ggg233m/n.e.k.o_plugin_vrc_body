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
    blocking_uncertainties,
    stable_entity_id,
    stable_track_entity_id,
    vrchat_player_entity_id,
)


class WorldStateStoreTests(unittest.TestCase):
    def test_blocking_uncertainties_filters_capability_boundaries_only(self) -> None:
        self.assertEqual(
            blocking_uncertainties(["depth_unavailable", "ocr_unavailable", "opencv_hog_person_only"]),
            [],
        )
        # 白名单之外默认阻断，包括尚未出现过的新编码。
        self.assertEqual(
            blocking_uncertainties(["depth_unavailable", "observation_stale"]),
            ["observation_stale"],
        )
        self.assertEqual(blocking_uncertainties(["future_code"]), ["future_code"])
        self.assertEqual(blocking_uncertainties(None), [])
        self.assertEqual(blocking_uncertainties([]), [])

    def test_safe_navigation_ignores_capability_boundary_uncertainties(self) -> None:
        store = WorldStateStore()
        store.ingest(
            entities=[{"id": "person", "label": "person", "confidence": 0.9}],
            source="openvino",
            uncertainties=["depth_unavailable", "ocr_unavailable"],
        )
        self.assertTrue(store.delta(after_revision=0, wait_ms=0)["navigation"]["safe_navigation"])

        store.ingest(
            entities=[{"id": "person", "label": "person", "confidence": 0.9}],
            source="openvino",
            uncertainties=["depth_unavailable", "world_switched"],
        )
        self.assertFalse(store.delta(after_revision=0, wait_ms=0)["navigation"]["safe_navigation"])

    def test_stable_entity_id_and_track_id_fallback(self) -> None:
        self.assertEqual(stable_entity_id("yolo", "cup", 7), "yolo:cup:7")
        self.assertEqual(stable_entity_id("yolo", "cup:large", 7), "yolo:cup_large:7")
        self.assertEqual(stable_track_entity_id("yolo", 7), "yolo:track:7")
        self.assertEqual(vrchat_player_entity_id("usr:123"), "vrchat:player:usr_123")
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

    def test_remove_entities_is_atomic_source_filtered_and_keeps_events(self) -> None:
        now = [10.0]
        store = WorldStateStore(clock=lambda: now[0])
        store.ingest(
            entities=[
                {
                    "id": "vrchat:player:alice",
                    "label": "player",
                    "source": "vrchat_log",
                    "confidence": 0.9,
                },
                {
                    "id": "vision:player:alice",
                    "label": "player",
                    "source": "vision",
                    "confidence": 0.8,
                },
                {
                    "id": "vrchat:player:bob",
                    "label": "player",
                    "source": "vrchat_log",
                    "confidence": 0.7,
                },
            ],
            source="test",
        )

        result = store.remove_entities(
            ["vrchat:player:alice", "vrchat:player:alice", "missing"],
            source="vrchat_log",
            events=[{
                "type": "player_left",
                "target_id": "vrchat:player:alice",
                "source": "vrchat_log",
                "confidence": 1.0,
            }],
        )

        self.assertEqual(
            {item["id"] for item in result["entities"]},
            {"vision:player:alice", "vrchat:player:bob"},
        )
        self.assertEqual(result["events"][0]["type"], "player_left")
        self.assertEqual(result["events"][0]["target_id"], "vrchat:player:alice")
        # 返回值包含本次变更；其核心世界快照应与后续读取一致。
        current = store.snapshot()
        self.assertEqual(current["entities"], result["entities"])
        self.assertEqual(current["events"], result["events"])
        self.assertEqual(current["status"], result["status"])

    def test_remove_entity_and_source_prefix_helpers(self) -> None:
        store = WorldStateStore(clock=lambda: 20.0)
        store.ingest(
            entities=[
                {"id": "vrchat:player:alice", "label": "player", "source": "vrchat_log"},
                {"id": "vrchat:player:bob", "label": "player", "source": "vrchat_log"},
                {"id": "vrchat:world:portal", "label": "portal", "source": "vrchat_log"},
                {"id": "vision:player:alice", "label": "player", "source": "vision"},
            ]
        )

        self.assertFalse(store.remove_entity("missing"))
        self.assertFalse(store.remove_entity("vision:player:alice", source="vrchat_log"))
        self.assertTrue(store.remove_entity("vrchat:player:alice", source="vrchat_log"))
        self.assertFalse(store.remove_entity("vrchat:player:alice", source="vrchat_log"))
        self.assertEqual(store.remove_entities_by_source("vrchat_log", prefix="vrchat:player:"), 1)
        self.assertEqual(store.remove_entities_by_source("vrchat_log", prefix="vrchat:player:"), 0)
        snapshot = store.snapshot()
        self.assertEqual(
            {item["id"] for item in snapshot["entities"]},
            {"vrchat:world:portal", "vision:player:alice"},
        )

    def test_remove_missing_entity_with_owner_fences_late_first_frame(self) -> None:
        now = [10.0]
        store = WorldStateStore(clock=lambda: now[0], default_ttl_s=10.0)
        self.assertFalse(store.remove_entity("vrchat:player:early", source="vrchat_log"))
        now[0] = 11.0
        stale = store.ingest(entities=[{
            "id": "vrchat:player:early",
            "label": "late",
            "source": "vrchat_log",
            "observed_at": 10.0,
        }], source="vrchat_log")
        self.assertEqual(stale["entities"], [])

    def test_ingest_lifecycle_removal_is_atomic_and_source_scoped(self) -> None:
        now = [30.0]
        store = WorldStateStore(clock=lambda: now[0])
        store.ingest(
            entities=[{
                "id": "vrchat:player:usr_1",
                "label": "player",
                "source": "vrchat_log",
            }]
        )
        result = store.ingest(
            entities=[{
                "id": "vrchat:player:usr_1",
                "label": "stale-reappearance",
                "source": "vrchat_log",
            }],
            events=[{
                "type": "player_left",
                "target_id": "vrchat:player:usr_1",
                "source": "vrchat_log",
                "confidence": 1.0,
            }],
            source="vrchat_log",
            remove_entity_ids="vrchat:player:usr_1",
            remove_source="vrchat_log",
        )
        self.assertEqual(result["changes"]["removed_entity_ids"], ["vrchat:player:usr_1"])
        self.assertEqual(result["changes"]["removed_entity_count"], 1)
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["events"][0]["type"], "player_left")

        # 来源过滤不能删掉同 ID 的其他来源实体。
        store.ingest(entities=[{
            "id": "shared-id",
            "label": "vision-object",
            "source": "vision",
        }])
        scoped = store.ingest(
            source="vrchat_log",
            remove_entity_ids=["shared-id"],
            remove_source="vrchat_log",
        )
        self.assertEqual(scoped["changes"]["removed_entity_ids"], [])
        self.assertEqual(scoped["entities"][0]["id"], "shared-id")
        with self.assertRaises(ValueError):
            store.ingest(remove_source="vrchat_log")

    def test_delayed_player_left_does_not_delete_newer_entity(self) -> None:
        now = [100.0]
        store = WorldStateStore(clock=lambda: now[0], default_ttl_s=10.0)
        store.ingest(
            entities=[{
                "id": "vrchat:player:u1",
                "label": "player",
                "source": "vrchat_log",
                "observed_at": 100.0,
            }],
            source="vrchat_log",
        )

        # 日志事件在 90 时刻观测到，但在更新的检测帧之后才到达，不能擦除新状态。
        now[0] = 101.0
        result = store.ingest(
            events=[{
                "type": "player_left",
                "target_id": "vrchat:player:u1",
                "source": "vrchat_log",
                "observed_at": 90.0,
            }],
            source="vrchat_log",
            observed_at=90.0,
            remove_entity_ids=["vrchat:player:u1"],
            remove_source="vrchat_log",
        )
        self.assertEqual(result["changes"]["removed_entity_ids"], [])
        self.assertEqual(result["entities"][0]["id"], "vrchat:player:u1")

    def test_delete_watermark_blocks_late_frame_but_allows_newer_reentry(self) -> None:
        now = [100.0]
        store = WorldStateStore(clock=lambda: now[0], default_ttl_s=10.0)
        store.ingest(entities=[{
            "id": "vrchat:player:u2",
            "label": "player",
            "source": "vrchat_log",
            "observed_at": 100.0,
        }], source="vrchat_log")

        now[0] = 101.0
        removed = store.ingest(
            source="vrchat_log",
            remove_entity_ids=["vrchat:player:u2"],
            remove_source="vrchat_log",
            observed_at=101.0,
        )
        self.assertEqual(removed["changes"]["removed_entity_ids"], ["vrchat:player:u2"])

        now[0] = 102.0
        stale = store.ingest(entities=[{
            "id": "vrchat:player:u2",
            "label": "stale",
            "source": "vrchat_log",
            "observed_at": 100.0,
        }], source="vrchat_log")
        self.assertEqual(stale["entities"], [])

        now[0] = 103.0
        fresh = store.ingest(entities=[{
            "id": "vrchat:player:u2",
            "label": "rejoined",
            "source": "vrchat_log",
            "observed_at": 102.0,
        }], source="vrchat_log")
        self.assertEqual(fresh["entities"][0]["label"], "rejoined")

    def test_lifecycle_schema_requires_source_and_full_removal_batch(self) -> None:
        store = WorldStateStore(max_removals=1, clock=lambda: 10.0)
        store.ingest(entities=[{
            "id": "target",
            "label": "object",
            "source": "vision",
        }])
        with self.assertRaisesRegex(ValueError, "remove_source is required"):
            store.ingest(remove_entity_ids=["target"])
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            store.ingest(remove_entity_ids=["target"], remove_source=" ")
        with self.assertRaisesRegex(ValueError, "at most 1"):
            store.ingest(
                remove_entity_ids=["missing", "target"],
                remove_source="vision",
            )
        self.assertEqual(store.snapshot()["entities"][0]["id"], "target")

    def test_player_left_must_match_removed_target_and_bulk_reset_rejects_it(self) -> None:
        store = WorldStateStore(clock=lambda: 10.0)
        with self.assertRaisesRegex(ValueError, "target_id"):
            store.ingest(
                events=[{
                    "type": "player_left",
                    "target_id": "a",
                    "source": "vrchat_log",
                }],
                source="vrchat_log",
                remove_entity_ids=["b"],
                remove_source="vrchat_log",
            )
        with self.assertRaisesRegex(ValueError, "cannot publish player_left"):
            store.remove_entities_by_source(
                "vrchat_log",
                events=[{
                    "type": "player_left",
                    "target_id": "vrchat:player:a",
                    "source": "vrchat_log",
                }],
            )

    def test_bulk_source_reset_fences_late_frames(self) -> None:
        now = [10.0]
        store = WorldStateStore(clock=lambda: now[0], default_ttl_s=10.0)
        store.ingest(entities=[{
            "id": "vrchat:player:u3",
            "label": "player",
            "source": "vrchat_log",
            "observed_at": 10.0,
        }], events=[{
            "type": "player_joined",
            "target_id": "vrchat:player:u3",
            "source": "vrchat_log",
        }], source="vrchat_log")
        now[0] = 11.0
        self.assertEqual(
            store.remove_entities_by_source(
                "vrchat_log",
                prefix="vrchat:player:",
                observed_at=10.0,
            ),
            1,
        )
        self.assertEqual(store.snapshot()["events"], [])
        now[0] = 12.0
        stale = store.ingest(entities=[{
            "id": "vrchat:player:u3",
            "label": "stale",
            "source": "vrchat_log",
            "observed_at": 10.0,
        }], source="vrchat_log")
        self.assertEqual(stale["entities"], [])

    def test_lifecycle_source_uses_canonical_owner_for_multi_source_entity(self) -> None:
        now = [10.0]
        store = WorldStateStore(clock=lambda: now[0], default_ttl_s=10.0)
        store.ingest(entities=[{
            "id": "shared",
            "label": "object",
            "source": ["vision", "vrchat_log"],
        }])
        result = store.ingest(
            source="vrchat_log",
            remove_entity_ids=["shared"],
            remove_source="vrchat_log",
        )
        self.assertEqual(result["changes"]["removed_entity_ids"], [])
        self.assertEqual(result["entities"][0]["id"], "shared")


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


class _LateFrameSource(_FrameSource):
    """模拟 close() 后才返回一帧的采集源，用于验证停止竞态。"""

    def __init__(self) -> None:
        super().__init__()
        self.read_entered = threading.Event()
        self.release_read = threading.Event()

    def read(self):
        self.read_entered.set()
        self.release_read.wait(2.0)
        # 即使 close() 已经被调用，也故意返回一帧迟到画面。
        self.frames += 1
        return {"frame": self.frames}


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
    def test_lifecycle_mapping_rejects_scalar_and_blank_source(self) -> None:
        runtime = VisionRuntime(WorldStateStore())
        with self.assertRaisesRegex(ValueError, "must be an array"):
            runtime.ingest({"remove_entity_ids": "entity"})
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            runtime.ingest({"remove_entity_ids": ["entity"], "remove_source": " "})
        with self.assertRaisesRegex(ValueError, "observation must be an object"):
            runtime.ingest(42)  # type: ignore[arg-type]

    def test_normal_observation_keeps_legacy_store_signature_compatible(self) -> None:
        class LegacyStore:
            def __init__(self):
                self.calls = []

            def ingest(self, entities, events, *, source, observed_at):
                self.calls.append((entities, events, source, observed_at))
                return {"entities": [], "events": [], "status": {}}

            def set_backend_status(self, *_args):
                return None

        store = LegacyStore()
        runtime = VisionRuntime(store)  # type: ignore[arg-type]
        runtime.ingest(VisionObservation(source="legacy"))
        self.assertEqual(store.calls[0][2], "legacy")

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

    def test_capture_only_worker_reports_frames_without_fabricating_world_entities(self) -> None:
        source = _FrameSource()
        store = WorldStateStore()
        runtime = VisionRuntime(store)
        worker = VisionWorker(runtime, source, interval_s=0.01, capture_only=True)
        self.assertTrue(worker.start())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and worker.status()["frames_captured"] < 1:
            time.sleep(0.005)
        worker.stop()
        status = worker.status()
        self.assertTrue(status["capture_only"])
        self.assertGreaterEqual(status["frames_captured"], 1)
        self.assertEqual(store.snapshot()["entities"], [])

    def test_late_frame_after_stop_cannot_reactivate_capture_or_write_world(self) -> None:
        source = _LateFrameSource()
        store = WorldStateStore()
        runtime = VisionRuntime(store, detector=_WorkerDetector())
        worker = VisionWorker(runtime, source, interval_s=0.01)
        self.assertTrue(worker.start())
        self.assertTrue(source.read_entered.wait(1.0))

        stopper = threading.Thread(target=worker.stop)
        stopper.start()
        # 让 stop() 先设置停止门，再释放阻塞的 read()。
        time.sleep(0.02)
        source.release_read.set()
        stopper.join(2.0)

        self.assertFalse(stopper.is_alive())
        self.assertFalse(runtime.capture_state()["active"])
        self.assertEqual(store.snapshot()["entities"], [])

    def test_stopped_worker_frame_is_rejected_by_runtime_ingest(self) -> None:
        store = WorldStateStore()
        runtime = VisionRuntime(store)
        runtime.set_capture_state(False, "manual_stop")
        result = runtime.ingest(
            VisionObservation(
                source="late_detector",
                entities=({"id": "stale", "label": "stale", "confidence": 0.9},),
            ),
            _reactivate=False,
        )
        self.assertFalse(result["available"])
        self.assertEqual(store.snapshot()["entities"], [])

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

    def test_detect_interval_throttles_inference_without_erasing_the_world(self) -> None:
        """检测占空比是 CPU 上限的第二个轴，但跳帧不能变成"看到了空场景"。

        限线程只管"一次推理占几个核"，限间隔才管"一秒推几次"，两者相乘才是
        实际占用。而跳过的帧必须**不写**世界状态：补一个空观测会把「这一帧
        没看」伪造成「这一帧什么都没有」，实体会在两次推理之间整体闪烁消失。
        不写则由 store 自己让 age_ms 长上去，「多久之前看到的」仍然是真话。
        """
        now = [0.0]
        detector = _Detector()
        runtime = VisionRuntime(
            WorldStateStore(clock=lambda: now[0]),
            detector=detector,
            detect_interval_s=0.5,
            clock=lambda: now[0],
        )

        runtime.process_frame(object())
        self.assertEqual(detector.calls, 1)
        # 间隔内的帧被跳过：推理次数不涨，但已有实体不能消失。
        for moment in (0.1, 0.25, 0.49):
            now[0] = moment
            snapshot = runtime.process_frame(object())
            self.assertEqual(detector.calls, 1, moment)
            self.assertEqual(len(snapshot["entities"]), 1, moment)

        now[0] = 0.5
        runtime.process_frame(object())
        self.assertEqual(detector.calls, 2)

        throttle = runtime.snapshot()["vision"]["detect_throttle"]
        self.assertEqual(throttle["interval_s"], 0.5)
        self.assertEqual(throttle["skipped_frames"], 3)

    def test_zero_detect_interval_leaves_every_frame_inferred(self) -> None:
        """0 表示不限速——这是基准测量和默认行为依赖的出口，不能被钳成有限值。"""
        now = [0.0]
        detector = _Detector()
        runtime = VisionRuntime(
            WorldStateStore(clock=lambda: now[0]),
            detector=detector,
            detect_interval_s=0.0,
            clock=lambda: now[0],
        )
        for index in range(4):
            now[0] = index * 0.001
            runtime.process_frame(object())
        self.assertEqual(detector.calls, 4)
        self.assertEqual(runtime.snapshot()["vision"]["detect_throttle"]["skipped_frames"], 0)

    def test_an_obscured_frame_is_not_inferred_at_all(self) -> None:
        """窗口被盖住时抓到的是上层窗口的像素，对它推理等于凭空造实体。

        DXGI 抓的是合成后的桌面，被遮挡时采集"成功"但内容是别的应用。检测器
        在浏览器或聊天窗里找到的人形会被写成世界里的玩家，而下游分不出真假。
        所以整帧不推理——检测器和 VLM 一个都不叫。
        """
        detector = _Detector()
        semantic = _Semantic()
        runtime = VisionRuntime(detector=detector, semantic=semantic)

        snapshot = runtime.process_frame(object(), source_obscured=True)

        self.assertEqual(detector.calls, 0)
        self.assertEqual(semantic.calls, 0)
        self.assertEqual(snapshot["entities"], [])
        self.assertIn("visual_capture_occluded", snapshot["uncertainties"])
        self.assertEqual(snapshot["vision"]["obscured_frames"], 1)

    def test_occlusion_uncertainty_stops_movement(self) -> None:
        """看不见就不该走：这条不确定性必须留在阻断集合里。

        ``INFORMATIONAL_UNCERTAINTIES`` 白名单里的编码不阻断移动。遮挡说的是
        "当前观测不可信"，不是"这项能力天生没有"，一旦被误加进白名单，机器人
        就会照着一屏浏览器画面继续导航。
        """
        self.assertEqual(
            blocking_uncertainties(["visual_capture_occluded"]),
            ["visual_capture_occluded"],
        )

    def test_obscured_frames_let_known_entities_age_out_instead_of_erasing_them(self) -> None:
        """遮挡不是离场证据，已有实体只能按各自 TTL 自然老化。

        强清会把"我暂时看不见"伪造成"它们走了"，Alt-Tab 一下世界就空掉；而 TTL
        路径保留了 age_ms，"多久之前看到的"仍然是真话。
        """
        now = [0.0]
        detector = _Detector()
        runtime = VisionRuntime(
            WorldStateStore(clock=lambda: now[0]),
            detector=detector,
            clock=lambda: now[0],
        )
        runtime.process_frame(object())
        self.assertEqual(len(runtime.snapshot()["entities"]), 1)

        for moment in (0.2, 0.4, 0.6):
            now[0] = moment
            snapshot = runtime.process_frame(object(), source_obscured=True)
            self.assertEqual(len(snapshot["entities"]), 1, moment)
            self.assertNotIn("changes", snapshot, moment)

        self.assertEqual(detector.calls, 1)
        self.assertGreater(snapshot["entities"][0]["age_ms"], 0.0)

    def test_obscured_writes_follow_the_detect_interval(self) -> None:
        """遮挡期间的声明也要限流，否则每帧都在推高世界修订号。

        遮挡可能持续几分钟。若每帧都写一次，消费者的增量流会被一条重复的
        不确定性刷满，真正的变化反而被挤出窗口。
        """
        now = [0.0]
        runtime = VisionRuntime(
            WorldStateStore(clock=lambda: now[0]),
            detector=_Detector(),
            detect_interval_s=0.5,
            clock=lambda: now[0],
        )
        first = runtime.process_frame(object(), source_obscured=True)["status"]["revision"]
        for moment in (0.1, 0.25, 0.49):
            now[0] = moment
            self.assertEqual(
                runtime.process_frame(object(), source_obscured=True)["status"]["revision"],
                first,
                moment,
            )

        now[0] = 0.5
        self.assertGreater(
            runtime.process_frame(object(), source_obscured=True)["status"]["revision"],
            first,
        )
        vision = runtime.snapshot()["vision"]
        self.assertEqual(vision["obscured_frames"], 5)
        self.assertEqual(vision["detect_throttle"]["skipped_frames"], 3)

    def test_worker_reads_occlusion_from_the_source_before_inferring(self) -> None:
        """遮挡状态由采集源报告，worker 只负责把它传下去。

        运行时拿不到窗口句柄，只有采集源知道自己抓的是不是目标窗口。
        """
        class ObscuredSource(_FrameSource):
            def status(self):
                return {"available": True, "window_found": True, "window_obscured": True}

        detector = _Detector()
        runtime = VisionRuntime(detector=detector)
        worker = VisionWorker(runtime=runtime, source=ObscuredSource(), interval_s=0.01)
        worker.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and runtime.status()["obscured_frames"] < 1:
                time.sleep(0.01)
        finally:
            worker.stop()

        self.assertGreaterEqual(runtime.status()["obscured_frames"], 1)
        self.assertEqual(detector.calls, 0)
        self.assertIn("visual_capture_occluded", runtime.snapshot()["uncertainties"])

    def test_worker_treats_an_unreadable_source_status_as_visible(self) -> None:
        """探测故障不能停掉全部感知：宁放不误封。

        把 status() 抛异常读成"被遮挡"，等于用一个诊断故障换掉整条视觉通路。
        """
        class BrokenStatusSource(_FrameSource):
            def status(self):
                raise RuntimeError("status unavailable")

        detector = _Detector()
        runtime = VisionRuntime(detector=detector)
        worker = VisionWorker(runtime=runtime, source=BrokenStatusSource(), interval_s=0.01)
        worker.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and detector.calls < 1:
                time.sleep(0.01)
        finally:
            worker.stop()

        self.assertGreaterEqual(detector.calls, 1)
        self.assertEqual(runtime.status()["obscured_frames"], 0)

    def test_backend_error_is_reported_without_raising_to_control_loop(self) -> None:
        class BrokenDetector(_Detector):
            def observe(self, frame, *, now):
                raise RuntimeError("detector offline")

        runtime = VisionRuntime(detector=BrokenDetector())
        snapshot = runtime.process_frame(object())
        self.assertIn("detector offline", snapshot["vision"]["last_error"])

    def test_stopped_capture_masks_stale_world_without_deleting_store(self) -> None:
        store = WorldStateStore()
        runtime = VisionRuntime(store)
        runtime.ingest(VisionObservation(
            source="test_detector",
            entities=({"id": "button", "label": "button", "confidence": 0.9},),
        ))
        self.assertTrue(runtime.snapshot()["available"])

        runtime.set_capture_state(False, "manual_stop")
        masked = runtime.snapshot()
        self.assertFalse(masked["available"])
        self.assertEqual(masked["entities"], [])
        self.assertIn("visual_capture_stopped", masked["uncertainties"])
        self.assertEqual(masked["capture_reason"], "manual_stop")
        self.assertEqual(store.snapshot()["entities"][0]["id"], "button")

        delta = runtime.delta(after_revision=0, wait_ms=0)
        self.assertFalse(delta["world"]["available"])
        self.assertEqual(delta["changes"]["entities"], [])
        self.assertFalse(delta["navigation"]["safe_navigation"])

        runtime.set_capture_state(True, "running")
        self.assertEqual(runtime.snapshot()["entities"][0]["id"], "button")


if __name__ == "__main__":
    unittest.main()
