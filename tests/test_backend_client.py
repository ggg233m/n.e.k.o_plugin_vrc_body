from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.client import (
    BackendClient,
    BackendRejected,
    BackendUnavailable,
    RemoteAutonomy,
    RemoteScheduler,
    RemoteVision,
)
from neko_anyadance_body.backend.service import BackendService, _effective_detector_interval_ms
from neko_anyadance_body.backend.vision import VisionObservation


class BackendClientTests(unittest.TestCase):
    def test_semantic_backend_can_be_configured_without_neko_host(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "VRC_VLM_ENDPOINT": "",
                "OPENAI_BASE_URL": "",
                "VRC_VLM_MODEL": "",
                "OPENAI_VLM_MODEL": "",
                "VRC_VLM_API_KEY": "session-secret",
            },
            clear=False,
        ):
            service = BackendService(
                {
                    "vision": {
                        "enabled": True,
                        "source": "none",
                        "local_backend": "none",
                        "semantic_backend": "openai_compatible",
                        "semantic_endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                        "semantic_model": "local-vlm",
                    }
                },
                Path.cwd(),
            )
        semantic = service.vision.semantic
        self.assertIsNotNone(semantic)
        self.assertEqual(semantic.endpoint, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(semantic.model, "local-vlm")
        self.assertEqual(semantic.api_key, "session-secret")

    def test_main_llm_semantic_goal_reuses_existing_conversation_bridge(self) -> None:
        class Detector:
            name = "test_detector"

            def status(self):
                return {"available": True}

            def observe(self, frame, *, now):
                return VisionObservation(
                    entities=({
                        "id": "avatar:session:test:1",
                        "label": "person",
                        "confidence": 0.95,
                        "bbox": [0.2, 0.2, 0.5, 0.9],
                    },),
                    source=self.name,
                    observed_at=now,
                    frame_id="paired-frame",
                )

        service = BackendService(
            {
                "vision": {
                    "enabled": True,
                    "source": "none",
                    "local_backend": "none",
                    "semantic_backend": "main_llm",
                    "frame_cache_interval_s": 0,
                }
            },
            Path.cwd(),
        )
        try:
            service.vision.set_backends(detector=Detector())
            service.vision.set_capture_state(True, "test")
            world = service.vision.process_frame(b"jpeg")
            service.autonomy_arm()
            goal = service.autonomy_goal(
                "寻找地图 NPC",
                "explore",
                selector={"semantic_type": "npc", "min_confidence": 0.7},
                based_on_revision=world["status"]["revision"],
            )
            self.assertTrue(goal["accepted"])
            self.assertTrue(goal["semantic_request"]["accepted"])

            request = service.main_llm_semantic_request()
            self.assertTrue(request["available"])
            self.assertIn("data_base64", request)
            committed = service.main_llm_semantic_commit(
                request["request_id"],
                request["revision"],
                [{
                    "target_id": "avatar:session:test:1",
                    "semantic_type": "npc",
                    "label": "地图 NPC",
                    "confidence": 0.9,
                }],
            )
            self.assertTrue(committed["accepted"])
            self.assertEqual(committed["bindings"][0]["target_id"], "avatar:session:test:1")

            # 导航总时限是被动语义请求的最终边界：超时后仍保持手动 arm，
            # 但目标和内存中的图片单槽都必须立刻释放，不能每 30 秒重新送图。
            renewed = service.autonomy_goal(
                "继续寻找地图 NPC",
                "explore",
                selector={"semantic_type": "npc", "min_confidence": 0.7},
                based_on_revision=world["status"]["revision"],
            )
            self.assertTrue(renewed["semantic_request"]["accepted"])
            self.assertTrue(service.main_llm_semantic_request()["available"])
            service._navigator_complete_goal("explore_duration_exhausted")
            completed = service.autonomy_snapshot()
            self.assertTrue(completed["armed"])
            self.assertIsNone(completed["goal"])
            self.assertEqual(
                service.main_llm_semantic_request()["reason"],
                "no_pending_request",
            )
        finally:
            service.vision.close()

    def test_remote_autonomy_forwards_exact_target_id(self) -> None:
        calls = []

        class RecordingClient:
            def fast_request(self, method, path, payload):
                calls.append((method, path, payload))
                return {"accepted": True}

        result = RemoteAutonomy(RecordingClient()).goal(
            "follow that player",
            "follow",
            "openvino:track:7",
            {"semantic_type": "player"},
            {"max_duration_s": 30},
            41,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(calls, [(
            "POST",
            "/autonomy/goal",
            {
                "text": "follow that player",
                "kind": "follow",
                "target_id": "openvino:track:7",
                "selector": {"semantic_type": "player"},
                "constraints": {"max_duration_s": 30},
                "based_on_revision": 41,
            },
        )])

    def test_remote_autonomy_forwards_short_frame_reference(self) -> None:
        calls = []

        class RecordingClient:
            def fast_request(self, method, path, payload):
                calls.append((method, path, payload))
                return {"accepted": True}

        result = RemoteAutonomy(RecordingClient()).goal(
            "follow T2",
            "follow",
            based_on_revision=42,
            target_ref="T2",
            frame_revision=42,
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(calls, [(
            "POST",
            "/autonomy/goal",
            {
                "text": "follow T2",
                "kind": "follow",
                "based_on_revision": 42,
                "target_ref": "T2",
                "frame_revision": 42,
            },
        )])

    def test_autonomy_goal_resolves_frame_ref_and_rejects_fake_target_id(self) -> None:
        """LLM只选 T 编号；后端解析稳定 ID，并拒绝描述文本冒充 ID。"""
        try:
            from PIL import Image
        except Exception:  # pragma: no cover - 取决于可选视觉依赖
            self.skipTest("Pillow is not installed")

        class Detector:
            name = "test_detector"

            def status(self):
                return {"available": True}

            def observe(self, frame, *, now):
                return VisionObservation(
                    entities=(
                        {
                            "id": "avatar:session:test:1",
                            "label": "person",
                            "confidence": 0.91,
                            "bbox": [0.05, 0.1, 0.45, 0.9],
                        },
                        {
                            "id": "avatar:session:test:2",
                            "label": "person",
                            "confidence": 0.94,
                            "bbox": [0.55, 0.1, 0.95, 0.9],
                        },
                    ),
                    source=self.name,
                    observed_at=now,
                    frame_id="selection-frame",
                )

        buffer = BytesIO()
        Image.new("RGB", (160, 100), (10, 10, 10)).save(buffer, format="JPEG", quality=80)
        service = BackendService(
            {
                "vision": {
                    "enabled": True,
                    "source": "none",
                    "local_backend": "none",
                    "semantic_backend": "none",
                    "frame_cache_interval_s": 0,
                }
            },
            Path.cwd(),
        )
        try:
            service.vision.set_backends(detector=Detector())
            service.vision.set_capture_state(True, "test")
            world = service.vision.process_frame(buffer.getvalue())
            frame = service.vision_frame(max_age_ms=3000, overlay=True)
            revision = frame["overlay"]["revision"]
            mapping = {
                item["ref"]: item["target_id"]
                for item in frame["overlay"]["candidates"]
            }
            self.assertEqual(
                set(mapping.values()),
                {"avatar:session:test:1", "avatar:session:test:2"},
            )
            selected_ref = next(
                ref for ref, target in mapping.items()
                if target == "avatar:session:test:2"
            )

            service.autonomy_arm()
            fake = service.autonomy_goal(
                "走过去看看",
                "approach",
                target_id="person[visible](front/unknown, conf=0.94)",
                based_on_revision=world["status"]["revision"],
            )
            self.assertFalse(fake["accepted"])
            self.assertEqual(fake["reason_code"], "target_id_not_visible")

            selected = service.autonomy_goal(
                "走向右边的人",
                "approach",
                target_ref=selected_ref.lower(),
                frame_revision=revision,
                based_on_revision=world["status"]["revision"],
            )
            self.assertTrue(selected["accepted"], selected)
            self.assertEqual(selected["resolved_by"], "frame_target_ref")
            self.assertEqual(selected["resolved_target_ref"], selected_ref)
            self.assertEqual(selected["resolved_target_id"], "avatar:session:test:2")
            self.assertEqual(selected["goal"]["target_id"], "avatar:session:test:2")

            expired = service.autonomy_goal(
                "跟随右边的人",
                "follow",
                target_ref=selected_ref,
                frame_revision=revision + 999,
            )
            self.assertFalse(expired["accepted"])
            self.assertEqual(expired["reason_code"], "target_ref_revision_unavailable")
        finally:
            service.vision.close()

    def test_remote_autonomy_intent_uses_single_high_level_endpoint(self) -> None:
        calls = []

        class RecordingClient:
            def fast_request(self, method, path, payload):
                calls.append((method, path, payload))
                return {"accepted": True, "resolved_target_id": "avatar:session:test:1"}

        result = RemoteAutonomy(RecordingClient()).intent(
            "approach",
            text="走向地图 NPC",
            target_type="npc",
            min_confidence=0.25,
            constraints={"max_duration_s": 30},
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(calls[0][0:2], ("POST", "/autonomy/intent"))
        self.assertEqual(calls[0][2]["target_type"], "npc")
        self.assertEqual(calls[0][2]["constraints"], {"max_duration_s": 30})

    def test_autonomy_intent_requires_manual_arm_and_only_resolves_unique_semantic_target(self) -> None:
        service = BackendService({}, Path.cwd())
        unarmed = service.autonomy_intent("approach", target_type="npc")
        self.assertFalse(unarmed["accepted"])
        self.assertEqual(unarmed["reason_code"], "manual_arm_required")

        service.autonomy_arm()
        service.vision.set_capture_state(True, "test")
        service.world_state.ingest(
            entities=[{
                "id": "avatar:session:test:1",
                "label": "地图 NPC",
                "confidence": 0.88,
                "bbox": [0.55, 0.2, 0.8, 0.9],
                "attributes": {
                    "semantic_type": "npc",
                    "semantic_verified": True,
                    "bearing_deg": 18.0,
                    "apparent_height": 0.7,
                },
            }],
            source="main_llm_vlm",
        )
        approached = service.autonomy_intent(
            "approach",
            text="走过去看看",
            target_type="npc",
        )
        self.assertTrue(approached["accepted"])
        self.assertEqual(approached["resolved_by"], "unique_semantic_target")
        self.assertEqual(approached["resolved_target_id"], "avatar:session:test:1")
        self.assertEqual(approached["goal"]["kind"], "approach_observe")

        service.world_state.ingest(
            entities=[{
                "id": "avatar:session:test:2",
                "label": "另一个 NPC",
                "confidence": 0.91,
                "bbox": [0.1, 0.2, 0.3, 0.85],
                "attributes": {
                    "semantic_type": "npc",
                    "semantic_verified": True,
                    "bearing_deg": -22.0,
                    "apparent_height": 0.65,
                },
            }],
            source="main_llm_vlm",
        )
        ambiguous = service.autonomy_intent("follow", target_type="npc")
        self.assertFalse(ambiguous["accepted"])
        self.assertEqual(ambiguous["reason_code"], "target_choice_required")
        self.assertEqual(len(ambiguous["candidates"]), 2)

    def test_autonomy_intent_resumes_after_one_main_llm_semantic_commit(self) -> None:
        """未分类 person 只触发一次看图；分类后不需要 LLM 再发移动命令。"""
        class Detector:
            name = "test_detector"

            def status(self):
                return {"available": True}

            def observe(self, frame, *, now):
                return VisionObservation(
                    entities=({
                        "id": "avatar:session:test:pending",
                        "label": "person",
                        "confidence": 0.95,
                        "bbox": [0.2, 0.2, 0.5, 0.9],
                        "attributes": {
                            "bearing_deg": 12.0,
                            "apparent_height": 0.32,
                        },
                    },),
                    source=self.name,
                    observed_at=now,
                    frame_id="pending-navigation-frame",
                )

        service = BackendService(
            {
                "vision": {
                    "enabled": True,
                    "source": "none",
                    "local_backend": "none",
                    "semantic_backend": "main_llm",
                    "frame_cache_interval_s": 0,
                }
            },
            Path.cwd(),
        )
        try:
            service.vision.set_backends(detector=Detector())
            service.vision.set_capture_state(True, "test")
            service.vision.process_frame(b"jpeg")
            service.autonomy_arm()

            pending = service.autonomy_intent(
                "approach",
                text="走过去看看",
                target_type="npc",
            )
            self.assertTrue(pending["accepted"], pending)
            self.assertTrue(pending["pending_semantic"])
            self.assertFalse(pending["movement_started"])
            self.assertEqual(pending["reason_code"], "semantic_target_pending")
            self.assertIsNone(service.autonomy_snapshot()["goal"])

            request = service.main_llm_semantic_request()
            self.assertTrue(request["available"])
            self.assertEqual(request["reason"], "agent_navigation_target_unresolved")
            committed = service.main_llm_semantic_commit(
                request["request_id"],
                request["revision"],
                [{
                    "target_id": "avatar:session:test:pending",
                    "semantic_type": "npc",
                    "label": "地图 NPC",
                    "confidence": 0.91,
                }],
            )

            resumed = committed["pending_navigation"]
            self.assertTrue(resumed["accepted"], committed)
            self.assertTrue(resumed["movement_started"])
            self.assertEqual(resumed["resolved_by"], "main_llm_semantic_commit")
            self.assertEqual(resumed["goal"]["kind"], "approach_observe")
            self.assertEqual(
                resumed["goal"]["target_id"],
                "avatar:session:test:pending",
            )
            self.assertIsNone(service.autonomy_snapshot()["pending_semantic_intent"])
        finally:
            service.vision.close()

    def test_remote_vision_transports_main_llm_semantic_request_and_commit(self) -> None:
        calls = []

        class RecordingClient:
            def request(self, method, path, payload=None):
                calls.append((method, path, payload))
                return {"available": True} if method == "GET" else {"accepted": True}

        vision = RemoteVision(RecordingClient())
        request = vision.semantic_request("semantic-request:test:1")
        committed = vision.semantic_commit(
            "semantic-request:test:2",
            42,
            [{
                "target_id": "avatar:1",
                "semantic_type": "npc",
                "label": "NPC",
                "confidence": 0.9,
            }],
        )

        self.assertTrue(request["available"])
        self.assertTrue(committed["accepted"])
        self.assertEqual(
            calls[0],
            (
                "GET",
                "/semantic/request?after_request_id=semantic-request%3Atest%3A1",
                None,
            ),
        )
        self.assertEqual(calls[1][0:2], ("POST", "/semantic/commit"))
        self.assertEqual(calls[1][2]["frame_revision"], 42)
        self.assertEqual(calls[1][2]["entities"][0]["semantic_type"], "npc")

    def test_detector_interval_uses_override_only_for_resolved_accelerators(self) -> None:
        config = SimpleNamespace(
            detector_interval_ms=500,
            detector_accelerator_interval_ms=100,
        )

        def detector(runtime: str, resolved_device: str):
            return SimpleNamespace(status=lambda: {
                "runtime": runtime,
                "resolved_device": resolved_device,
            })

        self.assertEqual(
            _effective_detector_interval_ms(config, detector("openvino", "GPU.1")), 100
        )
        self.assertEqual(
            _effective_detector_interval_ms(config, detector("openvino", "NPU")), 100
        )
        self.assertEqual(
            _effective_detector_interval_ms(
                config, detector("onnxruntime_cuda", "CUDA.2")
            ),
            100,
        )
        self.assertEqual(
            _effective_detector_interval_ms(config, detector("openvino", "CPU")), 500
        )
        self.assertEqual(
            _effective_detector_interval_ms(config, detector("onnxruntime", "CPU")), 500
        )
        self.assertEqual(_effective_detector_interval_ms(config, None), 500)

    def test_remote_vision_keeps_lifecycle_publish_as_a_structured_transport_call(self) -> None:
        client = BackendClient({}, Path.cwd())
        invalid = client.vision.ingest(42)  # type: ignore[arg-type]
        self.assertFalse(invalid["accepted"])
        self.assertEqual(invalid["reason_code"], "invalid_world_observation")
        unavailable = client.vision.ingest({"remove_entity_ids": ["x"]})
        self.assertFalse(unavailable["accepted"])
        self.assertEqual(unavailable["reason_code"], "backend_unavailable")
        class SerializationFailure:
            def request(self, *_args, **_kwargs):
                raise TypeError("not JSON serializable")

        malformed = RemoteVision(SerializationFailure()).ingest({"attributes": object()})
        self.assertFalse(malformed["accepted"])
        self.assertEqual(malformed["reason_code"], "invalid_world_observation")

    def test_http_schema_errors_are_rejected_not_reported_as_offline(self) -> None:
        client = BackendClient({}, Path.cwd())
        client.port = 12345
        client.token = "test"
        error = HTTPError(
            "http://127.0.0.1:12345/world/ingest",
            400,
            "bad observation",
            {},
            BytesIO(b'{"error":"ValueError: remove_source is required"}'),
        )
        with patch("neko_anyadance_body.backend.client.urlopen", side_effect=error):
            with self.assertRaises(BackendRejected) as raised:
                client.request("POST", "/world/ingest", {})
        self.assertEqual(raised.exception.status_code, 400)

        class RejectingClient:
            def request(self, *_args, **_kwargs):
                raise BackendRejected("unauthorized", status_code=401)

        unauthorized = RemoteVision(RejectingClient()).ingest({})
        self.assertFalse(unauthorized["accepted"])
        self.assertEqual(unauthorized["reason_code"], "backend_unavailable")

        class SchemaRejectingClient:
            def request(self, *_args, **_kwargs):
                raise BackendRejected("bad schema", status_code=422)

        schema = RemoteVision(SchemaRejectingClient()).ingest({})
        self.assertFalse(schema["accepted"])
        self.assertEqual(schema["reason_code"], "invalid_world_observation")

    def test_remote_osc_config_invalid_values_fall_back_safely(self) -> None:
        client = BackendClient({"vrchat_osc": {"enabled": "no", "input_pulse_ms": "bad"}}, Path.cwd())
        self.assertTrue(client.osc_config.enabled)
        self.assertEqual(client.osc_config.input_pulse_ms, 100)

    def test_osc_service_propagates_failures_and_rejects_unsafe_values(self) -> None:
        service = BackendService({}, Path.cwd())

        class FailingOsc:
            def __init__(self) -> None:
                self.calls = []

            def set_axis(self, *args):
                self.calls.append(args)
                return False, "send failed"

            def stop_axes(self, axes):
                self.calls.append(("stop_axes", axes))
                return True, None

            def stop_all_axes(self):
                self.calls.append(("stop_all_axes",))
                return True, None

        osc = FailingOsc()
        service.osc = osc  # type: ignore[assignment]

        failed = service.set_locomotion(1.0, 0.0, 1000)
        self.assertEqual(failed, (False, "send failed"))
        self.assertIn(("stop_axes", ("move_vertical", "move_horizontal")), osc.calls)
        self.assertFalse(service.set_locomotion(float("nan"), 0.0, 1000)[0])
        self.assertFalse(service.set_locomotion(1.0, 0.0, -1)[0])
        self.assertFalse(service.set_turn(0.0, float("inf"))[0])
        self.assertFalse(service.pulse_input("grab", "right", float("nan"))[0])
        self.assertFalse(service.pulse_input("grab", "right", -1)[0])

    def test_osc_service_propagates_stop_result(self) -> None:
        service = BackendService({}, Path.cwd())

        class Osc:
            def stop_all_axes(self):
                return False, "zero packet failed"

        service.osc = Osc()  # type: ignore[assignment]
        self.assertEqual(service.stop_movement(), (False, "zero packet failed"))

    def test_movement_reaches_vrchat_osc_even_when_vmc_is_primary(self) -> None:
        """走位与转向必须落到 OSC，不能被 VMC 路由吃掉。

        VMC 覆盖层把摇杆值写成 avatar 手臂 pose：看着像在推摇杆，VRChat 收不到
        ``/input/*``，人原地不动，而调用方拿到 ``accepted: true``。这是 turn
        长期"接受但不转"的根因——修好之前，下面的 set_axis 断言一条都到不了。
        """
        class RecordingOsc:
            def __init__(self) -> None:
                self.axes: list[tuple[str, float, float]] = []

            def set_axis(self, name, value, duration_s):
                self.axes.append((name, value, duration_s))
                return True, None

            def stop_all_axes(self):
                return True, None

        class RecordingScheduler:
            def __init__(self) -> None:
                self.submits: list[str] = []
                self.params: list[dict] = []

            def submit(self, kind, _params=None):
                self.submits.append(kind)
                self.params.append(dict(_params or {}))
                return {"accepted": True, "reason": None}

        service = BackendService({"input": {"primary": "anyadance"}}, Path.cwd())
        osc = RecordingOsc()
        scheduler = RecordingScheduler()
        service.osc = osc  # type: ignore[assignment]
        service.scheduler = scheduler  # type: ignore[assignment]

        self.assertEqual(service.set_turn(0.5, 500), (True, None))
        self.assertEqual(service.set_locomotion(1.0, 0.0, 1000), (True, None))

        sent = {name for name, _value, _duration in osc.axes}
        self.assertIn("move_vertical", sent)
        # 转向不能走 OSC：VR 模式下 look 轴是死地址，发了照样回 accepted。
        self.assertNotIn("look_horizontal", sent)
        # 它走调度器，直接转虚拟 HMD。
        self.assertIn("turn", scheduler.submits)
        # 摇杆轴一次都不该进调度器：那条路服务待机动作，不是游戏输入。
        self.assertNotIn("input_axes", scheduler.submits)

    def test_stop_movement_does_not_hide_an_osc_failure_behind_the_vmc_release(self) -> None:
        """OSC 停不下来就必须报失败，哪怕 VMC 覆盖层清干净了。

        移动轴现在只经由 OSC，只有它能真正让人停下。把 VMC 的成功当成整体成功，
        等于把"还在走"报成"已停住"，而调用方不会重试。
        """
        class FailingStopOsc:
            def stop_all_axes(self):
                return False, "zero packet failed"

        class AcceptingScheduler:
            def submit(self, _kind, _params=None):
                return {"accepted": True, "reason": None}

        service = BackendService({"input": {"primary": "anyadance"}}, Path.cwd())
        service.osc = FailingStopOsc()  # type: ignore[assignment]
        service.scheduler = AcceptingScheduler()  # type: ignore[assignment]

        self.assertEqual(service.stop_movement(), (False, "zero packet failed"))

    def test_movement_without_its_transport_fails_instead_of_reporting_success(self) -> None:
        """走位和转向各有各的通道，缺谁就该谁失败，不能互相顶替。

        走位只能靠 OSC；转向只能靠 AnyaDance，因为 VR 模式下 look 轴根本不生效，
        回落到 OSC 只会把"没转"重新包装成成功。
        """
        class AcceptingScheduler:
            def submit(self, _kind, _params=None):
                return {"accepted": True, "reason": None}

        service = BackendService({"input": {"primary": "anyadance"}}, Path.cwd())
        service.osc = None  # type: ignore[assignment]
        service.scheduler = AcceptingScheduler()  # type: ignore[assignment]

        # 转向不碰 OSC，所以 OSC 缺席也照样能转。
        self.assertEqual(service.set_turn(0.5, 500), (True, None))
        self.assertFalse(service.set_locomotion(1.0, 0.0, 1000)[0])

        service.scheduler = None  # type: ignore[assignment]
        turn_ok, turn_reason = service.set_turn(0.5, 500)
        self.assertFalse(turn_ok)
        self.assertIn("scheduler", turn_reason or "")

    def test_navigator_turn_uses_current_yaw_correction_semantics(self) -> None:
        # scheduler 的 delta 会叠加到旧 target；导航闭环必须基于当前实际 yaw 生成
        # 新的绝对目标，才能在上一段尚未结束时连续修正而不超调。
        class HeadingScheduler:
            def __init__(self) -> None:
                self.params: list[dict] = []

            def submit(self, kind, params):
                self.assert_kind = kind
                self.params.append(dict(params))
                return {"accepted": True}

        service = BackendService({"input": {"primary": "anyadance"}}, Path.cwd())
        scheduler = HeadingScheduler()
        service.scheduler = scheduler  # type: ignore[assignment]
        self.assertTrue(service._navigator_send_turn(20.0))
        self.assertEqual(scheduler.params, [{"correction_deg": 20.0}])
        self.assertTrue(service.navigator.snapshot()["turn"]["continuous_retarget"])

    def test_osc_batch_is_bounded_and_records_dispatch_latency(self) -> None:
        service = BackendService({}, Path.cwd())

        class Osc:
            def __init__(self) -> None:
                self.calls = []

            def set_axes(self, values, duration):
                self.calls.append(("axes", values, duration))
                return True, None

            def set_axis(self, axis, value, duration):
                self.calls.append(("axis", axis, value, duration))
                return True, None

            def stop_all_axes(self):
                self.calls.append(("stop",))
                return True, None

            def send_parameter(self, name, value):
                self.calls.append(("parameter", name, value))
                return True, None

        osc = Osc()
        service.osc = osc  # type: ignore[assignment]

        class AcceptingScheduler:
            def submit(self, _kind, _params=None):
                return {"accepted": True, "reason": None}

        service.scheduler = AcceptingScheduler()  # type: ignore[assignment]
        result = service.send_osc_batch([
            {"kind": "locomotion", "vertical": 0.4, "horizontal": 0.0, "duration_ms": 500},
            {"kind": "turn", "horizontal": -0.2, "duration_ms": 300},
            {"kind": "parameter", "name": "NEKO_Action", "value": 1},
        ])
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["results"]), 3)
        # 转向不再产生 OSC 调用，它走调度器。
        self.assertEqual(osc.calls[0][0], "axes")
        self.assertEqual(osc.calls[1][0], "parameter")

        self.assertFalse(service.send_osc_batch([{"kind": "noop"}] * 9)["accepted"])
        service.record_control_dispatch("/osc/batch", 0.0)
        metrics = service.control_metrics_snapshot()
        self.assertEqual(metrics["count"], 1)
        self.assertEqual(metrics["last_operation"], "/osc/batch")
        self.assertIn("/osc/batch", metrics["by_operation"])

    def test_scheduler_snapshot_contains_safe_awareness_when_backend_is_down(self) -> None:
        class DownClient:
            def request(self, *_args, **_kwargs):
                raise BackendUnavailable("offline")

        snapshot = RemoteScheduler(DownClient()).snapshot()
        self.assertEqual(snapshot["state"], "backend_unavailable")
        self.assertIn("awareness", snapshot)

    def test_expanded_clip_duration_limit_is_enforced_in_backend(self) -> None:
        service = BackendService({}, Path.cwd())
        service.config = replace(service.config, clip_max_duration_seconds=3.0)

        class FakeScheduler:
            def snapshot(self):
                return {"state": "idle", "safety_state": "normal"}

            def submit(self, *_args, **_kwargs):
                raise AssertionError("overlong clip must be rejected before scheduling")

        service.scheduler = FakeScheduler()
        clip = SimpleNamespace(is_pose=False, duration_s=2.0, name="long")
        result = service.submit(
            "play_clip",
            {"clip_name": "long", "speed": 1.0, "loop_count": 2, "_clip": clip},
        )
        self.assertFalse(result["accepted"])
        self.assertIn("must not exceed", result["reason"])

    def test_backend_submit_gates_declared_world_preconditions(self) -> None:
        service = BackendService({}, Path.cwd())

        class FakeScheduler:
            def __init__(self):
                self.submissions = []

            def snapshot(self):
                return {"state": "idle", "safety_state": "normal"}

            def submit(self, kind, params):
                self.submissions.append((kind, params))
                return {
                    "accepted": True,
                    "action_id": "a-1",
                    "state": "active",
                    "normalized_params": params,
                    "reason": None,
                    "safety_state": "normal",
                }

        scheduler = FakeScheduler()
        service.scheduler = scheduler
        condition = [{
            "kind": "entity_visible",
            "entity_id": "yolo:cup:7",
            "source": "yolo",
            "min_confidence": 0.8,
            "max_age_ms": 500,
        }]
        rejected = service.submit(
            "reach_and_grab",
            {"side": "right"},
            preconditions=condition,
        )
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["reason_code"], "world_precondition_failed")
        self.assertTrue(rejected["replan_required"])
        self.assertEqual(
            rejected["precondition_check"]["failures"][0]["code"],
            "entity_not_visible",
        )
        self.assertEqual(scheduler.submissions, [])
        feedback = service.cognition.snapshot()["feedback"][-1]
        self.assertEqual(feedback["reason_code"], "world_precondition_failed")
        self.assertEqual(
            feedback["precondition_check"]["failures"][0]["code"],
            "entity_not_visible",
        )

        conflicting_alias = service.submit(
            "reach_and_grab",
            {"side": "right", "_world_preconditions": []},
            preconditions=condition,
        )
        self.assertFalse(conflicting_alias["accepted"])
        self.assertEqual(
            conflicting_alias["precondition_check"]["failures"][0]["code"],
            "invalid_world_precondition",
        )
        self.assertEqual(scheduler.submissions, [])

        stopped = service.submit("stop", {}, preconditions=condition)
        self.assertTrue(stopped["accepted"])
        self.assertTrue(stopped["precondition_check"]["bypassed"])
        self.assertEqual(len(scheduler.submissions), 1)

        service.ingest_world({
            "source": "yolo",
            "entities": [{
                "id": "yolo:cup:7",
                "label": "cup",
                "confidence": 0.95,
                "source": ["yolo"],
                "ttl_s": 2.0,
            }],
        })
        accepted = service.submit(
            "reach_and_grab",
            {"side": "right"},
            preconditions=condition,
        )
        self.assertTrue(accepted["accepted"])
        self.assertTrue(accepted["precondition_check"]["passed"])
        self.assertEqual(len(scheduler.submissions), 2)

    def test_injected_vision_worker_is_owned_by_backend_lifecycle(self) -> None:
        class Source:
            name = "fake_source"

            def __init__(self):
                self.closed = False
                self.count = 0

            def status(self):
                return {"available": not self.closed}

            def read(self):
                if self.closed:
                    return None
                self.count += 1
                return self.count

            def close(self):
                self.closed = True

        class Detector:
            name = "fake_detector"

            def __init__(self):
                self.event = __import__("threading").Event()

            def status(self):
                return {"available": True, "backend": "test"}

            def observe(self, frame, *, now):
                self.event.set()
                return VisionObservation(
                    entities=({
                        "track_id": 1,
                        "label": "button",
                        "confidence": 0.9,
                        "ttl_s": 0.5,
                    },),
                    source="fake_detector",
                    observed_at=now,
                )

        source = Source()
        detector = Detector()
        service = BackendService(
            {
                "vision": {"enabled": True, "interval_ms": 10},
                "vmc_idle": {"enabled": False, "manage_host_output": False},
                "vrchat_osc": {"enabled": False},
                "driver_log": {"enabled": False},
            },
            Path.cwd(),
            dry_run=True,
            vision_source=source,
            vision_detector=detector,
        )
        try:
            service.start()
            self.assertTrue(detector.event.wait(1.0))
            deadline = time.monotonic() + 1.0
            snapshot = service.snapshot()
            while snapshot["vision_worker"]["frames_processed"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
                snapshot = service.snapshot()
            self.assertTrue(snapshot["world"]["available"])
            self.assertGreaterEqual(snapshot["vision_worker"]["frames_processed"], 1)
            self.assertEqual(snapshot["world"]["entities"][0]["id"], "fake_detector:button:1")
        finally:
            service.stop()
        self.assertTrue(source.closed)

    def test_vision_stop_start_rebuilds_closed_source(self) -> None:
        class Source:
            name = "restartable_test_source"

            def __init__(self, serial: int):
                self.serial = serial
                self.closed = False
                self.count = 0

            def status(self):
                return {
                    "available": not self.closed,
                    "name": self.name,
                    "serial": self.serial,
                }

            def read(self):
                if self.closed:
                    return None
                self.count += 1
                return self.count

            def close(self):
                self.closed = True

        class Detector:
            name = "restart_detector"

            def status(self):
                return {"available": True, "backend": "test"}

            def observe(self, frame, *, now):
                return VisionObservation(
                    entities=({
                        "id": "restart:marker",
                        "label": "marker",
                        "confidence": 0.9,
                        "ttl_s": 1.0,
                    },),
                    source="restart_detector",
                    observed_at=now,
                )

        created: list[Source] = []

        def factory() -> Source:
            source = Source(len(created) + 1)
            created.append(source)
            return source

        service = BackendService(
            {
                "vision": {
                    "enabled": True,
                    "source": "external",
                    "capture": "external",
                    "local_backend": "none",
                    "semantic_backend": "none",
                    "interval_ms": 10,
                },
                "vmc_idle": {"enabled": False, "manage_host_output": False},
                "vrchat_osc": {"enabled": False},
                "driver_log": {"enabled": False},
            },
            Path.cwd(),
            dry_run=True,
            vision_detector=Detector(),
            vision_source_factory=factory,
        )
        try:
            service.start()
            deadline = time.monotonic() + 1.0
            while not created or created[0].count < 1:
                if time.monotonic() >= deadline:
                    self.fail("initial vision source did not capture")
                time.sleep(0.01)
            first = created[0]
            stopped = service.vision_stop("test_stop")
            self.assertTrue(stopped["accepted"])
            self.assertFalse(stopped["running"])
            self.assertTrue(first.closed)
            self.assertEqual(service.perception()["worker"]["reason"], "test_stop")

            started = service.vision_start()
            self.assertTrue(started["accepted"])
            self.assertTrue(started["running"])
            self.assertGreaterEqual(len(created), 2)
            second = created[-1]
            self.assertIsNot(first, second)
            self.assertFalse(second.closed)
            deadline = time.monotonic() + 1.0
            while second.count < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(second.count, 1)
        finally:
            service.stop()

    def test_remote_vision_lifecycle_methods_use_control_routes(self) -> None:
        class FakeClient:
            def __init__(self):
                self.calls = []

            def fast_request(self, method, path, payload):
                self.calls.append((method, path, payload))
                return {"accepted": True, "running": path.endswith("start")}

        fake = FakeClient()
        remote = RemoteVision(fake)
        self.assertTrue(remote.start()["accepted"])
        self.assertTrue(remote.stop("user")["accepted"])
        self.assertEqual(
            fake.calls,
            [
                ("POST", "/vision/start", {}),
                ("POST", "/vision/stop", {"reason": "user"}),
            ],
        )

    def test_unavailable_optional_detector_keeps_capture_only_worker_running(self) -> None:
        class Source:
            name = "capture_only_source"

            def __init__(self):
                self.closed = False
                self.frames = 0

            def status(self):
                return {"available": not self.closed, "name": self.name}

            def read(self):
                if self.closed:
                    return None
                self.frames += 1
                return object()

            def close(self):
                self.closed = True

        class UnavailableDetector:
            name = "optional_detector"

            def status(self):
                return {"available": False, "last_error": "model is not installed"}

            def observe(self, _frame, *, now):
                raise AssertionError("unavailable detector must not be invoked")

        sources: list[Source] = []

        def factory() -> Source:
            source = Source()
            sources.append(source)
            return source

        service = BackendService(
            {
                "vision": {
                    "enabled": True,
                    "source": "external",
                    "capture": "external",
                    "local_backend": "none",
                    "semantic_backend": "none",
                    "interval_ms": 10,
                },
                "vmc_idle": {"enabled": False, "manage_host_output": False},
                "vrchat_osc": {"enabled": False},
                "driver_log": {"enabled": False},
            },
            Path.cwd(),
            dry_run=True,
            vision_detector=UnavailableDetector(),
            vision_source_factory=factory,
        )
        try:
            service.start()
            deadline = time.monotonic() + 1.0
            while (not sources or sources[0].frames < 1) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(sources)
            self.assertGreaterEqual(sources[0].frames, 1)
            worker = service.perception()["worker"]
            self.assertTrue(worker["running"])
            self.assertTrue(worker["capture_only"])
        finally:
            service.stop()

    def test_standalone_backend_process_health_and_shutdown(self) -> None:
        root = Path(__file__).resolve().parents[1]
        client = BackendClient(
            {
                "anyadance": {"port": 39570, "rate_hz": 60},
                "vmc_idle": {"enabled": False, "manage_host_output": False},
                "vrchat_osc": {"enabled": False},
                "driver_log": {"enabled": False},
            },
            root,
        )
        try:
            client.start(timeout_s=8.0)
            self.assertTrue(client.request("GET", "/health")["ok"])
            self.assertTrue(client.fast_request("GET", "/health")["ok"])
            fast_connection = client._fast_connection
            self.assertIsNotNone(fast_connection)
            self.assertTrue(client.fast_request("GET", "/health")["ok"])
            self.assertIs(client._fast_connection, fast_connection)
            batch = client.request(
                "POST",
                "/osc/batch",
                {"commands": [{"kind": "stop_movement"}]},
            )
            self.assertIn("dispatch_latency_ms", batch)
            self.assertEqual(client.scheduler.snapshot()["state"], "disabled")
            self.assertFalse(client.vision.snapshot()["available"])
            vision_start = client.vision.start()
            self.assertFalse(vision_start["accepted"])
            self.assertIn("disabled", vision_start["reason"])
            vision_stop = client.vision.stop("integration_test")
            self.assertTrue(vision_stop["accepted"])
            self.assertFalse(vision_stop["running"])
            invalid_lifecycle = client.vision.ingest({"remove_entity_ids": ["button"]})
            self.assertFalse(invalid_lifecycle["accepted"])
            self.assertEqual(invalid_lifecycle["reason_code"], "invalid_world_observation")
            self.assertEqual(invalid_lifecycle["status_code"], 400)
            world = client.request(
                "POST",
                "/world/ingest",
                {
                    "source": "test_detector",
                    "entities": [
                        {"id": "button", "label": "button", "confidence": 0.9},
                        {"id": "avatar", "label": "avatar"},
                        {
                            "id": "vrchat:player:usr_1",
                            "label": "player",
                            "source": "vrchat_log",
                            "confidence": 0.95,
                        },
                    ],
                },
            )
            self.assertTrue(world["available"])
            self.assertIn("button", {item["id"] for item in world["entities"]})
            left = client.vision.ingest(
                {
                    "source": "vrchat_log",
                    "events": [{
                        "type": "player_left",
                        "target_id": "vrchat:player:usr_1",
                        "confidence": 1.0,
                    }],
                    "remove_entity_ids": ["vrchat:player:usr_1"],
                    "remove_source": "vrchat_log",
                },
                ack_only=True,
            )
            self.assertTrue(left["accepted"])
            self.assertEqual(left["removed_entity_ids"], ["vrchat:player:usr_1"])
            self.assertNotIn(
                "vrchat:player:usr_1",
                {item["id"] for item in client.vision.snapshot()["entities"]},
            )
            ack = client.request(
                "POST",
                "/world/ingest",
                {
                    "source": "test_detector",
                    "frame_id": "frame-ack",
                    "entities": [{"id": "button", "label": "button", "confidence": 0.9}],
                    "ack_only": True,
                },
            )
            self.assertTrue(ack["accepted"])
            self.assertEqual(ack["frame_id"], "frame-ack")
            self.assertNotIn("entities", ack)
            gated = client.scheduler.submit(
                "arm_pose",
                {"side": "right", "elevation_deg": 90},
                preconditions=[{
                    "kind": "entity_visible",
                    "entity_id": "missing-target",
                    "min_confidence": 0.8,
                }],
            )
            self.assertFalse(gated["accepted"])
            self.assertEqual(gated["reason_code"], "world_precondition_failed")
            malformed_gate = client.scheduler.submit(
                "arm_pose",
                {"side": "right", "elevation_deg": 90},
                preconditions=42,
            )
            self.assertFalse(malformed_gate["accepted"])
            self.assertEqual(
                malformed_gate["precondition_check"]["failures"][0]["code"],
                "invalid_world_precondition",
            )
            plan = client.request(
                "POST",
                "/cognition/plan",
                {
                    "goal": "raise hand",
                    "action": "arm_pose",
                    "params": {"side": "right"},
                    "preconditions": [{
                        "kind": "entity_visible",
                        "entity_id": "button",
                        "min_confidence": 0.8,
                    }],
                },
            )
            self.assertEqual(plan["status"], "planned")
            self.assertTrue(plan["precondition_check"]["passed"])
            cognition = client.request("GET", "/cognition")
            self.assertEqual(cognition["plan"]["id"], plan["id"])
            self.assertEqual(cognition["state"]["mode"], "nominal")
            self.assertGreaterEqual(
                cognition["state"]["sources"]["world"]["runtime"]["entity_count"],
                1,
            )
            feedback = client.request(
                "POST",
                "/cognition/feedback",
                {"type": "world_changed", "data": {"entity": "button"}},
            )
            self.assertTrue(feedback["replan_required"])
            self.assertEqual(feedback["replan_reason"], "world_changed")
            rejected = client.scheduler.submit("play_clip", {"clip_name": "does-not-exist"})
            self.assertFalse(rejected["accepted"])
            self.assertIn("unknown preset clip", rejected["reason"])
        finally:
            client.stop()
        self.assertIsNone(client.process)

    def test_process_and_debug_cli_run_without_plugin_sdk(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "backend.toml"
            config_file.write_text(
                "[vmc_idle]\nenabled=false\nmanage_host_output=false\n"
                "[vrchat_osc]\nenabled=false\n[driver_log]\nenabled=false\n",
                encoding="utf-8",
            )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                port = int(probe.getsockname()[1])
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "backend" / "process.py"),
                    "--config-file", str(config_file),
                    "--config-dir", temp_dir,
                    "--offline",
                    "--port", str(port),
                    "--token", "dev",
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
            try:
                health = None
                for _ in range(60):
                    if process.poll() is not None:
                        break
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(root / "backend" / "debug_cli.py"),
                            "--port", str(port),
                            "--token", "dev",
                            "health",
                        ],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        creationflags=creationflags,
                    )
                    if result.returncode == 0:
                        health = json.loads(result.stdout)
                        break
                    time.sleep(0.1)
                self.assertIsNotNone(health)
                self.assertTrue(health["ok"])
                action = subprocess.run(
                    [
                        sys.executable,
                        str(root / "backend" / "debug_cli.py"),
                        "--port", str(port),
                        "--token", "dev",
                        "action", "--kind", "enable", "--json", "{}",
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                )
                self.assertEqual(action.returncode, 0, action.stderr)
                snapshot = subprocess.run(
                    [
                        sys.executable,
                        str(root / "backend" / "debug_cli.py"),
                        "--port", str(port),
                        "--token", "dev",
                        "snapshot",
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                )
                self.assertEqual(snapshot.returncode, 0, snapshot.stderr)
                snapshot_data = json.loads(snapshot.stdout)
                self.assertTrue(snapshot_data["backend"]["dry_run"])
                self.assertIsNone(snapshot_data["body"]["udp"]["local_port"])
            finally:
                subprocess.run(
                    [
                        sys.executable,
                        str(root / "backend" / "debug_cli.py"),
                        "--port", str(port),
                        "--token", "dev",
                        "shutdown",
                    ],
                    cwd=root,
                    capture_output=True,
                    creationflags=creationflags,
                )
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
                if process.stderr is not None:
                    process.stderr.close()


if __name__ == "__main__":
    unittest.main()
