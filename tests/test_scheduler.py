from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.model import neutral_frame
from neko_anyadance_body.nya import ClipLibrary
from neko_anyadance_body.scheduler import BodyScheduler, advance_deadline


class FakeTransport:
    def __init__(self, fail_after: int | None = None) -> None:
        self.lock = threading.Lock()
        self.packets: list[tuple[float, bytes, tuple[str, int]]] = []
        self.closed = False
        self.fail_after = fail_after

    def send(self, payload: bytes, address: tuple[str, int]) -> None:
        with self.lock:
            if self.fail_after is not None and len(self.packets) >= self.fail_after:
                raise OSError("synthetic UDP failure")
            self.packets.append((time.perf_counter(), payload, address))

    def close(self) -> None:
        self.closed = True

    @property
    def local_port(self) -> int | None:
        return None

    def count(self) -> int:
        with self.lock:
            return len(self.packets)

    def latest_payload(self) -> dict:
        with self.lock:
            return json.loads(self.packets[-1][1])


class FakeIdleFrameSource:
    def __init__(self) -> None:
        self.frame = neutral_frame()
        self.frame.devices["hmd"].position = (0.12, 1.42, -0.08)

    def latest_frame(self):
        return self.frame.clone()

    def snapshot(self):
        return {
            "enabled": True,
            "listen_address": "127.0.0.1:39539",
            "connection": "detected",
            "source_available": True,
        }


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeTransport()
        self.scheduler = BodyScheduler(PluginConfig(), transport=self.transport)
        self.scheduler.start()

    def tearDown(self) -> None:
        self.scheduler.shutdown()

    def enable(self) -> None:
        result = self.scheduler.submit("enable")
        self.assertTrue(result["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "idle")
        wait_until(lambda: self.transport.count() >= 3)

    def test_absolute_deadline_skips_expired_slots(self) -> None:
        deadline, skipped = advance_deadline(10.0, 10.025, 0.01)
        self.assertAlmostEqual(deadline, 10.03)
        self.assertEqual(skipped, 2)

    def test_disabled_rejects_motion(self) -> None:
        result = self.scheduler.submit("arm_pose", {
            "side": "right", "elevation_deg": 160.0, "plane": "front", "reach": 0.9,
            "palm": "neutral", "duration_ms": 200,
        })
        self.assertFalse(result["accepted"])
        self.assertEqual(result["state"], "disabled")

    def test_idle_state_relays_host_vmc_frame(self) -> None:
        self.scheduler.shutdown()
        source = FakeIdleFrameSource()
        self.transport = FakeTransport()
        self.scheduler = BodyScheduler(
            PluginConfig(),
            transport=self.transport,
            idle_frame_source=source,
        )
        self.scheduler.start()
        self.enable()
        wait_until(lambda: self.scheduler.snapshot()["idle_relay"]["applied"])
        payload = self.transport.latest_payload()
        self.assertEqual(payload["devices"]["hmd"]["pose"]["position"], [0.12, 1.42, -0.08])
        status = self.scheduler.snapshot()
        self.assertEqual(status["idle_relay"]["connection"], "detected")
        self.assertTrue(status["awareness"]["idle_relay"]["applied"])
        self.assertIn("VMC 待机姿态", status["awareness"]["summary"])

    def test_stop_is_accepted_even_when_normal_queue_is_full(self) -> None:
        self.enable()
        params = {
            "side": "right", "elevation_deg": 90.0, "plane": "front", "reach": 0.9,
            "palm": "neutral", "duration_ms": 500,
        }
        accepted = 0
        for _ in range(self.scheduler.config.max_queue_size + 2):
            if self.scheduler.submit("arm_pose", params)["accepted"]:
                accepted += 1
        self.assertGreaterEqual(accepted, 1)
        stopped = self.scheduler.submit("stop")
        self.assertTrue(stopped["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "stopped_latched")
        self.assertEqual(self.scheduler.snapshot()["queue_length"], 0)
        enable_again = self.scheduler.submit("enable")
        self.assertFalse(enable_again["accepted"])
        self.assertIn("body_reset", enable_again["reason"])

    def test_malformed_direct_command_does_not_latch_scheduler_fault(self) -> None:
        self.enable()
        result = self.scheduler.submit("arm_pose", {"side": "right"})
        self.assertTrue(result["accepted"])
        wait_until(
            lambda: (
                self.scheduler.snapshot()["behavior"]["last_decision"] is not None
                and self.scheduler.snapshot()["behavior"]["last_decision"].get("id") == result["action_id"]
            )
        )
        snapshot = self.scheduler.snapshot()
        self.assertNotEqual(snapshot["state"], "fault_latched")
        self.assertIn("rejected", snapshot["last_error"])

    def test_arm_pose_holds_and_status_reports_angle(self) -> None:
        self.enable()
        result = self.scheduler.submit("arm_pose", {
            "side": "right", "elevation_deg": 160.0, "plane": "front", "reach": 0.9,
            "palm": "neutral", "duration_ms": 200,
        })
        self.assertTrue(result["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        status = self.scheduler.snapshot()
        self.assertEqual(status["arms"]["right"]["elevation_deg"], 160.0)
        self.assertEqual(status["current_action"]["name"], "arm_pose")

    def test_awareness_reports_semantic_pose_and_completion(self) -> None:
        self.enable()
        result = self.scheduler.submit("arm_pose", {
            "side": "right", "elevation_deg": 160.0, "azimuth_deg": 0.0, "plane": None,
            "reach": 0.9, "palm": "neutral", "duration_ms": 200,
        })
        self.assertTrue(result["accepted"])
        self.assertFalse(result["target_pose_summary"]["completion_confirmed"])
        self.assertIn("160", result["target_pose_summary"]["description"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        awareness = self.scheduler.snapshot()["awareness"]
        self.assertEqual(awareness["motion"]["id"], result["action_id"])
        self.assertEqual(awareness["motion"]["phase"], "holding")
        self.assertTrue(awareness["motion"]["completed"])
        self.assertAlmostEqual(awareness["pose"]["right_arm"]["elevation_deg"], 160.0, delta=1.0)
        self.assertEqual(awareness["pose"]["right_arm"]["direction"], "forward")
        self.assertIn("正在保持", awareness["summary"])

    def test_awareness_tracks_motion_progress_and_action_transition(self) -> None:
        self.enable()
        first = self.scheduler.submit("hand", {
            "side": "right", "pose": "fist", "strength": 1.0, "duration_ms": 100,
        })
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        second = self.scheduler.submit("gesture", {"name": "wave", "side": "right", "intensity": 0.8})
        wait_until(lambda: self.scheduler.snapshot()["state"] == "moving")
        awareness = self.scheduler.snapshot()["awareness"]
        self.assertEqual(awareness["motion"]["id"], second["action_id"])
        self.assertEqual(awareness["motion"]["source"], "gesture")
        self.assertGreater(awareness["motion"]["remaining_seconds"], 0.0)
        self.assertEqual(awareness["previous_action"]["id"], first["action_id"])
        self.assertEqual(awareness["previous_action"]["outcome"], "interrupted")
        self.assertEqual(awareness["transition"]["to"]["name"], "wave")

    def test_reach_closes_grip_only_at_end(self) -> None:
        self.enable()
        result = self.scheduler.submit("reach_and_grab", {
            "side": "right", "height": "chest", "direction": "forward", "distance_m": 0.35,
            "duration_ms": 400,
        })
        self.assertTrue(result["accepted"])
        time.sleep(0.15)
        mid = self.transport.latest_payload()["inputs"]["right_controller"]
        self.assertFalse(mid["grip_click"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        final = self.transport.latest_payload()["inputs"]["right_controller"]
        self.assertTrue(final["grip_click"])
        self.assertEqual(final["grip_value"], 1.0)

    def test_reach_start_callback_receives_safe_duration(self) -> None:
        self.scheduler.shutdown()
        started: list[float] = []
        self.transport = FakeTransport()
        self.scheduler = BodyScheduler(
            PluginConfig(),
            transport=self.transport,
            motion_started_callback=lambda _command, duration: started.append(duration),
        )
        self.scheduler.start()
        self.enable()
        result = self.scheduler.submit("reach_and_grab", {
            "side": "right", "height": "chest", "direction": "forward", "distance_m": 0.70,
            "duration_ms": 100,
        })
        self.assertTrue(result["accepted"])
        wait_until(lambda: bool(started))
        self.assertGreater(started[0], 0.1)

    def test_expanded_sequence_limit_does_not_fault_scheduler(self) -> None:
        self.enable()
        def step(x: float) -> dict:
            return {
                "type": "move_hand", "side": "right", "relative_to": "hmd",
                "x_m": x, "y_m": -1.0, "z_m": -1.0, "palm": "neutral",
                "wrist_pitch_deg": 0.0, "wrist_yaw_deg": 0.0, "wrist_roll_deg": 0.0,
                "duration_ms": 100,
            }
        result = self.scheduler.submit("sequence", {
            "steps": [step(-1.0 if index % 2 == 0 else 1.0) for index in range(16)],
            "loop_count": 4,
        })
        self.assertTrue(result["accepted"])
        wait_until(lambda: (
            self.scheduler.snapshot()["behavior"]["last_decision"] or {}
        ).get("accepted") is False)
        snapshot = self.scheduler.snapshot()
        self.assertEqual(snapshot["safety_state"], "normal")
        self.assertIn("expanded sequence duration", snapshot["behavior"]["last_decision"]["reason"])

    def test_semantic_expression_uses_overlay_layer_and_returns_to_idle(self) -> None:
        self.enable()
        result = self.scheduler.submit("express", {
            "intent": "explain",
            "intent_label": "解释",
            "gesture": "explain",
            "side": "right",
            "energy": 0.4,
            "duration_ms": 600,
            "head_only": False,
        })
        self.assertTrue(result["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["behavior"]["mode"] == "expressing")
        active = self.scheduler.snapshot()
        self.assertEqual(active["state"], "idle")
        self.assertIsNone(active["current_action"])
        self.assertEqual(active["behavior"]["overlays"][0]["params"]["intent"], "explain")
        self.assertEqual(active["expression_motion"]["active"][0]["source"], "llm_intent")
        wait_until(lambda: self.scheduler.snapshot()["behavior"]["mode"] == "idle")

    def test_semantic_vmd_uses_protected_base_layer_and_reports_intent(self) -> None:
        self.enable()
        root = Path(__file__).resolve().parents[1]
        clip = ClipLibrary(root / "tests" / "fixtures", self.scheduler.config).load("sample_clip")
        result = self.scheduler.submit("semantic_clip", {
            "clip_name": clip.name,
            "speed": 3.0,
            "loop_count": 1,
            "transition_ms": 100,
            "anchor": True,
            "restore_after": True,
            "semantic_intent": "greet",
            "intent_label": "问候",
            "motion_source": "vmd_bake",
            "motion_label": "测试问候",
            "source_name": "test.vmd",
            "_clip": clip,
        })
        self.assertTrue(result["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["behavior"]["mode"] == "clip_expression")
        snapshot = self.scheduler.snapshot()
        self.assertEqual(snapshot["current_action"]["name"], "semantic_clip")
        self.assertEqual(snapshot["awareness"]["motion"]["source"], "semantic_vmd")
        self.assertEqual(snapshot["awareness"]["motion"]["semantic_intent"], "greet")
        wait_until(lambda: self.scheduler.snapshot()["state"] == "idle")

    def test_clip_protects_full_body_expression_but_allows_head_overlay(self) -> None:
        self.enable()
        root = Path(__file__).resolve().parents[1]
        clip = ClipLibrary(root / "tests" / "fixtures", self.scheduler.config).load("sample_clip")
        played = self.scheduler.submit("play_clip", {
            "clip_name": clip.name,
            "speed": 1.0,
            "loop_count": 1,
            "transition_ms": 100,
            "anchor": True,
            "restore_after": False,
            "_clip": clip,
        })
        self.assertTrue(played["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["behavior"]["mode"] == "clip")
        blocked = self.scheduler.submit("express", {
            "intent": "explain", "gesture": "explain", "side": "right",
            "energy": 0.4, "duration_ms": 700, "head_only": False,
        })
        self.assertFalse(blocked["accepted"])
        self.assertIn("protected", blocked["reason"])
        allowed = self.scheduler.submit("express", {
            "intent": "agree", "gesture": "nod", "side": "right",
            "energy": 0.3, "duration_ms": 600, "head_only": True,
        })
        self.assertTrue(allowed["accepted"])
        wait_until(lambda: bool(self.scheduler.snapshot()["behavior"]["overlays"]))
        snapshot = self.scheduler.snapshot()
        self.assertEqual(snapshot["behavior"]["base"]["mode"], "clip")
        self.assertEqual(snapshot["behavior"]["overlays"][0]["params"]["intent"], "agree")

    def test_move_hand_and_sequence_are_composable(self) -> None:
        self.enable()
        sequence = self.scheduler.submit("sequence", {
            "steps": [
                {
                    "type": "arm_pose", "side": "right", "elevation_deg": 110.0,
                    "azimuth_deg": -25.0, "plane": None, "reach": 0.8, "palm": "neutral",
                    "wrist_pitch_deg": 0.0, "wrist_yaw_deg": 0.0, "wrist_roll_deg": 20.0,
                    "duration_ms": 150,
                },
                {
                    "type": "move_hand", "side": "right", "relative_to": "chest",
                    "x_m": 0.30, "y_m": 0.05, "z_m": -0.40, "palm": "down",
                    "wrist_pitch_deg": 10.0, "wrist_yaw_deg": 15.0, "wrist_roll_deg": 30.0,
                    "duration_ms": 150,
                },
                {"type": "hand", "side": "right", "pose": "grip", "strength": 1.0, "duration_ms": 100},
                {"type": "wait", "duration_ms": 100},
            ],
            "loop_count": 1,
        })
        self.assertTrue(sequence["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        payload = self.transport.latest_payload()
        position = payload["devices"]["right_controller"]["pose"]["position"]
        self.assertAlmostEqual(position[0], 0.30, places=4)
        self.assertAlmostEqual(position[1], 1.20, places=4)
        self.assertAlmostEqual(position[2], -0.40, places=4)
        self.assertTrue(payload["inputs"]["right_controller"]["grip_click"])
        self.assertEqual(self.scheduler.snapshot()["current_action"]["name"], "sequence")

    def test_sequence_can_be_cancelled_by_action_id(self) -> None:
        self.enable()
        sequence = self.scheduler.submit("sequence", {
            "steps": [{"type": "wait", "duration_ms": 1500}],
            "loop_count": 1,
        })
        wait_until(lambda: self.scheduler.snapshot()["state"] == "moving")
        cancel = self.scheduler.submit("cancel", {"action_id": sequence["action_id"]})
        self.assertTrue(cancel["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        self.assertIsNone(self.scheduler.snapshot()["current_action"])

    def test_sequence_gesture_applies_minimum_safe_duration(self) -> None:
        self.enable()
        sequence = self.scheduler.submit("sequence", {
            "steps": [{
                "type": "gesture", "name": "wave", "side": "right",
                "intensity": 1.0, "duration_ms": 100,
            }],
            "loop_count": 1,
        })
        self.assertTrue(sequence["accepted"])
        wait_until(lambda: (
            self.scheduler.snapshot()["current_action"] or {}
        ).get("params", {}).get("applied_duration_ms", 0) > 100)
        params = self.scheduler.snapshot()["current_action"]["params"]
        self.assertGreater(params["applied_duration_ms"], 100)

    def test_preset_clip_plays_to_last_frame_and_holds(self) -> None:
        self.enable()
        root = Path(__file__).resolve().parents[1]
        clip = ClipLibrary(root / "tests" / "fixtures", self.scheduler.config).load("sample_clip")
        result = self.scheduler.submit("play_clip", {
            "clip_name": clip.name,
            "speed": 3.0,
            "loop_count": 1,
            "transition_ms": 100,
            "anchor": True,
            "restore_after": False,
            "_clip": clip,
        })
        self.assertTrue(result["accepted"])
        self.assertNotIn("_clip", result["normalized_params"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        status = self.scheduler.snapshot()
        self.assertEqual(status["current_action"]["clip_name"], "sample_clip")
        payload = self.transport.latest_payload()
        right_hand = payload["devices"]["right_controller"]["pose"]["position"]
        self.assertAlmostEqual(right_hand[1], 1.82, places=4)

    def test_preset_clip_can_restore_previous_pose(self) -> None:
        self.enable()
        root = Path(__file__).resolve().parents[1]
        clip = ClipLibrary(root / "tests" / "fixtures", self.scheduler.config).load("sample_clip")
        result = self.scheduler.submit("play_clip", {
            "clip_name": clip.name,
            "speed": 3.0,
            "loop_count": 1,
            "transition_ms": 100,
            "anchor": True,
            "restore_after": True,
            "_clip": clip,
        })
        self.assertTrue(result["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "moving")
        wait_until(lambda: self.scheduler.snapshot()["state"] == "idle")
        payload = self.transport.latest_payload()
        right_hand = payload["devices"]["right_controller"]["pose"]["position"]
        self.assertAlmostEqual(right_hand[0], 0.68, places=4)
        self.assertAlmostEqual(right_hand[1], 1.33, places=4)

    def test_stop_releases_inputs_and_reset_recovers(self) -> None:
        self.enable()
        self.scheduler.submit("hand", {"side": "right", "pose": "grip", "strength": 1.0, "duration_ms": 100})
        wait_until(lambda: self.scheduler.snapshot()["state"] == "holding")
        self.assertTrue(self.transport.latest_payload()["inputs"]["right_controller"]["grip_click"])

        stopped = self.scheduler.submit("stop")
        self.assertTrue(stopped["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "stopped_latched")
        wait_until(lambda: not self.transport.latest_payload()["inputs"]["right_controller"]["grip_click"])

        blocked = self.scheduler.submit("gesture", {"name": "wave", "side": "right", "intensity": 0.8})
        self.assertFalse(blocked["accepted"])
        reset = self.scheduler.submit("reset", {"duration_ms": 100})
        self.assertTrue(reset["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "idle")
        self.assertEqual(self.scheduler.snapshot()["safety_state"], "normal")

    def test_disable_sends_six_neutral_frames_then_stops(self) -> None:
        self.enable()
        before = self.transport.count()
        result = self.scheduler.submit("disable", {"duration_ms": 100})
        self.assertTrue(result["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "disabled")
        after = self.transport.count()
        self.assertGreaterEqual(after - before, 6)
        time.sleep(0.1)
        self.assertEqual(self.transport.count(), after)

    def test_output_rate_is_near_60_hz_without_bursts(self) -> None:
        self.enable()
        start_count = self.transport.count()
        time.sleep(1.0)
        end_count = self.transport.count()
        observed = end_count - start_count
        self.assertGreaterEqual(observed, 55)
        self.assertLessEqual(observed, 65)
        with self.transport.lock:
            times = [entry[0] for entry in self.transport.packets[-observed:]]
        intervals = [right - left for left, right in zip(times, times[1:])]
        self.assertTrue(intervals)
        self.assertGreater(min(intervals), 0.005)

    def test_transport_failure_latches_fault(self) -> None:
        self.scheduler.shutdown()
        failing = FakeTransport(fail_after=2)
        self.scheduler = BodyScheduler(PluginConfig(), transport=failing)
        self.scheduler.start()
        self.assertTrue(self.scheduler.submit("enable")["accepted"])
        wait_until(lambda: self.scheduler.snapshot()["state"] == "fault_latched")
        self.assertEqual(self.scheduler.snapshot()["safety_state"], "fault")
        self.assertIn("synthetic UDP failure", self.scheduler.snapshot()["last_error"])


if __name__ == "__main__":
    unittest.main()
