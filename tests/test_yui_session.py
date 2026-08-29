"""YUI 插件会话与日志层测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.yui_session import YuiSessionState
from yui_npc_controller.runtime.yui_transport import YuiReliableTransport
from yui_npc_controller.runtime.yui_log import YuiOutputLogTailer


ROOT = Path(__file__).resolve().parents[1]


def _vectors() -> dict[str, dict]:
    with (ROOT / "Docs" / "Protocols" / "YUI_NPC_TestVectors_v1.1.json").open(
        "r", encoding="utf-8"
    ) as handle:
        document = json.load(handle)
    return {item["id"]: item for item in document["vectors"]}


def _header(log_sequence: int, event_type: str, **body):
    return {
        "v": 1,
        "spec": "1.1",
        "session": 1193046,
        "world_id": "wrld_test_yui",
        "npc": "yui",
        "log_seq": log_sequence,
        "t": log_sequence / 100.0,
        "type": event_type,
        **body,
    }


class YuiSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectors = _vectors()

    def test_discover_projection_builds_catalog_and_privacy_safe_observation(self) -> None:
        state = YuiSessionState()
        for event in self.vectors["discover_hello_catalog"]["expected_logs"]:
            state.ingest(event)
        observation = state.observe()
        self.assertEqual(observation["session"], 1193046)
        self.assertEqual(observation["control_state"], "safe_idle")
        self.assertTrue(observation["operation_lifecycle"])
        self.assertTrue(observation["active_ops_authoritative"])
        self.assertEqual(observation["active_ops"], [])
        self.assertIn("greet_wave", observation["semantic_keys"]["action"])
        self.assertTrue(observation["players"])
        self.assertNotIn("name", observation["players"][0])
        named = state.observe(include_player_names=True)
        self.assertEqual(named["players"][0]["name"], "Driver")

    def test_missing_operation_lifecycle_keeps_required_empty_shape(self) -> None:
        state = YuiSessionState()
        for event in self.vectors["operation_lifecycle_capability_disabled"]["expected_logs"]:
            state.ingest(event)
        observation = state.observe()
        self.assertFalse(observation["operation_lifecycle"])
        self.assertFalse(observation["active_ops_authoritative"])
        self.assertEqual(observation["active_ops"], [])

    def test_log_gap_is_reported_without_inventing_events(self) -> None:
        state = YuiSessionState()
        state.ingest(_header(1, "sys.boot", ready=True))
        state.ingest(_header(3, "sys.telemetry", dropped_since_last_report=1, dropped_total=1,
                             first_dropped_log_seq=2, last_dropped_log_seq=2, wrap_count=0))
        observation = state.observe()
        self.assertFalse(observation["log_complete"])
        self.assertEqual(observation["log_gaps"], [(2, 2)])

    def test_complete_snapshot_replaces_stale_player_set(self) -> None:
        state = YuiSessionState()
        state.ingest(_header(1, "player.join", slot=1, pid=9, name="旧玩家"))
        state.ingest(_header(
            2,
            "sys.snapshot",
            snapshot_seq=7,
            part=1,
            parts=1,
            section="players",
            data={"players": [{"slot": 2, "d": 3.0, "brg": 90.0}]},
        ))
        observation = state.observe()
        self.assertEqual([player["slot"] for player in observation["players"]], [2])

    def test_host_waiters_use_observed_position_and_operation_terminal(self) -> None:
        state = YuiSessionState()
        state.ingest(_header(
            1,
            "npc.state",
            state="moving",
            estop=False,
            pos=[1.0, 0.0, 2.0],
            yaw=0.0,
            vel=[0.0, 0.0, 1.2],
            speed=1.2,
            grounded=True,
            mode="goto",
            anim_id=-1,
            expression_id=-1,
            active_ops=["1193046:7:ABCD"],
        ))
        near = state.wait_for_npc_near(1.1, 2.1, 0.2, 0.0)
        self.assertIsNotNone(near)
        self.assertEqual(near["speed"], 1.2)
        state.ingest(_header(
            2,
            "npc.operation_cancelled",
            op_id="1193046:7:ABCD",
            kind="goto",
            request_seq=7,
            request_hash="ABCD",
            elapsed_ms=500,
            reason="replaced",
        ))
        operation = state.wait_for_operation("1193046:7:ABCD", 0.0)
        self.assertEqual(operation["status"], "cancelled")
        self.assertEqual(operation["reason"], "replaced")

    def test_owner_error_clears_host_authorization(self) -> None:
        state = YuiSessionState()
        state.session = 1193046
        state.set_host_arm_authorized(True)
        state.ingest(_header(
            1,
            "npc.ack",
            seq=1,
            cmd_id=11,
            cmd="HEARTBEAT",
            request_hash="A2BC",
            ok=False,
            replayed=False,
            state="external",
            err="not_owner",
        ))
        self.assertFalse(state.host_arm_authorized)

    def test_player_touch_enters_recent_observation_and_notifies_listener(self) -> None:
        state = YuiSessionState()
        received = []
        state.add_event_listener(received.append)
        state.ingest(_header(1, "player.touch", slot=2, zone="head", intensity=0.7))
        observation = state.observe()
        self.assertEqual(observation["recent_social_events"][0]["type"], "player.touch")
        self.assertEqual(received[0]["zone"], "head")

    def test_wait_for_ack_ignores_history_before_generation(self) -> None:
        state = YuiSessionState()
        old = _header(
            1,
            "npc.ack",
            seq=10,
            cmd_id=2,
            cmd="GOTO_XZ",
            request_hash="D34C",
            ok=False,
            replayed=False,
            state="external",
            err="target_out_of_bounds",
        )
        state.ingest(old)
        generation = state.ack_generation
        result = []

        def wait() -> None:
            result.append(state.wait_for_ack(10, 2, "D34C", 0.5, after_arrival_index=generation))

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.02)
        replay = dict(old, log_seq=2, replayed=True)
        state.ingest(replay)
        thread.join(timeout=1.0)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0])
        self.assertTrue(result[0].replayed)

    def test_reliable_transport_correlates_ack_and_marks_long_operation_accepted(self) -> None:
        state = YuiSessionState()
        state.session = 1193046
        state.control_state = "external"
        state.capabilities = ("operation_lifecycle",)
        sent = []
        transport = YuiReliableTransport(
            sent.append,
            state,
            ack_timeout_s=0.2,
            command_deadline_s=0.5,
        )
        outcome = []

        def send() -> None:
            outcome.append(transport.send_command("TURN_TO", (455, 0, 0, 0, 0, 0)))

        thread = threading.Thread(target=send)
        thread.start()
        deadline = time.monotonic() + 0.5
        while len(sent) < 10 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(len(sent), 10)
        note = sent[-1]
        from yui_npc_controller.runtime.yui_protocol import command_request_hash
        request_hash = command_request_hash(4, note.value, (455, 0, 0, 0, 0, 0))
        state.ingest(_header(
            1,
            "npc.ack",
            seq=note.value,
            cmd_id=4,
            cmd="TURN_TO",
            request_hash=request_hash,
            ok=True,
            replayed=False,
            state="moving",
        ))
        thread.join(timeout=1.0)
        self.assertEqual(outcome[0].status, "accepted")
        self.assertEqual(outcome[0].kind, "turn")
        self.assertEqual(outcome[0].operation_id, f"1193046:{note.value}:{request_hash}")

    def test_estop_bypasses_normal_frame_and_can_use_any_channel(self) -> None:
        state = YuiSessionState()
        sent = []
        transport = YuiReliableTransport(sent.append, state)
        frame = transport.send_estop(channel=2, acknowledge=False)
        self.assertEqual(frame.sequence, 0)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].channel, 2)
        self.assertEqual(sent[0].number, 0x7F)

    def test_ack_timeout_requests_snapshot_before_returning_unknown(self) -> None:
        state = YuiSessionState()
        state.session = 1193046
        state.control_state = "external"
        state.capabilities = ("snapshot", "operation_lifecycle")
        sent = []
        transport = YuiReliableTransport(
            sent.append,
            state,
            ack_timeout_s=0.02,
            command_deadline_s=0.03,
        )
        outcome = transport.send_command("TURN_TO", (455, 0, 0, 0, 0, 0))
        self.assertEqual(outcome.status, "unknown")
        self.assertTrue(outcome.snapshot_requested)
        notes = [event.number for event in sent if event.type == "note_on"]
        self.assertEqual(notes, [4, 4, 15])

    def test_output_log_tailer_reads_only_new_marked_events(self) -> None:
        state = YuiSessionState()
        tailer = YuiOutputLogTailer(
            state,
            log_path=ROOT / "tests" / "fixtures" / "yui_output_log.txt",
            from_end=False,
            poll_interval_s=0.02,
        )
        tailer.start()
        deadline = time.monotonic() + 1.0
        while state.last_log_sequence != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        tailer.stop()
        self.assertEqual(state.last_log_sequence, 1)
        snapshot = tailer.snapshot()
        self.assertEqual(snapshot["events_read"], 1)
        self.assertEqual(snapshot["decode_errors"], 0)

    def test_output_log_tailer_waits_for_newline_before_consuming_event(self) -> None:
        state = YuiSessionState()
        event = _header(1, "sys.boot", ready=True)
        line = (
            "2026.08.29 00:00:00 Log - [NEKO]"
            + json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        split_at = line.index(b'"ready"') + 8

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output_log_test.txt"
            path.write_bytes(b"")
            tailer = YuiOutputLogTailer(
                state,
                log_path=path,
                from_end=False,
                poll_interval_s=0.02,
            )
            tailer.start()
            try:
                with path.open("ab", buffering=0) as handle:
                    handle.write(line[:split_at])
                    time.sleep(0.08)
                    partial = tailer.snapshot()
                    self.assertEqual(partial["lines_read"], 0)
                    self.assertEqual(partial["events_read"], 0)
                    self.assertEqual(partial["decode_errors"], 0)

                    handle.write(line[split_at:])
                    deadline = time.monotonic() + 1.0
                    while state.last_log_sequence != 1 and time.monotonic() < deadline:
                        time.sleep(0.01)
            finally:
                tailer.stop()

        self.assertEqual(state.last_log_sequence, 1)
        completed = tailer.snapshot()
        self.assertEqual(completed["lines_read"], 1)
        self.assertEqual(completed["events_read"], 1)
        self.assertEqual(completed["decode_errors"], 0)


if __name__ == "__main__":
    unittest.main()
