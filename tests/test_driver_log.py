from __future__ import annotations

import json
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import DriverLogConfig
from neko_anyadance_body.driver_log import DriverLogListener, parse_driver_log_event


# Copied from AnyaDance docs/protocol.md so the parser stays aligned with the
# published contract rather than with an assumption about it.
COMMAND_EVENT = {
    "version": 1,
    "event": "command_processed",
    "sequence": 42,
    "suppressed": 613,
    "detail": "accepted 2 device entries; clamped Y for 1",
    "source": {"host": "127.0.0.1", "port": 54321},
    "command": {
        "protocol": "pose_frame",
        "bytes": 347,
        "accepted": True,
        "devices": ["hmd", "left_controller"],
        "y_clamped": ["hmd"],
        "payload": '{"version":1,...}',
    },
}
HAPTIC_EVENT = {
    "version": 1,
    "event": "haptic_vibration",
    "sequence": 7,
    "suppressed": 0,
    "detail": "0.125 s at 160.5 Hz, amplitude 0.75",
    "device": "right_controller",
    "haptic": {"duration_seconds": 0.125, "frequency_hz": 160.5, "amplitude": 0.75},
}


def datagram(base: dict, **overrides) -> bytes:
    payload = {**base, **overrides}
    return json.dumps(payload).encode("utf-8")


class DriverLogParseTests(unittest.TestCase):
    def test_documented_command_event_parses(self) -> None:
        event = parse_driver_log_event(datagram(COMMAND_EVENT))
        assert event is not None
        self.assertEqual(event["type"], "command_processed")
        self.assertEqual(event["sequence"], 42)
        self.assertEqual(event["suppressed"], 613)
        self.assertEqual(event["source"], {"host": "127.0.0.1", "port": 54321})
        self.assertTrue(event["command"]["accepted"])
        self.assertEqual(event["command"]["bytes"], 347)
        self.assertEqual(event["command"]["devices"], ["hmd", "left_controller"])
        self.assertEqual(event["command"]["y_clamped"], ["hmd"])

    def test_documented_haptic_event_parses(self) -> None:
        event = parse_driver_log_event(datagram(HAPTIC_EVENT))
        assert event is not None
        self.assertEqual(event["type"], "haptic_vibration")
        self.assertEqual(event["device"], "right_controller")
        self.assertAlmostEqual(event["haptic"]["frequency_hz"], 160.5)
        self.assertAlmostEqual(event["haptic"]["amplitude"], 0.75)

    def test_unknown_event_name_still_parses_with_envelope(self) -> None:
        # The group is designed to grow; a receiver must skip, not fail.
        event = parse_driver_log_event(
            datagram({"version": 1, "event": "future_event", "sequence": 3, "detail": "x"})
        )
        assert event is not None
        self.assertEqual(event["type"], "unknown")
        self.assertEqual(event["event"], "future_event")
        self.assertEqual(event["sequence"], 3)

    def test_unknown_fields_are_ignored(self) -> None:
        event = parse_driver_log_event(datagram(COMMAND_EVENT, future_field={"a": 1}))
        assert event is not None
        self.assertEqual(event["type"], "command_processed")
        self.assertNotIn("future_field", event)

    def test_unknown_version_is_ignored(self) -> None:
        self.assertIsNone(parse_driver_log_event(datagram(COMMAND_EVENT, version=2)))

    def test_malformed_payloads_are_rejected(self) -> None:
        self.assertIsNone(parse_driver_log_event(b""))
        self.assertIsNone(parse_driver_log_event(b"not json"))
        self.assertIsNone(parse_driver_log_event(b"[1, 2, 3]"))
        self.assertIsNone(parse_driver_log_event(b'{"version":1,"event":"x"}'))
        self.assertIsNone(parse_driver_log_event(datagram(COMMAND_EVENT, sequence=-1)))
        self.assertIsNone(parse_driver_log_event(datagram(COMMAND_EVENT, event="")))
        self.assertIsNone(parse_driver_log_event(b'{"version":1,"event":"x","sequence":NaN}'))

    def test_oversized_payload_is_truncated(self) -> None:
        event = parse_driver_log_event(
            datagram(COMMAND_EVENT, command={**COMMAND_EVENT["command"], "payload": "x" * 9000})
        )
        assert event is not None
        self.assertEqual(len(event["command"]["payload"]), 2048)


class DriverLogListenerTests(unittest.TestCase):
    """Deterministic: the public ingest path is exercised without a socket."""

    def listener(self, **overrides) -> DriverLogListener:
        config = DriverLogConfig(**overrides)
        return DriverLogListener(config, clock=lambda: 100.0)

    def test_command_event_updates_counters_and_last_command(self) -> None:
        listener = self.listener()
        self.assertTrue(listener.ingest_packet(datagram(COMMAND_EVENT), now=100.0))
        status = listener.snapshot()
        self.assertEqual(status["accepted_commands"], 1)
        self.assertEqual(status["rejected_commands"], 0)
        self.assertEqual(status["y_clamped_commands"], 1)
        self.assertEqual(status["decoded_events"], 1)
        self.assertEqual(status["last_command"]["source"], "127.0.0.1:54321")
        self.assertEqual(status["connection"], "detected")

    def test_rejected_command_is_counted_separately(self) -> None:
        listener = self.listener()
        listener.ingest_packet(
            datagram(COMMAND_EVENT, command={**COMMAND_EVENT["command"], "accepted": False, "y_clamped": []}),
            now=100.0,
        )
        status = listener.snapshot()
        self.assertEqual(status["accepted_commands"], 0)
        self.assertEqual(status["rejected_commands"], 1)
        self.assertEqual(status["y_clamped_commands"], 0)

    def test_haptic_event_records_last_haptic(self) -> None:
        listener = self.listener()
        self.assertTrue(listener.ingest_packet(datagram(HAPTIC_EVENT), now=100.0))
        status = listener.snapshot()
        self.assertEqual(status["last_haptic"]["device"], "right_controller")
        self.assertEqual(status["accepted_commands"], 0)
        # A haptic report alone is not proof a pose command arrived.
        self.assertEqual(status["connection"], "unknown")

    def test_unknown_event_is_counted_not_rejected(self) -> None:
        listener = self.listener()
        self.assertTrue(
            listener.ingest_packet(
                datagram({"version": 1, "event": "future_event", "sequence": 1}), now=100.0
            )
        )
        status = listener.snapshot()
        self.assertEqual(status["unknown_events"], 1)
        self.assertEqual(status["rejected_packets"], 0)

    def test_malformed_datagram_increments_rejected(self) -> None:
        listener = self.listener()
        self.assertFalse(listener.ingest_packet(b"not json", now=100.0))
        status = listener.snapshot()
        self.assertEqual(status["rejected_packets"], 1)
        self.assertEqual(status["decoded_events"], 0)

    def test_sequence_gap_counts_loss(self) -> None:
        listener = self.listener()
        listener.ingest_packet(datagram(HAPTIC_EVENT, sequence=1), now=100.0)
        listener.ingest_packet(datagram(HAPTIC_EVENT, sequence=5), now=100.0)
        self.assertEqual(listener.snapshot()["lost_events"], 3)

    def test_repeated_sequence_is_a_duplicate(self) -> None:
        listener = self.listener()
        listener.ingest_packet(datagram(COMMAND_EVENT, sequence=1), now=100.0)
        self.assertFalse(listener.ingest_packet(datagram(COMMAND_EVENT, sequence=1), now=100.0))
        status = listener.snapshot()
        self.assertEqual(status["duplicate_events"], 1)
        self.assertEqual(status["accepted_commands"], 1)

    def test_late_arrival_inside_window_is_not_counted_as_loss(self) -> None:
        listener = self.listener()
        listener.ingest_packet(datagram(HAPTIC_EVENT, sequence=4), now=100.0)
        listener.ingest_packet(datagram(HAPTIC_EVENT, sequence=6), now=100.0)
        self.assertEqual(listener.snapshot()["lost_events"], 1)
        # Sequence 5 arriving late must not be counted a second time.
        listener.ingest_packet(datagram(HAPTIC_EVENT, sequence=5), now=100.0)
        status = listener.snapshot()
        self.assertEqual(status["lost_events"], 1)
        self.assertEqual(status["decoded_events"], 3)

    def test_multiple_sources_are_tracked(self) -> None:
        listener = self.listener()
        listener.ingest_packet(datagram(COMMAND_EVENT, sequence=1), now=100.0)
        listener.ingest_packet(
            datagram(COMMAND_EVENT, sequence=2, source={"host": "127.0.0.1", "port": 9999}),
            now=100.0,
        )
        self.assertEqual(
            listener.snapshot()["senders"], ["127.0.0.1:54321", "127.0.0.1:9999"]
        )

    def test_connection_goes_stale_after_the_configured_window(self) -> None:
        current = [100.0]
        listener = DriverLogListener(DriverLogConfig(stale_after_ms=3000), clock=lambda: current[0])
        listener.ingest_packet(datagram(COMMAND_EVENT), now=current[0])
        self.assertEqual(listener.snapshot()["connection"], "detected")
        current[0] += 5.0
        self.assertEqual(listener.snapshot()["connection"], "stale")

    def test_disabled_listener_never_binds(self) -> None:
        listener = DriverLogListener(DriverLogConfig(enabled=False))
        listener.start()
        try:
            self.assertFalse(listener.thread_alive)
            self.assertFalse(listener.snapshot()["receiver_listening"])
        finally:
            listener.stop()


if __name__ == "__main__":
    unittest.main()
