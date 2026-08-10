from __future__ import annotations

import socket
import struct
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import VrchatOscConfig
from neko_anyadance_body.osc import (
    OscProtocolError,
    VrchatOscBridge,
    decode_osc_packet,
    encode_osc_message,
    normalize_parameter_value,
    validate_parameter_name,
)


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


class OscProtocolTests(unittest.TestCase):
    def test_round_trip_supported_types(self) -> None:
        packet = encode_osc_message(
            "/avatar/parameters/Test",
            (True, False, 17, 0.25, "文字", b"abc", None),
        )
        messages = decode_osc_packet(packet)
        self.assertEqual(len(messages), 1)
        address, arguments = messages[0]
        self.assertEqual(address, "/avatar/parameters/Test")
        self.assertEqual(arguments[:3], (True, False, 17))
        self.assertAlmostEqual(arguments[3], 0.25)
        self.assertEqual(arguments[4:], ("文字", b"abc", None))

    def test_bundle_is_flattened(self) -> None:
        first = encode_osc_message("/avatar/change", ("avtr_test",))
        second = encode_osc_message("/avatar/parameters/NEKO_Action", (3,))
        bundle = (
            b"#bundle\x00"
            + struct.pack(">Q", 1)
            + struct.pack(">i", len(first))
            + first
            + struct.pack(">i", len(second))
            + second
        )
        self.assertEqual(
            decode_osc_packet(bundle),
            [
                ("/avatar/change", ("avtr_test",)),
                ("/avatar/parameters/NEKO_Action", (3,)),
            ],
        )

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(OscProtocolError):
            encode_osc_message("not/an/address", (1,))
        with self.assertRaises(ValueError):
            validate_parameter_name("bad/name")
        with self.assertRaises(ValueError):
            normalize_parameter_value(float("nan"))
        with self.assertRaises(ValueError):
            normalize_parameter_value("1")


class OscBridgeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target.bind(("127.0.0.1", 0))
        self.target.settimeout(1.0)
        self.listen_port = _free_udp_port()
        self.bridge = VrchatOscBridge(VrchatOscConfig(
            send_host="127.0.0.1",
            send_port=self.target.getsockname()[1],
            listen_host="127.0.0.1",
            listen_port=self.listen_port,
            allowed_sender="127.0.0.1",
            input_pulse_ms=30,
            awareness_parameters=("NEKO_Action",),
        ))
        self.bridge.start()

    def tearDown(self) -> None:
        self.bridge.stop()
        self.target.close()

    def test_parameter_send_feedback_receive_and_input_release(self) -> None:
        sent, reason = self.bridge.send_parameter("NEKO_Action", 4)
        self.assertTrue(sent, reason)
        packet, _ = self.target.recvfrom(4096)
        self.assertEqual(
            decode_osc_packet(packet),
            [("/avatar/parameters/NEKO_Action", (4,))],
        )

        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(encode_osc_message("/avatar/change", ("avtr_test",)), ("127.0.0.1", self.listen_port))
            sender.sendto(
                encode_osc_message("/avatar/parameters/NEKO_Action", (4,)),
                ("127.0.0.1", self.listen_port),
            )
        finally:
            sender.close()
        deadline = time.monotonic() + 1.0
        while self.bridge.snapshot()["parameter_count"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        awareness = self.bridge.awareness()
        self.assertEqual(awareness["connection"], "detected")
        self.assertEqual(awareness["avatar_id"], "avtr_test")
        self.assertEqual(awareness["parameters"]["NEKO_Action"]["value"], 4)
        self.assertFalse(awareness["pose_feedback_available"])

        sent, reason = self.bridge.pulse_input("grab", "right", 30)
        self.assertTrue(sent, reason)
        values: list[int] = []
        deadline = time.monotonic() + 1.0
        while len(values) < 2 and time.monotonic() < deadline:
            packet, _ = self.target.recvfrom(4096)
            address, arguments = decode_osc_packet(packet)[0]
            if address == "/input/GrabRight":
                values.append(arguments[0])
        self.assertEqual(values, [1, 0])


if __name__ == "__main__":
    unittest.main()
