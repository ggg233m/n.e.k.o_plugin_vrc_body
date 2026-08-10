from __future__ import annotations

import json
import math
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import SafetyConfig
from neko_anyadance_body.model import CONTROLLER_IDS, DEVICE_IDS, neutral_frame
from neko_anyadance_body.protocol import MAX_PACKET_BYTES, encode_frame


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.safety = SafetyConfig()

    def test_neutral_frame_encodes_complete_snapshot(self) -> None:
        encoded = encode_frame(neutral_frame(), self.safety)
        self.assertLess(len(encoded), MAX_PACKET_BYTES)
        payload = json.loads(encoded)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(set(payload["devices"]), set(DEVICE_IDS))
        self.assertEqual(set(payload["inputs"]), set(CONTROLLER_IDS))
        for device in payload["devices"].values():
            self.assertEqual(len(device["pose"]["position"]), 3)
            self.assertEqual(len(device["pose"]["rotation_xyzw"]), 4)
        for controller in payload["inputs"].values():
            self.assertEqual(set(controller["finger_bends"]), {"thumb", "index", "middle", "ring", "pinky"})
            self.assertIn("grip_click", controller)
            self.assertIn("joystick_x", controller)

    def test_nan_is_rejected(self) -> None:
        frame = neutral_frame()
        frame.devices["hmd"].position = (math.nan, 1.5, 0.0)
        with self.assertRaisesRegex(ValueError, "NaN or Infinity"):
            encode_frame(frame, self.safety)

    def test_oversized_position_and_bad_quaternion_are_rejected(self) -> None:
        frame = neutral_frame()
        frame.devices["left_foot"].position = (-3.1, 0.26, 0.1)
        with self.assertRaisesRegex(ValueError, "safety bounds"):
            encode_frame(frame, self.safety)
        frame = neutral_frame()
        frame.devices["hmd"].rotation = (0.0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "quaternion"):
            encode_frame(frame, self.safety)

    def test_json_encoder_never_emits_nonstandard_nan(self) -> None:
        encoded = encode_frame(neutral_frame(), self.safety)
        self.assertNotIn(b"NaN", encoded)
        self.assertNotIn(b"Infinity", encoded)


if __name__ == "__main__":
    unittest.main()

