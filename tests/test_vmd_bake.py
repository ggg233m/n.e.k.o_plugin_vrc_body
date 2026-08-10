from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.nya import parse_nya
from neko_anyadance_body.vmd_bake import convert_solved_file, retarget_solved_document


def _joint(position: list[float]) -> dict[str, list[float]]:
    return {"p": position, "q": [0.0, 0.0, 0.0, 1.0]}


def _standing_joints() -> dict[str, dict[str, list[float]]]:
    return {
        "pelvis": _joint([0.0, 0.9, 0.0]),
        "head": _joint([0.0, 1.5, 0.0]),
        "left_shoulder": _joint([-0.18, 1.35, 0.0]),
        "right_shoulder": _joint([0.18, 1.35, 0.0]),
        "left_elbow": _joint([-0.4, 1.35, 0.0]),
        "right_elbow": _joint([0.4, 1.35, 0.0]),
        "left_wrist": _joint([-0.6, 1.35, 0.0]),
        "right_wrist": _joint([0.6, 1.35, 0.0]),
        "left_ankle": _joint([-0.1, 0.1, 0.0]),
        "right_ankle": _joint([0.1, 0.1, 0.0]),
        "left_toe": _joint([-0.1, 0.0, 0.1]),
        "right_toe": _joint([0.1, 0.0, 0.1]),
    }


def _document() -> dict[str, object]:
    rest = _standing_joints()
    second = _standing_joints()
    second["head"] = _joint([0.0, 1.6, 0.0])
    return {
        "format": "anyadance_mmd_solved",
        "version": 1,
        "fps": 30.0,
        "model": "synthetic",
        "rest": rest,
        "frames": [
            {
                "t": 0.0,
                "j": rest,
                "fl": [0.0, 0.2, 0.4, 0.6, 0.8],
                "fr": [0.8, 0.6, 0.4, 0.2, 0.0],
            },
            {"t": 1.0, "j": second},
        ],
    }


class VmdBakeTests(unittest.TestCase):
    def test_retarget_matches_anyadance_standing_fixture(self) -> None:
        result = retarget_solved_document(_document(), target_height_m=1.62, hand_reach_scale=1.0)
        first = result["frames"][0]
        devices = first["devices"]
        self.assertAlmostEqual(devices["hmd"]["p"][1], 1.6, places=5)
        self.assertAlmostEqual(devices["hip"]["p"][1], 0.9, places=5)
        self.assertAlmostEqual(devices["left_foot"]["p"][1], 0.1, places=5)
        self.assertAlmostEqual(devices["right_controller"]["p"][0], 0.642, places=5)
        self.assertEqual(first["fingers"]["left"], [0.0, 0.2, 0.4, 0.6, 0.8])
        for device in devices.values():
            self.assertAlmostEqual(sum(value * value for value in device["q"]), 1.0, places=5)

    def test_output_is_accepted_by_plugin_clip_parser(self) -> None:
        result = retarget_solved_document(_document())
        clip = parse_nya(json.dumps(result), name="real_vmd", config=PluginConfig())
        self.assertEqual(len(clip.frames), 2)
        self.assertAlmostEqual(clip.duration_s, 1.0)
        self.assertTrue(clip.frames[0].has_fingers)

    def test_avatar_proportion_profile_adjusts_width_and_leg_height(self) -> None:
        result = retarget_solved_document(
            _document(),
            target_height_m=1.62,
            hand_reach_scale=1.0,
            body_width_scale=1.2,
            leg_length_scale=1.1,
            hip_height_offset_m=0.05,
        )
        devices = result["frames"][0]["devices"]
        self.assertAlmostEqual(devices["right_controller"]["p"][0], 0.642 * 1.2, places=5)
        self.assertAlmostEqual(devices["hmd"]["p"][1], 1.73, places=5)
        self.assertAlmostEqual(devices["right_foot"]["p"][1], 0.1, places=5)
        self.assertEqual(result["bake_profile"]["leg_length_scale"], 1.1)

    def test_file_conversion_is_atomic_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solved = root / "solved.json"
            output = root / "motion.nya"
            solved.write_text(json.dumps(_document()), encoding="utf-8")
            result = convert_solved_file(solved, output)
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".nya.tmp").exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["frames"], result["frames"])

    def test_rejects_non_monotonic_timestamps(self) -> None:
        document = _document()
        document["frames"][1]["t"] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            retarget_solved_document(document)


if __name__ == "__main__":
    unittest.main()
