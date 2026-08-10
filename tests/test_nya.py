from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body import nya as nya_module
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.model import neutral_frame
from neko_anyadance_body.nya import ClipLibrary, parse_nya, sample_clip


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class NyaClipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PluginConfig()
        self.library = ClipLibrary(ROOT / "motions", self.config)

    def test_builtin_catalog_contains_real_vmd_presets(self) -> None:
        catalog = self.library.list()
        self.assertEqual(catalog["invalid_clips"], [])
        names = {clip["name"] for clip in catalog["clips"]}
        self.assertEqual(names, {
            "vmd_greeting", "vmd_v_sign", "vmd_showcase", "vmd_model_pose",
            "vmd_stretch", "vmd_shooting_pose", "vmd_silly_walk", "vmd_turn",
            "vmd_idle_01", "vmd_idle_02", "vmd_idle_03", "vmd_idle_04", "vmd_idle_05",
        })
        self.assertTrue(all(clip["frame_count"] > 400 for clip in catalog["clips"]))
        self.assertEqual(catalog["motion_catalog"]["errors"], [])
        self.assertEqual(catalog["motion_catalog"]["missing_clips"], [])

    def test_fast_catalog_does_not_read_or_parse_uncached_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "large.nya").write_text("not-json", encoding="utf-8")
            library = ClipLibrary(root, self.config)
            with patch.object(Path, "read_text", side_effect=AssertionError("catalog read payload")):
                catalog = library.catalog()
            self.assertEqual(catalog["invalid_clips"], [])
            self.assertEqual(catalog["clips"][0]["name"], "large")
            self.assertFalse(catalog["clips"][0]["indexed"])
            self.assertEqual(catalog["unindexed_count"], 1)
            self.assertEqual(catalog["cache"]["parse_count"], 0)

    def test_repeated_load_uses_signature_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dance.nya").write_text(
                (FIXTURES / "sample_clip.nya").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            library = ClipLibrary(root, self.config)
            with patch.object(nya_module, "parse_nya", wraps=nya_module.parse_nya) as parser:
                first = library.load("dance")
                second = library.load("dance")
            self.assertIs(first, second)
            self.assertEqual(parser.call_count, 1)
            self.assertEqual(library.catalog()["cache"]["cache_hits"], 1)

    def test_cache_is_invalidated_when_clip_signature_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "dance.nya"
            source = json.loads((FIXTURES / "sample_clip.nya").read_text(encoding="utf-8"))
            path.write_text(json.dumps(source), encoding="utf-8")
            library = ClipLibrary(root, self.config)
            with patch.object(nya_module, "parse_nya", wraps=nya_module.parse_nya) as parser:
                first = library.load("dance")
                source["model"] = "changed-model-with-a-different-file-size"
                path.write_text(json.dumps(source), encoding="utf-8")
                second = library.load("dance")
            self.assertEqual(parser.call_count, 2)
            self.assertNotEqual(first.model, second.model)
            self.assertEqual(second.model, "changed-model-with-a-different-file-size")

    def test_clip_sampling_interpolates_and_anchors_xz(self) -> None:
        clip = ClipLibrary(FIXTURES, self.config).load("sample_clip")
        self.assertEqual(len(clip.frames), 3)
        self.assertEqual(clip.times, tuple(frame.time_s for frame in clip.frames))
        self.assertAlmostEqual(clip.duration_s, 1.2)
        base = neutral_frame()
        base.devices["hmd"].position = (1.0, 1.5, -2.0)
        sample = sample_clip(clip, 0.6, base=base, offset_x=1.0, offset_z=-2.0)
        self.assertEqual(sample.devices["hmd"].position, (1.0, 1.5, -2.0))
        self.assertAlmostEqual(sample.devices["right_controller"].position[0], 1.3)
        self.assertAlmostEqual(sample.devices["right_controller"].position[2], -2.25)

    def test_finger_frames_drive_grip_without_overwriting_other_inputs(self) -> None:
        clip = ClipLibrary(FIXTURES, self.config).load("grip_clip")
        base = neutral_frame()
        base.controllers["left_controller"].trigger_click = True
        end = sample_clip(clip, clip.duration_s, base=base, offset_x=0.0, offset_z=0.0)
        self.assertTrue(end.controllers["right_controller"].grip_click)
        self.assertEqual(end.controllers["right_controller"].grip_value, 1.0)
        self.assertTrue(end.controllers["left_controller"].trigger_click)

    def test_path_traversal_and_unknown_names_are_rejected(self) -> None:
        for name in ("../secret", "C:\\secret", "trailing ", "."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.library.load(name)
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            self.library.load("does_not_exist")

    def test_invalid_times_and_non_finite_values_are_rejected(self) -> None:
        source = json.loads((FIXTURES / "sample_clip.nya").read_text(encoding="utf-8"))
        source["frames"][1]["t"] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_nya(json.dumps(source), name="bad_times", config=self.config)
        source = json.loads((FIXTURES / "sample_clip.nya").read_text(encoding="utf-8"))
        source["frames"][0]["devices"]["hmd"]["p"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid constant|finite"):
            parse_nya(json.dumps(source), name="bad_nan", config=self.config)

    def test_library_reports_invalid_files_without_hiding_valid_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.nya").write_text("not-json", encoding="utf-8")
            (root / "good.nya").write_text(
                (FIXTURES / "sample_clip.nya").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            catalog = ClipLibrary(root, self.config).list()
            self.assertEqual([item["name"] for item in catalog["clips"]], ["good"])
            self.assertEqual(catalog["invalid_clips"][0]["name"], "bad")

    def test_unicode_clip_names_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "人间万朵红.nya").write_text(
                (FIXTURES / "sample_clip.nya").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            clip = ClipLibrary(root, self.config).load("人间万朵红")
            self.assertEqual(clip.name, "人间万朵红")

    def test_y_above_driver_ceiling_is_clamped_like_anyadance(self) -> None:
        source = json.loads((FIXTURES / "sample_clip.nya").read_text(encoding="utf-8"))
        source["frames"][1]["devices"]["right_controller"]["p"][1] = 2.019
        clip = parse_nya(json.dumps(source), name="clamped", config=self.config)
        self.assertEqual(clip.frames[1].frame.devices["right_controller"].position[1], 2.0)

    def test_default_file_limit_accepts_large_pretty_printed_dances(self) -> None:
        self.assertEqual(self.config.clip_max_file_bytes, 64 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
