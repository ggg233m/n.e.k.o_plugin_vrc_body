from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.motion_catalog import MotionCatalog


class MotionCatalogTests(unittest.TestCase):
    def test_selects_by_intent_side_intensity_and_rotates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for name, side, intensity in (("calm", "neutral", 0.2), ("active", "right", 0.8)):
                (root / f"{name}.nya").write_text("{}", encoding="utf-8")
                entries.append({
                    "name": name, "label": name, "description": name,
                    "intents": ["idle"], "tags": ["test"], "source_kind": "vmd_bake",
                    "source_name": f"{name}.vmd", "body_scope": "full_body", "side": side,
                    "intensity": intensity, "recommended_speed": 1.0, "transition_ms": 400,
                    "restore_after": True, "loop_count": 1,
                })
            (root / "catalog.json").write_text(json.dumps({"version": 1, "motions": entries}), encoding="utf-8")
            catalog = MotionCatalog(root)
            self.assertEqual(catalog.select("idle", intensity=0.1)["name"], "calm")
            self.assertEqual(catalog.select("idle", side="right", intensity=0.9)["name"], "active")
            self.assertIsNone(catalog.select("missing"))

    def test_reports_invalid_metadata_and_missing_clip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalog.json").write_text(json.dumps({
                "version": 1,
                "motions": [
                    {"name": "../bad", "intents": ["idle"]},
                    {
                        "name": "missing", "label": "Missing", "description": "Missing clip",
                        "intents": ["idle"], "tags": ["test"], "source_kind": "vmd_bake",
                        "source_name": "missing.vmd", "body_scope": "full_body", "side": "neutral",
                        "intensity": 0.2, "recommended_speed": 1.0, "transition_ms": 400,
                        "restore_after": True, "loop_count": 1,
                    },
                ],
            }), encoding="utf-8")
            summary = MotionCatalog(root).summary()
            self.assertTrue(summary["errors"])
            self.assertEqual(summary["missing_clips"], ["missing"])


if __name__ == "__main__":
    unittest.main()
