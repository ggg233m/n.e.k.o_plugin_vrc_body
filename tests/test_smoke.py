from __future__ import annotations

import ast
from pathlib import Path
import tomllib
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body import tool_defs
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.instructions import BODY_AI_INSTRUCTIONS
from neko_anyadance_body.motion import GESTURE_NAMES


ROOT = Path(__file__).resolve().parents[1]


class PluginSmokeTests(unittest.TestCase):
    def test_manifest_and_entry_class_match(self) -> None:
        with (ROOT / "plugin.toml").open("rb") as handle:
            manifest = tomllib.load(handle)
        plugin = manifest["plugin"]
        self.assertEqual(plugin["id"], "neko_anyadance_body")
        self.assertEqual(plugin["entry"], "plugins.neko_anyadance_body:NekoAnyadanceBodyPlugin")
        self.assertFalse(manifest["plugin_runtime"]["auto_start"])
        self.assertEqual(manifest["vmc_idle"]["listen_port"], 39539)
        self.assertTrue(manifest["vmc_idle"]["manage_host_output"])

        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("NekoAnyadanceBodyPlugin", classes)

    def test_plugin_declares_no_runtime_dependencies(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["dependencies"], [])

    def test_all_tool_definition_modules_execute(self) -> None:
        self.assertEqual(tool_defs.BODY_PLAY_CLIP["parameters"]["properties"]["anchor"]["default"], True)
        self.assertEqual(tool_defs.BODY_PLAY_CLIP["parameters"]["properties"]["restore_after"]["default"], False)
        self.assertEqual(tool_defs.BODY_AWARENESS["name"], "body_awareness")
        self.assertEqual(tool_defs.BODY_AVATAR_PARAMETER["name"], "body_avatar_parameter")
        self.assertEqual(tool_defs.BODY_VRCHAT_INPUT["name"], "body_vrchat_input")
        self.assertEqual(tool_defs.WORLD_OBSERVE["name"], "world_observe")
        self.assertEqual(tool_defs.BODY_EXPRESS["name"], "body_express")
        reach_preconditions = tool_defs.BODY_REACH_AND_GRAB["parameters"]["properties"]["preconditions"]
        self.assertEqual(reach_preconditions["maxItems"], 16)
        condition_kinds = set(
            reach_preconditions["items"]["properties"]["kind"]["enum"]
        )
        self.assertEqual(
            condition_kinds,
            {"world_available", "entity_visible", "event_recent"},
        )
        self.assertIn("preconditions", BODY_AI_INSTRUCTIONS)
        self.assertIn("idle", tool_defs.BODY_EXPRESS["parameters"]["properties"]["intent"]["enum"])
        self.assertIn("body_awareness", BODY_AI_INSTRUCTIONS)
        self.assertIn("accepted=true", BODY_AI_INSTRUCTIONS)
        declared_gestures = tool_defs.BODY_GESTURE["parameters"]["properties"]["name"]["enum"]
        self.assertEqual(set(declared_gestures), set(GESTURE_NAMES))

    def test_behavior_config_is_bounded(self) -> None:
        config = PluginConfig.from_mapping({
            "behavior": {
                "default_crossfade_ms": 450,
                "protect_full_body_motion": False,
                "prefer_vmd_expressions": False,
                "transition_history_size": 24,
            }
        })
        self.assertEqual(config.behavior.default_crossfade_ms, 450)
        self.assertFalse(config.behavior.protect_full_body_motion)
        self.assertFalse(config.behavior.prefer_vmd_expressions)
        self.assertEqual(config.behavior.transition_history_size, 24)
        with self.assertRaisesRegex(ValueError, "transition_history_size"):
            PluginConfig.from_mapping({"behavior": {"transition_history_size": 2}})

    def test_vision_config_is_disabled_and_bounded_by_default(self) -> None:
        default = PluginConfig.from_mapping({})
        self.assertFalse(default.vision.enabled)
        self.assertEqual(default.vision.source, "none")
        configured = PluginConfig.from_mapping({
            "vision": {
                "enabled": True,
                "source": "external",
                "interval_ms": 50,
                "queue_size": 2,
            }
        })
        self.assertTrue(configured.vision.enabled)
        self.assertEqual(configured.vision.interval_ms, 50)
        self.assertEqual(configured.vision.queue_size, 2)
        with self.assertRaisesRegex(ValueError, "vision.source"):
            PluginConfig.from_mapping({"vision": {"source": "unknown"}})
        with self.assertRaisesRegex(ValueError, "vision.interval_ms"):
            PluginConfig.from_mapping({"vision": {"interval_ms": 5}})

    def test_integer_config_values_are_not_silently_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "anyadance.port"):
            PluginConfig.from_mapping({"anyadance": {"port": 39570.5}})

    def test_vrchat_osc_config_is_bounded(self) -> None:
        config = PluginConfig.from_mapping({
            "vrchat_osc": {
                "enabled": True,
                "send_port": 9100,
                "listen_port": 9101,
                "input_pulse_ms": 80,
                "awareness_parameters": ["NEKO_Action", "NEKO_Holding"],
            }
        })
        self.assertEqual(config.vrchat_osc.send_port, 9100)
        self.assertEqual(config.vrchat_osc.listen_port, 9101)
        self.assertEqual(config.vrchat_osc.input_pulse_ms, 80)
        self.assertEqual(config.vrchat_osc.awareness_parameters, ("NEKO_Action", "NEKO_Holding"))

        with self.assertRaisesRegex(ValueError, "input_pulse_ms"):
            PluginConfig.from_mapping({"vrchat_osc": {"input_pulse_ms": 5}})

    def test_vmc_idle_config_is_bounded(self) -> None:
        config = PluginConfig.from_mapping({
            "vmc_idle": {
                "enabled": True,
                "listen_port": 39540,
                "stale_after_ms": 750,
            }
        })
        self.assertEqual(config.vmc_idle.listen_port, 39540)
        self.assertEqual(config.vmc_idle.stale_after_ms, 750)
        self.assertTrue(config.vmc_idle.manage_host_output)
        with self.assertRaisesRegex(ValueError, "stale_after_ms"):
            PluginConfig.from_mapping({"vmc_idle": {"stale_after_ms": 20}})

    def test_safety_config_is_bounded(self) -> None:
        config = PluginConfig.from_mapping({
            "safety": {"max_position_abs_m": 30.0, "max_y_m": 25.0}
        })
        self.assertEqual(config.safety.max_position_abs_m, 30.0)
        self.assertEqual(config.safety.max_y_m, 25.0)

        with self.assertRaisesRegex(ValueError, "max_y_m"):
            PluginConfig.from_mapping({"safety": {"max_y_m": 25.1}})
        # A Y ceiling above the all-axis position bound would make the .nya
        # loader clamp to a value the frame validator then rejects.
        with self.assertRaisesRegex(ValueError, "max_position_abs_m"):
            PluginConfig.from_mapping({"safety": {"max_y_m": 10.0}})


if __name__ == "__main__":
    unittest.main()
