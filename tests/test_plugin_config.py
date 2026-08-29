"""插件配置安全门和 manifest 冲突声明测试。"""

from pathlib import Path
import tomllib
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.config import YuiPluginConfig


ROOT = Path(__file__).resolve().parents[1]


def _conflicts(manifest: dict) -> set[str]:
    dependencies = manifest.get("plugin", {}).get("dependency", [])
    return {
        item["id"]
        for item in dependencies
        if isinstance(item, dict) and item.get("conflicts") is True
    }


class YuiPluginConfigTests(unittest.TestCase):
    def test_default_configuration_keeps_all_sensitive_gates_closed(self) -> None:
        config = YuiPluginConfig.from_mapping({})
        self.assertFalse(config.host_arm_authorized)
        self.assertFalse(config.free_coordinate_navigation)
        self.assertFalse(config.include_player_names)
        self.assertFalse(config.enable_wander_tool)

    def test_boolean_security_fields_reject_string_coercion(self) -> None:
        with self.assertRaisesRegex(ValueError, "host_arm_authorized"):
            YuiPluginConfig.from_mapping({"host_arm_authorized": "false"})
        with self.assertRaisesRegex(ValueError, "enable_wander_tool"):
            YuiPluginConfig.from_mapping({"enable_wander_tool": "true"})

    def test_log_source_and_deadlines_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "只能配置一个"):
            YuiPluginConfig.from_mapping({"log_path": "a", "log_directory": "b"})
        with self.assertRaisesRegex(ValueError, "command_deadline_s"):
            YuiPluginConfig.from_mapping(
                {"ack_timeout_s": 2.0, "command_deadline_s": 1.0}
            )

    def test_yui_manifest_declares_anydance_conflict(self) -> None:
        with (ROOT / "plugin.toml").open("rb") as handle:
            yui_manifest = tomllib.load(handle)
        self.assertIn("neko_anyadance_body", _conflicts(yui_manifest))


if __name__ == "__main__":
    unittest.main()
