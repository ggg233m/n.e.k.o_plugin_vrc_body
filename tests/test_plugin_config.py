"""插件配置安全门和 manifest 冲突声明测试。"""

import importlib.util
from pathlib import Path
import sys
import tomllib
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.config import YuiPluginConfig
from yui_npc_controller.runtime.yui_session import YuiSessionState


ROOT = Path(__file__).resolve().parents[1]


def _plugin_class():
    """用独立包名装载宿主入口，避免覆盖纯 runtime 测试使用的包桩。"""
    name = "yui_npc_controller_plugin_test"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("无法装载 YUI 插件入口")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.YuiNpcControllerPlugin


def _conflicts(manifest: dict) -> set[str]:
    dependencies = manifest.get("plugin", {}).get("dependency", [])
    return {
        item["id"]
        for item in dependencies
        if isinstance(item, dict) and item.get("conflicts") is True
    }


class YuiPluginConfigTests(unittest.TestCase):
    def test_default_configuration_keeps_optional_data_and_tools_closed(self) -> None:
        config = YuiPluginConfig.from_mapping({})
        self.assertFalse(config.free_coordinate_navigation)
        self.assertFalse(config.include_player_names)
        self.assertFalse(config.enable_wander_tool)

    def test_boolean_security_fields_reject_string_coercion(self) -> None:
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

    def test_unchanged_runtime_state_does_not_rebuild_tool_schema(self) -> None:
        plugin = _plugin_class()(None)
        session = YuiSessionState()
        session.session = 7
        session.control_state = "external"
        session.capabilities = ("operation_lifecycle",)

        class CountingSurface:
            free_coordinate_navigation = False
            include_player_names = False
            enable_wander_tool = False

            def __init__(self) -> None:
                self.calls = 0

            def definitions(self):
                self.calls += 1
                return []

        surface = CountingSurface()
        plugin._session = session
        plugin._surface = surface
        plugin._registered_yui_tools = set()
        plugin._tool_signature = ""
        plugin._tool_state_key = None

        plugin._refresh_llm_tools()
        plugin._refresh_llm_tools()
        self.assertEqual(surface.calls, 1)

        # moving/external 的工具面相同，高频 npc.state 不应触发 schema 重算。
        session.control_state = "moving"
        plugin._refresh_llm_tools()
        self.assertEqual(surface.calls, 1)


if __name__ == "__main__":
    unittest.main()
