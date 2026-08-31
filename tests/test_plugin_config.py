"""插件配置安全门和 manifest 冲突声明测试。"""

import ast
import importlib.util
import json
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

    def test_host_diagnostic_entries_are_hidden_from_automatic_agent(self) -> None:
        module = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        host_entries = {
            "yui_connect",
            "yui_clear_estop",
            "yui_disconnect",
            "yui_status",
            "yui_reload_config",
        }
        hidden_entries: set[str] = set()
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if not isinstance(decorator.func, ast.Name) or decorator.func.id != "plugin_entry":
                    continue
                keywords = {item.arg: item.value for item in decorator.keywords if item.arg}
                entry_id = keywords.get("id")
                metadata = keywords.get("metadata")
                if (
                    isinstance(entry_id, ast.Constant)
                    and entry_id.value in host_entries
                    and isinstance(metadata, ast.Name)
                    and metadata.id == "_HOST_ONLY_ENTRY_METADATA"
                ):
                    hidden_entries.add(str(entry_id.value))
        self.assertEqual(hidden_entries, host_entries)

        plugin_module = sys.modules[_plugin_class().__module__]
        self.assertEqual(
            plugin_module._HOST_ONLY_ENTRY_METADATA,
            {
                "agent_hidden": True,
                "agent_auto": False,
                "agent_exposed": False,
                "llm_exposed": False,
            },
        )

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

    def test_passive_world_context_is_private_complete_and_deduplicated(self) -> None:
        plugin = _plugin_class()(None)
        session = YuiSessionState()
        session.spec_version = "1.3"
        session.session = 77
        session.control_state = "external"
        session.catalog_revision = 9
        session.capabilities = (
            "operation_lifecycle",
            "world_map",
            "semantic_navigation",
            "region_localization",
            "local_navigation",
        )
        session.npc_state = {
            "pos": [1.0, 0.0, 2.0],
            "active_ops": [],
            "location": {
                "localized": True,
                "region_key": "ground_floor",
                "floor_label": "G",
                "nearest_anchor": {
                    "semantic_key": "spawn",
                    "d": 1.2,
                    "brg": 20.0,
                },
            },
        }
        session.players[0] = {
            "slot": 0,
            "pid": 123,
            "name": "不得注入的玩家名",
            "d": 3.0,
            "brg": 15.0,
        }
        session.catalogs["anchor"][0] = {
            "id": 0,
            "semantic_key": "spawn",
            "pos": [0.0, 0.0, 0.0],
        }
        session.catalogs["region"][0] = {
            "id": 0,
            "semantic_key": "ground_floor",
            "description_zh": "一楼",
            "entry_anchor_id": 0,
            "explorable": True,
        }
        session.catalogs["entity"][0] = {
            "id": 0,
            "semantic_key": "central_obstacle",
            "description_zh": "中央障碍物",
            "center": [2.0, 0.0, 2.0],
            "approach_anchor_id": 0,
            "orbitable": True,
        }

        class FakeAdapter:
            def observe(self, *, include_player_names=False):
                result = session.observe(include_player_names=include_player_names)
                result["world"] = session.nearby_world(limit=8)
                return result

            @staticmethod
            def plan_status():
                return {"status": "failed", "error": "plan_not_found"}

        plugin._session = session
        plugin._adapter = FakeAdapter()
        plugin._surface = object()
        plugin._registered_yui_tools = {"npc.observe", "npc.move_relative"}
        pushed = []
        plugin.push_message = lambda **kwargs: pushed.append(kwargs) or {"ok": True}

        self.assertTrue(plugin._push_context_snapshot(force=True))
        self.assertFalse(plugin._push_context_snapshot())
        self.assertEqual(len(pushed), 1)
        message = pushed[0]
        self.assertEqual(message["ai_behavior"], "read")
        self.assertEqual(message["visibility"], [])
        self.assertEqual(message["coalesce_key"], "yui_world_context")
        body = message["parts"][0]["text"]
        self.assertTrue(body.startswith("YUI_WORLD_CONTEXT "))
        payload = json.loads(body.removeprefix("YUI_WORLD_CONTEXT "))
        self.assertEqual(payload["world"]["session"], 77)
        self.assertEqual(payload["world"]["location"]["region_key"], "ground_floor")
        self.assertEqual(payload["world"]["players"][0]["slot"], 0)
        self.assertEqual(payload["catalog"]["revision"], 9)
        self.assertEqual(payload["available_tools"], ["npc.move_relative", "npc.observe"])
        nearby_keys = {
            item.get("semantic_key")
            for item in payload["world"]["world"]["items"]
        }
        self.assertIn("central_obstacle", nearby_keys)
        encoded = json.dumps(payload, ensure_ascii=False)
        for private_field in ('"name"', '"pid"', '"pos"', '"center"', '"x"', '"y"', '"z"'):
            self.assertNotIn(private_field, encoded)

        # 玩家距离高频变化不重复注入；楼层/区域变化必须产生新上下文。
        session.players[0]["d"] = 2.0
        self.assertFalse(plugin._push_context_snapshot())
        session.npc_state["location"]["region_key"] = "upper_floor"
        session.npc_state["location"]["floor_label"] = "L1"
        self.assertTrue(plugin._push_context_snapshot())
        self.assertEqual(len(pushed), 2)


if __name__ == "__main__":
    unittest.main()
