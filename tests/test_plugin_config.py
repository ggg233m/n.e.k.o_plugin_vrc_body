"""插件配置安全门和 manifest 冲突声明测试。"""

import asyncio
import ast
import importlib.util
import json
from pathlib import Path
import sys
import threading
import tomllib
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.config import YuiAutonomyConfig, YuiPluginConfig
from yui_npc_controller.runtime.tool_surface import YuiToolSurface
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
        self.assertTrue(config.chat_bridge.enabled)
        self.assertEqual(config.chat_bridge.display_seconds, 15)
        self.assertTrue(config.player_chat.enabled)
        self.assertEqual(config.player_chat.max_chars, 144)
        self.assertEqual(config.player_chat.cooldown_s, 2.0)
        self.assertFalse(config.autonomy.enabled)
        self.assertFalse(config.autonomy.auto_connect)
        self.assertFalse(config.autonomy.intent_model.enabled)
        self.assertEqual(config.autonomy.intent_model.api_key_env, "TEST_API")
        self.assertEqual(config.autonomy.intent_model.endpoint, "")
        self.assertFalse(config.autonomy.intent_model.chat_context.enabled)

    def test_autonomy_configuration_is_nested_and_strict(self) -> None:
        config = YuiPluginConfig.from_mapping({
            "chat_bridge": {
                "enabled": True,
                "source": "recent_file",
                "poll_interval_s": 0.5,
                "display_seconds": 8,
                "max_pages": 4,
                "max_file_bytes": 2097152,
            },
            "player_chat": {
                "enabled": True,
                "max_chars": 144,
                "cooldown_s": 2.0,
            },
            "autonomy": {
                "enabled": True,
                "auto_connect": True,
                "decision_interval_s": 0.5,
                "resume_delay_s": 8,
                "walk_speed_mps": 1.0,
                "dwell_range_s": [8, 20],
                "explore_range_s": [15, 35],
                "social_cooldown_s": 60,
                "llm_inspiration_range_s": [180, 360],
                "intent_model": {
                    "enabled": True,
                    "endpoint": "https://relay.example.com/v1/chat/completions",
                    "model": "gemini-3.7-flash",
                    "api_key_env": "TEST_API",
                    "timeout_s": 20,
                    "min_interval_s": 30,
                    "temperature": 0.7,
                    "max_output_tokens": 700,
                    "chat_context": {
                        "enabled": True,
                        "source": "recent_file",
                        "max_turns": 6,
                        "max_chars": 6000,
                        "poll_interval_s": 1,
                        "max_file_bytes": 2097152,
                    },
                },
            }
        })
        self.assertTrue(config.chat_bridge.enabled)
        self.assertEqual(config.chat_bridge.max_pages, 4)
        self.assertTrue(config.player_chat.enabled)
        self.assertTrue(config.autonomy.enabled)
        self.assertTrue(config.autonomy.auto_connect)
        self.assertEqual(config.autonomy.walk_speed_mps, 1.0)
        self.assertEqual(config.autonomy.dwell_range_s, (8.0, 20.0))
        self.assertTrue(config.autonomy.intent_model.enabled)
        self.assertEqual(
            config.autonomy.intent_model.endpoint,
            "https://relay.example.com/v1/chat/completions",
        )
        self.assertEqual(config.autonomy.intent_model.temperature, 0.7)
        self.assertTrue(config.autonomy.intent_model.chat_context.enabled)
        self.assertEqual(config.autonomy.intent_model.chat_context.max_turns, 6)
        with self.assertRaisesRegex(ValueError, "dwell_range_s"):
            YuiPluginConfig.from_mapping({"autonomy": {"dwell_range_s": [20, 8]}})
        with self.assertRaisesRegex(ValueError, "walk_speed_mps"):
            YuiPluginConfig.from_mapping({"autonomy": {"walk_speed_mps": 0}})
        with self.assertRaisesRegex(ValueError, "chat_context.source"):
            YuiPluginConfig.from_mapping({
                "autonomy": {"intent_model": {"chat_context": {"source": "conversation_bus"}}}
            })
        with self.assertRaisesRegex(ValueError, "chat_bridge.source"):
            YuiPluginConfig.from_mapping({
                "chat_bridge": {"source": "conversation_bus"}
            })
        with self.assertRaisesRegex(ValueError, "player_chat.max_chars"):
            YuiPluginConfig.from_mapping({"player_chat": {"max_chars": 145}})

    def test_intent_model_endpoint_is_empty_until_configured(self) -> None:
        for value in ("", "   "):
            with self.subTest(endpoint=value):
                config = YuiPluginConfig.from_mapping({
                    "autonomy": {"intent_model": {"enabled": True, "endpoint": value}}
                })
                self.assertEqual(config.autonomy.intent_model.endpoint, "")
        omitted = YuiPluginConfig.from_mapping({"autonomy": {"intent_model": {}}})
        self.assertEqual(omitted.autonomy.intent_model.endpoint, "")
        with self.assertRaisesRegex(ValueError, "intent_model.endpoint"):
            YuiPluginConfig.from_mapping({
                "autonomy": {"intent_model": {"endpoint": "http://relay.example.com/v1/chat/completions"}}
            })

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
            "yui_autonomy_start",
            "yui_autonomy_pause",
            "yui_autonomy_status",
            "yui_autonomy_intent_probe",
            "yui_chat_bridge_status",
            "yui_player_chat_status",
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

    def test_neko_registration_reuses_mcp_execute_plan_schema(self) -> None:
        plugin = _plugin_class()(None)
        session = YuiSessionState()
        session.spec_version = "1.3"
        session.session = 7
        session.control_state = "external"
        session.capabilities = (
            "goto",
            "navmesh",
            "anchors",
            "operation_lifecycle",
            "world_map",
            "semantic_navigation",
            "region_localization",
            "local_navigation",
        )
        session.max_speed_mps = 2.0
        session.catalogs["anchor"][0] = {
            "id": 0,
            "semantic_key": "spawn_point",
            "pos": [0.0, 0.0, 0.0],
        }
        session.catalogs["entity"][0] = {
            "id": 0,
            "semantic_key": "central_obstacle",
            "approach_anchor_id": 0,
            "orbitable": True,
        }
        surface = YuiToolSurface(object(), session)  # type: ignore[arg-type]
        source_definition = next(
            item for item in surface.definitions() if item.name == "npc.execute_plan"
        )
        registered = {}
        plugin.register_llm_tool = (
            lambda **kwargs: registered.setdefault(kwargs["name"], kwargs) or True
        )
        plugin.unregister_llm_tool = lambda _name: True
        plugin._session = session
        plugin._surface = surface
        plugin._registered_yui_tools = set()
        plugin._tool_signature = ""
        plugin._tool_state_key = None

        plugin._refresh_llm_tools()

        projected = registered["npc.execute_plan"]
        self.assertEqual(projected["parameters"], source_definition.input_schema)
        self.assertEqual(
            source_definition.as_mcp_tool()["inputSchema"],
            projected["parameters"],
        )
        self.assertNotIn("npc.plan_step", registered)

    def test_neko_handler_marks_failed_tool_result_as_error(self) -> None:
        plugin = _plugin_class()(None)
        failure = {
            "status": "failed",
            "error": "behavior_graph_invalid",
            "detail": "arguments.graph.nodes[0].target_key 为必填字段",
            "midi_sent": False,
        }

        class FailedSurface:
            @staticmethod
            def call(_name, _arguments):
                return dict(failure)

        plugin._surface = FailedSurface()
        plugin._refresh_llm_tools = lambda: []
        plugin._push_context_snapshot = lambda: False
        handler = plugin._make_tool_handler("npc.execute_plan")

        result = asyncio.run(handler(graph={"entry": "root", "nodes": []}))

        self.assertTrue(result["is_error"])
        self.assertEqual(result["error"], "behavior_graph_invalid")
        self.assertEqual(result["output"], failure)

    def test_neko_handler_rejects_stale_callback_after_disconnect(self) -> None:
        plugin = _plugin_class()(None)
        plugin._surface = None
        handler = plugin._make_tool_handler("npc.navigate")

        result = asyncio.run(handler(target_key="upper_observation"))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "not_connected")
        self.assertFalse(result["midi_sent"])
        self.assertNotIn("output", result)

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
        self.assertEqual(payload["context_revision"], 2)
        self.assertTrue(payload["connection"]["connected"])
        self.assertTrue(payload["connection"]["fresh"])
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
        instructions = "\n".join(payload["instructions"])
        self.assertIn("本轮必须先调用 npc.observe", instructions)
        self.assertIn("严禁回复", instructions)
        self.assertIn("accepted 只表示已受理", instructions)
        self.assertIn("只供内部追踪", instructions)
        self.assertIn("禁止在面向用户的可朗读回复中输出", instructions)
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

    def test_autonomy_and_social_events_never_start_host_chat(self) -> None:
        source = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('ai_behavior="respond"'), 1)

        plugin = _plugin_class()(None)
        pushed = []
        plugin.push_message = lambda **kwargs: pushed.append(kwargs) or {"ok": True}
        plugin._handle_session_event({"type": "social.wave", "slot": 2})
        plugin._handle_session_event({"type": "player.touch", "slot": 2})
        self.assertEqual(pushed, [])

        plugin._handle_session_event({"type": "sys.err", "code": "test_failure"})
        self.assertEqual(len(pushed), 1)
        self.assertEqual(pushed[0]["visibility"], ["hud"])
        self.assertEqual(pushed[0]["ai_behavior"], "blind")
        self.assertNotIn("respond", json.dumps(pushed[0], ensure_ascii=False))

    def test_only_explicit_world_chat_submit_starts_host_reply(self) -> None:
        plugin = _plugin_class()(None)
        session = YuiSessionState()
        session.session = 31
        session.players[2] = {"slot": 2, "pid": 90210, "name": "不应上传的名字"}
        plugin._session = session
        plugin._registered_yui_tools = {
            "npc.approach", "npc.follow", "npc.observe",
        }
        pushed = []
        plugin.push_message = lambda **kwargs: pushed.append(kwargs) or {"submitted": True}

        class _CurrentCharacterProvider:
            def poll(self, *, force=False):
                self.forced = force

            @staticmethod
            def status():
                return {"current_character": "测试猫娘"}

        class _ReplyDisplay:
            provider = _CurrentCharacterProvider()

        plugin._reply_display = _ReplyDisplay()

        plugin._handle_session_event({"type": "sys.chat_input_ready", "ready": True})
        plugin._handle_session_event({
            "type": "player.chat_submit",
            "session": 31,
            "slot": 2,
            "pid": 90210,
            "submit_seq": 7,
            "text": "过来",
        })

        self.assertEqual(len(pushed), 1)
        message = pushed[0]
        self.assertEqual(message["visibility"], [])
        self.assertEqual(message["ai_behavior"], "respond")
        self.assertEqual(message["source"], "yui_npc_controller.world_chat")
        self.assertEqual(message["target_lanlan"], "测试猫娘")
        request_text = message["parts"][0]["text"]
        self.assertTrue(request_text.startswith("YUI_WORLD_CHAT_REQUEST "))
        request = json.loads(request_text.removeprefix("YUI_WORLD_CHAT_REQUEST "))
        self.assertEqual(request["player_slot"], 2)
        self.assertEqual(request["player_text"], "过来")
        self.assertTrue(request["world_context"]["connection"]["connected"])
        self.assertIn("npc.approach", request["world_context"]["available_tools"])
        self.assertTrue(any(
            "npc.approach" in item for item in request["turn_contract"]
        ))
        self.assertTrue(any(
            "npc.observe" in item for item in request["turn_contract"]
        ))
        self.assertEqual(message["metadata"]["projection"], "model_context")
        self.assertNotIn("不应上传的名字", json.dumps(message, ensure_ascii=False))
        self.assertNotIn("text", message["metadata"])

        # 相同提交、伪造槽位和冷却期内的新提交均不得再次触发。
        plugin._handle_session_event({
            "type": "player.chat_submit", "session": 31, "slot": 2,
            "pid": 90210, "submit_seq": 7, "text": "重复",
        })
        plugin._handle_session_event({
            "type": "player.chat_submit", "session": 31, "slot": 2,
            "pid": 123, "submit_seq": 8, "text": "伪造",
        })
        plugin._handle_session_event({
            "type": "player.chat_submit", "session": 31, "slot": 2,
            "pid": 90210, "submit_seq": 8, "text": "太快",
        })
        self.assertEqual(len(pushed), 1)
        status = plugin._player_chat_status_snapshot()
        self.assertTrue(status["world_ui_ready"])
        self.assertNotIn("今天去看看照片吧", json.dumps(status, ensure_ascii=False))

    def test_offline_context_expires_stale_world_and_forbids_action_promises(self) -> None:
        plugin = _plugin_class()(None)
        plugin._registered_yui_tools = {"npc.observe", "npc.navigate"}
        pushed = []
        plugin.push_message = lambda **kwargs: pushed.append(kwargs) or {"ok": True}

        self.assertTrue(
            plugin._push_context_unavailable("plugin_started_disconnected", force=True)
        )
        self.assertFalse(plugin._push_context_unavailable("plugin_started_disconnected"))

        self.assertEqual(len(pushed), 1)
        message = pushed[0]
        self.assertEqual(message["coalesce_key"], "yui_world_context")
        self.assertFalse(message["metadata"]["connected"])
        body = message["parts"][0]["text"]
        payload = json.loads(body.removeprefix("YUI_WORLD_CONTEXT "))
        self.assertFalse(payload["connection"]["connected"])
        self.assertFalse(payload["connection"]["fresh"])
        self.assertFalse(payload["world"]["available"])
        self.assertEqual(payload["available_tools"], [])
        self.assertIsNone(payload["plan"])
        instructions = "\n".join(payload["instructions"])
        self.assertIn("此前注入的全部位置", instructions)
        self.assertIn("不得声称刚刚观察过世界", instructions)
        self.assertIn("连接成功后必须由用户重新发起", instructions)
        self.assertIn("YUI 未连接", instructions)
        self.assertEqual(plugin._context_push_status["state"], "deduplicated")

    def test_close_control_unregisters_tools_and_pushes_offline_context(self) -> None:
        plugin = _plugin_class()(None)
        closed = []
        released = []
        unregistered = []
        pushed = []

        class Closable:
            @staticmethod
            def close():
                closed.append(True)

        class Lease:
            @staticmethod
            def release():
                released.append(True)

        plugin._adapter = Closable()
        plugin._transport = Closable()
        plugin._surface = object()
        plugin._driver_lease = Lease()
        plugin._registered_yui_tools = {"npc.observe", "npc.navigate"}
        plugin.unregister_llm_tool = lambda name: unregistered.append(name) or True
        plugin.push_message = lambda **kwargs: pushed.append(kwargs) or {"ok": True}

        plugin._close_control(reason="host_disconnected")

        self.assertEqual(unregistered, ["npc.navigate", "npc.observe"])
        self.assertEqual(len(closed), 2)
        self.assertEqual(len(released), 1)
        self.assertIsNone(plugin._adapter)
        self.assertIsNone(plugin._transport)
        self.assertIsNone(plugin._surface)
        self.assertIsNone(plugin._driver_lease)
        self.assertEqual(plugin._registered_yui_tools, set())
        self.assertEqual(plugin._context_push_status["state"], "offline_sent")
        payload = json.loads(
            pushed[-1]["parts"][0]["text"].removeprefix("YUI_WORLD_CONTEXT ")
        )
        self.assertEqual(payload["connection"]["reason"], "host_disconnected")

    def test_session_rebuild_preserves_physical_midi_sink(self) -> None:
        plugin = _plugin_class()(None)
        adapter_closed = []
        heartbeat_stopped = []
        transport_closed = []
        sink_closed = []

        class Adapter:
            @staticmethod
            def close():
                adapter_closed.append(True)

        class Transport:
            @staticmethod
            def stop_heartbeat():
                heartbeat_stopped.append(True)

            @staticmethod
            def close():
                transport_closed.append(True)

        class Sink:
            @staticmethod
            def close():
                sink_closed.append(True)

        sink = Sink()
        plugin._adapter = Adapter()
        plugin._transport = Transport()
        plugin._midi_sink = sink
        plugin._push_context_unavailable = lambda *_args, **_kwargs: True

        plugin._close_control(reason="connect_ack_timeout", preserve_midi=True)

        self.assertEqual(adapter_closed, [True])
        self.assertEqual(heartbeat_stopped, [True])
        self.assertEqual(transport_closed, [])
        self.assertEqual(sink_closed, [])
        self.assertIs(plugin._midi_sink, sink)
        self.assertIsNone(plugin._transport)

    def test_ack_timeout_rebuild_requests_make_before_break_midi_refresh(self) -> None:
        plugin = _plugin_class()(None)
        closed = []
        plugin._close_control = lambda **kwargs: closed.append(kwargs)

        rebuilt = plugin._rebuild_failed_connection({
            "status": "unknown",
            "error": "ack_timeout",
            "session_rebuild_required": True,
        })

        self.assertTrue(rebuilt)
        self.assertTrue(plugin._midi_refresh_required)
        self.assertEqual(closed, [{
            "reason": "connect_ack_timeout",
            "preserve_midi": True,
        }])

    def test_midi_refresh_opens_new_sink_before_closing_stale_sink(self) -> None:
        plugin_class = _plugin_class()
        module = sys.modules[plugin_class.__module__]
        plugin = plugin_class(None)
        events = []

        class Sink:
            def __init__(self, name):
                self.name = name
                self.closed = False

            def close(self):
                self.closed = True
                events.append(f"close:{self.name}")

        stale = Sink("stale")
        fresh = Sink("fresh")
        plugin._midi_sink = stale
        plugin._midi_refresh_required = True
        original_factory = module.MidoOutputSink

        def open_fresh(port_name):
            self.assertEqual(port_name, "NEKO_MIDI")
            self.assertFalse(stale.closed)
            events.append("open:fresh")
            return fresh

        module.MidoOutputSink = open_fresh
        try:
            sink, replaced = plugin._acquire_midi_sink()
        finally:
            module.MidoOutputSink = original_factory

        self.assertTrue(replaced)
        self.assertIs(sink, fresh)
        self.assertIs(plugin._midi_sink, fresh)
        self.assertFalse(plugin._midi_refresh_required)
        self.assertEqual(events, ["open:fresh", "close:stale"])

    def test_startup_announces_disconnected_state_without_registering_tools(self) -> None:
        plugin = _plugin_class()(None)
        pushed = []

        async def load_config():
            return YuiPluginConfig()

        plugin._load_config = load_config
        plugin._configure_reply_display = lambda: None
        plugin._start_log_tailer = lambda: None
        plugin.push_message = lambda **kwargs: pushed.append(kwargs) or {"ok": True}

        result = asyncio.run(plugin.startup())

        self.assertEqual(result.value["status"], "ready")
        self.assertFalse(result.value["result"]["midi_open"])
        self.assertEqual(result.value["result"]["llm_tools"], [])
        payload = json.loads(
            pushed[-1]["parts"][0]["text"].removeprefix("YUI_WORLD_CONTEXT ")
        )
        self.assertFalse(payload["connection"]["connected"])
        self.assertEqual(
            payload["connection"]["reason"], "plugin_started_disconnected"
        )

    def test_auto_connect_starts_autonomy_after_successful_handshake(self) -> None:
        plugin = _plugin_class()(None)
        plugin._config = YuiPluginConfig(
            autonomy=YuiAutonomyConfig(enabled=True, auto_connect=True)
        )
        plugin._manual_disconnect = False
        plugin._session = type(
            "Session",
            (),
            {
                "discovery_ready": True,
                "control_state": "external",
                "npc_state": {"state": "external"},
            },
        )()
        started = []
        order = []

        class Adapter:
            @staticmethod
            def connect(_claim_code):
                return {"status": "succeeded"}

        class Autonomy:
            @staticmethod
            def status():
                return {"running": False, "pause_reason": "not_started"}

            @staticmethod
            def start():
                started.append(True)
                order.append("autonomy")
                return {"state": "ready", "running": True, "pause_reason": None}

        plugin._ensure_control = lambda: Adapter()
        plugin._autonomy = Autonomy()
        plugin._refresh_llm_tools = lambda: order.append("tools") or []
        plugin._push_context_snapshot = lambda **_kwargs: order.append("context") or True

        asyncio.run(plugin._auto_connect_loop())
        self.assertEqual(started, [True])
        self.assertEqual(order, ["autonomy", "tools", "context"])

    def test_auto_connect_worker_survives_lifecycle_loop_and_retries(self) -> None:
        plugin = _plugin_class()(None)
        plugin._config = YuiPluginConfig(
            autonomy=YuiAutonomyConfig(enabled=True, auto_connect=True)
        )
        plugin._manual_disconnect = False
        first_attempt = threading.Event()
        connected = threading.Event()
        started = threading.Event()
        attempts = []
        rebuilt = []

        class FailedAdapter:
            @staticmethod
            def connect(_claim_code):
                attempts.append(len(attempts) + 1)
                first_attempt.set()
                return {
                    "status": "unknown",
                    "error": "ack_timeout",
                    "session_rebuild_required": True,
                }

        class ConnectedAdapter:
            @staticmethod
            def connect(_claim_code):
                attempts.append(len(attempts) + 1)
                connected.set()
                return {"status": "succeeded"}

        class Autonomy:
            running = False

            def status(self):
                return {
                    "running": self.running,
                    "pause_reason": None if self.running else "not_started",
                }

            def start(self):
                self.running = True
                started.set()
                return self.status()

        adapters = iter((FailedAdapter(), ConnectedAdapter()))
        plugin._ensure_control = lambda: next(adapters)
        plugin._close_control = lambda *, reason, **_kwargs: rebuilt.append(reason)
        plugin._autonomy = Autonomy()
        plugin._refresh_llm_tools = lambda: []
        plugin._push_context_snapshot = lambda **_kwargs: True

        try:
            # 调度发生在一个马上销毁的 lifecycle loop 内；连接线程仍须存活并重试。
            async def schedule_from_lifecycle_loop():
                plugin._schedule_auto_connect()

            asyncio.run(schedule_from_lifecycle_loop())
            self.assertTrue(first_attempt.wait(1.0))
            self.assertTrue(plugin._auto_connect_thread.is_alive())
            self.assertTrue(connected.wait(3.5))
            self.assertTrue(started.wait(1.0))
            self.assertEqual(attempts, [1, 2])
            self.assertEqual(rebuilt, ["connect_ack_timeout"])
            self.assertEqual(plugin._connect_rebuilds, 1)
            self.assertIsNone(plugin._last_connect_error)
        finally:
            plugin._cancel_auto_connect_task()

    def test_manual_connect_rebuilds_failed_chain_and_resumes_auto_connect(self) -> None:
        plugin = _plugin_class()(None)
        plugin._config = YuiPluginConfig(
            autonomy=YuiAutonomyConfig(enabled=True, auto_connect=True)
        )
        cancelled = []
        rebuilt = []
        scheduled = []

        class Adapter:
            @staticmethod
            def connect(_claim_code):
                return {
                    "status": "unknown",
                    "error": "ack_timeout",
                    "session_rebuild_required": True,
                }

        plugin._cancel_auto_connect_task = lambda: cancelled.append(True)
        plugin._ensure_control = lambda: Adapter()
        plugin._close_control = lambda *, reason, **_kwargs: rebuilt.append(reason)
        plugin._ensure_auto_connect_worker = lambda: scheduled.append(True) or True

        response = asyncio.run(plugin.yui_connect())

        self.assertEqual(cancelled, [True])
        self.assertEqual(rebuilt, ["connect_ack_timeout"])
        self.assertEqual(scheduled, [True])
        self.assertEqual(response.value["status"], "unknown")
        self.assertTrue(response.value["reconnect_scheduled"])
        self.assertEqual(plugin._connect_rebuilds, 1)
        self.assertEqual(plugin._last_connect_error, "ack_timeout")

    def test_ready_session_event_recovers_autonomy_startup_race(self) -> None:
        plugin = _plugin_class()(None)
        plugin._config = YuiPluginConfig(
            autonomy=YuiAutonomyConfig(enabled=True, auto_connect=True)
        )
        plugin._manual_disconnect = False
        plugin._session = type(
            "Session",
            (),
            {
                "discovery_ready": True,
                "control_state": "external",
                "npc_state": {"state": "safe_idle"},
            },
        )()
        started = []

        class Autonomy:
            @staticmethod
            def status():
                return {"running": False, "pause_reason": "not_started"}

            @staticmethod
            def start():
                started.append(True)
                return {"state": "ready"}

        plugin._autonomy = Autonomy()
        plugin._refresh_llm_tools = lambda: []
        plugin._push_context_snapshot = lambda **_kwargs: True

        # ACK 已切换 control_state，但积压遥测仍是 safe_idle 时不能抢跑。
        plugin._handle_session_event({"type": "npc.ack"})
        self.assertEqual(started, [])

        plugin._session.npc_state["state"] = "external"
        plugin._handle_session_event({"type": "npc.state"})
        self.assertEqual(started, [True])

        # 人工暂停不能被后续高频 npc.state 自动解除。
        plugin._autonomy.status = lambda: {
            "running": False,
            "pause_reason": "manual_pause",
        }
        plugin._handle_session_event({"type": "npc.state"})
        self.assertEqual(started, [True])

    def test_tailer_thread_starts_autonomy_without_event_loop(self) -> None:
        plugin = _plugin_class()(None)
        plugin._config = YuiPluginConfig(
            autonomy=YuiAutonomyConfig(enabled=True, auto_connect=True)
        )
        plugin._manual_disconnect = False
        plugin._event_loop = None
        plugin._session = type(
            "Session",
            (),
            {
                "discovery_ready": True,
                "control_state": "external",
                "npc_state": {"state": "external"},
            },
        )()
        started = []

        class Autonomy:
            running = False

            def status(self):
                return {
                    "running": self.running,
                    "pause_reason": None if self.running else "not_started",
                }

            def start(self):
                self.running = True
                started.append(True)
                return self.status()

        plugin._autonomy = Autonomy()
        plugin._on_session_event({"type": "npc.state", "state": "external"})
        plugin._on_session_event({"type": "npc.state", "state": "external"})
        self.assertEqual(started, [True])

    def test_world_restart_recovers_auto_connect_without_lifecycle_loop(self) -> None:
        plugin = _plugin_class()(None)
        plugin._config = YuiPluginConfig(
            autonomy=YuiAutonomyConfig(enabled=True, auto_connect=True)
        )
        plugin._manual_disconnect = False
        plugin._event_loop = None
        plugin._session = type("Session", (), {"session": 0})()
        scheduled = []
        plugin._schedule_auto_connect = lambda: scheduled.append(True)

        plugin._on_session_event({"type": "sys.boot", "session": 0})

        self.assertEqual(scheduled, [True])
        self.assertTrue(plugin._midi_refresh_required)
        self.assertEqual(
            plugin._player_chat_status_snapshot()["state"],
            "waiting_for_world",
        )

    def test_world_restart_does_not_duplicate_or_override_manual_disconnect(self) -> None:
        plugin = _plugin_class()(None)
        plugin._config = YuiPluginConfig(
            autonomy=YuiAutonomyConfig(enabled=True, auto_connect=True)
        )
        plugin._event_loop = None
        plugin._session = type("Session", (), {"session": 0})()
        scheduled = []
        plugin._schedule_auto_connect = lambda: scheduled.append(True)
        plugin._auto_connect_thread = type("Thread", (), {"is_alive": lambda _self: True})()

        plugin._on_session_event({
            "type": "npc.ack",
            "session": 0,
            "ok": False,
            "err": "not_handshaken",
        })
        self.assertEqual(scheduled, [])

        plugin._auto_connect_thread = None
        plugin._manual_disconnect = True
        plugin._on_session_event({"type": "sys.boot", "session": 0})
        self.assertEqual(scheduled, [])

    def test_tailer_thread_forwards_explicit_player_chat_without_event_loop(self) -> None:
        plugin = _plugin_class()(None)
        plugin._event_loop = None
        session = YuiSessionState()
        session.session = 31
        session.players[2] = {"slot": 2, "pid": 90210}
        plugin._session = session
        pushed = []
        plugin.push_message = lambda **kwargs: pushed.append(kwargs) or {"submitted": True}
        plugin._current_host_character = lambda: "测试猫娘"

        plugin._on_session_event({"type": "sys.chat_input_ready", "ready": True})
        plugin._on_session_event({
            "type": "player.chat_submit",
            "session": 31,
            "slot": 2,
            "pid": 90210,
            "submit_seq": 9,
            "text": "常驻日志线程提交测试",
        })

        self.assertEqual(len(pushed), 1)
        self.assertEqual(pushed[0]["visibility"], [])
        self.assertEqual(pushed[0]["ai_behavior"], "respond")
        self.assertIn("YUI_WORLD_CHAT_REQUEST", pushed[0]["parts"][0]["text"])
        self.assertEqual(pushed[0]["source"], "yui_npc_controller.world_chat")
        self.assertIn("常驻日志线程提交测试", pushed[0]["parts"][0]["text"])
        self.assertEqual(pushed[0]["target_lanlan"], "测试猫娘")
        status = plugin._player_chat_status_snapshot()
        self.assertTrue(status["world_ui_ready"])
        self.assertEqual(status["state"], "submitted")

    def test_intent_worker_survives_without_host_event_loop(self) -> None:
        plugin = _plugin_class()(None)
        delivered = threading.Event()

        class Provider:
            config = type("Config", (), {"enabled": True, "min_interval_s": 0.0})()

            @staticmethod
            async def request(_context):
                return {
                    "status": "succeeded",
                    "intent": {"motivation": "测试"},
                    "latency_ms": 1,
                    "format": "json_schema",
                }

        class Autonomy:
            @staticmethod
            def status():
                return {"running": True}

            @staticmethod
            def offer_intent(intent, token):
                if intent["motivation"] == "测试" and token == "1:1":
                    delivered.set()
                return True

        plugin._event_loop = None
        plugin._intent_provider = Provider()
        plugin._autonomy = Autonomy()
        plugin._start_intent_worker()
        try:
            plugin._queue_autonomy_inspiration({
                "request_token": "1:1",
                "reason": "startup",
                "context": {"catalog": {}},
            })
            self.assertTrue(delivered.wait(timeout=1.0))
        finally:
            plugin._stop_intent_worker()


if __name__ == "__main__":
    unittest.main()
