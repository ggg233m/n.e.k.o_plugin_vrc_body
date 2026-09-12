"""面板只能保存安全白名单，密钥只写不回显，不能修改导航边界。"""
import unittest
import asyncio
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.control_panel import persist_profile_compat, settings_view, validated_patch
from yui_npc_controller.runtime.config import YuiPluginConfig


class ControlPanelTests(unittest.TestCase):
    def test_dashboard_projects_model_panel_without_general_snapshot_contents(self):
        from test_plugin_config import _plugin_class
        cls = _plugin_class()
        instance = cls.__new__(cls)
        instance._config = YuiPluginConfig.from_mapping({"autonomy":{"intent_model":{"api_key":"test-dashboard-secret"}}})
        instance._load_config = AsyncMock(return_value=instance._config)
        instance._status_snapshot = Mock(return_value={"control_ready": False, "midi_open": False, "world": {"private": "world payload"}})
        instance._manual_disconnect = False
        instance._adapter = None
        instance._session = None
        instance._motion_backend = Mock()
        instance._motion_backend.snapshot.return_value = {"ready": False}
        instance.config = Mock()
        instance.config.profile_active = AsyncMock(return_value="test")
        instance._intent_provider = Mock()
        instance._intent_provider.panel_status.return_value = {"requests": 3, "last_call": {"output": "model result"}}
        view = asyncio.run(instance.yui_dashboard())
        self.assertNotIn("test-dashboard-secret", str(view))
        self.assertTrue(view["key_configured"])
        self.assertEqual(view["intent_model"]["requests"], 3)
        self.assertEqual(view["intent_model"]["last_call"]["output"], "model result")
        self.assertNotIn("world", view)
        self.assertEqual(view["action_progress"]["state"], "not_initialized")

    def test_save_uses_active_profile_without_reloading(self):
        from test_plugin_config import _plugin_class
        cls = _plugin_class()
        instance = cls.__new__(cls)
        instance._runtime_lock = asyncio.Lock()
        instance.config = type("ConfigStub", (), {})()
        instance.config.profile_active = AsyncMock(return_value="home")
        instance.config.dump = AsyncMock(return_value={})
        instance.config.profile_update = AsyncMock()
        instance.config.update = AsyncMock()
        asyncio.run(instance.yui_panel_save(changes={"chat_bridge.display_seconds": 15}, expected_profile="home"))
        instance.config.profile_update.assert_awaited_once_with("home", {"yui": {"chat_bridge": {"display_seconds": 15}}})
        instance.config.update.assert_not_awaited()
        instance.config.profile_update.reset_mock()
        asyncio.run(instance.yui_panel_save(changes={"autonomy.enabled": True}, expected_profile="old"))
        instance.config.profile_update.assert_not_awaited()

    def test_save_reports_validation_and_host_failures_separately(self):
        from test_plugin_config import _plugin_class
        cls = _plugin_class()
        instance = cls.__new__(cls)
        instance._runtime_lock = asyncio.Lock()
        instance.logger = Mock()
        instance.config = type("ConfigStub", (), {})()
        instance.config.profile_active = AsyncMock(return_value=None)
        instance.config.dump = AsyncMock(return_value={})
        instance.config.update = AsyncMock()
        invalid = asyncio.run(instance.yui_panel_save(
            changes={"log_path": "a", "log_directory": "b"},
            expected_profile=None,
        ))
        self.assertIn("设置校验失败：log_path 与 log_directory 只能配置一个", invalid.message)
        instance.config.update.side_effect = OSError("不得透传的本机路径")
        failed = asyncio.run(instance.yui_panel_save(
            changes={"log_path": "", "log_directory": ""},
            expected_profile=None,
        ))
        self.assertEqual(failed.message, "宿主配置服务写入失败，请刷新配置档后重试")
        self.assertNotIn("本机路径", failed.message)

    def test_save_uses_plugin_compatibility_only_for_missing_profile_write_channel(self):
        from test_plugin_config import _plugin_class
        cls = _plugin_class()
        module = __import__(cls.__module__, fromlist=["TransportError"])
        instance = cls.__new__(cls)
        instance._runtime_lock = asyncio.Lock()
        instance.logger = Mock()
        instance.config = type("ConfigStub", (), {})()
        instance.config.profile_active = AsyncMock(return_value="test")
        instance.config.dump = AsyncMock(return_value={})
        instance.config.profile_get = AsyncMock(return_value={"yui": {"claim_code": 7}})
        instance.config.profile_update = AsyncMock(side_effect=module.TransportError(
            "failed to upsert profile: Unknown request type: PLUGIN_CONFIG_PROFILE_UPSERT"
        ))
        with patch.object(module, "persist_profile_compat") as compat:
            result = asyncio.run(instance.yui_panel_save(
                changes={"chat_bridge.display_seconds": 15},
                expected_profile="test",
            ))
        self.assertEqual(result.value["status"], "saved")
        compat.assert_called_once()
        self.assertEqual(
            compat.call_args.args[1]["yui"],
            {"claim_code": 7, "chat_bridge": {"display_seconds": 15}},
        )

    def test_profile_compatibility_writer_is_scoped_and_atomic(self):
        root = Path(__file__).resolve().parent / "fixtures" / "profile_compat"
        profiles_path = root / "profiles.toml"
        target = root / "profiles" / "test.toml"
        try:
            profiles_path.write_text(
                '[config_profiles]\nactive = "test"\n\n[config_profiles.files]\ntest = "profiles/test.toml"\n',
                encoding="utf-8",
            )
            target.write_text("", encoding="utf-8")
            persist_profile_compat(
                "test",
                {"yui": {"log_path": "", "autonomy": {"enabled": True}}},
                [root],
            )
            with target.open("rb") as stream:
                saved = tomllib.load(stream)
            self.assertEqual(saved["yui"]["log_path"], "")
            self.assertTrue(saved["yui"]["autonomy"]["enabled"])
            self.assertEqual(list((root / "profiles").glob(".yui_profile_*.toml")), [])

            profiles_path.write_text(
                '[config_profiles.files]\ntest = "../outside.toml"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "超出插件配置目录"):
                persist_profile_compat("test", {"yui": {}}, [root])
        finally:
            profiles_path.unlink(missing_ok=True)
            target.unlink(missing_ok=True)

    def test_view_is_allowlisted(self):
        view = settings_view(YuiPluginConfig())
        self.assertEqual(view["chat_bridge.display_seconds"], 10)
        self.assertEqual(view["autonomy.intent_model.api_key"], "")
        self.assertIn("autonomy.intent_model.endpoint", view)

    def test_patch_preserves_unrelated_configuration(self):
        raw = {"yui": {"midi_port": "NEKO_MIDI", "chat_bridge": {"display_seconds": 10}}}
        patch = validated_patch(raw, {"chat_bridge.display_seconds": 15})
        self.assertEqual(patch, {"yui": {"chat_bridge": {"display_seconds": 15}}})
        self.assertEqual(raw["yui"]["chat_bridge"]["display_seconds"], 10)

    def test_reject_unknown_and_invalid_values(self):
        for changes in ({}, {"autonomy.enabled": 1}, {"chat_bridge.display_seconds": True},
                        {"chat_bridge.display_seconds": 26}, {"chat_bridge.max_pages": 0},
                        {"autonomy.intent_model.api_key_env": "INVALID-NAME"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validated_patch({}, changes)


    def test_model_key_is_write_only_and_can_be_cleared(self):
        from yui_npc_controller.runtime.intent import AutonomyIntentProvider
        secret = "test-only-private-key"
        raw = {"yui": {"autonomy": {"intent_model": {"api_key": secret}}}}
        config = YuiPluginConfig.from_mapping(raw["yui"])
        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, str(settings_view(config)))
        self.assertEqual(AutonomyIntentProvider(config.autonomy.intent_model)._api_key(), secret)
        self.assertEqual(validated_patch(raw, {"autonomy.intent_model.api_key": ""}), {"yui": {}})
        cleared = validated_patch(raw, {"autonomy.intent_model.clear_api_key": True})
        self.assertEqual(cleared, {"yui": {"autonomy": {"intent_model": {"api_key": ""}}}})
        self.assertEqual(raw["yui"]["autonomy"]["intent_model"]["api_key"], secret)
        with self.assertRaises(ValueError):
            validated_patch(raw, {"autonomy.intent_model.api_key": "new", "autonomy.intent_model.clear_api_key": True})
        with self.assertRaises(ValueError):
            validated_patch({}, {"autonomy.intent_model.api_key": "key\r\nheader"})

    def test_connection_fields_roundtrip_and_validate(self):
        changes = {"midi_port": "NEKO_MIDI", "claim_code": 42,
                   "log_path": "Editor.log", "log_directory": "",
                   "autonomy.intent_model.endpoint": "https://example.test/v1/chat/completions",
                   "autonomy.intent_model.model": "example-model",
                   "autonomy.intent_model.timeout_s": 12.5,
                   "ardy.endpoint": "http://127.0.0.1:2346"}
        patch = validated_patch({}, changes)
        config = YuiPluginConfig.from_mapping(patch["yui"])
        self.assertEqual(config.log_path, "Editor.log")
        self.assertEqual(config.autonomy.intent_model.timeout_s, 12.5)
        empty_logs = validated_patch({}, {"log_path": "", "log_directory": ""})
        empty_config = YuiPluginConfig.from_mapping(empty_logs["yui"])
        self.assertIsNone(empty_config.log_path)
        self.assertIsNone(empty_config.log_directory)
        for invalid in ({"log_path":"a", "log_directory":"b"},
                        {"ardy.endpoint":"http://example.test"},
                        {"autonomy.intent_model.timeout_s":float('nan')},
                        {"claim_code":16384}):
            with self.assertRaises(ValueError):
                validated_patch({}, invalid)
