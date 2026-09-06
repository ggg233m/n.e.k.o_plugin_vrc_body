"""面板只能保存安全白名单，不能修改凭据或导航边界。"""
import unittest
import asyncio
from unittest.mock import AsyncMock, Mock
import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.control_panel import settings_view, validated_patch
from yui_npc_controller.runtime.config import YuiPluginConfig


class ControlPanelTests(unittest.TestCase):
    def test_dashboard_projects_model_panel_without_general_snapshot_contents(self):
        from test_plugin_config import _plugin_class
        cls = _plugin_class()
        instance = cls.__new__(cls)
        instance._config = YuiPluginConfig()
        instance._load_config = AsyncMock(return_value=instance._config)
        instance._status_snapshot = Mock(return_value={"control_ready": False, "midi_open": False, "world": {"private": "world payload"}})
        instance._manual_disconnect = False
        instance._adapter = None
        instance.config = Mock()
        instance.config.profile_active = AsyncMock(return_value="test")
        instance._intent_provider = Mock()
        instance._intent_provider.panel_status.return_value = {"requests": 3, "last_call": {"output": "model result"}}
        view = asyncio.run(instance.yui_dashboard())
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

    def test_view_is_allowlisted(self):
        view = settings_view(YuiPluginConfig())
        self.assertEqual(view["chat_bridge.display_seconds"], 10)
        self.assertFalse(any("api_key" in key or "endpoint" in key for key in view))

    def test_patch_preserves_unrelated_configuration(self):
        raw = {"yui": {"midi_port": "NEKO_MIDI", "chat_bridge": {"display_seconds": 10}}}
        patch = validated_patch(raw, {"chat_bridge.display_seconds": 15})
        self.assertEqual(patch, {"yui": {"chat_bridge": {"display_seconds": 15}}})
        self.assertEqual(raw["yui"]["chat_bridge"]["display_seconds"], 10)

    def test_reject_unknown_and_invalid_values(self):
        for changes in ({}, {"autonomy.enabled": 1}, {"chat_bridge.display_seconds": True},
                        {"chat_bridge.display_seconds": 26}, {"chat_bridge.max_pages": 0},
                        {"autonomy.intent_model.api_key_env": "OTHER"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validated_patch({}, changes)
