"""配置重载后真实刷新线程的生命周期回归；不连接宿主或 MIDI。"""

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, Mock

from test_plugin_config import _plugin_class


class ReloadRefreshTests(unittest.TestCase):
    def make_plugin(self):
        plugin = _plugin_class()(None)
        plugin._load_config = AsyncMock(return_value=plugin._config)
        # 保留真实的关闭、后端重建和刷新线程，只替换外部集成。
        for name in (
            "_configure_intent_provider", "_configure_reply_display",
            "_start_log_tailer", "_schedule_auto_connect", "_status_snapshot",
        ):
            setattr(plugin, name, Mock(return_value={}))
        return plugin

    def test_repeated_reload_consumes_motion_wakeup(self):
        plugin = self.make_plugin()
        refreshed = threading.Event()
        plugin._host_refresh.callback = refreshed.set
        plugin._host_refresh.start()
        try:
            for _ in range(2):
                previous = plugin._host_refresh.thread
                result = asyncio.run(plugin.yui_reload_config())
                self.assertEqual(result.value["status"], "reloaded")
                self.assertFalse(previous.is_alive())
                self.assertIsNot(plugin._host_refresh.thread, previous)
                self.assertTrue(plugin._host_refresh.thread.is_alive())
                # 等待关闭时遗留的唤醒被消费，再验证新的运动事件。
                self.assertTrue(refreshed.wait(1.0))
                refreshed.clear()
                plugin._motion_changed()
                self.assertTrue(refreshed.wait(1.0))
        finally:
            plugin._close_runtime()
        self.assertFalse(plugin._host_refresh.thread.is_alive())

    def test_reload_failure_stops_restarted_worker(self):
        plugin = self.make_plugin()
        plugin._host_refresh.callback = Mock()
        plugin._configure_reply_display.side_effect = RuntimeError("测试重载失败")
        try:
            result = asyncio.run(plugin.yui_reload_config())
            self.assertIn("测试重载失败", result.message)
            self.assertIsNotNone(plugin._host_refresh.thread)
            self.assertFalse(plugin._host_refresh.thread.is_alive())
        finally:
            plugin._close_runtime()
