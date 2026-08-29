"""独立 MCP 只暴露规范工具，不暴露宿主管理入口。"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from yui_npc_controller.mcp_server import StdioMcpServer, main


class FakeRuntime:
    def tools(self):
        return [
            {
                "name": "npc.observe",
                "description": "observe",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        ]

    def call(self, name, arguments):
        if dict(arguments or {}).get("fail"):
            return {"status": "failed", "error": "invalid_arguments", "midi_sent": False}
        return {"name": name, "arguments": dict(arguments or {}), "status": "succeeded"}


class McpServerTests(unittest.TestCase):
    def _invoke(self, request):
        output = io.StringIO()
        with patch("sys.stdout", output):
            StdioMcpServer(FakeRuntime()).handle(request)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_tools_list_contains_no_host_admin_tools(self) -> None:
        messages = self._invoke({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {item["name"] for item in messages[0]["result"]["tools"]}
        self.assertEqual(names, {"npc.observe"})
        self.assertFalse(names & {"connect", "disconnect", "status", "clear_estop"})

    def test_tools_call_returns_structured_content(self) -> None:
        messages = self._invoke(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "npc.observe", "arguments": {}},
            }
        )
        self.assertEqual(messages[0]["result"]["structuredContent"]["status"], "succeeded")

    def test_failed_tool_result_sets_mcp_is_error(self) -> None:
        messages = self._invoke(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "npc.observe", "arguments": {"fail": True}},
            }
        )
        self.assertTrue(messages[0]["result"]["isError"])

    def test_ctrl_c_is_clean_shutdown(self) -> None:
        runtime = SimpleNamespace(start=Mock(), close=Mock(), tools=Mock(return_value=[]))
        with (
            patch("yui_npc_controller.mcp_server.YuiMcpRuntime", return_value=runtime),
            patch.object(StdioMcpServer, "serve", side_effect=KeyboardInterrupt),
        ):
            self.assertEqual(main([]), 0)
        runtime.start.assert_called_once_with()
        runtime.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
