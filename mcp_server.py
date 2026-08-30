"""不依赖第三方 MCP SDK 的 YUI stdio MCP 服务器。

stdout 只输出一行一个 JSON-RPC 消息；人工连接通过进程启动参数完成，不注册为
模型工具。连接后由宿主内部把地图 NPC 切入可控态。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping

from .runtime import (
    MidoOutputSink,
    YuiDriverLease,
    YuiOutputLogTailer,
    YuiReliableTransport,
    YuiSemanticAdapter,
    YuiSessionState,
    YuiToolSurface,
)


class YuiMcpRuntime:
    """持有独立 MCP 进程的日志、MIDI、会话和工具面。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session = YuiSessionState()
        self.tailer = YuiOutputLogTailer(
            self.session,
            log_path=args.log_path,
            log_directory=args.log_directory,
            from_end=not args.log_from_start,
            poll_interval_s=args.log_poll_interval,
        )
        self.lease: YuiDriverLease | None = None
        self.transport: YuiReliableTransport | None = None
        self.adapter: YuiSemanticAdapter | None = None
        self.surface: YuiToolSurface | None = None

    def start(self) -> None:
        # 无 --connect 时只跟随日志，绝不打开 MIDI 或发送命令。
        self.tailer.start()
        if not self.args.connect:
            return
        lease = YuiDriverLease(self.args.midi)
        lease.acquire()
        sink: MidoOutputSink | None = None
        try:
            sink = MidoOutputSink(self.args.midi)
            transport = YuiReliableTransport(
                sink,
                self.session,
                ack_timeout_s=self.args.ack_timeout,
                command_deadline_s=self.args.command_deadline,
                heartbeat_interval_s=self.args.heartbeat_interval,
            )
            adapter = YuiSemanticAdapter(
                transport,
                self.session,
                free_coordinate_navigation=self.args.free_coordinate_navigation,
            )
            connected = adapter.connect(self.args.claim_code)
            if connected.get("status") != "succeeded":
                raise RuntimeError(
                    f"连接失败: {connected.get('error') or connected.get('detail') or connected}"
                )
            surface = YuiToolSurface(
                adapter,
                self.session,
                free_coordinate_navigation=self.args.free_coordinate_navigation,
                include_player_names=self.args.include_player_names,
                enable_wander_tool=self.args.enable_wander_tool,
                command_deadline_s=self.args.command_deadline,
            )
        except Exception:
            if sink is not None:
                sink.close()
            lease.release()
            raise
        self.lease = lease
        self.transport = transport
        self.adapter = adapter
        self.surface = surface

    def tools(self) -> list[dict[str, Any]]:
        if self.surface is None:
            return []
        return [definition.as_mcp_tool() for definition in self.surface.definitions()]

    def call(self, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        if self.surface is None:
            return {
                "status": "failed",
                "error": "not_connected",
                "detail": "MCP 必须由人工使用 --connect 启动后才暴露 YUI 工具",
                "midi_sent": False,
            }
        return self.surface.call(name, arguments)

    def close(self) -> None:
        if self.adapter is not None:
            self.adapter.close()
        if self.transport is not None:
            self.transport.close()
        if self.lease is not None:
            self.lease.release()
        self.tailer.close()


class StdioMcpServer:
    def __init__(self, runtime: YuiMcpRuntime) -> None:
        self.runtime = runtime
        self._tool_signature = self._current_tool_signature()

    def _current_tool_signature(self) -> str:
        return json.dumps(self.runtime.tools(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _write(message: Mapping[str, Any]) -> None:
        sys.stdout.write(json.dumps(dict(message), ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def _notify_tools_changed(self) -> None:
        current = self._current_tool_signature()
        if current == self._tool_signature:
            return
        self._tool_signature = current
        self._write({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})

    def _result(self, request_id: Any, result: Any) -> None:
        self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _error(self, request_id: Any, code: int, message: str, data: Any = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        self._write({"jsonrpc": "2.0", "id": request_id, "error": error})

    def handle(self, request: Mapping[str, Any]) -> None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params")
        if method == "notifications/initialized":
            return
        if method == "initialize":
            requested_version = (
                params.get("protocolVersion")
                if isinstance(params, Mapping)
                else "2024-11-05"
            )
            self._result(
                request_id,
                {
                    "protocolVersion": requested_version or "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "yui-npc-controller", "version": "0.4.0"},
                    "instructions": "连接与 CLEAR_ESTOP 不属于模型工具；地图 NPC 在宿主连接后自动进入可控态，npc.arm 不存在。",
                },
            )
            return
        if method == "ping":
            self._result(request_id, {})
            return
        if method == "tools/list":
            self._result(request_id, {"tools": self.runtime.tools()})
            return
        if method == "tools/call":
            if not isinstance(params, Mapping) or not isinstance(params.get("name"), str):
                self._error(request_id, -32602, "tools/call 缺少字符串 name")
                return
            arguments = params.get("arguments")
            if arguments is not None and not isinstance(arguments, Mapping):
                self._error(request_id, -32602, "tools/call.arguments 必须是对象")
                return
            try:
                result = self.runtime.call(str(params["name"]), arguments)
            except Exception as exc:
                self._result(
                    request_id,
                    {
                        "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                        "isError": True,
                    },
                )
                return
            self._result(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                        }
                    ],
                    "structuredContent": result,
                    "isError": result.get("status") == "failed",
                },
            )
            self._notify_tools_changed()
            return
        if request_id is not None:
            self._error(request_id, -32601, f"未知方法: {method}")

    def serve(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, Mapping):
                    raise ValueError("JSON-RPC 请求必须是对象")
                self.handle(request)
            except Exception as exc:
                self._error(None, -32700, f"JSON 解析失败: {exc}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YUI NPC v1.1/v1.2/v1.3 stdio MCP server")
    parser.add_argument("--connect", action="store_true", help="由操作者显式打开 MIDI 并执行 DISCOVER")
    parser.add_argument("--midi", default="NEKO_MIDI")
    parser.add_argument("--claim-code", type=int, default=0)
    parser.add_argument("--free-coordinate-navigation", action="store_true")
    parser.add_argument("--include-player-names", action="store_true")
    parser.add_argument("--enable-wander-tool", action="store_true")
    parser.add_argument("--log-path")
    parser.add_argument("--log-directory")
    parser.add_argument("--log-from-start", action="store_true")
    parser.add_argument("--log-poll-interval", type=float, default=0.05)
    parser.add_argument("--ack-timeout", type=float, default=2.0)
    parser.add_argument("--command-deadline", type=float, default=5.0)
    parser.add_argument("--heartbeat-interval", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.log_path and args.log_directory:
        raise SystemExit("--log-path 与 --log-directory 只能设置一个")
    if not 0 <= args.claim_code <= 16383:
        raise SystemExit("--claim-code 必须是 0..16383")
    runtime = YuiMcpRuntime(args)
    try:
        runtime.start()
        StdioMcpServer(runtime).serve()
    except KeyboardInterrupt:
        # 人工停止 stdio MCP 是正常生命周期；关闭传输并静默返回成功。
        return 0
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
