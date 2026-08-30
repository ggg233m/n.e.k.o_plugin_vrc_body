"""N.E.K.O 的独立 YUI NPC 控制插件标准入口。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any


# 独立插件把第三方依赖同步到自身 vendor；不污染宿主解释器。
_VENDOR = Path(__file__).resolve().parent / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

try:
    from plugin.sdk.plugin import Err, NekoPluginBase, Ok, lifecycle, neko_plugin, plugin_entry
except ImportError:
    # 单元测试环境没有 N.E.K.O SDK，使用最小桩保持核心可导入。
    class NekoPluginBase:  # type: ignore[no-redef]
        def __init__(self, ctx: Any = None) -> None:
            self.ctx = ctx

    class Ok:  # type: ignore[no-redef]
        def __init__(self, value: Any = None) -> None:
            self.value = value

    class Err:  # type: ignore[no-redef]
        def __init__(self, message: str = "") -> None:
            self.message = message

    def _decorator(*args: Any, **kwargs: Any) -> Any:
        return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else (lambda func: func)

    lifecycle = neko_plugin = plugin_entry = _decorator  # type: ignore[assignment]

from .runtime import (
    MidoOutputSink,
    YuiDriverLease,
    YuiOutputLogTailer,
    YuiPluginConfig,
    YuiReliableTransport,
    YuiSemanticAdapter,
    YuiSessionState,
    YuiToolSurface,
)


def _object_schema(
    properties: dict[str, Any] | None = None,
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


@neko_plugin
class YuiNpcControllerPlugin(NekoPluginBase):
    """只负责 N.E.K.O 集成；协议事实和工具 schema 全部留在 runtime。"""

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        try:
            self.logger = self.enable_file_logging(log_level="INFO")
        except Exception:
            self.logger = getattr(ctx, "logger", None)
        self._config = YuiPluginConfig()
        self._session: YuiSessionState | None = None
        self._tailer: YuiOutputLogTailer | None = None
        self._transport: YuiReliableTransport | None = None
        self._adapter: YuiSemanticAdapter | None = None
        self._surface: YuiToolSurface | None = None
        self._driver_lease: YuiDriverLease | None = None
        self._registered_yui_tools: set[str] = set()
        self._tool_signature = ""
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._runtime_lock = asyncio.Lock()

    async def _load_config(self) -> YuiPluginConfig:
        raw = await self.config.dump(timeout=5.0)
        root = raw if isinstance(raw, dict) else {}
        section = root.get("yui")
        return YuiPluginConfig.from_mapping(section if isinstance(section, dict) else {})

    def _start_log_tailer(self) -> None:
        if self._session is None:
            self._session = YuiSessionState()
            self._session.add_event_listener(self._on_session_event)
        self._tailer = YuiOutputLogTailer(
            self._session,
            log_path=self._config.log_path,
            log_directory=self._config.log_directory,
            from_end=self._config.log_from_end,
            poll_interval_s=self._config.log_poll_interval_s,
        )
        self._tailer.start()

    def _ensure_control(self) -> YuiSemanticAdapter:
        if self._adapter is not None:
            return self._adapter
        if self._session is None:
            raise RuntimeError("YUI 插件尚未启动")
        lease = YuiDriverLease(self._config.midi_port)
        lease.acquire()
        sink: MidoOutputSink | None = None
        try:
            sink = MidoOutputSink(self._config.midi_port)
            transport = YuiReliableTransport(
                sink,
                self._session,
                ack_timeout_s=self._config.ack_timeout_s,
                command_deadline_s=self._config.command_deadline_s,
                heartbeat_interval_s=self._config.heartbeat_interval_s,
            )
            adapter = YuiSemanticAdapter(
                transport,
                self._session,
                free_coordinate_navigation=self._config.free_coordinate_navigation,
            )
        except Exception:
            if sink is not None:
                sink.close()
            lease.release()
            raise
        self._driver_lease = lease
        self._transport = transport
        self._adapter = adapter
        self._surface = YuiToolSurface(
            adapter,
            self._session,
            host_arm_authorized=self._config.host_arm_authorized,
            free_coordinate_navigation=self._config.free_coordinate_navigation,
            include_player_names=self._config.include_player_names,
            enable_wander_tool=self._config.enable_wander_tool,
            command_deadline_s=self._config.command_deadline_s,
        )
        return adapter

    def _unregister_yui_tools(self) -> None:
        for name in sorted(self._registered_yui_tools):
            try:
                self.unregister_llm_tool(name)
            except Exception:
                pass
        self._registered_yui_tools.clear()
        self._tool_signature = ""

    def _make_tool_handler(self, tool_name: str):
        async def handler(**arguments: Any) -> dict[str, Any]:
            surface = self._surface
            if surface is None:
                return {
                    "status": "failed",
                    "error": "not_connected",
                    "detail": "尚未由宿主连接 YUI 世界",
                    "midi_sent": False,
                }
            result = await asyncio.to_thread(surface.call, tool_name, arguments)
            self._refresh_llm_tools()
            return result

        return handler

    def _refresh_llm_tools(self) -> list[str]:
        definitions = self._surface.definitions() if self._surface is not None else []
        signature = json.dumps(
            [
                {
                    "name": item.name,
                    "description": item.description,
                    "schema": item.input_schema,
                    "timeout": item.timeout_s,
                }
                for item in definitions
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature == self._tool_signature:
            return sorted(self._registered_yui_tools)
        self._unregister_yui_tools()
        for definition in definitions:
            self.register_llm_tool(
                name=definition.name,
                description=definition.description,
                parameters=definition.input_schema,
                handler=self._make_tool_handler(definition.name),
                timeout=definition.timeout_s,
            )
            self._registered_yui_tools.add(definition.name)
        self._tool_signature = signature
        return sorted(self._registered_yui_tools)

    @staticmethod
    def _privacy_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: YuiNpcControllerPlugin._privacy_safe(item)
                for key, item in value.items()
                if key not in {"name", "pid", "driver_pid"}
            }
        if isinstance(value, list):
            return [YuiNpcControllerPlugin._privacy_safe(item) for item in value]
        return value

    def _on_session_event(self, event: dict[str, Any]) -> None:
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._handle_session_event, dict(event))

    def _handle_session_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type in {
            "sys.session",
            "sys.hello",
            "sys.catalog",
            "sys.watchdog",
            "player.leave",
            "npc.state",
            "npc.ack",
        }:
            self._refresh_llm_tools()
        high_salience = (
            event_type == "player.touch"
            or event_type.startswith("social.")
            or event_type in {"sys.err", "npc.operation_cancelled"}
            or (event_type == "npc.ack" and event.get("ok") is False)
        )
        if not high_salience:
            return
        safe_event = self._privacy_safe(event)
        text = "YUI NPC 发生需要关注的事件：" + json.dumps(
            safe_event,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            self.push_message(
                source="yui_npc_controller",
                visibility=[],
                ai_behavior="respond",
                parts=[{"type": "text", "text": text}],
                priority=80,
                coalesce_key=f"yui:{event_type}",
                metadata={"event_type": event_type},
            )
        except Exception as exc:
            logger = self.logger
            if logger is not None:
                try:
                    logger.warning("YUI 主动事件推送失败: %s", exc)
                except Exception:
                    pass

    def _status_snapshot(self) -> dict[str, Any]:
        return {
            "midi_open": self._transport is not None,
            "driver_lease": bool(self._driver_lease and self._driver_lease.acquired),
            "log": self._tailer.snapshot() if self._tailer is not None else None,
            "host_arm_authorized": (
                self._session.host_arm_authorized if self._session is not None else False
            ),
            "llm_tools": sorted(self._registered_yui_tools),
            "world": (
                self._session.observe(include_player_names=self._config.include_player_names)
                if self._session is not None
                else None
            ),
        }

    def _close_control(self) -> None:
        self._unregister_yui_tools()
        if self._session is not None:
            self._session.set_host_arm_authorized(False)
        if self._adapter is not None:
            self._adapter.close()
        if self._transport is not None:
            self._transport.close()
        if self._driver_lease is not None:
            self._driver_lease.release()
        self._transport = None
        self._adapter = None
        self._surface = None
        self._driver_lease = None

    def _close_runtime(self) -> None:
        self._close_control()
        if self._tailer is not None:
            self._tailer.close()
        if self._session is not None:
            self._session.remove_event_listener(self._on_session_event)
        self._tailer = None
        self._session = None

    @lifecycle(id="startup")
    async def startup(self, **_: Any):
        try:
            self._event_loop = asyncio.get_running_loop()
            self._config = await self._load_config()
            # 启动只跟随日志；不得打开 MIDI、DISCOVER、授权或 ARM。
            self._start_log_tailer()
            return Ok({"status": "ready", "result": self._status_snapshot()})
        except Exception as exc:
            self._close_runtime()
            return Err(f"{type(exc).__name__}: {exc}")

    @lifecycle(id="shutdown")
    async def shutdown(self, **_: Any):
        self._close_runtime()
        self._event_loop = None
        return Ok({"status": "stopped"})

    @plugin_entry(
        id="yui_connect",
        name="连接 YUI 世界 NPC",
        description="人工打开 MIDI 并发送 DISCOVER；连接不会自动授权或 ARM。",
        input_schema=_object_schema(),
        llm_result_fields=["status", "error", "detail"],
        timeout=15.0,
    )
    async def yui_connect(self, **_: Any):
        async with self._runtime_lock:
            try:
                adapter = self._ensure_control()
                result = await asyncio.to_thread(adapter.connect, self._config.claim_code)
                if result.get("status") == "succeeded":
                    # 这是宿主配置绑定到当前 session，不会调用 npc.arm。
                    adapter.authorize_arm(self._config.host_arm_authorized)
                    if self._surface is not None:
                        self._surface.host_arm_authorized = self._config.host_arm_authorized
                    result["llm_tools"] = self._refresh_llm_tools()
                return Ok(result)
            except Exception as exc:
                return Err(f"{type(exc).__name__}: {exc}")

    @plugin_entry(
        id="yui_authorize_arm",
        name="授权当前 YUI 会话 ARM",
        description="宿主人工设置当前 session 的 ARM 授权；不会发送 SET_CONTROL_MODE。",
        input_schema=_object_schema(
            {"authorized": {"type": "boolean"}},
            required=["authorized"],
        ),
        llm_result_fields=["authorized", "session", "control_state"],
        timeout=10.0,
    )
    async def yui_authorize_arm(self, *, authorized: bool, **_: Any):
        async with self._runtime_lock:
            if self._adapter is None or self._session is None or self._session.session <= 0:
                return Err("尚未完成当前 YUI session 的 DISCOVER")
            if authorized and self._session.control_state == "estop":
                return Err("ESTOP 已锁存，必须先由人工安全入口清除")
            result = self._adapter.authorize_arm(authorized)
            if self._surface is not None:
                self._surface.host_arm_authorized = bool(authorized)
            result["llm_tools"] = self._refresh_llm_tools()
            return Ok(result)

    @plugin_entry(
        id="yui_clear_estop",
        name="人工清除 YUI ESTOP",
        description="仅宿主安全入口可用；清除后保持 safe_idle，授权仍为 false。",
        input_schema=_object_schema(),
        llm_result_fields=["status", "error", "detail"],
        timeout=10.0,
    )
    async def yui_clear_estop(self, **_: Any):
        async with self._runtime_lock:
            if self._adapter is None:
                return Err("尚未连接 YUI 世界")
            result = await asyncio.to_thread(self._adapter.clear_estop)
            if self._surface is not None:
                self._surface.host_arm_authorized = False
            result["llm_tools"] = self._refresh_llm_tools()
            return Ok(result)

    @plugin_entry(
        id="yui_disconnect",
        name="断开 YUI 本地控制器",
        description="关闭心跳、MIDI 与本地驱动锁；Unity 将按 watchdog 回到 safe_idle。",
        input_schema=_object_schema(),
        llm_result_fields=["status"],
        timeout=10.0,
    )
    async def yui_disconnect(self, **_: Any):
        async with self._runtime_lock:
            self._close_control()
            return Ok({"status": "disconnected", "result": self._status_snapshot()})

    @plugin_entry(
        id="yui_status",
        name="查询 YUI 插件状态",
        description="读取 MIDI、驱动锁、VRChat 日志、会话和动态 LLM 工具诊断。",
        input_schema=_object_schema(),
        llm_result_fields=["summary"],
        timeout=10.0,
    )
    async def yui_status(self, **_: Any):
        snapshot = self._status_snapshot()
        world = snapshot.get("world") or {}
        log = snapshot.get("log") or {}
        snapshot["summary"] = (
            f"session={world.get('session', 0)} "
            f"state={world.get('control_state', 'unknown')} "
            f"midi_open={snapshot['midi_open']} "
            f"log_running={log.get('running', False)}"
        )
        return Ok(snapshot)

    @plugin_entry(
        id="yui_reload_config",
        name="重载 YUI 插件配置",
        description="停止当前控制器并重新读取配置；不会自动连接、授权或 ARM。",
        input_schema=_object_schema(),
        llm_result_fields=["status"],
        timeout=10.0,
    )
    async def yui_reload_config(self, **_: Any):
        async with self._runtime_lock:
            try:
                self._close_runtime()
                self._config = await self._load_config()
                self._start_log_tailer()
                return Ok({"status": "reloaded", "result": self._status_snapshot()})
            except Exception as exc:
                self._close_runtime()
                return Err(f"{type(exc).__name__}: {exc}")


__all__ = ["YuiNpcControllerPlugin"]
