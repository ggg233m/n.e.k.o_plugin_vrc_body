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


# 这些入口只供宿主菜单和人工安全通道使用。N.E.K.O 的后台 Agent 会在
# 主对话完成后再次评估普通插件入口；若不显式隐藏，会造成重复控制，甚至
# 把 CLEAR_ESTOP 暴露给模型。动态 npc.* 工具仍由主对话 LLM 独占。
_HOST_ONLY_ENTRY_METADATA: dict[str, object] = {
    "agent_hidden": True,
    "agent_auto": False,
    "agent_exposed": False,
    "llm_exposed": False,
}


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
        self._tool_state_key: tuple[Any, ...] | None = None
        self._last_context_signature = ""
        self._context_push_status: dict[str, Any] = {
            "state": "idle",
            "pushes": 0,
            "last_error": None,
        }
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
        self._tool_state_key = None

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
            self._push_context_snapshot()
            return result

        return handler

    def _current_tool_state_key(self) -> tuple[Any, ...]:
        """生成低成本工具可见性键，避免高频状态日志反复重建 schema。"""
        session = self._session
        surface = self._surface
        if session is None or surface is None:
            return (0,)
        control_group = (
            "armed"
            if session.control_state in {"external", "moving", "action"}
            else session.control_state
        )
        catalog_identity: tuple[Any, ...] = ()
        if session.discovery_ready:
            catalog_identity = tuple(
                (
                    kind,
                    tuple(
                        str(item.get("semantic_key") or item.get("name") or item_id)
                        for item_id, item in sorted(session.catalogs[kind].items())
                    ),
                )
                for kind in ("action", "expression", "anchor", "region", "entity")
            )
        return (
            session.session,
            session.discovery_ready,
            control_group,
            tuple(session.capabilities),
            session.catalog_revision,
            session.max_speed_mps,
            catalog_identity,
            surface.free_coordinate_navigation,
            surface.include_player_names,
            surface.enable_wander_tool,
        )

    def _refresh_llm_tools(self) -> list[str]:
        state_key = self._current_tool_state_key()
        if state_key == self._tool_state_key:
            return sorted(self._registered_yui_tools)
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
            self._tool_state_key = state_key
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
        self._tool_state_key = state_key
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

    def _model_context_payload(self) -> dict[str, Any] | None:
        """构建只含世界确认事实的 fast 模型上下文，不携带绝对坐标。"""
        session = self._session
        if session is None or session.session <= 0:
            return None
        if self._adapter is not None:
            observation = self._adapter.observe(
                include_player_names=self._config.include_player_names
            )
        else:
            observation = session.observe(
                include_player_names=self._config.include_player_names
            )
        if session.world_map_ready and "world" not in observation:
            observation["world"] = session.nearby_world(limit=8)

        plan: dict[str, Any] | None = None
        if self._adapter is not None:
            current = self._adapter.plan_status()
            if current.get("error") != "plan_not_found":
                plan = current

        payload: dict[str, Any] = {
            "context_type": "yui_world_context_v1",
            "instructions": [
                "这些是 Unity 和冻结目录确认的事实，不是视觉推断。",
                "操作前先使用 npc.observe 刷新状态，只能选择已发布的 semantic_key 或 player_slot。",
                "operation 的 failed、cancelled、unknown 都不得当作成功。",
            ],
            "world": observation,
            "catalog": {
                "revision": session.catalog_revision,
                "counts": {
                    kind: len(catalog)
                    for kind, catalog in session.catalogs.items()
                },
            },
            "available_tools": sorted(self._registered_yui_tools),
            "plan": plan,
        }
        payload = self._privacy_safe(payload)
        if session.spec_version in {"1.2", "1.3"}:
            payload = session._without_absolute_coordinates(payload)
        return payload

    @staticmethod
    def _context_signature(payload: dict[str, Any]) -> str:
        """只对语义状态签名，忽略高频距离和朝向变化。"""
        world = payload.get("world") if isinstance(payload.get("world"), dict) else {}
        location = world.get("location") if isinstance(world.get("location"), dict) else {}
        nearest = (
            location.get("nearest_anchor")
            if isinstance(location.get("nearest_anchor"), dict)
            else {}
        )
        players = world.get("players") if isinstance(world.get("players"), list) else []
        active_ops = (
            world.get("active_ops")
            if isinstance(world.get("active_ops"), list)
            else []
        )
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        stable = {
            "session": world.get("session"),
            "spec": world.get("spec"),
            "control_state": world.get("control_state"),
            "estop": world.get("estop"),
            "caps": world.get("caps"),
            "catalog_rev": world.get("catalog_rev"),
            "region_key": location.get("region_key"),
            "floor_label": location.get("floor_label"),
            "nearest_anchor": nearest.get("semantic_key"),
            "player_slots": sorted(
                item.get("slot")
                for item in players
                if isinstance(item, dict) and isinstance(item.get("slot"), int)
            ),
            "active_ops": sorted(
                (
                    str(item.get("operation_id") or item.get("op_id") or ""),
                    str(item.get("kind") or ""),
                    str(item.get("status") or ""),
                )
                for item in active_ops
                if isinstance(item, dict)
            ),
            "plan": (
                plan.get("plan_id"),
                plan.get("status"),
            ),
            "available_tools": payload.get("available_tools"),
        }
        return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _push_receipt_ok(receipt: Any) -> bool:
        if isinstance(receipt, dict):
            return receipt.get("ok", True) is not False
        ok = getattr(receipt, "ok", None)
        return ok is not False

    def _push_context_snapshot(self, *, force: bool = False) -> bool:
        """向宿主被动注入最新语义快照；相同状态只注入一次。"""
        if self._adapter is None or self._surface is None:
            return False
        payload = self._model_context_payload()
        if payload is None:
            return False
        signature = self._context_signature(payload)
        if not force and signature == self._last_context_signature:
            self._context_push_status["state"] = "deduplicated"
            return False
        text = "YUI_WORLD_CONTEXT " + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            receipt = self.push_message(
                source="yui_npc_controller",
                visibility=[],
                ai_behavior="read",
                parts=[{"type": "text", "text": text}],
                priority=20,
                coalesce_key="yui_world_context",
                metadata={
                    "context_type": "yui_world_context_v1",
                    "session": payload["world"].get("session"),
                },
            )
            if not self._push_receipt_ok(receipt):
                raise RuntimeError("宿主拒绝 YUI 上下文消息")
            self._last_context_signature = signature
            self._context_push_status.update({
                "state": "sent",
                "pushes": int(self._context_push_status.get("pushes", 0)) + 1,
                "last_error": None,
            })
            return True
        except Exception as exc:
            self._context_push_status.update({
                "state": "failed",
                "last_error": f"{type(exc).__name__}: {exc}",
            })
            logger = self.logger
            if logger is not None:
                try:
                    logger.warning("YUI 世界上下文注入失败: %s", exc)
                except Exception:
                    pass
            return False

    def _on_session_event(self, event: dict[str, Any]) -> None:
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._handle_session_event, dict(event))

    def _handle_session_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        refresh_event = event_type in {
            "sys.session",
            "sys.watchdog",
            "player.leave",
            "npc.state",
            "npc.ack",
        }
        if event_type in {"sys.hello", "sys.catalog"}:
            # DISCOVER 的目录按 20 行/s 分批到达。中间页不会形成可用工具面，
            # 每页都向宿主注销/注册工具会把一次连接放大成数十次 IPC 操作。
            refresh_event = bool(self._session and self._session.discovery_ready)
        if refresh_event:
            self._refresh_llm_tools()
            self._push_context_snapshot()
        high_salience = (
            event_type == "player.touch"
            or event_type.startswith("social.")
            or event_type in {"sys.err", "npc.operation_cancelled", "npc.operation_failed"}
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
        if self._adapter is not None:
            world = self._adapter.observe(
                include_player_names=self._config.include_player_names
            )
        elif self._session is not None:
            world = self._session.observe(
                include_player_names=self._config.include_player_names
            )
        else:
            world = None
        return {
            "midi_open": self._transport is not None,
            "driver_lease": bool(self._driver_lease and self._driver_lease.acquired),
            "log": self._tailer.snapshot() if self._tailer is not None else None,
            "control_ready": bool(
                self._session is not None
                and self._session.control_state in {"external", "moving", "action"}
            ),
            "llm_tools": sorted(self._registered_yui_tools),
            "context_injection": dict(self._context_push_status),
            "world": world,
        }

    def _close_control(self) -> None:
        self._unregister_yui_tools()
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
        self._last_context_signature = ""

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
            # 启动只跟随日志；不得打开 MIDI 或 DISCOVER。
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
        description="人工打开 MIDI，执行 DISCOVER，并由宿主把地图 NPC 切入可控态。",
        input_schema=_object_schema(),
        llm_result_fields=["status", "error", "detail"],
        timeout=15.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_connect(self, **_: Any):
        async with self._runtime_lock:
            try:
                adapter = self._ensure_control()
                result = await asyncio.to_thread(adapter.connect, self._config.claim_code)
                if result.get("status") == "succeeded":
                    result["llm_tools"] = self._refresh_llm_tools()
                    result["context_injected"] = self._push_context_snapshot(force=True)
                return Ok(result)
            except Exception as exc:
                return Err(f"{type(exc).__name__}: {exc}")

    @plugin_entry(
        id="yui_clear_estop",
        name="人工清除 YUI ESTOP",
        description="仅宿主安全入口可用；清除后由宿主恢复地图 NPC 控制态。",
        input_schema=_object_schema(),
        llm_result_fields=["status", "error", "detail"],
        timeout=10.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_clear_estop(self, **_: Any):
        async with self._runtime_lock:
            if self._adapter is None:
                return Err("尚未连接 YUI 世界")
            result = await asyncio.to_thread(self._adapter.clear_estop)
            result["llm_tools"] = self._refresh_llm_tools()
            return Ok(result)

    @plugin_entry(
        id="yui_disconnect",
        name="断开 YUI 本地控制器",
        description="关闭心跳、MIDI 与本地驱动锁；Unity 将按 watchdog 回到 safe_idle。",
        input_schema=_object_schema(),
        llm_result_fields=["status"],
        timeout=10.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
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
        metadata=_HOST_ONLY_ENTRY_METADATA,
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
        description="停止当前控制器并重新读取配置；不会自动连接。",
        input_schema=_object_schema(),
        llm_result_fields=["status"],
        timeout=10.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
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
