"""N.E.K.O 的独立 YUI NPC 控制插件标准入口。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import sys
import threading
import time
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
    AutonomyDirector,
    AutonomyIntentProvider,
    MainReplyDisplayBridge,
    RecentChatContextProvider,
    MidoOutputSink,
    YuiChatContextConfig,
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
        self._autonomy: AutonomyDirector | None = None
        self._intent_provider = AutonomyIntentProvider(
            self._config.autonomy.intent_model
        )
        self._chat_context_provider = RecentChatContextProvider(
            self._config.autonomy.intent_model.chat_context
        )
        self._reply_display: MainReplyDisplayBridge | None = None
        self._reply_display_send_lock = threading.RLock()
        self._player_chat_lock = threading.RLock()
        self._player_chat_seen: dict[tuple[int, int, int], float] = {}
        self._player_chat_last_by_slot: dict[int, float] = {}
        self._player_chat_status: dict[str, Any] = {
            "enabled": self._config.player_chat.enabled,
            "world_ui_ready": False,
            "state": "waiting_for_world",
            "accepted": 0,
            "rejected": 0,
            "last_error": None,
            "last_message_hash": None,
        }
        self._intent_request_thread: threading.Thread | None = None
        self._intent_request_stop = threading.Event()
        self._intent_request_condition = threading.Condition(threading.RLock())
        self._intent_request_active = False
        self._intent_pending_request: dict[str, Any] | None = None
        self._intent_last_started_at = 0.0
        self._auto_connect_task: asyncio.Task[Any] | None = None
        self._auto_connect_thread: threading.Thread | None = None
        self._auto_connect_stop = threading.Event()
        self._manual_disconnect = False
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

    def _configure_intent_provider(self) -> None:
        self._stop_intent_worker()
        self._intent_provider = AutonomyIntentProvider(
            self._config.autonomy.intent_model
        )
        self._chat_context_provider = RecentChatContextProvider(
            self._config.autonomy.intent_model.chat_context
        )
        self._intent_last_started_at = 0.0
        self._start_intent_worker()

    def _configure_reply_display(self) -> None:
        if self._reply_display is not None:
            self._reply_display.close()
        bridge_config = self._config.chat_bridge
        provider = RecentChatContextProvider(
            YuiChatContextConfig(
                enabled=bridge_config.enabled,
                source=bridge_config.source,
                max_turns=1,
                max_chars=32_000,
                poll_interval_s=bridge_config.poll_interval_s,
                max_file_bytes=bridge_config.max_file_bytes,
            )
        )
        self._reply_display = MainReplyDisplayBridge(
            provider,
            bridge_config,
            self._display_main_reply,
            conversation_fetcher=self._fetch_proactive_reply_records,
        )
        self._reply_display.start()

    def _fetch_proactive_reply_records(self) -> object:
        """读取宿主已有的主动回复记录，不修改或扩展 conversations 总线。"""

        bus = self.bus
        conversations = getattr(bus, "conversations", None) if bus is not None else None
        getter = getattr(conversations, "get", None)
        if not callable(getter):
            raise RuntimeError("conversation_bus_unavailable")
        result = getter(max_count=50, timeout=1.5)
        if inspect.isawaitable(result):
            # 此方法只允许由桥接工作线程调用；进入事件循环会导致无法安全同步等待。
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise RuntimeError("conversation_bus_async_result")
        return result

    def _display_main_reply(self, text: str, display_seconds: int) -> dict[str, Any]:
        """只显示已存在的主模型输出，不调用 push_message、TTS 或动作模型。"""
        with self._reply_display_send_lock:
            adapter = self._adapter
            if adapter is None:
                return {"status": "failed", "error": "not_connected"}
            return adapter.say(text, display_seconds=display_seconds)

    def _reset_player_chat_state(self, *, state: str) -> None:
        with self._player_chat_lock:
            self._player_chat_seen.clear()
            self._player_chat_last_by_slot.clear()
            self._player_chat_status.update({
                "enabled": self._config.player_chat.enabled,
                "world_ui_ready": False,
                "state": state,
                "last_error": None,
                "last_message_hash": None,
            })

    def _player_chat_status_snapshot(self) -> dict[str, Any]:
        with self._player_chat_lock:
            return {
                **self._player_chat_status,
                "max_chars": self._config.player_chat.max_chars,
                "cooldown_s": self._config.player_chat.cooldown_s,
                "activation": "local_follow_hotkey_T",
            }

    def _current_host_character(self) -> str | None:
        """只读磁盘中的当前猫娘，避免多角色会话把消息投递到错误对象。"""

        providers: list[RecentChatContextProvider] = []
        reply_display = self._reply_display
        if reply_display is not None:
            providers.append(reply_display.provider)
        if all(provider is not self._chat_context_provider for provider in providers):
            providers.append(self._chat_context_provider)

        for provider in providers:
            try:
                provider.poll(force=True)
                value = provider.status().get("current_character")
            except Exception:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _handle_player_chat_submit(self, event: dict[str, Any]) -> None:
        """只有世界内玩家显式提交能进入宿主 respond 通道。"""

        config = self._config.player_chat
        if not config.enabled:
            with self._player_chat_lock:
                self._player_chat_status.update({"state": "disabled", "last_error": None})
            return

        session = event.get("session")
        slot = event.get("slot")
        pid = event.get("pid")
        submit_sequence = event.get("submit_seq")
        text = event.get("text")
        valid_numbers = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (session, slot, pid, submit_sequence)
        )
        current_session = self._session.session if self._session is not None else 0
        player = (
            self._session.players.get(slot)
            if self._session is not None and isinstance(slot, int)
            else None
        )
        valid_player = bool(
            isinstance(player, dict)
            and isinstance(pid, int)
            and player.get("pid") == pid
        )
        normalized = (
            text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
            if isinstance(text, str)
            else ""
        )
        error: str | None = None
        if not valid_numbers:
            error = "invalid_identity"
        elif session <= 0 or session != current_session:
            error = "stale_session"
        elif not 0 <= slot <= 63 or pid <= 0 or submit_sequence <= 0:
            error = "invalid_identity"
        elif not valid_player:
            error = "slot_mismatch"
        elif not normalized:
            error = "empty_message"
        elif len(normalized) > config.max_chars or len(normalized.encode("utf-8")) > 576:
            error = "message_too_long"
        elif any(ord(character) < 32 for character in normalized):
            error = "invalid_character"

        now = time.monotonic()
        key = (
            int(session or 0),
            int(pid or 0),
            int(submit_sequence or 0),
        )
        with self._player_chat_lock:
            if error is None and key in self._player_chat_seen:
                error = "duplicate"
            last_at = self._player_chat_last_by_slot.get(int(slot or -1), float("-inf"))
            if error is None and now - last_at < config.cooldown_s:
                error = "rate_limited"
            if error is not None:
                self._player_chat_status["state"] = "rejected"
                self._player_chat_status["rejected"] = int(
                    self._player_chat_status.get("rejected", 0)
                ) + 1
                self._player_chat_status["last_error"] = error
                return

        target_character = self._current_host_character()
        if target_character is None:
            with self._player_chat_lock:
                self._player_chat_status["state"] = "rejected"
                self._player_chat_status["rejected"] = int(
                    self._player_chat_status.get("rejected", 0)
                ) + 1
                self._player_chat_status["last_error"] = "target_character_unavailable"
            return

        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        try:
            receipt = self.push_message(
                source="yui_npc_controller.world_chat",
                visibility=["chat"],
                ai_behavior="respond",
                parts=[{
                    "type": "text",
                    "text": f"VRChat 世界内玩家（player_slot={slot}）说：{normalized}",
                }],
                priority=90,
                target_lanlan=target_character,
                metadata={
                    "event_type": "player.chat_submit",
                    "session": session,
                    "player_slot": slot,
                    "submit_seq": submit_sequence,
                    "content_sha256": content_hash,
                },
            )
            submitted = self._push_receipt_ok(receipt)
        except Exception:
            submitted = False

        with self._player_chat_lock:
            if submitted:
                self._player_chat_seen[key] = now
                self._player_chat_last_by_slot[int(slot)] = now
                if len(self._player_chat_seen) > 256:
                    oldest = min(self._player_chat_seen, key=self._player_chat_seen.get)
                    self._player_chat_seen.pop(oldest, None)
                self._player_chat_status["state"] = "submitted"
                self._player_chat_status["accepted"] = int(
                    self._player_chat_status.get("accepted", 0)
                ) + 1
                self._player_chat_status["last_error"] = None
                self._player_chat_status["last_message_hash"] = content_hash
            else:
                self._player_chat_status["state"] = "submission_failed"
                self._player_chat_status["rejected"] = int(
                    self._player_chat_status.get("rejected", 0)
                ) + 1
                self._player_chat_status["last_error"] = "host_submission_failed"

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
        self._autonomy = AutonomyDirector(
            adapter,
            self._session,
            self._config.autonomy,
            inspiration_callback=(
                self._queue_autonomy_inspiration
                if self._config.autonomy.intent_model.enabled
                else None
            ),
            telemetry_callback=self._log_autonomy_event,
            chat_context_provider=self._chat_context_provider,
        )
        return adapter

    def _log_autonomy_event(self, event: dict[str, Any]) -> None:
        """写入不含密钥、正文、坐标和玩家姓名的结构化自主日志。"""
        allowed = {
            "event", "reason", "status", "error", "latency_ms", "format",
            "accepted", "mood", "activity_count", "avoid_count", "kind",
            "interest_count", "targets", "regions", "intent_activity_index",
            "decision_reason",
        }
        safe = {key: value for key, value in event.items() if key in allowed}
        logger = self.logger
        if logger is None:
            return
        try:
            logger.info(
                "YUI_AUTONOMY %s",
                json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
        except Exception:
            pass

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
            autonomy = self._autonomy
            if autonomy is not None:
                autonomy.before_explicit_tool(tool_name)
            try:
                result = await asyncio.to_thread(surface.call, tool_name, arguments)
            except Exception:
                if autonomy is not None:
                    autonomy.after_explicit_tool(
                        tool_name,
                        {"status": "failed", "error": "tool_exception"},
                    )
                raise
            if autonomy is not None:
                autonomy.after_explicit_tool(tool_name, result)
            self._refresh_llm_tools()
            self._push_context_snapshot()
            if result.get("status") == "failed":
                return {
                    "output": result,
                    "is_error": True,
                    "error": str(result.get("error") or "tool_failed"),
                }
            return result

        return handler

    def _queue_autonomy_inspiration(self, request: dict[str, Any]) -> None:
        """投递给插件自有线程；不依赖宿主生命周期事件循环。"""
        self._enqueue_intent_request(dict(request))

    def _enqueue_intent_request(self, request: dict[str, Any]) -> None:
        provider = self._intent_provider
        if not provider.config.enabled:
            return
        token = request.get("request_token")
        context = request.get("context")
        if not isinstance(token, str) or not isinstance(context, dict):
            return
        # 容量为 1：尚未发出的旧请求直接被最新上下文覆盖。
        with self._intent_request_condition:
            # 容量为 1：尚未发送的旧上下文由最新触发覆盖。
            self._intent_pending_request = {
                "request_token": token,
                "context": dict(context),
                "reason": str(request.get("reason") or "unknown"),
            }
            self._intent_request_condition.notify_all()

    def _start_intent_worker(self) -> None:
        if not self._intent_provider.config.enabled:
            return
        thread = self._intent_request_thread
        if thread is not None and thread.is_alive():
            return
        stop_event = threading.Event()
        self._intent_request_stop = stop_event
        thread = threading.Thread(
            target=self._intent_request_worker,
            args=(stop_event,),
            name="yui-autonomy-intent",
            daemon=True,
        )
        self._intent_request_thread = thread
        thread.start()

    def _intent_request_worker(self, stop_event: threading.Event) -> None:
        """常驻串行请求循环；宿主关闭 asyncio loop 后仍保持有效。"""
        while not stop_event.is_set():
            with self._intent_request_condition:
                while self._intent_pending_request is None and not stop_event.is_set():
                    self._intent_request_condition.wait(timeout=0.5)
                if stop_event.is_set():
                    return
                request = self._intent_pending_request
                self._intent_pending_request = None
            if request is None:
                continue

            # 限频等待期间继续吸收新触发，只保留最后一个尚未发送的请求。
            while not stop_event.is_set():
                wait_s = max(
                    0.0,
                    self._intent_provider.config.min_interval_s
                    - (time.monotonic() - self._intent_last_started_at),
                )
                if wait_s <= 0.0:
                    break
                stop_event.wait(min(wait_s, 0.25))
                with self._intent_request_condition:
                    if self._intent_pending_request is not None:
                        request = self._intent_pending_request
                        self._intent_pending_request = None
            if stop_event.is_set():
                return

            autonomy = self._autonomy
            if autonomy is None or not autonomy.status().get("running"):
                continue
            provider = self._intent_provider
            self._intent_last_started_at = time.monotonic()
            self._log_autonomy_event({
                "event": "intent_request_started",
                "reason": request.get("reason"),
            })
            with self._intent_request_condition:
                self._intent_request_active = True
            try:
                result = asyncio.run(provider.request(request["context"]))
            except Exception:
                result = {"status": "failed", "error": "request_worker_error"}
            finally:
                with self._intent_request_condition:
                    self._intent_request_active = False

            accepted = False
            if not stop_event.is_set() and result.get("status") == "succeeded":
                intent = result.get("intent")
                if isinstance(intent, dict):
                    accepted = autonomy.offer_intent(intent, request["request_token"])
            self._log_autonomy_event({
                "event": "intent_request_finished",
                "reason": request.get("reason"),
                "status": result.get("status"),
                "error": result.get("error"),
                "latency_ms": result.get("latency_ms"),
                "format": result.get("format"),
                "accepted": accepted,
            })

    def _cancel_intent_requests(self) -> None:
        with self._intent_request_condition:
            self._intent_pending_request = None
            self._intent_request_condition.notify_all()

    def _stop_intent_worker(self) -> None:
        self._cancel_intent_requests()
        stop_event = self._intent_request_stop
        thread = self._intent_request_thread
        stop_event.set()
        with self._intent_request_condition:
            self._intent_request_condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._intent_request_thread = None

    def _intent_request_status(self) -> dict[str, bool]:
        with self._intent_request_condition:
            thread = self._intent_request_thread
            return {
                "request_pending": self._intent_pending_request is not None,
                "worker_running": bool(thread is not None and thread.is_alive()),
                "request_active": self._intent_request_active,
            }

    def _cancel_auto_connect_task(self) -> None:
        stop_event = self._auto_connect_stop
        stop_event.set()
        thread = self._auto_connect_thread
        self._auto_connect_thread = None
        task = self._auto_connect_task
        self._auto_connect_task = None
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is not None and task is not current and not task.done():
            task.cancel()
        if thread is not None and thread is not threading.current_thread():
            # 未握手世界中的一次 DISCOVER 最长等待约一个 command deadline；
            # 先让常驻线程退出，再关闭 MIDI/驱动租约，避免迟到连接碰到已释放资源。
            thread.join(timeout=8.0)

    def _run_auto_connect_thread(self, stop_event: threading.Event) -> None:
        """在插件自有事件循环中运行自动连接，不依赖宿主生命周期 loop。"""
        try:
            if (
                not self._manual_disconnect
                and not stop_event.is_set()
                and self._session is not None
                and self._session.session == 0
                and self._adapter is not None
            ):
                # 世界已重启，但宿主进程仍持有旧 Adapter、心跳和自主计划。
                # 先完整释放旧控制链路，再以新 DISCOVER 建立全新会话。
                self._close_control(reason="world_restarted")
            asyncio.run(self._auto_connect_loop(stop_event=stop_event))
        except Exception as exc:
            logger = self.logger
            if logger is not None:
                try:
                    logger.info("YUI 自动连接线程结束: %s", type(exc).__name__)
                except Exception:
                    pass

    def _schedule_auto_connect(self) -> None:
        self._cancel_auto_connect_task()
        if not (self._config.autonomy.enabled and self._config.autonomy.auto_connect):
            return
        stop_event = threading.Event()
        self._auto_connect_stop = stop_event
        thread = threading.Thread(
            target=self._run_auto_connect_thread,
            args=(stop_event,),
            name="yui-autonomy-auto-connect",
            daemon=True,
        )
        self._auto_connect_thread = thread
        thread.start()

    def _ensure_auto_connect_worker(self) -> bool:
        """按需恢复已结束的自动连接线程；高频失败 ACK 不重复创建线程。"""

        if (
            self._manual_disconnect
            or not self._config.autonomy.enabled
            or not self._config.autonomy.auto_connect
        ):
            return False
        thread = self._auto_connect_thread
        if thread is not None and thread.is_alive():
            return False
        self._schedule_auto_connect()
        return True

    def _start_autonomy_after_connect(self) -> dict[str, Any] | None:
        """首次连接自动启动；STOP/ESTOP/人工暂停不会被重复连接解除。"""
        autonomy = self._autonomy
        if autonomy is None or not self._config.autonomy.enabled:
            return None
        status = autonomy.status()
        if status.get("running"):
            return status
        if status.get("pause_reason") in {None, "not_started"}:
            return autonomy.start()
        return status

    def _start_autonomy_for_ready_session(self) -> dict[str, Any] | None:
        """在世界真实进入可控态后补齐异步握手与工具注册之间的启动竞态。"""
        session = self._session
        if (
            self._manual_disconnect
            or not self._config.autonomy.auto_connect
            or session is None
            or not session.discovery_ready
            or session.control_state not in {"external", "moving", "action"}
            or session.npc_state.get("state") not in {"external", "moving", "action"}
        ):
            return None
        return self._start_autonomy_after_connect()

    async def _auto_connect_loop(
        self,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        """等待兼容世界出现并自动连接；人工断开后不再重试。"""
        while not self._manual_disconnect and not (
            stop_event is not None and stop_event.is_set()
        ):
            try:
                adapter = self._ensure_control()
                result = await asyncio.to_thread(adapter.connect, self._config.claim_code)
                if self._manual_disconnect or (
                    stop_event is not None and stop_event.is_set()
                ):
                    return
                if result.get("status") == "succeeded":
                    # Director 先启动并自行等待 _control_ready；握手阶段短暂的
                    # safe_idle 由 Director 的有界宽限处理，不能让 LLM IPC
                    # 成为规则自主的启动依赖。
                    self._start_autonomy_after_connect()
                    # LLM 工具注册和上下文 IPC 可能受宿主忙碌影响；规则自主
                    # 已先行启动，不能被这些可选集成步骤阻塞。
                    self._refresh_llm_tools()
                    self._push_context_snapshot(force=True)
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger = self.logger
                if logger is not None:
                    try:
                        logger.info("YUI 自主等待兼容世界: %s", exc)
                    except Exception:
                        pass
            if stop_event is None:
                await asyncio.sleep(2.0)
            else:
                await asyncio.to_thread(stop_event.wait, 2.0)

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
            "context_revision": 2,
            "connection": {
                "connected": True,
                "fresh": True,
                "session": session.session,
            },
            "instructions": [
                "这些是 Unity 和冻结目录确认的事实，不是视觉推断。",
                "用户询问当前位置、附近对象、玩家或当前状态时，本轮必须先调用 npc.observe；需要更大范围地图或路线时调用 npc.world_query。注入快照只能用于选择工具参数，不能冒充本轮实时观察。",
                "用户要求移动、导航、跟随、探索、注视、动作、表情、说话、停止或急停时，本轮必须在回复前调用匹配的 npc.* 工具；没有工具返回时，严禁回复‘开始’、‘马上’、‘正在执行’或‘已经完成’。",
                "只能选择当前 available_tools 和目录已发布的 semantic_key 或 player_slot；不可用时明确说明当前无法执行。",
                "工具返回 accepted 只表示已受理；plan_id 与 op_id 只供内部追踪及后续状态查询使用。除非用户明确询问编号，否则禁止在面向用户的可朗读回复中输出这些编号或 UUID。只有 operation/plan 的 succeeded 才能报告完成，failed、cancelled、unknown 都不得当作成功。",
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
    def _offline_context_payload(reason: str) -> dict[str, Any]:
        """构建离线失效通知，明确覆盖对话里残留的旧世界快照。"""

        return {
            "context_type": "yui_world_context_v1",
            "context_revision": 2,
            "connection": {
                "connected": False,
                "fresh": False,
                "reason": str(reason or "not_connected"),
            },
            "instructions": [
                "YUI 当前未连接；此前注入的全部位置、附近实体、玩家、目录和计划快照已经失效，不得继续引用。",
                "不得声称刚刚观察过世界，也不得根据旧快照回答‘我在哪’或‘附近有什么’。",
                "不得承诺移动、导航、跟随、探索、注视、动作、表情、说话、停止或急停将会执行。",
                "不得承诺连接后自动执行、稍后补执行或记住当前动作请求；连接成功后必须由用户重新发起。",
                "用户询问世界或要求控制 NPC 时，应明确回复：YUI 未连接，请先由宿主连接 YUI 世界 NPC。",
                "只有收到新的 connection.connected=true 上下文，并在本轮取得对应 npc.* 工具返回后，才能陈述实时事实或执行状态。",
            ],
            "world": {
                "available": False,
                "fresh": False,
            },
            "catalog": {
                "revision": None,
                "counts": {},
            },
            "available_tools": [],
            "plan": None,
        }

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
            "context_revision": payload.get("context_revision"),
            "connection": payload.get("connection"),
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
            if "submitted" in receipt:
                return receipt.get("submitted") is True
            return receipt.get("ok", True) is not False
        ok = getattr(receipt, "ok", None)
        return ok is not False

    def _push_context_payload(
        self,
        payload: dict[str, Any],
        *,
        force: bool = False,
        sent_state: str = "sent",
    ) -> bool:
        """把在线或离线上下文写入同一宿主通道，确保新状态覆盖旧语义。"""

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
                    "context_revision": payload.get("context_revision"),
                    "connected": bool(
                        isinstance(payload.get("connection"), dict)
                        and payload["connection"].get("connected") is True
                    ),
                    "session": payload.get("world", {}).get("session"),
                },
            )
            if not self._push_receipt_ok(receipt):
                raise RuntimeError("宿主拒绝 YUI 上下文消息")
            self._last_context_signature = signature
            self._context_push_status.update({
                "state": sent_state,
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

    def _push_context_snapshot(self, *, force: bool = False) -> bool:
        """向宿主被动注入最新在线语义快照；相同状态只注入一次。"""

        if self._adapter is None or self._surface is None:
            return False
        payload = self._model_context_payload()
        if payload is None:
            return False
        return self._push_context_payload(payload, force=force, sent_state="sent")

    def _push_context_unavailable(self, reason: str, *, force: bool = False) -> bool:
        """废止宿主对话中残留的在线快照，并禁止离线时假装观察或执行。"""

        return self._push_context_payload(
            self._offline_context_payload(reason),
            force=force,
            sent_state="offline_sent",
        )

    def _on_session_event(self, event: dict[str, Any]) -> None:
        event_copy = dict(event)
        event_type = str(event_copy.get("type") or "")
        world_restarted = (
            event_type == "sys.boot"
            and event_copy.get("session") == 0
        ) or (
            event_type == "npc.ack"
            and event_copy.get("session") == 0
            and event_copy.get("ok") is False
            and event_copy.get("err") == "not_handshaken"
        )
        if world_restarted:
            # 日志状态已先由 YuiSessionState 清成 session=0。这里不能依赖宿主
            # lifecycle loop；直接从常驻尾读线程唤起一次去重的后台重连。
            self._reset_player_chat_state(state="waiting_for_world")
            self._ensure_auto_connect_worker()
        if event_type == "player.chat_submit":
            # startup 生命周期回调返回后，宿主提供的 asyncio loop 可能已经不再
            # 运行。玩家提交是低频显式事件，直接在常驻日志线程校验并转发，
            # 避免消息只排进失活 loop 后永远无人处理。
            self._handle_player_chat_submit(event_copy)
            return
        if event_type == "sys.chat_input_ready":
            # 就绪状态同样必须独立于宿主生命周期 loop，供人工状态入口读取。
            with self._player_chat_lock:
                self._player_chat_status.update({
                    "world_ui_ready": bool(event_copy.get("ready")),
                    "state": "ready" if event_copy.get("ready") else "unavailable",
                    "last_error": None,
                })
            return
        if (
            event_type == "npc.state"
            and event_copy.get("state") in {"external", "moving", "action"}
        ):
            # 日志尾读线程已经完成 session 投影。规则自主在这里直接、线程安全
            # 地启动，不依赖可能被宿主 IPC 阻塞的 asyncio 事件循环。
            self._start_autonomy_for_ready_session()
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._handle_session_event, event_copy)

    def _handle_session_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "sys.session":
            self._reset_player_chat_state(state="waiting_for_world")
        elif event_type == "sys.chat_input_ready":
            with self._player_chat_lock:
                self._player_chat_status.update({
                    "world_ui_ready": bool(event.get("ready")),
                    "state": "ready" if event.get("ready") else "unavailable",
                    "last_error": None,
                })
        elif event_type == "player.chat_submit":
            self._handle_player_chat_submit(event)
            return
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
        if event_type in {"sys.hello", "npc.state", "npc.ack"}:
            # DISCOVER、SET_CONTROL_MODE、工具注册分别在不同线程完成。把首次
            # 自主启动最终绑定到日志确认的可控态，避免连接已成功却留在
            # not_started；已被 STOP、ESTOP 或人工暂停的状态不会被这里解除。
            self._start_autonomy_for_ready_session()
        # 社交事件由 Director 触发独立意图 API；这里不再启动宿主主动对话。
        failure_event = (
            event_type in {"sys.err", "npc.operation_failed"}
            or (event_type == "npc.ack" and event.get("ok") is False)
        )
        if not failure_event:
            return
        safe_event = self._privacy_safe(event)
        text = "YUI NPC 控制异常：" + json.dumps(
            safe_event,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            self.push_message(
                source="yui_npc_controller",
                visibility=["hud"],
                ai_behavior="blind",
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
            "auto_connect": {
                "enabled": bool(
                    self._config.autonomy.enabled
                    and self._config.autonomy.auto_connect
                ),
                "worker_running": bool(
                    self._auto_connect_thread is not None
                    and self._auto_connect_thread.is_alive()
                ),
                "manual_disconnect": self._manual_disconnect,
            },
            "autonomy": self._autonomy.status() if self._autonomy is not None else {
                "enabled": self._config.autonomy.enabled,
                "running": False,
                "state": "not_initialized",
            },
            "intent_model": {
                **self._intent_provider.status(),
                **self._intent_request_status(),
            },
            "chat_context": self._chat_context_provider.status(),
            "chat_bridge": (
                self._reply_display.status()
                if self._reply_display is not None
                else {
                    "enabled": self._config.chat_bridge.enabled,
                    "worker_running": False,
                    "last_result": "not_initialized",
                }
            ),
            "player_chat": self._player_chat_status_snapshot(),
            "world": world,
        }

    def _close_control(self, *, reason: str = "not_connected") -> None:
        self._cancel_intent_requests()
        self._unregister_yui_tools()
        if self._autonomy is not None:
            self._autonomy.close()
        with self._reply_display_send_lock:
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
        self._autonomy = None
        self._last_context_signature = ""
        self._push_context_unavailable(reason, force=True)

    def _close_runtime(self, *, reason: str = "plugin_stopped") -> None:
        self._reset_player_chat_state(state=reason)
        self._cancel_auto_connect_task()
        if self._reply_display is not None:
            self._reply_display.close()
            self._reply_display = None
        self._close_control(reason=reason)
        self._stop_intent_worker()
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
            self._configure_intent_provider()
            self._configure_reply_display()
            self._manual_disconnect = False
            # 通用配置仍只跟随日志；仅显式 autonomy.auto_connect 才打开 MIDI。
            self._start_log_tailer()
            self._push_context_unavailable("plugin_started_disconnected", force=True)
            self._schedule_auto_connect()
            return Ok({"status": "ready", "result": self._status_snapshot()})
        except Exception as exc:
            self._close_runtime(reason="startup_failed")
            return Err(f"{type(exc).__name__}: {exc}")

    @lifecycle(id="shutdown")
    async def shutdown(self, **_: Any):
        self._manual_disconnect = True
        self._close_runtime(reason="plugin_shutdown")
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
                self._manual_disconnect = False
                adapter = self._ensure_control()
                result = await asyncio.to_thread(adapter.connect, self._config.claim_code)
                if result.get("status") == "succeeded":
                    autonomy = self._start_autonomy_after_connect()
                    result["llm_tools"] = self._refresh_llm_tools()
                    result["context_injected"] = self._push_context_snapshot(force=True)
                    if autonomy is not None:
                        result["autonomy"] = autonomy
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
            self._manual_disconnect = True
            self._cancel_auto_connect_task()
            self._close_control(reason="host_disconnected")
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
        id="yui_autonomy_start",
        name="启动 YUI NPC 自主行为",
        description="人工启动或恢复宿主规则自主；ESTOP 未清除时不会启动。",
        input_schema=_object_schema(),
        llm_result_fields=["state", "pause_reason"],
        timeout=10.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_autonomy_start(self, **_: Any):
        if self._autonomy is None:
            return Err("尚未连接 YUI 世界")
        return Ok(self._autonomy.start())

    @plugin_entry(
        id="yui_autonomy_pause",
        name="暂停 YUI NPC 自主行为",
        description="人工持久暂停规则自主并停止当前自主计划。",
        input_schema=_object_schema(),
        llm_result_fields=["state", "pause_reason"],
        timeout=10.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_autonomy_pause(self, **_: Any):
        if self._autonomy is None:
            return Err("尚未连接 YUI 世界")
        return Ok(await asyncio.to_thread(self._autonomy.pause, "manual_pause"))

    @plugin_entry(
        id="yui_autonomy_status",
        name="查询 YUI NPC 自主状态",
        description="读取自主循环、移动占比、访问历史、失败退避和当前计划诊断。",
        input_schema=_object_schema(),
        llm_result_fields=["state", "pause_reason", "movement_ratio"],
        timeout=10.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_autonomy_status(self, **_: Any):
        intent_model = {
            **self._intent_provider.status(),
            **self._intent_request_status(),
        }
        chat_context = self._chat_context_provider.status()
        if self._autonomy is None:
            return Ok({
                "enabled": self._config.autonomy.enabled,
                "running": False,
                "state": "not_initialized",
                "intent_model": intent_model,
                "chat_context": chat_context,
            })
        result = self._autonomy.status()
        result["intent_model"] = intent_model
        result["chat_context"] = chat_context
        return Ok(result)

    @plugin_entry(
        id="yui_chat_bridge_status",
        name="查看 YUI 主对话头顶显示状态",
        description="查看 recent.json 到 NPC 头顶文本的只读桥接状态，不显示聊天正文。",
        input_schema=_object_schema(),
        llm_result_fields=["enabled", "worker_running", "last_result", "last_error"],
        timeout=5.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_chat_bridge_status(self, **_: Any):
        if self._reply_display is None:
            return Ok({
                "enabled": self._config.chat_bridge.enabled,
                "worker_running": False,
                "last_result": "not_initialized",
            })
        return Ok(self._reply_display.status())

    @plugin_entry(
        id="yui_player_chat_status",
        name="查看 YUI 世界内聊天输入状态",
        description="查看跟随式输入 UI、限频和宿主提交状态，不显示玩家正文。",
        input_schema=_object_schema(),
        llm_result_fields=["enabled", "world_ui_ready", "state", "last_error"],
        timeout=5.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_player_chat_status(self, **_: Any):
        return Ok(self._player_chat_status_snapshot())

    @plugin_entry(
        id="yui_autonomy_intent_probe",
        name="探测 YUI NPC 独立意图模型",
        description="验证 TEST_API、接口认证、模型和结构化响应；结果不会应用为 NPC 行为。",
        input_schema=_object_schema(),
        llm_result_fields=["status", "error", "latency_ms", "schema_valid"],
        timeout=30.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_autonomy_intent_probe(self, **_: Any):
        request_state = self._intent_request_status()
        if request_state["request_pending"] or request_state["request_active"]:
            return Ok({
                "status": "failed",
                "error": "intent_request_busy",
                "schema_valid": False,
            })
        await asyncio.to_thread(self._chat_context_provider.poll, force=True)
        result = await self._intent_provider.probe()
        result["chat_context"] = self._chat_context_provider.status()
        return Ok(result)

    @plugin_entry(
        id="yui_reload_config",
        name="重载 YUI 插件配置",
        description="停止当前控制器并重新读取配置；仅在 autonomy.auto_connect 开启时自动连接。",
        input_schema=_object_schema(),
        llm_result_fields=["status"],
        timeout=10.0,
        metadata=_HOST_ONLY_ENTRY_METADATA,
    )
    async def yui_reload_config(self, **_: Any):
        async with self._runtime_lock:
            try:
                self._close_runtime(reason="config_reload")
                self._config = await self._load_config()
                self._configure_intent_provider()
                self._configure_reply_display()
                self._manual_disconnect = False
                self._start_log_tailer()
                self._schedule_auto_connect()
                return Ok({"status": "reloaded", "result": self._status_snapshot()})
            except Exception as exc:
                self._close_runtime(reason="config_reload_failed")
                return Err(f"{type(exc).__name__}: {exc}")


__all__ = ["YuiNpcControllerPlugin"]
