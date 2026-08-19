"""N.E.K.O 插件入口：提供安全的 AnyaDance 身体控制。"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
import math
import threading
import time
from typing import Any, Iterable
import uuid

from plugin.sdk.plugin import NekoPluginBase, Ok, lifecycle, llm_tool, neko_plugin, plugin_entry, ui

from .backend.client import BackendClient, BackendUnavailable
from .behavior import EXPRESSION_INTENTS
from .config import PluginConfig
from .instructions import BODY_AI_INSTRUCTIONS
from .motion import GESTURE_NAMES
from .osc import normalize_parameter_value, validate_parameter_name
from .tool_defs import (
    BODY_ARM_POSE,
    BODY_AVATAR_PARAMETER,
    BODY_AWARENESS,
    BODY_CANCEL,
    BODY_CHATBOX,
    BODY_DISABLE,
    BODY_ENABLE,
    BODY_EXPRESS,
    BODY_GESTURE,
    BODY_HAND,
    BODY_LIST_CLIPS,
    BODY_LOCOMOTION,
    BODY_MOVE_HAND,
    BODY_PLAY_CLIP,
    BODY_REACH_AND_GRAB,
    BODY_RESET,
    BODY_SEQUENCE,
    BODY_STATUS,
    BODY_STOP,
    BODY_STOP_MOVEMENT,
    BODY_TURN,
    BODY_VRCHAT_INPUT,
    WORLD_OBSERVE,
)


def _enum(name: str, value: Any, allowed: Iterable[str]) -> str:
    normalized = str(value or "").strip().lower()
    choices = tuple(allowed)
    if normalized not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")
    return normalized


def _number(name: str, value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _integer(name: str, value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    parsed = int(numeric)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


_DEBUG_COMMAND_NAMES = (
    "body_enable",
    "body_disable",
    "body_stop",
    "body_reset",
    "body_cancel",
    "body_arm_pose",
    "body_move_hand",
    "body_hand",
    "body_reach_and_grab",
    "body_gesture",
    "body_express",
    "body_play_clip",
    "body_avatar_parameter",
    "body_vrchat_input",
)

@neko_plugin
class NekoAnyadanceBodyPlugin(NekoPluginBase):
    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        self._body_config = PluginConfig()
        self._raw_config: Mapping[str, Any] = {}
        self._backend_client: BackendClient | None = None
        self._scheduler: Any | None = None
        self._osc: Any | None = None
        self._driver_log: Any | None = None
        self._vmc_idle: Any | None = None
        self._host_vmc: Any | None = None
        # 视觉状态独立于 60 Hz 身体调度器；后端可以发布观测而不改变 VMC 待机路径。
        self._vision: Any | None = None
        self._ui_event_lock = threading.Lock()
        self._ui_events: deque[dict[str, Any]] = deque(maxlen=40)

    async def _load_config(self) -> PluginConfig:
        self._raw_config = {}
        try:
            raw = await self.config.dump()
            parsed = PluginConfig.from_mapping(raw)
            # 只有配置完整校验通过后，才把原始映射交给后端进程。
            self._raw_config = dict(raw) if isinstance(raw, Mapping) else {}
            return parsed
        except Exception as exc:
            self.logger.warning("Invalid AnyaDance body config; using safe defaults: %s", exc)
            return PluginConfig()

    @lifecycle(id="startup")
    async def on_startup(self, **_: Any):
        self._body_config = await self._load_config()
        # 重载插件会启动新的感知周期，同时重新建立后端运行时。
        if self._backend_client:
            await asyncio.to_thread(self._backend_client.stop)
        self._backend_client = BackendClient(
            self._raw_config,
            self.config_dir,
            logger=self.logger,
        )
        try:
            await asyncio.to_thread(self._backend_client.start)
        except BackendUnavailable as exc:
            self.logger.error("独立 AnyaDance 后端不可用：%s", exc)
            self._backend_client = None
        if self._backend_client:
            # 这些是远程兼容代理，真实对象位于后端进程而不是插件进程。
            self._scheduler = self._backend_client.scheduler
            self._osc = self._backend_client.osc
            self._driver_log = self._backend_client.driver_log
            self._vmc_idle = self._backend_client.vmc_idle
            self._host_vmc = self._backend_client.host_vmc
            self._vision = self._backend_client.vision
        else:
            self._scheduler = None
            self._osc = None
            self._driver_log = None
            self._vmc_idle = None
            self._host_vmc = None
            self._vision = None
        self._inject_ai_instructions()
        self.logger.info(
            "AnyaDance body scheduler ready (output disabled, target=%s:%s, rate=%s Hz)",
            self._body_config.host,
            self._body_config.port,
            self._body_config.rate_hz,
        )
        if self._body_config.vrchat_osc.enabled:
            osc_status = await asyncio.to_thread(self._osc.snapshot, include_parameters=False) if self._osc else {
                "send_target": f"{self._body_config.vrchat_osc.send_host}:{self._body_config.vrchat_osc.send_port}",
                "listen_address": f"{self._body_config.vrchat_osc.listen_host}:{self._body_config.vrchat_osc.listen_port}",
                "receiver_listening": False,
                "last_error": "backend unavailable",
            }
            self.logger.info(
                "VRChat OSC bridge ready (send=%s, listen=%s, receiver_listening=%s)",
                osc_status.get("send_target", f"{self._body_config.vrchat_osc.send_host}:{self._body_config.vrchat_osc.send_port}"),
                osc_status.get("listen_address", f"{self._body_config.vrchat_osc.listen_host}:{self._body_config.vrchat_osc.listen_port}"),
                osc_status.get("receiver_listening", False),
            )
        if self._body_config.vmc_idle.enabled:
            vmc_status = await asyncio.to_thread(self._vmc_idle.snapshot) if self._vmc_idle else {
                "listen_address": f"{self._body_config.vmc_idle.listen_host}:{self._body_config.vmc_idle.listen_port}",
                "receiver_listening": False,
                "last_error": "backend unavailable",
            }
            self.logger.info(
                "N.E.K.O VMC idle relay ready (listen=%s, receiver_listening=%s)",
                vmc_status.get("listen_address", f"{self._body_config.vmc_idle.listen_host}:{self._body_config.vmc_idle.listen_port}"),
                vmc_status.get("receiver_listening", False),
            )
        if self._body_config.driver_log.enabled:
            driver_log_status = await asyncio.to_thread(self._driver_log.snapshot) if self._driver_log else {
                "listen_address": f"{self._body_config.driver_log.multicast_group}:{self._body_config.driver_log.listen_port}",
                "receiver_listening": False,
                "last_error": "backend unavailable",
            }
            self.logger.info(
                "AnyaDance driver log listener ready (group=%s, receiver_listening=%s)",
                driver_log_status.get("listen_address", f"{self._body_config.driver_log.multicast_group}:{self._body_config.driver_log.listen_port}"),
                driver_log_status.get("receiver_listening", False),
            )
        return Ok({"status": "ready", "output_enabled": False})

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_: Any):
        if self._backend_client:
            await asyncio.to_thread(self._backend_client.stop)
        self._backend_client = None
        self._scheduler = None
        self._osc = None
        self._driver_log = None
        self._host_vmc = None
        self._vmc_idle = None
        self._vision = None
        return Ok({"status": "stopped"})

    def _inject_ai_instructions(self) -> None:
        try:
            self.push_message(
                source="neko_anyadance_body",
                ai_behavior="read",
                parts=[{"type": "text", "text": BODY_AI_INSTRUCTIONS}],
                priority=0,
            )
        except Exception as exc:
            self.logger.warning("Could not inject AnyaDance body awareness instructions: %s", exc)


    def _submit(
        self,
        kind: str,
        params: dict[str, Any] | None = None,
        *,
        preconditions: Any = None,
    ) -> dict[str, Any]:
        if not self._scheduler:
            return {
                "accepted": False,
                "action_id": None,
                "state": "shutdown",
                "normalized_params": {},
                "reason": "plugin scheduler is not initialized",
                "safety_state": "fault",
            }
        if preconditions is None:
            return self._scheduler.submit(kind, params)
        return self._scheduler.submit(kind, params, preconditions=preconditions)

    async def _submit_async(
        self,
        kind: str,
        params: dict[str, Any] | None = None,
        *,
        preconditions: Any = None,
    ) -> dict[str, Any]:
        """在线程池中执行一次后端动作提交，避免阻塞插件事件循环。"""
        return await asyncio.to_thread(
            self._submit,
            kind,
            params,
            preconditions=preconditions,
        )

    async def _invalid(self, reason: str) -> dict[str, Any]:
        if self._scheduler:
            state = await asyncio.to_thread(self._scheduler.snapshot)
        else:
            state = {"state": "shutdown", "safety_state": "fault"}
        return {
            "accepted": False,
            "action_id": None,
            "state": state["state"],
            "normalized_params": {},
            "reason": reason,
            "safety_state": state["safety_state"],
        }

    def _duration(self, value: Any, default: int) -> int:
        maximum = self._body_config.safety.max_action_duration_ms
        return _integer("duration_ms", default if value is None else value, minimum=100, maximum=maximum)

    async def _osc_result(
        self,
        *,
        accepted: bool,
        normalized_params: dict[str, Any],
        reason: str | None,
    ) -> dict[str, Any]:
        snapshot = await asyncio.to_thread(self._scheduler.snapshot) if self._scheduler else {
            "state": "shutdown",
            "safety_state": "fault",
        }
        return {
            "accepted": accepted,
            "action_id": str(uuid.uuid4()),
            "state": "sent" if accepted else snapshot["state"],
            "normalized_params": normalized_params,
            "reason": reason,
            "safety_state": snapshot["safety_state"],
            "transport": "vrchat_osc_udp",
            "delivery_confirmed": False,
        }

    async def _release_osc_inputs(self) -> None:
        if self._osc:
            await asyncio.to_thread(self._osc.cancel_scheduled_inputs, release=True)

    def _driver_log_snapshot(self) -> dict[str, Any]:
        if not self._driver_log:
            return {
                "enabled": False,
                "connection": "unknown",
                "receiver_listening": False,
                "last_error": "driver log listener is not initialized",
            }
        return self._driver_log.snapshot()

    @staticmethod
    def _apply_driver_log_to_udp(snapshot: dict[str, Any], driver_log: Mapping[str, Any]) -> None:
        """用驱动上报事实替换无法由 UDP 验证的字段。

        姿态协议没有响应，因此遥测组关闭或没有上报时，两个字段保持不可验证
        的默认值；旧版驱动不发送上报时，这里也不会擅自改变状态。
        """
        udp = snapshot.get("udp")
        if not isinstance(udp, dict) or not driver_log.get("enabled"):
            return
        connection = driver_log.get("connection")
        if connection not in {"detected", "stale"}:
            return
        senders = [str(item) for item in driver_log.get("senders") or []]
        local_port = udp.get("local_port")

        def is_local_sender(origin: Any) -> bool:
            return (
                local_port is not None
                and isinstance(origin, str)
                and origin.endswith(f":{local_port}")
            )

        # 过期状态没有活动发送端：driver_log 会在超过过期窗口后主动清理发送端。
        # 因此还要检查最后一条命令的来源，才能把过期状态归属到本地发送端。
        last_command = driver_log.get("last_command")
        last_source = last_command.get("source") if isinstance(last_command, Mapping) else None
        owns_recent_report = any(is_local_sender(sender) for sender in senders)

        # 只有确认至少一个发送端来自本插件的 UDP 源端口时，才能认定驱动连接状态。
        # 否则遥测可能属于其他进程，连接状态应继续保持未知。
        if owns_recent_report or (connection == "stale" and is_local_sender(last_source)):
            udp["connected"] = connection

        if connection == "stale":
            return
        if local_port is None:
            snapshot["concurrent_sender_detection"] = "detected_unattributed"
            return

        # 遥测来源是发送端地址，而 udp["target"] 是以“主机:端口”保存的目标地址。
        # 本机 UDP 源端口通常唯一，因此按端口排除本插件发送端，避免把目标地址
        # 与来源地址混为一谈。
        local_port_text = str(local_port)
        others = [item for item in senders if not item.endswith(f":{local_port_text}")]
        udp["other_senders"] = others
        snapshot["concurrent_sender_detection"] = "concurrent" if others else "exclusive"

    @staticmethod
    def _driver_delivery_awareness(
        snapshot: Mapping[str, Any], driver_log: Mapping[str, Any]
    ) -> dict[str, Any]:
        """向模型说明驱动是否实际收到命令。"""
        connection = str(driver_log.get("connection", "unknown"))
        concurrent = str(snapshot.get("concurrent_sender_detection", "unsupported"))
        accepted = int(driver_log.get("accepted_commands", 0) or 0)
        rejected = int(driver_log.get("rejected_commands", 0) or 0)
        y_clamped = int(driver_log.get("y_clamped_commands", 0) or 0)
        if not driver_log.get("enabled"):
            summary = "驱动遥测已禁用；只能确认本地发送成功，无法确认 AnyaDance 驱动收到。"
        elif connection == "detected":
            summary = f"AnyaDance 驱动已确认收到命令（接受 {accepted} 条，拒绝 {rejected} 条）。"
            if y_clamped:
                summary += f"其中 {y_clamped} 条被驱动钳制了 Y 高度。"
            if concurrent == "concurrent":
                summary += "检测到其他程序也在向驱动发送姿态，可能造成抖动。"
        elif connection == "stale":
            summary = "驱动曾确认收到命令，但最近一段时间没有新的上报。"
        elif connection == "listening":
            summary = "已加入驱动遥测组但尚未收到上报；驱动可能未运行或未启用该通道。"
        else:
            summary = "驱动遥测不可用；只能确认本地发送成功。"
        return {
            "enabled": bool(driver_log.get("enabled")),
            "connection": connection,
            "accepted_commands": accepted,
            "rejected_commands": rejected,
            "y_clamped_commands": y_clamped,
            "concurrent_sender_detection": concurrent,
            "summary": summary,
        }

    def _ui_event(self, command: str, arguments: Mapping[str, Any], payload: Any) -> None:
        value = getattr(payload, "value", payload)
        result = value if isinstance(value, Mapping) else {}
        safe_arguments: dict[str, Any] = {}
        for key, item in arguments.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                safe_arguments[str(key)[:80]] = item
            else:
                safe_arguments[str(key)[:80]] = str(item)[:240]
        event = {
            "at_unix": time.time(),
            "command": command,
            "arguments": safe_arguments,
            "accepted": bool(result.get("accepted", True)),
            "state": result.get("state"),
            "action_id": result.get("action_id"),
            "reason": result.get("reason"),
        }
        with self._ui_event_lock:
            self._ui_events.append(event)

    @ui.context(id="debug_dashboard", title="AnyaDance 身体调试台")
    async def debug_dashboard_context(self):
        driver_log = await asyncio.to_thread(self._driver_log_snapshot)
        if self._scheduler:
            body = await asyncio.to_thread(self._scheduler.snapshot)
            self._apply_driver_log_to_udp(body, driver_log)
            awareness = dict(body.get("awareness") or {})
        else:
            body = {
                "state": "shutdown",
                "safety_state": "fault",
                "output_enabled": False,
                "current_action": None,
                "queue_length": 0,
                "udp": {"connected": "unknown", "sent_packets": 0, "send_failures": 0},
                "metrics": {"actual_hz": 0.0, "skipped_frames": 0},
            }
            awareness = {
                "summary": "身体调度器尚未初始化。",
                "motion": None,
                "pose": {},
            }
        osc = await asyncio.to_thread(self._osc.snapshot) if self._osc else {
            "enabled": False,
            "connection": "unknown",
            "receiver_listening": False,
            "last_error": "OSC bridge is not initialized",
        }
        # Hosted UI 每秒刷新一次上下文。目录扫描只统计文件并复用缓存元数据，
        # 不能在插件事件循环中解析大型 .nya 内容。
        backend_client = self._backend_client
        clip_library = (
            await asyncio.to_thread(lambda: backend_client.catalog())
            if backend_client
            else {
                "clips": [],
                "invalid_clips": [],
                "directory": None,
            }
        )
        with self._ui_event_lock:
            events = list(self._ui_events)
        return {
            "version": "0.13.9",
            "updated_at_unix": time.time(),
            "body": body,
            "awareness": awareness,
            "vrchat_osc": osc,
            "driver_log": driver_log,
            "host_vmc": await asyncio.to_thread(self._host_vmc.snapshot) if self._host_vmc else {
                "managed": False,
                "active": False,
                "last_error": "host VMC controller is not initialized",
            },
            "world": await asyncio.to_thread(self._vision.snapshot) if self._vision else {
                "available": False,
                "uncertainties": ["backend_unavailable"],
            },
            "clips": clip_library,
            "config": {
                "anyadance_target": f"{self._body_config.host}:{self._body_config.port}",
                "rate_hz": self._body_config.rate_hz,
                "prefer_vmd_expressions": self._body_config.behavior.prefer_vmd_expressions,
                "vmc_idle_listen_address": f"{self._body_config.vmc_idle.listen_host}:{self._body_config.vmc_idle.listen_port}",
                "osc_send_target": f"{self._body_config.vrchat_osc.send_host}:{self._body_config.vrchat_osc.send_port}",
                "osc_listen_address": f"{self._body_config.vrchat_osc.listen_host}:{self._body_config.vrchat_osc.listen_port}",
                "driver_log_address": f"{self._body_config.driver_log.multicast_group}:{self._body_config.driver_log.listen_port}",
            },
            "ui_events": events,
        }

    @ui.action(
        id="debug_command",
        label="执行调试命令",
        tone="primary",
        group="debug",
        order=10,
        refresh_context=False,
    )
    @plugin_entry(
        id="debug_command",
        name="执行身体调试命令",
        description="仅供插件 Hosted UI 调试台调用已有身体和 VRChat OSC 命令。",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": list(_DEBUG_COMMAND_NAMES)},
                "arguments": {"type": "object", "default": {}},
            },
            "required": ["command"],
        },
    )
    async def debug_command(self, command: Any = "", arguments: Any = None, **_: Any):
        normalized = str(command or "").strip()
        params = dict(arguments) if isinstance(arguments, Mapping) else {}
        if normalized not in _DEBUG_COMMAND_NAMES:
            state = await asyncio.to_thread(self._scheduler.snapshot) if self._scheduler else {
                "state": "shutdown",
                "safety_state": "fault",
            }
            result = Ok({
                "accepted": False,
                "action_id": None,
                "state": state["state"],
                "normalized_params": {},
                "reason": "unsupported debug command",
                "safety_state": state["safety_state"],
            })
            self._ui_event(normalized or "unknown", params, result)
            return result
        handler = getattr(self, normalized, None)
        if not callable(handler):
            result = Ok({"accepted": False, "reason": "debug command handler is unavailable"})
            self._ui_event(normalized, params, result)
            return result
        try:
            result = await handler(**params)
        except Exception as exc:
            self.logger.warning("Body debug UI command failed (%s): %s", normalized, exc)
            state = await asyncio.to_thread(self._scheduler.snapshot) if self._scheduler else {
                "state": "shutdown",
                "safety_state": "fault",
            }
            result = Ok({
                "accepted": False,
                "action_id": None,
                "state": state["state"],
                "normalized_params": {},
                "reason": str(exc)[:500],
                "safety_state": state["safety_state"],
            })
        self._ui_event(normalized, params, result)
        return result

    @llm_tool(**BODY_ENABLE)
    async def body_enable(self, **_: Any):
        return Ok(await self._submit_async("enable"))

    @llm_tool(**BODY_DISABLE)
    async def body_disable(self, **_: Any):
        result = await self._submit_async("disable", {"duration_ms": self._body_config.default_duration_ms})
        if result.get("accepted"):
            await self._release_osc_inputs()
        return Ok(result)

    @llm_tool(**BODY_ARM_POSE)
    async def body_arm_pose(
        self,
        *,
        side: Any = "",
        elevation_deg: Any = None,
        azimuth_deg: Any = None,
        plane: Any = None,
        reach: Any = 0.9,
        palm: Any = "neutral",
        wrist_pitch_deg: Any = 0.0,
        wrist_yaw_deg: Any = 0.0,
        wrist_roll_deg: Any = 0.0,
        duration_ms: Any = None,
        **_: Any,
    ):
        try:
            normalized_azimuth = None if azimuth_deg is None else _number(
                "azimuth_deg", azimuth_deg, minimum=-180.0, maximum=180.0
            )
            normalized_plane = None if normalized_azimuth is not None else _enum(
                "plane", "front" if plane is None else plane, ("front", "side")
            )
            params = {
                "side": _enum("side", side, ("left", "right", "both")),
                "elevation_deg": _number("elevation_deg", elevation_deg, minimum=0.0, maximum=180.0),
                "azimuth_deg": normalized_azimuth,
                "plane": normalized_plane,
                "reach": _number("reach", reach, minimum=0.3, maximum=1.0),
                "palm": _enum("palm", palm, ("neutral", "forward", "down", "inward")),
                "wrist_pitch_deg": _number("wrist_pitch_deg", wrist_pitch_deg, minimum=-90.0, maximum=90.0),
                "wrist_yaw_deg": _number("wrist_yaw_deg", wrist_yaw_deg, minimum=-180.0, maximum=180.0),
                "wrist_roll_deg": _number("wrist_roll_deg", wrist_roll_deg, minimum=-180.0, maximum=180.0),
                "duration_ms": self._duration(duration_ms, self._body_config.default_duration_ms),
            }
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        return Ok(await self._submit_async("arm_pose", params))

    @llm_tool(**BODY_MOVE_HAND)
    async def body_move_hand(
        self,
        *,
        side: Any = "",
        relative_to: Any = "chest",
        x_m: Any = None,
        y_m: Any = None,
        z_m: Any = None,
        palm: Any = "neutral",
        wrist_pitch_deg: Any = 0.0,
        wrist_yaw_deg: Any = 0.0,
        wrist_roll_deg: Any = 0.0,
        duration_ms: Any = None,
        **_: Any,
    ):
        try:
            params = {
                "side": _enum("side", side, ("left", "right")),
                "relative_to": _enum("relative_to", relative_to, ("hmd", "chest", "hip")),
                "x_m": _number("x_m", x_m, minimum=-1.0, maximum=1.0),
                "y_m": _number("y_m", y_m, minimum=-1.0, maximum=1.0),
                "z_m": _number("z_m", z_m, minimum=-1.0, maximum=1.0),
                "palm": _enum("palm", palm, ("neutral", "forward", "down", "inward")),
                "wrist_pitch_deg": _number("wrist_pitch_deg", wrist_pitch_deg, minimum=-90.0, maximum=90.0),
                "wrist_yaw_deg": _number("wrist_yaw_deg", wrist_yaw_deg, minimum=-180.0, maximum=180.0),
                "wrist_roll_deg": _number("wrist_roll_deg", wrist_roll_deg, minimum=-180.0, maximum=180.0),
                "duration_ms": self._duration(duration_ms, self._body_config.default_duration_ms),
            }
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        return Ok(await self._submit_async("move_hand", params))

    @llm_tool(**BODY_HAND)
    async def body_hand(
        self,
        *,
        side: Any = "",
        pose: Any = "",
        strength: Any = 1.0,
        duration_ms: Any = None,
        **_: Any,
    ):
        try:
            params = {
                "side": _enum("side", side, ("left", "right", "both")),
                "pose": _enum("pose", pose, ("open", "fist", "grip", "point")),
                "strength": _number("strength", strength, minimum=0.0, maximum=1.0),
                "duration_ms": self._duration(duration_ms, 300),
            }
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        return Ok(await self._submit_async("hand", params))

    @llm_tool(**BODY_REACH_AND_GRAB)
    async def body_reach_and_grab(
        self,
        *,
        side: Any = "",
        height: Any = "",
        direction: Any = "forward",
        distance_m: Any = 0.35,
        duration_ms: Any = None,
        preconditions: Any = None,
        **_: Any,
    ):
        try:
            params = {
                "side": _enum("side", side, ("left", "right")),
                "height": _enum("height", height, ("waist", "chest", "head")),
                "direction": _enum("direction", direction, ("forward", "inward", "outward")),
                "distance_m": _number("distance_m", distance_m, minimum=0.15, maximum=0.70),
                "duration_ms": self._duration(duration_ms, 700),
            }
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        result = await self._submit_async(
            "reach_and_grab",
            params,
            preconditions=preconditions,
        )
        result["grip_engaged"] = bool(result["accepted"])
        result["object_held"] = "unknown"
        result["vrchat_osc_grab_scheduled"] = bool(
            result["accepted"]
            and self._osc
            and self._osc.config.enabled
            and self._osc.thread_alive
        )
        return Ok(result)

    @llm_tool(**BODY_GESTURE)
    async def body_gesture(
        self,
        *,
        name: Any = "",
        side: Any = "right",
        intensity: Any = 0.8,
        **_: Any,
    ):
        try:
            params = {
                "name": _enum("name", name, GESTURE_NAMES),
                "side": _enum("side", side, ("left", "right", "both")),
                "intensity": _number("intensity", intensity, minimum=0.0, maximum=1.0),
            }
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        return Ok(await self._submit_async("gesture", params))

    @llm_tool(**BODY_EXPRESS)
    async def body_express(
        self,
        *,
        intent: Any = "",
        side: Any = "auto",
        intensity: Any = None,
        duration_ms: Any = None,
        **_: Any,
    ):
        try:
            normalized_intent = _enum("intent", intent, EXPRESSION_INTENTS)
            normalized_side = _enum("side", side, ("auto", "left", "right", "both"))
            normalized_intensity = None if intensity is None else _number(
                "intensity", intensity, minimum=0.0, maximum=1.0
            )
            normalized_duration = None if duration_ms is None else _integer(
                "duration_ms", duration_ms, minimum=500, maximum=5000
            )
        except (KeyError, ValueError) as exc:
            return Ok(await self._invalid(str(exc)))
        if not self._backend_client:
            return Ok(await self._invalid("backend is not initialized"))
        # VMD 选择、片段加载和程序化回退都由后端负责；插件只校验面向 LLM 的输入，
        # 然后通过本机回环 IPC 发送一个粗粒度语义命令。
        result = await asyncio.to_thread(
            self._backend_client.semantic_express,
            {
                "intent": normalized_intent,
                "side": normalized_side,
                "intensity": normalized_intensity,
                "duration_ms": normalized_duration,
            },
        )
        return Ok(result)

    def _normalize_sequence_step(self, raw: Any, index: int) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError(f"steps[{index}] must be an object")
        kind = _enum(f"steps[{index}].type", raw.get("type"), ("arm_pose", "hand", "move_hand", "gesture", "wait"))
        prefix = f"steps[{index}]"
        if kind == "wait":
            return {"type": kind, "duration_ms": self._duration(raw.get("duration_ms"), 500)}
        if kind == "gesture":
            return {
                "type": kind,
                "name": _enum(f"{prefix}.name", raw.get("name"), GESTURE_NAMES),
                "side": _enum(f"{prefix}.side", raw.get("side", "right"), ("left", "right", "both")),
                "intensity": _number(f"{prefix}.intensity", raw.get("intensity", 0.8), minimum=0.0, maximum=1.0),
                "duration_ms": self._duration(raw.get("duration_ms"), 1200),
            }
        if kind == "hand":
            return {
                "type": kind,
                "side": _enum(f"{prefix}.side", raw.get("side"), ("left", "right", "both")),
                "pose": _enum(f"{prefix}.pose", raw.get("pose"), ("open", "fist", "grip", "point")),
                "strength": _number(f"{prefix}.strength", raw.get("strength", 1.0), minimum=0.0, maximum=1.0),
                "duration_ms": self._duration(raw.get("duration_ms"), 300),
            }
        wrist = {
            "palm": _enum(f"{prefix}.palm", raw.get("palm", "neutral"), ("neutral", "forward", "down", "inward")),
            "wrist_pitch_deg": _number(f"{prefix}.wrist_pitch_deg", raw.get("wrist_pitch_deg", 0.0), minimum=-90.0, maximum=90.0),
            "wrist_yaw_deg": _number(f"{prefix}.wrist_yaw_deg", raw.get("wrist_yaw_deg", 0.0), minimum=-180.0, maximum=180.0),
            "wrist_roll_deg": _number(f"{prefix}.wrist_roll_deg", raw.get("wrist_roll_deg", 0.0), minimum=-180.0, maximum=180.0),
        }
        if kind == "move_hand":
            return {
                "type": kind,
                "side": _enum(f"{prefix}.side", raw.get("side"), ("left", "right")),
                "relative_to": _enum(f"{prefix}.relative_to", raw.get("relative_to", "chest"), ("hmd", "chest", "hip")),
                "x_m": _number(f"{prefix}.x_m", raw.get("x_m"), minimum=-1.0, maximum=1.0),
                "y_m": _number(f"{prefix}.y_m", raw.get("y_m"), minimum=-1.0, maximum=1.0),
                "z_m": _number(f"{prefix}.z_m", raw.get("z_m"), minimum=-1.0, maximum=1.0),
                **wrist,
                "duration_ms": self._duration(raw.get("duration_ms"), self._body_config.default_duration_ms),
            }
        return {
            "type": kind,
            "side": _enum(f"{prefix}.side", raw.get("side"), ("left", "right", "both")),
            "elevation_deg": _number(f"{prefix}.elevation_deg", raw.get("elevation_deg"), minimum=0.0, maximum=180.0),
            "azimuth_deg": _number(f"{prefix}.azimuth_deg", raw.get("azimuth_deg", 0.0), minimum=-180.0, maximum=180.0),
            "plane": None,
            "reach": _number(f"{prefix}.reach", raw.get("reach", 0.9), minimum=0.3, maximum=1.0),
            **wrist,
            "duration_ms": self._duration(raw.get("duration_ms"), self._body_config.default_duration_ms),
        }

    @llm_tool(**BODY_SEQUENCE)
    async def body_sequence(self, *, steps: Any = None, loop_count: Any = 1, **_: Any):
        try:
            if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
                raise ValueError("steps must contain between 1 and 16 action objects")
            loops = _integer("loop_count", loop_count, minimum=1, maximum=4)
            normalized_steps = [self._normalize_sequence_step(step, index) for index, step in enumerate(steps)]
            requested_total = sum(step["duration_ms"] for step in normalized_steps) * loops
            if requested_total > 30000:
                raise ValueError("sequence requested duration must not exceed 30000 ms")
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        return Ok(await self._submit_async("sequence", {"steps": normalized_steps, "loop_count": loops}))

    @llm_tool(**BODY_CANCEL)
    async def body_cancel(self, *, action_id: Any = "", **_: Any):
        normalized = str(action_id or "").strip()
        if len(normalized) > 128:
            return Ok(await self._invalid("action_id must be at most 128 characters"))
        result = await self._submit_async("cancel", {"action_id": normalized or None})
        if result.get("accepted"):
            await self._release_osc_inputs()
        return Ok(result)

    @llm_tool(**BODY_LIST_CLIPS)
    async def body_list_clips(self, **_: Any):
        if not self._backend_client:
            return Ok({"clips": [], "invalid_clips": [], "reason": "backend is not initialized"})
        return Ok(await asyncio.to_thread(self._backend_client.list_clips))

    @llm_tool(**BODY_PLAY_CLIP)
    async def body_play_clip(
        self,
        *,
        clip_name: Any = "",
        speed: Any = 1.0,
        loop_count: Any = 1,
        transition_ms: Any = None,
        anchor: Any = True,
        restore_after: Any = False,
        **_: Any,
    ):
        try:
            name = str(clip_name or "").strip()
            if not name or len(name) > 256:
                raise ValueError("clip_name must not be empty and must be at most 256 characters")
            normalized_speed = _number("speed", speed, minimum=0.25, maximum=3.0)
            loops = _integer("loop_count", loop_count, minimum=1, maximum=10)
            transition = self._body_config.behavior.default_crossfade_ms if transition_ms is None else _integer(
                "transition_ms", transition_ms, minimum=0, maximum=5000
            )
            anchored = _boolean("anchor", anchor)
            restore = _boolean("restore_after", restore_after)
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        result = await asyncio.to_thread(
            self._submit,
            "play_clip",
            {
                "clip_name": name,
                "speed": normalized_speed,
                "loop_count": loops,
                "transition_ms": transition,
                "anchor": anchored,
                "restore_after": restore,
            },
        )
        return Ok(result)

    @llm_tool(**BODY_AVATAR_PARAMETER)
    async def body_avatar_parameter(self, *, name: Any = "", value: Any = None, **_: Any):
        try:
            parameter = validate_parameter_name(name)
            normalized_value = normalize_parameter_value(value)
        except ValueError as exc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params={},
                reason=str(exc),
            ))
        normalized = {"name": parameter, "value": normalized_value}
        if not self._osc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params=normalized,
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = await asyncio.to_thread(
            self._osc.send_parameter,
            parameter,
            normalized_value,
        )
        return Ok(await self._osc_result(
            accepted=accepted,
            normalized_params=normalized,
            reason=reason,
        ))

    @llm_tool(**BODY_VRCHAT_INPUT)
    async def body_vrchat_input(
        self,
        *,
        action: Any = "",
        side: Any = "",
        hold_ms: Any = None,
        **_: Any,
    ):
        try:
            normalized = {
                "action": _enum("action", action, ("grab", "use", "drop")),
                "side": _enum("side", side, ("left", "right")),
                "hold_ms": _integer(
                    "hold_ms",
                    self._body_config.vrchat_osc.input_pulse_ms if hold_ms is None else hold_ms,
                    minimum=20,
                    maximum=1000,
                ),
            }
        except ValueError as exc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params={},
                reason=str(exc),
            ))
        if not self._osc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params=normalized,
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = await asyncio.to_thread(
            self._osc.pulse_input,
            normalized["action"],
            normalized["side"],
            normalized["hold_ms"],
        )
        result = await self._osc_result(
            accepted=accepted,
            normalized_params=normalized,
            reason=reason,
        )
        result["object_held"] = "unknown"
        return Ok(result)

    @llm_tool(**BODY_STOP)
    async def body_stop(self, **_: Any):
        result = await self._submit_async("stop")
        await self._release_osc_inputs()
        return Ok(result)

    @llm_tool(**BODY_RESET)
    async def body_reset(self, *, duration_ms: Any = None, **_: Any):
        try:
            duration = self._duration(duration_ms, self._body_config.default_duration_ms)
        except ValueError as exc:
            return Ok(await self._invalid(str(exc)))
        result = await self._submit_async("reset", {"duration_ms": duration})
        if result.get("accepted"):
            await self._release_osc_inputs()
        return Ok(result)

    @llm_tool(**BODY_STATUS)
    async def body_status(self, **_: Any):
        if not self._scheduler:
            return Ok({
                "state": "shutdown",
                "output_enabled": False,
                "reason": "scheduler is not initialized",
                "idle_relay": await asyncio.to_thread(self._vmc_idle.snapshot) if self._vmc_idle else {"enabled": False},
                "vrchat_osc": await asyncio.to_thread(self._osc.snapshot) if self._osc else {"enabled": False},
                "driver_log": await asyncio.to_thread(self._driver_log_snapshot),
            })
        snapshot = await asyncio.to_thread(self._scheduler.snapshot)
        snapshot["vrchat_osc"] = await asyncio.to_thread(self._osc.snapshot) if self._osc else {
            "enabled": False,
            "connection": "unknown",
            "last_error": "OSC bridge is not initialized",
        }
        driver_log = await asyncio.to_thread(self._driver_log_snapshot)
        snapshot["driver_log"] = driver_log
        self._apply_driver_log_to_udp(snapshot, driver_log)
        return Ok(snapshot)

    @llm_tool(**BODY_AWARENESS)
    async def body_awareness(self, **_: Any):
        if not self._scheduler:
            return Ok({
                "state": "shutdown",
                "output_enabled": False,
                "safety_state": "fault",
                "summary": "身体调度器尚未初始化。",
                "motion": None,
                "previous_action": None,
                "transition": None,
                "pose": {},
                "idle_relay": await asyncio.to_thread(self._vmc_idle.snapshot) if self._vmc_idle else {"enabled": False},
                "vrchat_osc": await asyncio.to_thread(self._osc.awareness) if self._osc else {"enabled": False},
            })
        snapshot = await asyncio.to_thread(self._scheduler.snapshot)
        driver_log = await asyncio.to_thread(self._driver_log_snapshot)
        self._apply_driver_log_to_udp(snapshot, driver_log)
        return Ok({
            "state": snapshot["state"],
            "output_enabled": snapshot["output_enabled"],
            "safety_state": snapshot["safety_state"],
            "queue_length": snapshot["queue_length"],
            **snapshot["awareness"],
            "driver_delivery": self._driver_delivery_awareness(snapshot, driver_log),
            "vrchat_osc": await asyncio.to_thread(self._osc.awareness) if self._osc else {
                "enabled": False,
                "connection": "unknown",
                "summary": "VRChat OSC 桥接器尚未初始化。",
                "parameters": {},
                "pose_feedback_available": False,
                "pickup_confirmation_available": False,
            },
        })

    @llm_tool(**WORLD_OBSERVE)
    async def world_observe(self, **_: Any):
        """返回最新的视觉世界快照，不阻塞插件事件循环。"""
        if self._vision is None:
            return Ok({
                "available": False,
                "uncertainties": ["backend_unavailable"],
            })
        return Ok(await asyncio.to_thread(self._vision.snapshot))

    @llm_tool(**BODY_LOCOMOTION)
    async def body_locomotion(
        self,
        *,
        vertical: Any = 0.0,
        horizontal: Any = 0.0,
        duration_ms: Any = 1000,
        **_: Any,
    ):
        try:
            normalized = {
                "vertical": _number("vertical", vertical, minimum=-1.0, maximum=1.0),
                "horizontal": _number("horizontal", horizontal, minimum=-1.0, maximum=1.0),
                "duration_ms": _integer("duration_ms", duration_ms, minimum=100, maximum=10000),
            }
        except ValueError as exc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params={},
                reason=str(exc),
            ))
        if not self._osc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params=normalized,
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = await asyncio.to_thread(
            self._osc.set_locomotion,
            normalized["vertical"],
            normalized["horizontal"],
            normalized["duration_ms"],
        )
        return Ok(await self._osc_result(
            accepted=accepted,
            normalized_params=normalized,
            reason=reason,
        ))

    @llm_tool(**BODY_TURN)
    async def body_turn(
        self,
        *,
        horizontal: Any = None,
        duration_ms: Any = 500,
        **_: Any,
    ):
        try:
            if horizontal is None:
                raise ValueError("horizontal is required")
            normalized = {
                "horizontal": _number("horizontal", horizontal, minimum=-1.0, maximum=1.0),
                "duration_ms": _integer("duration_ms", duration_ms, minimum=100, maximum=10000),
            }
        except ValueError as exc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params={},
                reason=str(exc),
            ))
        if not self._osc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params=normalized,
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = await asyncio.to_thread(
            self._osc.set_turn,
            normalized["horizontal"],
            normalized["duration_ms"],
        )
        return Ok(await self._osc_result(
            accepted=accepted,
            normalized_params=normalized,
            reason=reason,
        ))

    @llm_tool(**BODY_STOP_MOVEMENT)
    async def body_stop_movement(self, **_: Any):
        if not self._osc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params={},
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = await asyncio.to_thread(self._osc.stop_movement)
        return Ok(await self._osc_result(
            accepted=accepted,
            normalized_params={},
            reason=reason,
        ))

    @llm_tool(**BODY_CHATBOX)
    async def body_chatbox(
        self,
        *,
        text: Any = "",
        immediate: Any = True,
        **_: Any,
    ):
        try:
            message = str(text or "").strip()
            if not message or len(message) > 144:
                raise ValueError("text must be between 1 and 144 characters")
            normalized = {
                "text": message,
                "immediate": _boolean("immediate", immediate),
            }
        except ValueError as exc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params={},
                reason=str(exc),
            ))
        if not self._osc:
            return Ok(await self._osc_result(
                accepted=False,
                normalized_params=normalized,
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = await asyncio.to_thread(
            self._osc.send_chatbox,
            normalized["text"],
            normalized["immediate"],
        )
        return Ok(await self._osc_result(
            accepted=accepted,
            normalized_params=normalized,
            reason=reason,
        ))


__all__ = ["NekoAnyadanceBodyPlugin"]
