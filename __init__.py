"""N.E.K.O plugin entry point for safe AnyaDance body control."""

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

from .behavior import EXPRESSION_INTENTS, resolve_expression
from .config import PluginConfig
from .driver_log import DriverLogListener
from .host_vmc import HostVmcController
from .instructions import BODY_AI_INSTRUCTIONS
from .motion import GESTURE_NAMES
from .nya import ClipLibrary
from .osc import VrchatOscBridge, normalize_parameter_value, validate_parameter_name
from .scheduler import BodyCommand, BodyScheduler
from .vmc_idle import VmcIdleRelay
from .tool_defs import (
    BODY_ARM_POSE,
    BODY_AVATAR_PARAMETER,
    BODY_AWARENESS,
    BODY_CANCEL,
    BODY_DISABLE,
    BODY_ENABLE,
    BODY_EXPRESS,
    BODY_GESTURE,
    BODY_HAND,
    BODY_LIST_CLIPS,
    BODY_MOVE_HAND,
    BODY_PLAY_CLIP,
    BODY_REACH_AND_GRAB,
    BODY_RESET,
    BODY_SEQUENCE,
    BODY_STATUS,
    BODY_STOP,
    BODY_VRCHAT_INPUT,
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
        self._scheduler: BodyScheduler | None = None
        self._clip_library: ClipLibrary | None = None
        self._osc: VrchatOscBridge | None = None
        self._driver_log: DriverLogListener | None = None
        self._vmc_idle: VmcIdleRelay | None = None
        self._host_vmc: HostVmcController | None = None
        self._vmc_calibration_stop = threading.Event()
        self._vmc_calibration_thread: threading.Thread | None = None
        self._expression_side_count = 0
        self._motion_intent_counts: dict[str, int] = {}
        self._ui_event_lock = threading.Lock()
        self._ui_events: deque[dict[str, Any]] = deque(maxlen=40)

    async def _load_config(self) -> PluginConfig:
        try:
            raw = await self.config.dump()
            return PluginConfig.from_mapping(raw)
        except Exception as exc:
            self.logger.warning("Invalid AnyaDance body config; using safe defaults: %s", exc)
            return PluginConfig()

    def _stop_vmc_calibration(self) -> None:
        self._vmc_calibration_stop.set()
        thread = self._vmc_calibration_thread
        if thread and thread.is_alive():
            thread.join(timeout=min(4.0, self._body_config.vmc_idle.host_api_timeout_seconds + 0.5))
        self._vmc_calibration_thread = None

    def _start_vmc_calibration(self) -> None:
        host_vmc = self._host_vmc
        relay = self._vmc_idle
        if host_vmc is None or relay is None or not host_vmc.snapshot()["active"]:
            return
        stop_event = threading.Event()
        self._vmc_calibration_stop = stop_event

        def calibrate() -> None:
            calibrated = host_vmc.calibrate_rest_pose(
                lambda: relay.reset_calibration(reason="host_t_pose"),
                timeout_seconds=120.0,
                stop_event=stop_event,
            )
            if not calibrated and not stop_event.is_set():
                # Never leave the relay held forever: fall back to calibrating
                # against the next complete frame instead of the host T pose.
                relay.reset_calibration(reason="t_pose_unavailable")

        self._vmc_calibration_thread = threading.Thread(
            target=calibrate,
            name="neko-vmc-rest-calibration",
            daemon=True,
        )
        self._vmc_calibration_thread.start()

    @lifecycle(id="startup")
    async def on_startup(self, **_: Any):
        self._body_config = await self._load_config()
        self._clip_library = ClipLibrary(
            self.config_dir / self._body_config.clip_directory,
            self._body_config,
        )
        if self._scheduler:
            await asyncio.to_thread(self._scheduler.shutdown)
        self._stop_vmc_calibration()
        if self._host_vmc:
            await asyncio.to_thread(self._host_vmc.stop)
            self._host_vmc = None
        if self._vmc_idle:
            await asyncio.to_thread(self._vmc_idle.stop)
        self._vmc_idle = VmcIdleRelay(
            self._body_config.vmc_idle,
            self._body_config.profile,
            logger=self.logger,
        )
        if (
            self._body_config.vmc_idle.enabled
            and self._body_config.vmc_idle.manage_host_output
        ):
            self._vmc_idle.hold_calibration(reason="waiting_for_host_t_pose")
        self._vmc_idle.start()
        self._host_vmc = HostVmcController(
            self._body_config.vmc_idle,
            logger=self.logger,
        )
        await asyncio.to_thread(self._host_vmc.start)
        self._start_vmc_calibration()
        self._scheduler = BodyScheduler(
            self._body_config,
            logger=self.logger,
            idle_frame_source=self._vmc_idle,
            motion_started_callback=self._on_motion_started,
        )
        self._scheduler.start()
        if self._osc:
            await asyncio.to_thread(self._osc.stop)
        self._osc = VrchatOscBridge(self._body_config.vrchat_osc, logger=self.logger)
        self._osc.start()
        if self._driver_log:
            await asyncio.to_thread(self._driver_log.stop)
        self._driver_log = DriverLogListener(self._body_config.driver_log, logger=self.logger)
        self._driver_log.start()
        self._inject_ai_instructions()
        self.logger.info(
            "AnyaDance body scheduler ready (output disabled, target=%s:%s, rate=%s Hz)",
            self._body_config.host,
            self._body_config.port,
            self._body_config.rate_hz,
        )
        if self._body_config.vrchat_osc.enabled:
            osc_status = self._osc.snapshot(include_parameters=False)
            self.logger.info(
                "VRChat OSC bridge ready (send=%s, listen=%s, receiver_listening=%s)",
                osc_status["send_target"],
                osc_status["listen_address"],
                osc_status["receiver_listening"],
            )
        if self._body_config.vmc_idle.enabled:
            vmc_status = self._vmc_idle.snapshot()
            self.logger.info(
                "N.E.K.O VMC idle relay ready (listen=%s, receiver_listening=%s)",
                vmc_status["listen_address"],
                vmc_status["receiver_listening"],
            )
        if self._body_config.driver_log.enabled:
            driver_log_status = self._driver_log.snapshot()
            self.logger.info(
                "AnyaDance driver log listener ready (group=%s, receiver_listening=%s)",
                driver_log_status["listen_address"],
                driver_log_status["receiver_listening"],
            )
        return Ok({"status": "ready", "output_enabled": False})

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_: Any):
        self._stop_vmc_calibration()
        if self._driver_log:
            await asyncio.to_thread(self._driver_log.stop)
            self._driver_log = None
        if self._osc:
            await asyncio.to_thread(self._osc.stop)
            self._osc = None
        if self._scheduler:
            await asyncio.to_thread(self._scheduler.shutdown)
        if self._host_vmc:
            await asyncio.to_thread(self._host_vmc.stop)
            self._host_vmc = None
        if self._vmc_idle:
            await asyncio.to_thread(self._vmc_idle.stop)
            self._vmc_idle = None
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


    def _submit(self, kind: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._scheduler:
            return {
                "accepted": False,
                "action_id": None,
                "state": "shutdown",
                "normalized_params": {},
                "reason": "plugin scheduler is not initialized",
                "safety_state": "fault",
            }
        return self._scheduler.submit(kind, params)

    def _on_motion_started(self, command: BodyCommand, duration_s: float) -> None:
        """Schedule reach-and-grab input from the scheduler's safe duration."""
        if command.kind != "reach_and_grab" or self._osc is None:
            return
        action_id = command.action_id
        side = str(command.params["side"])

        def action_is_current() -> bool:
            if not self._scheduler:
                return False
            snapshot = self._scheduler.snapshot()
            current = snapshot.get("current_action") or {}
            return (
                snapshot.get("output_enabled") is True
                and current.get("id") == action_id
                and current.get("name") == "reach_and_grab"
            )

        self._osc.schedule_input_pulse(
            "grab",
            side,
            delay_s=max(0.0, duration_s * 0.85),
            guard=action_is_current,
        )

    def _invalid(self, reason: str) -> dict[str, Any]:
        state = self._scheduler.snapshot() if self._scheduler else {"state": "shutdown", "safety_state": "fault"}
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

    def _osc_result(
        self,
        *,
        accepted: bool,
        normalized_params: dict[str, Any],
        reason: str | None,
    ) -> dict[str, Any]:
        snapshot = self._scheduler.snapshot() if self._scheduler else {
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

    def _release_osc_inputs(self) -> None:
        if self._osc:
            self._osc.cancel_scheduled_inputs(release=True)

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
        """Replace the unverifiable UDP fields with driver-reported facts.

        The pose protocol has no response, so both fields stay at their
        unverifiable defaults whenever the telemetry group is off or silent --
        an older driver simply never reports and nothing here changes.
        """
        udp = snapshot.get("udp")
        if not isinstance(udp, dict) or not driver_log.get("enabled"):
            return
        connection = driver_log.get("connection")
        if connection not in {"detected", "stale"}:
            return
        udp["connected"] = connection

        senders = [str(item) for item in driver_log.get("senders") or []]
        local_port = udp.get("local_port")
        if local_port is None:
            snapshot["concurrent_sender_detection"] = "detected_unattributed"
            return
        others = [item for item in senders if not item.endswith(f":{local_port}")]
        udp["other_senders"] = others
        snapshot["concurrent_sender_detection"] = "concurrent" if others else "exclusive"

    @staticmethod
    def _driver_delivery_awareness(
        snapshot: Mapping[str, Any], driver_log: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Tell the model whether the driver actually received the commands."""
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
        driver_log = self._driver_log_snapshot()
        if self._scheduler:
            body = self._scheduler.snapshot()
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
        osc = self._osc.snapshot() if self._osc else {
            "enabled": False,
            "connection": "unknown",
            "receiver_listening": False,
            "last_error": "OSC bridge is not initialized",
        }
        # Hosted UI refreshes this context every second.  A catalog scan only
        # stats files and reuses cached metadata; it must never parse a large
        # .nya payload on the plugin event loop.
        clip_library = self._clip_library.catalog() if self._clip_library else {
            "clips": [],
            "invalid_clips": [],
            "directory": None,
        }
        with self._ui_event_lock:
            events = list(self._ui_events)
        return {
            "version": "0.13.9",
            "updated_at_unix": time.time(),
            "body": body,
            "awareness": awareness,
            "vrchat_osc": osc,
            "driver_log": driver_log,
            "host_vmc": self._host_vmc.snapshot() if self._host_vmc else {
                "managed": False,
                "active": False,
                "last_error": "host VMC controller is not initialized",
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
            result = Ok({
                "accepted": False,
                "action_id": None,
                "state": self._scheduler.snapshot()["state"] if self._scheduler else "shutdown",
                "normalized_params": {},
                "reason": "unsupported debug command",
                "safety_state": self._scheduler.snapshot()["safety_state"] if self._scheduler else "fault",
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
            result = Ok({
                "accepted": False,
                "action_id": None,
                "state": self._scheduler.snapshot()["state"] if self._scheduler else "shutdown",
                "normalized_params": {},
                "reason": str(exc)[:500],
                "safety_state": self._scheduler.snapshot()["safety_state"] if self._scheduler else "fault",
            })
        self._ui_event(normalized, params, result)
        return result

    @llm_tool(**BODY_ENABLE)
    async def body_enable(self, **_: Any):
        return Ok(self._submit("enable"))

    @llm_tool(**BODY_DISABLE)
    async def body_disable(self, **_: Any):
        result = self._submit("disable", {"duration_ms": self._body_config.default_duration_ms})
        if result.get("accepted"):
            self._release_osc_inputs()
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
            return Ok(self._invalid(str(exc)))
        return Ok(self._submit("arm_pose", params))

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
            return Ok(self._invalid(str(exc)))
        return Ok(self._submit("move_hand", params))

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
            return Ok(self._invalid(str(exc)))
        return Ok(self._submit("hand", params))

    @llm_tool(**BODY_REACH_AND_GRAB)
    async def body_reach_and_grab(
        self,
        *,
        side: Any = "",
        height: Any = "",
        direction: Any = "forward",
        distance_m: Any = 0.35,
        duration_ms: Any = None,
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
            return Ok(self._invalid(str(exc)))
        result = self._submit("reach_and_grab", params)
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
            return Ok(self._invalid(str(exc)))
        return Ok(self._submit("gesture", params))

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
            alternate_side = "left" if self._expression_side_count % 2 else "right"
            params = resolve_expression(
                normalized_intent,
                side=normalized_side,
                intensity=normalized_intensity,
                duration_ms=normalized_duration,
                alternate_side=alternate_side,
            )
        except (KeyError, ValueError) as exc:
            return Ok(self._invalid(str(exc)))

        if self._body_config.behavior.prefer_vmd_expressions and self._clip_library is not None:
            selection_index = self._motion_intent_counts.get(normalized_intent, 0)
            metadata = self._clip_library.select_for_intent(
                normalized_intent,
                side=normalized_side,
                intensity=normalized_intensity,
                sequence_index=selection_index,
            )
            if metadata is not None:
                try:
                    clip = await asyncio.to_thread(self._clip_library.load, metadata["name"])
                    speed = float(metadata["recommended_speed"])
                    if normalized_duration is not None and not clip.is_pose:
                        requested_seconds = normalized_duration / 1000.0
                        speed = min(3.0, max(0.25, clip.duration_s / requested_seconds))
                    result = self._submit("semantic_clip", {
                        "clip_name": clip.name,
                        "speed": speed,
                        "loop_count": int(metadata["loop_count"]),
                        "transition_ms": int(metadata["transition_ms"]),
                        "anchor": True,
                        "restore_after": bool(metadata["restore_after"]),
                        "semantic_intent": normalized_intent,
                        "intent_label": params["intent_label"],
                        "motion_source": str(metadata["source_kind"]),
                        "motion_label": str(metadata["label"]),
                        "source_name": str(metadata["source_name"]),
                        "requested_intensity": normalized_intensity,
                        "requested_duration_ms": normalized_duration,
                        "_clip": clip,
                    })
                except (KeyError, OSError, ValueError) as exc:
                    self.logger.warning(
                        "Semantic VMD clip %s could not be loaded; using procedural fallback: %s",
                        metadata.get("name"),
                        exc,
                    )
                else:
                    if result.get("accepted"):
                        self._motion_intent_counts[normalized_intent] = selection_index + 1
                        if normalized_side == "auto":
                            self._expression_side_count += 1
                    return Ok(result)

        result = self._submit("express", params)
        if result.get("accepted") and normalized_side == "auto":
            self._expression_side_count += 1
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
            return Ok(self._invalid(str(exc)))
        return Ok(self._submit("sequence", {"steps": normalized_steps, "loop_count": loops}))

    @llm_tool(**BODY_CANCEL)
    async def body_cancel(self, *, action_id: Any = "", **_: Any):
        normalized = str(action_id or "").strip()
        if len(normalized) > 128:
            return Ok(self._invalid("action_id must be at most 128 characters"))
        result = self._submit("cancel", {"action_id": normalized or None})
        if result.get("accepted"):
            self._release_osc_inputs()
        return Ok(result)

    @llm_tool(**BODY_LIST_CLIPS)
    async def body_list_clips(self, **_: Any):
        if not self._clip_library:
            return Ok({"clips": [], "invalid_clips": [], "reason": "clip library is not initialized"})
        # Explicit listing indexes every clip.  JSON decoding a large dance is
        # CPU-heavy, so keep it off the SDK/event-loop thread.
        return Ok(await asyncio.to_thread(self._clip_library.list))

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
        if not self._clip_library:
            return Ok(self._invalid("clip library is not initialized"))
        try:
            name = str(clip_name or "").strip()
            # First use parses and validates the clip; subsequent calls use the
            # signature-aware resident cache.  Parsing happens in a worker so
            # the debug panel and other plugin entries remain responsive.
            clip = await asyncio.to_thread(self._clip_library.load, name)
            normalized_speed = _number("speed", speed, minimum=0.25, maximum=3.0)
            loops = _integer("loop_count", loop_count, minimum=1, maximum=10)
            transition = self._body_config.behavior.default_crossfade_ms if transition_ms is None else _integer(
                "transition_ms", transition_ms, minimum=0, maximum=5000
            )
            anchored = _boolean("anchor", anchor)
            restore = _boolean("restore_after", restore_after)
            playback_seconds = 0.0 if clip.is_pose else (clip.duration_s / normalized_speed) * loops
            if playback_seconds > self._body_config.clip_max_duration_seconds:
                raise ValueError(
                    f"expanded clip playback must not exceed {self._body_config.clip_max_duration_seconds:g} seconds"
                )
        except (ValueError, OSError) as exc:
            return Ok(self._invalid(str(exc)))
        return Ok(self._submit("play_clip", {
            "clip_name": clip.name,
            "speed": normalized_speed,
            "loop_count": loops,
            "transition_ms": transition,
            "anchor": anchored,
            "restore_after": restore,
            "_clip": clip,
        }))

    @llm_tool(**BODY_AVATAR_PARAMETER)
    async def body_avatar_parameter(self, *, name: Any = "", value: Any = None, **_: Any):
        try:
            parameter = validate_parameter_name(name)
            normalized_value = normalize_parameter_value(value)
        except ValueError as exc:
            return Ok(self._osc_result(
                accepted=False,
                normalized_params={},
                reason=str(exc),
            ))
        normalized = {"name": parameter, "value": normalized_value}
        if not self._osc:
            return Ok(self._osc_result(
                accepted=False,
                normalized_params=normalized,
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = self._osc.send_parameter(parameter, normalized_value)
        return Ok(self._osc_result(
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
            return Ok(self._osc_result(
                accepted=False,
                normalized_params={},
                reason=str(exc),
            ))
        if not self._osc:
            return Ok(self._osc_result(
                accepted=False,
                normalized_params=normalized,
                reason="VRChat OSC bridge is not initialized",
            ))
        accepted, reason = self._osc.pulse_input(
            normalized["action"],
            normalized["side"],
            normalized["hold_ms"],
        )
        result = self._osc_result(
            accepted=accepted,
            normalized_params=normalized,
            reason=reason,
        )
        result["object_held"] = "unknown"
        return Ok(result)

    @llm_tool(**BODY_STOP)
    async def body_stop(self, **_: Any):
        result = self._submit("stop")
        self._release_osc_inputs()
        return Ok(result)

    @llm_tool(**BODY_RESET)
    async def body_reset(self, *, duration_ms: Any = None, **_: Any):
        try:
            duration = self._duration(duration_ms, self._body_config.default_duration_ms)
        except ValueError as exc:
            return Ok(self._invalid(str(exc)))
        result = self._submit("reset", {"duration_ms": duration})
        if result.get("accepted"):
            self._release_osc_inputs()
        return Ok(result)

    @llm_tool(**BODY_STATUS)
    async def body_status(self, **_: Any):
        if not self._scheduler:
            return Ok({
                "state": "shutdown",
                "output_enabled": False,
                "reason": "scheduler is not initialized",
                "idle_relay": self._vmc_idle.snapshot() if self._vmc_idle else {"enabled": False},
                "vrchat_osc": self._osc.snapshot() if self._osc else {"enabled": False},
                "driver_log": self._driver_log_snapshot(),
            })
        snapshot = self._scheduler.snapshot()
        snapshot["vrchat_osc"] = self._osc.snapshot() if self._osc else {
            "enabled": False,
            "connection": "unknown",
            "last_error": "OSC bridge is not initialized",
        }
        driver_log = self._driver_log_snapshot()
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
                "idle_relay": self._vmc_idle.snapshot() if self._vmc_idle else {"enabled": False},
                "vrchat_osc": self._osc.awareness() if self._osc else {"enabled": False},
            })
        snapshot = self._scheduler.snapshot()
        driver_log = self._driver_log_snapshot()
        self._apply_driver_log_to_udp(snapshot, driver_log)
        return Ok({
            "state": snapshot["state"],
            "output_enabled": snapshot["output_enabled"],
            "safety_state": snapshot["safety_state"],
            "queue_length": snapshot["queue_length"],
            **snapshot["awareness"],
            "driver_delivery": self._driver_delivery_awareness(snapshot, driver_log),
            "vrchat_osc": self._osc.awareness() if self._osc else {
                "enabled": False,
                "connection": "unknown",
                "summary": "VRChat OSC 桥接器尚未初始化。",
                "parameters": {},
                "pose_feedback_available": False,
                "pickup_confirmation_available": False,
            },
        })


__all__ = ["NekoAnyadanceBodyPlugin"]
