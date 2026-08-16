"""Single-owner real-time scheduler for AnyaDance body output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import copy
import itertools
import math
import queue
import socket
import threading
import time
import uuid
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from .behavior import BehaviorStateMachine, expression_admission
from .config import PluginConfig
from .model import CONTROLLER_IDS, FrameState, neutral_frame, neutralize_inputs
from .nya import NyaClip, sample_clip
from .motion import (
    GESTURE_DURATIONS,
    apply_hand_pose,
    arm_pose_target,
    gesture_frame,
    interpolate_frame,
    move_hand_target,
    reach_target,
)
from .protocol import encode_frame, validate_frame
from .expression_motion import (
    EXPRESSION_GESTURES,
    ExpressionOverlay,
    apply_expression_overlay,
    sample_expression,
)


class DatagramTransport(Protocol):
    def send(self, payload: bytes, address: tuple[str, int]) -> None: ...
    def close(self) -> None: ...


class IdleFrameSource(Protocol):
    def latest_frame(self) -> FrameState | None: ...
    def snapshot(self) -> Dict[str, Any]: ...


class UdpTransport:
    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)

    def send(self, payload: bytes, address: tuple[str, int]) -> None:
        self._socket.sendto(payload, address)

    def close(self) -> None:
        self._socket.close()


@dataclass(frozen=True)
class BodyCommand:
    kind: str
    action_id: str
    params: Dict[str, Any]


@dataclass(order=True)
class PrioritizedCommand:
    priority: int
    sequence: int
    command: BodyCommand = field(compare=False)


@dataclass
class ActiveMotion:
    action_id: str
    name: str
    started_at: float
    duration_s: float
    sampler: Callable[[float], FrameState]
    completion: str
    normalized_params: Dict[str, Any]


class CommandRejected(ValueError):
    """A normalized command cannot be run without faulting the scheduler."""


@dataclass
class SequenceSegment:
    offset_s: float
    duration_s: float
    sampler: Callable[[float], FrameState]
    end_frame: FrameState


def advance_deadline(deadline: float, now: float, period: float) -> tuple[float, int]:
    """Advance an absolute clock deadline, skipping expired slots without catch-up bursts."""
    if now <= deadline:
        return deadline + period, 0
    missed = int((now - deadline) // period)
    return deadline + (missed + 1) * period, missed


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _quat_angle_deg(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dot = abs(sum(left * right for left, right in zip(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


class BodyScheduler:
    NORMAL_COMMANDS = {
        "arm_pose", "hand", "move_hand", "reach_and_grab", "gesture", "sequence", "play_clip",
        "express", "semantic_clip",
    }

    def __init__(
        self,
        config: PluginConfig,
        *,
        logger: Any = None,
        transport: Optional[DatagramTransport] = None,
        idle_frame_source: Optional[IdleFrameSource] = None,
        motion_started_callback: Callable[[BodyCommand, float], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.logger = logger
        self._transport = transport
        self._owns_transport = transport is None
        self._idle_frame_source = idle_frame_source
        self._motion_started_callback = motion_started_callback
        self._clock = clock
        self._sleep = sleeper
        self._commands: queue.PriorityQueue[PrioritizedCommand] = queue.PriorityQueue(config.max_queue_size)
        self._sequence = itertools.count()
        self._submit_lock = threading.Lock()
        self._urgent_lock = threading.Lock()
        self._urgent_stop: Optional[BodyCommand] = None
        self._snapshot_lock = threading.Lock()
        self._snapshot: Dict[str, Any] = self._initial_snapshot()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # The following fields are owned by the scheduler thread after start().
        self._frame = neutral_frame()
        self._output_frame = self._frame.clone()
        self._enabled = False
        self._state = "disabled"
        self._safety_state = "normal"
        self._active: Optional[ActiveMotion] = None
        self._current_action: Optional[Dict[str, Any]] = None
        self._previous_action: Optional[Dict[str, Any]] = None
        self._transition: Optional[Dict[str, Any]] = None
        self._disable_frames_remaining = 0
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._sent_packets = 0
        self._send_failures = 0
        self._skipped_frames = 0
        self._last_send_at: Optional[float] = None
        self._send_intervals: deque[float] = deque(maxlen=120)
        self._last_lateness_ms = 0.0
        self._max_lateness_ms = 0.0
        self._arm_state: Dict[str, Any] = {
            "left": {"mode": "neutral", "elevation_deg": None, "plane": None, "reach": None},
            "right": {"mode": "neutral", "elevation_deg": None, "plane": None, "reach": None},
        }
        self._hand_state = {"left": "open", "right": "open"}
        self._expression_overlays: list[ExpressionOverlay] = []
        self._last_expression: Optional[Dict[str, Any]] = None
        self._idle_relay_applied = False
        self._idle_relay_error: Optional[str] = None
        self._behavior = BehaviorStateMachine(
            history_size=self.config.behavior.transition_history_size
        )

    def _initial_snapshot(self) -> Dict[str, Any]:
        return {
            "state": "disabled",
            "safety_state": "normal",
            "output_enabled": False,
            "udp": {
                "target": f"{self.config.host}:{self.config.port}",
                "connected": "unknown",
                "last_send_at_monotonic": None,
                "sent_packets": 0,
                "send_failures": 0,
            },
            "current_action": None,
            "awareness": {
                "motion": None,
            "expression_overlays": [],
                "previous_action": None,
                "transition": None,
                "pose": {},
                "summary": "身体输出未启用。",
                "updated_at_monotonic": None,
            },
            "queue_length": 0,
            "arms": {},
            "hands": {},
            "expression_motion": {"active": [], "last": None},
            "idle_relay": {
                "enabled": self.config.vmc_idle.enabled,
                "listen_address": f"{self.config.vmc_idle.listen_host}:{self.config.vmc_idle.listen_port}",
                "connection": "unknown",
                "source_available": False,
                "applied": False,
            },
            "behavior": {
                "mode": "disabled",
                "phase": "disabled",
                "base": None,
                "overlays": [],
                "active_layers": [],
                "previous_base": None,
                "transition": None,
                "history": [],
                "last_decision": None,
                "policy_version": 1,
            },
            "metrics": {"actual_hz": 0.0, "skipped_frames": 0, "last_lateness_ms": 0.0, "max_lateness_ms": 0.0},
            "last_error": None,
            "error_count": 0,
            "concurrent_sender_detection": "unsupported",
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="neko-anyadance-body", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        if thread and thread.is_alive() and self.logger:
            self.logger.warning("AnyaDance scheduler did not stop within %.1f seconds", timeout)

    @property
    def thread_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> Dict[str, Any]:
        with self._snapshot_lock:
            return copy.deepcopy(self._snapshot)

    def submit(self, kind: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        action_id = str(uuid.uuid4())
        current = self.snapshot()
        state = current["state"]
        if not self.thread_alive:
            return self._rejection(action_id, state, "scheduler is not running")
        if kind in self.NORMAL_COMMANDS and state in {"disabled", "stopped_latched", "fault_latched", "shutdown"}:
            return self._rejection(action_id, state, f"body output cannot accept motions while state is {state}")
        if kind == "reset" and state == "disabled":
            return self._rejection(action_id, state, "body output is disabled; call body_enable first")
        if kind == "enable" and state != "disabled":
            return self._rejection(action_id, state, "body output is already enabled")
        if kind == "disable" and state == "disabled":
            return self._rejection(action_id, state, "body output is already disabled")
        if kind in {"express", "semantic_clip"} and self.config.behavior.protect_full_body_motion:
            gesture = str(params.get("gesture", "")) if kind == "express" else "__full_body_clip__"
            allowed, reason = expression_admission(current, gesture)
            if not allowed:
                return self._rejection(action_id, state, reason or "expression is blocked by current behavior")

        if kind == "stop":
            priority = 0
        elif kind in {"disable", "reset", "enable", "cancel"}:
            priority = 1
        elif kind in {"express", "semantic_clip"}:
            priority = 15
        else:
            priority = 10
        command = BodyCommand(kind=kind, action_id=action_id, params=params)
        public_params = {key: value for key, value in params.items() if not key.startswith("_")}
        if kind == "stop":
            # Emergency stop never competes for capacity with the bounded normal queue.
            # Repeated stops coalesce to the newest action id.
            with self._urgent_lock:
                self._urgent_stop = command
            predicted = "disabled" if state == "disabled" else "stopped_latched"
            return {
                "accepted": True,
                "action_id": action_id,
                "state": predicted,
                "normalized_params": public_params,
                "target_pose_summary": self._target_pose_summary(kind, public_params),
                "reason": None,
                "safety_state": current["safety_state"],
            }
        with self._submit_lock:
            item = PrioritizedCommand(priority, next(self._sequence), command)
            try:
                self._commands.put_nowait(item)
            except queue.Full:
                return self._rejection(action_id, state, "motion command queue is full")
        if kind == "enable":
            predicted = "idle"
        elif kind == "cancel":
            predicted = "holding"
        else:
            predicted = "moving"
        return {
            "accepted": True,
            "action_id": action_id,
            "state": predicted,
            "normalized_params": public_params,
            "target_pose_summary": self._target_pose_summary(kind, public_params),
            "reason": None,
            "safety_state": current["safety_state"],
        }

    def _rejection(self, action_id: str, state: str, reason: str) -> Dict[str, Any]:
        return {
            "accepted": False,
            "action_id": action_id,
            "state": state,
            "normalized_params": {},
            "reason": reason,
            "safety_state": self.snapshot().get("safety_state", "normal"),
        }

    @staticmethod
    def _target_pose_summary(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the intended result without implying the motion has completed."""
        if kind == "arm_pose":
            azimuth = params.get("azimuth_deg")
            direction = params.get("plane") if azimuth is None else f"方位角 {azimuth:g}°"
            description = (
                f"{params.get('side', '指定')}手臂抬升到 {params.get('elevation_deg', 0):g}°，"
                f"方向 {direction}，伸展 {params.get('reach', 0):g}。"
            )
        elif kind == "move_hand":
            description = (
                f"{params.get('side', '指定')}手移动到相对 {params.get('relative_to')} 的 "
                f"({params.get('x_m', 0):g}, {params.get('y_m', 0):g}, {params.get('z_m', 0):g}) 米。"
            )
        elif kind == "hand":
            description = f"{params.get('side', '指定')}手变为 {params.get('pose')}，强度 {params.get('strength', 0):g}。"
        elif kind == "reach_and_grab":
            description = (
                f"{params.get('side', '指定')}手向 {params.get('height')} 高度、{params.get('direction')} 方向"
                f"伸出 {params.get('distance_m', 0):g} 米并触发握持；是否拿到物体未知。"
            )
        elif kind == "gesture":
            description = f"播放 {params.get('name')} 手势，结束后恢复动作前姿态。"
        elif kind == "sequence":
            description = f"播放 {len(params.get('steps', []))} 步动作序列，共 {params.get('loop_count', 1)} 轮。"
        elif kind in {"play_clip", "semantic_clip"}:
            description = (
                f"播放{'语义 VMD' if kind == 'semantic_clip' else '预制'}动作“{params.get('clip_name')}”，"
                f"速度 {params.get('speed', 1):g} 倍，"
                f"循环 {params.get('loop_count', 1)} 次。"
            )
        elif kind == "express":
            description = (
                f"以{params.get('intent_label', params.get('intent', '语义'))}意图播放"
                f"低优先级 {params.get('gesture')} 表达动作；高优先级全身动作会受到保护。"
            )
        else:
            descriptions = {
                "enable": "启用身体输出并采用标准 T Pose。",
                "disable": "平滑返回标准 T Pose 后停止输出。",
                "stop": "立即冻结当前合法姿态并释放控制器输入。",
                "reset": "解除安全锁定并平滑返回标准 T Pose。",
                "cancel": "取消当前动作并停在已经到达的合法姿态。",
            }
            description = descriptions.get(kind, kind)
        return {"kind": kind, "description": description, "completion_confirmed": False}

    def _run(self) -> None:
        period = 1.0 / self.config.rate_hz
        deadline = self._clock()
        try:
            if self._transport is None:
                self._transport = UdpTransport()
            while not self._stop_event.is_set():
                now = self._clock()
                if now < deadline:
                    self._sleep(min(deadline - now, period))
                    continue
                lateness_ms = max(0.0, (now - deadline) * 1000.0)
                self._last_lateness_ms = lateness_ms
                self._max_lateness_ms = max(self._max_lateness_ms, lateness_ms)
                next_deadline, skipped = advance_deadline(deadline, now, period)
                deadline = next_deadline
                self._skipped_frames += skipped

                try:
                    self._process_one_command(now)
                    self._sample_active_motion(now)
                    self._sample_idle_relay()
                    self._sample_expression_motion(now)
                    if self._enabled:
                        self._send_current_frame(now)
                        if self._disable_frames_remaining > 0:
                            self._disable_frames_remaining -= 1
                            if self._disable_frames_remaining == 0:
                                self._enabled = False
                                self._state = "disabled"
                                self._active = None
                                self._clear_current_action(now, "completed")
                    self._publish_snapshot()
                except Exception as exc:  # scheduler must fail safe instead of dying
                    self._enter_fault(exc)
                    self._publish_snapshot()
        finally:
            self._shutdown_output(period)

    def _process_one_command(self, now: float) -> None:
        with self._urgent_lock:
            urgent = self._urgent_stop
            self._urgent_stop = None
        if urgent is not None:
            self._handle_stop(urgent, now)
            return
        try:
            item = self._commands.get_nowait()
        except queue.Empty:
            return
        try:
            command = item.command
            kind = command.kind
            if kind == "stop":
                self._handle_stop(command, now)
                return
            if kind == "enable":
                self._frame = neutral_frame()
                self._output_frame = self._frame.clone()
                self._enabled = True
                self._state = "idle"
                self._safety_state = "normal"
                self._active = None
                self._clear_current_action(now, "replaced")
                self._last_error = None
                self._drop_expression_overlays(now, "replaced")
                self._behavior.reset(now=now, outcome="replaced")
                return
            if kind == "disable":
                self._clear_commands()
                self._drop_expression_overlays(now, "replaced")
                self._start_target_motion(command, now, neutral_frame(), completion="disable")
                return
            if kind == "reset":
                self._clear_commands()
                self._drop_expression_overlays(now, "replaced")
                self._safety_state = "normal"
                self._last_error = None
                self._start_target_motion(command, now, neutral_frame(), completion="idle")
                self._arm_state = {
                    "left": {"mode": "neutral", "elevation_deg": None, "plane": None, "reach": None},
                    "right": {"mode": "neutral", "elevation_deg": None, "plane": None, "reach": None},
                }
                self._hand_state = {"left": "open", "right": "open"}
                return
            if kind == "cancel":
                target_id = command.params.get("action_id")
                if self._active is not None and (not target_id or self._active.action_id == target_id):
                    self._active = None
                    self._state = "holding"
                    self._clear_current_action(now, "cancelled")
                elif self._current_action is not None and (
                    not target_id or self._current_action.get("id") == target_id
                ):
                    self._clear_current_action(now, "cancelled")
                return
            if kind in {"express", "semantic_clip"} and self.config.behavior.protect_full_body_motion:
                live_snapshot = {
                    "state": self._state,
                    "behavior": self._behavior.snapshot(runtime_state=self._state, now=now),
                }
                gesture = str(command.params.get("gesture", "")) if kind == "express" else "__full_body_clip__"
                allowed, reason = expression_admission(live_snapshot, gesture)
                if not allowed:
                    self._behavior.reject(
                        action_id=command.action_id,
                        kind=kind,
                        now=now,
                        reason=reason or "expression is blocked by current behavior",
                    )
                    return
            if kind == "express":
                self._start_expression_overlay(command, now)
                return
            self._cancel_expression_overlays(now)
            if kind == "arm_pose":
                target = arm_pose_target(
                    self._frame,
                    profile=self.config.profile,
                    **{key: value for key, value in command.params.items() if key != "duration_ms"},
                )
                validate_frame(target, self.config.safety)
                self._start_target_motion(command, now, target, completion="holding")
                for side in self._selected_sides(command.params["side"]):
                    self._arm_state[side] = {
                        "mode": "angle",
                        "elevation_deg": command.params["elevation_deg"],
                        "azimuth_deg": command.params.get("azimuth_deg"),
                        "plane": command.params.get("plane"),
                        "reach": command.params["reach"],
                    }
                return
            if kind == "hand":
                target = apply_hand_pose(
                    self._frame,
                    **{key: value for key, value in command.params.items() if key != "duration_ms"},
                )
                validate_frame(target, self.config.safety)
                self._start_target_motion(command, now, target, completion="holding")
                for side in self._selected_sides(command.params["side"]):
                    self._hand_state[side] = command.params["pose"]
                return
            if kind == "move_hand":
                target = move_hand_target(
                    self._frame,
                    profile=self.config.profile,
                    **{key: value for key, value in command.params.items() if key != "duration_ms"},
                )
                validate_frame(target, self.config.safety)
                self._start_target_motion(command, now, target, completion="holding")
                side = command.params["side"]
                self._arm_state[side] = {
                    "mode": "local_target",
                    "relative_to": command.params["relative_to"],
                    "position_m": {
                        "x": command.params["x_m"],
                        "y": command.params["y_m"],
                        "z": command.params["z_m"],
                    },
                }
                return
            if kind == "reach_and_grab":
                self._start_reach_and_grab(command, now)
                self._arm_state[command.params["side"]] = {
                    "mode": "reach",
                    "height": command.params["height"],
                    "direction": command.params["direction"],
                    "distance_m": command.params["distance_m"],
                }
                self._hand_state[command.params["side"]] = "grip"
                return
            if kind == "gesture":
                self._start_gesture(command, now)
                return
            if kind == "sequence":
                self._start_sequence(command, now)
                return
            if kind in {"play_clip", "semantic_clip"}:
                self._start_clip(command, now)
                return
            raise ValueError(f"unknown scheduler command: {kind}")
        except CommandRejected as exc:
            self._behavior.reject(
                action_id=command.action_id,
                kind=command.kind,
                now=now,
                reason=str(exc),
            )
        finally:
            self._commands.task_done()

    def _handle_stop(self, command: BodyCommand, now: float) -> None:
        self._clear_commands()
        self._active = None
        self._drop_expression_overlays(now, "stopped")
        neutralize_inputs(self._frame)
        self._state = "stopped_latched" if self._enabled else "disabled"
        self._safety_state = "stopped" if self._enabled else "normal"
        self._set_current_action(
            {"id": command.action_id, "name": "body_stop", "progress": 1.0},
            now,
            policy_kind=command.kind,
            policy_params=command.params,
        )

    def _start_expression_overlay(self, command: BodyCommand, now: float) -> None:
        gesture = str(command.params.get("gesture", ""))
        if gesture not in EXPRESSION_GESTURES:
            raise ValueError(f"unsupported expression gesture: {gesture}")
        intent = str(command.params.get("intent", ""))[:64] or None
        # One expression lane mirrors AnimationMixer's newest-action fade:
        # cancel the outgoing overlay, then blend the new delta over the live base.
        self._cancel_expression_overlays(now)
        overlay = ExpressionOverlay(
            action_id=command.action_id,
            gesture=gesture,
            side=str(command.params.get("side", "right")),
            energy=float(command.params.get("energy", 0.4)),
            started_at=now,
            duration_s=float(command.params.get("duration_ms", 1200)) / 1000.0,
            reference=self._frame.clone(),
            source="llm_intent",
            intent=intent,
        )
        self._expression_overlays.append(overlay)
        self._behavior.activate_overlay(
            action_id=command.action_id,
            kind=command.kind,
            now=now,
            params=command.params,
        )
        self._last_expression = {
            "id": command.action_id,
            "gesture": gesture,
            "intent": intent,
            "source": overlay.source,
            "side": overlay.side,
            "energy": round(overlay.energy, 3),
            "started_at_monotonic": now,
            "state": "moving",
        }

    def _drop_expression_overlays(self, now: float, outcome: str) -> None:
        for overlay in self._expression_overlays:
            self._behavior.finish_overlay(
                action_id=overlay.action_id,
                now=now,
                outcome=outcome,
            )
        self._expression_overlays.clear()

    def _cancel_expression_overlays(self, now: float) -> None:
        for overlay in self._expression_overlays:
            if overlay.cancelled_at is None:
                overlay.cancelled_at = now

    def _sample_expression_motion(self, now: float) -> None:
        output = self._frame.clone()
        remaining: list[ExpressionOverlay] = []
        for overlay in self._expression_overlays:
            sampled, channels, weight = sample_expression(overlay, now, self.config.profile)
            output = apply_expression_overlay(output, overlay.reference, sampled, channels, weight)
            if not overlay.expired(now):
                remaining.append(overlay)
            else:
                outcome = "completed" if overlay.cancelled_at is None else "cancelled"
                self._behavior.finish_overlay(
                    action_id=overlay.action_id,
                    now=now,
                    outcome=outcome,
                )
                if self._last_expression and self._last_expression.get("id") == overlay.action_id:
                    self._last_expression["state"] = outcome
                    self._last_expression["ended_at_monotonic"] = now
        self._expression_overlays = remaining
        validate_frame(output, self.config.safety)
        self._output_frame = output

    def _sample_idle_relay(self) -> None:
        self._idle_relay_applied = False
        source = self._idle_frame_source
        if source is None or not self._enabled or self._state != "idle" or self._active is not None:
            return
        try:
            frame = source.latest_frame()
        except Exception as exc:
            self._idle_relay_error = f"idle VMC source failed: {exc}"
            return
        if frame is None:
            return
        try:
            validate_frame(frame, self.config.safety)
        except ValueError as exc:
            self._idle_relay_error = f"idle VMC frame rejected: {exc}"
            return
        self._frame = frame
        self._idle_relay_applied = True
        self._idle_relay_error = None

    def _start_target_motion(self, command: BodyCommand, now: float, target: FrameState, *, completion: str) -> None:
        start = self._frame.clone()
        requested = float(command.params.get("duration_ms", self.config.default_duration_ms)) / 1000.0
        duration = max(requested, self._minimum_safe_duration(start, target))
        self._active = ActiveMotion(
            action_id=command.action_id,
            name=command.kind,
            started_at=now,
            duration_s=duration,
            sampler=lambda progress, a=start, b=target: interpolate_frame(a, b, progress),
            completion=completion,
            normalized_params={**command.params, "applied_duration_ms": round(duration * 1000.0)},
        )
        self._state = "moving"
        self._set_current_action(
            {"id": command.action_id, "name": command.kind, "progress": 0.0},
            now,
            policy_kind=command.kind,
            policy_params=command.params,
        )

    def _start_reach_and_grab(self, command: BodyCommand, now: float) -> None:
        start = self._frame.clone()
        target_open = reach_target(self._frame, profile=self.config.profile, **{
            key: value for key, value in command.params.items() if key != "duration_ms"
        })
        target_grip = apply_hand_pose(target_open, side=command.params["side"], pose="grip", strength=1.0)
        validate_frame(target_grip, self.config.safety)
        requested = command.params["duration_ms"] / 1000.0
        duration = max(requested, self._minimum_safe_duration(start, target_open))

        def sample(progress: float) -> FrameState:
            result = interpolate_frame(start, target_open, progress)
            if progress >= 0.85:
                grip_progress = (progress - 0.85) / 0.15
                return interpolate_frame(result, target_grip, grip_progress)
            return result

        self._active = ActiveMotion(
            action_id=command.action_id,
            name=command.kind,
            started_at=now,
            duration_s=duration,
            sampler=sample,
            completion="holding",
            normalized_params={**command.params, "applied_duration_ms": round(duration * 1000.0)},
        )
        self._state = "moving"
        self._set_current_action(
            {"id": command.action_id, "name": command.kind, "progress": 0.0},
            now,
            policy_kind=command.kind,
            policy_params=command.params,
        )
        callback = self._motion_started_callback
        if callback is not None:
            try:
                callback(command, duration)
            except Exception as exc:
                if self.logger:
                    self.logger.warning("motion-start callback failed: %s", exc)

    def _start_gesture(self, command: BodyCommand, now: float) -> None:
        start = self._frame.clone()
        duration = GESTURE_DURATIONS[command.params["name"]]
        self._active = ActiveMotion(
            action_id=command.action_id,
            name=command.kind,
            started_at=now,
            duration_s=duration,
            sampler=lambda progress, base=start: gesture_frame(base, progress=progress, profile=self.config.profile, **command.params),
            completion="idle",
            normalized_params={**command.params, "applied_duration_ms": round(duration * 1000.0)},
        )
        self._state = "moving"
        self._set_current_action(
            {
                "id": command.action_id,
                "name": command.params["name"],
                "action_type": "gesture",
                "progress": 0.0,
            },
            now,
            policy_kind=command.kind,
            policy_params=command.params,
        )

    def _start_sequence(self, command: BodyCommand, now: float) -> None:
        segments: list[SequenceSegment] = []
        current = self._frame.clone()
        offset = 0.0
        steps = command.params["steps"]
        for _ in range(command.params["loop_count"]):
            for step in steps:
                kind = step["type"]
                start = current.clone()
                requested = step["duration_ms"] / 1000.0
                if kind == "wait":
                    target = start.clone()
                    duration = requested
                    sampler = lambda progress, base=start: base.clone()
                elif kind == "gesture":
                    target = start.clone()
                    duration = requested
                    sampler = lambda progress, base=start, spec=step: gesture_frame(
                        base,
                        name=spec["name"],
                        side=spec["side"],
                        intensity=spec["intensity"],
                        progress=progress,
                        profile=self.config.profile,
                    )
                else:
                    if kind == "arm_pose":
                        target = arm_pose_target(
                            start,
                            profile=self.config.profile,
                            **{key: value for key, value in step.items() if key not in {"type", "duration_ms"}},
                        )
                    elif kind == "hand":
                        target = apply_hand_pose(
                            start,
                            **{key: value for key, value in step.items() if key not in {"type", "duration_ms"}},
                        )
                    elif kind == "move_hand":
                        target = move_hand_target(
                            start,
                            profile=self.config.profile,
                            **{key: value for key, value in step.items() if key not in {"type", "duration_ms"}},
                        )
                    else:
                        raise ValueError(f"unsupported sequence step: {kind}")
                    validate_frame(target, self.config.safety)
                    duration = max(requested, self._minimum_safe_duration(start, target))
                    sampler = lambda progress, a=start, b=target: interpolate_frame(a, b, progress)
                segments.append(SequenceSegment(offset, duration, sampler, target.clone()))
                offset += duration
                current = target.clone()

        total_duration = max(offset, 1.0 / self.config.rate_hz)
        if total_duration > 30.0:
            raise CommandRejected("expanded sequence duration must not exceed 30000 ms")

        def sample(progress: float) -> FrameState:
            elapsed = min(total_duration, max(0.0, progress) * total_duration)
            for segment in segments:
                if elapsed <= segment.offset_s + segment.duration_s:
                    local = (elapsed - segment.offset_s) / max(segment.duration_s, 1e-6)
                    return segment.sampler(min(1.0, max(0.0, local)))
            return current.clone()

        self._active = ActiveMotion(
            action_id=command.action_id,
            name="sequence",
            started_at=now,
            duration_s=total_duration,
            sampler=sample,
            completion="holding",
            normalized_params={
                "step_count": len(steps),
                "loop_count": command.params["loop_count"],
                "applied_duration_ms": round(total_duration * 1000.0),
            },
        )
        self._state = "moving"
        self._set_current_action(
            {"id": command.action_id, "name": "sequence", "progress": 0.0, "step_count": len(steps)},
            now,
            policy_kind=command.kind,
            policy_params=command.params,
        )

    def _start_clip(self, command: BodyCommand, now: float) -> None:
        clip: NyaClip = command.params["_clip"]
        base = self._frame.clone()
        anchor = command.params["anchor"]
        if anchor:
            first_hmd = clip.frames[0].frame.devices["hmd"].position
            current_hmd = base.devices["hmd"].position
            offset_x = current_hmd[0] - first_hmd[0]
            offset_z = current_hmd[2] - first_hmd[2]
        else:
            offset_x = 0.0
            offset_z = 0.0
        first = sample_clip(clip, 0.0, base=base, offset_x=offset_x, offset_z=offset_z)
        last = sample_clip(clip, clip.duration_s, base=base, offset_x=offset_x, offset_z=offset_z)
        validate_frame(first, self.config.safety)
        validate_frame(last, self.config.safety)

        requested_transition = command.params["transition_ms"] / 1000.0
        transition_in = max(requested_transition, self._minimum_safe_duration(base, first))
        speed = command.params["speed"]
        loops = command.params["loop_count"]
        playback_duration = 0.0 if clip.is_pose else (clip.duration_s / speed) * loops
        restore_after = command.params["restore_after"]
        transition_out = max(requested_transition, self._minimum_safe_duration(last, base)) if restore_after else 0.0
        total_duration = max(transition_in + playback_duration + transition_out, 1.0 / self.config.rate_hz)

        def sample(progress: float) -> FrameState:
            elapsed = min(total_duration, max(0.0, progress) * total_duration)
            if elapsed < transition_in:
                return interpolate_frame(base, first, elapsed / max(transition_in, 1e-6))
            elapsed -= transition_in
            if playback_duration > 0.0 and elapsed < playback_duration:
                source_time = elapsed * speed
                local_time = source_time % clip.duration_s
                return sample_clip(clip, local_time, base=base, offset_x=offset_x, offset_z=offset_z)
            if restore_after:
                elapsed = max(0.0, elapsed - playback_duration)
                return interpolate_frame(last, base, min(1.0, elapsed / max(transition_out, 1e-6)))
            return last.clone()

        action_name = "semantic_clip" if command.kind == "semantic_clip" else "play_clip"
        semantic_fields = {
            key: copy.deepcopy(command.params[key])
            for key in ("semantic_intent", "intent_label", "motion_source", "motion_label", "source_name")
            if key in command.params
        }
        self._active = ActiveMotion(
            action_id=command.action_id,
            name=action_name,
            started_at=now,
            duration_s=total_duration,
            sampler=sample,
            completion="idle" if restore_after else "holding",
            normalized_params={
                "clip_name": clip.name,
                "speed": speed,
                "loop_count": loops,
                "anchor": anchor,
                "restore_after": restore_after,
                "applied_transition_ms": round(transition_in * 1000.0),
                "applied_duration_ms": round(total_duration * 1000.0),
                **semantic_fields,
            },
        )
        self._state = "moving"
        self._set_current_action(
            {
                "id": command.action_id,
                "name": action_name,
                "clip_name": clip.name,
                "progress": 0.0,
                **semantic_fields,
            },
            now,
            policy_kind=command.kind,
            policy_params=command.params,
        )

    def _sample_active_motion(self, now: float) -> None:
        motion = self._active
        if motion is None:
            return
        progress = min(1.0, max(0.0, (now - motion.started_at) / max(motion.duration_s, 1e-6)))
        self._frame = motion.sampler(progress)
        validate_frame(self._frame, self.config.safety)
        if self._current_action is not None:
            self._current_action["progress"] = round(progress, 4)
            self._current_action["params"] = motion.normalized_params
        if progress < 1.0:
            return
        self._active = None
        if motion.completion == "disable":
            self._disable_frames_remaining = 6
            self._state = "moving"
        elif motion.completion == "holding":
            self._state = "holding"
        else:
            self._state = "idle"
            self._clear_current_action(now, "completed")

    @staticmethod
    def _action_descriptor(action: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        descriptor = {
            key: copy.deepcopy(action[key])
            for key in (
                "id", "name", "action_type", "clip_name", "progress", "step_count",
                "semantic_intent", "intent_label", "motion_source", "motion_label", "source_name",
            )
            if key in action
        }
        params = action.get("params")
        if isinstance(params, dict):
            for key in (
                "side", "pose", "name", "clip_name", "loop_count", "speed",
                "semantic_intent", "intent_label", "motion_source", "motion_label", "source_name",
            ):
                if key in params:
                    descriptor[key] = copy.deepcopy(params[key])
        return descriptor

    def _set_current_action(
        self,
        action: Dict[str, Any],
        now: float,
        *,
        policy_kind: str,
        policy_params: Mapping[str, Any] | None = None,
    ) -> None:
        action.setdefault("started_at_monotonic", now)
        previous = self._action_descriptor(self._current_action)
        if previous is not None:
            previous["outcome"] = "interrupted"
            previous["ended_at_monotonic"] = now
            self._previous_action = previous
        self._transition = {
            "from": previous,
            "to": self._action_descriptor(action),
            "started_at_monotonic": now,
        }
        self._current_action = action
        self._behavior.activate_base(
            action_id=str(action.get("id", "")),
            kind=policy_kind,
            now=now,
            params=policy_params,
        )

    def _clear_current_action(self, now: float, outcome: str) -> None:
        action_id = str(self._current_action.get("id", "")) if self._current_action else None
        previous = self._action_descriptor(self._current_action)
        if previous is not None:
            previous["outcome"] = outcome
            previous["ended_at_monotonic"] = now
            self._previous_action = previous
            self._transition = {
                "from": previous,
                "to": None,
                "started_at_monotonic": now,
            }
        self._current_action = None
        self._behavior.finish_base(action_id=action_id, now=now, outcome=outcome)

    def _minimum_safe_duration(self, start: FrameState, target: FrameState) -> float:
        linear = max(
            _distance(start.devices[name].position, target.devices[name].position)
            for name in start.devices
        ) / self.config.safety.max_linear_speed_mps
        angular = max(
            _quat_angle_deg(start.devices[name].rotation, target.devices[name].rotation)
            for name in start.devices
        ) / self.config.safety.max_angular_speed_dps
        return max(linear, angular, 1.0 / self.config.rate_hz)

    def _send_current_frame(self, now: float) -> None:
        if self._transport is None:
            raise RuntimeError("UDP transport is not initialized")
        try:
            payload = encode_frame(self._output_frame, self.config.safety)
            self._transport.send(payload, (self.config.host, self.config.port))
        except Exception:
            self._send_failures += 1
            raise
        if self._last_send_at is not None:
            self._send_intervals.append(max(0.0, now - self._last_send_at))
        self._last_send_at = now
        self._sent_packets += 1

    def _enter_fault(self, exc: Exception) -> None:
        now = self._clock()
        self._error_count += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._active = None
        self._drop_expression_overlays(now, "faulted")
        self._frame = neutral_frame()
        self._output_frame = self._frame.clone()
        neutralize_inputs(self._frame)
        self._state = "fault_latched" if self._enabled else "disabled"
        self._safety_state = "fault" if self._enabled else "normal"
        self._clear_current_action(now, "faulted")
        self._clear_commands()
        if self.logger:
            self.logger.exception("AnyaDance scheduler entered fail-safe state: %s", exc)

    def _shutdown_output(self, period: float) -> None:
        if self._enabled and self._transport is not None:
            self._frame = neutral_frame()
            self._output_frame = self._frame.clone()
            for _ in range(6):
                try:
                    self._send_current_frame(self._clock())
                except Exception as exc:
                    self._last_error = f"shutdown send failed: {exc}"
                    break
                self._sleep(period)
        self._enabled = False
        self._active = None
        now = self._clock()
        self._drop_expression_overlays(now, "shutdown")
        self._state = "shutdown"
        self._clear_current_action(now, "shutdown")
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:
                pass
        self._publish_snapshot()

    def _clear_commands(self) -> None:
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                return
            else:
                self._commands.task_done()

    @staticmethod
    def _selected_sides(side: str) -> tuple[str, ...]:
        return ("left", "right") if side == "both" else (side,)

    @staticmethod
    def _direction_label(azimuth_deg: float) -> str:
        if -45.0 <= azimuth_deg <= 45.0:
            return "forward"
        if 45.0 < azimuth_deg < 135.0:
            return "right"
        if -135.0 < azimuth_deg < -45.0:
            return "left"
        return "backward"

    @staticmethod
    def _elevation_label(elevation_deg: float) -> str:
        if elevation_deg < 30.0:
            return "lowered"
        if elevation_deg < 70.0:
            return "low"
        if elevation_deg <= 110.0:
            return "horizontal"
        if elevation_deg < 155.0:
            return "raised"
        return "overhead"

    @staticmethod
    def _hand_pose(controller: Any) -> str:
        bends = controller.finger_bends
        values = tuple(float(bends.get(name, 0.0)) for name in ("thumb", "index", "middle", "ring", "pinky"))
        if controller.grip_click or controller.grip_value >= 0.75:
            return "grip"
        if values[1] < 0.3 and min(values[2:]) > 0.6:
            return "point"
        average = sum(values) / len(values)
        if average < 0.15:
            return "open"
        if average > 0.8:
            return "fist"
        return "partially_closed"

    @staticmethod
    def _quat_to_euler_deg(quat: tuple[float, float, float, float]) -> Dict[str, float]:
        x, y, z, w = quat
        sin_pitch = 2.0 * (w * x + y * z)
        cos_pitch = 1.0 - 2.0 * (x * x + y * y)
        pitch = math.atan2(sin_pitch, cos_pitch)
        sin_yaw = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        yaw = math.asin(sin_yaw)
        sin_roll = 2.0 * (w * z + x * y)
        cos_roll = 1.0 - 2.0 * (y * y + z * z)
        roll = math.atan2(sin_roll, cos_roll)
        return {
            "pitch_deg": round(math.degrees(pitch), 1),
            "yaw_deg": round(math.degrees(yaw), 1),
            "roll_deg": round(math.degrees(roll), 1),
        }

    def _semantic_pose(self) -> Dict[str, Any]:
        frame = self._output_frame
        hmd = frame.devices["hmd"]
        pose: Dict[str, Any] = {"head": self._quat_to_euler_deg(hmd.rotation)}
        for side, sign in (("left", -1.0), ("right", 1.0)):
            shoulder = (
                hmd.position[0] + sign * self.config.profile.shoulder_width_m / 2.0,
                hmd.position[1] - self.config.profile.shoulder_drop_m,
                hmd.position[2],
            )
            hand = frame.devices[f"{side}_controller"].position
            vector = tuple(hand[index] - shoulder[index] for index in range(3))
            distance = max(_distance(hand, shoulder), 1e-9)
            elevation = math.degrees(math.acos(max(-1.0, min(1.0, -vector[1] / distance))))
            azimuth = math.degrees(math.atan2(vector[0], -vector[2]))
            controller = frame.controllers[f"{side}_controller"]
            pose[f"{side}_arm"] = {
                "elevation_deg": round(elevation, 1),
                "azimuth_deg": round(azimuth, 1),
                "reach": round(distance / self.config.profile.arm_length_m, 2),
                "elevation": self._elevation_label(elevation),
                "direction": self._direction_label(azimuth),
                "hand_position_m": {"x": round(hand[0], 3), "y": round(hand[1], 3), "z": round(hand[2], 3)},
            }
            pose[f"{side}_hand"] = {
                "pose": self._hand_pose(controller),
                "rotation": self._quat_to_euler_deg(frame.devices[f"{side}_controller"].rotation),
                "rotation_xyzw": [
                    round(float(value), 5)
                    for value in frame.devices[f"{side}_controller"].rotation
                ],
                "grip_engaged": bool(controller.grip_click),
                "grip_value": round(float(controller.grip_value), 3),
                "finger_bends": {key: round(float(value), 3) for key, value in controller.finger_bends.items()},
            }
        return pose

    def _motion_awareness(self, now: float) -> Optional[Dict[str, Any]]:
        action = self._action_descriptor(self._current_action)
        if action is None:
            return None
        motion = self._active
        started_at = float(self._current_action.get("started_at_monotonic", now))
        if motion is not None and motion.action_id == action.get("id"):
            duration = motion.duration_s
            elapsed = min(duration, max(0.0, now - motion.started_at))
            remaining = max(0.0, duration - elapsed)
        else:
            params = self._current_action.get("params", {})
            duration = float(params.get("applied_duration_ms", 0.0)) / 1000.0 if isinstance(params, dict) else 0.0
            elapsed = duration if self._state in {"holding", "stopped_latched"} else max(0.0, now - started_at)
            remaining = 0.0
        name = str(action.get("name", "unknown"))
        if name == "semantic_clip":
            source = "semantic_vmd"
        elif name == "play_clip":
            source = "preset_clip"
        elif name == "sequence":
            source = "sequence"
        elif action.get("action_type") == "gesture" or name in {"wave", "nod", "bow"}:
            source = "gesture"
        elif name in {"body_stop", "enable", "disable", "reset", "cancel"}:
            source = "system"
        else:
            source = "pose_command"
        return {
            **action,
            "source": source,
            "phase": self._state,
            "elapsed_seconds": round(elapsed, 3),
            "remaining_seconds": round(remaining, 3),
            "duration_seconds": round(duration, 3),
            "completed": motion is None and self._state in {"holding", "idle", "stopped_latched", "disabled"},
        }

    def _awareness_summary(self, motion: Optional[Dict[str, Any]], pose: Dict[str, Any]) -> str:
        if not self._enabled:
            return "身体输出未启用。"
        if self._state == "fault_latched":
            return "身体输出处于故障锁定，正在发送安全姿态。"
        if self._state == "stopped_latched":
            prefix = "身体已急停并冻结当前姿态，控制器输入已释放。"
        elif motion:
            label = motion.get("clip_name") or motion.get("name")
            if self._state == "moving":
                prefix = f"正在执行“{label}”，进度 {float(motion.get('progress', 0.0)) * 100:.0f}%。"
            elif self._state == "holding":
                prefix = f"“{label}”已到达目标，正在保持该姿态。"
            else:
                prefix = f"当前动作是“{label}”。"
        elif self._expression_overlays:
            label = self._expression_overlays[-1].gesture
            prefix = f"正在执行低强度“{label}”表达动作。"
        elif self._idle_relay_applied:
            prefix = "正在中转 N.E.K.O 宿主的 VMC 待机姿态。"
        else:
            prefix = "身体输出已启用，当前没有进行中的动作。"
        elevation_cn = {"lowered": "下垂", "low": "低位", "horizontal": "水平", "raised": "抬起", "overhead": "举过头顶"}
        direction_cn = {"forward": "向前", "right": "向右", "left": "向左", "backward": "向后"}
        hand_cn = {"open": "张开", "fist": "握拳", "grip": "抓握", "point": "指向", "partially_closed": "半握"}
        parts = []
        for side, side_cn in (("left", "左"), ("right", "右")):
            arm = pose[f"{side}_arm"]
            hand = pose[f"{side}_hand"]
            parts.append(
                f"{side_cn}臂{elevation_cn[arm['elevation']]}{direction_cn[arm['direction']]}，"
                f"{side_cn}手{hand_cn[hand['pose']]}"
            )
        return prefix + " " + "；".join(parts) + "。"

    def _idle_relay_snapshot(self) -> Dict[str, Any]:
        source = self._idle_frame_source
        if source is None:
            status: Dict[str, Any] = {
                "enabled": self.config.vmc_idle.enabled,
                "listen_address": f"{self.config.vmc_idle.listen_host}:{self.config.vmc_idle.listen_port}",
                "connection": "unknown",
                "source_available": False,
            }
        else:
            try:
                status = dict(source.snapshot())
            except Exception as exc:
                status = {
                    "enabled": self.config.vmc_idle.enabled,
                    "listen_address": f"{self.config.vmc_idle.listen_host}:{self.config.vmc_idle.listen_port}",
                    "connection": "error",
                    "source_available": False,
                    "last_error": f"idle VMC status failed: {exc}",
                }
        status["applied"] = self._idle_relay_applied
        if self._idle_relay_error:
            status["frame_error"] = self._idle_relay_error
        return status

    def _build_awareness(self, now: float) -> Dict[str, Any]:
        pose = self._semantic_pose()
        motion = self._motion_awareness(now)
        return {
            "motion": motion,
            "expression_overlays": [
                {
                    "id": overlay.action_id,
                    "gesture": overlay.gesture,
                    "intent": overlay.intent,
                    "source": overlay.source,
                    "side": overlay.side,
                    "energy": round(overlay.energy, 3),
                    "progress": round(overlay.progress(now), 4),
                }
                for overlay in self._expression_overlays
            ],
            "previous_action": copy.deepcopy(self._previous_action),
            "transition": copy.deepcopy(self._transition),
            "behavior": self._behavior.snapshot(runtime_state=self._state, now=now),
            "idle_relay": self._idle_relay_snapshot(),
            "pose": pose,
            "summary": self._awareness_summary(motion, pose),
            "updated_at_monotonic": now,
        }

    def _publish_snapshot(self) -> None:
        now = self._clock()
        actual_hz = 0.0
        if self._send_intervals:
            mean_interval = sum(self._send_intervals) / len(self._send_intervals)
            actual_hz = 1.0 / mean_interval if mean_interval > 0 else 0.0
        snapshot = {
            "state": self._state,
            "safety_state": self._safety_state,
            "output_enabled": self._enabled,
            "udp": {
                "target": f"{self.config.host}:{self.config.port}",
                "connected": "unknown",
                "last_send_at_monotonic": self._last_send_at,
                "sent_packets": self._sent_packets,
                "send_failures": self._send_failures,
            },
            "current_action": copy.deepcopy(self._current_action),
            "awareness": self._build_awareness(now),
            "queue_length": self._commands.qsize() + (1 if self._urgent_stop is not None else 0),
            "arms": copy.deepcopy(self._arm_state),
            "hands": copy.deepcopy(self._hand_state),
            "expression_motion": {
                "active": [
                    {
                        "id": overlay.action_id,
                        "gesture": overlay.gesture,
                        "intent": overlay.intent,
                        "source": overlay.source,
                        "side": overlay.side,
                        "energy": round(overlay.energy, 3),
                        "progress": round(overlay.progress(now), 4),
                        "cancelled": overlay.cancelled_at is not None,
                    }
                    for overlay in self._expression_overlays
                ],
                "last": copy.deepcopy(self._last_expression),
            },
            "idle_relay": self._idle_relay_snapshot(),
            "behavior": self._behavior.snapshot(runtime_state=self._state, now=now),
            "metrics": {
                "actual_hz": round(actual_hz, 2),
                "skipped_frames": self._skipped_frames,
                "last_lateness_ms": round(self._last_lateness_ms, 3),
                "max_lateness_ms": round(self._max_lateness_ms, 3),
            },
            "last_error": self._last_error,
            "error_count": self._error_count,
            "concurrent_sender_detection": "unsupported",
        }
        with self._snapshot_lock:
            self._snapshot = snapshot
