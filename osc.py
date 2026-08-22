"""Minimal, dependency-free VRChat OSC transport and state cache."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
import heapq
import itertools
import math
import socket
import struct
import threading
import time
from typing import Any, Callable, Iterable

from .config import VrchatOscConfig


MAX_OSC_PACKET_BYTES = 65535
_INPUT_ADDRESSES = {
    ("grab", "left"): "/input/GrabLeft",
    ("grab", "right"): "/input/GrabRight",
    ("use", "left"): "/input/UseLeft",
    ("use", "right"): "/input/UseRight",
    ("drop", "left"): "/input/DropLeft",
    ("drop", "right"): "/input/DropRight",
}
_AXIS_ADDRESSES = {
    "move_vertical": "/input/Vertical",
    "move_horizontal": "/input/Horizontal",
    "look_horizontal": "/input/LookHorizontal",
}
_BUTTON_ADDRESSES = {
    "move_forward": "/input/MoveForward",
    "move_backward": "/input/MoveBackward",
    "move_left": "/input/MoveLeft",
    "move_right": "/input/MoveRight",
    "look_left": "/input/LookLeft",
    "look_right": "/input/LookRight",
    "run": "/input/Run",
    "jump": "/input/Jump",
}
_INPUT_HOLD_MIN_MS = 20
_INPUT_HOLD_MAX_MS = 1000

# VRChat 内置 Avatar 参数。名称取自 VRChat 官方 Avatar Parameters 文档，但**本仓库
# 无法验证实机上的实际回传**：built-in 参数只有在 avatar 的参数列表里存在时才会被
# VRChat 驱动并经 OSC 发回 9001。因此所有读取路径都必须能接受「一个都收不到」，
# 返回 available=false 而不是猜一个速度出来。
_VELOCITY_AXIS_PARAMETERS = ("VelocityX", "VelocityY", "VelocityZ")
BUILTIN_MOTION_PARAMETERS = _VELOCITY_AXIS_PARAMETERS + ("Upright", "Grounded", "AngularY")

# 「最后一次回传说自己是静止的」的判定阈值。低于它时链路沉默与取值不矛盾
# （没变化自然没有包），可用性保持；高于它时沉默与取值矛盾，说明链路是在运动中
# 断掉的。上界由实测卡墙速度 0.08 m/s 定死：顶着墙推摇杆必须仍算「在动」，否则
# 撞墙时若速度恒定不发包，就会被读成「静止且可信」。真正的静止恒为 0，所以只需
# 一个能盖住浮点噪声、又明显低于 0.08 的值。
_RESTING_SPEED_MPS = 0.05


def _numeric(value: Any) -> float | None:
    """把 OSC 标量读成有限浮点；bool 不算数值（Grounded 单独处理）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


class OscProtocolError(ValueError):
    """Raised when an OSC packet is malformed or unsupported."""


def _padded(raw: bytes) -> bytes:
    return raw + (b"\x00" * ((-len(raw)) % 4))


def _osc_string(value: str) -> bytes:
    if "\x00" in value:
        raise OscProtocolError("OSC strings cannot contain NUL")
    return _padded(value.encode("utf-8") + b"\x00")


def _validate_address(address: str) -> str:
    if not isinstance(address, str) or not address.startswith("/") or "\x00" in address:
        raise OscProtocolError("OSC address must start with '/' and contain no NUL")
    if len(address.encode("utf-8")) > 1024:
        raise OscProtocolError("OSC address is too long")
    return address


def validate_parameter_name(name: Any) -> str:
    normalized = str(name or "").strip()
    if (
        not normalized
        or len(normalized) > 128
        or "/" in normalized
        or "\x00" in normalized
        or any(char in normalized for char in "*?[]{}")
    ):
        raise ValueError("parameter name must be 1-128 characters without '/', NUL, or OSC wildcards")
    return normalized


def normalize_parameter_value(value: Any) -> bool | int | float:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not -(2**31) <= value < 2**31:
            raise ValueError("integer avatar parameter must fit signed 32-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("float avatar parameter must be finite")
        return value
    raise ValueError("avatar parameter value must be a boolean, integer, or finite float")


def encode_osc_message(address: str, arguments: Iterable[Any] = ()) -> bytes:
    """Encode one OSC 1.0 message using only types needed by VRChat."""
    _validate_address(address)
    tags: list[str] = []
    payload: list[bytes] = []
    for value in arguments:
        if isinstance(value, bool):
            tags.append("T" if value else "F")
        elif isinstance(value, int):
            if not -(2**31) <= value < 2**31:
                raise OscProtocolError("OSC int must fit signed 32-bit range")
            tags.append("i")
            payload.append(struct.pack(">i", value))
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise OscProtocolError("OSC float must be finite")
            tags.append("f")
            payload.append(struct.pack(">f", value))
        elif isinstance(value, str):
            tags.append("s")
            payload.append(_osc_string(value))
        elif isinstance(value, (bytes, bytearray, memoryview)):
            blob = bytes(value)
            tags.append("b")
            payload.append(struct.pack(">i", len(blob)) + _padded(blob))
        elif value is None:
            tags.append("N")
        else:
            raise OscProtocolError(f"unsupported OSC argument type: {type(value).__name__}")
    packet = _osc_string(address) + _osc_string("," + "".join(tags)) + b"".join(payload)
    if len(packet) > MAX_OSC_PACKET_BYTES:
        raise OscProtocolError("OSC packet exceeds UDP payload limit")
    return packet


def _read_string(packet: bytes, offset: int) -> tuple[str, int]:
    if offset < 0 or offset >= len(packet):
        raise OscProtocolError("truncated OSC string")
    end = packet.find(b"\x00", offset)
    if end < 0:
        raise OscProtocolError("unterminated OSC string")
    next_offset = (end + 4) & ~3
    if next_offset > len(packet):
        raise OscProtocolError("truncated OSC string padding")
    try:
        value = packet[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OscProtocolError("OSC string is not valid UTF-8") from exc
    return value, next_offset


def _unpack(packet: bytes, offset: int, fmt: str, size: int) -> tuple[Any, int]:
    if offset + size > len(packet):
        raise OscProtocolError("truncated OSC argument")
    return struct.unpack_from(fmt, packet, offset)[0], offset + size


def _decode_message(packet: bytes) -> tuple[str, tuple[Any, ...]]:
    address, offset = _read_string(packet, 0)
    _validate_address(address)
    tags, offset = _read_string(packet, offset)
    if not tags.startswith(","):
        raise OscProtocolError("OSC type tag string must start with ','")
    arguments: list[Any] = []
    for tag in tags[1:]:
        if tag == "i":
            value, offset = _unpack(packet, offset, ">i", 4)
        elif tag == "f":
            value, offset = _unpack(packet, offset, ">f", 4)
            if not math.isfinite(value):
                raise OscProtocolError("OSC float must be finite")
        elif tag == "s":
            value, offset = _read_string(packet, offset)
        elif tag == "b":
            length, offset = _unpack(packet, offset, ">i", 4)
            if length < 0 or offset + length > len(packet):
                raise OscProtocolError("invalid OSC blob length")
            value = packet[offset:offset + length]
            offset = (offset + length + 3) & ~3
            if offset > len(packet):
                raise OscProtocolError("truncated OSC blob padding")
        elif tag == "T":
            value = True
        elif tag == "F":
            value = False
        elif tag == "N":
            value = None
        elif tag == "h":
            value, offset = _unpack(packet, offset, ">q", 8)
        elif tag == "d":
            value, offset = _unpack(packet, offset, ">d", 8)
            if not math.isfinite(value):
                raise OscProtocolError("OSC double must be finite")
        else:
            raise OscProtocolError(f"unsupported OSC type tag: {tag!r}")
        arguments.append(value)
    if any(byte != 0 for byte in packet[offset:]):
        raise OscProtocolError("unexpected trailing OSC data")
    return address, tuple(arguments)


def decode_osc_packet(packet: bytes, *, _depth: int = 0) -> list[tuple[str, tuple[Any, ...]]]:
    """Decode an OSC message or bundle into a flat list of messages."""
    if not isinstance(packet, bytes) or not packet or len(packet) > MAX_OSC_PACKET_BYTES:
        raise OscProtocolError("invalid OSC packet size")
    if _depth > 8:
        raise OscProtocolError("OSC bundle nesting is too deep")
    if not packet.startswith(b"#bundle\x00"):
        return [_decode_message(packet)]
    if len(packet) < 16:
        raise OscProtocolError("truncated OSC bundle")
    offset = 16  # '#bundle\0' plus the 64-bit timetag.
    messages: list[tuple[str, tuple[Any, ...]]] = []
    while offset < len(packet):
        length, offset = _unpack(packet, offset, ">i", 4)
        if length <= 0 or offset + length > len(packet):
            raise OscProtocolError("invalid OSC bundle element length")
        messages.extend(decode_osc_packet(packet[offset:offset + length], _depth=_depth + 1))
        offset += length
    return messages


@dataclass(order=True)
class _ScheduledInput:
    deadline: float
    sequence: int
    action: str = field(compare=False)
    side: str = field(compare=False)
    pressed: bool = field(compare=False)
    guard: Callable[[], bool] | None = field(compare=False, default=None)
    pulse_id: int | None = field(compare=False, default=None)


class VrchatOscBridge:
    """Own the OSC sockets, cache avatar feedback, and schedule safe input pulses."""

    def __init__(
        self,
        config: VrchatOscConfig,
        *,
        logger: Any = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.logger = logger
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._send_socket: socket.socket | None = None
        self._receive_socket: socket.socket | None = None
        self._scheduled: list[_ScheduledInput] = []
        self._schedule_sequence = itertools.count()
        self._pulse_sequence = itertools.count()
        self._started_pulses: set[int] = set()
        self._parameters: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._avatar_id: str | None = None
        self._avatar_changed_at_unix: float | None = None
        self._held_inputs: set[tuple[str, str]] = set()
        self._held_buttons: set[str] = set()
        # {axis: (value, expires_at, generation)}.  The generation prevents
        # an old expiry callback from zeroing a newer command for the same axis.
        self._active_axes: dict[str, tuple[float, float, int]] = {}
        self._axis_generation = itertools.count()
        self._receiver_listening = False
        self._sent_packets = 0
        self._send_failures = 0
        self._received_packets = 0
        self._decoded_messages = 0
        self._rejected_packets = 0
        self._last_send_at_unix: float | None = None
        self._last_receive_at_unix: float | None = None
        self._last_receive_at_monotonic: float | None = None
        self._last_error: str | None = None

    @property
    def thread_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if not self.config.enabled or self.thread_alive:
            return
        self._stop_event.clear()
        self._wake_event.clear()
        try:
            self._send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._send_socket.setblocking(False)
        except OSError as exc:
            self._record_error(f"could not create OSC send socket: {exc}")
            return
        try:
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            receiver.bind((self.config.listen_host, self.config.listen_port))
            receiver.settimeout(0.02)
            self._receive_socket = receiver
            with self._lock:
                self._receiver_listening = True
        except OSError as exc:
            self._receive_socket = None
            self._record_error(
                f"could not listen on {self.config.listen_host}:{self.config.listen_port}: {exc}"
            )
        self._thread = threading.Thread(target=self._run, name="neko-vrchat-osc", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self.config.enabled and self._send_socket is not None:
            self.cancel_scheduled_inputs(release=True)
        self._stop_event.set()
        self._wake_event.set()
        receiver = self._receive_socket
        if receiver is not None:
            try:
                receiver.close()
            except OSError:
                pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout)
        sender = self._send_socket
        if sender is not None:
            try:
                sender.close()
            except OSError:
                pass
        with self._lock:
            self._receiver_listening = False
        self._receive_socket = None
        self._send_socket = None
        self._thread = None

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message[:500]
        if self.logger:
            self.logger.warning("VRChat OSC: %s", message)

    def _send(self, address: str, arguments: Iterable[Any]) -> tuple[bool, str | None]:
        if not self.config.enabled:
            return False, "VRChat OSC is disabled"
        sender = self._send_socket
        if sender is None:
            return False, "VRChat OSC sender is not initialized"
        try:
            packet = encode_osc_message(address, arguments)
            sender.sendto(packet, (self.config.send_host, self.config.send_port))
        except (OSError, OscProtocolError) as exc:
            with self._lock:
                self._send_failures += 1
            self._record_error(str(exc))
            return False, str(exc)
        with self._lock:
            self._sent_packets += 1
            self._last_send_at_unix = self._wall_clock()
        return True, None

    def send_parameter(self, name: Any, value: Any) -> tuple[bool, str | None]:
        parameter = validate_parameter_name(name)
        normalized = normalize_parameter_value(value)
        return self._send(f"/avatar/parameters/{parameter}", (normalized,))

    def send_chatbox(self, text: Any, *, immediate: bool = True) -> tuple[bool, str | None]:
        """通过 /chatbox/input 发送聊天框文本。"""
        if not isinstance(text, str):
            return False, "text must be a string"
        if not isinstance(immediate, bool):
            return False, "immediate must be a boolean"
        message = text.replace("\x00", "").strip()
        if not message or len(message) > 144:
            return False, "text must be between 1 and 144 characters"
        return self._send("/chatbox/input", (message, immediate, False))

    def send_input(self, action: str, side: str, pressed: bool) -> tuple[bool, str | None]:
        if not isinstance(pressed, bool):
            return False, "pressed must be a boolean"
        normalized_action = str(action).strip().lower()
        normalized_side = str(side).strip().lower()
        address = _INPUT_ADDRESSES.get((normalized_action, normalized_side))
        if address is None:
            return False, "action/side must identify grab, use, or drop for left or right"
        with self._lock:
            sent, reason = self._send(address, (1 if pressed else 0,))
            if sent:
                key = (normalized_action, normalized_side)
                if pressed:
                    self._held_inputs.add(key)
                else:
                    self._held_inputs.discard(key)
        return sent, reason

    def send_button(self, button: str, pressed: bool) -> tuple[bool, str | None]:
        """Send one of VRChat's button-style input addresses.

        Button inputs require OSC integers (1/0), whereas movement axes use
        floats.  Keeping this path separate prevents callers from accidentally
        sending a float to e.g. ``/input/Jump``.
        """
        if not isinstance(pressed, bool):
            return False, "pressed must be a boolean"
        normalized = str(button).strip().lower().replace("-", "_")
        address = _BUTTON_ADDRESSES.get(normalized)
        if address is None:
            return False, f"unknown button: {button}"
        with self._lock:
            sent, reason = self._send(address, (1 if pressed else 0,))
            if sent:
                if pressed:
                    self._held_buttons.add(normalized)
                else:
                    self._held_buttons.discard(normalized)
        return sent, reason

    @staticmethod
    def _normalize_hold_ms(value: Any, default: int) -> int | None:
        if value is None:
            value = default
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(numeric)
            or not numeric.is_integer()
            or not _INPUT_HOLD_MIN_MS <= numeric <= _INPUT_HOLD_MAX_MS
        ):
            return None
        return int(numeric)

    def pulse_input(self, action: str, side: str, hold_ms: int | None = None) -> tuple[bool, str | None]:
        duration_ms = self._normalize_hold_ms(hold_ms, self.config.input_pulse_ms)
        if duration_ms is None:
            return False, f"hold_ms must be an integer between {_INPUT_HOLD_MIN_MS} and {_INPUT_HOLD_MAX_MS}"
        sent, reason = self.send_input(action, side, True)
        if not sent:
            return False, reason
        try:
            self._schedule_input(action, side, False, duration_ms / 1000.0)
        except (TypeError, ValueError) as exc:
            # Do not leave a pressed VRChat input behind if scheduling fails.
            self.send_input(action, side, False)
            return False, str(exc)
        return True, None

    def schedule_input_pulse(
        self,
        action: str,
        side: str,
        *,
        delay_s: float,
        hold_ms: int | None = None,
        guard: Callable[[], bool] | None = None,
    ) -> bool:
        if not self.config.enabled or self._send_socket is None:
            return False
        if isinstance(delay_s, bool):
            return False
        try:
            delay = float(delay_s)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(delay) or delay < 0.0:
            return False
        duration_ms = self._normalize_hold_ms(hold_ms, self.config.input_pulse_ms)
        if duration_ms is None:
            return False
        pulse_id = next(self._pulse_sequence)
        self._schedule_input(action, side, True, delay, guard=guard, pulse_id=pulse_id)
        self._schedule_input(
            action,
            side,
            False,
            delay + duration_ms / 1000.0,
            pulse_id=pulse_id,
        )
        return True

    def _schedule_input(
        self,
        action: str,
        side: str,
        pressed: bool,
        delay_s: float,
        *,
        guard: Callable[[], bool] | None = None,
        pulse_id: int | None = None,
    ) -> None:
        normalized_action = str(action).strip().lower()
        normalized_side = str(side).strip().lower()
        if (normalized_action, normalized_side) not in _INPUT_ADDRESSES:
            raise ValueError("invalid VRChat OSC input")
        event = _ScheduledInput(
            self._clock() + max(0.0, delay_s),
            next(self._schedule_sequence),
            normalized_action,
            normalized_side,
            bool(pressed),
            guard,
            pulse_id,
        )
        with self._lock:
            heapq.heappush(self._scheduled, event)
        self._wake_event.set()

    def cancel_scheduled_inputs(self, *, release: bool) -> None:
        with self._lock:
            self._scheduled.clear()
            self._started_pulses.clear()
            held = tuple(self._held_inputs)
            held_buttons = tuple(self._held_buttons)
        if release:
            # Release known pressed inputs, then send all supported releases so
            # a lost local state update can never leave a VRChat button stuck.
            keys = set(held) | set(_INPUT_ADDRESSES)
            for action, side in sorted(keys):
                self.send_input(action, side, False)
            # Release button-style movement inputs too.  This is intentionally
            # separate from axis zeroing because VRChat expects integer 0 for
            # buttons and float 0.0 for axes.
            buttons = set(held_buttons) | set(_BUTTON_ADDRESSES)
            for button in sorted(buttons):
                self.send_button(button, False)
            self.stop_all_axes()
        self._wake_event.set()

    @staticmethod
    def _normalize_axis_value(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
            raise ValueError(f"{name} must be between -1 and 1")
        return numeric

    @staticmethod
    def _normalize_axis_name(axis: Any) -> str:
        normalized = str(axis).strip().lower().replace("-", "_")
        if normalized not in _AXIS_ADDRESSES:
            raise ValueError(f"unknown axis: {axis}")
        return normalized

    def set_axes(
        self,
        values: Mapping[str, Any],
        duration_s: float = 0.0,
    ) -> tuple[bool, str | None]:
        """Set several float axes as one serialized command.

        Validation, packets, active-state bookkeeping, and rollback all happen
        under one lock. This prevents an emergency stop from landing between
        the two packets of a diagonal locomotion command.
        """
        if not isinstance(values, Mapping) or not values:
            return False, "axis values must be a non-empty object"
        if isinstance(duration_s, bool):
            return False, "axis duration must be a finite non-negative number"
        try:
            duration = float(duration_s)
        except (TypeError, ValueError, OverflowError):
            return False, "axis duration must be a finite non-negative number"
        if not math.isfinite(duration) or duration < 0.0:
            return False, "axis duration must be finite and non-negative"

        normalized: list[tuple[str, float]] = []
        try:
            for raw_axis, raw_value in values.items():
                axis = self._normalize_axis_name(raw_axis)
                numeric = self._normalize_axis_value(raw_value, "axis value")
                if duration == 0.0 and numeric != 0.0:
                    return False, "a non-zero axis value requires a positive duration"
                normalized.append((axis, numeric))
        except ValueError as exc:
            return False, str(exc)

        failures: list[str] = []
        with self._lock:
            for axis, numeric in normalized:
                sent, reason = self._send(_AXIS_ADDRESSES[axis], (numeric,))
                if not sent:
                    failures.append(f"{axis}: {reason or 'send failed'}")
                    break

            if failures:
                # Release every requested axis, including the one whose
                # non-zero packet failed, and retain bookkeeping when a zero
                # packet itself cannot be sent.
                for axis in dict.fromkeys(item[0] for item in normalized):
                    released, release_reason = self._send(_AXIS_ADDRESSES[axis], (0.0,))
                    if released:
                        self._active_axes.pop(axis, None)
                    else:
                        failures.append(f"{axis} rollback: {release_reason or 'send failed'}")
            else:
                expires_at = self._clock() + duration
                for axis, numeric in normalized:
                    if duration > 0.0:
                        self._active_axes[axis] = (
                            numeric,
                            expires_at,
                            next(self._axis_generation),
                        )
                    else:
                        self._active_axes.pop(axis, None)

        self._wake_event.set()
        if failures:
            return False, "; ".join(failures)[:500]
        return True, None

    def set_axis(self, axis: str, value: float, duration_s: float = 0.0) -> tuple[bool, str | None]:
        """持续设置一个移动/转向轴。"""
        return self.set_axes({axis: value}, duration_s)

    def stop_axes(self, axes: Iterable[str] | None = None) -> tuple[bool, str | None]:
        """立即归零指定的轴；没有指定时归零全部已知轴。

        Zero packets are sent while holding the state lock so a concurrent
        command cannot be inserted between state clearing and the safety
        release. Failed releases for active axes remain tracked and are
        retried by the expiry loop or a subsequent stop call.
        """
        if axes is None:
            requested = list(_AXIS_ADDRESSES)
        else:
            requested = []
            for raw_axis in axes:
                normalized = str(raw_axis).strip().lower().replace("-", "_")
                if normalized not in _AXIS_ADDRESSES:
                    return False, f"unknown axis: {raw_axis}"
                if normalized not in requested:
                    requested.append(normalized)
        failures: list[str] = []
        with self._lock:
            for axis in requested:
                address = _AXIS_ADDRESSES[axis]
                sent, reason = self._send(address, (0.0,))
                if sent:
                    self._active_axes.pop(axis, None)
                else:
                    failures.append(f"{axis}: {reason or 'send failed'}")
        self._wake_event.set()
        if failures:
            return False, "; ".join(failures)[:500]
        return True, None

    def stop_all_axes(self) -> tuple[bool, str | None]:
        """立即归零所有已知移动轴."""
        return self.stop_axes()

    def _run_axis_expirations(self) -> None:
        """检查并归零过期的轴。"""
        now = self._clock()
        expired: list[tuple[str, int]] = []
        with self._lock:
            for axis, (_value, expires_at, generation) in list(self._active_axes.items()):
                if now >= expires_at:
                    expired.append((axis, generation))
            # Keep the lock while sending the release.  This makes the
            # generation check and the physical zero packet one atomic action
            # relative to set_axis().
            for axis, generation in expired:
                current = self._active_axes.get(axis)
                if current is None or current[2] != generation:
                    continue
                sent, _reason = self._send(_AXIS_ADDRESSES[axis], (0.0,))
                if sent:
                    self._active_axes.pop(axis, None)

    def _run_due_inputs(self) -> None:
        now = self._clock()
        due: list[_ScheduledInput] = []
        with self._lock:
            while self._scheduled and self._scheduled[0].deadline <= now:
                due.append(heapq.heappop(self._scheduled))
        for event in due:
            if event.guard is not None:
                try:
                    if not event.guard():
                        continue
                except Exception as exc:
                    self._record_error(f"scheduled input guard failed: {exc}")
                    continue
            if event.pulse_id is not None and not event.pressed:
                with self._lock:
                    if event.pulse_id not in self._started_pulses:
                        continue
            sent, _ = self.send_input(event.action, event.side, event.pressed)
            if event.pulse_id is not None:
                with self._lock:
                    if event.pressed:
                        if sent:
                            self._started_pulses.add(event.pulse_id)
                    else:
                        self._started_pulses.discard(event.pulse_id)

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if isinstance(value, bytes):
            return {"blob_bytes": len(value)}
        return value

    def _handle_message(self, address: str, arguments: tuple[Any, ...]) -> None:
        now_wall = self._wall_clock()
        if address == "/avatar/change" and arguments:
            avatar_id = str(arguments[0])[:256]
            with self._lock:
                if avatar_id != self._avatar_id:
                    self._parameters.clear()
                self._avatar_id = avatar_id
                self._avatar_changed_at_unix = now_wall
            return
        prefix = "/avatar/parameters/"
        if address.startswith(prefix) and arguments:
            name = address[len(prefix):]
            if not name or len(name) > 256:
                return
            record = {"value": self._safe_value(arguments[0]), "received_at_unix": now_wall}
            with self._lock:
                self._parameters.pop(name, None)
                self._parameters[name] = record
                while len(self._parameters) > self.config.parameter_cache_size:
                    self._parameters.popitem(last=False)

    def _receive_once(self) -> None:
        receiver = self._receive_socket
        if receiver is None:
            self._wake_event.wait(0.02)
            self._wake_event.clear()
            return
        try:
            packet, sender = receiver.recvfrom(MAX_OSC_PACKET_BYTES)
        except socket.timeout:
            return
        except OSError as exc:
            if not self._stop_event.is_set():
                self._record_error(f"OSC receive failed: {exc}")
            return
        if self.config.allowed_sender and sender[0] != self.config.allowed_sender:
            with self._lock:
                self._rejected_packets += 1
            return
        try:
            messages = decode_osc_packet(packet)
        except OscProtocolError as exc:
            with self._lock:
                self._rejected_packets += 1
            self._record_error(f"rejected OSC packet: {exc}")
            return
        now_mono = self._clock()
        now_wall = self._wall_clock()
        with self._lock:
            self._received_packets += 1
            self._decoded_messages += len(messages)
            self._last_receive_at_monotonic = now_mono
            self._last_receive_at_unix = now_wall
            self._last_error = None
        for address, arguments in messages:
            self._handle_message(address, arguments)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._run_due_inputs()
            self._run_axis_expirations()
            self._receive_once()
        with self._lock:
            self._receiver_listening = False

    def snapshot(self, *, include_parameters: bool = True) -> dict[str, Any]:
        now = self._clock()
        # 诊断面板要能在未 arm 时也看到「VRChat 有没有在回传移动」——导航器的
        # last_motion 只在 tick 里刷新，授权解除后就冻住了。
        motion = self.motion_feedback()
        with self._lock:
            age_ms = None
            if self._last_receive_at_monotonic is not None:
                age_ms = max(0.0, (now - self._last_receive_at_monotonic) * 1000.0)
            result: dict[str, Any] = {
                "enabled": self.config.enabled,
                "send_target": f"{self.config.send_host}:{self.config.send_port}",
                "listen_address": f"{self.config.listen_host}:{self.config.listen_port}",
                "receiver_listening": self._receiver_listening,
                "thread_alive": self.thread_alive,
                "connection": "detected" if self._last_receive_at_monotonic is not None else "unknown",
                "avatar_id": self._avatar_id,
                "avatar_changed_at_unix": self._avatar_changed_at_unix,
                "parameter_count": len(self._parameters),
                "held_inputs": [f"{action}_{side}" for action, side in sorted(self._held_inputs)],
                "held_buttons": sorted(self._held_buttons),
                "scheduled_inputs": len(self._scheduled),
                "active_axes": {
                    axis: {
                        "value": value,
                        "remaining_ms": max(0.0, (expires_at - now) * 1000.0),
                    }
                    for axis, (value, expires_at, _generation) in sorted(self._active_axes.items())
                },
                "sent_packets": self._sent_packets,
                "send_failures": self._send_failures,
                "received_packets": self._received_packets,
                "decoded_messages": self._decoded_messages,
                "rejected_packets": self._rejected_packets,
                "last_send_at_unix": self._last_send_at_unix,
                "last_receive_at_unix": self._last_receive_at_unix,
                "last_receive_age_ms": age_ms,
                "last_error": self._last_error,
                "delivery_confirmation": "unavailable",
                "motion": motion,
            }
            if include_parameters:
                result["parameters"] = dict(self._parameters)
            return result

    def motion_feedback(self, *, max_age_ms: int = 2000) -> dict[str, Any]:
        """从 VRChat 内置 Avatar 参数读取实际移动反馈。

        这是全仓库唯一一条「VRChat 说我动了没有」的通路：AnyaDance 与 OSC 都只能
        确认本机发送成功，`accepted=true` 从来不代表角色真的移动了。有了它，导航
        器才能区分「正在前进」和「顶着墙推摇杆」。

        两个必须记住的前提：

        1. **参数名未经实机验证。** 内置参数只有在 avatar 的参数列表里存在时才会
           被驱动并回传；名字对不上或 avatar 没配，就一个都收不到。此时返回
           ``available=false`` 并给出 ``expected``，绝不猜一个速度出来。
        2. **静止时的沉默要靠取值自证，链路年龄单独判不出来。** VRChat 的参数是
           变化驱动的：站着不动时速度恒为 0，于是**一个包都不会来**，链路年龄和
           取值年龄一起变老。所以「安静」和「断了」无法只靠入站流量区分，得看
           最后那个取值跟沉默是否自相矛盾：

           - 最后报的是 ~0 速度 → 沉默正是「没变化」的证据，可用性保持，速度
             继续读 0。这是站住不动时唯一正确的读法，也是唯一能确认「我没在动」
             的时刻，不能丢。
           - 最后报的是在动 → 沉默与之矛盾：真在移动就会持续有速度更新。持续
             沉默说明链路是在运动中断掉的，此时把最后已知速度当现值就是撒谎，
             按 ``feedback_stale`` 判不可用。

           取值年龄始终单独报告为 ``value_age_ms``，不参与可用性判定。
        """
        try:
            limit_ms = max(0, int(max_age_ms))
        except (TypeError, ValueError, OverflowError):
            limit_ms = 2000
        now_mono = self._clock()
        now_wall = self._wall_clock()
        with self._lock:
            link_age_ms = (
                None
                if self._last_receive_at_monotonic is None
                else max(0.0, (now_mono - self._last_receive_at_monotonic) * 1000.0)
            )
            records = {
                name: dict(self._parameters[name])
                for name in BUILTIN_MOTION_PARAMETERS
                if name in self._parameters
            }
        result: dict[str, Any] = {
            "available": False,
            "reason": None,
            "expected": list(BUILTIN_MOTION_PARAMETERS),
            "present": sorted(records),
            "link_age_ms": None if link_age_ms is None else round(link_age_ms, 1),
            "value_age_ms": None,
            "speed_mps": None,
            "horizontal_speed_mps": None,
            "vertical_speed_mps": None,
            "angular_speed": None,
            "grounded": None,
            "upright": None,
        }
        if not self.config.enabled:
            result["reason"] = "osc_disabled"
            return result
        if link_age_ms is None:
            result["reason"] = "no_feedback_received"
            return result

        axes: dict[str, float] = {}
        newest_received: float | None = None
        for name in _VELOCITY_AXIS_PARAMETERS:
            record = records.get(name)
            if record is None:
                continue
            value = _numeric(record.get("value"))
            if value is None:
                continue
            axes[name] = value
            received = _numeric(record.get("received_at_unix"))
            if received is not None and (newest_received is None or received > newest_received):
                newest_received = received
        if not axes:
            # 参数名可能不对，也可能 avatar 根本没配。两者都不是「速度为零」。
            result["reason"] = "velocity_parameters_absent"
            return result

        vx = axes.get("VelocityX", 0.0)
        vy = axes.get("VelocityY", 0.0)
        vz = axes.get("VelocityZ", 0.0)
        horizontal = math.hypot(vx, vz)
        speed = math.sqrt(horizontal * horizontal + vy * vy)

        # 链路安静下来时，唯一能判断「安静」还是「断了」的证据就是最后那个取值：
        # 静止时没有变化所以本就该没有包，在动时却必须持续有速度更新。用平移速度
        # 判定，不看 AngularY——原地匀速转身时 AngularY 恒定同样不发包，而那时
        # 平移速度确实是 0，把它算成「在动」会白丢一次有效读数。
        if limit_ms and link_age_ms > limit_ms and speed > _RESTING_SPEED_MPS:
            result["reason"] = "feedback_stale"
            return result

        result["available"] = True
        result["value_age_ms"] = (
            None if newest_received is None else round(max(0.0, (now_wall - newest_received) * 1000.0), 1)
        )
        result["horizontal_speed_mps"] = round(horizontal, 4)
        result["vertical_speed_mps"] = round(vy, 4)
        result["speed_mps"] = round(speed, 4)
        if limit_ms and link_age_ms > limit_ms:
            # 可用，但要让面板看出这是「靠静止取值兜住的沉默」而不是活跃回传。
            result["reason"] = "link_quiet_at_rest"

        angular = _numeric((records.get("AngularY") or {}).get("value"))
        if angular is not None:
            result["angular_speed"] = round(angular, 4)
        upright = _numeric((records.get("Upright") or {}).get("value"))
        if upright is not None:
            result["upright"] = round(upright, 4)
        grounded = (records.get("Grounded") or {}).get("value")
        if isinstance(grounded, bool):
            result["grounded"] = grounded
        return result

    def awareness(self) -> dict[str, Any]:
        with self._lock:
            parameters = {
                name: dict(self._parameters[name])
                for name in self.config.awareness_parameters
                if name in self._parameters
            }
            detected = self._last_receive_at_monotonic is not None
            avatar_id = self._avatar_id
        motion = self.motion_feedback()
        # 内置移动参数已经由 ``motion`` 汇总；再在摘要里逐个列出来只会把真正需要
        # 人读的动作状态参数淹掉。
        summary_parameters = {
            name: record
            for name, record in parameters.items()
            if name not in BUILTIN_MOTION_PARAMETERS
        }
        if not self.config.enabled:
            summary = "VRChat OSC 已禁用。"
        elif not detected:
            summary = "尚未收到 VRChat OSC 回传；发送状态不能证明 VRChat 已接收。"
        elif summary_parameters:
            values = "、".join(f"{name}={record['value']}" for name, record in summary_parameters.items())
            summary = f"已收到 VRChat OSC；动作参数：{values}。"
        else:
            summary = "已收到 VRChat OSC，但尚无配置的动作状态参数。"
        if motion["available"]:
            summary += f"实测水平速度 {motion['horizontal_speed_mps']} m/s。"
        elif detected and self.config.enabled:
            summary += "VRChat 未回传内置移动参数，无法确认自己是否真的在移动。"
        return {
            "enabled": self.config.enabled,
            "connection": "detected" if detected else "unknown",
            "avatar_id": avatar_id,
            "parameters": parameters,
            "motion": motion,
            "summary": summary,
            "pose_feedback_available": False,
            "pickup_confirmation_available": False,
        }


__all__ = [
    "BUILTIN_MOTION_PARAMETERS",
    "MAX_OSC_PACKET_BYTES",
    "OscProtocolError",
    "VrchatOscBridge",
    "decode_osc_packet",
    "encode_osc_message",
    "normalize_parameter_value",
    "validate_parameter_name",
]
