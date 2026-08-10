"""Minimal, dependency-free VRChat OSC transport and state cache."""

from __future__ import annotations

from collections import OrderedDict
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
        self._parameters: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._avatar_id: str | None = None
        self._avatar_changed_at_unix: float | None = None
        self._held_inputs: set[tuple[str, str]] = set()
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

    def send_input(self, action: str, side: str, pressed: bool) -> tuple[bool, str | None]:
        normalized_action = str(action).strip().lower()
        normalized_side = str(side).strip().lower()
        address = _INPUT_ADDRESSES.get((normalized_action, normalized_side))
        if address is None:
            return False, "action/side must identify grab, use, or drop for left or right"
        sent, reason = self._send(address, (1 if pressed else 0,))
        if sent:
            with self._lock:
                key = (normalized_action, normalized_side)
                if pressed:
                    self._held_inputs.add(key)
                else:
                    self._held_inputs.discard(key)
        return sent, reason

    def pulse_input(self, action: str, side: str, hold_ms: int | None = None) -> tuple[bool, str | None]:
        duration_ms = self.config.input_pulse_ms if hold_ms is None else hold_ms
        sent, reason = self.send_input(action, side, True)
        if not sent:
            return False, reason
        self._schedule_input(action, side, False, duration_ms / 1000.0)
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
        duration_ms = self.config.input_pulse_ms if hold_ms is None else hold_ms
        self._schedule_input(action, side, True, max(0.0, delay_s), guard=guard)
        self._schedule_input(action, side, False, max(0.0, delay_s) + duration_ms / 1000.0)
        return True

    def _schedule_input(
        self,
        action: str,
        side: str,
        pressed: bool,
        delay_s: float,
        *,
        guard: Callable[[], bool] | None = None,
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
        )
        with self._lock:
            heapq.heappush(self._scheduled, event)
        self._wake_event.set()

    def cancel_scheduled_inputs(self, *, release: bool) -> None:
        with self._lock:
            self._scheduled.clear()
            held = tuple(self._held_inputs)
        if release:
            # Release known pressed inputs, then send all supported releases so
            # a lost local state update can never leave a VRChat button stuck.
            keys = set(held) | set(_INPUT_ADDRESSES)
            for action, side in sorted(keys):
                self.send_input(action, side, False)
        self._wake_event.set()

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
            self.send_input(event.action, event.side, event.pressed)

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
            self._receive_once()
        with self._lock:
            self._receiver_listening = False

    def snapshot(self, *, include_parameters: bool = True) -> dict[str, Any]:
        now = self._clock()
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
                "scheduled_inputs": len(self._scheduled),
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
            }
            if include_parameters:
                result["parameters"] = dict(self._parameters)
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
        if not self.config.enabled:
            summary = "VRChat OSC 已禁用。"
        elif not detected:
            summary = "尚未收到 VRChat OSC 回传；发送状态不能证明 VRChat 已接收。"
        elif parameters:
            values = "、".join(f"{name}={record['value']}" for name, record in parameters.items())
            summary = f"已收到 VRChat OSC；动作参数：{values}。"
        else:
            summary = "已收到 VRChat OSC，但尚无配置的动作状态参数。"
        return {
            "enabled": self.config.enabled,
            "connection": "detected" if detected else "unknown",
            "avatar_id": avatar_id,
            "parameters": parameters,
            "summary": summary,
            "pose_feedback_available": False,
            "pickup_confirmation_available": False,
        }


__all__ = [
    "MAX_OSC_PACKET_BYTES",
    "OscProtocolError",
    "VrchatOscBridge",
    "decode_osc_packet",
    "encode_osc_message",
    "normalize_parameter_value",
    "validate_parameter_name",
]
