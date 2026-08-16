"""Subscribe to AnyaDance's driver telemetry multicast group.

The driver multicasts one JSON datagram per event on a loopback-only group with
TTL 0, whether or not anyone is listening.  Subscribing is the only way to learn
what the driver actually did with a pose command: the pose protocol itself has
no response, so a successful ``sendto`` proves nothing beyond the local socket.

The event schema contract is published in AnyaDance's ``docs/protocol.md``.  Two
rules from it shape this module: unknown event names and unknown fields must be
ignored rather than treated as errors, and ordering, deduplication, and loss
detection must use ``sequence`` alone -- never the wall clock or arrival order.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import math
import socket
import struct
import threading
import time
from typing import Any, Callable

from .config import DriverLogConfig

MAX_DRIVER_LOG_PACKET_BYTES = 65507
DRIVER_LOG_PROTOCOL_VERSION = 1
MAX_PAYLOAD_CHARS = 2048
MAX_DETAIL_CHARS = 500
_MAX_TRACKED_SENDERS = 16


def _reject_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant {value}")


def _text(value: Any, limit: int) -> str:
    return str(value)[:limit] if isinstance(value, (str, int, float)) else ""


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    result = float(value)
    return result if math.isfinite(result) else 0.0


def _whole(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _device_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item[:64] for item in value if isinstance(item, str)][:16]


def parse_driver_log_event(payload: bytes) -> dict[str, Any] | None:
    """Decode one telemetry datagram, or return None if it is unusable.

    A well-formed event of an unrecognized type decodes with ``type`` set to
    ``"unknown"``; the group is designed to grow, so skipping is the contract.
    """
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        return None
    if len(payload) > MAX_DRIVER_LOG_PACKET_BYTES:
        return None
    try:
        root = json.loads(bytes(payload).decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(root, dict):
        return None
    if root.get("version") != DRIVER_LOG_PROTOCOL_VERSION:
        return None

    sequence = root.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        return None
    name = root.get("event")
    if not isinstance(name, str) or not name:
        return None

    event: dict[str, Any] = {
        "type": "unknown",
        "event": name[:64],
        "sequence": sequence,
        "timestamp_ms": _whole(root.get("timestamp_ms")),
        "suppressed": _whole(root.get("suppressed")),
        "detail": _text(root.get("detail"), MAX_DETAIL_CHARS),
    }

    if name == "command_processed":
        source = root.get("source") if isinstance(root.get("source"), dict) else {}
        command = root.get("command") if isinstance(root.get("command"), dict) else {}
        port = _whole(source.get("port"))
        event["type"] = "command_processed"
        event["source"] = {
            "host": _text(source.get("host"), 64),
            "port": port if 0 <= port <= 65535 else 0,
        }
        event["command"] = {
            "protocol": _text(command.get("protocol"), 32),
            "bytes": _whole(command.get("bytes")),
            "accepted": command.get("accepted") is True,
            "devices": _device_names(command.get("devices")),
            "y_clamped": _device_names(command.get("y_clamped")),
            # The original datagram is up to 8 KiB; only a prefix is retained so
            # a status snapshot stays small enough to hand to a language model.
            "payload": _text(command.get("payload"), MAX_PAYLOAD_CHARS),
        }
    elif name == "haptic_vibration":
        haptic = root.get("haptic") if isinstance(root.get("haptic"), dict) else {}
        event["type"] = "haptic_vibration"
        event["device"] = _text(root.get("device"), 32)
        event["haptic"] = {
            "duration_seconds": _finite(haptic.get("duration_seconds")),
            "frequency_hz": _finite(haptic.get("frequency_hz")),
            "amplitude": _finite(haptic.get("amplitude")),
        }
    return event


class DriverLogListener:
    """Join the driver log group and summarize what the driver reports."""

    def __init__(
        self,
        config: DriverLogConfig,
        *,
        logger: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.logger = logger
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._receiver_listening = False
        self._received_packets = 0
        self._decoded_events = 0
        self._rejected_packets = 0
        self._unknown_events = 0
        self._duplicate_events = 0
        self._lost_events = 0
        self._accepted_commands = 0
        self._rejected_commands = 0
        self._y_clamped_commands = 0
        self._seen_sequences: OrderedDict[int, None] = OrderedDict()
        self._next_sequence: int | None = None
        self._last_command: dict[str, Any] | None = None
        self._last_haptic: dict[str, Any] | None = None
        self._last_command_at: float | None = None
        self._senders: OrderedDict[str, int] = OrderedDict()
        self._last_error: str | None = None

    @property
    def thread_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _membership(self) -> bytes:
        return struct.pack(
            "=4s4s",
            socket.inet_aton(self.config.multicast_group),
            socket.inet_aton(self.config.interface_host),
        )

    def start(self) -> None:
        if not self.config.enabled or self.thread_alive:
            return
        self._stop_event.clear()
        receiver: socket.socket | None = None
        try:
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # Several local processes share this port by design, so the option
            # has to be set before the bind rather than after it.
            receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind the port on any address: binding the group address itself
            # does not work on Windows.
            receiver.bind(("", self.config.listen_port))
            receiver.settimeout(0.2)
            # The driver sends from loopback with TTL 0, so a membership on any
            # other interface never sees the traffic.
            receiver.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, self._membership())
        except OSError as exc:
            if receiver is not None:
                try:
                    receiver.close()
                except OSError:
                    pass
            with self._lock:
                self._last_error = f"driver log listen failed: {exc}"
                self._receiver_listening = False
            if self.logger:
                self.logger.warning("AnyaDance driver log listener could not join: %s", exc)
            return
        self._socket = receiver
        with self._lock:
            self._receiver_listening = True
            self._last_error = None
        self._thread = threading.Thread(target=self._run, name="neko-anyadance-driver-log", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        receiver = self._socket
        self._socket = None
        if receiver is not None:
            try:
                receiver.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, self._membership())
            except OSError:
                pass
            try:
                receiver.close()
            except OSError:
                pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        with self._lock:
            self._receiver_listening = False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            receiver = self._socket
            if receiver is None:
                break
            try:
                packet, sender = receiver.recvfrom(MAX_DRIVER_LOG_PACKET_BYTES)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    with self._lock:
                        self._last_error = f"driver log receive failed: {exc}"
                break
            self.ingest_packet(packet, sender=sender, now=self._clock())
        with self._lock:
            self._receiver_listening = False

    def ingest_packet(
        self,
        packet: bytes,
        *,
        sender: tuple[str, int] = ("127.0.0.1", 0),
        now: float | None = None,
    ) -> bool:
        """Decode one datagram. Public for deterministic protocol tests."""
        timestamp = self._clock() if now is None else now
        event = parse_driver_log_event(packet)
        if event is None:
            with self._lock:
                self._rejected_packets += 1
            return False
        with self._lock:
            self._received_packets += 1
            if not self._track_sequence_locked(event["sequence"]):
                self._duplicate_events += 1
                return False
            self._decoded_events += 1
            self._apply_event_locked(event, timestamp)
            self._last_error = None
        return True

    def _track_sequence_locked(self, sequence: int) -> bool:
        """Return False for a repeat. Loss is counted from forward gaps only."""
        if sequence in self._seen_sequences:
            return False
        self._seen_sequences[sequence] = None
        while len(self._seen_sequences) > self.config.history_size:
            self._seen_sequences.popitem(last=False)
        if self._next_sequence is None:
            self._next_sequence = sequence + 1
            return True
        if sequence >= self._next_sequence:
            # A gap means datagrams were dropped; a late arrival inside the
            # window is simply reordered and was already counted when the gap
            # opened, so it must not be counted again.
            self._lost_events += sequence - self._next_sequence
            self._next_sequence = sequence + 1
        return True

    def _apply_event_locked(self, event: dict[str, Any], now: float) -> None:
        kind = event["type"]
        if kind == "command_processed":
            command = event["command"]
            source = event["source"]
            if command["accepted"]:
                self._accepted_commands += 1
            else:
                self._rejected_commands += 1
            if command["y_clamped"]:
                self._y_clamped_commands += 1
            origin = f"{source['host']}:{source['port']}"
            self._senders[origin] = self._senders.get(origin, 0) + 1
            while len(self._senders) > _MAX_TRACKED_SENDERS:
                self._senders.popitem(last=False)
            self._last_command = {
                "sequence": event["sequence"],
                "accepted": command["accepted"],
                "bytes": command["bytes"],
                "devices": command["devices"],
                "y_clamped": command["y_clamped"],
                "source": origin,
                "detail": event["detail"],
                "suppressed": event["suppressed"],
                "payload": command["payload"],
                "at_monotonic": now,
            }
            self._last_command_at = now
        elif kind == "haptic_vibration":
            haptic = event["haptic"]
            self._last_haptic = {
                "sequence": event["sequence"],
                "device": event["device"],
                "duration_seconds": haptic["duration_seconds"],
                "frequency_hz": haptic["frequency_hz"],
                "amplitude": haptic["amplitude"],
                "detail": event["detail"],
                "at_monotonic": now,
            }
        else:
            self._unknown_events += 1

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            age_ms = (
                max(0.0, (now - self._last_command_at) * 1000.0)
                if self._last_command_at is not None
                else None
            )
            fresh = age_ms is not None and age_ms <= self.config.stale_after_ms
            if self._last_error and not self._receiver_listening:
                connection = "error"
            elif fresh:
                connection = "detected"
            elif self._last_command_at is not None:
                connection = "stale"
            elif self._receiver_listening:
                connection = "listening"
            else:
                connection = "unknown"
            return {
                "enabled": self.config.enabled,
                "listen_address": f"{self.config.multicast_group}:{self.config.listen_port}",
                "interface_host": self.config.interface_host,
                "receiver_listening": self._receiver_listening,
                "connection": connection,
                "stale_after_ms": self.config.stale_after_ms,
                "last_command_age_ms": round(age_ms, 1) if age_ms is not None else None,
                "received_packets": self._received_packets,
                "decoded_events": self._decoded_events,
                "rejected_packets": self._rejected_packets,
                "unknown_events": self._unknown_events,
                "duplicate_events": self._duplicate_events,
                "lost_events": self._lost_events,
                "accepted_commands": self._accepted_commands,
                "rejected_commands": self._rejected_commands,
                "y_clamped_commands": self._y_clamped_commands,
                "last_command": dict(self._last_command) if self._last_command else None,
                "last_haptic": dict(self._last_haptic) if self._last_haptic else None,
                "senders": list(self._senders),
                "last_error": self._last_error,
            }


__all__ = [
    "DRIVER_LOG_PROTOCOL_VERSION",
    "MAX_DRIVER_LOG_PACKET_BYTES",
    "DriverLogListener",
    "parse_driver_log_event",
]
