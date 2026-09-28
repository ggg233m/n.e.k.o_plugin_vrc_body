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
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import DriverLogConfig

MAX_DRIVER_LOG_PACKET_BYTES = 65507
DRIVER_LOG_PROTOCOL_VERSION = 1
# 动作时间轴记录格式版本。与 DRIVER_LOG_PROTOCOL_VERSION 分开：时间轴是我们
# 自己落盘的产物，生命周期比驱动遥测契约长。
ACTION_TIMELINE_PROTOCOL_VERSION = 1
MAX_PAYLOAD_CHARS = 2048
MAX_DETAIL_CHARS = 500
_MAX_TRACKED_SENDERS = 16
_ACTION_ID_CHARS = 64


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


def _hmd_yaw_from_payload(payload: Any) -> float | None:
    """从驱动 command_processed 的 JSON payload 提取 HMD yaw。"""
    if not isinstance(payload, str) or not payload:
        return None
    try:
        root = json.loads(payload)
    except (TypeError, ValueError):
        return None
    devices = root.get("devices") if isinstance(root, dict) else None
    hmd = devices.get("hmd") if isinstance(devices, dict) else None
    pose = hmd.get("pose") if isinstance(hmd, dict) else None
    rotation = pose.get("rotation_xyzw") if isinstance(pose, dict) else None
    if not isinstance(rotation, list) or len(rotation) != 4:
        return None
    try:
        x, y, z, w = (float(value) for value in rotation)
        if not all(math.isfinite(value) for value in (x, y, z, w)):
            return None
        return math.degrees(math.atan2(
            2.0 * (w * y + x * z),
            1.0 - 2.0 * (y * y + z * z),
        ))
    except (TypeError, ValueError, OverflowError):
        return None


def _hmd_pose_from_payload(payload: Any) -> dict[str, Any] | None:
    """提取驱动实际接受的 HMD 位置和四元数（位置单位沿用驱动协议）。"""
    if not isinstance(payload, str) or not payload:
        return None
    try:
        root = json.loads(payload, parse_constant=_reject_constant)
    except (TypeError, ValueError):
        return None
    devices = root.get("devices") if isinstance(root, dict) else None
    hmd = devices.get("hmd") if isinstance(devices, dict) else None
    pose = hmd.get("pose") if isinstance(hmd, dict) else None
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    rotation = pose.get("rotation_xyzw")
    if not isinstance(position, list) or len(position) != 3:
        return None
    if not isinstance(rotation, list) or len(rotation) != 4:
        return None
    try:
        values = [float(value) for value in [*position, *rotation]]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return {
        "position_xyz": values[:3],
        "rotation_xyzw": values[3:],
    }


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
    version = root.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != DRIVER_LOG_PROTOCOL_VERSION
    ):
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
        hmd_yaw = _hmd_yaw_from_payload(event["command"]["payload"])
        if hmd_yaw is not None:
            event["hmd_yaw_deg"] = hmd_yaw
        hmd_pose = _hmd_pose_from_payload(event["command"]["payload"])
        if hmd_pose is not None:
            event["hmd_pose"] = hmd_pose
    elif name == "haptic_vibration":
        haptic = root.get("haptic") if isinstance(root.get("haptic"), dict) else {}
        event["type"] = "haptic_vibration"
        event["device"] = _text(root.get("device"), 32)
        event["haptic"] = {
            "duration_seconds": _finite(haptic.get("duration_seconds")),
            "frequency_hz": _finite(haptic.get("frequency_hz")),
            "amplitude": _finite(haptic.get("amplitude")),
        }
    elif name == "action_timeline":
        event["type"] = "action_timeline"
        event["action"] = _parse_action_fields(root)
    return event


def _parse_action_fields(root: dict[str, Any]) -> dict[str, Any]:
    """解析动作时间轴字段。缺字段一律降级为中性值，绝不抛错。

    ``frame_index`` / ``frame_timestamp`` **不在这里解**：驱动不知道视频帧号，
    只有本地 :class:`VideoTimebase` 能把同一个 monotonic 时刻换算成帧号。
    这条分工是"视频、帧索引、动作日志共享同一时间基准"的实现方式——
    三方都以录制锚点为原点，谁都不用文件名/墙钟时间互推。
    """
    command = root.get("input_command")
    command = command if isinstance(command, dict) else {}
    velocity = root.get("osc_velocity")
    velocity = velocity if isinstance(velocity, dict) else {}
    turn = root.get("turn_intent")
    turn = turn if isinstance(turn, dict) else {}
    send_result = root.get("actual_send_result")
    ack = root.get("driver_ack")
    return {
        "monotonic_ms": _whole(root.get("monotonic_ms")),
        "goal_id": _text(root.get("goal_id"), _ACTION_ID_CHARS),
        "episode_id": _text(root.get("episode_id"), _ACTION_ID_CHARS),
        "input_command": {
            "forward": _finite(command.get("forward")),
            "strafe": _finite(command.get("strafe")),
            "jump": command.get("jump") is True,
            "run": command.get("run") is True,
            "source": _text(command.get("source"), 32),
        },
        "actual_send_result": send_result if send_result in ("sent", "failed") else "unknown",
        "driver_ack": ack if ack in ("accepted", "rejected") else "none",
        "osc_velocity": {
            "vx": _finite(velocity.get("vx")),
            "vy": _finite(velocity.get("vy")),
            "vz": _finite(velocity.get("vz")),
        },
        "turn_intent": {
            "yaw_delta": _finite(turn.get("yaw_delta")),
            "pitch_delta": _finite(turn.get("pitch_delta")),
            "source": _text(turn.get("source"), 32),
        },
    }


class DriverLogListener:
    """Join the driver log group and summarize what the driver reports."""

    def __init__(
        self,
        config: DriverLogConfig,
        *,
        logger: Any = None,
        clock: Callable[[], float] = time.monotonic,
        on_action: Callable[[dict[str, Any]], None] | None = None,
        on_hmd: Callable[[dict[str, Any], float], None] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self._clock = clock
        # 动作时间轴的接入点：监听器只负责"看到"事件，落盘由 ActionLogRecorder
        # 负责（它持有 VideoTimebase，才能把 monotonic 换算成帧号）。
        self._on_action = on_action
        self._on_hmd = on_hmd
        self._action_events = 0
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
        self._last_hmd_yaw_deg: float | None = None
        self._last_hmd_position_xyz: list[float] | None = None
        self._last_hmd_rotation_xyzw: list[float] | None = None
        self._last_hmd_at: float | None = None
        self._senders: OrderedDict[str, float] = OrderedDict()
        self._last_error: str | None = None
        # 与 _last_error 分开：ingest_packet 在成功解码后会清空 _last_error，
        # 若把落盘失败写在那里面，会被这次清空吞掉，故障就静默了。
        self._last_action_error: str | None = None

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
            self._senders[origin] = now
            self._senders.move_to_end(origin)
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
            if command.get("accepted") is True and event.get("hmd_yaw_deg") is not None:
                self._last_hmd_yaw_deg = float(event["hmd_yaw_deg"])
                self._last_hmd_at = now
                hmd_pose = event.get("hmd_pose")
                if isinstance(hmd_pose, dict):
                    self._last_hmd_position_xyz = list(hmd_pose.get("position_xyz") or [])
                    self._last_hmd_rotation_xyzw = list(hmd_pose.get("rotation_xyzw") or [])
                if self._on_hmd is not None:
                    try:
                        self._on_hmd({
                            "yaw_deg": event["hmd_yaw_deg"],
                            "position_xyz": self._last_hmd_position_xyz,
                            "rotation_xyzw": self._last_hmd_rotation_xyzw,
                            "sequence": event["sequence"],
                            "driver_timestamp_ms": event["timestamp_ms"],
                        }, now)
                    except Exception as exc:  # noqa: BLE001 - 读取器故障不能拖垮遥测
                        self._last_error = f"HMD telemetry sink failed: {exc}"[:500]
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
        elif kind == "action_timeline":
            self._action_events += 1
            if self._on_action is not None:
                try:
                    self._on_action(event)
                except Exception as exc:  # noqa: BLE001 - 落盘失败不得拖垮监听线程
                    self._last_action_error = f"action timeline sink failed: {exc}"
        else:
            self._unknown_events += 1

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        now = self._clock() if now is None else now
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
            stale_threshold_s = self.config.stale_after_ms / 1000.0
            active_senders = [
                origin for origin, last_seen in self._senders.items()
                if now - last_seen <= stale_threshold_s
            ]
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
                "action_events": self._action_events,
                "duplicate_events": self._duplicate_events,
                "lost_events": self._lost_events,
                "accepted_commands": self._accepted_commands,
                "rejected_commands": self._rejected_commands,
                "y_clamped_commands": self._y_clamped_commands,
                "last_command": dict(self._last_command) if self._last_command else None,
                "last_haptic": dict(self._last_haptic) if self._last_haptic else None,
                "hmd_pose_available": self._last_hmd_at is not None,
                "last_hmd_yaw_deg": self._last_hmd_yaw_deg,
                "last_hmd_position_xyz": list(self._last_hmd_position_xyz) if self._last_hmd_position_xyz else None,
                "last_hmd_rotation_xyzw": list(self._last_hmd_rotation_xyzw) if self._last_hmd_rotation_xyzw else None,
                "last_hmd_age_ms": (
                    round(max(0.0, (now - self._last_hmd_at) * 1000.0), 1)
                    if self._last_hmd_at is not None else None
                ),
                "senders": active_senders,
                "last_error": self._last_error,
                "last_action_error": self._last_action_error,
            }


# ------------------------------------------------------------------ 动作时间轴
#
# 为什么必须有它
# --------------
# 本轮之前**没有任何动作日志**：录屏 `2026-09-18 07-31-11.mkv` 没有配套指令/速度记录，
# 于是"路线历史"这条独立通道只能兑现 2/5 个子项（已确认 episode 的顺序、已确认地点
# 之间的访问连接），另外 3 项（转向指令、前进持续时间、OSC 速度积分距离）无从计算。
# 直接后果：`ep39` 这类视觉不可分辨的地点只能靠图结构共识消歧，少了一条独立证据。
#
# 时间基准（硬约束）
# ------------------
# 视频帧、帧索引、动作日志**共享同一个原点**：录制开始时抓取一次的 monotonic 锚点。
#
#     frame_timestamp(f) = anchor_monotonic + f / fps
#
# 禁止用文件名时间、墙钟时间或脚本启动时间互相推算——它们的抖动会让"某一帧对应的
# 指令"整体偏移几百毫秒，而回环确认恰恰是在这个尺度上判定的。
#
# 本模块只记录"当时要求身体做什么 / 指令是否真的发出 / 驱动是否接受 / 角色实际速度"，
# **不**用于把 AngularY 积分成朝向（那条约束没有变）。


class VideoTimebase:
    """把视频帧索引与动作事件锁在同一个单调时钟基准上。"""

    def __init__(self, fps: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if (not isinstance(fps, (int, float)) or isinstance(fps, bool)
                or not math.isfinite(float(fps)) or fps <= 0):
            # NaN 会让 `fps <= 0` 为假从而溜过去，之后每个时间戳都是 NaN——
            # 这种"静默毒化"比直接报错难查得多。
            raise ValueError("fps must be a finite positive number")
        self._fps = float(fps)
        self._clock = clock
        self._anchor: float | None = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def anchor(self) -> float | None:
        return self._anchor

    def begin(self) -> float:
        """抓取录制锚点。只应在采集真正开始时调用一次。"""
        self._anchor = self._clock()
        return self._anchor

    def _require_anchor(self) -> float:
        if self._anchor is None:
            raise RuntimeError("VideoTimebase.begin() must be called before use")
        return self._anchor

    def frame_timestamp(self, frame_index: int) -> float:
        """帧号 -> 单调时间（秒）。"""
        return self._require_anchor() + int(frame_index) / self._fps

    def frame_index_at(self, monotonic_time: float) -> int:
        """单调时间（秒）-> 最近帧号。事件与帧由此对齐到同一基准。"""
        return int(round((float(monotonic_time) - self._require_anchor()) * self._fps))

    def to_record(self) -> dict[str, Any]:
        return {"fps": self._fps, "anchor_monotonic": self._anchor}

    @classmethod
    def from_record(cls, record: dict[str, Any], *,
                    clock: Callable[[], float] = time.monotonic) -> "VideoTimebase":
        """从落盘记录恢复同一基准（离线分析时必须用它，不能重新 begin）。"""
        tb = cls(float(record["fps"]), clock=clock)
        anchor = record.get("anchor_monotonic")
        tb._anchor = None if anchor is None else float(anchor)
        return tb


class ActionLogRecorder:
    """把动作事件按同一时间基准落成 JSONL。

    每行字段固定为：
        monotonic_time, frame_timestamp, frame_index, goal_id, episode_id,
        input_command, actual_send_result, driver_ack, osc_velocity, turn_intent
    """

    def __init__(self, path: str | Path, timebase: VideoTimebase, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.path = Path(path)
        self.timebase = timebase
        self._clock = clock
        self._handle = None
        self._written = 0
        self._last_error: str | None = None

    @property
    def written(self) -> int:
        return self._written

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> bool:
        if self._handle is not None:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
            header = {"protocol_version": ACTION_TIMELINE_PROTOCOL_VERSION,
                      "record": "timebase", **self.timebase.to_record()}
            self._handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            self._handle.flush()
        except OSError as exc:
            self._last_error = f"action timeline open failed: {exc}"
            self._handle = None
            return False
        return True

    def stop(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def ingest(self, event: dict[str, Any],
               monotonic_time: float | None = None) -> bool:
        """把一条已解析的 ``action_timeline`` 事件写盘，并补上帧号。"""
        if event.get("type") != "action_timeline":
            return False
        return self.record(
            action=event.get("action", {}),
            sequence=event.get("sequence", 0),
            monotonic_time=monotonic_time,
        )

    def record(self, *, action: dict[str, Any], sequence: int = 0,
               monotonic_time: float | None = None) -> bool:
        if self._handle is None and not self.start():
            return False
        now = self._clock() if monotonic_time is None else float(monotonic_time)
        # 驱动自报的 monotonic_ms 优先作为事件时刻；缺失则用本地时钟。
        reported = action.get("monotonic_ms") or 0
        event_time = reported / 1000.0 if reported else now
        try:
            frame_index = self.timebase.frame_index_at(event_time)
        except RuntimeError as exc:
            self._last_error = str(exc)
            return False
        row = {
            "protocol_version": ACTION_TIMELINE_PROTOCOL_VERSION,
            "record": "action",
            "sequence": int(sequence),
            "monotonic_time": round(event_time, 6),
            "frame_index": frame_index,
            "frame_timestamp": round(self.timebase.frame_timestamp(frame_index), 6),
            "goal_id": action.get("goal_id", ""),
            "episode_id": action.get("episode_id", ""),
            "input_command": action.get("input_command", {}),
            "actual_send_result": action.get("actual_send_result", "unknown"),
            "driver_ack": action.get("driver_ack", "none"),
            "osc_velocity": action.get("osc_velocity", {}),
            "turn_intent": action.get("turn_intent", {}),
        }
        try:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._handle.flush()
        except OSError as exc:
            self._last_error = f"action timeline write failed: {exc}"
            return False
        self._written += 1
        return True


class ActionTimeline:
    """运行时动作时间轴：路线历史通道的**唯一写入口**。

    与 :class:`ActionLogRecorder` 的分工：Recorder 只负责"把一行写对"，
    ActionTimeline 负责**运行时语义**——何时锚定时间基准、谁提供 goal/episode、
    什么情况必须拒写。三条硬规则写进实现里：

    1. **不伪造**：未启用 / 未锚定 / 文件打不开时，``record_*`` 一律 no-op 并返回
       ``False``。调用方据此知道"本次录制没有动作日志"，从而按既定规则**自动降级
       为视觉代理**，而不是拿到一份看起来正常、实则全 0 的假路线历史。
    2. **不推断回执**：本机 ``sendto`` 成功只写 ``actual_send_result="sent"``；
       ``driver_ack`` 永远保持 ``"none"``——只有驱动自报的 ack（经
       :meth:`ingest_driver_event`）才算数。
    3. **同一时间基准**：帧号只由唯一的 :class:`VideoTimebase` 换算。锚点必须与
       **录像起点**重合：录像侧要么调用 :meth:`begin`，要么把 :meth:`to_record`
       写进自己的 sidecar 再用 :meth:`VideoTimebase.from_record <from_record>`
       恢复——绝不能用文件名/墙钟时间反推。
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        *,
        enabled: bool = True,
        goal_id: str = "",
        dedup_window_s: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._enabled = bool(enabled)
        self._timebase = VideoTimebase(fps, clock=clock)
        self._recorder = ActionLogRecorder(path, self._timebase, clock=clock)
        self._goal_id = _text(goal_id, _ACTION_ID_CHARS)
        self._episode_id = ""
        self._velocity = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        self._dedup_window = max(0.0, float(dedup_window_s))
        self._last_key: tuple[Any, ...] | None = None
        self._last_emit_at: float | None = None
        self._begun = False
        self._skipped = 0

    # ---- 生命周期 ----------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        """是否可以落盘。未启用或未 ``begin()`` 时为 False。"""
        return self._enabled and self._begun

    @property
    def fps(self) -> float:
        return self._timebase.fps

    @property
    def written(self) -> int:
        return self._recorder.written

    @property
    def skipped(self) -> int:
        return self._skipped

    @property
    def last_error(self) -> str | None:
        return self._recorder.last_error

    def to_record(self) -> dict[str, Any]:
        """给录像侧对齐用：把它写进视频 sidecar，离线即可恢复同一基准。"""
        return self._timebase.to_record()

    def begin(self) -> bool:
        """抓锚点并打开文件。**只应在录制真正开始时调用一次**。"""
        if not self._enabled:
            return False
        self._timebase.begin()
        ok = self._recorder.start()
        self._begun = bool(ok)
        return self._begun

    def close(self) -> None:
        self._recorder.stop()
        self._begun = False

    # ---- 会话上下文 --------------------------------------------------
    def set_goal(self, goal_id: Any) -> None:
        self._goal_id = _text(goal_id, _ACTION_ID_CHARS)

    def set_episode(self, episode_id: Any) -> None:
        """在线切分器（若存在）在 episode 建立/结束时调用；不存在则保持空串。

        空串是有意义的：离线 :func:`episode_action_summary` 用 ``frame_index``
        与 episode 表对齐，不需要运行时先知道 episode。
        """
        self._episode_id = _text(episode_id, _ACTION_ID_CHARS)

    def note_velocity(self, vx: Any = 0.0, vy: Any = 0.0, vz: Any = 0.0) -> None:
        """记录最近一次 VRChat 回传速度；写入**其后**每一行。"""
        self._velocity = {"vx": _finite(vx), "vy": _finite(vy), "vz": _finite(vz)}

    # ---- 写入 --------------------------------------------------------
    def _emit(
        self,
        *,
        forward: float | None = None,
        strafe: float | None = None,
        yaw_delta: float | None = None,
        sent: bool | None = None,
        source: str = "",
    ) -> bool:
        if not self.active:
            return False
        key = (forward, strafe, yaw_delta, sent, self._episode_id, self._goal_id)
        now = self._clock()
        if (self._dedup_window > 0 and key == self._last_key
                and self._last_emit_at is not None
                and now - self._last_emit_at < self._dedup_window):
            self._skipped += 1
            return False
        action = {
            "goal_id": self._goal_id,
            "episode_id": self._episode_id,
            "input_command": {
                "forward": _finite(forward), "strafe": _finite(strafe),
                "source": _text(source, 32),
            },
            "actual_send_result": (
                "unknown" if sent is None else ("sent" if sent else "failed")
            ),
            # 绝不从本机发送成功推断驱动回执。
            "driver_ack": "none",
            "osc_velocity": dict(self._velocity),
            "turn_intent": {
                "yaw_delta": _finite(yaw_delta), "source": _text(source, 32),
            },
        }
        written = self._recorder.record(action=action, monotonic_time=now)
        if written:
            self._last_key = key
            self._last_emit_at = now
        return written

    def record_command(
        self, *, forward: Any, strafe: Any, sent: bool,
        source: str = "navigator",
    ) -> bool:
        """记录一条轴向指令。``forward``=y（前进），``strafe``=x（横移）。"""
        return self._emit(forward=_finite(forward), strafe=_finite(strafe),
                          sent=bool(sent), source=source)

    def record_turn(self, *, yaw_delta: Any, sent: bool,
                    source: str = "navigator") -> bool:
        """记录一条转向**意图**（``turn_intent`` 是输入意图，不是实际角度）。"""
        return self._emit(yaw_delta=_finite(yaw_delta), sent=bool(sent),
                          source=source)

    def ingest_driver_event(self, event: dict[str, Any]) -> bool:
        """:class:`DriverLogListener` 的 ``on_action`` sink。

        驱动自报的行会带自己的 ``driver_ack`` / ``osc_velocity``，因此这里的
        ack 是**真回执**；本地指令行的 ack 则永远是 ``"none"``。
        """
        if not self.active:
            return False
        return self._recorder.ingest(event)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "active": self.active,
            "fps": self._timebase.fps,
            "anchor_monotonic": self._timebase.anchor,
            "path": str(self._recorder.path),
            "records": self._recorder.written,
            "dedup_skipped": self._skipped,
            "goal_id": self._goal_id,
            "episode_id": self._episode_id,
            "last_error": self._recorder.last_error,
        }


def load_action_timeline(path: str | Path) -> tuple[VideoTimebase | None, list[dict]]:
    """读回 JSONL：返回 (时间基准, 动作记录列表)。坏行跳过，不抛错。"""
    timebase: VideoTimebase | None = None
    rows: list[dict] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get("record") == "timebase":
                    try:
                        timebase = VideoTimebase.from_record(row)
                    except (KeyError, ValueError, TypeError):
                        timebase = None
                    continue
                if row.get("record") == "action":
                    rows.append(row)
    except OSError:
        return None, []
    return timebase, rows


def episode_action_summary(path: str | Path,
                           episodes: Sequence[Sequence[int]],
                           labels: Iterable[object] | None = None,
                           *, fps: float = 20.0) -> list[dict]:
    """把动作日志按 episode 汇总成**路线历史**（离线，只读）。

    episode 用帧区间 ``[a, b)`` 表示（与 ``PlaceEpisode`` 同一坐标系）。

    OSC 前进距离用**真实时间间隔**积分（``monotonic_time`` 的逐行 Δt），**不按帧数
    累加**——帧率一变、或写盘循环抖动/丢行，``1/fps`` 的等间隔假设就会静默算错距离。
    Δt 不可用（缺 ``monotonic_time``／非单调／大于 5 s 的断点）时退回 ``1/fps`` 计一个
    时间片，并在 ``osc_distance_dt_fallbacks`` 里如实计数，绝不假装是真间隔。
    """
    timebase, rows = load_action_timeline(path)
    eff_fps = timebase.fps if timebase is not None else float(fps)
    dt_default = 1.0 / eff_fps if eff_fps > 0 else 0.05
    labels = list(labels) if labels is not None else None

    def _episode_of(frame_index: int) -> int | None:
        for i, (a, b) in enumerate(episodes):
            if int(a) <= frame_index < int(b):
                return i
        return None

    # 先把每行的距离增量按"该行所属 episode"归集，Δt 取该行与前一行的真实间隔。
    per_ep = [0.0] * len(episodes)
    n_dt_real = n_dt_fallback = 0
    prev_t: float | None = None
    for r in rows:
        raw_t = r.get("monotonic_time")
        t = float(raw_t) if isinstance(raw_t, (int, float)) and not isinstance(
            raw_t, bool) and math.isfinite(float(raw_t)) else None
        dt = None
        if t is not None and prev_t is not None:
            gap = t - prev_t
            if 0.0 < gap <= 5.0:
                dt = gap
        if dt is None:
            dt = dt_default
            n_dt_fallback += 1
        else:
            n_dt_real += 1
        if t is not None:
            prev_t = t
        ei = _episode_of(int(r.get("frame_index", -1)))
        if ei is not None:
            v = r.get("osc_velocity", {})
            per_ep[ei] += math.hypot(_finite(v.get("vx")),
                                     _finite(v.get("vz"))) * dt

    out: list[dict] = []
    for idx, (a, b) in enumerate(episodes):
        a, b = int(a), int(b)
        inside = [r for r in rows if a <= int(r.get("frame_index", -1)) < b]
        fwd = sum(1 for r in inside if float(r["input_command"].get("forward", 0.0)) > 0)
        back = sum(1 for r in inside if float(r["input_command"].get("forward", 0.0)) < 0)
        left = sum(1 for r in inside if float(r["turn_intent"].get("yaw_delta", 0.0)) > 0)
        right = sum(1 for r in inside if float(r["turn_intent"].get("yaw_delta", 0.0)) < 0)
        sent = sum(1 for r in inside if r.get("actual_send_result") == "sent")
        acked = sum(1 for r in inside if r.get("driver_ack") == "accepted")
        out.append({
            "episode": idx,
            "frames": [a, b],
            "label": (labels[idx] if labels is not None and idx < len(labels) else None),
            "n_records": len(inside),
            "forward_frames": fwd, "backward_frames": back,
            "turn_left_frames": left, "turn_right_frames": right,
            "commands_sent": sent, "commands_acked": acked,
            "send_failure_rate": (round(1.0 - sent / len(inside), 4)
                                  if inside else None),
            "ack_rate": round(acked / sent, 4) if sent else None,
            "osc_forward_distance": round(per_ep[idx], 4),
            "osc_distance_dt_real": n_dt_real,
            "osc_distance_dt_fallbacks": n_dt_fallback,
            "goal_ids": sorted({r.get("goal_id", "") for r in inside if r.get("goal_id")}),
        })
    return out


__all__ = [
    "ACTION_TIMELINE_PROTOCOL_VERSION",
    "DRIVER_LOG_PROTOCOL_VERSION",
    "MAX_DRIVER_LOG_PACKET_BYTES",
    "ActionLogRecorder",
    "ActionTimeline",
    "DriverLogListener",
    "VideoTimebase",
    "episode_action_summary",
    "load_action_timeline",
    "parse_driver_log_event",
]
