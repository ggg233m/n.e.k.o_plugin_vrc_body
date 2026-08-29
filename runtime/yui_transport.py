"""独立 YUI NPC 插件的可靠 MIDI 发送与心跳通道。

传输层不替世界做成功判断：普通命令必须等 ``npc.ack``，长操作在 ACK 后仍只
是 ``accepted``。HEARTBEAT 与 ESTOP 不占普通命令的单未决槽。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .yui_protocol import (
    COMMAND_IDS,
    CommandFrame,
    MidiEvent,
    TextTransaction,
    encode_command,
    encode_text_transaction,
    encode_upper_body_frame,
    preload_command_constraints,
)
from .yui_session import YuiAck, YuiSessionState


MidiSink = Callable[[MidiEvent], None]


@dataclass(frozen=True)
class YuiCommandOutcome:
    """与 §17.2 对齐的确定性命令结果。"""

    status: str
    kind: str | None
    wire_sequence: int
    request_hash: str
    operation_id: str | None
    error: str | None
    detail: str | None
    ack_replayed: bool
    session_rebuild_required: bool = False
    snapshot_requested: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "wire_seq": self.wire_sequence,
            "request_hash": self.request_hash,
            "op_id": self.operation_id,
            "error": self.error,
            "detail": self.detail,
            "ack_replayed": self.ack_replayed,
            "session_rebuild_required": self.session_rebuild_required,
            "snapshot_requested": self.snapshot_requested,
        }


@dataclass(frozen=True)
class YuiTextOutcome:
    transaction: TextTransaction
    begin: YuiCommandOutcome
    commit: YuiCommandOutcome | None

    @property
    def status(self) -> str:
        return self.commit.status if self.commit is not None else self.begin.status


class MidoOutputSink:
    """可选的 ``mido`` 输出适配器；未安装时不影响后端其余功能。"""

    def __init__(self, port_name: str = "NEKO_MIDI") -> None:
        try:
            import mido  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "YUI MIDI 输出需要可选依赖 mido 和 python-rtmidi"
            ) from exc
        self._mido = mido
        self.requested_port_name = port_name
        self.port_name = self._resolve_port_name(port_name, mido.get_output_names())
        self._port = mido.open_output(self.port_name)

    @staticmethod
    def _resolve_port_name(port_name: str, available: Sequence[str]) -> str:
        """兼容 RtMidi 在 Windows 端口名后自动追加的数字索引。"""
        if port_name in available:
            return port_name

        prefix = port_name + " "
        indexed = [
            name
            for name in available
            if name.startswith(prefix) and name[len(prefix) :].isdigit()
        ]
        if len(indexed) == 1:
            return indexed[0]
        if len(indexed) > 1:
            raise RuntimeError(
                f"MIDI 输出端口 {port_name!r} 对应多个 RtMidi 端口，拒绝自动选择：{indexed!r}"
            )
        raise OSError(
            f"找不到 MIDI 输出端口 {port_name!r}；当前可用端口：{list(available)!r}"
        )

    def __call__(self, event: MidiEvent) -> None:
        if event.type == "cc":
            message = self._mido.Message(
                "control_change",
                channel=event.channel,
                control=event.number,
                value=event.value,
            )
        else:
            message = self._mido.Message(
                "note_on",
                channel=event.channel,
                note=event.number,
                velocity=event.value,
            )
        self._port.send(message)

    def close(self) -> None:
        self._port.close()


class YuiReliableTransport:
    """单普通未决命令、独立 1Hz 心跳和原样重发的传输器。"""

    def __init__(
        self,
        sink: MidiSink,
        session: YuiSessionState,
        *,
        ack_timeout_s: float = 2.0,
        command_deadline_s: float = 5.0,
        heartbeat_interval_s: float = 1.0,
    ) -> None:
        # 构造期加载冻结常量；缺文件时提前失败，不等到第一条命令。
        preload_command_constraints()
        self._sink = sink
        self.session = session
        self.ack_timeout_s = max(0.01, float(ack_timeout_s))
        self.command_deadline_s = max(self.ack_timeout_s, float(command_deadline_s))
        self.heartbeat_interval_s = max(0.05, float(heartbeat_interval_s))
        self._send_lock = threading.RLock()
        self._sequence_lock = threading.Lock()
        self._normal_lock = threading.RLock()
        self._rate_lock = threading.Condition(threading.RLock())
        self._normal_command_times: deque[float] = deque()
        self._reliable_command_times: deque[float] = deque()
        self._upper_body_frame_times: deque[float] = deque()
        self._text_payload_times: deque[float] = deque()
        self._next_sequence = 1
        self._upper_body_sequence = 1
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_enabled = False
        self._upper_body_active = False
        self._text_transfer_active = False

    def _allocate_sequence(self) -> int:
        with self._sequence_lock:
            value = self._next_sequence
            self._next_sequence = 1 if value >= 127 else value + 1
            return value

    def _allocate_upper_body_sequence(self) -> int:
        with self._sequence_lock:
            value = self._upper_body_sequence
            self._upper_body_sequence = 1 if value >= 127 else value + 1
            return value

    @staticmethod
    def _trim_window(window: deque[float], now: float) -> None:
        while window and now - window[0] >= 1.0:
            window.popleft()

    def _wait_normal_rate_slot(self) -> None:
        """为 1Hz 心跳预留一条可靠命令容量。"""
        with self._rate_lock:
            while True:
                now = time.monotonic()
                self._trim_window(self._normal_command_times, now)
                maximum = 1 if self._upper_body_active else 3
                if len(self._normal_command_times) < maximum:
                    self._normal_command_times.append(now)
                    self._reliable_command_times.append(now)
                    return
                self._rate_lock.wait(max(0.001, 1.0 - (now - self._normal_command_times[0])))

    def _record_priority_reliable(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            self._trim_window(self._reliable_command_times, now)
            self._reliable_command_times.append(now)
            self._rate_lock.notify_all()

    def _send_events(self, events: Sequence[MidiEvent]) -> None:
        with self._send_lock:
            for event in events:
                self._sink(event)

    @staticmethod
    def _expected_session(frame: CommandFrame, current_session: int) -> int:
        if frame.command == "DISCOVER":
            return frame.parameters[0] + (frame.parameters[1] << 14)
        return current_session

    @staticmethod
    def _operation_kind(frame: CommandFrame) -> str | None:
        if frame.command == "GOTO_XZ":
            return "goto"
        if frame.command == "TURN_TO":
            return "turn"
        if frame.command in {"LOOK_AT", "LOOK_AT_XYZ"}:
            return "look"
        if frame.command == "PLAY_ANIM":
            return "action"
        if frame.command == "SET_EXPRESSION" and frame.parameters[3] != 127:
            return "expression"
        if frame.command == "SET_MODE":
            return {1: "follow", 3: "wander"}.get(frame.parameters[3])
        return None

    def _outcome_from_ack(self, frame: CommandFrame, ack: YuiAck) -> YuiCommandOutcome:
        kind = self._operation_kind(frame)
        operation_id = (
            f"{ack.session}:{frame.sequence}:{frame.request_hash}"
            if ack.ok and kind is not None and self.session.operation_lifecycle
            else None
        )
        return YuiCommandOutcome(
            status=("accepted" if kind is not None else "succeeded") if ack.ok else "failed",
            kind=kind,
            wire_sequence=frame.sequence,
            request_hash=frame.request_hash,
            operation_id=operation_id,
            error=ack.error,
            detail=ack.detail,
            ack_replayed=ack.replayed,
        )

    def _send_prebuilt_normal(self, frame: CommandFrame) -> YuiCommandOutcome:
        if frame.command in {"HEARTBEAT", "ESTOP"}:
            raise ValueError("HEARTBEAT/ESTOP 必须走各自的优先通道")
        with self._normal_lock:
            self._wait_normal_rate_slot()
            expected_session = self._expected_session(frame, self.session.session)
            after_ack = self.session.ack_generation
            started = time.monotonic()
            self._send_events(frame.events)
            ack = self.session.wait_for_ack(
                frame.sequence,
                frame.command_id,
                frame.request_hash,
                self.ack_timeout_s,
                session=expected_session,
                after_arrival_index=after_ack,
            )
            if ack is None:
                # 只原样重发一次；旧 seq/hash/寄存器不得重建。
                self._wait_normal_rate_slot()
                self._send_events(frame.events)
                remaining = max(0.0, self.command_deadline_s - (time.monotonic() - started))
                ack = self.session.wait_for_ack(
                    frame.sequence,
                    frame.command_id,
                    frame.request_hash,
                    remaining,
                    session=expected_session,
                    after_arrival_index=after_ack,
                )
            if ack is None:
                snapshot_requested = self._request_snapshot_evidence(frame)
                return YuiCommandOutcome(
                    status="unknown",
                    kind=self._operation_kind(frame),
                    wire_sequence=frame.sequence,
                    request_hash=frame.request_hash,
                    operation_id=None,
                    error="ack_timeout",
                    detail=(
                        "未取得可关联 ACK；已请求 snapshot 取证，结果仍必须视为 unknown"
                        if snapshot_requested
                        else "未取得可关联 ACK，且当前世界未发布 snapshot；必须重建 session"
                    ),
                    ack_replayed=False,
                    session_rebuild_required=True,
                    snapshot_requested=snapshot_requested,
                )
            return self._outcome_from_ack(frame, ack)

    def _request_snapshot_evidence(self, failed_frame: CommandFrame) -> bool:
        """ACK 最终超时后尝试一次独立快照；不得递归或改写原命令结论。"""
        if (
            failed_frame.command == "SNAPSHOT_REQUEST"
            or "snapshot" not in self.session.capabilities
            or self.session.session <= 0
        ):
            return False
        try:
            snapshot = encode_command("SNAPSHOT_REQUEST", self._allocate_sequence())
            self._wait_normal_rate_slot()
            after_ack = self.session.ack_generation
            self._send_events(snapshot.events)
            self.session.wait_for_ack(
                snapshot.sequence,
                snapshot.command_id,
                snapshot.request_hash,
                self.ack_timeout_s,
                session=self.session.session,
                after_arrival_index=after_ack,
            )
            return True
        except Exception:
            return False

    def send_command(
        self,
        command: str | int,
        parameters: Sequence[Any] = (0, 0, 0, 0, 0, 0),
    ) -> YuiCommandOutcome:
        command_id = COMMAND_IDS.get(command.strip().upper()) if isinstance(command, str) else int(command)
        if command_id == COMMAND_IDS["HEARTBEAT"]:
            frame = self.send_heartbeat()
            return YuiCommandOutcome(
                status="accepted",
                kind=None,
                wire_sequence=frame.sequence,
                request_hash=frame.request_hash,
                operation_id=None,
                error=None,
                detail="HEARTBEAT ACK 仅作诊断，不等待也不重发",
                ack_replayed=False,
            )
        if command_id == COMMAND_IDS["ESTOP"]:
            frame = self.send_estop()
            return YuiCommandOutcome(
                status="accepted",
                kind=None,
                wire_sequence=frame.sequence,
                request_hash=frame.request_hash,
                operation_id=None,
                error=None,
                detail="ESTOP 已进入最高优先级发送路径",
                ack_replayed=False,
            )
        with self._normal_lock:
            frame = encode_command(command, self._allocate_sequence(), parameters)
            return self._send_prebuilt_normal(frame)

    def send_heartbeat(self) -> CommandFrame:
        frame = encode_command("HEARTBEAT", self._allocate_sequence())
        self._record_priority_reliable()
        self._send_events(frame.events)
        return frame

    def send_estop(self, *, channel: int = 0, acknowledge: bool = True) -> CommandFrame:
        sequence = self._allocate_sequence() if acknowledge else 0
        frame = encode_command("ESTOP", sequence, estop_channel=channel)
        # ESTOP 不等待预算、不等待普通命令锁，也不依赖队列腾位。
        self._send_events(frame.events)
        self.session.set_host_arm_authorized(False)
        return frame

    def set_heartbeat_enabled(self, enabled: bool) -> None:
        self._heartbeat_enabled = bool(enabled)

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="yui-midi-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_interval_s):
            if not self._heartbeat_enabled or self.session.session <= 0:
                continue
            try:
                self.send_heartbeat()
            except Exception:
                # 后端状态接口负责暴露 sink 故障；心跳线程不能因单次设备异常永久退出。
                continue

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, self.heartbeat_interval_s * 2.0))
        self._heartbeat_thread = None

    def send_upper_body(self, values: Mapping[str, Any]) -> tuple[MidiEvent, ...] | None:
        """发送一帧名义 20Hz 上身流；文本事务和非执行态直接暂停。"""
        if self._text_transfer_active:
            return None
        if self.session.control_state not in {"external", "moving", "action"}:
            return None
        if "upper_body_stream" not in self.session.capabilities:
            return None
        with self._rate_lock:
            now = time.monotonic()
            self._trim_window(self._reliable_command_times, now)
            self._trim_window(self._upper_body_frame_times, now)
            reliable_count = len(self._reliable_command_times)
            maximum_frames = 20 if reliable_count <= 1 else 19 if reliable_count == 2 else 0
            if len(self._upper_body_frame_times) >= maximum_frames:
                return None
            self._upper_body_frame_times.append(now)
        frame = encode_upper_body_frame(self._allocate_upper_body_sequence(), values)
        self._upper_body_active = True
        self._send_events(frame)
        return frame

    def mark_upper_body_inactive(self) -> None:
        self._upper_body_active = False

    def _send_text_payload(self, payload: Sequence[MidiEvent]) -> None:
        for event in payload:
            with self._rate_lock:
                while True:
                    now = time.monotonic()
                    self._trim_window(self._text_payload_times, now)
                    if len(self._text_payload_times) < 159:
                        self._text_payload_times.append(now)
                        break
                    self._rate_lock.wait(max(0.001, 1.0 - (now - self._text_payload_times[0])))
            self._send_events((event,))

    def send_text(
        self,
        text: str,
        *,
        transfer_sequence: int,
        display_seconds: int = 0,
    ) -> YuiTextOutcome:
        with self._normal_lock:
            begin_sequence = self._allocate_sequence()
            commit_sequence = self._allocate_sequence()
            transaction = encode_text_transaction(
                text,
                transfer_sequence=transfer_sequence,
                begin_sequence=begin_sequence,
                commit_sequence=commit_sequence,
                display_seconds=display_seconds,
            )
            self._text_transfer_active = True
            self._upper_body_active = False
            try:
                begin = self._send_prebuilt_normal(transaction.begin)
                if begin.status not in {"accepted", "succeeded"}:
                    return YuiTextOutcome(transaction=transaction, begin=begin, commit=None)
                self._send_text_payload(transaction.payload)
                commit = self._send_prebuilt_normal(transaction.commit)
                return YuiTextOutcome(transaction=transaction, begin=begin, commit=commit)
            finally:
                self._text_transfer_active = False

    def close(self) -> None:
        self.stop_heartbeat()
        close = getattr(self._sink, "close", None)
        if callable(close):
            close()


__all__ = [
    "MidoOutputSink",
    "MidiSink",
    "YuiCommandOutcome",
    "YuiReliableTransport",
    "YuiTextOutcome",
]
