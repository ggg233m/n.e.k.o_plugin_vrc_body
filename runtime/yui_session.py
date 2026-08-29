"""独立 YUI NPC 插件的上行日志会话投影与 ACK 关联。

世界日志是执行结果的唯一回传。本模块只保存已经观测到的事实；缺失字段、日志
序号 gap 或未发布 capability 不会被补成猜测值。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Callable, Mapping

from .yui_protocol import COMMAND_NAMES, parse_neko_log_line


@dataclass(frozen=True)
class YuiAck:
    """一条经过公共头解析的命令 ACK。"""

    session: int
    sequence: int
    command_id: int
    command: str
    request_hash: str
    ok: bool
    replayed: bool
    state: str
    error: str | None
    detail: str | None
    log_sequence: int
    arrival_index: int = 0

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "YuiAck":
        if event.get("type") != "npc.ack":
            raise ValueError("event 不是 npc.ack")
        sequence = event.get("seq")
        command_id = event.get("cmd_id")
        request_hash = event.get("request_hash")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or not 1 <= sequence <= 127:
            raise ValueError("npc.ack.seq 必须是 1..127")
        if not isinstance(command_id, int) or isinstance(command_id, bool) or not 0 <= command_id <= 127:
            raise ValueError("npc.ack.cmd_id 必须是 0..127")
        if not isinstance(request_hash, str) or len(request_hash) != 4:
            raise ValueError("npc.ack.request_hash 必须是 4 位十六进制")
        try:
            int(request_hash, 16)
        except ValueError as exc:
            raise ValueError("npc.ack.request_hash 必须是 4 位十六进制") from exc
        ok = event.get("ok")
        replayed = event.get("replayed")
        if not isinstance(ok, bool) or not isinstance(replayed, bool):
            raise ValueError("npc.ack.ok/replayed 必须是布尔值")
        error = event.get("err")
        if not ok and not isinstance(error, str):
            raise ValueError("失败 ACK 必须包含 err")
        if ok and error is not None:
            raise ValueError("成功 ACK 不得包含 err")
        command = event.get("cmd")
        expected_name = COMMAND_NAMES.get(command_id, "UNKNOWN")
        if command != expected_name:
            raise ValueError(f"npc.ack.cmd 与 cmd_id 不一致，应为 {expected_name}")
        return cls(
            session=int(event["session"]),
            sequence=sequence,
            command_id=command_id,
            command=command,
            request_hash=request_hash.upper(),
            ok=ok,
            replayed=replayed,
            state=str(event.get("state")),
            error=error,
            detail=event.get("detail") if isinstance(event.get("detail"), str) else None,
            log_sequence=int(event["log_seq"]),
        )


class YuiSessionState:
    """把事件流投影成可供后端和 LLM 使用的最小当前状态。"""

    def __init__(self, *, ack_history_size: int = 256, recent_event_size: int = 64) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._acks: deque[YuiAck] = deque(maxlen=max(16, int(ack_history_size)))
        self._ack_generation = 0
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=max(16, int(recent_event_size)))
        self._snapshot_parts: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
        self._snapshot_part_counts: dict[tuple[int, int], int] = {}
        self._catalog_pages: dict[str, set[int]] = {}
        self._catalog_expected_pages: dict[str, int] = {}
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []
        self.session = 0
        self.world_id: str | None = None
        self.world_name: str | None = None
        self.driver_pid: int | None = None
        self.control_state = "unhandshaken"
        self.estop = False
        self.capabilities: tuple[str, ...] = ()
        self.capability_bits = 0
        self.wire_bounds: tuple[float, ...] | None = None
        self.activity_bounds: tuple[float, ...] | None = None
        self.max_speed_mps: float | None = None
        self.catalog_revision: int | None = None
        self.catalogs: dict[str, dict[int, dict[str, Any]]] = {
            "action": {},
            "expression": {},
            "text_preset": {},
            "anchor": {},
        }
        self.players: dict[int, dict[str, Any]] = {}
        self.npc_state: dict[str, Any] = {"active_ops": []}
        self.voice_state: dict[str, Any] = {
            "state": "disabled",
            "error_code": None,
            "url_loaded": False,
            "last_speech_seq": None,
        }
        self.text_state: dict[str, Any] = {
            "transfer_seq": None,
            "utf8_bytes": 0,
            "crc16": None,
            "display_until_server_ms": None,
            "text": None,
        }
        self.operations: dict[str, dict[str, Any]] = {}
        self.last_log_sequence: int | None = None
        self.log_wrap_count = 0
        self.log_gaps: list[tuple[int, int]] = []
        self.log_complete = True
        self._expect_log_sequence_one = False
        self.host_arm_authorized = False

    @property
    def operation_lifecycle(self) -> bool:
        return "operation_lifecycle" in self.capabilities

    @property
    def ack_generation(self) -> int:
        with self._condition:
            return self._ack_generation

    @property
    def active_ops_authoritative(self) -> bool:
        return self.operation_lifecycle

    def set_host_arm_authorized(self, authorized: bool) -> None:
        with self._condition:
            self.host_arm_authorized = bool(authorized)

    def add_event_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        if not callable(listener):
            raise TypeError("listener 必须可调用")
        with self._condition:
            if listener not in self._event_listeners:
                self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._condition:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

    def _reset_for_new_session(self, session: int) -> None:
        self.session = session
        self.control_state = "safe_idle"
        self.estop = False
        self.capabilities = ()
        self.capability_bits = 0
        self.wire_bounds = None
        self.activity_bounds = None
        self.max_speed_mps = None
        self.catalog_revision = None
        for catalog in self.catalogs.values():
            catalog.clear()
        self._catalog_pages.clear()
        self._catalog_expected_pages.clear()
        self.players.clear()
        self.npc_state = {"active_ops": []}
        self.operations.clear()
        self._snapshot_parts.clear()
        self._snapshot_part_counts.clear()
        self.host_arm_authorized = False

    def _track_log_sequence(self, event: Mapping[str, Any]) -> None:
        sequence = int(event["log_seq"])
        event_type = event["type"]
        if event_type == "sys.log_wrap":
            if sequence != 2_147_483_647:
                self.log_complete = False
            self.last_log_sequence = sequence
            self.log_wrap_count = max(self.log_wrap_count, int(event.get("wrap_count", 0)))
            self._expect_log_sequence_one = True
            return

        if self._expect_log_sequence_one:
            if sequence != 1:
                self.log_complete = False
                self.log_gaps.append((1, max(1, sequence - 1)))
            self._expect_log_sequence_one = False
        elif self.last_log_sequence is not None:
            expected = self.last_log_sequence + 1
            if sequence != expected:
                self.log_complete = False
                if sequence > expected:
                    self.log_gaps.append((expected, sequence - 1))
                else:
                    self.log_gaps.append((sequence, self.last_log_sequence))
        self.last_log_sequence = sequence

    def ingest_line(self, line: str | bytes) -> dict[str, Any] | None:
        event = parse_neko_log_line(line)
        if event is not None:
            self.ingest(event)
        return event

    def ingest(self, event: Mapping[str, Any]) -> None:
        """吸收一条已解析事件；未知 type 保留为最近事件但不报错。"""
        event_copy = dict(event)
        event_type = event_copy.get("type")
        if not isinstance(event_type, str):
            raise ValueError("event.type 必须是字符串")
        with self._condition:
            self._track_log_sequence(event_copy)
            self.world_id = str(event_copy.get("world_id") or self.world_id or "") or None

            if event_type == "npc.ack":
                self._ack_generation += 1
                ack = replace(YuiAck.from_event(event_copy), arrival_index=self._ack_generation)
                self._acks.append(ack)
                self.control_state = ack.state
                self.estop = ack.state == "estop"
                if self.estop or ack.error in {"not_owner", "ownership_failed"}:
                    self.host_arm_authorized = False
                self._condition.notify_all()
            elif event_type == "sys.session":
                new_session = int(event_copy["new_session"])
                previous_session = self.session
                if new_session != previous_session:
                    preserved_estop = bool(event_copy.get("estop_preserved", False))
                    self._reset_for_new_session(new_session)
                    if preserved_estop:
                        self.estop = True
                        self.control_state = "estop"
                self.driver_pid = int(event_copy["driver_pid"])
            elif event_type == "sys.hello":
                self.session = int(event_copy["session"])
                self.world_name = str(event_copy["world_name"])
                self.capabilities = tuple(str(item) for item in event_copy.get("caps", []))
                self.capability_bits = int(event_copy.get("cap_bits", 0))
                self.catalog_revision = int(event_copy["catalog_rev"])
                wire_bounds = event_copy.get("wire_bounds")
                activity_bounds = event_copy.get("activity_bounds")
                if isinstance(wire_bounds, list) and len(wire_bounds) == 6:
                    self.wire_bounds = tuple(float(item) for item in wire_bounds)
                if isinstance(activity_bounds, list) and len(activity_bounds) == 6:
                    self.activity_bounds = tuple(float(item) for item in activity_bounds)
                self.max_speed_mps = float(event_copy["max_speed"])
            elif event_type == "sys.catalog":
                self._ingest_catalog(event_copy)
            elif event_type == "player.join":
                slot = int(event_copy["slot"])
                self.players[slot] = {
                    "slot": slot,
                    "pid": int(event_copy["pid"]),
                    "name": str(event_copy["name"]),
                }
            elif event_type == "player.leave":
                slot = int(event_copy["slot"])
                leaving = self.players.pop(slot, None)
                leaving_pid = int(event_copy["pid"])
                if leaving_pid == self.driver_pid or (leaving and leaving.get("pid") == self.driver_pid):
                    self.host_arm_authorized = False
                    self.driver_pid = None
            elif event_type == "player.pose":
                for player in event_copy.get("players", []):
                    if not isinstance(player, Mapping) or "slot" not in player:
                        continue
                    slot = int(player["slot"])
                    merged = dict(self.players.get(slot, {"slot": slot}))
                    merged.update(dict(player))
                    self.players[slot] = merged
            elif event_type == "npc.state":
                self.npc_state = {
                    key: value
                    for key, value in event_copy.items()
                    if key not in {"v", "spec", "session", "world_id", "npc", "log_seq", "t", "type"}
                }
                self.control_state = str(event_copy["state"])
                self.estop = bool(event_copy["estop"])
                if self.estop:
                    self.host_arm_authorized = False
            elif event_type.startswith("npc.operation_"):
                self._ingest_operation(event_copy)
            elif event_type == "voice.state":
                self.voice_state.update({
                    "state": event_copy.get("state"),
                    "error_code": event_copy.get("error_code"),
                    "url_loaded": bool(event_copy.get("url_loaded", False)),
                })
            elif event_type == "voice.cue":
                self.voice_state["last_speech_seq"] = event_copy.get("speech_seq")
            elif event_type == "npc.text_displayed":
                self.text_state.update({
                    "transfer_seq": event_copy.get("transfer_seq"),
                    "utf8_bytes": event_copy.get("utf8_bytes", 0),
                    "crc16": event_copy.get("crc16"),
                    "text": event_copy.get("text"),
                })
            elif event_type == "npc.text_cleared":
                self.text_state.update({
                    "transfer_seq": None,
                    "utf8_bytes": 0,
                    "crc16": None,
                    "display_until_server_ms": None,
                    "text": None,
                })
            elif event_type == "sys.watchdog":
                self.control_state = "safe_idle"
                self.host_arm_authorized = False
            elif event_type == "sys.snapshot":
                self._ingest_snapshot_part(event_copy)

            if event_type == "player.touch" or event_type.startswith(("touch.", "social.")):
                self._recent_events.append(event_copy)

            # 宿主诊断等待器依赖状态/操作事件，而不仅是 ACK；所有有效事件均唤醒。
            self._condition.notify_all()

            listeners = tuple(self._event_listeners)

        # 监听器可能触发宿主 IPC，必须在状态锁之外调用。
        for listener in listeners:
            try:
                listener(dict(event_copy))
            except Exception:
                # 日志投影是唯一事实源；外部通知失败不能阻断后续事件。
                continue

    def _ingest_catalog(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("kind"))
        if kind not in self.catalogs:
            return
        page = int(event.get("page", 1))
        pages = int(event.get("pages", 1))
        self._catalog_pages.setdefault(kind, set()).add(page)
        self._catalog_expected_pages[kind] = pages
        for item in event.get("items", []):
            if isinstance(item, Mapping) and isinstance(item.get("id"), int):
                self.catalogs[kind][int(item["id"])] = dict(item)

    def _ingest_operation(self, event: Mapping[str, Any]) -> None:
        op_id = event.get("op_id")
        if not isinstance(op_id, str):
            return
        record = dict(self.operations.get(op_id, {}))
        record.update({
            "op_id": op_id,
            "kind": event.get("kind"),
            "request_seq": event.get("request_seq"),
            "request_hash": event.get("request_hash"),
        })
        event_type = str(event["type"])
        if event_type == "npc.operation_started":
            record.update({"status": "running", "expected_end_ms": event.get("expected_end_ms")})
        elif event_type == "npc.operation_completed":
            record.update({
                "status": "succeeded",
                "elapsed_ms": event.get("elapsed_ms"),
                "result": event.get("result"),
            })
        elif event_type == "npc.operation_cancelled":
            record.update({
                "status": "cancelled",
                "elapsed_ms": event.get("elapsed_ms"),
                "reason": event.get("reason"),
            })
        self.operations[op_id] = record

    def _ingest_snapshot_part(self, event: Mapping[str, Any]) -> None:
        snapshot_sequence = int(event["snapshot_seq"])
        part = int(event["part"])
        parts = int(event["parts"])
        key = (int(event["session"]), snapshot_sequence)
        bucket = self._snapshot_parts.setdefault(key, {})
        bucket[part] = dict(event)
        self._snapshot_part_counts[key] = parts
        if len(bucket) != parts or any(index not in bucket for index in range(1, parts + 1)):
            return
        # 完整快照描述的是某一时刻的权威集合，不能保留上一快照中已经消失的玩家。
        if any(str(item.get("section")) == "players" for item in bucket.values()):
            self.players.clear()
        for index in range(1, parts + 1):
            item = bucket[index]
            self._apply_snapshot_section(str(item["section"]), item.get("data"))
        self._snapshot_parts.pop(key, None)
        self._snapshot_part_counts.pop(key, None)

    def _apply_snapshot_section(self, section: str, data: Any) -> None:
        if not isinstance(data, Mapping):
            return
        if section == "session":
            self.driver_pid = data.get("driver_pid") if isinstance(data.get("driver_pid"), int) else None
            self.control_state = str(data.get("control_state", self.control_state))
            self.estop = bool(data.get("estop", self.estop))
            self.capabilities = tuple(str(item) for item in data.get("caps", self.capabilities))
            if self.estop or self.driver_pid is None:
                self.host_arm_authorized = False
        elif section == "npc":
            self.npc_state = dict(data)
        elif section == "players":
            for player in data.get("players", []):
                if isinstance(player, Mapping) and isinstance(player.get("slot"), int):
                    self.players[int(player["slot"])] = dict(player)
        elif section == "voice":
            self.voice_state = dict(data)
        elif section == "text":
            self.text_state = dict(data)

    def find_ack(
        self,
        sequence: int,
        command_id: int,
        request_hash: str,
        *,
        session: int | None = None,
        after_arrival_index: int = 0,
    ) -> YuiAck | None:
        normalized_hash = request_hash.upper()
        with self._condition:
            for ack in reversed(self._acks):
                if (
                    ack.sequence == sequence
                    and ack.command_id == command_id
                    and ack.request_hash == normalized_hash
                    and (session is None or ack.session == session)
                    and ack.arrival_index > after_arrival_index
                ):
                    return ack
        return None

    def wait_for_ack(
        self,
        sequence: int,
        command_id: int,
        request_hash: str,
        timeout_s: float,
        *,
        session: int | None = None,
        after_arrival_index: int = 0,
    ) -> YuiAck | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while True:
                ack = self.find_ack(
                    sequence,
                    command_id,
                    request_hash,
                    session=session,
                    after_arrival_index=after_arrival_index,
                )
                if ack is not None:
                    return ack
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def wait_for_operation(self, operation_id: str, timeout_s: float) -> dict[str, Any] | None:
        """等待宿主诊断操作进入终态；超时返回 None，不臆测成功。"""
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id 必须是非空字符串")
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        terminal = {"succeeded", "cancelled", "failed", "unknown"}
        with self._condition:
            while True:
                operation = self.operations.get(operation_id)
                if operation is not None and operation.get("status") in terminal:
                    return dict(operation)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def wait_for_npc_near(
        self,
        x: float,
        z: float,
        distance_m: float,
        timeout_s: float,
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any] | None:
        """等待 NPC 进入预切半径；若关联操作先结束或超时则返回 None。"""
        radius = float(distance_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("distance_m 必须是正有限数")
        target_x = float(x)
        target_z = float(z)
        if not math.isfinite(target_x) or not math.isfinite(target_z):
            raise ValueError("x/z 必须是有限数")
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        terminal = {"succeeded", "cancelled", "failed", "unknown"}
        with self._condition:
            while True:
                state = dict(self.npc_state)
                position = state.get("pos")
                if isinstance(position, list) and len(position) == 3:
                    dx = float(position[0]) - target_x
                    dz = float(position[2]) - target_z
                    if dx * dx + dz * dz <= radius * radius:
                        return state
                if operation_id is not None:
                    operation = self.operations.get(operation_id)
                    if operation is not None and operation.get("status") in terminal:
                        return None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def observe(self, *, include_player_names: bool = False) -> dict[str, Any]:
        """生成 §17.3 的最小观察摘要，默认不暴露真实显示名。"""
        with self._condition:
            active_ops = self.npc_state.get("active_ops", [])
            if not self.operation_lifecycle or not isinstance(active_ops, list):
                active_ops = []
            npc = dict(self.npc_state)
            npc["active_ops"] = list(active_ops)
            players = []
            for slot in sorted(self.players):
                source = self.players[slot]
                item = {
                    key: source.get(key)
                    for key in ("slot", "d", "brg", "yaw", "vr")
                    if key in source
                }
                if include_player_names and "name" in source:
                    item["name"] = source["name"]
                players.append(item)
            return {
                "session": self.session,
                "world_id": self.world_id,
                "control_state": self.control_state,
                "estop": self.estop,
                "caps": list(self.capabilities),
                "operation_lifecycle": self.operation_lifecycle,
                "active_ops_authoritative": self.active_ops_authoritative,
                "active_ops": list(active_ops),
                "npc": npc,
                "players": players,
                "recent_social_events": [
                    {
                        key: value
                        for key, value in event.items()
                        if key not in {"name", "pid"}
                    }
                    for event in self._recent_events
                ],
                "voice": dict(self.voice_state),
                "text": dict(self.text_state),
                "semantic_keys": {
                    kind: [
                        item.get("semantic_key")
                        for _item_id, item in sorted(catalog.items())
                        if isinstance(item.get("semantic_key"), str)
                    ]
                    for kind, catalog in self.catalogs.items()
                    if kind in {"action", "expression", "anchor"}
                },
                "log_complete": self.log_complete,
                "log_gaps": list(self.log_gaps),
            }


__all__ = ["YuiAck", "YuiSessionState"]
