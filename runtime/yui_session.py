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
from typing import Any, Callable, Iterable, Mapping

from .yui_protocol import COMMAND_NAMES, normalize_bearing, parse_neko_log_line


def _spec_at_least(spec: str | None, minimum_minor: int) -> bool:
    """只比较冻结的 1.x 小版本；未知格式采取失败关闭。"""
    if not isinstance(spec, str):
        return False
    parts = spec.split(".")
    return len(parts) == 2 and parts[0] == "1" and parts[1].isdigit() and int(parts[1]) >= minimum_minor


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
        self._event_history: deque[tuple[float, dict[str, Any]]] = deque(maxlen=128)
        self._snapshot_parts: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
        self._snapshot_part_counts: dict[tuple[int, int], int] = {}
        self._catalog_pages: dict[str, set[int]] = {}
        self._catalog_expected_pages: dict[str, int] = {}
        self._catalog_expected_counts: dict[str, int] = {}
        self._hello_session = 0
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []
        self.session = 0
        self.spec_version: str | None = None
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
            "region": {},
            "entity": {},
            "route_edge": {},
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

    @property
    def world_map_ready(self) -> bool:
        return _spec_at_least(self.spec_version, 2) and "world_map" in self.capabilities

    @property
    def semantic_navigation(self) -> bool:
        return _spec_at_least(self.spec_version, 2) and "semantic_navigation" in self.capabilities

    @property
    def region_localization(self) -> bool:
        return _spec_at_least(self.spec_version, 3) and "region_localization" in self.capabilities

    @property
    def local_navigation(self) -> bool:
        return _spec_at_least(self.spec_version, 3) and "local_navigation" in self.capabilities

    @property
    def discovery_ready(self) -> bool:
        """当前会话的 hello 与声明目录是否已经完整投影。"""
        with self._condition:
            return self._discovery_ready_locked(self.session)

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
        self._catalog_expected_counts.clear()
        self._hello_session = 0
        self.players.clear()
        self.npc_state = {"active_ops": []}
        self.operations.clear()
        self._snapshot_parts.clear()
        self._snapshot_part_counts.clear()

        # 授权只能绑定已经明确收到的当前 session；首次建联也必须清除预授权。
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
            event_spec = event_copy.get("spec")
            if isinstance(event_spec, str):
                self.spec_version = event_spec

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
                self._hello_session = self.session
                self.world_name = str(event_copy["world_name"])
                self.capabilities = tuple(str(item) for item in event_copy.get("caps", []))
                self.capability_bits = int(event_copy.get("cap_bits", 0))
                self.catalog_revision = int(event_copy["catalog_rev"])
                counts = event_copy.get("catalog_counts")
                self._catalog_expected_counts = {
                    str(kind): int(count)
                    for kind, count in counts.items()
                    if kind in self.catalogs
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                } if isinstance(counts, Mapping) else {}
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
            self._event_history.append((time.monotonic(), event_copy))

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
        elif event_type == "npc.operation_failed":
            record.update({
                "status": "failed",
                "elapsed_ms": event.get("elapsed_ms"),
                "error": event.get("err"),
                "detail": event.get("detail"),
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

    def wait_for_session(self, session: int, timeout_s: float) -> bool:
        """等待 ``sys.session`` 把指定会话投影为当前会话。

        Unity 对 DISCOVER 的日志顺序是先 ``npc.ack``、后 ``sys.session``。宿主授权
        必须等后者落地，否则新会话的安全重置会立即清掉刚写入的授权。
        """
        expected = int(session)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while self.session != expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def _discovery_ready_locked(self, session: int) -> bool:
        if self.session != session or self._hello_session != session or self.catalog_revision is None:
            return False
        return all(
            len(self.catalogs[kind]) == expected
            for kind, expected in self._catalog_expected_counts.items()
        )

    def wait_for_discovery(self, session: int, timeout_s: float) -> bool:
        """等待当前会话的 ``sys.hello`` 和其声明的全部目录项。

        ``sys.session`` 只证明安全边界已经切换；能力和目录在随后到达。宿主只有
        等这些事实完整投影后才能宣布连接完成，否则刚出现的语义工具会误报
        ``target_missing``。
        """
        expected = int(session)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while not self._discovery_ready_locked(expected):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

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

    def anchor_view(self, *, limit: int = 8) -> dict[str, Any]:
        """按距离给出有界的 anchor 语义视图；绝不输出绝对坐标。

        规范 §17.1 要求不得让模型复制 anchor 坐标，因此 pos/yaw 只用于本地
        计算相对量，不进入返回值。位置未知时省略几何而不是估算。
        """
        bound = max(1, int(limit))
        with self._condition:
            catalog = self.catalogs.get("anchor", {})
            total = len(catalog)
            npc_position = self.npc_state.get("pos")
            npc_yaw = self.npc_state.get("yaw")
            position_known = (
                isinstance(npc_position, list)
                and len(npc_position) == 3
                and all(isinstance(item, (int, float)) for item in npc_position)
                and isinstance(npc_yaw, (int, float))
            )

            entries: list[tuple[float | None, int, dict[str, Any]]] = []
            for anchor_id, item in catalog.items():
                semantic_key = item.get("semantic_key")
                if not isinstance(semantic_key, str):
                    continue
                view: dict[str, Any] = {"semantic_key": semantic_key}
                description = item.get("description_zh")
                if isinstance(description, str):
                    view["description_zh"] = description
                tags = item.get("tags")
                if isinstance(tags, list):
                    view["tags"] = [tag for tag in tags if isinstance(tag, str)]

                distance: float | None = None
                anchor_position = item.get("pos")
                if position_known and isinstance(anchor_position, list) and len(anchor_position) == 3:
                    dx = float(anchor_position[0]) - float(npc_position[0])
                    dz = float(anchor_position[2]) - float(npc_position[2])
                    distance = math.hypot(dx, dz)
                    view["d"] = round(distance, 1)
                    # 世界方位角按 X 东 / Z 北，减去朝向得相对角；brg 右正。
                    world_bearing = math.degrees(math.atan2(dx, dz))
                    view["brg"] = round(normalize_bearing(world_bearing - float(npc_yaw)))
                entries.append((distance, int(anchor_id), view))

            if position_known:
                entries.sort(key=lambda entry: (entry[0] is None, entry[0] or 0.0, entry[1]))
            else:
                entries.sort(key=lambda entry: entry[1])

            selected = entries[:bound]
            return {
                "anchors": [view for _distance, _anchor_id, view in selected],
                "total": total,
                "omitted": max(0, total - len(selected)),
                "position_known": position_known,
            }

    @staticmethod
    def _safe_tags(value: Any) -> list[str]:
        return [str(item) for item in value] if isinstance(value, list) else []

    @staticmethod
    def _safe_relative(
        position: Any,
        npc_position: Any,
        npc_yaw: Any,
    ) -> tuple[float, int] | None:
        if not (
            isinstance(position, list)
            and len(position) == 3
            and isinstance(npc_position, list)
            and len(npc_position) == 3
            and isinstance(npc_yaw, (int, float))
        ):
            return None
        try:
            dx = float(position[0]) - float(npc_position[0])
            dz = float(position[2]) - float(npc_position[2])
        except (TypeError, ValueError):
            return None
        world_bearing = math.degrees(math.atan2(dx, dz))
        return round(math.hypot(dx, dz), 1), round(normalize_bearing(world_bearing - float(npc_yaw)))

    def _semantic_projection(
        self,
        kind: str,
        item: Mapping[str, Any],
        *,
        include_relative: bool,
    ) -> dict[str, Any]:
        """投影给模型的目录项；绝不复制 center/pos/yaw 或内部数字 id。"""
        projected: dict[str, Any] = {
            "kind": kind,
            "semantic_key": item.get("semantic_key"),
        }
        for key in (
            "description_zh",
            "floor_label",
            "region_key",
            "traversal",
            "bidirectional",
            "explorable",
            "orbitable",
            "orbit_min_radius",
            "orbit_max_radius",
        ):
            if key in item:
                projected[key] = item[key]
        tags = self._safe_tags(item.get("tags"))
        if tags:
            projected["tags"] = tags
        if kind == "route_edge":
            from_id = item.get("from_anchor_id")
            to_id = item.get("to_anchor_id")
            from_item = self.catalogs["anchor"].get(from_id) if isinstance(from_id, int) else None
            to_item = self.catalogs["anchor"].get(to_id) if isinstance(to_id, int) else None
            if from_item is not None:
                projected["from_key"] = from_item.get("semantic_key")
            if to_item is not None:
                projected["to_key"] = to_item.get("semantic_key")
        if include_relative:
            position = item.get("pos") if kind == "anchor" else item.get("center")
            if kind == "region":
                entry_id = item.get("entry_anchor_id")
                entry = self.catalogs["anchor"].get(entry_id) if isinstance(entry_id, int) else None
                position = None if entry is None else entry.get("pos")
            relative = self._safe_relative(
                position,
                self.npc_state.get("pos"),
                self.npc_state.get("yaw"),
            )
            if relative is not None:
                projected["d"], projected["brg"] = relative
        return projected

    def nearby_world(self, *, limit: int = 8) -> dict[str, Any]:
        """返回 v1.2 附近语义摘要；v1.1 固定为 unavailable。"""
        bound = min(8, max(1, int(limit)))
        with self._condition:
            if not self.world_map_ready:
                return {"available": False, "items": [], "total": 0, "omitted": 0}
            candidates: list[tuple[float, str, dict[str, Any]]] = []
            for kind in ("region", "entity", "anchor"):
                for item in self.catalogs[kind].values():
                    projection = self._semantic_projection(kind, item, include_relative=True)
                    distance = projection.get("d")
                    candidates.append((float(distance) if isinstance(distance, (int, float)) else math.inf, str(projection.get("semantic_key", "")), projection))
            candidates.sort(key=lambda value: (value[0], value[1]))
            selected = [item for _distance, _key, item in candidates[:bound]]
            return {
                "available": True,
                "items": selected,
                "total": len(candidates),
                "omitted": max(0, len(candidates) - len(selected)),
            }

    def _target_anchor_id(self, semantic_key: str) -> int | None:
        for anchor_id, item in self.catalogs["anchor"].items():
            if item.get("semantic_key") == semantic_key:
                return anchor_id
        for kind, field in (("entity", "approach_anchor_id"), ("region", "entry_anchor_id")):
            for item in self.catalogs[kind].values():
                if item.get("semantic_key") == semantic_key and isinstance(item.get(field), int):
                    return int(item[field])
        return None

    def _reachable_anchor_ids(self, source_key: str) -> set[int]:
        source = self._target_anchor_id(source_key)
        if source is None and source_key == "npc":
            npc_position = self.npc_state.get("pos")
            nearest: tuple[float, int] | None = None
            for anchor_id, item in self.catalogs["anchor"].items():
                relative = self._safe_relative(item.get("pos"), npc_position, self.npc_state.get("yaw"))
                if relative is not None and (nearest is None or relative[0] < nearest[0]):
                    nearest = (relative[0], anchor_id)
            source = None if nearest is None else nearest[1]
        if source is None:
            return set()
        adjacency: dict[int, set[int]] = {}
        for edge in self.catalogs["route_edge"].values():
            start = edge.get("from_anchor_id")
            end = edge.get("to_anchor_id")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            adjacency.setdefault(start, set()).add(end)
            if bool(edge.get("bidirectional")):
                adjacency.setdefault(end, set()).add(start)
        reached = {source}
        frontier = [source]
        while frontier:
            current = frontier.pop(0)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        return reached

    def world_query(
        self,
        *,
        query: str | None = None,
        region_key: str | None = None,
        tags: Iterable[str] | None = None,
        reachable_from: str | None = None,
        kinds: Iterable[str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """对作者发布的语义地图执行确定性查询，静态可达性只来自 route_edge。"""
        with self._condition:
            if not self.world_map_ready:
                return {"status": "failed", "error": "unsupported_capability", "detail": "世界未发布 v1.2 world_map", "midi_sent": False}
            allowed_kinds = {"region", "entity", "anchor", "route_edge"}
            selected_kinds = set(kinds or allowed_kinds)
            if not selected_kinds or not selected_kinds <= allowed_kinds:
                return {"status": "failed", "error": "invalid_param", "detail": "kinds 包含未知目录类型", "midi_sent": False}
            requested_tags = {str(item).casefold() for item in (tags or [])}
            needle = (query or "").strip().casefold()
            reachable = self._reachable_anchor_ids(reachable_from) if reachable_from else None
            results: list[dict[str, Any]] = []
            for kind in ("region", "entity", "anchor", "route_edge"):
                if kind not in selected_kinds:
                    continue
                for _item_id, item in sorted(self.catalogs[kind].items()):
                    if region_key and item.get("region_key") != region_key:
                        continue
                    item_tags = {tag.casefold() for tag in self._safe_tags(item.get("tags"))}
                    if requested_tags and not requested_tags <= item_tags:
                        continue
                    if needle:
                        haystack = " ".join((str(item.get("semantic_key", "")), str(item.get("description_zh", "")), *sorted(item_tags))).casefold()
                        if needle not in haystack:
                            continue
                    if reachable is not None:
                        anchor_id = None
                        if kind == "anchor":
                            anchor_id = item.get("id")
                        elif kind == "entity":
                            anchor_id = item.get("approach_anchor_id")
                        elif kind == "region":
                            anchor_id = item.get("entry_anchor_id")
                        elif kind == "route_edge":
                            anchor_id = item.get("from_anchor_id")
                        if not isinstance(anchor_id, int) or anchor_id not in reachable:
                            continue
                    results.append(self._semantic_projection(kind, item, include_relative=True))
            results.sort(key=lambda item: (str(item.get("kind")), str(item.get("semantic_key", item.get("from_key", "")))))
            try:
                offset = int(cursor or "0")
            except ValueError:
                return {"status": "failed", "error": "invalid_param", "detail": "cursor 必须是十进制偏移", "midi_sent": False}
            page_size = min(20, max(1, int(limit)))
            if offset < 0:
                return {"status": "failed", "error": "invalid_param", "detail": "cursor 不得为负数", "midi_sent": False}
            page = results[offset:offset + page_size]
            next_offset = offset + len(page)
            return {
                "status": "succeeded",
                "items": page,
                "total": len(results),
                "next_cursor": str(next_offset) if next_offset < len(results) else None,
                "static_reachability": reachable_from is not None,
                "midi_sent": False,
            }

    def has_recent_event(self, event_type: str, within_ms: int) -> bool:
        """供行为图白名单条件查询；只做精确 type 匹配。"""
        threshold = time.monotonic() - max(1, int(within_ms)) / 1000.0
        with self._condition:
            return any(arrived >= threshold and event.get("type") == event_type for arrived, event in self._event_history)

    @classmethod
    def _without_absolute_coordinates(cls, value: Any) -> Any:
        """v1.2+ 模型投影移除绝对世界坐标，保留 d/brg/yaw 等相对事实。"""
        absolute_keys = {
            "pos",
            "target",
            "target_pos",
            "center",
            "origin",
            "hit_pos",
            "x",
            "y",
            "z",
        }
        if isinstance(value, Mapping):
            return {
                str(key): cls._without_absolute_coordinates(item)
                for key, item in value.items()
                if key not in absolute_keys
            }
        if isinstance(value, list):
            return [cls._without_absolute_coordinates(item) for item in value]
        return value

    def _location_projection_locked(self) -> dict[str, Any]:
        """只信任 Unity 发布的 Region 命中事实，不从 Anchor 或坐标反推楼层。"""
        unavailable = {
            "localized": False,
            "region_key": None,
            "floor_label": None,
            "nearest_anchor": None,
        }
        if not self.region_localization:
            return unavailable
        raw = self.npc_state.get("location")
        if not isinstance(raw, Mapping):
            return unavailable
        localized = raw.get("localized") is True
        projected: dict[str, Any] = {
            "localized": localized,
            "region_key": raw.get("region_key") if localized and isinstance(raw.get("region_key"), str) else None,
            "floor_label": raw.get("floor_label") if localized and isinstance(raw.get("floor_label"), str) else None,
            "nearest_anchor": None,
        }
        nearest = raw.get("nearest_anchor")
        if isinstance(nearest, Mapping) and isinstance(nearest.get("semantic_key"), str):
            item: dict[str, Any] = {"semantic_key": nearest["semantic_key"]}
            distance = nearest.get("d")
            bearing = nearest.get("brg")
            if isinstance(distance, (int, float)) and math.isfinite(float(distance)) and float(distance) >= 0.0:
                item["d"] = round(float(distance), 1)
            if isinstance(bearing, (int, float)) and math.isfinite(float(bearing)):
                item["brg"] = round(normalize_bearing(float(bearing)))
            projected["nearest_anchor"] = item
        return projected

    def observe(self, *, include_player_names: bool = False) -> dict[str, Any]:
        """生成 §17.3 的最小观察摘要，默认不暴露真实显示名。"""
        with self._condition:
            active_ops = self.npc_state.get("active_ops", [])
            if not self.operation_lifecycle or not isinstance(active_ops, list):
                active_ops = []
            npc = dict(self.npc_state)
            npc["active_ops"] = list(active_ops)
            hide_absolute_coordinates = _spec_at_least(self.spec_version, 2)
            if hide_absolute_coordinates:
                npc = self._without_absolute_coordinates(npc)
                active_ops_view = self._without_absolute_coordinates(list(active_ops))
            else:
                active_ops_view = list(active_ops)
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
            observation = {
                "spec": self.spec_version,
                "session": self.session,
                "world_id": self.world_id,
                "control_state": self.control_state,
                "estop": self.estop,
                "caps": list(self.capabilities),
                "operation_lifecycle": self.operation_lifecycle,
                "active_ops_authoritative": self.active_ops_authoritative,
                "active_ops": active_ops_view,
                "npc": npc,
                "players": players,
                "recent_social_events": [
                    self._without_absolute_coordinates({
                        key: value
                        for key, value in event.items()
                        if key not in {"name", "pid"}
                    }) if hide_absolute_coordinates else {
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
                    if kind in {"action", "expression", "anchor", "region", "entity"}
                },
                "catalog_rev": self.catalog_revision,
                "catalog_counts": {
                    kind: len(catalog)
                    for kind, catalog in self.catalogs.items()
                },
                "log_complete": self.log_complete,
                # N.E.K.O 的 MessagePack 传输层会拒绝 tuple；对外结果必须保持
                # JSON 原生类型，不能把内部的区间 tuple 直接泄漏给工具调用方。
                "log_gaps": [[start, end] for start, end in self.log_gaps],
            }
            if _spec_at_least(self.spec_version, 3):
                observation["location"] = self._location_projection_locked()
            return observation


__all__ = ["YuiAck", "YuiSessionState"]
