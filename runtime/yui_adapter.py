"""独立 YUI NPC 插件的语义适配层。

LLM 只接触 semantic_key、anchor_key 和 player_slot；量化值、命令 ID、CRC 与
MIDI 事件全部留在确定性 Python 代码中。世界 ACK 始终是最终权威。
"""

from __future__ import annotations

import math
import secrets
import threading
from typing import Any, Mapping
import uuid

from .behavior_plan import BehaviorPlanManager, single_node_graph
from .yui_protocol import (
    SESSION_MAX,
    encode_position_q14,
    encode_speed_q7,
    encode_yaw_q14,
)
from .yui_session import YuiSessionState
from .yui_transport import YuiCommandOutcome, YuiReliableTransport


def _local_result(error: str, detail: str, **extra: Any) -> dict[str, Any]:
    result = {
        "request_id": f"host-{uuid.uuid4()}",
        "status": "failed",
        "error": error,
        "detail": detail,
        "midi_sent": False,
    }
    result.update(extra)
    return result


class YuiSemanticAdapter:
    """把冻结的最小工具面展开为串行可靠命令。"""

    def __init__(
        self,
        transport: YuiReliableTransport,
        session: YuiSessionState,
        *,
        free_coordinate_navigation: bool = False,
    ) -> None:
        self.transport = transport
        self.session = session
        self.free_coordinate_navigation = bool(free_coordinate_navigation)
        self._semantic_lock = threading.RLock()
        self._next_action_sequence = 1
        self._next_transfer_sequence = 1
        self._next_speech_sequence = 1
        # 只记录本 Adapter 实例亲自完成的握手。新建宿主控制器不能借用日志中
        # 残留的旧 session 跳过 DISCOVER，但同一持久实例重复点“连接”应立即返回。
        self._connected_session = 0
        self.plan_manager = BehaviorPlanManager(self, session)

    @staticmethod
    def _advance_14bit(value: int) -> int:
        return 1 if value >= 16383 else value + 1

    def _allocate_action_sequence(self) -> int:
        value = self._next_action_sequence
        self._next_action_sequence = self._advance_14bit(value)
        return value

    def _allocate_transfer_sequence(self) -> int:
        value = self._next_transfer_sequence
        self._next_transfer_sequence = self._advance_14bit(value)
        return value

    def _allocate_speech_sequence(self) -> int:
        value = self._next_speech_sequence
        self._next_speech_sequence = self._advance_14bit(value)
        return value

    def _require_tool(self, *capabilities: str, operation: bool = False) -> dict[str, Any] | None:
        if self.session.session <= 0:
            return _local_result("not_handshaken", "尚未完成 DISCOVER")
        missing = [item for item in capabilities if item not in self.session.capabilities]
        if missing:
            return _local_result(
                "unsupported_capability",
                f"世界未发布 capability: {', '.join(missing)}",
            )
        if operation and not self.session.operation_lifecycle:
            return _local_result(
                "unsupported_capability",
                "世界未发布 operation_lifecycle，不能向 LLM 暴露长操作工具",
            )
        if self.session.control_state == "safe_idle":
            return _local_result(
                "control_not_ready",
                "当前为 safe_idle；地图 NPC 应由宿主重新连接并自动进入 external",
            )
        if self.session.control_state == "estop":
            return _local_result("estop_latched", "ESTOP 已锁存，只能由人工安全路径清除")
        if self.session.control_state not in {"external", "moving", "action"}:
            return _local_result("invalid_state", f"当前状态 {self.session.control_state} 不允许执行")
        return None

    @staticmethod
    def _outcome(outcome: YuiCommandOutcome, **extra: Any) -> dict[str, Any]:
        result = outcome.as_dict()
        result["request_id"] = f"host-{uuid.uuid4()}"
        result["midi_sent"] = True
        result.update(extra)
        return result

    def _send_command(
        self, command: str, parameters: Any
    ) -> "YuiCommandOutcome | dict[str, Any]":
        """发送普通命令；把本地编码错误转为 _local_result dict 而不是抛出异常。"""
        from .yui_protocol import YuiProtocolError
        try:
            outcome = self.transport.send_command(command, parameters)
            if outcome.error in {"not_owner", "ownership_failed", "session_conflict"}:
                self._connected_session = 0
            return outcome
        except YuiProtocolError as exc:
            return _local_result("invalid_param", str(exc))

    def _send_text(self, text: str, *, transfer_sequence: int, display_seconds: int) -> Any:
        """发送文本事务；本地编码错误是已知失败，不得伪装成 unknown。"""
        from .yui_protocol import YuiProtocolError
        try:
            return self.transport.send_text(
                text,
                transfer_sequence=transfer_sequence,
                display_seconds=display_seconds,
            )
        except YuiProtocolError as exc:
            bad = YuiCommandOutcome(
                status="failed",
                kind=None,
                wire_sequence=0,
                request_hash="0000",
                operation_id=None,
                error="invalid_param",
                detail=str(exc),
                ack_replayed=False,
            )
            return type("_FailedText", (), {
                "transaction": None,
                "begin": bad,
                "commit": None,
                "status": "unknown",
            })()

    def connect(self, claim_code: int, *, session: int | None = None) -> dict[str, Any]:
        """建立地图 NPC 会话并由宿主自动进入 ``external``。"""
        if isinstance(claim_code, bool) or not isinstance(claim_code, int) or not 0 <= claim_code <= 16383:
            return _local_result("invalid_param", "claim_code 必须是 0..16383 的整数")
        if (
            session is None
            and self._connected_session > 0
            and self._connected_session == self.session.session
            and self.session.discovery_ready
        ):
            # 宿主是长驻进程；重复连接若再次 DISCOVER，会强制世界重放完整目录，
            # 不但增加数秒等待，还会让宿主反复重建动态工具。当前实例和 session
            # 都健康时直接复用；断开会创建新 Adapter，owner 错误也会清除此标记。
            if self.session.control_state == "estop":
                return _local_result(
                    "estop_latched",
                    "ESTOP 已锁存，只能由人工安全入口清除",
                    requested_session=self.session.session,
                    already_connected=True,
                )
            if self.session.control_state == "safe_idle":
                with self._semantic_lock:
                    control_outcome = self.transport.send_command(
                        "SET_CONTROL_MODE", (0, 0, 0, 1, 0, 0)
                    )
                if control_outcome.status == "succeeded":
                    self.transport.set_heartbeat_enabled(True)
                    self.transport.start_heartbeat()
                return self._outcome(
                    control_outcome,
                    requested_session=self.session.session,
                    already_connected=True,
                    auto_control=control_outcome.status == "succeeded",
                    rediscovered=False,
                )
            self.transport.set_heartbeat_enabled(True)
            self.transport.start_heartbeat()
            return {
                "request_id": f"host-{uuid.uuid4()}",
                "status": "succeeded",
                "already_connected": True,
                "requested_session": self.session.session,
                "midi_sent": False,
                "error": None,
            }
        session_value = session if session is not None else secrets.randbelow(SESSION_MAX) + 1
        if isinstance(session_value, bool) or not isinstance(session_value, int) or not 1 <= session_value <= SESSION_MAX:
            return _local_result("invalid_param", "session 必须是非零 28-bit 整数")
        parameters = (
            session_value & 0x3FFF,
            (session_value >> 14) & 0x3FFF,
            claim_code,
            0,
            0,
            0,
        )
        with self._semantic_lock:
            self.session.set_host_arm_authorized(False)
            discover_outcome = self.transport.send_command("DISCOVER", parameters)
            if discover_outcome.status == "succeeded":
                # Unity 先写 DISCOVER ACK、再写 sys.session。只有具体会话已经落地，
                # 宿主才能安全继续目录同步和内部控制态切换。
                session_timeout = float(getattr(self.transport, "command_deadline_s", 5.0))
                if not self.session.wait_for_session(session_value, session_timeout):
                    return _local_result(
                        "session_event_timeout",
                        "DISCOVER 已 ACK，但未在期限内收到同一会话的 sys.session",
                        requested_session=session_value,
                    )
                # DISCOVER 目录由世界限速分批写入；正式 NEKO Home 的目录总量
                # 会稳定超过普通命令的 5 秒 deadline。ACK/session 仍使用命令
                # deadline，但完整目录必须给出独立宽限，避免自动连接不断换
                # session、让上一轮目录永远收不齐。
                discovery_timeout = max(session_timeout, 12.0)
                if not self.session.wait_for_discovery(session_value, discovery_timeout):
                    return _local_result(
                        "discovery_timeout",
                        "当前会话的 sys.hello 或声明目录未在期限内完整到达",
                        requested_session=session_value,
                    )
                # 地图 NPC 的活动范围和可用目标已由世界 capability、目录、NavMesh
                # 与 ownership 共同约束，不再把 ARM 作为 LLM 权限门。SET_CONTROL_MODE
                # 是宿主连接流程的内部可靠步骤，模型工具面从不暴露 npc.arm。
                control_outcome = self.transport.send_command(
                    "SET_CONTROL_MODE", (0, 0, 0, 1, 0, 0)
                )
                if control_outcome.status != "succeeded":
                    return self._outcome(
                        control_outcome,
                        requested_session=session_value,
                        already_connected=False,
                        auto_control=False,
                        discover_wire_seq=discover_outcome.wire_sequence,
                        discover_request_hash=discover_outcome.request_hash,
                    )
                self._connected_session = session_value
                # 完整连接后再启动心跳；首次模型操作不会与目录同步或 ARM 往返竞争。
                self.transport.set_heartbeat_enabled(True)
                self.transport.start_heartbeat()
                return self._outcome(
                    control_outcome,
                    requested_session=session_value,
                    already_connected=False,
                    auto_control=True,
                    discover_wire_seq=discover_outcome.wire_sequence,
                    discover_request_hash=discover_outcome.request_hash,
                )
            return self._outcome(
                discover_outcome,
                requested_session=session_value,
                already_connected=False,
                auto_control=False,
            )

    def observe(self, *, include_player_names: bool = False) -> dict[str, Any]:
        result = self.session.observe(include_player_names=include_player_names)
        if self.session.world_map_ready:
            result["world"] = self.session.nearby_world(limit=8)
        return result

    def world_query(self, **arguments: Any) -> dict[str, Any]:
        """只读查询作者发布的 v1.2 语义地图。"""
        return self.session.world_query(**arguments)

    def _goto_parameters(
        self,
        x: Any,
        z: Any,
        *,
        yaw: Any = None,
        speed_mps: Any = None,
    ) -> tuple[int, int, int, int, int, int]:
        bounds = self.session.wire_bounds
        activity = self.session.activity_bounds
        maximum_speed = self.session.max_speed_mps
        if bounds is None or activity is None or maximum_speed is None:
            raise ValueError("尚未收到包含 bounds/max_speed 的 sys.hello")
        x_value = float(x)
        z_value = float(z)
        if not all(math.isfinite(item) for item in (x_value, z_value)):
            raise ValueError("x/z 必须是有限数值")
        if not activity[0] <= x_value <= activity[3] or not activity[2] <= z_value <= activity[5]:
            raise ValueError("目标越出 activity bounds")
        speed = maximum_speed if speed_mps is None else float(speed_mps)
        has_yaw = yaw is not None
        return (
            encode_position_q14(x_value, bounds[0], bounds[3]),
            encode_position_q14(z_value, bounds[2], bounds[5]),
            encode_yaw_q14(yaw) if has_yaw else 0,
            encode_speed_q7(speed, maximum_speed),
            1 if has_yaw else 0,
            0,
        )

    def go_to(self, anchor_key: str, *, speed_mps: float | None = None) -> dict[str, Any]:
        blocked = self._require_tool("goto", "navmesh", "anchors", operation=True)
        if blocked is not None:
            return blocked
        if self.session.semantic_navigation:
            return self.navigate_wire(anchor_key, speed_mps=speed_mps)
        anchor = next(
            (
                item
                for item in self.session.catalogs["anchor"].values()
                if item.get("semantic_key") == anchor_key
            ),
            None,
        )
        if anchor is None:
            return _local_result("anchor_not_found", f"目录中没有 anchor {anchor_key!r}")
        position = anchor.get("pos")
        if not isinstance(position, list) or len(position) != 3:
            return _local_result("catalog_invalid", "anchor.pos 不是 XYZ 三元组")
        try:
            parameters = self._goto_parameters(
                position[0],
                position[2],
                yaw=anchor.get("yaw") if anchor.get("has_yaw") else None,
                speed_mps=speed_mps,
            )
        except (TypeError, ValueError) as exc:
            return _local_result("invalid_param", str(exc))
        with self._semantic_lock:
            outcome = self._send_command("GOTO_XZ", parameters)
            if isinstance(outcome, dict):
                return outcome
        return self._outcome(outcome, semantic_key=anchor_key)

    def go_to_xyz(
        self,
        x: float,
        z: float,
        *,
        yaw: float | None = None,
        speed_mps: float | None = None,
    ) -> dict[str, Any]:
        if not self.free_coordinate_navigation:
            return _local_result(
                "tool_disabled",
                "free_coordinate_navigation 未显式启用，npc.go_to_xyz 不应出现在工具 schema",
            )
        blocked = self._require_tool("goto", "navmesh", operation=True)
        if blocked is not None:
            return blocked
        try:
            parameters = self._goto_parameters(x, z, yaw=yaw, speed_mps=speed_mps)
        except (TypeError, ValueError) as exc:
            return _local_result("invalid_param", str(exc))
        with self._semantic_lock:
            outcome = self._send_command("GOTO_XZ", parameters)
        return self._outcome(outcome)

    def _semantic_target_anchor(self, target_key: str) -> tuple[dict[str, Any] | None, str | None]:
        """把 Anchor/Entity/Region 的全局语义键解析为 Anchor 目录项。"""
        for anchor in self.session.catalogs["anchor"].values():
            if anchor.get("semantic_key") == target_key:
                return anchor, "anchor"
        for kind, field in (("entity", "approach_anchor_id"), ("region", "entry_anchor_id")):
            for item in self.session.catalogs[kind].values():
                if item.get("semantic_key") != target_key:
                    continue
                anchor_id = item.get(field)
                if isinstance(anchor_id, int):
                    return self.session.catalogs["anchor"].get(anchor_id), kind
                return None, kind
        return None, None

    def navigate_wire(self, target_key: str, *, speed_mps: float | None = None) -> dict[str, Any]:
        """v1.2 内部语义导航；目标 id 与速度不暴露给模型。"""
        blocked = self._require_tool(
            "goto", "navmesh", "anchors", "world_map", "semantic_navigation", operation=True,
        )
        if blocked is not None:
            return blocked
        if not isinstance(target_key, str) or not target_key:
            return _local_result("invalid_param", "target_key 必须是非空字符串")
        anchor, target_kind = self._semantic_target_anchor(target_key)
        if anchor is None:
            return _local_result("target_missing", f"目录中没有可导航目标 {target_key!r}")
        anchor_id = anchor.get("id")
        if isinstance(anchor_id, bool) or not isinstance(anchor_id, int) or not 0 <= anchor_id <= 126:
            return _local_result("catalog_invalid", "目标的 approach/entry anchor id 非法")
        maximum_speed = self.session.max_speed_mps
        if maximum_speed is None:
            return _local_result("not_ready", "尚未收到 max_speed")
        try:
            speed = maximum_speed if speed_mps is None else float(speed_mps)
            speed_q7 = encode_speed_q7(speed, maximum_speed)
        except (TypeError, ValueError) as exc:
            return _local_result("invalid_param", str(exc))
        with self._semantic_lock:
            outcome = self._send_command("GOTO_ANCHOR", (0, 0, 0, anchor_id, speed_q7, 0))
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(outcome, semantic_key=target_key, target_kind=target_kind)

    def orbit_wire(
        self,
        target_key: str,
        *,
        radius_m: float = 2.0,
        laps: int = 1,
        direction: str = "cw",
        speed_mps: float | None = None,
        face_target: bool = True,
    ) -> dict[str, Any]:
        """发送单条 ORBIT_ENTITY；圆周执行和无缝切点完全由 Unity 负责。"""
        blocked = self._require_tool(
            "goto", "navmesh", "world_map", "semantic_navigation", operation=True,
        )
        if blocked is not None:
            return blocked
        entity = next(
            (item for item in self.session.catalogs["entity"].values() if item.get("semantic_key") == target_key),
            None,
        )
        if entity is None or not bool(entity.get("orbitable")):
            return _local_result("target_missing", f"目录中没有可绕行实体 {target_key!r}")
        entity_id = entity.get("id")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int) or not 0 <= entity_id <= 126:
            return _local_result("catalog_invalid", "entity id 必须是 0..126")
        if isinstance(laps, bool) or not isinstance(laps, int) or not 1 <= laps <= 3:
            return _local_result("invalid_param", "laps 必须是 1..3 的整数")
        if direction not in {"cw", "ccw"}:
            return _local_result("invalid_param", "direction 必须是 cw|ccw")
        try:
            radius = float(radius_m)
            minimum = max(0.25, float(entity.get("orbit_min_radius", 0.25)))
            maximum = min(5.0, float(entity.get("orbit_max_radius", 5.0)))
        except (TypeError, ValueError):
            return _local_result("catalog_invalid", "entity 绕行半径元数据非法")
        if not math.isfinite(radius) or not minimum <= radius <= maximum:
            return _local_result("invalid_param", f"radius_m 必须位于实体发布范围 {minimum}..{maximum}")
        maximum_speed = self.session.max_speed_mps
        if maximum_speed is None:
            return _local_result("not_ready", "尚未收到 max_speed")
        try:
            speed_q7 = encode_speed_q7(maximum_speed if speed_mps is None else speed_mps, maximum_speed)
        except (TypeError, ValueError) as exc:
            return _local_result("invalid_param", str(exc))
        flags = (1 if direction == "ccw" else 0) | ((laps - 1) << 1) | (8 if face_target else 0)
        with self._semantic_lock:
            outcome = self._send_command(
                "ORBIT_ENTITY",
                (int(math.floor(radius * 1000.0 + 0.5)), 0, 0, entity_id, speed_q7, flags),
            )
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(
            outcome,
            semantic_key=target_key,
            radius_m=radius,
            laps=laps,
            direction=direction,
            face_target=bool(face_target),
        )

    def move_relative_wire(
        self,
        bearing_deg: float,
        distance_m: float,
        *,
        speed_mps: float | None = None,
        face_travel: bool = True,
        allow_shorter: bool = True,
    ) -> dict[str, Any]:
        """发送单条相对移动命令；路径缩短和严格方位由 Unity 完成。"""
        blocked = self._require_tool(
            "goto", "navmesh", "world_map", "semantic_navigation", "local_navigation", operation=True,
        )
        if blocked is not None:
            return blocked
        if isinstance(distance_m, bool) or isinstance(bearing_deg, bool):
            return _local_result("invalid_param", "bearing_deg/distance_m 必须是有限数值")
        if not isinstance(face_travel, bool) or not isinstance(allow_shorter, bool):
            return _local_result("invalid_param", "face_travel/allow_shorter 必须是布尔值")
        try:
            distance = float(distance_m)
            bearing = float(bearing_deg)
        except (TypeError, ValueError, OverflowError):
            return _local_result("invalid_param", "bearing_deg/distance_m 必须是有限数值")
        if not math.isfinite(distance) or not 0.25 <= distance <= 10.0:
            return _local_result("invalid_param", "distance_m 必须位于 0.25..10.0")
        if not math.isfinite(bearing):
            return _local_result("invalid_param", "bearing_deg 必须是有限数值")
        maximum_speed = self.session.max_speed_mps
        if maximum_speed is None:
            return _local_result("not_ready", "尚未收到 max_speed")
        try:
            speed_q7 = encode_speed_q7(maximum_speed if speed_mps is None else speed_mps, maximum_speed)
            bearing_q14 = encode_yaw_q14(bearing)
        except (TypeError, ValueError) as exc:
            return _local_result("invalid_param", str(exc))
        flags = (1 if face_travel else 0) | (2 if allow_shorter else 0)
        distance_mm = int(math.floor(distance * 1000.0 + 0.5))
        with self._semantic_lock:
            outcome = self._send_command(
                "MOVE_RELATIVE",
                (distance_mm, 0, bearing_q14, 0, speed_q7, flags),
            )
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(
            outcome,
            bearing_deg=bearing % 360.0,
            distance_m=distance,
            face_travel=face_travel,
            allow_shorter=allow_shorter,
        )

    def turn_relative_wire(self, delta_deg: float) -> dict[str, Any]:
        """按当前世界 yaw 做有限相对转身；角度由宿主生成，不暴露给模型。"""
        blocked = self._require_tool(operation=True)
        if blocked is not None:
            return blocked
        if isinstance(delta_deg, bool) or not isinstance(delta_deg, (int, float)):
            return _local_result("invalid_param", "delta_deg 必须是有限数值")
        delta = float(delta_deg)
        yaw = self.session.npc_state.get("yaw")
        if not math.isfinite(delta) or not -180.0 <= delta <= 180.0 or abs(delta) < 5.0:
            return _local_result("invalid_param", "delta_deg 必须位于 -180..180 且绝对值不小于 5")
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(float(yaw)):
            return _local_result("not_ready", "尚未收到 NPC yaw")
        target_yaw = (float(yaw) + delta) % 360.0
        with self._semantic_lock:
            outcome = self._send_command(
                "TURN_TO",
                (encode_yaw_q14(target_yaw), 0, 0, 0, 0, 0),
            )
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(
            outcome,
            delta_deg=delta,
            target_yaw=target_yaw,
        )

    def explore_region_wire(
        self,
        region_key: str,
        *,
        duration_ms: int,
        strategy: str = "unvisited",
        speed_mps: float | None = None,
    ) -> dict[str, Any]:
        """v1.3 连续区域探索；整个期限只创建一个 Unity operation。"""
        blocked = self._require_tool(
            "goto", "navmesh", "anchors", "world_map", "semantic_navigation", "local_navigation", operation=True,
        )
        if blocked is not None:
            return blocked
        region = next(
            (item for item in self.session.catalogs["region"].values() if item.get("semantic_key") == region_key),
            None,
        )
        if region is None:
            return _local_result("target_missing", f"目录中没有区域 {region_key!r}")
        if not bool(region.get("explorable")):
            return _local_result("target_missing", f"区域 {region_key!r} 未发布 explorable=true")
        region_id = region.get("id")
        if isinstance(region_id, bool) or not isinstance(region_id, int) or not 0 <= region_id <= 126:
            return _local_result("catalog_invalid", "region id 必须是 0..126")
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or not 1000 <= duration_ms <= 600_000:
            return _local_result("invalid_param", "duration_ms 必须是 1000..600000 的整数")
        if strategy not in {"unvisited", "patrol"}:
            return _local_result("invalid_param", "strategy 必须是 unvisited|patrol")
        maximum_speed = self.session.max_speed_mps
        if maximum_speed is None:
            return _local_result("not_ready", "尚未收到 max_speed")
        try:
            speed_q7 = encode_speed_q7(maximum_speed if speed_mps is None else speed_mps, maximum_speed)
        except (TypeError, ValueError) as exc:
            return _local_result("invalid_param", str(exc))
        duration_deciseconds = (duration_ms + 99) // 100
        with self._semantic_lock:
            outcome = self._send_command(
                "EXPLORE_REGION",
                (duration_deciseconds, 0, 0, region_id, speed_q7, 0 if strategy == "unvisited" else 1),
            )
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(
            outcome,
            region_key=region_key,
            duration_ms=duration_deciseconds * 100,
            strategy=strategy,
        )

    def navigate(self, target_key: str, *, speed_mps: float | None = None, replace_active: bool = False) -> dict[str, Any]:
        blocked = self._require_tool("goto", "navmesh", "anchors", "world_map", "semantic_navigation", operation=True)
        if blocked is not None:
            return blocked
        if self._semantic_target_anchor(target_key)[0] is None:
            return _local_result("target_missing", f"目录中没有可导航目标 {target_key!r}")
        return self.plan_manager.submit(
            single_node_graph("navigate", target_key=target_key, **({} if speed_mps is None else {"speed_mps": speed_mps})),
            replace_active=replace_active,
        )

    def orbit(
        self,
        target_key: str,
        *,
        radius_m: float = 2.0,
        laps: int = 1,
        direction: str = "cw",
        speed_mps: float | None = None,
        face_target: bool = True,
        replace_active: bool = False,
    ) -> dict[str, Any]:
        blocked = self._require_tool("goto", "navmesh", "anchors", "world_map", "semantic_navigation", operation=True)
        if blocked is not None:
            return blocked
        entity = next(
            (item for item in self.session.catalogs["entity"].values() if item.get("semantic_key") == target_key),
            None,
        )
        if entity is None or not bool(entity.get("orbitable")):
            return _local_result("target_missing", f"目录中没有可绕行实体 {target_key!r}")
        arguments: dict[str, Any] = {
            "target_key": target_key,
            "radius_m": radius_m,
            "laps": laps,
            "direction": direction,
            "face_target": face_target,
        }
        if speed_mps is not None:
            arguments["speed_mps"] = speed_mps
        return self.plan_manager.submit(single_node_graph("orbit", **arguments), replace_active=replace_active)

    def move_relative(
        self,
        bearing_deg: float,
        distance_m: float,
        *,
        speed_mps: float | None = None,
        face_travel: bool = True,
        allow_shorter: bool = True,
        replace_active: bool = False,
    ) -> dict[str, Any]:
        blocked = self._require_tool(
            "goto", "navmesh", "world_map", "semantic_navigation", "local_navigation", operation=True,
        )
        if blocked is not None:
            return blocked
        arguments: dict[str, Any] = {
            "bearing_deg": bearing_deg,
            "distance_m": distance_m,
            "face_travel": face_travel,
            "allow_shorter": allow_shorter,
        }
        if speed_mps is not None:
            arguments["speed_mps"] = speed_mps
        return self.plan_manager.submit(
            single_node_graph("move_relative", **arguments),
            replace_active=replace_active,
        )

    def approach(
        self,
        player_slot: int,
        *,
        distance_m: float = 1.5,
        speed_mps: float | None = None,
        face_target: bool = True,
        replace_active: bool = False,
    ) -> dict[str, Any]:
        blocked = self._require_tool("follow", "navmesh", "world_map", "semantic_navigation", operation=True)
        if blocked is not None:
            return blocked
        if player_slot not in self.session.players:
            return _local_result("slot_unknown", f"player_slot {player_slot} 当前未分配")
        arguments: dict[str, Any] = {
            "player_slot": player_slot,
            "distance_m": distance_m,
            "face_target": face_target,
        }
        if speed_mps is not None:
            arguments["speed_mps"] = speed_mps
        return self.plan_manager.submit(single_node_graph("approach", **arguments), replace_active=replace_active)

    def explore(
        self,
        region_key: str,
        *,
        duration_s: int = 60,
        strategy: str = "unvisited",
        speed_mps: float | None = None,
        replace_active: bool = False,
    ) -> dict[str, Any]:
        blocked = self._require_tool("goto", "navmesh", "anchors", "world_map", "semantic_navigation", operation=True)
        if blocked is not None:
            return blocked
        if isinstance(duration_s, bool) or not isinstance(duration_s, int):
            return _local_result("invalid_param", "duration_s 必须是整数")
        if not 1 <= duration_s <= 600:
            return _local_result("invalid_param", "duration_s 必须是 1..600 的整数")
        region = next(
            (item for item in self.session.catalogs["region"].values() if item.get("semantic_key") == region_key),
            None,
        )
        if region is None:
            return _local_result("target_missing", f"目录中没有区域 {region_key!r}")
        if self.session.local_navigation and not bool(region.get("explorable")):
            return _local_result("target_missing", f"区域 {region_key!r} 未发布 explorable=true")
        if strategy not in {"unvisited", "patrol"}:
            return _local_result("invalid_param", "strategy 必须是 unvisited|patrol")
        arguments: dict[str, Any] = {
            "region_key": region_key,
            "duration_ms": duration_s * 1000,
            "strategy": strategy,
        }
        if speed_mps is not None:
            arguments["speed_mps"] = speed_mps
        return self.plan_manager.submit(single_node_graph("explore", **arguments), replace_active=replace_active)

    def execute_plan(self, graph: Mapping[str, Any], *, replace_active: bool = False) -> dict[str, Any]:
        blocked = self._require_tool("world_map", "semantic_navigation", operation=True)
        if blocked is not None:
            return blocked
        return self.plan_manager.submit(graph, replace_active=replace_active)

    def plan_status(self, plan_id: str | None = None) -> dict[str, Any]:
        return self.plan_manager.status(plan_id)

    def plan_cancel(self, plan_id: str) -> dict[str, Any]:
        return self.plan_manager.cancel(plan_id)

    def follow_wire(self, player_slot: int, *, speed_mps: float | None = None) -> dict[str, Any]:
        blocked = self._require_tool("follow", "navmesh", operation=True)
        if blocked is not None:
            return blocked
        if player_slot not in self.session.players:
            return _local_result("slot_unknown", f"player_slot {player_slot} 当前未分配")
        with self._semantic_lock:
            substeps: list[dict[str, Any]] = []
            if speed_mps is not None:
                maximum_speed = self.session.max_speed_mps
                if maximum_speed is None:
                    return _local_result("not_ready", "尚未收到 max_speed")
                try:
                    speed_q7 = encode_speed_q7(speed_mps, maximum_speed)
                except (TypeError, ValueError) as exc:
                    return _local_result("invalid_param", str(exc))
                speed = self._send_command("SET_SPEED", (0, 0, 0, speed_q7, 0, 0))
                if isinstance(speed, dict):
                    return speed
                substeps.append(speed.as_dict())
                if speed.status != "succeeded":
                    return self._outcome(speed, failed_step="SET_SPEED", substeps=substeps)
            target = self._send_command("SET_TARGET", (0, 0, 0, player_slot, 0, 0))
            if isinstance(target, dict):
                return target
            substeps.append(target.as_dict())
            if target.status != "succeeded":
                return self._outcome(target, failed_step="SET_TARGET", substeps=substeps)
            follow = self._send_command("SET_MODE", (0, 0, 0, 1, 0, 0))
            if isinstance(follow, dict):
                return follow
            substeps.append(follow.as_dict())
            return self._outcome(
                follow,
                player_slot=player_slot,
                substeps=substeps,
            )

    def follow(self, player_slot: int) -> dict[str, Any]:
        return self.follow_wire(player_slot)

    @staticmethod
    def _duration_seconds(duration_ms: Any, *, maximum_ms: int = 127000) -> int:
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or not 0 <= duration_ms <= maximum_ms
        ):
            raise ValueError(f"duration_ms 必须是 0..{maximum_ms} 的整数")
        return 0 if duration_ms == 0 else (duration_ms + 999) // 1000

    def look_at(
        self,
        *,
        duration_ms: int,
        player_slot: int | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> dict[str, Any]:
        blocked = self._require_tool(operation=True)
        if blocked is not None:
            return blocked
        has_player = player_slot is not None
        has_any_coordinate = any(value is not None for value in (x, y, z))
        has_all_coordinates = all(value is not None for value in (x, y, z))
        if has_player == has_any_coordinate or (has_any_coordinate and not has_all_coordinates):
            return _local_result(
                "invalid_param",
                "player_slot 与完整 x/y/z 必须二选一",
            )
        try:
            seconds = self._duration_seconds(duration_ms)
        except ValueError as exc:
            return _local_result("invalid_param", str(exc))

        if has_player:
            if duration_ms != 0:
                return _local_result(
                    "invalid_param",
                    "冻结 LOOK_AT wire 不携带时长；玩家注视必须使用 duration_ms=0，并由 npc.stop 清除",
                )
            if (
                isinstance(player_slot, bool)
                or not isinstance(player_slot, int)
                or not 0 <= player_slot <= 63
            ):
                return _local_result("invalid_param", "player_slot 必须是 0..63 的整数")
            if player_slot not in self.session.players:
                return _local_result("slot_unknown", f"player_slot {player_slot} 当前未分配")
            with self._semantic_lock:
                outcome = self._send_command("LOOK_AT", (0, 0, 0, player_slot, 0, 0))
            if isinstance(outcome, dict):
                return outcome
            return self._outcome(outcome, player_slot=player_slot, duration_ms=0)

        bounds = self.session.wire_bounds
        activity = self.session.activity_bounds
        if bounds is None or activity is None:
            return _local_result("not_ready", "尚未收到包含 bounds 的 sys.hello")
        try:
            values = (float(x), float(y), float(z))
        except (TypeError, ValueError) as exc:
            return _local_result("invalid_param", f"x/y/z 必须是数值: {exc}")
        if not all(math.isfinite(value) for value in values):
            return _local_result("invalid_param", "x/y/z 必须是有限数值")
        if not all(activity[index] <= values[index] <= activity[index + 3] for index in range(3)):
            return _local_result("invalid_param", "注视目标越出 activity bounds")
        parameters = (
            encode_position_q14(values[0], bounds[0], bounds[3]),
            encode_position_q14(values[1], bounds[1], bounds[4]),
            encode_position_q14(values[2], bounds[2], bounds[5]),
            127,
            seconds,
            0,
        )
        with self._semantic_lock:
            outcome = self._send_command("LOOK_AT_XYZ", parameters)
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(outcome, target={"x": values[0], "y": values[1], "z": values[2]}, duration_ms=duration_ms)

    def look_at_semantic_wire(self, target_key: str) -> dict[str, Any]:
        """把语义目标解析为 Unity 已发布位置并持续注视，调用方负责清除。"""
        if not isinstance(target_key, str) or not target_key:
            return _local_result("invalid_param", "target_key 必须是非空字符串")
        position: Any = None
        for item in self.session.catalogs["anchor"].values():
            if item.get("semantic_key") == target_key:
                position = item.get("pos")
                break
        if position is None:
            for item in self.session.catalogs["entity"].values():
                if item.get("semantic_key") != target_key:
                    continue
                position = item.get("center")
                if not isinstance(position, list) or len(position) != 3:
                    anchor_id = item.get("approach_anchor_id")
                    anchor = (
                        self.session.catalogs["anchor"].get(anchor_id)
                        if isinstance(anchor_id, int)
                        else None
                    )
                    position = None if anchor is None else anchor.get("pos")
                break
        if position is None:
            for item in self.session.catalogs["region"].values():
                if item.get("semantic_key") != target_key:
                    continue
                anchor_id = item.get("entry_anchor_id")
                anchor = (
                    self.session.catalogs["anchor"].get(anchor_id)
                    if isinstance(anchor_id, int)
                    else None
                )
                position = None if anchor is None else anchor.get("pos")
                break
        if not isinstance(position, list) or len(position) != 3:
            return _local_result("target_missing", "语义目标没有可用注视位置")
        return self.look_at(
            x=position[0],
            y=position[1],
            z=position[2],
            duration_ms=0,
        )

    def clear_look_wire(self) -> dict[str, Any]:
        """行为执行器的有限注视收尾；LOOK_AT P3=127 是冻结的清除语义。"""
        blocked = self._require_tool(operation=True)
        if blocked is not None:
            return blocked
        with self._semantic_lock:
            outcome = self._send_command("LOOK_AT", (0, 0, 0, 127, 0, 0))
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(outcome, cleared=True)

    def act(self, action_key: str, *, player_slot: int | None = None, loop: bool = False) -> dict[str, Any]:
        blocked = self._require_tool("actions", operation=True)
        if blocked is not None:
            return blocked
        action = next(
            (
                item
                for item in self.session.catalogs["action"].values()
                if item.get("semantic_key") == action_key
            ),
            None,
        )
        if action is None:
            return _local_result("action_not_found", f"目录中没有 action {action_key!r}")
        if loop and not bool(action.get("loopable", False)):
            return _local_result("invalid_param", "该动作不允许 loop")
        if action.get("target_required") == "player":
            if player_slot is None:
                return _local_result("target_missing", "该动作需要 player_slot")
            if player_slot not in self.session.players:
                return _local_result("slot_unknown", f"player_slot {player_slot} 当前未分配")
        with self._semantic_lock:
            substeps: list[dict[str, Any]] = []
            if player_slot is not None:
                target = self._send_command("SET_TARGET", (0, 0, 0, player_slot, 0, 0))
                if isinstance(target, dict):
                    return target
                substeps.append(target.as_dict())
                if target.status != "succeeded":
                    return self._outcome(target, failed_step="SET_TARGET", substeps=substeps)
            action_sequence = self._allocate_action_sequence()
            play = self._send_command(
                "PLAY_ANIM",
                (action_sequence, 0, 0, int(action["id"]), 1 if loop else 0, 0),
            )
            if isinstance(play, dict):
                return play
            substeps.append(play.as_dict())
            return self._outcome(
                play,
                semantic_key=action_key,
                action_sequence=action_sequence,
                substeps=substeps,
            )

    def set_expression(self, expression_key: str, duration_ms: int) -> dict[str, Any]:
        blocked = self._require_tool("expressions", operation=True)
        if blocked is not None:
            return blocked
        expression = next(
            (
                item
                for item in self.session.catalogs["expression"].values()
                if item.get("semantic_key") == expression_key
            ),
            None,
        )
        if expression is None:
            return _local_result(
                "expression_not_found",
                f"目录中没有 expression {expression_key!r}",
            )
        try:
            seconds = self._duration_seconds(duration_ms)
        except ValueError as exc:
            return _local_result("invalid_param", str(exc))
        expression_id = expression.get("id")
        if (
            isinstance(expression_id, bool)
            or not isinstance(expression_id, int)
            or not 0 <= expression_id <= 126
        ):
            return _local_result("catalog_invalid", "expression id 必须是 0..126")
        with self._semantic_lock:
            outcome = self._send_command(
                "SET_EXPRESSION",
                (0, 0, 0, expression_id, 127, seconds),
            )
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(
            outcome,
            semantic_key=expression_key,
            duration_ms=duration_ms,
        )

    def wander(self) -> dict[str, Any]:
        blocked = self._require_tool("wander", "navmesh", operation=True)
        if blocked is not None:
            return blocked
        with self._semantic_lock:
            outcome = self._send_command("SET_MODE", (0, 0, 0, 3, 0, 0))
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(outcome)

    def say(
        self,
        text: str,
        *,
        action_key: str | None = None,
        estimated_delay_ms: int | None = None,
        duration_ms: int | None = None,
        display_seconds: int = 0,
    ) -> dict[str, Any]:
        blocked = self._require_tool("text_utf8")
        if blocked is not None:
            return blocked
        if (
            isinstance(display_seconds, bool)
            or not isinstance(display_seconds, int)
            or not 0 <= display_seconds <= 127
        ):
            return _local_result("invalid_param", "display_seconds 必须是 0..127 的整数")
        action_result: dict[str, Any] | None = None
        action_sequence = 0
        if action_key is not None:
            action_result = self.act(action_key)
            if action_result.get("status") not in {"accepted", "succeeded"}:
                result = dict(action_result)
                result["failed_step"] = "PLAY_ANIM"
                return result
            action_sequence = int(action_result.get("action_sequence") or 0)
        with self._semantic_lock:
            transfer_sequence = self._allocate_transfer_sequence()
            result = self._send_text(
                text,
                transfer_sequence=transfer_sequence,
                display_seconds=display_seconds,
            )
        if isinstance(result, dict):
            return result
        transaction = result.transaction
        response: dict[str, Any] = {
            "request_id": f"host-{uuid.uuid4()}",
            "status": result.status,
            "transfer_sequence": transfer_sequence,
            "utf8_bytes": len(transaction.utf8_bytes) if transaction is not None else 0,
            "crc16": transaction.crc16 if transaction is not None else "0000",
            "begin": result.begin.as_dict(),
            "commit": result.commit.as_dict() if result.commit is not None else None,
            "midi_sent": transaction is not None,
            "error": (
                result.commit.error
                if result.commit is not None
                else result.begin.error
            ),
            "action": action_result,
            "speech_cue": None,
        }
        if result.status not in {"accepted", "succeeded"}:
            return response
        if "voice_stream" not in self.session.capabilities:
            return response
        if duration_ms is None:
            return response
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or not 250 <= duration_ms <= 31750
        ):
            return _local_result("invalid_param", "duration_ms 必须是 250..31750 的整数")
        delay = 0 if estimated_delay_ms is None else estimated_delay_ms
        if (
            isinstance(delay, bool)
            or not isinstance(delay, int)
            or not 0 <= delay <= 12700
        ):
            return _local_result("invalid_param", "estimated_delay_ms 必须是 0..12700 的整数")
        speech_sequence = self._allocate_speech_sequence()
        cue = self._send_command(
            "SPEECH_CUE",
            (
                speech_sequence,
                transfer_sequence,
                action_sequence,
                (delay + 99) // 100,
                (duration_ms + 249) // 250,
                0,
            ),
        )
        response["speech_cue"] = cue if isinstance(cue, dict) else cue.as_dict()
        if isinstance(cue, dict) or cue.status not in {"accepted", "succeeded"}:
            response["status"] = "failed"
            response["error"] = (
                cue.get("error") if isinstance(cue, dict) else cue.error
            )
        return response

    def say_preset(self, preset_key: str, *, display_seconds: int = 0) -> dict[str, Any]:
        """显示世界发布的静态文案；模型只选择目录键，不接触 preset id。"""
        blocked = self._require_tool("text_preset")
        if blocked is not None:
            return blocked
        if isinstance(display_seconds, bool) or not isinstance(display_seconds, int) or not 0 <= display_seconds <= 127:
            return _local_result("invalid_param", "display_seconds 必须是 0..127 的整数")
        preset = next(
            (
                item
                for item in self.session.catalogs["text_preset"].values()
                if item.get("semantic_key") == preset_key or item.get("name") == preset_key
            ),
            None,
        )
        if preset is None:
            return _local_result("text_preset_not_found", f"目录中没有 text preset {preset_key!r}")
        preset_id = preset.get("id")
        if isinstance(preset_id, bool) or not isinstance(preset_id, int) or not 0 <= preset_id <= 126:
            return _local_result("catalog_invalid", "text preset id 必须是 0..126")
        with self._semantic_lock:
            outcome = self._send_command(
                "TEXT_PRESET",
                (0, 0, 0, preset_id, display_seconds, 0),
            )
            if isinstance(outcome, dict):
                return outcome
        return self._outcome(outcome, preset_key=preset_key)

    def request_snapshot_evidence(self) -> dict[str, Any]:
        """operation 超时时请求一次快照；快照只能补证据，不能改写 unknown。"""
        if "snapshot" not in self.session.capabilities or self.session.session <= 0:
            return _local_result("unsupported_capability", "世界未发布 snapshot")
        with self._semantic_lock:
            outcome = self._send_command("SNAPSHOT_REQUEST", (0, 0, 0, 0, 0, 0))
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(outcome)

    def _stop_plan_domains(self, domains: frozenset[str] | set[str]) -> bool:
        """后台计划内部停止路径；不反向触发 plan_manager 取消。"""
        if not domains:
            return False
        if len(domains) > 1 or any(item in domains for item in {"look", "expression", "text"}):
            command, parameters = "STOP", (0, 0, 0, 0, 0, 0)
        elif "movement" in domains:
            command, parameters = "SET_MODE", (0, 0, 0, 0, 0, 0)
        elif "action" in domains:
            command, parameters = "STOP_ACTION", (0, 0, 0, 0, 0, 0)
        else:
            return False
        with self._semantic_lock:
            outcome = self._send_command(command, parameters)
        return not isinstance(outcome, dict) or bool(outcome.get("midi_sent"))

    def stop(self, scope: str = "all", *, _from_plan: bool = False) -> dict[str, Any]:
        blocked = self._require_tool()
        if blocked is not None and not (
            scope == "all" and self.session.control_state == "safe_idle"
        ):
            return blocked
        command = scope.strip().lower()
        if command == "all":
            name, parameters = "STOP", (0, 0, 0, 0, 0, 0)
        elif command == "movement":
            name, parameters = "SET_MODE", (0, 0, 0, 0, 0, 0)
        elif command == "action":
            name, parameters = "STOP_ACTION", (0, 0, 0, 0, 0, 0)
        else:
            return _local_result("invalid_param", "scope 必须是 all|movement|action")
        if not _from_plan:
            self.plan_manager.cancel_for_scope(command)
        with self._semantic_lock:
            outcome = self._send_command(name, parameters)
        if isinstance(outcome, dict):
            return outcome
        return self._outcome(outcome, scope=command)

    def clear_estop(self) -> dict[str, Any]:
        """仅供宿主安全入口调用；清除后恢复地图 NPC 的宿主控制态。"""
        self.plan_manager.cancel_all("estop_clear")
        self.session.set_host_arm_authorized(False)
        if self.session.control_state != "estop":
            return _local_result("invalid_state", "当前未锁存 ESTOP")
        with self._semantic_lock:
            clear_outcome = self._send_command("CLEAR_ESTOP", (0, 0, 0, 0, 0, 0))
            if isinstance(clear_outcome, dict):
                return clear_outcome
            if clear_outcome.status != "succeeded":
                return self._outcome(clear_outcome, control_state="estop")
            control_outcome = self._send_command(
                "SET_CONTROL_MODE", (0, 0, 0, 1, 0, 0)
            )
        if isinstance(control_outcome, dict):
            return control_outcome
        return self._outcome(
            control_outcome,
            control_state="external",
            estop_cleared=True,
            clear_wire_seq=clear_outcome.wire_sequence,
            clear_request_hash=clear_outcome.request_hash,
        )

    def estop(self, reason: str = "") -> dict[str, Any]:
        self.plan_manager.cancel_all("estop")
        frame = self.transport.send_estop()
        return {
            "request_id": f"host-{uuid.uuid4()}",
            "status": "accepted",
            "wire_seq": frame.sequence,
            "request_hash": frame.request_hash,
            "reason": str(reason)[:160],
            "midi_sent": True,
            "error": None,
        }

    def close(self) -> None:
        self._connected_session = 0
        self.plan_manager.close()


__all__ = ["YuiSemanticAdapter"]
