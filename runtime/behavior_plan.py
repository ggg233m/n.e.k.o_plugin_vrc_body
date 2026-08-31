"""YUI v1.2/v1.3 后台行为图执行器。

模型只提交受限 JSON 图；本模块在独立线程中编译、调度并等待 Unity operation
证据。它不解释自然语言、不执行表达式，也不会把 cancelled/failed/unknown
冒充成功。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Mapping, Sequence, TYPE_CHECKING
import uuid

if TYPE_CHECKING:  # pragma: no cover
    from .yui_adapter import YuiSemanticAdapter
    from .yui_session import YuiSessionState


PLAN_STATUSES = ("accepted", "running", "succeeded", "cancelled", "failed", "unknown")
TERMINAL_STATUSES = frozenset({"succeeded", "cancelled", "failed", "unknown"})
CONTROL_TYPES = frozenset({"sequence", "selector", "parallel", "repeat", "retry", "timeout", "condition"})
LEAF_TYPES = frozenset({"navigate", "approach", "follow", "orbit", "explore", "move_relative", "look_at", "act", "set_expression", "say", "wait", "stop"})
NODE_TYPES = CONTROL_TYPES | LEAF_TYPES
MOVEMENT_TYPES = frozenset({"navigate", "approach", "follow", "orbit", "explore", "move_relative"})
MAX_NODES = 64
MAX_DEPTH = 8
MAX_REPEAT = 10
MAX_RETRY_ATTEMPTS = 3
MAX_PARALLEL_CHILDREN = 4
MAX_WAIT_MS = 60_000
MAX_PLAN_SECONDS = 600.0
PLAN_HISTORY_SIZE = 16


class BehaviorGraphError(ValueError):
    """提交的行为图违反冻结 v1.2 约束。"""


def _result(status: str, *, error: str | None = None, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    if status not in PLAN_STATUSES:
        raise ValueError(f"未知计划状态: {status}")
    value: dict[str, Any] = {"status": status, "error": error, "detail": detail}
    value.update(extra)
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise BehaviorGraphError(f"{name} 必须是 {minimum}..{maximum} 的整数")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BehaviorGraphError(f"{name} 必须是数字")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise BehaviorGraphError(f"{name} 必须位于 {minimum}..{maximum}")
    return parsed


@dataclass
class BehaviorPlan:
    plan_id: str
    graph: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    entry: str
    domains: frozenset[str]
    session_id: int
    catalog_revision: int | None
    driver_pid: int | None
    status: str = "accepted"
    error: str | None = None
    detail: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_reason: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    node_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=128))

    def public(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "error": self.error,
            "detail": self.detail,
            "domains": sorted(self.domains),
            "session": self.session_id,
            "catalog_rev": self.catalog_revision,
            "driver_pid": self.driver_pid,
            "elapsed_ms": round(((self.finished_at or now) - (self.started_at or self.created_at)) * 1000),
            "node_status": {key: dict(value) for key, value in self.node_status.items()},
            "evidence": list(self.evidence),
            "done": self.status in TERMINAL_STATUSES,
        }


class BehaviorGraphCompiler:
    """验证 JSON 图并计算控制域；不做任何运行时 I/O。"""

    def __init__(self, session: "YuiSessionState") -> None:
        self.session = session

    def compile(self, graph: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str, frozenset[str]]:
        if not isinstance(graph, Mapping):
            raise BehaviorGraphError("graph 必须是对象")
        unknown_root = set(graph) - {"entry", "nodes"}
        if unknown_root:
            raise BehaviorGraphError(f"graph 包含未知字段: {', '.join(sorted(unknown_root))}")
        entry = graph.get("entry")
        raw_nodes = graph.get("nodes")
        if not isinstance(entry, str) or not entry:
            raise BehaviorGraphError("graph.entry 必须是非空字符串")
        if not isinstance(raw_nodes, list) or not 1 <= len(raw_nodes) <= MAX_NODES:
            raise BehaviorGraphError(f"graph.nodes 必须包含 1..{MAX_NODES} 项")
        nodes: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(raw_nodes):
            if not isinstance(raw, Mapping):
                raise BehaviorGraphError(f"nodes[{index}] 必须是对象")
            node = dict(raw)
            node_id = node.get("id")
            node_type = node.get("type")
            if not isinstance(node_id, str) or not 1 <= len(node_id) <= 48:
                raise BehaviorGraphError(f"nodes[{index}].id 必须是 1..48 字符串")
            if node_id in nodes:
                raise BehaviorGraphError(f"节点 id 重复: {node_id}")
            if node_type not in NODE_TYPES:
                raise BehaviorGraphError(f"节点 {node_id} type 不受支持: {node_type}")
            self._validate_node(node)
            nodes[node_id] = node
        if entry not in nodes:
            raise BehaviorGraphError("entry 未引用已声明节点")
        inbound: dict[str, int] = {key: 0 for key in nodes}
        for node in nodes.values():
            for child in self._children(node):
                if child not in nodes:
                    raise BehaviorGraphError(f"节点 {node['id']} 引用了不存在的节点 {child}")
                inbound[child] += 1
                if inbound[child] > 1:
                    raise BehaviorGraphError(f"节点 {child} 被多个父节点共享；v1.2 要求行为树形展开")
        visited: set[str] = set()
        active: set[str] = set()

        def walk(node_id: str, depth: int) -> frozenset[str]:
            if depth > MAX_DEPTH:
                raise BehaviorGraphError(f"行为图深度超过 {MAX_DEPTH}")
            if node_id in active:
                raise BehaviorGraphError("行为图包含回边；循环只能使用 repeat")
            active.add(node_id)
            visited.add(node_id)
            node = nodes[node_id]
            child_domains = [walk(child, depth + 1) for child in self._children(node)]
            active.remove(node_id)
            domains = set(self._leaf_domains(node))
            for child_domain in child_domains:
                domains.update(child_domain)
            if node["type"] == "parallel":
                movement_children = sum("movement" in item for item in child_domains)
                if movement_children > 1:
                    raise BehaviorGraphError(f"parallel {node_id} 同时包含多个移动分支")
                if movement_children and self._contains_blocking_action(node_id, nodes):
                    raise BehaviorGraphError(f"parallel {node_id} 将 movement=block 动作与移动并行")
            return frozenset(domains)

        domains = walk(entry, 1)
        unreachable = sorted(set(nodes) - visited)
        if unreachable:
            raise BehaviorGraphError(f"行为图包含不可达节点: {', '.join(unreachable)}")
        return {"entry": entry, "nodes": [dict(nodes[key]) for key in nodes]}, nodes, entry, domains

    @staticmethod
    def _children(node: Mapping[str, Any]) -> list[str]:
        node_type = node["type"]
        if node_type in {"sequence", "selector", "parallel"}:
            return list(node["children"])
        if node_type in {"repeat", "retry", "timeout"}:
            return [str(node["child"])]
        return []

    @staticmethod
    def _leaf_domains(node: Mapping[str, Any]) -> set[str]:
        node_type = node["type"]
        if node_type in MOVEMENT_TYPES:
            return {"movement"}
        if node_type == "look_at":
            return {"look"}
        if node_type == "act":
            return {"action"}
        if node_type == "set_expression":
            return {"expression"}
        if node_type == "say":
            return {"text"}
        if node_type == "stop":
            scope = node.get("scope", "all")
            return {"movement", "look", "action", "expression", "text"} if scope == "all" else {str(scope)}
        return set()

    def _contains_blocking_action(self, node_id: str, nodes: Mapping[str, Mapping[str, Any]]) -> bool:
        node = nodes[node_id]
        if node["type"] == "act":
            action_key = node.get("action_key")
            return any(
                item.get("semantic_key") == action_key and item.get("movement") == "block"
                for item in self.session.catalogs["action"].values()
            )
        return any(self._contains_blocking_action(child, nodes) for child in self._children(node))

    def _validate_node(self, node: Mapping[str, Any]) -> None:
        node_id = str(node["id"])
        node_type = str(node["type"])
        common = {"id", "type"}
        allowed: dict[str, set[str]] = {
            "sequence": common | {"children"},
            "selector": common | {"children"},
            "parallel": common | {"children", "join"},
            "repeat": common | {"child", "count"},
            "retry": common | {"child", "max_attempts", "delay_ms"},
            "timeout": common | {"child", "timeout_ms"},
            "condition": common | {"predicate"},
            "navigate": common | {"target_key", "speed_mps"},
            "approach": common | {"player_slot", "distance_m", "speed_mps", "face_target"},
            "follow": common | {"player_slot", "duration_ms", "speed_mps"},
            "orbit": common | {"target_key", "radius_m", "laps", "direction", "speed_mps", "face_target"},
            "explore": common | {"region_key", "duration_ms", "strategy", "speed_mps"},
            "move_relative": common | {"bearing_deg", "distance_m", "speed_mps", "face_travel", "allow_shorter"},
            "look_at": common | {"player_slot", "duration_ms"},
            "act": common | {"action_key", "player_slot", "loop"},
            "set_expression": common | {"expression_key", "duration_ms"},
            "say": common | {"text", "duration_ms", "estimated_delay_ms", "action_key"},
            "wait": common | {"duration_ms"},
            "stop": common | {"scope"},
        }
        unknown = set(node) - allowed[node_type]
        if unknown:
            raise BehaviorGraphError(f"节点 {node_id} 包含未知字段: {', '.join(sorted(unknown))}")
        if node_type in {"sequence", "selector", "parallel"}:
            children = node.get("children")
            maximum = MAX_PARALLEL_CHILDREN if node_type == "parallel" else MAX_NODES
            if not isinstance(children, list) or not 1 <= len(children) <= maximum or not all(isinstance(item, str) and item for item in children):
                raise BehaviorGraphError(f"节点 {node_id}.children 非法")
            if len(set(children)) != len(children):
                raise BehaviorGraphError(f"节点 {node_id}.children 不得重复")
            if node_type == "parallel" and node.get("join", "all") not in {"all", "race"}:
                raise BehaviorGraphError(f"节点 {node_id}.join 必须是 all|race")
        elif node_type in {"repeat", "retry", "timeout"}:
            if not isinstance(node.get("child"), str) or not node["child"]:
                raise BehaviorGraphError(f"节点 {node_id}.child 必须是非空字符串")
            if node_type == "repeat":
                _integer(node.get("count"), f"节点 {node_id}.count", 1, MAX_REPEAT)
            elif node_type == "retry":
                _integer(node.get("max_attempts"), f"节点 {node_id}.max_attempts", 1, MAX_RETRY_ATTEMPTS)
                _integer(node.get("delay_ms", 0), f"节点 {node_id}.delay_ms", 0, 5000)
            else:
                _integer(node.get("timeout_ms"), f"节点 {node_id}.timeout_ms", 1, MAX_WAIT_MS)
        elif node_type == "condition":
            self._validate_predicate(node.get("predicate"), node_id)
        else:
            self._validate_leaf(node)

    def _validate_predicate(self, predicate: Any, node_id: str) -> None:
        if not isinstance(predicate, Mapping):
            raise BehaviorGraphError(f"节点 {node_id}.predicate 必须是对象")
        kind = predicate.get("type")
        allowed_fields = {
            "player_present": {"type", "player_slot"},
            "player_distance": {"type", "player_slot", "op", "value_m"},
            "event_seen": {"type", "event_type", "within_ms"},
            "node_status": {"type", "node_id", "status"},
            "control_state": {"type", "value"},
            "estop": {"type", "value"},
            "elapsed_ms": {"type", "op", "value"},
        }
        if kind not in allowed_fields:
            raise BehaviorGraphError(f"节点 {node_id} 使用未授权条件 {kind}")
        unknown = set(predicate) - allowed_fields[str(kind)]
        if unknown:
            raise BehaviorGraphError(
                f"节点 {node_id}.predicate 包含未知字段: {', '.join(sorted(unknown))}"
            )
        if kind in {"player_present", "player_distance"}:
            _integer(predicate.get("player_slot"), f"节点 {node_id}.player_slot", 0, 63)
        if kind == "player_distance":
            if predicate.get("op") not in {"lt", "lte", "gt", "gte"}:
                raise BehaviorGraphError(f"节点 {node_id}.op 非法")
            _number(predicate.get("value_m"), f"节点 {node_id}.value_m", 0.0, 1000.0)
        elif kind == "event_seen":
            if not isinstance(predicate.get("event_type"), str) or not predicate["event_type"]:
                raise BehaviorGraphError(f"节点 {node_id}.event_type 必须是非空字符串")
            _integer(predicate.get("within_ms", 5000), f"节点 {node_id}.within_ms", 1, MAX_WAIT_MS)
        elif kind == "node_status":
            if not isinstance(predicate.get("node_id"), str) or predicate.get("status") not in PLAN_STATUSES:
                raise BehaviorGraphError(f"节点 {node_id} 的 node_status 条件非法")
        elif kind == "control_state" and predicate.get("value") not in {"safe_idle", "external", "moving", "action", "estop"}:
            raise BehaviorGraphError(f"节点 {node_id}.value 不是控制状态")
        elif kind == "estop" and not isinstance(predicate.get("value"), bool):
            raise BehaviorGraphError(f"节点 {node_id}.value 必须是布尔值")
        elif kind == "elapsed_ms":
            if predicate.get("op") not in {"lt", "lte", "gt", "gte"}:
                raise BehaviorGraphError(f"节点 {node_id}.op 非法")
            _integer(predicate.get("value"), f"节点 {node_id}.value", 0, int(MAX_PLAN_SECONDS * 1000))

    def _validate_leaf(self, node: Mapping[str, Any]) -> None:
        node_id = str(node["id"])
        node_type = str(node["type"])
        for field_name in ("target_key", "region_key", "action_key", "expression_key", "text"):
            if field_name in node and (not isinstance(node[field_name], str) or not node[field_name]):
                raise BehaviorGraphError(f"节点 {node_id}.{field_name} 必须是非空字符串")
        if node_type in {"navigate", "orbit"} and "target_key" not in node:
            raise BehaviorGraphError(f"节点 {node_id}.target_key 为必填")
        if node_type == "explore" and "region_key" not in node:
            raise BehaviorGraphError(f"节点 {node_id}.region_key 为必填")
        if node_type in {"approach", "follow", "look_at"}:
            _integer(node.get("player_slot"), f"节点 {node_id}.player_slot", 0, 63)
        if node_type == "approach":
            _number(node.get("distance_m", 1.5), f"节点 {node_id}.distance_m", 0.5, 5.0)
            if "face_target" in node and not isinstance(node["face_target"], bool):
                raise BehaviorGraphError(f"节点 {node_id}.face_target 必须是布尔值")
        if node_type == "orbit":
            _number(node.get("radius_m", 2.0), f"节点 {node_id}.radius_m", 0.25, 5.0)
            _integer(node.get("laps", 1), f"节点 {node_id}.laps", 1, 3)
            if node.get("direction", "cw") not in {"cw", "ccw"}:
                raise BehaviorGraphError(f"节点 {node_id}.direction 必须是 cw|ccw")
            if "face_target" in node and not isinstance(node["face_target"], bool):
                raise BehaviorGraphError(f"节点 {node_id}.face_target 必须是布尔值")
        if node_type == "move_relative":
            if not self.session.local_navigation:
                raise BehaviorGraphError(
                    f"节点 {node_id}.move_relative 需要 local_navigation capability"
                )
            _number(node.get("bearing_deg"), f"节点 {node_id}.bearing_deg", -math.inf, math.inf)
            _number(node.get("distance_m"), f"节点 {node_id}.distance_m", 0.25, 10.0)
            for field_name in ("face_travel", "allow_shorter"):
                if field_name in node and not isinstance(node[field_name], bool):
                    raise BehaviorGraphError(f"节点 {node_id}.{field_name} 必须是布尔值")
        if "speed_mps" in node:
            maximum = self.session.max_speed_mps or 2.0
            _number(node["speed_mps"], f"节点 {node_id}.speed_mps", 0.0, maximum)
        if node_type in {"follow", "look_at", "wait"}:
            _integer(node.get("duration_ms"), f"节点 {node_id}.duration_ms", 1, MAX_WAIT_MS)
        if node_type == "explore":
            minimum_duration = 1000 if self.session.local_navigation else 1
            _integer(node.get("duration_ms"), f"节点 {node_id}.duration_ms", minimum_duration, int(MAX_PLAN_SECONDS * 1000))
            if node.get("strategy", "unvisited") not in {"unvisited", "patrol"}:
                raise BehaviorGraphError(f"节点 {node_id}.strategy 必须是 unvisited|patrol")
        if node_type == "set_expression":
            _integer(node.get("duration_ms", 0), f"节点 {node_id}.duration_ms", 0, MAX_WAIT_MS)
        if node_type == "act" and "action_key" not in node:
            raise BehaviorGraphError(f"节点 {node_id}.action_key 为必填")
        if node_type == "set_expression" and "expression_key" not in node:
            raise BehaviorGraphError(f"节点 {node_id}.expression_key 为必填")
        if node_type == "say" and "text" not in node:
            raise BehaviorGraphError(f"节点 {node_id}.text 为必填")
        if node_type == "say":
            if not 1 <= len(str(node["text"]).encode("utf-8")) <= 384:
                raise BehaviorGraphError(f"节点 {node_id}.text 必须是 1..384 UTF-8 字节")
            if "duration_ms" in node:
                _integer(node["duration_ms"], f"节点 {node_id}.duration_ms", 250, 31_750)
            if "estimated_delay_ms" in node:
                _integer(node["estimated_delay_ms"], f"节点 {node_id}.estimated_delay_ms", 0, 12_700)
        if node_type == "act":
            if "player_slot" in node:
                _integer(node["player_slot"], f"节点 {node_id}.player_slot", 0, 63)
            if "loop" in node and not isinstance(node["loop"], bool):
                raise BehaviorGraphError(f"节点 {node_id}.loop 必须是布尔值")
        if node_type == "wait":
            _integer(node.get("duration_ms"), f"节点 {node_id}.duration_ms", 1, MAX_WAIT_MS)
        if node_type == "stop" and node.get("scope", "all") not in {"all", "movement", "action"}:
            raise BehaviorGraphError(f"节点 {node_id}.scope 必须是 all|movement|action")


class BehaviorPlanManager:
    """单 session 后台计划调度器；stdio 与 Agent 调用线程不会被长行为占用。"""

    def __init__(self, adapter: "YuiSemanticAdapter", session: "YuiSessionState") -> None:
        self.adapter = adapter
        self.session = session
        self._condition = threading.Condition(threading.RLock())
        self._pending: deque[BehaviorPlan] = deque()
        self._history: deque[BehaviorPlan] = deque()
        self._plans: dict[str, BehaviorPlan] = {}
        self._active: BehaviorPlan | None = None
        self._closed = False
        self.session.add_event_listener(self._on_session_event)
        self._worker = threading.Thread(target=self._worker_loop, name="yui-behavior-plan", daemon=True)
        self._worker.start()

    def submit(self, graph: Mapping[str, Any], *, replace_active: bool = False) -> dict[str, Any]:
        try:
            normalized, nodes, entry, domains = BehaviorGraphCompiler(self.session).compile(graph)
        except BehaviorGraphError as exc:
            return _result(
                "failed",
                error="behavior_graph_invalid",
                detail=str(exc),
                midi_sent=False,
            )
        if self.session.session <= 0 or self.session.control_state not in {"external", "moving", "action"}:
            return _result("failed", error="invalid_state", detail="地图 NPC 尚未进入宿主控制态", midi_sent=False)
        plan = BehaviorPlan(
            plan_id=f"plan-{uuid.uuid4()}",
            graph=normalized,
            nodes=nodes,
            entry=entry,
            domains=domains,
            session_id=self.session.session,
            catalog_revision=self.session.catalog_revision,
            driver_pid=self.session.driver_pid,
        )
        replaced: BehaviorPlan | None = None
        with self._condition:
            self._prune_history_locked(make_room=True)
            live_plans = ([self._active] if self._active is not None else []) + list(self._pending)
            current = next(
                (
                    item for item in live_plans
                    if item.status not in TERMINAL_STATUSES
                    and "movement" in item.domains
                    and "movement" in plan.domains
                ),
                None,
            )
            if current is not None:
                if not replace_active:
                    return _result("failed", error="plan_conflict", detail=f"移动计划 {current.plan_id} 仍在运行；需要 replace_active=true", active_plan_id=current.plan_id, midi_sent=False)
                replaced = current
                self._cancel_locked(current, "replaced")
            if len(self._history) >= PLAN_HISTORY_SIZE:
                return _result(
                    "failed",
                    error="plan_capacity",
                    detail=f"会话内同时可保留的计划不得超过 {PLAN_HISTORY_SIZE} 个",
                    midi_sent=False,
                )
            self._plans[plan.plan_id] = plan
            self._history.append(plan)
            self._pending.append(plan)
            self._condition.notify_all()
        if replaced is not None:
            self.adapter._stop_plan_domains(replaced.domains)
        response = plan.public()
        response.update({"replaced_plan_id": None if replaced is None else replaced.plan_id, "midi_sent": replaced is not None})
        return response

    def status(self, plan_id: str | None = None) -> dict[str, Any]:
        with self._condition:
            plan = self._active if plan_id is None else self._plans.get(plan_id)
            if plan is None:
                return _result("failed", error="plan_not_found", detail="未找到计划", midi_sent=False)
            result = plan.public()
            result["midi_sent"] = False
            return result

    def cancel(self, plan_id: str) -> dict[str, Any]:
        with self._condition:
            plan = self._plans.get(plan_id)
            if plan is None:
                return _result("failed", error="plan_not_found", detail=f"未找到计划 {plan_id}", midi_sent=False)
            if plan.status in TERMINAL_STATUSES:
                result = plan.public()
                result["midi_sent"] = False
                return result
            self._cancel_locked(plan, "explicit_stop")
        sent = self.adapter._stop_plan_domains(plan.domains)
        result = plan.public()
        result["midi_sent"] = bool(sent)
        return result

    def cancel_for_scope(self, scope: str, *, reason: str = "explicit_stop") -> None:
        wanted = {"movement", "look", "action", "expression", "text"} if scope == "all" else {scope}
        with self._condition:
            plan = self._active
            if plan is not None and plan.status not in TERMINAL_STATUSES and plan.domains & wanted:
                self._cancel_locked(plan, reason)
            for pending in self._pending:
                if pending.status not in TERMINAL_STATUSES and pending.domains & wanted:
                    self._cancel_locked(pending, reason)

    def cancel_all(self, reason: str) -> None:
        with self._condition:
            if self._active is not None and self._active.status not in TERMINAL_STATUSES:
                self._cancel_locked(self._active, reason)
            for plan in self._pending:
                if plan.status not in TERMINAL_STATUSES:
                    self._cancel_locked(plan, reason)

    def close(self) -> None:
        domains: set[str] = set()
        with self._condition:
            if self._active is not None and self._active.status not in TERMINAL_STATUSES:
                domains.update(self._active.domains)
            for plan in self._pending:
                if plan.status not in TERMINAL_STATUSES:
                    domains.update(plan.domains)
        self.cancel_all("disconnect")
        if domains and self.session.session > 0 and self.session.control_state in {"external", "moving", "action"}:
            self.adapter._stop_plan_domains(domains)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.session.remove_event_listener(self._on_session_event)
        self._worker.join(timeout=2.0)

    def _cancel_locked(self, plan: BehaviorPlan, reason: str) -> None:
        plan.cancel_reason = reason
        plan.cancel_event.set()
        if plan.status == "accepted":
            plan.status = "cancelled"
            plan.detail = f"计划在启动前取消: {reason}"
            plan.finished_at = time.monotonic()
        self._condition.notify_all()

    def _prune_history_locked(self, *, make_room: bool = False) -> None:
        """只保留最近 16 个会话内诊断记录。

        运行中或排队中的计划不可被淘汰；容量全被非终态计划占用时，
        submit 显式返回 plan_capacity，不会创建无法查询的孤儿计划。
        """
        limit = PLAN_HISTORY_SIZE - 1 if make_room else PLAN_HISTORY_SIZE
        while len(self._history) > limit:
            removable = next(
                (item for item in self._history if item.status in TERMINAL_STATUSES),
                None,
            )
            if removable is None:
                return
            self._history.remove(removable)
            self._plans.pop(removable.plan_id, None)

    def _on_session_event(self, event: dict[str, Any]) -> None:
        with self._condition:
            event_type = event.get("type")
            candidates = ([self._active] if self._active is not None else []) + list(self._pending)
            for plan in candidates:
                if plan.status in TERMINAL_STATUSES:
                    continue
                reason: str | None = None
                if event_type == "sys.session" and int(event.get("new_session", 0)) != plan.session_id:
                    reason = "session_reset"
                elif event_type == "sys.hello" and int(event.get("catalog_rev", -1)) != (plan.catalog_revision if plan.catalog_revision is not None else -1):
                    reason = "catalog_changed"
                elif event_type == "sys.watchdog":
                    reason = "watchdog"
                elif event_type == "player.leave" and int(event.get("pid", -1)) == plan.driver_pid:
                    reason = "player_left"
                elif event_type == "npc.state" and bool(event.get("estop")):
                    reason = "estop"
                elif event_type == "npc.ack" and event.get("err") in {"not_owner", "ownership_failed"}:
                    reason = "ownership_lost"
                if reason is not None:
                    self._cancel_locked(plan, reason)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                plan = self._pending.popleft()
                if plan.status == "cancelled":
                    continue
                if (
                    plan.session_id != self.session.session
                    or plan.catalog_revision != self.session.catalog_revision
                    or plan.driver_pid != self.session.driver_pid
                ):
                    self._cancel_locked(plan, "session_binding_changed")
                    continue
                self._active = plan
                plan.status = "running"
                plan.started_at = time.monotonic()
            deadline = plan.started_at + MAX_PLAN_SECONDS
            try:
                outcome = self._execute_node(plan, plan.entry, deadline, None)
            except Exception as exc:  # 安全兜底：后台线程不能静默死亡。
                outcome = _result("failed", error="internal_error", detail=f"{type(exc).__name__}: {exc}")
            with self._condition:
                if plan.cancel_event.is_set():
                    plan.status = "cancelled"
                    plan.error = None
                    plan.detail = f"计划已取消: {plan.cancel_reason or 'cancelled'}"
                else:
                    plan.status = str(outcome["status"])
                    plan.error = outcome.get("error")
                    plan.detail = outcome.get("detail")
                plan.finished_at = time.monotonic()
                if self._active is plan:
                    self._active = None
                self._prune_history_locked()
                self._condition.notify_all()

    def _cancelled(self, plan: BehaviorPlan, local_cancel: threading.Event | None) -> bool:
        return plan.cancel_event.is_set() or (local_cancel is not None and local_cancel.is_set())

    def _execute_node(
        self,
        plan: BehaviorPlan,
        node_id: str,
        deadline: float,
        local_cancel: threading.Event | None,
    ) -> dict[str, Any]:
        if self._cancelled(plan, local_cancel):
            return _result("cancelled", detail=plan.cancel_reason or "parallel_cancel")
        if time.monotonic() >= deadline:
            return _result("failed", error="timeout", detail="节点执行超时")
        node = plan.nodes[node_id]
        node_type = node["type"]
        plan.node_status[node_id] = {"status": "running", "started_ms": round((time.monotonic() - (plan.started_at or plan.created_at)) * 1000)}
        if node_type == "sequence":
            outcome = self._sequence(plan, node, deadline, local_cancel)
        elif node_type == "selector":
            outcome = self._selector(plan, node, deadline, local_cancel)
        elif node_type == "parallel":
            outcome = self._parallel(plan, node, deadline, local_cancel)
        elif node_type == "repeat":
            outcome = self._repeat(plan, node, deadline, local_cancel)
        elif node_type == "retry":
            outcome = self._retry(plan, node, deadline, local_cancel)
        elif node_type == "timeout":
            node_deadline = min(deadline, time.monotonic() + node["timeout_ms"] / 1000.0)
            outcome = self._execute_node(plan, node["child"], node_deadline, local_cancel)
            if outcome["status"] == "unknown" and time.monotonic() >= node_deadline:
                outcome = _result("failed", error="timeout", detail=f"节点 {node_id} 超时")
        elif node_type == "condition":
            outcome = _result("succeeded") if self._predicate(plan, node["predicate"]) else _result("failed", error="condition_false", detail=f"条件节点 {node_id} 不成立")
        else:
            outcome = self._leaf(plan, node, deadline, local_cancel)
        plan.node_status[node_id] = {
            "status": outcome["status"],
            "error": outcome.get("error"),
            "detail": outcome.get("detail"),
            "finished_ms": round((time.monotonic() - (plan.started_at or plan.created_at)) * 1000),
        }
        plan.evidence.append({"node_id": node_id, "type": node_type, **plan.node_status[node_id]})
        return outcome

    def _sequence(self, plan: BehaviorPlan, node: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        for child in node["children"]:
            outcome = self._execute_node(plan, child, deadline, local_cancel)
            if outcome["status"] != "succeeded":
                return outcome
        return _result("succeeded")

    def _selector(self, plan: BehaviorPlan, node: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        last = _result("failed", error="selector_exhausted", detail="所有 fallback 均失败")
        for child in node["children"]:
            outcome = self._execute_node(plan, child, deadline, local_cancel)
            if outcome["status"] == "succeeded":
                return outcome
            if outcome["status"] in {"unknown", "cancelled"}:
                return outcome
            last = outcome
        return last

    def _parallel(self, plan: BehaviorPlan, node: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        children = list(node["children"])
        results: list[dict[str, Any] | None] = [None] * len(children)
        branch_cancel = threading.Event()
        lock = threading.Lock()

        def run(index: int, child: str) -> None:
            result = self._execute_node(plan, child, deadline, branch_cancel)
            with lock:
                results[index] = result
                if node.get("join", "all") == "race" and result["status"] == "succeeded":
                    branch_cancel.set()
                elif node.get("join", "all") == "all" and result["status"] != "succeeded":
                    branch_cancel.set()

        threads = [threading.Thread(target=run, args=(index, child), daemon=True) for index, child in enumerate(children)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            branch_cancel.set()
            return _result("failed", error="timeout", detail="parallel 子节点超时")
        complete = [item for item in results if item is not None]
        if node.get("join", "all") == "race":
            success = next((item for item in complete if item["status"] == "succeeded"), None)
            if success is not None:
                return success
            unknown = next((item for item in complete if item["status"] == "unknown"), None)
            return unknown or (complete[-1] if complete else _result("failed", error="parallel_failed"))
        failure = next((item for item in complete if item["status"] == "unknown"), None)
        failure = failure or next((item for item in complete if item["status"] == "failed"), None)
        failure = failure or next((item for item in complete if item["status"] == "cancelled"), None)
        return failure or _result("succeeded")

    def _repeat(self, plan: BehaviorPlan, node: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        for _index in range(int(node["count"])):
            outcome = self._execute_node(plan, node["child"], deadline, local_cancel)
            if outcome["status"] != "succeeded":
                return outcome
        return _result("succeeded")

    def _retry(self, plan: BehaviorPlan, node: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        last = _result("failed", error="retry_exhausted")
        for attempt in range(int(node["max_attempts"])):
            last = self._execute_node(plan, node["child"], deadline, local_cancel)
            if last["status"] in {"succeeded", "cancelled", "unknown"}:
                return last
            if attempt + 1 < int(node["max_attempts"]) and not self._wait(plan, int(node.get("delay_ms", 0)), deadline, local_cancel):
                return _result("cancelled", detail="retry 等待被取消")
        return last

    @staticmethod
    def _compare(left: float, op: str, right: float) -> bool:
        return {"lt": left < right, "lte": left <= right, "gt": left > right, "gte": left >= right}[op]

    def _predicate(self, plan: BehaviorPlan, predicate: Mapping[str, Any]) -> bool:
        kind = predicate["type"]
        if kind == "player_present":
            return int(predicate["player_slot"]) in self.session.players
        if kind == "player_distance":
            player = self.session.players.get(int(predicate["player_slot"]))
            distance = None if player is None else player.get("d")
            return isinstance(distance, (int, float)) and self._compare(float(distance), str(predicate["op"]), float(predicate["value_m"]))
        if kind == "event_seen":
            return self.session.has_recent_event(str(predicate["event_type"]), int(predicate.get("within_ms", 5000)))
        if kind == "node_status":
            return plan.node_status.get(str(predicate["node_id"]), {}).get("status") == predicate["status"]
        if kind == "control_state":
            return self.session.control_state == predicate["value"]
        if kind == "estop":
            return self.session.estop is predicate["value"]
        elapsed = (time.monotonic() - (plan.started_at or plan.created_at)) * 1000.0
        return self._compare(elapsed, str(predicate["op"]), float(predicate["value"]))

    def _wait(self, plan: BehaviorPlan, duration_ms: int, deadline: float, local_cancel: threading.Event | None) -> bool:
        end = min(deadline, time.monotonic() + max(0, duration_ms) / 1000.0)
        while time.monotonic() < end:
            if self._cancelled(plan, local_cancel):
                return False
            time.sleep(min(0.05, end - time.monotonic()))
        return not self._cancelled(plan, local_cancel) and time.monotonic() <= deadline

    def _await_operation(self, plan: BehaviorPlan, outcome: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        status = outcome.get("status")
        if status == "succeeded":
            return _result("succeeded", wire=outcome)
        if status in {"failed", "unknown", "cancelled"}:
            return _result(str(status), error=outcome.get("error"), detail=outcome.get("detail"), wire=outcome)
        operation_id = outcome.get("op_id") or outcome.get("operation_id")
        if status != "accepted" or not isinstance(operation_id, str):
            return _result("unknown", error="operation_missing", detail="命令已接受但没有可关联 operation_id", wire=outcome)
        while time.monotonic() < deadline:
            if self._cancelled(plan, local_cancel):
                return _result("cancelled", detail=plan.cancel_reason or "parallel_cancel", wire=outcome)
            operation = self.session.wait_for_operation(operation_id, min(0.1, max(0.0, deadline - time.monotonic())))
            if operation is None:
                continue
            terminal = operation.get("status")
            if terminal == "succeeded":
                return _result("succeeded", operation=operation, wire=outcome)
            if terminal in {"failed", "cancelled", "unknown"}:
                return _result(str(terminal), error=operation.get("error") or operation.get("reason"), detail="Unity operation 未成功", operation=operation, wire=outcome)
        self.adapter.request_snapshot_evidence()
        return _result("unknown", error="operation_timeout", detail="operation 无终态证据；已请求 snapshot", wire=outcome)

    def _leaf(self, plan: BehaviorPlan, node: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        kind = node["type"]
        if kind == "wait":
            return _result("succeeded") if self._wait(plan, int(node["duration_ms"]), deadline, local_cancel) else _result("cancelled", detail="等待被取消")
        if kind == "condition":
            return _result("succeeded") if self._predicate(plan, node["predicate"]) else _result("failed", error="condition_false")
        if kind == "navigate":
            return self._await_operation(plan, self.adapter.navigate_wire(str(node["target_key"]), speed_mps=node.get("speed_mps")), deadline, local_cancel)
        if kind == "orbit":
            return self._await_operation(plan, self.adapter.orbit_wire(str(node["target_key"]), radius_m=float(node.get("radius_m", 2.0)), laps=int(node.get("laps", 1)), direction=str(node.get("direction", "cw")), speed_mps=node.get("speed_mps"), face_target=bool(node.get("face_target", True))), deadline, local_cancel)
        if kind == "move_relative":
            return self._await_operation(
                plan,
                self.adapter.move_relative_wire(
                    float(node["bearing_deg"]),
                    float(node["distance_m"]),
                    speed_mps=node.get("speed_mps"),
                    face_travel=bool(node.get("face_travel", True)),
                    allow_shorter=bool(node.get("allow_shorter", True)),
                ),
                deadline,
                local_cancel,
            )
        if kind == "approach":
            slot = int(node["player_slot"])
            outcome = self.adapter.follow_wire(slot, speed_mps=node.get("speed_mps"))
            accepted = self._accepted_or_failure(outcome)
            if accepted is not None:
                return accepted
            wanted = float(node.get("distance_m", 1.5))
            while time.monotonic() < deadline and not self._cancelled(plan, local_cancel):
                player = self.session.players.get(slot)
                if player is None:
                    return _result("failed", error="slot_unknown", detail="接近期间玩家离开")
                distance = player.get("d")
                if isinstance(distance, (int, float)) and float(distance) <= wanted:
                    self.adapter._stop_plan_domains(frozenset({"movement"}))
                    if bool(node.get("face_target", True)):
                        look = self.adapter.look_at(duration_ms=0, player_slot=slot)
                        if look.get("status") in {"failed", "unknown"}:
                            return _result(str(look["status"]), error=look.get("error"), detail=look.get("detail"))
                    return _result("succeeded", distance_m=float(distance))
                time.sleep(0.05)
            if self._cancelled(plan, local_cancel):
                return _result("cancelled")
            self.adapter.request_snapshot_evidence()
            return _result("unknown", error="approach_timeout", detail="接近玩家未取得距离终态证据")
        if kind == "follow":
            outcome = self.adapter.follow_wire(int(node["player_slot"]), speed_mps=node.get("speed_mps"))
            failure = self._accepted_or_failure(outcome)
            if failure is not None:
                return failure
            if not self._wait(plan, int(node["duration_ms"]), deadline, local_cancel):
                return _result("cancelled")
            self.adapter._stop_plan_domains(frozenset({"movement"}))
            return _result("succeeded")
        if kind == "explore":
            return self._explore(plan, node, deadline, local_cancel)
        if kind == "look_at":
            look = self.adapter.look_at(player_slot=int(node["player_slot"]), duration_ms=0)
            failure = self._accepted_or_failure(look)
            if failure is not None:
                return failure
            if not self._wait(plan, int(node["duration_ms"]), deadline, local_cancel):
                return _result("cancelled") if self._cancelled(plan, local_cancel) else _result("failed", error="timeout")
            cleared = self.adapter.clear_look_wire()
            return _result("succeeded", wire=cleared) if cleared.get("status") in {"succeeded", "accepted"} else _result(str(cleared.get("status", "failed")), error=cleared.get("error"), detail=cleared.get("detail"), wire=cleared)
        if kind == "act":
            return self._await_operation(plan, self.adapter.act(str(node["action_key"]), player_slot=node.get("player_slot"), loop=bool(node.get("loop", False))), deadline, local_cancel)
        if kind == "set_expression":
            return self._await_operation(plan, self.adapter.set_expression(str(node["expression_key"]), int(node.get("duration_ms", 0))), deadline, local_cancel)
        if kind == "say":
            outcome = self.adapter.say(text=str(node["text"]), estimated_delay_ms=node.get("estimated_delay_ms"), duration_ms=node.get("duration_ms"), action_key=node.get("action_key"))
            return _result("succeeded", wire=outcome) if outcome.get("status") in {"succeeded", "accepted"} else _result(str(outcome.get("status", "failed")), error=outcome.get("error"), detail=outcome.get("detail"), wire=outcome)
        if kind == "stop":
            outcome = self.adapter.stop(str(node.get("scope", "all")), _from_plan=True)
            return _result("succeeded", wire=outcome) if outcome.get("status") == "succeeded" else _result(str(outcome.get("status", "failed")), error=outcome.get("error"), detail=outcome.get("detail"), wire=outcome)
        return _result("failed", error="unsupported_node", detail=f"未实现节点 {kind}")

    @staticmethod
    def _accepted_or_failure(outcome: Mapping[str, Any]) -> dict[str, Any] | None:
        if outcome.get("status") == "accepted":
            return None
        if outcome.get("status") == "succeeded":
            return None
        return _result(str(outcome.get("status", "failed")), error=outcome.get("error"), detail=outcome.get("detail"), wire=outcome)

    def _explore(self, plan: BehaviorPlan, node: Mapping[str, Any], deadline: float, local_cancel: threading.Event | None) -> dict[str, Any]:
        region_key = str(node["region_key"])
        if self.session.local_navigation:
            return self._await_operation(
                plan,
                self.adapter.explore_region_wire(
                    region_key,
                    duration_ms=int(node.get("duration_ms", 60_000)),
                    strategy=str(node.get("strategy", "unvisited")),
                    speed_mps=node.get("speed_mps"),
                ),
                deadline,
                local_cancel,
            )
        anchors = [
            item for _item_id, item in sorted(self.session.catalogs["anchor"].items())
            if item.get("region_key") == region_key
        ]
        if not anchors:
            return _result("failed", error="target_missing", detail=f"区域 {region_key} 没有已发布 Anchor")
        strategy = str(node.get("strategy", "unvisited"))
        if strategy not in {"unvisited", "patrol"}:
            return _result("failed", error="invalid_param", detail="strategy 必须是 unvisited|patrol")
        end = min(deadline, time.monotonic() + int(node.get("duration_ms", 60_000)) / 1000.0)
        index = 0
        visited: set[str] = set()
        completed = 0
        while time.monotonic() < end and not self._cancelled(plan, local_cancel):
            available = [item for item in anchors if strategy == "patrol" or item.get("semantic_key") not in visited]
            if not available:
                return _result("succeeded", visited=sorted(visited))
            item = available[index % len(available)]
            index += 1
            key = str(item.get("semantic_key"))
            outcome = self._await_operation(plan, self.adapter.navigate_wire(key, speed_mps=node.get("speed_mps")), end, local_cancel)
            if outcome["status"] != "succeeded":
                return outcome
            visited.add(key)
            completed += 1
        if self._cancelled(plan, local_cancel):
            return _result("cancelled")
        return _result("succeeded", visited=sorted(visited), completed=completed)


def single_node_graph(node_type: str, **arguments: Any) -> dict[str, Any]:
    """为类型化工具构造一节点行为图。"""
    node = {"id": "root", "type": node_type, **arguments}
    return {"entry": "root", "nodes": [node]}


__all__ = [
    "BehaviorGraphCompiler",
    "BehaviorGraphError",
    "BehaviorPlan",
    "BehaviorPlanManager",
    "PLAN_STATUSES",
    "single_node_graph",
]
