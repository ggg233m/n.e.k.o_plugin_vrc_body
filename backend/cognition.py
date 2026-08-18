"""独立后端使用的小型确定性认知循环。

实时肢体调度器仍是 60 Hz 输出的唯一事实来源。本模块只保存有界的观测数据、
估计数据新鲜度与置信度，并把高层计划归一化为严格的 JSON 结构。后续接入 LLM
时只需产出相同格式的计划输入即可；它无需运行在控制线程中，也不需要直接产生
设备帧。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import threading
import time
import uuid
from typing import Any, Callable, Iterable, Mapping


def _text(value: Any, *, default: str = "", limit: int = 128) -> str:
    result = str(value if value is not None else default).strip()
    return result[:limit]


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _confidence(value: Any) -> float:
    return min(1.0, max(0.0, _finite(value, 0.0)))


def _safe(value: Any, *, depth: int = 0) -> Any:
    """保证适配器与 LLM 之间传递的数据是有界且 JSON 安全的。"""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if isinstance(value, str):
        return value[:512]
    # 计划 schema 的正常层级会到 goal -> steps -> preconditions -> condition。
    # 保留该结构，同时仍通过每层条目数和字符串长度限制总数据规模。
    if depth > 5:
        return _text(value, limit=256)
    if isinstance(value, Mapping):
        return {
            _text(key, limit=64): _safe(item, depth=depth + 1)
            for key, item in list(value.items())[:32]
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe(item, depth=depth + 1) for item in list(value)[:32]]
    return _text(value, limit=256)


def _string_sequence(value: Any, *, limit: int = 8) -> tuple[str, ...] | None:
    """归一化一个有界字符串数组，输入非法时返回 ``None``。"""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return None
    result: list[str] = []
    for item in list(value)[:limit]:
        if not isinstance(item, str):
            return None
        text = item.strip()[:96]
        if text:
            result.append(text)
    return tuple(result)


def _optional_threshold(value: Any, *, name: str, maximum: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum:g}")
    return result


def _strict_optional_string(value: Any, *, name: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    result = value.strip()
    if len(result) > limit:
        raise ValueError(f"{name} must contain at most {limit} characters")
    return result or None


def _aliased_string(
    value: Mapping[str, Any],
    primary: str,
    alias: str,
    *,
    limit: int,
) -> str | None:
    left = _strict_optional_string(value.get(primary), name=primary, limit=limit)
    right = _strict_optional_string(value.get(alias), name=alias, limit=limit)
    if primary in value and alias in value and left != right:
        raise ValueError(f"{primary} conflicts with alias {alias}")
    return left if primary in value else right


@dataclass(frozen=True)
class WorldPrecondition:
    """一个动作对最新世界状态的显式、可序列化约束。"""

    kind: str
    entity_id: str | None = None
    event_type: str | None = None
    target_id: str | None = None
    source: str | None = None
    label: str | None = None
    state: str | None = None
    min_confidence: float | None = None
    max_age_ms: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorldPrecondition":
        aliases = {
            "entity": "entity_visible",
            "visible_entity": "entity_visible",
            "event": "event_recent",
            "available": "world_available",
        }
        raw_kinds = [
            _strict_optional_string(value.get(name), name=name, limit=48)
            for name in ("kind", "type")
            if name in value
        ]
        normalized_kinds = [aliases.get(item or "", item or "") for item in raw_kinds]
        if len(set(normalized_kinds)) > 1:
            raise ValueError("kind conflicts with alias type")
        kind = normalized_kinds[0] if normalized_kinds else ""
        if kind not in {"world_available", "entity_visible", "event_recent"}:
            raise ValueError(
                "precondition kind must be world_available, entity_visible, or event_recent"
            )
        allowed_fields = {
            "world_available": {"kind", "type", "max_age_ms"},
            "entity_visible": {
                "kind", "type", "entity_id", "id", "source", "label", "state",
                "min_confidence", "max_age_ms",
            },
            "event_recent": {
                "kind", "type", "event_type", "event", "target_id", "source",
                "min_confidence", "max_age_ms",
            },
        }[kind]
        unknown = sorted(str(key) for key in value if str(key) not in allowed_fields)
        if unknown:
            raise ValueError(f"unknown precondition fields: {', '.join(unknown)}")
        entity_id = _aliased_string(value, "entity_id", "id", limit=96)
        event_type = _aliased_string(value, "event_type", "event", limit=64)
        if kind == "entity_visible" and entity_id is None:
            raise ValueError("entity_visible precondition requires entity_id")
        if kind == "event_recent" and event_type is None:
            raise ValueError("event_recent precondition requires event_type")
        min_confidence = _optional_threshold(
            value.get("min_confidence"), name="min_confidence", maximum=1.0
        )
        max_age_ms = _optional_threshold(
            value.get("max_age_ms"), name="max_age_ms", maximum=60_000.0
        )
        if kind in {"entity_visible", "event_recent"} and min_confidence is None:
            min_confidence = 0.5
        if kind == "event_recent" and max_age_ms is None:
            max_age_ms = 2_000.0
        condition = cls(
            kind=kind,
            entity_id=entity_id,
            event_type=event_type,
            target_id=_strict_optional_string(
                value.get("target_id"), name="target_id", limit=96
            ),
            source=_strict_optional_string(value.get("source"), name="source", limit=48),
            label=_strict_optional_string(value.get("label"), name="label", limit=64),
            state=_strict_optional_string(value.get("state"), name="state", limit=64),
            min_confidence=min_confidence,
            max_age_ms=max_age_ms,
        )
        if kind == "world_available" and any((
            condition.entity_id,
            condition.event_type,
            condition.target_id,
            condition.source,
            condition.label,
            condition.state,
            condition.min_confidence is not None,
        )):
            raise ValueError("world_available only supports max_age_ms")
        if kind == "entity_visible" and any((condition.event_type, condition.target_id)):
            raise ValueError("entity_visible does not support event_type or target_id")
        if kind == "event_recent" and any((condition.entity_id, condition.label, condition.state)):
            raise ValueError("event_recent does not support entity_id, label, or state")
        return condition

    def to_dict(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "entity_id": self.entity_id,
            "event_type": self.event_type,
            "target_id": self.target_id,
            "source": self.source,
            "label": self.label,
            "state": self.state,
            "min_confidence": self.min_confidence,
            "max_age_ms": self.max_age_ms,
        }
        return {key: value for key, value in result.items() if value is not None}


class WorldPreconditionGate:
    """依据世界状态快照检查动作前置条件，不接触实时控制线程。"""

    @staticmethod
    def normalize(value: Any) -> tuple[WorldPrecondition, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("preconditions must be an array")
        if len(value) > 16:
            raise ValueError("preconditions must contain at most 16 items")
        result: list[WorldPrecondition] = []
        for index, item in enumerate(value):
            if isinstance(item, WorldPrecondition):
                item = item.to_dict()
            if not isinstance(item, Mapping):
                raise ValueError(f"preconditions[{index}] must be an object")
            try:
                result.append(WorldPrecondition.from_mapping(item))
            except ValueError as exc:
                raise ValueError(f"preconditions[{index}]: {exc}") from exc
        return tuple(result)

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        return _finite(value, default)

    @staticmethod
    def _source_matches(item: Mapping[str, Any], source: str | None) -> bool:
        if source is None:
            return True
        raw = item.get("source")
        sources = raw if isinstance(raw, (list, tuple)) else [raw]
        return source in {str(name) for name in sources if name is not None}

    @staticmethod
    def _failure(
        index: int,
        condition: WorldPrecondition,
        code: str,
        message: str,
        *,
        observed: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "index": index,
            "code": code,
            "message": message[:256],
            "precondition": condition.to_dict(),
        }
        if observed is not None:
            result["observed"] = _safe(observed)
        return result

    def evaluate_normalized(
        self,
        conditions: Iterable[WorldPrecondition],
        world: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = tuple(conditions)
        if not normalized:
            return {"passed": True, "checked": 0, "preconditions": [], "failures": []}
        snapshot = world if isinstance(world, Mapping) else {}
        entities = [
            item for item in (snapshot.get("entities") or ()) if isinstance(item, Mapping)
        ]
        events = [
            item for item in (snapshot.get("events") or ()) if isinstance(item, Mapping)
        ]
        failures: list[dict[str, Any]] = []
        for index, condition in enumerate(normalized):
            if condition.kind == "world_available":
                if not bool(snapshot.get("available")):
                    failures.append(self._failure(
                        index,
                        condition,
                        "world_unavailable",
                        "no recent world observation is available",
                    ))
                    continue
                age_ms = self._number(
                    (snapshot.get("status") or {}).get("last_observation_age_ms"),
                    math.inf,
                )
                if condition.max_age_ms is not None and age_ms > condition.max_age_ms:
                    failures.append(self._failure(
                        index,
                        condition,
                        "world_observation_too_old",
                        f"latest world observation age {age_ms:g} ms exceeds {condition.max_age_ms:g} ms",
                        observed={"age_ms": age_ms},
                    ))
                continue

            if condition.kind == "entity_visible":
                entity = next(
                    (item for item in entities if str(item.get("id") or "") == condition.entity_id),
                    None,
                )
                if entity is None or entity.get("visible") is False:
                    failures.append(self._failure(
                        index,
                        condition,
                        "entity_not_visible",
                        f"entity '{condition.entity_id}' is missing, expired, or not visible",
                    ))
                    continue
                confidence = self._number(entity.get("confidence"))
                age_ms = self._number(entity.get("age_ms"), math.inf)
                if condition.min_confidence is not None and confidence < condition.min_confidence:
                    failures.append(self._failure(
                        index,
                        condition,
                        "entity_low_confidence",
                        f"entity '{condition.entity_id}' confidence {confidence:g} is below {condition.min_confidence:g}",
                        observed=entity,
                    ))
                    continue
                if condition.max_age_ms is not None and age_ms > condition.max_age_ms:
                    failures.append(self._failure(
                        index,
                        condition,
                        "entity_observation_too_old",
                        f"entity '{condition.entity_id}' age {age_ms:g} ms exceeds {condition.max_age_ms:g} ms",
                        observed=entity,
                    ))
                    continue
                if condition.label is not None and str(entity.get("label") or "") != condition.label:
                    failures.append(self._failure(
                        index,
                        condition,
                        "entity_label_mismatch",
                        f"entity '{condition.entity_id}' does not have label '{condition.label}'",
                        observed=entity,
                    ))
                    continue
                if condition.state is not None and str(entity.get("state") or "") != condition.state:
                    failures.append(self._failure(
                        index,
                        condition,
                        "entity_state_mismatch",
                        f"entity '{condition.entity_id}' is not in state '{condition.state}'",
                        observed=entity,
                    ))
                    continue
                if not self._source_matches(entity, condition.source):
                    failures.append(self._failure(
                        index,
                        condition,
                        "entity_source_mismatch",
                        f"entity '{condition.entity_id}' was not observed by source '{condition.source}'",
                        observed=entity,
                    ))
                continue

            candidates = [
                item
                for item in events
                if str(item.get("type") or item.get("kind") or "") == condition.event_type
                and (
                    condition.target_id is None
                    or str(item.get("target_id") or "") == condition.target_id
                )
                and self._source_matches(item, condition.source)
            ]
            event = min(
                candidates,
                key=lambda item: self._number(item.get("age_ms"), math.inf),
                default=None,
            )
            if event is None:
                failures.append(self._failure(
                    index,
                    condition,
                    "event_not_recent",
                    f"event '{condition.event_type}' was not observed for the requested target/source",
                ))
                continue
            confidence = self._number(event.get("confidence"))
            age_ms = self._number(event.get("age_ms"), math.inf)
            if condition.min_confidence is not None and confidence < condition.min_confidence:
                failures.append(self._failure(
                    index,
                    condition,
                    "event_low_confidence",
                    f"event '{condition.event_type}' confidence {confidence:g} is below {condition.min_confidence:g}",
                    observed=event,
                ))
                continue
            if condition.max_age_ms is not None and age_ms > condition.max_age_ms:
                failures.append(self._failure(
                    index,
                    condition,
                    "event_observation_too_old",
                    f"event '{condition.event_type}' age {age_ms:g} ms exceeds {condition.max_age_ms:g} ms",
                    observed=event,
                ))
        return {
            "passed": not failures,
            "checked": len(normalized),
            "preconditions": [item.to_dict() for item in normalized],
            "failures": failures,
        }

    def evaluate(self, value: Any, world: Mapping[str, Any] | None) -> dict[str, Any]:
        try:
            conditions = self.normalize(value)
        except ValueError as exc:
            return {
                "passed": False,
                "checked": 0,
                "preconditions": [],
                "failures": [{
                    "index": None,
                    "code": "invalid_world_precondition",
                    "message": str(exc)[:256],
                }],
            }
        return self.evaluate_normalized(conditions, world)


@dataclass(frozen=True)
class ObservationRecord:
    """来自感知适配器的一条带时间戳的观测记录。"""

    source: str
    kind: str
    data: Mapping[str, Any]
    confidence: float
    observed_at: float
    frame_id: str | None = None

    def to_dict(self, *, now: float) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "confidence": round(self.confidence, 4),
            "age_ms": round(max(0.0, now - self.observed_at) * 1000.0, 1),
            "frame_id": self.frame_id,
            "data": _safe(self.data),
        }


class StateEstimator:
    """融合新鲜度与置信度，不把缺失数据当作 false 处理。"""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_observations: int = 128,
        stale_after_s: float = 5.0,
    ) -> None:
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        self._clock = clock
        self._observations: deque[ObservationRecord] = deque(maxlen=max_observations)
        self._stale_after_s = min(60.0, max(0.1, float(stale_after_s)))
        self._lock = threading.RLock()

    def ingest(
        self,
        source: str,
        kind: str,
        data: Mapping[str, Any] | None = None,
        *,
        confidence: float = 0.0,
        observed_at: float | None = None,
        frame_id: str | None = None,
    ) -> dict[str, Any]:
        now = self._clock()
        timestamp = min(now, _finite(observed_at, now))
        record = ObservationRecord(
            source=_text(source, default="unknown", limit=48) or "unknown",
            kind=_text(kind, default="unknown", limit=64) or "unknown",
            data=_safe(data if isinstance(data, Mapping) else {}) or {},
            confidence=_confidence(confidence),
            observed_at=timestamp,
            frame_id=_text(frame_id, limit=96) or None,
        )
        with self._lock:
            self._observations.append(record)
        return self.snapshot(now=now)

    def snapshot(
        self,
        *,
        now: float | None = None,
        source_status: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        timestamp = self._clock() if now is None else now
        with self._lock:
            records = list(self._observations)
        recent = [
            item for item in records
            if timestamp - item.observed_at <= self._stale_after_s
        ]
        sources: dict[str, dict[str, Any]] = {}
        for item in records:
            status = sources.setdefault(
                item.source,
                {"last_seen_ms": None, "confidence": 0.0, "observation_count": 0, "stale": True},
            )
            status["observation_count"] += 1
            age_ms = max(0.0, timestamp - item.observed_at) * 1000.0
            if status["last_seen_ms"] is None or age_ms < status["last_seen_ms"]:
                status["last_seen_ms"] = round(age_ms, 1)
                status["confidence"] = round(item.confidence, 4)
                status["stale"] = age_ms > self._stale_after_s * 1000.0
        for name, status in (source_status or {}).items():
            target = sources.setdefault(_text(name, limit=48), {})
            target["runtime"] = _safe(status)

        uncertainties: list[str] = []
        if not recent:
            uncertainties.append("no_recent_cognition_observation")
        if any(item.confidence < 0.5 for item in recent):
            uncertainties.append("low_confidence_observation")
        if any(status.get("stale") for status in sources.values()):
            uncertainties.append("stale_observation_source")
        mode = "nominal" if recent and not uncertainties else ("degraded" if recent else "unknown")
        return {
            "mode": mode,
            "updated_at_monotonic": timestamp,
            "uncertainties": list(dict.fromkeys(uncertainties)),
            "sources": sources,
            "observations": [item.to_dict(now=timestamp) for item in recent[-32:]],
        }


@dataclass(frozen=True)
class PlanStep:
    id: str
    action: str
    params: Mapping[str, Any]
    timeout_s: float = 5.0
    expected: tuple[str, ...] = ()
    abort_if: tuple[str, ...] = ()
    preconditions: tuple[WorldPrecondition, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "params": _safe(self.params),
            "timeout_s": self.timeout_s,
            "expected": list(self.expected),
            "abort_if": list(self.abort_if),
            "preconditions": [item.to_dict() for item in self.preconditions],
        }


@dataclass(frozen=True)
class Plan:
    id: str
    goal: Mapping[str, Any]
    steps: tuple[PlanStep, ...]
    status: str
    reason: str | None = None
    created_at_monotonic: float = 0.0
    current_index: int = 0
    replan_required: bool = False
    precondition_check: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": _safe(self.goal),
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "reason": self.reason,
            "created_at_monotonic": self.created_at_monotonic,
            "current_index": self.current_index,
            "replan_required": self.replan_required,
            "precondition_check": _safe(self.precondition_check),
        }


class SkillPlanner:
    """确定性的基线规划器；LLM 输出可以对齐同一套 schema。"""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock

    def plan(self, goal: Mapping[str, Any] | None) -> Plan:
        now = self._clock()
        value = _safe(goal if isinstance(goal, Mapping) else {}) or {}
        plan_id = str(uuid.uuid4())
        raw_steps = value.get("steps") if isinstance(value, Mapping) else None
        if raw_steps is None and isinstance(value, Mapping) and value.get("action"):
            shorthand = {
                "action": value.get("action"),
                "params": value.get("params") or {},
            }
            for field in (
                "timeout_s",
                "expected",
                "abort_if",
                "preconditions",
                "world_preconditions",
            ):
                if field in value:
                    shorthand[field] = value[field]
            raw_steps = [shorthand]
        if not isinstance(raw_steps, (list, tuple)) or not raw_steps:
            return Plan(
                id=plan_id,
                goal=value,
                steps=(),
                status="blocked",
                reason="goal must contain an action or a non-empty steps list",
                created_at_monotonic=now,
            )
        steps: list[PlanStep] = []
        for index, raw in enumerate(list(raw_steps)[:16]):
            if not isinstance(raw, Mapping):
                continue
            action = _text(raw.get("action") or raw.get("kind"), limit=64)
            params = raw.get("params")
            if not action or not isinstance(params, Mapping):
                continue
            timeout_s = min(300.0, max(0.1, _finite(raw.get("timeout_s"), 5.0)))
            expected = _string_sequence(raw.get("expected"))
            abort_if = _string_sequence(raw.get("abort_if"))
            if expected is None or abort_if is None:
                return Plan(
                    id=plan_id,
                    goal=value,
                    steps=(),
                    status="blocked",
                    reason="expected and abort_if must be string arrays",
                    created_at_monotonic=now,
                )
            try:
                if "preconditions" in raw and "world_preconditions" in raw:
                    raise ValueError(
                        "preconditions must not be combined with alias world_preconditions"
                    )
                preconditions = WorldPreconditionGate.normalize(
                    raw.get("preconditions", raw.get("world_preconditions"))
                )
            except ValueError as exc:
                message = f"step {index + 1}: {exc}"
                return Plan(
                    id=plan_id,
                    goal=value,
                    steps=(),
                    status="blocked",
                    reason=message,
                    created_at_monotonic=now,
                    replan_required=True,
                    precondition_check={
                        "passed": False,
                        "checked": 0,
                        "preconditions": [],
                        "failures": [{
                            "index": index,
                            "code": "invalid_world_precondition",
                            "message": message[:256],
                        }],
                    },
                )
            steps.append(PlanStep(
                str(index + 1),
                action,
                _safe(params),
                timeout_s,
                expected,
                abort_if,
                preconditions,
            ))
        if not steps:
            return Plan(
                id=plan_id,
                goal=value,
                steps=(),
                status="blocked",
                reason="goal contains no valid action steps",
                created_at_monotonic=now,
            )
        return Plan(
            id=plan_id,
            goal=value,
            steps=tuple(steps),
            status="planned",
            created_at_monotonic=now,
        )


class CognitionRuntime:
    """由 BackendService 持有的有界状态估计 / 规划 / 反馈状态。"""

    def __init__(
        self,
        source_provider: Callable[[], Mapping[str, Mapping[str, Any]]],
        *,
        world_provider: Callable[[], Mapping[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._source_provider = source_provider
        self._world_provider = world_provider or (lambda: {})
        self.estimator = StateEstimator(clock=clock)
        self.planner = SkillPlanner(clock=clock)
        self.precondition_gate = WorldPreconditionGate()
        self._lock = threading.RLock()
        self._plan: Plan | None = None
        self._feedback: deque[dict[str, Any]] = deque(maxlen=32)
        self._replan_reason: str | None = None
        self._action_count = 0

    def observe(
        self,
        source: str,
        kind: str,
        data: Mapping[str, Any] | None = None,
        *,
        confidence: float = 0.0,
        observed_at: float | None = None,
        frame_id: str | None = None,
    ) -> dict[str, Any]:
        return self.estimator.ingest(
            source,
            kind,
            data,
            confidence=confidence,
            observed_at=observed_at,
            frame_id=frame_id,
        )

    def plan(self, goal: Mapping[str, Any] | None) -> dict[str, Any]:
        plan = self.planner.plan(goal)
        if plan.status == "planned" and plan.steps:
            check = self.check_preconditions(plan.steps[0].preconditions)
            plan = replace(plan, precondition_check=check)
            if not check["passed"]:
                plan = replace(
                    plan,
                    status="blocked",
                    reason="world preconditions are not satisfied",
                    replan_required=True,
                )
        with self._lock:
            self._plan = plan
            self._replan_reason = (
                None
                if plan.status == "planned"
                else (
                    "world_precondition_failed"
                    if plan.precondition_check and not plan.precondition_check.get("passed")
                    else plan.reason
                )
            )
        return plan.to_dict()

    def check_preconditions(self, value: Any) -> dict[str, Any]:
        """检查动作前置条件；无条件时不读取世界状态。"""
        try:
            conditions = self.precondition_gate.normalize(value)
        except ValueError:
            return self.precondition_gate.evaluate(value, {})
        if not conditions:
            return self.precondition_gate.evaluate_normalized(conditions, {})
        try:
            world = self._world_provider()
        except Exception as exc:
            return {
                "passed": False,
                "checked": len(conditions),
                "preconditions": [item.to_dict() for item in conditions],
                "failures": [{
                    "index": None,
                    "code": "world_state_unavailable",
                    "message": f"{type(exc).__name__}: {exc}"[:256],
                }],
            }
        return self.precondition_gate.evaluate_normalized(conditions, world)

    def record_action(self, kind: str, result: Mapping[str, Any]) -> None:
        accepted = bool(result.get("accepted"))
        now = self._clock()
        with self._lock:
            self._action_count += 1
            self._feedback.append({
                "type": "action_result",
                "action": _text(kind, limit=64),
                "accepted": accepted,
                "action_id": _text(result.get("action_id"), limit=96) or None,
                "state": _text(result.get("state"), limit=48) or None,
                "reason": _text(result.get("reason"), limit=256) or None,
                "reason_code": _text(result.get("reason_code"), limit=96) or None,
                "replan_required": bool(result.get("replan_required")),
                "precondition_check": _safe(result.get("precondition_check")),
                "at_monotonic": now,
            })
            if not accepted:
                self._replan_reason = (
                    _text(result.get("replan_reason"), limit=96)
                    or _text(result.get("reason_code"), limit=96)
                    or "execution_rejected"
                )

    def feedback(self, value: Mapping[str, Any] | None) -> dict[str, Any]:
        data = _safe(value if isinstance(value, Mapping) else {}) or {}
        kind = _text(data.get("type") or data.get("kind"), default="feedback", limit=64)
        now = self._clock()
        with self._lock:
            self._feedback.append({"type": kind, "data": data, "at_monotonic": now})
            if kind in {"timeout", "failed", "uncertain", "world_changed", "user_interrupt"}:
                self._replan_reason = kind
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        try:
            sources = self._source_provider()
        except Exception as exc:
            sources = {"runtime": {"error": f"{type(exc).__name__}: {exc}"[:256]}}
        with self._lock:
            plan = self._plan.to_dict() if self._plan is not None else None
            feedback = list(self._feedback)[-16:]
            replan_reason = self._replan_reason
            action_count = self._action_count
        return {
            "state": self.estimator.snapshot(now=now, source_status=sources),
            "plan": plan,
            "replan_required": replan_reason is not None,
            "replan_reason": replan_reason,
            "feedback": feedback,
            "metrics": {"action_count": action_count},
        }


__all__ = [
    "CognitionRuntime",
    "ObservationRecord",
    "Plan",
    "PlanStep",
    "SkillPlanner",
    "StateEstimator",
    "WorldPrecondition",
    "WorldPreconditionGate",
]
