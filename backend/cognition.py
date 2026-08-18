"""独立后端使用的小型确定性认知循环。

实时肢体调度器仍是 60 Hz 输出的唯一事实来源。本模块只保存有界的观测数据、
估计数据新鲜度与置信度，并把高层计划归一化为严格的 JSON 结构。后续接入 LLM
时只需产出相同格式的计划输入即可；它无需运行在控制线程中，也不需要直接产生
设备帧。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
    if depth > 3:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "params": _safe(self.params),
            "timeout_s": self.timeout_s,
            "expected": list(self.expected),
            "abort_if": list(self.abort_if),
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
            for field in ("timeout_s", "expected", "abort_if"):
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
            steps.append(PlanStep(str(index + 1), action, _safe(params), timeout_s, expected, abort_if))
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._source_provider = source_provider
        self.estimator = StateEstimator(clock=clock)
        self.planner = SkillPlanner(clock=clock)
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
        with self._lock:
            self._plan = plan
            self._replan_reason = None if plan.status == "planned" else plan.reason
        return plan.to_dict()

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
                "at_monotonic": now,
            })
            if not accepted:
                self._replan_reason = "execution_rejected"

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
]
