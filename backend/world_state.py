"""线程安全、带不确定性标记的视觉世界状态。

身体调度器刻意不依赖本模块。视觉检测器和语义模型可以在此发布观测，同时
AnyaDance 继续独立运行 60 Hz 控制循环。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Iterable, Mapping


def _text(value: Any, *, default: str = "", limit: int = 128) -> str:
    result = str(value if value is not None else default).strip()
    return result[:limit]


def _confidence(value: Any, default: float = 0.0) -> float:
    try:
        result = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError):
        result = default
    if not math.isfinite(result):
        result = default
    return min(1.0, max(0.0, result))


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    """在输出交给 LLM 前，将模型结果限制为有界的 JSON 类数据。"""
    if depth > 2:
        return _text(value, limit=256)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, Mapping):
        return {
            _text(key, limit=64): _safe_value(item, depth=depth + 1)
            for key, item in list(value.items())[:24]
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, depth=depth + 1) for item in list(value)[:24]]
    return _text(value, limit=256)


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    values = tuple(_finite(item, math.nan) for item in value)
    if not all(math.isfinite(item) for item in values):
        return None
    # 视觉适配器发布归一化屏幕坐标；钳制可以防止错误检测结果污染后续规划提示。
    return tuple(min(1.0, max(0.0, item)) for item in values)  # type: ignore[return-value]


def _sources(value: Any, default: str) -> tuple[str, ...]:
    raw = list(value)[:8] if isinstance(value, (list, tuple, set)) else [value or default]
    result: list[str] = []
    for item in raw:
        name = _text(item, limit=48)
        if name and name not in result:
            result.append(name)
    return tuple(result or [default])


@dataclass(frozen=True)
class WorldEntity:
    """一个有界的视觉世界实体假设。"""

    id: str
    label: str
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    state: str = "unknown"
    attributes: Mapping[str, Any] = None  # type: ignore[assignment]
    relations: tuple[Mapping[str, Any], ...] = ()
    source: tuple[str, ...] = ()
    observed_at: float = 0.0
    ttl_s: float = 2.0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        now: float,
        default_source: str = "vision",
        default_ttl_s: float = 2.0,
    ) -> "WorldEntity":
        entity_id = _text(value.get("id"), limit=96)
        if not entity_id:
            raise ValueError("world entity id must not be empty")
        attributes = value.get("attributes")
        safe_attributes = (
            {str(key)[:64]: _safe_value(item) for key, item in list(attributes.items())[:24]}
            if isinstance(attributes, Mapping)
            else {}
        )
        raw_relations = value.get("relations")
        relations = tuple(
            {
                str(key)[:64]: _safe_value(item)
                for key, item in list(item.items())[:16]
            }
            for item in (list(raw_relations)[:16] if isinstance(raw_relations, (list, tuple)) else ())
            if isinstance(item, Mapping)
        )
        ttl_s = min(60.0, max(0.1, _finite(value.get("ttl_s"), default_ttl_s)))
        observed_at = min(now, _finite(value.get("observed_at"), now))
        return cls(
            id=entity_id,
            label=_text(value.get("label"), default="unknown", limit=64) or "unknown",
            confidence=_confidence(value.get("confidence")),
            bbox=_bbox(value.get("bbox")),
            state=_text(value.get("state"), default="unknown", limit=64) or "unknown",
            attributes=safe_attributes,
            relations=relations,
            source=_sources(value.get("source"), default_source),
            observed_at=observed_at,
            ttl_s=ttl_s,
        )

    def expired(self, now: float) -> bool:
        return now > self.observed_at + self.ttl_s

    def to_dict(self, *, now: float) -> dict[str, Any]:
        age_ms = max(0.0, (now - self.observed_at) * 1000.0)
        return {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "state": self.state,
            "attributes": dict(self.attributes or {}),
            "relations": [dict(item) for item in self.relations],
            "source": list(self.source),
            "age_ms": round(age_ms, 1),
            "ttl_ms": round(max(0.0, self.ttl_s * 1000.0 - age_ms), 1),
            "visible": not self.expired(now),
        }


@dataclass(frozen=True)
class WorldEvent:
    """检测器或 VLM 发出的短时事件假设。"""

    kind: str
    target_id: str | None = None
    confidence: float = 0.0
    data: Mapping[str, Any] = None  # type: ignore[assignment]
    source: tuple[str, ...] = ()
    observed_at: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        now: float,
        default_source: str = "vision",
    ) -> "WorldEvent":
        return cls(
            kind=_text(value.get("type") or value.get("kind"), default="unknown", limit=64) or "unknown",
            target_id=_text(value.get("target_id"), limit=96) or None,
            confidence=_confidence(value.get("confidence")),
            data=_safe_value(value.get("data") if isinstance(value.get("data"), Mapping) else {}) or {},
            source=_sources(value.get("source"), default_source),
            observed_at=min(now, _finite(value.get("observed_at"), now)),
        )

    def to_dict(self, *, now: float) -> dict[str, Any]:
        return {
            "type": self.kind,
            "target_id": self.target_id,
            "confidence": round(self.confidence, 4),
            "data": dict(self.data or {}),
            "source": list(self.source),
            "age_ms": round(max(0.0, (now - self.observed_at) * 1000.0), 1),
        }


class WorldStateStore:
    """供感知层和规划层共享的有界线程安全存储。"""

    def __init__(
        self,
        *,
        max_entities: int = 128,
        event_history_size: int = 64,
        default_ttl_s: float = 2.0,
        clock: Any = time.monotonic,
    ) -> None:
        if max_entities < 1 or event_history_size < 1:
            raise ValueError("world state bounds must be positive")
        self.max_entities = max_entities
        self.event_history_size = event_history_size
        self.default_ttl_s = min(60.0, max(0.1, float(default_ttl_s)))
        self._clock = clock
        self._lock = threading.RLock()
        self._entities: dict[str, WorldEntity] = {}
        self._events: deque[WorldEvent] = deque(maxlen=event_history_size)
        self._backend: dict[str, dict[str, Any]] = {}
        self._last_observation_at: float | None = None
        self._observation_count = 0

    def set_backend_status(self, name: str, status: Mapping[str, Any]) -> None:
        backend_name = _text(name, default="unknown", limit=48) or "unknown"
        with self._lock:
            self._backend[backend_name] = {
                str(key)[:64]: _safe_value(item) for key, item in list(status.items())[:24]
            }

    def ingest(
        self,
        entities: Iterable[WorldEntity | Mapping[str, Any]] = (),
        events: Iterable[WorldEvent | Mapping[str, Any]] = (),
        *,
        source: str = "vision",
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        now = self._clock() if observed_at is None else _finite(observed_at, self._clock())
        normalized_entities: list[WorldEntity] = []
        for item in entities:
            try:
                entity = item if isinstance(item, WorldEntity) else WorldEntity.from_mapping(
                    item,
                    now=now,
                    default_source=source,
                    default_ttl_s=self.default_ttl_s,
                )
            except (TypeError, ValueError):
                continue
            normalized_entities.append(entity)
        normalized_events: list[WorldEvent] = []
        for item in events:
            try:
                event = item if isinstance(item, WorldEvent) else WorldEvent.from_mapping(
                    item,
                    now=now,
                    default_source=source,
                )
            except (TypeError, ValueError):
                continue
            normalized_events.append(event)
        with self._lock:
            for entity in normalized_entities:
                self._entities[entity.id] = entity
            if len(self._entities) > self.max_entities:
                ranked = sorted(
                    self._entities.values(),
                    key=lambda item: (item.observed_at, item.confidence),
                    reverse=True,
                )[: self.max_entities]
                self._entities = {item.id: item for item in ranked}
            self._events.extend(normalized_events)
            self._last_observation_at = now
            self._observation_count += 1
        return self.snapshot(now=now)

    def clear(self) -> None:
        with self._lock:
            self._entities.clear()
            self._events.clear()
            self._last_observation_at = None
            self._observation_count = 0

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = self._clock() if now is None else now
        with self._lock:
            visible = [item for item in self._entities.values() if not item.expired(timestamp)]
            visible.sort(key=lambda item: (item.confidence, item.observed_at), reverse=True)
            recent_events = [
                item.to_dict(now=timestamp)
                for item in self._events
                if timestamp - item.observed_at <= max(5.0, self.default_ttl_s * 4.0)
            ]
            age_ms = (
                None
                if self._last_observation_at is None
                else round(max(0.0, timestamp - self._last_observation_at) * 1000.0, 1)
            )
            return {
                "available": bool(visible or recent_events),
                "entities": [item.to_dict(now=timestamp) for item in visible],
                "events": recent_events,
                "uncertainties": [] if visible else ["no_recent_visual_observation"],
                "status": {
                    "entity_count": len(visible),
                    "event_count": len(recent_events),
                    "last_observation_age_ms": age_ms,
                    "observation_count": self._observation_count,
                },
                "backends": {key: dict(value) for key, value in self._backend.items()},
            }


__all__ = ["WorldEntity", "WorldEvent", "WorldStateStore"]
