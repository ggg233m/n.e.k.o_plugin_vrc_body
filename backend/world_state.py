"""线程安全、带不确定性标记的视觉世界状态。

身体调度器刻意不依赖本模块。视觉检测器和语义模型可以在此发布观测，同时
AnyaDance 继续独立运行 60 Hz 控制循环。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


VRCHAT_LOG_SOURCE = "vrchat_log"
VRCHAT_PLAYER_ID_PREFIX = "vrchat:player:"

# 这些来源发布的是逐帧视觉假设，不是可以跨后端进程复用的世界事实。显式的
# memory_scope 是新适配器首选；来源表与 ID 规则用于清理旧版本已经写盘的数据。
_TRANSIENT_VISION_SOURCES = frozenset({
    "openvino",
    "onnxruntime",
    "opencv_hog",
    "yolo",
    "local_detector",
})
_TRANSIENT_MEMORY_SCOPES = frozenset({
    "observation",
    "frame",
    "track",
    "session",
    "transient",
})


def _transient_entity_id(value: Any) -> bool:
    entity_id = _text(value, limit=96).lower()
    return (
        entity_id.startswith("avatar:session:")
        or entity_id.startswith("synthetic:")
        or ":track:" in entity_id
    )


def _transient_source(value: Any) -> bool:
    source = _text(value, limit=48).lower()
    return (
        source in _TRANSIENT_VISION_SOURCES
        or source.startswith("synthetic")
        or source.endswith("_test")
    )

# 这些不确定性描述的是感知能力的永久边界，而不是当前观测不可信。检测器正常
# 工作时它们会一直存在，所以不能让它们阻断移动：否则检测器越正常，导航越死。
# 白名单之外的一切（世界切换、观测过期、并发发送者等）仍然阻断——未知的新
# 编码默认按阻断处理，加检测器不会意外放松安全边界。
INFORMATIONAL_UNCERTAINTIES = frozenset({
    "depth_unavailable",
    "ocr_unavailable",
    "opencv_hog_person_only",
})


def blocking_uncertainties(values: Any) -> list[str]:
    """过滤掉仅供参考的能力边界标记，返回真正应当停止移动的不确定性。"""
    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in values:
        text = _text(item, limit=160)
        if text and text not in INFORMATIONAL_UNCERTAINTIES and text not in result:
            result.append(text)
    return result


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


def stable_entity_id(source: Any, label: Any, track_id: Any) -> str:
    """生成带类别的兼容实体 ID。

    这个旧接口保留给明确需要类别分段的调用方；如果类别可能抖动，
    请使用 :func:`stable_track_entity_id`，否则同一 track 会被拆成多个实体。
    """
    parts: list[str] = []
    for name, value, limit in (
        ("source", source, 48),
        ("label", label, 64),
        ("track_id", track_id, 96),
    ):
        text = _text(value, limit=limit)
        if not text:
            raise ValueError(f"stable entity id requires {name}")
        # 冒号是协议分隔符；替换掉输入中的冒号，避免产生歧义 ID。
        parts.append(text.replace(":", "_"))
    return ":".join(parts)


def stable_track_entity_id(source: Any, track_id: Any) -> str:
    """生成不受检测类别抖动影响的跨帧实体 ID。"""
    source_text = _text(source, limit=48)
    track_text = _text(track_id, limit=96)
    if not source_text:
        raise ValueError("stable track entity id requires source")
    if not track_text:
        raise ValueError("stable track entity id requires track_id")
    return f"{source_text.replace(':', '_')}:track:{track_text.replace(':', '_')}"


def vrchat_player_entity_id(user_id: Any) -> str:
    """生成世界日志与视觉适配器共用的 VRChat 玩家实体 ID。"""
    normalized = _text(user_id, limit=96)
    if not normalized:
        raise ValueError("VRChat player user_id must not be empty")
    return f"{VRCHAT_PLAYER_ID_PREFIX}{normalized.replace(':', '_')}"


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
        source = _sources(value.get("source"), default_source)
        label = _text(value.get("label"), default="unknown", limit=64) or "unknown"
        entity_id = _text(value.get("id"), limit=96)
        if not entity_id and value.get("track_id") is not None:
            # Preserve the original implicit ID format for existing callers and
            # precondition references. New detectors should provide an explicit
            # ID from stable_track_entity_id() to avoid label churn.
            entity_id = stable_entity_id(source[0], label, value.get("track_id"))
        if not entity_id:
            raise ValueError("world entity id or track_id must not be empty")
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
            label=label,
            confidence=_confidence(value.get("confidence")),
            bbox=_bbox(value.get("bbox")),
            state=_text(value.get("state"), default="unknown", limit=64) or "unknown",
            attributes=safe_attributes,
            relations=relations,
            source=source,
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
        max_removals: int = 256,
        lifecycle_watermark_limit: int = 4096,
        event_history_size: int = 64,
        default_ttl_s: float = 2.0,
        clock: Any = time.monotonic,
        persistence_path: str | Path | None = None,
        persist_world: bool = False,
        persist_players: bool = False,
    ) -> None:
        if (
            max_entities < 1
            or max_removals < 1
            or lifecycle_watermark_limit < 1
            or event_history_size < 1
        ):
            raise ValueError("world state bounds must be positive")
        self.max_entities = max_entities
        self.max_removals = min(4096, int(max_removals))
        # 至少保留一个完整的删除批次作为围栏；当调用方的世界适配器玩家进出窗口较长时，可选择更大的上限。
        self.lifecycle_watermark_limit = min(
            65536,
            max(self.max_removals, int(lifecycle_watermark_limit)),
        )
        self.event_history_size = event_history_size
        self.default_ttl_s = min(60.0, max(0.1, float(default_ttl_s)))
        self._clock = clock
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._persist_world = bool(persist_world and self._persistence_path)
        self._persist_players = bool(persist_players)
        # 缓存稳定序列化结果。视觉 worker 可能 10 Hz 调用 ingest；如果真正可持久
        # 化的事实没有变化，就不应每帧 replace 同一个文件。
        self._last_persistence_serialized: str | None = None
        self._persistence_write_count = 0
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._entities: dict[str, WorldEntity] = {}
        self._events: deque[WorldEvent] = deque(maxlen=event_history_size)
        self._backend: dict[str, dict[str, Any]] = {}
        self._last_observation_at: float | None = None
        self._uncertainties: list[str] = []
        self._observation_count = 0
        self._revision = 0
        self._last_changes: dict[str, Any] = {"removed_entity_ids": [], "removed_entity_count": 0}
        # 精确的生命周期删除和来源/世界重置都会留下水印；
        # 否则延迟的检测器帧可能会复活已离场的实体。
        self._delete_watermarks: dict[tuple[str, str], float] = {}
        self._delete_event_times: dict[tuple[str, str], float] = {}
        self._source_reset_watermarks: dict[tuple[str, str | None], float] = {}
        self._source_reset_event_times: dict[tuple[str, str | None], float] = {}
        # 按所有者索引的视图使高频实体检查无需扫描
        # 无关来源的重置围栏
        self._source_reset_index: dict[str, dict[str | None, float]] = {}
        if self._persist_world:
            self._load_persisted()

    def _persist_allowed_entity(self, entity: WorldEntity) -> bool:
        attributes = entity.attributes or {}
        memory_scope = _text(attributes.get("memory_scope"), limit=32).lower()
        identity_scope = _text(attributes.get("identity_scope"), limit=32).lower()
        if (
            _transient_entity_id(entity.id)
            or memory_scope in _TRANSIENT_MEMORY_SCOPES
            or identity_scope in {"track", "session"}
            or bool(attributes.get("track_entity_id"))
            or any(_transient_source(source) for source in entity.source)
        ):
            return False
        if self._persist_players:
            return True
        return not entity.id.startswith(VRCHAT_PLAYER_ID_PREFIX) and not any(
            source.startswith("vrchat_player") or source == VRCHAT_LOG_SOURCE
            for source in entity.source
        )

    def _persist_allowed_event(self, event: WorldEvent) -> bool:
        if _transient_entity_id(event.target_id) or any(
            _transient_source(source) for source in event.source
        ):
            return False
        if self._persist_players:
            return True
        kind = event.kind.lower()
        return "chat" not in kind and not kind.startswith("player_") and not (
            event.target_id or ""
        ).startswith(VRCHAT_PLAYER_ID_PREFIX)

    def _persist_locked(self) -> None:
        path = self._persistence_path
        if not self._persist_world or path is None:
            return
        try:
            now = self._clock()
            entities = []
            for item in self._entities.values():
                if not self._persist_allowed_entity(item):
                    continue
                serialized = item.to_dict(now=now)
                # age/visible 是当前进程单调时钟的派生值，既不能跨进程复用，也会
                # 让内容哈希每帧变化。加载时仍按既有策略给事实新的短期 TTL。
                for key in ("age_ms", "ttl_ms", "visible"):
                    serialized.pop(key, None)
                entities.append(serialized)
                if len(entities) >= self.max_entities:
                    break
            events = []
            for item in self._events:
                if not self._persist_allowed_event(item):
                    continue
                serialized_event = item.to_dict(now=now)
                serialized_event.pop("age_ms", None)
                events.append(serialized_event)
                if len(events) >= self.event_history_size:
                    break
            payload = {"version": 1, "entities": entities, "events": events}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if encoded == self._last_persistence_serialized:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
            self._last_persistence_serialized = encoded
            self._persistence_write_count += 1
        except Exception:
            # Persistence is advisory and must never interfere with control or
            # perception updates.
            return

    def _load_persisted(self) -> None:
        path = self._persistence_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            now = self._clock()
            for raw in (payload.get("entities") or ())[: self.max_entities]:
                if not isinstance(raw, Mapping):
                    continue
                entity = WorldEntity.from_mapping(raw, now=now, default_source="world_memory", default_ttl_s=max(30.0, self.default_ttl_s))
                if self._persist_allowed_entity(entity):
                    self._entities[entity.id] = entity
            for raw in (payload.get("events") or ())[: self.event_history_size]:
                if not isinstance(raw, Mapping):
                    continue
                event = WorldEvent.from_mapping(raw, now=now, default_source="world_memory")
                if self._persist_allowed_event(event):
                    self._events.append(event)
            # 旧版本可能包含视觉轨迹、会话 ID 和每帧变化的 age_ms。加载后立刻按
            # 新规则重写一次，不必等下一帧才能清除磁盘上的脏数据。
            self._persist_locked()
        except Exception:
            return

    def set_backend_status(self, name: str, status: Mapping[str, Any]) -> None:
        backend_name = _text(name, default="unknown", limit=48) or "unknown"
        with self._lock:
            self._backend[backend_name] = {
                str(key)[:64]: _safe_value(item) for key, item in list(status.items())[:24]
            }

    def _normalize_remove_ids(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            # 保持旧版进程内便捷 API 的兼容性；HTTP 和 VisionRuntime 适配器
            # 在到达此兼容路径之前会拒绝标量 JSON 值。
            value = (value,)
        elif isinstance(value, bytes):
            raise ValueError("remove_entity_ids must be an array")
        try:
            iterator = iter(value)
        except TypeError as exc:
            raise ValueError("remove_entity_ids must be an array") from exc
        result: list[str] = []
        for item in iterator:
            if not isinstance(item, str):
                raise ValueError("remove_entity_ids must contain strings")
            entity_id = _text(item, limit=96)
            if not entity_id:
                raise ValueError("remove_entity_ids must not contain empty IDs")
            if entity_id not in result:
                if len(result) >= self.max_removals:
                    raise ValueError(
                        f"remove_entity_ids must contain at most {self.max_removals} items"
                    )
                result.append(entity_id)
        return result

    @staticmethod
    def _normalize_remove_source(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("remove_source must be a non-empty string")
        source = value.strip()
        if not source or len(source) > 48:
            raise ValueError("remove_source must be a non-empty string")
        return source

    def _watermark_blocks(self, entity: WorldEntity) -> bool:
        owner = entity.source[0] if entity.source else ""
        if not owner:
            return False
        exact = self._delete_watermarks.get((owner, entity.id), -math.inf)
        for prefix, watermark in self._source_reset_index.get(owner, {}).items():
            if prefix is None or entity.id.startswith(prefix):
                exact = max(exact, watermark)
        return entity.observed_at <= exact

    def _clear_superseded_watermarks(self, entity: WorldEntity) -> None:
        owner = entity.source[0] if entity.source else ""
        if not owner:
            return
        key = (owner, entity.id)
        watermark = self._delete_watermarks.get(key)
        if watermark is not None and entity.observed_at > watermark:
            self._delete_watermarks.pop(key, None)
            self._delete_event_times.pop(key, None)

    def _prune_lifecycle_watermarks(self) -> None:
        """Bound tombstone memory while retaining the newest receive fences."""
        limit = self.lifecycle_watermark_limit
        if len(self._delete_watermarks) > limit:
            newest = sorted(
                self._delete_watermarks.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
            self._delete_watermarks = dict(newest)
            self._delete_event_times = {
                key: self._delete_event_times[key]
                for key, _ in newest
                if key in self._delete_event_times
            }
        if len(self._source_reset_watermarks) > limit:
            newest_resets = sorted(
                self._source_reset_watermarks.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
            self._source_reset_watermarks = dict(newest_resets)
            self._source_reset_event_times = {
                key: self._source_reset_event_times[key]
                for key, _ in newest_resets
                if key in self._source_reset_event_times
            }
        self._source_reset_index = {}
        for (source, prefix), watermark in self._source_reset_watermarks.items():
            source_index = self._source_reset_index.setdefault(source, {})
            source_index[prefix] = max(source_index.get(prefix, -math.inf), watermark)

    def _drop_events_for_source_reset(self, source: str, prefix: str | None) -> None:
        """Remove stale source events when a world/source reset is committed."""
        retained: list[WorldEvent] = []
        for event in self._events:
            owner = event.source[0] if event.source else ""
            if owner != source:
                retained.append(event)
                continue
            if prefix is not None and (
                not event.target_id or not event.target_id.startswith(prefix)
            ):
                retained.append(event)
        self._events = deque(retained, maxlen=self.event_history_size)

    def ingest(
        self,
        entities: Iterable[WorldEntity | Mapping[str, Any]] = (),
        events: Iterable[WorldEvent | Mapping[str, Any]] = (),
        *,
        source: str = "vision",
        observed_at: float | None = None,
        remove_entity_ids: Iterable[Any] = (),
        remove_source: str | None = None,
        uncertainties: Iterable[Any] = (),
    ) -> dict[str, Any]:
        """写入观测，并可在同一批次撤销明确声明的实体生命周期。

        ``remove_entity_ids``/``remove_source`` 只用于发布者明确知道实体已
        离场的情况（例如玩家退出）；``remove_source`` 仅过滤这些明确 ID，
        不会在未提供 ID 时清空整个来源。普通检测漏帧不应使用它们。
        """
        received_at = self._clock()
        now = (
            received_at
            if observed_at is None
            else min(received_at, _finite(observed_at, received_at))
        )
        normalized_entities: list[WorldEntity] = []
        for item in entities:
            try:
                raw_entity: Mapping[str, Any] = (
                    {
                        "id": item.id,
                        "label": item.label,
                        "confidence": item.confidence,
                        "bbox": item.bbox,
                        "state": item.state,
                        "attributes": item.attributes,
                        "relations": item.relations,
                        "source": item.source,
                        "observed_at": item.observed_at,
                        "ttl_s": item.ttl_s,
                    }
                    if isinstance(item, WorldEntity)
                    else item
                )
                entity = WorldEntity.from_mapping(
                    raw_entity,
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
                raw_event: Mapping[str, Any] = (
                    {
                        "type": item.kind,
                        "target_id": item.target_id,
                        "confidence": item.confidence,
                        "data": item.data,
                        "source": item.source,
                        "observed_at": item.observed_at,
                    }
                    if isinstance(item, WorldEvent)
                    else item
                )
                event = WorldEvent.from_mapping(
                    raw_event,
                    now=now,
                    default_source=source,
                )
            except (TypeError, ValueError):
                continue
            normalized_events.append(event)
        normalized_remove_ids = self._normalize_remove_ids(remove_entity_ids)
        normalized_uncertainties = [
            _text(item, limit=160) for item in (uncertainties or ())
            if _text(item, limit=160)
        ][:16]
        remove_source_name: str | None = None
        if remove_source is not None:
            remove_source_name = self._normalize_remove_source(remove_source)
        if remove_source_name is not None and not normalized_remove_ids:
            raise ValueError(
                "remove_source requires remove_entity_ids; "
                "use remove_entities_by_source for bulk cleanup"
            )
        if normalized_remove_ids and remove_source_name is None:
            raise ValueError("remove_source is required when removing entity IDs")
        for event in normalized_events:
            if event.kind != "player_left":
                continue
            if not event.target_id or event.target_id not in normalized_remove_ids:
                raise ValueError(
                    "player_left target_id must be included in remove_entity_ids"
                )
            if remove_source_name is None or not event.source or event.source[0] != remove_source_name:
                raise ValueError("player_left source must match remove_source")
            if remove_source_name == VRCHAT_LOG_SOURCE and not event.target_id.startswith(
                VRCHAT_PLAYER_ID_PREFIX
            ):
                raise ValueError("VRChat player_left target_id has an invalid namespace")
        removed_entity_ids: list[str] = []
        with self._lock:
            # 先建立删除水位，再处理 upsert；同一时间戳的离场优先，
            # 迟到的旧检测帧不会把玩家复活。
            for entity_id in normalized_remove_ids:
                if remove_source_name is None:
                    continue
                entity = self._entities.get(entity_id)
                # A source-scoped lifecycle command must not create a tombstone
                # for an entity owned by another source. Otherwise a shared ID
                # could be blocked by an unrelated publisher.
                if entity is not None and (
                    not entity.source or entity.source[0] != remove_source_name
                ):
                    continue
                delete_key = (remove_source_name, entity_id)
                previous_watermark = self._delete_watermarks.get(delete_key, -math.inf)
                self._delete_watermarks[delete_key] = max(previous_watermark, received_at)
                previous_event_time = self._delete_event_times.get(delete_key, -math.inf)
                self._delete_event_times[delete_key] = max(previous_event_time, now)
                if entity is None:
                    continue
                if entity.observed_at > self._delete_event_times[delete_key]:
                    continue
                del self._entities[entity_id]
                removed_entity_ids.append(entity_id)
            for entity in normalized_entities:
                if self._watermark_blocks(entity):
                    continue
                previous = self._entities.get(entity.id)
                if previous is None or entity.observed_at >= previous.observed_at:
                    self._entities[entity.id] = entity
                    self._clear_superseded_watermarks(entity)
            if len(self._entities) > self.max_entities:
                ranked = sorted(
                    self._entities.values(),
                    key=lambda item: (item.observed_at, item.confidence),
                    reverse=True,
                )[: self.max_entities]
                self._entities = {item.id: item for item in ranked}
            self._events.extend(normalized_events)
            self._prune_lifecycle_watermarks()
            self._last_observation_at = max(self._last_observation_at or now, now)
            self._uncertainties = normalized_uncertainties
            self._observation_count += 1
            self._revision += 1
            snapshot = self.snapshot(now=received_at)
            if normalized_remove_ids:
                snapshot["changes"] = {
                    "removed_entity_ids": removed_entity_ids,
                    "removed_entity_count": len(removed_entity_ids),
                }
            self._last_changes = dict(snapshot.get("changes") or {
                "removed_entity_ids": [],
                "removed_entity_count": 0,
            })
            self._persist_locked()
            self._changed.notify_all()
        return snapshot

    def remove_entities(
        self,
        entity_ids: Iterable[Any] = (),
        *,
        source: str | None = None,
        events: Iterable[WorldEvent | Mapping[str, Any]] = (),
        observed_at: float | None = None,
        event_source: str | None = None,
    ) -> dict[str, Any]:
        """原子删除实体并（可选）追加事件，返回删除后的快照。

        ``source`` 是保护性过滤器；事件和删除会走与普通 ingest 相同的
        规范化及时间戳钳制路径，避免维护两套生命周期逻辑。
        """
        event_source_name = event_source or source or "world_state"
        return self.ingest(
            events=events,
            source=event_source_name,
            observed_at=observed_at,
            remove_entity_ids=entity_ids,
            remove_source=source,
        )

    def remove_entity(
        self,
        entity_id: Any,
        *,
        source: str | None = None,
    ) -> bool:
        """删除一个实体，返回是否实际删除。"""
        normalized_id = _text(entity_id, limit=96)
        if not normalized_id:
            return False
        # 先在同一把 RLock 下确认目标存在；RLock 允许随后复用统一的
        # ingest 事务，并在调用者明确给出 owner 时为“先离场、后首帧”的
        # 情况留下 tombstone。
        with self._lock:
            entity = self._entities.get(normalized_id)
            if entity is None:
                # 知晓所属来源的调用方可能会在首个检测帧到达之前就发布真正的离场事件。
                # 此时应保留一个墓碑（tombstone），防止迟到的旧帧复活已离场的实体 ID。
                # 若没有所属来源，则无法确定安全的作用域。
                if source is not None:
                    source_name = self._normalize_remove_source(source)
                    self.remove_entities([normalized_id], source=source_name)
                return False
            if source is not None:
                source_name = self._normalize_remove_source(source)
                if not entity.source or entity.source[0] != source_name:
                    return False
            elif len(entity.source) == 1:
                source_name = entity.source[0]
            else:
                raise ValueError("source is required for a multi-source entity")
            result = self.remove_entities([normalized_id], source=source_name)
            return bool((result.get("changes") or {}).get("removed_entity_count"))

    def remove_entities_by_source(
        self,
        source: str,
        *,
        prefix: str | None = None,
        observed_at: float | None = None,
        events: Iterable[WorldEvent | Mapping[str, Any]] = (),
        event_source: str | None = None,
    ) -> int:
        """按来源清理实体并可在同一事务追加事件，返回实际删除数量。

        ``prefix`` 可用于只清理来源下的一类实体，例如
        ``vrchat:player:``。清理会记录 source reset watermark，阻止队列中的
        旧帧重新插入已清理实体。
        """
        source_name = self._normalize_remove_source(source)
        prefix_value = None if prefix is None else _text(prefix, limit=96)
        if prefix is not None and not prefix_value:
            raise ValueError("prefix must not be empty")
        received_at = self._clock()
        reset_at = (
            received_at
            if observed_at is None
            else min(received_at, _finite(observed_at, received_at))
        )
        normalized_events: list[WorldEvent] = []
        default_event_source = (
            self._normalize_remove_source(event_source)
            if event_source is not None
            else source_name
        )
        for item in events:
            if isinstance(item, WorldEvent):
                raw_event: Mapping[str, Any] = {
                    "type": item.kind,
                    "target_id": item.target_id,
                    "confidence": item.confidence,
                    "data": item.data,
                    "source": item.source,
                    "observed_at": item.observed_at,
                }
            elif isinstance(item, Mapping):
                raw_event = item
            else:
                raise ValueError("events must contain objects")
            normalized_events.append(
                WorldEvent.from_mapping(raw_event, now=reset_at, default_source=default_event_source)
            )
        if any(event.kind == "player_left" for event in normalized_events):
            raise ValueError(
                "remove_entities_by_source cannot publish player_left; "
                "use remove_entities with explicit remove_entity_ids"
            )
        with self._lock:
            reset_key = (source_name, prefix_value)
            self._source_reset_watermarks[reset_key] = max(
                self._source_reset_watermarks.get(reset_key, -math.inf), received_at
            )
            self._source_reset_index.setdefault(source_name, {})[prefix_value] = max(
                self._source_reset_index.get(source_name, {}).get(prefix_value, -math.inf),
                received_at,
            )
            self._source_reset_event_times[reset_key] = max(
                self._source_reset_event_times.get(reset_key, -math.inf), reset_at
            )
            removed = [
                entity_id
                for entity_id, entity in self._entities.items()
                if entity.source
                and entity.source[0] == source_name
                and (prefix_value is None or entity_id.startswith(prefix_value))
                and entity.observed_at <= reset_at
            ]
            for entity_id in removed:
                self._entities.pop(entity_id, None)
                self._delete_watermarks[(source_name, entity_id)] = max(
                    self._delete_watermarks.get((source_name, entity_id), -math.inf),
                    received_at,
                )
                self._delete_event_times[(source_name, entity_id)] = max(
                    self._delete_event_times.get((source_name, entity_id), -math.inf),
                    reset_at,
                )
            self._drop_events_for_source_reset(source_name, prefix_value)
            self._events.extend(normalized_events)
            self._prune_lifecycle_watermarks()
            self._last_observation_at = max(self._last_observation_at or reset_at, reset_at)
            self._observation_count += 1
            self._revision += 1
            self._last_changes = {
                "removed_entity_ids": list(removed),
                "removed_entity_count": len(removed),
            }
            self._persist_locked()
            self._changed.notify_all()
            return len(removed)

    def clear(self) -> None:
        with self._lock:
            self._entities.clear()
            self._events.clear()
            self._delete_watermarks.clear()
            self._delete_event_times.clear()
            self._source_reset_watermarks.clear()
            self._source_reset_event_times.clear()
            self._source_reset_index.clear()
            self._last_observation_at = None
            self._uncertainties = []
            self._observation_count = 0
            self._revision += 1
            self._last_changes = {"removed_entity_ids": [], "removed_entity_count": 0}
            self._persist_locked()
            self._changed.notify_all()

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
                "uncertainties": list(self._uncertainties) if self._uncertainties else ([] if visible else ["no_recent_visual_observation"]),
                "status": {
                    "entity_count": len(visible),
                    "event_count": len(recent_events),
                    "last_observation_age_ms": age_ms,
                    "observation_count": self._observation_count,
                    "revision": self._revision,
                    "lifecycle_watermark_count": (
                        len(self._delete_watermarks) + len(self._source_reset_watermarks)
                    ),
                },
                "backends": {key: dict(value) for key, value in self._backend.items()},
                "memory": {
                    "persist_world": self._persist_world,
                    "persist_players": self._persist_players,
                    "transient_entities_persisted": False,
                    "persistence_write_count": self._persistence_write_count,
                    "raw_frames_persisted": False,
                    "chat_persisted": False,
                },
            }

    def delta(
        self,
        after_revision: int = 0,
        *,
        wait_ms: int = 250,
        limit: int = 16,
    ) -> dict[str, Any]:
        """Wait for a newer revision and return a bounded world delta.

        The store does not retain raw frames or player history.  When a caller
        falls behind, the latest bounded snapshot is returned with
        ``coalesced=true``; callers use the revision cursor to avoid replaying
        it.  Waiting happens on a condition variable and never touches the
        120 Hz body scheduler.
        """
        try:
            cursor = max(0, int(after_revision))
        except (TypeError, ValueError, OverflowError):
            cursor = 0
        try:
            timeout_s = min(2.0, max(0.0, float(wait_ms) / 1000.0))
        except (TypeError, ValueError, OverflowError):
            timeout_s = 0.25
        try:
            item_limit = min(64, max(1, int(limit)))
        except (TypeError, ValueError, OverflowError):
            item_limit = 16
        deadline = self._clock() + timeout_s
        with self._changed:
            while self._revision <= cursor:
                remaining = deadline - self._clock()
                if remaining <= 0.0:
                    break
                self._changed.wait(timeout=remaining)
            snapshot = self.snapshot()
            revision = int((snapshot.get("status") or {}).get("revision", self._revision))
            change_payload = dict(self._last_changes)
            return {
                "revision": revision,
                "after_revision": cursor,
                "changed": revision > cursor,
                "coalesced": revision > cursor + 1,
                "world": snapshot,
                "navigation": {
                    "status": "unknown",
                    "safe_navigation": (
                        bool(snapshot.get("available"))
                        and not blocking_uncertainties(snapshot.get("uncertainties"))
                    ),
                },
                "social": {
                    "status": "unknown",
                    "players_persisted": False,
                    "chat_persisted": False,
                },
                "uncertainty": list(snapshot.get("uncertainties") or ()),
                "changes": {
                    "entities": list(snapshot.get("entities") or [])[:item_limit],
                    "events": list(snapshot.get("events") or [])[:item_limit],
                    "removed_entity_ids": list(change_payload.get("removed_entity_ids") or ())[: self.max_removals],
                    "removed_entity_count": int(change_payload.get("removed_entity_count", 0) or 0),
                },
            }


__all__ = [
    "INFORMATIONAL_UNCERTAINTIES",
    "VRCHAT_LOG_SOURCE",
    "VRCHAT_PLAYER_ID_PREFIX",
    "WorldEntity",
    "WorldEvent",
    "WorldStateStore",
    "blocking_uncertainties",
    "stable_entity_id",
    "stable_track_entity_id",
    "vrchat_player_entity_id",
]
