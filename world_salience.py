"""把世界增量分类成「值得叫醒 agent」还是「只记进上下文」。

这里只做纯函数判断，不碰 SDK、不发消息，因此可以被单元测试直接覆盖；
``__init__.py`` 的世界桥只负责接线和限速。

拆出来的原因是两个具体缺陷：

1. 原来的去重签名只包含 ``(id, state, label)``。但本地检测器把 ``state``
   恒定写成 ``visible``、``events`` 恒定为空，于是「有人径直朝我走来」这种
   全程 id/label/state 都不变的过程产生**零次**推送——距离、方位这些真正在
   变的量一个都不在签名里。
2. 原来所有推送都用 ``ai_behavior="read"``，宿主把它映射成 ``passive``，只
   装饰下一次由用户触发的回合。也就是说没有任何观测能让 agent 主动开口，
   「注意到有人挥手并回应」在架构上不可达。

两者的共同前提是：先能把「场面变了多少」量化出来。
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Iterable

# 表观高度分档。导航器用 ``apparent_height`` 做距离闭环（target 0.55），所以
# 这里的档位边界按「社交距离」而不是等距划分：远处的人挪动几个百分点不值得
# 打断，走进对话距离则必须知道。
_PROXIMITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.45, "very_close"),
    (0.28, "close"),
    (0.15, "mid"),
    (0.06, "far"),
)
_PROXIMITY_ORDER: tuple[str, ...] = ("unknown", "very_far", "far", "mid", "close", "very_close")

# 方位分档。八分方位足以表达「在我左前方」而不会因为几度抖动反复触发。
_BEARING_SECTORS: tuple[tuple[float, str], ...] = (
    (11.25, "front"),
    (33.75, "front_slight"),
    (67.5, "side"),
    (112.5, "wide"),
)

# 会主动叫醒 agent 的事件类型。检测器目前不产出事件，但 VLM 与将来的世界日志
# 会；先把语义定下来，免得接进来时又变成静默的 read。
_SOCIAL_EVENT_KINDS: frozenset[str] = frozenset({
    "wave",
    "waving",
    "greeting",
    "gesture",
    "pointing",
    "speaking",
    "talking",
    "chat",
    "player_joined",
    "player_left",
    "approach",
    "looking_at_me",
})


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def proximity_band(apparent_height: Any, *, clipped: bool = False) -> str:
    """把表观高度量化成有限档位。

    ``clipped=True`` 表示检测框贴到画面边缘，真实高度不可测、数值会饱和；
    这时不能假装知道远近，返回 ``unknown``——不伪造是这套架构的硬约束。
    """
    if clipped:
        return "unknown"
    height = _finite(apparent_height)
    if height is None or height <= 0.0:
        return "unknown"
    for threshold, name in _PROXIMITY_BANDS:
        if height >= threshold:
            return name
    return "very_far"


def bearing_sector(bearing_deg: Any) -> str:
    """把方位角量化成八分方位，带左右。"""
    bearing = _finite(bearing_deg)
    if bearing is None:
        return "unknown"
    magnitude = abs(bearing)
    for threshold, name in _BEARING_SECTORS:
        if magnitude < threshold:
            if name == "front":
                return "front"
            return f"{name}_{'right' if bearing > 0 else 'left'}"
    return f"behind_{'right' if bearing > 0 else 'left'}"


def _attributes(entity: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = entity.get("attributes")
    return attributes if isinstance(attributes, Mapping) else {}


def entity_digest(entity: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """一个实体在签名里的稳定摘要。

    含 id/label/state 是为了保留原有行为；额外的距离档和方位档才是让「有人
    走近」「有人绕到侧面」能被看见的部分。
    """
    attributes = _attributes(entity)
    return (
        str(entity.get("id")),
        str(entity.get("label")),
        str(entity.get("state")),
        proximity_band(
            attributes.get("apparent_height"),
            clipped=bool(attributes.get("apparent_height_clipped")),
        ),
        bearing_sector(attributes.get("bearing_deg")),
    )


def delta_signature(changes: Mapping[str, Any]) -> str:
    """世界增量的去重签名。

    与旧实现的差别只有一处：实体摘要现在包含量化后的距离和方位。量化是刻意
    的——直接用原始浮点会让每一帧都不同，等于取消去重。
    """
    entities = changes.get("entities") or ()
    events = changes.get("events") or ()
    removed = changes.get("removed_entity_ids") or ()
    return repr((
        tuple(sorted(
            entity_digest(item)
            for item in entities
            if isinstance(item, Mapping)
        )),
        tuple(sorted(
            (str(item.get("type") or item.get("kind")), str(item.get("target_id")))
            for item in events
            if isinstance(item, Mapping)
        )),
        tuple(sorted(str(item) for item in list(removed)[:32])),
    ))


def _person_like(entity: Mapping[str, Any]) -> bool:
    label = str(entity.get("label") or "").lower()
    return "person" in label or "player" in label or "avatar" in label


def _closer_than(current: str, previous: str) -> bool:
    """当前档位是否比之前更近。``unknown`` 不参与比较，避免把不确定当成靠近。"""
    if current == "unknown" or previous == "unknown":
        return False
    return _PROXIMITY_ORDER.index(current) > _PROXIMITY_ORDER.index(previous)


def classify(
    changes: Mapping[str, Any],
    previous: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """判断这次增量是否值得主动叫醒 agent。

    ``previous`` 是上一次的 ``{entity_id: (proximity_band, bearing_sector)}``，
    由调用方保存；没有历史时只能判断「出现了新的人」。

    返回 ``{"wake": bool, "reasons": [...], "entity_states": {...}}``。
    刻意保守：只有社交上确实需要反应的变化才置 ``wake``，其余继续走 ``read``
    进上下文。误报的代价是 agent 无缘无故开口，比漏报更难忍受。
    """
    history = dict(previous or {})
    reasons: list[str] = []
    entity_states: dict[str, tuple[str, str]] = {}

    for item in changes.get("entities") or ():
        if not isinstance(item, Mapping):
            continue
        entity_id = str(item.get("id") or "")
        if not entity_id:
            continue
        attributes = _attributes(item)
        band = proximity_band(
            attributes.get("apparent_height"),
            clipped=bool(attributes.get("apparent_height_clipped")),
        )
        sector = bearing_sector(attributes.get("bearing_deg"))
        entity_states[entity_id] = (band, sector)
        if not _person_like(item):
            continue
        label = str(item.get("label") or "unknown")[:48]
        if entity_id not in history:
            # 新出现的人。跟踪器 TTL 只有 1.5 s，短暂遮挡会让同一个人重新分配
            # ID；限定在「已经在对话距离内出现」可以压掉大部分这类抖动。
            if band in {"very_close", "close", "mid"}:
                reasons.append(f"{label} 出现在 {sector}（{band}）")
            continue
        previous_band, previous_sector = history[entity_id]
        if _closer_than(band, previous_band) and band in {"very_close", "close"}:
            reasons.append(f"{label} 正在靠近（{previous_band} → {band}，{sector}）")
        elif previous_sector != sector and band in {"very_close", "close"}:
            reasons.append(f"{label} 移动到 {sector}（{band}）")

    for item in changes.get("events") or ():
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or item.get("kind") or "").lower()
        if kind in _SOCIAL_EVENT_KINDS:
            reasons.append(f"事件：{kind}")

    for entity_id in list(changes.get("removed_entity_ids") or ())[:8]:
        previous_state = history.get(str(entity_id))
        # 只有在对话距离内消失才值得反应；远处的人走出画面是常态。
        if previous_state and previous_state[0] in {"very_close", "close"}:
            reasons.append("身边有人离开了视野")

    return {
        "wake": bool(reasons),
        "reasons": reasons[:6],
        "entity_states": entity_states,
    }


def describe_entities(entities: Iterable[Any], limit: int = 12) -> list[str]:
    """给世界文本用的实体描述，带上距离档和方位。

    原来的文本只有 ``label[state](confidence)``；agent 拿到它无法回答「谁离我
    最近」。这里补上的两个量都是量化档位，不是精确值——它们本来就只是视觉
    猜测，写成精确数字会诱导过度解读。
    """
    described: list[str] = []
    for item in list(entities)[:limit]:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or item.get("id") or "unknown")[:80]
        attributes = _attributes(item)
        band = proximity_band(
            attributes.get("apparent_height"),
            clipped=bool(attributes.get("apparent_height_clipped")),
        )
        sector = bearing_sector(attributes.get("bearing_deg"))
        confidence = item.get("confidence")
        state = str(item.get("state") or "")[:48]
        parts = [label]
        if state:
            parts.append(f"[{state}]")
        parts.append(f"({sector}/{band}, conf={confidence})")
        described.append("".join(parts))
    return described


__all__ = [
    "bearing_sector",
    "classify",
    "delta_signature",
    "describe_entities",
    "entity_digest",
    "proximity_band",
]
