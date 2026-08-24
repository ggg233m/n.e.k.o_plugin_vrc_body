"""语义搜索目标的有界本地状态机。

Explorer 不调用 LLM，也不解释自由文本。慢模型只需给出 selector 和约束；本地
状态机随后在新鲜观测上执行「扫描—短前进—再扫描」，找到候选后只负责把它保持
在视野中央，不会把 explore 偷换成 approach。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Mapping


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _attributes(entity: Mapping[str, Any]) -> Mapping[str, Any]:
    value = entity.get("attributes")
    return value if isinstance(value, Mapping) else {}


def _semantic_type(entity: Mapping[str, Any]) -> str:
    attributes = _attributes(entity)
    return str(
        attributes.get("semantic_type")
        or entity.get("semantic_type")
        or entity.get("label")
        or "unknown"
    ).replace("\x00", "").strip().lower()[:32]


def _bearing(entity: Mapping[str, Any]) -> float | None:
    attributes = _attributes(entity)
    for key in ("bearing_deg", "bearing", "angle_deg"):
        value = _finite(attributes.get(key))
        if value is not None:
            return min(180.0, max(-180.0, value))
    bbox = entity.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        left = _finite(bbox[0])
        right = _finite(bbox[2])
        if left is not None and right is not None:
            return min(180.0, max(-180.0, (((left + right) * 0.5) - 0.5) * 90.0))
    return None


@dataclass(frozen=True)
class ExploreDirective:
    state: str
    reason: str
    target_id: str | None = None
    bearing_deg: float | None = None
    turn_deg: float = 0.0
    forward_axis: float = 0.0


class ExplorerStateMachine:
    """只在内存中保存一次 explore 目标的有限搜索进度。"""

    _TYPE_MATCHES = {
        "npc": frozenset({"npc"}),
        "player": frozenset({"player", "avatar", "person", "humanoid"}),
        "avatar": frozenset({"avatar", "player", "person", "humanoid"}),
        "person": frozenset({"npc", "player", "avatar", "person", "humanoid"}),
        "humanoid": frozenset({"npc", "player", "avatar", "person", "humanoid"}),
        "object": frozenset({"object"}),
    }

    def __init__(
        self,
        *,
        scan_turn_deg: float = 45.0,
        turns_per_scan: int = 8,
        advance_ticks: int = 8,
        default_max_scan_turns: int = 16,
        default_max_duration_s: float = 90.0,
        default_forward_axis: float = 0.45,
    ) -> None:
        self.scan_turn_deg = min(90.0, max(10.0, float(scan_turn_deg)))
        self.turns_per_scan = min(16, max(4, int(turns_per_scan)))
        self.advance_ticks = min(40, max(1, int(advance_ticks)))
        self.default_max_scan_turns = min(32, max(1, int(default_max_scan_turns)))
        self.default_max_duration_s = min(600.0, max(1.0, float(default_max_duration_s)))
        self.default_forward_axis = min(0.6, max(0.05, float(default_forward_axis)))
        self._lock = threading.RLock()
        self._goal_key: tuple[Any, ...] | None = None
        self._goal_age_s: float | None = None
        self._phase = "idle"
        self._scan_turns = 0
        self._cycle_turns = 0
        self._advance_steps = 0
        self._locked_target_id: str | None = None
        self._last_reason = "not_started"

    @staticmethod
    def _goal_signature(goal: Mapping[str, Any]) -> tuple[Any, ...]:
        selector = goal.get("selector") if isinstance(goal.get("selector"), Mapping) else {}
        constraints = goal.get("constraints") if isinstance(goal.get("constraints"), Mapping) else {}
        return (
            str(goal.get("kind") or ""),
            str(goal.get("text") or ""),
            tuple(sorted((str(key), repr(value)) for key, value in selector.items())),
            tuple(sorted((str(key), repr(value)) for key, value in constraints.items())),
            goal.get("based_on_revision"),
        )

    def reset(self, reason: str = "reset") -> None:
        with self._lock:
            self._goal_key = None
            self._goal_age_s = None
            self._phase = "idle"
            self._scan_turns = 0
            self._cycle_turns = 0
            self._advance_steps = 0
            self._locked_target_id = None
            self._last_reason = str(reason)[:96]

    def _refresh_goal_locked(self, goal: Mapping[str, Any]) -> None:
        signature = self._goal_signature(goal)
        age_s = _finite(goal.get("age_seconds"))
        renewed = signature != self._goal_key or (
            age_s is not None and self._goal_age_s is not None and age_s < self._goal_age_s
        )
        if renewed:
            self._goal_key = signature
            self._phase = "scan"
            self._scan_turns = 0
            self._cycle_turns = 0
            self._advance_steps = 0
            self._locked_target_id = None
            self._last_reason = "goal_started"
        self._goal_age_s = age_s

    @classmethod
    def _matches(cls, entity: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
        if entity.get("visible") is False:
            return False
        confidence = _finite(entity.get("confidence")) or 0.0
        minimum = _finite(selector.get("min_confidence"))
        if confidence < (0.4 if minimum is None else minimum):
            return False
        expected_type = str(selector.get("semantic_type") or "").strip().lower()
        actual_type = _semantic_type(entity)
        if expected_type:
            if actual_type not in cls._TYPE_MATCHES.get(expected_type, frozenset({expected_type})):
                return False
            # npc/object 不能只凭通用 person 框猜；必须等语义 worker 确认。
            if expected_type in {"npc", "object"} and not bool(
                _attributes(entity).get("semantic_verified")
            ):
                return False
        expected_label = str(selector.get("label") or "").strip().casefold()
        if expected_label:
            actual_label = str(entity.get("label") or "").casefold()
            if expected_label not in actual_label:
                return False
        return True

    def _select_candidate(
        self,
        entities: Any,
        selector: Mapping[str, Any],
        skip_ids: set[str],
    ) -> Mapping[str, Any] | None:
        if not isinstance(entities, (list, tuple)):
            return None
        candidates: list[tuple[float, Mapping[str, Any]]] = []
        for raw in entities:
            if not isinstance(raw, Mapping) or not self._matches(raw, selector):
                continue
            entity_id = str(raw.get("id") or "")[:96]
            if not entity_id or entity_id in skip_ids:
                continue
            confidence = _finite(raw.get("confidence")) or 0.0
            bearing = abs(_bearing(raw) or 0.0)
            apparent = _finite(_attributes(raw).get("apparent_height")) or 0.0
            # 确定性优先，其次选择更大、更靠近画面中央的候选。
            score = confidence + min(0.2, apparent * 0.2) - min(0.2, bearing / 900.0)
            candidates.append((score, raw))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def decide(
        self,
        goal: Mapping[str, Any],
        world: Mapping[str, Any],
        *,
        max_turn_deg: float,
        turn_gain: float,
        bearing_deadband_deg: float,
        navigator_max_forward_axis: float,
        skip_ids: set[str] | None = None,
    ) -> ExploreDirective:
        selector = goal.get("selector") if isinstance(goal.get("selector"), Mapping) else None
        if not selector:
            return ExploreDirective("stop", "explore_selector_required")
        constraints = goal.get("constraints") if isinstance(goal.get("constraints"), Mapping) else {}
        max_duration_s = _finite(constraints.get("max_duration_s")) or self.default_max_duration_s
        max_scan_turns = int(_finite(constraints.get("max_scan_turns")) or self.default_max_scan_turns)
        max_forward_axis = _finite(constraints.get("max_forward_axis"))
        forward_axis = min(
            navigator_max_forward_axis,
            self.default_forward_axis if max_forward_axis is None else max_forward_axis,
        )
        age_s = _finite(goal.get("age_seconds")) or 0.0

        with self._lock:
            self._refresh_goal_locked(goal)
            if age_s > max_duration_s:
                self._phase = "complete"
                self._last_reason = "explore_duration_exhausted"
                return ExploreDirective("stop", self._last_reason, self._locked_target_id)

            entities = world.get("entities")
            selected: Mapping[str, Any] | None = None
            if self._locked_target_id:
                if isinstance(entities, (list, tuple)):
                    selected = next((
                        raw for raw in entities
                        if isinstance(raw, Mapping)
                        and str(raw.get("id") or "") == self._locked_target_id
                        and raw.get("visible") is not False
                    ), None)
            else:
                selected = self._select_candidate(entities, selector, skip_ids or set())
                if selected is not None:
                    self._locked_target_id = str(selected.get("id") or "")[:96] or None
                    self._phase = "locked"

            if selected is not None and self._locked_target_id:
                bearing = _bearing(selected)
                if bearing is None:
                    self._last_reason = "explore_target_bearing_unknown"
                    return ExploreDirective("stop", self._last_reason, self._locked_target_id)
                if abs(bearing) > bearing_deadband_deg:
                    turn = min(max_turn_deg, max(-max_turn_deg, -bearing * turn_gain))
                    self._last_reason = "explore_target_off_center"
                    return ExploreDirective(
                        "turn",
                        self._last_reason,
                        self._locked_target_id,
                        bearing,
                        turn,
                    )
                self._last_reason = "explore_target_found"
                return ExploreDirective(
                    "found",
                    self._last_reason,
                    self._locked_target_id,
                    bearing,
                )

            if self._scan_turns >= max_scan_turns:
                self._phase = "complete"
                self._last_reason = (
                    "explore_target_lost" if self._locked_target_id else "explore_search_exhausted"
                )
                return ExploreDirective("stop", self._last_reason, self._locked_target_id)

            if self._phase == "advance" and self._locked_target_id is None:
                self._last_reason = "explore_advance"
                return ExploreDirective("advance", self._last_reason, forward_axis=forward_axis)

            self._phase = "scan"
            self._last_reason = (
                "explore_reacquire_scan" if self._locked_target_id else "explore_scan"
            )
            return ExploreDirective(
                "turn",
                self._last_reason,
                self._locked_target_id,
                turn_deg=self.scan_turn_deg,
            )

    def record_applied(self, reason: str, succeeded: bool) -> None:
        """只在控制命令真正提交后消耗搜索预算。"""
        if not succeeded:
            return
        with self._lock:
            if reason in {"explore_scan", "explore_reacquire_scan"}:
                self._scan_turns += 1
                self._cycle_turns += 1
                if self._cycle_turns >= self.turns_per_scan and self._locked_target_id is None:
                    self._phase = "advance"
                    self._advance_steps = 0
            elif reason == "explore_advance":
                self._advance_steps += 1
                if self._advance_steps >= self.advance_ticks:
                    self._phase = "scan"
                    self._cycle_turns = 0
                    self._advance_steps = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._phase,
                "last_reason": self._last_reason,
                "scan_turns": self._scan_turns,
                "cycle_turns": self._cycle_turns,
                "advance_steps": self._advance_steps,
                "locked_target_id": self._locked_target_id,
                "policy": "scan_advance_scan_then_center",
                "llm_calls_in_loop": 0,
            }


__all__ = ["ExploreDirective", "ExplorerStateMachine"]
