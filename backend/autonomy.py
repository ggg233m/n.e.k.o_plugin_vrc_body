"""会话级 VRChat 自主操作护栏。

本模块刻意仅负责授权与时效检查，不涉及感知或高频运动控制。规划器可依据此状态决定执行何种操作，而身体调度器仍是唯一负责写入 AnyaDance 动画帧的组件。
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping

from .world_state import blocking_uncertainties


ALLOWED_GOAL_KINDS = frozenset({
    "explore", "approach", "approach_observe", "follow", "interact", "socialize",
})
TARGETED_GOAL_KINDS = frozenset({
    "approach", "approach_observe", "follow", "interact", "socialize",
})
ALLOWED_SELECTOR_TYPES = frozenset({"npc", "player", "avatar", "person", "humanoid", "object"})
_SELECTOR_KEYS = frozenset({"semantic_type", "label", "min_confidence"})
_CONSTRAINT_KEYS = frozenset({
    "max_duration_s", "max_scan_turns", "max_forward_axis",
    "settle_seconds", "observe_seconds",
})


def _finite_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return parsed


def _normalize_selector(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("selector must be an object")
    unknown = set(value) - _SELECTOR_KEYS
    if unknown:
        raise ValueError(f"selector contains unsupported fields: {', '.join(sorted(map(str, unknown)))}")
    result: dict[str, Any] = {}
    semantic_type = str(value.get("semantic_type") or "").replace("\x00", "").strip().lower()
    if semantic_type:
        if semantic_type not in ALLOWED_SELECTOR_TYPES:
            raise ValueError("selector.semantic_type is unsupported")
        result["semantic_type"] = semantic_type
    label = str(value.get("label") or "").replace("\x00", "").strip()[:64]
    if label:
        result["label"] = label
    if value.get("min_confidence") is not None:
        result["min_confidence"] = _finite_number(
            value.get("min_confidence"),
            "selector.min_confidence",
            0.0,
            1.0,
        )
    if not result:
        raise ValueError("selector must contain semantic_type or label")
    return result


def _normalize_constraints(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("constraints must be an object")
    unknown = set(value) - _CONSTRAINT_KEYS
    if unknown:
        raise ValueError(f"constraints contains unsupported fields: {', '.join(sorted(map(str, unknown)))}")
    result: dict[str, Any] = {}
    if value.get("max_duration_s") is not None:
        result["max_duration_s"] = _finite_number(
            value.get("max_duration_s"),
            "constraints.max_duration_s",
            1.0,
            600.0,
        )
    if value.get("max_scan_turns") is not None:
        raw_turns = value.get("max_scan_turns")
        if isinstance(raw_turns, bool):
            raise ValueError("constraints.max_scan_turns must be an integer")
        try:
            turns = int(raw_turns)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("constraints.max_scan_turns must be an integer") from exc
        if turns != raw_turns or not 1 <= turns <= 32:
            raise ValueError("constraints.max_scan_turns must be between 1 and 32")
        result["max_scan_turns"] = turns
    if value.get("max_forward_axis") is not None:
        result["max_forward_axis"] = _finite_number(
            value.get("max_forward_axis"),
            "constraints.max_forward_axis",
            0.05,
            1.0,
        )
    if value.get("settle_seconds") is not None:
        result["settle_seconds"] = _finite_number(
            value.get("settle_seconds"),
            "constraints.settle_seconds",
            0.2,
            3.0,
        )
    if value.get("observe_seconds") is not None:
        result["observe_seconds"] = _finite_number(
            value.get("observe_seconds"),
            "constraints.observe_seconds",
            0.5,
            10.0,
        )
    return result or None


def _normalize_revision(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("based_on_revision must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("based_on_revision must be a non-negative integer") from exc
    if parsed != value or parsed < 0:
        raise ValueError("based_on_revision must be a non-negative integer")
    return parsed


@dataclass
class AutonomyGoal:
    kind: str
    text: str
    target_id: str | None
    selector: dict[str, Any] | None
    constraints: dict[str, Any] | None
    based_on_revision: int | None
    created_at: float


class AutonomyRuntime:
    """Manual-arm, expiring authorization state for public-instance actions."""

    def __init__(
        self,
        *,
        world_provider: Callable[[], Mapping[str, Any]],
        release_inputs: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
        session_ttl_s: float = 1800.0,
    ) -> None:
        self._world_provider = world_provider
        self._release_inputs = release_inputs
        self._clock = clock
        self._session_ttl_s = min(3600.0, max(60.0, float(session_ttl_s)))
        self._lock = threading.RLock()
        self._state = "disarmed"
        self._reason = "manual_arm_required"
        self._armed_until: float | None = None
        self._goal: AutonomyGoal | None = None
        self._world_revision = 0

    def _disarm_locked(self, reason: str) -> None:
        self._state = "disarmed"
        self._reason = str(reason)[:160]
        self._armed_until = None
        self._goal = None

    def arm(self, *, ttl_s: float | None = None) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            ttl = self._session_ttl_s if ttl_s is None else min(self._session_ttl_s, max(60.0, float(ttl_s)))
            self._state = "armed"
            self._reason = "manual_session_arm"
            self._armed_until = now + ttl
            self._goal = None
            return self.snapshot()

    def disarm(self, reason: str = "manual_disarm") -> dict[str, Any]:
        with self._lock:
            self._disarm_locked(reason)
        self._release_inputs()
        return self.snapshot()

    def stop(self, reason: str = "autonomy_stop") -> dict[str, Any]:
        with self._lock:
            self._goal = None
            if self._state in {"armed", "degraded", "stopping"}:
                self._state = "stopping"
                self._reason = str(reason)[:160]
        self._release_inputs()
        with self._lock:
            if self._state == "stopping":
                self._disarm_locked(str(reason)[:160])
        return self.snapshot()

    def complete_goal(self, reason: str = "goal_complete") -> dict[str, Any]:
        """结束当前目标但保留手动 arm，会话可继续接收下一条明确目标。"""
        with self._lock:
            self._goal = None
            if self._state in {"armed", "degraded"}:
                self._state = "armed"
                self._reason = str(reason or "goal_complete")[:160]
        self._release_inputs()
        return self.snapshot()

    def submit_goal(
        self,
        text: Any,
        kind: Any = "explore",
        target_id: Any = None,
        selector: Any = None,
        constraints: Any = None,
        based_on_revision: Any = None,
    ) -> dict[str, Any]:
        normalized_text = str(text or "").strip()[:256]
        normalized_kind = str(kind or "explore").strip().lower()
        normalized_target_id = str(target_id or "").replace("\x00", "").strip()[:96] or None
        try:
            normalized_selector = _normalize_selector(selector)
            normalized_constraints = _normalize_constraints(constraints)
            normalized_revision = _normalize_revision(based_on_revision)
        except ValueError as exc:
            return {"accepted": False, **self.snapshot(), "reason": str(exc)}
        if not normalized_text:
            return {"accepted": False, **self.snapshot(), "reason": "goal must not be empty"}
        if normalized_kind not in ALLOWED_GOAL_KINDS:
            return {
                "accepted": False,
                **self.snapshot(),
                "reason": "unsupported autonomy goal kind",
            }
        # 接近、跟随和交互会驱动 avatar，不能靠“person”等标签模糊选择。
        # 调用方必须从最新世界快照里显式锁定一个实体，避免海报、镜像或旁人被
        # 高置信度检测框误选后触发移动。
        if normalized_kind in TARGETED_GOAL_KINDS and normalized_target_id is None:
            return {
                "accepted": False,
                **self.snapshot(),
                "reason": "target_id is required for targeted autonomy goals",
            }
        with self._lock:
            self._refresh_locked()
            if self._state != "armed":
                result = {"accepted": False, "reason": self._reason, **self.snapshot()}
                return result
            if normalized_revision is not None and normalized_revision > self._world_revision:
                return {
                    "accepted": False,
                    **self.snapshot(),
                    "reason": "based_on_revision is newer than current world revision",
                }
            self._goal = AutonomyGoal(
                normalized_kind,
                normalized_text,
                normalized_target_id,
                normalized_selector,
                normalized_constraints,
                normalized_revision,
                self._clock(),
            )
            self._reason = "goal_accepted_pending_perception"
            return {"accepted": True, "reason": None, **self.snapshot()}

    def update_world(self, world: Mapping[str, Any] | None) -> None:
        with self._lock:
            expired = self._refresh_locked()
            if expired:
                self._release_inputs()
                return
            if not isinstance(world, Mapping):
                return
            status = world.get("status") if isinstance(world.get("status"), Mapping) else {}
            try:
                revision = int(status.get("revision", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                revision = 0
            if revision < self._world_revision:
                return
            self._world_revision = revision
            # degraded 也要往下走：它是「授权还在，但暂时看不见」，必须能恢复。
            # stopping/disarmed 是终态，不复活。
            if self._state not in {"armed", "degraded"}:
                return
            events = world.get("events") if isinstance(world.get("events"), (list, tuple)) else ()
            if any(
                isinstance(event, Mapping)
                and str(event.get("type") or event.get("kind") or "").lower()
                in {"world_changed", "world_switch", "instance_changed", "world_loaded"}
                for event in events
            ):
                self._disarm_locked("world_changed")
                self._release_inputs()
                return
            if self._goal is None:
                return
            if not bool(world.get("available")):
                status_age = None
                try:
                    status_age = float(status.get("last_observation_age_ms"))
                except (TypeError, ValueError, OverflowError):
                    pass
                uncertainties = set(blocking_uncertainties(world.get("uncertainties")))
                fresh_empty_observation = (
                    world.get("capture_active") is not False
                    and status_age is not None
                    and status_age <= 2500.0
                    and not (uncertainties - {"no_recent_visual_observation"})
                )
                fresh_empty_goal = fresh_empty_observation and (
                    (
                        self._goal.kind == "explore"
                        and self._goal.selector is not None
                    )
                    or self._goal.kind == "approach_observe"
                )
                if fresh_empty_goal:
                    self._state = "armed"
                    self._reason = (
                        "goal_waiting_for_explorer"
                        if self._goal.kind == "explore"
                        else "goal_reacquiring_target"
                    )
                    return
                self._state = "degraded"
                self._reason = "world_observation_unknown"
                self._release_inputs()
                return
            if self._state == "degraded":
                # 看回来了就恢复。world.available 只表示「此刻有没有没过期的实体」，
                # 而人的 TTL 才 1.5 秒——检测漏一帧半就会掉一次。没有这条恢复边的话，
                # 一次瞬时丢失就把整个会话锁死在 degraded，导航器从此只报
                # autonomy_not_armed，实机上 2.5 秒就复现了。
                self._state = "armed"
            # This first runtime only authorizes goals. Navigation policy is
            # supplied by the perception/planner layer and must explicitly
            # submit bounded controller commands; no blind movement is emitted.
            self._reason = "goal_waiting_for_planner"

    def _refresh_locked(self) -> bool:
        if self._armed_until is not None and self._clock() >= self._armed_until:
            self._disarm_locked("session_ttl_expired")
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        expired = False
        with self._lock:
            expired = self._refresh_locked()
            now = self._clock()
            result = {
                "state": self._state,
                "reason": self._reason,
                "armed": self._state in {"armed", "degraded", "stopping"},
                "armed_until_monotonic": self._armed_until,
                "remaining_seconds": (
                    None if self._armed_until is None else round(max(0.0, self._armed_until - now), 2)
                ),
                "world_revision": self._world_revision,
                "goal": None if self._goal is None else {
                    "kind": self._goal.kind,
                    "text": self._goal.text,
                    "target_id": self._goal.target_id,
                    "selector": None if self._goal.selector is None else dict(self._goal.selector),
                    "constraints": (
                        None if self._goal.constraints is None else dict(self._goal.constraints)
                    ),
                    "based_on_revision": self._goal.based_on_revision,
                    "age_seconds": round(max(0.0, now - self._goal.created_at), 2),
                },
                "capabilities": {
                    "current_instance_actions": True,
                    "chat": True,
                    "world_switch": False,
                    "friend_graph": False,
                    "voice": False,
                },
            }
        if expired:
            self._release_inputs()
        return result


__all__ = [
    "ALLOWED_GOAL_KINDS",
    "ALLOWED_SELECTOR_TYPES",
    "AutonomyRuntime",
    "TARGETED_GOAL_KINDS",
]
