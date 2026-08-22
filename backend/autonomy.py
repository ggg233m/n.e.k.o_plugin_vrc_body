"""Session-scoped VRChat autonomy guardrails.

This module deliberately owns authorization and freshness, not perception or
high-frequency movement.  A planner can use the state to decide what to do,
while the body scheduler remains the only writer of AnyaDance frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping


ALLOWED_GOAL_KINDS = frozenset({"explore", "approach", "follow", "interact", "socialize"})


@dataclass
class AutonomyGoal:
    kind: str
    text: str
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

    def submit_goal(self, text: Any, kind: Any = "explore") -> dict[str, Any]:
        normalized_text = str(text or "").strip()[:256]
        normalized_kind = str(kind or "explore").strip().lower()
        if not normalized_text:
            return {"accepted": False, "reason": "goal must not be empty", **self.snapshot()}
        if normalized_kind not in ALLOWED_GOAL_KINDS:
            return {"accepted": False, "reason": "unsupported autonomy goal kind", **self.snapshot()}
        with self._lock:
            self._refresh_locked()
            if self._state != "armed":
                result = {"accepted": False, "reason": self._reason, **self.snapshot()}
                return result
            self._goal = AutonomyGoal(normalized_kind, normalized_text, self._clock())
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


__all__ = ["AutonomyRuntime", "ALLOWED_GOAL_KINDS"]
