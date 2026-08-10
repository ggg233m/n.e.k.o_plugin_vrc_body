"""Deterministic layered behavior state for body motion arbitration."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MotionPolicy:
    mode: str
    layer: str
    priority: int
    interruptible: bool


@dataclass(frozen=True)
class ExpressionProfile:
    gesture: str
    default_side: str
    default_intensity: float
    default_duration_ms: int
    label: str
    head_only: bool = False


MOTION_POLICIES: dict[str, MotionPolicy] = {
    "enable": MotionPolicy("enabling", "safety", 100, False),
    "disable": MotionPolicy("disabling", "safety", 100, False),
    "reset": MotionPolicy("resetting", "safety", 100, False),
    "stop": MotionPolicy("stopped", "safety", 100, False),
    "cancel": MotionPolicy("cancelling", "safety", 100, False),
    "reach_and_grab": MotionPolicy("interaction", "base", 90, True),
    "sequence": MotionPolicy("sequence", "base", 85, True),
    "arm_pose": MotionPolicy("posing", "base", 80, True),
    "move_hand": MotionPolicy("posing", "base", 80, True),
    "hand": MotionPolicy("posing", "hand", 80, True),
    "play_clip": MotionPolicy("clip", "base", 75, True),
    "gesture": MotionPolicy("gesture", "base", 70, True),
    "semantic_clip": MotionPolicy("clip_expression", "expression_base", 40, True),
    "express": MotionPolicy("expressing", "expression", 40, True),
}
DEFAULT_POLICY = MotionPolicy("acting", "base", 60, True)


EXPRESSION_PROFILES: dict[str, ExpressionProfile] = {
    "greet": ExpressionProfile("wave", "right", 0.55, 1350, "问候"),
    "agree": ExpressionProfile("nod", "right", 0.34, 950, "同意", head_only=True),
    "disagree": ExpressionProfile("deny", "right", 0.40, 1050, "否定", head_only=True),
    "explain": ExpressionProfile("explain", "alternate", 0.40, 1500, "解释"),
    "present": ExpressionProfile("offer", "alternate", 0.48, 1450, "展示"),
    "think": ExpressionProfile("think", "alternate", 0.34, 1500, "思考"),
    "celebrate": ExpressionProfile("celebrate", "both", 0.65, 1500, "庆祝"),
    "question": ExpressionProfile("tilt", "right", 0.28, 1100, "疑问", head_only=True),
    "emphasize": ExpressionProfile("point", "alternate", 0.46, 1350, "强调"),
    "beckon": ExpressionProfile("beckon", "alternate", 0.50, 1500, "招手靠近"),
    "comfort": ExpressionProfile("comfort", "alternate", 0.38, 1450, "安慰"),
    "apologize": ExpressionProfile("apologize", "right", 0.42, 1450, "道歉"),
    "surprise": ExpressionProfile("surprise", "both", 0.52, 1300, "惊讶"),
    "shrug": ExpressionProfile("shrug", "both", 0.40, 1450, "无奈"),
    "clap": ExpressionProfile("clap", "both", 0.58, 1550, "鼓掌"),
    "laugh": ExpressionProfile("laugh", "right", 0.36, 1250, "发笑", head_only=True),
    "sigh": ExpressionProfile("sigh", "right", 0.34, 1400, "叹气", head_only=True),
    "idle": ExpressionProfile("sigh", "right", 0.25, 1800, "待机", head_only=True),
    "pose": ExpressionProfile("surprise", "both", 0.55, 1800, "摆姿势"),
    "stretch": ExpressionProfile("celebrate", "both", 0.72, 1800, "舒展"),
    "playful": ExpressionProfile("wave", "right", 0.75, 1800, "活泼表现"),
}
EXPRESSION_INTENTS = tuple(EXPRESSION_PROFILES)
HEAD_ONLY_GESTURES = frozenset(
    profile.gesture for profile in EXPRESSION_PROFILES.values() if profile.head_only
)


def policy_for(kind: str) -> MotionPolicy:
    return MOTION_POLICIES.get(kind, DEFAULT_POLICY)


def resolve_expression(
    intent: str,
    *,
    side: str,
    intensity: float | None,
    duration_ms: int | None,
    alternate_side: str,
) -> dict[str, Any]:
    profile = EXPRESSION_PROFILES[intent]
    resolved_side = side
    if side == "auto":
        resolved_side = profile.default_side
        if resolved_side == "alternate":
            resolved_side = alternate_side
    return {
        "intent": intent,
        "intent_label": profile.label,
        "gesture": profile.gesture,
        "side": resolved_side,
        "energy": profile.default_intensity if intensity is None else intensity,
        "duration_ms": profile.default_duration_ms if duration_ms is None else duration_ms,
        "head_only": profile.head_only,
    }


def expression_admission(snapshot: Mapping[str, Any], gesture: str) -> tuple[bool, str | None]:
    """Guard low-priority expression overlays without weakening explicit tools."""
    state = str(snapshot.get("state", "shutdown"))
    if state in {"disabled", "stopped_latched", "fault_latched", "shutdown"}:
        return False, f"expression cannot run while scheduler state is {state}"
    behavior = snapshot.get("behavior")
    base = behavior.get("base") if isinstance(behavior, Mapping) else None
    base_mode = str(base.get("mode", "")) if isinstance(base, Mapping) else ""
    if state == "moving" and base_mode in {"interaction", "sequence", "clip", "clip_expression", "gesture"}:
        if gesture not in HEAD_ONLY_GESTURES:
            return False, f"full-body expression is protected by active {base_mode} motion"
    return True, None


class BehaviorStateMachine:
    """Scheduler-owned behavior/layer state with bounded transition history."""

    def __init__(self, *, history_size: int = 16) -> None:
        self._base: dict[str, Any] | None = None
        self._overlays: dict[str, dict[str, Any]] = {}
        self._previous_base: dict[str, Any] | None = None
        self._transition: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._last_decision: dict[str, Any] | None = None

    @staticmethod
    def _public_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            return {}
        return {
            str(key): copy.deepcopy(value)
            for key, value in params.items()
            if not str(key).startswith("_")
        }

    @staticmethod
    def _finished(action: dict[str, Any], *, now: float, outcome: str) -> dict[str, Any]:
        result = copy.deepcopy(action)
        result["outcome"] = outcome
        result["ended_at_monotonic"] = now
        return result

    def activate_base(
        self,
        *,
        action_id: str,
        kind: str,
        now: float,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        policy = policy_for(kind)
        previous = None
        if self._base is not None:
            previous = self._finished(self._base, now=now, outcome="interrupted")
            self._previous_base = previous
            self._history.append(previous)
        current = {
            "id": action_id,
            "kind": kind,
            "mode": policy.mode,
            "layer": policy.layer,
            "priority": policy.priority,
            "interruptible": policy.interruptible,
            "started_at_monotonic": now,
            "params": self._public_params(params),
        }
        self._base = current
        self._last_decision = {
            "accepted": True,
            "id": action_id,
            "kind": kind,
            "reason": None,
            "at_monotonic": now,
        }
        self._transition = {
            "from": copy.deepcopy(previous),
            "to": copy.deepcopy(current),
            "started_at_monotonic": now,
            "strategy": "current_pose_snapshot_crossfade",
        }

    def finish_base(self, *, action_id: str | None, now: float, outcome: str) -> None:
        if self._base is None or (action_id and self._base.get("id") != action_id):
            return
        previous = self._finished(self._base, now=now, outcome=outcome)
        self._previous_base = previous
        self._history.append(previous)
        self._transition = {
            "from": copy.deepcopy(previous),
            "to": None,
            "started_at_monotonic": now,
            "strategy": "current_pose_snapshot_crossfade",
        }
        self._base = None

    def activate_overlay(
        self,
        *,
        action_id: str,
        kind: str,
        now: float,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        policy = policy_for(kind)
        self._overlays[action_id] = {
            "id": action_id,
            "kind": kind,
            "mode": policy.mode,
            "layer": policy.layer,
            "priority": policy.priority,
            "interruptible": policy.interruptible,
            "started_at_monotonic": now,
            "params": self._public_params(params),
        }
        self._last_decision = {
            "accepted": True,
            "id": action_id,
            "kind": kind,
            "reason": None,
            "at_monotonic": now,
        }

    def reject(self, *, action_id: str, kind: str, now: float, reason: str) -> None:
        decision = {
            "accepted": False,
            "id": action_id,
            "kind": kind,
            "reason": reason,
            "at_monotonic": now,
            "outcome": "rejected",
        }
        self._last_decision = decision
        self._history.append(copy.deepcopy(decision))

    def finish_overlay(self, *, action_id: str, now: float, outcome: str) -> None:
        action = self._overlays.pop(action_id, None)
        if action is not None:
            self._history.append(self._finished(action, now=now, outcome=outcome))

    def reset(self, *, now: float, outcome: str) -> None:
        self.finish_base(action_id=None, now=now, outcome=outcome)
        for action_id in tuple(self._overlays):
            self.finish_overlay(action_id=action_id, now=now, outcome=outcome)

    def snapshot(self, *, runtime_state: str, now: float) -> dict[str, Any]:
        base = copy.deepcopy(self._base)
        overlays = [copy.deepcopy(value) for value in self._overlays.values()]
        if base is not None:
            base["elapsed_seconds"] = round(max(0.0, now - base["started_at_monotonic"]), 3)
        for overlay in overlays:
            overlay["elapsed_seconds"] = round(max(0.0, now - overlay["started_at_monotonic"]), 3)

        if runtime_state == "disabled":
            mode = "disabled"
        elif runtime_state == "stopped_latched":
            mode = "stopped"
        elif runtime_state == "fault_latched":
            mode = "fault"
        elif runtime_state == "shutdown":
            mode = "shutdown"
        elif base is not None:
            mode = str(base["mode"])
        elif overlays:
            mode = "expressing"
        elif runtime_state == "holding":
            mode = "holding"
        else:
            mode = "idle"
        return {
            "mode": mode,
            "phase": runtime_state,
            "base": base,
            "overlays": overlays,
            "active_layers": ([base["layer"]] if base is not None else [])
            + [item["layer"] for item in overlays],
            "previous_base": copy.deepcopy(self._previous_base),
            "transition": copy.deepcopy(self._transition),
            "history": list(copy.deepcopy(self._history)),
            "last_decision": copy.deepcopy(self._last_decision),
            "policy_version": 1,
        }


__all__ = [
    "BehaviorStateMachine",
    "EXPRESSION_INTENTS",
    "EXPRESSION_PROFILES",
    "HEAD_ONLY_GESTURES",
    "MOTION_POLICIES",
    "MotionPolicy",
    "expression_admission",
    "policy_for",
    "resolve_expression",
]
