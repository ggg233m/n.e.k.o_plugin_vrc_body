"""AnyaDance 控制器的有界闭环导航。

该导航器刻意设计为小型局部控制循环。它从不调用模型、等待 HTTP 请求，也不会在世界状态未知时凭空生成移动指令。高层自主目标会被转换为简短的、最新优先的控制器更新，且每次更新都必须以新鲜且可见的实体观测作为门控条件。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import threading
import time
from typing import Any, Callable, Mapping

from .world_state import blocking_uncertainties


AxisSender = Callable[[str, float, float, int], bool]
ReleaseSender = Callable[[str], None]
SnapshotProvider = Callable[[], Mapping[str, Any]]

# 目标贴到画面上下边时表观高度会饱和，无法再作为距离的单调函数。此时按一个
# 更低的比例判定「已到达」，避免因为读数封顶而一直前进撞上对方。安全边界留在
# 代码里而不是配置项。
_CLIPPED_REACH_FACTOR = 0.6


@dataclass(frozen=True)
class NavigatorConfig:
    """Safety limits for the local 10 Hz navigation loop."""

    tick_hz: float = 10.0
    pulse_ms: int = 220
    max_forward_axis: float = 0.28
    max_turn_axis: float = 0.35
    max_observation_age_ms: int = 1000
    min_confidence: float = 0.55
    bearing_deadband_deg: float = 8.0
    target_distance_m: float = 1.25
    # 表观高度（归一化检测框高度）是二维检测器唯一能实测的接近度指标。它随
    # avatar 体型等比缩放，因此停止距离也随体型缩放，符合「个人空间随体型
    # 变化」的直觉；而按固定身高反算的米制距离对矮/高 avatar 会差出数倍。
    target_apparent_height: float = 0.55
    max_goal_age_s: float = 60.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.tick_hz) or not 2.0 <= self.tick_hz <= 30.0:
            raise ValueError("navigator.tick_hz must be between 2 and 30")
        if not 100 <= int(self.pulse_ms) <= 1000:
            raise ValueError("navigator.pulse_ms must be between 100 and 1000")
        if not 0.05 <= float(self.max_forward_axis) <= 0.6:
            raise ValueError("navigator.max_forward_axis must be between 0.05 and 0.6")
        if not 0.05 <= float(self.max_turn_axis) <= 0.8:
            raise ValueError("navigator.max_turn_axis must be between 0.05 and 0.8")
        if not 100 <= int(self.max_observation_age_ms) <= 5000:
            raise ValueError("navigator.max_observation_age_ms must be between 100 and 5000")
        if not 0.1 <= float(self.min_confidence) <= 1.0:
            raise ValueError("navigator.min_confidence must be between 0.1 and 1")
        if not 1.0 <= float(self.bearing_deadband_deg) <= 30.0:
            raise ValueError("navigator.bearing_deadband_deg must be between 1 and 30")
        if not 0.25 <= float(self.target_distance_m) <= 5.0:
            raise ValueError("navigator.target_distance_m must be between 0.25 and 5")
        if not 0.05 <= float(self.target_apparent_height) <= 0.95:
            raise ValueError("navigator.target_apparent_height must be between 0.05 and 0.95")
        if not 5.0 <= float(self.max_goal_age_s) <= 600.0:
            raise ValueError("navigator.max_goal_age_s must be between 5 and 600")


@dataclass(frozen=True)
class NavigationDecision:
    state: str
    reason: str
    target_id: str | None = None
    bearing_deg: float | None = None
    distance_m: float | None = None
    side: str | None = None
    x: float = 0.0
    y: float = 0.0
    pulse_ms: int = 0
    revision: int = 0
    observed_age_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "target_id": self.target_id,
            "bearing_deg": None if self.bearing_deg is None else round(self.bearing_deg, 2),
            "distance_m": None if self.distance_m is None else round(self.distance_m, 3),
            "side": self.side,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "pulse_ms": self.pulse_ms,
            "revision": self.revision,
            "observed_age_ms": None if self.observed_age_ms is None else round(self.observed_age_ms, 1),
        }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _spatial_hint(entity: Mapping[str, Any]) -> tuple[float | None, float | None, float | None, bool]:
    """Read a detector-neutral bearing/distance/apparent-size hint.

    Detectors should publish ``attributes.bearing_deg`` plus either
    ``attributes.apparent_height`` (normalized bbox height, the only approach
    metric a 2-D detector can actually measure) or ``attributes.distance_m``
    when a depth adapter supplies real metric depth.  Bboxes and relative
    positions are accepted as conservative fallbacks so a new detector can be
    integrated incrementally.

    Returns ``(bearing_deg, distance_m, apparent_height, apparent_clipped)``.
    """

    attributes = _mapping(entity.get("attributes")) or {}
    bearing = None
    for key in ("bearing_deg", "bearing", "angle_deg"):
        bearing = _finite(attributes.get(key))
        if bearing is not None:
            break
    if bearing is None:
        bearing_rad = _finite(attributes.get("bearing_rad"))
        if bearing_rad is not None:
            bearing = math.degrees(bearing_rad)

    distance = None
    for key in ("distance_m", "distance"):
        distance = _finite(attributes.get(key))
        if distance is not None:
            break

    position = attributes.get("relative_position", attributes.get("position"))
    if isinstance(position, Mapping):
        px = _finite(position.get("x"))
        pz = _finite(position.get("z"))
        if bearing is None and px is not None and pz is not None:
            bearing = math.degrees(math.atan2(px, -pz if abs(pz) > 1e-6 else 1e-6))
        if distance is None and px is not None and pz is not None:
            distance = math.sqrt(px * px + pz * pz)
    elif isinstance(position, (list, tuple)) and len(position) >= 3:
        px = _finite(position[0])
        pz = _finite(position[2])
        if bearing is None and px is not None and pz is not None:
            bearing = math.degrees(math.atan2(px, -pz if abs(pz) > 1e-6 else 1e-6))
        if distance is None and px is not None and pz is not None:
            distance = math.sqrt(px * px + pz * pz)

    if bearing is None:
        bbox = entity.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            left, _top, right, _bottom = (_finite(item) for item in bbox)
            if left is not None and right is not None and right >= left:
                center = (left + right) / 2.0
                # A bbox normally uses normalized coordinates; reject pixels
                # rather than turning wildly from an uncalibrated source.
                if -0.25 <= center <= 1.25:
                    bearing = (center - 0.5) * 90.0

    apparent = _finite(attributes.get("apparent_height"))
    clipped = bool(attributes.get("apparent_height_clipped"))
    if apparent is None:
        # Fall back to the bbox itself so a detector that only publishes boxes
        # can still close the approach loop.  Pixel-scale boxes are rejected
        # rather than guessed at.
        bbox = entity.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            _left, top, _right, bottom = (_finite(item) for item in bbox)
            if top is not None and bottom is not None and bottom >= top and bottom <= 1.25:
                apparent = bottom - top
                if not clipped:
                    clipped = top <= 0.001 or bottom >= 0.999
    if apparent is not None and not 0.0 < apparent <= 1.0:
        apparent = None

    if bearing is not None:
        bearing = _clamp(bearing, -180.0, 180.0)
    if distance is not None and (distance < 0.0 or distance > 100.0):
        distance = None
    return bearing, distance, apparent, clipped


class LocalNavigator:
    """A bounded local controller driven by fresh world observations."""

    def __init__(
        self,
        *,
        world_provider: SnapshotProvider,
        goal_provider: SnapshotProvider,
        send_axes: AxisSender,
        release_inputs: ReleaseSender,
        config: NavigatorConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or NavigatorConfig()
        self._world_provider = world_provider
        self._goal_provider = goal_provider
        self._send_axes = send_axes
        self._release_inputs = release_inputs
        self._clock = clock
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_side: str | None = None
        self._last_decision = NavigationDecision("idle", "not_started")
        self._last_error: str | None = None
        self._tick_count = 0
        self._command_count = 0
        self._stop_count = 0

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="neko-local-navigator", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        self._safe_release()

    @property
    def thread_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def tick(self) -> NavigationDecision:
        now = self._clock()
        with self._lock:
            self._tick_count += 1
        try:
            goal_state = self._goal_provider()
            world = self._world_provider()
            decision = self._decide(goal_state, world, now)
            self._apply(decision)
            with self._lock:
                self._last_error = None
                self._last_decision = decision
            return decision
        except Exception as exc:
            self._safe_release()
            decision = NavigationDecision("degraded", f"navigator_error:{type(exc).__name__}")
            with self._lock:
                self._last_error = str(exc)[:240]
                self._last_decision = decision
            return decision

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            decision = self._last_decision.to_dict()
            return {
                "enabled": True,
                "running": self.thread_alive,
                "tick_hz": self.config.tick_hz,
                "tick_count": self._tick_count,
                "command_count": self._command_count,
                "stop_count": self._stop_count,
                "active_side": self._active_side,
                "last_decision": decision,
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        period = 1.0 / self.config.tick_hz
        deadline = self._clock()
        while not self._stop_event.is_set():
            now = self._clock()
            if now < deadline:
                self._stop_event.wait(min(period, deadline - now))
                continue
            deadline = now + period
            self.tick()
        self._safe_release()

    def _decide(
        self,
        goal_state: Mapping[str, Any] | None,
        world: Mapping[str, Any] | None,
        now: float,
    ) -> NavigationDecision:
        if not isinstance(goal_state, Mapping) or goal_state.get("state") != "armed":
            return NavigationDecision("idle", "autonomy_not_armed")
        goal = _mapping(goal_state.get("goal"))
        if not goal:
            return NavigationDecision("idle", "no_goal")
        age_s = _finite(goal.get("age_seconds"))
        if age_s is not None and age_s > self.config.max_goal_age_s:
            return NavigationDecision("stop", "goal_expired")
        if not isinstance(world, Mapping) or not bool(world.get("available")):
            return NavigationDecision("stop", "world_unknown")
        uncertainties = blocking_uncertainties(world.get("uncertainties"))
        if uncertainties:
            return NavigationDecision("stop", "world_uncertain")
        status = _mapping(world.get("status")) or {}
        revision = int(_finite(status.get("revision")) or 0)
        observed_age = _finite(status.get("last_observation_age_ms"))
        if observed_age is None or observed_age > self.config.max_observation_age_ms:
            return NavigationDecision("stop", "observation_stale", revision=revision, observed_age_ms=observed_age)

        entity = self._select_target(goal, world.get("entities"))
        if entity is None:
            return NavigationDecision("stop", "target_not_visible", revision=revision, observed_age_ms=observed_age)
        target_id = str(entity.get("id") or "")[:96] or None
        confidence = _finite(entity.get("confidence")) or 0.0
        if entity.get("visible") is False or confidence < self.config.min_confidence:
            return NavigationDecision("stop", "target_low_confidence", target_id=target_id, revision=revision, observed_age_ms=observed_age)
        bearing, distance, apparent, apparent_clipped = _spatial_hint(entity)
        if bearing is None:
            return NavigationDecision("stop", "target_bearing_unknown", target_id=target_id, revision=revision, observed_age_ms=observed_age)
        if abs(bearing) > self.config.bearing_deadband_deg:
            turn = _clamp(
                bearing / 45.0 * self.config.max_turn_axis,
                -self.config.max_turn_axis,
                self.config.max_turn_axis,
            )
            return NavigationDecision(
                "turn",
                "target_off_center",
                target_id,
                bearing,
                distance,
                "right",
                turn,
                0.0,
                self.config.pulse_ms,
                revision,
                observed_age,
            )
        # 优先使用表观高度：这是二维检测器唯一实测得到的接近度指标。必须排在
        # ``distance is None`` 之前，否则不带深度适配器的检测器永远走不到这里。
        if apparent is not None:
            target_apparent = self.config.target_apparent_height
            reach_apparent = (
                target_apparent * _CLIPPED_REACH_FACTOR if apparent_clipped else target_apparent
            )
            if apparent >= reach_apparent:
                return NavigationDecision(
                    "reached",
                    "target_in_interaction_range",
                    target_id,
                    bearing,
                    distance,
                    revision=revision,
                    observed_age_ms=observed_age,
                )
            forward = _clamp(
                (target_apparent - apparent) / target_apparent * self.config.max_forward_axis,
                0.05,
                self.config.max_forward_axis,
            )
            return NavigationDecision(
                "advance",
                "target_centered",
                target_id,
                bearing,
                distance,
                "left",
                0.0,
                forward,
                self.config.pulse_ms,
                revision,
                observed_age,
            )
        if distance is None:
            return NavigationDecision("stop", "target_distance_unknown", target_id, bearing, None, revision=revision, observed_age_ms=observed_age)
        if distance <= self.config.target_distance_m:
            return NavigationDecision("reached", "target_in_interaction_range", target_id, bearing, distance, revision=revision, observed_age_ms=observed_age)
        forward = _clamp(
            (distance - self.config.target_distance_m) / max(distance, 0.25) * self.config.max_forward_axis,
            0.05,
            self.config.max_forward_axis,
        )
        return NavigationDecision(
            "advance",
            "target_centered",
            target_id,
            bearing,
            distance,
            "left",
            0.0,
            forward,
            self.config.pulse_ms,
            revision,
            observed_age,
        )

    def _select_target(self, goal: Mapping[str, Any], raw_entities: Any) -> Mapping[str, Any] | None:
        if not isinstance(raw_entities, (list, tuple)):
            return None
        text = str(goal.get("text") or "").strip().lower()
        target_id = str(goal.get("target_id") or "").strip().lower()
        tokens = [token for token in re.split(r"[^\w\u3400-\u9fff-]+", text) if len(token) >= 2]
        candidates: list[tuple[int, float, Mapping[str, Any]]] = []
        for raw in raw_entities:
            if not isinstance(raw, Mapping):
                continue
            label = str(raw.get("label") or "").strip().lower()
            entity_id = str(raw.get("id") or "").strip().lower()
            if raw.get("visible") is False:
                continue
            confidence = _finite(raw.get("confidence")) or 0.0
            score = 0
            if target_id and entity_id == target_id:
                score += 100
            if entity_id and entity_id in text:
                score += 20
            if label and label in text:
                score += 15
            score += sum(2 for token in tokens if token in label or token in entity_id)
            if score <= 0 and goal.get("kind") == "explore" and label in {"door", "portal", "entrance", "path", "入口", "门", "传送门"}:
                score = 1
            if score > 0:
                candidates.append((score, confidence, raw))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _apply(self, decision: NavigationDecision) -> None:
        if decision.state == "turn" and decision.side:
            if self._active_side == "left":
                self._release_inputs("left")
            if self._send_axes("right", decision.x, decision.y, decision.pulse_ms):
                with self._lock:
                    self._active_side = "right"
                    self._command_count += 1
            return
        if decision.state == "advance" and decision.side:
            if self._active_side == "right":
                self._release_inputs("right")
            if self._send_axes("left", decision.x, decision.y, decision.pulse_ms):
                with self._lock:
                    self._active_side = "left"
                    self._command_count += 1
            return
        self._safe_release()

    def _safe_release(self) -> None:
        with self._lock:
            active = self._active_side
            self._active_side = None
        if active is not None:
            try:
                self._release_inputs("all")
            finally:
                with self._lock:
                    self._stop_count += 1


__all__ = ["LocalNavigator", "NavigationDecision", "NavigatorConfig"]
