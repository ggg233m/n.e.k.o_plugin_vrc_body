"""YUI NPC 宿主常驻自主行为循环。

本模块只消费 Unity 发布的语义地图、Region、Anchor、玩家槽位和 operation
终态。它不使用绝对自由坐标，也不把 ``accepted`` 当作完成。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .behavior_plan import TERMINAL_STATUSES
from .chat_context import RecentChatContextProvider
from .config import YuiAutonomyConfig
from .yui_adapter import YuiSemanticAdapter
from .yui_session import YuiSessionState


class AutonomyStimulusProvider(Protocol):
    """第二阶段可实现的异步刺激接口；第一阶段不得采图。"""

    async def get_stimulus(self, context: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """返回一个可选的低频兴趣信号。"""


class NoopAutonomyStimulusProvider:
    """第一阶段固定使用的无操作实现。"""

    async def get_stimulus(self, context: Mapping[str, Any]) -> Mapping[str, Any] | None:
        del context
        return None


@dataclass(slots=True)
class _PlanRecord:
    plan_id: str
    kind: str
    targets: tuple[str, ...]
    regions: tuple[str, ...]
    movement: bool
    started_at: float
    planned_dwell_s: float = 0.0
    intent_activity_index: int | None = None
    decision_reason: str = "rule"
    route_signature: tuple[str, ...] | None = None
    cross_region: bool = False
    interest_override_target: str | None = None
    intent_token: str | None = None


@dataclass(slots=True)
class _SocialStimulus:
    event_type: str
    player_slot: int
    created_at: float


@dataclass(slots=True)
class _IntentFragment:
    motivation: str
    mood: str
    activities: tuple[dict[str, Any], ...]
    avoid_targets: frozenset[str]
    interests: tuple[dict[str, Any], ...]
    received_at: float
    expires_at: float
    request_token: str
    activity_index: int = 0
    used_route_overrides: set[str] = field(default_factory=set)
    cross_region_count: int = 0


class AutonomyDirector:
    """以一秒级频率维持规则自主，同时让显式控制和安全态绝对优先。"""

    _READ_ONLY_TOOLS = frozenset({"npc.observe", "npc.world_query", "npc.plan_status"})
    _PERSISTENT_PAUSE_TOOLS = frozenset({"npc.stop", "npc.estop"})

    def __init__(
        self,
        adapter: YuiSemanticAdapter,
        session: YuiSessionState,
        config: YuiAutonomyConfig,
        *,
        stimulus_provider: AutonomyStimulusProvider | None = None,
        chat_context_provider: RecentChatContextProvider | None = None,
        inspiration_callback: Callable[[dict[str, Any]], None] | None = None,
        telemetry_callback: Callable[[dict[str, Any]], None] | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.adapter = adapter
        self.session = session
        self.config = config
        self.stimulus_provider = stimulus_provider or NoopAutonomyStimulusProvider()
        self.chat_context_provider = chat_context_provider
        self._inspiration_callback = inspiration_callback
        self._telemetry_callback = telemetry_callback
        self._rng = rng or random.Random()
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._desired_running = False
        self._pause_reason = "disabled" if not config.enabled else "not_started"
        self._active: _PlanRecord | None = None
        self._social_queue: deque[_SocialStimulus] = deque(maxlen=8)
        self._social_last_seen: dict[int, float] = {}
        self._recent_targets: deque[str] = deque(maxlen=12)
        self._recent_regions: deque[str] = deque(maxlen=8)
        self._route_history: dict[tuple[str, ...], float] = {}
        self._failures: dict[str, tuple[int, float]] = {}
        self._movement_seconds = 0.0
        self._dwell_seconds = 0.0
        self._plans_started = 0
        self._plans_completed = 0
        self._plans_failed = 0
        self._explicit_plan_id: str | None = None
        self._explicit_operation_id: str | None = None
        self._explicit_started_at = 0.0
        self._resume_at = 0.0
        self._startup_grace_until = 0.0
        self._last_decision_at = 0.0
        self._next_inspiration_at = self._new_inspiration_deadline(self._clock())
        self._intent: _IntentFragment | None = None
        self._pending_intent: _IntentFragment | None = None
        self._intent_request_serial = 0
        self._latest_intent_token: str | None = None
        self._startup_intent_needed = True
        self._request_after_explicit = False
        self._last_intent_request_reason: str | None = None
        self._last_intent_outcome: str | None = None
        self._last_decision_reason = "rule_fallback"
        self._last_social_event: dict[str, Any] | None = None
        self._fallback_active = True
        self._last_cross_region_at = float("-inf")
        self._last_movement_at = self._clock()
        self.session.add_event_listener(self._on_session_event)

    def _emit_telemetry(self, event: str, **fields: Any) -> None:
        """发送脱敏结构化诊断；日志故障不能影响自主循环。"""
        callback = self._telemetry_callback
        if callback is None:
            return
        try:
            callback({"event": event, **fields})
        except Exception:
            pass

    def start(self) -> dict[str, Any]:
        """启动或恢复自主；ESTOP 未清除时保持安全暂停。"""
        with self._condition:
            if not self.config.enabled:
                self._pause_reason = "disabled_by_config"
                return self.status()
            if self.session.estop or self.session.control_state == "estop":
                self._desired_running = False
                self._pause_reason = "estop"
                return self.status()
            self._desired_running = True
            self._pause_reason = ""
            self._resume_at = self._clock()
            self._startup_intent_needed = True
            # SET_CONTROL_MODE ACK 与 npc.state 投影可能短暂乱序。只在首次启动
            # 给出有界宽限；超过宽限的 safe_idle/watchdog 仍按安全暂停处理。
            self._startup_grace_until = self._clock() + 5.0
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._loop,
                    name="yui-autonomy-director",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()
            return self.status()

    def pause(self, reason: str = "manual_pause") -> dict[str, Any]:
        """持久暂停并立即取消自主来源计划。"""
        with self._condition:
            self._desired_running = False
            self._pause_reason = str(reason or "manual_pause")
            self._explicit_plan_id = None
            self._explicit_operation_id = None
            self._clear_intent_locked("paused")
        self.adapter.plan_manager.cancel_origin("autonomy", self._pause_reason)
        return self.status()

    def close(self) -> None:
        self.pause("director_closed")
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, self.config.decision_interval_s * 2.0))
        self.session.remove_event_listener(self._on_session_event)

    def before_explicit_tool(self, tool_name: str) -> None:
        """显式执行工具调用前抢占自主计划；只读工具不影响自主。"""
        if tool_name in self._READ_ONLY_TOOLS:
            return
        now = self._clock()
        with self._condition:
            self._explicit_plan_id = None
            self._explicit_operation_id = None
            self._explicit_started_at = now
            self._resume_at = float("inf")
            self._pause_reason = "explicit_control"
            self._clear_intent_locked("explicit_control")
            self._request_after_explicit = True
        self.adapter.plan_manager.cancel_origin("autonomy", "explicit_control")

    def after_explicit_tool(self, tool_name: str, result: Mapping[str, Any]) -> None:
        """普通命令在真实终态后延迟恢复；STOP/ESTOP 始终保持暂停。"""
        if tool_name in self._READ_ONLY_TOOLS:
            return
        if tool_name in self._PERSISTENT_PAUSE_TOOLS:
            self.pause("estop" if tool_name == "npc.estop" else "explicit_stop")
            return
        now = self._clock()
        with self._condition:
            plan_id = result.get("plan_id")
            operation_id = result.get("op_id") or result.get("operation_id")
            self._explicit_plan_id = plan_id if isinstance(plan_id, str) else None
            self._explicit_operation_id = (
                operation_id if isinstance(operation_id, str) else None
            )
            if self._explicit_plan_id is None and self._explicit_operation_id is None:
                self._resume_at = now + self.config.resume_delay_s
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            ratio_total = self._movement_seconds + self._dwell_seconds
            ratio = self._movement_seconds / ratio_total if ratio_total > 0.0 else 0.5
            now = self._clock()
            if not self.config.enabled:
                state = "disabled"
            elif not self._desired_running:
                state = "paused"
            elif self._explicit_plan_id or self._explicit_operation_id or now < self._resume_at:
                state = "explicit_control"
            elif self._active is not None:
                state = "executing"
            elif not self._control_ready():
                state = "waiting_control"
            else:
                state = "ready"
            intent = self._intent
            pending = self._pending_intent
            return {
                "enabled": self.config.enabled,
                "running": self._desired_running,
                "state": state,
                "pause_reason": self._pause_reason or None,
                "active_plan_id": None if self._active is None else self._active.plan_id,
                "active_kind": None if self._active is None else self._active.kind,
                "movement_ratio": round(ratio, 3),
                "movement_seconds": round(self._movement_seconds, 1),
                "dwell_seconds": round(self._dwell_seconds, 1),
                "plans_started": self._plans_started,
                "plans_completed": self._plans_completed,
                "plans_failed": self._plans_failed,
                "recent_targets": list(self._recent_targets),
                "recent_regions": list(self._recent_regions),
                "recent_routes": [
                    list(signature)
                    for signature, expires_at in sorted(self._route_history.items())
                    if expires_at > self._clock()
                ],
                "recent_route_count": sum(
                    1 for expires_at in self._route_history.values() if expires_at > now
                ),
                "blacklist": {
                    key: round(until - now, 1)
                    for key, (_count, until) in self._failures.items()
                    if until > now
                },
                "social_queue": len(self._social_queue),
                "current_intent": None if intent is None else {
                    "motivation": intent.motivation,
                    "mood": intent.mood,
                    "activity_index": intent.activity_index,
                    "activity_count": len(intent.activities),
                    "interest_count": len(intent.interests),
                    "current_activity": (
                        None
                        if intent.activity_index >= len(intent.activities)
                        else dict(intent.activities[intent.activity_index])
                    ),
                    "expires_in_s": round(max(0.0, intent.expires_at - now), 1),
                },
                "pending_intent": pending is not None,
                "last_intent_request_reason": self._last_intent_request_reason,
                "last_intent_outcome": self._last_intent_outcome,
                "last_decision_reason": self._last_decision_reason,
                "fallback_active": self._fallback_active,
                "chat_context": (
                    self.chat_context_provider.status()
                    if self.chat_context_provider is not None
                    else {
                        "enabled": False,
                        "source": "recent_file",
                        "file_state": "not_configured",
                        "turn_count": 0,
                    }
                ),
            }

    def offer_intent(self, value: Mapping[str, Any], request_token: str) -> bool:
        """接收独立 API 已严格校验的生活片段；只在活动边界应用。"""
        now = self._clock()
        with self._condition:
            if (
                not self._desired_running
                or request_token != self._latest_intent_token
                or not isinstance(value.get("motivation"), str)
                or not isinstance(value.get("mood"), str)
                or not isinstance(value.get("activities"), list)
            ):
                self._last_intent_outcome = "stale_or_paused"
                return False
            ttl_s = value.get("ttl_s", 240)
            if isinstance(ttl_s, bool) or not isinstance(ttl_s, int) or not 60 <= ttl_s <= 600:
                self._last_intent_outcome = "invalid_ttl"
                return False
            activities = tuple(
                dict(item)
                for item in value["activities"]
                if isinstance(item, Mapping)
            )
            if len(activities) != len(value["activities"]) or not 2 <= len(activities) <= 4:
                self._last_intent_outcome = "invalid_activities"
                return False
            avoid = value.get("avoid_targets", [])
            if not isinstance(avoid, list):
                self._last_intent_outcome = "invalid_avoid_targets"
                return False
            interests = value.get("interests", [])
            if not isinstance(interests, list) or len(interests) > 4:
                self._last_intent_outcome = "invalid_interests"
                return False
            normalized_interests: list[dict[str, Any]] = []
            for item in interests:
                if not isinstance(item, Mapping):
                    self._last_intent_outcome = "invalid_interests"
                    return False
                target_key = item.get("target_key")
                strength = item.get("strength")
                interest_ttl = item.get("ttl_s")
                if (
                    not isinstance(target_key, str)
                    or isinstance(strength, bool)
                    or not isinstance(strength, (int, float))
                    or not 0.0 <= float(strength) <= 1.0
                    or isinstance(interest_ttl, bool)
                    or not isinstance(interest_ttl, int)
                    or not 60 <= interest_ttl <= 600
                ):
                    self._last_intent_outcome = "invalid_interests"
                    return False
                normalized_interests.append({
                    "target_key": target_key,
                    "strength": float(strength),
                    "expires_at": now + interest_ttl,
                })
            self._pending_intent = _IntentFragment(
                motivation=str(value["motivation"]),
                mood=str(value["mood"]),
                activities=activities,
                avoid_targets=frozenset(str(item) for item in avoid),
                interests=tuple(normalized_interests),
                received_at=now,
                expires_at=now + ttl_s,
                request_token=request_token,
            )
            self._last_intent_outcome = "queued"
            self._condition.notify_all()
            self._emit_telemetry(
                "intent_queued",
                mood=str(value["mood"]),
                activity_count=len(activities),
                avoid_count=len(avoid),
                interest_count=len(normalized_interests),
            )
            return True

    def _loop(self) -> None:
        while not self._stop_event.wait(self.config.decision_interval_s):
            try:
                self._tick()
            except Exception:
                # 自主循环必须可恢复；任何单次决策错误都不得杀死常驻线程。
                with self._condition:
                    self._pause_reason = "decision_error_retrying"

    def _tick(self) -> None:
        now = self._clock()
        with self._condition:
            if not self._desired_running:
                return
            startup_intent_needed = self._startup_intent_needed
            if startup_intent_needed:
                self._startup_intent_needed = False
        chat_reason = self._poll_chat_context()
        if chat_reason is not None:
            self._request_intent(chat_reason)
        elif startup_intent_needed:
            self._request_intent("startup")
        if self.session.estop or self.session.control_state == "estop":
            self.pause("estop")
            return
        if self.session.control_state == "safe_idle":
            with self._condition:
                if now < self._startup_grace_until:
                    return
            self.pause("watchdog")
            return
        self._update_explicit_wait(now)
        with self._condition:
            if self._explicit_plan_id or self._explicit_operation_id or now < self._resume_at:
                return
        if self._active is not None:
            self._update_active(now)
        social = self._pop_social()
        if social is not None:
            if self._active is not None:
                previous = self._active
                self.adapter.plan_manager.cancel_origin("autonomy", "social_stimulus")
                self._record_elapsed(previous, now, include_planned_dwell=False)
                self._active = None
            if self._submit_social(social, now):
                return
        if self._active is not None:
            return
        if not self._control_ready() or self._has_unowned_active_operation():
            return
        if now - self._last_decision_at < self.config.decision_interval_s:
            return
        self._last_decision_at = now
        self._activate_pending_intent(now)
        if self._submit_intent_activity(now):
            return
        self._maybe_request_inspiration(now)
        self._submit_routine(now)

    def _poll_chat_context(self) -> str | None:
        """读取落盘聊天；角色切换时先失效所有旧角色意图 token。"""
        provider = self.chat_context_provider
        if provider is None or not provider.config.enabled:
            return None
        update = provider.poll()
        if not update.changed:
            return None
        if update.character_changed:
            with self._condition:
                self._clear_intent_locked("character_changed")
            return "character_changed"
        return "chat_updated"

    def _update_explicit_wait(self, now: float) -> None:
        with self._condition:
            plan_id = self._explicit_plan_id
            operation_id = self._explicit_operation_id
        terminal = False
        if plan_id is not None:
            terminal = self.adapter.plan_manager.status(plan_id).get("status") in TERMINAL_STATUSES
        elif operation_id is not None:
            operation = self.session.operations.get(operation_id)
            terminal = bool(operation and operation.get("status") in TERMINAL_STATUSES)
        if terminal:
            with self._condition:
                self._explicit_plan_id = None
                self._explicit_operation_id = None
                self._resume_at = now + self.config.resume_delay_s
                self._pause_reason = "resume_delay"
                request_after_explicit = self._request_after_explicit
                self._request_after_explicit = False
            if request_after_explicit:
                self._request_intent("explicit_completed")

    def _update_active(self, now: float) -> None:
        record = self._active
        if record is None:
            return
        result = self.adapter.plan_manager.status(record.plan_id)
        status = result.get("status")
        if status not in TERMINAL_STATUSES:
            return
        self._record_elapsed(
            record,
            now,
            include_planned_dwell=status == "succeeded",
        )
        request_reason: str | None = None
        if status == "succeeded":
            self._plans_completed += 1
            if record.movement:
                self._last_movement_at = now
            if record.route_signature is not None:
                self._route_history[record.route_signature] = now + 600.0
            if record.cross_region:
                self._last_cross_region_at = now
            for target in record.targets:
                self._recent_targets.append(target)
                self._failures.pop(target, None)
            for region in record.regions:
                self._recent_regions.append(region)
            if record.intent_activity_index is not None:
                with self._condition:
                    intent = self._intent
                    if (
                        intent is not None
                        and intent.activity_index == record.intent_activity_index
                    ):
                        if record.interest_override_target is not None:
                            intent.used_route_overrides.add(record.interest_override_target)
                        if record.cross_region and record.intent_token == intent.request_token:
                            intent.cross_region_count += 1
                        intent.activity_index += 1
                        self._last_intent_outcome = "activity_succeeded"
                        if intent.activity_index >= len(intent.activities):
                            self._intent = None
                            self._fallback_active = True
                            self._last_intent_outcome = "fragment_completed"
                            request_reason = "fragment_completed"
        elif status != "cancelled" or "explicit_control" not in str(result.get("detail") or ""):
            self._plans_failed += 1
            for target in record.targets:
                count, _until = self._failures.get(target, (0, 0.0))
                count += 1
                self._failures[target] = (count, now + min(300.0, 15.0 * (2 ** (count - 1))))
            if record.intent_activity_index is not None:
                with self._condition:
                    self._intent = None
                    self._fallback_active = True
                    self._last_intent_outcome = "activity_failed"
                request_reason = "fragment_failed"
        self._emit_telemetry(
            "activity_terminal",
            status=str(status or "unknown"),
            kind=record.kind,
            targets=list(record.targets),
            regions=list(record.regions),
            intent_activity_index=record.intent_activity_index,
            decision_reason=record.decision_reason,
        )
        self._active = None
        self._prune_route_history(now)
        if request_reason is not None:
            self._request_intent(request_reason)

    def _control_ready(self) -> bool:
        caps = set(self.session.capabilities)
        return bool(
            self.session.session > 0
            and self.session.discovery_ready
            and self.session.control_state in {"external", "moving", "action"}
            and not self.session.estop
            and self.session.operation_lifecycle
            and {"goto", "navmesh", "anchors", "world_map", "semantic_navigation"} <= caps
        )

    def _has_unowned_active_operation(self) -> bool:
        active = self.session.npc_state.get("active_ops")
        return isinstance(active, list) and bool(active)

    def _new_inspiration_deadline(self, now: float) -> float:
        lower, upper = self.config.llm_inspiration_range_s
        return now + self._rng.uniform(lower, upper)

    def _maybe_request_inspiration(self, now: float) -> None:
        if now < self._next_inspiration_at:
            return
        self._next_inspiration_at = self._new_inspiration_deadline(now)
        self._request_intent("fallback_timer")

    def _clear_intent_locked(self, outcome: str) -> None:
        self._intent = None
        self._pending_intent = None
        self._intent_request_serial += 1
        self._latest_intent_token = None
        self._fallback_active = True
        self._last_intent_outcome = outcome

    def _record_elapsed(
        self,
        record: _PlanRecord,
        now: float,
        *,
        include_planned_dwell: bool = True,
    ) -> None:
        elapsed = max(0.0, now - record.started_at)
        if not record.movement:
            self._dwell_seconds += elapsed
            return
        dwell = (
            min(max(0.0, record.planned_dwell_s), elapsed)
            if include_planned_dwell
            else 0.0
        )
        self._dwell_seconds += dwell
        self._movement_seconds += max(0.0, elapsed - dwell)

    @staticmethod
    def _safe_tags(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        return [
            item
            for item in value
            if isinstance(item, str) and item
        ][:8]

    def _intent_context(self, reason: str) -> dict[str, Any]:
        """构建只含语义事实和相对玩家信息的独立模型上下文。"""
        with self.session._condition:
            raw_location = self.session.npc_state.get("location") or {}
            nearest = raw_location.get("nearest_anchor") if isinstance(raw_location, Mapping) else None
            location = {
                "region_key": raw_location.get("region_key") if isinstance(raw_location, Mapping) else None,
                "floor_label": raw_location.get("floor_label") if isinstance(raw_location, Mapping) else None,
                "nearest_anchor": (
                    nearest.get("semantic_key") if isinstance(nearest, Mapping) else None
                ),
            }
            players = [
                {
                    "slot": slot,
                    "distance_m": item.get("d"),
                    "bearing_deg": item.get("brg"),
                }
                for slot, item in sorted(self.session.players.items())
                if isinstance(slot, int) and isinstance(item, Mapping)
            ]
            catalog: dict[str, list[dict[str, Any]]] = {}
            for kind in ("anchor", "region", "entity", "action"):
                projected: list[dict[str, Any]] = []
                for item in self.session.catalogs.get(kind, {}).values():
                    key = item.get("semantic_key")
                    if not isinstance(key, str):
                        continue
                    value: dict[str, Any] = {"semantic_key": key}
                    for field in ("region_key", "description_zh", "explorable", "orbitable"):
                        field_value = item.get(field)
                        if isinstance(field_value, (str, bool)):
                            value[field] = field_value
                    tags = self._safe_tags(item.get("tags"))
                    if tags:
                        value["tags"] = tags
                    projected.append(value)
                catalog[kind] = projected

        with self._condition:
            ratio_total = self._movement_seconds + self._dwell_seconds
            ratio = self._movement_seconds / ratio_total if ratio_total > 0.0 else 0.5
            current = self._intent
            previous_intent = None if current is None else {
                "motivation": current.motivation,
                "mood": current.mood,
                "activity_index": current.activity_index,
                "activity_count": len(current.activities),
            }
            context = {
                "reason": reason,
                "location": location,
                "players": players,
                "catalog": catalog,
                "recent_targets": list(self._recent_targets),
                "recent_regions": list(self._recent_regions),
                "recent_routes": [
                    list(signature)
                    for signature, expires_at in sorted(self._route_history.items())
                    if expires_at > self._clock()
                ],
                "failed_targets": [
                    key for key, (_count, until) in self._failures.items() if until > self._clock()
                ],
                "movement_ratio": round(ratio, 3),
                "previous_intent": previous_intent,
                "last_outcome": self._last_intent_outcome,
                "trigger_event": dict(self._last_social_event) if self._last_social_event else None,
                "instruction": (
                    "生成连贯的 2 到 4 项短生活活动；优先表达自然兴趣，"
                    "不要为了覆盖地图而机械巡逻。可以原地观察、转身或在附近随意走动，"
                    "只有真的对远处目标感兴趣时才跨区域。"
                ),
            }
            provider = self.chat_context_provider
            if provider is not None and provider.config.enabled:
                context["recent_conversation"] = provider.context()
            return context

    def _request_intent(self, reason: str) -> None:
        callback = self._inspiration_callback
        if callback is None:
            return
        with self._condition:
            if not self._desired_running:
                return
            self._intent_request_serial += 1
            token = f"{self.session.session}:{self._intent_request_serial}"
            self._latest_intent_token = token
            self._last_intent_request_reason = reason
        payload = {
            "request_token": token,
            "reason": reason,
            "context": self._intent_context(reason),
        }
        try:
            callback(payload)
            self._emit_telemetry("intent_requested", reason=reason)
        except Exception:
            with self._condition:
                if self._latest_intent_token == token:
                    self._last_intent_outcome = "request_queue_failed"

    def _activate_pending_intent(self, now: float) -> None:
        with self._condition:
            pending = self._pending_intent
            if pending is None:
                if self._intent is not None and self._intent.expires_at <= now:
                    self._intent = None
                    self._fallback_active = True
                    self._last_intent_outcome = "expired"
                return
            self._pending_intent = None
            if pending.expires_at <= now:
                self._last_intent_outcome = "expired_before_apply"
                return
            self._intent = pending
            self._fallback_active = False
            self._last_intent_outcome = "active"
            self._emit_telemetry(
                "intent_activated",
                mood=pending.mood,
                activity_count=len(pending.activities),
            )

    def _submit_intent_activity(self, now: float) -> bool:
        with self._condition:
            intent = self._intent
            if intent is None:
                return False
            if intent.expires_at <= now or intent.activity_index >= len(intent.activities):
                self._intent = None
                self._fallback_active = True
                self._last_intent_outcome = "expired" if intent.expires_at <= now else "fragment_completed"
                reason = "fragment_expired" if intent.expires_at <= now else "fragment_completed"
            else:
                activity_index = intent.activity_index
                activity = dict(intent.activities[activity_index])
                avoid_targets = intent.avoid_targets
                reason = ""
        if reason:
            self._request_intent(reason)
            return False

        compiled = self._compile_intent_activity(activity, avoid_targets, now)
        if compiled is None:
            with self._condition:
                if self._intent is intent:
                    self._intent = None
                    self._fallback_active = True
                    self._last_intent_outcome = "activity_unavailable"
            self._request_intent("fragment_failed")
            return False
        (
            graph,
            kind,
            targets,
            regions,
            movement,
            planned_dwell_s,
            decision_reason,
            route_signature,
            cross_region,
            interest_override_target,
        ) = compiled
        submitted = self._submit(
            graph,
            kind=f"intent_{kind}",
            targets=targets,
            regions=regions,
            movement=movement,
            now=now,
            planned_dwell_s=planned_dwell_s,
            intent_activity_index=activity_index,
            decision_reason=decision_reason,
            route_signature=route_signature,
            cross_region=cross_region,
            interest_override_target=interest_override_target,
            intent_token=intent.request_token,
        )
        if submitted:
            with self._condition:
                self._fallback_active = False
                self._last_decision_reason = decision_reason
        return submitted

    def _on_session_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "sys.watchdog":
            # 自动连接前日志尾部可能仍在输出旧会话的 watchdog。Director 尚未
            # 启动时忽略它；运行中的 watchdog 仍保持持久安全暂停。
            with self._condition:
                if not self._desired_running or self._clock() < self._startup_grace_until:
                    return
            self.pause("watchdog")
            return
        if event_type == "npc.state" and bool(event.get("estop")):
            self.pause("estop")
            return
        if event_type not in {"player.touch", "social.wave", "social.gaze", "social.approach"}:
            return
        if event_type == "social.gaze" and event.get("on") is not True:
            return
        slot = event.get("slot")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot not in self.session.players:
            return
        now = self._clock()
        request_intent = False
        with self._condition:
            if not self._desired_running:
                return
            if now - self._social_last_seen.get(slot, float("-inf")) < self.config.social_cooldown_s:
                return
            self._social_last_seen[slot] = now
            self._social_queue.append(_SocialStimulus(event_type, slot, now))
            self._last_social_event = {"type": event_type, "player_slot": slot}
            request_intent = True
            self._condition.notify_all()
        if request_intent:
            self._request_intent("social_event")

    def _pop_social(self) -> _SocialStimulus | None:
        with self._condition:
            while self._social_queue:
                item = self._social_queue.popleft()
                if item.player_slot in self.session.players:
                    return item
        return None

    def _submit_social(self, stimulus: _SocialStimulus, now: float) -> bool:
        actions = {
            str(item.get("semantic_key"))
            for item in self.session.catalogs["action"].values()
            if isinstance(item.get("semantic_key"), str)
        }
        preferred = {
            "social.wave": "greet_wave",
            "social.gaze": "listen",
            "social.approach": "greet",
            "player.touch": "agree_nod",
        }.get(stimulus.event_type)
        nodes: list[dict[str, Any]] = [
            {"id": "root", "type": "sequence", "children": ["approach", "look"]},
            {
                "id": "approach",
                "type": "approach",
                "player_slot": stimulus.player_slot,
                "distance_m": 1.5,
                "face_target": True,
            },
            {
                "id": "look",
                "type": "look_at",
                "player_slot": stimulus.player_slot,
                "duration_ms": 1800,
            },
        ]
        if preferred in actions:
            nodes[0]["children"].append("act")
            nodes.append({
                "id": "act",
                "type": "act",
                "action_key": preferred,
                "player_slot": stimulus.player_slot,
                "loop": False,
            })
        return self._submit(
            {"entry": "root", "nodes": nodes},
            kind="social",
            targets=(f"player_slot:{stimulus.player_slot}",),
            regions=(),
            movement=True,
            now=now,
            replace_active=True,
        )

    @staticmethod
    def _item_tags(item: Mapping[str, Any]) -> set[str]:
        raw = item.get("tags")
        if not isinstance(raw, (list, tuple, set)):
            return set()
        return {value for value in raw if isinstance(value, str) and value}

    def _select_activity_target(
        self,
        activity: Mapping[str, Any],
        avoid_targets: frozenset[str],
        snapshot: Mapping[str, Any],
        now: float,
    ) -> tuple[str, str, dict[str, Any]] | None:
        exact = activity.get("target_key")
        requested_tags = {
            value
            for value in activity.get("tags", [])
            if isinstance(value, str)
        }
        activity_kind = activity.get("kind")
        current_region = snapshot["location"].get("region_key")
        choices: list[tuple[float, str, str, dict[str, Any]]] = []
        source_kinds = ("region",) if activity_kind == "explore" else ("anchor", "region", "entity")
        collections = {"anchor": "anchors", "region": "regions", "entity": "entities"}
        for kind in source_kinds:
            values = snapshot[collections[kind]].values()
            for item in values:
                key = item.get("semantic_key")
                if not isinstance(key, str) or key in avoid_targets or self._blacklisted(key, now):
                    continue
                if kind == "region" and activity_kind == "explore" and not bool(item.get("explorable")):
                    continue
                if exact is not None and key != exact:
                    continue
                item_tags = self._item_tags(item)
                if requested_tags and not requested_tags.intersection(item_tags):
                    continue
                region = key if kind == "region" else item.get("region_key")
                score = self._candidate_score(key, region, current_region)
                if exact == key:
                    score += 20.0
                score += min(9.0, 3.0 * len(requested_tags.intersection(item_tags)))
                choices.append((score, kind, key, dict(item)))
        if not choices:
            return None
        choices.sort(key=lambda value: value[0], reverse=True)
        top_score = choices[0][0]
        top = [value for value in choices if value[0] >= top_score - 0.75]
        _score, kind, key, item = self._rng.choice(top)
        return kind, key, item

    def _semantic_route_nodes(
        self,
        target_kind: str,
        target_key: str,
        target: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]], bool] | None:
        anchors: dict[int, dict[str, Any]] = snapshot["anchors"]
        edges: dict[int, dict[str, Any]] = snapshot["edges"]
        location = snapshot["location"]
        nearest = location.get("nearest_anchor") if isinstance(location, Mapping) else None
        nearest_key = nearest.get("semantic_key") if isinstance(nearest, Mapping) else None
        anchor_by_key = {
            str(item.get("semantic_key")): anchor_id
            for anchor_id, item in anchors.items()
            if isinstance(item.get("semantic_key"), str)
        }
        source_id = anchor_by_key.get(str(nearest_key)) if nearest_key is not None else None
        if target_kind == "anchor":
            destination_id = target.get("id")
            if not isinstance(destination_id, int):
                destination_id = anchor_by_key.get(target_key)
        elif target_kind == "region":
            destination_id = target.get("entry_anchor_id")
        else:
            destination_id = target.get("approach_anchor_id")
        if not isinstance(destination_id, int) or destination_id not in anchors:
            return None
        if source_id == destination_id:
            return [], [], False
        paths = self._route_paths(source_id, anchors, edges)
        if edges and source_id is not None and destination_id not in paths:
            return None
        route_ids = paths.get(destination_id, [destination_id])
        if route_ids and source_id == route_ids[0]:
            route_ids = route_ids[1:]
        children: list[str] = []
        nodes: list[dict[str, Any]] = []
        for index, anchor_id in enumerate(route_ids):
            anchor = anchors.get(anchor_id)
            anchor_key = None if anchor is None else anchor.get("semantic_key")
            if not isinstance(anchor_key, str):
                continue
            node_id = f"route{index}"
            children.append(node_id)
            nodes.append({"id": node_id, "type": "navigate", "target_key": anchor_key})
        return children, nodes, bool(children)

    @staticmethod
    def _bounded_dwell_s(requested: int) -> int:
        """动作混合交给模型；宿主只保留协议允许的时长边界。"""
        return min(60, max(5, requested))

    def _compile_intent_activity(
        self,
        activity: Mapping[str, Any],
        avoid_targets: frozenset[str],
        now: float,
    ) -> tuple[
        dict[str, Any],
        str,
        tuple[str, ...],
        tuple[str, ...],
        bool,
        float,
        str,
        tuple[str, ...] | None,
        bool,
        str | None,
    ] | None:
        kind = activity.get("kind")
        if kind not in {
            "visit", "explore", "linger", "socialize", "perform", "observe", "local_roam",
        }:
            return None
        duration_s = activity.get("duration_s")
        if isinstance(duration_s, bool) or not isinstance(duration_s, int):
            return None
        duration_s = self._bounded_dwell_s(duration_s)
        snapshot = self._map_snapshot()

        if kind in {"observe", "local_roam"}:
            target: tuple[str, str, dict[str, Any]] | None = None
            if activity.get("target_key") is not None or activity.get("tags"):
                target = self._select_activity_target(
                    activity,
                    avoid_targets,
                    snapshot,
                    now,
                )
                if target is None:
                    return None
            requested_slot = activity.get("player_slot")
            player_slot = (
                requested_slot
                if isinstance(requested_slot, int) and requested_slot in self.session.players
                else None
            )
            if kind == "observe" and target is None and player_slot is None:
                return None

            children: list[str] = []
            nodes: list[dict[str, Any]] = []
            movement = False
            if kind == "local_roam":
                style = activity.get("style")
                if style not in {
                    "stay_and_look", "turn_left", "turn_right", "meander", "small_loop",
                }:
                    return None
                if style in {"turn_left", "turn_right"}:
                    delta = self._rng.uniform(30.0, 85.0)
                    if style == "turn_left":
                        delta = -delta
                    children.append("turn")
                    nodes.append({"id": "turn", "type": "turn_relative", "delta_deg": delta})
                    movement = True
                elif style == "stay_and_look":
                    # 没有视觉输入时也避免完全僵直：只做一次轻微随机转头式转身，
                    # 随后在新朝向驻足；不会凭空生成注视坐标。
                    delta = self._rng.uniform(12.0, 32.0)
                    if self._rng.random() < 0.5:
                        delta = -delta
                    children.append("turn")
                    nodes.append({"id": "turn", "type": "turn_relative", "delta_deg": delta})
                    movement = True
                elif style in {"meander", "small_loop"} and snapshot.get("local_navigation"):
                    if style == "small_loop":
                        step_count = 4
                        bearings = [90.0] * step_count
                        distance = self._rng.uniform(0.5, 0.9)
                        distances = [distance] * step_count
                    else:
                        step_count = 2 if duration_s < 15 else 3
                        bearings = [self._rng.uniform(-100.0, 100.0) for _ in range(step_count)]
                        distances = [self._rng.uniform(0.5, 1.5) for _ in range(step_count)]
                    for index, (bearing, distance) in enumerate(zip(bearings, distances)):
                        node_id = f"step{index}"
                        children.append(node_id)
                        nodes.append({
                            "id": node_id,
                            "type": "move_relative",
                            "bearing_deg": bearing,
                            "distance_m": distance,
                            "face_travel": True,
                            "allow_shorter": True,
                        })
                    movement = True

            if player_slot is not None:
                children.append("observe")
                nodes.append({
                    "id": "observe",
                    "type": "look_at",
                    "player_slot": player_slot,
                    "duration_ms": duration_s * 1000,
                })
            elif target is not None:
                children.append("observe")
                nodes.append({
                    "id": "observe",
                    "type": "look_at_target",
                    "target_key": target[1],
                    "duration_ms": duration_s * 1000,
                })
            else:
                children.append("linger")
                nodes.append({
                    "id": "linger",
                    "type": "wait",
                    "duration_ms": duration_s * 1000,
                })
            action_key = activity.get("action_key")
            if isinstance(action_key, str):
                children.append("act")
                action: dict[str, Any] = {
                    "id": "act",
                    "type": "act",
                    "action_key": action_key,
                    "loop": False,
                }
                if player_slot is not None:
                    action["player_slot"] = player_slot
                nodes.append(action)
            graph = (
                {"entry": children[0], "nodes": nodes}
                if len(children) == 1
                else {
                    "entry": "root",
                    "nodes": [
                        {"id": "root", "type": "sequence", "children": children},
                        *nodes,
                    ],
                }
            )
            targets = (
                (f"player_slot:{player_slot}",)
                if player_slot is not None
                else ((target[1],) if target is not None else ())
            )
            return (
                graph,
                str(kind),
                targets,
                (),
                movement,
                float(duration_s),
                "llm_observe_interest" if kind == "observe" else f"llm_local_{activity.get('style')}",
                None,
                False,
                None,
            )

        if kind == "socialize":
            players = [
                (slot, item)
                for slot, item in self.session.players.items()
                if isinstance(slot, int) and isinstance(item, Mapping)
            ]
            requested_slot = activity.get("player_slot")
            if requested_slot is not None:
                players = [item for item in players if item[0] == requested_slot]
            if not players:
                return None
            slot, _player = min(
                players,
                key=lambda item: float(item[1].get("d", float("inf"))),
            )
            children = ["approach", "look"]
            nodes: list[dict[str, Any]] = [
                {
                    "id": "approach",
                    "type": "approach",
                    "player_slot": slot,
                    "distance_m": 1.5,
                    "face_target": True,
                },
                {
                    "id": "look",
                    "type": "look_at",
                    "player_slot": slot,
                    "duration_ms": 1800,
                },
            ]
            action_key = activity.get("action_key")
            if isinstance(action_key, str):
                children.append("act")
                nodes.append({
                    "id": "act",
                    "type": "act",
                    "action_key": action_key,
                    "player_slot": slot,
                    "loop": False,
                })
            children.append("linger")
            nodes.append({"id": "linger", "type": "wait", "duration_ms": duration_s * 1000})
            return (
                {"entry": "root", "nodes": [{"id": "root", "type": "sequence", "children": children}, *nodes]},
                "socialize",
                (f"player_slot:{slot}",),
                (),
                True,
                float(duration_s),
                "llm_social_interest",
                None,
                False,
                None,
            )

        target: tuple[str, str, dict[str, Any]] | None = None
        if activity.get("target_key") is not None or activity.get("tags"):
            target = self._select_activity_target(activity, avoid_targets, snapshot, now)
            if target is None:
                return None
        if kind in {"visit", "explore"} and target is None:
            return None

        children: list[str] = []
        nodes: list[dict[str, Any]] = []
        movement = False
        targets: tuple[str, ...] = ()
        regions: tuple[str, ...] = ()
        decision_reason = "llm_current_location"
        route_signature: tuple[str, ...] | None = None
        cross_region = False
        interest_override_target: str | None = None
        if target is not None:
            target_kind, target_key, target_item = target
            route = self._semantic_route_nodes(
                target_kind,
                target_key,
                target_item,
                snapshot,
            )
            if route is None:
                return None
            route_children, route_nodes, movement = route
            children.extend(route_children)
            nodes.extend(route_nodes)
            targets = (target_key,)
            region_key = target_key if target_kind == "region" else target_item.get("region_key")
            if isinstance(region_key, str):
                regions = (region_key,)
            current_region = snapshot["location"].get("region_key")
            cross_region = bool(
                movement
                and isinstance(region_key, str)
                and isinstance(current_region, str)
                and region_key != current_region
            )
            with self._condition:
                current_intent = self._intent
                if (
                    cross_region
                    and current_intent is not None
                    and current_intent.cross_region_count >= 1
                ):
                    return None
            nearest = snapshot["location"].get("nearest_anchor")
            source_key = (
                nearest.get("semantic_key")
                if isinstance(nearest, Mapping)
                else None
            )
            route_keys = tuple(
                str(node["target_key"])
                for node in route_nodes
                if node.get("type") == "navigate" and isinstance(node.get("target_key"), str)
            )
            route_signature = self._canonical_route_signature(source_key, route_keys)
            admitted, interest_override_target = self._route_admission(
                route_signature,
                target_key,
                now,
                allow_interest_override=True,
            )
            if not admitted:
                return None
            decision_reason = (
                "llm_interest_route_override"
                if interest_override_target is not None
                else (
                    "llm_exact_target"
                    if activity.get("target_key") == target_key
                    else "llm_tag_match"
                )
            )

        planned_dwell_s = 0.0
        if kind == "explore":
            if target is None or target[0] != "region" or not snapshot.get("local_navigation"):
                return None
            children.append("explore")
            nodes.append({
                "id": "explore",
                "type": "explore",
                "region_key": target[1],
                "duration_ms": duration_s * 1000,
                "strategy": "unvisited",
            })
            movement = True
        else:
            action_key = activity.get("action_key")
            if isinstance(action_key, str):
                action: dict[str, Any] = {
                    "id": "act",
                    "type": "act",
                    "action_key": action_key,
                    "loop": False,
                }
                player_slot = activity.get("player_slot")
                if isinstance(player_slot, int) and player_slot in self.session.players:
                    action["player_slot"] = player_slot
                children.append("act")
                nodes.append(action)
            children.append("linger")
            nodes.append({"id": "linger", "type": "wait", "duration_ms": duration_s * 1000})
            planned_dwell_s = float(duration_s)

        if not children:
            return None
        if len(children) == 1:
            graph = {"entry": children[0], "nodes": nodes}
        else:
            graph = {
                "entry": "root",
                "nodes": [{"id": "root", "type": "sequence", "children": children}, *nodes],
            }
        return (
            graph,
            str(kind),
            targets,
            regions,
            movement,
            planned_dwell_s,
            decision_reason,
            route_signature,
            cross_region,
            interest_override_target,
        )

    def _submit_routine(self, now: float) -> None:
        self._fallback_active = True
        self._last_decision_reason = "rule_fallback"
        snapshot = self._map_snapshot()
        # 不再追逐固定移动占比。通常优先做原地观察、转身或附近闲逛；只有
        # 45 秒都没有任何移动时才由规则防止永久站立。
        startup_semantic = self._plans_started == 0
        force_movement = now - self._last_movement_at >= 45.0
        prefer_local = bool(
            not startup_semantic
            and (
                force_movement
            or self._rng.random() < 0.65
            or not snapshot["anchors"]
            )
        )
        if prefer_local:
            if force_movement and snapshot.get("local_navigation"):
                style = "meander"
            elif force_movement:
                style = self._rng.choice(["turn_left", "turn_right"])
            else:
                style = self._rng.choice([
                    "stay_and_look",
                    "stay_and_look",
                    "turn_left",
                    "turn_right",
                    "meander",
                    "small_loop",
                ])
            duration_s = int(round(self._rng.uniform(*self.config.dwell_range_s)))
            compiled = self._compile_intent_activity(
                {
                    "kind": "local_roam",
                    "style": style,
                    "duration_s": max(5, min(60, duration_s)),
                },
                frozenset(),
                now,
            )
            if compiled is not None:
                (
                    graph,
                    kind,
                    targets,
                    regions,
                    movement,
                    planned_dwell_s,
                    _reason,
                    route_signature,
                    cross_region,
                    _override,
                ) = compiled
                if self._submit(
                    graph,
                    kind=f"rule_{kind}",
                    targets=targets,
                    regions=regions,
                    movement=movement,
                    now=now,
                    planned_dwell_s=planned_dwell_s,
                    decision_reason=f"rule_local_{style}",
                    route_signature=route_signature,
                    cross_region=cross_region,
                ):
                    self._last_decision_reason = f"rule_local_{style}"
                    return

        candidates = self._movement_candidates(snapshot, now)
        if not candidates:
            # 没有安全可达且未重复的语义路线时驻足，不为凑路线再次往返。
            duration = int(round(self._rng.uniform(*self.config.dwell_range_s) * 1000.0))
            self._submit(
                {"entry": "rest", "nodes": [{"id": "rest", "type": "wait", "duration_ms": duration}]},
                kind="dwell_no_target",
                targets=(),
                regions=(),
                movement=False,
                now=now,
            )
            return
        candidates.sort(key=lambda item: item[0], reverse=True)
        top = candidates[: min(3, len(candidates))]
        (
            _score,
            graph,
            kind,
            targets,
            regions,
            route_signature,
            cross_region,
        ) = self._rng.choice(top)
        self._submit(
            graph,
            kind=kind,
            targets=targets,
            regions=regions,
            movement=True,
            now=now,
            decision_reason="rule_semantic_fallback",
            route_signature=route_signature,
            cross_region=cross_region,
        )
        self._fallback_active = True
        self._last_decision_reason = "rule_semantic_fallback"

    def _map_snapshot(self) -> dict[str, Any]:
        with self.session._condition:  # 同一 runtime 内读取日志投影的一致快照。
            return {
                "anchors": {key: dict(value) for key, value in self.session.catalogs["anchor"].items()},
                "regions": {key: dict(value) for key, value in self.session.catalogs["region"].items()},
                "entities": {
                    key: dict(value)
                    for key, value in self.session.catalogs.get("entity", {}).items()
                },
                "actions": {
                    key: dict(value)
                    for key, value in self.session.catalogs.get("action", {}).items()
                },
                "edges": {key: dict(value) for key, value in self.session.catalogs["route_edge"].items()},
                "location": dict(self.session.npc_state.get("location") or {}),
                "local_navigation": self.session.local_navigation,
            }

    def _movement_candidates(
        self,
        snapshot: Mapping[str, Any],
        now: float,
    ) -> list[
        tuple[
            float,
            dict[str, Any],
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...] | None,
            bool,
        ]
    ]:
        anchors: dict[int, dict[str, Any]] = snapshot["anchors"]
        regions: dict[int, dict[str, Any]] = snapshot["regions"]
        edges: dict[int, dict[str, Any]] = snapshot["edges"]
        location = snapshot["location"]
        nearest = location.get("nearest_anchor") if isinstance(location, Mapping) else None
        nearest_key = nearest.get("semantic_key") if isinstance(nearest, Mapping) else None
        anchor_by_key = {
            str(item.get("semantic_key")): anchor_id
            for anchor_id, item in anchors.items()
            if isinstance(item.get("semantic_key"), str)
        }
        source_id = anchor_by_key.get(str(nearest_key)) if nearest_key is not None else None
        paths = self._route_paths(source_id, anchors, edges)
        current_region = location.get("region_key") if isinstance(location, Mapping) else None
        candidates: list[
            tuple[
                float,
                dict[str, Any],
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...] | None,
                bool,
            ]
        ] = []

        for anchor_id, item in anchors.items():
            key = item.get("semantic_key")
            if not isinstance(key, str) or key == nearest_key or self._blacklisted(key, now):
                continue
            path = paths.get(anchor_id)
            if edges and source_id is not None and path is None:
                continue
            route_ids = path[1:] if path else [anchor_id]
            route_keys = tuple(
                str(anchors[item_id].get("semantic_key"))
                for item_id in route_ids
                if item_id in anchors and isinstance(anchors[item_id].get("semantic_key"), str)
            )
            if not route_keys:
                continue
            target_region = item.get("region_key")
            cross_region = bool(
                isinstance(target_region, str)
                and isinstance(current_region, str)
                and target_region != current_region
            )
            if cross_region and now - self._last_cross_region_at < 180.0:
                continue
            route_signature = self._canonical_route_signature(
                str(nearest_key) if isinstance(nearest_key, str) else None,
                route_keys,
            )
            admitted, _override = self._route_admission(
                route_signature,
                key,
                now,
                allow_interest_override=False,
            )
            if not admitted:
                continue
            score = self._candidate_score(key, target_region, current_region)
            graph = self._navigate_graph(route_keys)
            candidates.append((
                score,
                graph,
                "route",
                (key,),
                (str(target_region),) if isinstance(target_region, str) else (),
                route_signature,
                cross_region,
            ))

        if snapshot.get("local_navigation"):
            for _region_id, region in regions.items():
                key = region.get("semantic_key")
                if not isinstance(key, str) or not bool(region.get("explorable")) or self._blacklisted(key, now):
                    continue
                duration = int(round(self._rng.uniform(*self.config.explore_range_s) * 1000.0))
                entry_id = region.get("entry_anchor_id")
                route_ids = paths.get(entry_id) if isinstance(entry_id, int) else None
                children: list[str] = []
                nodes: list[dict[str, Any]] = []
                if route_ids:
                    for index, anchor_id in enumerate(route_ids[1:]):
                        anchor = anchors.get(anchor_id)
                        target = None if anchor is None else anchor.get("semantic_key")
                        if isinstance(target, str):
                            node_id = f"route{index}"
                            children.append(node_id)
                            nodes.append({"id": node_id, "type": "navigate", "target_key": target})
                route_keys = tuple(
                    str(node["target_key"])
                    for node in nodes
                    if node.get("type") == "navigate" and isinstance(node.get("target_key"), str)
                )
                cross_region = bool(
                    isinstance(current_region, str) and key != current_region
                )
                if cross_region and now - self._last_cross_region_at < 180.0:
                    continue
                route_signature = self._canonical_route_signature(
                    str(nearest_key) if isinstance(nearest_key, str) else None,
                    route_keys,
                )
                admitted, _override = self._route_admission(
                    route_signature,
                    key,
                    now,
                    allow_interest_override=False,
                )
                if not admitted:
                    continue
                children.append("explore")
                nodes.append({
                    "id": "explore",
                    "type": "explore",
                    "region_key": key,
                    "duration_ms": duration,
                    "strategy": "unvisited",
                })
                graph = {
                    "entry": "root",
                    "nodes": [{"id": "root", "type": "sequence", "children": children}, *nodes],
                }
                score = self._candidate_score(key, key, current_region)
                candidates.append((
                    score,
                    graph,
                    "explore",
                    (key,),
                    (key,),
                    route_signature,
                    cross_region,
                ))
        return candidates

    def _candidate_score(self, key: str, region: Any, current_region: Any) -> float:
        recent = list(self._recent_targets)
        penalty = 0.0
        if key in recent:
            # 越近期访问的目标惩罚越大，较早记录会自然衰减。
            penalty = 4.0 / (1.0 + recent[::-1].index(key))
        local_bonus = 1.0 if isinstance(region, str) and region == current_region else 0.0
        return 5.0 + local_bonus - penalty + self._rng.random()

    def _blacklisted(self, key: str, now: float) -> bool:
        failure = self._failures.get(key)
        return failure is not None and failure[1] > now

    def _prune_route_history(self, now: float) -> None:
        expired = [key for key, expires_at in self._route_history.items() if expires_at <= now]
        for key in expired:
            self._route_history.pop(key, None)

    @staticmethod
    def _canonical_route_signature(
        source_key: str | None,
        route_keys: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        ordered: list[str] = []
        if isinstance(source_key, str) and source_key:
            ordered.append(source_key)
        for key in route_keys:
            if key and (not ordered or ordered[-1] != key):
                ordered.append(key)
        if len(ordered) < 2:
            return None
        forward = tuple(ordered)
        reverse = tuple(reversed(ordered))
        return min(forward, reverse)

    def _route_interest_override(self, target_key: str, now: float) -> str | None:
        intent = self._intent
        if intent is None or target_key in intent.used_route_overrides:
            return None
        for item in intent.interests:
            if (
                item.get("target_key") == target_key
                and float(item.get("strength", 0.0)) >= 0.7
                and float(item.get("expires_at", 0.0)) > now
            ):
                return target_key
        return None

    def _route_admission(
        self,
        signature: tuple[str, ...] | None,
        target_key: str,
        now: float,
        *,
        allow_interest_override: bool,
    ) -> tuple[bool, str | None]:
        if signature is None:
            return True, None
        self._prune_route_history(now)
        if self._route_history.get(signature, 0.0) <= now:
            return True, None
        if allow_interest_override:
            override = self._route_interest_override(target_key, now)
            if override is not None:
                return True, override
        return False, None

    @staticmethod
    def _route_paths(
        source_id: int | None,
        anchors: Mapping[int, Mapping[str, Any]],
        edges: Mapping[int, Mapping[str, Any]],
    ) -> dict[int, list[int]]:
        if source_id is None or source_id not in anchors:
            return {}
        adjacency: dict[int, set[int]] = {}
        for edge in edges.values():
            start = edge.get("from_anchor_id")
            end = edge.get("to_anchor_id")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            adjacency.setdefault(start, set()).add(end)
            if bool(edge.get("bidirectional")):
                adjacency.setdefault(end, set()).add(start)
        paths = {source_id: [source_id]}
        queue = deque([source_id])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in paths or neighbor not in anchors:
                    continue
                paths[neighbor] = [*paths[current], neighbor]
                queue.append(neighbor)
        return paths

    @staticmethod
    def _navigate_graph(route_keys: tuple[str, ...]) -> dict[str, Any]:
        if len(route_keys) == 1:
            return {
                "entry": "route0",
                "nodes": [{"id": "route0", "type": "navigate", "target_key": route_keys[0]}],
            }
        children = [f"route{index}" for index in range(len(route_keys))]
        return {
            "entry": "root",
            "nodes": [
                {"id": "root", "type": "sequence", "children": children},
                *[
                    {"id": node_id, "type": "navigate", "target_key": target}
                    for node_id, target in zip(children, route_keys)
                ],
            ],
        }

    def _submit(
        self,
        graph: Mapping[str, Any],
        *,
        kind: str,
        targets: tuple[str, ...],
        regions: tuple[str, ...],
        movement: bool,
        now: float,
        replace_active: bool = False,
        planned_dwell_s: float = 0.0,
        intent_activity_index: int | None = None,
        decision_reason: str = "rule",
        route_signature: tuple[str, ...] | None = None,
        cross_region: bool = False,
        interest_override_target: str | None = None,
        intent_token: str | None = None,
    ) -> bool:
        # 世界协议在省略 speed_mps 时会采用 max_speed（正式世界为 2m/s），
        # 这会让日常自主行为进入 Run 动画。自主来源统一补成步行速度；只有
        # 未来明确标注了速度的必要赶路行为才保留其显式值。用户直接调用的
        # 显式工具不经过这里，因此仍可自行选择跑速。
        prepared_graph = self._with_default_walk_speed(graph)
        result = self.adapter.plan_manager.submit(
            prepared_graph,
            replace_active=replace_active,
            origin="autonomy",
        )
        plan_id = result.get("plan_id")
        if result.get("status") != "accepted" or not isinstance(plan_id, str):
            if result.get("error") not in {"explicit_control_active", "plan_conflict"}:
                self._plans_failed += 1
                for target in targets:
                    count, _until = self._failures.get(target, (0, 0.0))
                    count += 1
                    self._failures[target] = (count, now + min(300.0, 15.0 * (2 ** (count - 1))))
            return False
        self._active = _PlanRecord(
            plan_id,
            kind,
            targets,
            regions,
            movement,
            now,
            planned_dwell_s,
            intent_activity_index,
            decision_reason,
            route_signature,
            cross_region,
            interest_override_target,
            intent_token,
        )
        self._plans_started += 1
        self._pause_reason = ""
        self._emit_telemetry(
            "activity_started",
            kind=kind,
            targets=list(targets),
            regions=list(regions),
            intent_activity_index=intent_activity_index,
            decision_reason=decision_reason,
        )
        return True

    def _with_default_walk_speed(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        movement_types = {
            "navigate", "approach", "follow", "orbit", "explore", "move_relative",
        }
        configured = float(self.config.walk_speed_mps)
        world_max = getattr(self.session, "max_speed_mps", None)
        speed = min(configured, float(world_max)) if isinstance(world_max, (int, float)) else configured
        prepared = dict(graph)
        nodes: list[Any] = []
        raw_nodes = graph.get("nodes")
        if not isinstance(raw_nodes, list):
            return prepared
        for raw_node in raw_nodes:
            if not isinstance(raw_node, Mapping):
                nodes.append(raw_node)
                continue
            node = dict(raw_node)
            if node.get("type") in movement_types and "speed_mps" not in node:
                node["speed_mps"] = speed
            nodes.append(node)
        prepared["nodes"] = nodes
        return prepared


__all__ = [
    "AutonomyDirector",
    "AutonomyStimulusProvider",
    "NoopAutonomyStimulusProvider",
]
