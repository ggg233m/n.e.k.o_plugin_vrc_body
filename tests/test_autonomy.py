"""宿主常驻自主循环、语义路线和控制仲裁测试。"""

from __future__ import annotations

import asyncio
from collections import deque
import random
import threading
import time
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.autonomy import (
    AutonomyDirector,
    NoopAutonomyStimulusProvider,
)
from yui_npc_controller.runtime.chat_context import ChatContextUpdate
from yui_npc_controller.runtime.behavior_plan import BehaviorGraphCompiler
from yui_npc_controller.runtime.config import YuiAutonomyConfig


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSession:
    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._listeners = []
        self.session = 9
        self.discovery_ready = True
        self.control_state = "external"
        self.estop = False
        self.operation_lifecycle = True
        self.local_navigation = True
        self.max_speed_mps = 2.0
        self.capabilities = (
            "goto",
            "navmesh",
            "anchors",
            "world_map",
            "semantic_navigation",
            "operation_lifecycle",
            "local_navigation",
        )
        self.players = {2: {"slot": 2, "d": 3.0}}
        self.operations = {}
        self.npc_state = {
            "active_ops": [],
            "location": {
                "localized": True,
                "region_key": "a",
                "nearest_anchor": {"semantic_key": "a0", "d": 0.1, "brg": 0},
            },
        }
        self.catalogs = {
            "anchor": {
                0: {"id": 0, "semantic_key": "a0", "region_key": "a", "tags": ["quiet"]},
                1: {"id": 1, "semantic_key": "b0", "region_key": "b", "tags": ["view"]},
                2: {"id": 2, "semantic_key": "c0", "region_key": "c", "tags": ["social"]},
            },
            "region": {
                0: {"id": 0, "semantic_key": "a", "entry_anchor_id": 0, "explorable": True},
                1: {"id": 1, "semantic_key": "b", "entry_anchor_id": 1, "explorable": True},
                2: {"id": 2, "semantic_key": "c", "entry_anchor_id": 2, "explorable": True},
            },
            "route_edge": {
                0: {"from_anchor_id": 0, "to_anchor_id": 1, "bidirectional": True},
                1: {"from_anchor_id": 1, "to_anchor_id": 2, "bidirectional": True},
            },
            "action": {
                0: {"semantic_key": "greet"},
                1: {"semantic_key": "greet_wave"},
                2: {"semantic_key": "listen"},
                3: {"semantic_key": "agree_nod"},
            },
            "entity": {},
        }

    def add_event_listener(self, listener):
        self._listeners.append(listener)

    def remove_event_listener(self, listener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def emit(self, event):
        for listener in tuple(self._listeners):
            listener(dict(event))


class FakePlanManager:
    def __init__(self) -> None:
        self.submissions = []
        self.statuses = {}
        self.cancelled = []

    def submit(self, graph, *, replace_active=False, origin="explicit"):
        plan_id = f"plan-{len(self.submissions) + 1}"
        self.submissions.append({
            "plan_id": plan_id,
            "graph": graph,
            "replace_active": replace_active,
            "origin": origin,
        })
        self.statuses[plan_id] = "accepted"
        return {"plan_id": plan_id, "status": "accepted", "error": None}

    def status(self, plan_id=None):
        status = self.statuses.get(plan_id, "failed")
        return {
            "plan_id": plan_id,
            "status": status,
            "error": "plan_not_found" if status == "failed" and plan_id not in self.statuses else None,
            "detail": None,
        }

    def cancel_origin(self, origin, reason, *, stop_domains=True):
        self.cancelled.append((origin, reason, stop_domains))
        for plan_id, status in list(self.statuses.items()):
            if status not in {"succeeded", "failed", "cancelled", "unknown"}:
                self.statuses[plan_id] = "cancelled"
        return True


class FakeAdapter:
    def __init__(self) -> None:
        self.plan_manager = FakePlanManager()


class FakeChatContextProvider:
    def __init__(self, updates, turns) -> None:
        self.config = type("ChatConfig", (), {"enabled": True})()
        self.updates = deque(updates)
        self.turns = list(turns)
        self.character = "然然"

    def poll(self):
        if self.updates:
            return self.updates.popleft()
        return ChatContextUpdate(False, False, "same")

    def context(self):
        return {"source": "recent_file", "untrusted": True, "turns": list(self.turns)}

    def status(self):
        return {
            "enabled": True,
            "source": "recent_file",
            "current_character": self.character,
            "file_state": "available",
            "turn_count": len(self.turns),
            "revision": "same",
        }


class AutonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.session = FakeSession()
        self.adapter = FakeAdapter()
        self.config = YuiAutonomyConfig(
            enabled=True,
            decision_interval_s=1.0,
            resume_delay_s=8.0,
            dwell_range_s=(8.0, 8.0),
            explore_range_s=(15.0, 15.0),
            social_cooldown_s=60.0,
            llm_inspiration_range_s=(180.0, 180.0),
        )
        self.director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            rng=random.Random(1),
            clock=self.clock,
        )
        # 单元测试直接驱动 tick，不启动后台线程。
        self.director._desired_running = True
        self.director._pause_reason = ""

    def tearDown(self) -> None:
        self.session.remove_event_listener(self.director._on_session_event)

    def test_autonomy_movement_defaults_to_walk_speed(self) -> None:
        graph = {
            "entry": "root",
            "nodes": [
                {"id": "root", "type": "sequence", "children": ["go", "near", "roam"]},
                {"id": "go", "type": "navigate", "target_key": "b0"},
                {"id": "near", "type": "approach", "player_slot": 2, "distance_m": 1.5},
                {"id": "roam", "type": "move_relative", "bearing_deg": 0, "distance_m": 1},
            ],
        }
        self.assertTrue(self.director._submit(
            graph,
            kind="test_walk",
            targets=("b0",),
            regions=("b",),
            movement=True,
            now=self.clock(),
        ))
        submitted = self.adapter.plan_manager.submissions[-1]["graph"]
        movement = [node for node in submitted["nodes"] if node["type"] != "sequence"]
        self.assertTrue(all(node["speed_mps"] == 1.0 for node in movement))
        self.assertNotIn("speed_mps", graph["nodes"][1])

    def test_noop_stimulus_provider_never_returns_visual_input(self) -> None:
        result = asyncio.run(NoopAutonomyStimulusProvider().get_stimulus({"x": 1}))
        self.assertIsNone(result)

    def test_first_routine_is_visible_movement(self) -> None:
        self.clock.advance(1.1)
        self.director._tick()

        submission = self.adapter.plan_manager.submissions[0]
        node_types = {node.get("type") for node in submission["graph"]["nodes"]}
        self.assertTrue({"navigate", "explore"} & node_types)
        self.assertNotEqual(self.director.status()["active_kind"], "dwell")

    def test_stale_watchdog_before_start_does_not_block_new_session(self) -> None:
        self.director._desired_running = False
        self.director._pause_reason = "not_started"

        self.session.emit({"type": "sys.watchdog"})
        self.assertEqual(self.director.status()["pause_reason"], "not_started")

        self.director._desired_running = True
        self.session.emit({"type": "sys.watchdog"})
        self.assertFalse(self.director.status()["running"])
        self.assertEqual(self.director.status()["pause_reason"], "watchdog")

    def test_startup_safe_idle_has_bounded_grace_then_pauses(self) -> None:
        self.session.control_state = "safe_idle"
        self.director._startup_grace_until = 5.0

        self.clock.advance(4.9)
        self.director._tick()
        self.assertTrue(self.director.status()["running"])

        self.clock.advance(0.2)
        self.director._tick()
        self.assertFalse(self.director.status()["running"])
        self.assertEqual(self.director.status()["pause_reason"], "watchdog")

    def test_route_planner_uses_published_edges_and_waits_for_terminal(self) -> None:
        self.director._movement_seconds = 0.0
        self.director._dwell_seconds = 10.0
        self.clock.advance(1.1)
        self.director._tick()

        self.assertEqual(len(self.adapter.plan_manager.submissions), 1)
        submission = self.adapter.plan_manager.submissions[0]
        self.assertEqual(submission["origin"], "autonomy")
        nodes = submission["graph"]["nodes"]
        route_targets = [node.get("target_key") for node in nodes if node.get("type") == "navigate"]
        self.assertTrue(route_targets or any(node.get("type") == "explore" for node in nodes))

        # accepted 不是完成；下一轮不得提交第二个计划。
        self.clock.advance(2.0)
        self.director._tick()
        self.assertEqual(len(self.adapter.plan_manager.submissions), 1)

        self.adapter.plan_manager.statuses[submission["plan_id"]] = "succeeded"
        self.clock.advance(20.0)
        self.director._tick()
        self.assertGreaterEqual(self.director.status()["plans_completed"], 1)
        self.assertTrue(
            self.director.status()["recent_targets"]
            or self.director.status()["recent_regions"]
        )

    def test_explicit_plan_preempts_and_resumes_only_after_terminal_plus_delay(self) -> None:
        self.director._movement_seconds = 0.0
        self.director._dwell_seconds = 10.0
        self.clock.advance(1.1)
        self.director._tick()
        first_count = len(self.adapter.plan_manager.submissions)

        self.director.before_explicit_tool("npc.navigate")
        self.adapter.plan_manager.statuses["explicit"] = "running"
        self.director.after_explicit_tool(
            "npc.navigate", {"status": "accepted", "plan_id": "explicit"}
        )
        self.assertIn(("autonomy", "explicit_control", True), self.adapter.plan_manager.cancelled)

        self.clock.advance(30.0)
        self.director._tick()
        self.assertEqual(len(self.adapter.plan_manager.submissions), first_count)
        self.adapter.plan_manager.statuses["explicit"] = "succeeded"
        self.director._tick()
        self.clock.advance(7.9)
        self.director._tick()
        self.assertEqual(len(self.adapter.plan_manager.submissions), first_count)
        self.clock.advance(0.2)
        self.director._tick()
        self.assertGreater(len(self.adapter.plan_manager.submissions), first_count)

    def test_stop_and_estop_require_manual_restart(self) -> None:
        self.director.before_explicit_tool("npc.stop")
        self.director.after_explicit_tool("npc.stop", {"status": "succeeded"})
        self.assertFalse(self.director.status()["running"])
        self.assertEqual(self.director.status()["pause_reason"], "explicit_stop")

        self.director._desired_running = True
        self.session.estop = True
        self.director._tick()
        self.assertFalse(self.director.status()["running"])
        self.assertEqual(self.director.status()["pause_reason"], "estop")

    def test_social_event_is_rate_limited_per_player_and_preempts_routine(self) -> None:
        self.director._movement_seconds = 0.0
        self.director._dwell_seconds = 10.0
        self.clock.advance(1.1)
        self.director._tick()
        self.session.emit({"type": "social.wave", "slot": 2})
        self.session.emit({"type": "player.touch", "slot": 2})
        self.assertEqual(self.director.status()["social_queue"], 1)

        self.clock.advance(1.1)
        self.director._tick()
        latest = self.adapter.plan_manager.submissions[-1]
        self.assertEqual(latest["replace_active"], True)
        self.assertTrue(any(node.get("type") == "approach" for node in latest["graph"]["nodes"]))
        self.assertEqual(self.director.status()["social_queue"], 0)

    def test_failed_target_enters_bounded_backoff(self) -> None:
        self.director._movement_seconds = 0.0
        self.director._dwell_seconds = 10.0
        self.clock.advance(1.1)
        self.director._tick()
        submitted = self.adapter.plan_manager.submissions[-1]
        self.adapter.plan_manager.statuses[submitted["plan_id"]] = "failed"
        self.clock.advance(2.0)
        self.director._tick()
        blacklist = self.director.status()["blacklist"]
        self.assertTrue(blacklist)
        self.assertTrue(all(0.0 < seconds <= 300.0 for seconds in blacklist.values()))

    def test_llm_fragment_waits_for_current_terminal_then_drives_exact_target(self) -> None:
        requests = []
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            inspiration_callback=lambda request: requests.append(request),
            rng=random.Random(2),
            clock=self.clock,
        )
        director._desired_running = True
        director._pause_reason = ""
        try:
            self.clock.advance(1.1)
            director._tick()
            self.assertEqual(requests[0]["reason"], "startup")
            fallback = self.adapter.plan_manager.submissions[-1]
            accepted = director.offer_intent(
                {
                    "motivation": "想去更远的社交角看看，再安静待一会儿。",
                    "mood": "quiet",
                    "activities": [
                        {"kind": "visit", "target_key": "c0", "duration_s": 8},
                        {"kind": "linger", "tags": ["quiet"], "duration_s": 10},
                    ],
                    "avoid_targets": [],
                    "ttl_s": 240,
                },
                requests[0]["request_token"],
            )
            self.assertTrue(accepted)

            # 当前规则活动仍未终态，LLM 意图不能突然改向。
            self.clock.advance(2.0)
            director._tick()
            self.assertEqual(self.adapter.plan_manager.submissions[-1], fallback)

            self.adapter.plan_manager.statuses[fallback["plan_id"]] = "succeeded"
            self.clock.advance(8.0)
            director._tick()
            intent_plan = self.adapter.plan_manager.submissions[-1]
            targets = [
                node.get("target_key")
                for node in intent_plan["graph"]["nodes"]
                if node.get("type") == "navigate"
            ]
            self.assertIn("c0", targets)
            self.assertTrue(any(
                node.get("type") == "wait" for node in intent_plan["graph"]["nodes"]
            ))
            BehaviorGraphCompiler(self.session).compile(intent_plan["graph"])
            self.assertEqual(director.status()["last_decision_reason"], "llm_exact_target")
            self.assertFalse(director.status()["fallback_active"])
        finally:
            self.session.remove_event_listener(director._on_session_event)

    def test_chat_revision_is_injected_and_requests_new_intent(self) -> None:
        requests = []
        provider = FakeChatContextProvider(
            [ChatContextUpdate(True, False, "new-revision")],
            [{"user": "最近对窗边很感兴趣", "assistant": "可以过去看看"}],
        )
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            inspiration_callback=requests.append,
            chat_context_provider=provider,  # type: ignore[arg-type]
            clock=self.clock,
        )
        director._desired_running = True
        director._pause_reason = ""
        try:
            self.clock.advance(1.1)
            director._tick()
            self.assertEqual(requests[-1]["reason"], "chat_updated")
            recent = requests[-1]["context"]["recent_conversation"]
            self.assertTrue(recent["untrusted"])
            self.assertEqual(recent["turns"][0]["user"], "最近对窗边很感兴趣")
            self.assertEqual(director.status()["chat_context"]["current_character"], "然然")
        finally:
            self.session.remove_event_listener(director._on_session_event)

    def test_character_switch_invalidates_old_intent_token(self) -> None:
        requests = []
        provider = FakeChatContextProvider(
            [ChatContextUpdate(False, False, "old")],
            [{"user": "旧角色聊天", "assistant": "旧回答"}],
        )
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            inspiration_callback=requests.append,
            chat_context_provider=provider,  # type: ignore[arg-type]
            clock=self.clock,
        )
        director._desired_running = True
        director._request_intent("startup")
        stale_token = requests[-1]["request_token"]
        provider.character = "新角色"
        provider.turns = []
        provider.updates.append(ChatContextUpdate(True, True, None))
        try:
            director._poll_chat_context()
            director._request_intent("character_changed")
            self.assertFalse(director.offer_intent({
                "motivation": "旧角色想法",
                "mood": "quiet",
                "activities": [
                    {"kind": "linger", "duration_s": 5},
                    {"kind": "linger", "duration_s": 5},
                ],
                "avoid_targets": [],
                "ttl_s": 60,
            }, stale_token))
            self.assertEqual(requests[-1]["context"]["recent_conversation"]["turns"], [])
        finally:
            self.session.remove_event_listener(director._on_session_event)

    def test_observe_and_local_roam_compile_to_internal_terminal_operations(self) -> None:
        observed = self.director._compile_intent_activity(
            {"kind": "observe", "target_key": "b0", "duration_s": 12},
            frozenset(),
            self.clock(),
        )
        self.assertIsNotNone(observed)
        observe_graph = observed[0]
        self.assertTrue(any(node.get("type") == "look_at_target" for node in observe_graph["nodes"]))
        BehaviorGraphCompiler(self.session).compile(observe_graph)

        roamed = self.director._compile_intent_activity(
            {"kind": "local_roam", "style": "meander", "duration_s": 15},
            frozenset(),
            self.clock(),
        )
        self.assertIsNotNone(roamed)
        steps = [node for node in roamed[0]["nodes"] if node.get("type") == "move_relative"]
        self.assertGreaterEqual(len(steps), 2)
        self.assertTrue(all(0.5 <= node["distance_m"] <= 1.5 for node in steps))
        BehaviorGraphCompiler(self.session).compile(roamed[0])

    def test_route_cooldown_allows_one_strong_interest_override(self) -> None:
        requests = []
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            inspiration_callback=requests.append,
            clock=self.clock,
        )
        director._desired_running = True
        director._request_intent("startup")
        signature = ("a0", "b0")
        director._route_history[signature] = self.clock() + 600.0
        self.assertTrue(director.offer_intent({
            "motivation": "又想回窗边看看",
            "mood": "curious",
            "activities": [
                {"kind": "visit", "target_key": "b0", "duration_s": 8},
                {"kind": "linger", "duration_s": 8},
            ],
            "avoid_targets": [],
            "interests": [{"target_key": "b0", "strength": 0.8, "ttl_s": 300}],
            "ttl_s": 300,
        }, requests[-1]["request_token"]))
        director._activate_pending_intent(self.clock())
        try:
            allowed, override = director._route_admission(
                signature, "b0", self.clock(), allow_interest_override=True
            )
            self.assertTrue(allowed)
            self.assertEqual(override, "b0")
            director._intent.used_route_overrides.add("b0")  # type: ignore[union-attr]
            allowed_again, override_again = director._route_admission(
                signature, "b0", self.clock(), allow_interest_override=True
            )
            self.assertFalse(allowed_again)
            self.assertIsNone(override_again)
        finally:
            self.session.remove_event_listener(director._on_session_event)

    def test_stale_intent_and_pause_cannot_reactivate_autonomy(self) -> None:
        requests = []
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            inspiration_callback=lambda request: requests.append(request),
            clock=self.clock,
        )
        director._desired_running = True
        director._request_intent("startup")
        token = requests[-1]["request_token"]
        director._request_intent("newer_event")
        value = {
            "motivation": "旧想法",
            "mood": "quiet",
            "activities": [
                {"kind": "linger", "duration_s": 5},
                {"kind": "linger", "duration_s": 5},
            ],
            "avoid_targets": [],
            "ttl_s": 60,
        }
        self.assertFalse(director.offer_intent(value, token))
        director.pause("manual_pause")
        self.assertFalse(director.offer_intent(value, requests[-1]["request_token"]))
        self.assertIsNone(director.status()["current_intent"])
        self.session.remove_event_listener(director._on_session_event)

    def test_social_event_requests_new_intent_without_waiting_for_model(self) -> None:
        requests = []
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            inspiration_callback=lambda request: requests.append(request),
            clock=self.clock,
        )
        director._desired_running = True
        self.session.emit({"type": "social.wave", "slot": 2})
        self.assertEqual(requests[-1]["reason"], "social_event")
        self.assertEqual(requests[-1]["context"]["trigger_event"]["player_slot"], 2)
        self.assertEqual(director.status()["social_queue"], 1)
        self.session.remove_event_listener(director._on_session_event)

    def test_structured_telemetry_covers_intent_queue(self) -> None:
        telemetry: list[dict[str, object]] = []
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            self.config,
            inspiration_callback=lambda _request: None,
            telemetry_callback=telemetry.append,
            clock=self.clock,
        )
        director._desired_running = True
        director._request_intent("startup")
        token = director._latest_intent_token
        self.assertIsInstance(token, str)
        self.assertTrue(director.offer_intent({
            "motivation": "想去窗边安静看看",
            "mood": "quiet",
            "activities": [
                {"type": "visit", "target_key": "b0", "duration_s": 8},
                {"type": "linger", "target_key": "b0", "duration_s": 8},
            ],
            "avoid_targets": [],
            "ttl_s": 240,
        }, token))
        events = [item["event"] for item in telemetry]
        self.assertIn("intent_requested", events)
        self.assertIn("intent_queued", events)
        self.session.remove_event_listener(director._on_session_event)

    def test_close_stops_background_thread(self) -> None:
        director = AutonomyDirector(
            self.adapter,  # type: ignore[arg-type]
            self.session,  # type: ignore[arg-type]
            YuiAutonomyConfig(enabled=True, decision_interval_s=0.01),
        )
        director.start()
        time.sleep(0.03)
        director.close()
        self.assertIsNotNone(director._thread)
        self.assertFalse(director._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
