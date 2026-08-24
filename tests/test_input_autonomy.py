from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.autonomy import AutonomyRuntime
from neko_anyadance_body.backend.vision import OpenAICompatibleSemanticBackend
from neko_anyadance_body.backend.world_state import WorldStateStore
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.scheduler import BodyScheduler


class _Transport:
    local_port = 39501

    def __init__(self) -> None:
        self.packets: list[bytes] = []
        self.lock = threading.Lock()

    def send(self, payload: bytes, _target: tuple[str, int]) -> None:
        with self.lock:
            self.packets.append(payload)

    def close(self) -> None:
        return


def _wait(predicate, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


class ControllerInputTests(unittest.TestCase):
    def test_overlay_is_latest_wins_and_auto_releases(self) -> None:
        transport = _Transport()
        scheduler = BodyScheduler(PluginConfig(), transport=transport)
        scheduler.start()
        try:
            self.assertTrue(scheduler.submit("enable")["accepted"])
            _wait(lambda: scheduler.snapshot()["state"] == "idle")
            self.assertTrue(scheduler.submit("input_axes", {
                "side": "left", "x": 0.75, "y": 0.25, "duration_ms": 500,
            })["accepted"])
            self.assertTrue(scheduler.submit("input_button", {
                "side": "right", "button": "trigger", "pressed": True,
                "value": 1.0, "hold_ms": 100,
            })["accepted"])
            _wait(lambda: scheduler.snapshot()["controller_input"]["axes"].get("left", {}).get("x") == 0.75)
            with transport.lock:
                frame = json.loads(transport.packets[-1])
            self.assertAlmostEqual(frame["inputs"]["left_controller"]["joystick_x"], 0.75)
            self.assertTrue(frame["inputs"]["right_controller"]["trigger_click"])
            _wait(lambda: "right:trigger" not in scheduler.snapshot()["controller_input"]["buttons"])
            self.assertIn("left", scheduler.snapshot()["controller_input"]["axes"])
            self.assertTrue(scheduler.submit("input_release", {"side": "all"})["accepted"])
            _wait(lambda: not scheduler.snapshot()["controller_input"]["axes"])
        finally:
            scheduler.shutdown()


class AutonomyAndWorldTests(unittest.TestCase):
    def test_complete_goal_keeps_arm_but_clears_goal_and_releases_inputs(self) -> None:
        released: list[str] = []
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: released.append("release"),
            session_ttl_s=600.0,
        )
        runtime.arm()
        self.assertTrue(runtime.submit_goal(
            "find the npc", "explore", selector={"semantic_type": "npc"}
        )["accepted"])

        completed = runtime.complete_goal("explore_duration_exhausted")

        self.assertTrue(completed["armed"])
        self.assertEqual(completed["state"], "armed")
        self.assertIsNone(completed["goal"])
        self.assertEqual(completed["reason"], "explore_duration_exhausted")
        self.assertEqual(released, ["release"])

    def test_autonomy_requires_arm_and_ttl_releases(self) -> None:
        now = [0.0]
        released: list[str] = []
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: released.append("release"),
            clock=lambda: now[0],
            session_ttl_s=60.0,
        )
        self.assertFalse(runtime.submit_goal("find a portal")["accepted"])
        self.assertTrue(runtime.arm()["armed"])
        self.assertTrue(runtime.submit_goal("find a portal")["accepted"])
        now[0] = 61.0
        self.assertFalse(runtime.snapshot()["armed"])
        self.assertIn("release", released)

    def test_degraded_recovers_when_the_target_comes_back_into_view(self) -> None:
        """看不见了要降级，看回来了必须恢复——degraded 不能是死状态。

        world.available 只表示「此刻有没有没过期的实体」，而人的 TTL 是 1.5 秒，
        检测漏一帧半就会掉一次。少了恢复边的话，一次瞬时丢失就把会话永久锁死，
        导航器从此只报 autonomy_not_armed。实机跑真实检测时 2.5 秒就复现了。
        """
        now = [0.0]
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
            clock=lambda: now[0],
            session_ttl_s=600.0,
        )
        self.assertTrue(runtime.arm()["armed"])
        self.assertTrue(runtime.submit_goal(
            "walk to the person", "approach", "vision:person:1"
        )["accepted"])

        def world(available: bool, revision: int) -> dict:
            return {"available": available, "entities": [], "events": [],
                    "status": {"revision": revision}}

        runtime.update_world(world(True, 1))
        self.assertEqual(runtime.snapshot()["state"], "armed")

        runtime.update_world(world(False, 2))
        degraded = runtime.snapshot()
        self.assertEqual(degraded["state"], "degraded")
        self.assertEqual(degraded["reason"], "world_observation_unknown")

        runtime.update_world(world(True, 3))
        recovered = runtime.snapshot()
        self.assertEqual(recovered["state"], "armed")
        # 导航器要的是 state == "armed"，光有 armed=True 不够：degraded 也算 armed。
        self.assertTrue(recovered["armed"])
        self.assertIsNotNone(recovered["goal"])

    def test_finite_approach_stays_armed_on_a_fresh_empty_frame(self) -> None:
        """目标离开画面时要让本地重捕获计时，而不是先把整个会话降级。"""
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
            session_ttl_s=600.0,
        )
        runtime.arm()
        self.assertTrue(runtime.submit_goal(
            "过去看看", "approach_observe", "vision:person:1"
        )["accepted"])

        runtime.update_world({
            "available": False,
            "capture_active": True,
            "entities": [],
            "events": [],
            "uncertainties": ["no_recent_visual_observation"],
            "status": {"revision": 1, "last_observation_age_ms": 80},
        })

        state = runtime.snapshot()
        self.assertEqual(state["state"], "armed")
        self.assertEqual(state["reason"], "goal_reacquiring_target")

    def test_targeted_goal_requires_and_preserves_exact_entity_id(self) -> None:
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
            session_ttl_s=600.0,
        )
        runtime.arm()

        missing = runtime.submit_goal("walk to the person", "approach")
        self.assertFalse(missing["accepted"])
        self.assertEqual(
            missing["reason"],
            "target_id is required for targeted autonomy goals",
        )

        accepted = runtime.submit_goal(
            "walk to the person",
            "approach",
            "vision:person:7",
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["goal"]["target_id"], "vision:person:7")

        finite = runtime.submit_goal(
            "walk over and look",
            "approach_observe",
            "vision:person:7",
            constraints={"settle_seconds": 0.4, "observe_seconds": 1.2},
        )
        self.assertTrue(finite["accepted"])
        self.assertEqual(finite["goal"]["kind"], "approach_observe")
        self.assertEqual(finite["goal"]["constraints"]["observe_seconds"], 1.2)

    def test_explore_goal_preserves_selector_constraints_and_revision(self) -> None:
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
            session_ttl_s=600.0,
        )
        runtime.arm()
        runtime.update_world({
            "available": True,
            "entities": [],
            "events": [],
            "status": {"revision": 12},
        })

        accepted = runtime.submit_goal(
            "find the npc",
            "explore",
            selector={"semantic_type": "npc", "min_confidence": 0.7},
            constraints={"max_duration_s": 120, "max_scan_turns": 8, "max_forward_axis": 0.4},
            based_on_revision=12,
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["goal"]["selector"]["semantic_type"], "npc")
        self.assertEqual(accepted["goal"]["constraints"]["max_scan_turns"], 8)
        self.assertEqual(accepted["goal"]["based_on_revision"], 12)

    def test_depart_is_local_but_wander_requires_an_llm_planned_step(self) -> None:
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
        )
        runtime.arm()

        depart = runtime.submit_goal(
            "离开这里",
            "depart",
            constraints={"max_duration_s": 2.0, "max_forward_axis": 0.35},
        )
        self.assertTrue(depart["accepted"], depart)
        self.assertIsNone(depart["goal"]["target_id"])

        missing_direction = runtime.submit_goal(
            "随便逛逛",
            "wander",
            constraints={"max_duration_s": 2.0, "max_forward_axis": 0.45},
        )
        self.assertFalse(missing_direction["accepted"])
        self.assertIn("turn_deg", missing_direction["reason"])

        too_long = runtime.submit_goal(
            "随便逛逛",
            "wander",
            constraints={"turn_deg": -30.0, "max_duration_s": 4.0},
        )
        self.assertFalse(too_long["accepted"])

        wander = runtime.submit_goal(
            "往右前方逛逛",
            "wander",
            constraints={"turn_deg": -30.0, "max_duration_s": 2.0, "max_forward_axis": 0.45},
        )
        self.assertTrue(wander["accepted"], wander)
        self.assertIsNone(wander["goal"]["selector"])
        self.assertEqual(wander["goal"]["constraints"]["turn_deg"], -30.0)

    def test_fresh_empty_observation_keeps_selector_explore_armed(self) -> None:
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
            session_ttl_s=600.0,
        )
        runtime.arm()
        self.assertTrue(runtime.submit_goal(
            "find the npc",
            "explore",
            selector={"semantic_type": "npc"},
        )["accepted"])

        runtime.update_world({
            "available": False,
            "entities": [],
            "events": [],
            "uncertainties": ["no_recent_visual_observation", "depth_unavailable"],
            "status": {"revision": 1, "last_observation_age_ms": 20},
        })

        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["state"], "armed")
        self.assertEqual(snapshot["reason"], "goal_waiting_for_explorer")

    def test_goal_rejects_unknown_fields_and_future_revision(self) -> None:
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
            session_ttl_s=600.0,
        )
        runtime.arm()
        runtime.update_world({
            "available": True,
            "entities": [],
            "events": [],
            "status": {"revision": 4},
        })

        invalid = runtime.submit_goal("find", selector={"semantic_type": "npc", "prompt": "x"})
        self.assertFalse(invalid["accepted"])
        self.assertIn("unsupported fields", invalid["reason"])
        future = runtime.submit_goal("find", based_on_revision=5)
        self.assertFalse(future["accepted"])
        self.assertEqual(future["reason"], "based_on_revision is newer than current world revision")

    def test_degraded_does_not_resurrect_a_disarmed_session(self) -> None:
        """恢复边不能把已经 disarm 的会话弄活。

        这里真正拦住它的是 disarm 清掉了 goal（goal is None 会提前返回），不是状态
        守卫——把守卫改成 `if False` 这个测试照样过。写清楚是为了别人改 goal 检查时
        知道这条断言依赖的是什么。
        """
        now = [0.0]
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: None,
            clock=lambda: now[0],
            session_ttl_s=600.0,
        )
        runtime.arm()
        runtime.submit_goal("walk to the person", "approach", "vision:person:1")
        runtime.disarm("manual_disarm")
        self.assertIsNone(runtime.snapshot()["goal"])

        runtime.update_world({"available": True, "entities": [], "events": [],
                              "status": {"revision": 9}})
        after = runtime.snapshot()
        self.assertEqual(after["state"], "disarmed")
        self.assertFalse(after["armed"])

    def test_world_delta_revision_and_player_memory_boundary(self) -> None:
        store = WorldStateStore(clock=lambda: 1.0)
        first = store.delta(wait_ms=0)
        self.assertFalse(first["changed"])
        store.ingest(
            entities=[
                {"id": "world:cup", "label": "cup", "source": "vision"},
                {"id": "vrchat:player:user", "label": "player", "source": "vrchat_log"},
            ],
            events=[{"type": "chat_message", "data": {"text": "secret"}}],
            source="vision",
        )
        delta = store.delta(first["revision"], wait_ms=0)
        self.assertTrue(delta["changed"])
        self.assertGreater(delta["revision"], first["revision"])
        self.assertIn("world:cup", [item["id"] for item in delta["changes"]["entities"]])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.json"
            persisted = WorldStateStore(
                clock=lambda: 1.0,
                persistence_path=path,
                persist_world=True,
                persist_players=False,
            )
            persisted.ingest(
                entities=[
                    {"id": "world:cup", "label": "cup", "source": "vision"},
                    {"id": "vrchat:player:user", "label": "player", "source": "vrchat_log"},
                ],
                events=[{"type": "chat_message", "data": {"text": "secret"}}],
                source="vision",
            )
            loaded = WorldStateStore(
                clock=lambda: 1.0,
                persistence_path=path,
                persist_world=True,
                persist_players=False,
            ).snapshot()
            self.assertEqual([item["id"] for item in loaded["entities"]], ["world:cup"])
            self.assertEqual(loaded["events"], [])

    def test_semantic_backend_structures_json_and_enforces_rate_limit(self) -> None:
        class _Response:
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": json.dumps({
                        "entities": [{"id": "world:door", "label": "door", "confidence": 0.8}],
                        "events": [],
                        "uncertainties": ["depth_unknown"],
                    })}}],
                }).encode("utf-8")

        backend = OpenAICompatibleSemanticBackend(
            endpoint="http://127.0.0.1/v1/chat/completions",
            model="test",
            max_per_minute=1,
            request_fn=lambda _request, _timeout: _Response(),
        )
        observation = backend.observe(b"jpeg", world={}, now=1.0)
        self.assertEqual(observation.entities[0]["id"], "world:door")
        self.assertEqual(observation.uncertainties, ("depth_unknown",))
        with self.assertRaises(RuntimeError):
            backend.observe(b"jpeg", world={}, now=1.1)


if __name__ == "__main__":
    unittest.main()
