"""YUI v1.2 受限行为图与后台调度测试。"""

from __future__ import annotations

import time
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.behavior_plan import (
    BehaviorGraphCompiler,
    BehaviorGraphError,
    BehaviorPlanManager,
    single_node_graph,
)
from yui_npc_controller.runtime.yui_session import YuiSessionState


class FakeAdapter:
    def __init__(self, session: YuiSessionState) -> None:
        self.session = session
        self.calls: list[tuple[str, object]] = []
        self.navigate_results: list[str] = []
        self.stops = 0

    def navigate_wire(self, target_key, *, speed_mps=None):
        self.calls.append(("navigate", target_key))
        status = self.navigate_results.pop(0) if self.navigate_results else "succeeded"
        return {"status": status, "error": None if status == "succeeded" else "no_path", "detail": None, "op_id": None}

    def orbit_wire(self, target_key, **kwargs):
        self.calls.append(("orbit", target_key))
        return {"status": "succeeded", "error": None, "detail": None, "op_id": None}

    def move_relative_wire(self, bearing_deg, distance_m, **kwargs):
        self.calls.append(("move_relative", (bearing_deg, distance_m, kwargs)))
        return {"status": "succeeded", "error": None, "detail": None, "op_id": None}

    def explore_region_wire(self, region_key, **kwargs):
        self.calls.append(("explore_region", (region_key, kwargs)))
        return {"status": "succeeded", "error": None, "detail": None, "op_id": None}

    def follow_wire(self, player_slot, *, speed_mps=None):
        self.calls.append(("follow", player_slot))
        return {"status": "accepted", "op_id": "follow-op", "error": None, "detail": None}

    def look_at(self, **kwargs):
        self.calls.append(("look_at", kwargs.get("player_slot")))
        return {"status": "succeeded", "error": None, "detail": None}

    def clear_look_wire(self):
        self.calls.append(("clear_look", None))
        return {"status": "succeeded", "error": None, "detail": None}

    def act(self, action_key, **kwargs):
        self.calls.append(("act", action_key))
        return {"status": "succeeded", "error": None, "detail": None}

    def set_expression(self, expression_key, duration_ms):
        self.calls.append(("expression", expression_key))
        return {"status": "succeeded", "error": None, "detail": None}

    def say(self, **kwargs):
        self.calls.append(("say", kwargs["text"]))
        return {"status": "succeeded", "error": None, "detail": None}

    def stop(self, scope="all", *, _from_plan=False):
        self.calls.append(("stop", scope))
        return {"status": "succeeded", "error": None, "detail": None}

    def _stop_plan_domains(self, domains):
        self.stops += 1
        return True

    def request_snapshot_evidence(self):
        self.calls.append(("snapshot", None))
        return {"status": "succeeded"}


def _graph(*nodes, entry="root"):
    return {"entry": entry, "nodes": list(nodes)}


class BehaviorPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = YuiSessionState()
        self.session.session = 1193046
        self.session.spec_version = "1.2"
        self.session.control_state = "external"
        self.session.catalog_revision = 3
        self.session.driver_pid = 1
        self.session.capabilities = (
            "goto", "navmesh", "anchors", "operation_lifecycle", "world_map", "semantic_navigation",
        )
        self.session.max_speed_mps = 2.0
        self.session.catalogs["anchor"][0] = {"id": 0, "semantic_key": "plaza", "region_key": "ground", "pos": [0, 0, 0]}
        self.session.catalogs["region"][0] = {"id": 0, "semantic_key": "ground", "entry_anchor_id": 0}
        self.session.catalogs["entity"][0] = {
            "id": 0, "semantic_key": "pillar", "approach_anchor_id": 0,
            "orbitable": True, "orbit_min_radius": 1.0, "orbit_max_radius": 3.0,
        }
        self.adapter = FakeAdapter(self.session)
        self.manager = BehaviorPlanManager(self.adapter, self.session)

    def tearDown(self) -> None:
        self.manager.close()

    def _wait(self, plan_id: str, timeout: float = 1.0):
        end = time.monotonic() + timeout
        result = self.manager.status(plan_id)
        while result["status"] not in {"succeeded", "failed", "cancelled", "unknown"} and time.monotonic() < end:
            time.sleep(0.01)
            result = self.manager.status(plan_id)
        return result

    def test_sequence_runs_in_background_without_plan_step(self) -> None:
        result = self.manager.submit(_graph(
            {"id": "root", "type": "sequence", "children": ["go", "orbit"]},
            {"id": "go", "type": "navigate", "target_key": "plaza"},
            {"id": "orbit", "type": "orbit", "target_key": "pillar"},
        ))
        self.assertEqual(result["status"], "accepted")
        done = self._wait(result["plan_id"])
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(self.adapter.calls, [("navigate", "plaza"), ("orbit", "pillar")])

    def test_retry_is_explicit_and_bounded(self) -> None:
        self.adapter.navigate_results = ["failed", "succeeded"]
        result = self.manager.submit(_graph(
            {"id": "root", "type": "retry", "child": "go", "max_attempts": 2, "delay_ms": 0},
            {"id": "go", "type": "navigate", "target_key": "plaza"},
        ))
        done = self._wait(result["plan_id"])
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(len(self.adapter.calls), 2)

    def test_unknown_never_becomes_success(self) -> None:
        self.adapter.navigate_results = ["unknown", "succeeded"]
        result = self.manager.submit(_graph(
            {"id": "root", "type": "retry", "child": "go", "max_attempts": 2},
            {"id": "go", "type": "navigate", "target_key": "plaza"},
        ))
        done = self._wait(result["plan_id"])
        self.assertEqual(done["status"], "unknown")
        self.assertEqual(self.adapter.calls, [("navigate", "plaza")], "unknown 不得重试或当作成功")

    def test_replace_active_requires_explicit_flag(self) -> None:
        self.session.players[2] = {"slot": 2, "pid": 2, "d": 5.0}
        first = self.manager.submit(single_node_graph("follow", player_slot=2, duration_ms=500))
        time.sleep(0.03)
        rejected = self.manager.submit(single_node_graph("navigate", target_key="plaza"))
        self.assertEqual(rejected["error"], "plan_conflict")
        second = self.manager.submit(single_node_graph("navigate", target_key="plaza"), replace_active=True)
        self.assertEqual(second["status"], "accepted")
        self.assertGreaterEqual(self.adapter.stops, 0)
        self.assertEqual(self._wait(first["plan_id"])["status"], "cancelled")
        self.assertEqual(self._wait(second["plan_id"])["status"], "succeeded")

    def test_only_last_sixteen_plan_records_are_retained(self) -> None:
        plan_ids: list[str] = []
        for _index in range(17):
            submitted = self.manager.submit(single_node_graph("navigate", target_key="plaza"))
            plan_ids.append(submitted["plan_id"])
            self.assertEqual(self._wait(submitted["plan_id"])["status"], "succeeded")
        self.assertEqual(self.manager.status(plan_ids[0])["error"], "plan_not_found")
        self.assertEqual(self.manager.status(plan_ids[-1])["status"], "succeeded")

    def test_session_change_cancels_running_plan(self) -> None:
        result = self.manager.submit(single_node_graph("wait", duration_ms=500))
        time.sleep(0.03)
        self.session.ingest({
            "v": 1, "spec": "1.2", "session": 2, "world_id": "wrld_test", "npc": "yui",
            "log_seq": 1, "t": 1.0, "type": "sys.session", "new_session": 2,
            "driver_pid": 1, "reset": True, "estop_preserved": False,
        })
        self.assertEqual(self._wait(result["plan_id"])["status"], "cancelled")

    def test_parallel_rejects_two_movement_branches(self) -> None:
        with self.assertRaises(BehaviorGraphError):
            BehaviorGraphCompiler(self.session).compile(_graph(
                {"id": "root", "type": "parallel", "children": ["a", "b"]},
                {"id": "a", "type": "navigate", "target_key": "plaza"},
                {"id": "b", "type": "orbit", "target_key": "pillar"},
            ))

    def test_cycle_and_unbounded_repeat_are_rejected(self) -> None:
        for graph in (
            _graph({"id": "root", "type": "repeat", "child": "root", "count": 2}),
            _graph(
                {"id": "root", "type": "repeat", "child": "wait", "count": 11},
                {"id": "wait", "type": "wait", "duration_ms": 1},
            ),
        ):
            with self.subTest(graph=graph), self.assertRaises(BehaviorGraphError):
                BehaviorGraphCompiler(self.session).compile(graph)

    def test_structural_limits_are_all_enforced_by_compiler(self) -> None:
        too_many_nodes = _graph(
            {"id": "root", "type": "sequence", "children": [f"n{i}" for i in range(64)]},
            *(
                {"id": f"n{i}", "type": "wait", "duration_ms": 1}
                for i in range(64)
            ),
        )
        too_deep = _graph(
            *(
                {"id": f"n{i}", "type": "timeout", "child": f"n{i + 1}", "timeout_ms": 1}
                for i in range(8)
            ),
            {"id": "n8", "type": "wait", "duration_ms": 1},
            entry="n0",
        )
        too_many_parallel = _graph(
            {"id": "root", "type": "parallel", "children": [f"n{i}" for i in range(5)]},
            *(
                {"id": f"n{i}", "type": "wait", "duration_ms": 1}
                for i in range(5)
            ),
        )
        bad_retry = _graph(
            {"id": "root", "type": "retry", "child": "wait", "max_attempts": 4},
            {"id": "wait", "type": "wait", "duration_ms": 1},
        )
        bad_timeout = _graph(
            {"id": "root", "type": "timeout", "child": "wait", "timeout_ms": 60_001},
            {"id": "wait", "type": "wait", "duration_ms": 1},
        )
        for graph in (
            too_many_nodes,
            too_deep,
            too_many_parallel,
            bad_retry,
            bad_timeout,
        ):
            with self.subTest(graph=graph), self.assertRaises(BehaviorGraphError):
                BehaviorGraphCompiler(self.session).compile(graph)

    def test_invalid_graph_uses_model_visible_error_code(self) -> None:
        result = self.manager.submit(
            _graph({"id": "root", "type": "navigate"})
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "behavior_graph_invalid")
        self.assertIn("target_key 为必填", result["detail"])

    def test_v12_rejects_v13_move_relative_node(self) -> None:
        result = self.manager.submit(single_node_graph(
            "move_relative",
            bearing_deg=0.0,
            distance_m=1.0,
        ))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "behavior_graph_invalid")
        self.assertIn("local_navigation capability", result["detail"])
        self.assertEqual(self.adapter.calls, [])

    def test_natural_language_condition_is_rejected(self) -> None:
        for predicate in (
            {"type": "ask_llm", "prompt": "玩家看起来开心吗"},
            {
                "type": "player_present",
                "player_slot": 0,
                "prompt": "请自行判断玩家是否开心",
            },
        ):
            with self.subTest(predicate=predicate), self.assertRaises(BehaviorGraphError):
                BehaviorGraphCompiler(self.session).compile(_graph({
                    "id": "root", "type": "condition", "predicate": predicate,
                }))

    def test_v13_move_relative_and_explore_are_single_wire_operations(self) -> None:
        self.session.spec_version = "1.3"
        self.session.capabilities += ("region_localization", "local_navigation")
        self.session.catalogs["region"][0]["explorable"] = True
        moved = self.manager.submit(single_node_graph(
            "move_relative",
            bearing_deg=270.0,
            distance_m=2.5,
            face_travel=False,
            allow_shorter=True,
        ))
        self.assertEqual(self._wait(moved["plan_id"])["status"], "succeeded")
        explored = self.manager.submit(single_node_graph(
            "explore", region_key="ground", duration_ms=20_000, strategy="patrol",
        ))
        self.assertEqual(self._wait(explored["plan_id"])["status"], "succeeded")
        self.assertEqual([name for name, _value in self.adapter.calls], ["move_relative", "explore_region"])

    def test_v13_move_relative_rejects_unknown_fields_and_bad_distance(self) -> None:
        self.session.spec_version = "1.3"
        self.session.capabilities += ("local_navigation",)
        for graph in (
            single_node_graph("move_relative", bearing_deg=0.0, distance_m=0.2),
            single_node_graph("move_relative", bearing_deg=0.0, distance_m=1.0, x=4.0),
        ):
            with self.subTest(graph=graph), self.assertRaises(BehaviorGraphError):
                BehaviorGraphCompiler(self.session).compile(graph)


if __name__ == "__main__":
    unittest.main()
