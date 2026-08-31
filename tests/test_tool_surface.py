"""固定 npc.* 工具面、能力门和本地驱动锁测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime import (
    YuiDriverLease,
    YuiDriverLeaseError,
    YuiSemanticAdapter,
    YuiSessionState,
    YuiToolSurface,
)
from yui_npc_controller.runtime.behavior_plan import NODE_TYPES
from yui_npc_controller.runtime.yui_transport import YuiCommandOutcome


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def send_command(self, command, parameters=(0, 0, 0, 0, 0, 0)):
        values = tuple(parameters)
        self.calls.append((command, values))
        kind = None
        if command == "GOTO_XZ":
            kind = "goto"
        elif command == "PLAY_ANIM":
            kind = "action"
        elif command in {"LOOK_AT", "LOOK_AT_XYZ"}:
            kind = "look"
        elif command == "SET_EXPRESSION":
            kind = "expression"
        elif command == "SET_MODE" and values[3] in {1, 3}:
            kind = "follow" if values[3] == 1 else "wander"
        return YuiCommandOutcome(
            status="accepted" if kind else "succeeded",
            kind=kind,
            wire_sequence=10,
            request_hash="ABCD",
            operation_id=f"1193046:10:ABCD" if kind else None,
            error=None,
            detail=None,
            ack_replayed=False,
        )

    def send_estop(self):
        from yui_npc_controller.runtime.yui_protocol import encode_command

        return encode_command("ESTOP", 12)


class ToolSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = YuiSessionState()
        self.session.session = 1193046
        self.session.control_state = "safe_idle"
        self.transport = RecordingTransport()
        self.adapter = YuiSemanticAdapter(self.transport, self.session)
        self.surface = YuiToolSurface(
            self.adapter,
            self.session,
            free_coordinate_navigation=True,
            enable_wander_tool=True,
        )

    def tearDown(self) -> None:
        self.adapter.close()

    def _names(self) -> set[str]:
        return {definition.name for definition in self.surface.definitions()}

    def _ready(self) -> None:
        self.session.control_state = "external"
        self.session.capabilities = (
            "goto",
            "navmesh",
            "follow",
            "wander",
            "actions",
            "expressions",
            "text_preset",
            "text_utf8",
            "ray_scan",
            "touch",
            "player_pose",
            "snapshot",
            "social_signals",
            "anchors",
            "operation_lifecycle",
        )
        self.session.wire_bounds = (-12.0, -1.0, -22.0, 12.0, 5.0, 22.0)
        self.session.activity_bounds = (-10.0, 0.0, -20.0, 10.0, 4.0, 20.0)
        self.session.max_speed_mps = 2.0
        self.session.players[3] = {"slot": 3, "pid": 99, "name": "不应默认泄露"}
        self.session.catalogs["anchor"][0] = {
            "id": 0,
            "semantic_key": "spawn",
            "pos": [0.0, 0.0, 0.0],
            "has_yaw": False,
        }
        self.session.catalogs["action"][0] = {
            "id": 0,
            "semantic_key": "cat_pose",
            "target_required": "none",
            "loopable": False,
        }
        self.session.catalogs["expression"][0] = {
            "id": 0,
            "semantic_key": "idle",
        }

    def _advanced_v13(self) -> dict[str, object]:
        self._ready()
        self.surface.free_coordinate_navigation = False
        self.surface.enable_wander_tool = False
        self.session.spec_version = "1.3"
        self.session.capabilities += (
            "world_map",
            "semantic_navigation",
            "region_localization",
            "local_navigation",
        )
        self.session.catalogs["anchor"][1] = {
            "id": 1,
            "semantic_key": "upper_observation",
            "pos": [0.0, 2.0, 0.0],
            "has_yaw": False,
        }
        self.session.catalogs["anchor"][2] = {
            "id": 2,
            "semantic_key": "spawn_point",
            "pos": [0.0, 0.0, 0.0],
            "has_yaw": False,
        }
        self.session.catalogs["region"][0] = {
            "id": 0,
            "semantic_key": "ground_floor",
            "entry_anchor_id": 2,
            "explorable": True,
        }
        self.session.catalogs["entity"][0] = {
            "id": 0,
            "semantic_key": "central_obstacle",
            "approach_anchor_id": 2,
            "orbitable": True,
        }
        return {item.name: item for item in self.surface.definitions()}

    def test_safe_idle_never_exposes_arm(self) -> None:
        self.assertEqual(self._names(), {"npc.observe", "npc.estop"})
        self.assertNotIn("npc.arm", self._names())
        self.assertNotIn("npc.clear_estop", self._names())

    def test_complete_surface_uses_only_frozen_model_tool_names(self) -> None:
        self._ready()
        self.assertEqual(
            self._names(),
            {
                "npc.observe",
                "npc.estop",
                "npc.go_to",
                "npc.go_to_xyz",
                "npc.follow",
                "npc.look_at",
                "npc.act",
                "npc.set_expression",
                "npc.say",
                "npc.stop",
                "npc.wander",
            },
        )
        for forbidden in ("connect", "status", "disconnect", "snapshot", "ray_scan", "wait_operation"):
            self.assertTrue(all(forbidden not in name for name in self._names()))

    def test_v12_tools_never_expose_wire_or_coordinate_fields(self) -> None:
        """v1.2 编排工具只接受语义参数；坐标必须留在确定性代码里。"""
        self._ready()
        self.session.spec_version = "1.2"
        self.session.capabilities += ("world_map", "semantic_navigation")
        self.session.catalogs["region"][0] = {"id": 0, "semantic_key": "ground", "entry_anchor_id": 0}
        self.session.catalogs["entity"][0] = {
            "id": 0, "semantic_key": "pillar", "approach_anchor_id": 0,
            "orbitable": True, "orbit_min_radius": 1.0, "orbit_max_radius": 3.0,
        }
        definitions = {item.name: item for item in self.surface.definitions()}
        for name in ("npc.navigate", "npc.orbit", "npc.explore", "npc.execute_plan", "npc.plan_cancel"):
            properties = set(definitions[name].input_schema.get("properties", {}))
            self.assertEqual(properties & {"x", "y", "z", "yaw", "pos"}, set(), name)
        self.assertNotIn("npc.plan_step", definitions)

    def test_v13_adds_relative_move_only_when_local_navigation_is_published(self) -> None:
        self._ready()
        self.session.spec_version = "1.3"
        self.session.capabilities += ("world_map", "semantic_navigation", "region_localization")
        self.session.catalogs["region"][0] = {
            "id": 0, "semantic_key": "ground_floor", "entry_anchor_id": 0, "explorable": True,
        }
        self.assertNotIn("npc.move_relative", self._names())
        self.session.capabilities += ("local_navigation",)
        definitions = {item.name: item for item in self.surface.definitions()}
        self.assertIn("npc.move_relative", definitions)
        properties = set(definitions["npc.move_relative"].input_schema["properties"])
        self.assertEqual(properties & {"x", "y", "z", "yaw", "pos"}, set())
        self.assertEqual(
            {"bearing_deg", "distance_m", "speed_mps", "face_travel", "allow_shorter", "replace_active"},
            properties,
        )

    def test_default_full_v13_surface_matches_neko_fast_model_contract(self) -> None:
        """完整世界只暴露规范工具；宿主入口、恢复入口和坐标工具必须隐藏。"""
        self._ready()
        self.surface.free_coordinate_navigation = False
        self.surface.enable_wander_tool = False
        self.session.spec_version = "1.3"
        self.session.capabilities += (
            "world_map",
            "semantic_navigation",
            "region_localization",
            "local_navigation",
        )
        self.session.catalogs["region"][0] = {
            "id": 0,
            "semantic_key": "ground_floor",
            "entry_anchor_id": 0,
            "explorable": True,
        }
        self.session.catalogs["entity"][0] = {
            "id": 0,
            "semantic_key": "central_obstacle",
            "approach_anchor_id": 0,
            "orbitable": True,
        }
        self.assertEqual(
            self._names(),
            {
                "npc.observe",
                "npc.estop",
                "npc.world_query",
                "npc.plan_status",
                "npc.go_to",
                "npc.navigate",
                "npc.approach",
                "npc.orbit",
                "npc.explore",
                "npc.execute_plan",
                "npc.plan_cancel",
                "npc.move_relative",
                "npc.follow",
                "npc.look_at",
                "npc.act",
                "npc.set_expression",
                "npc.say",
                "npc.stop",
            },
        )
        for hidden in {
            "npc.arm",
            "npc.clear_estop",
            "npc.go_to_xyz",
            "npc.wander",
            "npc.connect",
            "npc.status",
            "npc.disconnect",
            "npc.snapshot",
            "npc.ray_scan",
            "npc.wait_operation",
        }:
            self.assertNotIn(hidden, self._names())

    def test_realtime_queries_and_execution_tools_forbid_language_only_success(self) -> None:
        definitions = self._advanced_v13()

        self.assertIn("本轮必须先调用本工具", definitions["npc.observe"].description)
        self.assertIn("本轮必须调用本工具", definitions["npc.world_query"].description)
        for name in {
            "npc.navigate",
            "npc.approach",
            "npc.orbit",
            "npc.explore",
            "npc.execute_plan",
            "npc.move_relative",
            "npc.follow",
            "npc.look_at",
            "npc.act",
            "npc.set_expression",
            "npc.say",
            "npc.stop",
            "npc.estop",
        }:
            description = definitions[name].description
            self.assertIn("必须在回复前调用本工具", description, name)
            self.assertIn("未取得工具返回不得承诺", description, name)
            self.assertIn("accepted 仅表示已受理", description, name)
            self.assertIn("只供内部追踪", description, name)
            self.assertIn("禁止在面向用户的可朗读回复中输出", description, name)

    def test_execute_plan_schema_is_complete_strict_and_compact(self) -> None:
        definition = self._advanced_v13()["npc.execute_plan"]
        graph = definition.input_schema["properties"]["graph"]
        self.assertEqual(graph["required"], ["entry", "nodes"])
        self.assertFalse(graph["additionalProperties"])
        self.assertEqual(graph["properties"]["nodes"]["maxItems"], 64)
        branches = graph["properties"]["nodes"]["items"]["anyOf"]
        schema_types = {
            branch["properties"]["type"]["enum"][0]
            for branch in branches
        }
        self.assertEqual(schema_types, set(NODE_TYPES))
        self.assertTrue(all(branch["additionalProperties"] is False for branch in branches))
        self.assertIn('"type":"sequence"', definition.description)
        self.assertIn('"type":"orbit"', definition.description)

        # 工具定义需要覆盖完整白名单，但不能无界膨胀 fast 模型上下文。
        encoded = json.dumps(
            definition.as_mcp_tool(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertLess(len(encoded), 16_000)

    def test_v12_execute_plan_schema_hides_v13_relative_move(self) -> None:
        self._ready()
        self.session.spec_version = "1.2"
        self.session.capabilities += ("world_map", "semantic_navigation")
        definitions = {item.name: item for item in self.surface.definitions()}
        graph = definitions["npc.execute_plan"].input_schema["properties"]["graph"]
        branches = graph["properties"]["nodes"]["items"]["anyOf"]
        schema_types = {
            branch["properties"]["type"]["enum"][0]
            for branch in branches
        }

        self.assertNotIn("move_relative", schema_types)
        result = self.surface.call(
            "npc.execute_plan",
            {
                "graph": {
                    "entry": "move",
                    "nodes": [
                        {
                            "id": "move",
                            "type": "move_relative",
                            "bearing_deg": 0.0,
                            "distance_m": 1.0,
                        }
                    ],
                }
            },
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "behavior_graph_invalid")
        self.assertIn("type 不在允许值中", result["detail"])

    def test_execute_plan_accepts_documented_sequence_shape(self) -> None:
        self._advanced_v13()
        result = self.surface.call(
            "npc.execute_plan",
            {
                "graph": {
                    "entry": "root",
                    "nodes": [
                        {
                            "id": "root",
                            "type": "sequence",
                            "children": ["orbit", "upper", "home"],
                        },
                        {
                            "id": "orbit",
                            "type": "orbit",
                            "target_key": "central_obstacle",
                            "laps": 1,
                            "direction": "cw",
                        },
                        {
                            "id": "upper",
                            "type": "navigate",
                            "target_key": "upper_observation",
                        },
                        {
                            "id": "home",
                            "type": "navigate",
                            "target_key": "spawn_point",
                        },
                    ],
                }
            },
        )
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["plan_id"].startswith("plan-"))

    def test_execute_plan_reports_discriminated_field_errors(self) -> None:
        self._advanced_v13()
        missing_target = self.surface.call(
            "npc.execute_plan",
            {
                "graph": {
                    "entry": "go",
                    "nodes": [{"id": "go", "type": "navigate"}],
                }
            },
        )
        self.assertEqual(missing_target["status"], "failed")
        self.assertEqual(missing_target["error"], "behavior_graph_invalid")
        self.assertEqual(
            missing_target["detail"],
            "arguments.graph.nodes[0].target_key 为必填字段",
        )

        bad_reference = self.surface.call(
            "npc.execute_plan",
            {
                "graph": {
                    "entry": "root",
                    "nodes": [
                        {"id": "root", "type": "sequence", "children": ["missing"]}
                    ],
                }
            },
        )
        self.assertEqual(bad_reference["error"], "behavior_graph_invalid")
        self.assertIn("引用了不存在的节点 missing", bad_reference["detail"])

    def test_operation_tools_and_free_coordinates_are_strictly_hidden(self) -> None:
        self._ready()
        self.surface.free_coordinate_navigation = False
        self.assertNotIn("npc.go_to_xyz", self._names())
        self.session.capabilities = tuple(
            item for item in self.session.capabilities if item != "operation_lifecycle"
        )
        names = self._names()
        for name in (
            "npc.go_to",
            "npc.go_to_xyz",
            "npc.follow",
            "npc.look_at",
            "npc.act",
            "npc.set_expression",
            "npc.wander",
        ):
            self.assertNotIn(name, names)

    def test_expression_coordinate_look_and_wander_expand_deterministically(self) -> None:
        self._ready()
        expression = self.surface.call(
            "npc.set_expression",
            {"expression_key": "idle", "duration_ms": 1000},
        )
        self.assertEqual(expression["status"], "accepted")
        self.assertEqual(self.transport.calls[-1], ("SET_EXPRESSION", (0, 0, 0, 0, 127, 1)))
        look = self.surface.call(
            "npc.look_at",
            {"x": 0.0, "y": 2.0, "z": 0.0, "duration_ms": 1000},
        )
        self.assertEqual(look["status"], "accepted")
        self.assertEqual(self.transport.calls[-1][0], "LOOK_AT_XYZ")
        wander = self.surface.call("npc.wander", {})
        self.assertEqual(wander["status"], "accepted")
        self.assertEqual(self.transport.calls[-1], ("SET_MODE", (0, 0, 0, 3, 0, 0)))

    def test_schema_is_enforced_before_any_midi_is_sent(self) -> None:
        self._ready()
        before = len(self.transport.calls)
        invalid_wander = self.surface.call("npc.wander", {"distance": 1})
        self.assertEqual(invalid_wander["status"], "failed")
        self.assertEqual(invalid_wander["error"], "invalid_arguments")
        self.assertFalse(invalid_wander["midi_sent"])
        self.assertEqual(len(self.transport.calls), before)

        invalid_look = self.surface.call(
            "npc.look_at",
            {"player_slot": 3, "x": 0.0, "y": 1.0, "z": 0.0, "duration_ms": 1000},
        )
        self.assertEqual(invalid_look["error"], "invalid_arguments")
        self.assertEqual(len(self.transport.calls), before)

        missing_text = self.surface.call("npc.say", {})
        self.assertEqual(missing_text["error"], "invalid_arguments")
        self.assertEqual(len(self.transport.calls), before)

    def test_observe_omits_player_names_by_default(self) -> None:
        self._ready()
        observed = self.surface.call("npc.observe", {})
        self.assertNotIn("name", observed["players"][0])


class DriverLeaseTests(unittest.TestCase):
    def test_second_local_controller_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "driver.lock"
            first = YuiDriverLease("NEKO_MIDI", lock_path=path).acquire()
            second = YuiDriverLease("NEKO_MIDI", lock_path=path)
            try:
                with self.assertRaises(YuiDriverLeaseError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()


if __name__ == "__main__":
    unittest.main()
