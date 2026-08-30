"""固定 npc.* 工具面、能力门和本地驱动锁测试。"""

from __future__ import annotations

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
            host_arm_authorized=True,
            free_coordinate_navigation=True,
            enable_wander_tool=True,
        )

    def tearDown(self) -> None:
        self.adapter.close()

    def _names(self) -> set[str]:
        return {definition.name for definition in self.surface.definitions()}

    def _ready(self) -> None:
        self.session.control_state = "external"
        self.session.set_host_arm_authorized(True)
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

    def test_arm_requires_current_session_host_authorization(self) -> None:
        self.assertEqual(self._names(), {"npc.observe", "npc.estop"})
        self.session.set_host_arm_authorized(True)
        self.assertEqual(self._names(), {"npc.observe", "npc.estop", "npc.arm"})
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
