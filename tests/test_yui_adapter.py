"""YUI 插件适配层测试。"""

from __future__ import annotations

import threading
import time
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.yui_adapter import YuiSemanticAdapter
from yui_npc_controller.runtime.yui_session import YuiSessionState
from yui_npc_controller.runtime.yui_transport import YuiCommandOutcome, YuiReliableTransport


def _outcome(status: str = "succeeded", *, error: str | None = None) -> YuiCommandOutcome:
    return YuiCommandOutcome(
        status=status,
        kind=None,
        wire_sequence=10,
        request_hash="ABCD",
        operation_id=None,
        error=error,
        detail=None,
        ack_replayed=False,
    )


class RecordingTransport:
    def __init__(self) -> None:
        self.calls = []
        self.results = []
        self.heartbeat_enabled = False
        self.heartbeat_started = False
        self.command_deadline_s = 0.5

    def send_command(self, command, parameters=(0, 0, 0, 0, 0, 0)):
        self.calls.append((command, tuple(parameters)))
        if self.results:
            return self.results.pop(0)
        kind = {
            "GOTO_XZ": "goto",
            "SET_MODE": "follow" if parameters[3] == 1 else None,
            "PLAY_ANIM": "action",
        }.get(command)
        result = _outcome("accepted" if kind else "succeeded")
        if kind is None:
            return result
        return YuiCommandOutcome(
            status="accepted",
            kind=kind,
            wire_sequence=result.wire_sequence,
            request_hash=result.request_hash,
            operation_id="1193046:10:ABCD",
            error=None,
            detail=None,
            ack_replayed=False,
        )

    def set_heartbeat_enabled(self, enabled):
        self.heartbeat_enabled = bool(enabled)

    def start_heartbeat(self):
        self.heartbeat_started = True

    def send_estop(self):
        from yui_npc_controller.runtime.yui_protocol import encode_command
        return encode_command("ESTOP", 12)


def _real_adapter(state: YuiSessionState) -> tuple[YuiSemanticAdapter, list]:
    """适配器 + 真实传输层（同步 no-op sink，ACK 超时极短）。"""
    sent: list = []
    transport = YuiReliableTransport(
        sent.append, state, ack_timeout_s=0.05, command_deadline_s=0.1,
    )
    return YuiSemanticAdapter(transport, state), sent


class YuiSemanticAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = YuiSessionState()
        self.transport = RecordingTransport()
        self.adapter = YuiSemanticAdapter(self.transport, self.state)

    def _ready(self) -> None:
        self.state.session = 1193046
        self.state.control_state = "external"
        self.state.capabilities = (
            "goto",
            "follow",
            "actions",
            "text_preset",
            "text_utf8",
            "navmesh",
            "anchors",
            "operation_lifecycle",
        )
        self.state.wire_bounds = (-12.0, -1.0, -22.0, 12.0, 5.0, 22.0)
        self.state.activity_bounds = (-10.0, 0.0, -20.0, 10.0, 4.0, 20.0)
        self.state.max_speed_mps = 2.0

    def test_connect_waits_until_declared_catalog_is_complete(self) -> None:
        result: list[dict] = []
        thread = threading.Thread(
            target=lambda: result.append(self.adapter.connect(0, session=1193046))
        )
        thread.start()
        base = {
            "v": 1,
            "spec": "1.2",
            "session": 1193046,
            "world_id": "wrld_test",
            "npc": "yui",
            "t": 0.0,
        }
        self.state.ingest({
            **base,
            "log_seq": 1,
            "type": "sys.session",
            "new_session": 1193046,
            "driver_pid": 7,
            "reset": False,
            "estop_preserved": False,
        })
        self.state.ingest({
            **base,
            "log_seq": 2,
            "type": "sys.hello",
            "world_name": "测试世界",
            "caps": ["world_map"],
            "cap_bits": 0,
            "catalog_rev": 2,
            "catalog_counts": {"entity": 1},
            "wire_bounds": [-5, -1, -5, 5, 5, 5],
            "activity_bounds": [-4, 0, -4, 4, 4, 4],
            "max_speed": 2.0,
        })
        time.sleep(0.02)
        self.assertTrue(thread.is_alive())
        self.state.ingest({
            **base,
            "log_seq": 3,
            "type": "sys.catalog",
            "kind": "entity",
            "page": 1,
            "pages": 1,
            "items": [{"id": 0, "semantic_key": "central_obstacle"}],
        })
        thread.join(timeout=1.0)
        self.assertEqual(result[0]["status"], "succeeded")
        self.assertTrue(result[0]["auto_control"])
        self.assertTrue(self.transport.heartbeat_started)

        # 真实世界会由 SET_CONTROL_MODE ACK 投影到 external；测试传输器在这里补齐。
        self.state.control_state = "external"
        repeated = self.adapter.connect(0)
        self.assertEqual(repeated["status"], "succeeded")
        self.assertTrue(repeated["already_connected"])
        self.assertFalse(repeated["midi_sent"])
        self.assertEqual(
            [command for command, _parameters in self.transport.calls],
            ["DISCOVER", "SET_CONTROL_MODE"],
            "首次连接应自动进入控制态，重复连接不得重放目录或控制命令",
        )

        self.state.control_state = "safe_idle"
        recovered = self.adapter.connect(0)
        self.assertEqual(recovered["status"], "succeeded")
        self.assertTrue(recovered["already_connected"])
        self.assertFalse(recovered["rediscovered"])
        self.assertEqual(
            [command for command, _parameters in self.transport.calls],
            ["DISCOVER", "SET_CONTROL_MODE", "SET_CONTROL_MODE"],
            "watchdog 回到 safe_idle 后只恢复控制态，不应重放目录",
        )

        self.state.control_state = "estop"
        estop_result = self.adapter.connect(0)
        self.assertEqual(estop_result["error"], "estop_latched")
        self.assertEqual(len(self.transport.calls), 3)

    def test_go_to_uses_only_published_anchor_and_world_bounds(self) -> None:
        self._ready()
        self.state.catalogs["anchor"][0] = {
            "id": 0,
            "semantic_key": "stage_center",
            "pos": [0.0, 0.0, 4.0],
            "yaw": 180.0,
            "has_yaw": True,
        }
        result = self.adapter.go_to("stage_center", speed_mps=1.0)
        self.assertEqual(result["status"], "accepted")
        command, parameters = self.transport.calls[-1]
        self.assertEqual(command, "GOTO_XZ")
        self.assertEqual(parameters, (8192, 9681, 8192, 64, 1, 0))
        missing = self.adapter.go_to("unconfigured_place")
        self.assertEqual(missing["error"], "anchor_not_found")
        self.assertEqual(len(self.transport.calls), 1)

    def test_free_coordinate_navigation_is_absent_by_default(self) -> None:
        self._ready()
        result = self.adapter.go_to_xyz(1.0, 2.0)
        self.assertEqual(result["error"], "tool_disabled")
        self.assertEqual(self.transport.calls, [])

    def test_follow_stops_before_second_command_when_target_step_fails(self) -> None:
        self._ready()
        self.state.players[3] = {"slot": 3, "pid": 7}
        self.transport.results.append(_outcome("failed", error="slot_unknown"))
        result = self.adapter.follow(3)
        self.assertEqual(result["failed_step"], "SET_TARGET")
        self.assertEqual(self.transport.calls, [("SET_TARGET", (0, 0, 0, 3, 0, 0))])

    def test_operation_tools_hidden_without_lifecycle_capability(self) -> None:
        self._ready()
        self.state.capabilities = tuple(
            capability
            for capability in self.state.capabilities
            if capability != "operation_lifecycle"
        )
        result = self.adapter.follow(0)
        self.assertEqual(result["error"], "unsupported_capability")
        self.assertEqual(self.transport.calls, [])

    def test_catalog_bad_action_id_returns_dict_not_exception(self) -> None:
        """世界目录上报越界 action id（违反 PLAY_ANIM P3 range [0,126]）时返回 dict，不抛异常。"""
        state = YuiSessionState()
        state.session = 1193046
        state.control_state = "external"
        state.capabilities = ("actions", "operation_lifecycle")
        # action id=127 违反冻结常量 PLAY_ANIM range.P3=[0,126]
        state.catalogs["action"][0] = {"id": 127, "semantic_key": "bad_action", "loopable": False}
        adapter, sent = _real_adapter(state)
        result = adapter.act("bad_action")
        self.assertIsInstance(result, dict, "必须返回 dict，不得抛异常")
        self.assertFalse(result.get("midi_sent", True), "编码失败时不应有 MIDI 发出")
        self.assertEqual(sent, [], "编码失败时不应有任何事件进入 sink")

    def test_catalog_bad_player_slot_returns_dict_not_exception(self) -> None:
        """世界上报越界 player_slot（违反 SET_TARGET allowed_ranges [[0,63],[127,127]]）时返回 dict。"""
        state = YuiSessionState()
        state.session = 1193046
        state.control_state = "external"
        state.capabilities = ("follow", "navmesh", "operation_lifecycle")
        # slot=64 不在 [0,63] 也不在 [127,127]
        state.players[64] = {"slot": 64, "pid": 99}
        adapter, sent = _real_adapter(state)
        result = adapter.follow(64)
        self.assertIsInstance(result, dict, "必须返回 dict，不得抛异常")
        self.assertFalse(result.get("midi_sent", True), "编码失败时不应有 MIDI 发出")
        self.assertEqual(sent, [], "编码失败时不应有任何事件进入 sink")

    def test_text_preset_uses_catalog_key_and_frozen_registers(self) -> None:
        self._ready()
        self.state.catalogs["text_preset"][2] = {
            "id": 2,
            "name": "preset_2",
            "text": "稍等一下…",
        }
        result = self.adapter.say_preset("preset_2", display_seconds=7)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            self.transport.calls[-1],
            ("TEXT_PRESET", (0, 0, 0, 2, 7, 0)),
        )
        missing = self.adapter.say_preset("not_published")
        self.assertEqual(missing["error"], "text_preset_not_found")

    def test_v12_semantic_tools_fail_locally_when_capability_or_catalog_is_missing(self) -> None:
        self._ready()
        self.state.spec_version = "1.2"
        missing_capability = self.adapter.navigate("upper_observation")
        self.assertEqual(missing_capability["error"], "unsupported_capability")

        self.state.capabilities += ("world_map", "semantic_navigation")
        missing_target = self.adapter.navigate("upper_observation")
        self.assertEqual(missing_target["error"], "target_missing")
        missing_entity = self.adapter.orbit("central_obstacle")
        self.assertEqual(missing_entity["error"], "target_missing")
        missing_region = self.adapter.explore("upper_floor")
        self.assertEqual(missing_region["error"], "target_missing")
        missing_player = self.adapter.approach(4)
        self.assertEqual(missing_player["error"], "slot_unknown")
        self.assertEqual(self.transport.calls, [])

    def test_v12_semantic_tools_require_operation_lifecycle(self) -> None:
        self._ready()
        self.state.spec_version = "1.2"
        self.state.capabilities += ("world_map", "semantic_navigation")
        self.state.capabilities = tuple(item for item in self.state.capabilities if item != "operation_lifecycle")
        result = self.adapter.execute_plan({"entry": "root", "nodes": [{"id": "root", "type": "wait", "duration_ms": 1}]})
        self.assertEqual(result["error"], "unsupported_capability")
        self.assertEqual(self.transport.calls, [])

    def test_v13_relative_move_and_region_explore_encode_once(self) -> None:
        self._ready()
        self.state.spec_version = "1.3"
        self.state.capabilities += (
            "world_map", "semantic_navigation", "region_localization", "local_navigation",
        )
        self.state.catalogs["region"][2] = {
            "id": 2,
            "semantic_key": "upper_floor",
            "entry_anchor_id": 0,
            "explorable": True,
        }
        moved = self.adapter.move_relative_wire(
            90.0, 2.0, speed_mps=1.0, face_travel=True, allow_shorter=True,
        )
        self.assertEqual(moved["status"], "succeeded")
        self.assertEqual(self.transport.calls[-1], ("MOVE_RELATIVE", (2000, 0, 4096, 0, 64, 3)))

        explored = self.adapter.explore_region_wire(
            "upper_floor", duration_ms=20_000, strategy="unvisited", speed_mps=1.0,
        )
        self.assertEqual(explored["status"], "succeeded")
        self.assertEqual(self.transport.calls[-1], ("EXPLORE_REGION", (200, 0, 0, 2, 64, 0)))

    def test_v13_rejects_unpublished_or_unexplorable_region_without_midi(self) -> None:
        self._ready()
        self.state.spec_version = "1.3"
        self.state.capabilities += ("world_map", "semantic_navigation", "local_navigation")
        self.state.catalogs["region"][1] = {
            "id": 1, "semantic_key": "stairway", "entry_anchor_id": 0, "explorable": False,
        }
        result = self.adapter.explore("stairway", duration_s=20)
        self.assertEqual(result["error"], "target_missing")
        self.assertEqual(self.transport.calls, [])

        self.state.capabilities = tuple(item for item in self.state.capabilities if item != "local_navigation")
        missing_capability = self.adapter.move_relative(0.0, 1.0)
        self.assertEqual(missing_capability["error"], "unsupported_capability")
        self.assertEqual(self.transport.calls, [])


if __name__ == "__main__":
    unittest.main()
