"""YUI 插件适配层测试。"""

from __future__ import annotations

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

    def test_arm_never_auto_authorizes(self) -> None:
        self.state.session = 1193046
        self.state.control_state = "safe_idle"
        rejected = self.adapter.arm()
        self.assertEqual(rejected["error"], "arm_not_authorized")
        self.assertEqual(self.transport.calls, [])
        self.adapter.authorize_arm(True)
        accepted = self.adapter.arm()
        self.assertEqual(accepted["status"], "succeeded")
        self.assertEqual(self.transport.calls, [("SET_CONTROL_MODE", (0, 0, 0, 1, 0, 0))])

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


if __name__ == "__main__":
    unittest.main()
