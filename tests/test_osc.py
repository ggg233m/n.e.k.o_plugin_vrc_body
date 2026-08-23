from __future__ import annotations

import math
import socket
import struct
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import VrchatOscConfig
from neko_anyadance_body.osc import (
    OscProtocolError,
    VrchatOscBridge,
    decode_osc_packet,
    encode_osc_message,
    normalize_parameter_value,
    validate_parameter_name,
)


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


class OscProtocolTests(unittest.TestCase):
    def test_round_trip_supported_types(self) -> None:
        packet = encode_osc_message(
            "/avatar/parameters/Test",
            (True, False, 17, 0.25, "文字", b"abc", None),
        )
        messages = decode_osc_packet(packet)
        self.assertEqual(len(messages), 1)
        address, arguments = messages[0]
        self.assertEqual(address, "/avatar/parameters/Test")
        self.assertEqual(arguments[:3], (True, False, 17))
        self.assertAlmostEqual(arguments[3], 0.25)
        self.assertEqual(arguments[4:], ("文字", b"abc", None))

    def test_bundle_is_flattened(self) -> None:
        first = encode_osc_message("/avatar/change", ("avtr_test",))
        second = encode_osc_message("/avatar/parameters/NEKO_Action", (3,))
        bundle = (
            b"#bundle\x00"
            + struct.pack(">Q", 1)
            + struct.pack(">i", len(first))
            + first
            + struct.pack(">i", len(second))
            + second
        )
        self.assertEqual(
            decode_osc_packet(bundle),
            [
                ("/avatar/change", ("avtr_test",)),
                ("/avatar/parameters/NEKO_Action", (3,)),
            ],
        )

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(OscProtocolError):
            encode_osc_message("not/an/address", (1,))
        with self.assertRaises(ValueError):
            validate_parameter_name("bad/name")
        with self.assertRaises(ValueError):
            normalize_parameter_value(float("nan"))
        with self.assertRaises(ValueError):
            normalize_parameter_value("1")

    def test_guarded_pulse_does_not_emit_release_if_press_is_skipped(self) -> None:
        now = [0.0]
        sent: list[tuple[str, str, bool]] = []
        bridge = VrchatOscBridge(
            VrchatOscConfig(input_pulse_ms=20),
            clock=lambda: now[0],
        )
        bridge._send_socket = object()  # type: ignore[assignment]
        bridge.send_input = lambda action, side, pressed: (  # type: ignore[method-assign]
            sent.append((action, side, pressed)) or (True, None)
        )
        self.assertTrue(bridge.schedule_input_pulse(
            "grab", "right", delay_s=1.0, guard=lambda: False,
        ))
        now[0] = 1.1
        bridge._run_due_inputs()
        now[0] = 1.2
        bridge._run_due_inputs()
        self.assertEqual(sent, [])

    def test_axis_rejects_nonfinite_and_unbounded_commands(self) -> None:
        sent: list[tuple[str, tuple[object, ...]]] = []
        bridge = VrchatOscBridge(VrchatOscConfig())
        bridge._send = lambda address, arguments: (  # type: ignore[method-assign]
            sent.append((address, tuple(arguments))) or (True, None)
        )

        for value, duration in (
            (float("nan"), 1.0),
            (float("inf"), 1.0),
            (1.0, float("nan")),
            (1.0, -1.0),
            (1.0, 0.0),
        ):
            accepted, reason = bridge.set_axis("move_vertical", value, duration)
            self.assertFalse(accepted, (value, duration, reason))
        self.assertEqual(sent, [])

    def test_set_axes_rolls_back_partial_locomotion(self) -> None:
        sent: list[tuple[str, tuple[object, ...]]] = []
        bridge = VrchatOscBridge(VrchatOscConfig())

        def send(address: str, arguments: tuple[object, ...]):
            sent.append((address, tuple(arguments)))
            if address == "/input/Horizontal" and arguments != (0.0,):
                return False, "horizontal failed"
            return True, None

        bridge._send = send  # type: ignore[method-assign]
        accepted, reason = bridge.set_axes(
            {"move_vertical": 1.0, "move_horizontal": 1.0},
            1.0,
        )
        self.assertFalse(accepted)
        self.assertIn("horizontal failed", reason or "")
        self.assertEqual(
            sent,
            [
                ("/input/Vertical", (1.0,)),
                ("/input/Horizontal", (1.0,)),
                ("/input/Vertical", (0.0,)),
                ("/input/Horizontal", (0.0,)),
            ],
        )
        self.assertEqual(bridge.snapshot()["active_axes"], {})

    def test_newer_axis_command_wins_over_older_expiration(self) -> None:
        now = [0.0]
        sent: list[tuple[str, tuple[object, ...]]] = []
        bridge = VrchatOscBridge(VrchatOscConfig(), clock=lambda: now[0])
        bridge._send = lambda address, arguments: (
            sent.append((address, tuple(arguments))) or (True, None)
        )  # type: ignore[method-assign]

        self.assertTrue(bridge.set_axis("move_vertical", 0.25, 1.0)[0])
        now[0] = 0.5
        self.assertTrue(bridge.set_axis("move_vertical", 0.75, 3.0)[0])
        now[0] = 1.1
        bridge._run_axis_expirations()

        active = bridge.snapshot()["active_axes"]["move_vertical"]
        self.assertEqual(active["value"], 0.75)
        self.assertEqual(sent[-1], ("/input/Vertical", (0.75,)))

    def test_stop_all_axes_resets_known_axes_without_active_state(self) -> None:
        sent: list[tuple[str, tuple[object, ...]]] = []
        bridge = VrchatOscBridge(VrchatOscConfig())
        bridge._send = lambda address, arguments: (  # type: ignore[method-assign]
            sent.append((address, tuple(arguments))) or (True, None)
        )

        accepted, reason = bridge.stop_all_axes()
        self.assertTrue(accepted, reason)
        self.assertEqual(
            sent,
            [
                ("/input/Vertical", (0.0,)),
                ("/input/Horizontal", (0.0,)),
                ("/input/LookHorizontal", (0.0,)),
            ],
        )

    def test_button_path_uses_integer_payload(self) -> None:
        sent: list[tuple[str, tuple[object, ...]]] = []
        bridge = VrchatOscBridge(VrchatOscConfig())
        bridge._send = lambda address, arguments: (  # type: ignore[method-assign]
            sent.append((address, tuple(arguments))) or (True, None)
        )

        accepted, reason = bridge.send_button("jump", True)
        self.assertTrue(accepted, reason)
        self.assertEqual(sent, [("/input/Jump", (1,))])
        accepted, reason = bridge.send_button("jump", False)
        self.assertTrue(accepted, reason)
        self.assertEqual(sent[-1], ("/input/Jump", (0,)))

    def test_pulse_rejects_invalid_hold_without_pressing(self) -> None:
        sent: list[tuple[str, tuple[object, ...]]] = []
        bridge = VrchatOscBridge(VrchatOscConfig())
        bridge._send = lambda address, arguments: (  # type: ignore[method-assign]
            sent.append((address, tuple(arguments))) or (True, None)
        )

        for hold_ms in (float("nan"), float("inf"), -1, 1001, True):
            accepted, reason = bridge.pulse_input("grab", "right", hold_ms)
            self.assertFalse(accepted, (hold_ms, reason))
        self.assertEqual(sent, [])

    def test_chatbox_rejects_invalid_types_and_length(self) -> None:
        sent: list[tuple[str, tuple[object, ...]]] = []
        bridge = VrchatOscBridge(VrchatOscConfig())
        bridge._send = lambda address, arguments: (  # type: ignore[method-assign]
            sent.append((address, tuple(arguments))) or (True, None)
        )

        self.assertFalse(bridge.send_chatbox("x" * 145)[0])
        self.assertFalse(bridge.send_chatbox("hello", immediate="false")[0])  # type: ignore[arg-type]
        accepted, reason = bridge.send_chatbox("hello", immediate=False)
        self.assertTrue(accepted, reason)
        self.assertEqual(sent, [("/chatbox/input", ("hello", False, False))])


class OscMotionFeedbackTests(unittest.TestCase):
    """VRChat 内置 Velocity 参数是唯一能说明「我真的动了没有」的回传。"""

    def setUp(self) -> None:
        self.now = [100.0]
        self.wall = [1_700_000_000.0]
        self.bridge = VrchatOscBridge(
            VrchatOscConfig(),
            clock=lambda: self.now[0],
            wall_clock=lambda: self.wall[0],
        )

    def _feed(self, name: str, value: object) -> None:
        self.bridge._handle_message(f"/avatar/parameters/{name}", (value,))
        self.bridge._last_receive_at_monotonic = self.now[0]

    def test_missing_builtins_report_unavailable_not_zero_speed(self) -> None:
        # 最重要的一条：avatar 没配这些参数时不能读成「速度为零」，否则导航器
        # 会把「读不到」当成「卡住了」，一armed 就立刻停车。
        self._feed("NEKO_Action", 1)
        motion = self.bridge.motion_feedback()
        self.assertFalse(motion["available"])
        self.assertEqual(motion["reason"], "velocity_parameters_absent")
        self.assertIsNone(motion["speed_mps"])
        self.assertIn("VelocityX", motion["expected"])

    def test_no_feedback_at_all_is_distinguished_from_absent_parameters(self) -> None:
        motion = self.bridge.motion_feedback()
        self.assertFalse(motion["available"])
        self.assertEqual(motion["reason"], "no_feedback_received")

    def test_horizontal_speed_ignores_vertical_component(self) -> None:
        # 跳跃/下落时 VelocityY 很大而水平速度为零；用三维速度判断卡墙会把
        # 「贴着墙往下滑」误判成「正在前进」。
        self._feed("VelocityX", 3.0)
        self._feed("VelocityY", -9.0)
        self._feed("VelocityZ", 4.0)
        motion = self.bridge.motion_feedback()
        self.assertTrue(motion["available"], motion["reason"])
        self.assertAlmostEqual(motion["horizontal_speed_mps"], 5.0, places=3)
        self.assertAlmostEqual(motion["vertical_speed_mps"], -9.0, places=3)
        self.assertAlmostEqual(motion["speed_mps"], math.hypot(5.0, 9.0), places=3)

    def test_clear_path_reports_forward_ratio_near_one(self) -> None:
        # 实测：畅通前进时 VelocityZ 主导，X 只有 ~1e-7 的数值噪声。
        self._feed("VelocityX", -1.1920928955078125e-07)
        self._feed("VelocityZ", 2.6666667461395264)
        motion = self.bridge.motion_feedback()
        self.assertTrue(motion["available"], motion["reason"])
        self.assertAlmostEqual(motion["forward_ratio"], 1.0, places=3)
        self.assertAlmostEqual(motion["slip_ratio"], 0.0, places=3)

    def test_wall_slide_collapses_forward_ratio_while_speed_stays_high(self) -> None:
        # 斜撞墙时角色控制器把移动投影到墙面上：速度模长还很大，但前进分量塌了。
        # 只看 horizontal_speed_mps 看不出这个差别——这正是要导出比值的原因。
        self._feed("VelocityX", 2.5)
        self._feed("VelocityZ", 0.4)
        motion = self.bridge.motion_feedback()
        self.assertTrue(motion["available"], motion["reason"])
        self.assertGreater(motion["horizontal_speed_mps"], 2.0)  # 速度并不低
        self.assertLess(abs(motion["forward_ratio"]), 0.25)      # 但没在前进
        self.assertGreater(motion["slip_ratio"], 0.9)            # 正沿墙面滑行

    def test_slip_ratio_sign_distinguishes_which_way_the_wall_deflects(self) -> None:
        # 滑行方向就是可通行方向，符号不能丢，否则绕行会朝墙里拐。
        self._feed("VelocityX", -2.5)
        self._feed("VelocityZ", 0.4)
        motion = self.bridge.motion_feedback()
        self.assertLess(motion["slip_ratio"], -0.9)

    def test_ratios_are_none_at_rest_rather_than_zero(self) -> None:
        # 0.0 会被读成「正对着墙」，与「站着不动」是完全不同的结论。
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 0.0)
        motion = self.bridge.motion_feedback()
        self.assertTrue(motion["available"], motion["reason"])
        self.assertEqual(motion["horizontal_speed_mps"], 0.0)
        self.assertIsNone(motion["forward_ratio"])
        self.assertIsNone(motion["slip_ratio"])

    def test_wall_slide_ratio_ignores_vertical_slide(self) -> None:
        # 贴着墙下滑时 Y 很大；把它算进分母会把「前进被挡住」稀释成还在走。
        self._feed("VelocityX", 0.0)
        self._feed("VelocityY", -9.0)
        self._feed("VelocityZ", 2.0)
        motion = self.bridge.motion_feedback()
        self.assertAlmostEqual(motion["forward_ratio"], 1.0, places=3)

    def test_fresh_vertical_velocity_does_not_refresh_old_horizontal_motion(self) -> None:
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 1.0)
        self.now[0] += 3.0
        self.wall[0] += 3.0
        self._feed("VelocityY", -2.0)
        motion = self.bridge.motion_feedback(max_age_ms=2000)
        self.assertFalse(motion["available"])
        self.assertEqual(motion["reason"], "velocity_feedback_quiet")

    def test_old_zero_is_not_kept_alive_by_other_osc_traffic(self) -> None:
        # 实机确认 VelocityX/Z 只在角色移动时回传。其他参数还在更新也不能给旧的
        # 0 速度续命，否则下一次前进会把「上一次静止」误判成「这次撞墙」。
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 0.0)
        self.now[0] += 5.0
        self.wall[0] += 5.0
        self.bridge._last_receive_at_monotonic = self.now[0]  # 心跳仍在，只是速度没变
        motion = self.bridge.motion_feedback()
        self.assertFalse(motion["available"])
        self.assertEqual(motion["reason"], "velocity_feedback_quiet")
        self.assertGreaterEqual(motion["value_age_ms"], 5000.0)
        self.assertLess(motion["link_age_ms"], 1.0)

    def test_dead_link_reports_stale_rather_than_last_known_speed(self) -> None:
        # 最后一次回传说自己在动，之后彻底没声音：真在移动就会持续有速度更新，
        # 所以沉默与取值矛盾，只能判链路在运动中断了。
        self._feed("VelocityX", 1.0)
        self._feed("VelocityZ", 0.0)
        self.now[0] += 30.0
        motion = self.bridge.motion_feedback(max_age_ms=2000)
        self.assertFalse(motion["available"])
        self.assertEqual(motion["reason"], "velocity_feedback_quiet")
        self.assertIsNone(motion["speed_mps"])

    def test_silence_after_a_resting_reading_becomes_unavailable(self) -> None:
        """静止沉默是正常现象，但旧读数不能冒充当前实时速度。"""
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 0.0)
        self.now[0] += 600.0
        self.wall[0] += 600.0
        motion = self.bridge.motion_feedback(max_age_ms=2000)
        self.assertFalse(motion["available"])
        self.assertIsNone(motion["horizontal_speed_mps"])
        self.assertEqual(motion["reason"], "velocity_feedback_quiet")
        self.assertGreaterEqual(motion["link_age_ms"], 600_000.0)

    def test_pushing_into_a_wall_counts_as_moving_so_silence_is_still_stale(self) -> None:
        """卡墙实测速度 0.08 m/s，必须仍算「在动」。

        顶着墙时速度低但恒定，VRChat 同样可能不再发包。若把这个读数算成「静止
        且可信」，沉默就会被无限兜住——而这正是卡墙判据要抓的场景，等于反过来
        把它废掉。静止阈值必须留在实测卡墙速度之下。
        """
        self._feed("VelocityX", 0.08)
        self._feed("VelocityZ", 0.0)
        self.now[0] += 30.0
        motion = self.bridge.motion_feedback(max_age_ms=2000)
        self.assertFalse(motion["available"])
        self.assertEqual(motion["reason"], "velocity_feedback_quiet")

    def test_fresh_zero_sample_is_temporarily_available(self) -> None:
        # 即使实机通常不发静止 0，刚收到的样本仍是有效事实；只是不能永久缓存。
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 0.0)
        motion = self.bridge.motion_feedback(max_age_ms=2000)
        self.assertTrue(motion["available"], motion["reason"])
        self.assertIsNone(motion["reason"])

    def test_grounded_and_upright_are_read_without_faking_missing_ones(self) -> None:
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 0.0)
        self._feed("Grounded", True)
        motion = self.bridge.motion_feedback()
        self.assertIs(motion["grounded"], True)
        self.assertIsNone(motion["upright"])
        self.assertIsNone(motion["angular_speed"])

    def test_awareness_surfaces_motion_and_keeps_action_summary_readable(self) -> None:
        self._feed("NEKO_Action", 2)
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 0.0)
        awareness = self.bridge.awareness()
        self.assertTrue(awareness["motion"]["available"])
        # 内置参数仍在 parameters 里（面板要看原始回传），但摘要只列动作状态，
        # 否则六个速度分量会把真正要读的东西淹掉。
        self.assertIn("VelocityX", awareness["parameters"])
        self.assertIn("NEKO_Action=2", awareness["summary"])
        self.assertNotIn("VelocityX=", awareness["summary"])

    def test_awareness_says_so_when_builtins_never_arrive(self) -> None:
        self._feed("NEKO_Action", 2)
        awareness = self.bridge.awareness()
        self.assertFalse(awareness["motion"]["available"])
        self.assertIn("无法确认自己是否真的在移动", awareness["summary"])

    def test_snapshot_carries_motion_for_the_unarmed_debug_panel(self) -> None:
        # 面板在未授权时读不到导航器的采样（那是 tick 里刷新的），只能靠快照。
        self._feed("VelocityX", 0.0)
        self._feed("VelocityZ", 1.5)
        motion = self.bridge.snapshot()["motion"]
        self.assertTrue(motion["available"], motion["reason"])
        self.assertAlmostEqual(motion["horizontal_speed_mps"], 1.5, places=3)


class OscBridgeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target.bind(("127.0.0.1", 0))
        self.target.settimeout(1.0)
        self.listen_port = _free_udp_port()
        self.bridge = VrchatOscBridge(VrchatOscConfig(
            send_host="127.0.0.1",
            send_port=self.target.getsockname()[1],
            listen_host="127.0.0.1",
            listen_port=self.listen_port,
            allowed_sender="127.0.0.1",
            input_pulse_ms=30,
            awareness_parameters=("NEKO_Action",),
        ))
        self.bridge.start()

    def tearDown(self) -> None:
        self.bridge.stop()
        self.target.close()

    def test_parameter_send_feedback_receive_and_input_release(self) -> None:
        sent, reason = self.bridge.send_parameter("NEKO_Action", 4)
        self.assertTrue(sent, reason)
        packet, _ = self.target.recvfrom(4096)
        self.assertEqual(
            decode_osc_packet(packet),
            [("/avatar/parameters/NEKO_Action", (4,))],
        )

        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(encode_osc_message("/avatar/change", ("avtr_test",)), ("127.0.0.1", self.listen_port))
            sender.sendto(
                encode_osc_message("/avatar/parameters/NEKO_Action", (4,)),
                ("127.0.0.1", self.listen_port),
            )
        finally:
            sender.close()
        deadline = time.monotonic() + 1.0
        while self.bridge.snapshot()["parameter_count"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        awareness = self.bridge.awareness()
        self.assertEqual(awareness["connection"], "detected")
        self.assertEqual(awareness["avatar_id"], "avtr_test")
        self.assertEqual(awareness["parameters"]["NEKO_Action"]["value"], 4)
        self.assertFalse(awareness["pose_feedback_available"])

        sent, reason = self.bridge.pulse_input("grab", "right", 30)
        self.assertTrue(sent, reason)
        values: list[int] = []
        deadline = time.monotonic() + 1.0
        while len(values) < 2 and time.monotonic() < deadline:
            packet, _ = self.target.recvfrom(4096)
            address, arguments = decode_osc_packet(packet)[0]
            if address == "/input/GrabRight":
                values.append(arguments[0])
        self.assertEqual(values, [1, 0])


if __name__ == "__main__":
    unittest.main()
