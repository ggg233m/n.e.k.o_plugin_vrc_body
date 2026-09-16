from __future__ import annotations

import math
import threading
from types import SimpleNamespace
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import BodyProfile, VmcIdleConfig
from neko_anyadance_body.model import LEFT_CANONICAL_QUAT, RIGHT_CANONICAL_QUAT
from neko_anyadance_body.osc import encode_osc_message
from neko_anyadance_body.vmc_idle import VmcIdleRelay, _quat_multiply


def _bone(name: str, x: float, y: float, z: float = 0.0):
    return "/VMC/Ext/Bone/Pos", (name, x, y, z, 0.0, 0.0, 0.0, 1.0)


def _finger_bone(name: str, degrees: float = 0.0):
    half_angle = math.radians(degrees) / 2.0
    return "/VMC/Ext/Bone/Pos", (
        name, 0.0, 0.0, 0.0,
        math.sin(half_angle), 0.0, 0.0, math.cos(half_angle),
    )


def _complete_frame_messages():
    return [
        ("/VMC/Ext/OK", (1,)),
        ("/VMC/Ext/T", (0.0,)),
        ("/VMC/Ext/Root/Pos", ("root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        _bone("Hips", 0.0, 1.0),
        _bone("Spine", 0.0, 0.2),
        _bone("Chest", 0.0, 0.2),
        _bone("UpperChest", 0.0, 0.0),
        _bone("Neck", 0.0, 0.2),
        _bone("Head", 0.0, 0.2),
        _bone("LeftShoulder", 0.0, 0.0),
        _bone("LeftUpperArm", -0.2, 0.15),
        _bone("LeftLowerArm", -0.3, 0.0),
        _bone("LeftHand", -0.25, 0.0),
        _bone("RightShoulder", 0.0, 0.0),
        _bone("RightUpperArm", 0.2, 0.15),
        _bone("RightLowerArm", 0.3, 0.0),
        _bone("RightHand", 0.25, 0.0),
        _bone("LeftUpperLeg", -0.1, -0.4),
        _bone("LeftLowerLeg", 0.0, -0.4),
        _bone("LeftFoot", 0.0, -0.2),
        _bone("LeftToes", 0.0, -0.1, -0.1),
        _bone("RightUpperLeg", 0.1, -0.4),
        _bone("RightLowerLeg", 0.0, -0.4),
        _bone("RightFoot", 0.0, -0.2),
        _bone("RightToes", 0.0, -0.1, -0.1),
        ("/VMC/Ext/T", (1.0 / 60.0,)),
    ]


class VmcIdleRelayTests(unittest.TestCase):
    def test_missing_upper_chest_calibrates_and_tracks_chest_rotation(self) -> None:
        # 复现真实 53 骨骼流：上胸骨缺省，但胸部旋转仍须传到头和双手。
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: 10.0)
        messages = [item for item in _complete_frame_messages()
                    if not (item[0] == "/VMC/Ext/Bone/Pos" and item[1][0] == "UpperChest")]
        relay.reset_calibration(reason="host_t_pose")
        relay.ingest_messages(messages, now=10.0)
        frame = relay.latest_frame()
        self.assertIsNotNone(frame)
        self.assertTrue(relay.snapshot()["calibration"]["calibrated"])
        self.assertEqual(relay.snapshot()["incomplete_frames"], 0)
        assert frame is not None
        rotation = (0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))
        rotated = [(address, (*args[:4], *rotation))
                   if address == "/VMC/Ext/Bone/Pos" and args[0] == "Chest"
                   else (address, args) for address, args in messages]
        relay.ingest_messages(rotated, now=10.0)
        moved = relay.latest_frame()
        self.assertIsNotNone(moved)
        assert moved is not None
        # VMC 左手坐标输入在接收端转换为 AnyaDance 的右手坐标。
        output_rotation = (0.0, -rotation[1], 0.0, rotation[3])
        for device in ("hmd", "left_controller", "right_controller"):
            expected = _quat_multiply(output_rotation, frame.devices[device].rotation)
            for actual, value in zip(moved.devices[device].rotation, expected):
                self.assertAlmostEqual(actual, value, places=6)
        self.assertEqual(relay.snapshot()["accepted_frames"], 2)

    def test_listening_receiver_is_not_reported_as_unknown(self) -> None:
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile())
        with relay._lock:
            relay._receiver_listening = True
        self.assertEqual(relay.snapshot()["connection"], "listening")

    def test_calibration_hold_rejects_live_frames_until_host_t_pose(self) -> None:
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: 10.0)
        relay.hold_calibration(reason="waiting_for_host_t_pose")
        relay.ingest_messages(_complete_frame_messages(), now=10.0)
        self.assertIsNone(relay.latest_frame())
        held = relay.snapshot()["calibration"]
        self.assertTrue(held["held"])
        self.assertFalse(held["calibrated"])

        relay.reset_calibration(reason="host_t_pose")
        relay.ingest_messages(_complete_frame_messages(), now=10.0)
        self.assertIsNotNone(relay.latest_frame())
        self.assertFalse(relay.snapshot()["calibration"]["held"])

    def test_humanoid_local_transforms_become_six_anyadance_devices(self) -> None:
        current = [10.0]
        relay = VmcIdleRelay(
            VmcIdleConfig(stale_after_ms=500),
            BodyProfile(height_m=1.5),
            clock=lambda: current[0],
        )
        relay.ingest_messages(_complete_frame_messages(), now=current[0])

        frame = relay.latest_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(set(frame.devices), {
            "hmd", "left_controller", "right_controller", "hip", "left_foot", "right_foot",
        })
        expected_positions = {
            "hmd": (0.0, 1.50, 0.0),
            "left_controller": (-0.68, 1.33, -0.10),
            "right_controller": (0.68, 1.33, -0.10),
            "hip": (0.0, 1.07, -0.05),
            "left_foot": (-0.09, 0.26, 0.10),
            "right_foot": (0.09, 0.26, 0.10),
        }
        for device, expected_position in expected_positions.items():
            for actual, expected in zip(frame.devices[device].position, expected_position):
                self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(frame.devices["left_controller"].rotation, LEFT_CANONICAL_QUAT):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(frame.devices["right_controller"].rotation, RIGHT_CANONICAL_QUAT):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(relay.snapshot()["accepted_frames"], 1)
        self.assertEqual(
            set(relay.snapshot()["calibration"]["position_mount_offsets_m"]),
            set(expected_positions),
        )

        current[0] += 0.6
        self.assertIsNone(relay.latest_frame())
        self.assertFalse(relay.snapshot()["source_available"])

    def test_udp_decoder_accepts_only_configured_sender(self) -> None:
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile())
        packet = encode_osc_message("/VMC/Ext/OK", [1])
        self.assertFalse(relay.ingest_packet(packet, sender=("192.0.2.8", 1234), now=1.0))
        self.assertTrue(relay.ingest_packet(packet, sender=("127.0.0.1", 1234), now=1.1))
        status = relay.snapshot()
        self.assertEqual(status["received_packets"], 1)
        self.assertEqual(status["rejected_packets"], 1)

    def test_parent_arm_rotation_reaches_controller_rotation(self) -> None:
        current = [10.0]
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        relay.ingest_messages(_complete_frame_messages(), now=current[0])

        half_sqrt = 2.0 ** -0.5
        rotated_messages = []
        for address, arguments in _complete_frame_messages():
            if address == "/VMC/Ext/Bone/Pos" and arguments[0] == "LeftUpperArm":
                arguments = (*arguments[:4], half_sqrt, 0.0, 0.0, half_sqrt)
            rotated_messages.append((address, arguments))
        current[0] += 1.0 / 60.0
        relay.ingest_messages(rotated_messages, now=current[0])

        frame = relay.latest_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertNotEqual(frame.devices["left_controller"].rotation, LEFT_CANONICAL_QUAT)
        for actual, expected in zip(frame.devices["right_controller"].rotation, RIGHT_CANONICAL_QUAT):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_missing_intermediate_bone_rejects_frame(self) -> None:
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: 10.0)
        incomplete = [
            message
            for message in _complete_frame_messages()
            if not (
                message[0] == "/VMC/Ext/Bone/Pos"
                and message[1][0] == "Neck"
            )
        ]
        relay.ingest_messages(incomplete, now=10.0)
        self.assertIsNone(relay.latest_frame())
        self.assertEqual(relay.snapshot()["incomplete_frames"], 1)

    def test_tracker_mount_offset_rotates_with_driving_bone(self) -> None:
        current = [10.0]
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        relay.ingest_messages(_complete_frame_messages(), now=current[0])
        rest = relay.latest_frame()
        self.assertIsNotNone(rest)
        assert rest is not None

        half_sqrt = 2.0 ** -0.5
        rotated_messages = []
        for address, arguments in _complete_frame_messages():
            if address == "/VMC/Ext/Bone/Pos" and arguments[0] == "Hips":
                # Unity/VMC +90 degrees about Y reflects to -90 degrees in the
                # right-handed relay coordinates.
                arguments = (*arguments[:4], 0.0, half_sqrt, 0.0, half_sqrt)
            rotated_messages.append((address, arguments))
        current[0] += 1.0 / 60.0
        relay.ingest_messages(rotated_messages, now=current[0])
        rotated = relay.latest_frame()
        self.assertIsNotNone(rotated)
        assert rotated is not None

        # The hip tracker mount is offset from the anatomical hip joint.  Its
        # horizontal component must turn with the hip instead of staying as a
        # fixed world-space translation.
        rest_offset = (
            rest.devices["hip"].position[0],
            rest.devices["hip"].position[2],
        )
        rotated_offset = (
            rotated.devices["hip"].position[0],
            rotated.devices["hip"].position[2],
        )
        self.assertNotEqual(rotated_offset, rest_offset)

    def test_vmc_finger_bones_become_anyadance_finger_bends(self) -> None:
        current = [10.0]
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        rest_messages = _complete_frame_messages()
        rest_messages[-1:-1] = [
            _finger_bone("LeftIndexProximal"),
            _finger_bone("LeftIndexIntermediate"),
            _finger_bone("LeftIndexDistal"),
            _finger_bone("RightThumbProximal"),
            _finger_bone("RightThumbIntermediate"),
            _finger_bone("RightThumbDistal"),
        ]
        relay.ingest_messages(rest_messages, now=current[0])

        curled_messages = _complete_frame_messages()
        curled_messages[-1:-1] = [
            _finger_bone("LeftIndexProximal", 90.0),
            _finger_bone("LeftIndexIntermediate", 80.0),
            _finger_bone("LeftIndexDistal", 80.0),
            _finger_bone("RightThumbProximal", 5.0),
            _finger_bone("RightThumbIntermediate", 90.0),
            _finger_bone("RightThumbDistal", 90.0),
        ]
        current[0] += 1.0 / 60.0
        relay.ingest_messages(curled_messages, now=current[0])

        frame = relay.latest_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertAlmostEqual(frame.controllers["left_controller"].finger_bends["index"], 1.0, places=6)
        self.assertAlmostEqual(frame.controllers["right_controller"].finger_bends["thumb"], 1.0, places=6)
        self.assertEqual(frame.controllers["left_controller"].finger_bends["middle"], 0.0)
        self.assertEqual(frame.controllers["right_controller"].finger_bends["index"], 0.0)

    def test_unavailable_marker_discards_pose_and_calibration(self) -> None:
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: 3.0)
        relay.ingest_messages(_complete_frame_messages(), now=3.0)
        self.assertIsNotNone(relay.latest_frame())
        relay.ingest_messages([("/VMC/Ext/OK", (0,))], now=3.1)
        self.assertIsNone(relay.latest_frame())
        self.assertFalse(relay.snapshot()["source_available"])
        self.assertFalse(relay.snapshot()["calibration"]["calibrated"])
        self.assertEqual(relay.snapshot()["calibration"]["position_mount_offsets_m"], {})

    def test_host_t_pose_replaces_animation_dependent_wrist_and_finger_rest(self) -> None:
        current = [10.0]
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        animated_rest = _complete_frame_messages()
        animated_rest[-1:-1] = [
            _finger_bone("LeftIndexProximal", 35.0),
            _finger_bone("LeftIndexIntermediate", 25.0),
            _finger_bone("LeftIndexDistal", 20.0),
        ]
        relay.ingest_messages(animated_rest, now=current[0])

        relay.reset_calibration(reason="host_t_pose")
        t_pose = _complete_frame_messages()
        t_pose[-1:-1] = [
            _finger_bone("LeftIndexProximal"),
            _finger_bone("LeftIndexIntermediate"),
            _finger_bone("LeftIndexDistal"),
        ]
        current[0] += 1.0 / 60.0
        relay.ingest_messages(t_pose, now=current[0])
        calibrated = relay.latest_frame()
        self.assertIsNotNone(calibrated)
        assert calibrated is not None
        for actual, expected in zip(calibrated.devices["left_controller"].rotation, LEFT_CANONICAL_QUAT):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(calibrated.controllers["left_controller"].finger_bends["index"], 0.0)
        calibration = relay.snapshot()["calibration"]
        self.assertEqual(calibration["generation"], 1)
        self.assertEqual(calibration["reason"], "host_t_pose")
        self.assertTrue(calibration["calibrated"])

        live_pose = []
        half_angle = math.radians(45.0) / 2.0
        for address, arguments in animated_rest:
            if address == "/VMC/Ext/Bone/Pos" and arguments[0] == "LeftHand":
                arguments = (
                    *arguments[:4],
                    math.sin(half_angle), 0.0, 0.0, math.cos(half_angle),
                )
            live_pose.append((address, arguments))
        current[0] += 1.0 / 60.0
        relay.ingest_messages(live_pose, now=current[0])
        live = relay.latest_frame()
        self.assertIsNotNone(live)
        assert live is not None
        self.assertAlmostEqual(
            live.controllers["left_controller"].finger_bends["index"],
            (35.0 + 25.0 + 20.0) / (90.0 + 80.0 + 80.0),
            places=6,
        )
        expected_wrist = _quat_multiply(
            # The helper message is in Unity/VMC left-handed coordinates;
            # _vmc_transform reflects X/Y quaternion components back to the
            # right-handed AnyaDance space before FK.
            (-math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)),
            LEFT_CANONICAL_QUAT,
        )
        for actual, expected in zip(live.devices["left_controller"].rotation, expected_wrist):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_host_t_pose_calibration_rejects_arm_down_transition_frame(self) -> None:
        current = [10.0]
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        relay.hold_calibration(reason="waiting_for_host_t_pose")
        relay.reset_calibration(reason="host_t_pose")

        half_sqrt = 2.0 ** -0.5
        arm_down = []
        for address, arguments in _complete_frame_messages():
            if address == "/VMC/Ext/Bone/Pos" and arguments[0] == "LeftUpperArm":
                arguments = (*arguments[:4], 0.0, 0.0, half_sqrt, half_sqrt)
            elif address == "/VMC/Ext/Bone/Pos" and arguments[0] == "RightUpperArm":
                arguments = (*arguments[:4], 0.0, 0.0, -half_sqrt, half_sqrt)
            arm_down.append((address, arguments))
        relay.ingest_messages(arm_down, now=current[0])
        self.assertIsNone(relay.latest_frame())
        rejected = relay.snapshot()["calibration"]
        self.assertTrue(rejected["waiting_for_t_pose_frame"])
        self.assertEqual(rejected["rejected_frames"], 1)

        current[0] += 1.0 / 60.0
        relay.ingest_messages(_complete_frame_messages(), now=current[0])
        accepted = relay.latest_frame()
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertEqual(accepted.devices["left_controller"].position, (-0.68, 1.33, -0.10))
        self.assertFalse(relay.snapshot()["calibration"]["waiting_for_t_pose_frame"])

    def test_host_output_restart_requires_authoritative_rest_pose(self) -> None:
        """宿主暂停后不得用恢复时的动画姿势重新自锁腕部与手指零点。"""
        current = [10.0]
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        relay.reset_calibration(reason="host_t_pose")
        relay.ingest_messages(_complete_frame_messages(), now=current[0])
        self.assertIsNotNone(relay.latest_frame())
        self.assertTrue(relay.has_baseline())
        self.assertFalse(relay.needs_recalibration())

        # 页面冻结 / VMC 输出重新握手：宿主发 OK=0，基准与姿态一起作废。
        relay.ingest_messages([("/VMC/Ext/OK", (0,))], now=current[0])
        self.assertIsNone(relay.latest_frame())
        self.assertTrue(relay.needs_recalibration())

        # 恢复后的第一个完整帧是动画姿势，它不能成为新的静止基准。
        current[0] += 3.0
        half_sqrt = 2.0 ** -0.5
        animated = []
        for address, arguments in _complete_frame_messages():
            if address == "/VMC/Ext/Bone/Pos" and arguments[0] == "LeftUpperArm":
                arguments = (*arguments[:4], 0.0, 0.0, half_sqrt, half_sqrt)
            animated.append((address, arguments))
        animated[0] = ("/VMC/Ext/OK", (1,))
        relay.ingest_messages(animated, now=current[0])
        self.assertIsNone(relay.latest_frame())
        self.assertFalse(relay.snapshot()["calibration"]["calibrated"])
        self.assertTrue(relay.needs_recalibration())

        # 补一次 T Pose 之后才允许重建基准，且重建结果必须是正确的腕部零点。
        current[0] += 1.0 / 60.0
        relay.ingest_messages(_complete_frame_messages(), now=current[0])
        calibrated = relay.latest_frame()
        self.assertIsNotNone(calibrated)
        assert calibrated is not None
        for actual, expected in zip(
            calibrated.devices["left_controller"].rotation, LEFT_CANONICAL_QUAT
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertFalse(relay.needs_recalibration())

    def test_stall_discards_partial_frame_instead_of_mixing_across_gap(self) -> None:
        """卡顿前的半帧不得与恢复后的新帧拼成一个跨卡顿的混合帧。"""
        current = [10.0]
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        partial = [
            message
            for message in _complete_frame_messages()
            if message[0] != "/VMC/Ext/T"
            and not (message[0] == "/VMC/Ext/Bone/Pos" and message[1][0] == "Neck")
        ]
        relay.ingest_messages(partial, now=current[0])

        # 卡顿 3 秒后恢复，新帧仍然缺 Neck；旧帧残留的 Neck 不能补上这个缺口。
        current[0] += 3.0
        resumed = [
            message
            for message in _complete_frame_messages()
            if not (message[0] == "/VMC/Ext/Bone/Pos" and message[1][0] == "Neck")
        ]
        relay.ingest_messages(resumed, now=current[0])
        self.assertIsNone(relay.latest_frame())
        self.assertEqual(relay.snapshot()["incomplete_frames"], 1)


class ManualRecalibrationTests(unittest.TestCase):
    """面板「重新校准 VMC」按钮的后端语义。

    这颗按钮要救的是「零点锁错、动作一直是歪的」。它的核心约束不是成功路径，
    而是**按了不能比不按更糟**：清掉基准之后若没有人去重建，中转会一直拒绝普通
    帧、角色停在最后一帧，从「歪着能动」变成「正着不动」。
    """

    class _StubService:
        """只借 ``BackendService.vmc_recalibrate`` 一个方法，不启动真后端。"""

        def __init__(self, relay, *, manage_host_output: bool, worker_alive: bool) -> None:
            self.vmc_idle = relay
            self._lock = threading.RLock()
            self.config = SimpleNamespace(
                vmc_idle=VmcIdleConfig(manage_host_output=manage_host_output)
            )
            self._vmc_calibration_thread = (
                SimpleNamespace(is_alive=lambda: True) if worker_alive else None
            )

    def _service(self, relay, *, manage_host_output: bool = True, worker_alive: bool = True):
        from neko_anyadance_body.backend.service import BackendService

        stub = self._StubService(
            relay, manage_host_output=manage_host_output, worker_alive=worker_alive
        )
        stub.vmc_recalibrate = BackendService.vmc_recalibrate.__get__(stub, type(stub))
        return stub

    def _calibrated_relay(self, current: list[float]) -> VmcIdleRelay:
        relay = VmcIdleRelay(VmcIdleConfig(), BodyProfile(), clock=lambda: current[0])
        relay.reset_calibration(reason="host_t_pose")
        relay.ingest_messages(_complete_frame_messages(), now=current[0])
        assert relay.latest_frame() is not None
        return relay

    def test_default_path_hands_the_baseline_back_to_the_t_pose_worker(self) -> None:
        current = [10.0]
        relay = self._calibrated_relay(current)
        result = self._service(relay).vmc_recalibrate()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["mode"], "host_t_pose")
        # 旧基准作废，并且明确要求一次权威静止姿势——校准线程下一轮就会去请求。
        self.assertTrue(relay.needs_recalibration())
        self.assertTrue(relay.snapshot()["calibration"]["waiting_for_t_pose_frame"])
        self.assertFalse(relay.snapshot()["calibration"]["calibrated"])

        # 期间的普通动画帧仍然不能顶替静止姿势，否则等于没校准。
        current[0] += 1.0
        half_sqrt = 2.0 ** -0.5
        animated = [
            (address, (*arguments[:4], 0.0, 0.0, half_sqrt, half_sqrt))
            if address == "/VMC/Ext/Bone/Pos" and arguments[0] == "LeftUpperArm"
            else (address, arguments)
            for address, arguments in _complete_frame_messages()
        ]
        relay.ingest_messages(animated, now=current[0])
        self.assertIsNone(relay.latest_frame())

    def test_request_is_refused_outright_when_nobody_can_rebuild_the_baseline(self) -> None:
        """没有校准线程时必须原样拒绝：清了基准没人重建，角色会永久停在最后一帧。"""
        for label, kwargs in (
            ("托管宿主输出已关闭", {"manage_host_output": False}),
            ("校准线程没在跑", {"worker_alive": False}),
        ):
            with self.subTest(label):
                current = [10.0]
                relay = self._calibrated_relay(current)
                before = relay.snapshot()["calibration"]
                result = self._service(relay, **kwargs).vmc_recalibrate()
                self.assertFalse(result["accepted"])
                # 拒绝要指出退路，否则用户只会对着一颗永远失败的按钮反复点。
                self.assertIn("accept_current_pose=true", result["reason"])
                # 关键：状态一个字段都没动，中转照常工作。
                self.assertEqual(relay.snapshot()["calibration"], before)
                self.assertIsNotNone(relay.latest_frame())
                self.assertFalse(relay.needs_recalibration())

    def test_fallback_locks_the_next_ordinary_frame_without_waiting_for_a_t_pose(self) -> None:
        """宿主给不出 T Pose 时的唯一出口：下一个完整帧直接成为新基准。"""
        current = [10.0]
        relay = self._calibrated_relay(current)
        result = self._service(
            relay, manage_host_output=False, worker_alive=False
        ).vmc_recalibrate(accept_current_pose=True)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["mode"], "accept_current_pose")
        self.assertFalse(relay.snapshot()["calibration"]["calibrated"])
        # 不等 T Pose，所以也不该把活儿丢给一个并不存在的校准线程。
        self.assertFalse(relay.snapshot()["calibration"]["waiting_for_t_pose_frame"])
        self.assertFalse(relay.needs_recalibration())

        current[0] += 1.0
        half_sqrt = 2.0 ** -0.5
        animated = [
            (address, (*arguments[:4], 0.0, 0.0, half_sqrt, half_sqrt))
            if address == "/VMC/Ext/Bone/Pos" and arguments[0] == "LeftUpperArm"
            else (address, arguments)
            for address, arguments in _complete_frame_messages()
        ]
        relay.ingest_messages(animated, now=current[0])
        self.assertIsNotNone(relay.latest_frame())
        self.assertTrue(relay.snapshot()["calibration"]["calibrated"])

    def test_disabled_relay_reports_instead_of_pretending_to_recalibrate(self) -> None:
        current = [10.0]
        relay = self._calibrated_relay(current)
        stub = self._service(relay)
        stub.config = SimpleNamespace(vmc_idle=VmcIdleConfig(enabled=False))
        result = stub.vmc_recalibrate()
        self.assertFalse(result["accepted"])
        self.assertIsNotNone(relay.latest_frame())


if __name__ == "__main__":
    unittest.main()
