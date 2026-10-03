from __future__ import annotations

import math
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import BodyProfile
from neko_anyadance_body.model import LEFT_CANONICAL_QUAT, RIGHT_CANONICAL_QUAT, neutral_frame, quat_norm_sq
from neko_anyadance_body.motion import (
    GESTURE_NAMES,
    apply_hand_pose,
    arm_pose_target,
    axis_angle,
    gesture_frame,
    move_hand_target,
    quat_multiply,
    reach_target,
)


def apply_quat(quat: tuple[float, float, float, float],
               vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """把四元数作用到向量上（q ⊗ v ⊗ q⁻¹）。用来做不依赖实现公式的物理判据。"""
    x, y, z, w = quat
    inverse = (-x, -y, -z, w)
    rotated = quat_multiply(quat_multiply(quat, (vec[0], vec[1], vec[2], 0.0)), inverse)
    return (rotated[0], rotated[1], rotated[2])


class MotionGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = BodyProfile()
        self.frame = neutral_frame()

    def arm(self, side: str, angle: float, plane: str):
        return arm_pose_target(
            self.frame,
            side=side,
            elevation_deg=angle,
            plane=plane,
            reach=1.0,
            palm="neutral",
            profile=self.profile,
        )

    def test_front_plane_angle_convention(self) -> None:
        down = self.arm("right", 0.0, "front").devices["right_controller"].position
        horizontal = self.arm("right", 90.0, "front").devices["right_controller"].position
        up = self.arm("right", 180.0, "front").devices["right_controller"].position
        self.assertAlmostEqual(down[0], 0.18, places=6)
        self.assertAlmostEqual(down[1], 0.74, places=6)
        self.assertAlmostEqual(horizontal[1], 1.32, places=6)
        self.assertAlmostEqual(horizontal[2], -0.58, places=6)
        self.assertAlmostEqual(up[1], 1.90, places=6)

    def test_side_plane_is_mirrored(self) -> None:
        result = self.arm("both", 90.0, "side")
        left = result.devices["left_controller"].position
        right = result.devices["right_controller"].position
        self.assertAlmostEqual(left[0], -right[0], places=6)
        self.assertAlmostEqual(left[1], right[1], places=6)
        self.assertAlmostEqual(left[2], right[2], places=6)
        self.assertAlmostEqual(right[0], 0.76, places=6)

    def test_hand_rotation_follows_arm_direction(self) -> None:
        t_pose = self.arm("both", 90.0, "side")
        for actual, expected in zip(t_pose.devices["left_controller"].rotation, LEFT_CANONICAL_QUAT):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(t_pose.devices["right_controller"].rotation, RIGHT_CANONICAL_QUAT):
            self.assertAlmostEqual(actual, expected, places=6)

        forward = self.arm("right", 90.0, "front")
        raised = self.arm("right", 180.0, "front")
        self.assertNotEqual(
            forward.devices["right_controller"].rotation,
            t_pose.devices["right_controller"].rotation,
        )
        self.assertNotEqual(
            raised.devices["right_controller"].rotation,
            forward.devices["right_controller"].rotation,
        )

    def test_azimuth_allows_arbitrary_horizontal_direction(self) -> None:
        rightward = arm_pose_target(
            self.frame,
            side="right",
            elevation_deg=90,
            azimuth_deg=90,
            reach=1.0,
            palm="neutral",
            profile=self.profile,
        ).devices["right_controller"].position
        diagonal = arm_pose_target(
            self.frame,
            side="right",
            elevation_deg=90,
            azimuth_deg=-45,
            reach=1.0,
            palm="neutral",
            profile=self.profile,
        ).devices["right_controller"].position
        self.assertAlmostEqual(rightward[0], 0.76, places=6)
        self.assertAlmostEqual(rightward[2], 0.0, places=6)
        self.assertLess(diagonal[0], 0.18)
        self.assertLess(diagonal[2], 0.0)

    def test_palm_presets_produce_unit_quaternions(self) -> None:
        for side in ("left", "right"):
            for palm in ("neutral", "forward", "down", "inward"):
                target = arm_pose_target(
                    self.frame,
                    side=side,
                    elevation_deg=120,
                    plane="front",
                    reach=0.9,
                    palm=palm,
                    profile=self.profile,
                )
                self.assertAlmostEqual(quat_norm_sq(target.devices[f"{side}_controller"].rotation), 1.0, places=6)

    def test_hand_grip_and_point(self) -> None:
        grip = apply_hand_pose(self.frame, side="right", pose="grip", strength=0.8)
        controller = grip.controllers["right_controller"]
        self.assertTrue(controller.grip_click)
        self.assertEqual(controller.grip_value, 0.8)
        self.assertTrue(all(value == 0.8 for value in controller.finger_bends.values()))

        point = apply_hand_pose(self.frame, side="left", pose="point", strength=1.0)
        fingers = point.controllers["left_controller"].finger_bends
        self.assertEqual(fingers["index"], 0.0)
        self.assertEqual(fingers["middle"], 1.0)
        self.assertFalse(point.controllers["left_controller"].grip_click)

    def test_reach_uses_semantic_height_and_direction(self) -> None:
        forward = reach_target(
            self.frame,
            side="right",
            height="chest",
            direction="forward",
            distance_m=0.35,
            profile=self.profile,
        ).devices["right_controller"].position
        outward = reach_target(
            self.frame,
            side="right",
            height="chest",
            direction="outward",
            distance_m=0.35,
            profile=self.profile,
        ).devices["right_controller"].position
        self.assertAlmostEqual(forward[1], 1.15, places=6)
        self.assertAlmostEqual(forward[2], -0.35, places=6)
        self.assertGreater(outward[0], forward[0])

    def test_move_hand_uses_body_anchor_and_wrist_euler(self) -> None:
        target = move_hand_target(
            self.frame,
            side="right",
            relative_to="chest",
            x_m=0.30,
            y_m=0.10,
            z_m=-0.40,
            palm="down",
            wrist_pitch_deg=15,
            wrist_yaw_deg=20,
            wrist_roll_deg=45,
        )
        self.assertEqual(target.devices["right_controller"].position, (0.30, 1.25, -0.40))
        self.assertAlmostEqual(quat_norm_sq(target.devices["right_controller"].rotation), 1.0, places=6)
        self.assertNotEqual(
            target.devices["right_controller"].rotation,
            self.frame.devices["right_controller"].rotation,
        )

    def test_move_hand_direction_changes_rotation(self) -> None:
        forward = move_hand_target(
            self.frame, side="right", relative_to="chest",
            x_m=0.18, y_m=0.0, z_m=-0.4, palm="neutral", profile=self.profile,
        )
        outward = move_hand_target(
            self.frame, side="right", relative_to="chest",
            x_m=0.58, y_m=0.0, z_m=0.0, palm="neutral", profile=self.profile,
        )
        self.assertNotEqual(
            forward.devices["right_controller"].rotation,
            outward.devices["right_controller"].rotation,
        )

    def test_head_gestures_rotate_about_the_head_local_axis(self) -> None:
        """头部手势必须绕**头自身**的轴转，不是绕世界轴——否则角色一转身，点头就变侧倾。

        判据用物理不变量：绕哪个轴转，那个轴在世界系的方向就不变。两个细节很关键：

        * base 必须给非零朝向。IDENTITY_QUAT 下前乘与后乘等价，改不改都看不出来
          （现有用例的 base 全是 identity，所以这条是唯一能验它的）；
        * progress 必须落在手势幅度非零处。nod 的包络是 sin(2πp)²，在 p=0.5 处
          恰好为 0；shake_head 的 sin(4πp) 在 p=0.25/0.5 也都是 0——取到零点这条
          会变成恒真。
        """
        for name, axis, progress, base_axis, base_deg in (
            ("nod", (1.0, 0.0, 0.0), 0.25, (0.0, 1.0, 0.0), 90.0),
            ("bow", (1.0, 0.0, 0.0), 0.50, (0.0, 1.0, 0.0), 90.0),
            ("sigh", (1.0, 0.0, 0.0), 0.50, (0.0, 1.0, 0.0), 90.0),
            # 摇头绕 Y：base 只有 yaw 时头的本地 Y 恰好等于世界 Y，前乘后乘同解，
            # 判据天然区分不了。改用俯仰 base 让它成为有效用例。
            ("shake_head", (0.0, 1.0, 0.0), 0.375, (1.0, 0.0, 0.0), 30.0),
        ):
            base = neutral_frame()
            base.devices["hmd"].rotation = axis_angle(base_axis, base_deg)
            before = apply_quat(base.devices["hmd"].rotation, axis)
            moved = gesture_frame(base, name=name, side="right", intensity=1.0,
                                  progress=progress, profile=self.profile)
            after = apply_quat(moved.devices["hmd"].rotation, axis)
            for actual, expected in zip(after, before):
                self.assertAlmostEqual(actual, expected, places=6,
                                       msg=f"{name} 把旋转施加在了世界系而不是头本地系")

    def test_gestures_restore_start_frame(self) -> None:
        for name in GESTURE_NAMES:
            middle = gesture_frame(
                self.frame,
                name=name,
                side="right",
                intensity=0.8,
                progress=0.5,
                profile=self.profile,
            )
            self.assertTrue(all(math.isfinite(value) for device in middle.devices.values() for value in (*device.position, *device.rotation)))
            end = gesture_frame(
                self.frame,
                name=name,
                side="right",
                intensity=0.8,
                progress=1.0,
                profile=self.profile,
            )
            for device in self.frame.devices:
                for actual, expected in zip(end.devices[device].position, self.frame.devices[device].position):
                    self.assertAlmostEqual(actual, expected, places=12)
                for actual, expected in zip(end.devices[device].rotation, self.frame.devices[device].rotation):
                    self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
