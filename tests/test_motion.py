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
    gesture_frame,
    move_hand_target,
    reach_target,
)


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
