from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import BodyProfile
from neko_anyadance_body.model import neutral_frame
from neko_anyadance_body.expression_motion import (
    ExpressionOverlay,
    apply_expression_overlay,
    sample_expression,
)


class ExpressionMotionTests(unittest.TestCase):
    def test_overlay_is_relative_and_never_changes_controller_clicks(self) -> None:
        reference = neutral_frame()
        base = neutral_frame()
        base.devices["right_controller"].position = (0.50, 1.20, -0.20)
        base.controllers["right_controller"].grip_click = True
        base.controllers["right_controller"].grip_value = 1.0
        overlay = ExpressionOverlay(
            action_id="a",
            gesture="offer",
            side="right",
            energy=0.5,
            started_at=0.0,
            duration_s=1.0,
            reference=reference,
        )
        sampled, channels, weight = sample_expression(overlay, 0.55, BodyProfile())
        result = apply_expression_overlay(base, reference, sampled, channels, weight)
        self.assertNotEqual(result.devices["right_controller"].position, base.devices["right_controller"].position)
        self.assertEqual(result.devices["left_controller"].position, base.devices["left_controller"].position)
        self.assertTrue(result.controllers["right_controller"].grip_click)
        self.assertEqual(result.controllers["right_controller"].grip_value, 1.0)

    def test_gesture_returns_to_reference(self) -> None:
        reference = neutral_frame()
        overlay = ExpressionOverlay(
            action_id="a",
            gesture="nod",
            side="right",
            energy=0.4,
            started_at=0.0,
            duration_s=1.0,
            reference=reference,
        )
        sampled, channels, _ = sample_expression(overlay, 1.0, BodyProfile())
        result = apply_expression_overlay(reference, reference, sampled, channels, 1.0)
        self.assertEqual(result.devices["hmd"].position, reference.devices["hmd"].position)
        for actual, expected in zip(result.devices["hmd"].rotation, reference.devices["hmd"].rotation):
            self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
