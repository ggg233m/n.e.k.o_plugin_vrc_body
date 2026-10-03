from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import BodyProfile
from neko_anyadance_body.model import neutral_frame
from neko_anyadance_body.motion import axis_angle, quat_multiply
from neko_anyadance_body.expression_motion import (
    ExpressionOverlay,
    apply_expression_overlay,
    sample_expression,
)


def apply_quat(quat: tuple[float, float, float, float],
               vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """把四元数作用到向量上（q ⊗ v ⊗ q⁻¹）。用物理不变量做判据，不照抄实现公式。"""
    x, y, z, w = quat
    inverse = (-x, -y, -z, w)
    rotated = quat_multiply(quat_multiply(quat, (vec[0], vec[1], vec[2], 0.0)), inverse)
    return (rotated[0], rotated[1], rotated[2])


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

    def test_head_expression_delta_is_head_local_not_world(self) -> None:
        """表情叠加的 delta 必须是**头部本地**增量。

        这是全链路用例（_phase_frames → sample_expression → apply_expression_overlay）：
        `delta = gesture ⊗ R⁻¹` 与 `R⁻¹ ⊗ gesture` 必须与 _phase_frames 的乘法顺序
        **成对**翻转，只改一半的话叠加结果仍旧按世界系解释。

        ⚠️ base 必须与 reference **不同**：两者相同时 delta 的两种取法结果恰好相等
        （delta⊗B 与 B⊗D 都化简成 R⊗axis），这条会变成恒真。现实里 base 是当前帧、
        reference 是手势起始帧，本来就不是一个姿态。
        """
        reference = neutral_frame()
        reference.devices["hmd"].rotation = axis_angle((0.0, 1.0, 0.0), 90.0)
        base = neutral_frame()
        base.devices["hmd"].rotation = axis_angle((0.0, 1.0, 0.0), 10.0)     # 当前帧朝向
        before = apply_quat(base.devices["hmd"].rotation, (1.0, 0.0, 0.0))   # base 的本地 X 在世界系的方向
        overlay = ExpressionOverlay(
            action_id="a", gesture="nod", side="right", energy=0.9,
            started_at=0.0, duration_s=1.0, reference=reference,
        )
        # 0.48 之后进 stroke 段，幅度最大
        sampled, channels, weight = sample_expression(overlay, 0.62, BodyProfile())
        result = apply_expression_overlay(base, reference, sampled, channels, weight)
        after = apply_quat(result.devices["hmd"].rotation, (1.0, 0.0, 0.0))
        for actual, expected in zip(after, before):
            self.assertAlmostEqual(actual, expected, places=6,
                                   msg="表情 delta 仍是世界系增量")

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
