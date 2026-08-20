from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.navigator import LocalNavigator, NavigatorConfig


class NavigatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = {
            "available": True,
            "uncertainties": [],
            "status": {"revision": 3, "last_observation_age_ms": 80},
            "entities": [{
                "id": "vision:door:1",
                "label": "door",
                "confidence": 0.92,
                "visible": True,
                "attributes": {"bearing_deg": 0.0, "distance_m": 4.0},
            }],
        }
        self.goal = {
            "state": "armed",
            "goal": {"kind": "approach", "text": "approach the door", "age_seconds": 1.0},
        }
        self.sent: list[tuple[str, float, float, int]] = []
        self.released: list[str] = []
        self.navigator = LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=lambda side, x, y, pulse: self.sent.append((side, x, y, pulse)) or True,
            release_inputs=lambda side: self.released.append(side),
        )

    def test_centered_target_emits_bounded_forward_pulse(self) -> None:
        decision = self.navigator.tick()
        self.assertEqual(decision.state, "advance")
        self.assertEqual(self.sent[0][0], "left")
        self.assertGreater(self.sent[0][3], 0)
        self.assertLessEqual(self.sent[0][2], 0.28)
        self.assertGreater(self.sent[0][2], 0.0)

    def test_off_center_target_turns_without_forward_motion(self) -> None:
        self.world["entities"][0]["attributes"]["bearing_deg"] = 30.0
        decision = self.navigator.tick()
        self.assertEqual(decision.state, "turn")
        self.assertEqual(self.sent[0][0], "right")
        self.assertGreater(self.sent[0][1], 0.0)
        self.assertEqual(self.sent[0][2], 0.0)

    def test_unknown_or_stale_world_releases_active_axis(self) -> None:
        self.navigator.tick()
        self.assertEqual(self.navigator.snapshot()["active_side"], "left")
        self.world["uncertainties"] = ["depth_unknown"]
        decision = self.navigator.tick()
        self.assertEqual(decision.state, "stop")
        self.assertEqual(decision.reason, "world_uncertain")
        self.assertEqual(self.released, ["all"])
        self.assertIsNone(self.navigator.snapshot()["active_side"])

    def test_capability_boundary_uncertainties_do_not_block_movement(self) -> None:
        # 检测器正常工作时会一直报告没有深度/OCR。如果这些也阻断移动，那么
        # 检测器越正常，导航越死——必须只对真正可疑的观测停车。
        self.world["uncertainties"] = ["depth_unavailable", "ocr_unavailable"]
        decision = self.navigator.tick()
        self.assertEqual(decision.state, "advance")

    def test_unknown_uncertainty_codes_still_block(self) -> None:
        # 白名单之外的一切默认阻断：以后新增检测器不会意外放松安全边界。
        for code in ("world_switched", "observation_stale", "concurrent_sender", "brand_new_code"):
            with self.subTest(code=code):
                navigator = LocalNavigator(
                    world_provider=lambda: {**self.world, "uncertainties": [code]},
                    goal_provider=lambda: self.goal,
                    send_axes=lambda side, x, y, pulse: True,
                    release_inputs=lambda side: None,
                )
                decision = navigator.tick()
                self.assertEqual(decision.state, "stop")
                self.assertEqual(decision.reason, "world_uncertain")

    def test_apparent_height_drives_approach_without_metric_depth(self) -> None:
        # 二维检测器不发布 distance_m；表观高度必须能独立闭环，否则前进永不触发。
        attributes = self.world["entities"][0]["attributes"]
        attributes.pop("distance_m")
        attributes["apparent_height"] = 0.2
        decision = self.navigator.tick()
        self.assertEqual(decision.state, "advance")
        self.assertGreater(self.sent[0][2], 0.0)
        self.assertLessEqual(self.sent[0][2], 0.28)

        attributes["apparent_height"] = 0.6
        self.assertEqual(self.navigator.tick().state, "reached")

    def test_edge_clipped_target_reaches_instead_of_walking_closer(self) -> None:
        # 目标贴边时表观高度封顶，若仍按普通阈值判定就会一直前进撞上对方。
        attributes = self.world["entities"][0]["attributes"]
        attributes.pop("distance_m")
        attributes["apparent_height"] = 0.4
        attributes["apparent_height_clipped"] = True
        decision = self.navigator.tick()
        self.assertEqual(decision.state, "reached")
        self.assertEqual(self.sent, [])

    def test_metric_distance_still_works_for_depth_adapters(self) -> None:
        # 注入式深度适配器提供真实米制距离时，原路径必须保持可用。
        self.world["entities"][0]["attributes"]["distance_m"] = 0.5
        self.assertEqual(self.navigator.tick().state, "reached")

    def test_bbox_only_detector_can_still_close_the_approach_loop(self) -> None:
        self.world["entities"][0]["attributes"] = {"bearing_deg": 0.0}
        self.world["entities"][0]["bbox"] = [0.4, 0.4, 0.6, 0.6]
        self.assertEqual(self.navigator.tick().state, "advance")

    def test_missing_target_never_blindly_moves(self) -> None:
        self.world["entities"] = []
        decision = self.navigator.tick()
        self.assertEqual(decision.reason, "target_not_visible")
        self.assertEqual(self.sent, [])
        self.assertEqual(self.released, [])

    def test_config_rejects_unbounded_speed(self) -> None:
        with self.assertRaises(ValueError):
            NavigatorConfig(max_forward_axis=0.9)


if __name__ == "__main__":
    unittest.main()
