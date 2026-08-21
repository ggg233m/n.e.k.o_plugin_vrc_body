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


class NavigatorStallTests(unittest.TestCase):
    """「卡墙不自知」：检测器只看画面，永远不会报告前面有堵墙。"""

    def setUp(self) -> None:
        self.world = {
            "available": True,
            "uncertainties": [],
            "status": {"revision": 3, "last_observation_age_ms": 80},
            "entities": [{
                "id": "vision:person:1",
                "label": "person",
                "confidence": 0.92,
                "visible": True,
                "attributes": {"bearing_deg": 0.0, "apparent_height": 0.2},
            }],
        }
        self.goal = {
            "state": "armed",
            "goal": {"kind": "approach", "text": "walk to the person", "age_seconds": 1.0},
        }
        self.motion: dict[str, object] = {"available": True, "horizontal_speed_mps": 0.9}
        self.sent: list[tuple[str, float, float, int]] = []
        self.released: list[str] = []

    def _navigator(self, *, motion_provider=..., **overrides) -> LocalNavigator:
        return LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=lambda side, x, y, pulse: self.sent.append((side, x, y, pulse)) or True,
            release_inputs=lambda side: self.released.append(side),
            motion_provider=(lambda: self.motion) if motion_provider is ... else motion_provider,
            config=NavigatorConfig(stall_ticks=3, **overrides),
        )

    def test_moving_forward_never_trips_the_stall_guard(self) -> None:
        navigator = self._navigator()
        for _ in range(10):
            self.assertEqual(navigator.tick().state, "advance")
        self.assertFalse(navigator.snapshot()["stall"]["stalled"])

    def test_zero_velocity_while_advancing_stops_and_latches(self) -> None:
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        self.assertEqual(navigator.tick().state, "advance")
        self.assertEqual(navigator.tick().state, "advance")
        decision = navigator.tick()
        self.assertEqual(decision.state, "stop")
        self.assertEqual(decision.reason, "movement_stalled")
        self.assertEqual(self.released, ["all"])

        # 闩锁：停下之后速度当然还是 0，靠速度本身永远解不开。必须由换目标解除，
        # 否则导航器会在「停车 → 速度为零 → 继续判定卡住」里空转。
        self.assertEqual(navigator.tick().reason, "movement_stalled")
        snapshot = navigator.snapshot()
        self.assertTrue(snapshot["stall"]["stalled"])
        self.assertEqual(snapshot["stall"]["stall_count"], 1)

    def test_new_goal_clears_the_latch(self) -> None:
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(3):
            navigator.tick()
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])

        self.goal["goal"] = {"kind": "follow", "text": "follow the person", "age_seconds": 0.5}
        self.motion["horizontal_speed_mps"] = 0.9
        self.assertEqual(navigator.tick().state, "advance")
        self.assertFalse(navigator.snapshot()["stall"]["stalled"])

    def test_resubmitting_the_same_goal_also_clears_the_latch(self) -> None:
        # 同一句目标重新提交时文本不变，只有 age_seconds 会倒退。若不认这个信号，
        # 用户「再试一次」会被永远拒绝。
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(3):
            navigator.tick()
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])

        self.goal["goal"] = {"kind": "approach", "text": "walk to the person", "age_seconds": 0.1}
        self.motion["horizontal_speed_mps"] = 0.9
        self.assertEqual(navigator.tick().state, "advance")

    def test_missing_velocity_feedback_never_blocks_movement(self) -> None:
        # 收不到内置参数时「有没有卡住」不可知。把不可知当成卡住，等于在没配
        # 这些参数的 avatar 上直接废掉导航。
        navigator = self._navigator(motion_provider=None)
        for _ in range(10):
            self.assertEqual(navigator.tick().state, "advance")
        stall = navigator.snapshot()["stall"]
        self.assertFalse(stall["detectable"])
        self.assertFalse(stall["stalled"])

    def test_unavailable_motion_report_never_blocks_movement(self) -> None:
        self.motion = {"available": False, "reason": "velocity_parameters_absent"}
        navigator = self._navigator()
        for _ in range(10):
            self.assertEqual(navigator.tick().state, "advance")
        self.assertFalse(navigator.snapshot()["stall"]["detectable"])

    def test_turning_in_place_does_not_reset_the_stall_counter(self) -> None:
        # 顶着墙时导航器会在 advance/turn 之间抖动。若 turn 清零计数，卡墙就
        # 永远攒不够连续 tick，判据形同虚设。
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        self.assertEqual(navigator.tick().state, "advance")
        self.assertEqual(navigator.tick().state, "advance")

        self.world["entities"][0]["attributes"]["bearing_deg"] = 30.0
        self.assertEqual(navigator.tick().state, "turn")

        self.world["entities"][0]["attributes"]["bearing_deg"] = 0.0
        self.assertEqual(navigator.tick().reason, "movement_stalled")

    def test_config_rejects_unbounded_stall_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            NavigatorConfig(stall_speed_mps=5.0)
        with self.assertRaises(ValueError):
            NavigatorConfig(stall_ticks=1)


if __name__ == "__main__":
    unittest.main()
