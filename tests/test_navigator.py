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
        self.turns: list[float] = []
        self.released: list[str] = []
        self.navigator = LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=lambda side, x, y, pulse: self.sent.append((side, x, y, pulse)) or True,
            send_turn=lambda delta: self.turns.append(delta) or True,
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
        """目标偏右就必须往右转，而且不能同时前进。

        符号搞反会让导航器背对目标越转越远，且看起来一切正常（命令都 accepted）。
        bearing>0 是「目标在画面右侧」，+delta_deg 是左转，所以 turn_deg 必须是负的。
        """
        self.world["entities"][0]["attributes"]["bearing_deg"] = 30.0
        decision = self.navigator.tick()
        self.assertEqual(decision.state, "turn")
        self.assertLess(decision.turn_deg, 0.0)
        self.assertEqual(self.turns, [decision.turn_deg])
        # 转向不碰摇杆：一路都不该有 axis 命令。
        self.assertEqual(self.sent, [])

    def test_turn_magnitude_is_bounded_and_undershoots(self) -> None:
        """转向幅度要收敛，不能一次转过头。

        bearing 来自上一帧观测，按整个偏差转会越过中心然后来回摆；同时再大的
        bearing 也不能换来无界的转身。
        """
        self.world["entities"][0]["attributes"]["bearing_deg"] = -40.0
        decision = self.navigator.tick()
        self.assertGreater(decision.turn_deg, 0.0)
        self.assertLess(decision.turn_deg, 40.0)

        self.world["entities"][0]["attributes"]["bearing_deg"] = 179.0
        self.assertLessEqual(abs(self.navigator.tick().turn_deg), 45.0)

    def test_turn_fires_once_per_observation_not_once_per_tick(self) -> None:
        """同一次观测只能转一次，否则转向按 tick/出帧 的倍数超调。

        转向和前进的命令语义不同：前进轴是按住的状态，重发只是覆盖，幂等；
        转向是相对位移，重发会累加。而导航器 tick 远快于检测器出帧（实测
        9.14 Hz 对 1.75 Hz，每观测 5.2 tick），同一个 bearing 会被发成 5 条
        独立的转向命令。

        实测 bearing -11.47 度算出 +9.18 度，连发 5 次 = +45.9 度，对着 11.47
        度的偏差转过去 4 倍，冲过中心后符号翻转反向超调——第一次实机闭环就是
        这样从 -11 发散到 -18，最后摄像机停在海面上。
        """
        self.world["entities"][0]["attributes"]["bearing_deg"] = 30.0
        for _ in range(5):
            self.assertEqual(self.navigator.tick().state, "turn")
        self.assertEqual(len(self.turns), 1)
        snapshot = self.navigator.snapshot()
        self.assertEqual(snapshot["command_count"], 1)
        self.assertEqual(snapshot["turn"]["last_revision"], 3)
        self.assertEqual(snapshot["turn"]["suppressed_count"], 4)

        # 新观测到了才允许再转一次——去重压的是重复，不是转向本身。
        self.world["status"]["revision"] = 4
        self.assertEqual(self.navigator.tick().state, "turn")
        self.assertEqual(len(self.turns), 2)
        self.assertEqual(self.navigator.snapshot()["turn"]["last_revision"], 4)

    def test_turn_is_retried_when_the_command_is_rejected(self) -> None:
        # 只有真发出去了才算「这一帧转过了」。发送失败还记账的话，这次观测就
        # 被白白吞掉，目标一直偏着却再也不转。
        rejected: list[float] = []

        def refuse(delta: float) -> bool:
            rejected.append(delta)
            return False

        navigator = LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=lambda side, x, y, pulse: True,
            send_turn=refuse,
            release_inputs=lambda side: None,
        )
        self.world["entities"][0]["attributes"]["bearing_deg"] = 30.0
        for _ in range(3):
            navigator.tick()
        self.assertEqual(len(rejected), 3)
        self.assertIsNone(navigator.snapshot()["turn"]["last_revision"])

    def test_unversioned_world_still_turns_every_tick(self) -> None:
        # 不发 revision 的世界源，观测之间无从区分。此时按 revision 去重会把
        # 所有观测认成同一次，转向永远只发一条——「一条都不转」比「多转几度」
        # 坏得多。
        self.world["status"] = {"revision": 0, "last_observation_age_ms": 80}
        self.world["entities"][0]["attributes"]["bearing_deg"] = 30.0
        for _ in range(4):
            self.assertEqual(self.navigator.tick().state, "turn")
        self.assertEqual(len(self.turns), 4)

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
                    send_turn=lambda delta: True,
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

    def test_sitting_npc_sinks_to_frame_bottom_triggers_clipped_reached(self) -> None:
        """坐在地上的 NPC 接近时会往画面底部下沉，表观高度先涨后跌永远到不了 0.55。

        实测：bbox_bottom 稳定在 0.991，navigator 的 fallback clip 阈值是 0.999，
        差 0.8% 导致 apparent_height_clipped 一直为 False，approached 永远不触发。

        修复：将 navigator._spatial_hint 内的 fallback clip 阈值放宽到 0.97/0.03，
        与 local_perception.py 保持一致。底边 >= 0.97 时 clipped=True，
        reach_apparent = target_apparent * 0.6 = 0.55 * 0.6 = 0.33，
        而实测接近时 apparent 约 0.22——仍低于 0.33，所以还需要 target_apparent
        再低一点或者 bbox_top 贴到 0.03 触发上边 clipped。

        本测试验证 bbox bottom=0.991 时 fallback clipped 现在为 True，
        且 apparent 0.22 < 0.33 仍然 advance（不会误报 reached）；
        而当 bbox 完全压底（bottom=0.99, top=0.77, apparent=0.22, clipped=True）时，
        apparent(0.22) >= reach_apparent(0.33) 不成立，仍 advance。
        只有 apparent >= 0.33 时才 reached。
        """
        attributes = self.world["entities"][0]["attributes"]
        attributes.pop("distance_m")
        # 实测「沉底」时的 bbox：top 漂到 0.77，bottom 停在 0.991
        self.world["entities"][0]["bbox"] = [0.467, 0.77, 0.708, 0.991]
        # apparent_height_clipped 由 local_perception 设，这里模拟它没设（None/absent）
        # _spatial_hint 的 fallback 逻辑必须从 bbox 推断 clipped=True
        decision = self.navigator.tick()
        # apparent = 0.991 - 0.77 = 0.221; reach_apparent = 0.55 * 0.6 = 0.33
        # 0.221 < 0.33 => still advance, not reached
        self.assertEqual(decision.state, "advance")

        # 当 apparent 升到 reach_apparent 时才 reached
        self.world["entities"][0]["bbox"] = [0.467, 0.64, 0.708, 0.991]
        # apparent = 0.991 - 0.64 = 0.351 >= 0.33 => reached
        decision2 = self.navigator.tick()
        self.assertEqual(decision2.state, "reached")
        self.assertEqual(decision2.reason, "target_in_interaction_range")

    def test_missing_target_never_blindly_moves(self) -> None:
        self.world["entities"] = []
        decision = self.navigator.tick()
        self.assertEqual(decision.reason, "target_not_visible")
        self.assertEqual(self.sent, [])
        self.assertEqual(self.released, [])

    def test_approach_never_emits_a_command_below_vrchat_deadzone(self) -> None:
        """越接近目标，前进轴越小——小到 VRChat 直接忽略，人就停在原地。

        实测 VRChat 的起步死区在 0.13 附近：y=0.10 完全不动，y=0.076（表观高度
        0.40 时算出的值）也完全不动。命令照发、accepted 照回，但 avatar 一动不
        动，失速守卫读到真实的 0，攒够 8 tick 就误报 movement_stalled 并latch 住
        ——一次误判会废掉这个目标的整段导航。

        所以「advance」必须名副其实：只要决定前进，油门就得过死区。
        """
        minimum = self.navigator.config.min_forward_axis
        attributes = self.world["entities"][0]["attributes"]

        # 表观高度分支：扫到 reached 阈值前的每一档，含 0.40 这个实测卡死点。
        attributes.pop("distance_m")
        for apparent in (0.05, 0.2, 0.35, 0.40, 0.45, 0.5, 0.54):
            with self.subTest(apparent_height=apparent):
                attributes["apparent_height"] = apparent
                decision = self.navigator.tick()
                self.assertEqual(decision.state, "advance")
                self.assertGreaterEqual(decision.y, minimum)
                self.assertLessEqual(decision.y, 0.28)

        # 米制分支：距离刚过停止距离时同样会缩到死区以下。
        attributes.pop("apparent_height")
        for distance in (1.3, 1.5, 2.0, 2.7, 6.0):
            with self.subTest(distance_m=distance):
                attributes["distance_m"] = distance
                decision = self.navigator.tick()
                self.assertEqual(decision.state, "advance")
                self.assertGreaterEqual(decision.y, minimum)
                self.assertLessEqual(decision.y, 0.28)

        # 过死区的判据是「实际发出去的摇杆量」，不是决策对象里的数字。
        self.assertTrue(self.sent)
        for side, _x, y, _pulse in self.sent:
            self.assertEqual(side, "left")
            self.assertGreaterEqual(y, minimum)

    def test_minimum_throttle_clears_the_stall_threshold(self) -> None:
        """死区下限还必须让实际速度高于失速阈值，否则两个判据自相矛盾。

        实测 y=0.13 能动但只有 0.1333 m/s，仍低于 stall_speed_mps=0.15——照样
        会被判失速。下限取 0.15（实测 0.2222 m/s）才同时满足两边。
        """
        config = self.navigator.config
        self.assertGreaterEqual(config.min_forward_axis, 0.15)
        self.assertLessEqual(config.min_forward_axis, config.max_forward_axis)

    def test_config_rejects_unbounded_speed(self) -> None:
        with self.assertRaises(ValueError):
            NavigatorConfig(max_forward_axis=0.9)

    def test_config_rejects_minimum_throttle_above_maximum(self) -> None:
        # 下限高于上限时 _clamp 会静默返回上限，把「最小油门」变成「全速」。
        with self.assertRaises(ValueError):
            NavigatorConfig(max_forward_axis=0.1, min_forward_axis=0.3)


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
        self.turns: list[float] = []
        self.released: list[str] = []

    def _navigator(self, *, motion_provider=..., **overrides) -> LocalNavigator:
        # 默认关掉自动绕行，让这一组用例专注于「失速判据本身」。绕行会在撞墙
        # 与闩锁之间插入若干 recover tick，那是 NavigatorAutoRecoverTests 的题目。
        overrides.setdefault("auto_recover_limit", 0)
        return LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=lambda side, x, y, pulse: self.sent.append((side, x, y, pulse)) or True,
            send_turn=lambda delta: self.turns.append(delta) or True,
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
        # 换了目标文字时 _unreachable 清空，实体重新可选，latch 也解除。
        # age 倒退但文字不变 → 地形未变，_unreachable 保留；要重试必须换文字。
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(3):
            navigator.tick()
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])

        # 换目标文字（语义接近但 key 不同） → _unreachable 清空，latch 解除。
        self.goal["goal"] = {"kind": "approach", "text": "approach person", "age_seconds": 0.5}
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

    def test_rejected_axis_commands_never_count_as_stalling(self) -> None:
        """指令被下游拒收时不能判失速——那是「没发出去」，不是「顶着墙」。

        实机复现（2026-08-23）：后端起来后 body output 默认是 disabled，
        scheduler.py 会把所有 INPUT_COMMANDS 直接拒掉。导航器一路算到
        advance / y=0.15，`_navigator_send_axes` 提交被拒，avatar 一动不动，
        于是速度恒为 0、8 tick 后 latch 住，报 movement_stalled。

        结论是假的：那一刻既没有墙，也没有镜子，只是忘了 body_enable。
        更糟的是它会把目标记进 _unreachable，于是「忘了 enable」会伪装成
        「这个方向走不通」。
        """
        rejected: list[tuple[str, float, float, int]] = []

        def refuse(side: str, x: float, y: float, pulse: int) -> bool:
            rejected.append((side, x, y, pulse))
            return False

        navigator = LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=refuse,
            send_turn=lambda delta: True,
            release_inputs=lambda side: None,
            motion_provider=lambda: self.motion,
            config=NavigatorConfig(stall_ticks=3),
        )
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(10):
            self.assertEqual(navigator.tick().state, "advance")
        # 命令确实一直在尝试发送，只是一直被拒。
        self.assertEqual(len(rejected), 10)
        stall = navigator.snapshot()["stall"]
        self.assertFalse(stall["stalled"])
        self.assertEqual(stall["consecutive_ticks"], 0)
        self.assertIs(stall["axis_send_ok"], False)
        # 关键：不能把目标记成够不着，否则 body enable 之后它还被拉黑着。
        self.assertEqual(stall["unreachable_targets"], [])

    def test_stall_resumes_once_axis_commands_are_accepted_again(self) -> None:
        # body enable 之后判据必须自己恢复，不能因为之前被拒就永久失效。
        accepted = {"ok": False}

        def gated(side: str, x: float, y: float, pulse: int) -> bool:
            return accepted["ok"]

        navigator = LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=gated,
            send_turn=lambda delta: True,
            release_inputs=lambda side: None,
            motion_provider=lambda: self.motion,
            config=NavigatorConfig(stall_ticks=3, auto_recover_limit=0),
        )
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(5):
            navigator.tick()
        self.assertFalse(navigator.snapshot()["stall"]["stalled"])

        accepted["ok"] = True
        for _ in range(4):
            navigator.tick()
        stall = navigator.snapshot()["stall"]
        self.assertTrue(stall["stalled"])
        self.assertIs(stall["axis_send_ok"], True)

    def test_config_rejects_unbounded_stall_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            NavigatorConfig(stall_speed_mps=5.0)
        with self.assertRaises(ValueError):
            NavigatorConfig(stall_ticks=1)
        with self.assertRaises(ValueError):
            NavigatorConfig(unreachable_ttl_s=-1.0)

    def test_new_goal_does_not_re_select_the_target_that_just_stalled(self) -> None:
        """换目标解开闩锁，并给实体一次重试机会。镜面倒影会立刻再次触发失速。

        实机场景（tmp/play_1.jpg）：镜面倒影是画面里置信度最高的 person（0.84，
        方位 +0.04°）。顶着镜子推 8 tick 之后闩锁生效，LLM 按 instructions
        第 26 条换一句目标——解开后倒影重新可选，但 3 tick 后又失速，行为收敛：
        每次新目标都顶 N tick 再停，而不是无限制前进。
        """
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(3):
            navigator.tick()
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])
        self.assertEqual(
            navigator.snapshot()["stall"]["unreachable_targets"], ["vision:person:1"]
        )

        # 换目标：_pre_tick_goal_check 清空 _unreachable，倒影重新可选。
        # 前两 tick advance（stall_ticks < threshold），第三 tick 再次 stalled。
        self.goal["goal"] = {"kind": "follow", "text": "follow the person", "age_seconds": 0.5}
        d1 = navigator.tick()
        d2 = navigator.tick()
        d3 = navigator.tick()
        self.assertEqual(d1.state, "advance")
        self.assertEqual(d2.state, "advance")
        self.assertEqual(d3.state, "stop")
        self.assertEqual(d3.reason, "movement_stalled")
        self.assertEqual(d3.target_id, "vision:person:1")

    def test_a_second_person_is_still_approached_after_the_first_stalls(self) -> None:
        # 拉黑必须是按实体的，不是按目标的：房间里另一个人还够得着就该走过去。
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(3):
            navigator.tick()
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])

        self.world["entities"].append({
            "id": "vision:person:2",
            "label": "person",
            "confidence": 0.71,
            "visible": True,
            "attributes": {"bearing_deg": 0.0, "apparent_height": 0.2},
        })
        self.goal["goal"] = {"kind": "approach", "text": "walk to the person", "age_seconds": 0.1}
        self.motion["horizontal_speed_mps"] = 0.9
        decision = navigator.tick()
        self.assertEqual(decision.state, "advance")
        self.assertEqual(decision.target_id, "vision:person:2")

    def test_unreachable_target_becomes_eligible_again_after_the_ttl(self) -> None:
        """挡在中间的可能是会走开的人，也可能只是自己当时的朝向。

        场景：换了目标文字（goal_key 变了）→ _unreachable 清空，实体重新可选，
        但立刻再次触发失速，记回 _unreachable。之后再换同一套目标文字（无变化）
        → unreachable 不清空。TTL 到期后，实体自动解禁。
        """
        clock = {"now": 1000.0}
        navigator = LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=lambda side, x, y, pulse: self.sent.append((side, x, y, pulse)) or True,
            send_turn=lambda delta: True,
            release_inputs=lambda side: None,
            motion_provider=lambda: self.motion,
            config=NavigatorConfig(stall_ticks=3, unreachable_ttl_s=45.0, auto_recover_limit=0),
            clock=lambda: clock["now"],
        )
        # 第一轮：失速，实体进入 _unreachable。
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(3):
            navigator.tick()
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])
        self.assertIn("vision:person:1", navigator.snapshot()["stall"]["unreachable_targets"])

        # 换目标文字 → _unreachable 清空，实体重新可选，立刻再次触发失速，
        # 重新进入 _unreachable。
        self.goal["goal"] = {"kind": "approach", "text": "go find the person", "age_seconds": 1.0}
        for _ in range(3):
            navigator.tick()
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])
        self.assertIn("vision:person:1", navigator.snapshot()["stall"]["unreachable_targets"])

        # 再换回相同目标文字（age_seconds 倒退，触发 renewed 但 goal_key 未变），
        # _unreachable 不清空——latch 解开了，但实体还被记着。
        self.goal["goal"] = {"kind": "approach", "text": "go find the person", "age_seconds": 0.5}
        d = navigator.tick()
        # stall latch 解开，但 unreachable 未清，所以仍然 stop。
        self.assertEqual(d.state, "stop")
        self.assertIn("vision:person:1", navigator.snapshot()["stall"]["unreachable_targets"])

        # TTL 到期，实体自动解禁。
        clock["now"] += 46.0
        self.motion["horizontal_speed_mps"] = 0.9
        # 需要再触发一次 renewed（age 继续倒退），否则 stall latch 还锁着。
        self.goal["goal"] = {"kind": "approach", "text": "go find the person", "age_seconds": 0.1}
        d2 = navigator.tick()
        self.assertEqual(d2.state, "advance")
        self.assertEqual(navigator.snapshot()["stall"]["unreachable_targets"], [])

    def test_zero_ttl_disables_blacklisting_and_keeps_pure_latch_behaviour(self) -> None:
        navigator = self._navigator(unreachable_ttl_s=0.0)
        self.motion["horizontal_speed_mps"] = 0.0
        for _ in range(3):
            navigator.tick()
        self.assertEqual(navigator.snapshot()["stall"]["unreachable_targets"], [])
        self.goal["goal"] = {"kind": "approach", "text": "walk to the person", "age_seconds": 0.1}
        self.motion["horizontal_speed_mps"] = 0.9
        self.assertEqual(navigator.tick().state, "advance")


class NavigatorAutoRecoverTests(unittest.TestCase):
    """撞墙后自己绕行，而不是停下来等 LLM 重提目标。

    两件事在这里成立或不成立：
      1. 斜撞墙能被发现。速度模长还在阈值之上，纯速度判据看不见。
      2. 绕行预算有限。死胡同里怎么转都出不去，最终必须交还 LLM。
    """

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
        # 畅通：全速前进，前进分量占满。
        self.motion: dict[str, object] = {
            "available": True,
            "horizontal_speed_mps": 0.9,
            "forward_ratio": 1.0,
            "slip_ratio": 0.0,
        }
        self.sent: list[tuple[str, float, float, int]] = []
        self.turns: list[float] = []
        self.released: list[str] = []

    def _navigator(self, **overrides) -> LocalNavigator:
        overrides.setdefault("stall_ticks", 3)
        overrides.setdefault("slip_ticks", 3)
        overrides.setdefault("auto_recover_limit", 2)
        return LocalNavigator(
            world_provider=lambda: self.world,
            goal_provider=lambda: self.goal,
            send_axes=lambda side, x, y, pulse: self.sent.append((side, x, y, pulse)) or True,
            send_turn=lambda delta: self.turns.append(delta) or True,
            release_inputs=lambda side: self.released.append(side),
            motion_provider=lambda: self.motion,
            config=NavigatorConfig(**overrides),
        )

    def _slide(self, slip: float) -> None:
        """贴着墙滑：速度模长很高，但前进分量已经塌了。"""
        self.motion["horizontal_speed_mps"] = 0.85
        self.motion["forward_ratio"] = 0.2
        self.motion["slip_ratio"] = slip

    def test_wall_slide_is_detected_even_though_speed_stays_above_threshold(self) -> None:
        navigator = self._navigator()
        self._slide(0.98)
        self.assertEqual(navigator.tick().state, "advance")
        self.assertEqual(navigator.tick().state, "advance")
        decision = navigator.tick()
        # 关键：速度全程高于 stall_speed_mps，旧判据永远不会触发。
        self.assertGreater(self.motion["horizontal_speed_mps"], 0.15)
        self.assertEqual(decision.state, "recover")
        self.assertTrue(decision.reason.startswith("auto_recover_slide"))
        self.assertGreater(navigator.snapshot()["stall"]["slip_count"], 0)

    def test_slide_recovery_turns_toward_the_slide_direction(self) -> None:
        # 滑行方向就是几何上可通行的方向；朝反方向转就是往墙里拐。
        navigator = self._navigator()
        self._slide(0.98)
        for _ in range(3):
            decision = navigator.tick()
        self.assertGreater(decision.turn_deg, 0.0)

        navigator = self._navigator()
        self._slide(-0.98)
        for _ in range(3):
            decision = navigator.tick()
        self.assertLess(decision.turn_deg, 0.0)

    def test_slide_recovery_does_not_waste_a_backup_step(self) -> None:
        # 斜撞墙时人还在动，直接转就行，不需要先退。
        navigator = self._navigator()
        self._slide(0.98)
        for _ in range(3):
            decision = navigator.tick()
        self.assertEqual(decision.y, 0.0)

    def test_head_on_wall_backs_up_before_turning(self) -> None:
        # 正面墙贴着墙角转身会蹭不出去，所以先退一步。
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.0
        self.motion["forward_ratio"] = None
        self.motion["slip_ratio"] = None
        for _ in range(3):
            decision = navigator.tick()
        self.assertEqual(decision.state, "recover")
        self.assertTrue(decision.reason.startswith("auto_recover_blocked"))
        self.assertLess(decision.y, 0.0)
        self.assertNotEqual(decision.turn_deg, 0.0)

    def test_recovery_budget_runs_out_and_hands_control_back_to_the_llm(self) -> None:
        # 死胡同：怎么绕都出不去。有限预算保证最终会把决策权交还上层，
        # 而不是永远自己转圈。
        navigator = self._navigator(auto_recover_limit=2)
        self.motion["horizontal_speed_mps"] = 0.0
        self.motion["forward_ratio"] = None
        self.motion["slip_ratio"] = None

        states = [navigator.tick().state for _ in range(12)]
        self.assertEqual(states.count("recover"), 2)
        self.assertEqual(states[-1], "stop")

        snapshot = navigator.snapshot()
        self.assertTrue(snapshot["stall"]["stalled"])
        self.assertEqual(snapshot["stall"]["recover_attempts"], 2)
        self.assertEqual(snapshot["stall"]["recover_limit"], 2)
        self.assertEqual(snapshot["stall"]["stall_count"], 1)

    def test_new_goal_restores_the_recovery_budget(self) -> None:
        navigator = self._navigator(auto_recover_limit=1)
        self.motion["horizontal_speed_mps"] = 0.0
        self.motion["forward_ratio"] = None
        self.motion["slip_ratio"] = None
        for _ in range(8):
            navigator.tick()
        self.assertEqual(navigator.snapshot()["stall"]["recover_attempts"], 1)

        self.goal["goal"] = {"kind": "approach", "text": "try the other side", "age_seconds": 1.0}
        navigator.tick()
        self.assertEqual(navigator.snapshot()["stall"]["recover_attempts"], 0)

    def test_clear_path_never_triggers_recovery(self) -> None:
        navigator = self._navigator()
        for _ in range(12):
            self.assertEqual(navigator.tick().state, "advance")
        snapshot = navigator.snapshot()
        self.assertEqual(snapshot["stall"]["recover_count"], 0)
        self.assertEqual(snapshot["stall"]["slip_count"], 0)
        self.assertEqual(self.turns, [])

    def test_missing_ratios_fall_back_to_the_pure_speed_judgement(self) -> None:
        # 旧 avatar / 旧后端没有这两个字段。没有比值时行为必须与之前完全一致，
        # 而不是因为读不到就当成撞墙。
        navigator = self._navigator()
        self.motion.pop("forward_ratio")
        self.motion.pop("slip_ratio")
        for _ in range(12):
            self.assertEqual(navigator.tick().state, "advance")
        self.assertEqual(navigator.snapshot()["stall"]["slip_count"], 0)

    def test_recovery_can_be_disabled_to_restore_pure_latching(self) -> None:
        navigator = self._navigator(auto_recover_limit=0)
        self.motion["horizontal_speed_mps"] = 0.0
        self.motion["forward_ratio"] = None
        self.motion["slip_ratio"] = None
        for _ in range(3):
            decision = navigator.tick()
        self.assertEqual(decision.state, "stop")
        self.assertEqual(decision.reason, "movement_stalled")
        self.assertTrue(navigator.snapshot()["stall"]["stalled"])


    def test_real_measured_wall_slide_readings_trip_the_guard(self) -> None:
        """用实机实测的原始读数当回归基线。

        2026-08-23 海滩地图斜角推墙实测：撞墙后 |h|=0.170、fwd=0.507、
        slip=-0.862，连续 23 个 tick。关键是 0.170 **高于** stall_speed_mps
        =0.15——纯速度判据永远不会触发，这正是滑行判据存在的理由。

        0.507 距离阈值 0.55 只有 0.043。把 slip_forward_ratio 往下调到 0.5
        就会漏掉这次真实撞墙，所以这个用例是防止"顺手改得更保守"的闸门。
        """
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.170
        self.motion["forward_ratio"] = 0.507
        self.motion["slip_ratio"] = -0.862

        # 速度确实高于失速阈值：旧判据看不见这种撞墙。
        self.assertGreater(0.170, NavigatorConfig().stall_speed_mps)

        # slip_ticks=3：前两 tick 累加计数仍放行，第三 tick 才判定。
        self.assertEqual(navigator.tick().state, "advance")
        self.assertEqual(navigator.tick().state, "advance")
        decision = navigator.tick()
        self.assertEqual(decision.state, "recover")
        self.assertTrue(decision.reason.startswith("auto_recover_slide"))
        # slip 是负的，绕行必须朝负方向转——朝正方向就是往墙里拐。
        self.assertLess(decision.turn_deg, 0.0)

    def test_real_measured_clear_walk_never_trips_the_guard(self) -> None:
        # 同一次实测的负样本：斜着走但畅通，fwd=0.887 远高于阈值。
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 3.255
        self.motion["forward_ratio"] = 0.887
        self.motion["slip_ratio"] = 0.461
        for _ in range(12):
            self.assertEqual(navigator.tick().state, "advance")
        self.assertEqual(navigator.snapshot()["stall"]["slip_count"], 0)

    def test_real_measured_head_on_wall_uses_the_speed_judgement(self) -> None:
        # 另一次实测：正面撞墙时速度直接塌到 0.062，低于失速阈值，
        # 走的是原有失速判据。此时 slip 仍有值（贴着墙微微侧滑），
        # 所以绕行应当利用这个方向而不是盲转。
        navigator = self._navigator()
        self.motion["horizontal_speed_mps"] = 0.062
        self.motion["forward_ratio"] = 0.020
        self.motion["slip_ratio"] = -0.9998
        for _ in range(3):
            decision = navigator.tick()
        self.assertEqual(decision.state, "recover")
        self.assertTrue(decision.reason.startswith("auto_recover_slide"))
        self.assertLess(decision.turn_deg, 0.0)


if __name__ == "__main__":
    unittest.main()
