"""方向性死路记忆的离线验证。

这里刻意不接后端：``DirectionMemory`` 是纯逻辑，喂进实测采样就能验证。用到的数字
全部来自实机采集（tmp/px_open2、tmp/px_approach 等），不是构造的理想值。
"""
from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.direction_memory import (
    DirectionMemory,
    SegmentOutcome,
    integrate_progress,
    normalize_bearing,
    sector_center_deg,
    sector_of,
)


class BearingMathTests(unittest.TestCase):
    def test_normalize_keeps_project_sign_convention(self) -> None:
        # 正左负右，和 navigator 的 turn_sign_convention 一致。
        self.assertEqual(normalize_bearing(0.0), 0.0)
        self.assertEqual(normalize_bearing(45.0), 45.0)
        self.assertEqual(normalize_bearing(-45.0), -45.0)
        self.assertEqual(normalize_bearing(360.0), 0.0)
        self.assertEqual(normalize_bearing(190.0), -170.0)
        self.assertEqual(normalize_bearing(-190.0), 170.0)

    def test_sector_merges_small_heading_jitter(self) -> None:
        # 同一堵墙不该因为几度抖动被记成两条边。
        self.assertEqual(sector_of(0.0), sector_of(20.0))
        self.assertEqual(sector_of(0.0), sector_of(-20.0))
        # 45° 扇区下 wander 的 +25° 和 recover 的 +55° 落在同一格，这是刻意的：
        # 两者都是「左前方」，撞到的很可能就是同一堵墙，分开记反而会让同一面墙
        # 攒不够 confident_block。真正需要分开的是左右和前后。
        self.assertEqual(sector_of(25.0), sector_of(55.0))
        self.assertNotEqual(sector_of(25.0), sector_of(-25.0))
        self.assertNotEqual(sector_of(0.0), sector_of(90.0))
        self.assertNotEqual(sector_of(0.0), sector_of(180.0))

    def test_sector_center_round_trips(self) -> None:
        for deg in (0.0, 45.0, 90.0, -45.0, -90.0, 135.0):
            self.assertEqual(sector_of(sector_center_deg(sector_of(deg))), sector_of(deg))


class DirectionMemoryTests(unittest.TestCase):
    def test_repeated_block_in_same_sector_becomes_confident(self) -> None:
        mem = DirectionMemory(ttl_s=45.0)
        first = mem.record(SegmentOutcome(bearing_deg=25.0, blocked=True), now=100.0)
        self.assertEqual(first.empirical_state, "blocked")
        self.assertFalse(first.confident_block)

        # 同一堵墙第二次撞：升级为高置信，这才是「别再选这个方向」的依据。
        second = mem.record(SegmentOutcome(bearing_deg=30.0, blocked=True), now=101.0)
        self.assertTrue(second.confident_block)
        self.assertEqual(second.blocked_count, 2)
        self.assertEqual(mem.blocked_sectors(now=101.0), [second.sector])

    def test_cleared_segment_upgrades_state_but_keeps_block_history(self) -> None:
        mem = DirectionMemory(ttl_s=45.0)
        mem.record(SegmentOutcome(bearing_deg=-45.0, blocked=True), now=10.0)
        entry = mem.record(
            SegmentOutcome(bearing_deg=-45.0, blocked=False, progress_m=3.2),
            now=11.0,
        )
        self.assertEqual(entry.empirical_state, "verified_free")
        # 走通了不代表历史清零：这个方向曾经受阻，上层该知道它不稳。
        self.assertEqual(entry.blocked_count, 1)
        self.assertEqual(entry.cleared_count, 1)
        self.assertAlmostEqual(entry.last_progress_m, 3.2)

    def test_records_expire_so_a_stale_wall_does_not_block_forever(self) -> None:
        mem = DirectionMemory(ttl_s=30.0)
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=0.0)
        self.assertEqual(mem.state_of(0.0, now=29.0), "blocked")
        # 没有绝对定位，走远之后旧证据不再描述当前周边——必须过期。
        self.assertEqual(mem.state_of(0.0, now=31.0), "unknown")
        self.assertEqual(mem.blocked_sectors(now=31.0), [])

    def test_recording_never_produces_a_predicted_state(self) -> None:
        # 实测态只有三个。预测分是独立字段，不会伪装成一种实测结果。
        mem = DirectionMemory()
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=False, progress_m=1.0), now=5.0)
        mem.record(SegmentOutcome(bearing_deg=90.0, blocked=True), now=5.0)
        records = mem.advice(now=5.0)["records"]
        states = {r["empirical_state"] for r in records}
        self.assertTrue(states <= {"verified_free", "blocked"})
        self.assertTrue(all(r["predicted_score"] is None for r in records))


class PredictionTests(unittest.TestCase):
    def test_prediction_is_stored_without_touching_empirical_state(self) -> None:
        mem = DirectionMemory()
        mem.predict({"left": 0.8, "forward": 0.2, "right": 0.5}, now=0.0)
        advice = mem.advice(now=0.0)
        # 预测不产生实测结论，也不该让这个方向看起来「已经试过」。
        self.assertTrue(
            all(r["empirical_state"] == "unknown" for r in advice["records"])
        )
        self.assertEqual(advice["blocked_bearings_deg"], [])
        by_bearing = {r["bearing_deg"]: r["predicted_score"] for r in advice["records"]}
        self.assertAlmostEqual(by_bearing[45.0], 0.8)
        self.assertAlmostEqual(by_bearing[0.0], 0.2)
        self.assertAlmostEqual(by_bearing[-45.0], 0.5)

    def test_scores_are_clamped_and_bad_keys_are_skipped(self) -> None:
        mem = DirectionMemory()
        mem.predict(
            {
                "left": 5.0,
                "right": -2.0,
                "sideways": 0.9,
                "90": 0.4,
                "nan": 0.8,
                "inf": 0.7,
            },
            now=0.0,
        )
        by_bearing = {
            r["bearing_deg"]: r["predicted_score"] for r in mem.advice(now=0.0)["records"]
        }
        self.assertAlmostEqual(by_bearing[45.0], 1.0)
        self.assertAlmostEqual(by_bearing[-45.0], 0.0)
        # 数字字符串键仍然可用；拼错的方向名跳过而不是让整段闲逛失败。
        self.assertAlmostEqual(by_bearing[90.0], 0.4)
        self.assertEqual(len(by_bearing), 3)

    def test_prediction_expires_much_sooner_than_evidence(self) -> None:
        mem = DirectionMemory(ttl_s=45.0, prediction_ttl_s=5.0)
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=0.0)
        mem.predict({"forward": 0.9}, now=0.0)
        self.assertIsNotNone(mem.advice(now=4.0)["records"][0]["predicted_score"])
        later = mem.advice(now=10.0)["records"][0]
        # 预测过期，但撞墙的实测证据还在。
        self.assertIsNone(later["predicted_score"])
        self.assertEqual(later["empirical_state"], "blocked")

    def test_disagreement_between_prediction_and_reality_is_reported(self) -> None:
        mem = DirectionMemory()
        mem.predict({"left": 0.8}, now=0.0)
        entry = mem.record(SegmentOutcome(bearing_deg=45.0, blocked=True), now=1.0)
        self.assertEqual(entry.prediction_outcome, "false_positive")
        advice = mem.advice(now=1.0)
        self.assertEqual(len(advice["prediction_disagreements"]), 1)
        disagreement = advice["prediction_disagreements"][0]
        self.assertEqual(disagreement["bearing_deg"], 45.0)
        self.assertAlmostEqual(disagreement["predicted_score"], 0.8)
        self.assertEqual(disagreement["empirical_state"], "blocked")

    def test_midrange_prediction_is_not_counted_as_a_miss(self) -> None:
        # 0.5 等于没表态，撞了也不该记成预测失败，否则摘要全是噪声。
        mem = DirectionMemory()
        mem.predict({"forward": 0.5}, now=0.0)
        entry = mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=1.0)
        self.assertEqual(entry.prediction_outcome, "confirmed")
        self.assertEqual(mem.advice(now=1.0)["prediction_disagreements"], [])

    def test_refusal_only_applies_to_confident_blocks(self) -> None:
        mem = DirectionMemory()
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=0.0)
        # 撞一次可能是被人挡了一下，拒绝会显得莫名其妙。
        self.assertFalse(mem.should_refuse(0.0, now=0.0))
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=1.0)
        self.assertTrue(mem.should_refuse(0.0, now=1.0))
        # 拒绝只针对那个扇区，不扩散成「四周都不能走」。
        self.assertFalse(mem.should_refuse(180.0, now=1.0))

    def test_memory_exposes_no_route_chooser(self) -> None:
        # 闲逛路线归主 LLM。这里出现选路方法就等于导航器夺权。
        mem = DirectionMemory()
        for name in ("best_direction", "choose_direction", "pick_bearing", "best_bearing"):
            self.assertFalse(hasattr(mem, name), name)
        self.assertEqual(mem.advice(now=0.0)["route_choice_owner"], "main_llm")

    def test_crowded_directions_are_separate_from_traversability(self) -> None:
        mem = DirectionMemory()
        mem.mark_dynamic_obstacle([45.0], now=0.0)
        advice = mem.advice(now=0.0)
        # 看到人只说明可能有动态障碍，说明不了有没有墙。
        self.assertEqual(advice["crowded_bearings_deg"], [45.0])
        self.assertEqual(advice["records"][0]["empirical_state"], "unknown")
        self.assertIsNone(advice["records"][0]["predicted_score"])

    def test_clearing_predictions_keeps_empirical_evidence(self) -> None:
        mem = DirectionMemory()
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=0.0)
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=1.0)
        mem.predict({"forward": 0.9}, now=1.0)
        mem.mark_dynamic_obstacle([0.0], now=1.0)
        mem.clear_predictions()
        advice = mem.advice(now=1.0)
        self.assertIsNone(advice["records"][0]["predicted_score"])
        self.assertEqual(advice["crowded_bearings_deg"], [])
        # 换目标不该让撞过两次的墙消失。
        self.assertEqual(advice["confident_block_bearings_deg"], [0.0])
        self.assertTrue(mem.should_refuse(0.0, now=1.0))

    def test_advice_declares_that_there_is_no_world_localization(self) -> None:
        mem = DirectionMemory()
        mem.record(SegmentOutcome(bearing_deg=55.0, blocked=True, turned=True), now=1.0)
        advice = mem.advice(now=1.0)
        # 摘要必须自己说清「这不是地图」，否则上层会把扇区当坐标用。
        self.assertFalse(advice["world_localization_available"])
        self.assertEqual(advice["position_reference"], "relative_to_current_heading")
        self.assertEqual(advice["heading_anchor_source"], "scheduler_virtual_hmd")
        self.assertEqual(advice["turn_sign_convention"], "positive_left_negative_right")
        self.assertEqual(advice["blocked_bearings_deg"], [sector_center_deg(sector_of(55.0))])
        self.assertTrue(advice["records"][0]["turned_during_segment"])


class HeadingAnchorTests(unittest.TestCase):
    """扇区锚在虚拟 HMD yaw 上，不是「相对当前朝向」的裸角度。"""

    def test_turning_away_does_not_refuse_a_different_wall(self) -> None:
        mem = DirectionMemory()
        # 朝向 0° 时向左 25° 撞了两次，该扇区确认封死。
        mem.record(
            SegmentOutcome(bearing_deg=25.0, blocked=True, heading_deg=0.0), now=0.0
        )
        mem.record(
            SegmentOutcome(bearing_deg=25.0, blocked=True, heading_deg=0.0), now=1.0
        )
        self.assertTrue(mem.should_refuse(25.0, now=1.0, heading_deg=0.0))
        # 人转过 90° 之后，同一个「向左 25°」指向世界里完全不同的地方，不能再拒绝。
        # 不带锚点混用相对角时这里会误判成封死，把可走的方向永久堵死。
        self.assertFalse(mem.should_refuse(25.0, now=1.0, heading_deg=90.0))
        # 那堵墙现在位于「相对当前朝向 -65°」，仍然应该被拒绝——记忆没丢，只是换算了。
        self.assertTrue(mem.should_refuse(-65.0, now=1.0, heading_deg=90.0))

    def test_advice_reports_bearings_relative_to_current_heading(self) -> None:
        mem = DirectionMemory()
        mem.record(
            SegmentOutcome(bearing_deg=0.0, blocked=True, heading_deg=0.0), now=0.0
        )
        # 站在原地看：那堵墙就在正前方。
        self.assertEqual(mem.advice(now=0.0, heading_deg=0.0)["blocked_bearings_deg"], [0.0])
        # 转过 90° 之后再看：同一堵墙变成了右手边，主 LLM 可以直接把这个角度当
        # turn_deg 用，不需要自己补偿转过多少。
        self.assertEqual(
            mem.advice(now=0.0, heading_deg=90.0)["blocked_bearings_deg"], [-90.0]
        )

    def test_reset_drops_evidence_anchored_to_the_previous_world(self) -> None:
        mem = DirectionMemory()
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=0.0)
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=1.0)
        self.assertTrue(mem.should_refuse(0.0, now=1.0))
        # 换世界后 yaw 锚点不再连续，旧扇区指向的是上个世界的墙。
        mem.reset()
        self.assertFalse(mem.should_refuse(0.0, now=1.0))
        self.assertEqual(mem.advice(now=1.0)["records"], [])


class ProgressGateTests(unittest.TestCase):
    """「收到速度样本」不等于「真的走通了」。"""

    def test_zero_speed_samples_do_not_prove_the_direction_is_free(self) -> None:
        # 顶着墙时 VRChat 照样回传速度包，值就是 0.0——而 0.0 是个有限数。
        # 只看「有没有样本」会把撞墙记成 verified_free，和这套记忆的目的相反。
        mem = DirectionMemory()
        entry = mem.record(
            SegmentOutcome(
                bearing_deg=0.0,
                blocked=False,
                progress_m=0.0,
                evidence_available=True,
            ),
            now=0.0,
        )
        self.assertEqual(entry.empirical_state, "unknown")
        self.assertEqual(entry.cleared_count, 0)

    def test_short_segment_below_threshold_stays_unknown(self) -> None:
        # 失速计数还没到阈值就先到期的短路段：没撞上，但也没走出距离。
        mem = DirectionMemory()
        entry = mem.record(
            SegmentOutcome(bearing_deg=0.0, blocked=False, progress_m=0.1), now=0.0
        )
        self.assertEqual(entry.empirical_state, "unknown")

    def test_real_progress_upgrades_to_verified_free(self) -> None:
        # 实机 px_open2：1.05 m/s 巡航两秒 = 2.1m，远超门槛。
        mem = DirectionMemory()
        entry = mem.record(
            SegmentOutcome(bearing_deg=0.0, blocked=False, progress_m=2.1), now=0.0
        )
        self.assertEqual(entry.empirical_state, "verified_free")
        self.assertEqual(entry.cleared_count, 1)

    def test_progressless_segment_does_not_launder_a_confident_block(self) -> None:
        mem = DirectionMemory()
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=0.0)
        mem.record(SegmentOutcome(bearing_deg=0.0, blocked=True), now=1.0)
        # 一段没走出距离的尝试不该把撞过两次的墙洗成可走。
        mem.record(
            SegmentOutcome(bearing_deg=0.0, blocked=False, progress_m=0.05), now=2.0
        )
        self.assertTrue(mem.should_refuse(0.0, now=2.0))

    def test_prediction_is_not_settled_without_real_progress(self) -> None:
        # 没有实测结论就不结算预测，否则「不知道」会反过来给看图判断背书。
        mem = DirectionMemory()
        mem.predict({"forward": 0.9}, now=0.0)
        entry = mem.record(
            SegmentOutcome(bearing_deg=0.0, blocked=False, progress_m=0.0), now=1.0
        )
        self.assertIsNone(entry.prediction_outcome)


class ProgressIntegrationTests(unittest.TestCase):
    def test_open_cruise_samples_integrate_to_a_plausible_distance(self) -> None:
        # 实机 px_open2：speed 稳定 1.05 m/s。这里给 forward_ratio=1.0 的理想直行段。
        samples = [{"speed": 1.05, "forward_ratio": 1.0, "dt": 0.25} for _ in range(8)]
        distance, turned = integrate_progress(samples)
        self.assertAlmostEqual(distance, 1.05 * 2.0, places=6)
        self.assertFalse(turned)

    def test_wall_slide_samples_are_rejected(self) -> None:
        # 实机 px_open2 实测 forward_ratio=0.675：速度模长不低，但大部分是沿墙横移。
        samples = [{"speed": 1.05, "forward_ratio": 0.675, "dt": 0.25} for _ in range(8)]
        distance, _ = integrate_progress(samples)
        self.assertEqual(distance, 0.0)

    def test_missing_velocity_is_not_treated_as_zero_speed(self) -> None:
        # 静止时 VRChat 不回传速度；把沉默当 0 会把「不知道」记成「没动」。
        samples = [
            {"speed": None, "forward_ratio": None, "dt": 0.25},
            {"speed": 1.5, "forward_ratio": 0.95, "dt": 0.25},
        ]
        distance, _ = integrate_progress(samples)
        self.assertAlmostEqual(distance, 1.5 * 0.25, places=6)

    def test_turn_during_segment_is_reported_even_if_nothing_integrates(self) -> None:
        # AngularY 只在移动期间产生新样本，所以「转过」这个事实必须独立于积分结果。
        samples = [{"speed": 0.9, "forward_ratio": 0.4, "dt": 0.2, "turned": True}]
        distance, turned = integrate_progress(samples)
        self.assertEqual(distance, 0.0)
        self.assertTrue(turned)


if __name__ == "__main__":
    unittest.main()
