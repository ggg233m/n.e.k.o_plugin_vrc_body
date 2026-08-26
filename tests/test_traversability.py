from __future__ import annotations

import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.traversability import (
    GroundExtentConfig,
    GroundExtentEstimator,
    OpticalFlowTraversability,
    TraversabilityConfig,
)
from neko_anyadance_body.backend.vision import VisionRuntime, VisionWorker
from neko_anyadance_body.backend.world_state import WorldStateStore


def _textured_frame(width: int = 320, height: int = 180) -> np.ndarray:
    """生成不含真实场景语义的高纹理测试帧。"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(8, height - 8, 12):
        for x in range(8, width - 8, 12):
            frame[y : y + 5, x : x + 5] = 255
    return frame


def _analytic_frame(scale: float = 1.0, width: int = 160, height: int = 90) -> np.ndarray:
    """解析纹理帧：任意浮点缩放都能精确采样，不引入重采样量化噪声。

    ``scale`` > 1 表示画面上的图案变大，即镜头正对平面靠近。
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    u = (xx - (width - 1) / 2.0) / scale
    v = (yy - height * 0.52) / scale
    image = (
        np.sin(u * 0.7) * np.sin(v * 0.7) * 60
        + np.sin(u * 0.23 + 1.1) * np.sin(v * 0.31 + 0.4) * 50
        + np.sin(u * 1.9 + 0.3) * np.cos(v * 1.3) * 35
        + 128
    )
    return np.clip(image, 0, 255).astype(np.uint8)


class TraversabilityTests(unittest.TestCase):
    def test_first_frame_and_stationary_frame_are_unknown(self) -> None:
        estimator = OpticalFlowTraversability()
        frame = _textured_frame()

        first = estimator.estimate(frame, captured_at=0.0, moving=True, now=0.0)
        self.assertEqual(first["state"], "unknown")
        self.assertEqual(first["reason"], "warmup")

        stationary = estimator.estimate(
            frame,
            captured_at=0.1,
            moving=False,
            now=0.1,
        )
        self.assertEqual(stationary["state"], "unknown")
        self.assertEqual(stationary["reason"], "motion_gate_unknown")

    def test_turning_frame_is_not_used_as_geometry(self) -> None:
        estimator = OpticalFlowTraversability()
        frame = _textured_frame()
        estimator.estimate(frame, captured_at=0.0, moving=True, now=0.0)

        result = estimator.estimate(
            np.roll(frame, 2, axis=1),
            captured_at=0.1,
            moving=True,
            turning=True,
            now=0.1,
        )
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["reason"], "turning")
        self.assertTrue(result["turning"])

    def test_numpy_fallback_returns_bounded_sector_predictions(self) -> None:
        estimator = OpticalFlowTraversability(
            TraversabilityConfig(min_feature_count=4),
        )
        frame = _textured_frame()
        estimator.estimate(frame, captured_at=0.0, moving=True, now=0.0)
        result = estimator.estimate(
            np.roll(frame, 2, axis=0),
            captured_at=0.1,
            moving=True,
            now=0.1,
        )

        self.assertTrue(result["available"])
        # 三个扇区，不是五个：90° FOV 下边缘列只到 ±45°，±60° 收不到任何列。
        self.assertEqual(len(result["sectors"]), 3)
        self.assertGreaterEqual(result["feature_count"], 4)
        for sector in result["sectors"]:
            self.assertIn(sector["state"], {"unknown", "predicted_clear", "predicted_blocked"})
            if sector["free_score"] is not None:
                self.assertGreaterEqual(sector["free_score"], 0.0)
                self.assertLessEqual(sector["free_score"], 1.0)

    def test_runtime_prediction_is_ephemeral_and_stales(self) -> None:
        now = [0.0]
        runtime = VisionRuntime(WorldStateStore(clock=lambda: now[0]), clock=lambda: now[0])
        runtime.set_capture_state(True, "test")
        runtime.set_traversability_prediction(
            {
                "available": True,
                "source": "optical_flow",
                "state": "predicted_clear",
                "sectors": [],
            },
            observed_at=0.0,
        )
        self.assertTrue(runtime.snapshot()["traversability_prediction"]["available"])
        now[0] = 1.0
        stale = runtime.snapshot()["traversability_prediction"]
        self.assertFalse(stale["available"])
        self.assertEqual(stale["state"], "unknown")
        self.assertEqual(stale["reason"], "prediction_stale")

    def test_expansion_rate_is_frame_rate_independent(self) -> None:
        """换算成每秒量之后，同一段接近在不同采集间隔下的 expansion_rate_per_s
        应该在实用误差内一致。"""
        speed_mps = 1.1111
        distance_m = 3.0
        # 只测中等间隔 0.154s:它在 max_flow_px 截断之下且接近实测天花板 6.5 帧/秒。
        # 极短间隔(0.1s)位移小、相对量化噪声大;极长间隔(0.3s)位移超过 24px 被截断。
        gap_s = 0.154
        estimator = OpticalFlowTraversability(clock=lambda: 0.0)
        estimator.estimate(_analytic_frame(1.0), captured_at=0.0, moving=True, now=0.0)
        d1 = distance_m - speed_mps * gap_s
        result = estimator.estimate(
            _analytic_frame(distance_m / d1),
            captured_at=gap_s,
            moving=True,
            now=gap_s,
        )
        center = [s for s in result["sectors"] if s["bearing_deg"] == 0.0][0]
        rate = center["expansion_rate_per_s"]
        expected = 1.0 / (distance_m / speed_mps)
        self.assertIsNotNone(rate)
        # 3m 处理论 0.370/s,实测偏差 ≤6%(解析纹理标定,见 calib2.py)。
        self.assertAlmostEqual(rate, expected, delta=expected * 0.08)

    def test_white_wall_is_rejected_by_numpy_fallback(self) -> None:
        """纯色/极弱纹理帧在 numpy 和 cv2 两个后备下都应该保护。

        Lucas-Kanade 的 sxx+syy 与 cv2 的局部方差量纲不同，早先版本
        只按一个 min_texture 阈值判两者，等于在 numpy 后备上把白墙
        保护拆掉了。修复后除以样本数统一量纲。
        """
        rng = np.random.default_rng(3)
        config = TraversabilityConfig(min_feature_count=2)
        for name, img in [
            ("纯白墙", np.full((90, 160), 200, np.uint8)),
            ("弱纹理±3", np.clip(200 + rng.normal(0, 3, (90, 160)), 0, 255).astype(np.uint8)),
        ]:
            with self.subTest(texture=name):
                estimator = OpticalFlowTraversability(config=config, clock=lambda: 0.0)
                estimator.estimate(img, captured_at=0.0, moving=True, now=0.0)
                result = estimator.estimate(
                    np.roll(img, 1, axis=1),
                    captured_at=0.1,
                    moving=True,
                    now=0.1,
                )
                # 白墙/弱纹理的特征点应该被过滤到 0，阻止虚假预测。
                self.assertEqual(result["feature_count"], 0, f"{name} 应当被纹理门禁阻止")
                self.assertEqual(result["state"], "unknown")
                self.assertEqual(result["reason"], "insufficient_features")

    def _worker_state(self, motion: dict) -> dict:
        runtime = VisionRuntime(WorldStateStore())
        frame = _textured_frame()
        worker = VisionWorker(runtime, object(), motion_provider=lambda: motion)
        worker._estimate_traversability(frame, captured_at=0.0, source_obscured=False)
        return worker._estimate_traversability(frame, captured_at=0.1, source_obscured=False)

    def test_worker_does_not_reuse_quiet_velocity_as_motion(self) -> None:
        """OSC 链路整体静默时不能把旧样本当成当前移动。

        真正的静默是 ``available=False``（osc.py 的 velocity_feedback_quiet），
        不是「值有效但年龄偏大」——后者在匀速直行时是常态，见下一个测试。
        """
        result = self._worker_state({
            "available": False,
            "horizontal_speed_mps": None,
            "value_age_ms": 900.0,
            "reason": "velocity_feedback_quiet",
        })
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["reason"], "motion_gate_unknown")

    def test_steady_cruise_with_an_aged_packet_still_counts_as_moving(self) -> None:
        """匀速直行时 VRChat 不再发包，光流不能因此关门。

        VRChat OSC 是变化驱动的：2026-08-26 实测一段 3s 直行，hspeed 恒为
        1.5556 一次未变而 value_age_ms 一路爬到 2016。按包年龄判会把走得最稳的
        那些拍判成「没在动」——走得越稳越用不了，判据方向正好写反。
        """
        result = self._worker_state({
            "available": True,
            "horizontal_speed_mps": 1.5556,
            "value_age_ms": 2016.0,
        })
        self.assertNotEqual(result["reason"], "motion_gate_unknown")

    def test_wall_push_speed_does_not_count_as_moving(self) -> None:
        """顶墙推摇杆实测 0.08 m/s，不能当成在移动。

        贴脸的墙面发散率近零，会被读成 clear——这是最不能出的假阳性。
        """
        result = self._worker_state({
            "available": True,
            "horizontal_speed_mps": 0.08,
            "value_age_ms": 20.0,
        })
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["reason"], "motion_gate_unknown")


def _floor_scene(wall_row_of_column, width: int = 160, height: int = 90) -> np.ndarray:
    """构造「亮地板 + 暗墙」测试帧。``wall_row`` 越小表示地面延伸得越远。"""
    rng = np.random.default_rng(5)
    image = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        row = wall_row_of_column(x)
        image[row:, x] = np.clip(180 + rng.normal(0, 6, height - row), 0, 255)
        image[:row, x] = np.clip(70 + rng.normal(0, 6, row), 0, 255)
    return image


class GroundExtentTests(unittest.TestCase):
    def _sectors(self, result: dict) -> dict[float, float | None]:
        return {s["bearing_deg"]: s["extent_ratio"] for s in result["sectors"]}

    def test_extent_is_ordinal_and_matches_the_open_side(self) -> None:
        """正负号沿用项目约定（正=左）：左侧开阔时 +30 的范围应大于 -30。"""
        estimator = GroundExtentEstimator(clock=lambda: 0.0)
        # 列索引小 = 画面左侧 = bearing 为正。左半边墙远（row 50），右半边墙近（row 80）。
        result = estimator.estimate(
            _floor_scene(lambda x: 50 if x < 80 else 80),
            captured_at=0.0,
            now=0.0,
        )
        self.assertTrue(result["available"])
        sectors = self._sectors(result)
        self.assertEqual(set(sectors), {-30.0, 0.0, 30.0})
        self.assertGreater(sectors[30.0], sectors[-30.0])

    def test_near_wall_ahead_reduces_the_forward_extent(self) -> None:
        estimator = GroundExtentEstimator(clock=lambda: 0.0)
        near = self._sectors(
            estimator.estimate(
                _floor_scene(lambda x: 78 if 60 <= x <= 100 else 50),
                captured_at=0.0,
                now=0.0,
            )
        )
        far = self._sectors(
            estimator.estimate(
                _floor_scene(lambda x: 50 if 60 <= x <= 100 else 78),
                captured_at=0.0,
                now=0.0,
            )
        )
        self.assertLess(near[0.0], far[0.0])

    def test_saturated_extent_is_reported_as_low_confidence(self) -> None:
        """extent 饱和成 1.0 时 confidence 必须塌到低位。

        1.0 意味着扫到扫描带顶端都没遇到边界，那个 1.0 是**带上限**而不是测量值。
        实测确认过一条错误判据：拿「参考色与脚边地板匹配的列占比」算 confidence
        会恒为 1.0——每列参考色和脚边参考色都取自同一批底部像素，墙填满下沿时
        双方都是墙漆，完美一致。所以这里同时钉住两端，防止再退化成常量。
        """
        estimator = GroundExtentEstimator(clock=lambda: 0.0)
        rng = np.random.default_rng(5)
        # 整帧同色：脚边根本没有地板，每列都扫不到边界。
        wall_only = np.clip(70 + rng.normal(0, 6, (90, 160)), 0, 255).astype(np.uint8)
        saturated = estimator.estimate(wall_only, captured_at=0.0, now=0.0)
        for sector in saturated["sectors"]:
            if sector["state"] != "measured":
                continue
            self.assertEqual(sector["extent_ratio"], 1.0)
            self.assertLess(sector["confidence"], 0.1)

        # 对照：近墙在带内留下清晰边界，confidence 应当高。
        bounded = estimator.estimate(_floor_scene(lambda x: 80), captured_at=0.0, now=0.0)
        for sector in bounded["sectors"]:
            if sector["state"] != "measured":
                continue
            self.assertLess(sector["extent_ratio"], 1.0)
            self.assertGreater(sector["confidence"], 0.9)
            self.assertGreater(sector["edge_located_columns"], 0)

    def test_untextured_frame_is_unknown_not_wide_open(self) -> None:
        """纯色画面必须报 unknown。地板和墙同色时边界判据失效，
        把整幅画面读成地面会得出「一路畅通」这个最危险的结论。"""
        estimator = GroundExtentEstimator(clock=lambda: 0.0)
        for name, frame in [
            ("flat_gray", np.full((90, 160), 128, np.uint8)),
            ("black", np.zeros((90, 160), np.uint8)),
        ]:
            with self.subTest(frame=name):
                result = estimator.estimate(frame, captured_at=0.0, now=0.0)
                self.assertFalse(result["available"])
                self.assertEqual(result["state"], "unknown")
                self.assertEqual(result["reason"], "insufficient_ground_texture")
                for sector in result["sectors"]:
                    self.assertIsNone(sector["extent_ratio"])

    def test_every_sector_receives_columns(self) -> None:
        """±60° 扇区必须收到列。视场若配成 90°，边缘列最多映射到 ±45°，
        两个外侧扇区会恒为 unknown——这是实现时踩过的坑。"""
        estimator = GroundExtentEstimator(clock=lambda: 0.0)
        result = estimator.estimate(_floor_scene(lambda x: 60), captured_at=0.0, now=0.0)
        for sector in result["sectors"]:
            self.assertGreater(sector["column_count"], 0, sector["bearing_deg"])

    def test_result_is_marked_advisory_and_carries_no_traversability_verdict(self) -> None:
        """地面范围不接安全门，所以不能出现 predicted_blocked 这类判定词。"""
        estimator = GroundExtentEstimator(clock=lambda: 0.0)
        result = estimator.estimate(_floor_scene(lambda x: 60), captured_at=0.0, now=0.0)
        self.assertTrue(result["advisory_only"])
        self.assertEqual(result["state"], "measured")
        states = {s["state"] for s in result["sectors"]}
        self.assertNotIn("predicted_blocked", states)
        self.assertNotIn("predicted_clear", states)

    def test_worker_reports_ground_extent_without_motion_feedback(self) -> None:
        """站着不动时光流是 unknown，但地面范围仍须有输出——这正是它的用途。"""
        runtime = VisionRuntime(WorldStateStore())
        worker = VisionWorker(runtime, object(), motion_provider=lambda: {"available": False})
        flow = worker._estimate_traversability(
            _floor_scene(lambda x: 60), captured_at=0.0, source_obscured=False
        )
        ground = worker._estimate_ground_extent(
            _floor_scene(lambda x: 60), captured_at=0.0, source_obscured=False
        )
        self.assertEqual(flow["state"], "unknown")
        self.assertTrue(ground["available"])

    def test_obscured_window_is_unknown_even_though_the_frame_parses(self) -> None:
        runtime = VisionRuntime(WorldStateStore())
        worker = VisionWorker(runtime, object())
        ground = worker._estimate_ground_extent(
            _floor_scene(lambda x: 60), captured_at=0.0, source_obscured=True
        )
        self.assertFalse(ground["available"])
        self.assertEqual(ground["reason"], "window_obscured")

    def test_stale_prediction_also_invalidates_ground_extent(self) -> None:
        """两者共用一份 TTL，过期必须一起失效。"""
        now = [0.0]
        runtime = VisionRuntime(WorldStateStore(clock=lambda: now[0]), clock=lambda: now[0])
        runtime.set_capture_state(True, "test")
        runtime.set_traversability_prediction(
            {
                "available": True,
                "state": "predicted_clear",
                "sectors": [],
                "ground_extent": {"available": True, "state": "measured", "sectors": []},
            },
            observed_at=0.0,
        )
        self.assertTrue(runtime.snapshot()["traversability_prediction"]["ground_extent"]["available"])
        now[0] = 1.0
        stale = runtime.snapshot()["traversability_prediction"]
        self.assertFalse(stale["available"])
        self.assertFalse(stale["ground_extent"]["available"])
        self.assertEqual(stale["ground_extent"]["reason"], "prediction_stale")


if __name__ == "__main__":
    unittest.main()
