# -*- coding: utf-8 -*-
"""M0 契约测试（2026-11-06）：wp≈0 不停车 / 五种停止源 / mode 单向映射。

这组测试只用 policy_modes + policy_executor（numpy 级），**不加载模型、不碰 torch**，
所以 `.venv` 与 `.venv_xpu` 都能跑。

动机（每一条都是真实踩过的坑）：
* ``|wp| < wp_stop`` 曾把整块动作清零，导致「未定位 / 地图外 / 自由探索」
  与「已到终点」不可区分，策略看起来接管了却永远不动。
* ``mode`` 在 policy 侧是 0=follow/3=stop，在 navmesh 侧曾是「跟随器轴没输出 ⇒ 3」，
  同一个数字两种语义，下游无法判断到底是谁要求停。
* ``stop_recall`` 必须说清是在哪种停止源下测的，否则会和门控数字混在一起用。
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

import policy_modes as pm  # noqa: E402
import policy_executor as pe  # noqa: E402


def _block(v=0.4):
    return [[v, v, 0.0]] * pe.K_STEPS


class WpNearZeroTests(unittest.TestCase):
    """wp≈0 = 「没有路径先验」，绝不能变成停车命令。"""

    def test_decide_stop_ignores_wp_near_zero(self):
        d = pm.decide_stop(wp_near_zero=True)
        self.assertFalse(d["stop"], "wp≈0 竟然触发停车")
        self.assertIsNone(d["source"])
        # 注记仍然如实带出，只是不参与决策
        self.assertTrue(d["detail"]["wp_near_zero"])
        self.assertEqual(d["detail"]["path_prior"], "none")

    def test_decide_stop_wp_present_also_not_stop(self):
        d = pm.decide_stop(wp_near_zero=False)
        self.assertFalse(d["stop"])
        self.assertEqual(d["detail"]["path_prior"], "soft")

    def test_apply_stop_to_block_only_zeros_when_stop(self):
        b = _block(0.5)
        self.assertEqual(pm.apply_stop_to_block(b, False), b)
        z = pm.apply_stop_to_block(b, True)
        self.assertTrue(all(row == [0.0, 0.0, 0.0] for row in z))
        self.assertEqual(len(z), len(b))

    def test_executor_keeps_block_when_wp_is_zero(self):
        """执行器这一层也不能因为 wp≈0 清零：meta 里根本没有 wp 的判定。"""
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        ex.submit(_block(0.5), 100.0, {"stop": False, "wp": [0.0, 0.0]})
        self.assertFalse(ex._gated)
        self.assertNotEqual(float(ex._block.sum()), 0.0)


class StopSourceTests(unittest.TestCase):
    """五种来源各自能触发；优先级只影响 source 归因。"""

    def test_estop(self):
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "STOP"
            ex.stop_file = p
            self.assertFalse(ex._estop_requested())
            p.write_text("x", encoding="utf-8")
            self.assertTrue(ex._estop_requested())
            d = pm.decide_stop(estop=ex._estop_requested())
            self.assertEqual(d["source"], "estop")

    def test_stale(self):
        self.assertEqual(pm.decide_stop(stale=True)["source"], "stale")

    def test_safety_gate(self):
        self.assertEqual(pm.decide_stop(safety_veto=True)["source"], "safety_gate")

    def test_nav_state_done_and_stop(self):
        for s in ("done", "stop", "DONE"):
            self.assertEqual(pm.decide_stop(nav_state=s)["source"], "nav_state", s)

    def test_nav_state_idle_is_not_a_stop(self):
        """idle 刻意**不**算停止：独立 VLA 接管时导航器长期 idle。"""
        self.assertFalse(pm.decide_stop(nav_state="idle")["stop"])
        self.assertFalse(pm.nav_state_is_stop("idle"))

    def test_stop_head_threshold(self):
        self.assertFalse(pm.decide_stop(stop_head_p=None)["stop"])
        self.assertFalse(pm.decide_stop(stop_head_p=0.2)["stop"])
        self.assertEqual(pm.decide_stop(stop_head_p=0.7)["source"], "stop_head")
        self.assertEqual(pm.decide_stop(stop_head_p=0.5,
                                        stop_head_thresh=0.5)["source"], "stop_head")

    def test_priority_estop_beats_everything(self):
        d = pm.decide_stop(estop=True, stale=True, safety_veto=True,
                           nav_state="done", stop_head_p=0.9)
        self.assertEqual(d["source"], "estop")

    def test_priority_chain(self):
        self.assertEqual(pm.decide_stop(stale=True, safety_veto=True,
                                       nav_state="done")["source"], "stale")
        self.assertEqual(pm.decide_stop(safety_veto=True, nav_state="done")["source"],
                         "safety_gate")
        self.assertEqual(pm.decide_stop(nav_state="done", stop_head_p=0.9)["source"],
                         "nav_state")

    def test_executor_zeroes_on_safety_veto_and_records_source(self):
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        ex.submit(_block(0.6), 100.0, {"safety_veto": True})
        self.assertTrue(ex._gated)
        self.assertEqual(float(abs(ex._block).sum()), 0.0)
        self.assertEqual(ex._stop_source, "safety_gate")

    def test_executor_zeroes_on_stale(self):
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        ex.submit(_block(0.6), 100.0, {"stale": True})
        self.assertEqual(float(abs(ex._block).sum()), 0.0)
        self.assertEqual(ex._stop_source, "stale")

    def test_executor_accepts_legacy_gated_alias(self):
        """旧调用方只传 `gated` 时必须照旧停车（语义已改为 stop，键名仍兼容）。"""
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        ex.submit(_block(0.6), 100.0, {"gated": True, "stop_source": "nav_state"})
        self.assertTrue(ex._gated)
        self.assertEqual(ex._stop_source, "nav_state")

    def test_sources_declared_once(self):
        self.assertEqual(pm.STOP_SOURCES,
                         ("estop", "stale", "safety_gate", "safety_gate_unlocalized",
                          "nav_state", "stop_head"))
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        self.assertEqual(tuple(ex.n_halts_by_source), pm.STOP_SOURCES)


class LocalizationGateTests(unittest.TestCase):
    """定位安全门：与 wp / nav mode 完全独立（2026-11-06 第二步）。

    六条硬契约（用户逐条要求）。注意 ``localization_ok=None`` 表示**没有定位
    子系统**（独立 VLA 无 navmesh），不适用、不停车。
    """

    def test_1_wp_zero_localized_explore_allows_motion(self):
        d = pm.decide_stop(wp_near_zero=True, localization_ok=True,
                           nav_state="explore")
        self.assertFalse(d["stop"])
        self.assertIsNone(d["source"])
        self.assertEqual(d["detail"]["path_prior"], "none")

    def test_2_wp_zero_unlocalized_stops_with_dedicated_source(self):
        d = pm.decide_stop(wp_near_zero=True, localization_ok=False,
                           localization_reason="unlocalized", nav_state="explore")
        self.assertTrue(d["stop"])
        self.assertEqual(d["source"], "safety_gate_unlocalized")
        self.assertEqual(d["reason"], "unlocalized")

    def test_3_no_astar_localized_allows_motion(self):
        """没有 A*（valid=False / mode 无路径先验）但定位正常 ⇒ 允许运动。"""
        d = pm.decide_stop(wp_near_zero=True, localization_ok=True, nav_state=None)
        self.assertFalse(d["stop"])

    def test_4_idle_localized_does_not_auto_stop(self):
        d = pm.decide_stop(localization_ok=True, nav_state="idle", wp_near_zero=True)
        self.assertFalse(d["stop"])

    def test_5_nav_state_stop_done_is_nav_state_not_safety_gate(self):
        for s in ("done", "stop"):
            d = pm.decide_stop(localization_ok=True, nav_state=s)
            self.assertTrue(d["stop"], s)
            self.assertEqual(d["source"], "nav_state", s)
            self.assertNotIn("safety_gate", d["source"])

    def test_6_pose_stale_is_safety_gate_with_reason(self):
        d = pm.decide_stop(localization_ok=False, localization_reason="pose_stale")
        self.assertEqual(d["source"], "safety_gate_unlocalized")
        self.assertEqual(d["reason"], "pose_stale")
        # 纯函数层的同一条判定（nav_online 就调用它）
        loc = pm.decide_localization("localized", 2.0, pose_stale_s=0.5)
        self.assertFalse(loc["ok"])
        self.assertEqual(loc["reason"], "pose_stale")

    def test_7_stop_head_is_parallel_with_safety_gate(self):
        """stop_head 与 safety_gate **并联**：两个都能停，安全门优先级更高。"""
        d = pm.decide_stop(stop_head_p=0.95, localization_ok=True)
        self.assertTrue(d["stop"])
        self.assertEqual(d["source"], "stop_head")
        d = pm.decide_stop(safety_veto=True, stop_head_p=0.95, localization_ok=True)
        self.assertEqual(d["source"], "safety_gate")
        d = pm.decide_stop(nav_state="stop", stop_head_p=0.95, localization_ok=True)
        self.assertEqual(d["source"], "nav_state")
        d = pm.decide_stop(stop_head_p=0.95, localization_ok=False,
                           localization_reason="unlocalized")
        self.assertEqual(d["source"], "safety_gate_unlocalized")
        d = pm.decide_stop(stop_head_p=0.10, localization_ok=True)
        self.assertFalse(d["stop"])

    def test_8_stop_head_missing_is_not_a_stop(self):
        d = pm.decide_stop(stop_head_p=None, localization_ok=True)
        self.assertFalse(d["stop"])

    def test_decide_localization_cases(self):
        loc = pm.decide_localization(None, None)
        self.assertTrue(loc["ok"])
        self.assertFalse(loc["detail"]["applicable"])
        loc = pm.decide_localization("localized", 0.1, pose_stale_s=0.5)
        self.assertTrue(loc["ok"])
        self.assertIsNone(loc["reason"])
        loc = pm.decide_localization("unlocalized", 0.1)
        self.assertEqual(loc["reason"], "unlocalized")
        loc = pm.decide_localization("unknown", 0.1, state_reason="not_relocalized")
        self.assertEqual(loc["reason"], "unknown")
        # state=unknown 且 reason=pose_stale：报更具体的 pose_stale
        loc = pm.decide_localization("unknown", None, state_reason="pose_stale")
        self.assertEqual(loc["reason"], "pose_stale")
        # 质量阈值当前是占位（quality_max=None）：给多大的 quality 都不影响
        loc = pm.decide_localization("localized", 0.1, quality=999.0, quality_max=None)
        self.assertTrue(loc["ok"])
        self.assertEqual(loc["detail"]["quality_max"], None)
        # 接入真实指标后的语义（留桩，证明开关是通的）
        loc = pm.decide_localization("localized", 0.1, quality=3.0, quality_max=1.0)
        self.assertEqual(loc["reason"], "quality_over_threshold")

    def test_executor_enforces_localization_gate(self):
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        ex.submit(_block(0.6), 100.0,
                  {"localization_ok": False, "localization_reason": "pose_stale"})
        self.assertTrue(ex._gated)
        self.assertEqual(float(abs(ex._block).sum()), 0.0)
        self.assertEqual(ex._stop_source, "safety_gate_unlocalized")
        self.assertEqual(ex._stop_reason, "pose_stale")

    def test_executor_no_localization_info_does_not_stop(self):
        """无导航子系统（localization_ok=None）时不得停车，否则独立 VLA 永远不动。"""
        ex = pe.PolicyExecutor(send_axes=lambda x, y, d: True, dry_run=True)
        ex.submit(_block(0.6), 100.0, {"localization_ok": None})
        self.assertFalse(ex._gated)
        self.assertNotEqual(float(abs(ex._block).sum()), 0.0)


class ModeMappingTests(unittest.TestCase):
    """NavSession.mode → 运行时停止决策；**不再进入记忆卡**（v2，2026-11-06）。"""

    def test_from_session_mode_table(self):
        self.assertEqual(pm.from_session_mode("goto"), pm.MODE_FOLLOW)
        self.assertEqual(pm.from_session_mode("explore"), pm.MODE_EXPLORE)
        self.assertEqual(pm.from_session_mode("idle"), pm.MODE_STOP)
        self.assertEqual(pm.from_session_mode("done"), pm.MODE_STOP)
        self.assertEqual(pm.from_session_mode("replan"), pm.MODE_REPLAN)
        # 未知状态按 follow：状态名漂移不该凭空造出停车条件
        self.assertEqual(pm.from_session_mode("weird"), pm.MODE_FOLLOW)
        self.assertEqual(pm.from_session_mode(None), pm.MODE_FOLLOW)

    def test_policy_text_uses_shared_constants(self):
        import policy_text as pt
        self.assertEqual(pt.MODE_STOP, pm.MODE_STOP)
        self.assertEqual(pt.MODE_NAMES, pm.MODE_NAMES)
        # 卡 v2：mode **不进**模型输入（2026-11-06 拍板）。
        txt = pt.build_policy_text((1.0, 2.0), None, (0.0, 0.0))
        self.assertNotIn("[mode]", txt)
        with self.assertRaises(ValueError):
            pt.assert_card_clean("[mode] 2 [wp] 1.00 2.00")

    def test_follower_axes_is_not_a_mode(self):
        self.assertEqual(pm.from_follower_axes(0.0, 0.0), "hold")
        self.assertEqual(pm.from_follower_axes(0.0, 1.0), "turn")
        self.assertEqual(pm.from_follower_axes(1.0, 0.0), "drive")
        self.assertNotIn(pm.from_follower_axes(0.0, 0.0),
                         [v for v in pm.MODE_NAMES.values()])

    def test_nav_online_uses_single_mapping(self):
        """nav_online 里不得再出现第二套 _policy_mode 推导（源码级契约）。"""
        src = (ROOT / "backend" / "nav_online.py").read_text(encoding="utf-8")
        self.assertIn("from_session_mode(session_mode)", src)
        self.assertNotIn("self._policy_mode =", src)
        self.assertIn("_follower_motion_mode", src)


class MetricContractTests(unittest.TestCase):
    """summary.json 口径的源码级契约（避免有人把字段悄悄删掉）。"""

    def test_train_summary_fields_present(self):
        src = (ROOT / "research" / "tools" / "train_policy_lora.py").read_text(
            encoding="utf-8")
        for key in ('"stop_recall_source"', '"near_collision"', '"traj_dev_m"',
                    '"mse12"', '"best_metric"', '"selection"'):
            self.assertIn(key, src, "summary.json 缺少 %s" % key)
        self.assertIn('--best-metric', src)
        self.assertIn('--min-stop-recall', src)

    def test_runtime_has_no_wp_zeroing(self):
        """runtime 里不得再出现 `if gated: block = zeros` 那条旧逻辑。"""
        src = (ROOT / "research" / "tools" / "policy_runtime.py").read_text(
            encoding="utf-8")
        self.assertNotIn("gated = float(np.hypot", src)
        self.assertIn("wp_near_zero", src)
        self.assertIn("decide_stop", src)


if __name__ == "__main__":
    unittest.main()
