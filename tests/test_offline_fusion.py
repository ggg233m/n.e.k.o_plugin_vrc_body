# -*- coding: utf-8 -*-
"""``research/tools/offline_fusion.py`` 的回归测试。

为什么单独有这个文件：2026-10-08 那次改动里，``main()`` 里一段内联代码用了尚未赋值的
``H, W``（`UnboundLocalError`），而当时的验证脚本是直接调 ``aggregate`` / ``classify`` 的，
**绕过了命令行入口** ⇒ 55 个测试全绿，而工具根本跑不起来，还被提交了。
教训：**"函数级验证通过"不等于"入口能跑"**。这里两层都盖。

第一层（必跑，无外部依赖）：``veto_grid`` 的稀疏→稠密对齐。
第二层（录制不在就 skip）：真的跑一次 ``main()``。
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "research" / "tools" / "offline_fusion.py"


def _load_tool():
    """按路径加载工具模块（``research/`` 不是包，没有 ``__init__.py``）。"""
    spec = importlib.util.spec_from_file_location("offline_fusion_under_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


OF = _load_tool()


def _smallest_recording() -> Path | None:
    """关键帧最少的那个录制。冒烟测试要跑整条重放，挑最小的以控制耗时
    （最大那场 3000+ kf，单跑就要一分多钟，不该压在单测里）。"""
    cands = [p.parent for p in (ROOT / "navmesh_recordings").glob("*/events.jsonl")]
    if not cands:
        return None
    return min(cands, key=lambda d: len(list((d / "kf").glob("*.npz"))) or 10 ** 9)


REC = _smallest_recording()


class VetoGridTests(unittest.TestCase):
    """``veto_grid``：稀疏格键 → 稠密 ``(H, W)`` 掩码。"""

    @staticmethod
    def _agg(ix, iy, ray_clear=True, sparse=None):
        """构造一个最小 agg：``ray_veto`` 被替换掉，只测对齐。"""
        n = len(ix)
        return dict(ix=np.asarray(ix, np.int64), iy=np.asarray(iy, np.int64),
                    traj=np.zeros((0, 2)), n_kf=0, n_refresh=0, pose_fallback=0,
                    cam_h=1.73, ray_clear=ray_clear,
                    ray_hit_key=np.zeros(0, np.int64), ray_hit_n=np.zeros(0),
                    ray_mis_key=np.zeros(0, np.int64), ray_mis_n=np.zeros(0),
                    _sparse=sparse)

    def _run(self, agg, x0, y0, shape, sparse):
        orig = OF.ray_veto
        OF.ray_veto = lambda a, **kw: np.asarray(sparse, bool)
        try:
            return OF.veto_grid(agg, x0, y0, shape)
        finally:
            OF.ray_veto = orig

    def test_offsets_map_to_the_right_cells(self) -> None:
        # flat = (ix - x0) * W + (iy - y0)，与 dense() 的铺法逐字一致。
        agg = self._agg([3, 4, 5], [7, 7, 9])
        x0, y0, W = 3, 7, 5
        shape = (4, W)
        got = self._run(agg, x0, y0, shape, [True, False, True])
        self.assertIsNotNone(got)
        # 显式算 flat，不写 2*W 这种"看起来对"的式子（第一版就是那样写错的）
        f_a = (3 - x0) * W + (7 - y0)          # 0
        f_b = (4 - x0) * W + (7 - y0)          # 5
        f_c = (5 - x0) * W + (9 - y0)          # 12
        self.assertEqual((f_a, f_b, f_c), (0, 5, 12))
        self.assertTrue(got.flat[f_a], "(3,7) 的票是 True")
        self.assertFalse(got.flat[f_b], "(4,7) 的票是 False")
        self.assertTrue(got.flat[f_c], "(5,9) 的票是 True")
        self.assertEqual(int(got.sum()), 2)
        # 分辨力：把 W 用错（当成 H）会把这些格铺到别处，格数就不对
        self.assertEqual(got.shape, shape)

    def test_none_when_ray_clear_off(self) -> None:
        agg = self._agg([0], [0], ray_clear=False)
        self.assertIsNone(OF.veto_grid(agg, 0, 0, (2, 2)))

    def test_out_of_range_raises_instead_of_silently_wrapping(self) -> None:
        """越界必须**报错**：`flat` 越界在 numpy 里是 IndexError，但负下标会静默环绕。"""
        agg = self._agg([-5], [0])
        with self.assertRaises(ValueError):
            self._run(agg, 0, 0, (2, 2), [True])


@unittest.skipUnless(REC is not None, "navmesh_recordings/ 不在（录制不入库）")
class MainSmokeTests(unittest.TestCase):
    """真的跑一次 ``main()`` —— 拦的正是"函数都对、入口跑不起来"那一类。"""

    def test_main_runs_end_to_end(self) -> None:
        argv = ["offline_fusion.py", REC.name, "--out", str(ROOT / ".tmp" / "smoke_of")]
        old_argv = sys.argv
        sys.argv = argv
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                OF.main()                     # 不写 prior：只跑渲染与指标
            out = buf.getvalue()
        finally:
            sys.argv = old_argv
        self.assertIn("fused_rings", out, "跑完了都没输出指标 ⇒ main 没走到底")
        self.assertIn("transitions", out)


class AlignGateTests(unittest.TestCase):
    """``gate_align_quality``：``--write-prior`` 前的**对齐残差闸**。

    背景：世界树的 ``bridge_min`` 只保证约束**条数**、不保证残差小 ⇒
    **「进了树」≠「质量合格」**。2026-10-15 手工核对发现，留出严重不达标
    （0.566 / 1.220 / 5.143 m，甚至留出 n=0 根本无法验证）的会话照样能写进先验 ⇒
    错 gauge 混进图 = 废图 + 假障碍自困，且不可逆。这里把它变成自动闸。

    判据只用**留出残差**，不看旋转：``gauge.R_dev_deg`` 与质量不相关
    （001523 旋转 11.01° 但留出 0.270 最好；223033 旋转 3.63° 却 0.566 不合格）。
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.world = Path(self.tmp.name)
        xs = self.world / "xsession"
        xs.mkdir(parents=True)
        self._put(xs, "good", 0.270, 41)         # 与 001523 同档
        self._put(xs, "edge", 0.354, 8)          # 现有先验里最差的那场，必须放行
        self._put(xs, "bad_resid", 1.220, 6)     # 033237
        self._put(xs, "mid_resid", 0.566, 10)    # 223033
        self._put(xs, "few_holdout", 0.100, 2)   # 残差小但样本不足，中位不可信
        self._put(xs, "no_holdout", None, 0)     # 完全无法验证
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def _put(xs: Path, sid: str, med: float | None, n: int) -> None:
        (xs / f"align_{sid}_into_root.json").write_text(
            json.dumps({"new_sid": sid, "old_sid": "root", "base_sid": "root",
                        "holdout": {"n": 0, "med_m": None} if med is None
                        else {"n": n, "med_m": med}}),
            encoding="utf-8")

    def test_root_session_passes(self) -> None:
        """坐标系根就是基准，没有可对齐的对象 ⇒ 必须放行，否则永远写不出先验。"""
        self.assertEqual(OF.gate_align_quality(self.world, ["root"], "root"), [])

    def test_good_and_edge_pass(self) -> None:
        """现有先验三场 0.169 / 0.270 / 0.354 必须全过 —— 闸不该把已验证合格的挡在外面。"""
        self.assertEqual(OF.gate_align_quality(self.world, ["good", "edge"], "root"), [])

    def test_bad_residual_rejected(self) -> None:
        p = OF.gate_align_quality(self.world, ["bad_resid", "mid_resid"], "root")
        self.assertEqual(len(p), 2)
        self.assertIn("1.220", p[0])
        self.assertIn("0.566", p[1])

    def test_few_holdout_rejected(self) -> None:
        """残差再小，留出 2 个的中位也没有统计力 ⇒ 不能放行。"""
        p = OF.gate_align_quality(self.world, ["few_holdout"], "root")
        self.assertEqual(len(p), 1)
        self.assertIn("2", p[0])

    def test_no_holdout_rejected(self) -> None:
        self.assertEqual(len(OF.gate_align_quality(self.world, ["no_holdout"], "root")), 1)

    def test_missing_alignment_rejected(self) -> None:
        """从来没对齐过的会话：位姿可能根本不在主 gauge 下。"""
        p = OF.gate_align_quality(self.world, ["never_aligned"], "root")
        self.assertEqual(len(p), 1)
        self.assertIn("对齐记录", p[0])

    def test_mixed_batch_reports_only_bad(self) -> None:
        p = OF.gate_align_quality(
            self.world, ["root", "good", "bad_resid", "no_holdout"], "root")
        self.assertEqual(len(p), 2, "根与合格场不该出现在问题列表里")

    def test_threshold_has_discriminating_power(self) -> None:
        """反向验证：阈值放大到荒谬值后，**阈值类**问题必须全部放行。

        ⇒ 证明拦截真的来自阈值，而不是"任何带 align 记录的都会被拒"这种假通过。
        只放受阈值控制的两类（残差超限、样本不足）；``no_holdout`` 不在此列，见下一条。
        """
        self.assertEqual(
            OF.gate_align_quality(self.world, ["bad_resid", "few_holdout"],
                                  "root", max_m=100.0, min_n=0),
            [], "放宽阈值后仍被拒 ⇒ 这条闸没有分辨力")

    def test_unverifiable_cannot_be_bought_off(self) -> None:
        """「无法验证」是硬拒，放宽阈值也买不通 —— 与「残差超限」是两类不同的问题。

        没有留出样本就是没有证据，调大阈值并不会让它变得有证据。若这条挂了，说明
        有人把 ``unverified`` 也塞进了阈值分支 ⇒ 就等于开了个后门。
        """
        p = OF.gate_align_quality(self.world, ["no_holdout"], "root",
                                  max_m=100.0, min_n=0)
        self.assertEqual(len(p), 1, "放宽阈值后 n=0 竟被放行 ⇒ 闸被开后门了")

    def test_default_threshold_matches_known_good(self) -> None:
        """默认阈值必须落在「已知合格 0.354」与「已知不合格 0.566」之间。"""
        self.assertGreater(OF.MAX_HOLDOUT_M, 0.354)
        self.assertLess(OF.MAX_HOLDOUT_M, 0.566)


class WorldScaleGateTests(unittest.TestCase):
    """``check_world_scale``：world_scale **不一致不再拒绝**，降级为警告。

    依据（2026-10-16 实证，``.tmp/ws_grid_invariance.py``）：同一场录制只改 world_scale
    （0.755 ↔ 0.840），``aggregate → dense → classify`` 全链产出的三态栅格**逐格完全一致**。
    离线建图全程在追踪米空间，ws 只用于"追踪米 ↔ 世界米"的对外换算，不进栅格
    ⇒ 换 avatar 的会话，栅格仍在同一追踪米空间。

    ⚠️ 这不是「放宽阈值让它成功」：真正的尺度风险改由**对齐残差闸**把关（直接判据），
    ws 一致与否只是个错代理指标。
    """

    def test_missing_is_still_fatal(self) -> None:
        """缺 ws ⇒ 先验记不下基准尺度，仍然是致命错。"""
        fatal, _ws, _warn = OF.check_world_scale({"a": 0.755, "b": None})
        self.assertIsNotNone(fatal)
        self.assertIn("缺 world_scale", fatal)

    def test_consistent_is_clean(self) -> None:
        fatal, ws, warn = OF.check_world_scale({"a": 0.755, "b": 0.755})
        self.assertIsNone(fatal)
        self.assertIsNone(warn)
        self.assertAlmostEqual(ws, 0.755)

    def test_inconsistent_warns_instead_of_refusing(self) -> None:
        """核心：换 avatar 不再拦 ⇒ 但必须留下警告，别悄悄放行。"""
        fatal, ws, warn = OF.check_world_scale({"a": 0.755, "b": 0.840})
        self.assertIsNone(fatal, "ws 不一致不该再拒绝：栅格已实证与 ws 无关")
        self.assertIsNotNone(warn, "不拒绝但必须告警，静默放行会更难查")
        self.assertAlmostEqual(ws, 0.755)

    def test_root_scale_is_reported(self) -> None:
        """基准 ws 必须是有效正数，写先验时要存它。"""
        _fatal, ws, _warn = OF.check_world_scale({"a": 0.840, "b": 0.755, "c": 0.900})
        self.assertTrue(np.isfinite(ws) and ws > 0.0)


if __name__ == "__main__":
    unittest.main()
