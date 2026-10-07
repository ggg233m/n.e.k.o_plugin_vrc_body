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
import sys
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


if __name__ == "__main__":
    unittest.main()
