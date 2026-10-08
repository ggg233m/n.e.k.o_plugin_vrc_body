# -*- coding: utf-8 -*-
"""``research.tools.pose_graph4.PoseGraph4`` 的残差对拍回归门。

背景：`Docs/archive/回环优化耗时剖析（2026-09-29）.md`。残差函数从「纯 Python 逐边循环」改成
「整批 einsum 向量化」，单次 9.5 ms → 0.157 ms、整体 ÷15。
**向量化写错了不报错，只会把解算器带到别处**（剖析时两次写错公式：一次把 inv(Z) 的旋转
又转置了，一次把平移项写成 R(t − t_Z)，残差能差到 205、cost 从 302 跑到 13028）。
所以这个门的任务就是：**向量化实现必须与朴素实现逐元素一致，且优化结果不能变。**

按 `python tests/test_posegraph4_residual.py` 直接跑（本仓库 tests 无法用 pytest 收集）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.tools.pose_graph4 import PoseGraph4  # noqa: E402

N_NODES = 12
ATOL = 1e-9          # 残差逐元素容差（实测两者差 1.6e-13）
COST_RTOL = 0.02     # optimize 末值 cost 相对容差（实测差 0.1%）


def _step(rng: np.random.Generator) -> np.ndarray:
    """一节航位推算：带 roll/pitch（不能只有 yaw，否则 _tilt 恒为单位阵，测不出 tilt 相关的错）。"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rng.normal(0, [0.25, 0.25, 0.6])).as_matrix()
    T[:3, 3] = rng.normal(0, 0.3, 3)
    return T


def build(seed: int = 0, n: int = N_NODES, perturb: float = 0.05):
    """造一张图：n 个节点，若干条被扰动过的回环边（残差非零）。"""
    rng = np.random.default_rng(seed)
    odom = {}
    T = np.eye(4)
    for k in range(1, n + 1):
        T = T @ _step(rng)
        odom[k] = T.copy()
    pg = PoseGraph4(odom)
    for k in range(1, n + 1):
        pg.add_node(k)
    loops = []
    for a in (1, 3, 5, 8):
        b = a + 4
        if b > n:
            continue
        Z = np.linalg.inv(odom[a]) @ odom[b]
        Z[:3, 3] += rng.normal(0, perturb, 3)          # 回环观测与 odom 不一致 ⇒ 残差非零
        loops.append((a, b, Z))
    for a, b, Z in loops:
        pg.add_loop(a, b, Z)
    return pg


class ResidualParityTests(unittest.TestCase):
    """向量化残差 == 朴素残差。"""

    def test_parity_at_current_x(self):
        pg = build()
        c = pg.prepare()
        x0 = np.concatenate([pg.x[k] for k in pg.ids[1:]])
        dv = float(np.max(np.abs(pg.residual(x0, c) - pg.residual_reference(x0, c))))
        self.assertLess(dv, ATOL, f"与朴素残差的最大绝对差 {dv:.3e} 超过 {ATOL:.0e}")

    def test_parity_at_perturbed_x(self):
        """不能只在当前 x 上比——解算器会在别处求值，那里也必须一致。"""
        pg = build()
        c = pg.prepare()
        x0 = np.concatenate([pg.x[k] for k in pg.ids[1:]])
        rng = np.random.default_rng(1)
        worst = 0.0
        for scale in (1e-3, 1e-2, 0.1, 1.0):
            x = x0 + rng.normal(0, scale, x0.shape)
            worst = max(worst, float(np.max(np.abs(pg.residual(x, c) - pg.residual_reference(x, c)))))
        self.assertLess(worst, ATOL, f"扰动后最大绝对差 {worst:.3e}")

    def test_zero_input_control(self):
        """零输入控制组：位姿与回环全是单位阵 ⇒ 残差应全零。两个实现都得是零。"""
        odom = {k: np.eye(4) for k in range(1, N_NODES + 1)}
        pg = PoseGraph4(odom)
        for k in range(1, N_NODES + 1):
            pg.add_node(k)
        pg.add_loop(1, 5, np.eye(4))
        pg.add_loop(2, 9, np.eye(4))
        c = pg.prepare()
        z = np.zeros(4 * (N_NODES - 1))
        self.assertLess(np.max(np.abs(pg.residual(z, c))), 1e-12)
        self.assertLess(np.max(np.abs(pg.residual_reference(z, c))), 1e-12)

    def test_residual_is_not_trivially_zero(self):
        """守卫：对照组之外残差必须真的非零，否则上面几条会在常数函数上白绿。"""
        pg = build()
        c = pg.prepare()
        x0 = np.concatenate([pg.x[k] for k in pg.ids[1:]])
        self.assertGreater(float(np.max(np.abs(pg.residual(x0, c)))), 1e-3)


class OptimizeEquivalenceTests(unittest.TestCase):
    """换成向量化残差后，optimize 的结果不能变。"""

    def test_cost_and_solution_match(self):
        c_ref = build().optimize(reference_residual=True)
        pg_vec = build()
        c_vec = pg_vec.optimize()
        self.assertGreater(c_ref, 0.0, "回环被扰动过，cost 不应为 0（守卫）")
        self.assertLess(abs(c_vec - c_ref) / max(c_ref, 1e-9), COST_RTOL,
                        f"cost 变化过大：朴素 {c_ref:.4f} vs 向量化 {c_vec:.4f}")

        pg_ref = build()
        pg_ref.optimize(reference_residual=True)
        d = max(float(np.max(np.abs(pg_vec.x[k] - pg_ref.x[k]))) for k in pg_vec.ids[1:])
        self.assertLess(d, 0.05, f"优化后位姿差 {d:.4f}（追踪米/弧度）过大")

    def test_optimize_actually_reduces_cost(self):
        """守卫：optimize 必须真的在降 cost，否则上面那条会在两个一样大的数上白绿。"""
        pg = build()
        c = pg.prepare()
        x0 = np.concatenate([pg.x[k] for k in pg.ids[1:]])
        before = float(np.sum(0.5 * pg.residual(x0, c) ** 2))
        after = pg.optimize(max_nfev=50)
        self.assertLess(after, before, f"优化后 cost {after:.4f} 未低于优化前 {before:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
