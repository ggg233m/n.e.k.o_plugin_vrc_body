# -*- coding: utf-8 -*-
"""点云导出的**口径**必须与建图侧一致 —— 这里钉住那条口径。

背景（2026-10-08）：`pointcloud_view` 的距离闸写的是 `np.linalg.norm(p, axis=1)`
（三维范数），而 mapper 是 `np.hypot(p[:,0], p[:,1])`（水平半径）。
base 系里地板点在 z ≈ −cam_h ⇒ 三维范数把远处**地面**算长、误丢。
实测 044153：水平口径留 79.95% / 三维口径留 73.20%，少留 8.44% 的应留点。

这类"同一个阈值的两套算法"在本仓已经出现过两次（在线/离线判定、轨迹掩码），
所以不靠"注释里写清楚"，直接断言两者**在同一批点上取同一子集**。
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_VIEW = ROOT / "research" / "tools" / "pointcloud_view.py"
_MAPPING = ROOT / "backend" / "nav_mapping.py"


def _range_gate_expr(path: Path) -> set[str]:
    """找出该文件里"距离闸"那一行用到的距离函数名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = f"{getattr(node.func.value, 'id', '?')}.{node.func.attr}"
            if name in ("np.hypot", "np.linalg.norm"):
                found.add(name)
    return found


class RangeGateCaliberTests(unittest.TestCase):
    def test_mapper_uses_horizontal_radius(self) -> None:
        self.assertIn("np.hypot", _range_gate_expr(_MAPPING))
        self.assertNotIn("np.linalg.norm", _range_gate_expr(_MAPPING))

    def test_viewer_uses_the_same_gate_as_mapper(self) -> None:
        """导出侧不得再用三维范数做距离闸（那会把远处地板误丢）。"""
        got = _range_gate_expr(_VIEW)
        self.assertIn("np.hypot", got, "导出必须用水平半径")
        self.assertNotIn("np.linalg.norm", got,
                         "导出出现三维范数 ⇒ 与 mapper 口径不一致（044153 上少留 8.44% 应留点）")

    def test_two_calibers_actually_differ(self) -> None:
        """证明"这两套算法确实会给出不同子集"——否则上面两条断言没有分辨力。"""
        rng = np.random.default_rng(0)
        p = np.column_stack([rng.uniform(-6, 6, 4000), rng.uniform(-6, 6, 4000),
                             rng.uniform(-2.0, 2.0, 4000)])
        rm = 5.0
        horiz = np.hypot(p[:, 0], p[:, 1]) <= rm
        three = np.linalg.norm(p, axis=1) <= rm
        diff = int((horiz & ~three).sum())
        self.assertGreater(diff, 0, "两套口径应当真的有差，否则这个测试没必要存在")
        # 差异点必然在半径边缘，且 |z| 越大差得越多
        if diff:
            d = p[horiz & ~three]
            self.assertTrue(np.all(np.hypot(d[:, 0], d[:, 1]) > 4.0))


if __name__ == "__main__":
    unittest.main()