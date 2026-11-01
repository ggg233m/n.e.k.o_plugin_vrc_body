# -*- coding: utf-8 -*-
"""路径形状输入：让 VLA 看得见「前面要左拐」。

为什么需要（2026-11-02 归因）：VLA 原先只拿 local_goal 一个前视点，而那与
pure-pursuit 用的是**同一行代码**（nav_follow.local_goal）。于是模型的信息集与几何
控制器完全相同，轨迹必然是 A* 的时间重参数化 —— 「不会绕障碍」是**输出空间缺转向**
的直接推论（剖析 §15.17），不是训练不足。

这里只解决「看不见弯」：把沿路径前进的 k 个点写进文本。**不解决转向**（头 A 仍只输出
vx/vz），转向仍由跟随器代管。要真正会绕，得让输出含 omega 并重训（M1）。

注意：启用会偏离训练分布（训练集没有这些 token），所以默认关闭、只做 A/B。

坐标系注意：nav_follow 的 (dx, dz) = (右, 前)，theta=0 时**前方是世界 +x**。
这条由 local_goal 的既有实现决定（d[0]*fy - d[1]*fx / d[0]*fx + d[1]*fy），
path_ahead 必须与它逐字一致，否则喂给模型的点与航点不在同一系。
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research" / "tools"))

from nav_follow import PathFollower  # noqa: E402


class PathAheadTests(unittest.TestCase):
    def test_matches_local_goal_for_first_point(self) -> None:
        """第 1 个点必须与同 lookahead 的 local_goal 逐位相等。

        这是最重要的一条：两者不同系的话，模型会同时收到两个矛盾的方向，
        比只给一个点更糟。
        """
        for path, th in (([[0.0, 0.0], [10.0, 0.0]], 0.0),
                         ([[0.0, 0.0], [0.0, 10.0]], 0.0),
                         ([[0.0, 0.0], [5.0, 5.0]], math.pi / 4),
                         ([[0.0, 0.0], [3.0, -4.0]], -0.7)):
            f = PathFollower(path)
            goal = f.local_goal((0.0, 0.0), th, 0.8)
            ahead = f.path_ahead((0.0, 0.0), th, steps=1, step_m=0.8)[0]
            self.assertAlmostEqual(goal[0], ahead[0], places=9, msg=f"{path} th={th}")
            self.assertAlmostEqual(goal[1], ahead[1], places=9, msg=f"{path} th={th}")

    def test_straight_path_points_stay_ahead(self) -> None:
        """直线路径：所有点都在正前方（dz>0, dx≈0），且 dz 递增。"""
        f = PathFollower([[0.0, 0.0], [10.0, 0.0]])
        pts = f.path_ahead((0.0, 0.0), theta=0.0, steps=3, step_m=0.8)
        self.assertEqual(len(pts), 3)
        for dx, dz in pts:
            self.assertAlmostEqual(dx, 0.0, places=6)
            self.assertGreater(dz, 0.0)
        self.assertLess(pts[0][1], pts[1][1])
        self.assertLess(pts[1][1], pts[2][1])

    def test_curved_path_shows_lateral_offset(self) -> None:
        """折线路径：越远的点横向偏移越大 —— 这正是「前面要左拐」的信号。

        单点前视版只给 0.8 m，看不出弯；k 个点会在 dx 上体现出来。
        """
        f = PathFollower([[0.0, 0.0], [0.8, 0.0], [0.8, 3.0], [3.0, 3.0]])
        pts = f.path_ahead((0.0, 0.0), theta=0.0, steps=3, step_m=0.8)
        self.assertEqual(len(pts), 3)
        near_dx = abs(pts[0][0])
        far_dx = max(abs(pts[-1][0]), abs(pts[-2][0]))
        self.assertGreater(far_dx, near_dx + 1e-6, f"远处没体现出横向偏移：{pts}")

    def test_theta_rotates_into_body_frame(self) -> None:
        """theta=+90（朝世界 +y）时，原前方世界 +x 变成机体系「右侧」：dz≈0, dx>0。

        右手系、朝向从 +x 逆时针（同 nav_follow 模块 docstring）。
        """
        f = PathFollower([[0.0, 0.0], [10.0, 0.0]])
        pts = f.path_ahead((0.0, 0.0), theta=math.pi / 2, steps=1, step_m=0.8)
        dx, dz = pts[0]
        self.assertAlmostEqual(dz, 0.0, places=6)
        self.assertGreater(dx, 0.0)

    def test_short_path_clamps_to_end(self) -> None:
        """路径比 k*step_m 短时，后面的点钳到终点，不许越界外推。"""
        f = PathFollower([[0.0, 0.0], [1.0, 0.0]])
        pts = f.path_ahead((0.0, 0.0), theta=0.0, steps=3, step_m=0.8)
        self.assertEqual(len(pts), 3)
        for dx, dz in pts:
            self.assertAlmostEqual(dx, 0.0, places=6)
            self.assertLessEqual(dz, 1.0 + 1e-6)

    def test_steps_zero_returns_empty(self) -> None:
        f = PathFollower([[0.0, 0.0], [5.0, 0.0]])
        self.assertEqual(f.path_ahead((0.0, 0.0), 0.0, steps=0), [])


if __name__ == "__main__":
    unittest.main()