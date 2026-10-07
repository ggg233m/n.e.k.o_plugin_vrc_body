# -*- coding: utf-8 -*-
r"""轨迹 → 栅格格掩码的**唯一实现**。所有工具都从这里取，不许再各推一遍旋转/转置。

为什么抽出这个模块
------------------
2026-10-08 核查发现：``mapping_gate._trail_mask``、``q_tier_ab.ring_masks``、
``q_tier_ab.near_ground_mask``、``precision_probe.corridor_blockers`` **四处**都把轨迹点
写成了 ``cv2.circle(m, (int((p[1]-oy)/r), int((p[0]-ox)/r)))``。``NavGrid`` 的行轴是 **y 且自下而上**
（``row = H-1-(y-oy)/res``，见 ``nav_grid.to_cell``），这四处的写法把 x 当成了行、又漏了上下翻转。

后果不是报错，是**量错了地方**：044153 上门的 ``_trail_mask(0.5)`` 与正确掩码的 IoU = **0.090**
（重叠 3041 / 各 18469 格），落在掩码里的障碍格从 505 变成 **899**。也就是"走廊障碍"这个
用来判断"能不能走过去"的指标，一直在数一片**离轨迹很远的镜像区域**。同一录制上，正确朝向
的轨迹点 93.7% 落在 FREE、2.77% 落在 OCC；按转置读法只有 60.4% 落在 FREE。

为什么四处都写错：``q_tier_ab`` 的变量名叫 ``cx/cy`` 而 ``cx`` 其实是行号，抄名字的人
再配一个 ``cv2.circle(m, (cx, cy))`` 就翻了。本模块只吃 ``NavGrid.to_cell``，不自己算格子。
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["trail_cells", "trail_mask"]


def trail_cells(ng, trail) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """轨迹点（**追踪米**，未乘 world_scale 的 T_map 平面坐标）→ ``(row, col, inb)``。

    口径只此一处：``to_cell`` 吃**世界米**，所以这里乘 ``world_scale`` 再交出去
    （``to_cell`` 内部会除回来，净效果是 ``(p - origin)/res``，与 ``origin_xy_m`` 同口径）。
    """
    p = np.asarray(trail, np.float64).reshape(-1, 2)
    ws = ng.meta.world_scale
    h, w = ng.grid.shape
    row = np.empty(len(p), np.int64)
    col = np.empty(len(p), np.int64)
    for i, q in enumerate(p):
        r, c = ng.to_cell([float(q[0]) * ws, float(q[1]) * ws])
        row[i], col[i] = r, c
    inb = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    return row, col, inb


def trail_mask(ng, trail, radius_m: float, *, connect: bool = True) -> np.ndarray:
    """轨迹两侧 ``radius_m``（**追踪米**）内的格掩码（``NavGrid`` 行序）。

    ⚠️ **单位**：``radius_m`` 与 ``trail`` 同为追踪米，与 ``origin_xy_m`` 同口径，
    所以格半径 = ``radius_m / resolution_m``。这**不是** ``NavGrid.walked_mask`` 的口径——
    那个吃的是导航系**世界米**折线、格半径 = ``half_m / cell_world_m``。两者差一个
    ``world_scale``（044153 上 0.755 ⇒ 差 1.32×）。项目在单位上栽过（C30 的视差/网格混用），
    所以这里把口径写死在 docstring 里，不留给调用方猜。

    ``connect=True``：连成折线（默认）。身体走的是**连续**路径，而关键帧位姿只是它的采样，
    相邻采样点之间可能隔好几米——只按点画圆会在采样稀疏处留下一串空洞，把"走廊"量小了。
    这与运行时的 ``NavGrid.walked_mask`` 用的是同一套画法（那里也是折线），所以门量的走廊
    与 ``build`` 认定的走廊是同一个东西。

    ``connect=False``：逐点画圆（旧实现语义，用于把"朝向修复"与"连线修复"分开计量）。
    """
    p = np.asarray(trail, np.float64).reshape(-1, 2)
    h, w = ng.grid.shape
    m = np.zeros((h, w), np.uint8)
    rpx = max(1, int(round(radius_m / ng.meta.resolution_m)))
    row, col, inb = trail_cells(ng, p)
    if connect and len(p) >= 2:
        # 折线要 uv=(col,row)；只保留在界内的点，否则越界的点会把线拽回来。
        uv = np.column_stack([col[inb], row[inb]]).astype(np.int32)
        if len(uv) >= 2:
            cv2.polylines(m, [uv.reshape(-1, 1, 2)], False, 1, max(1, 2 * rpx))
        return m > 0
    for r, c in zip(row[inb], col[inb]):
        cv2.circle(m, (int(c), int(r)), rpx, 1, -1)
    return m > 0


def _selftest() -> None:
    """有分辨力的自检：**必须**能区分正确与转置两种读法，否则这个自检等于没有。

    教训（本仓已有一次）：自检的两个分支算同一个表达式 = 没有分辨力，比没有自检更糟。
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from backend.nav_grid import NavGrid, GridMeta

    h, w, res = 20, 30, 0.1                       # H≠W，转置必然越界 ⇒ 能抓住
    g = np.zeros((h, w), np.uint8)
    ng = NavGrid(g, GridMeta(res, (0.0, 0.0), 1.0))
    p = (0.55, 1.25)                              # (x, y) 追踪米
    r, c = ng.to_cell([p[0], p[1]])
    assert (r, c) == (h - 1 - 12, 5), (r, c)      # 手工算一遍：row=7, col=5
    m = trail_mask(ng, [p], 0.05, connect=False)
    assert m[r, c], "正确格必须落在掩码里"
    # 转置读法（旧的 bug）落在别处 —— 断言它**不**等于正确格，证明自检有分辨力
    r_wrong, c_wrong = int(p[0] / res), int(p[1] / res)
    assert (r_wrong, c_wrong) != (r, c), "转置读法与正确读法重合 ⇒ 这个自检没有分辨力"
    assert not m[r_wrong, c_wrong], "转置读法不该在掩码里（旧 bug 会在这里为真）"
    # 连线语义：两点之间必须连通
    m2 = trail_mask(ng, [(0.55, 1.25), (2.55, 1.25)], 0.05, connect=True)
    assert m2[r, c] and m2[r, 25], "折线两端都要在掩码里"
    assert m2[r, 15], "折线中点必须在掩码里（逐点画圆会漏掉它）"
    print("trail_geom self-test: PASS")


if __name__ == "__main__":
    _selftest()
