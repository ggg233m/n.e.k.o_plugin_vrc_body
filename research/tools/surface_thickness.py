# -*- coding: utf-8 -*-
r"""密度无关的表面厚度：**固定半径邻域主平面残差**（拖影的可用指标）。

为什么不能用「同心环间距」或「体素内点数」当拖影量（2026-10-27 实测）：
同一份基线点云，原样测环距 54 cm，随机抽稀到 34% 就变 **16 cm** —— 同一份数据两个答案，
该指标被**点密度**污染。本工具改用与密度无关的口径：在每个锚点周围取**固定半径**的点，
做主平面拟合，取最小特征值方向的 std 作为「面厚」。单帧自厚基准约 3.9–4.7 cm。

⚠️ 另一个陷阱（同一天踩到）：拿面厚去评价一个**删了点**的版本会得到假改善。
实测把 5 m 距离闸收到 3 m，面厚 4.0→1.9 cm，但把基线**截到同一批空间格**再测是 3.0 cm
⇒ 2.1 cm 的改善里 1.0 cm 纯粹来自「少画了远场」。评价删点版本必须用 --same-cells。

用法：
    .venv/Scripts/python.exe research/tools/surface_thickness.py --html .tmp/pointcloud/world.html
    # 两个导出对照（自动截到共同空间格）
    .venv/Scripts/python.exe research/tools/surface_thickness.py --html a.html b.html --same-cells

报告口径：p50 / p75 / p90（cm）。p50 是主体面的厚度，p90 是拖影尾巴。
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
STRIDE = 8          # pointcloud_view 主布局：u16 x, u16 y, u8 z, u8 sid, u8 t, u8 hits
XYO, XYS, ZS, ZO = 32.0, 1000.0, 50.0, 1.0


def load_html(path: Path) -> np.ndarray:
    """从 pointcloud_view 导出的自包含 HTML 里取点坐标（世界系米）。"""
    h = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'const RAW = atob\("([A-Za-z0-9+/=]*)"\)', h)
    if not m:
        raise SystemExit(f"[fail] {path} 里找不到 RAW 载荷（不是 pointcloud_view 的导出？）")
    a = np.frombuffer(base64.b64decode(m.group(1)), np.uint8)
    if len(a) % STRIDE:
        raise SystemExit(f"[fail] {path} 载荷长度 {len(a)} 不是 {STRIDE} 的整数倍")
    a = a.reshape(-1, STRIDE)
    x = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)) / XYS - XYO
    y = (a[:, 2].astype(np.int32) | (a[:, 3].astype(np.int32) << 8)) / XYS - XYO
    z = a[:, 4].astype(np.float64) / ZS - ZO
    return np.column_stack([x, y, z])


def voxelize(P: np.ndarray, v: float) -> np.ndarray:
    """体素降采样到统一网格 —— 不同导出的点密度不同，不先统一就没法比。"""
    q = np.floor(P / v).astype(np.int64)
    key = q[:, 0] * (1 << 42) + q[:, 1] * (1 << 21) + q[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    acc = np.zeros((len(uk), 3))
    np.add.at(acc, inv, P)
    return acc / np.bincount(inv).astype(float)[:, None]


def thickness(V: np.ndarray, anchors: np.ndarray, radius: float, minpts: int,
              flat_ratio: float = 0.25) -> np.ndarray:
    """锚点周围固定半径内做主平面拟合，返回最小特征值方向 std（米）。

    flat_ratio 用来只保留**像平面**的邻域（角/杆/边不算）：若第一特征值 > ratio × 第二
    特征值，说明这一点不像一张面，跳过。不跳的话墙角会被当成很厚的面。
    """
    tree = cKDTree(V)
    out = []
    for c in anchors:
        idx = tree.query_ball_point(c, radius, workers=-1)
        if len(idx) < minpts:
            continue
        Q = V[idx] - V[idx].mean(0)
        w, _ = np.linalg.eigh(Q.T @ Q / len(Q))
        if w[0] > flat_ratio * w[1]:
            continue
        out.append(np.sqrt(max(w[0], 0.0)))
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("html", nargs="+", help="pointcloud_view 导出的 HTML（1 个=单档；2 个=对照）")
    ap.add_argument("--grid", type=float, default=0.15,
                    help="统一体素边长（米），默认 0.15。两个导出必须用同一值才可比")
    ap.add_argument("--radius", type=float, default=0.35,
                    help="面拟合邻域半径（米），默认 0.35。要显著大于 --grid")
    ap.add_argument("--minpts", type=int, default=14, help="邻域内最少点数，默认 14")
    ap.add_argument("--anchors", type=int, default=1500, help="抽样锚点数，默认 1500")
    ap.add_argument("--same-cells", action="store_true",
                    help="只保留所有导出都覆盖到的空间格（评价删点版本时必须开）")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    paths = [Path(p) for p in a.html]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"[fail] 没有这个文件：{p}")
    raw = [load_html(p) for p in paths]
    V = [voxelize(P, a.grid) for P in raw]
    for p, P_, Vv in zip(paths, raw, V):
        print(f"[load ] {p.name:32s} 点 {len(P_):9,} → 体素 {len(Vv):9,}", flush=True)

    def cellset(X):
        q = np.floor(X / a.grid).astype(np.int64)
        return q[:, 0] * (1 << 42) + q[:, 1] * (1 << 21) + q[:, 2]

    rng = np.random.default_rng(a.seed)
    base = V[0]
    if a.same_cells and len(V) > 1:
        common = set(cellset(V[0]).tolist())
        for X in V[1:]:
            common &= set(cellset(X).tolist())
        if not common:
            raise SystemExit("[fail] 两个导出没有共同空间格（坐标系不同？）")
        keep = np.array([c in common for c in cellset(base).tolist()])
        base = base[keep]
        print(f"[cells] 共同空间格 {len(common):,}（占首个导出 {100*keep.mean():.1f}%）", flush=True)
    n = min(a.anchors, len(base))
    anchors = base[rng.choice(len(base), n, replace=False)]

    print()
    print(f"{'导出':34s} {'点':>9s} {'体素':>9s} {'面厚p50':>9s} {'p75':>7s} {'p90':>7s}")
    for p, P_, Vv in zip(paths, raw, V):
        th = thickness(Vv, anchors, a.radius, a.minpts)
        if not len(th):
            print(f"{p.name:34s}   [skip] 没有有效平面邻域（--radius 太小或点太稀）")
            continue
        q = np.percentile(th, [50, 75, 90]) * 100
        print(f"{p.name:34s} {len(P_):9,} {len(Vv):9,} {q[0]:8.2f}cm {q[1]:6.2f} {q[2]:6.2f}")
    print()
    print("判读：单帧自厚基准 3.9–4.7 cm。p50 明显高于它 = 同一张面被多帧错开画了多份。")
    print("⚠️ 面厚只反映叠了几层，不能单独用来判位姿好坏（见模块 docstring 的删点陷阱）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
