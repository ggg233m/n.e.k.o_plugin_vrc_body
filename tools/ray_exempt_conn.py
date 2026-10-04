# -*- coding: utf-8 -*-
"""``ray_near_exempt`` 的**第三把尺子**：连通性。

    python -m tools.ray_exempt_conn [录制名...]

## 为什么需要第三把

``tools/ray_exempt_ab.py`` 给的两把尺子是：

  尺子 1  走过∩OCC 率          抓路径上的虚假障碍
  尺子 2  近距平面证据的漏放    抓路径外的真结构被清掉

它最后报一个"交换比"（044153 上 135/29 = 4.7:1）。但这个比值的**分子在路径外、
分母在路径上**，前者只影响地图完不完整，后者影响这条道还走不走得通——两者价值
不等价，比值本身证明不了"划算"。

真正决定价格的是：多封死的那几十格**有没有把路堵断**。这就是本工具量的东西。

## 量什么（豁免前后各一次，同一批点对）

  1. 可走区（``center``）的连通分量数——变多 ⇒ 有区域被彻底隔离
  2. 从轨迹起点出发的可达格数——能到的地方少了多少
  3. **相邻采样点对的规划成功率**——"走过的地方还能不能走"（短程，最敏感）
  4. 首点→各采样点的规划成功率 + 长度增量——长程绕行代价

采样点只从**最长的一条折线**上取：跨段的跳跃不是真实走过的路径，拼起来会造出
不存在的点对。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import KeyframeGridMapper, MapperConfig   # noqa: E402
from tools.precision_probe import load                             # noqa: E402

RECS = ("20261001_044153", "20260929_045615")
SAMPLES = 40          # 采样点数（沿最长折线按弧长均匀取）
MIN_SEP_M = 1.0       # 相邻点对太近就没意义，间隔不足就跳过


def _longest(polys) -> np.ndarray:
    best, blen = None, -1.0
    for p in polys:
        a = np.asarray(p, float)
        if len(a) < 2:
            continue
        L = float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
        if L > blen:
            best, blen = a, L
    return np.zeros((0, 2)) if best is None else best


def _resample(poly: np.ndarray, n: int) -> list[tuple[float, float]]:
    """按弧长均匀取 n 个点（含首尾）。"""
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return [tuple(poly[0])]
    ts = np.linspace(0.0, s[-1], n)
    out = [tuple(poly[0])]
    for t in ts[1:]:
        i = int(np.searchsorted(s, t, side="right")) - 1
        i = min(max(i, 0), len(d) - 1)
        u = 0.0 if d[i] <= 0 else (t - s[i]) / d[i]
        out.append(tuple(poly[i] + u * (poly[i + 1] - poly[i])))
    return out


def build(frames, exempt: bool):
    c = MapperConfig()
    c.ray_near_exempt = exempt
    m = KeyframeGridMapper(c)
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    ng.build(radius_m=0.25, walked=m.walked())
    return ng, m


def metrics(ng, pts):
    """连通性四件套。"""
    center = ng.center
    # 1) 连通分量数（去掉背景 0）
    n_comp, lab = cv2.connectedComponents(center.astype(np.uint8), connectivity=8)
    n_comp -= 1

    # 2) 起点可达域
    rc0 = ng.to_cell(pts[0])
    reach = 0
    if ng.inside(rc0) and center[rc0]:
        reach = int((lab == lab[rc0]).sum())
    else:
        snap = ng.snap_to_center(pts[0], 0.5)
        if snap is not None:
            reach = int((lab == lab[snap[0]]).sum())

    # 3) 相邻点对
    adj_ok = adj_tot = 0
    for a, b in zip(pts, pts[1:]):
        if float(np.hypot(*(np.asarray(b) - np.asarray(a)))) < MIN_SEP_M:
            continue
        adj_tot += 1
        if ng.plan(a, b).accepted:
            adj_ok += 1

    # 4) 首点 → 各点
    far_ok = far_tot = 0
    lengths: list[float] = []
    for p in pts[1:]:
        if float(np.hypot(*(np.asarray(p) - np.asarray(pts[0])))) < MIN_SEP_M:
            continue
        far_tot += 1
        r = ng.plan(pts[0], p)
        if r.accepted:
            far_ok += 1
            lengths.append(r.length_m)
    return {"n_comp": n_comp, "reach": reach, "adj": (adj_ok, adj_tot),
            "far": (far_ok, far_tot), "len": lengths}


def main() -> None:
    for name in (sys.argv[1:] or list(RECS)):
        frames = load(ROOT / "navmesh_recordings" / name)
        if not frames:
            print(f"\n=== {name} === 没有关键帧")
            continue
        print(f"\n=== {name} ===  {len(frames)} 关键帧")
        base = None
        for v in (False, True):
            ng, m = build(frames, v)
            pts = _resample(_longest(m.walked()), SAMPLES)
            mt = metrics(ng, pts)
            if base is None:
                base = mt
                print(f"  豁免={str(v):>5s}  分量={mt['n_comp']:4d}  可达格={mt['reach']:6d}  "
                      f"相邻通={mt['adj'][0]:3d}/{mt['adj'][1]:<3d}  "
                      f"长程通={mt['far'][0]:3d}/{mt['far'][1]:<3d}  "
                      f"长程长度中位数={np.median(mt['len']):.2f} m   （基线）")
            else:
                adj_lost = base["adj"][0] - mt["adj"][0]
                far_lost = base["far"][0] - mt["far"][0]
                common = min(len(base["len"]), len(mt["len"]))
                dl = (np.median(mt["len"][:common]) - np.median(base["len"][:common])) if common else float("nan")
                print(f"  豁免={str(v):>5s}  分量={mt['n_comp']:4d}  可达格={mt['reach']:6d}  "
                      f"相邻通={mt['adj'][0]:3d}/{mt['adj'][1]:<3d}  "
                      f"长程通={mt['far'][0]:3d}/{mt['far'][1]:<3d}  "
                      f"长程长度中位数={np.median(mt['len']):.2f} m")
                print(f"        Δ分量={mt['n_comp'] - base['n_comp']:+d}  "
                      f"Δ可达={mt['reach'] - base['reach']:+d}  "
                      f"相邻断裂={adj_lost}  长程断裂={far_lost}  "
                      f"Δ长度中位数={dl:+.2f} m")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
