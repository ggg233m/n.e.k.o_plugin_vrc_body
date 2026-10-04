# -*- coding: utf-8 -*-
r"""回环自标定：用**回环本身**当内部锚点，量 OSC 链与几何链的标度比。

    python research/tools/loop_selfcal.py 20261001_044153

为什么不需要外部标定物
--------------------
``Docs/尺度标定门依赖-9%分歧（2026-09-24）.md`` 说："谁是准的（OSC 路程 vs 深度几何）
需要一个**独立几何锚点**才能判……本实验只能量出『它们差多少』，量不出『谁对』。"

**回环就是那个锚点，而且它就在盘上。** 每次回环同时给出两个独立的量：

* **几何链** ``offset_m`` —— PnP 在双目特征上解出的同一地点相对位移（世界米）
* **OSC 链** ``|T_dr[b] - T_dr[a]| × world_scale`` —— 速度积分的同段位移（世界米）

两者量的是同一段距离，来源完全独立 ⇒ 它们的比值就是标度比：

    s = |OSC 位移| / |几何位移|

* ``s = 1`` ⇒ 两链一致
* ``s ≠ 1`` ⇒ 差的那部分就是整个地图的**系统性**尺度误差，随距离线性累积

⚠️ 用 ``T_dr``（纯航位推算）而不是 ``T_map``（回环修正后）：后者已经被这次回环本身
改过了，拿它算等于用结论验证结论。

数据来源：``events.jsonl`` 的 ``kind:"kf"`` 事件里 ``loops`` 字段（回环对 + 几何位移）
与同事件的 ``dist_m``；航位推算位姿在 ``kf/<id>.npz`` 的 ``T_dr``。**全程离线，不碰在线代码。**

筛选口径
--------
只取通过 ``rot_err_deg`` 门（默认 6°）的回环——PnP 的转角必须与 HMD 绝对朝向一致，
这是最强的外点过滤，混进来的回环会把中位数拖偏。再按几何位移长度分档，
因为**短基线的比值噪声大**（0.13 m 的位移上 5 cm 的误差就是 38%）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(rec: Path) -> tuple[dict[int, np.ndarray], list[dict]]:
    tr: dict[int, np.ndarray] = {}
    for f in (rec / "kf").glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        tr[int(f.stem)] = np.asarray(z["T_dr"], np.float64)[:2, 3].copy()
    loops: list[dict] = []
    for line in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("kind") == "kf" and e.get("loops"):
            loops += [dict(L, _b=e["k"]) for L in e["loops"]]
    return tr, loops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="*", default=["20261001_044153"])
    ap.add_argument("--yaw-tol", type=float, default=6.0)
    ap.add_argument("--min-offset-m", type=float, default=0.5)
    args = ap.parse_args()

    for name in args.recs:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.exists():
            print(f"[skip] {name}", file=sys.stderr)
            continue
        meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
        ws = float(meta["config"]["mapper"]["world_scale"])
        tr, loops = load(rec)
        print(f"\n=== {name}  world_scale={ws}  原始回环 {len(loops)} 条 ===")
        if not loops:
            print("无回环，无法自标定")
            continue

        rows = []
        for L in loops:
            a, b = int(L["a"]), int(L["b"])
            if a not in tr or b not in tr:
                continue
            if float(L.get("rot_err_deg", 99)) > args.yaw_tol:
                continue
            geo = float(L["offset_m"])                     # 几何链，世界米
            osc = float(np.hypot(*(tr[b] - tr[a])) * ws)    # OSC 链，世界米
            if geo <= 1e-6 or osc <= 1e-6:
                continue
            rows.append({"a": a, "b": b, "geo_m": geo, "osc_m": osc,
                         "s": osc / geo, "inliers": int(L.get("inliers", 0)),
                         "cov": float(L.get("coverage", 0.0)),
                         "reproj": float(L.get("reproj_px", 0.0))})
        if not rows:
            print("转角门之后没有可用回环")
            continue

        s = np.array([r["s"] for r in rows])
        geo = np.array([r["geo_m"] for r in rows])
        print(f"通过转角门的回环 {len(rows)}/{len(loops)}")
        print(f"几何位移 中位 {np.median(geo):.2f} m   （p10 {np.percentile(geo,10):.2f} / "
              f"p90 {np.percentile(geo,90):.2f}）")
        print(f"★ 标度比 s = |OSC| / |几何|")
        print(f"    中位 {np.median(s):.4f}   IQR [{np.percentile(s,25):.4f}, "
              f"{np.percentile(s,75):.4f}]   MAD {np.median(np.abs(s-np.median(s))):.4f}")
        print(f"    → OSC 链比几何链{'大' if np.median(s)>1 else '小'} "
              f"{abs(np.median(s)-1)*100:.1f}%")

        big = geo >= args.min_offset_m
        if big.sum() >= 3:
            sb = s[big]
            print(f"    仅取几何位移 ≥{args.min_offset_m} m（{int(big.sum())} 条，基线长、"
                  f"比值更可信）：中位 {np.median(sb):.4f}  IQR "
                  f"[{np.percentile(sb,25):.4f}, {np.percentile(sb,75):.4f}]")
        else:
            print(f"    ⚠️ 几何位移 ≥{args.min_offset_m} m 的只有 {int(big.sum())} 条，"
                  f"长基线样本不足，比值仍被短基线噪声主导")

        # 按几何位移分档：比值应当随基线变长而收敛，若不收敛说明不是尺度问题
        edges = [0.0, 0.3, 0.8, 1.5, 3.0, 1e9]
        print("\n    按几何位移分档（比值若随基线收敛 ⇒ 是尺度；若发散 ⇒ 是别的误差）")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (geo >= lo) & (geo < hi)
            if m.sum() >= 3:
                print(f"      {lo:4.1f}–{hi if hi<1e8 else float('inf'):6.1f} m : "
                      f"{int(m.sum()):3d} 条  s 中位 {np.median(s[m]):.4f}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
