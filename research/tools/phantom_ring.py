# -*- coding: utf-8 -*-
r"""假障碍的**空间指纹**：它该长成"观测扇形外沿的一圈"。

机理由 `Docs/虚假障碍-根因与分层修复（2026-10-05）.md` §一 给出：

    dz = z^2/(fx*b) * dd        fx=202.5, b=0.126  =>  z^2/25.5

z=1.5m 时 1 像素视差误差 = 0.088 m，z=3m 时 0.353 m，z=5m 时 0.980 m。
而障碍高度带是 (ground_tol 0.30, obst_top 2.0)，**只有 1.7 m 宽**。

⇒ 两个**可falsify的预登记预测**：
    P1. 障碍占比应随**观测距离**急剧上升（真结构不该有这个趋势）。
    P2. 反之，近带（<1.5 m）的障碍占比应与真结构比例相当，且**不随距离变**
        如果那里还有真障碍的话。

本工具不看 q_tiers（它正是为了治这个病），而是在**关闭 q_tiers** 的栅格上量
"每个已观测格被判 OCC 的概率 vs 它的最近观测距离"。
这是规则**之外**的一维（规则用票数，这里用几何距离），所以不循环。
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import FREE, OCC, UNK  # noqa: E402
import mapping_gate as MG                       # noqa: E402

BANDS = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 5.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", nargs="*", default=None)
    a = ap.parse_args()
    names = a.rec or ["20261001_044153", "20260929_045615", "20261010_033237"]
    for name in names:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.is_dir():
            print("%s not found" % name); continue
        cfg_rec, info = MG.recorded_config(rec)
        # **关掉 q_tiers**：量的是病本身，不是药效
        ng, trail, st, m = MG.replay(rec, dataclasses.replace(cfg_rec, q_tiers=False))
        g = ng.grid
        m2 = ng.meta
        H, W = g.shape

        # 每个已观测格(MIN 1 keyframe)到最近关键帧相机的水平距离
        keys = sorted(m._pose)
        Ts = np.array([m._pose[k] for k in keys])
        rr, cc = np.mgrid[0:H, 0:W]
        cx = m2.origin_xy_m[0] + (cc + 0.5) * m2.resolution_m
        cy = m2.origin_xy_m[1] + (H - 1 - rr + 0.5) * m2.resolution_m
        cent = np.stack([cx.ravel(), cy.ravel()], -1)
        P = Ts[:, :2, 3]
        # 网格分块算最近距离，避免 N*K 爆内存
        dmin = np.full(len(cent), np.inf)
        for i in range(0, len(cent), 200000):
            chunk = cent[i:i + 200000]
            d = np.sqrt(((chunk[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            dmin[i:i + 200000] = d.min(axis=1)
        dmin = dmin.reshape(H, W)

        seen = (g != UNK)
        print("=" * 78)
        print("rec %s   range_m=%s  q_tiers=OFF" % (name, info.get("range_m")))
        print("=" * 78)
        print("  %-10s %9s %9s %9s %8s" % ("dist band", "seen", "OCC", "OCC%", "share"))
        tot_occ = int((g == OCC).sum())
        rows = []
        for lo, hi in BANDS:
            sel = seen & (dmin >= lo) & (dmin < hi)
            n = int(sel.sum())
            if n == 0:
                continue
            o = int((sel & (g == OCC)).sum())
            rows.append((lo, hi, n, o, 100.0 * o / n, 100.0 * o / max(tot_occ, 1)))
        for lo, hi, n, o, pct, share in rows:
            print("  %-10s %9d %9d %8.2f%% %7.1f%%" % ("%.1f-%.1fm" % (lo, hi), n, o, pct, share))
        # 预测 P1：占比随距离单调上升？
        ps = [r[4] for r in rows]
        mono = all(ps[i] <= ps[i + 1] + 1e-9 for i in range(len(ps) - 1))
        print()
        print("  P1 障碍占比随观测距离单调上升? %s" % ("YES" if mono else "NO"))
        if len(ps) >= 2:
            print("     近带(%.1fm) %.2f%%  ->  远带(%.1fm) %.2f%%   (%.1fx)"
                  % (rows[0][0], ps[0], rows[-1][1], ps[-1], ps[-1] / max(ps[0], 1e-9)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
