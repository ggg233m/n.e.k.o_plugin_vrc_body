"""点云"塞满"体检：内部自由空间里到底有没有点、有多少。

用法::

    python research/tools/pc_fill_probe.py --world wrld_home-7cf435ea \\
        --rec 20261001_044153 20261005_235232 20261006_001523 \\
        --session 20261001_044153 20261005_235237 20261006_001523

点云只该长在**表面**上：地板一层、墙面一层、桌沿一层。若一个 xy 格里有十几层高度，
那些多出来的层就是落在自由空间里的点（视差噪声 / 弱纹理误匹配 / 沿射线的飞点），
画出来就是"整个内部被点云塞满，没有空的区域"。

产出四组数：

1. **每格高度层数**：xy 格（默认 0.25 m）× z 分箱（默认 0.20 m），统计有点的箱数
2. **每格点数**分布
3. **高度直方图**：全场 z 分布（地面主峰该在 0）
4. **点到最近轨迹点的距离**：``mapper.range_m=5.0``，真实点不该远离轨迹 6 m 以上

``--only-air`` 只统计空中点（z > 阈值），用于把"地面那一层"排除掉单看弥漫程度。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.nav_xsession import session_pose_table  # noqa: E402
from pointcloud_view import prior_cam_h, estimate_cam_h  # noqa: E402

sys.path.insert(0, str(ROOT / "research" / "tools"))

CAM_H_FALLBACK = 1.6
Z_MIN, Z_MAX = -0.6, 3.4


def load(rec: str, sid: str, wdir: Path, cam_h: float, every: int, vox: float):
    tab = session_pose_table(wdir, sid)
    pose = {int(k): np.asarray(T, np.float64) for k, T in zip(tab["ids"], tab["T_map"])}
    kfs = sorted((ROOT / "navmesh_recordings" / rec).glob("kf/*.npz"), key=lambda p: int(p.stem))
    chunks, traj = [], []
    for f in kfs:
        k = int(f.stem)
        if k % every:
            continue
        T = pose.get(k)
        if T is None:
            continue
        with np.load(f) as z:
            if "pts" not in z.files:
                continue
            p = np.asarray(z["pts"], np.float32)
        pw = p @ T[:3, :3].T
        xy = pw[:, :2] + T[:2, 3]
        zw = pw[:, 2] + cam_h
        sel = (zw > Z_MIN) & (zw < Z_MAX) & np.isfinite(xy).all(axis=1)
        if sel.any():
            chunks.append(np.column_stack([xy[sel], zw[sel]]))
        traj.append(T[:2, 3])
    allp = np.concatenate(chunks)
    if vox > 0:
        key = np.floor(allp / vox).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        c = np.zeros((len(cnt), 3), np.float64)
        np.add.at(c, inv, allp)
        allp = c / cnt[:, None]
    return allp, np.asarray(traj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--rec", nargs="+", required=True)
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--vox", type=float, default=0.08)
    ap.add_argument("--cell", type=float, default=0.25, help="xy 格边长（米）")
    ap.add_argument("--zbin", type=float, default=0.20, help="高度分箱（米）")
    ap.add_argument("--air-min", type=float, default=0.45, help="高于此值算空中点（米）")
    args = ap.parse_args()

    wdir = ROOT / "navmesh_memory" / args.world
    pm = prior_cam_h(wdir)
    t0 = time.perf_counter()
    Ps, trajs = [], []
    for rec, sid in zip(args.rec, args.session):
        h = pm.get(sid) or estimate_cam_h(ROOT / "navmesh_recordings" / rec)
        P, tr = load(rec, sid, wdir, h, max(1, args.every), float(args.vox))
        print(f"[load ] {rec} → {sid}  cam_h={h:.3f}  点 {len(P):,}  轨迹 {len(tr)}", flush=True)
        Ps.append(P)
        trajs.append(tr)
    P = np.concatenate(Ps)
    TR = np.concatenate(trajs)
    print(f"[all  ] 合计 {len(P):,} 点  体素 {args.vox} m  ({time.perf_counter()-t0:.1f}s)\n")

    # 1) 每格高度层数
    ix = np.floor(P[:, 0] / args.cell).astype(np.int64)
    iy = np.floor(P[:, 1] / args.cell).astype(np.int64)
    iz = np.floor(P[:, 2] / args.zbin).astype(np.int64)
    ckey = ix * 8192 + iy                      # 偏移打包防负
    cu, cinv = np.unique(ckey, return_inverse=True)
    ncell = len(cu)
    lay = np.zeros(ncell, np.int64)
    np.add.at(lay, cinv, 1)                    # 每格总点数
    # 层数 = 该格内不同的 (高度箱) 个数：先算逐点的 格索引×箱号，再去重回数
    nb = int((P[:, 2].max() - P[:, 2].min()) / args.zbin) + 8
    bkey = cinv.astype(np.int64) * nb + (iz - iz.min())
    cell_of_bin = np.unique(bkey) // nb
    layers = np.zeros(ncell, np.int64)
    np.add.at(layers, cell_of_bin, 1)

    print(f"=== 每格高度层数（xy {args.cell} m × z {args.zbin} m；真实表面 1–3 层，弥漫则十几层）===")
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 10), (11, 999)):
        m = (layers >= lo) & (layers <= hi)
        print(f"  {lo if hi < 999 else '≥11':>3}–{hi if hi < 999 else '+':<3} 层: "
              f"{m.sum():7,d} 格 ({m.mean():6.1%})")
    print(f"  层数 中位 {np.median(layers):.0f}  均值 {layers.mean():.1f}  "
          f"p90 {np.percentile(layers, 90):.0f}  max {layers.max()}")

    print(f"\n=== 每格点数（xy {args.cell} m）===")
    print(f"  中位 {np.median(lay):.1f}  均值 {lay.mean():.1f}  "
          f"p90 {np.percentile(lay, 90):.0f}  max {lay.max()}  总格数 {ncell:,}")

    # 2) 高度直方图
    print(f"\n=== 高度直方图（{args.zbin} m 一档）===")
    lo, hi = float(np.floor(P[:, 2].min() * 5) / 5), float(np.ceil(P[:, 2].max() * 5) / 5)
    edges = np.arange(lo, hi + 1e-9, args.zbin)
    h, _ = np.histogram(P[:, 2], bins=edges)
    for a, c in zip(edges[:-1], h):
        print(f"  {a:+.2f}~{a+args.zbin:+.2f} {100*c/len(P):5.1f}% {'#'*int(50*c/max(1, h.max()))}")

    # 3) 点到最近轨迹点的距离
    from scipy.spatial import cKDTree
    d, _ = cKDTree(TR).query(P[:, :2], k=1, workers=-1)
    print(f"\n=== 点到最近轨迹点距离（mapper.range_m=5.0，不该大量 >6 m）===")
    print(f"  中位 {np.median(d):.2f}  p90 {np.percentile(d,90):.2f}  p99 {np.percentile(d,99):.2f}  "
          f"max {d.max():.2f}   >6 m 占 {(d>6).mean():.2%}")

    # 4) 空中点的弥漫程度（排除地面那一层）
    air = P[:, 2] > args.air_min
    print(f"\n=== 只看空中点（z > {args.air_min} m，占 {air.mean():.1%}）===")
    if air.any():
        A = P[air]
        ax = np.floor(A[:, 0] / args.cell).astype(np.int64)
        ay = np.floor(A[:, 1] / args.cell).astype(np.int64)
        akey = ax * 8192 + ay
        au, ainv = np.unique(akey, return_inverse=True)
        cnt = np.zeros(len(au), np.int64)
        np.add.at(cnt, ainv, 1)
        print(f"  空中点 {len(A):,}  占 {len(au):,} 格  "
              f"每格中位 {np.median(cnt):.0f}  p90 {np.percentile(cnt,90):.0f}  max {cnt.max()}")
        occupied = len(au) * args.cell ** 2
        bbox = (P[:, 0].max()-P[:, 0].min()) * (P[:, 1].max()-P[:, 1].min())
        print(f"  空中点铺开面积 {occupied:.0f} m²  / 包围盒面积 {bbox:.0f} m²  "
              f"= {occupied/bbox:.1%}")
    print(f"\n[done] {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
