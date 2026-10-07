"""点云分场体检：定位"某一场看起来不对"到底错在哪。

用法::

    python research/tools/pc_session_audit.py --world wrld_home-7cf435ea \\
        --rec 20261001_044153 20261005_235232 20261006_001523 \\
        --session 20261001_044153 20261005_235237 20261006_001523

三种位姿源两两对照——**同一批点、只换位姿**，差异就只可能来自位姿：

* ``tree``   : ``session_pose_table``（世界树并树后的位姿，pointcloud_view 用的就是这个）
* ``kf``     : 录制档 ``kf/*.npz`` 里的 ``T_map``（会话内位姿，offline_fusion 的 fallback）
* ``final``  : 录制档 ``final_poses.npz``（该场自己的最终位姿）

产出三组数：

1. 每场规模：点数、包围盒、位姿轨迹里程
2. **跨场重合**：A 抽样点对 B 的最近邻距离（中位 / p90 / <0.30 m 比例）——对齐好则贴到体素地板，
   某一场错位则它对别人的距离会显著抬升
3. **同场换算位姿的差异**：tree vs kf vs final 两两之间的平移差（中位 / 最大）——
   若某场 tree 与 kf 差得离谱，说明它并树那一步有问题，而不是点云本身有问题
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.nav_xsession import session_pose_table  # noqa: E402

CAM_H = 1.756
Z_MIN, Z_MAX = -0.6, 3.4


def voxelize(xyz: np.ndarray, vox: float) -> np.ndarray:
    """体素降采样（取格内质心）。"""
    if len(xyz) == 0:
        return xyz
    key = np.floor(xyz / vox).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    c = np.zeros((len(cnt), 3), np.float64)
    np.add.at(c, inv, xyz)
    return c / cnt[:, None]


def load_poses(world_dir: Path, rec: str, sid: str, src: str):
    """返回 {kf_id: T(4x4)} 或抛错。"""
    if src == "tree":
        tab = session_pose_table(world_dir, sid)
        if tab is None:
            raise SystemExit(f"[fail] {sid} 没有位姿表")
        return ({int(k): np.asarray(T, np.float64) for k, T in zip(tab["ids"], tab["T_map"])},
                str(tab.get("source")), str(tab.get("base_sid")))
    rec_dir = ROOT / "navmesh_recordings" / rec
    if src == "kf":
        out = {}
        for f in sorted(rec_dir.glob("kf/*.npz"), key=lambda p: int(p.stem)):
            with np.load(f) as z:
                if "T_map" in z.files:
                    out[int(f.stem)] = np.asarray(z["T_map"], np.float64)
        return out, "kf:T_map", rec
    fp = rec_dir / "final_poses.npz"
    if not fp.exists():
        return {}, "final:missing", rec
    z = np.load(fp)
    return ({int(k): np.asarray(T, np.float64) for k, T in zip(z["ids"], z["T"])}, "final_poses", rec)


def build_cloud(rec: str, pose_of: dict, every: int, vox: float):
    """一场 → 世界系点云（体素后）+ 轨迹里程。"""
    kfs = sorted((ROOT / "navmesh_recordings" / rec).glob("kf/*.npz"), key=lambda p: int(p.stem))
    chunks, traj = [], []
    n_raw = 0
    for f in kfs:
        k = int(f.stem)
        if k % every:
            continue
        T = pose_of.get(k)
        if T is None:
            continue
        with np.load(f) as z:
            if "pts" not in z.files:
                continue
            p = np.asarray(z["pts"], np.float32)
        n_raw += len(p)
        pw = p @ T[:3, :3].T
        xy = pw[:, :2] + T[:2, 3]
        zw = pw[:, 2] + CAM_H
        sel = (zw > Z_MIN) & (zw < Z_MAX) & np.isfinite(xy).all(axis=1)
        if sel.any():
            chunks.append(np.column_stack([xy[sel], zw[sel]]))
        traj.append(T[:2, 3])
    if not chunks:
        return np.zeros((0, 3)), 0.0, n_raw
    allp = np.concatenate(chunks)
    tr = np.asarray(traj)
    path = float(np.abs(np.diff(tr, axis=0)).sum()) if len(tr) > 1 else 0.0
    return voxelize(allp.astype(np.float64), vox), path, n_raw


def nn_stats(a: np.ndarray, b: np.ndarray, sample: int, rng: np.random.Generator):
    """a 抽样点到 b 的最近邻距离统计。"""
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return dict(med=float("nan"), p90=float("nan"), frac03=float("nan"))
    idx = rng.choice(len(a), size=min(sample, len(a)), replace=False)
    tree = cKDTree(b)
    d, _ = tree.query(a[idx], k=1, workers=-1)
    return dict(med=float(np.median(d)), p90=float(np.percentile(d, 90)),
                frac03=float((d < 0.30).mean()))


def pose_diff(pa: dict, pb: dict):
    """两套位姿的平移差（共有关键帧）。"""
    ks = sorted(set(pa) & set(pb))
    if not ks:
        return None
    a = np.asarray([pa[k][:2, 3] for k in ks])
    b = np.asarray([pb[k][:2, 3] for k in ks])
    d = np.hypot(*(a - b).T)
    return dict(n=len(ks), med=float(np.median(d)), p90=float(np.percentile(d, 90)),
                mx=float(d.max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--rec", nargs="+", required=True)
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--vox", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()
    if len(args.rec) != len(args.session):
        raise SystemExit("[fail] --rec 与 --session 必须一一对应")

    wdir = ROOT / "navmesh_memory" / args.world
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    print(f"=== 位姿源对照（每场 tree / kf / final 的平移差）===")
    poses = {}
    for rec, sid in zip(args.rec, args.session):
        poses[(rec, "tree")] = load_poses(wdir, rec, sid, "tree")
        poses[(rec, "kf")] = load_poses(wdir, rec, sid, "kf")
        poses[(rec, "final")] = load_poses(wdir, rec, sid, "final")
        (pt, src, base), (pk, _, _), (pf, _, _) = (poses[(rec, "tree")], poses[(rec, "kf")],
                                                   poses[(rec, "final")])
        dk = pose_diff(pt, pk)
        df = pose_diff(pt, pf)
        print(f"  {rec} → {sid}   tree.src={src} base={base}")
        print(f"     tree vs kf    : {dk}")
        print(f"     tree vs final : {df}")

    print(f"\n=== 点云规模（位姿源 = tree，every={args.every}, vox={args.vox}）===")
    clouds = {}
    for rec, sid in zip(args.rec, args.session):
        c, path, n_raw = build_cloud(rec, poses[(rec, "tree")][0], args.every, args.vox)
        clouds[rec] = c
        bb = c.max(0) - c.min(0) if len(c) else np.zeros(3)
        print(f"  {rec}: 原始 {n_raw:,} → 体素 {len(c):,}   "
              f"bbox {bb[0]:.1f}×{bb[1]:.1f}×{bb[2]:.1f} m   里程 {path:.1f} m")

    print(f"\n=== 跨场重合（A 抽样点 → B 的最近邻；对齐好应贴 ~{args.vox} m 体素地板）===")
    recs = list(args.rec)
    print(f"{'A':>18} → {'B':>18} {'med':>7} {'p90':>7} {'<0.30m':>8}")
    for a in recs:
        for b in recs:
            if a == b:
                continue
            s = nn_stats(clouds[a], clouds[b], args.sample, rng)
            print(f"{a:>18} → {b:>18} {s['med']:7.3f} {s['p90']:7.3f} {s['frac03']:8.1%}")

    print(f"\n=== 每场 vs 其余两场合并 ===")
    for a in recs:
        others = np.concatenate([clouds[b] for b in recs if b != a])
        s = nn_stats(clouds[a], others, args.sample, rng)
        print(f"  {a}: med {s['med']:.3f}  p90 {s['p90']:.3f}  <0.30m {s['frac03']:.1%}")

    print(f"\n[done] {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
