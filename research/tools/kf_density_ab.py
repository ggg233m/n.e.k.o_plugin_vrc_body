# -*- coding: utf-8 -*-
r"""关键帧密度 A/B：加密关键帧到底能不能降低点云面厚？**必须同时控制速度与帧数**。

背景（2026-10-27）：失配 ∝ 位移（0.3 m→8 cm、2 m→99 cm），于是很自然推断「把关键帧
压密能按比例降拖影」。这个推断**被本工具证伪**。

三个必须同时控制的混杂：
  1. **速度**：录制里「密采」的帧恰好也在低速段（实测密采 0.55 m/s vs 疏采 3.62 m/s，
     差 6.6 倍）。不控速度会得到 1.92 vs 2.95 cm 的假改善，其实几乎全是速度差。
  2. **帧数**：密采组帧数天然更多，更多层叠上去本身就更厚。必须等量抽样。
  3. **位置**：两组必须落在同一批空间格，否则不是同一块面。

结论（20261026_184318，三次等量抽样）：密采 1.38 cm vs 疏采 1.06 cm —— **密采反而厚**。
原因：加密增加的是**层数**；每层错位虽小，总厚度不线性下降。

⚠️ 归因边界：本工具比较的是「录制里天然存在的局部间隔差异」，不是独立造出的更密帧。
低速段的密采主要由**超时/快转**触发（相邻 dt 中位 1.05 s），与「距离触发」的密度不完全等价。

用法：
    .venv/Scripts/python.exe research/tools/kf_density_ab.py 20261026_184318
    .venv/Scripts/python.exe research/tools/kf_density_ab.py 20261026_184318 --dense-max 0.40 --trials 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REC_ROOT = ROOT / "navmesh_recordings"


def load_kfs(rec: Path):
    """返回 (ids, T_dr, t, dist_m, pts 列表, meta)。pts 已按 mapper.range_m 做水平半径闸。"""
    ev = [json.loads(x) for x in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kf = [e for e in ev if e.get("kind") == "kf"]
    ids = np.array([e["k"] for e in kf])
    tt = np.array([float(e["t"]) for e in kf])
    dm = np.array([float(e["dist_m"]) for e in kf])
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    rng_m = float(meta["config"]["mapper"]["range_m"])
    T, pts = [], []
    for k in ids:
        d = np.load(rec / "kf" / f"{int(k):06d}.npz")
        p = np.asarray(d["pts"], np.float32)
        m = np.isfinite(p).all(axis=1) & (np.hypot(p[:, 0], p[:, 1]) <= rng_m)
        pts.append(p[m])
        T.append(np.asarray(d["T_dr"], float))
    return ids, np.array(T), tt, dm, pts, meta


def world_cloud(sel_idx, T, pts, cam_h, gc_fix):
    """把选中的关键帧投到世界系（与 pointcloud_view 的 z 口径一致）。"""
    ch = []
    for i in sel_idx:
        p = pts[i]
        if len(p) == 0:
            continue
        Td = T[i]
        pw = p @ Td[:3, :3].T
        xy = pw[:, :2] + Td[:2, 3]
        zw = pw[:, 2] + cam_h - gc_fix
        s = (zw > -1.0) & (zw < 4.0)
        ch.append(np.column_stack([xy[s], zw[s]]))
    return np.concatenate(ch) if ch else np.zeros((0, 3))


def voxelize(P, v):
    q = np.floor(P / v).astype(np.int64)
    key = q[:, 0] * (1 << 42) + q[:, 1] * (1 << 21) + q[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    acc = np.zeros((len(uk), 3))
    np.add.at(acc, inv, P)
    return acc / np.bincount(inv).astype(float)[:, None]


def thickness(V, anchors, radius, minpts):
    tree = cKDTree(V)
    out = []
    for c in anchors:
        idx = tree.query_ball_point(c, radius, workers=-1)
        if len(idx) < minpts:
            continue
        Q = V[idx] - V[idx].mean(0)
        w, _ = np.linalg.eigh(Q.T @ Q / len(Q))
        if w[0] > 0.25 * w[1]:
            continue
        out.append(np.sqrt(max(w[0], 0.0)))
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec", help="录制目录名")
    ap.add_argument("--dense-max", type=float, default=0.40,
                    help="局部间隔 <= 该值算密采（米），默认 0.40")
    ap.add_argument("--speed-pct", type=float, default=33.0,
                    help="只取速度最低的百分位，控制速度混杂（默认 33）")
    ap.add_argument("--trials", type=int, default=3, help="等量抽样次数")
    ap.add_argument("--grid", type=float, default=0.10)
    ap.add_argument("--radius", type=float, default=0.30)
    ap.add_argument("--minpts", type=int, default=14)
    ap.add_argument("--anchors", type=int, default=900)
    ap.add_argument("--cam-h", type=float, default=None, help="默认取 meta 的 cam_h_default")
    a = ap.parse_args()

    rec = REC_ROOT / a.rec
    if not rec.is_dir():
        raise SystemExit(f"[fail] 没有这个录制：{rec}")
    ids, T, tt, dm, pts, meta = load_kfs(rec)
    s = float(meta["config"]["world_scale"])
    cam_h = a.cam_h if a.cam_h is not None else float(meta["config"]["mapper"]["cam_h_default"])

    pos = T[:, :3, 3]
    gap = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * s
    dt = np.diff(tt)
    spd = gap / np.maximum(dt, 1e-9)
    n = len(ids)
    loc = np.empty(n)
    loc[0], loc[-1] = gap[0], gap[-1]
    loc[1:-1] = np.minimum(gap[:-1], gap[1:])
    vs = np.empty(n)
    vs[0], vs[-1] = spd[0], spd[-1]
    vs[1:-1] = (spd[:-1] + spd[1:]) / 2

    print(f"{a.rec}: {n} 关键帧, world_scale {s:.4f}, cam_h {cam_h:.3f}")
    print(f"  局部间隔 p10/p50/p90 = {np.percentile(loc,10):.2f}/{np.median(loc):.2f}/{np.percentile(loc,90):.2f} m")
    print(f"  速度     p10/p50/p90 = {np.percentile(vs,10):.2f}/{np.median(vs):.2f}/{np.percentile(vs,90):.2f} m/s")

    dense = loc <= a.dense_max
    slow = vs <= np.percentile(vs, a.speed_pct)
    d_idx = np.flatnonzero(dense & slow)
    s_idx = np.flatnonzero(~dense & slow)
    print()
    print(f"[控速] 低速且密采 {len(d_idx)} 帧 / 低速且疏采 {len(s_idx)} 帧")
    if not len(d_idx) or not len(s_idx):
        print("[fail] 某一组为空；放宽 --speed-pct 或 --dense-max")
        return 1
    vm_d, vm_s = float(np.median(vs[d_idx])), float(np.median(vs[s_idx]))
    print("  密采组速度中位 %.2f m/s（疏采组 %.2f）" % (vm_d, vm_s))
    if max(vm_d, vm_s) > 1.5 * max(min(vm_d, vm_s), 1e-6):
        print("  ⚠️ 两组速度仍差 %.1f 倍：低速段里「密采」几乎总是更慢，"
              "这个混杂**消不掉**。" % (max(vm_d, vm_s) / max(min(vm_d, vm_s), 1e-6)))
        print("     把它当**上界证据**读：即便密采更慢（本该更薄），面厚仍然没有变薄。")

    gc_fix = 0.0
    B = voxelize(world_cloud(s_idx, T, pts, cam_h, gc_fix), a.grid)
    rng = np.random.default_rng(0)
    m = min(len(d_idx), len(s_idx))
    print()
    print("%3s %5s %10s %10s %8s" % ("次", "抽帧", "密采面厚", "疏采面厚", "共同格"))
    res = []
    for trial in range(a.trials):
        sub = rng.choice(d_idx, size=m, replace=False)
        A = voxelize(world_cloud(sub, T, pts, cam_h, gc_fix), a.grid)
        d, _ = cKDTree(B).query(A, k=1, workers=-1)
        ov = d < 0.30
        if int(ov.sum()) < 200:
            print("%3d   [skip] 共同格太少 %d" % (trial + 1, int(ov.sum())))
            continue
        anchors = A[ov][rng.choice(int(ov.sum()), min(a.anchors, int(ov.sum())), replace=False)]
        ta = thickness(A, anchors, a.radius, a.minpts)
        tb = thickness(B, anchors, a.radius, a.minpts)
        if not len(ta) or not len(tb):
            print("%3d   [skip] 无有效平面邻域" % (trial + 1))
            continue
        res.append((np.median(ta) * 100, np.median(tb) * 100))
        print("%3d %5d %9.2fcm %9.2fcm %8d" % (
            trial + 1, m, res[-1][0], res[-1][1], int(ov.sum())))
    print()
    if not res:
        print("[fail] 没有一次成功；检查 --grid/--radius")
        return 1
    r = np.asarray(res)
    print("平均：密采 %.2f cm  vs  疏采 %.2f cm  ⇒ 差 %+.2f cm" % (
        r[:, 0].mean(), r[:, 1].mean(), r[:, 0].mean() - r[:, 1].mean()))
    print("判读：差值为正 = 密采更厚 ⇒ 加密关键帧不能降面厚（已实测证伪）。")
    print("⚠️ 不要拿未控混杂的数字（不控速度/帧数）当结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
