# -*- coding: utf-8 -*-
r"""q_tiers 清障的复制性裁定（split-half）—— 判"清掉的格是不是只在一次观测里出现过"。

原理：把同一场录制的关键帧**隔一取一**分成两半（交错 ⇒ 两半覆盖同一条轨迹），
各自独立建图（都用扁平基础规则 q_tiers=False）。两半的噪声与位姿是**独立**的。
只在两半都**观测到**的格上问：基础障碍是否**复制**。

若 q_tiers 清掉的主要是"只在单一半出现"的格 ⇒ 它清的是不可复制的噪声。
若它把"两半都复现"的格也大量清掉 ⇒ 有误删真结构的风险。

⚠️ 只报数、不下结论；结论写文档并附本判据。
⚠️ 用 monkeypatch 复用 mapping_gate.replay 本体，**不另写一份回放**（本仓栽过规则重写的坑）。
"""
from __future__ import annotations
import dataclasses, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import OCC, FREE, UNK  # noqa: E402
import mapping_gate as MG  # noqa: E402
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]


def subset_replay(rec, cfg, keep):
    """复用 MG.replay，只在事件流上过滤关键帧（保持回放语义逐行不变）。"""
    orig = MG.load_events
    def filt(r):
        return [e for e in orig(r)
                if not (e.get("kind") == "kf" and int(e["k"]) not in keep)]
    MG.load_events = filt
    try:
        return MG.replay(rec, cfg)
    finally:
        MG.load_events = orig


def observed(m):
    """该 mapper 的"已观测"点计数（near+far），已裁成 NavGrid 行序。"""
    cov = m.coverage_counts()
    if cov is None:
        return None
    return m._cut_acc(cov["near"] + cov["far"])


def to_common(arr, meta, X0, Y0, H2, W2):
    """把一个与 NavGrid 同布局的数组放进公共格架（各次回放范围不同，须按追踪米对齐）。

    原点对齐到 res 整数倍（见 rasterize docstring）⇒ 偏移是精确整数格。
    NavGrid 行 0 = 最大 y；origin_xy_m = 最小 x / 最小 y 角。
    """
    r = meta.resolution_m; H, W = arr.shape
    dcol = int(round((meta.origin_xy_m[0] - X0) / r))
    drowb = int(round((meta.origin_xy_m[1] - Y0) / r))       # 自下而上的行偏移
    out = np.zeros((H2, W2), arr.dtype)
    r0 = H2 - 1 - drowb - (H - 1)
    out[r0:r0 + H, dcol:dcol + W] = arr
    return out


def common_frame(metas):
    xs = [m.origin_xy_m[0] for m, _ in metas]
    ys = [m.origin_xy_m[1] for m, _ in metas]
    r = metas[0][0].resolution_m
    X0, Y0 = min(xs), min(ys)
    X1 = max(m.origin_xy_m[0] + s[1] * r for m, s in metas)
    Y1 = max(m.origin_xy_m[1] + s[0] * r for m, s in metas)
    W2 = int(round((X1 - X0) / r)); H2 = int(round((Y1 - Y0) / r))
    return X0, Y0, H2, W2, r


def centers_common(X0, Y0, H2, W2, r):
    rr, cc = np.mgrid[0:H2, 0:W2]
    x = X0 + (cc + 0.5) * r
    y = Y0 + (H2 - 1 - rr + 0.5) * r
    return np.stack([x, y], -1).reshape(-1, 2)


def dense_trail(trail, step_m=0.1):
    p = np.asarray(trail, np.float64).reshape(-1, 2)
    if len(p) < 2:
        return p
    segs = []
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        n = max(2, int(np.hypot(*(b - a)) / step_m) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        segs.append(a + t * (b - a))
    segs.append(p[-1:])
    return np.concatenate(segs, 0)


def centers_tracking(ng):
    m = ng.meta; H, W = ng.grid.shape
    r, c = np.mgrid[0:H, 0:W]
    x = m.origin_xy_m[0] + (c + 0.5) * m.resolution_m
    y = m.origin_xy_m[1] + (H - 1 - r + 0.5) * m.resolution_m
    return np.stack([x, y], -1).reshape(-1, 2)


def run(name):
    rec = ROOT / "navmesh_recordings" / name
    cfg_rec, info = MG.recorded_config(rec)
    base = dataclasses.replace(cfg_rec, q_tiers=False)
    on_cfg = dataclasses.replace(cfg_rec, q_tiers=True)

    evs = MG.load_events(rec)
    ids = [int(e["k"]) for e in evs if e.get("kind") == "kf"]
    even, odd = set(ids[0::2]), set(ids[1::2])

    ngA, trA, _, mA = subset_replay(rec, base, even)
    ngB, _, _, mB = subset_replay(rec, base, odd)
    ngF, trail, _, mF = MG.replay(rec, base)
    ngOn = MG.replay(rec, on_cfg)[0]

    metas = [(ngA.meta, ngA.grid.shape), (ngB.meta, ngB.grid.shape),
             (ngF.meta, ngF.grid.shape), (ngOn.meta, ngOn.grid.shape)]
    X0, Y0, H2, W2, r = common_frame(metas)

    occA = to_common(ngA.grid, ngA.meta, X0, Y0, H2, W2) == OCC
    occB = to_common(ngB.grid, ngB.meta, X0, Y0, H2, W2) == OCC
    occF = to_common(ngF.grid, ngF.meta, X0, Y0, H2, W2) == OCC
    occOn = to_common(ngOn.grid, ngOn.meta, X0, Y0, H2, W2) == OCC
    # 覆盖数组已被 _cut_acc 裁到该 mapper 自己最后那张栅格的布局 ⇒ 用同一份 meta
    obsA = to_common(observed(mA), ngA.meta, X0, Y0, H2, W2) > 0.5
    obsB = to_common(observed(mB), ngB.meta, X0, Y0, H2, W2) > 0.5

    cleared = occF & ~occOn
    universe = occF & obsA & obsB
    rep = universe & occA & occB
    single = universe & (occA ^ occB)

    dist = np.full((H2, W2), np.nan)
    if cKDTree is not None and len(trail) >= 2:
        d, _ = cKDTree(dense_trail(trail)).query(centers_common(X0, Y0, H2, W2, r))
        dist = d.reshape(H2, W2)

    def rate(mask):
        n = int(mask.sum())
        c = int((mask & cleared).sum())
        return n, c, (100.0 * c / n if n else float("nan"))

    print("=" * 92)
    print(f"{name}  kf={len(ids)} (even {len(even)} / odd {len(odd)})  range_m={info.get('range_m')}")
    print(f"  基础障碍 occ(F)={int(occF.sum())}  两半都观测到的={int(universe.sum())}")
    print(f"  {'集合':<22}{'格数':>8}{'被清掉':>8}{'清掉率%':>10}")
    for lbl, mk in (("两半都复现(rep)", rep), ("只在一半(single)", single), ("universe 全体", universe)):
        n, c, r = rate(mk)
        print(f"  {lbl:<22}{n:>8}{c:>8}{r:>10.1f}")
    print("\n  [按到轨迹距离分层] 层内比 rep vs single 的清掉率")
    print(f"  {'距离带':<12}{'rep n':>8}{'rep清%':>9}{'single n':>10}{'single清%':>11}")
    for lo, hi in BANDS:
        band = (dist >= lo) & (dist < hi)
        n1, _, r1 = rate(rep & band)
        n2, _, r2 = rate(single & band)
        if n1 == 0 and n2 == 0:
            continue
        lbl = f"{lo:g}-{hi:g}m" if hi < 1e8 else f">{lo:g}m"
        print(f"  {lbl:<12}{n1:>8}{r1:>9.1f}{n2:>10}{r2:>11.1f}")
    print("  （清掉率 = 该集合中被 q_tiers 清掉的比例；距离=追踪米）")


def main(recs):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for n in recs:
        try:
            run(n)
        except Exception as e:
            import traceback; traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["20261001_044153"]))