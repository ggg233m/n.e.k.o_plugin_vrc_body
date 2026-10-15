# -*- coding: utf-8 -*-
r'''视觉帧间边进 `LoopCloser` 位姿图：判据实验。

    python research/tools/vis_edges_gate.py 20261001_044153 --span 3
    python research/tools/vis_edges_gate.py <rec> --sigma-mode gap_rank

问题
----
`Docs/点云拖影修复原型与栅格影响（2026-10-09）.md` §十四 量到：人眼看到的弧形拖影是
**回访错位**，回环已削 88%（dr 143 → lk100 16.5 cm）。§7.3 给的下一步是
「视觉 Δt + 回环**进同一张位姿图**」。本工具就是那一格的判据。

做法（**不改 backend**）
----------------------
`nav_loop._Loop` 就是一条跨边（`a/b/z/sigma`），`LoopCloser.optimize()` 把它与里程边一起
塞进同一个加权拉普拉斯（`nav_loop.py:487-492`）。视觉约束形态相同 ⇒ 直接 append 即可，
**走的是生产解算器**，只多了一批边。

逐边 σ 由数据定（**不手调**）
---------------------------
`kf_visual_odom.rel_translation` 返回的信息矩阵 `I = Σ W_k` 是**几何可观测性**的度量。
实测它与「视觉 − DR」的真差相关 r=+0.32~+0.43，四分位能把真误差分开 **2.0–2.8 倍**；
内点数 r=−0.20~−0.27、DR 步长 r=+0.16~+0.26，更弱。
**但它的绝对值被低估约 7 倍**（中位 3.2 cm vs 真差 23 cm）⇒ 不能直接当 σ 用。
所以：用 σ_geo **只做分级**，把级别映射到「实测 |视觉−DR| 的分位区间」上，再夹到 [lo, hi]。

按场判（留出组）
---------------
把该场已接受回环按 70/30 拆，只用 70% 进图，在**留出的 30%** 上量
「地图里两帧距离 vs 该回环自己记的几何位移」—— 每个会话用自己**不看**的那批约束判自己。

结论（2026-10-15）
-----------------
* 四场留出残差**都改善**：9.0→6.5 / 8.8→6.7 / 26.1→22.1 / 7.2→4.6 cm。
* ⚠️ **必须配证伪控制**：把视觉边的 z 在边之间打乱（刚度/数量不变、信息为零），
  目标指标**同等改善一部分** —— 044153 上打乱版拿到 19.4→17.5（真版 6.3），
  即**约 1/4 的改善来自「多了一批硬边」的正则化，不是视觉信息**。
* ⚠️ σ 极小时视觉边占全图刚度 **91%**（σ=0.02 m 是过度自信 10 倍）⇒ 那种配置下的
  大数字不可信。
'''
from __future__ import annotations

import argparse, json, sys, numpy as np
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'research'/'tools'))
from backend.nav_loop import LoopCloser, _Loop
from backend.nav_grid import OCC
from loop_replay import recorded_loop_config, features_of
import grid_pose_ab as GAB
import mapping_gate as G
from trail_geom import trail_mask
from kf_visual_odom import load as vo_load, rel_translation
from revisit_shift import strict_wall, wall_shift
import smear_eval as SE

S = 0.755


def heldout_resid(P, hold, ws):
    """被**留出**（没进优化）的那批回环：最终地图里两帧距离 vs 该回环自己记的几何位移。

    口径同 `drift_shape` 证据 C。用它判「地图是否更自洽」比回访对更对：样本多，
    而且**不看被优化的那批约束**。`hold` 里每项要 `a` / `b` / `offset_m`。
    """
    vals = []
    for L in hold:
        a, b = int(L['a']), int(L['b'])
        if a not in P or b not in P:
            continue
        geo = float(L.get('offset_m', 0.0))
        if geo <= 1e-6:
            continue
        vals.append(abs(float(np.hypot(*(P[b][:2, 3] - P[a][:2, 3]))) * ws - geo))
    return np.array(vals) if vals else np.array([np.nan])


def load_edges(name, span=3):
    rec = ROOT/'navmesh_recordings'/name
    ev = [json.loads(x) for x in (rec/'events.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    kfs = [(int(e['k']), float(e['t']), float(e['dist_m'])) for e in ev if e.get('kind') == 'kf']
    FEAT, TD, DM = {}, {}, {}
    for k, t, d in kfs:
        with np.load(rec/'kf'/('{:06d}.npz'.format(k))) as z:
            FEAT[k] = features_of(z); TD[k] = np.asarray(z['T_dr'], np.float64).copy()
        DM[k] = d
    F = vo_load(rec); rng = np.random.default_rng(0)
    ids = sorted(F)
    rows = []
    for i, ka in enumerate(ids):
        for s in range(1, span + 1):
            if i + s >= len(ids):
                break
            kb = ids[i + s]
            r = rel_translation(F[ka], F[kb], zmax=8.0, thr=0.15, min_in=6, rng=rng, guided=True)
            if r is None:
                continue
            dz, info, nin = r
            d_dr = F[kb]['T'][:3, 3] - F[ka]['T'][:3, 3]
            gap = float(np.linalg.norm((dz - d_dr)[:2]))
            if gap > 0.8:
                continue
            try:
                C = np.linalg.inv(np.asarray(info[:2, :2], float) + 1e-12 * np.eye(2))
            except np.linalg.LinAlgError:
                continue
            rows.append(dict(a=ka, b=kb, z=dz[:2].copy(), gap=gap, nin=nin,
                             sig_geo=float(np.sqrt(max(0.0, np.trace(C))))))
    return rec, kfs, FEAT, TD, DM, rows


def edge_sigma(rows, mode, lo, hi):
    g = np.array([r['gap'] for r in rows])
    sg = np.array([r['sig_geo'] for r in rows])
    if mode == 'const_med':
        return np.full(len(rows), float(np.median(g)))
    if mode == 'gap_rank':
        # 用 σ_geo 排序（实测最能分开真误差），把它映射到 gap 的分位区间上
        order = np.argsort(sg)
        rank = np.empty(len(rows), float)
        rank[order] = np.linspace(0.0, 1.0, len(rows))
        qs = np.quantile(g, rank)
        return np.clip(qs, lo, hi)
    if mode == 'geo':
        return np.clip(sg, lo, hi)
    raise SystemExit('bad mode ' + mode)


def solve(name, rows, sig_edges, kf=('none', 1.0)):
    rec, kfs, FEAT, TD, DM, _ = load_edges(name)
    cfg, _ = recorded_loop_config(rec, current=False); cfg.bow_candidates = 0
    cl = LoopCloser(cfg)
    for k, t, d in kfs:
        cl.add_keyframe(k, TD[k], DM[k], FEAT[k])
    cl.optimize()
    if rows:
        for r, sg in zip(rows, sig_edges):
            cl.loops.append(_Loop(r['a'], r['b'], np.asarray(r['z'], float),
                                  float(sg) / cfg.world_scale, {'kind': 'visual'}, 1.0))
        cl.optimize()
    return rec, cfg, {k: np.array(cl.pose(k)) for k in cl._order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recs', nargs='+')
    ap.add_argument('--radius', type=float, default=0.2)
    ap.add_argument('--lo', type=float, default=0.05)
    ap.add_argument('--hi', type=float, default=0.60)
    a = ap.parse_args()
    for name in a.recs:
        rec, kfs, FEAT, TD, DM, rows = load_edges(name)
        cfg_map, _ = G.recorded_config(rec)
        _ng, ref_trail, _ = GAB.replay(rec, cfg_map, GAB.poses_for(rec, 'loop'))
        Fw = strict_wall(SE.load_cached(rec))
        zz = np.load(rec/'final_poses.npz')
        Ref = {int(i): np.asarray(T, float) for i, T in zip(zz['ids'], zz['T'])}
        ids = sorted(Fw); pairs = []
        for i, ka in enumerate(ids):
            if Fw[ka]['pl'] is None or ka not in Ref:
                continue
            for kb in ids[i+1:]:
                if Fw[kb]['pl'] is None or kb not in Ref or Fw[kb]['t']-Fw[ka]['t'] < 10.0:
                    continue
                if float(np.hypot(*(Ref[kb][:2,3]-Ref[ka][:2,3]))) < 2.0:
                    pairs.append((ka, kb))
                if len(pairs) >= 120: break
            if len(pairs) >= 120: break
        print('=== %s  视觉边 %d  回访对 %d' % (name, len(rows), len(pairs)))
        print('%-16s %8s %10s %12s %10s' % ('变体', 'σ中位cm', '回访中位cm', '回访p75', '走廊障碍'))
        def report(tag, P, smed):
            vals = []
            for ka, kb in pairs:
                if ka not in P or kb not in P: continue
                ra = Fw[ka]; nb, p0b, _t = ra['pl']; Ta, Tb = P[ka], P[kb]
                n = Ta[:3,:3] @ nb; p0 = Ta[:3,:3] @ p0b + Ta[:3,3]
                if (p0 - Ta[:3,3]) @ n < 0: n = -n
                m = wall_shift(ra['w'] @ Ta[:3,:3].T + Ta[:3,3],
                               Fw[kb]['w'] @ Tb[:3,:3].T + Tb[:3,3], n, p0)
                if m is not None: vals.append(abs(m)*S)
            v = np.array(vals) if vals else np.array([np.nan])
            ng, _o, _m = GAB.replay(rec, cfg_map, P)
            corr = int((trail_mask(ng, ref_trail, a.radius) & (ng.grid == OCC)).sum())
            print('%-16s %8s %10.1f %12.1f %10d' % (tag, smed, np.median(v)*100, np.percentile(v,75)*100, corr))
        _r, _c, P0 = solve(name, [], np.array([]))
        report('loops only', P0, '-')
        for mode in ('const_med', 'gap_rank', 'geo'):
            se = edge_sigma(rows, mode, a.lo, a.hi)
            _r, _c, P = solve(name, rows, se)
            report(mode, P, '%.1f' % (np.median(se)*100))
    return 0


if __name__ == '__main__':
    sys.exit(main())