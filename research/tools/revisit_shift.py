# -*- coding: utf-8 -*-
r'''判据实验：单帧位姿修正（视觉 Δt）能不能修**回访错位**？

已知（`点云拖影修复原型与栅格影响（2026-10-09）` §二/§三）：视觉 Δt 把**相邻帧**的主墙面
整面偏移从 11.7–15.9 cm 修到 3.1–6.9 cm（≈ 单帧自厚，4 场）。
已知（`OSC速度尺度与回访拖影（2026-10-12）` §三）：人眼看到的弧形拖影是**回访错位** ——
同一面墙在不同**经过**里错开 DR 108 cm / 回环后 lk100 18–28 cm。

两者是不同的自由度：视觉 Δt 只动**相邻帧之间**的相对位姿；回访错位要求**相隔几十米的两趟**
互相对齐。本实验量后者在三种位姿下的残余，并给出**单帧视觉的天花板**：
直接问「把这一对回访帧的相对位姿换成用它们自己解出的视觉平移」。
'''
import argparse, json, math, sys, numpy as np
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'research'/'tools'))
import smear_eval as SE

S = 0.755
CAMH = 1.75            # 地图单位；严格口径的常数（.tmp/smear_strict.py:8）


def strict_wall(F):
    rng = np.random.default_rng(0)
    for k, r in F.items():
        P = r['pts']; z = P[:, 2] + CAMH
        r['w'] = P[(z > 0.3 / S) & (z < 1.6 / S)]
        r['pl'] = SE.fit_vertical_plane(r['w'], rng, max_vz=0.06, min_in=600)
    return F


def wall_shift(PA, PB, n, p0):
    '''两片严格墙面点集沿法向的**带符号**中位差（同 `smear_strict.smed` 口径）。'''
    def agg(P):
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        E = np.column_stack([e1, np.cross(n, e1)])
        X = P - p0
        kk = np.floor(X @ E / SE.CELL).astype(np.int64)
        key = kk[:, 0] * 1_000_003 + kk[:, 1]
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        h = X @ n
        o = np.lexsort((h, inv))
        st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        hs = h[o]
        return u, cnt, 0.5 * (hs[st + (cnt - 1) // 2] + hs[st + cnt // 2])
    ua, ca, ma = agg(PA); ub, cb, mb = agg(PB)
    _, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    ok = (ca[ia] >= SE.MINPTS) & (cb[ib] >= SE.MINPTS) & (np.abs(ma[ia]) < 0.1)
    if int(ok.sum()) < 5:
        return None
    return float(np.median(mb[ib][ok] - ma[ia][ok]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('--dt', type=float, default=10.0)
    ap.add_argument('--near', type=float, default=2.0)
    ap.add_argument('--max-pairs', type=int, default=120)
    ap.add_argument('--srcs', default='lk100')
    a = ap.parse_args()
    rec = SE.REC_ROOT / a.rec
    F = strict_wall(SE.load_cached(rec))
    ids = sorted(F)
    print('%s：严格墙面 %d/%d 帧' % (a.rec, sum(1 for r in F.values() if r['pl'] is not None), len(F)))

    pos = {k: F[k]['T'] for k in ids}
    srcs = {'dr': pos}
    loop = {}
    if (rec / 'final_poses.npz').exists():
        z = np.load(rec / 'final_poses.npz')
        loop = {int(k): np.asarray(T, float) for k, T in zip(z['ids'], z['T'])}
        srcs['sync'] = loop
    for nm in a.srcs.split(','):
        p = rec / (nm + '.npz')
        if p.exists():
            z = np.load(p)
            srcs[nm] = {int(k): np.asarray(T, float) for k, T in zip(z['ids'], z['T'])}
    Ref = loop if loop else pos
    tt = {k: F[k]['t'] for k in ids}

    pairs = []
    for i, ka in enumerate(ids):
        if F[ka]['pl'] is None or ka not in Ref:
            continue
        for kb in ids[i + 1:]:
            if F[kb]['pl'] is None or kb not in Ref or tt[kb] - tt[ka] < a.dt:
                continue
            d = float(np.hypot(*(Ref[kb][:2, 3] - Ref[ka][:2, 3])))
            if d < a.near:
                pairs.append((ka, kb, d))
            if len(pairs) >= a.max_pairs:
                break
        if len(pairs) >= a.max_pairs:
            break
    if not pairs:
        print('没有回访对')
        return 1
    print('回访对 %d 对；参考位姿下距离 中位 %.2f m' % (len(pairs), np.median([p[2] for p in pairs])))

    print()
    print('%-8s %8s %8s %8s' % ('位姿', '对数', '|偏移|中位cm', 'p75 cm'))
    for nm, TP in srcs.items():
        vals = []
        for ka, kb, _d in pairs:
            Ta, Tb = TP.get(ka), TP.get(kb)
            if Ta is None or Tb is None:
                continue
            ra = F[ka]; nb, p0b, _th = ra['pl']
            n = Ta[:3, :3] @ nb
            p0 = Ta[:3, :3] @ p0b + Ta[:3, 3]
            if (p0 - Ta[:3, 3]) @ n < 0:
                n = -n
            PA = ra['w'] @ Ta[:3, :3].T + Ta[:3, 3]
            PB = F[kb]['w'] @ Tb[:3, :3].T + Tb[:3, 3]
            m = wall_shift(PA, PB, n, p0)
            if m is not None:
                vals.append(abs(m) * S)
        if vals:
            v = np.array(vals)
            print('%-8s %8d %8.1f %8.1f' % (nm, len(v), np.median(v) * 100, np.percentile(v, 75) * 100))

    print()
    print('判读要点：dr → lk100 的降幅 = 回环削掉多少；vo_g6 若不降反升，' )
    print('说明「只串相邻帧」的视觉里程把误差累积到了回访对上（§六 目视已见 vo_g6 全局漂走）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())