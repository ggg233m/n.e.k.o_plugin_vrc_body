# -*- coding: utf-8 -*-
r'''仪器标定：(T_map − T_dr) 的仿射「尺度」到底是不是 DR 尺度误差的读数？

    python research/tools/affine_caliber.py 20261001_044153
    python research/tools/affine_caliber.py 20261001_044153 --odo-sigma 0.5

为什么需要它
----------
`drift_shape.py` 的证据 B 把 ``r = p_map − p_dr ≈ A·p_dr + t`` 分解出的 ``scale`` 当作
「DR 链的尺度偏差」。**本工具证明这个读数是错的**：往瓶颈里灌一个**已知**的尺度误差，
看仿射尺度报不报得出来。

做法（受控注入，不碰图像/PnP/回环配置）
------------------------------
只把喂给 ``LoopCloser.add_keyframe`` 的 ``T_dr`` 平面位置乘一个已知倍率 ``f``
（等价于 ``world_scale`` 或 ``osc_speed_scale`` 设错）。**回环约束全部来自双目 PnP，
完全不变**，所以这是干净的输入-输出实验：

* ``f < 1`` ⇒ DR 少算位移 ⇒ 地图相对 DR 偏大 ⇒ 期望仿射尺度 ≈ ``+(1−f)``
* ``f > 1`` ⇒ DR 多算位移 ⇒ 期望仿射尺度 ≈ ``−(f−1)``

实测传函（三场一致）
------------------
**增益只有 0.05~0.25**，即注入 10% 的 DR 尺度误差，仿射尺度只动 0.5~2.5 个百分点。
所以本仓库到处引用的「045615 尺度 −9.91%」**不能**读成「DR 长了 9.91%」。

机制：里程边比回环边硬得多（``optimize()`` 里 ``ow = 1/σ_odo²``，``σ_odo = 0.03`` 米，
而回环 ``σ_loop = 0.1`` 米 ⇒ 每条里程边刚度是回环边的 ~11 倍，且条数远多）
⇒ 地图尺寸基本跟着 DR 走，回环改不动它。直接证据是 ``map_rms/dr_rms ≈ 1.00``，
且它对 ``f`` 的响应只有 −0.06~−0.08 / 10pp。用 ``--odo-sigma 0.5`` 放松里程边，
该响应升到 −0.16~−0.28，仿射尺度也开始跟着动 ⇒ 机制确认。

⚠️ 反面结论同样重要：仿射尺度**在固定数据上随 ``σ_odo`` 变**（044153 的 ``f=1`` 从
+0.77% 变到 −0.94%）。它因此不是 DR 精度的读数，而是**位姿图权重的读数**。
'''
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'research' / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'research' / 'tools'))

from backend.nav_loop import LoopCloser                    # noqa: E402
from loop_replay import features_of, recorded_loop_config   # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def affine(P_dr: np.ndarray, P_mp: np.ndarray, w: np.ndarray | None = None) -> dict:
    """r = A·p_dr + t 的无权 / 加权最小二乘，拆出尺度、旋转、剪切（drift_shape 同口径）。"""
    r = P_mp - P_dr
    p = P_dr
    n = len(p)
    X = np.zeros((2 * n, 6)); y = np.zeros(2 * n)
    X[0::2, 0], X[0::2, 1], X[0::2, 2] = p[:, 0], p[:, 1], 1.0
    X[1::2, 3], X[1::2, 4], X[1::2, 5] = p[:, 0], p[:, 1], 1.0
    y[0::2], y[1::2] = r[:, 0], r[:, 1]
    if w is None:
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = float(np.sqrt(np.mean((X @ sol - y) ** 2)))
        raw = float(np.sqrt(np.mean(y ** 2)))
    else:
        ww = np.repeat(w, 2)
        sw = np.sqrt(ww)[:, None]
        sol, *_ = np.linalg.lstsq(X * sw, y * np.sqrt(ww), rcond=None)
        m = ww > 0
        res = float(np.sqrt(np.average((X @ sol - y)[m] ** 2, weights=ww[m])))
        raw = float(np.sqrt(np.average(y ** 2, weights=ww)))
    A = np.array([[sol[0], sol[1]], [sol[3], sol[4]]])
    s11 = 0.5 * (A[0, 0] - A[1, 1]); s12 = 0.5 * (A[0, 1] + A[1, 0])
    return {'scale': 0.5 * (A[0, 0] + A[1, 1]) * 100,
            'rot': math.degrees(0.5 * (A[1, 0] - A[0, 1])),
            'shear': math.sqrt(2.0 * (s11 ** 2 + s12 ** 2)) * 100,
            'res': res * 100, 'raw': raw * 100}


def rms_radius(P: np.ndarray) -> float:
    """轨迹相对自身质心的 rms 半径（米）。尺度误差正是作用在这个量上。"""
    return float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('recs', nargs='*', default=['20261001_044153'])
    ap.add_argument('--factors', default='0.70,0.85,1.00,1.15,1.30',
                    help='注入倍率 f（逗号分隔）；1.00 是未注入的基准')
    ap.add_argument('--odo-sigma', type=float, default=None,
                    help='覆盖 odo_sigma0_m / odo_sigma_frac（录制值 0.03）；放大会让回环说得上话')
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()
    factors = [float(v) for v in args.factors.split(',')]

    out = []
    for name in args.recs:
        rec = ROOT / 'navmesh_recordings' / name
        if not rec.exists():
            print(f'[skip] {name}', file=sys.stderr)
            continue
        ev = load_jsonl(rec / 'events.jsonl')
        kfs = [(int(e['k']), float(e['t']), float(e['dist_m'])) for e in ev if e.get('kind') == 'kf']
        FEAT, TD, DM = {}, {}, {}
        for k, t, d in kfs:
            with np.load(rec / 'kf' / f'{k:06d}.npz') as z:
                FEAT[k] = features_of(z); TD[k] = np.asarray(z['T_dr'], np.float64).copy()
            DM[k] = d
        print(f'\n=== {name} ===  {len(kfs)} 关键帧'
              + (f'   σ_odo 覆盖为 {args.odo_sigma:g}' if args.odo_sigma else ''))
        print(f"{'f':>6}{'loops':>7}{'dr_rms_m':>10}{'map_rms_m':>11}{'map/dr':>9}"
              f"{'scale%':>9}{'rot_deg':>9}{'shear%':>9}{'|r|中位cm':>11}{'res_cm':>9}")
        base = None
        rows = []
        for f in factors:
            cfg, _ = recorded_loop_config(rec, current=False); cfg.bow_candidates = 0
            if args.odo_sigma is not None:
                cfg.odo_sigma0_m = args.odo_sigma; cfg.odo_sigma_frac = args.odo_sigma
            cl = LoopCloser(cfg)
            nloop = 0
            for k, t, d in kfs:
                T = TD[k].copy(); T[:2, 3] = T[:2, 3] * f
                nloop += len(cl.add_keyframe(k, T, DM[k], FEAT[k]))
            poses = cl.poses(); ids = sorted(poses)
            P_dr = np.array([TD[k][:2, 3] * f for k in ids])
            P_mp = np.array([np.asarray(poses[k], float)[:2, 3] for k in ids])
            q = affine(P_dr, P_mp)
            drr, mrr = rms_radius(P_dr), rms_radius(P_mp)
            rmed = float(np.median(np.hypot(*(P_mp - P_dr).T))) * 100
            print(f"{f:>6.2f}{nloop:>7}{drr:>10.2f}{mrr:>11.2f}{mrr / max(drr, 1e-9):>9.4f}"
                  f"{q['scale']:>+9.2f}{q['rot']:>+9.3f}{q['shear']:>9.2f}{rmed:>11.1f}{q['res']:>9.1f}")
            rows.append({'f': f, 'loops': nloop, 'map_over_dr': mrr / max(drr, 1e-9), **q})
            if abs(f - 1.0) < 1e-9:
                base = rows[-1]
        if base and len(rows) > 1:
            lo = min(rows, key=lambda r: r['f']); hi = max(rows, key=lambda r: r['f'])
            span_f = hi['f'] - lo['f']
            gain = (hi['scale'] - lo['scale']) / (-100 * span_f) if span_f else float('nan')
            print(f"  ★ 传函增益 |Δscale| / |Δf| = {abs(gain):.3f}"
                  f"  （=1 才说明「仿射尺度」可以直接当 DR 尺度误差读）")
            print(f"  ★ map_rms/dr_rms 从 {lo['map_over_dr']:.4f}（f={lo['f']:g}）"
                  f" 到 {hi['map_over_dr']:.4f}（f={hi['f']:g}）"
                  f"  ⇒ {'地图跟着 DR 变，回环拉不住' if abs(hi['map_over_dr'] - lo['map_over_dr']) < 0.1 else '回环确实在拉地图'}")
        out.append({'rec': name, 'rows': rows})
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'\n已写入 {args.json}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())