# -*- coding: utf-8 -*-
r'''方案 ④：**局部**修正 —— 只动「能解出视觉约束」的相邻对，不重解全局表。

## 为什么要它（2026-10-28 实测）

同一份录制、同一批「能解出视觉 Δt」的 111 对相邻帧上：

| 位姿源 | 相邻帧墙面失配 |
|---|---|
| 视觉 Δt oracle（逐对直接修正）| **5.7 cm** |
| `vo_g6`（全局最小二乘）| 10.2 cm |
| `joint_poses`（全局）| 10.1 cm |
| `final_poses`（无视觉）| 20.9 cm |
| 单帧自厚（下界）| ~4.0 cm |

⇒ 全局解**在有约束的地方也只到 10 cm**，而同样的约束逐对用能到 5.7 cm。
差的 4.5 cm 来自「20% 的好约束要和 80% 的 DR 软约束一起解」——
全局最小二乘会把视觉残差平摊到整串无约束帧上。

## 做法

不做全局优化。对每个**成功解出视觉 Δt** 的相邻对 (a,b)：
  1. 把 a 的位姿固定；
  2. 只把 b 的位姿平移到 a + Δt；
  3. 误差**不外推**，就在 (a,b) 这一对上消化。

这样在能约束的对上，残差 = 视觉解自身的精度；在不能约束的对上维持原样。
代价：位移序列不再满足「相邻位移求和 = 总路程」，**不是物理一致的轨迹**，
只适合「按帧看局部一致性」的用途（点云导出就是这样用的）。

⚠️ 因此**不能**拿它当 SLAM 位姿用：跨帧范围一拉长，连续性就断了。

用法：
    python research/tools/local_vo_fix.py 20261026_184318
    python research/tools/local_vo_fix.py 20261026_184318 --out localfix_poses.npz --span 1
'''
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "tools"))

import kf_visual_odom as V      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec", help="录制目录名（navmesh_recordings/ 下）")
    ap.add_argument("--base", default="final_poses", help="基准位姿表（默认 final_poses）")
    ap.add_argument("--out", default="localfix_poses", help="输出表名")
    ap.add_argument("--span", type=int, default=1, help="约束跨度（默认 1；>1 实测覆盖率崩，见文档）")
    ap.add_argument("--zmax", type=float, default=5.0)
    ap.add_argument("--thr", type=float, default=0.10, help="3D-3D 内点阈值 m")
    ap.add_argument("--min-in", type=int, default=10, help="最少内点")
    ap.add_argument("--max-corr", type=float, default=1.0,
                    help="视觉与 DR 帧间差的接受上限 m（超过视为误匹配）")
    a = ap.parse_args()

    rec = V.REC_ROOT / a.rec
    if not rec.is_dir():
        raise SystemExit(f"[fail] 没有这个录制：{rec}")
    F = V.load(rec)
    ids = sorted(F)

    bp = rec / f"{a.base}.npz"
    if not bp.exists():
        raise SystemExit(f"[fail] 没有基准位姿表 {bp}")
    with np.load(bp) as z:
        base = {int(k): np.asarray(T, np.float64) for k, T in zip(z["ids"], z["T"])}
    use = [k for k in ids if k in base]
    print(f"[{a.rec}] 帧 {len(ids)}，基准表覆盖 {len(use)}（{a.base}）")

    rng = np.random.default_rng(0)
    pos = {k: base[k][:2, 3].copy() for k in use}
    n_ok = n_rej = 0
    gaps = []

    # 顺序推进：a 用**已修正后**的位置，b 用 Δt 相对它落点。
    for i in range(len(use) - a.span):
        ka, kb = use[i], use[i + a.span]
        r = V.rel_translation(F[ka], F[kb], zmax=a.zmax, thr=a.thr,
                              min_in=a.min_in, rng=rng)
        if r is None:
            continue
        dt = np.asarray(r[0], np.float64)[:2]
        dr = base[kb][:2, 3] - base[ka][:2, 3]
        gap = float(np.linalg.norm(dt - dr))
        if gap > a.max_corr:
            n_rej += 1
            continue
        n_ok += 1
        gaps.append(gap)
        pos[kb] = pos[ka] + dt

    print(f"  接受 {n_ok}，拒绝(超 max_corr) {n_rej}，"
          f"成功率 {100*n_ok/max(len(use)-a.span,1):.0f}%")
    if gaps:
        g = np.array(gaps)
        print(f"  |视觉Δt − DR| : med {np.median(g)*100:.1f} cm  p90 {np.percentile(g,90)*100:.1f} cm")

    T_out = np.stack([base[k] for k in use])
    T_out[:, :2, 3] = np.stack([pos[k] for k in use])
    dest = rec / f"{a.out}.npz"
    np.savez_compressed(dest, ids=np.array(use), T=T_out)
    shift = np.linalg.norm(T_out[:, :2, 3] - np.stack([base[k][:2, 3] for k in use]), axis=1)
    print(f"  相对基准的位移：med {np.median(shift):.3f} m  max {shift.max():.3f} m")
    print(f"  写出 {dest}")
    print()
    print("⚠️ 这不是物理一致的轨迹：只保证「有约束的那一对」靠视觉对齐，")
    print("   误差不外推。适合按帧看局部一致性，**不能**当 SLAM 位姿用。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())