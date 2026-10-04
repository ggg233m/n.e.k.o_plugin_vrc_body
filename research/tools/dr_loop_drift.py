# -*- coding: utf-8 -*-
r"""用 RTAB-Map 回环当"同一地点"证据，量 OSC+HMD 航位推算的**真实漂移**。

    python research/tools/dr_loop_drift.py .slam_probe/stereo_seq/run6 .tmp/rtabmap_run6/f2m_lite_r1/rtabmap.db
    python research/tools/dr_loop_drift.py .slam_probe/stereo_seq/run5_ipd126 .tmp/rtabmap_run5_ipd126/f2m_lite_r1/rtabmap.db --yaw-sign +1

**来历**：原脚本是 `.tmp/dr_drift/dr_loop_drift.py`（2026-09-27 实验），
`Docs/双目序列run5-7结论汇总（2026-09-27）.md` §3 记「脚本写了但**输出没有留在 `.tmp/`**
⇒ 结论无产物，需重跑」，§7.4 又把它列为待补实验。本次（2026-10-05）把它搬进仓库
并修好两处漂移：① 依赖 `stereo_seq_ground_truth` 的路径假设还停在旧的 `tools/`
（该文件已迁到 `research/tools/`）；② 输出路径原来靠 `parents[3]` 反推、随 db 布局脆断。

**它量的是什么**：与 `loop_selfcal.py` 同源但不同一个量 ——

* `loop_selfcal.py` 只比**长度**（`s = |OSC| / |几何|`），一个标量；
* 本脚本比**矢量**：把 DR 的世界位移按 a 时刻 HMD yaw 转回头部系，与回环变换给的同段位移相减，
  得到**漂移矢量**。因此它能回答"差在长度还是差在方向"。

⚠️ **盲区与 `loop_selfcal` 相同**：只能看到**被 RTAB-Map 接受**的回环（幸存者偏差）；
且它依赖 `build_ground_truth` 的 OSC 时间偏移与 yaw 符号标定，这两项错了会整体偏。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "research" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stereo_seq_ground_truth import build_ground_truth      # noqa: E402
from backend.pose_math import advance                       # noqa: E402

S = 0.755          # 世界米/追踪米（与 mapper 同值）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("db")
    ap.add_argument("--yaw-sign", type=float, default=-1.0)
    ap.add_argument("--osc-offset", type=float, default=0.13)
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "dr_loop_drift")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    stamp = dict(con.execute("select id, stamp - 1000000000.0 from Node"))
    loops = []
    for f, t, blob in con.execute(
            "select from_id, to_id, transform from Link where type=1 and from_id > to_id"):
        m = np.frombuffer(blob, np.float32).reshape(3, 4)
        loops.append((t, f, m[:, 3].astype(float)))          # 旧 a=t，新 b=f

    ids = sorted(stamp)
    ft = np.array([stamp[i] for i in ids])
    gt = build_ground_truth(Path(args.seq), ft, args.yaw_sign, args.osc_offset)
    x, z, yaw, valid = gt["x"], gt["z"], gt["yaw"], gt["valid"]
    idx = {i: k for k, i in enumerate(ids)}
    path = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])

    rows = []
    for ia, ib, tr in loops:
        ka, kb = idx[ia], idx[ib]
        if not (valid[ka] and valid[kb]):
            continue
        dx, dz = x[kb] - x[ka], z[kb] - z[ka]
        lr, lf = advance(0.0, 0.0, dx, dz, -yaw[ka])          # 世界 → a 时刻头部系（右, 前）
        loop_rf = S * np.array([-tr[1], tr[0]])               # base (x 前, y 左) → (右, 前)
        rows.append(dict(a=ia, b=ib, tb=float(ft[kb]), dt=float(ft[kb] - ft[ka]),
                         path=float(path[kb] - path[ka]), dr=[lr, lf], loop=loop_rf.tolist(),
                         loop_z=float(S * tr[2]),
                         err_vec=float(np.hypot(lr - loop_rf[0], lf - loop_rf[1])),
                         err_vec_flip=float(np.hypot(lr + loop_rf[0], lf + loop_rf[1])),
                         err_lb=float(abs(np.hypot(dx, dz) - np.hypot(*loop_rf)))))
    if not rows:
        print("没有落在有效窗口内的回环")
        return 0

    rows.sort(key=lambda r: r["tb"])
    ev = np.array([r["err_vec"] for r in rows])
    ef = np.array([r["err_vec_flip"] for r in rows])
    use = "err_vec" if np.median(ev) <= np.median(ef) else "err_vec_flip"
    print(f"回环 {len(rows)} 条（有效窗口内）；变换方向检查：中位 正向 {np.median(ev):.2f} m / "
          f"反向 {np.median(ef):.2f} m → 取 {use}")
    print(f"{'a→b 帧':>12} {'t_b':>7} {'间隔s':>7} {'DR路程m':>9} {'|回环|m':>8} "
          f"{'漂移m':>7} {'下界m':>7} {'%路程':>7}")
    for r in rows:
        e = r[use]
        print(f"{r['a']:5d}→{r['b']:5d} {r['tb']:7.1f} {r['dt']:7.1f} {r['path']:9.1f} "
              f"{np.hypot(*r['loop']):8.2f} {e:7.2f} {r['err_lb']:7.2f} "
              f"{100*e/max(r['path'],1e-6):6.1f}")

    E = np.array([r[use] for r in rows])
    P = np.array([r["path"] for r in rows])
    print(f"漂移中位 {np.median(E):.2f} m、最大 {E.max():.2f} m；"
          f"对应路程中位 {np.median(P):.1f} m；漂移/路程 中位 {np.median(E/P):.1%}")
    # ★ 与 drift_shape 证据 A 同一判据：E = a + b·P，看"地板"与"累积斜率"哪个在
    b, a = np.polyfit(P, E, 1)
    pred = a + b * P
    ss = 1.0 - ((E - pred) ** 2).sum() / max(((E - E.mean()) ** 2).sum(), 1e-12)
    print(f"拟合 E = a + b·L： 地板 a = {a*100:+.1f} cm，斜率 b = {b*100:.3f} %/m，R² {ss:.3f}")
    if abs(a) > 0.05 and abs(b) < 0.002:
        print("  判读：地板显著、斜率近零 ⇒ 与路程**无关**的固定偏差，不是累积漂移")
    elif b > 0.002:
        print(f"  判读：斜率显著 ⇒ 有随路程累积的成分（{P.max():.0f} m 上折合 {b*P.max()*100:.0f} cm）")

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{Path(args.seq).name}.json"
    json.dump(dict(seq=Path(args.seq).name, db=str(args.db), use=use,
                   yaw_sign=args.yaw_sign, osc_offset=args.osc_offset,
                   n=len(rows), floor_cm=a * 100, slope_pct_per_m=b * 100, r2=ss,
                   rows=rows), open(dest, "w", encoding="utf-8"), indent=1)
    print(f"产物：{dest}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
