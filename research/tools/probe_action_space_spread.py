# -*- coding: utf-8 -*-
"""补充：动作空间里到底有没有「分歧」这件事 —— 分方向与幅值两维看。

为什么要补（2026-11-02）：probe_action_multimodality 报「单峰」，但它的全局 R 就已经是 0.982，
打乱对照 0.983 —— 也就是说**方向在整个数据集里几乎不变**，这个测试没有分辨力。
原因很可能是：动作是机体系 (vx, vz)，而我们**把 ω 交给了跟随器**，
人「在岔路口往左还是往右」的选择根本没有进入动作空间。

本脚本把这件事量清楚：
  1. 机体系方向角的分布（负角=左，正角=右）；
  2. 幅值（速度）的分布；
  3. 近邻内「方向」与「幅值」各自的离散度 —— 看多峰性到底在哪一维；
  4. 有没有任何一个子集呈现真正的双峰（比如快/慢两种走法）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    d = np.load(a.feats, allow_pickle=True)
    A, M = d["act"], d["mask"]
    spd0 = np.linalg.norm(A[:, 0, :], axis=-1)
    sel = (M[:, 0] > 0) & (spd0 > 0.1)
    a0 = A[sel][:, 0, :]
    n = len(a0)
    print("[data] 移动样本 %d" % n, flush=True)

    ang = np.degrees(np.arctan2(a0[:, 0], a0[:, 1]))   # atan2(vx 右, vz 前)
    spd = np.linalg.norm(a0, axis=-1)
    q = lambda x, p: float(np.percentile(x, p))

    out = dict(
        n=int(n),
        angle_deg=dict(p05=q(ang, 5), p25=q(ang, 25), p50=q(ang, 50),
                       p75=q(ang, 75), p95=q(ang, 95),
                       abs_p95=float(np.percentile(np.abs(ang), 95)),
                       frac_abs_gt_20=float((np.abs(ang) > 20).mean()),
                       frac_abs_gt_45=float((np.abs(ang) > 45).mean())),
        speed=dict(p05=q(spd, 5), p50=q(spd, 50), p95=q(spd, 95),
                   cv=float(spd.std() / max(spd.mean(), 1e-9))),
    )

    print()
    print("  机体系方向角（度，负=左 正=右）:")
    print("    p05=%.1f p25=%.1f p50=%.1f p75=%.1f p95=%.1f"
          % (out["angle_deg"]["p05"], out["angle_deg"]["p25"], out["angle_deg"]["p50"],
             out["angle_deg"]["p75"], out["angle_deg"]["p95"]))
    print("    |角|>20度 占 %.2f%%，|角|>45度 占 %.2f%%"
          % (100 * out["angle_deg"]["frac_abs_gt_20"],
             100 * out["angle_deg"]["frac_abs_gt_45"]))
    print("  速度: p05=%.2f p50=%.2f p95=%.2f  CV=%.2f"
          % (out["speed"]["p05"], out["speed"]["p50"], out["speed"]["p95"], out["speed"]["cv"]),
          flush=True)

    # 方向与速度的联合：看有没有「慢-左 / 快-右」这种子结构
    h, xe, ye = np.histogram2d(ang, spd, bins=[8, 6])
    out["joint_hist"] = dict(angle_edges=[float(x) for x in xe],
                             speed_edges=[float(y) for y in ye],
                             counts=h.astype(int).tolist())
    print("  方向 x 速度 联合直方图（行=方向 8 档，列=速度 6 档）:")
    for i, row in enumerate(h.astype(int)):
        print("    %.0f..%.0f : %s" % (xe[i], xe[i + 1], " ".join("%5d" % c for c in row)))

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "action_space_spread.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "action_space_spread.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
