# -*- coding: utf-8 -*-
r"""不可约方差下界：**按距离匹配**而不是按 k 取邻居，避开「重复样本聚集」的偏差。

为什么重做（2026-11-03）：
  上一版用 kNN(k=8)，结果「仅 body」下界 0.48 反而优于「全部 8 维」1.19 —— 数学上不可能。
  查出原因：录制里速度是离散上报的，大量样本共享**完全相同**的 body 值，
  于是 k=8 的邻居平均距离只有 **0.001**，等于在同一个点上求方差，严重低估。

正确做法：对每个 val 样本，取训练集里**距离小于 r** 的所有样本，用它们的标签方差
近似 Var[a|x]。r 取多个值，看下界随 r 如何变化（r 越小越接近真下界但样本越少）。
同时报告「有多少样本能在 r 内找到 >=min_n 个邻居」，样本不够时不出数。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()

    import train_policy_lora as T
    s = T.build_index(Path(a.dataset), None)
    tr = [x for x in s if x["split"] == "train"]
    va = [x for x in s if x["split"] == "val"][: a.val]

    def feats(v):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            o[i, :2] = wp; o[i, 2:5] = tp; o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    Xtr0, Xva0 = feats(tr), feats(va)
    mu, sd = Xtr0.mean(0), Xtr0.std(0) + 1e-9
    Xtr0 = (Xtr0 - mu) / sd; Xva0 = (Xva0 - mu) / sd
    S = a.steps
    Atr = np.stack([np.asarray(x["act"])[:, :2] for x in tr]).reshape(len(tr), -1)
    Mtr = np.stack([np.asarray(x["mask"]) for x in tr])
    Ava = np.stack([np.asarray(x["act"])[:, :2] for x in va]).reshape(len(va), -1)
    Mva = np.stack([np.asarray(x["mask"]) for x in va])
    oktr = Mtr[:, :S].sum(-1) == S; okva = Mva[:, :S].sum(-1) == S
    Xtr0, A = Xtr0[oktr], Atr[oktr][:, : 2 * S]
    Xva0, Av = Xva0[okva], Ava[okva][:, : 2 * S]
    print("[data] train=%d val=%d  steps=%d" % (len(A), len(Av), S), flush=True)

    print()
    print("  %-8s %-10s %8s %14s" % ("半径r", "有邻居的样本", "平均邻居数", "邻域方差"))
    for r in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        r2 = r * r
        vals, ns = [], []
        for i in range(len(Xva0)):
            d = ((Xtr0 - Xva0[i]) ** 2).sum(1)
            m = d <= r2
            c = int(m.sum())
            if c >= a.min_n:
                vals.append(float(A[m].var(axis=0).mean()))
                ns.append(c)
        pct = 100.0 * len(vals) / len(Xva0)
        if vals:
            print("  %-8.2f %-10s %8.0f %14.4f"
                  % (r, "%.0f%%" % pct, float(np.mean(ns)), float(np.mean(vals))))
        else:
            print("  %-8.2f %-10s %8s %14s" % (r, "%.0f%%" % pct, "-", "样本不足"))

    print()
    print("  参照：copy 基线 %.4f / 模型实测 1.7569"
          % float((((Av[:, :2] - Av) ** 2).sum(-1)).mean()))
    print("  判读：只信「有邻居的样本」接近 100%% 且邻居数足够的那几行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
