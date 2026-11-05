# -*- coding: utf-8 -*-
r"""数值输入里还有 47% 没榨干，钱丢在哪一条通道上。

已知（2026-11-03）：
  kNN 用 8 个数值输入（wp dx/dz、topo 3 项、body vx/vz、mode）就能到 mse 1.197，
  而模型只到 1.76。这说明**数值输入本身含有模型没利用的信息**。

逐个通道做「剔除实验」：每次只用一部分输入特征做 kNN，看 mse 变化，
从而给每条通道的信息量排序。若某通道单独就逼近 1.197，说明它才是主力，
模型没学到它 = 训练问题；若所有单通道都远高于 1.197、要全部合起来才行，
说明模型需要更好的非线性组合能力（头容量问题）。

另外做一件关键的事：**只用 body（上一拍速度）**能到多少 —— 这是「抄捷径」的量化。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


NAMES = ["wp_dx", "wp_dz", "topo_p", "topo_l", "topo_r", "body_vx", "body_vz", "mode"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--val", type=int, default=300)
    a = ap.parse_args()

    import train_policy_lora as T
    samples = T.build_index(Path(a.dataset), None)
    tr = [s for s in samples if s["split"] == "train"]
    va = [s for s in samples if s["split"] == "val"][: a.val]

    def feats(v):
        out = np.zeros((len(v), 8), dtype=np.float64)
        for i, s in enumerate(v):
            wp = s["wp"]; tp = s.get("topo") or (0.0, 0.0, 0.0)
            bd = s.get("body") or (0.0, 0.0)
            out[i, :2] = wp; out[i, 2:5] = tp; out[i, 5:7] = bd
            out[i, 7] = float(s.get("mode", 0))
        return out

    Xtr0, Xva0 = feats(tr), feats(va)
    mu, sd = Xtr0.mean(0), Xtr0.std(0) + 1e-9
    Xtr0 = (Xtr0 - mu) / sd; Xva0 = (Xva0 - mu) / sd
    S = a.steps
    Atr = np.stack([np.asarray(s["act"])[:, :2] for s in tr])
    Mtr = np.stack([np.asarray(s["mask"]) for s in tr])
    Ava = np.stack([np.asarray(s["act"])[:, :2] for s in va])
    Mva = np.stack([np.asarray(s["mask"]) for s in va])
    ok_tr = Mtr[:, :S].sum(-1) == S; ok_va = Mva[:, :S].sum(-1) == S
    Xtr0, A = Xtr0[ok_tr], Atr[ok_tr][:, :S]
    Xva0, Av = Xva0[ok_va], Ava[ok_va][:, :S]

    def knn_mse(cols):
        Xa, Xb = Xtr0[:, cols], Xva0[:, cols]
        tot = 0.0
        for i in range(len(Xb)):
            d = ((Xa - Xb[i]) ** 2).sum(1)
            idx = np.argpartition(d, a.k)[: a.k]
            tot += float(A[idx].var(axis=0).sum(-1).mean())
        return tot / len(Xb)

    groups = {
        "全部 8 维": list(range(8)),
        "仅 body (vx,vz)": [5, 6],
        "仅 wp (dx,dz)": [0, 1],
        "仅 topo (3项)": [2, 3, 4],
        "wp+body": [0, 1, 5, 6],
        "wp+body+topo": [0, 1, 2, 3, 4, 5, 6],
    }
    print("[data] train_ok=%d val_ok=%d k=%d" % (len(A), len(Av), a.k), flush=True)
    print()
    print("  %-20s %10s" % ("输入组合", "kNN下界"))
    base = None
    for name, cols in groups.items():
        v = knn_mse(cols)
        if base is None:
            base = v
        print("  %-20s %10.4f" % (name, v))

    print()
    print("  参照：copy 基线 %.4f / 模型实测 1.7569"
          % float((((Av[:, :1] - Av) ** 2).sum(-1)).mean()))
    print()
    print("  判读：")
    print("   * 若「仅 body」就接近全部 8 维 ⇒ 其他通道几乎不提供信息，模型抄 body 是理性的；")
    print("   * 若「全部」明显低于任一单组 ⇒ 信息在**组合**里，需要非线性（头容量/特征问题）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
