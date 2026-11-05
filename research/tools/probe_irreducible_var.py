# -*- coding: utf-8 -*-
r"""回归模型到上限了吗 —— 直接估「给定输入后标签还剩多少不可约方差」。

判据（贝叶斯意义）：对任何从输入 x 预测 a 的估计器，

    E[(a - f(x))^2] >= E[Var[a|x]]

右边就是**不可约下界**，与模型容量、训练时长都无关。回归头恰好取到这个下界
（它学的就是 E[a|x]），所以「回归头是不是到顶」等价于「当前 MSE 是不是已经贴近
E[Var[a|x]]」。

估法：在模型实际能看到的**数值输入**（记忆卡那几个数）上做 k 近邻，
用邻居标签的方差近似 Var[a|x]。这会给一个**偏高**的估计（邻居的图像仍可能不同），
所以要这么读：

  * 邻居方差 ≈ 模型 mse  → 模型已在这条输入能给的极限附近；
  * 邻居方差 << 模型 mse → 还有信息没学到（模型/训练问题，不是架构上限）。

同时把 copy / zero 两个平凡基线放进去做参照。
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
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()

    import train_policy_lora as T
    samples = T.build_index(Path(a.dataset), None)
    tr = [s for s in samples if s["split"] == "train"]
    va = [s for s in samples if s["split"] == "val"][:400]
    print("[data] train=%d val=%d" % (len(tr), len(va)), flush=True)

    def feats(v):
        out = np.zeros((len(v), 8), dtype=np.float64)
        for i, s in enumerate(v):
            wp = s["wp"]
            tp = s.get("topo") or (0.0, 0.0, 0.0)
            bd = s.get("body") or (0.0, 0.0)
            out[i, 0] = wp[0]; out[i, 1] = wp[1]
            out[i, 2:5] = tp
            out[i, 5] = bd[0]; out[i, 6] = bd[1]
            out[i, 7] = float(s.get("mode", 0))
        return out

    Xtr = feats(tr)
    Xva = feats(va)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    Atr = np.stack([np.asarray(s["act"])[:, :2] for s in tr])       # (N,12,2)
    Mtr = np.stack([np.asarray(s["mask"]) for s in tr])
    Ava = np.stack([np.asarray(s["act"])[:, :2] for s in va])
    Mva = np.stack([np.asarray(s["mask"]) for s in va])

    S = a.steps
    # 只用「前 S 步都有 mask」的样本来算，避免把 pad 当信号
    ok_tr = Mtr[:, :S].sum(-1) == S
    ok_va = Mva[:, :S].sum(-1) == S
    print("[mask] train_ok=%d val_ok=%d" % (ok_tr.sum(), ok_va.sum()), flush=True)

    Xtr2 = Xtr[ok_tr]; Atr2 = Atr[ok_tr][:, :S]
    Xva2 = Xva[ok_va]; Ava2 = Ava[ok_va][:, :S]

    # kNN：分块算距离，避免 400x7000 的一次性内存
    k = a.k
    within = []
    for i in range(len(Xva2)):
        d = ((Xtr2 - Xva2[i]) ** 2).sum(1)
        idx = np.argpartition(d, k)[:k]
        nb = Atr2[idx]                       # (k,S,2)
        within.append(float(nb.var(axis=0).sum(-1).mean()))
    irr = float(np.mean(within))

    # 对照基线（同一批 val 子集）
    def mse_of(pred):
        e = ((pred - Ava2) ** 2).sum(-1)
        return float(e.mean())

    copy = mse_of(np.repeat(Ava2[:, :1], S, axis=1))
    zero = mse_of(np.zeros_like(Ava2))
    mean_pred = mse_of(np.repeat(Atr2.mean(0, keepdims=True), len(Ava2), axis=0))

    print()
    print("  样本：val %d 条（前 %d 步都有 mask）" % (len(Ava2), S))
    print("  zero 基线            %.4f" % zero)
    print("  copy 基线（抄上一步）  %.4f" % copy)
    print("  常数均值基线          %.4f" % mean_pred)
    print("  **kNN 邻域方差下界**  **%.4f**  (k=%d)" % (irr, k))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
