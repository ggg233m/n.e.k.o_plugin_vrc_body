# -*- coding: utf-8 -*-
r"""独立复现队友的「45 参数二次 ridge」，验证 1.2392 是否可信。

队友（policy-designer, task-14）声称：
  只用 8 个数值特征、不含图像、不含 LM，一个二次 ridge 做到 mse8 = 1.2392，
  比 0.8B+LoRA+Linear 的 1.7569 好 29.5%。

这个结论如果成立，会直接改变路线（说明主干丢信息）。所以要**自己独立复现**，
并且做三个必须的对照，否则很容易得出虚高的数：

  A. 二次特征扩张后的维度：8 个原始 + 8 个平方 + 28 个交叉 = 44，加偏置 45 参数。
  B. **必须用训练集 fit、验证集评**——若在同一批上 fit+评，ridge 会严重过拟合
     （45 参数对 400 样本也不算少，但真正的陷阱是**同 session 时间相邻样本**）。
  C. **必须检查验证集与训练集是否来自同一 session**：录制里相邻时间点几乎相同，
     若 val 样本在 train 里有「时间近邻」，ridge 等于在背答案，误差会虚低。

本脚本同时输出「随机切分」与「按 session 切分」两种口径，差异就是泄漏量。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def quad(X: np.ndarray) -> np.ndarray:
    """8 -> 44：原值 + 平方 + 上三角交叉项。"""
    n, d = X.shape
    feats = [X]
    feats.append(X ** 2)
    cross = []
    for i in range(d):
        for j in range(i + 1, d):
            cross.append((X[:, i] * X[:, j])[:, None])
    feats.append(np.concatenate(cross, axis=1))
    return np.concatenate(feats, axis=1)


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ Y
    return np.linalg.solve(A, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--lam", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 1.0])
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_policy_lora as T
    s = T.build_index(Path(a.dataset), None)
    S = a.steps

    def feats(v):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            o[i, :2] = wp; o[i, 2:5] = tp; o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    def labels(v):
        A = np.stack([np.asarray(x["act"])[:, :2] for x in v]).reshape(len(v), -1)
        M = np.stack([np.asarray(x["mask"]) for x in v])
        ok = M[:, :S].sum(-1) == S
        return A[ok][:, : 2 * S], ok

    tr = [x for x in s if x["split"] == "train"]
    va = [x for x in s if x["split"] == "val"]
    print("[data] train=%d val=%d" % (len(tr), len(va)), flush=True)

    # ---- 口径 1：项目原有 split（val 是时间上的后 20%）----
    def run(train_l, val_l, tag):
        Xtr, Xva = feats(train_l), feats(val_l)
        Ytr, oktr = labels(train_l)
        Yva, okva = labels(val_l)
        Xtr, Xva = Xtr[oktr], Xva[okva]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
        Ftr, Fva = quad(Xtr), quad(Xva)
        best = None
        for lam in a.lam:
            W = ridge_fit(Ftr, Ytr, lam)
            pred = Fva @ W
            mse = float((((pred - Yva) ** 2).sum(1)).mean())
            if best is None or mse < best[0]:
                best = (mse, lam)
        print("  %-22s n_tr=%5d n_va=%5d  dim=%d  best_lam=%-8g  mse8=%.4f"
              % (tag, len(Xtr), len(Xva), Ftr.shape[1], best[1], best[0]))
        return best[0]

    run(tr, va, "原有 split")

    # ---- 口径 2：按 session 切分（val 只留未见过的 session）----
    sids = sorted({x["sid"] for x in tr})
    if len(sids) >= 2:
        hold = sids[-1]
        tr2 = [x for x in tr if x["sid"] != hold]
        va2 = [x for x in va if x["sid"] == hold]
        if va2 and tr2:
            run(tr2, va2, "按 session 切(%s)" % hold)

    # ---- 对照基线（同一批 val）----
    Yva, okva = labels(va)
    copy = float((((Yva[:, :2] - Yva) ** 2).sum(1)).mean())
    print()
    print("  同批 val 的 copy 基线 = %.4f" % copy)
    print("  模型实测（w6_far）    = 1.8496")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
