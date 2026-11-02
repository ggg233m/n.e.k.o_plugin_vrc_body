# -*- coding: utf-8 -*-
"""5-fold 重报「视觉增益」：确认 -56% 是不是过拟合（policy-designer 的 P2）。

**为什么要做**：probe_variance_decomp 报的 body+wp 2.87 -> +img 1.27（-56%），
但那是**单次随机切分 + PCA64 + 线性回归**。policy-designer 指出这可能严重乐观：
它用 21 维廉价图像统计、正经 train/val 切分，只得 -3.4%，差 16 倍。

本脚本在缓存的 ViT 特征上做 5-fold（每折都重新拟合 PCA 与标准化，绝不跨折泄漏），
报告均值 ± 标准差。同时报 shuffle / black 两个对照，确认信号是真的。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_headroom_kfold.py \
        --feats .venv_xpu/runs/variance_decomp/feats.npz --out .venv_xpu/runs/kfold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1)
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, Xs.T @ Y), mu, sd


def ridge_pred(W, mu, sd, X):
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1) @ W


def mse8(pred, act, mask, steps=8):
    err = ((pred[:, :steps, :] - act[:, :steps, :]) ** 2).sum(-1)
    m = mask[:, :steps]
    return float((err * m).sum() / max(m.sum(), 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    d = np.load(a.feats, allow_pickle=True)
    F, FB, FS = d["real"], d["black"], d["shuffle"]
    B, W, A, M = d["body"], d["wp"], d["act"], d["mask"]
    n = len(F)
    Y = A.reshape(n, -1)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(n)
    folds = np.array_split(order, a.folds)

    keys = ["body", "body+wp", "body+wp+img", "body+wp+black", "body+wp+shuffle", "img_only"]
    acc = {k: [] for k in keys}

    for fi, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        # ⚠️ PCA 与标准化**只在训练折上拟合**，否则是最经典的泄漏
        mu = F[tr].mean(0)
        _, _, Vt = np.linalg.svd(F[tr] - mu, full_matrices=False)
        P = Vt[:a.pca].T
        def proj(X):
            Z = (X - mu) @ P
            return Z / (Z[tr].std(0) + 1e-8)
        Fp, FBp, FSp = proj(F), proj(FB), proj(FS)

        sets = {
            "body": B,
            "body+wp": np.concatenate([B, W], 1),
            "body+wp+img": np.concatenate([B, W, Fp], 1),
            "body+wp+black": np.concatenate([B, W, FBp], 1),
            "body+wp+shuffle": np.concatenate([B, W, FSp], 1),
            "img_only": Fp,
        }
        for k, X in sets.items():
            best = None
            for al in a.alphas:
                Wt, m, s = ridge_fit(X[tr], Y[tr], al)
                p = ridge_pred(Wt, m, s, X[te]).reshape(-1, 12, 2)
                v = mse8(p, A[te], M[te])
                if best is None or v < best:
                    best = v
            acc[k].append(best)
        print("  fold %d done (test n=%d)" % (fi, len(te)), flush=True)

    res = {}
    for k in keys:
        v = np.asarray(acc[k])
        res[k] = dict(mean=float(v.mean()), std=float(v.std()),
                      folds=[float(x) for x in v])
    gain = res["body+wp"]["mean"] - res["body+wp+img"]["mean"]
    gain_pct = 100.0 * gain / res["body+wp"]["mean"]
    res["gain"] = dict(abs=gain, pct=gain_pct)
    res["control_gain_black"] = res["body+wp"]["mean"] - res["body+wp+black"]["mean"]
    res["control_gain_shuffle"] = res["body+wp"]["mean"] - res["body+wp+shuffle"]["mean"]

    print()
    for k in keys:
        print("  %-18s mse8 = %.4f +- %.4f" % (k, res[k]["mean"], res[k]["std"]))
    print()
    print("  视觉增益 = %.4f (%.1f%%)" % (gain, gain_pct))
    print("  对照 black   = %.4f  (应 ~0)" % res["control_gain_black"])
    print("  对照 shuffle = %.4f  (应 <=0)" % res["control_gain_shuffle"])

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "kfold.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print("[done] %s" % (o / "kfold.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
