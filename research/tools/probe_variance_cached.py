# -*- coding: utf-8 -*-
"""用缓存好的视觉特征做方差分解（不重跑 ViT，秒级）。

前置：eval_vision_ablation / probe_variance_decomp 先产出 feats.npz。
本脚本回答的是「**哪一部分视觉信息有用**」，这是上线决策真正需要的：

* real      —— 真实图像的 ViT 特征
* black     —— 全黑图（视觉无信息）
* shuffle   —— 别的样本的图像（场景错但仍然自然）

如果 real 明显好于 shuffle，说明模型能从图像读出「当前场景是什么」；
如果 shuffle 也明显好于 black，说明它读的是「有没有画面」这种低级统计量，
而不是场景内容 —— 那是伪相关，换个世界就失效。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_variance_cached.py \
        --feats .venv_xpu/runs/variance_decomp/feats.npz --out .venv_xpu/runs/variance_decomp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ridge_fit(X, Y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1)
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, Xs.T @ Y), mu, sd


def ridge_pred(W, mu, sd, X):
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1) @ W


def masked_mse_np(pred, act, mask, steps=8):
    err = ((pred[:, :steps, :] - act[:, :steps, :]) ** 2).sum(-1)
    m = mask[:, :steps]
    return float((err * m).sum() / max(m.sum(), 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args(argv)

    d = np.load(a.feats, allow_pickle=True)
    F, B, W, A, M = d["real"], d["body"], d["wp"], d["act"], d["mask"]
    n = len(F)
    ntr = int(n * 0.8)
    print("[data] n=%d (train %d / test %d)" % (n, ntr, n - ntr), flush=True)

    # PCA 只在训练侧拟合
    mu = F[:ntr].mean(0)
    _, _, Vt = np.linalg.svd(F[:ntr] - mu, full_matrices=False)
    P = Vt[:a.pca].T
    Fp = (F - mu) @ P
    sd = Fp[:ntr].std(0) + 1e-8
    Fp = Fp / sd

    Y = A.reshape(n, -1)
    Mte, Ate = M[ntr:], A[ntr:]

    def score(X):
        if X.shape[1] == 0:
            P_te = np.zeros((n - ntr, Y.shape[1]))
        else:
            Wt, m, s = ridge_fit(X[:ntr], Y[:ntr], alpha=a.alpha)
            P_te = ridge_pred(Wt, m, s, X[ntr:])
        return masked_mse_np(P_te.reshape(-1, 12, 2), Ate, Mte, steps=8)

    sets = {
        "zero": np.zeros((n, 0)),
        "body": B,
        "wp_only": W,
        "img_only": Fp,
        "body+wp": np.concatenate([B, W], 1),
        "body+wp+img": np.concatenate([B, W, Fp], 1),
    }
    res = {}
    for k, X in sets.items():
        res[k] = score(X)
        print("  %-14s mse8=%.4f" % (k, res[k]), flush=True)

    copy = np.repeat(B[ntr:][:, None, :], 12, axis=1)
    res["copy_current"] = masked_mse_np(copy, Ate, Mte, steps=8)
    print("  %-14s mse8=%.4f" % ("copy_current", res["copy_current"]), flush=True)

    out = dict(n=n, pca=a.pca, alpha=a.alpha, mse8=res,
               vision_gain=res["body+wp"] - res["body+wp+img"],
               img_alone_vs_zero=res["zero"] - res["img_only"])
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "variance_cached.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print("  视觉增益 (body+wp -> +img) = %.4f" % out["vision_gain"])
    print("  [done] %s" % (o / "variance_cached.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
