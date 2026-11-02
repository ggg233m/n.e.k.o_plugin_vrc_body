# -*- coding: utf-8 -*-
"""用缓存的视觉特征检查：视觉增益是不是「泄漏」？

**为什么要查**：probe_variance_decomp 得到
  body+wp         mse8 = 2.8652
  body+wp+img     mse8 = 1.2672   （视觉让误差降 56%）
  body+wp+shuffle mse8 = 3.0943   （换成别的场景的图，比不给图还差）

最后一行说明视觉特征确实带场景信息、不是噪声。但还有一个 confound：
样本在时间上相邻、同一段走廊长得一样，随机切 train/test 会让「同一地点」
同时出现在两边 —— 那测到的可能只是「我在哪」的记忆，不是「怎么走」。

**做法**：两组切分，比同一个信息集的表现
  * temporal —— 每场内部按时间前 80% 训练、后 20% 测试（原口径）
  * session  —— 留出整个世界（10 场里挑 2 场只做测试）

若 session 切分下视觉增益大幅缩水 → 视觉学到的是场景记忆，换世界就失效。
若仍然显著 → 视觉信息是真的可迁移的。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_variance_split.py \
        --feats .venv_xpu/runs/variance_decomp/feats.npz
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


def mse8(pred, act, mask, steps=8):
    err = ((pred[:, :steps, :] - act[:, :steps, :]) ** 2).sum(-1)
    m = mask[:, :steps]
    return float((err * m).sum() / max(m.sum(), 1.0))


def run_split(name, tr, te, F, B, W, A, M, pca, alpha=1.0):
    """在给定切分上跑三组信息集，返回 mse8。"""
    mu = F[tr].mean(0)
    _, _, Vt = np.linalg.svd(F[tr] - mu, full_matrices=False)
    P = Vt[:pca].T
    Fp = (F - mu) @ P
    sd = Fp[tr].std(0) + 1e-8
    Fp = Fp / sd

    n = len(F)
    Y = A.reshape(n, -1)
    res = {}
    sets = {
        "body": B,
        "body+wp": np.concatenate([B, W], 1),
        "body+wp+img": np.concatenate([B, W, Fp], 1),
    }
    for k, X in sets.items():
        Wt, m, s = ridge_fit(X[tr], Y[tr], alpha=alpha)
        p = ridge_pred(Wt, m, s, X[te]).reshape(-1, 12, 2)
        res[k] = mse8(p, A[te], M[te])
    res["mir"] = res["body+wp"] - res["body+wp+img"]
    res["n_train"], res["n_test"] = int(len(tr)), int(len(te))
    print("  %-10s train=%-5d test=%-5d  body=%.4f  +wp=%.4f  +img=%.4f  视觉增益=%.4f"
          % (name, len(tr), len(te), res["body"], res["body+wp"],
             res["body+wp+img"], res["mir"]), flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--only-sessions", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    d = np.load(a.feats, allow_pickle=True)
    F, B, W, A, M = d["real"], d["body"], d["wp"], d["act"], d["mask"]
    sid = d["sid"]
    n = len(F)
    out = {"n": int(n), "pca": a.pca, "splits": {}}

    # ---- 1. temporal：每场内部前 80% / 后 20%（reproduce 原口径）----
    idx = np.arange(n)
    tr_t, te_t = [], []
    for s in np.unique(sid):
        sel = idx[sid == s]
        if len(sel) < 10:
            continue
        cut = int(len(sel) * 0.8)
        tr_t.extend(sel[:cut])
        te_t.extend(sel[cut:])
    out["splits"]["temporal"] = run_split(
        "temporal", np.array(tr_t), np.array(te_t), F, B, W, A, M, a.pca)

    # ---- 2. session：留出整个世界 ----
    uniq = sorted(np.unique(sid))
    held = a.only_sessions if a.only_sessions else uniq[-2:]
    te_s = idx[np.isin(sid, held)]
    tr_s = idx[~np.isin(sid, held)]
    print("[held-out worlds] %s" % list(held), flush=True)
    out["held_out"] = list(held)
    out["splits"]["session"] = run_split(
        "session", tr_s, te_s, F, B, W, A, M, a.pca)

    # ---- 3. 反向：用被留出的那两场训练、在其余场测试（排除样本量因素）----
    out["splits"]["session_rev"] = run_split(
        "session_rev", te_s, tr_s, F, B, W, A, M, a.pca)

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "variance_split.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "variance_split.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
