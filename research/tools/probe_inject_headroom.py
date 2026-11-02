# -*- coding: utf-8 -*-
"""特征注入的上限验证：把视觉特征**绕过 LM** 直接接进读出层，能不能拿到那 -56%？

**为什么要做**（2026-11-02）：ViT LoRA 已证伪（视觉依赖度 0.983 -> 0.994）。
policy-designer 的判断是「通路瓶颈」——112 个视觉 token 的信息必须经线性注意力
汇总到末位 token 才能到输出头，这条路太窄。

本脚本直接验证这个判断的**上限**：完全绕过 LM，只用
  [视觉特征池化 (64 维 PCA)] + [body, wp]
做线性读出，看能到什么水平。和之前 variance_decomp 的区别：

* 之前是把视觉特征当成「额外输入」加进同一个回归，测的是增量；
* 这里是**只靠视觉**（不加 body/wp）能不能预测动作 —— 如果单独就有解释力，
  说明特征本身携带动作信息，值得接进 head；
* 同时报告「视觉特征 → 动作」这个映射在**留出整个世界**时的表现，
  排除场景记忆。

输出三个数：img_only / body+wp / body+wp+img，各在 temporal 与 session 两种切分下。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_inject_headroom.py \
        --feats .venv_xpu/runs/variance_decomp/feats.npz --out .venv_xpu/runs/inject_headroom
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0],
                    help="扫一组正则强度，报告最好的那个（线性探针的上限）")
    a = ap.parse_args(argv)

    d = np.load(a.feats, allow_pickle=True)
    F, B, W, A, M = d["real"], d["body"], d["wp"], d["act"], d["mask"]
    FB = d["black"]
    sid = d["sid"]
    n = len(F)

    def build_splits():
        idx = np.arange(n)
        out = {}
        tr, te = [], []
        for s in np.unique(sid):
            sel = idx[sid == s]
            cut = int(len(sel) * 0.8)
            tr.extend(sel[:cut])
            te.extend(sel[cut:])
        out["temporal"] = (np.array(tr), np.array(te))
        uniq = sorted(np.unique(sid))
        held = uniq[-2:]
        te2 = idx[np.isin(sid, held)]
        tr2 = idx[~np.isin(sid, held)]
        out["session"] = (tr2, te2)
        return out

    splits = build_splits()
    res = {}
    for sname, (tr, te) in splits.items():
        mu = F[tr].mean(0)
        _, _, Vt = np.linalg.svd(F[tr] - mu, full_matrices=False)
        P = Vt[:a.pca].T
        Fp = (F - mu) @ P
        sd = Fp[tr].std(0) + 1e-8
        Fp = Fp / sd
        FBp = ((FB - mu) @ P) / sd

        Y = A.reshape(n, -1)
        sets = {
            "img_only": Fp,
            "body+wp": np.concatenate([B, W], 1),
            "body+wp+img": np.concatenate([B, W, Fp], 1),
            "body+wp+black": np.concatenate([B, W, FBp], 1),
        }
        res[sname] = {}
        for k, X in sets.items():
            best = None
            for al in a.alphas:
                Wt, m, s = ridge_fit(X[tr], Y[tr], alpha=al)
                p = ridge_pred(Wt, m, s, X[te]).reshape(-1, 12, 2)
                v = mse8(p, A[te], M[te])
                if best is None or v < best[0]:
                    best = (v, al)
            res[sname][k] = dict(mse8=best[0], alpha=best[1])
            print("  %-9s %-14s mse8=%.4f (alpha=%.1f)"
                  % (sname, k, best[0], best[1]), flush=True)
        copy = np.repeat(B[te][:, None, :], 12, axis=1)
        res[sname]["copy_current"] = dict(mse8=mse8(copy, A[te], M[te]), alpha=None)
        print("  %-9s %-14s mse8=%.4f" % (sname, "copy_current",
                                          res[sname]["copy_current"]["mse8"]), flush=True)

        # 上限：视觉单独能不能达到「body+wp」的水平
        res[sname]["headroom"] = dict(
            img_only_vs_body_wp=res[sname]["body+wp"]["mse8"] - res[sname]["img_only"]["mse8"],
            inject_gain=res[sname]["body+wp"]["mse8"] - res[sname]["body+wp+img"]["mse8"],
        )

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "inject_headroom.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "inject_headroom.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
