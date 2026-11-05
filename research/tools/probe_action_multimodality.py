# -*- coding: utf-8 -*-
"""判决测量：在「几乎相同的状态」下，人的动作是不是多峰的？

为什么这样测（2026-11-02）：换流匹配头部的前提是「同一 (图像, wp, body) 下有多个合法动作」。
这件事**是数据属性，不是模型属性** —— 用 MC dropout 看模型输出散不散只能反映模型不确定性，
反映不了标签里有没有多峰。所以这里直接在冻结特征上做近邻分析：

  1. 用 [PCA(冻结 ViT 特征), wp, body] 构造状态空间；
  2. 对每个样本找 K 个最近邻（状态几乎相同）；
  3. 看邻居们的**动作方向**是否一致：
       R = |mean(单位向量)|，R→1 表示方向一致（单峰），R→0 表示方向发散（多峰或纯噪声）
  4. **必须和打乱对照比**：把邻居的动作随机置换后重算 R，得到「无关系基线」。
     只有观测 R 显著高于打乱 R，才能说「状态确实决定方向」。
  5. 再对邻居方向做 2-means：两簇都 >= 20% 且簇心夹角 > 30 度，记为一个多峰邻域。

判据（写进 §6.2 第 3 条）：
  * 多峰邻域占比 < 10% 且 R 明显高于打乱 => **单峰**，流匹配整体取消；
  * 多峰邻域占比 > 30%            => 确实多峰，换头才有依据。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_action_multimodality.py \
        --feats .venv_xpu/runs/variance_decomp/feats.npz --out .venv_xpu/runs/multimodal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def two_means_dirs(D, iters=20, seed=0):
    """对单位方向的二维点做 2-means（余弦距离）。返回 (标签, 两个簇心)。"""
    rng = np.random.default_rng(seed)
    i, j = rng.choice(len(D), 2, replace=False)
    C = np.stack([D[i], D[j]])
    for _ in range(iters):
        sim = D @ C.T
        lab = np.argmax(sim, axis=1)
        for k in range(2):
            m = lab == k
            if m.any():
                c = D[m].sum(0)
                C[k] = c / max(np.linalg.norm(c), 1e-9)
    return lab, C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--knn", type=int, default=25)
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args(argv)

    d = np.load(a.feats, allow_pickle=True)
    F, B, W, A, M = d["real"], d["body"], d["wp"], d["act"], d["mask"]
    n = len(F)
    print("[data] n=%d" % n, flush=True)

    # 只用 step0 有效、且真的在动（move）的样本：idle 的动作恒 0，放进来会稀释
    spd0 = np.linalg.norm(A[:, 0, :], axis=-1)
    sel = (M[:, 0] > 0) & (spd0 > 0.1)
    print("[data] 移动样本 %d" % sel.sum(), flush=True)
    F, B, W, A, M = F[sel], B[sel], W[sel], A[sel], M[sel]
    n = len(F)

    # ---- 状态空间 ----
    mu = F.mean(0)
    _, _, Vt = np.linalg.svd(F - mu, full_matrices=False)
    Z = (F - mu) @ Vt[:a.pca].T
    Z = Z / (Z.std(0) + 1e-8)
    # 特征、wp、body 各自标准化后等权拼接
    S = np.concatenate([Z,
                        (W - W.mean(0)) / (W.std(0) + 1e-8),
                        (B - B.mean(0)) / (B.std(0) + 1e-8)], axis=1)

    dirs = unit(A[:, 0, :])          # (n,2) 单位方向
    glob_R = float(np.linalg.norm(dirs.mean(0)))
    print("[stat] 全局方向集中度 R=%.3f（1=所有人朝同一边）" % glob_R, flush=True)

    rng = np.random.default_rng(0)
    R_obs, R_shuf, multi = [], [], []
    k = min(a.knn, n - 1)
    for start in range(0, n, a.batch):
        stop = min(start + a.batch, n)
        dist = ((S[start:stop, None, :] - S[None, :, :]) ** 2).sum(-1)
        for r in range(stop - start):
            row = dist[r]
            idx = np.argpartition(row, k + 1)[:k + 1]
            idx = idx[idx != (start + r)][:k]
            D = dirs[idx]
            R_obs.append(float(np.linalg.norm(D.mean(0))))
            perm = rng.permutation(len(dirs))[:k]
            R_shuf.append(float(np.linalg.norm(dirs[perm].mean(0))))
            lab, C = two_means_dirs(D)
            share = min((lab == 0).mean(), (lab == 1).mean())
            ang = float(np.degrees(np.arccos(np.clip(C[0] @ C[1], -1, 1))))
            multi.append(1 if (share >= 0.2 and ang >= 30.0) else 0)
        if (start // a.batch) % 2 == 0:
            print("  [knn] %d/%d" % (stop, n), flush=True)

    R_obs = np.asarray(R_obs); R_shuf = np.asarray(R_shuf); multi = np.asarray(multi)
    out = dict(
        n=int(n), pca=a.pca, knn=int(k),
        global_R=glob_R,
        R_obs=dict(mean=float(R_obs.mean()), p05=float(np.percentile(R_obs, 5)),
                   p50=float(np.percentile(R_obs, 50))),
        R_shuffle=dict(mean=float(R_shuf.mean()), p95=float(np.percentile(R_shuf, 95))),
        multimodal_frac=float(multi.mean()),
        verdict=None,
    )
    if out["multimodal_frac"] < 0.10:
        out["verdict"] = "single_mode：流匹配整体取消"
    elif out["multimodal_frac"] > 0.30:
        out["verdict"] = "multimodal：换头有依据"
    else:
        out["verdict"] = "灰区：需更换状态度量或加大 knn 再判"

    print()
    print("  近邻状态下的方向集中度 R_obs = %.3f  (p50 %.3f)"
          % (out["R_obs"]["mean"], out["R_obs"]["p50"]))
    print("  打乱对照         R_shuf = %.3f" % out["R_shuffle"]["mean"])
    print("  多峰邻域占比     = %.1f%%" % (100 * out["multimodal_frac"]))
    print("  判决：%s" % out["verdict"])

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "multimodality.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "multimodality.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
