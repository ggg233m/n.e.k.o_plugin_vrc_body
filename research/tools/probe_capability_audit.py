# -*- coding: utf-8 -*-
"""只读：causal 数值基线 vs 视觉基线，按 world_key 做 LWO。

回答四个问题：
  Q1 causal 数值特征在 LWO 下能到多少（速度 / 转向意图）
  Q2 加视觉（ViT pooled）能提升多少
  Q3 LWO 10 折是否可执行，方差多大
  Q4 当前数据能支撑哪类任务（自然驾驶风格 vs 目标驱动探索）
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

V4 = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v4causal")
V3 = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
S = 8


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def feats(v):
    o = np.zeros((len(v), 8))
    for i, x in enumerate(v):
        o[i, :2] = x["wp"]
        o[i, 2:5] = x.get("topo") or (0, 0, 0)
        o[i, 5:7] = x.get("body") or (0, 0)
        o[i, 7] = float(x.get("mode", 0))
    return o


def m_of(p, a, m):
    sq = ((p - a) ** 2).sum(-1)
    return float((sq * m).sum() / max(m.sum(), 1.0))


def lwo(ds_path, tag):
    s = T.build_index(ds_path, None)
    sids = sorted({x["sid"] for x in s})
    fold_spd, fold_cp, fold_omg, fold_omg0 = [], [], [], []
    for hold in sids:
        tr = [x for x in s if x["sid"] != hold]
        va = [x for x in s if x["sid"] == hold]
        if len(tr) < 200 or len(va) < 20:
            continue
        Xtr, Xva = feats(tr), feats(va)
        Atr = np.stack([np.asarray(x["act"])[:S] for x in tr])
        Ava = np.stack([np.asarray(x["act"])[:S] for x in va])
        Mtr = np.stack([np.asarray(x["mask"])[:S] for x in tr])
        Mva = np.stack([np.asarray(x["mask"])[:S] for x in va])
        ok = Mtr.sum(-1) == S
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr, Fva = quad((Xtr[ok]-mu)/sd), quad((Xva-mu)/sd)
        best = (1e18, None)
        for lam in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
            Y = Atr[ok].reshape(int(ok.sum()), -1)
            W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Y)
            p = (Fva@W).reshape(len(Fva), S, -1)
            e = m_of(p[..., :2], Ava[..., :2], Mva)
            if e < best[0]: best = (e, p)
        p = best[1]
        cp = np.repeat(Ava[:, :1, :2], S, axis=1)
        fold_spd.append(m_of(p[..., :2], Ava[..., :2], Mva))
        fold_cp.append(m_of(cp, Ava[..., :2], Mva))
        fold_omg.append(m_of(p[..., 2:3], Ava[..., 2:3], Mva))
        fold_omg0.append(m_of(np.zeros_like(Ava[..., 2:3]), Ava[..., 2:3], Mva))
    a = np.array(fold_spd); c = np.array(fold_cp)
    o = np.array(fold_omg); z = np.array(fold_omg0)
    print("  %-26s spd=%.4f±%.4f copy=%.4f±%.4f  omg=%.4f zero=%.4f (可解释 %.1f%%)"
          % (tag, a.mean(), a.std(), c.mean(), c.std(), o.mean(), z.mean(),
             100*(1-o.mean()/z.mean())))
    return a, c, o, z


print("=" * 100)
print("  LWO 10 折：causal（去未来）vs 旧 far（含未来 wp）")
print("=" * 100)
print("  %-26s %s" % ("数据集", "结果"))
lwo(V4, "v4causal (wp=0)")
lwo(V3, "v3far (wp=未来4m)")
