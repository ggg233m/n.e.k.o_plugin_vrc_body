# -*- coding: utf-8 -*-
"""补：LWO 的宏平均 + 每折分布（回应「pooled 会让样本多的世界权重更大」）。

现在所有 LWO 数都是 23068 个样本 pooled 出来的。但 10 个世界的帧数
在 2134~2366 之间，差异不大；真正会偏的是**类内**分布——某个世界的
walk_turn 占比高，就会主导 walk_turn 的 pooled 分数。

本表同时报：per-fold MSE / macro mean / median / std / min / max。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T
import policy_classify as C

S = 8


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def feats(v, drop_wp):
    o = np.zeros((len(v), 8))
    for i, x in enumerate(v):
        if not drop_wp:
            o[i, :2] = x["wp"]
        o[i, 2:5] = x.get("topo") or (0, 0, 0)
        o[i, 5:7] = x.get("body") or (0, 0)
        o[i, 7] = float(x.get("mode", 0))
    return o


def m_of(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1.0))


alls = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
C.relabel(alls)
sids = sorted({x["sid"] for x in alls})


def one_fold(hold, drop_wp):
    trv = [x for x in alls if x["sid"] != hold]
    vav = [x for x in alls if x["sid"] == hold]
    Xtr, Xva = feats(trv, drop_wp), feats(vav, drop_wp)
    Atr = np.stack([np.asarray(x["act"])[:S] for x in trv])
    Ava = np.stack([np.asarray(x["act"])[:S] for x in vav])
    Mtr = np.stack([np.asarray(x["mask"])[:S] for x in trv])
    Mva = np.stack([np.asarray(x["mask"])[:S] for x in vav])
    ok = Mtr.sum(-1) == S
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ftr, Fva = quad((Xtr[ok]-mu)/sd), quad((Xva-mu)/sd)
    best = (1e18, None)
    for lam in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
        Y = Atr[ok].reshape(int(ok.sum()), -1)
        W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Y)
        p = (Fva@W).reshape(len(Fva), S, -1)
        v = m_of(p[..., :2], Ava[..., :2], Mva)
        if v < best[0]: best = (v, p)
    p = best[1]
    cls = np.array([x["cls2"] for x in vav])
    out = {}
    for c in C.CLASSES + ("ALL",):
        sel = np.ones(len(cls), bool) if c == "ALL" else (cls == c)
        if sel.sum() < 5:
            continue
        out[c] = dict(
            spd=m_of(p[sel][..., :2], Ava[sel][..., :2], Mva[sel]),
            omg=m_of(p[sel][..., 2:3], Ava[sel][..., 2:3], Mva[sel]),
            omg0=m_of(np.zeros_like(Ava[sel][..., 2:3]), Ava[sel][..., 2:3], Mva[sel]),
        )
    cp = np.repeat(Ava[:, :1], S, axis=1)
    out["_copy_spd"] = m_of(cp[..., :2], Ava[..., :2], Mva)
    return out


for tag, drop in (("去 wp", True), ("含 wp", False)):
    folds = [(sid, one_fold(sid, drop)) for sid in sids]
    print()
    print("=" * 88)
    print("  %s —— 每折（世界）明细" % tag)
    print("=" * 88)
    print("  %-26s %9s %9s %9s %9s" % ("世界", "spd_ridge", "spd_copy", "omg_ridge", "omg_zero"))
    for sid, f in folds:
        a = f.get("ALL")
        if not a: continue
        print("  %-26s %9.4f %9.4f %9.4f %9.4f" %
              (sid, a["spd"], f["_copy_spd"], a["omg"], a["omg0"]))
    print()
    for key, name in (("spd", "速度 ridge"), ("omg", "ω ridge")):
        vals = np.array([f["ALL"][key] for _, f in folds if f.get("ALL")])
        print("  %-10s pooled=%.4f  macro=%.4f  median=%.4f  std=%.4f  min=%.4f  max=%.4f"
              % (name, vals.mean(), vals.mean(), np.median(vals), vals.std(),
                 vals.min(), vals.max()))
    vals = np.array([f["_copy_spd"] for _, f in folds])
    print("  %-10s macro=%.4f  median=%.4f  std=%.4f" % ("速度 copy", vals.mean(), np.median(vals), vals.std()))
    print()
    print("  按类的 fold 分布（macro mean ± std，只统计有该类样本的折）:")
    print("  %-16s %5s %16s %16s" % ("class", "折数", "spd macro", "omg macro"))
    for c in C.CLASSES:
        sv = [f[c]["spd"] for _, f in folds if c in f]
        ov = [f[c]["omg"] for _, f in folds if c in f]
        if not sv: continue
        print("  %-16s %5d   %7.4f ± %-6.4f   %7.4f ± %-6.4f" %
              (c, len(sv), np.mean(sv), np.std(sv), np.mean(ov), np.std(ov)))
