# -*- coding: utf-8 -*-
"""omega 上限：8 个数值特征能预测 ω 到什么程度？

背景：训练日志里 ω 误差长期停在 ~2.2（mse），但没人知道这个数
是「已经到上限」还是「配置浪费了 74% 的样本」。

对照：同口径下速度两维的上限（ridge / MLP）。
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


def feats(v, drop_wp=True):
    o = np.zeros((len(v), 8))
    for i, x in enumerate(v):
        if not drop_wp:
            o[i, :2] = x["wp"]
        o[i, 2:5] = x.get("topo") or (0, 0, 0)
        o[i, 5:7] = x.get("body") or (0, 0)
        o[i, 7] = float(x.get("mode", 0))
    return o


def masked(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1.0))


alls = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
C.relabel(alls)
sids = sorted({x["sid"] for x in alls})
print("[data] %d 样本 / %d 世界" % (len(alls), len(sids)))

results = {}
for tgt in ("speed", "omega"):
    Pr, Ar, Mr, Cr = [], [], [], []
    for hold in sids:
        trv = [x for x in alls if x["sid"] != hold]
        vav = [x for x in alls if x["sid"] == hold]
        Xtr, Xva = feats(trv), feats(vav)
        if tgt == "speed":
            Atr = np.stack([np.asarray(x["act"])[:S, :2] for x in trv])
            Ava = np.stack([np.asarray(x["act"])[:S, :2] for x in vav])
        else:
            Atr = np.stack([np.asarray(x["act"])[:S, 2:3] for x in trv])
            Ava = np.stack([np.asarray(x["act"])[:S, 2:3] for x in vav])
        Mtr = np.stack([np.asarray(x["mask"])[:S] for x in trv])
        Mva = np.stack([np.asarray(x["mask"])[:S] for x in vav])
        ok = Mtr.sum(-1) == S
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr, Fva = quad((Xtr[ok]-mu)/sd), quad((Xva-mu)/sd)
        Y = Atr[ok].reshape(int(ok.sum()), -1)
        best = (1e18, None)
        for lam in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
            W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Y)
            p = (Fva@W).reshape(len(Fva), S, -1)
            v = masked(p, Ava, Mva)
            if v < best[0]: best = (v, p)
        Pr.append(best[1]); Ar.append(Ava); Mr.append(Mva)
        Cr.extend([x["cls2"] for x in vav])
    P = np.concatenate(Pr); Aa = np.concatenate(Ar); Mk = np.concatenate(Mr); Cz = np.array(Cr)
    zero = masked(np.zeros_like(Aa), Aa, Mk)
    copy_pred = np.repeat(Aa[:, :1], S, axis=1)
    copy = masked(copy_pred, Aa, Mk)
    print()
    print("===== 目标: %s （LWO 10 折，前 %d 步）=====" % (tgt, S))
    print("  zero=%.4f  copy=%.4f  ridge=%.4f" % (zero, copy, masked(P, Aa, Mk)))
    print("  %-16s %5s %9s %9s %9s" % ("class", "n", "ridge", "copy", "zero"))
    for c in C.CLASSES + ("ALL",):
        sel = np.ones(len(Cz), bool) if c == "ALL" else (Cz == c)
        if sel.sum() == 0: continue
        zp = np.zeros_like(Aa)
        print("  %-16s %5d %9.4f %9.4f %9.4f" % (
            c, int(sel.sum()), masked(P[sel], Aa[sel], Mk[sel]),
            masked(copy_pred[sel], Aa[sel], Mk[sel]), masked(zp[sel], Aa[sel], Mk[sel])))
    results[tgt] = (zero, copy, masked(P, Aa, Mk))

print()
print("[对比] ω 的可解释比例")
z, c, r = results["omega"]
print("  zero=%.4f -> ridge=%.4f，解释掉 %.1f%%" % (z, r, 100*(1-r/z)))
