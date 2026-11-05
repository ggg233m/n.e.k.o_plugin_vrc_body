# -*- coding: utf-8 -*-
"""对照组：把「视觉差」和「子集小 + 1024 维」两个因素分开。

probe_visual_class_lwo 只用 260/折（2340 训练样本）跑 1024 维 ridge，
这个配置本身就该被强正则压死。本探针在**完全相同的 260/折子集**上重跑
纯数值特征（去 wp），若数值也掉到 3.x，说明是子集/维度问题；
若数值仍是 2.8，说明视觉特征确实更差。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

S = 8

def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)

def mse_of(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1e-9)), float(mask.sum())

def feats(v, drop_wp=True):
    o = np.zeros((len(v), 8))
    for i, x in enumerate(v):
        if not drop_wp:
            o[i, :2] = x["wp"]
        o[i, 2:5] = x.get("topo") or (0, 0, 0)
        o[i, 5:7] = x.get("body") or (0, 0)
        o[i, 7] = float(x.get("mode", 0))
    return o

alls = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
sids = sorted({x["sid"] for x in alls})
PER_FOLD = 260

# 复刻 probe_visual_class_lwo 的选样：对整个 session 做等距抽样
sub = {}
for sid in sids:
    va = [x for x in alls if x["sid"] == sid]
    if len(va) > PER_FOLD:
        step = len(va) / PER_FOLD
        va = [va[int(i * step)] for i in range(PER_FOLD)]
    sub[sid] = va

for tag, dw in (("去 wp", True), ("含 wp", False)):
    Pr, Ar, Mr, Cr = [], [], [], []
    for hold in sids:
        trv, vav = [], sub[hold]
        for s2 in sids:
            if s2 != hold:
                trv.extend(sub[s2])
        Xtr, Xva = feats(trv, dw), feats(vav, dw)
        Atr = np.stack([np.asarray(x["act"])[:S, :2] for x in trv])
        Mtr = np.stack([np.asarray(x["mask"])[:S] for x in trv])
        Ava = np.stack([np.asarray(x["act"])[:S, :2] for x in vav])
        Mva = np.stack([np.asarray(x["mask"])[:S] for x in vav])
        ok = Mtr.sum(-1) == S
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr, Fva = quad((Xtr[ok]-mu)/sd), quad((Xva-mu)/sd)
        Y = Atr[ok].reshape(int(ok.sum()), -1)
        best = (1e18, None)
        for lam in (1e-6,1e-4,1e-2,1.0,10.0,100.0,1000.0):
            W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Y)
            p = (Fva@W).reshape(len(Fva), S, 2)
            v = mse_of(p, Ava, Mva)[0]
            if v < best[0]: best = (v, p)
        Pr.append(best[1]); Ar.append(Ava); Mr.append(Mva); Cr.extend([x["cls"] for x in vav])
    P = np.concatenate(Pr); Aa = np.concatenate(Ar); Mk = np.concatenate(Mr); C = np.array(Cr)
    copy = np.repeat(Aa[:, :1], S, axis=1)
    print("===== %s（同 260/折 子集，LWO）=====" % tag)
    print("  %-6s %7s %8s %9s %9s %8s" % ("class","n","mask%","ridge","copy","vs copy"))
    for cl in ("idle","move","trans","ALL"):
        sel = np.ones(len(C), bool) if cl=="ALL" else (C==cl)
        if sel.sum()==0: continue
        m,nm = mse_of(P[sel], Aa[sel], Mk[sel]); c,_ = mse_of(copy[sel], Aa[sel], Mk[sel])
        rel = "    n/a" if c < 1e-6 else "%+7.1f%%" % (100*(m-c)/c)
        print("  %-6s %7d %7.1f%% %9.4f %9.4f %s" % (cl, int(sel.sum()), 100*nm/Mk.sum(), m, c, rel))
    print()
