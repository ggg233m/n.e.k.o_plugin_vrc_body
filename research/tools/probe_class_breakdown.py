# -*- coding: utf-8 -*-
"""分动作类型拆开看：任务是真的不可学，还是被 63% 静止样本稀释了。

copy 基线在同一批样本上跟 ridge 比。如果「运动中」样本 ridge 也打不过 copy，
那这个任务定义就彻底没救；如果只有 idle 拖累，那说明指标口径错了、任务还活着。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)

def masked(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return (sq * mask), mask

def run(ds, drop_wp, S=8):
    s = T.build_index(Path(ds), None)
    sids = sorted({x["sid"] for x in s})
    P, Aa, Mk, C = [], [], [], []
    for hold in sids:
        tr = [x for x in s if x["sid"] != hold]
        va = [x for x in s if x["sid"] == hold]
        if not tr or not va: continue
        def feats(v):
            o = np.zeros((len(v), 8))
            for i, x in enumerate(v):
                if not drop_wp:
                    o[i, :2] = x["wp"]
                o[i, 2:5] = x.get("topo") or (0, 0, 0)
                o[i, 5:7] = x.get("body") or (0, 0)
                o[i, 7] = float(x.get("mode", 0))
            return o
        Xtr, Xva = feats(tr), feats(va)
        Atr = np.stack([np.asarray(x["act"])[:S, :2] for x in tr])
        Mtr = np.stack([np.asarray(x["mask"])[:S] for x in tr])
        Ava = np.stack([np.asarray(x["act"])[:S, :2] for x in va])
        Mva = np.stack([np.asarray(x["mask"])[:S] for x in va])
        ok = Mtr.sum(-1) == S
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr, Fva = quad((Xtr[ok]-mu)/sd), quad((Xva-mu)/sd)
        Y = Atr[ok].reshape(ok.sum(), -1)
        best = (1e18, None)
        for lam in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0):
            W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Y)
            m = masked((Fva@W).reshape(len(Fva), S, 2), Ava, Mva)[0]
            v = float(m.sum()/max(Mva.sum(),1))
            if v < best[0]: best = (v, (Fva@W).reshape(len(Fva), S, 2))
        P.append(best[1]); Aa.append(Ava); Mk.append(Mva)
        C.extend([x["cls"] for x in va])
    P, Aa, Mk = np.concatenate(P), np.concatenate(Aa), np.concatenate(Mk)
    C = np.array(C)
    copy = np.repeat(Aa[:, :1], S, axis=1)
    zero = np.zeros_like(Aa)
    rows = []
    for cl in ("idle", "move", "trans", "ALL"):
        sel = np.ones(len(C), bool) if cl == "ALL" else (C == cl)
        if sel.sum() == 0: continue
        def mse(pr):
            sq, mk = masked(pr[sel], Aa[sel], Mk[sel])
            return float(sq.sum()/max(mk.sum(),1)), float(mk.sum())
        r, nm = mse(P); c, _ = mse(copy); z, _ = mse(zero)
        rows.append((cl, int(sel.sum()), 100*nm/max(Mk.sum(),1), r, c, z))
    return rows

for tag, ds, dw in (("含 wp", "G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far", False),
                    ("去 wp", "G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far", True)):
    print("===== %s（LWO 10 折，前 8 步）=====" % tag)
    print("  %-6s %7s %8s %9s %9s %9s %8s" % ("class","n","mask%","ridge","copy","zero","vs copy"))
    for cl, n, mpct, r, c, z in run(ds, dw):
        print("  %-6s %7d %7.1f%% %9.4f %9.4f %9.4f %+7.1f%%" % (cl, n, mpct, r, c, z, 100*(r-c)/c))
    print()
