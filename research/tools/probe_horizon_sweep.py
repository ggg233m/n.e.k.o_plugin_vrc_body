# -*- coding: utf-8 -*-
"""官方接口 vs 当前实现：200ms 动作块是否比 1.2s 更可学？

官方（README §3）：一次调用 = 一个 200ms 决策 = 6 步 x 33ms + 一个鼠标位移。
当前实现：12 步 x 100ms = 1.2s 连续 (vx,vz,omega) 回归。

本探针做 horizon sweep，同一批 LWO 10 折、同一组特征：
  H=1 步(0.1s) / H=2 步(0.2s=官方) / H=4 步(0.4s) / H=8 步(0.8s) / H=12 步(1.2s=现状)
每个 horizon 同时报：
  (a) 连续回归：omega 与 speed 的 MSE / 相对 zero 的可解释比例
  (b) 离散分类：转向意图(左/直/右) 与 移动状态(停/走) 的 accuracy
如果 200ms 明显好于 1.2s，说明当前实现选错了时间尺度。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

ROOT = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
THR_W = 0.3
THR_V = 0.1


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def feats(v):
    o = np.zeros((len(v), 6))
    for i, x in enumerate(v):
        tp = x.get("topo") or (0, 0, 0)
        bd = x.get("body") or (0, 0)
        o[i, :3] = tp
        o[i, 3:5] = bd
        o[i, 5] = float(x.get("mode", 0))
    return o


def mse(p, a, m):
    sq = ((p - a) ** 2).sum(-1)
    return float((sq * m).sum() / max(m.sum(), 1.0))


alls = T.build_index(ROOT, None)
sids = sorted({x["sid"] for x in alls})
print("[data] %d 样本 / %d 世界" % (len(alls), len(sids)))

HORIZONS = [(1, "0.1s"), (2, "0.2s=官方"), (4, "0.4s"), (8, "0.8s"), (12, "1.2s=现状")]

print()
print("=" * 92)
print("  (a) 连续回归：omega —— 不同 horizon 的可解释比例")
print("=" * 92)
print("  %-12s %10s %10s %10s %10s" % ("horizon", "omega_mse", "zero", "可解释", "speed_mse"))

for H, label in HORIZONS:
    Ps, As, Ms = [], [], []
    for hold in sids:
        trv = [x for x in alls if x["sid"] != hold]
        vav = [x for x in alls if x["sid"] == hold]
        Xtr, Xva = feats(trv), feats(vav)
        Atr = np.stack([np.asarray(x["act"])[:H] for x in trv])
        Ava = np.stack([np.asarray(x["act"])[:H] for x in vav])
        Mtr = np.stack([np.asarray(x["mask"])[:H] for x in trv])
        Mva = np.stack([np.asarray(x["mask"])[:H] for x in vav])
        ok = Mtr.sum(-1) == H
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr, Fva = quad((Xtr[ok]-mu)/sd), quad((Xva-mu)/sd)
        Y = Atr[ok].reshape(int(ok.sum()), -1)
        best = (1e18, None)
        for lam in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
            W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Y)
            p = (Fva@W).reshape(len(Fva), H, -1)
            e = mse(p[..., 2:3], Ava[..., 2:3], Mva)
            if e < best[0]: best = (e, p)
        Ps.append(best[1]); As.append(Ava); Ms.append(Mva)
    Pp = np.concatenate(Ps); Aa = np.concatenate(As); Mk = np.concatenate(Ms)
    z = mse(np.zeros_like(Aa[..., 2:3]), Aa[..., 2:3], Mk)
    om = mse(Pp[..., 2:3], Aa[..., 2:3], Mk)
    sp = mse(Pp[..., :2], Aa[..., :2], Mk)
    print("  %-12s %10.4f %10.4f %9.1f%% %10.4f" %
          (label, om, z, 100*(1-om/z), sp))

print()
print("=" * 92)
print("  (b) 离散分类：转向意图 / 移动状态 —— 不同 horizon 的 accuracy")
print("=" * 92)
print("  %-12s %14s %14s %12s" % ("horizon", "转向意图acc", "多数类基线", "移动状态acc"))

for H, label in HORIZONS:
    accs, bases, mv_accs = [], [], []
    for hold in sids:
        tri = [i for i, x in enumerate(alls) if x["sid"] != hold]
        vai = [i for i, x in enumerate(alls) if x["sid"] == hold]
        if len(tri) < 200 or len(vai) < 20: continue
        Xtr = feats([alls[i] for i in tri]); Xva = feats([alls[i] for i in vai])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr, Fva = quad((Xtr-mu)/sd), quad((Xva-mu)/sd)
        # 转向意图：窗口内 omega 均值三分
        def wlabel(idx):
            out = []
            for i in idx:
                a = np.asarray(alls[i]["act"])[:H, 2]
                m = np.asarray(alls[i]["mask"])[:H] > 0
                out.append(float(a[m].mean()) if m.any() else 0.0)
            return np.array(out)
        wtr, wva = wlabel(tri), wlabel(vai)
        ytr = np.where(wtr > THR_W, 2, np.where(wtr < -THR_W, 0, 1))
        yva = np.where(wva > THR_W, 2, np.where(wva < -THR_W, 0, 1))
        Yh = np.eye(3)[ytr]
        best = (-1, None)
        for lam in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
            W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Yh)
            pr = (Fva@W).argmax(1)
            a1 = float((pr == yva).mean())
            if a1 > best[0]: best = (a1, pr)
        accs.append(best[0])
        bases.append(float(np.bincount(ytr, minlength=3).max()/len(ytr)))
        # 移动状态：窗口内最大速度 > THR_V
        def mlabel(idx):
            out = []
            for i in idx:
                a = np.asarray(alls[i]["act"])[:H, :2]
                m = np.asarray(alls[i]["mask"])[:H] > 0
                sp = np.linalg.norm(a, axis=1)
                out.append(1 if (sp[m] > THR_V).any() else 0)
            return np.array(out)
        mtr, mva = mlabel(tri), mlabel(vai)
        Yh2 = np.eye(2)[mtr]
        b2 = (-1,)
        for lam in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
            W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Yh2)
            pr = (Fva@W).argmax(1)
            a2 = float((pr == mva).mean())
            if a2 > b2[0]: b2 = (a2,)
        mv_accs.append(b2[0])
    a = np.array(accs); bb = np.array(bases); mv = np.array(mv_accs)
    print("  %-12s %13.4f %14.4f %11.4f" % (label, a.mean(), bb.mean(), mv.mean()))
