# -*- coding: utf-8 -*-
"""从已缓存的 (H, NX, A, M, SID) 直接算四臂 LWO，不再走模型前向。

为什么要另写：probe_causal_vision.py 用 `*>` 重定向，PowerShell 会缓冲到进程结束，
日志在跑完前一直是 0 字节。特征已经缓存在 .tmp/feat_causal_lwo.npz，直接读即可。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")

CACHE = Path(".tmp/feat_causal_lwo.npz")
z = np.load(CACHE, allow_pickle=True)
H, SID, NX, A, M = z["H"], z["SID"], z["NX"], z["A"], z["M"]
S = 8
print("[data] H=%s  NX=%s  A=%s  世界数=%d" % (H.shape, NX.shape, A.shape, len(set(SID.tolist()))))


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def m_of(p, a, m):
    sq = ((p - a) ** 2).sum(-1)
    return float((sq * m).sum() / max(m.sum(), 1.0))


def run(X, tag):
    fold, foldc = [], []
    for hold in sorted(set(SID.tolist())):
        tr = SID != hold
        va = SID == hold
        if tr.sum() < 200 or va.sum() < 20:
            continue
        ok = M[tr][:, :S].sum(-1) == S
        Xt = X[tr][ok]
        Y = A[tr][ok][:, :S, :2].reshape(int(ok.sum()), -1)
        Xv = X[va]
        Av = A[va][:, :S, :2]
        Mv = M[va][:, :S]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        Xt = (Xt - mu) / sd
        Xv = (Xv - mu) / sd
        best = (1e18, None)
        for lam in (1.0, 10.0, 100.0, 1000.0, 1e4):
            W = np.linalg.solve(Xt.T@Xt + lam*np.eye(Xt.shape[1]), Xt.T@Y)
            p = (Xv@W).reshape(len(Xv), S, 2)
            e = m_of(p, Av, Mv)
            if e < best[0]: best = (e, p)
        fold.append(best[0])
        foldc.append(m_of(np.repeat(Av[:, :1], S, axis=1), Av, Mv))
    a = np.array(fold); c = np.array(foldc)
    print("  %-38s dim=%-5d spd=%.4f±%.4f  vs copy %+.1f%%"
          % (tag, X.shape[1], a.mean(), a.std(), 100*(a.mean()/c.mean()-1)))


mu0, sd0 = NX.mean(0), NX.std(0) + 1e-9
mu1, sd1 = H.mean(0), H.std(0) + 1e-9
NXs = (NX - mu0) / sd0
Hs = (H - mu1) / sd1

print()
print("=" * 92)
print("  LWO 10 折（240/折，前 %d 步速度），a 因果标签" % S)
print("=" * 92)
run(quad(NXs), "A 仅因果数值（44维二次）")
run(Hs, "B 仅图像（1024维 ViT pooled）")
run(np.concatenate([Hs, NXs], axis=1), "C 图像+数值（1030维线性）")
run(np.concatenate([Hs, quad(NXs)], axis=1), "D 图像+数值二次（1068维）")
run(np.concatenate([Hs, NXs, quad(NXs)], axis=1), "E 全拼（1074维）")
