# -*- coding: utf-8 -*-
"""omega 上限：加入 HMD yaw rate 与多帧 omega 历史后再测。

回应评审（2026-11-05）：omega 不可预测不能从当前 8 维数值 + ridge 推出。
HMD yaw rate、多帧 omega 历史、目标方向都可能补上缺失变量。
本探针加前两项（数据现成），目标方向没有。

对照口径固定：同一批 LWO 10 折、前 8 步、只算第 3 维 omega。
  A 含wp 8维 / B 去wp 6维 / C = B + yaw rate / D = C + omega 历史 / E 只用 omega 历史
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T
import policy_classify as C

S = 8
ROOT = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
# ⚠️ hmd_frames.jsonl 在**录制根目录**，不在数据集目录。第一版写错了 ROOT，
# 导致 load_yaw_rate() 返回 0 场，C/D 两组退化成和 B/E 完全一样（已复现验证）。
REC = Path("G:/vrc_recording")


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def m_of(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1.0))


def load_yaw_rate():
    out = {}
    for sess in sorted(p for p in REC.iterdir() if p.is_dir() and (p / "hmd_frames.jsonl").exists()):
        rows = [json.loads(l) for l in (sess / "hmd_frames.jsonl").open("r", encoding="utf-8")]
        if len(rows) < 10:
            continue
        t = np.array([r["t"] for r in rows], dtype=np.float64)
        q = np.array([r["hmd"]["rotation_xyzw"] for r in rows], dtype=np.float64)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
        yaw = np.unwrap(yaw)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.gradient(yaw, t)
        rate = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
        out[sess.name] = (t, rate)
    return out


def zoh(t, v, tq):
    if len(t) == 0:
        return 0.0
    j = int(np.searchsorted(t, tq, side="right") - 1)
    j = max(0, min(j, len(t) - 1))
    return float(v[j])


print("[load] HMD yaw rate ...", flush=True)
YR = load_yaw_rate()
print("[load] %d 场有 HMD" % len(YR), flush=True)
alls = T.build_index(ROOT, None)
C.relabel(alls)
sids = sorted({x["sid"] for x in alls})
hist = {}
for x in alls:
    hist[(x["sid"], round(float(x["t"]), 3))] = np.asarray(x["act"])[0]


def features(x, mode):
    tp = x.get("topo") or (0.0, 0.0, 0.0)
    bd = x.get("body") or (0.0, 0.0)
    f = [float(tp[0]), float(tp[1]), float(tp[2]),
         float(bd[0]), float(bd[1]), float(x.get("mode", 0))]
    if "wp" in mode:
        f = [float(x["wp"][0]), float(x["wp"][1])] + f
    t0 = float(x["t"])
    if "yaw" in mode:
        t, r = YR.get(x["sid"], (np.array([]), np.array([])))
        cur = zoh(t, r, t0)
        lo = int(np.searchsorted(t, t0 - 1.0, side="left"))
        hi = int(np.searchsorted(t, t0, side="right"))
        seg = r[lo:hi] if hi > lo else np.array([0.0])
        f += [cur, float(np.mean(seg)), float(np.std(seg)),
              float(np.max(np.abs(seg))) if seg.size else 0.0]
    if "hist" in mode:
        for k in (1, 2, 3, 5):
            h = hist.get((x["sid"], round(t0 - 0.2 * k, 3)))
            f += [float(h[2]) if h is not None and len(h) > 2 else 0.0]
    return f


MODES = [
    ("A 含wp 8维", "wp topo body mode"),
    ("B 去wp 6维", "topo body mode"),
    ("C = B + yaw rate", "topo body mode yaw"),
    ("D = C + omega 历史", "topo body mode yaw hist"),
    ("E 只用 omega 历史", "hist"),
]

for name, mode in MODES:
    Ps, As, Ms, Cz = [], [], [], []
    ndim = len(features(alls[0], mode))
    for hold in sids:
        trv = [x for x in alls if x["sid"] != hold]
        vav = [x for x in alls if x["sid"] == hold]
        Xtr = np.array([features(x, mode) for x in trv], dtype=np.float64)
        Xva = np.array([features(x, mode) for x in vav], dtype=np.float64)
        Atr = np.stack([np.asarray(x["act"])[:S, 2:3] for x in trv])
        Ava = np.stack([np.asarray(x["act"])[:S, 2:3] for x in vav])
        Mtr = np.stack([np.asarray(x["mask"])[:S] for x in trv])
        Mva = np.stack([np.asarray(x["mask"])[:S] for x in vav])
        ok = Mtr.sum(-1) == S
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Tt, Vv = (Xtr[ok]-mu)/sd, (Xva-mu)/sd
        if Tt.shape[1] <= 16:
            Tt, Vv = quad(Tt), quad(Vv)
        Y = Atr[ok].reshape(int(ok.sum()), -1)
        best = (1e18, None)
        for lam in (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
            W = np.linalg.solve(Tt.T@Tt + lam*np.eye(Tt.shape[1]), Tt.T@Y)
            p = (Vv@W).reshape(len(Vv), S, 1)
            v = m_of(p, Ava, Mva)
            if v < best[0]: best = (v, p)
        Ps.append(best[1]); As.append(Ava); Ms.append(Mva)
        Cz.extend([x["cls2"] for x in vav])
    Pp = np.concatenate(Ps); Aa = np.concatenate(As); Mk = np.concatenate(Ms)
    Cz = np.array(Cz)
    zero = m_of(np.zeros_like(Aa), Aa, Mk)
    allm = m_of(Pp, Aa, Mk)
    print()
    print("  %-20s dim=%-4d pooled=%.4f  zero=%.4f  可解释=%.1f%%" %
          (name, ndim, allm, zero, 100*(1-allm/zero)))
    for c in C.CLASSES:
        sel = (Cz == c)
        if sel.sum() < 5:
            continue
        z = m_of(np.zeros_like(Aa[sel]), Aa[sel], Mk[sel])
        m = m_of(Pp[sel], Aa[sel], Mk[sel])
        print("      %-16s n=%-6d mse=%.4f  zero=%.4f  可解释=%.1f%%" %
              (c, int(sel.sum()), m, z, 100*(1-m/z)))
