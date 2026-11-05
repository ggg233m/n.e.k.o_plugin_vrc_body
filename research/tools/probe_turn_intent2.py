# -*- coding: utf-8 -*-
"""转向意图分类：补齐 omega 历史与 HMD yaw rate 后重测。

上一版（probe_turn_intent.py）只用 topo+body(vx,vz)+mode，分类 acc 62.71%，
多数类基线 62.58% —— 等于没学到。但根因是 body 只有两维：
    train_policy_lora.py:112  body = (lab[ip]["act"][0][:2])   # 丢掉了 omega
**模型根本看不到自己刚才在往哪转。**

本探针在完全相同的 LWO 10 折上逐级补特征：
  F0 现状       topo(3) + body(2) + mode(1)                = 6
  F1 +omega 历史 最近 1/2/3/5 帧 act[0][2]                 = 10
  F2 +HMD yaw    当前值/1s均值/1s标准差/1s最大绝对         = 14
  F3 只用omega历史                                          = 4
  F4 body+omega历史（无 topo/mode）                         = 10
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
REC = Path("G:/vrc_recording")
WIN = (3, 6)
THR = 0.3


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def load_yaw():
    out = {}
    for sess in sorted(p for p in REC.iterdir() if p.is_dir() and (p / "hmd_frames.jsonl").exists()):
        rows = [json.loads(l) for l in (sess / "hmd_frames.jsonl").open("r", encoding="utf-8")]
        if len(rows) < 10:
            continue
        t = np.array([r["t"] for r in rows], dtype=np.float64)
        q = np.array([r["hmd"]["rotation_xyzw"] for r in rows], dtype=np.float64)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = np.unwrap(np.arctan2(2.0*(w*y + x*z), 1.0 - 2.0*(y*y + z*z)))
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.gradient(yaw, t)
        out[sess.name] = (t, np.nan_to_num(rate))
    return out


def zoh(t, v, tq):
    if len(t) == 0: return 0.0
    j = max(0, min(int(np.searchsorted(t, tq, side="right") - 1), len(t)-1))
    return float(v[j])


def intent_of(act, mask):
    w = np.asarray(act)[WIN[0]:WIN[1], 2]
    ok = np.asarray(mask)[WIN[0]:WIN[1]] > 0
    if not ok.any(): return -1
    mw = float(w[ok].mean())
    return 2 if mw > THR else (0 if mw < -THR else 1)


print("[load] yaw rate ...", flush=True)
YR = load_yaw()
alls = T.build_index(ROOT, None)
C.relabel(alls)
sids = sorted({x["sid"] for x in alls})
hist = {}
for x in alls:
    hist[(x["sid"], round(float(x["t"]), 3))] = np.asarray(x["act"])[0]
lab = np.array([intent_of(x["act"], x["mask"]) for x in alls])
print("[data] %d 样本，意图分布 左%d 直%d 右%d" % (
    len(alls), int((lab==0).sum()), int((lab==1).sum()), int((lab==2).sum())))


def feat(x, mode):
    tp = x.get("topo") or (0.0, 0.0, 0.0)
    bd = x.get("body") or (0.0, 0.0)
    t0 = float(x["t"])
    f = []
    if "base" in mode:
        f += [float(tp[0]), float(tp[1]), float(tp[2]), float(bd[0]), float(bd[1]),
              float(x.get("mode", 0))]
    elif "body" in mode:
        f += [float(bd[0]), float(bd[1])]
    if "hist" in mode:
        for k in (1, 2, 3, 5):
            h = hist.get((x["sid"], round(t0 - 0.2*k, 3)))
            f += [float(h[2]) if h is not None and len(h) > 2 else 0.0]
    if "yaw" in mode:
        t, r = YR.get(x["sid"], (np.array([]), np.array([])))
        lo = max(0, int(np.searchsorted(t, t0-1.0, side="left")))
        hi = int(np.searchsorted(t, t0, side="right"))
        seg = r[lo:hi] if hi > lo else np.array([0.0])
        f += [zoh(t, r, t0), float(np.mean(seg)), float(np.std(seg)),
              float(np.max(np.abs(seg)))]
    return f


MODES = [
    ("F0 现状 topo+body+mode", "base"),
    ("F1 = F0 + omega 历史", "base hist"),
    ("F2 = F1 + HMD yaw", "base hist yaw"),
    ("F3 只用 omega 历史", "hist"),
    ("F4 body + omega 历史", "body hist"),
]

for name, mode in MODES:
    accs, f1s, bases = [], [], []
    nd = len(feat(alls[0], mode))
    for hold in sids:
        tri = [i for i, x in enumerate(alls) if x["sid"] != hold and lab[i] >= 0]
        vai = [i for i, x in enumerate(alls) if x["sid"] == hold and lab[i] >= 0]
        if len(tri) < 200 or len(vai) < 20: continue
        Xtr = np.array([feat(alls[i], mode) for i in tri])
        Xva = np.array([feat(alls[i], mode) for i in vai])
        ytr, yva = lab[tri], lab[vai]
        mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
        Tt, Vv = (Xtr-mu)/sd, (Xva-mu)/sd
        if Tt.shape[1] <= 16:
            Tt, Vv = quad(Tt), quad(Vv)
        Yh = np.eye(3)[ytr]
        best = (-1, None)
        for lam in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
            W = np.linalg.solve(Tt.T@Tt + lam*np.eye(Tt.shape[1]), Tt.T@Yh)
            pr = (Vv@W).argmax(1)
            a = float((pr == yva).mean())
            if a > best[0]: best = (a, pr)
        pr = best[1]
        accs.append(best[0])
        bases.append(float(np.bincount(ytr, minlength=3).max()/len(ytr)))
        f1v = []
        for c in range(3):
            tp = ((pr==c)&(yva==c)).sum(); fp = ((pr==c)&(yva!=c)).sum(); fn = ((pr!=c)&(yva==c)).sum()
            if tp == 0: continue
            p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
            f1v.append(2*p*r/max(p+r,1e-9))
        f1s.append(float(np.mean(f1v)) if f1v else 0.0)
    a = np.array(accs); bb = np.array(bases)
    print()
    print("  %-26s dim=%-4d acc=%.4f±%.4f  F1=%.4f  基线=%.4f  提升=%+.1f%%" %
          (name, nd, a.mean(), a.std(), np.mean(f1s), bb.mean(), 100*(a.mean()/bb.mean()-1)))
