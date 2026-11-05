# -*- coding: utf-8 -*-
"""诚实的基准表：按正交 4 类 + 速度/ω 分开报。

为什么必须分开（2026-11-05 实测）：
  * still / walk_straight 占 26% 样本，但 zero 和 copy 基线都≈0，
    任何模型在这两类上都拿不到区分度 —— 混进聚合就是白送分。
  * ω 的可解释比例只有 1.1%（ridge 2.3734 vs zero 2.3986），
    把它和速度混在同一个 mse 里，等于用一个学不会的量惩罚所有模型。

所以本表只报三个数，且不合并：
    speed_mse（前 8 步，两维）  /  omega_mse（第 3 维）  /  各自的 copy、zero 基线
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
    return float((sq * mask).sum() / max(mask.sum(), 1.0)), float(mask.sum())


alls = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
C.relabel(alls)
sids = sorted({x["sid"] for x in alls})
print("[data] %d 样本 / %d 世界 / val 抽样口径见下" % (len(alls), len(sids)))


def run_lwo(drop_wp):
    """返回 (pred_speed, pred_omega, act_speed, act_omega, mask, cls)。"""
    Ps, Po, As, Ao, Ms, Cz = [], [], [], [], [], []
    for hold in sids:
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
            v = m_of(p[..., :2], Ava[..., :2], Mva)[0]
            if v < best[0]: best = (v, p)
        p = best[1]
        Ps.append(p[..., :2]); Po.append(p[..., 2:3])
        As.append(Ava[..., :2]); Ao.append(Ava[..., 2:3])
        Ms.append(Mva); Cz.extend([x["cls2"] for x in vav])
    return (np.concatenate(Ps), np.concatenate(Po), np.concatenate(As),
            np.concatenate(Ao), np.concatenate(Ms), np.array(Cz))


for tag, drop in (("去 wp（部署口径）", True), ("含 wp（泄漏上限）", False)):
    Ps, Po, As, Ao, Ms, Cz = run_lwo(drop)
    cp_s = np.repeat(As[:, :1], S, axis=1)
    cp_o = np.repeat(Ao[:, :1], S, axis=1)
    print()
    print("=" * 96)
    print("  %s" % tag)
    print("=" * 96)
    print("  %-16s %6s %8s %8s %8s %8s %8s %8s" %
          ("class", "n", "spd_ridge", "spd_copy", "spd_zero",
           "omg_ridge", "omg_copy", "omg_zero"))
    for c in C.CLASSES + ("ALL",):
        sel = np.ones(len(Cz), bool) if c == "ALL" else (Cz == c)
        if sel.sum() == 0:
            continue
        s_r, _ = m_of(Ps[sel], As[sel], Ms[sel])
        s_c, _ = m_of(cp_s[sel], As[sel], Ms[sel])
        s_z, _ = m_of(np.zeros_like(As[sel]), As[sel], Ms[sel])
        o_r, _ = m_of(Po[sel], Ao[sel], Ms[sel])
        o_c, _ = m_of(cp_o[sel], Ao[sel], Ms[sel])
        o_z, _ = m_of(np.zeros_like(Ao[sel]), Ao[sel], Ms[sel])
        print("  %-16s %6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" %
              (c, int(sel.sum()), s_r, s_c, s_z, o_r, o_c, o_z))
    # 决策类聚合：剔除 still / walk_straight（26% 白送分）
    dec = (Cz == "turn_in_place") | (Cz == "walk_turn")
    s_r, w = m_of(Ps[dec], As[dec], Ms[dec])
    s_c, _ = m_of(cp_s[dec], As[dec], Ms[dec])
    o_r, _ = m_of(Po[dec], Ao[dec], Ms[dec])
    print("  %-16s %6d %8.4f %8.4f %8.4f" %
          ("[决策类聚合]", int(dec.sum()), s_r, s_c, o_r))
    print("  %% 权重: still %.1f  turn %.1f  straight %.1f  walk_turn %.1f" % tuple(
        100*np.mean(Cz == k) for k in C.CLASSES))
