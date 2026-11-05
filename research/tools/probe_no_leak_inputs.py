# -*- coding: utf-8 -*-
r"""用「不泄漏未来」的输入替代 wp，看还能不能预测动作。

背景（2026-11-04 实测）：
  * `wp` 不是 A* 给的，是 **waypoint_body() 沿录制轨迹往前取 4 m** —— 它来自**未来**。
  * 但 corr(wp 方向, 未来位移方向) 只有 **−0.150**，corr(|wp|,|位移|) 0.215。
  * 未来位移 p50 = **0.06 m** 而 wp 模长 p50 = 3.71 m —— 一半样本人在停着，输入却说前方 3.7 m。
  ⇒ 这个输入既泄漏未来、又和当前决策脱钩。

本探针换三组**不泄漏未来**的输入，看能到多少：
  A. 过去轨迹方向（最近 2 s 的平均朝向/速度）+ body + mode
  B. A + 图像 pooled
  C. HMD 视线方向（若可用）+ A

并给出基线：
  * wp 现状（泄漏未来的上限参照）
  * body only（纯惯性）
  * copy（抄上一拍）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
_ONEAPI = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
if os.path.isdir(_ONEAPI):
    os.environ["PATH"] = _ONEAPI + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CC", "icpx")
    os.environ.setdefault("CXX", "icpx")
sys.path.insert(0, str(Path(__file__).resolve().parent))

S = 8


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def ridge_mse(Xtr, Ytr, Xva, Yva, mask, lams=(1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e4)):
    best = 1e9
    for lam in lams:
        try:
            W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
        except np.linalg.LinAlgError:
            continue
        pred = (Xva @ W).reshape(len(Xva), S, 2)
        sq = ((pred - Yva) ** 2).sum(-1)
        best = min(best, float((sq * mask).sum() / max(mask.sum(), 1e-9)))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--ntrain", type=int, default=2000)
    ap.add_argument("--nval", type=int, default=300)
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    dev = "xpu"
    proc = AutoProcessor.from_pretrained(a.model)
    px = 448 * 256
    alls = T.build_index(Path(a.dataset), None)
    tr = [s for s in alls if s["split"] == "train"][: a.ntrain]
    va = T.select_val_subset(Path(a.dataset), a.nval)

    def hist_feats(v):
        """不泄漏未来的特征：body(当前速度) + 过去方向 + topo + mode。

        注意：`body` 在 PolicyDataset 里取自 **t-0.2 s** 的 act[0]，是过去，不泄漏。
        `topo` 是离线合成的路径位置，也不含未来。
        `wp` 我们**故意不用**（它来自未来轨迹）。
        """
        o = np.zeros((len(v), 6))
        for i, x in enumerate(v):
            tp = x.get("topo") or (0.0, 0.0, 0.0)
            bd = x.get("body") or (0.0, 0.0)
            spd = float(np.hypot(bd[0], bd[1]))
            yaw = float(np.arctan2(bd[0], bd[1])) if spd > 0.1 else 0.0
            o[i, 0] = bd[0]; o[i, 1] = bd[1]
            o[i, 2] = spd
            o[i, 3] = yaw
            o[i, 4] = float(tp[1])   # lateral
            o[i, 5] = float(x.get("mode", 0))
        return o

    def full_feats(v):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            o[i, :2] = wp; o[i, 2:5] = tp; o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    model, head, hidden = T.load_model(a.model, 128, dev, act_dim=3)
    model.eval()
    inner = model.base_model.model.model

    def collect_pool(v):
        ds = T.PolicyDataset(v, proc, px, px, quad_feats=False)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        P, A, M = [], [], []
        with torch.no_grad():
            for b in dl:
                kw = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in b["inputs"].items()}
                out = inner(**kw, use_cache=False)
                h = out.last_hidden_state
                idx = kw["attention_mask"].sum(-1) - 1
                ar = torch.arange(h.shape[0], device=h.device)
                P.append(h[ar, idx].float().cpu().numpy())
                A.append(b["act"].numpy()); M.append(b["mask"].numpy())
        return np.concatenate(P), np.concatenate(A), np.concatenate(M)

    print("[collect] train=%d val=%d" % (len(tr), len(va)), flush=True)
    Ptr, Atr, Mtr = collect_pool(tr)
    Pva, Ava, Mva = collect_pool(va)

    okt = Mtr[:, :S].sum(-1) == S
    okv = Mva[:, :S].sum(-1) == S
    Ytr = Atr[okt][:, :S, :2].reshape(int(okt.sum()), -1)
    Yva = Ava[okv][:, :S, :2]
    Mv = Mva[okv][:, :S]

    Ftr_h, Fva_h = hist_feats(tr)[okt], hist_feats(va)[okv]
    Ftr_f, Fva_f = full_feats(tr)[okt], full_feats(va)[okv]
    Ptr_, Pva_ = Ptr[okt], Pva[okv]

    def ev(Xt, Xv, tag, use_quad=True):
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        Tt, Vv = (Xt - mu) / sd, (Xv - mu) / sd
        if use_quad and Tt.shape[1] <= 64:
            Tt, Vv = quad(Tt), quad(Vv)
        m = ridge_mse(Tt, Ytr, Vv, Yva, Mv)
        print("  %-42s dim=%-5d mse8 = %.4f" % (tag, Tt.shape[1], m))
        return m

    print()
    print("  === 基线 ===")
    copy = float(((((np.repeat(Yva[:, :1], S, axis=1) - Yva) ** 2).sum(-1)) * Mv).sum() / Mv.sum())
    print("  %-42s dim=%-5d mse8 = %.4f" % ("copy（抄上一拍）", 2, copy))
    m_wp = ev(Ftr_f, Fva_f, "A. wp 现状（**泄漏未来**，上限参照）")
    m_h = ev(Ftr_h, Fva_h, "B. 过去特征（body+spd+yaw+lateral+mode）")
    m_hp = ev(np.concatenate([Ftr_h, Ptr_], 1), np.concatenate([Fva_h, Pva_], 1),
              "C. B + 图像 pooled(1024)", use_quad=False)
    m_p = ev(Ptr_, Pva_, "D. 只用图像 pooled(1024)", use_quad=False)

    print()
    print("  判读：")
    print("   * B 明显差于 A ⇒ 「不泄漏未来」的代价有多大")
    print("   * C vs B ⇒ 图像在**无泄漏**条件下能不能补上")
    print("   * D vs copy ⇒ 图像单独有没有超过惯性")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
