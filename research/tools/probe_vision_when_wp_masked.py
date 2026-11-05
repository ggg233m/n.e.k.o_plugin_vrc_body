# -*- coding: utf-8 -*-
r"""拿掉 wp 的“泄漏”后，图像还有没有用？

背景（2026-11-04）：
  * 8 数值消融：全部 1.2063 / 去掉 wp 1.5584（差 29%）⇒ wp 是“去哪儿”的主通道
  * 但 wp 是 A* 算好的前视点，等于把方向答案直接给了模型
  * 图像在这套架构里被证实是可省的（black 比 real 还准）

本探针回答：**如果把 wp 削成“只有方向、没有精确距离”，图像能不能补上剩下的？**

三组对照（同一批 val，前 8 步，ridge 读出）：
  A. 完整 8 数值（现状）
  B. wp 只剩单位方向（归一化到 1.0）+ topo + body + mode —— 距离信息被抹掉
  C. B + 图像（视觉塔 pooled 特征）

判据：
  * 若 C ≈ B ⇒ 图像提供不了被抹掉的信息（模型学不会“自己决定去哪”）
  * 若 C 明显优于 B ⇒ **图像里有方向信息，只是当前被 wp 泄漏掩盖了**
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--quad", action="store_true", default=True)
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

    model, head, hidden = T.load_model(a.model, 128, dev, act_dim=3)
    model.eval()
    inner = model.base_model.model.model
    vis_cap = {}
    vis = getattr(inner, "visual", None)
    h = None
    if vis is not None:
        def hook(m, args, kwargs, out):
            vis_cap["f"] = out
        h = vis.register_forward_hook(hook, with_kwargs=True)

    def feats(v, drop_wp_dist: bool):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            if drop_wp_dist:
                # 只保留**单位方向**，距离信息抹掉
                n = max(np.hypot(wp[0], wp[1]), 1e-9)
                o[i, 0] = wp[0] / n; o[i, 1] = wp[1] / n
            else:
                o[i, 0] = wp[0]; o[i, 1] = wp[1]
            o[i, 2:5] = tp; o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    def collect(v, drop_wp_dist: bool, want_vis: bool):
        ds = T.PolicyDataset(v, proc, px, px, quad_feats=False)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        P, A, M = [], [], []
        with torch.no_grad():
            for b in dl:
                kw = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in b["inputs"].items()}
                out = inner(**kw, use_cache=False)
                hid = out.last_hidden_state
                idx = kw["attention_mask"].sum(-1) - 1
                ar = torch.arange(hid.shape[0], device=hid.device)
                P.append(hid[ar, idx].float().cpu().numpy())
                A.append(b["act"].numpy()); M.append(b["mask"].numpy())
        return (np.concatenate(P) if want_vis else None,
                feats(v, drop_wp_dist), np.concatenate(A), np.concatenate(M))

    def prep(Xtr, Xva, n_num: int = 8):
        """标准化；只对**前 n_num 个数值特征**做二次扩张，后面的是图像，保持线性。

        ⚠️ 不能整体 quad：拼接后是 44 + 1024 = 1068 维，二次扩张会到 57 万维，
        求解时要分配 2 TiB 内存（实测 numpy._core._exceptions._ArrayMemoryError）。
        """
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        T_, V_ = (Xtr - mu) / sd, (Xva - mu) / sd
        if not a.quad:
            return T_, V_
        Tn, Tv = quad(T_[:, :n_num]), quad(V_[:, :n_num])
        if T_.shape[1] > n_num:
            Tn = np.concatenate([Tn, T_[:, n_num:]], 1)
            Tv = np.concatenate([Tv, V_[:, n_num:]], 1)
        return Tn, Tv

    print("[collect] train=%d val=%d" % (len(tr), len(va)), flush=True)
    Ptr, Ntr_full, Atr, Mtr = collect(tr, False, True)
    Ntr_dir = feats(tr, True)
    Pva, Nva_full, Ava, Mva = collect(va, False, True)
    Nva_dir = feats(va, True)
    if h:
        h.remove()

    okt = Mtr[:, :S].sum(-1) == S
    okv = Mva[:, :S].sum(-1) == S
    Ytr = Atr[okt][:, :S, :2].reshape(int(okt.sum()), -1)
    Yva = Ava[okv][:, :S, :2]
    Mv = Mva[okv][:, :S]

    def ev(Xt, Xv, tag):
        Tt, Vv = prep(Xt[okt], Xv[okv])
        m = ridge_mse(Tt, Ytr, Vv, Yva, Mv)
        print("  %-40s dim=%-5d mse8 = %.4f" % (tag, Tt.shape[1], m))
        return m

    print()
    print("  === 三组对照（前 8 步，masked）===")
    mA = ev(Ntr_full, Nva_full, "A. 完整 8 数值（现状，含 wp 距离）")
    mB = ev(Ntr_dir, Nva_dir, "B. wp 只剩单位方向 + topo+body+mode")
    mC = ev(np.concatenate([Ntr_dir, Ptr], 1), np.concatenate([Nva_dir, Pva], 1),
            "C. B + 图像 pooled(1024)")
    print()
    print("  B → C 的改善：%+.1f%%" % (100.0 * (mB - mC) / max(mB, 1e-9)))
    if abs(mB - mC) / max(mB, 1e-9) < 0.05:
        print("  ⇒ **图像提供不了被抹掉的信息**：模型学不会「自己决定去哪」")
    else:
        print("  ⇒ 图像里有方向信息，只是被 wp 泄漏掩盖了")
    print()
    print("  A → B 的损失：%+.1f%%（抹掉 wp 距离的代价）" % (100.0 * (mA - mB) / max(mA, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
