# -*- coding: utf-8 -*-
r"""队友推荐的两个判决性实验（都纯 CPU，可与 GPU 训练并行）。

实验 1：ridge 的**数据缩放曲线**
  用 7206 / 11000 / 18365 / 23082 条训练样本，看 ridge 的 val mse8 怎么变。
    flat  → 加数据只是补课，追到 ridge 水平就到头
    下降  → 加数据是真突破
  这一个实验决定「接下来两个月是加数据还是换方案」。

实验 2：**同特征 MLP vs ridge**（44 维二次）
  逻辑：ridge 是 MLP 的子集，更大的假设类不该比子集更差。
  若 MLP 更差 ⇒ 只能是我的优化问题（印证「旁路失败≠表示问题」有漏洞）。

口径与 probe_ridge_masked 完全一致：select_val_subset(400) + 前 8 步 + mask 归一。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

S = 8


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def ridge_fit(X, Y, lam=100.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def masked_mse(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--nval", type=int, default=400)
    ap.add_argument("--sizes", type=int, nargs="+", default=[7206, 11000, 18365, 23082])
    ap.add_argument("--mlp-epochs", type=int, default=400)
    a = ap.parse_args()

    import train_policy_lora as T
    alls = T.build_index(Path(a.dataset), None)
    tr_all = [s for s in alls if s["split"] == "train"]
    va = T.select_val_subset(Path(a.dataset), a.nval)
    print("[data] train池=%d val=%d" % (len(tr_all), len(va)), flush=True)

    def feats(v):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            o[i, :2] = wp; o[i, 2:5] = tp; o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    def pack(v):
        A = np.stack([np.asarray(x["act"])[:, :2] for x in v])
        M = np.stack([np.asarray(x["mask"]) for x in v])
        return feats(v), A, M

    Nva, Ava, Mva = pack(va)
    Ava, Mva = Ava[:, :S], Mva[:, :S]
    mu_ref = None

    print()
    print("  === 实验 1：ridge 的数据缩放曲线 ===")
    print("  %8s %10s %10s" % ("n_train", "ridge", "copy"))
    curve = []
    for n in a.sizes:
        sub = tr_all[:n]
        Ntr, Atr, Mtr = pack(sub)
        okt = Mtr[:, :S].sum(-1) == S
        Ntr, Atr = Ntr[okt], Atr[okt][:, :S]
        if mu_ref is None:
            mu_ref = Ntr.mean(0), Ntr.std(0) + 1e-9
        # 归一化统计量固定用最大集合，保证各点可比
        mu, sd = Ntr.mean(0), Ntr.std(0) + 1e-9
        Ftr = quad((Ntr - mu) / sd)
        Fva = quad((Nva - mu) / sd)
        Ytr = Atr.reshape(len(Atr), -1)
        best = 1e9
        for lam in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
            W = ridge_fit(Ftr, Ytr, lam)
            pred = (Fva @ W).reshape(len(Fva), S, 2)
            best = min(best, masked_mse(pred, Ava, Mva))
        copy = masked_mse(np.repeat(Ava[:, :1], S, axis=1), Ava, Mva)
        curve.append((n, best, copy))
        print("  %8d %10.4f %10.4f" % (n, best, copy))

    if len(curve) >= 2:
        d0, d1 = curve[0][1], curve[-1][1]
        rel = 100.0 * (d0 - d1) / max(d0, 1e-9)
        print()
        print("  n: %d → %d，误差 %.4f → %.4f（变化 %.1f%%）"
              % (curve[0][0], curve[-1][0], d0, d1, rel))
        if abs(rel) < 3.0:
            print("  ⇒ **flat**：加数据只是补课，追到 ridge 水平就到头")
        elif rel > 0:
            print("  ⇒ **仍在下降**：加数据是真突破，值得投入")
        else:
            print("  ⇒ 反而变差，检查数据质量")

    # ---------------- 实验 2：MLP vs ridge ----------------
    print()
    print("  === 实验 2：同特征下 MLP vs ridge（44 维二次）===")
    n_use = min(18365, len(tr_all))
    sub = tr_all[:n_use]
    Ntr, Atr, Mtr = pack(sub)
    okt = Mtr[:, :S].sum(-1) == S
    Ntr, Atr = Ntr[okt], Atr[okt][:, :S]
    mu, sd = Ntr.mean(0), Ntr.std(0) + 1e-9
    Ftr = quad((Ntr - mu) / sd)
    Fva = quad((Nva - mu) / sd)
    Ytr = Atr.reshape(len(Atr), -1)

    best_r = 1e9
    for lam in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
        W = ridge_fit(Ftr, Ytr, lam)
        pred = (Fva @ W).reshape(len(Fva), S, 2)
        best_r = min(best_r, masked_mse(pred, Ava, Mva))
    print("  ridge（解析解）      mse8 = %.4f" % best_r)

    import torch
    torch.manual_seed(0)
    dev = "xpu" if torch.xpu.is_available() else "cpu"
    Xt = torch.tensor(Ftr, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    Xv = torch.tensor(Fva, dtype=torch.float32, device=dev)
    Av = torch.tensor(Ava, dtype=torch.float32, device=dev)
    Mv = torch.tensor(Mva, dtype=torch.float32, device=dev)

    for hidden, wd, lr in [(256, 1e-4, 1e-3), (256, 1e-2, 1e-3), (64, 1e-2, 1e-3)]:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(Xt.shape[1], hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, Yt.shape[1]),
        ).to(dev)
        opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)
        for ep in range(a.mlp_epochs):
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 4096):
                idx = perm[i:i + 4096]
                opt.zero_grad(set_to_none=True)
                p = mlp(Xt[idx])
                loss = ((p - Yt[idx]) ** 2).mean()
                loss.backward()
                opt.step()
        mlp.eval()
        with torch.no_grad():
            p = mlp(Xv).reshape(len(Xv), S, 2)
        m = masked_mse(p.cpu().numpy(), Av.cpu().numpy(), Mv.cpu().numpy())
        print("  MLP(h=%-4d wd=%-6g) mse8 = %.4f" % (hidden, wd, m))

    print()
    print("  判读：MLP 若显著差于 ridge ⇒ 是**优化问题**，不是「表示不够」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
