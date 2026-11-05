# -*- coding: utf-8 -*-
r"""逐层探测：24 层各自对「输出动作」贡献了什么，哪一层在拖后腿。

背景（2026-11-04 实测）：
  * 只用 8 个数值的二次扩张（44 维）+ MLP  → mse8 **1.1475**
  * 76 我们的模型（图像 + 0.8B 主干 + 5.41M LoRA） → **1.4823**
  ⇒ 主干 + 图像净贡献为负 23%。问题在主干内部，需要定位到层。

探针设计（不动权重，只读）：
  对每一层 L 的输出取 hidden（末位 token），单独用 ridge 读「前 8 步动作」，
  看**每层的表示里还剩多少动作信息**。三条曲线：
    (a) 原始 hidden（该层输出）
    (b) 减去该层的均值方向（去掉"公共偏置"后还剩什么）
    (c) 与输入数值特征拼接后的增益

若某层之后 mse 突然变差 ⇒ 那一层把动作相关信息丢了。
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


def ridge_mse(Xtr, Ytr, Xva, Yva, mask, lams=(1e-4, 1e-2, 1.0, 100.0, 1e4)):
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
    ap.add_argument("--ntrain", type=int, default=800)
    ap.add_argument("--nval", type=int, default=200)
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
    layers = inner.language_model.layers
    nl = len(layers)
    print("[model] layers=%d hidden=%d" % (nl, hidden), flush=True)

    caps = {}
    handles = []
    def mk(i):
        def hook(mod, args, kwargs, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[i] = h
        return hook
    for i, L in enumerate(layers):
        handles.append(L.register_forward_hook(mk(i), with_kwargs=True))

    def feats(v):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            o[i, :2] = wp; o[i, 2:5] = tp; o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    def collect(v):
        ds = T.PolicyDataset(v, proc, px, px)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        per_layer = {i: [] for i in range(nl)}
        A, M = [], []
        with torch.no_grad():
            for b in dl:
                kw = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in b["inputs"].items()}
                inner(**kw, use_cache=False)
                idx = kw["attention_mask"].sum(-1) - 1
                ar = torch.arange(4, device=dev)[: idx.shape[0]]
                for i in range(nl):
                    h = caps[i]
                    per_layer[i].append(h[ar, idx].float().cpu().numpy())
                A.append(b["act"].numpy()); M.append(b["mask"].numpy())
        return ({i: np.concatenate(v) for i, v in per_layer.items()},
                feats(v), np.concatenate(A), np.concatenate(M))

    print("[collect] train=%d val=%d (24 层 hook)" % (len(tr), len(va)), flush=True)
    Ptr, Ntr, Atr, Mtr = collect(tr)
    Pva, Nva, Ava, Mva = collect(va)
    for h in handles:
        h.remove()

    okt = Mtr[:, :S].sum(-1) == S
    okv = Mva[:, :S].sum(-1) == S
    Ytr = Atr[okt][:, :S, :2].reshape(int(okt.sum()), -1)
    Yva = Ava[okv][:, :S, :2]
    Mv = Mva[okv][:, :S]
    Ntr2, Nva2 = Ntr[okt], Nva[okv]

    print()
    print("  层  类型        hidden读出   +数值拼接   增益")
    for i in range(nl):
        Htr = Ptr[i][okt]; Hva = Pva[i][okv]
        # 标准化
        mu, sd = Htr.mean(0), Htr.std(0) + 1e-9
        Htr = (Htr - mu) / sd; Hva = (Hva - mu) / sd
        m_h = ridge_mse(Htr, Ytr, Hva, Yva, Mv)
        # 拼接数值
        nun_mu, nun_sd = Ntr2.mean(0), Ntr2.std(0) + 1e-9
        Nt = (Ntr2 - nun_mu) / nun_sd; Nv = (Nva2 - nun_mu) / nun_sd
        Ctr = np.concatenate([Htr, Nt], 1); Cva = np.concatenate([Hva, Nv], 1)
        m_c = ridge_mse(Ctr, Ytr, Cva, Yva, Mv)
        typ = "lin" if i % 4 != 3 else "FULL"
        print("  L%-2d %-6s %10.4f %12.4f %8.1f%%"
              % (i, typ, m_h, m_c, 100.0 * (m_h - m_c) / max(m_h, 1e-9)))

    print()
    Nmu, Nsd = Ntr2.mean(0), Ntr2.std(0) + 1e-9
    m_num = ridge_mse(quad((Ntr2 - Nmu) / Nsd), Ytr, quad((Nva2 - Nmu) / Nsd), Yva, Mv)
    print("  参照：只用 8 数值二次 = %.4f" % m_num)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
