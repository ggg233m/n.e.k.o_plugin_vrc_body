# -*- coding: utf-8 -*-
r"""A：不用 LM 末位 token，改用 **ViT patch tokens 的空间结构**读动作。

动机（2026-11-04 实测）：
  * LM 末位 token 的 pooled 1024 维：单独读动作 2.889，加进过去特征后把 1.53 拖到 2.89。
  * 主干信息量甚至不如 8 个数字（pooled 2.2978 vs 数值二次 1.1834）。
  * 假设：**避障/导航依赖空间结构**，而「末位 token 池化」把 patch 位置抹掉了。

本探针比较三种视觉读出：
  A. LM 末位 token（现状，对照）
  B. ViT patch tokens → 按 2D 网格做 4x4 平均池化 → 展平
  C. ViT patch tokens → 全局平均（丢掉空间，对照 B）

各自与「过去特征」拼接，看能否超过 B（不泄漏未来）的 1.5254。
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
    ap.add_argument("--grid", type=int, default=4, help="patch token 的空间池化格数")
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
    visual = getattr(inner, "visual", None)
    print("[model] visual=%s" % (visual is not None), flush=True)

    cap = {}
    if visual is not None:
        def hook(m, args, kwargs, out):
            cap["v"] = out
        visual.register_forward_hook(hook, with_kwargs=True)

    def hist_feats(v):
        o = np.zeros((len(v), 6))
        for i, x in enumerate(v):
            tp = x.get("topo") or (0.0, 0.0, 0.0)
            bd = x.get("body") or (0.0, 0.0)
            spd = float(np.hypot(bd[0], bd[1]))
            yaw = float(np.arctan2(bd[0], bd[1])) if spd > 0.1 else 0.0
            o[i] = [bd[0], bd[1], spd, yaw, float(tp[1]), float(x.get("mode", 0))]
        return o

    def extract_patches(out, bsz):
        """Qwen3.5 的 visual 输出结构（实测）：

            BaseModelOutputWithPooling:
              last_hidden_state: (896, 768)   = 2 帧 x 448 patch，**已把 batch 压平**
              pooler_output: tuple，每样本一个 (112, 1024)

        112 = 14 x 8，对应 448x256 按 32x32 patch 的网格。**pooler_output 才是
        空间池化后的表示**，但被拆成 tuple，之前当成单个向量用了。
        """
        po = getattr(out, "pooler_output", None)
        if isinstance(po, (tuple, list)) and len(po) == bsz:
            P = torch.stack([p for p in po], 0)   # (B, 112, 1024)
            return P, int(P.shape[1]), int(P.shape[2])
        t = getattr(out, "last_hidden_state", None)
        if torch.is_tensor(t) and t.dim() == 2:
            # ⚠️ last_hidden_state 是 (2帧*448patch, 768)，**不含 batch 维**；
            # 用它按 bsz 切会得到错的 per（末批不足时会算出 768 维）。
            # 只有 pooler_output 才是每样本一份，所以这里直接判失败，别硬切。
            return None, 0, 0
        if torch.is_tensor(t) and t.dim() == 3:
            return t, int(t.shape[1]), int(t.shape[2])
        return None, 0, 0

    def collect(v):
        ds = T.PolicyDataset(v, proc, px, px, quad_feats=False)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        pooled, patch_grid, patch_gap, A, M = [], [], [], [], []
        with torch.no_grad():
            for b in dl:
                kw = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in b["inputs"].items()}
                out = inner(**kw, use_cache=False)
                h = out.last_hidden_state
                idx = kw["attention_mask"].sum(-1) - 1
                ar = torch.arange(h.shape[0], device=h.device)
                pooled.append(h[ar, idx].float().cpu().numpy())

                pt, per, D = extract_patches(cap.get("v"), b["act"].shape[0])
                bsz = pt.shape[0] if pt is not None else 0
                if pt is None:
                    patch_grid.append(None); patch_gap.append(None)
                else:
                    P = pt.float().cpu()
                    # 固定网格 8x14 = 112（448x256 / 32x32 patch）。
                    # ⚠️ 池化必须产出**固定维度**：之前用 per 反推网格，末批不足 4 条时
                    # per 不同 -> usable 不同 -> 拼起来维度不一致（实测 16384 vs 12288）。
                    gh, gw = 8, 14
                    if gh * gw != per:
                        # 兜底：退化为全局平均，保持维度一致
                        patch_grid.append(None)
                    else:
                        # 8x14 的整除因子只有 (1,1),(1,2),(2,1),(2,2),(1,7),(1,14),(4,1),(4,2)...
                        # 用 4x2 的空间池化 -> 8 个格 x 1024 = 8192 维（保留粗空间结构）
                        ph, pw = 2, 2   # 8/2=4, 14/2=7 -> 池化后 4x7 = 28 个格
                        Q = P.view(P.shape[0], ph, gh // ph, pw, gw // pw, D).mean(dim=(2, 4))
                        patch_grid.append(Q.reshape(P.shape[0], -1).numpy())
                    patch_gap.append(P.mean(1).numpy())
                A.append(b["act"].numpy()); M.append(b["mask"].numpy())

        def cat_or_none(lst):
            if any(x is None for x in lst):
                return None
            return np.concatenate(lst)
        return (np.concatenate(pooled), cat_or_none(patch_grid),
                cat_or_none(patch_gap), np.concatenate(A), np.concatenate(M))

    print("[collect] train=%d val=%d grid=%d" % (len(tr), len(va), a.grid), flush=True)
    Ptr, Gtr, Atr_, Atr, Mtr = collect(tr)
    Pva, Gva, Ava_, Ava, Mva = collect(va)
    print("[patch] grid tokens train=%s val=%s gap train=%s"
          % (None if Gtr is None else Gtr.shape, None if Gva is None else Gva.shape,
             None if Atr_ is None else Atr_.shape), flush=True)

    okt = Mtr[:, :S].sum(-1) == S
    okv = Mva[:, :S].sum(-1) == S
    Ytr = Atr[okt][:, :S, :2].reshape(int(okt.sum()), -1)
    Yva = Ava[okv][:, :S, :2]
    Mv = Mva[okv][:, :S]
    Htr, Hva = hist_feats(tr)[okt], hist_feats(va)[okv]

    def ev(Xt, Xv, tag):
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        Tt, Vv = (Xt - mu) / sd, (Xv - mu) / sd
        m = ridge_mse(Tt, Ytr, Vv, Yva, Mv)
        print("  %-44s dim=%-6d mse8 = %.4f" % (tag, Tt.shape[1], m))
        return m

    print()
    m_hist = ev(Htr, Hva, "过去特征（无图像）")
    ev(Ptr[okt], Pva[okv], "A. 过去 + LM 末位 token")
    if Gtr is not None:
        ev(np.concatenate([Htr, Gtr[okt]], 1), np.concatenate([Hva, Gva[okv]], 1),
           "B. 过去 + ViT patch 空间池化(%dx%d)" % (a.grid, a.grid))
    if Atr_ is not None:
        ev(np.concatenate([Htr, Atr_[okt]], 1), np.concatenate([Hva, Ava_[okv]], 1),
           "C. 过去 + ViT 全局平均（丢空间）")
    print()
    print("  判读：B 若明显优于 A 和 C ⇒ **空间结构是缺的东西**，末位 token 池化丢了它")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
