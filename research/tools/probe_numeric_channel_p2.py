# -*- coding: utf-8 -*-
r"""P2：为什么主干用不好那 8 个数值，而 45 参数的 ridge 能到 1.2371？

已知（2026-11-04）：
  ridge（8 数值的二次扩张，45 参）  1.2371
  r8 模型（图像+记忆卡，经 0.8B）   1.4823
  head-only（冻结 LoRA）            2.7358   ← 主干在起作用

三个假设要分清：
  H1 数值信息在主干里**丢了** —— LM 的 24 层把精确标量稀释掉（task-14 的推断）
  H2 数值信息**还在**，但 head 没用出来 —— 读出方式不对
  H3 图像在**干扰** —— 主干把注意力放错地方

判别方法：在同一批 val 样本上，用**冻结主干输出**（pooled 1024 维）做线性/二次读，
与「只用 8 个数值」做同样的读，比较：
  * pooled → act 的可达误差；
  * 数值  → act 的可达误差。
若 pooled 能到 1.24 附近 ⇒ 信息在，是训练问题（H2）；
若 pooled 明显更差且数值更好 ⇒ 信息在主干里丢了（H1）或图像干扰（H3）。

进一步：把两者拼接（pooled + 数值）看是否优于各自单用 —— 若拼接不优于数值单用，
说明 pooled 里关于「怎么走」的信息基本是零。
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

S = 8  # 与 ridge 对齐：前 8 步


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def ridge_eval(Xtr, Ytr, Xva, Yva, lams=(1e-6, 1e-4, 1e-2, 1.0, 100.0)):
    best = 1e9
    for lam in lams:
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
        pred = (Xva @ W).reshape(len(Xva), S, 2)
        m = float(((((pred - Yva) ** 2).sum(-1)).sum()) / (len(Xva) * S))
        best = min(best, m)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--run", default=".venv_xpu/runs/r8/best")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--ntrain", type=int, default=3000)
    ap.add_argument("--nval", type=int, default=400)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    from peft_compat import load_adapter_compat

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    model, head, hidden = T.load_model(a.model, 128, dev, act_dim=3, lora_r=8)
    try:
        info = load_adapter_compat(model, Path(a.run) / "adapter", name="trained")
        print("[load] %s" % info, flush=True)
    except Exception as e:  # noqa: BLE001
        print("[warn] adapter: %r" % (e,), flush=True)
    model.eval()

    proc = AutoProcessor.from_pretrained(a.model)
    px = 448 * 256
    alls = T.build_index(Path(a.dataset), None)
    tr = [s for s in alls if s["split"] == "train"][: a.ntrain]
    va = T.select_val_subset(Path(a.dataset), a.nval)

    def feats(v):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            o[i, :2] = wp; o[i, 2:5] = tp; o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    def collect(v, want_pool: bool):
        """返回 (pooled, 数值, 标签, mask)。pooled 为 None 表示不要。"""
        ds = T.PolicyDataset(v, proc, px, px)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        P, A, M = [], [], []
        with torch.no_grad():
            for b in dl:
                kw = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in b["inputs"].items()}
                inner = model.base_model.model.model
                out = inner(**kw, use_cache=False)
                h = out.last_hidden_state
                idx = kw["attention_mask"].sum(-1) - 1
                ar = torch.arange(h.shape[0], device=h.device)
                P.append(h[ar, idx].float().cpu().numpy())
                A.append(b["act"].numpy()); M.append(b["mask"].numpy())
        P = np.concatenate(P) if want_pool else None
        return P, feats(v), np.concatenate(A), np.concatenate(M)

    print("[extract] train %d / val %d" % (len(tr), len(va)), flush=True)
    Ptr, Ntr, Atr, Mtr = collect(tr, True)
    Pva, Nva, Ava, Mva = collect(va, True)

    okt = Mtr[:, :S].sum(-1) == S
    okv = Mva[:, :S].sum(-1) == S
    Ytr = Atr[okt][:, :S, :2].reshape(int(okt.sum()), -1)
    Yva = Ava[okv][:, :S, :2]
    Ptr, Ntr = Ptr[okt], Ntr[okt]
    Pva, Nva = Pva[okv], Nva[okv]
    print("[mask] train_ok=%d val_ok=%d" % (okt.sum(), okv.sum()), flush=True)

    def prep(Xtr, Xva, use_quad=False):
        # ⚠️ 形参不能叫 quad —— 会遮蔽模块级函数 quad()，实测报
        # TypeError: 'bool' object is not callable
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = (Xtr - mu) / sd; B = (Xva - mu) / sd
        return (quad(A), quad(B)) if use_quad else (A, B)

    print()
    print("  同一批 val 上，不同特征源的 ridge 可达误差（前 8 步，masked）")
    cases = [
        ("只用 8 数值 (线性)", Ntr, Nva, False),
        ("只用 8 数值 (二次)", Ntr, Nva, True),
        ("只用 pooled 1024 (线性)", Ptr, Pva, False),
        ("pooled + 数值 (线性)", np.concatenate([Ptr, Ntr], 1), np.concatenate([Pva, Nva], 1), False),
    ]
    for tag, Xt, Xv, q in cases:
        Tt, Vv = prep(Xt, Xv, use_quad=q)
        m = ridge_eval(Tt, Ytr, Vv, Yva)
        print("  %-28s dim=%-5d mse8 = %.4f" % (tag, Tt.shape[1], m))

    print()
    print("  参照：copy 基线 %.4f / 训练日志里的 move8 (12步口径) 1.4823"
          % float(((((np.repeat(Yva[:, :1], S, axis=1) - Yva) ** 2).sum(-1)).sum())
                  / (len(Yva) * S)))
    print()
    print("  判读：")
    print("   * pooled 能到 ~1.24 ⇒ 信息在，是训练/读出问题")
    print("   * pooled 明显差于数值 ⇒ 信息在主干里丢了，或图像干扰")
    print("   * pooled+数值 ≈ 数值单用 ⇒ pooled 对「怎么走」几乎无信息")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
