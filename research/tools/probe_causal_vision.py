# -*- coding: utf-8 -*-
"""只读：causal 标签 + v3far 图像 = 视觉能补多少？

v4causal 是 --no-images 构建的（无图像），但它的 obs 里的 img_L 相对路径
与 v3far 完全一致（已核对前两行逐字相同）。所以可以把 v4causal 的**因果标签**
和 v3far 的**真实图像文件**配对，得到「去未来 + 有视觉」的组合。

对照三组（同一 LWO 10 折、同一批样本）：
  A 因果数值           v4causal 特征（wp=0, topo=历史）
  B 因果数值 + 图像    + ViT pooled 1024
  C 旧数值 + 图像      v3far（含未来 wp）
"""
import os
import sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
sys.path.insert(0, "research/tools")
import train_policy_lora as T

V4 = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v4causal")
V3 = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
CACHE = _REPO / ".tmp" / "feat_causal_lwo.npz"
S = 8
PER_FOLD = 240


def m_of(p, a, m):
    sq = ((p - a) ** 2).sum(-1)
    return float((sq * m).sum() / max(m.sum(), 1.0))


def feats(v):
    o = np.zeros((len(v), 8))
    for i, x in enumerate(v):
        o[i, :2] = x["wp"]
        o[i, 2:5] = x.get("topo") or (0, 0, 0)
        o[i, 5:7] = x.get("body") or (0, 0)
        o[i, 7] = float(x.get("mode", 0))
    return o


if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    H, SID, NX, A, M = z["H"], z["SID"], z["NX"], z["A"], z["M"]
    print("[cache] %s  H=%s" % (CACHE.name, H.shape))
else:
    import torch
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    proc = AutoProcessor.from_pretrained("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                      128, "xpu", act_dim=3)
    model.eval()
    inner = model.base_model.model.model
    px = 448 * 256
    causal = T.build_index(V4, None)
    sids = sorted({x["sid"] for x in causal})
    Hs, Ss, Ns, As, Ms = [], [], [], [], []
    for sid in sids:
        cv = [x for x in causal if x["sid"] == sid]
        if len(cv) > PER_FOLD:
            step = len(cv) / PER_FOLD
            cv = [cv[int(i * step)] for i in range(PER_FOLD)]
        # 把 img 路径指到 v3far（v4causal 没有图）
        for x in cv:
            rel = Path(x["img"]).relative_to(V4)
            x["img"] = V3 / rel
            x["prev_img"] = V3 / Path(x["prev_img"]).relative_to(V4)
        ds = T.PolicyDataset(cv, proc, px, px, drop_wp=False, quad_feats=False)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        P = []
        with torch.no_grad():
            for b in dl:
                kw = {k: (t.to("xpu") if torch.is_tensor(t) else t) for k, t in b["inputs"].items()}
                out = inner(**kw, use_cache=False)
                h = out.last_hidden_state
                idx = kw["attention_mask"].sum(-1) - 1
                ar = torch.arange(h.shape[0], device=h.device)
                P.append(h[ar, idx].float().cpu().numpy())
        Hs.append(np.concatenate(P))
        Ss.extend([sid] * len(cv))
        Ns.append(feats(cv))
        As.append(np.stack([np.asarray(x["act"])[:S] for x in cv]))
        Ms.append(np.stack([np.asarray(x["mask"])[:S] for x in cv]))
        print("  [collect] %s n=%d" % (sid, len(cv)), flush=True)
    H = np.concatenate(Hs); SID = np.array(Ss)
    NX = np.concatenate(Ns); A = np.concatenate(As); M = np.concatenate(Ms)
    np.savez_compressed(CACHE, H=H, SID=SID, NX=NX, A=A, M=M)

print("[data] %d 样本 / %d 世界" % (len(H), len(set(SID.tolist()))))


def run(X, tag):
    fold, foldc = [], []
    sids = sorted(set(SID.tolist()))
    for hold in sids:
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
        for lam in (1e-2, 1.0, 10.0, 100.0, 1000.0):
            W = np.linalg.solve(Xt.T@Xt + lam*np.eye(Xt.shape[1]), Xt.T@Y)
            p = (Xv@W).reshape(len(Xv), S, 2)
            e = m_of(p, Av, Mv)
            if e < best[0]: best = (e, p)
        fold.append(best[0])
        foldc.append(m_of(np.repeat(Av[:, :1], S, axis=1), Av, Mv))
    a = np.array(fold); c = np.array(foldc)
    print("  %-30s spd=%.4f±%.4f  copy=%.4f  vs copy %+.1f%%"
          % (tag, a.mean(), a.std(), c.mean(), 100*(a.mean()/c.mean()-1)))


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


print()
print("=" * 86)
print("  LWO 10 折（%d/折，前 %d 步速度）" % (PER_FOLD, S))
print("=" * 86)
mu0, sd0 = NX.mean(0), NX.std(0) + 1e-9
run(quad((NX - mu0)/sd0), "A 因果数值（44维二次）")
mu1, sd1 = H.mean(0), H.std(0) + 1e-9
# ⚠️ 2026-11-05 修：B 臂此前写成 run((H-mu1)/sd1)，与 D 臂（run(H)）**是同一段代码**，
# 所以 B 和 D 逐位相同（2.2227±0.7071）——那不是「视觉没用」，是 B 臂根本没加数值。
run(quad(np.concatenate([(H-mu1)/sd1, (NX-mu0)/sd0], axis=1)),
    "B 因果数值+图像（1068维，数值走二次）")
run(np.concatenate([(H-mu1)/sd1, (NX-mu0)/sd0], axis=1), "C 数值+图像拼接（1074维）")
run(H, "D 仅图像（1024维）")
