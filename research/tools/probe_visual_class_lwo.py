# -*- coding: utf-8 -*-
"""逐类 + 图像 pooled 特征的 LWO 对照。

回答唯一那个问题：**图像能不能让 move 类打过 copy？**
用 0.8B 主干最后一层 hidden（含视觉 token）做 ridge，按世界留出。
与 probe_class_breakdown 同一口径（前 8 步、masked_mse、copy 基线）。
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
_ONEAPI = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
if os.path.isdir(_ONEAPI):
    os.environ["PATH"] = _ONEAPI + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))

S = 8

def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)

def mse_of(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1e-9))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--per-fold", type=int, default=260, help="每折最多多少 val 样本（控制耗时）")
    ap.add_argument("--cache", default=str(_REPO / ".tmp/feat_cache_lwo.npz"))
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    dev = "xpu"
    px = 448 * 256
    alls = T.build_index(Path(a.dataset), None)

    cache = Path(a.cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        H, A, M, CLS, SID, OK = z["H"], z["A"], z["M"], z["CLS"], z["SID"], z["OK"]
        print("[cache] 载入 %s  H=%s" % (cache.name, H.shape))
    else:
        proc = AutoProcessor.from_pretrained(a.model)
        model, head, hidden = T.load_model(a.model, 128, dev, act_dim=3)
        model.eval()
        inner = model.base_model.model.model
        sids = sorted({x["sid"] for x in alls})
        Hs, As, Ms, Cs, Ss, Os = [], [], [], [], [], []
        for sid in sids:
            va = [x for x in alls if x["sid"] == sid]
            if len(va) > a.per_fold:
                step = len(va) / a.per_fold
                va = [va[int(i * step)] for i in range(a.per_fold)]
            ds = T.PolicyDataset(va, proc, px, px, quad_feats=False)
            dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                            collate_fn=T.make_collate(proc, px, px))
            P = []
            with torch.no_grad():
                for b in dl:
                    kw = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in b["inputs"].items()}
                    out = inner(**kw, use_cache=False)
                    h = out.last_hidden_state
                    idx = kw["attention_mask"].sum(-1) - 1
                    ar = torch.arange(h.shape[0], device=h.device)
                    P.append(h[ar, idx].float().cpu().numpy())
            H = np.concatenate(P)
            A = np.stack([np.asarray(x["act"])[:S, :2] for x in va])
            M = np.stack([np.asarray(x["mask"])[:S] for x in va])
            Hs.append(H); As.append(A); Ms.append(M)
            Cs.extend([x["cls"] for x in va])
            Ss.extend([sid] * len(va))
            Os.extend([x["split"] for x in va])
            print("  [collect] %s n=%d" % (sid, len(va)), flush=True)
        H = np.concatenate(Hs); A = np.concatenate(As); M = np.concatenate(Ms)
        CLS = np.array(Cs); SID = np.array(Ss); OK = np.array(Os) == "train"
        np.savez_compressed(cache, H=H, A=A, M=M, CLS=CLS, SID=SID, OK=OK)
        print("[cache] 写出 %s" % cache)

    sids = sorted(set(SID.tolist()))
    print("[data] 样本 %d  世界 %d" % (len(H), len(sids)))

    def feats_visual(i):
        return H[i]

    # LWO：每折用其余世界的 train 样本训练，留出世界的样本验证
    Pr, Ar, Mr, Cr = [], [], [], []
    for hold in sids:
        trm = (SID != hold) & OK
        vam = (SID == hold)
        if trm.sum() < 50 or vam.sum() < 10:
            continue
        okt = M[trm][:, :S].sum(-1) == S
        Xtr = H[trm][okt]
        Ytr = A[trm][okt][:, :S].reshape(int(okt.sum()), -1)
        Xva = H[vam]
        Ava = A[vam][:, :S]
        Mva = M[vam][:, :S]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        best = (1e18, None)
        for lam in (1e-2, 1.0, 10.0, 100.0, 1000.0, 1e4):
            W = np.linalg.solve(((Xtr-mu)/sd).T @ ((Xtr-mu)/sd) + lam*np.eye(Xtr.shape[1]),
                                ((Xtr-mu)/sd).T @ Ytr)
            p = (((Xva-mu)/sd) @ W).reshape(len(Xva), S, 2)
            v = mse_of(p, Ava, Mva)
            if v < best[0]:
                best = (v, p)
        Pr.append(best[1]); Ar.append(Ava); Mr.append(Mva); Cr.extend(CLS[vam].tolist())

    P = np.concatenate(Pr); Aa = np.concatenate(Ar); Mk = np.concatenate(Mr); C = np.array(Cr)
    copy = np.repeat(Aa[:, :1], S, axis=1)
    print()
    print("  %-6s %7s %8s %9s %9s %8s" % ("class","n","mask%","visual","copy","vs copy"))
    for cl in ("idle", "move", "trans", "ALL"):
        sel = np.ones(len(C), bool) if cl == "ALL" else (C == cl)
        if sel.sum() == 0:
            continue
        m = mse_of(P[sel], Aa[sel], Mk[sel])
        c = mse_of(copy[sel], Aa[sel], Mk[sel])
        w = 100 * Mk[sel].sum() / Mk.sum()
        rel = "   n/a" if c < 1e-6 else "%+7.1f%%" % (100*(m-c)/c)
        print("  %-6s %7d %7.1f%% %9.4f %9.4f %s" % (cl, int(sel.sum()), w, m, c, rel))
    return 0

raise SystemExit(main())
