# -*- coding: utf-8 -*-
"""消融四组 × 两套标签 × 行为分层：完整评测矩阵。

在同一批 400 个 causal val 样本上，对 A/B/C/D 四个模型分别在
  mean 标签（v5omega30）与 peak 标签（v6peak）下评分；
  并按行为类（turn_in_place / walk_turn / walk_straight / still / stop-look）分层。

所有 run 都是 causal topo + drop_wp=True，唯一差异是：
  A: mean 标签 + omega_norm
  B: peak 标签 + omega_norm
  C: mean 标签 + no-omega-norm
  D: peak 标签 + no-omega-norm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "research" / "tools"))

S = 8
OMEGA_THR = 0.3
V_STILL = 0.1


def mse(p, a, m):
    sq = ((p - a) ** 2).sum(-1)
    return float((sq * m).sum() / max(m.sum(), 1.0))


def ev(sig, thr):
    on = np.abs(sig) > thr
    if not on.any():
        return []
    d = np.diff(on.astype(np.int8))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if on[0]:
        s = [0] + s
    if on[-1]:
        e = e + [len(on)]
    return list(zip(s, e))


def recall(pred, act, mask):
    tp = tot = 0
    for i in range(len(act)):
        m = mask[i] > 0
        if m.sum() == 0:
            continue
        ea = ev(act[i][m], OMEGA_THR)
        if not ea:
            continue
        ep_ = ev(pred[i][m], OMEGA_THR)
        for x in ea:
            tot += x[1] - x[0]
            for y in ep_:
                tp += max(0, min(x[1], y[1]) - max(x[0], y[0]))
    return tp / max(tot, 1)


def pk_ratio(pred, act, mask):
    rs, ps = [], []
    for i in range(len(act)):
        m = mask[i] > 0
        if m.sum() == 0:
            continue
        a = np.abs(act[i][m])
        p = np.abs(pred[i][m])
        if a.max() > 1e-6:
            rs.append(a.max())
            ps.append(p.max())
    return float(np.mean(ps) / max(np.mean(rs), 1e-9)) if rs else float("nan")


def run_one(run_dir, ds_root, model_dir, n, dev="xpu"):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    import policy_classify as C
    from peft_compat import load_adapter_compat

    run = Path(run_dir)
    proc = AutoProcessor.from_pretrained(model_dir)
    px = 448 * 256
    val = T.select_val_subset(Path(ds_root), n)
    C.relabel(val)
    ra = json.loads((run.parent / "args.json").read_text(encoding="utf-8"))
    sd = torch.load(run / "head.pt", map_location="cpu")
    ad = 3 if sd["fc.weight"].shape[0] == 36 else 2
    model, head, _ = T.load_model(model_dir, 128, dev, act_dim=ad)
    load_adapter_compat(model, run / "adapter", name="trained")
    head.load_state_dict(sd)
    head.to(dev).eval()
    model.eval()

    ds = T.PolicyDataset(val, proc, px, px, drop_wp=ra.get("drop_wp", False),
                         quad_feats=ra.get("quad_feats", False))
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))
    Ps, As, Ms = [], [], []
    with torch.no_grad():
        for b in dl:
            o = T.forward_batch(model, head, b["inputs"], dev)
            Ps.append(o.float().cpu().numpy())
            As.append(b["act"].numpy())
            Ms.append(b["mask"].numpy())
    P_ = np.concatenate(Ps)[:, :S, :]
    A_ = np.concatenate(As)[:, :S, :]
    M_ = np.concatenate(Ms)[:, :S]
    cls = np.array([x["cls2"] for x in val])

    r = dict(n=int(len(P_)),
             mse_all=mse(P_, A_, M_),
             mse_speed=mse(P_[..., :2], A_[..., :2], M_),
             mse_omega=mse(P_[..., 2:3], A_[..., 2:3], M_),
             event_recall=recall(P_[..., 2], A_[..., 2], M_),
             peak_ratio=pk_ratio(P_[..., 2], A_[..., 2], M_))
    for c in ("turn_in_place", "walk_turn", "walk_straight", "still"):
        sel = cls == c
        if sel.sum() < 3:
            continue
        r["n_" + c] = int(sel.sum())
        r["mse_all_" + c] = mse(P_[sel], A_[sel], M_[sel])
        r["mse_omega_" + c] = mse(P_[sel][..., 2:3], A_[sel][..., 2:3], M_[sel])
    spd0 = np.linalg.norm(A_[:, 0, :2], axis=1)
    omg0 = np.abs(A_[:, 0, 2])
    look = (spd0 < V_STILL) & (omg0 > OMEGA_THR)
    if look.sum() >= 3:
        r["n_stoplool"] = int(look.sum())
        r["mse_all_stoplool"] = mse(P_[look], A_[look], M_[look])
        r["mse_omega_stoplool"] = mse(P_[look][..., 2:3], A_[look][..., 2:3], M_[look])
    cp = np.repeat(A_[:, :1], S, axis=1)
    r["copy_all"] = mse(cp, A_, M_)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = {}
    for rd in a.runs:
        rn = Path(rd).parent.name
        for ds in a.datasets:
            dn = Path(ds).name.replace("m0_policy_448_", "")
            k = rn + " @ " + dn
            print("[eval] " + k, flush=True)
            res[k] = run_one(rd, ds, a.model, a.n)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    keys = list(res)
    metrics = ["n", "mse_all", "mse_speed", "mse_omega", "event_recall", "peak_ratio",
               "copy_all",
               "n_turn_in_place", "mse_all_turn_in_place", "mse_omega_turn_in_place",
               "n_walk_turn", "mse_all_walk_turn", "mse_omega_walk_turn",
               "n_walk_straight", "mse_all_walk_straight",
               "n_still", "mse_all_still",
               "n_stoplool", "mse_all_stoplool", "mse_omega_stoplool"]
    print()
    hdr = "  %-30s" % "metric" + "".join("%16s" % k.split(" @ ")[0] + "/" + k.split(" @ ")[1][:4] for k in keys)
    for m in metrics:
        if not any(m in res[k] for k in keys):
            continue
        row = "  %-30s" % m
        for k in keys:
            v = res[k].get(m)
            row += "%22s" % ("-" if v is None else ("%.4f" % v if isinstance(v, float) else str(v)))
        print(row)
    return 0


raise SystemExit(main())
