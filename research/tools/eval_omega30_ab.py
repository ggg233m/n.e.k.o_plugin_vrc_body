# -*- coding: utf-8 -*-
"""正式离线评测：omega30 训练 run vs 旧 0.4s/单点ω run，同批对照。

固定样本顺序（select_val_subset），指标：
  整体 masked MSE / speed / omega / turn_in_place / walk_turn / stop-look / copy 基线
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


def mse(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1.0))


def evaluate(run_dir, model_dir, dataset, n, batch, dev="xpu"):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    import policy_classify as C
    from peft_compat import load_adapter_compat

    run = Path(run_dir)
    ds_root = Path(dataset)
    proc = AutoProcessor.from_pretrained(model_dir)
    px = 448 * 256

    # 固定样本（两组完全同一批、同一顺序）
    val = T.select_val_subset(ds_root, n)
    C.relabel(val)

    ra_path = run.parent / "args.json"
    ra = json.loads(ra_path.read_text(encoding="utf-8")) if ra_path.exists() else {}
    sd = torch.load(run / "head.pt", map_location="cpu")
    ad = 3 if sd["fc.weight"].shape[0] == 36 else 2
    model, head, _ = T.load_model(model_dir, 128, dev, act_dim=ad)
    info = load_adapter_compat(model, run / "adapter", name="trained")
    head.load_state_dict(sd)
    head.to(dev).eval()
    model.eval()

    ds = T.PolicyDataset(val, proc, px, px,
                         drop_wp=ra.get("drop_wp", False),
                         quad_feats=ra.get("quad_feats", False))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))
    Ps, As, Ms = [], [], []
    with torch.no_grad():
        for b in dl:
            o = T.forward_batch(model, head, b["inputs"], dev)
            Ps.append(o.float().cpu().numpy())
            As.append(b["act"].numpy())
            Ms.append(b["mask"].numpy())
    P = np.concatenate(Ps)[:, :S, :]
    A = np.concatenate(As)[:, :S, :]
    M = np.concatenate(Ms)[:, :S]
    cls = np.array([x["cls2"] for x in val])

    res = dict(run=str(run), n=int(len(P)), adapter=info, act_dim=ad)
    res["mse_all"] = mse(P, A, M)
    res["mse_speed"] = mse(P[..., :2], A[..., :2], M)
    res["mse_omega"] = mse(P[..., 2:3], A[..., 2:3], M)
    cp = np.repeat(A[:, :1], S, axis=1)
    res["copy_all"] = mse(cp, A, M)
    res["copy_speed"] = mse(cp[..., :2], A[..., :2], M)
    for c in ("turn_in_place", "walk_turn", "still", "walk_straight"):
        sel = cls == c
        if sel.sum() < 3:
            continue
        res["mse_all_" + c] = mse(P[sel], A[sel], M[sel])
        res["mse_omega_" + c] = mse(P[sel][..., 2:3], A[sel][..., 2:3], M[sel])
    spd0 = np.linalg.norm(A[:, 0, :2], axis=1)
    omg0 = np.abs(A[:, 0, 2])
    look = (spd0 < 0.1) & (omg0 > 0.3)
    if look.sum() >= 3:
        res["mse_all_stoplool"] = mse(P[look], A[look], M[look])
        res["n_stoplool"] = int(look.sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    all_res = {}
    for r in a.runs:
        print("[eval] %s" % r, flush=True)
        all_res[Path(r).parent.name] = evaluate(r, a.model, a.dataset, a.n, a.batch)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_res, indent=1, ensure_ascii=False), encoding="utf-8")

    keys = [k for k in all_res];
    print()
    hdr = "  %-24s" % "metric" + "".join("%14s" % k for k in keys)
    print(hdr)
    base = list(all_res.values())[0]
    for m in base:
        if not isinstance(base[m], (int, float)) or isinstance(base[m], bool):
            continue
        line = "  %-24s" % m + "".join("%14.4f" % all_res[k].get(m, float("nan")) for k in keys)
        print(line)
    return 0


raise SystemExit(main())
