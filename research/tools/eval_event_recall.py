# -*- coding: utf-8 -*-
"""事件召回评测：转向事件 / stop-look 事件的召回与峰值保留。

与 eval_omega30_ab 互补：后者报 MSE，本脚本报**事件级**指标
（MSE 对「事件被漏掉」不敏感，见 omega30 报告的 §6）。
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


def events(sig, thr):
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


def overlap(a, b):
    """两个区间的重叠长度。"""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def event_recall(pred, act, mask):
    """按样本（时间轴）算召回：真值事件有多少被预测事件覆盖。"""
    tp = tot = 0
    for i in range(len(act)):
        m = mask[i] > 0
        if m.sum() == 0:
            continue
        a = act[i][m]
        p = pred[i][m]
        ev_a = events(a, OMEGA_THR)
        if not ev_a:
            continue
        ev_p = events(p, OMEGA_THR)
        for ea in ev_a:
            tot += ea[1] - ea[0]
            for ep in ev_p:
                tp += overlap(ea, ep)
    return tp / max(tot, 1), tot


def peak_ratio(pred, act, mask):
    """预测峰值 / 真值峰值（1.0 = 完美保幅，<1 = 削峰）。"""
    ra, rp = [], []
    for i in range(len(act)):
        m = mask[i] > 0
        if m.sum() == 0:
            continue
        a = np.abs(act[i][m])
        p = np.abs(pred[i][m])
        if a.max() > 1e-6:
            ra.append(a.max())
            rp.append(p.max())
    if not ra:
        return float("nan")
    return float(np.mean(rp) / max(np.mean(ra), 1e-9))


def evaluate(run_dir, model_dir, dataset, n, batch, dev="xpu"):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    from peft_compat import load_adapter_compat

    run = Path(run_dir)
    proc = AutoProcessor.from_pretrained(model_dir)
    px = 448 * 256
    val = T.select_val_subset(Path(dataset), n)

    ra = {}
    ap = run.parent / "args.json"
    if ap.exists():
        ra = json.loads(ap.read_text(encoding="utf-8"))
    sd = torch.load(run / "head.pt", map_location="cpu")
    ad = 3 if sd["fc.weight"].shape[0] == 36 else 2
    model, head, _ = T.load_model(model_dir, 128, dev, act_dim=ad)
    load_adapter_compat(model, run / "adapter", name="trained")
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
    P = np.concatenate(Ps)[:, :S, 2]
    A = np.concatenate(As)[:, :S, 2]
    M = np.concatenate(Ms)[:, :S]
    rec, tot = event_recall(P, A, M)
    return dict(run=str(run), n=int(len(P)),
                turn_event_recall=float(rec), truth_event_steps=int(tot),
                peak_ratio=peak_ratio(P, A, M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for r in a.runs:
        print("[eval] %s" % r, flush=True)
        res[Path(r).parent.name] = evaluate(r, a.model, a.dataset, a.n, a.batch)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    print("  %-24s %14s %14s" % ("metric", *[k for k in res]))
    for m in ("n", "truth_event_steps", "turn_event_recall", "peak_ratio"):
        print("  %-24s" % m + "".join("%14.4f" % res[k].get(m, float("nan")) for k in res))
    return 0


raise SystemExit(main())
