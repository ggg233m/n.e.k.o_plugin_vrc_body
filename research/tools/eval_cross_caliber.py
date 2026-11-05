# -*- coding: utf-8 -*-
"""严格交叉评测：2 模型 × 2 标签口径。

为什么需要它（2026-11-05）：
  前一轮用 `v5omega30`（新 30 Hz 标签）同时评两个模型，对 `r8_nowp`（旧标签训练）
  是**跨口径评分**——它的 ω 天然吃亏。要分离「模型差异」和「标签口径差异」，
  必须让**两个模型都在两套标签下各评一次**。

严格性保证（已实测）：
  * `select_val_subset(v3far, 400)` 与 `select_val_subset(v5omega30, 400)` 的
    (sid, t) **完全一致**（400/400）
  * 两数据集的图像像素 **SHA256 相同**（同一批 JPEG）
  * 速度标签 **完全相同**（逐位）
  * 只有 ω 标签不同（平均绝对差 0.3987）
  ⇒ 唯一变量是「用哪套 ω 标签评分」，满足严格对照。

⚠️ 输入构造注意：两个 run 都是 `drop_wp=True`，所以 v3far 的 far-wp 与
   v5omega30 的 causal-wp **都被置 0**，输入口径一致。这是严格性的前提，
   若将来有 run 用 drop_wp=False，本脚本会打印警告。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "research" / "tools"))

S = 8
OMEGA_THR = 0.3


def mse(p, a, m):
    sq = ((p - a) ** 2).sum(-1)
    return float((sq * m).sum() / max(m.sum(), 1.0))


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


def event_recall(pred, act, mask):
    tp = tot = 0
    for i in range(len(act)):
        m = mask[i] > 0
        if m.sum() == 0:
            continue
        ea = events(act[i][m], OMEGA_THR)
        if not ea:
            continue
        ep = events(pred[i][m], OMEGA_THR)
        for x in ea:
            tot += x[1] - x[0]
            for y in ep:
                tp += max(0, min(x[1], y[1]) - max(x[0], y[0]))
    return tp / max(tot, 1)


def metrics(pred, act, mask):
    r = {}
    r["mse_all"] = mse(pred, act, mask)
    r["mse_speed"] = mse(pred[..., :2], act[..., :2], mask)
    r["mse_omega"] = mse(pred[..., 2:3], act[..., 2:3], mask)
    r["event_recall"] = float(event_recall(pred[..., 2], act[..., 2], mask))
    pa, aa = [], []
    for i in range(len(act)):
        m = mask[i] > 0
        if m.sum() == 0:
            continue
        a = np.abs(act[i][m, 2])
        p = np.abs(pred[i][m, 2])
        if a.max() > 1e-6:
            aa.append(a.max())
            pa.append(p.max())
    r["peak_ratio"] = float(np.mean(pa) / max(np.mean(aa), 1e-9)) if aa else float("nan")
    return r


def run_one(run_dir, ds_root, model_dir, n, dev="xpu"):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    from peft_compat import load_adapter_compat

    run = Path(run_dir)
    proc = AutoProcessor.from_pretrained(model_dir)
    px = 448 * 256
    val = T.select_val_subset(Path(ds_root), n)
    ra = json.loads((run.parent / "args.json").read_text(encoding="utf-8"))
    dw = ra.get("drop_wp", False)
    if not dw:
        print("  [WARN] drop_wp=False，两数据集 wp 口径不同，严格性受损", flush=True)

    sd = torch.load(run / "head.pt", map_location="cpu")
    ad = 3 if sd["fc.weight"].shape[0] == 36 else 2
    model, head, _ = T.load_model(model_dir, 128, dev, act_dim=ad)
    load_adapter_compat(model, run / "adapter", name="trained")
    head.load_state_dict(sd)
    head.to(dev).eval()
    model.eval()

    ds = T.PolicyDataset(val, proc, px, px, drop_wp=dw,
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
    P = np.concatenate(Ps)[:, :S, :]
    A = np.concatenate(As)[:, :S, :]
    M = np.concatenate(Ms)[:, :S]
    return metrics(P, A, M)


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
        tag_run = Path(rd).parent.name
        for ds in a.datasets:
            tag_ds = Path(ds).name.replace("m0_policy_448_", "")
            key = tag_run + " @ " + tag_ds
            print("[eval] " + key, flush=True)
            res[key] = run_one(rd, ds, a.model, a.n)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    keys = list(res)
    print()
    print("  %-22s" % "metric" + "".join("%22s" % k for k in keys))
    for m in ("mse_all", "mse_speed", "mse_omega", "event_recall", "peak_ratio"):
        print("  %-22s" % m + "".join("%22.4f" % res[k].get(m, float("nan")) for k in keys))
    return 0


raise SystemExit(main())
