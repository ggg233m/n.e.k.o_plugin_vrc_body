# -*- coding: utf-8 -*-
"""配对显著性检验：histframe 0.2s vs 0.4s（同一 checkpoint、同一批样本）。"""
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def pick_prev(samples, gap_s, obs_by_sid):
    """按 gap_s 重选 prev_img（与 eval_histframe_controlled.pick_prev 同一实现）。

    为什么不 import：eval_histframe_controlled 在模块级 raise SystemExit(main())，
    import 会先触发 argparse 报错（实测 exit code 2）。这里内联，避免副作用。
    """
    out = []
    for s in samples:
        tt = float(s["t"])
        o = obs_by_sid[s["sid"]]
        ts, imgs = o["t"], o["img"]
        j = int(np.searchsorted(ts, tt - gap_s, side="right") - 1)
        j = max(0, min(j, len(ts) - 1))
        j2 = max(0, min(j + 1, len(ts) - 1))
        k = j if abs(ts[j] - (tt - gap_s)) <= abs(ts[j2] - (tt - gap_s)) else j2
        ok = abs(ts[k] - (tt - gap_s)) <= 0.15
        d = dict(s)
        if ok:
            d["prev_img"] = imgs[k]
        d["_prev_ok"] = bool(ok)
        out.append(d)
    return out


def per_sample_mse(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    w = np.maximum(mask.sum(-1), 1.0)
    return (sq * mask).sum(-1) / w


def main():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    import policy_classify as C
    from peft_compat import load_adapter_compat

    run = Path(".venv_xpu/runs/r8_nowp/epoch_00")
    root = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    mdl = "G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3"
    dev = "xpu"
    proc = AutoProcessor.from_pretrained(mdl)
    px = 448 * 256
    val = T.select_val_subset(root, 120)
    C.relabel(val)
    obs_by = {}
    for sid in sorted({x["sid"] for x in val}):
        raw = T.read_jsonl(root / sid / "policy_obs.jsonl")
        obs_by[sid] = dict(
            t=np.array([r["t"] for r in raw], dtype=np.float64),
            img=[root / sid / r["img_L"] for r in raw])
    ra = json.loads((run.parent / "args.json").read_text(encoding="utf-8"))
    sd = torch.load(run / "head.pt", map_location="cpu")
    ad = 3 if sd["fc.weight"].shape[0] == 36 else 2
    model, head, _ = T.load_model(mdl, 128, dev, act_dim=ad)
    load_adapter_compat(model, run / "adapter", name="trained")
    head.load_state_dict(sd)
    head.to(dev).eval()
    model.eval()

    per = {}
    for gap in (0.4, 0.2):
        sm = pick_prev(val, gap, obs_by)
        ds = T.PolicyDataset(sm, proc, px, px,
                             drop_wp=ra.get("drop_wp", False),
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
        P = np.concatenate(Ps)[:, :8, :]
        A = np.concatenate(As)[:, :8, :]
        M = np.concatenate(Ms)[:, :8]
        per[gap] = per_sample_mse(P, A, M)
        print("[gap=%.1f] mean=%.4f" % (gap, per[gap].mean()), flush=True)

    d = per[0.2] - per[0.4]
    n = int(len(d))
    mu = float(d.mean())
    sdv = float(d.std(ddof=1))
    se = sdv / sqrt(n)
    t = mu / max(se, 1e-12)
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    win = int((d < 0).sum())

    print()
    print("=" * 68)
    print("  配对检验 0.2s vs 0.4s（同 ckpt / 同样本 / 同顺序）")
    print("=" * 68)
    print("  n=%d  mean_diff=%.6f  sd=%.6f  SE=%.6f" % (n, mu, sdv, se))
    print("  t=%.3f  p=%.4f (two-sided)" % (t, p))
    print("  0.2s 更优样本 = %d/%d = %.1f%%" % (win, n, 100.0 * win / n))
    print("  -> %s" % ("显著 (p<0.05)" if p < 0.05 else "不显著 (p>=0.05)"))

    out = Path(".venv_xpu/runs/hist_ab/paired_test.json")
    out.write_text(json.dumps(dict(
        n=n, mean_04=float(per[0.4].mean()), mean_02=float(per[0.2].mean()),
        mean_diff=mu, sd_diff=sdv, se=se, t=t, p_two_sided=p, win=win,
        verdict=("significant" if p < 0.05 else "not_significant")),
        indent=1), encoding="utf-8")
    return 0


raise SystemExit(main())