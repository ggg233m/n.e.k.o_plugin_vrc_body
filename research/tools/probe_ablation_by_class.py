# -*- coding: utf-8 -*-
"""视觉消融 × 动作类：图像对 move 类到底有没有用。

复用 eval_vision_ablation 的四个条件（base/black/shuffle/same），
但按 idle/move/trans 拆开报 mse8。口径与训练侧一致（只取速度两维）。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "research/tools")

import train_policy_lora as T
from peft_compat import load_adapter_compat
import eval_vision_ablation as E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=".venv_xpu/runs/r8_nowp/epoch_00")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()

    from transformers import AutoProcessor
    dev = "xpu"
    proc = AutoProcessor.from_pretrained(a.model)
    px = 448 * 256
    alls = T.build_index(Path(a.dataset), None)
    val = T.select_val_subset(Path(a.dataset), a.n)

    run = Path(a.run)
    run_args = json.loads((run.parent / "args.json").read_text(encoding="utf-8"))
    head_sd = torch.load(run / "head.pt", map_location="cpu")
    act_dim = 3 if head_sd["fc.weight"].shape[0] == 36 else 2
    model, head, hidden = T.load_model(a.model, 128, dev, act_dim=act_dim)
    st = load_adapter_compat(model, run / "adapter", name="trained")
    print("[adapter] %s" % st, flush=True)
    head.load_state_dict(head_sd)
    head.to(dev).eval(); model.eval()

    ds = T.PolicyDataset(val, proc, px, px,
                         drop_wp=run_args.get("drop_wp", False),
                         quad_feats=run_args.get("quad_feats", False))
    collate = T.make_collate(proc, px, px)
    modes = ("base", "black", "shuffle", "same")
    loaders = {m: torch.utils.data.DataLoader(
                   E.AblateDataset(ds, a.n, m), batch_size=4, shuffle=False,
                   num_workers=0, collate_fn=collate)
               for m in modes}
    print("[caliber] drop_wp=%s" % run_args.get("drop_wp", False), flush=True)

    res = {}
    for m in ("base", "black", "shuffle", "same"):
        pred, act, mask = E.predict(model, head, loaders[m], dev, injector=None,
                                    want_labels=True)
        res[m] = (pred, act, mask)
        print("  [%s] done" % m, flush=True)

    _, act, mask = res["base"]
    cls = np.array([x["cls"] for x in val])
    print()
    print("  %-6s %6s %9s %9s %9s %9s %10s" %
          ("class", "n", "base", "black", "shuffle", "same", "real-vs-black"))
    for c in ("idle", "move", "trans", "ALL"):
        sel = np.ones(len(cls), bool) if c == "ALL" else (cls == c)
        if sel.sum() == 0: continue
        row = {}
        for m in ("base", "black", "shuffle", "same"):
            row[m] = E.mse8_of(res[m][0][sel], act[sel], mask[sel])
        gap = row["black"] - row["base"]
        print("  %-6s %6d %9.4f %9.4f %9.4f %9.4f %+9.4f (%+.1f%%)" %
              (c, int(sel.sum()), row["base"], row["black"], row["shuffle"],
               row["same"], gap, 100 * gap / max(row["base"], 1e-9)))
    return 0

raise SystemExit(main())
