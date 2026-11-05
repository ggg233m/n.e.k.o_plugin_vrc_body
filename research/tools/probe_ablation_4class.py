# -*- coding: utf-8 -*-
"""视觉消融 x 正交 4 类（still / turn_in_place / walk_straight / walk_turn）。

替换掉旧的 idle/move/trans 口径（那个只看平移，对视觉有结构性负向偏置）。
口径与训练侧一致：mse8 只取速度两维（train_policy_lora.masked_mse）。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "research/tools")

import train_policy_lora as T
import policy_classify as C
from peft_compat import load_adapter_compat
import eval_vision_ablation as E


def mse8(pred, act, mask):
    err = ((pred[:, :8, :2] - act[:, :8, :2]) ** 2).sum(-1)
    m = mask[:, :8]
    return float((err * m).sum() / max(m.sum(), 1.0))


def mse_omega(pred, act, mask):
    """ω 单列：这是新分类真正的意义——still/turn_in_place 的差别全在这一维。"""
    if pred.shape[-1] < 3:
        return float("nan")
    err = ((pred[:, :8, 2] - act[:, :8, 2]) ** 2)
    m = mask[:, :8]
    return float((err * m).sum() / max(m.sum(), 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=".venv_xpu/runs/r8_nowp/epoch_00")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    from transformers import AutoProcessor
    dev = "xpu"
    proc = AutoProcessor.from_pretrained(a.model)
    px = 448 * 256
    val = T.select_val_subset(Path(a.dataset), a.n)
    C.relabel(val)

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
    modes = ("base", "black", "shuffle")
    loaders = {m: torch.utils.data.DataLoader(
                   E.AblateDataset(ds, a.n, m), batch_size=a.batch, shuffle=False,
                   num_workers=0, collate_fn=collate)
               for m in modes}
    print("[caliber] drop_wp=%s run=%s" % (run_args.get("drop_wp", False), run.name), flush=True)

    res = {}
    for m in modes:
        pred, act, mask = E.predict(model, head, loaders[m], dev, injector=None,
                                    want_labels=True)
        res[m] = (pred, act, mask)
        print("  [%s] done" % m, flush=True)

    _, act, mask = res["base"]
    cls = np.array([x["cls2"] for x in val])
    print()
    print("  %-16s %5s %8s %8s %8s %9s %10s %9s" %
          ("class", "n", "base", "black", "shuffle", "w_base", "real-black", "real-shuf"))
    for c in C.CLASSES + ("ALL",):
        sel = np.ones(len(cls), bool) if c == "ALL" else (cls == c)
        if sel.sum() == 0: continue
        b = mse8(res["base"][0][sel], act[sel], mask[sel])
        bl = mse8(res["black"][0][sel], act[sel], mask[sel])
        sh = mse8(res["shuffle"][0][sel], act[sel], mask[sel])
        wb = mse_omega(res["base"][0][sel], act[sel], mask[sel])
        print("  %-16s %5d %8.4f %8.4f %8.4f %9.4f %+9.1f%% %+8.1f%%" %
              (c, int(sel.sum()), b, bl, sh, wb,
               100*(bl-b)/max(b,1e-9), 100*(sh-b)/max(b,1e-9)))
    return 0


raise SystemExit(main())
