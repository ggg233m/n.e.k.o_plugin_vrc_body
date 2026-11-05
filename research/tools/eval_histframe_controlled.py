# -*- coding: utf-8 -*-
"""受控离线评测：同一 checkpoint，只换历史视觉帧间隔 0.2 s vs 0.4 s。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "research" / "tools"))

S = 8


def pick_prev(samples, gap_s, obs_by_sid):
    out = []
    for s in samples:
        t = float(s["t"])
        o = obs_by_sid[s["sid"]]
        ts, imgs = o["t"], o["img"]
        j = int(np.searchsorted(ts, t - gap_s, side="right") - 1)
        j = max(0, min(j, len(ts) - 1))
        j2 = max(0, min(j + 1, len(ts) - 1))
        k = j if abs(ts[j] - (t - gap_s)) <= abs(ts[j2] - (t - gap_s)) else j2
        ok = abs(ts[k] - (t - gap_s)) <= 0.15
        d = dict(s)
        if ok:
            d["prev_img"] = imgs[k]
        d["_prev_ok"] = bool(ok)
        out.append(d)
    return out


def mse(pred, act, mask):
    sq = ((pred - act) ** 2).sum(-1)
    return float((sq * mask).sum() / max(mask.sum(), 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    import policy_classify as C
    from peft_compat import load_adapter_compat

    dev = "xpu"
    ds_root = Path(a.dataset)
    proc = AutoProcessor.from_pretrained(a.model)
    px = 448 * 256

    # 固定样本：用 select_val_subset 保证两组完全同一批、同一顺序
    val = T.select_val_subset(ds_root, a.n)
    C.relabel(val)
    print("[data] val n=%d" % len(val), flush=True)

    # 取每场全部 obs，供按不同 gap 重选 prev
    obs_by_sid = {}
    for sid in sorted({x["sid"] for x in val}):
        raw = T.read_jsonl(ds_root / sid / "policy_obs.jsonl")
        obs_by_sid[sid] = dict(
            t=np.array([r["t"] for r in raw], dtype=np.float64),
            img=[ds_root / sid / r["img_L"] for r in raw])

    run = Path(a.run)
    run_args = json.loads((run.parent / "args.json").read_text(encoding="utf-8"))
    head_sd = torch.load(run / "head.pt", map_location="cpu")
    act_dim = 3 if head_sd["fc.weight"].shape[0] == 36 else 2
    model, head, _ = T.load_model(a.model, 128, dev, act_dim=act_dim)
    info = load_adapter_compat(model, run / "adapter", name="trained")
    head.load_state_dict(head_sd)
    head.to(dev).eval(); model.eval()
    print("[ckpt] %s adapter=%s act_dim=%d" % (run.name, info, act_dim), flush=True)

    results = {}
    for gap in (0.4, 0.2):
        samples = pick_prev(val, gap, obs_by_sid)
        n_ok = sum(1 for s in samples if s["_prev_ok"])
        ds = T.PolicyDataset(samples, proc, px, px,
                             drop_wp=run_args.get("drop_wp", False),
                             quad_feats=run_args.get("quad_feats", False))
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        P_, A_, M_ = [], [], []
        with torch.no_grad():
            for b in dl:
                out = T.forward_batch(model, head, b["inputs"], dev)
                P_.append(out.float().cpu().numpy())
                A_.append(b["act"].numpy()); M_.append(b["mask"].numpy())
        P = np.concatenate(P_)[:, :S, :]
        Aa = np.concatenate(A_)[:, :S, :]
        Mk = np.concatenate(M_)[:, :S]
        cls = np.array([x["cls2"] for x in samples])
        res = dict(gap=gap, n=len(samples), prev_hit=n_ok)
        res["mse_all"] = mse(P, Aa, Mk)
        res["mse_speed"] = mse(P[..., :2], Aa[..., :2], Mk)
        res["mse_omega"] = mse(P[..., 2:3], Aa[..., 2:3], Mk)
        cp = np.repeat(Aa[:, :1], S, axis=1)
        res["copy_all"] = mse(cp, Aa, Mk)
        res["copy_speed"] = mse(cp[..., :2], Aa[..., :2], Mk)
        for c in ("turn_in_place", "walk_turn", "still"):
            sel = cls == c
            if sel.sum() < 3:
                continue
            res["mse_all_" + c] = mse(P[sel], Aa[sel], Mk[sel])
            res["mse_omega_" + c] = mse(P[sel][..., 2:3], Aa[sel][..., 2:3], Mk[sel])
        # stop-look：速度<0.1 且 |omega|>0.3
        spd0 = np.linalg.norm(Aa[:, 0, :2], axis=1)
        omg0 = np.abs(Aa[:, 0, 2])
        look = (spd0 < 0.1) & (omg0 > 0.3)
        if look.sum() >= 3:
            res["mse_all_stoplool"] = mse(P[look], Aa[look], Mk[look])
            res["n_stoplool"] = int(look.sum())
        results["gap_%.1f" % gap] = res
        print("[gap=%.1f] n=%d prev_hit=%d mse_all=%.4f speed=%.4f omega=%.4f"
              % (gap, res["n"], n_ok, res["mse_all"], res["mse_speed"], res["mse_omega"]),
              flush=True)
        del ds, dl

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    k2, k4 = results["gap_0.2"], results["gap_0.4"]
    print()
    print("=" * 84)
    print("  受控对照：同一 checkpoint，只换历史帧间隔")
    print("=" * 84)
    print("  %-22s %12s %12s %10s" % ("指标", "0.4s(旧)", "0.2s(新)", "变化"))
    for key in ("mse_all", "mse_speed", "mse_omega", "copy_all", "copy_speed"):
        v4, v2 = k4.get(key), k2.get(key)
        if v4 is None or v2 is None:
            continue
        print("  %-22s %12.4f %12.4f %+9.1f%%" % (key, v4, v2, 100 * (v2 - v4) / v4))
    for key in sorted(set(k4) & set(k2)):
        if not key.startswith(("mse_all_", "mse_omega_")):
            continue
        v4, v2 = k4[key], k2[key]
        print("  %-22s %12.4f %12.4f %+9.1f%%" % (key, v4, v2, 100 * (v2 - v4) / v4))
    return 0


raise SystemExit(main())
