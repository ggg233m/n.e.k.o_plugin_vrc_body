# -*- coding: utf-8 -*-
"""在 hold-out 上评估训练好的 policy LoRA，并扫描输出死区（stop deadband）。

用途：M0 第一轮的 val 显示 idle「停住召回」只有 0%，但 idle MSE 已很低（RMS≈0.12 m/s）。
本脚本加载指定 run 的 ``adapter/ + head.pt``，在 ``split=val`` 上：
  1. 复算原始指标（与训练时 evaluate 同口径，作 sanity check）；
  2. 对预测逐步做「|v| < 阈 → 置零」的死区，扫 0.10/0.12/0.15/0.20；
  3. 同时报 move mse8 / walk 召回 / idle 停住召回 / trans 起步召回，
     以及死区造成的「误杀」（move 第 0 步被归零、trans 起不来）。

用法::

    .venv_xpu\\Scripts\\python.exe research/tools/eval_policy_lora.py \\
        --runs h:\\...\\.venv_xpu\\runs\\m0_s3_448\\best h:\\...\\epoch_03 \\
        --out  h:\\...\\.venv_xpu\\runs\\eval_deadband
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_policy_lora import (  # noqa: E402
    K_STEPS, V_STILL, PolicyDataset, build_index, forward_batch, load_model,
    make_collate, masked_mse,
)


def infer(model, head, loader, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    model.eval()
    head.eval()
    preds, acts, masks, clss, wps = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(forward_batch(model, head, batch["inputs"], device).float().cpu())
            acts.append(batch["act"])
            masks.append(batch["mask"])
            wps.append(batch["wps"])
            clss.extend(batch["cls"])
    return torch.cat(preds), torch.cat(acts), torch.cat(masks), np.array(clss), torch.cat(wps)


def deadband(pred: torch.Tensor, thr: float) -> torch.Tensor:
    if thr <= 0:
        return pred
    norm = pred.norm(dim=-1, keepdim=True)
    return torch.where(norm < thr, torch.zeros_like(pred), pred)


def metrics(pred: torch.Tensor, act: torch.Tensor, mask: torch.Tensor,
            cls: np.ndarray) -> dict:
    out = {}
    for c in ("move", "trans", "idle"):
        sel = cls == c
        if not sel.any():
            continue
        p, a, m = pred[sel], act[sel], mask[sel]
        d = dict(n=int(sel.sum()), mse8=float(masked_mse(p, a, m, 8)))
        if c == "move":
            ok = m[:, 0] > 0
            d["walk_recall"] = float(((p[:, 0].norm(dim=-1) > V_STILL) & ok).sum() / ok.sum().clamp_min(1))
        if c == "idle":
            first8 = p[:, :8].norm(dim=-1)
            ok = m[:, :8].sum(-1) == 8
            d["stop_recall"] = float(((first8 < V_STILL).all(-1) & ok).sum() / ok.sum().clamp_min(1))
        if c == "trans":
            first8 = p[:, :8].norm(dim=-1)
            ok = m[:, :8].sum(-1) > 0
            d["start_recall"] = float((((first8 > V_STILL) & (m[:, :8] > 0)).any(-1) & ok).sum() /
                                      ok.sum().clamp_min(1))
        out[c] = d
    return out


def killed_by_deadband(pred: torch.Tensor, mask: torch.Tensor, cls: np.ndarray,
                       thr: float) -> dict:
    """死区把「本来要动」的样本杀掉的比例（mean 在 mask 有效的第 0 步上）。"""
    if thr <= 0:
        return {"move_step0_killed": 0.0, "trans_first8_killed": 0.0}
    raw = pred.norm(dim=-1).detach().cpu().numpy()
    m = mask.detach().cpu().numpy() > 0
    mv = cls == "move"
    raw_m, m_m = raw[mv], m[mv]
    valid0 = m_m[:, 0]
    killed_m = float((raw_m[valid0, 0] < thr).mean()) if valid0.any() else 0.0
    tr = cls == "trans"
    raw_t, m_t = raw[tr], m[tr]
    valid8 = m_t[:, :8].sum(-1) > 0
    killed_t = 0.0
    if valid8.any():
        killed_t = float(((raw_t[:, :8] < thr) & m_t[:, :8]).all(-1)[valid8].mean())
    return {"move_step0_killed": round(killed_m, 4), "trans_first8_killed": round(killed_t, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="policy LoRA hold-out 评估 + 死区扫描")
    ap.add_argument("--dataset", default=r"G:\vrc_recording\vrc_model\dataset\m0_policy_448")
    ap.add_argument("--model", default=r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3")
    ap.add_argument("--runs", nargs="+", required=True, help="含 adapter/ 与 head.pt 的 run 目录")
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.10, 0.12, 0.15, 0.20])
    ap.add_argument("--wp-thresholds", nargs="+", type=float, default=[0.0, 0.02, 0.05])
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = int(a.size[0]) * int(a.size[1])

    samples = build_index(Path(a.dataset), None)
    val = [s for s in samples if s["split"] == "val"]
    if a.max_val:
        val = val[:a.max_val]
    print(f"[data] val={len(val)}", flush=True)
    ds = PolicyDataset(val, processor, min_px, max_px)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=make_collate(processor, min_px, max_px))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"val": len(val), "runs": {}}
    for run in a.runs:
        run = Path(run)
        print(f"[load] {run}", flush=True)
        model, head, _ = load_model(a.model, 128, device)
        model.load_adapter(str(run / "adapter"), adapter_name="trained", is_trainable=False)
        model.set_adapter("trained")
        head.load_state_dict(torch.load(run / "head.pt", map_location=device))
        pred, act, mask, cls, wps = infer(model, head, dl, device)
        rows = {}
        for thr in a.thresholds:
            p = deadband(pred, thr)
            m = metrics(p, act, mask, cls)
            m.update(killed_by_deadband(pred, mask, cls, thr))
            rows[f"{thr:.2f}"] = m
            mv, idl, tr = m.get("move", {}), m.get("idle", {}), m.get("trans", {})
            print("  thr=%.2f  move8=%.3f walk=%.1f%% kill0=%.1f%% | idle8=%.3f stop=%.1f%% | "
                  "trans8=%.3f start=%.1f%% kill8=%.1f%%"
                  % (thr, mv.get("mse8", float("nan")), 100 * mv.get("walk_recall", 0),
                     100 * m["move_step0_killed"], idl.get("mse8", float("nan")),
                     100 * idl.get("stop_recall", 0), tr.get("mse8", float("nan")),
                     100 * tr.get("start_recall", 0), 100 * m["trans_first8_killed"]), flush=True)

        # 航点门控：|wp| < thr → 24 维输出整体置 0（方案 §5.2 的「站立航点≈0」）
        wp_rows = {}
        wpn = wps.norm(dim=-1).numpy()
        mnp = mask.numpy() > 0
        for thr in a.wp_thresholds:
            gate = torch.from_numpy(wpn < thr)
            p = torch.where(gate[:, None, None], torch.zeros_like(pred), pred)
            m = metrics(p, act, mask, cls)
            mv, ok = (cls == "move"), (cls == "move") & mnp[:, 0]
            tr, ok8 = (cls == "trans"), (cls == "trans") & (mnp[:, :8].sum(-1) > 0)
            m["move_step0_killed"] = round(float((wpn[ok] < thr).mean()) if ok.any() else 0.0, 4)
            m["trans_first8_killed"] = round(float((wpn[ok8] < thr).mean()) if ok8.any() else 0.0, 4)
            wp_rows[f"{thr:.2f}"] = m
            mv_d, idl_d, tr_d = m.get("move", {}), m.get("idle", {}), m.get("trans", {})
            print("  wp<%.2f  move8=%.3f walk=%.1f%% kill0=%.1f%% | idle8=%.3f stop=%.1f%% | "
                  "trans8=%.3f start=%.1f%% kill8=%.1f%%"
                  % (thr, mv_d.get("mse8", float("nan")), 100 * mv_d.get("walk_recall", 0),
                     100 * m["move_step0_killed"], idl_d.get("mse8", float("nan")),
                     100 * idl_d.get("stop_recall", 0), tr_d.get("mse8", float("nan")),
                     100 * tr_d.get("start_recall", 0), 100 * m["trans_first8_killed"]), flush=True)
        # 基线（与 pred 无关）：抄当前 / 全零
        base = {}
        for c in ("move", "trans", "idle"):
            sel = cls == c
            if not sel.any():
                continue
            a_c, m_c = act[sel], mask[sel]
            base[c] = {
                "zero8": float(masked_mse(torch.zeros_like(a_c), a_c, m_c, 8)),
                "copy8": float(masked_mse(a_c[:, :1].repeat(1, K_STEPS, 1), a_c, m_c, 8)),
            }
        result["runs"][str(run)] = {"thresholds": rows, "wp_gate": wp_rows, "baseline": base}
        del model, head
        torch.xpu.empty_cache() if device == "xpu" else None
    (out / "deadband_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(f"[done] {out / 'deadband_eval.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
