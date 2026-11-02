# -*- coding: utf-8 -*-
"""「不许比跟随器差」下限（VRC_POLICY_FLOOR）的离线 A/B：它到底托住了多少速度？

**要回答的问题**（2026-11-02，用户问「重训能不能提升效率」的前提）：

在线实测策略稳态 2.83 m/s。但 SafetyGate 的 floor 是「跟随器本拍轴值 x 4.0 x floor_frac」，
跟随器满轴时下限就是 4.0 x 0.6 = 2.4 m/s —— **2.83 里有多少是模型自己的意愿、
多少是被下限托住的，从来没分开过。**

分不开就不知道重训该往哪使劲：模型要是本来就只想到 2.4，加 [mode] 会有大收益；
要是它本来想跑 3.5、只是被门限截断，那该改的是门限而不是模型。

**做法**：在 hold-out 上跑真模型，按 **模型原始输出**（未经任何门限）统计速度，
再按 SafetyGate 的实际算法套上不同 floor_frac，看终值差多少。

与在线不完全等价（在线 floor 来自跟随器的真实轴值，这里只能用代理），
所以同时给三档代理：0.0 / 0.6（生产默认）/ 1.0（跟满轴）。

用法::

    .venv_xpu/Scripts/python.exe research/tools/eval_floor_ab.py \
        --runs .venv_xpu/runs/m0_s3_448/best --out .venv_xpu/runs/eval_floor_ab
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
    K_STEPS, PolicyDataset, build_index, forward_batch, load_model, make_collate,
)

# 方案 §9.1 的实测速度平台：满轴 4.0 m/s（前进）
FULL_AXIS_MPS = 4.0
V_STILL = 0.1


def apply_floor(pred: np.ndarray, floor_mps: float) -> np.ndarray:
    """复刻 policy_executor.SafetyGate 的下限逻辑：只抬 vz，不碰 vx。"""
    b = pred.copy()
    if floor_mps > 0.0:
        b[:, :, 1] = np.maximum(b[:, :, 1], floor_mps)
    return b


def summarize(pred: np.ndarray, act: np.ndarray, mask: np.ndarray, cls: np.ndarray,
              floor_mps: float) -> dict:
    """在 move 类上统计「模型想走多快」vs「门限后走多快」。"""
    p = apply_floor(pred, floor_mps)
    out = {}
    mv = cls == "move"
    if not mv.any():
        return out
    for name, arr in (("raw", pred[mv]), ("gated", p[mv])):
        # 第 0 步是「现在该走多快」，最贴近在线稳态
        spd0 = np.linalg.norm(arr[:, 0, :], axis=-1)
        # 前 8 步（2 Hz 实际执行的窗口）均值
        spd8 = np.linalg.norm(arr[:, :8, :], axis=-1).mean(-1)
        out[name] = dict(
            step0_p50=float(np.percentile(spd0, 50)),
            step0_p95=float(np.percentile(spd0, 95)),
            step0_mean=float(spd0.mean()),
            first8_p50=float(np.percentile(spd8, 50)),
            first8_mean=float(spd8.mean()),
            vz_step0_p50=float(np.percentile(arr[:, 0, 1], 50)),
        )
    # 有多少步被下限改动过（说明模型自己低于下限）
    raw_vz = pred[mv][:, :, 1]
    out["floored_frac"] = float((raw_vz < floor_mps).mean()) if floor_mps > 0 else 0.0
    # 真值参照
    a = act[mv]
    out["truth"] = dict(
        step0_p50=float(np.percentile(np.linalg.norm(a[:, 0, :], axis=-1), 50)),
        first8_mean=float(np.linalg.norm(a[:, :8, :], axis=-1).mean(-1).mean()),
    )
    out["n"] = int(mv.sum())
    return out


@torch.no_grad()
def infer(model, head, loader, device):
    preds, acts, masks, clss = [], [], [], []
    for batch in loader:
        preds.append(forward_batch(model, head, batch["inputs"], device).float().cpu().numpy())
        acts.append(batch["act"].numpy())
        masks.append(batch["mask"].numpy())
        clss.extend(batch["cls"])
    return (np.concatenate(preds), np.concatenate(acts),
            np.concatenate(masks), np.array(clss))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="floor_frac 的离线 A/B")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--floors", nargs="+", type=float, default=[0.0, 0.3, 0.6])
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
    print("[data] val=%d" % len(val), flush=True)
    ds = PolicyDataset(val, processor, min_px, max_px)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=make_collate(processor, min_px, max_px))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"val": len(val), "full_axis_mps": FULL_AXIS_MPS, "runs": {}}
    for run in a.runs:
        run = Path(run)
        print("[load] %s" % run, flush=True)
        model, head, _ = load_model(a.model, 128, device)
        model.load_adapter(str(run / "adapter"), adapter_name="trained", is_trainable=False)
        model.set_adapter("trained")
        head.load_state_dict(torch.load(run / "head.pt", map_location=device))
        model.eval()
        pred, act, mask, cls = infer(model, head, dl, device)
        rows = {}
        for frac in a.floors:
            # 代理：假设跟随器满轴（最坏情况），下限 = 4.0 x frac
            rows["%.2f" % frac] = summarize(pred, act, mask, cls, FULL_AXIS_MPS * frac)
            r = rows["%.2f" % frac]
            print("  floor=%.2f (=%.2f m/s) raw step0 p50=%.2f  gated p50=%.2f  "
                  "floored=%.1f%%  truth p50=%.2f"
                  % (frac, FULL_AXIS_MPS * frac, r["raw"]["step0_p50"],
                     r["gated"]["step0_p50"], 100 * r["floored_frac"],
                     r["truth"]["step0_p50"]), flush=True)
        result["runs"][str(run)] = rows
    (out_dir / "floor_ab.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (out_dir / "floor_ab.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
