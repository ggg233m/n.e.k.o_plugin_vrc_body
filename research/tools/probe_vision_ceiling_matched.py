# -*- coding: utf-8 -*-
"""同一批样本上，比「可用视觉增益」与「模型实际吃下的视觉增益」。

起因（2026-11-02，doc-fixer 指出）：Lead 之前把两个不同分母的数当成同一个，
得出「视觉已到顶」的错误结论：
  * 探针 26.6% —— 分母是「无图」，比的是 body+wp+black -> body+wp+img
  * 模型 26.5% —— 分母是「真实图」，比的是 real -> shuffle
两个分母不同，数值接近纯属巧合。

本脚本把两者放到**同一批样本**上重算，只比较同一个量：
  「不给视觉信息」 -> 「给真实视觉信息」 的 MSE 下降百分比
    * 探针：body+wp+black -> body+wp+img（线性 ridge，5-fold）
    * 模型：black -> real（直接前向）

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_vision_ceiling_matched.py \
        --run .venv_xpu/runs/m0_s3_448/best --n 400 --out .venv_xpu/runs/vision_ceiling
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1)
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, Xs.T @ Y), mu, sd


def ridge_pred(W, mu, sd, X):
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1) @ W


def mse8(pred, act, mask, steps=8):
    err = ((pred[:, :steps, :] - act[:, :steps, :]) ** 2).sum(-1)
    m = mask[:, :steps]
    return float((err * m).sum() / max(m.sum(), 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor
    from train_policy_lora import (ActionHead, PolicyDataset, build_index,
                                   forward_batch, load_model, make_collate)

    device = "xpu" if torch.xpu.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = 448 * 256

    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"][:a.n]
    print("[data] 同一批 %d 条 val 样本" % len(samples), flush=True)
    ds = PolicyDataset(samples, processor, min_px, max_px, prompt_format="legacy")

    # ---------- 1. 探针：冻结 ViT 特征，同一批样本 ----------
    from transformers import AutoModelForImageTextToText
    vm = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to(device)
    vm.eval()
    visual = vm.model.visual

    def vit_feat(item):
        out = processor(text=[item["text"]], videos=[item["video"]], return_tensors="pt",
                        padding=True, min_pixels=min_px, max_pixels=max_px,
                        cap_pixels_per_frame=False, do_sample_frames=False)
        pv = out["pixel_values_videos"].to(device)
        grid = out["video_grid_thw"].to(device)
        with torch.no_grad():
            f = visual(pv, grid_thw=grid)
        po = f.pooler_output
        if isinstance(po, (tuple, list)):
            po = torch.stack([x for x in po if torch.is_tensor(x)], 0)[0]
        return po.float().cpu().numpy().mean(0)

    F, FB, B, W, A, M = [], [], [], [], [], []
    for i in range(len(samples)):
        it = ds[i]
        v = it["video"]
        F.append(vit_feat(it))
        it["video"] = np.zeros_like(v)
        FB.append(vit_feat(it))
        B.append(samples[i]["body"]); W.append(samples[i]["wp"])
        A.append(samples[i]["act"]); M.append(samples[i]["mask"])
        if (i + 1) % 100 == 0:
            print("  [feat] %d/%d" % (i + 1, len(samples)), flush=True)
    F = np.asarray(F, np.float32); FB = np.asarray(FB, np.float32)
    B = np.asarray(B, np.float32); W = np.asarray(W, np.float32)
    act = np.asarray(A, np.float32); mask = np.asarray(M, np.float32)
    Y = act.reshape(len(act), -1)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(F))
    folds = np.array_split(order, a.folds)
    probe = {"black": [], "img": []}
    for fi, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        mu = F[tr].mean(0)
        _, _, Vt = np.linalg.svd(F[tr] - mu, full_matrices=False)
        P = Vt[:a.pca].T
        # ⚠️ 标准化必须用**同一套** sd（取自真实图那一路）。
        # 早先每一路各用自己的 sd，黑图特征方差极小时 sd≈0，除法直接炸到 MSE 1.6e4，
        # 于是探针增益报成 100% —— 那是数值事故，不是结论。
        sd_ref = ((F - mu) @ P)[tr].std(0) + 1e-8

        def proj(X):
            return ((X - mu) @ P) / sd_ref
        for key, FEAT in (("black", FB), ("img", F)):
            X = np.concatenate([B, W, proj(FEAT)], 1)
            Wt, m, s = ridge_fit(X[tr], Y[tr], 1.0)
            p = ridge_pred(Wt, m, s, X[te]).reshape(-1, 12, 2)
            probe[key].append(mse8(p, act[te], mask[te]))
    pb, pi = float(np.mean(probe["black"])), float(np.mean(probe["img"]))

    # ---------- 2. 模型：同一批样本，black vs real ----------
    model, head, hidden = load_model(a.model, 128, device)
    model.load_adapter(str(Path(a.run) / "adapter"), adapter_name="trained", is_trainable=False)
    model.set_adapter("trained")
    head.load_state_dict(torch.load(Path(a.run) / "head.pt", map_location=device))
    model.eval()
    collate = make_collate(processor, min_px, max_px)
    import torch as T

    def model_mse(black: bool):
        errs, ms = [], []
        for i in range(0, len(samples), 8):
            chunk = []
            for j in range(i, min(i + 8, len(samples))):
                it = ds[j]
                if black:
                    it["video"] = np.zeros_like(it["video"])
                chunk.append(it)
            b = collate(chunk)
            with T.no_grad():
                r = forward_batch(model, head, b["inputs"], device)
            if isinstance(r, tuple):
                r = r[0]
            errs.append(r.float().cpu().numpy())
            ms.append(b["act"].numpy())
            M_ = b["mask"].numpy()
            ms[-1] = (ms[-1], M_)
        # 逐批累加 masked mse
        tot, wt = 0.0, 0.0
        for p, (ac, mk) in zip(errs, ms):
            e = ((p[:, :8, :] - ac[:, :8, :]) ** 2).sum(-1)
            mm = mk[:, :8]
            tot += float((e * mm).sum()); wt += float(mm.sum())
        return tot / max(wt, 1.0)

    mb = model_mse(True)
    mr = model_mse(False)

    out = dict(
        n=len(samples), pca=a.pca, folds=a.folds,
        probe=dict(black=pb, img=pi, gain_pct=100 * (pb - pi) / pb),
        model=dict(black=mb, real=mr, gain_pct=100 * (mb - mr) / mb),
    )
    out["capture_ratio"] = (out["model"]["gain_pct"] / out["probe"]["gain_pct"]
                           if out["probe"]["gain_pct"] > 0 else None)

    print()
    print("  同一批 %d 条样本，同一个量：「无视觉 -> 有视觉」的 MSE 下降" % out["n"])
    print("    探针（冻结 ViT + ridge）: %.4f -> %.4f  = %.1f%%"
          % (pb, pi, out["probe"]["gain_pct"]))
    print("    模型（直接前向）       : %.4f -> %.4f  = %.1f%%"
          % (mb, mr, out["model"]["gain_pct"]))
    print("    ⇒ 模型吃下上限的 %.0f%%" % (100 * out["capture_ratio"]))

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "vision_ceiling_matched.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "vision_ceiling_matched.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
