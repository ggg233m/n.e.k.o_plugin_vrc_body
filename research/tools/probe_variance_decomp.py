# -*- coding: utf-8 -*-
"""方差分解：在 move 类上，视觉到底贡献了多少解释力？

**要回答的问题**（2026-11-02）：视觉消融显示换图只让动作块转 2 度，
但那是**模型**的行为。真正要问的是：**这个任务里视觉有没有可用的解释力**。

做法：对同一个 hold-out，比较四组「信息集」对真值标签的拟合能力。
不看模型，只看数据本身（线性探针），所以结果与训练超参无关。

* base   —— 只给 body（上一拍速度）
* +wp    —— 再加 wp
* +img   —— 再加图像特征（用**冻结 ViT** 的 pooler 输出，1024 维，PCA 到 64 维）
* +black —— 图像换成全黑，作为「视觉无信息」的对照组

判据：如果 +img 显著优于 +wp，视觉有信息；如果 +img ≈ +wp ≈ +black，
那视觉在这批数据里对标签零贡献 —— 加输入、重训都改变不了。

补充两组基线：全零、抄当前。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_variance_decomp.py \
        --out .venv_xpu/runs/variance_decomp --n 1200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """闭式 ridge：X (n,d) -> Y (n,k)。标准化后解正规方程。"""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    A[-1, -1] -= alpha  # 不惩罚偏置
    return np.linalg.solve(A, Xs.T @ Y), mu, sd


def ridge_pred(W, mu, sd, X):
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    return Xs @ W


def masked_mse_np(pred: np.ndarray, act: np.ndarray, mask: np.ndarray, steps: int = 8) -> float:
    """与训练/评估口径一致：前 steps 步、(vx,vz) 两维平方和，按 mask 加权。"""
    err = ((pred[:, :steps, :] - act[:, :steps, :]) ** 2).sum(-1)
    m = mask[:, :steps]
    return float((err * m).sum() / max(m.sum(), 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor
    from train_policy_lora import PolicyDataset, build_index, build_index as _bi  # noqa: F401

    device = "xpu" if torch.xpu.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = 448 * 256
    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"]
    # 只用 move 类：idle 是「全零」，会把所有方法的 MSE 稀释
    samples = [s for s in samples if s["cls"] == "move"][:a.n]
    print("[data] move 样本 %d" % len(samples), flush=True)

    ds = PolicyDataset(samples, processor, min_px, max_px)

    # ---- 图像特征：用冻结 ViT 的 pooler 输出 ----
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to(device)
    model.eval()
    visual = model.model.visual

    def vit_feat(item):
        out = processor(text=[item["text"]], videos=[item["video"]], return_tensors="pt",
                        padding=True, min_pixels=min_px, max_pixels=max_px,
                        cap_pixels_per_frame=False, do_sample_frames=False)
        pv = out["pixel_values_videos"].to(device)
        grid = out["video_grid_thw"].to(device)
        with torch.no_grad():
            f = visual(pv, grid_thw=grid)
        # 对 112 个视觉 token 取均值池化 -> (1024,)
        return f.pooler_output.float().cpu().numpy().mean(0)

    # 追加两路对照，回答「模型读的是场景内容，还是『有没有画面』」：
    #   shuffle —— 换成别的样本的图（场景错，但仍是自然图像）
    #   same    —— 两帧都用当前帧（抹掉时间差分）
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(samples))
    feats, feats_black, feats_shuf, body, wp, acts, masks = [], [], [], [], [], [], []
    for i in range(len(samples)):
        item = ds[i]
        v = item["video"]
        feats.append(vit_feat(item))
        item["video"] = np.zeros_like(v)
        feats_black.append(vit_feat(item))
        j = int(perm[i])
        if j == i:
            j = (j + 1) % len(samples)
        item["video"] = ds[j]["video"].copy()
        feats_shuf.append(vit_feat(item))
        body.append(samples[i]["body"])
        wp.append(samples[i]["wp"])
        acts.append(samples[i]["act"])
        masks.append(samples[i]["mask"])
        if (i + 1) % 100 == 0:
            print("  [feat] %d/%d" % (i + 1, len(samples)), flush=True)

    F = np.asarray(feats, dtype=np.float32)
    FB = np.asarray(feats_black, dtype=np.float32)
    FS = np.asarray(feats_shuf, dtype=np.float32)
    B = np.asarray(body, dtype=np.float32)
    W = np.asarray(wp, dtype=np.float32)
    A = np.asarray(acts, dtype=np.float32)
    M = np.asarray(masks, dtype=np.float32)

    # ⚠️ 特征**先落盘再分析**：ViT 那一趟要 5 min，而分析段改一行就可能抛异常
    # （已经因此白跑两趟）。落盘之后用 probe_variance_cached.py 秒级重算。
    _o = Path(a.out)
    _o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_o / "feats.npz", real=F, black=FB, shuffle=FS, body=B, wp=W,
                        act=A, mask=M, sid=np.array([s["sid"] for s in samples]))
    print("[feats] 已缓存 %s" % (_o / "feats.npz"), flush=True)

    # PCA（在训练侧拟合，避免偷看）
    ntr = int(len(F) * 0.8)
    mu = F[:ntr].mean(0)
    U, S, Vt = np.linalg.svd(F[:ntr] - mu, full_matrices=False)
    P = Vt[:a.pca].T
    Fp = (F - mu) @ P
    FBp = (FB - mu) @ P

    # 标准化特征尺度。⚠️ 三个条件必须用**同一个** sd：早先只给 Fp/FBp 调了 norm01，
    # shuffle 那条直接引用未定义的 sd 抛 NameError（跑满 5 min 才炸）。
    sd = Fp[:ntr].std(0) + 1e-8
    Fp = Fp / sd
    FBp = (FBp / sd) if len(FB) else FBp
    FSp = ((FS - mu) @ P) / sd if len(FS) else np.zeros((len(F), 1), np.float32)

    Y = A.reshape(len(A), -1)  # (n, 24)

    sets = {
        "zero": np.zeros((len(B), 1), np.float32),
        "body": B,
        "+wp": np.concatenate([B, W], 1),
        "+wp+img": np.concatenate([B, W, Fp], 1),
        "+wp+black": np.concatenate([B, W, FBp], 1),
        "+wp+shuffle": np.concatenate([B, W, FSp], 1),
        "img_only": Fp,
        "wp_only": W,
    }

    out = {"n": len(samples), "pca": a.pca, "results": {}}
    for name, X in sets.items():
        Xtr, Xte = X[:ntr], X[ntr:]
        Ytr, Yte = Y[:ntr], Y[ntr:]
        Mte, Ate = M[ntr:], A[ntr:]
        if name == "zero":
            P_te = np.zeros_like(Yte)
        else:
            Wt, m, s = ridge_fit(Xtr, Ytr, alpha=1.0)
            P_te = ridge_pred(Wt, m, s, Xte)
        mse = masked_mse_np(P_te.reshape(-1, 12, 2), Ate, Mte, steps=8)
        # 抄当前基线：12 步都填 body
        copy = np.repeat(B[ntr:][:, None, :], 12, axis=1)
        mse_copy = masked_mse_np(copy, Ate, Mte, steps=8)
        out["results"][name] = dict(mse8=mse, n_train=ntr, n_test=len(Xte))
        print("  %-11s mse8=%.4f" % (name, mse), flush=True)
    out["baseline_copy_current"] = mse_copy
    print("  %-11s mse8=%.4f" % ("copy_current", mse_copy), flush=True)

    best = min(v["mse8"] for k, v in out["results"].items() if k != "zero")
    out["best"] = best
    out["vision_gain"] = out["results"]["+wp"]["mse8"] - out["results"]["+wp+img"]["mse8"]
    out["black_gain"] = out["results"]["+wp"]["mse8"] - out["results"]["+wp+black"]["mse8"]
    print()
    print("  视觉增益 (+wp -> +wp+img)   = %.4f" % out["vision_gain"])
    print("  对照   (+wp -> +wp+black)   = %.4f  （应当 ~0，验证方法本身）"
          % out["black_gain"])

    # ⚠️ 早先写成 Path(a.out).parent.mkdir —— 建的是 .venv_xpu/runs，不是 a.out 本身，
    # 于是 json 写不出来（FileNotFoundError），1200 条特征的 5 分钟白跑。
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "variance_decomp.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 特征存下来，后面重算不用再跑 ViT（这一趟是 5 min 的大头）
    np.savez_compressed(o / "feats.npz", real=F, black=FB, shuffle=FS, body=B, wp=W,
                        act=A, mask=M, sid=np.array([s["sid"] for s in samples]))
    print("[done] %s" % (o / "variance_decomp.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
