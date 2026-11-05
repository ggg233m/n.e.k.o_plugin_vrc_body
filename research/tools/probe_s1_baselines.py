# -*- coding: utf-8 -*-
r"""S1：三组**无 A\*** 基线（body / body+图像 / body+图像+因果历史）。

口径（2026-11-06 拍板，正式结果只认这一套）
--------------------------------------------
* 数据：``m0_policy_448_v5omega30``（10 session = 10 world / 23,082 窗口 / 78 min）。
* 指标：**完整 12 步** masked mse（12 × (vx, vz)），另报 move / trans / idle 分类。
* 折：world 名排序 → ``--seed`` 打乱 → ``array_split`` 成 5 折（每折 2 个世界）。
  数据集里 session 与 world 一一对应，所以「会话隔离」与「留一世界」在此等价。
* 读出：二次 ridge（原值 + 平方 + 上三角交叉），``--alpha`` 固定（默认 10.0）。
  多 alpha 时用**训练折内 world-holdout** 选，绝不在测试折上选。
* 视觉：冻结视觉塔（GameScaling-0.8B-S3）输出 **112 个 token**（两帧经
  ``temporal_patch_size=2`` 合并成一条序列）× 1024 维，**fp16 落盘缓存**，
  后续 ridge 与 4-query 读出实验复用同一份缓存。S1 只把 112 个 token 做均值池化。

三组（A\* 路径第四组标 **TBD**，不提前计算）
--------------------------------------------
    body                —— 上一拍本体速度 (vx, vz)，2 维
    body+img            —— body + 视觉池化（每折只在训练折上拟合 PCA64）
    body+img+causal     —— body + 视觉 + 因果历史 num44（wp=0 / topo 只含历史 /
                            body / mode 的 8 维二次扩张，44 维）

⚠️ 因果历史这组的目的**不是找增益**，而是确立「相对 body 接近 0」这个结论
（历史 8 步口径实测：``body_wp_causal − body_only = 0.0006``）。报告里不得把
任何一组的绝对值当成「模型能力」。

用法::

    # smoke（2,000 窗，结果**不进正式报告**、不得写进基线表）
    .venv_xpu/Scripts/python.exe research/tools/probe_s1_baselines.py \
        --n 2000 --tag smoke --out .venv_xpu/runs/s1_baselines

    # 全量（23,082 窗，正式结果）
    .venv_xpu/Scripts/python.exe research/tools/probe_s1_baselines.py \
        --tag full --out .venv_xpu/runs/s1_baselines
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

K_STEPS = 12
V_STILL = 0.1                     # 与 policy_executor / train evaluate 同一阈值
STOP_STEPS = 8                    # stop_recall 只看动作块前 8 步（与训练评测同口径）
STOP_RECALL_SOURCE = "ridge_action_head_first8_all_channels_below_V_STILL_idle_windows"
VISION_TOKEN_N = 112      # 448×256，patch16×spatial_merge2，两帧 temporal_merge2
VISION_DIM = 1024
CACHE_SPEC = "vision_tokens_fp16_v1:112x1024:video2frames"

# 「图像 + A* 路径」= TBD：依赖 astar_guidance.jsonl 的这些字段（build_policy_dataset
# 的 guide 契约）。当前数据集 0 个 astar_guidance.jsonl ⇒ 不计算，只登记。
ASTAR_TBD_FIELDS = ["path", "waypoint", "valid", "confidence", "pose_age_s",
                    "sigma_m", "mode", "control_mode", "requested", "reason",
                    "age_s", "source"]


# ---------------------------------------------------------------- 数值工具
def quad(X: np.ndarray) -> np.ndarray:
    """d 维 → d + d + d(d−1)/2（原值 + 平方 + 上三角交叉）。8 维 → 44。"""
    n, d = X.shape
    cross = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cross, axis=1)


def ridge_fit(X, Y, lam):
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1)
    A = Xs.T @ Xs + float(lam) * np.eye(Xs.shape[1])
    A[-1, -1] -= float(lam)          # 偏置不参与正则
    return np.linalg.solve(A, Xs.T @ Y), mu, sd


def ridge_pred(W, mu, sd, X):
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1) @ W


def masked_mse(pred, act, mask, steps=K_STEPS):
    """pred (n, steps*2) / act (n, steps, 2) / mask (n, steps) → 逐窗口平方和均值。"""
    p = np.asarray(pred).reshape(-1, steps, 2)
    a = np.asarray(act)[:, :steps, :2]
    err = ((p - a) ** 2).sum(-1)
    m = np.asarray(mask)[:, :steps]
    return float((err * m).sum() / max(m.sum(), 1.0))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 数据
def load_world_map(dataset: Path) -> dict:
    """session → world。manifest.json 里逐场有 ``world``；缺失时退回 session 名。"""
    mf = dataset / "manifest.json"
    if not mf.exists():
        return {}
    try:
        obj = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {k: str(v.get("world") or k) for k, v in (obj.get("sessions") or {}).items()}


def sample_key(s: dict) -> str:
    return "%s|%.4f" % (s["sid"], float(s["t"]))


def numeric7(s: dict) -> list:
    """数值旁路 7 维：wp dx/dz、topo 3、body vx/vz。**不含 mode**（v2 卡口径）。"""
    wp = s.get("wp") or (0.0, 0.0)
    tp = s.get("topo") or (0.0, 0.0, 0.0)
    bd = s.get("body") or (0.0, 0.0)
    return [float(wp[0]), float(wp[1]), float(tp[0]), float(tp[1]), float(tp[2]),
            float(bd[0]), float(bd[1])]


def body_feats(s: dict) -> list:
    bd = s.get("body") or (0.0, 0.0)
    return [float(bd[0]), float(bd[1])]


# ---------------------------------------------------------------- 视觉缓存
def cache_ident(dataset: Path, model: str, keys: list) -> tuple[str, dict]:
    mf = dataset / "manifest.json"
    payload = {
        "dataset": str(dataset),
        "manifest_sha256": sha256_file(mf) if mf.exists() else "missing",
        "model": str(model),
        "spec": CACHE_SPEC,
        "keys_sha256": sha256_text("\n".join(keys)),
        "n": len(keys),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))[:12], payload


def extract_vision(model_dir: str, samples: list, cache_dir: Path, ident: str,
                   payload: dict, batch: int, device: str, px: int) -> dict:
    """逐场跑**只视觉塔**前向，把 (112,1024) fp16 存成 ``<sid>.npy``。

    只调 ``inner.visual``：不走 0.8B LM（12 层 ViT vs 24 层 LM，实测耗时差一个量级），
    且视觉塔冻结 ⇒ 特征与 LM/LoRA 无关，可被后续读出实验直接复用。
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    cache_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    inner = model.model
    vis = inner.visual
    vdtype = next(vis.parameters()).dtype

    index = {"ident": ident, "payload": payload, "spec": CACHE_SPEC,
             "dtype": "float16", "shape_per_sample": [VISION_TOKEN_N, VISION_DIM],
             "sessions": {}, "keys": []}
    t_all = time.time()
    for sess in sorted({s["sid"] for s in samples}):
        sess_samples = [s for s in samples if s["sid"] == sess]
        out_path = cache_dir / ("%s.npy" % sess)
        if out_path.exists():
            arr = np.load(out_path, mmap_mode="r")
            if arr.shape == (len(sess_samples), VISION_TOKEN_N, VISION_DIM):
                index["sessions"][sess] = dict(n=len(sess_samples),
                                               file=out_path.name,
                                               bytes=int(out_path.stat().st_size),
                                               reused=True)
                index["keys"] += [sample_key(s) for s in sess_samples]
                print("  [cache] %s 复用 n=%d" % (sess, len(sess_samples)), flush=True)
                continue
        ds = T.PolicyDataset(sess_samples, proc, px, px, drop_wp=True, quad_feats=False)
        dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=T.make_collate(proc, px, px))
        chunks = []
        t0 = time.time()
        with torch.no_grad():
            for bi, b in enumerate(dl):
                pv = b["inputs"].get("pixel_values_videos")
                vg = b["inputs"].get("video_grid_thw")
                if pv is None or vg is None:
                    raise RuntimeError("processor 没产出 video 输入，检查 make_collate 口径")
                pv = pv.to(device).to(vdtype)
                vg = vg.to(device).clone()
                vg[:, 0] = 1          # 两帧已被 temporal_patch_size=2 合并成一条序列
                o = vis(pv, grid_thw=vg)
                po = o.pooler_output
                bs = len(b["cls"])
                # ⚠️ pv.shape[0] 是**展平后的 patch 数**（B*448），不是 batch size。
                if po.shape[0] != bs * VISION_TOKEN_N:
                    raise RuntimeError("视觉 token 总数 %d != batch %d × %d" %
                                       (po.shape[0], bs, VISION_TOKEN_N))
                feat = po.view(bs, VISION_TOKEN_N, po.shape[-1])
                chunks.append(feat.float().cpu().numpy().astype(np.float16))
                if (bi + 1) % 100 == 0:
                    print("    %s batch %d/%d  %.1fs" %
                          (sess, bi + 1, len(dl), time.time() - t0), flush=True)
        arr = np.concatenate(chunks, axis=0)
        assert arr.shape[0] == len(sess_samples), (arr.shape, len(sess_samples))
        np.save(out_path, arr)
        index["sessions"][sess] = dict(n=len(sess_samples), file=out_path.name,
                                       bytes=int(out_path.stat().st_size), reused=False)
        index["keys"] += [sample_key(s) for s in sess_samples]
        print("  [cache] %s 抽取 n=%d  %.1fs" % (sess, len(sess_samples),
                                                 time.time() - t0), flush=True)
    index["total_bytes"] = int(sum(v["bytes"] for v in index["sessions"].values()))
    index["extract_s"] = round(time.time() - t_all, 1)
    (cache_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    del model
    return index


def load_vision_pooled(cache_dir: Path, index: dict, samples: list) -> np.ndarray:
    """按样本顺序读缓存 → 112 token 均值 → (n,1024) float32（约 94 MB）。"""
    keys = index["keys"]
    want = [sample_key(s) for s in samples]
    if keys != want:
        raise RuntimeError("缓存 keys 与当前样本顺序不一致（hash 变了却复用了旧缓存）")
    out = np.zeros((len(want), VISION_DIM), dtype=np.float32)
    rows = 0
    for sess, info in index["sessions"].items():
        arr = np.load(cache_dir / info["file"])          # (n,112,1024) fp16
        out[rows:rows + info["n"]] = arr.mean(axis=1).astype(np.float32)
        rows += info["n"]
    assert rows == len(want), (rows, len(want))
    return out


# ---------------------------------------------------------------- 折与特征
def make_folds(worlds: np.ndarray, n_folds: int, seed: int) -> list:
    """world 名排序 → seed 打乱 → array_split（每折尽量均分世界）。"""
    uniq = np.array(sorted(set(worlds.tolist())))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    return [sorted(uniq[p].tolist()) for p in np.array_split(perm, n_folds)]


def pca_fit(X: np.ndarray, k: int) -> tuple:
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:int(k)].T
    return mu, P


def pca_apply(X, mu, P):
    return (X - mu) @ P


def pick_alpha(X, Y, mask, worlds, alphas) -> float:
    """训练折内 world-holdout 选 alpha（绝不用测试折）。

    只有传多个 alpha 时才真正做内层选择；单个 alpha 直接返回（正式口径是固定 10.0）。
    """
    if len(alphas) == 1:
        return float(alphas[0])
    uniq = sorted(set(worlds.tolist()))
    scores = {float(a): [] for a in alphas}
    for hold in uniq:
        te = worlds == hold
        tr = ~te
        if te.sum() < 20 or tr.sum() < 200:
            continue
        for al in alphas:
            W, mu, sd = ridge_fit(X[tr], Y[tr], al)
            scores[float(al)].append(
                _mse_from_pred(ridge_pred(W, mu, sd, X[te]), Y[te], mask[te]))
    if not scores[float(alphas[0])]:
        return float(alphas[0])
    return float(min(scores, key=lambda k: float(np.mean(scores[k]))))


def _mse_from_pred(pred, Y, mask):
    """Y 是 (n, steps*2) 但遮蔽步被填了 0；用 mask 展开到通道。"""
    p = np.asarray(pred).reshape(len(pred), K_STEPS, 2)
    y = np.asarray(Y).reshape(len(Y), K_STEPS, 2)
    m = np.asarray(mask)[:, :K_STEPS]
    return float(((((p - y) ** 2).sum(-1)) * m).sum() / max(m.sum(), 1.0))


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v5omega30")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--out", default=".venv_xpu/runs/s1_baselines")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--cache-dir", default=".venv_xpu/cache")
    ap.add_argument("--n", type=int, default=0, help="0 = 全量；>0 = 均匀抽 n 窗（smoke）")
    ap.add_argument("--steps", type=int, default=K_STEPS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca", type=int, default=64, help="0 = 不做 PCA（1024 维直入 ridge）")
    ap.add_argument("--alpha", type=float, nargs="+", default=[10.0])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--no-vision", action="store_true", help="只跑数值组（不含图像），用于快速校验")
    ap.add_argument("--diag-numeric", action="store_true",
                    help="额外跑 body+topo / body+mode / body+topo+mode 三个**诊断**臂，"
                         "用来把 body+img+causal 的增益拆到「历史」与「特权 mode」两处")
    a = ap.parse_args(argv)

    import train_policy_lora as T

    dataset = Path(a.dataset)
    world_map = load_world_map(dataset)
    samples = T.build_index(dataset, None)
    samples = [s for s in samples
               if int(np.asarray(s["mask"]).sum()) > 0 and s["act"].shape[0] >= a.steps]
    if a.n > 0 and a.n < len(samples):
        # 均匀抽：保持世界/类别分布，且完全确定（无随机）
        idx = np.linspace(0, len(samples) - 1, a.n).round().astype(int)
        samples = [samples[i] for i in sorted(set(idx.tolist()))]
    samples.sort(key=lambda s: (s["sid"], float(s["t"])))
    n = len(samples)
    worlds = np.array([world_map.get(s["sid"], s["sid"]) for s in samples])
    cls = np.array([str(s["cls"]) for s in samples])
    body = np.array([body_feats(s) for s in samples], dtype=np.float32)
    num7 = np.array([numeric7(s) for s in samples], dtype=np.float32)
    topo = np.array([list(s.get("topo") or (0.0, 0.0, 0.0))[:3] for s in samples],
                    dtype=np.float32)
    act = np.stack([np.asarray(s["act"])[:a.steps, :2] for s in samples]).astype(np.float32)
    mask = np.stack([np.asarray(s["mask"])[:a.steps] for s in samples]).astype(np.float32)
    full = mask.sum(-1) == a.steps
    keys = [sample_key(s) for s in samples]
    # 数据集口径 vs 可评估口径：manifest.totals.kept 含 14 窗 cls='none'（mask 全 0），
    # build_index 按既有口径跳过；这 14 窗的 masked mse 权重恒为 0，数值上无法计入。
    dataset_windows = None
    _mf = dataset / "manifest.json"
    if _mf.exists():
        try:
            dataset_windows = int((json.loads(_mf.read_text(encoding="utf-8")).get("totals")
                                   or {}).get("kept") or 0) or None
        except Exception:  # noqa: BLE001
            dataset_windows = None
    dropped = (dataset_windows - n) if dataset_windows else None
    print("[data] n=%d  full_mask=%d  worlds=%d  sessions=%d  (manifest kept=%s, 未用=%s)" %
          (n, int(full.sum()), len(set(worlds.tolist())), len({s["sid"] for s in samples}),
           dataset_windows, dropped), flush=True)

    # ---- 视觉缓存（哈希 = 数据 manifest + 模型 + spec + 样本键序列）----
    ident, payload = cache_ident(dataset, a.model, keys)
    cache_root = Path(a.cache_dir)
    if not cache_root.is_absolute():
        cache_root = _REPO / cache_root
    cache_dir = cache_root / ("vision_%s_%s" % (dataset.name, ident))
    vis_pooled = None
    cache_info = None
    if not a.no_vision:
        print("[cache] %s" % cache_dir, flush=True)
        if (cache_dir / "index.json").exists():
            index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
            if index.get("keys") != keys:
                print("[cache] keys 不匹配，重抽", flush=True)
                index = extract_vision(a.model, samples, cache_dir, ident, payload,
                                       a.batch, a.device, 448 * 256)
        else:
            index = extract_vision(a.model, samples, cache_dir, ident, payload,
                                   a.batch, a.device, 448 * 256)
        cache_info = dict(dir=str(cache_dir), ident=ident, size_bytes=index["total_bytes"],
                          n_files=len(index["sessions"]), spec=CACHE_SPEC,
                          dtype="float16", shape_per_sample=[VISION_TOKEN_N, VISION_DIM],
                          payload=index["payload"])
        vis_pooled = load_vision_pooled(cache_dir, index, samples)
        print("[cache] pooled %s  %.1f MB" % (vis_pooled.shape,
                                             vis_pooled.nbytes / 1e6), flush=True)

    # ---- 折 -------
    folds = make_folds(worlds, a.folds, a.seed)
    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    arms_res = {}
    data_listing = []
    t_start = time.time()
    for fi, test_worlds in enumerate(folds):
        te = np.isin(worlds, test_worlds)
        tr = ~te
        tr_full = tr & full
        if tr_full.sum() < 20 or te.sum() < 5:
            raise RuntimeError("折 %d 样本过少：train %d / eval %d" %
                               (fi, tr_full.sum(), te.sum()))
        Ytr = act[tr_full].reshape(int(tr_full.sum()), -1)
        Yte = act[te]
        Mte = mask[te]

        # 视觉 PCA：拟合只用训练折（用整个训练折，投影只取 tr_full / te 行）
        Vtr = Vte = None
        if vis_pooled is not None:
            if a.pca > 0:
                mu, P = pca_fit(vis_pooled[tr], a.pca)
                Vtr, Vte = pca_apply(vis_pooled[tr_full], mu, P), \
                           pca_apply(vis_pooled[te], mu, P)
            else:
                mu, sd = vis_pooled[tr].mean(0), vis_pooled[tr].std(0) + 1e-8
                Vtr, Vte = (vis_pooled[tr_full] - mu) / sd, (vis_pooled[te] - mu) / sd

        # 因果历史 num35：标准化后再二次扩张（统计量只用训练折，**不含 mode**）
        n_mu, n_sd = num7[tr].mean(0), num7[tr].std(0) + 1e-8
        Ctr = quad((num7[tr_full] - n_mu) / n_sd)
        Cte = quad((num7[te] - n_mu) / n_sd)

        arms = {
            "body": (body[tr_full], body[te]),
            "body+img": (None, None),
            "body+img+causal": (None, None),
        }
        if Vtr is not None:
            arms["body+img"] = (np.concatenate([body[tr_full], Vtr], 1),
                                np.concatenate([body[te], Vte], 1))
            arms["body+img+causal"] = (np.concatenate([body[tr_full], Vtr, Ctr], 1),
                                       np.concatenate([body[te], Vte, Cte], 1))
        if a.diag_numeric:
            # 只做归因：topo 是纯历史（无 mode）。mode 已被移出模型输入（v2 卡），
            # 不再构造 body+mode 诊断臂 —— 它属于特权状态，不是感知能力。
            arms["body+topo"] = (np.concatenate([body[tr_full], topo[tr_full]], 1),
                                 np.concatenate([body[te], topo[te]], 1))
        # copy：12 步都抄当前 body（不看任何输入）
        copy_pred = np.repeat(body[te][:, None, :2], a.steps, axis=1)
        copy_mse = float((((copy_pred - act[te]) ** 2).sum(-1) * Mte).sum() /
                         max(Mte.sum(), 1.0))

        fold_rec = {"fold": fi, "worlds_test": test_worlds,
                    "worlds_train": sorted(set(worlds[tr].tolist())),
                    "n_eval": int(te.sum()), "n_train_full": int(tr_full.sum()),
                    "eval_world_dist": {w: int((worlds[te] == w).sum())
                                        for w in test_worlds},
                    "eval_class_dist": {c: int((cls[te] == c).sum())
                                        for c in sorted(set(cls[te].tolist()))},
                    "train_class_dist": {c: int((cls[tr_full] == c).sum())
                                         for c in sorted(set(cls[tr_full].tolist()))},
                    "copy_mse12": copy_mse, "alpha": None,
                    "arms": {}}

        for name, (Xtr, Xte) in arms.items():
            if Xtr is None:
                continue
            alpha = pick_alpha(Xtr, Ytr, mask[tr_full], worlds[tr_full], a.alpha)
            W, mu, sd = ridge_fit(Xtr, Ytr, alpha)
            pred = ridge_pred(W, mu, sd, Xte)
            m12 = masked_mse(pred, act[te], Mte, a.steps)
            by_cls = {}
            for c in sorted(set(cls[te].tolist())):
                sel = cls[te] == c
                by_cls[c] = masked_mse(pred[sel], act[te][sel], Mte[sel], a.steps)
            # stop_recall：idle 窗里，动作头前 8 步是否全通道 < V_STILL（与训练评测同口径）
            p3 = np.asarray(pred).reshape(len(pred), a.steps, -1)
            idle_sel = cls[te] == "idle"
            if idle_sel.sum() > 0:
                _pi = np.abs(p3[idle_sel][:, :STOP_STEPS]).max(-1).max(-1)
                stop_recall = float((_pi < V_STILL).mean())
            else:
                stop_recall = None
            fold_rec["arms"][name] = dict(mse12=m12, by_class=by_cls, alpha=alpha,
                                          dim=int(Xtr.shape[1]),
                                          idle_mse12=by_cls.get("idle"),
                                          stop_recall=stop_recall,
                                          stop_recall_source=STOP_RECALL_SOURCE)
            fold_rec["alpha"] = alpha
            arms_res.setdefault(name, []).append(m12)
        data_listing.append(fold_rec)
        print("  fold %d test=%s  body=%.4f  body+img=%s  body+img+causal=%s  (%.0fs)" %
              (fi, ",".join(test_worlds),
               fold_rec["arms"]["body"]["mse12"],
               ("%.4f" % fold_rec["arms"]["body+img"]["mse12"]) if "body+img" in fold_rec["arms"] else "-",
               ("%.4f" % fold_rec["arms"]["body+img+causal"]["mse12"]) if "body+img+causal" in fold_rec["arms"] else "-",
               time.time() - t_start), flush=True)

    # ---- 汇总 ----=
    res = {}
    for name, vals in arms_res.items():
        v = np.asarray(vals, dtype=np.float64)
        per_cls = {}
        for c in ("move", "trans", "idle"):
            cv = [f["arms"][name]["by_class"].get(c) for f in data_listing]
            cv = [x for x in cv if x is not None]
            if cv:
                per_cls[c] = dict(mean=float(np.mean(cv)), std=float(np.std(cv)),
                                  folds=[float(x) for x in cv])
        res[name] = dict(mean=float(v.mean()), std=float(v.std()),
                         folds=[float(x) for x in vals], by_class=per_cls,
                         n_folds=len(vals),
                         idle_mse12=per_cls.get("idle"),
                         stop_recall=dict(
                             mean=float(np.mean(_sr)), std=float(np.std(_sr)),
                             folds=[float(x) for x in _sr],
                             source=STOP_RECALL_SOURCE,
                             definition="idle 窗中动作头前 %d 步全通道 |v| < %.2f 的比例"
                                        % (STOP_STEPS, V_STILL)) if (
                                 _sr := [x for x in (f["arms"][name].get("stop_recall")
                                                     for f in data_listing)
                                         if x is not None]) else None)
    pair = {}
    for a1, a2 in (("body+img", "body"),
                   ("body+img+causal", "body+img"),
                   ("body+img+causal", "body")):
        if a1 in res and a2 in res:
            d = np.asarray(res[a1]["folds"]) - np.asarray(res[a2]["folds"])
            pair["%s - %s" % (a1, a2)] = dict(mean=float(d.mean()), std=float(d.std()),
                                              folds=[float(x) for x in d])
    cv = [f["copy_mse12"] for f in data_listing]
    res["copy"] = dict(mean=float(np.mean(cv)), std=float(np.std(cv)),
                       folds=[float(x) for x in cv], n_folds=len(cv), note="12 步全填当前 body")

    summary = {
        "tag": a.tag,
        "note": "M0 = 停止语义和评测口径收敛完成；本表是**离线 ridge 基线**，不是模型能力通过。",
        "caveats": [
            "数据集 manifest.totals.kept=23082，其中 14 窗 cls='none' / mask 全 0"
            "（无任何动作标签），build_index 按既有口径跳过（train_policy_lora.py:111）"
            "⇒ 可评估 23068 窗；这 14 窗对 masked mse 权重恒为 0，数值上不可计入。",
            "body+img+causal 的 num35 **不含 mode**（v2 卡口径）；mode 只用于运行时"
            "停止决策，已从模型输入删除（2026-11-06）。",
            "训练侧报告的一并口径：mse12 / idle_mse12 / stop_recall（首个 8 步）+ "
            "stop_recall_source；stop_recall 在这里由 ridge 动作头直接判定，"
            "不是模型 + stop_head 的串联结果。",
            "copy 基线 = 12 步全填当前 body（不看任何输入），用来判断 ridge 是否连惯性外推都打不过。",
            "「图像+A* 路径」组 TBD：astar_guidance.jsonl 全仓 0 个，按拍板不提前计算。",
        ],
        "config": {
            "dataset": str(dataset), "model": str(a.model), "n_windows": n,
            "full_mask_windows": int(full.sum()), "steps": a.steps, "folds": a.folds,
            "seed": a.seed, "pca": a.pca, "alpha": [float(x) for x in a.alpha],
            "metric": "masked mse over %d steps x (vx,vz)" % a.steps,
            "ridge": "quadratic (x, x^2, upper-tri cross), bias unregularized",
            "folds_definition": "world names sorted -> seed-shuffled -> array_split",
            "no_vision": bool(a.no_vision),
            "diag_numeric": bool(a.diag_numeric),
            "card_version": 2, "numeric_dim": 7, "mode_in_input": False,
        },
        "cache": cache_info,
        "arms": res,
        "pair_diff": pair,
        "arms_tbd": [{
            "name": "body+img+astar_path",
            "status": "TBD",
            "reason": "当前数据集 astar_guidance.jsonl 为 0 个，按拍板不提前计算",
            "requires_fields": ASTAR_TBD_FIELDS,
        }],
        "data_listing": {
            "n_windows": n, "full_mask_windows": int(full.sum()),
            "partial_mask_windows": int(((mask.sum(-1) > 0) & ~full).sum()),
            "dataset_windows": dataset_windows, "dropped_unlabeled_windows": dropped,
            "worlds": sorted(set(worlds.tolist())),
            "world_dist": {w: int((worlds == w).sum()) for w in sorted(set(worlds.tolist()))},
            "class_dist": {c: int((cls == c).sum()) for c in sorted(set(cls.tolist()))},
            "per_session": {sid: int(sum(1 for s in samples if s["sid"] == sid))
                            for sid in sorted({s["sid"] for s in samples})},
            "per_fold": data_listing,
        },
        "elapsed_s": round(time.time() - t_start, 1),
    }
    summary["data_listing"]["per_session_world"] = {
        sid: world_map.get(sid, sid) for sid in summary["data_listing"]["per_session"]}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    for name in ("copy",) + tuple(k for k in res if k != "copy"):
        print("  %-20s mse12 = %.4f ± %.4f" % (name, res[name]["mean"], res[name]["std"]))
    for k, v in pair.items():
        print("  %-28s Δ = %+.4f ± %.4f" % (k, v["mean"], v["std"]))
    print("[done] %s" % (out_dir / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
