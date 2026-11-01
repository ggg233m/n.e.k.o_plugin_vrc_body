# -*- coding: utf-8 -*-
"""M0 头 A 训练：Qwen3.5 / GameScaling-0.8B + LoRA(r=32) + 12 步速度回归头。

口径见 ``Docs/VRChat具身agent工程方案v3（2026-10-27）.md`` §1、§5.2、§9.2、§11：

* **主干**：``AutoModelForImageTextToText`` 载 GameScaling-0.8B / Qwen3.5-0.8B，bf16，**全部冻结**；
* **LoRA**：r=32，α=64，只加在语言模型的 12 种 Linear（21.65M）；ViT 不加（M1 才试末尾 4 block）；
* **输入**：左目两帧（当前 + 0.4 s 前，5 Hz 网格上就是往前 2 个样本），沿 ``temporal_patch_size=2``
  打成一个 patch；默认图 448×256（内容 448×252，Qwen grid [1,16,28] → **112 视觉 token**）；
* **输出**：``Linear(1024 → 24)`` 回归头，12 步 × (vx, vz)，M0–M1 用 MSE（M2 才换流匹配头）；
* **loss**：按 ``policy_labels.jsonl`` 的 ``mask`` 加权 MSE，``mask=0`` 的步不计；
* **重采样**：训练集按 ``move : trans : idle = 2 : 1 : 1``（§5.2，防「输出 0」塌缩）；
* **验证**：``split=val``（每场末 20%，§5.3），按 §5.2 的 G7 口径报「全零 / 抄当前」两条基线。

数据来自 ``research/tools/build_policy_dataset.py`` 的输出。

用法::

    # 冒烟（3 步，显存/接缝检查）
    .venv_xpu\\Scripts\\python.exe research/tools/train_policy_lora.py \\
        --dataset G:\\vrc_recording\\vrc_model\\dataset\\m0_policy \\
        --out G:\\vrc_recording\\vrc_model\\runs\\smoke --max-steps 3 --batch 1

    # M0 正式训练（4 epoch）
    .venv_xpu\\Scripts\\python.exe research/tools/train_policy_lora.py \\
        --model G:\\vrc_recording\\vrc_model\\weights\\GameScaling-0.8B-S3 \\
        --dataset G:\\vrc_recording\\vrc_model\\dataset\\m0_policy \\
        --out G:\\vrc_recording\\vrc_model\\runs\\m0_s3_lora
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

K_STEPS = 12
V_STILL = 0.1
TARGET_MODULES = [
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------- 数据
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(dataset: Path, sessions: list[str] | None) -> list[dict]:
    """把每场的 policy_obs / policy_labels 按 t 对齐，并给每个样本找「0.4 s 前」和「0.2 s 前」。"""
    samples: list[dict] = []
    sess_dirs = sorted(p for p in dataset.iterdir() if p.is_dir() and (p / "policy_obs.jsonl").exists())
    if sessions:
        sess_dirs = [p for p in sess_dirs if p.name in set(sessions)]
    for sess in sess_dirs:
        obs = read_jsonl(sess / "policy_obs.jsonl")
        lab = read_jsonl(sess / "policy_labels.jsonl")
        assert len(obs) == len(lab), sess.name
        t = np.array([o["t"] for o in obs], dtype=np.float64)
        for i, (o, r) in enumerate(zip(obs, lab)):
            if r["cls"] == "none" or sum(r["mask"]) == 0:
                continue
            # 0.4 s 前的帧：在 5 Hz 网格上即往前 2 个样本（容忍 ±0.15 s）
            j = int(np.searchsorted(t, o["t"] - 0.4, side="right") - 1)
            j = max(0, min(j, len(t) - 1))
            j2 = max(0, min(j + 1, len(t) - 1))
            k4 = j if abs(t[j] - (o["t"] - 0.4)) <= abs(t[j2] - (o["t"] - 0.4)) else j2
            prev_img = o["img_L"]
            if abs(t[k4] - (o["t"] - 0.4)) <= 0.15:
                prev_img = obs[k4]["img_L"]
            # 本体状态（不许用 t+lag 的标签）：取前一个样本（t−0.2 s）的 act[0]
            ip = int(np.searchsorted(t, o["t"] - 0.2, side="right") - 1)
            body = lab[ip]["act"][0] if ip >= 0 and abs(t[ip] - (o["t"] - 0.2)) <= 0.15 else [0.0, 0.0]
            samples.append(dict(
                sid=sess.name, t=o["t"], cls=r["cls"], split=r["split"],
                img=sess / o["img_L"], prev_img=sess / prev_img,
                act=np.asarray(r["act"], dtype=np.float32).reshape(K_STEPS, 2),
                mask=np.asarray(r["mask"], dtype=np.float32),
                wp=r["wp"], body=body,
            ))
    return samples


class PolicyDataset(Dataset):
    def __init__(self, samples: list[dict], processor, min_pixels: int, max_pixels: int):
        self.s = samples
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.s)

    def __getitem__(self, i: int):
        s = self.s[i]
        cur = np.asarray(Image.open(s["img"]).convert("RGB"))
        prev = np.asarray(Image.open(s["prev_img"]).convert("RGB"))
        video = np.stack([prev, cur])  # (T=2, H, W, 3)，T 沿 temporal_patch_size 打成一个 patch
        text = ("Body velocity vx=%.2f vz=%.2f m/s. Local goal dx=%.2f dz=%.2f m. "
                "Predict the next 12 body velocities at 0.1 s steps."
                % (s["body"][0], s["body"][1], s["wp"][0], s["wp"][1]))
        # 必须走 chat template，否则输入里没有 <|video_pad|>，视觉特征对不上任何 token
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)
        return dict(video=video, text=text, act=s["act"], mask=s["mask"],
                    wp=np.asarray(s["wp"], dtype=np.float32),
                    cls=s["cls"], split=s["split"], sid=s["sid"], t=s["t"])


def make_collate(processor, min_pixels: int, max_pixels: int):
    def collate(batch: list[dict]):
        videos = [b["video"] for b in batch]
        texts = [b["text"] for b in batch]
        inputs = processor(text=texts, videos=videos, return_tensors="pt",
                           padding=True, min_pixels=min_pixels, max_pixels=max_pixels,
                           cap_pixels_per_frame=False, do_sample_frames=False)
        return dict(
            inputs=inputs,
            act=torch.from_numpy(np.stack([b["act"] for b in batch])),
            mask=torch.from_numpy(np.stack([b["mask"] for b in batch])),
            wps=torch.from_numpy(np.stack([b["wp"] for b in batch])),
            cls=[b["cls"] for b in batch], split=[b["split"] for b in batch],
            sid=[b["sid"] for b in batch], t=[b["t"] for b in batch],
        )
    return collate


# ---------------------------------------------------------------- 模型
class ActionHead(nn.Module):
    def __init__(self, hidden: int, out: int = K_STEPS * 2):
        super().__init__()
        self.fc = nn.Linear(hidden, out)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_model(model_dir: str, chunk: int, device: str):
    from transformers import AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model

    if chunk:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        mq.torch_chunk_gated_delta_rule = functools.partial(
            mq.torch_chunk_gated_delta_rule, chunk_size=chunk)
        print(f"[chunk] chunk_size={chunk}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.to(device)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    hidden = model.config.text_config.hidden_size
    head = ActionHead(hidden).to(device)
    return model, head, hidden


def forward_batch(model, head, inputs, device: str):
    """绕过 LM head：直接取 Qwen3_5Model 的 last_hidden_state（省 248k 词表的 logits）。"""
    base = model.base_model.model  # PeftModel -> LoraModel -> Qwen3_5ForConditionalGeneration
    inner = base.model             # Qwen3_5Model（视觉塔 + LM），LoRA 已注入
    kw = {}
    for k in ("input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
              "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
        v = inputs.get(k, None)
        if v is not None:
            kw[k] = v.to(device)
    out = inner(**kw, use_cache=False)
    h = out.last_hidden_state
    idx = kw["attention_mask"].sum(-1) - 1
    pooled = h[torch.arange(h.shape[0], device=h.device), idx]
    return head(pooled.float()).view(-1, K_STEPS, 2)


def masked_mse(pred: torch.Tensor, act: torch.Tensor, mask: torch.Tensor,
               steps: int | None = None) -> torch.Tensor:
    if steps is not None:
        act, mask = act[:, :steps], mask[:, :steps]
    err = ((pred[:, :act.shape[1]] - act) ** 2).sum(-1)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------- 验证
@torch.no_grad()
def evaluate(model, head, loader, device: str) -> dict:
    model.eval()
    preds, acts, masks, clss = [], [], [], []
    for batch in loader:
        pred = forward_batch(model, head, batch["inputs"], device)
        preds.append(pred.float().cpu())
        acts.append(batch["act"])
        masks.append(batch["mask"])
        clss.extend(batch["cls"])
    pred = torch.cat(preds)
    act = torch.cat(acts)
    mask = torch.cat(masks)
    cls = np.array(clss)
    out = {"n": int(len(cls))}
    for c in ("move", "trans", "idle"):
        sel = cls == c
        if not sel.any():
            continue
        p, a, m = pred[sel], act[sel], mask[sel]
        out[c] = dict(
            n=int(sel.sum()),
            mse8=float(masked_mse(p, a, m, 8)),
            mse_all=float(masked_mse(p, a, m)),
            zero8=float(masked_mse(torch.zeros_like(a), a, m, 8)),
            copy8=float(masked_mse(a[:, :1].repeat(1, K_STEPS, 1), a, m, 8)),
        )
        if c == "move":
            v0 = p[:, 0].norm(dim=-1)
            ok = m[:, 0] > 0
            out[c]["walk_recall"] = float(((v0 > V_STILL) & ok).sum() / ok.sum().clamp_min(1))
        if c == "idle":
            first8 = p[:, :8].norm(dim=-1)
            ok = m[:, :8].sum(-1) == 8
            out[c]["stop_recall"] = float(((first8 < V_STILL).all(-1) & ok).sum() / ok.sum().clamp_min(1))
        if c == "trans":
            first8 = p[:, :8].norm(dim=-1)
            ok = m[:, :8].sum(-1) > 0
            out[c]["start_recall"] = float((((first8 > V_STILL) & (m[:, :8] > 0)).any(-1) & ok).sum() /
                                           ok.sum().clamp_min(1))
    model.train()
    return out


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M0 头 A：Qwen3.5/GameScaling-0.8B + LoRA + 12 步回归头")
    ap.add_argument("--model", default=r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default=r"G:\vrc_recording\vrc_model\dataset\m0_policy")
    ap.add_argument("--out", default=r"G:\vrc_recording\vrc_model\runs\m0_lora")
    ap.add_argument("--sessions", nargs="*", default=None, help="只用这些场（冒烟用）")
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256], metavar=("W", "H"),
                    help="policy 图尺寸，必须与 build_policy_dataset 一致")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=128, help="GatedDeltaNet chunk_size；0=默认 64")
    ap.add_argument("--max-steps", type=int, default=0, help=">0 只跑这么多步（冒烟）")
    ap.add_argument("--max-val", type=int, default=0, help=">0 只评这么多验证样本（冒烟）")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-resample", action="store_true", help="关掉 2:1:1 重采样")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = a.device if a.device != "auto" else (
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} args={vars(a)}", flush=True)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = int(a.size[0]) * int(a.size[1])  # 别让默认 min_pixels=65536 改掉我们的尺寸

    samples = build_index(Path(a.dataset), a.sessions)
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    if not train or not val:
        print("数据切分里没有 train 或 val，检查 build_policy_dataset 的 split 字段")
        return 2
    if a.max_val:
        val = val[: a.max_val]
    print(f"[data] train={len(train)} val={len(val)} cls={dict(Counter(s['cls'] for s in train))}", flush=True)

    ds_train = PolicyDataset(train, processor, min_px, max_px)
    ds_val = PolicyDataset(val, processor, min_px, max_px)
    collate = make_collate(processor, min_px, max_px)

    if a.no_resample:
        sampler = None
        shuffle = True
    else:
        # 2:1:1；权重取「目标比例 / 实际数量」，每 epoch 内生效
        ratios = {"move": 2.0, "trans": 1.0, "idle": 1.0}
        cnt = Counter(s["cls"] for s in train)
        w = torch.tensor([ratios[s["cls"]] / cnt[s["cls"]] for s in train], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(train), replacement=True)
        shuffle = False
    dl_train = DataLoader(ds_train, batch_size=a.batch, sampler=sampler, shuffle=shuffle,
                          num_workers=a.num_workers, collate_fn=collate,
                          persistent_workers=a.num_workers > 0)
    dl_val = DataLoader(ds_val, batch_size=a.batch, shuffle=False,
                        num_workers=a.num_workers, collate_fn=collate,
                        persistent_workers=a.num_workers > 0)

    model, head, hidden = load_model(a.model, a.chunk, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in trainable)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"[model] hidden={hidden} lora={n_lora/1e6:.2f}M head={n_head/1e3:.1f}k "
          f"peak_alloc={torch.xpu.memory_allocated()/2**30:.2f} GiB" if device == "xpu" else
          f"[model] hidden={hidden} lora={n_lora/1e6:.2f}M head={n_head/1e3:.1f}k", flush=True)

    opt = torch.optim.AdamW([
        {"params": trainable, "lr": a.lr_lora, "weight_decay": a.weight_decay},
        {"params": list(head.parameters()), "lr": a.lr_head, "weight_decay": 0.0},
    ])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=1), encoding="utf-8")
    log_path = out / "train_log.jsonl"
    log_f = log_path.open("a", encoding="utf-8")

    def save(tag: str):
        d = out / tag
        (d / "adapter").mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d / "adapter")
        torch.save(head.state_dict(), d / "head.pt")
        (d / "meta.json").write_text(json.dumps(
            {"hidden": hidden, "steps": K_STEPS, "model": a.model, "tag": tag}, indent=1), encoding="utf-8")

    best = (math.inf, None)
    step = 0
    model.train()
    for epoch in range(a.epochs):
        t_ep = time.perf_counter()
        for batch in dl_train:
            t_step = time.perf_counter()
            if step == 0:
                print("[inputs] " + json.dumps(
                    {k: (list(v.shape) if torch.is_tensor(v) else str(type(v).__name__))
                     for k, v in batch["inputs"].items()}, ensure_ascii=False)
                    + " grid=" + str(batch["inputs"].get("video_grid_thw", None)), flush=True)
            pred = forward_batch(model, head, batch["inputs"], device)
            loss = masked_mse(pred, batch["act"].to(device), batch["mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), a.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.perf_counter() - t_step
            loss_val = loss.item()
            if step % 10 == 0 or step == 1:
                print(f"[train] ep={epoch} step={step} loss={loss_val:.4f} "
                      f"{dt*1000:.0f} ms/step {a.batch/dt:.1f} 样本/s", flush=True)
            log_f.write(json.dumps({"ep": epoch, "step": step, "loss": round(loss_val, 5),
                                    "ms": round(dt * 1000, 1)}) + "\n")
            log_f.flush()
            if a.max_steps and step >= a.max_steps:
                break
        val_metrics = evaluate(model, head, dl_val, device)
        val_metrics["ep"] = epoch
        print(f"[val] ep={epoch} {json.dumps(val_metrics, ensure_ascii=False)}", flush=True)
        log_f.write(json.dumps(val_metrics, ensure_ascii=False) + "\n")
        log_f.flush()
        save(f"epoch_{epoch:02d}")
        mse8 = val_metrics.get("move", {}).get("mse8", math.inf)
        if mse8 < best[0]:
            best = (mse8, epoch)
            save("best")
        print(f"[epoch] ep={epoch} {time.perf_counter() - t_ep:.1f}s move8={mse8:.3f} best={best}", flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    log_f.close()
    (out / "summary.json").write_text(json.dumps(
        {"best_move_mse8": best[0], "best_epoch": best[1], "steps": step,
         "peak_gib": round(torch.xpu.max_memory_allocated() / 2 ** 30, 2)
         if device == "xpu" else None}, indent=1), encoding="utf-8")
    print(f"[done] out={out} best={best}", flush=True)
    if device == "xpu":
        print(f"[mem] peak={torch.xpu.max_memory_allocated()/2**30:.2f} GiB "
              f"reserved={torch.xpu.memory_reserved()/2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
