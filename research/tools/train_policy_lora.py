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
    """训练期的两条捷径切断（2026-11-02 视觉消融 + 方差分解之后新增）。

    背景：训出来的模型换图后动作块 cos=0.983（几乎不看图），但同一批视觉特征
    能让线性探针的误差降 56%（2.87 -> 1.27），留出整个世界时增益更大（1.45）。
    所以不是图里没信息，是训练停在了「抄 body」的局部最优上 —— body 单独就有
    mse 3.92，而模型最终只做到 1.72。

    两个 dropout 分别切一条捷径，只在 training=True 时生效：
      * body_dropout   —— 把 body 置 0，逼模型不能只靠上一拍速度；
      * vision_dropout —— 把两帧置黑，逼模型别把「视觉特征恒定」当基线，
        同时充当方案 §11.4 的 geometry dropout 契约（缺失时退化为无视觉）。
    """

    def __init__(self, samples: list[dict], processor, min_pixels: int, max_pixels: int,
                 training: bool = False, vision_dropout: float = 0.0,
                 body_dropout: float = 0.0, seed: int = 0):
        self.s = samples
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.training = bool(training)
        self.vision_dropout = float(vision_dropout) if training else 0.0
        self.body_dropout = float(body_dropout) if training else 0.0
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.s)

    def __getitem__(self, i: int):
        s = self.s[i]
        cur = np.asarray(Image.open(s["img"]).convert("RGB"))
        prev = np.asarray(Image.open(s["prev_img"]).convert("RGB"))
        video = np.stack([prev, cur])  # (T=2, H, W, 3)，T 沿 temporal_patch_size 打成一个 patch
        body = s["body"]
        if self.body_dropout > 0.0 or self.vision_dropout > 0.0:
            # 每个 epoch 用不同随机流，否则同一张图永远被同一路 dropout 命中
            rng = np.random.default_rng((self.seed * 1000003 + self.epoch) * 7919 + i)
            if self.vision_dropout > 0.0 and rng.random() < self.vision_dropout:
                video = np.zeros_like(video)
            if self.body_dropout > 0.0 and rng.random() < self.body_dropout:
                body = (0.0, 0.0)
        text = ("Body velocity vx=%.2f vz=%.2f m/s. Local goal dx=%.2f dz=%.2f m. "
                "Predict the next 12 body velocities at 0.1 s steps."
                % (body[0], body[1], s["wp"][0], s["wp"][1]))
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
# ---------------------------------------------------------------- 视觉注入
class VisualInjector(nn.Module):
    """把视觉塔输出**绕开 LM** 直接接进动作头（2026-11-02，policy-designer 评审后）。

    为什么要绕开：ViT LoRA 实验已证伪「解冻 ViT」这条路（视觉敏感度 1-cos
    从 0.017 掉到 0.006，且 shuffle 反而比 black 扰动小，退化成存在性检测器）。
    policy-designer 的判断是通路瓶颈 —— 112 个视觉 token 必须经 22 层线性注意力
    汇总到末位 token 才能到输出头。这里直接把视觉特征接到 head 输入。

    **不用 global mean**：112 个 token 是 8x14 空间网格，全局平均等于又做一次
    112->1 的漏斗，把「哪边有东西」抹平 —— 而那正是避障要的量。改成 2x2 分块
    均值（每块 4x7 个 token）保留粗空间分布，再各过一个共享 Linear 压到 proj 维。

    参数量：1024xproj + proj*4，proj=256 时约 290k。
    """

    def __init__(self, dim: int, grid=(8, 14), blocks: int = 2, proj: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.grid_h, self.grid_w = int(grid[0]), int(grid[1])
        self.blocks = int(blocks)
        self.proj = nn.Linear(self.dim, int(proj))
        n = self.blocks * self.blocks
        self.out_dim = n * int(proj)

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        """vis: (B, N, D) 或 (B*N, D)。返回 (B, out_dim)。"""
        if vis.dim() == 2:
            b = self._batch if self._batch else 1
            vis = vis.view(b, -1, vis.shape[-1])
        b, n, d = vis.shape
        gh, gw = self.grid_h, self.grid_w
        if n != gh * gw:
            # 形状对不上就退回全局均值（保持可跑，并在日志里能看出来）
            return self.proj(vis.mean(1)).repeat(1, self.blocks * self.blocks)
        x = vis.view(b, gh, gw, d).permute(0, 3, 1, 2)      # (B, D, gh, gw)
        k = self.blocks
        x = x.view(b, d, k, gh // k, k, gw // k).mean(dim=(3, 5))  # (B, D, k, k)
        x = x.permute(0, 2, 3, 1).reshape(b, k * k, d)      # (B, k*k, D)
        return self.proj(x).reshape(b, -1)


class ActionHead(nn.Module):
    """``extra > 0`` 时接受「文本池化 + 视觉注入」拼接后的输入。"""

    def __init__(self, hidden: int, out: int = K_STEPS * 2, extra: int = 0):
        super().__init__()
        self.in_dim = int(hidden) + int(extra)
        self.fc = nn.Linear(self.in_dim, out)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, vis: torch.Tensor | None = None) -> torch.Tensor:
        if vis is not None:
            x = torch.cat([x, vis], dim=-1)
        return self.fc(x)


class EventHead(nn.Module):
    """事件头：预测未来 1.2 s 内会不会发生 go / stop（policy-designer 的 P3）。

    为什么要它：加容量不产生压力，反而给坍缩提供自由度 —— ViT LoRA 让视觉特征
    更容易被忽略就是这个机制。事件是 **body 预测不了的**：你在走多快，和「接下来
    会不会起步/停下」是两件事（实测 body 单独 AUC 0.63，图像统计 0.69，满速带 0.77）。
    标签零成本，从 act/mask 推。
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(int(in_dim), 2)   # [go, stop]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def event_labels(act: torch.Tensor, mask: torch.Tensor,
                 still_mps: float = 0.1) -> torch.Tensor:
    """(B, K, 2) + (B, K) -> (B, 2) 的 [go, stop] 二值标签。"""
    spd = act.norm(dim=-1)                       # (B, K)
    valid = mask > 0
    moving = (spd > still_mps) & valid
    # ⚠️ first 是 (B,)，rest 是 (B,K-1)；直接 & 会按最后一维对齐（2 vs 11）报错，
    # 必须显式补一维。
    first = moving[:, 0:1]                       # (B, 1)
    rest = moving[:, 1:]                         # (B, K-1)
    rest_valid = valid[:, 1:]
    go = ((~first) & rest & rest_valid).any(dim=1)
    stop = (first & (~rest) & rest_valid).any(dim=1)
    return torch.stack([go, stop], dim=-1).float()


def load_model(model_dir: str, chunk: int, device: str, vit_lora_blocks: int = 0,
               vit_lora_r: int = 16):
    """``vit_lora_blocks > 0`` 时给视觉塔**末尾 N 个 block** 加 LoRA（方案 §11.1 的 M1 试验）。

    为什么必须加（2026-11-02）：实测可训练参数 21,645,312 全部在 LM，视觉塔占 0.00%，
    TARGET_MODULES 在视觉塔一个都没命中（视觉塔的 Linear 叶名是 qkv / proj /
    linear_fc1 / linear_fc2）。也就是说 **ViT 从来没有被这个任务训练过**。
    而方差分解显示它的特征里含有能让误差降 56% 的信息 —— 需要给它梯度通道。

    只挂末尾 N 个 block（不动前面），理由与方案 §11.1 一致：目标差异在高层语义，
    且参数量可控（每 block 约 0.3M）。
    """
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
    # ---- 视觉塔 LoRA：只挂末尾 N 个 block 的 attn.qkv / attn.proj ----
    # 视觉塔的 Linear 叶名与 LM 不同（qkv / proj / linear_fc1 / linear_fc2），
    # 所以 TARGET_MODULES 覆盖不到它。
    #
    # ⚠️ 必须和 LM 的 LoRA **一次性注入**：早先是先 get_peft_model(LM) 再
    # get_peft_model(visual)，结果层级变成 PeftModel>LoraModel>PeftModel>LoraModel，
    # forward_batch 里 `base_model.model` 拿到的是 LoraModel 而不是
    # Qwen3_5ForConditionalGeneration，前向直接 AttributeError。
    vit_targets: list[str] = []
    vit_desc = ""
    if vit_lora_blocks > 0:
        import re as _re
        visual = model.model.visual
        n_blocks = len(visual.blocks)
        start = max(0, n_blocks - int(vit_lora_blocks))
        for name, _mod in visual.named_modules():
            m = _re.match(r"blocks\.(\d+)\.attn\.(qkv|proj)$", name)
            if m and int(m.group(1)) >= start:
                vit_targets.append("model.visual." + name)
        if not vit_targets:
            raise RuntimeError("ViT LoRA: 没匹配到任何视觉模块，检查命名")
        vit_desc = "blocks %d..%d, %d 个模块" % (start, n_blocks - 1, len(vit_targets))

    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=list(TARGET_MODULES) + vit_targets, task_type="CAUSAL_LM"))
    if vit_targets:
        n_vit = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and "visual" in n.lower())
        print("[vit-lora] %s, 可训练 %d" % (vit_desc, n_vit), flush=True)

    hidden = model.config.text_config.hidden_size
    head = ActionHead(hidden).to(device)
    return model, head, hidden


def build_extra_heads(hidden: int, device, inject_visual: bool, visual_proj: int,
                      visual_grid, event_head: bool, event_weight: float):
    """按开关造注入器/事件头，并返回 (injector, event_head, extra_dim, params)。"""
    injector = None
    extra = 0
    params = []
    if inject_visual:
        injector = VisualInjector(hidden, grid=tuple(visual_grid), proj=visual_proj).to(device)
        extra = injector.out_dim
        params += list(injector.parameters())
    ev = None
    if event_head:
        ev = EventHead(hidden + extra).to(device)
        params += list(ev.parameters())
    return injector, ev, extra, params


def _pname_of(model, target):
    for n, p in model.named_parameters():
        if p is target:
            return n
    return ""


def _extract_visual_feat(out, batch: int):
    """从视觉塔返回值里取出 (B, N, D) 的 token 特征。

    ⚠️ 实测形状（2026-11-02，Qwen3.5-0.8B，grid [1,16,28]）：
        out.last_hidden_state = (B*N_img, 768)      <- 不是我们要的
        out.pooler_output    = ((112,1024), (112,1024), ...)  <- **每样本一个张量的 tuple**
    早先按「pooler_output 是单个张量」写，会拿到 tuple 里的第一个样本，
    然后按 batch 去 view，报 "shape [2,-1,1024] invalid for input of size 50176"。
    也试过递归找「第一个 >=2 维张量」，结果抓到 last_hidden_state (896,768)，
    维度不对（768 不是 1024），同样会错。所以这里按语义精确取。
    """
    po = getattr(out, "pooler_output", None)
    if po is None:
        return None
    if torch.is_tensor(po):
        if po.dim() == 3:
            return po
        if po.dim() == 2 and batch > 0 and po.shape[0] % batch == 0:
            return po.view(batch, -1, po.shape[-1])
        return None
    if isinstance(po, (tuple, list)):
        parts = [x for x in po if torch.is_tensor(x) and x.dim() == 2]
        if len(parts) == batch and len(parts) > 0:
            if len({tuple(x.shape) for x in parts}) == 1:
                return torch.stack(parts, 0)
        if len(parts) == 1 and parts[0].shape[0] % max(batch, 1) == 0:
            return parts[0].view(batch, -1, parts[0].shape[-1])
    return None


def _find_visual_tensor(obj, depth: int = 0):
    """从视觉塔的返回值里挖出 token 特征张量。

    ⚠️ 实测 `Qwen3_5VisionModel` 返回的是 `BaseModelOutputWithPooling`，而它的
    `pooler_output` 字段**本身又是一个 tuple**（不是张量）。所以不能只试一层：
    早先按 `feat.pooler_output` 直接当张量用，shape 报 AttributeError，
    注入静默失效（head 输入 1024 而权重是 2048，报 mat1/mat2 不匹配）。
    这里递归找第一个二维以上、最后一维等于 hidden 的张量。
    """
    if depth > 4:
        return None
    if torch.is_tensor(obj):
        return obj if obj.dim() >= 2 else None
    for name in ("pooler_output", "last_hidden_state"):
        v = getattr(obj, name, None)
        if v is not None:
            r = _find_visual_tensor(v, depth + 1)
            if r is not None:
                return r
    if isinstance(obj, (tuple, list)):
        for x in obj:
            r = _find_visual_tensor(x, depth + 1)
            if r is not None:
                return r
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            if v is None or isinstance(v, (str, int, float, bool)):
                continue
            r = _find_visual_tensor(v, depth + 1)
            if r is not None:
                return r
    return None


def forward_batch(model, head, inputs, device: str, injector=None, event_head=None):
    """绕过 LM head，取 last_hidden_state 的末位 token 池化 —— 视需要再接视觉注入。

    ``injector`` 非 None 时：在同一次前向里用 forward hook 取视觉塔输出，
    绕开 LM 直接拼进动作头。**必须复用这一次前向的视觉输出**，不能另跑一次
    visual（p95 会从 164 ms 翻上去，G1 直接挂 —— policy-designer 的 P4）。
    """
    base = model.base_model.model  # PeftModel -> LoraModel -> Qwen3_5ForConditionalGeneration
    inner = base.model             # Qwen3_5Model（视觉塔 + LM），LoRA 已注入
    kw = {}
    for k in ("input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
              "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
        v = inputs.get(k, None)
        if v is not None:
            kw[k] = v.to(device)

    vis_cap = {}
    handle = None
    if injector is not None:
        visual = getattr(inner, "visual", None)
        if visual is not None:
            def _hook(module, args, kwargs, output):
                vis_cap["feat"] = output
            handle = visual.register_forward_hook(_hook, with_kwargs=True)

    try:
        out = inner(**kw, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    h = out.last_hidden_state
    idx = kw["attention_mask"].sum(-1) - 1
    ar = torch.arange(h.shape[0], device=h.device)
    pooled = h[ar, idx].float()

    vis = None
    if injector is not None:
        feat = _extract_visual_feat(vis_cap.get("feat"), h.shape[0])
        if feat is not None:
            injector._batch = h.shape[0]
            vis = injector(feat.float())

    pred = head(pooled, vis).view(-1, K_STEPS, 2)
    if event_head is not None:
        ev_in = torch.cat([pooled, vis], -1) if vis is not None else pooled
        return pred, event_head(ev_in)
    return pred


def masked_mse(pred: torch.Tensor, act: torch.Tensor, mask: torch.Tensor,
               steps: int | None = None) -> torch.Tensor:
    if steps is not None:
        act, mask = act[:, :steps], mask[:, :steps]
    err = ((pred[:, :act.shape[1]] - act) ** 2).sum(-1)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------- 验证
@torch.no_grad()
def evaluate(model, head, loader, device: str, injector=None, event_head=None) -> dict:
    model.eval()
    preds, acts, masks, clss = [], [], [], []
    ev_logits, ev_true = [], []
    for batch in loader:
        r = forward_batch(model, head, batch["inputs"], device,
                          injector=injector, event_head=event_head)
        if isinstance(r, tuple):
            pred, ev = r
            ev_logits.append(ev.float().cpu())
            ev_true.append(event_labels(batch["act"], batch["mask"]))
        else:
            pred = r
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

    # 事件头 AUC：go / stop 两类分别报。body 单独 0.63、图像统计 0.69 是外部参照线。
    if ev_logits:
        lg = torch.cat(ev_logits)
        tr = torch.cat(ev_true)
        out["event"] = {}
        for j, nm in enumerate(("go", "stop")):
            y, s = tr[:, j].numpy(), lg[:, j].numpy()
            pos, neg = s[y > 0.5], s[y <= 0.5]
            if pos.size and neg.size:
                # AUC = P(正样本分数 > 负样本分数)，用秩公式避免 sklearn 依赖
                allv = np.concatenate([pos, neg])
                order = np.argsort(allv, kind="mergesort")
                ranks = np.empty_like(order, dtype=np.float64)
                ranks[order] = np.arange(1, allv.size + 1)
                rp = ranks[:pos.size].sum()
                auc = (rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
                out["event"][nm] = dict(n_pos=int(pos.size), n_neg=int(neg.size),
                                        auc=float(auc))
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
    # ---- 视觉承重相关的配方（2026-11-02 方差分解之后新增，默认全关=旧行为）----
    ap.add_argument("--vit-lora-blocks", type=int, default=0,
                    help="视觉塔末尾 N 个 block 加 LoRA；0=全冻结（旧行为）")
    ap.add_argument("--vit-lora-r", type=int, default=16)
    ap.add_argument("--vision-dropout", type=float, default=0.0,
                    help="两帧置黑的概率（训练期），逼模型不把视觉当恒定输入")
    ap.add_argument("--body-dropout", type=float, default=0.0,
                    help="body 置 0 的概率（训练期）。已被证伪（+4.6~5.7%%），默认关")
    ap.add_argument("--inject-visual", action="store_true",
                    help="把视觉塔输出绕开 LM 直接接进动作头（2x2 分块池化）")
    ap.add_argument("--visual-proj", type=int, default=256)
    ap.add_argument("--visual-grid", nargs=2, type=int, default=[8, 14])
    ap.add_argument("--event-head", action="store_true",
                    help="挂事件头（未来 1.2 s 是否 go/stop），给视觉加压")
    ap.add_argument("--event-weight", type=float, default=0.3)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile 内层模型。CPU 派发是训练瓶颈（见 CPU派发开销剖析 §2）")
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

    ds_train = PolicyDataset(train, processor, min_px, max_px, training=True,
                             vision_dropout=a.vision_dropout,
                             body_dropout=a.body_dropout, seed=a.seed)
    ds_val = PolicyDataset(val, processor, min_px, max_px)  # 验证集永不做 dropout
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

    model, head, hidden = load_model(a.model, a.chunk, device,
                                     vit_lora_blocks=a.vit_lora_blocks,
                                     vit_lora_r=a.vit_lora_r)
    injector, ev_head, extra_dim, extra_params = build_extra_heads(
        hidden, device, a.inject_visual, a.visual_proj, a.visual_grid,
        a.event_head, a.event_weight)
    if a.inject_visual:
        # 注入维度变了，head 必须重建（旧 head.pt 形状对不上）
        head = ActionHead(hidden, extra=extra_dim).to(device)

    # ---- torch.compile 内层模型 ----
    # 为什么：训练每步 1174 ms，而数据准备只占 27 ms、GPU 占用只有 50%。
    # Docs/CPU派发开销剖析-2026-11-01.md §2 实测 eager 前向 110,644 个 kernel / 388 ms，
    # compile 后 48,454 / 179 ms —— 派发时间几乎严格正比于 kernel 数。
    # 训练 ≈ 前向 + 反传 ≈ 3x 前向 kernel，388x3 ≈ 1.16 s，与实测 1174 ms 吻合。
    # 所以瓶颈是 CPU 提交 kernel，不是显卡算力。
    if a.compile:
        # ⚠️ 必须先把 oneAPI 工具链塞进环境，否则 Inductor 报
        # "Failed to find C compiler. Please specify via CC environment variable"。
        # policy_runtime.setup_oneapi() 做的就是这件事，但本模块不能顶层 import 它
        # （policy_runtime 反过来 import 本模块，会循环），所以在这里惰性复制一遍。
        _bin = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
        import os as _os
        if _os.path.isdir(_bin):
            _os.environ["PATH"] = _bin + _os.pathsep + _os.environ.get("PATH", "")
            _os.environ.setdefault("ONEAPI_ROOT", r"D:\Program Files (x86)\Intel\oneAPI")
            _os.environ.setdefault("CC", "icpx")
            _os.environ.setdefault("CXX", "icpx")
            print("[compile] oneAPI 工具链已入 PATH (CC=%s)" % _os.environ["CC"], flush=True)
        else:
            print("[compile] 警告：oneAPI bin 不存在，编译大概率失败", flush=True)
        # 编译缓存默认落在 %TEMP%\torchinductor_<user>（实测一次 293 MB / 6235 个文件），
        # 而本脚本没走 policy_runtime.setup_oneapi()，所以不会自动指到仓库 .tc_cache。
        _cache = Path(__file__).resolve().parents[2] / ".tc_cache" / "inductor"
        _cache.mkdir(parents=True, exist_ok=True)
        _os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache))

        # Triton autotune 是**串行**的：一个 Python 线程逐个配置在 GPU 上试跑，
        # 所以 CPU 只有 1 核多、GPU 只有 18%，两边看着都「空闲」。实测 30 min 走不完，
        # 而训练本身只跑几十步。关掉 autotune 换编译时间。
        try:
            import torch._inductor.config as _ic
            _ic.max_autotune = False
            _ic.max_autotune_gemm = False
            _ic.coordinate_descent_tuning = False
            print("[compile] autotune 已关闭（编译时间优先）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print("[compile] 关 autotune 失败：%r" % (_e,), flush=True)
        print("[compile] torch.compile inner model …", flush=True)
        # model = PeftModel > base_model = LoraModel > model = Qwen3_5ForConditionalGeneration
        #                                              > model = Qwen3_5Model（要编译的是它）
        # ⚠️ 早先写成 model.base_model.model.model.model，多下了一层，
        # 拿到 None 就「跳过」了 —— A/B 两组都是未编译，白跑。
        _inner = model.base_model.model
        _lm = getattr(_inner, "model", None)
        if _lm is not None:
            _inner.model = torch.compile(_lm)
        else:
            print("[compile] 找不到内层 Qwen3_5Model，跳过", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in trainable)
    n_head = sum(p.numel() for p in head.parameters())
    if extra_params:
        print("[inject] visual=%s event=%s extra_dim=%d extra_params=%d"
              % (a.inject_visual, a.event_head, extra_dim,
                 sum(p.numel() for p in extra_params)), flush=True)
    print(f"[model] hidden={hidden} lora={n_lora/1e6:.2f}M head={n_head/1e3:.1f}k "
          f"peak_alloc={torch.xpu.memory_allocated()/2**30:.2f} GiB" if device == "xpu" else
          f"[model] hidden={hidden} lora={n_lora/1e6:.2f}M head={n_head/1e3:.1f}k", flush=True)

    opt = torch.optim.AdamW([
        {"params": trainable, "lr": a.lr_lora, "weight_decay": a.weight_decay},
        {"params": list(head.parameters()) + extra_params, "lr": a.lr_head,
         "weight_decay": 0.0},
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
        if injector is not None:
            torch.save(injector.state_dict(), d / "injector.pt")
        if ev_head is not None:
            torch.save(ev_head.state_dict(), d / "event_head.pt")
        (d / "meta.json").write_text(json.dumps(
            {"hidden": hidden, "steps": K_STEPS, "model": a.model, "tag": tag}, indent=1), encoding="utf-8")

    best = (math.inf, None)
    step = 0
    model.train()
    for epoch in range(a.epochs):
        # dropout 的随机流按 epoch 变，否则同一张图每个 epoch 都被同一路命中
        ds_train.epoch = epoch
        t_ep = time.perf_counter()
        for batch in dl_train:
            t_step = time.perf_counter()
            if step == 0:
                print("[inputs] " + json.dumps(
                    {k: (list(v.shape) if torch.is_tensor(v) else str(type(v).__name__))
                     for k, v in batch["inputs"].items()}, ensure_ascii=False)
                    + " grid=" + str(batch["inputs"].get("video_grid_thw", None)), flush=True)
            r = forward_batch(model, head, batch["inputs"], device,
                              injector=injector, event_head=ev_head)
            if isinstance(r, tuple):
                pred, ev_logit = r
            else:
                pred, ev_logit = r, None
            act_d = batch["act"].to(device)
            mask_d = batch["mask"].to(device)
            loss = masked_mse(pred, act_d, mask_d)
            if ev_logit is not None:
                # 事件头 loss：给视觉加压（body 预测不了 go/stop）。只对 mask 有效的块算。
                ev_y = event_labels(act_d, mask_d)
                valid_blk = mask_d[:, :8].sum(-1) > 0
                if bool(valid_blk.any()):
                    l_ev = torch.nn.functional.binary_cross_entropy_with_logits(
                        ev_logit[valid_blk], ev_y[valid_blk])
                    loss = loss + a.event_weight * l_ev
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable + list(head.parameters()) + extra_params, a.grad_clip)
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
        val_metrics = evaluate(model, head, dl_val, device,
                               injector=injector, event_head=ev_head)
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
