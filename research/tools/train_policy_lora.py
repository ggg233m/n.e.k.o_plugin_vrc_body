# -*- coding: utf-8 -*-
"""M0 头 A 训练：Qwen3.5 / GameScaling-0.8B + LoRA(r=32) + 12 步速度回归头。

口径见 ``Docs/VRChat具身agent工程方案v3（2026-10-27）.md`` §1、§5.2、§9.2、§11：

* **主干**：``AutoModelForImageTextToText`` 载 GameScaling-0.8B / Qwen3.5-0.8B，bf16，**全部冻结**；
* **LoRA**：r=32，α=64，只加在语言模型的 12 种 Linear（21.65M）；ViT 不加（M1 才试末尾 4 block）；
* **输入**：左目两帧（当前 + 0.2 s 前，5 Hz 网格上就是往前 1 个样本），沿 ``temporal_patch_size=2``
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
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

# ⚠️ 必须在 `import torch` **之前**设，而且必须在 `from pathlib import Path` **之后**
# （第一版写在了前面，直接 NameError: name 'Path' is not defined，脚本一行都没跑）。
# 为什么非得在 import torch 之前：torch 在 import 时就把 TORCHINDUCTOR_CACHE_DIR
# 写成它自己的默认值（%TEMP%\torchinductor_<user>），之后再 setdefault 已经晚了。
# 后果：每轮训练从零编译，白烧 6–8 分钟，缓存还丢在系统临时目录里。
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".tc_cache" / "inductor"),
)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_text import build_guidance_text, build_policy_text  # noqa: E402
from policy_histframe import HISTORY_FRAME_S  # noqa: E402  视觉历史帧间隔唯一真源
import input_schema  # noqa: E402  输入口径 manifest（推理侧按它重建输入卡）
import policy_readout  # noqa: E402  S2 读出（4-query / additive-gated fusion）

K_STEPS = 12
STEP_S = 0.1   # 与 policy_executor.STEP_S 同一口径（12 步 = 1.2 s 开环）
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


def select_val_subset(dataset: Path, n: int) -> list[dict]:
    """评测与基线共用的 **val 前 n 条** 选样口径。

    为什么要抽出来（2026-11-03）：ridge 与模型的对比一度不可比 —— 两边各自
    `build_index` 后取前 400，但因为过滤/排序不同，实际取到的是不同的 400 条，
    连 copy 基线都差 54%（1.80 vs 2.78）。任何跨模型对比都必须走同一个函数。
    """
    vals = [s for s in build_index(dataset, None) if s["split"] == "val"]
    return vals[: int(n)]


def build_index(dataset: Path, sessions: list[str] | None) -> list[dict]:
    """把每场的 policy_obs / policy_labels 按 t 对齐，并给每个样本找「历史视觉帧」和「本体状态帧」。

    历史视觉帧间隔见 policy_histframe.HISTORY_FRAME_S（当前 0.2 s）；
    本体状态固定取 t−0.2 s（与历史视觉帧同点，语义是「上一拍速度」，不随视觉间隔变）。
    """
    samples: list[dict] = []
    sess_dirs = sorted(p for p in dataset.iterdir() if p.is_dir() and (p / "policy_obs.jsonl").exists())
    if sessions:
        sess_dirs = [p for p in sess_dirs if p.name in set(sessions)]
    for sess in sess_dirs:
        obs = read_jsonl(sess / "policy_obs.jsonl")
        lab = read_jsonl(sess / "policy_labels.jsonl")
        _trav_path = sess / "traversability.jsonl"
        _trav_rows = read_jsonl(_trav_path) if _trav_path.exists() else []
        _trav_by_i = {int(x["i"]): x for x in _trav_rows if x.get("i") is not None}
        assert len(obs) == len(lab), sess.name
        t = np.array([o["t"] for o in obs], dtype=np.float64)
        for i, (o, r) in enumerate(zip(obs, lab)):
            if r["cls"] == "none" or sum(r["mask"]) == 0:
                continue
            # 历史视觉帧：在 5 Hz 网格上就是往前 HISTORY_FRAME_STEPS(=1) 个样本，
            # 即 t−0.2 s（与观测定点网格严格对齐）。容差 ±0.15 s 保留原有行为。
            # 2026-11-05 由 t−0.4 s 改为 t−0.2 s：实测端点漏检 26.9% → 17.8%。
            _hist_s = HISTORY_FRAME_S
            j = int(np.searchsorted(t, o["t"] - _hist_s, side="right") - 1)
            j = max(0, min(j, len(t) - 1))
            j2 = max(0, min(j + 1, len(t) - 1))
            k4 = j if abs(t[j] - (o["t"] - _hist_s)) <= abs(t[j2] - (o["t"] - _hist_s)) else j2
            prev_img = o["img_L"]
            if abs(t[k4] - (o["t"] - _hist_s)) <= 0.15:
                prev_img = obs[k4]["img_L"]
            # 本体状态（不许用 t+lag 的标签）：取前一个样本（t−0.2 s）的 act[0]
            ip = int(np.searchsorted(t, o["t"] - 0.2, side="right") - 1)
            body = (list(lab[ip]["act"][0][:2]) if ip >= 0 and abs(t[ip] - (o["t"] - 0.2)) <= 0.15 else [0.0, 0.0])
            samples.append(dict(
                sid=sess.name, t=o["t"], cls=r["cls"], split=r["split"],
                img=sess / o["img_L"], prev_img=sess / prev_img,
                act=np.asarray(r["act"], dtype=np.float32).reshape(K_STEPS, -1),
                mask=np.asarray(r["mask"], dtype=np.float32),
                wp=r["wp"], body=body,
                # 方案 §10.6 的记忆卡字段；老数据集没有就退化成 0（缺失值契约）
                mode=int(r.get("mode", 0) or 0),
                topo=tuple(r.get("topo") or (0.0, 0.0, 0.0)),
                path=r.get("path"),
                guide=r.get("guide"),
                control_mode=r.get("control_mode", "legacy"),
                traversability=_trav_by_i.get(int(o.get("i", -1))),
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

        **2026-11-03 修复一个评测陷阱**：默认值从 0.0 提到 **0.08**。
        之前训练里从没见过全黑输入，于是离线评测的 `black` 是分布外，
        实测 `black=1.2262 < real=1.4917`（黑比真图还准），把「模型忽略视觉」
        错误呈现成「黑图更好」。加了 dropout 后「无视觉」落进训练分布，
        black/real/shuffle 三者的比较才有意义。
    """

    def __init__(self, samples: list[dict], processor, min_pixels: int, max_pixels: int,
                 training: bool = False, vision_dropout: float = 0.0,
                 body_dropout: float = 0.0, seed: int = 0,
                 topo_dropout: float = 0.2, prompt_format: str = "card",
                 quad_feats: bool = False, drop_wp: bool = False,
                 path_in: bool = False, guidance_corrupt_p: float = 0.0,
                 guidance_jitter_m: float = 0.75,
                 guidance_stale_p: float = 0.15,
                 guidance_drop_p: float = 0.15):
        self.s = samples
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.training = bool(training)
        self.vision_dropout = float(vision_dropout) if training else 0.0
        self.body_dropout = float(body_dropout) if training else 0.0
        self.topo_dropout = float(topo_dropout) if training else 0.0
        self.seed = int(seed)
        # ⚠️ prompt 格式必须可回退。2026-11-02 换成记忆卡之后，**用新格式去评旧
        # checkpoint 会得出错误的绝对误差**（实测同一模型 real mse8 从 1.2467 涨到
        # 1.9644，全是因为喂了它没训过的格式）。旧 run 的 meta.json 里没有
        # prompt_format 字段，据此判定为 legacy。
        self.prompt_format = str(prompt_format or "card")
        # 二次特征直接拼进 head（P0-1b，2026-11-04）：8 -> 44 维
        self.quad_feats = bool(quad_feats)
        # 去掉 wp（2026-11-04）：wp 是 waypoint_body 沿录制轨迹**往前取 4 m** 算的，
        # 属于未来信息。实测含 wp 1.1991 vs 只过去特征 3.1518（差 2.6 倍），
        # 说明模型的预测能力几乎全来自这个泄漏。
        self.drop_wp = bool(drop_wp)
        self.path_in = bool(path_in)
        self.guidance_corrupt_p = float(guidance_corrupt_p) if training else 0.0
        self.guidance_jitter_m = float(guidance_jitter_m)
        self.guidance_stale_p = float(guidance_stale_p) if training else 0.0
        self.guidance_drop_p = float(guidance_drop_p) if training else 0.0
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.s)

    def __getitem__(self, i: int):
        s = self.s[i]
        cur = np.asarray(Image.open(s["img"]).convert("RGB"))
        prev = np.asarray(Image.open(s["prev_img"]).convert("RGB"))
        video = np.stack([prev, cur])  # (T=2, H, W, 3)，T 沿 temporal_patch_size 打成一个 patch
        body = s["body"]
        mode = int(s.get("mode", 0))
        topo = s.get("topo") or (0.0, 0.0, 0.0)
        guide = dict(s.get("guide") or {})
        guide.setdefault("requested", bool(s.get("path")))
        guide.setdefault("valid", bool(s.get("path")))
        guide.setdefault("confidence", 1.0 if guide.get("valid") else 0.0)
        guide.setdefault("pose_age_s", 0.0)
        guide.setdefault("sigma_m", 0.0)
        raw_path = s.get("path") or []
        path = [[float(p[0]), float(p[1])] for p in raw_path
                if isinstance(p, (list, tuple)) and len(p) >= 2]
        if (self.body_dropout > 0.0 or self.vision_dropout > 0.0 or self.topo_dropout > 0.0
                or self.guidance_corrupt_p > 0.0):
            # 每个 epoch 用不同随机流，否则同一张图永远被同一路 dropout 命中
            rng = np.random.default_rng((self.seed * 1000003 + self.epoch) * 7919 + i)
            if self.vision_dropout > 0.0 and rng.random() < self.vision_dropout:
                video = np.zeros_like(video)
            if self.body_dropout > 0.0 and rng.random() < self.body_dropout:
                body = (0.0, 0.0)
            # topo dropout：在线常常拿不到地点图（M0 完全没有），必须让「全 0」
            # 落在训练分布里，否则 M3 之前每次推理都是分布外。
            # 与方案 §10.6「缺失值一律写 0.0」、§11.4 geometry dropout 同一契约。
            if self.topo_dropout > 0.0 and rng.random() < self.topo_dropout:
                topo = (0.0, 0.0, 0.0)
                mode = 0
            if self.guidance_corrupt_p > 0.0 and guide.get("valid") and rng.random() < self.guidance_corrupt_p:
                r = rng.random()
                if r < self.guidance_drop_p:
                    guide.update(valid=False, confidence=0.0, reason="augmented_drop")
                    path = []
                elif r < self.guidance_drop_p + self.guidance_stale_p:
                    guide.update(valid=False, confidence=0.0, pose_age_s=2.0,
                                 reason="augmented_stale")
                    path = []
                else:
                    dx, dz = rng.normal(0.0, self.guidance_jitter_m, size=2)
                    guide.update(valid=True, confidence=min(float(guide.get("confidence", 1.0)), 0.25),
                                 reason="augmented_offset")
                    path = [[p[0] + float(dx), p[1] + float(dz)] for p in path]
                    if path:
                        guide["waypoint"] = [path[0][0], path[0][1]]
        # wp 进模型的唯一入口。legacy 分支漏赋值会让后面 204/225 行 NameError，
        # 且 legacy 生成的 text 会被下一行直接覆盖 —— 等于这个分支从来没成功跑过。
        wp_src = guide.get("waypoint") or s["wp"]
        _wp_in = (0.0, 0.0) if self.drop_wp or not guide.get("valid") else wp_src
        if self.prompt_format == "legacy":
            # 2026-11-02 之前的格式。只用于评估旧 checkpoint，不要用来训新模型。
            text = ("Body velocity vx=%.2f vz=%.2f m/s. Local goal dx=%.2f dz=%.2f m. "
                    "Predict the next 12 body velocities at 0.1 s steps."
                    % (body[0], body[1], _wp_in[0], _wp_in[1]))
        else:
            text = build_policy_text(mode, _wp_in, topo, body)
        if self.path_in:
            pts = " ".join("%.2f %.2f" % (float(a), float(b)) for a, b in path)
            text += " Path ahead (dx dz, right/fwd, every 0.8 m):" + pts
            text += build_guidance_text(guide)
        # 数值旁路用的 8 维（与记忆卡同源），P0-1 起直接送 head
        num_feat = [float(_wp_in[0]), float(_wp_in[1]),
                    float(topo[0]), float(topo[1]), float(topo[2]),
                    float(body[0]), float(body[1]), float(mode)]
        if self.quad_feats:
            # 二次扩张 8 -> 44（原值 + 平方 + 28 个交叉项），**直接拼进 head**。
            # 依据（P2 诊断 2026-11-04，同批 val 400、前 8 步）：
            #   8 数值线性 2.3916 → 二次 1.1834（−50.5%）
            #   pooled 1024 线性 2.2978；pooled+数值线性 2.2990（几乎无增益）
            # ⇒ 「怎么走」的信息几乎全在**交叉项**里，主干没保留它。
            _v = num_feat
            num_feat = _v + [x * x for x in _v]
            for _i in range(len(_v)):
                for _j in range(_i + 1, len(_v)):
                    num_feat.append(_v[_i] * _v[_j])
        # 必须走 chat template，否则输入里没有 <|video_pad|>，视觉特征对不上任何 token
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)
        return dict(video=video, text=text, act=s["act"], mask=s["mask"],
                    wp=np.asarray(_wp_in, dtype=np.float32),
                    num=np.asarray(num_feat, dtype=np.float32),
                    traversability=(None if s.get("traversability") is None else
                                    np.asarray(s["traversability"].get("traversability", []),
                                               dtype=np.float32)),
                    cls=s["cls"], split=s["split"], sid=s["sid"], t=s["t"])


def make_collate(processor, min_pixels: int, max_pixels: int):
    def collate(batch: list[dict]):
        videos = [b["video"] for b in batch]
        texts = [b["text"] for b in batch]
        inputs = processor(text=texts, videos=videos, return_tensors="pt",
                           padding=True, min_pixels=min_pixels, max_pixels=max_pixels,
                           cap_pixels_per_frame=False, do_sample_frames=False)
        _trav = [b.get("traversability") for b in batch]
        _trav_ok = all(x is not None and len(x) == 32 for x in _trav)
        return dict(
            inputs=inputs,
            act=torch.from_numpy(np.stack([b["act"] for b in batch])),
            mask=torch.from_numpy(np.stack([b["mask"] for b in batch])),
            wps=torch.from_numpy(np.stack([b["wp"] for b in batch])),
            nums=torch.from_numpy(np.stack([b["num"] for b in batch])),
            traversability=(torch.from_numpy(np.stack(_trav)).float() if _trav_ok else None),
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
    """``extra > 0`` 时接受「文本池化 + 视觉注入」拼接后的输入。

    ``use_hidden=False``（``--no-last-token``）：不吃 LM 末位 hidden，只吃读出模块
    产出的 extra —— 这是 4-query 读出的真正「绕过 LM」形态，序列长度/主干都不变。
    """

    def __init__(self, hidden: int, out: int = K_STEPS * 2, extra: int = 0,
                 use_hidden: bool = True):
        super().__init__()
        self.use_hidden = bool(use_hidden)
        self.in_dim = (int(hidden) if self.use_hidden else 0) + int(extra)
        self.fc = nn.Linear(self.in_dim, out)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor | None, *extra) -> torch.Tensor:
        """接受任意数量的额外拼接输入（视觉注入、数值旁路…），按顺序 cat 到末维。

        早先签名是 ``forward(x, vis=None)``；数值旁路加进来后需要拼第三块，
        所以改成变长。调用方 forward_batch 按 [vis, numeric] 顺序传。
        ``x`` 为 None 时（use_hidden=False）只用 extra；此时 extra 不能为空，
        否则在 0 维上 matmul —— 这里显式报错，不静默产出常数动作。
        """
        parts = ([x] if self.use_hidden else [])
        parts += [e for e in extra if e is not None]
        if not parts:
            raise RuntimeError("ActionHead 没有可用输入（use_hidden=False 且 extra 为空）")
        return self.fc(torch.cat(parts, dim=-1))


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
    spd = act[..., :2].norm(dim=-1)              # (B, K) —— 只看速度两维，别把 ω 算进速度
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


class TraversabilityHead(nn.Module):
    """从视觉 token 预测 8 扇区×4 距离环的局部可通行性。"""

    def __init__(self, visual_dim: int, out_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(int(visual_dim), int(out_dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        return self.fc(feat.float())


def load_model(model_dir: str, chunk: int, device: str, vit_lora_blocks: int = 0,
               vit_lora_r: int = 16, act_dim: int = 2, lora_r: int = 32,
               lora_alpha: int | None = None):
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

    # lora_alpha 默认保持 2×r（原 32/64 的比例），换 rank 时缩放不变。
    _alpha = int(lora_alpha) if lora_alpha else int(lora_r) * 2
    model = get_peft_model(model, LoraConfig(
        r=int(lora_r), lora_alpha=_alpha, lora_dropout=0.05, bias="none",
        target_modules=list(TARGET_MODULES) + vit_targets, task_type="CAUSAL_LM"))
    if vit_targets:
        n_vit = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and "visual" in n.lower())
        print("[vit-lora] %s, 可训练 %d" % (vit_desc, n_vit), flush=True)

    hidden = model.config.text_config.hidden_size
    head = ActionHead(hidden, out=K_STEPS * int(act_dim)).to(device)
    return model, head, hidden


def build_extra_heads(hidden: int, device, inject_visual: bool, visual_proj: int,
                      visual_grid, event_head: bool, event_weight: float,
                      readout=None):
    """按开关造注入器/事件头/读出模块，并返回 (injector, event_head, extra_dim, params)。

    ``readout``：S2 的读出模块（``policy_readout.build_readout`` 的产物），
    与 ``inject_visual`` 互斥（两者都吃视觉 token，同时开会拼两份视觉特征）。
    """
    if readout is not None and inject_visual:
        raise RuntimeError("--inject-visual 与 --vision-query/--fusion 互斥："
                           "两路都吃同一份视觉 token，同时开会把视觉拼两遍")
    injector = None
    extra = 0
    params = []
    if inject_visual:
        injector = VisualInjector(hidden, grid=tuple(visual_grid), proj=visual_proj).to(device)
        extra = injector.out_dim
        params += list(injector.parameters())
    if readout is not None:
        extra += int(readout.out_dim)
        params += list(readout.parameters())
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


class NumericBypass(nn.Module):
    """把记忆卡那 8 个数值**直接送进 head**，不经 24 层 LM。

    为什么需要（2026-11-03 实测）：一个只用 8 个数值、45 个参数的二次 ridge 做到
    mse8 **1.2392**，比 0.8B+LoRA+Linear 的 **1.7569** 好 29.5%。机制推断：
    精确标量要穿过 24 层（18 层 GatedDeltaNet 带遗忘门的递归状态）才能到末位 token，
    在长递归里会被稀释。旁路让这些数在 head 处保持原值。

    实现：标准化（用训练集统计）→ 小 MLP → 与 pooled 拼进 head。
    额外参数约 8*32 + 32*32 ≈ 1.3k，延迟增量可忽略。
    """

    def __init__(self, in_dim: int = 8, hidden: int = 64, out_dim: int = 32,
                 quad: bool = True):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.quad = bool(quad)
        # 二次扩张：d 维 → d + d + d(d-1)/2。8 维 → 8+8+28 = 44。
        # 为什么必须开（2026-11-03）：队友的 ridge 用的是二次特征（44 维，45 参数）做到
        # mse 1.2371；而只给 8 维线性旁路时模型只到 1.8609（比不加旁路的 1.8960 只好
        # 1.9%）。差别不在「旁路有没有用」，在**表示**：线性层表达不出
        # `wp × body` 这类交叉项，而 ridge 显式给了它们。
        self.feat_dim = in_dim * 2 + in_dim * (in_dim - 1) // 2 if self.quad else in_dim
        self.net = nn.Sequential(
            nn.Linear(self.feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim), nn.SiLU(),
        )
        self.out_dim = out_dim

    def _quad(self, x: torch.Tensor) -> torch.Tensor:
        """8 → 44：原值 + 平方 + 上三角交叉项（逐对相乘）。

        ⚠️ 不用 torch.triu_indices：它在这里报
        `RuntimeError: Kernel is incompatible with all devices in devs`（XPU 上没有该算子）。
        改为张量切片，纯 elementwise，任何设备都能跑。
        """
        if not self.quad:
            return x
        d = x.shape[-1]
        cross = []
        for i in range(d):
            for j in range(i + 1, d):
                cross.append(x[..., i:i + 1] * x[..., j:j + 1])
        if cross:
            return torch.cat([x, x * x] + cross, dim=-1)
        return torch.cat([x, x * x], dim=-1)

    def set_stats(self, mu, sd) -> None:
        self.mu.copy_(torch.as_tensor(mu, dtype=self.mu.dtype))
        self.sd.copy_(torch.as_tensor(sd, dtype=self.sd.dtype).clamp_min(1e-9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.to(self.mu.dtype).to(self.mu.device) - self.mu) / self.sd
        return self.net(self._quad(z))


def numeric_features(samples: list[dict]) -> list[list[float]]:
    """8 维数值：wp dx/dz、topo 3 项、body vx/vz、mode。与记忆卡同源。"""
    out = []
    for s in samples:
        wp = s.get("wp") or (0.0, 0.0)
        tp = s.get("topo") or (0.0, 0.0, 0.0)
        bd = s.get("body") or (0.0, 0.0)
        out.append([float(wp[0]), float(wp[1]), float(tp[0]), float(tp[1]),
                    float(tp[2]), float(bd[0]), float(bd[1]),
                    float(s.get("mode", 0))])
    return out


def forward_batch(model, head, inputs, device: str, injector=None, event_head=None,
                  traversability_head=None, bypass=None, numeric=None,
                  quad_feats_active=False, readout=None):
    """绕过 LM head，取 last_hidden_state 的末位 token 池化 —— 视需要再接视觉注入。

    ``injector`` 非 None 时：在同一次前向里用 forward hook 取视觉塔输出，
    绕开 LM 直接拼进动作头。**必须复用这一次前向的视觉输出**，不能另跑一次
    visual（p95 会从 164 ms 翻上去，G1 直接挂 —— policy-designer 的 P4）。

    ``readout`` 非 None 时：用同一份视觉 token 过 ``policy_readout`` 的读出模块
    （4-query 池化 / additive-gated fusion），结果进 head 的 extra 位。
    """
    base = model.base_model.model  # PeftModel -> LoraModel -> Qwen3_5ForConditionalGeneration
    inner = base.model             # Qwen3_5Model（视觉塔 + LM），LoRA 已注入
    kw = {}
    for k in ("input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
              "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
        v = inputs.get(k, None)
        if v is not None:
            kw[k] = v.to(device)

    need_vis = (injector is not None or traversability_head is not None
                or readout is not None)
    vis_cap = {}
    handle = None
    if need_vis:
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

    feat = _extract_visual_feat(vis_cap.get("feat"), h.shape[0])
    vis = None
    if injector is not None and feat is not None:
        injector._batch = h.shape[0]
        vis = injector(feat.float())

    # 自动跟随动作维度：2 通道（vx,vz）与 3 通道（vx,vz,ω）共用同一条路径。
    # ⚠️ 不能写 view(-1, K_STEPS, -1)：view 只允许一个待推断维（实测 RuntimeError:
    # only one dimension can be inferred）。所以先算最后一维再 reshape。
    # 数值旁路拼在 pooled 之后（P0-1，2026-11-03）
    cat = []
    if vis is not None:
        cat.append(vis)
    if readout is not None:
        if feat is None:
            raise RuntimeError("读出模块需要视觉 token，但视觉输出未捕获")
        _num = numeric.to(device) if numeric is not None else None
        cat.append(readout(feat.float(), _num))
    if bypass is not None and numeric is not None:
        # 注意：_num 不在 inputs（那是 processor 产出的）；调用方把 batch["nums"]
        # 以关键字 numeric= 传进来。
        cat.append(bypass(numeric.to(device)))
    elif quad_feats_active and numeric is not None:
        # quad_feats 模式：44 维二次特征**原样拼接**，不经 MLP。
        # 与 --numeric-bypass 的区别：后者是 8→32→32 的 MLP（1344~4960 参），
        # 在小数据上过拟合（实测 1.9984 比不加还差）；这里零额外参数，
        # head 只学一个线性映射，二次关系由输入显式提供。
        #
        # ⚠️ 必须由开关控制：batch 里**总是**有 nums，若无条件拼接，
        # 不开 quad 时 head 仍是 1024 维会报 mat1/mat2 (2x1032 and 1024x36)。
        cat.append(numeric.to(device))
    # use_hidden=False（--no-last-token）时 head 只吃 extra；ActionHead 对空输入报错。
    _x = pooled if getattr(head, "use_hidden", True) else None
    logits = head(_x, *cat) if cat else head(_x)
    act_dim = int(logits.shape[-1]) // K_STEPS
    pred = logits.view(-1, K_STEPS, act_dim)
    aux = []
    if event_head is not None:
        ev_in = torch.cat([pooled, vis], -1) if vis is not None else pooled
        aux.append(event_head(ev_in))
    if traversability_head is not None:
        if feat is None:
            raise RuntimeError("可通行性头需要视觉 token，但视觉输出未捕获")
        aux.append(traversability_head(feat.float()))
    return (pred, *aux) if aux else pred


def masked_mse(pred: torch.Tensor, act: torch.Tensor, mask: torch.Tensor,
               steps: int | None = None, chan_w: torch.Tensor | None = None) -> torch.Tensor:
    """按 mask 加权的 MSE；``chan_w`` 可给每个动作通道单独权重。

    为什么需要 chan_w（2026-11-03）：动作从 2 维扩到 3 维（vx, vz, ω）之后，
    ω 在 rad/s 下是**双峰**分布（50% 恰好 0、转弯段 ±2），比速度难回归得多。
    等权时它会把梯度主导走，实测训练在 step 60-70 出现 12.9/14.4 的尖峰、
    step 1250 直接 NaN。给 ω 一个独立权重是让三个通道能各自收敛。
    """
    if steps is not None:
        act, mask = act[:, :steps], mask[:, :steps]
    sq = (pred[:, :act.shape[1]] - act) ** 2
    if chan_w is not None:
        w = chan_w.to(sq.device).view(1, 1, -1)
        sq = sq * w
    err = sq.sum(-1)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def _metric_value(metrics: dict, name: str) -> float:
    """从 evaluate() 的嵌套 dict 取 ``"class.key"`` 指标；缺失返回 +inf（永不选它）。"""
    if "." in str(name):
        c, k = str(name).split(".", 1)
        return float((metrics.get(c) or {}).get(k, math.inf))
    return float(metrics.get(str(name), math.inf))


# ---------------------------------------------------------------- 验证
@torch.no_grad()
def evaluate(model, head, loader, device: str, injector=None, event_head=None,
             traversability_head=None, bypass=None, quad_active=False,
             readout=None) -> dict:
    model.eval()
    preds, acts, masks, clss = [], [], [], []
    ev_logits, ev_true = [], []
    trav_logits, trav_true = [], []
    for batch in loader:
        r = forward_batch(model, head, batch["inputs"], device,
                          injector=injector, event_head=event_head,
                          traversability_head=traversability_head,
                          bypass=bypass, numeric=batch.get("nums"),
                          quad_feats_active=quad_active, readout=readout)
        if isinstance(r, tuple):
            pred = r[0]
            pos = 1
            if event_head is not None:
                ev = r[pos]; pos += 1
                ev_logits.append(ev.float().cpu())
                ev_true.append(event_labels(batch["act"], batch["mask"]))
            if traversability_head is not None and batch.get("traversability") is not None:
                trav_logits.append(r[pos].float().cpu())
                trav_true.append(batch["traversability"].float().cpu())
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
    if trav_logits:
        tl = torch.cat(trav_logits)
        ty = torch.cat(trav_true)
        known = ty >= 0.0
        if bool(known.any()):
            out["traversability_bce"] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                tl[known], ty[known]))
    for c in ("move", "trans", "idle"):
        sel = cls == c
        if not sel.any():
            continue
        p, a, m = pred[sel], act[sel], mask[sel]
        # ⚠️ mse8 / zero8 / copy8 一律只取**速度两维**（p[..., :2]），
        # 这样加入 ω 之后，这些数与历史所有 run 仍然可比；ω 单列 omega8。
        pv, av = p[..., :2], a[..., :2]
        # 2026-11-06：**12 步全块口径**（只统计 12 步全部有效的样本）。
        # 为什么不复用 mse_all：mse_all 是「所有有效步」按步平均，短块（靠近录制
        # 结尾的残块）会占权重，与「整块一次执行」的部署口径不一致。
        full = m.sum(-1) >= K_STEPS
        mf = full.unsqueeze(-1).expand_as(m).to(m.dtype)
        out[c] = dict(
            n=int(sel.sum()),
            n_full=int(full.sum()),
            mse8=float(masked_mse(pv, av, m, 8)),
            mse_all=float(masked_mse(p, a, m)),
            zero8=float(masked_mse(torch.zeros_like(av), av, m, 8)),
            copy8=float(masked_mse(av[:, :1].repeat(1, K_STEPS, 1), av, m, 8)),
        )
        if bool(full.any()):
            out[c]["mse12"] = float(masked_mse(p, a, mf))
            out[c]["mse12_vel"] = float(masked_mse(pv, av, mf))
            # 轨迹偏差（开环）：把 12 步机体系速度积分成局部折线，比位置而不是速度。
            # 单位 = 世界米（STEP_S 秒一步）；body 系折线，所以是「这一块会走偏多少」。
            ppos = torch.cumsum(pv, dim=1) * STEP_S
            apos = torch.cumsum(av, dim=1) * STEP_S
            dev = (ppos - apos).norm(dim=-1)
            out[c]["traj_dev_m"] = float((dev * mf).sum() / mf.sum().clamp_min(1.0))
            out[c]["traj_dev_max_m"] = float(dev.max(dim=1).values[full].mean())
        if a.shape[-1] > 2:
            out[c]["omega8"] = float(masked_mse(p[..., 2:], a[..., 2:], m, 8))
            out[c]["omega_abs_mean"] = float(
                (p[..., 2:].abs().squeeze(-1) * m).sum() / m.sum().clamp_min(1.0))
        if c == "move":
            v0 = p[:, 0, :2].norm(dim=-1)
            ok = m[:, 0] > 0
            out[c]["walk_recall"] = float(((v0 > V_STILL) & ok).sum() / ok.sum().clamp_min(1))
        if c == "idle":
            first8 = p[:, :8, :2].norm(dim=-1)
            ok = m[:, :8].sum(-1) == 8
            out[c]["stop_recall"] = float(((first8 < V_STILL).all(-1) & ok).sum() / ok.sum().clamp_min(1))
        if c == "trans":
            first8 = p[:, :8, :2].norm(dim=-1)
            ok = m[:, :8].sum(-1) > 0
            out[c]["start_recall"] = float((((first8 > V_STILL) & (m[:, :8] > 0)).any(-1) & ok).sum() /
                                           ok.sum().clamp_min(1))

    # 2026-11-06：`stop_recall` 升为**顶层**字段，并显式标注它在哪个停止源下测。
    # 它测的是「模型自己的动作头在 idle 类窗口上前 8 步会不会全停」，属于
    # **动作头**来源；绝不能与执行器的 nav_state/stop_head/safety_gate 门控数字
    # 混用或相加（两者口径不同，一个基于标签窗口、一个基于在线否决计数）。
    if "idle" in out:
        out["stop_recall"] = out["idle"].get("stop_recall")
        out["stop_recall_source"] = "action_head_first8_idle_windows"
        out["stop_recall_definition"] = (
            "idle 类样本中，预测块前 8 步速度全部 < %.2f m/s 的比例（按块平均，"
            "只统计前 8 步标签全有效者）" % V_STILL)
    # 近碰/碰撞：collision.jsonl 尚不存在（见 Docs 缺口登记），
    # 这里如实报 unavailable，**不**用动作误差冒充碰撞率。
    out["near_collision"] = {"available": False, "count": None,
                             "source": "unavailable:collision.jsonl_not_produced"}
    out["protocol"] = {
        "k_steps": K_STEPS, "step_s": STEP_S,
        "mse": "move/trans/idle 三块各自报 mse8(前8步·速度两维)、mse12(12 步全块·全通道)",
        "selection": "见 run 根目录 summary.json 的 selection 字段",
    }

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
    # 默认 1e-4（2026-11-03 实测）：lr_head=1e-3 时 move8=1.8295，1e-4 时 1.7171（−6.1%）。
    # 注意：过拟合**不是**它造成的 —— 两个学习率都是 best_epoch=0、ep1 退化。
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--omega-loss-weight", type=float, default=1.0,
                    help="ω 通道的额外权重（1.0=等权）。真正解决尺度失衡靠 --omega-norm")
    ap.add_argument("--omega-norm", action="store_true", default=True,
                    help="按实测 std 归一化 ω 通道（默认开）。ω 是 rad/s 双峰分布，"
                         "等权会让它主导总损失")
    ap.add_argument("--no-omega-norm", dest="omega_norm", action="store_false")
    ap.add_argument("--omega-clip", type=float, default=9.17,
                    help="ω 标签硬裁剪（rad/s）。满量程 ±1024 unit ≈ 9.17 rad/s，"
                         "超过这个才是坏帧（2026-11-03 实测：之前设 3.0 砍掉 6.8%% 真实数据）")
    ap.add_argument("--chunk", type=int, default=128, help="GatedDeltaNet chunk_size；0=默认 64")
    ap.add_argument("--max-steps", type=int, default=0, help=">0 只跑这么多步（冒烟）")
    ap.add_argument("--max-val", type=int, default=0, help=">0 只评这么多验证样本（冒烟）")
    # 2026-11-06：选模口径。此前写死 `move.mse8`（前 8 步、只速度两维），
    # 与部署口径（整块 12 步、含 ω）不一致，也完全不看停止召回。
    ap.add_argument("--best-metric", default="move.mse12",
                    choices=["move.mse8", "move.mse12", "move.mse12_vel",
                             "move.mse_all", "idle.mse12", "trans.mse12"],
                    help="选模指标（越小越好）。默认 move.mse12 = 12 步全块口径")
    ap.add_argument("--min-stop-recall", type=float, default=0.0,
                    help="硬门：验证集 stop_recall 低于此值的 epoch 不参与选模（0=关闭）。"
                         "stop_recall 来源=动作头前 8 步（见 evaluate.stop_recall_source）")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # ---- 视觉承重相关的配方（2026-11-02 方差分解之后新增，默认全关=旧行为）----
    ap.add_argument("--vit-lora-blocks", type=int, default=0,
                    help="视觉塔末尾 N 个 block 加 LoRA；0=全冻结（旧行为）")
    ap.add_argument("--vit-lora-r", type=int, default=16)
    ap.add_argument("--freeze-lora", action="store_true",
                    help="训练时把 LoRA 也冻结，只训动作头（纯 head 对照）。"
                         "依据：78 分钟数据上 rank 越低越好（r8 1.4823 < r16 1.5587 < r64 1.6171）")
    ap.add_argument("--lora-r", type=int, default=32,
                    help="LM 侧 LoRA rank（默认 32）。数据只有 78 分钟，rank 越大越容易过拟合")
    ap.add_argument("--vision-dropout", type=float, default=0.08,
                    help="两帧置黑的概率（训练期）。默认 0.08 而不是 0：不造「无视觉」"
                         "样本的话，评测时的 black 输入是分布外，会得出「黑比真图更准」"
                         "的假结论（2026-11-03 实测踩到）。")
    ap.add_argument("--body-dropout", type=float, default=0.0,
                    help="body 置 0 的概率（训练期）。已被证伪（+4.6~5.7%%），默认关")
    ap.add_argument("--inject-visual", action="store_true",
                    help="把视觉塔输出绕开 LM 直接接进动作头（2x2 分块池化）")
    ap.add_argument("--visual-proj", type=int, default=256)
    ap.add_argument("--visual-grid", nargs=2, type=int, default=[8, 14])
    ap.add_argument("--event-head", action="store_true",
                    help="挂事件头（未来 1.2 s 是否 go/stop），给视觉加压")
    ap.add_argument("--event-weight", type=float, default=0.3)
    ap.add_argument("--traversability-head", action="store_true",
                    help="用 traversability.jsonl 训练 8 扇区×4 距离环可通行性辅助头")
    ap.add_argument("--traversability-weight", type=float, default=0.2)
    ap.add_argument("--prompt-format", default="card", choices=["card", "legacy"],
                    help="card=记忆卡(2026-11-02 起)，legacy=旧 Body velocity 格式")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile 内层模型。CPU 派发是训练瓶颈（见 CPU派发开销剖析 §2）")
    ap.add_argument("--drop-wp", action="store_true",
                    help="输入里去掉 wp（它来自未来轨迹，是泄漏）。"
                         "实测：含 wp 1.1991 vs 只过去特征 3.1518，差 2.6 倍")
    ap.add_argument("--path-in", action="store_true",
                    help="消费录制器保存的 A* 路径形状作为软引导；样本必须含 path 字段，"
                         "且必须配合引导损坏增强训练")
    ap.add_argument("--guidance-corrupt-p", type=float, default=0.30,
                    help="训练时损坏有效 A* 引导的概率（默认 0.30）")
    ap.add_argument("--guidance-jitter-m", type=float, default=0.75,
                    help="A* 引导横向/航向偏移增强的米数标准差")
    ap.add_argument("--guidance-stale-p", type=float, default=0.15,
                    help="引导损坏中模拟过期的比例")
    ap.add_argument("--guidance-drop-p", type=float, default=0.15,
                    help="引导损坏中模拟完全丢失的比例")
    ap.add_argument("--quad-feats", action="store_true",
                    help="把 8 个数值的二次扩张(44维)直接拼进 head 输入。零额外参数。"
                         "P2 诊断：8数值线性 2.39 → 二次 1.18")
    ap.add_argument("--numeric-bypass", action="store_true",
                    help="8 个数值直接接进 head（不经 LM）。P0-1 实验，硬门 mse8<=1.24")
    ap.add_argument("--bypass-no-quad", dest="bypass_quad", action="store_false",
                    default=True,
                    help="关闭二次特征扩张（默认开：8→44，与 ridge 的表示对齐）")
    # ---- S2 读出（2026-11-06）：默认全关 = 旧行为。本轮只实现，不启动训练。----
    ap.add_argument("--readout-only", action="store_true",
                    help="冻结主干（含 LoRA）只训读出 + 动作头；要求 --vision-query>0 "
                         "或 --fusion 非 none，否则没有可训的读出模块")
    ap.add_argument("--vision-query", type=int, default=0,
                    help="N 个可学习 query 对 112 个视觉 token 做注意力池化并绕过 LM；"
                         "0=关。N=4 时约 263k 参数，不改序列长度")
    ap.add_argument("--vision-query-dim", type=int, default=256,
                    help="query/投影维度（默认 256：1024→256 正好 262k 参数）")
    ap.add_argument("--fusion", default="none", choices=["none", "additive", "gated"],
                    help="视觉 + 数值 各自小编码器后相加；gated 用学到的 softmax 门控。"
                         "任务卡/A* 槽位当前无数据，留空")
    ap.add_argument("--fusion-out", type=int, default=256,
                    help="fusion 融合维度（默认 256）")
    ap.add_argument("--no-last-token", action="store_true",
                    help="动作头不吃 LM 末位 hidden，只吃读出模块输出；"
                         "必须与 --vision-query/--fusion 同时用")
    ap.add_argument("--symint-fallback", type=int, default=1024,
                    help="unbacked SymInt 的 fallback 值（默认 1024，Inductor 原值 8192）。"
                         "batch>4 时会按 8192 三次方算出 2^43，触发 maxGridSize 断言")
    ap.add_argument("--no-capture-scalars", dest="capture_scalars",
                    action="store_false", default=True,
                    help="关闭 dynamo 的 .item() 捕获（默认开；关掉会多 5 处分图）")
    ap.add_argument("--no-specialize-float", dest="specialize_float",
                    action="store_false", default=True,
                    help="关闭 dynamo float 特化（默认开；关闭会回到「Python float 当 "
                         "kernel 标量参数」的旧图，A770 上会撞 fp64 构建失败）")
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
    if a.path_in:
        _path_n = sum(1 for s in samples if isinstance(s.get("path"), list) and s.get("path"))
        if _path_n == 0:
            raise RuntimeError("--path-in 要求数据集含非空 path 字段；请先用 --astar-guidance 重建数据集")
        print("[guidance] path samples=%d/%d" % (_path_n, len(samples)), flush=True)
    if a.traversability_head:
        _trav_n = sum(1 for s in samples
                      if isinstance(s.get("traversability"), dict)
                      and len(s["traversability"].get("traversability", [])) == 32)
        if _trav_n != len(samples):
            raise RuntimeError("--traversability-head 要求每个样本都有 32 维 traversability 标签（%d/%d）"
                               % (_trav_n, len(samples)))
        print("[traversability] labelled samples=%d/%d" % (_trav_n, len(samples)), flush=True)
    # 动作维度从数据里读，不再写死 2：2026-11-03 起支持 (vx, vz, ω) 三通道
    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    if act_dim not in (2, 3):
        print("[data] 动作维度异常：%d（期望 2 或 3）" % act_dim)
        return 2
    print("[data] act_dim=%d" % act_dim, flush=True)
    # ω 通道的实测标准差：归一化与评估都用它，避免两处各拍一个数。
    omega_std = 0.0
    if act_dim > 2:
        _w = np.concatenate([np.asarray(s["act"])[:, 2] for s in train])
        _w = _w[np.abs(_w) <= a.omega_clip]
        omega_std = float(np.std(_w)) if _w.size else 0.0
        print("[data] omega rad/s: std=%.3f  p01=%.2f p50=%.2f p99=%.2f  clip=%s"
              % (omega_std, float(np.percentile(_w, 1)), float(np.percentile(_w, 50)),
                 float(np.percentile(_w, 99)), a.omega_clip), flush=True)

    ds_train = PolicyDataset(train, processor, min_px, max_px, training=True,
                             vision_dropout=a.vision_dropout,
                             body_dropout=a.body_dropout, seed=a.seed,
                             quad_feats=a.quad_feats, drop_wp=a.drop_wp,
                             path_in=a.path_in,
                             guidance_corrupt_p=a.guidance_corrupt_p,
                             guidance_jitter_m=a.guidance_jitter_m,
                             guidance_stale_p=a.guidance_stale_p,
                             guidance_drop_p=a.guidance_drop_p)
    ds_val = PolicyDataset(val, processor, min_px, max_px,
                           quad_feats=a.quad_feats,
                           drop_wp=a.drop_wp,
                           path_in=a.path_in)  # 验证集永不做 dropout 但要同样的特征
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
                                     vit_lora_r=a.vit_lora_r, act_dim=act_dim,
                                     lora_r=a.lora_r)
    # ---- S2 读出（2026-11-06）：默认全关，与旧路径逐位一致 ----------------
    if int(getattr(a, "vision_query", 0) or 0) > 0 and \
            str(getattr(a, "fusion", "none")) not in ("none", ""):
        raise SystemExit("--vision-query 与 --fusion 是两版对照，只能开一个")
    readout_kind = ("vision_query" if int(getattr(a, "vision_query", 0) or 0) > 0
                    else str(getattr(a, "fusion", "none") or "none"))
    readout = policy_readout.build_readout(
        readout_kind, q_dim=int(getattr(a, "vision_query_dim", 256) or 256),
        n_query=int(getattr(a, "vision_query", 4) or 4),
        out_dim=int(getattr(a, "fusion_out", 256) or 256))
    if getattr(a, "readout_only", False):
        if readout is None:
            raise SystemExit("--readout-only 需要 --vision-query>0 或 --fusion 非 none")
        a.freeze_lora = True      # 主干（含 LoRA）全冻，只训读出 + 动作头
    if getattr(a, "no_last_token", False) and readout is None:
        raise SystemExit("--no-last-token 必须配 --vision-query 或 --fusion，"
                         "否则动作头没有任何输入")
    injector, ev_head, extra_dim, extra_params = build_extra_heads(
        hidden, device, a.inject_visual, a.visual_proj, a.visual_grid,
        a.event_head, a.event_weight, readout=readout)
    trav_head = TraversabilityHead(hidden, 32).to(device) if a.traversability_head else None
    if trav_head is not None:
        extra_params = list(extra_params) + list(trav_head.parameters())

    # 数值旁路（P0-1，2026-11-03）：8 个数字直接进 head，不经 24 层。
    # 依据：45 参数二次 ridge 只用这 8 个数就到 1.2392，而 0.8B+LoRA 只有 1.7569。
    bypass = None
    if a.quad_feats:
        extra_dim += 44
        print("[quad] 数值二次扩张 8 -> 44 维直接拼进 head（无额外参数）", flush=True)
    if a.numeric_bypass:
        bypass = NumericBypass(quad=getattr(a, "bypass_quad", True)).to(device)
        _m = np.asarray(numeric_features(train), dtype=np.float64)
        bypass.set_stats(_m.mean(0), _m.std(0))
        extra_dim += bypass.out_dim
        extra_params = list(extra_params) + list(bypass.parameters())
        print("[bypass] numeric 8 -> %d, params=%d"
              % (bypass.out_dim, sum(p.numel() for p in bypass.parameters())), flush=True)

    if a.inject_visual or a.numeric_bypass or a.quad_feats or readout is not None:
        # 额外输入维度变了，head 必须重建（旧 head.pt 形状对不上）
        # --no-last-token：head 不吃 LM 末位 hidden（4-query 读出=真正绕 LM）
        head = ActionHead(hidden, out=K_STEPS * act_dim, extra=extra_dim,
                          use_hidden=not bool(getattr(a, "no_last_token", False))
                          ).to(device)

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
        # ⚠️ 只设环境变量**不管用**：torch._inductor.config 在 import 时就把
        # cache_dir 解析完了，之后再 setdefault 为时已晚。实测后果是编译产物
        # 一直写进 %TEMP%\torchinductor_<user>（涨到 400 MB / 9.3k 文件），
        # .tc_cache/inductor 纹丝不动（185.2 -> 185.2 MB），于是**每一轮训练都从零编译**，
        # 白烧 20–30 分钟。正确做法是直接改 config 属性。
        # 用 setdefault：外部想指定独立缓存目录（做冷编译 A/B）时不要被覆盖
        _os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache))

        # Triton autotune 是**串行**的：一个 Python 线程逐个配置在 GPU 上试跑，
        # 所以 CPU 只有 1 核多、GPU 只有 18%，两边看着都「空闲」。实测 30 min 走不完，
        # 而训练本身只跑几十步。关掉 autotune 换编译时间。
        try:
            import torch._inductor.config as _ic
            _ic.max_autotune = False
            _ic.max_autotune_gemm = False
            _ic.coordinate_descent_tuning = False
            # ⚠️ 本机 compile_threads 实测是 **1**（不是文档写的 min(32, cpu_count)），
            # 所以 Triton 内核编译完全串行：实测 8s 窗口只用 1.66 核 / 20 核，
            # 且进程树恒为 1（没有任何编译子进程）。显式调大才有并行。
            # 注意：Inductor 前端（FX trace → scheduler → codegen）本身仍是单线程，
            # 能并行的只有「内核编译」这一段。
            # ⚠️ **Windows 并行编译：env 与 worker_start 必须成对设**（2026-11-03 实测钉死）。
            #
            # 归因链有三层，缺一层就会误判：
            #   1. config.py:1465 的 TORCHINDUCTOR_COMPILE_THREADS 分支在 :1467 的
            #      `elif sys.platform == "win32": compile_threads = 1` **之前**，
            #      所以 env 确实能覆盖钳位 —— 实测 compile_threads = 8 生效。
            #   2. 但 worker_start_method 默认 "subprocess" → async_compile.py:329 走
            #      SubprocPool → subproc_pool.py:221 的 Popen(pass_fds=...) → Windows 抛
            #      "AssertionError: pass_fds not supported on Windows"。
            #   3. 设 TORCHINDUCTOR_WORKER_START=spawn 改走 TrackedProcessPoolExecutor
            #      （async_compile.py:334-344），完全不碰 pass_fds —— 实测 pool OK。
            #
            # 注意：async_compile.py 里搜不到 pass_fds，别据此以为限制没了；
            # 它藏在子模块 subproc_pool.py 与 autotune_process.py 里（各一处）。
            #
            # 但这不是死路（2026-11-03 实测）：async_compile.py:329 只有
            # worker_start_method == "subprocess" 才走 SubprocPool；设
            # TORCHINDUCTOR_WORKER_START=spawn 就走 TrackedProcessPoolExecutor(
            # mp_context=spawn)，完全不碰 pass_fds，实测能跑通。
            # 另外 config.py:1464 的 env 检查在 win32 判断**之前**，所以
            # TORCHINDUCTOR_COMPILE_THREADS 在 Windows 上是能覆盖的。
            _want = int(_os.environ.get("TORCH_INDUCTOR_COMPILE_THREADS") or 0)
            if _want <= 0:
                _want = min(16, _os.cpu_count() or 1)
            if _os.name == "nt" and _want > 1:
                _start = (_os.environ.get("TORCHINDUCTOR_WORKER_START") or "").lower()
                if _start == "spawn":
                    # spawn 每个 worker 都要重新 import torch（约 500 MB RSS），
                    # 16 个就 8 GB，容易把内存打满反而更慢。封顶 8。
                    if _want > 8:
                        _want = 8
                    print("[compile] Windows + WORKER_START=spawn：走 spawn 进程池，"
                          "compile_threads=%d 可用" % _want, flush=True)
                else:
                    print("[compile] Windows 默认走 SubprocPool（pass_fds），并行不可用；"
                          "compile_threads 固定 1。要并行请设 "
                          "TORCHINDUCTOR_WORKER_START=spawn", flush=True)
                    _want = 1
            _ic.compile_threads = _want
            # ⚠️ 不要写 _ic.cache_dir = ... —— 该属性不存在（实测 AttributeError），
            # 缓存位置只能靠环境变量 TORCHINDUCTOR_CACHE_DIR，且必须在 import torch 前设。
            # 本模块顶层就 import torch，所以这里设已经晚了；但保持设置无害，
            # 真正生效的是外部环境变量（见 README / 启动脚本）。
            print("[compile] autotune 关闭, compile_threads=%d, cache_env=%s"
                  % (_want, _os.environ.get("TORCHINDUCTOR_CACHE_DIR", "(未设)")), flush=True)
        except Exception as _e:  # noqa: BLE001
            print("[compile] 配置失败：%r" % (_e,), flush=True)
        # 2026-11-03：本会话实测「compile 编不出 decode 图」的真因是 fp64 标量参数
        # （Python float 被当运行期标量塞进 kernel，而 A770 has_fp64=False，IGC 直接
        # 报 Double type is not supported）。specialize_float 把 Python float 折成
        # 常量，decode 图随即编译成功：算子 6258 → 1824、wall 338 → 132 ms。
        # 训练侧同样有这个标量来源（rms_norm_eps 等），所以默认打开。
        # 关掉用 --no-specialize-float（会回到「标量当参数」的旧图）。
        try:
            # 2026-11-03 实测：这个开关把断图数从 28 降到 21（−25%）。
            # 原因见 probe_graph_breaks.py 的输出：有 5 处断在
            # 「Unsupported Tensor.item() call with capture_scalar_outputs=False」，
            # 打开后 .item() 被捕获成图内算子，那 5 处不再断。
            # 副作用：算子总数 3485 -> 3627（.item 本身也成了算子），但断图更少、
            # 图更大，Inductor 能融的范围更多。
            if getattr(a, "capture_scalars", True):
                torch._dynamo.config.capture_scalar_outputs = True
                print("[compile] capture_scalar_outputs=True", flush=True)
        except Exception as _e:  # noqa: BLE001
            print("[compile] capture_scalar_outputs 设置失败：%r" % (_e,), flush=True)
        try:
            if getattr(a, "specialize_float", True):
                torch._dynamo.config.specialize_float = True
                print("[compile] specialize_float=True（Python float 折成常量）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print("[compile] specialize_float 设置失败：%r" % (_e,), flush=True)
        # 2026-11-04：batch>4 报「Reduction config exceeds cudaDeviceProp maxGridSize」。
        # 根因（driver-analyst, task-15）**不是 batch 太大**：
        #   capture_scalar_outputs=True 让 grid_thw[0].item() 不再 graph break，
        #   被包成 unbacked SymInt，流进 get_vision_position_ids 的 meshgrid；
        #   Inductor 对 unbacked 符号取 _size_hinting.py:438 的 fallback=8192，
        #   三次方后 size_hints['x'] = 2^43，越过 _check_max_grid_x 的 2^31 断言。
        # 已证：纯整数 triton_config({'x': 2**43}) 直接断言；batch 1/2/4 → 2^40/41/42
        # 通过，6/8 → 2^43 挂，精确复现边界。
        # 修法：降 fallback，让虚拟尺寸不失控（1024 已验：最大 2^40，2^43 零出现）。
        try:
            _fb = int(getattr(a, "symint_fallback", 1024))
            if _fb > 0:
                import torch._inductor.config as _ic2
                _ic2.unbacked_symint_fallback = _fb
                print("[compile] unbacked_symint_fallback=%d" % _fb, flush=True)
        except Exception as _e:  # noqa: BLE001
            print("[compile] symint fallback 设置失败：%r" % (_e,), flush=True)
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
    # 纯 head 对照（2026-11-04）：把 LoRA 也冻上，只训动作头。
    # 依据：rank 扫描显示参数越少越好（r8 1.4823 < r16 1.5587 < r64 1.6171），
    # 而 45 参数 ridge 能到 1.2371 —— 需要知道「主干特征本身够不够」。
    if getattr(a, "freeze_lora", False):
        n_frozen = 0
        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n_frozen += 1
        print("[freeze] LoRA 全部冻结：%d 个张量，只训 head" % n_frozen, flush=True)
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

    # ⚠️ 空参数组会让 AdamW 报 ValueError（--freeze-lora 时 trainable 为空），
    # 所以按需过滤。
    groups = []
    if trainable:
        groups.append({"params": trainable, "lr": a.lr_lora,
                       "weight_decay": a.weight_decay})
    _hparams = list(head.parameters()) + extra_params
    if _hparams:
        groups.append({"params": _hparams, "lr": a.lr_head, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)

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
        if readout is not None:
            torch.save(readout.state_dict(), d / "readout.pt")
        if ev_head is not None:
            torch.save(ev_head.state_dict(), d / "event_head.pt")
        if trav_head is not None:
            torch.save(trav_head.state_dict(), d / "traversability_head.pt")
        (d / "meta.json").write_text(json.dumps(
            {"hidden": hidden, "steps": K_STEPS, "act_dim": act_dim,
         "model": a.model, "tag": tag,
             "prompt_format": a.prompt_format}, indent=1), encoding="utf-8")
        # ⚠️ 输入口径 manifest：推理侧靠它重建**同一张**输入卡。
        # 只写 args.json 不够 —— args.json 在 run 根目录，checkpoint 目录不带，
        # 且推理侧不知道该信哪个默认值。缺了它就会静默喂错输入（已实测两次）。
        _schema = input_schema.from_args(a)
        _schema["act_dim"] = int(act_dim)
        # 读出字段以**实际模块**为准（readout_dim 不能靠 args 猜）
        _schema.update(policy_readout.schema_fields(
            readout, no_last_token=bool(getattr(a, "no_last_token", False))))
        _schema["head_in_dim"] = int(getattr(head, "in_dim", hidden))
        input_schema.save(d, _schema)

    best = (math.inf, None)
    best_val = {}
    last_val = {}
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
                              injector=injector, event_head=ev_head,
                              traversability_head=trav_head,
                              bypass=bypass, numeric=batch.get("nums"),
                              quad_feats_active=a.quad_feats, readout=readout)
            if isinstance(r, tuple):
                pred = r[0]
                _pos = 1
                ev_logit = r[_pos] if ev_head is not None else None
                if ev_head is not None:
                    _pos += 1
                trav_logit = r[_pos] if trav_head is not None else None
            else:
                pred, ev_logit, trav_logit = r, None, None
            act_d = batch["act"].to(device)
            mask_d = batch["mask"].to(device)
            # ω 通道：先硬裁剪掉异常值，再按 --omega-loss-weight 单独加权。
            # 不裁剪的话，录制里有少量 AngularY 尖峰（换武器/瞬移）会在 rad/s 下变成
            # 几十的量级，一个 batch 里出现两三个就能把这一步的梯度带飞。
            chan_w = None
            if act_dim > 2:
                # ω：先按满量程裁剪（超过 ±9.17 rad/s 才是坏帧），再做**损失端归一化**。
                #
                # 为什么要归一化而不是「调权重」：ω 在 rad/s 下是双峰（50% 恰好 0、
                # 转弯段 ±2），且尾部到 ±9。等权 MSE 下它的贡献远大于速度，实测训练
                # 在 step 60-70 出现 12.9/14.4 尖峰、step 1250 NaN。手调权重等于拍脑袋；
                # 按实测 std 把三通道拉到同量级才是可解释的做法。
                #
                # 注意：pred 与 act 同时缩放，所以 mse 指标那一侧要还原（见 evaluate）。
                act_d = act_d.clone()
                act_d[..., 2] = act_d[..., 2].clamp(-a.omega_clip, a.omega_clip)
                if getattr(a, "omega_norm", True) and omega_std > 0:
                    wv = [1.0] * (act_dim - 1) + [1.0 / (omega_std ** 2)]
                    chan_w = torch.tensor(wv, dtype=act_d.dtype)
                elif a.omega_loss_weight != 1.0:
                    wv = [1.0] * (act_dim - 1) + [float(a.omega_loss_weight)]
                    chan_w = torch.tensor(wv, dtype=act_d.dtype)
            loss = masked_mse(pred, act_d, mask_d, chan_w=chan_w)
            if ev_logit is not None:
                # 事件头 loss：给视觉加压（body 预测不了 go/stop）。只对 mask 有效的块算。
                ev_y = event_labels(act_d, mask_d)
                valid_blk = mask_d[:, :8].sum(-1) > 0
                if bool(valid_blk.any()):
                    l_ev = torch.nn.functional.binary_cross_entropy_with_logits(
                        ev_logit[valid_blk], ev_y[valid_blk])
                    loss = loss + a.event_weight * l_ev
            if trav_logit is not None and batch.get("traversability") is not None:
                trav_y = batch["traversability"].to(device)
                known = trav_y >= 0.0
                if bool(known.any()):
                    l_trav = torch.nn.functional.binary_cross_entropy_with_logits(
                        trav_logit[known], trav_y[known])
                    loss = loss + a.traversability_weight * l_trav
            loss.backward()
            _clip = trainable + list(head.parameters()) + extra_params
            if _clip:
                torch.nn.utils.clip_grad_norm_(_clip, a.grad_clip)
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
                               injector=injector, event_head=ev_head,
                               traversability_head=trav_head, bypass=bypass,
                               quad_active=a.quad_feats, readout=readout)
        val_metrics["ep"] = epoch
        print(f"[val] ep={epoch} {json.dumps(val_metrics, ensure_ascii=False)}", flush=True)
        log_f.write(json.dumps(val_metrics, ensure_ascii=False) + "\n")
        log_f.flush()
        save(f"epoch_{epoch:02d}")
        score = _metric_value(val_metrics, a.best_metric)
        val_metrics["selection_score"] = score
        last_val = val_metrics
        # stop_recall 硬门：只影响「能不能当 best」，不影响训练本身。
        sr = val_metrics.get("stop_recall")
        eligible = (a.min_stop_recall <= 0.0) or (
            sr is not None and float(sr) >= float(a.min_stop_recall))
        if eligible and score < best[0]:
            best = (score, epoch)
            best_val = val_metrics
            save("best")
        print(f"[epoch] ep={epoch} {time.perf_counter() - t_ep:.1f}s "
              f"{a.best_metric}={score:.4f} stop_recall={sr} eligible={eligible} "
              f"best={best}", flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    log_f.close()
    # 2026-11-06：summary.json 从 4 个字段扩成「选模口径 + 三块全指标 + 停止召回
    # 来源 + 近碰可用性」，让跨会话对比不必再翻 train_log.jsonl。
    summary = {
        "best_metric": a.best_metric,
        "best_score": best[0],
        "best_epoch": best[1],
        "steps": step,
        "peak_gib": round(torch.xpu.max_memory_allocated() / 2 ** 30, 2)
        if device == "xpu" else None,
        "selection": {
            "metric": a.best_metric,
            "direction": "lower_is_better",
            "stop_recall_gate": a.min_stop_recall,
            "stop_recall_source": best_val.get("stop_recall_source"),
        },
        "best_val": best_val,
        "last_val": last_val,
        # 顶层速查（与 best_val 同值，避免下游各写一套取值路径）
        "stop_recall": best_val.get("stop_recall"),
        "stop_recall_source": best_val.get("stop_recall_source"),
        "move_mse12": (best_val.get("move") or {}).get("mse12"),
        "move_mse8": (best_val.get("move") or {}).get("mse8"),
        "near_collision": (best_val.get("near_collision")
                           or {"available": False, "count": None}),
        "traj_dev_m": {c: (best_val.get(c) or {}).get("traj_dev_m")
                       for c in ("move", "trans", "idle")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[done] out={out} best={best}", flush=True)
    if device == "xpu":
        print(f"[mem] peak={torch.xpu.max_memory_allocated()/2**30:.2f} GiB "
              f"reserved={torch.xpu.memory_reserved()/2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
