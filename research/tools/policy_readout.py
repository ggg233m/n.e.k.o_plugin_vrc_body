# -*- coding: utf-8 -*-
"""S2 读出模块：4-query 视觉读出 与 additive / gated fusion（2026-11-06）。

为什么单独成文件
----------------
训练侧（``train_policy_lora``）与推理侧（``policy_runtime``）必须用**同一份**
实现与**同一套**输入契约，否则又是一个「训练喂 A、推理喂 B、形状全对」的静默
错误。本模块只依赖 torch，不 import 任何训练/运行时模块，避免循环 import。

两版读出（用户拍板的 A/B，开关 ``--vision-query`` / ``--fusion``）
-------------------------------------------------------------------
1. ``VisionQueryReadout``：4 个可学习 query 对 112 个视觉 token 做**单头**注意力
   池化，**绕过 LM**，不改 token 序列长度，参数量约 263k：
       vis_proj 1024→256 (262,144) + query 4×256 (1,024) + LayerNorm 512 = 263,680。
   信息瓶颈在 256 维（4×256 输出），而不是 LM 末位 token 那 1024 维里被 24 层
   递归稀释过的一份摘要；代价是完全没有文本/任务卡语义。

2. ``AdditiveFusion``：视觉与数值（body + 因果历史 8 维）各一个小编码器，
   256 维空间里**相加**（``gated`` 时改成学到的 softmax 门控加权）。
   任务卡与 A* 两个槽位**当前留空**：``task.jsonl`` / ``astar_guidance.jsonl``
   在仓库里不存在（见 registry），不造 0 向量参与训练 —— 否则等于把「缺失」
   这一个状态教会模型。

两者的 ``out_dim`` 都进动作头的 extra 拼接位；``--no-last-token`` 时动作头
不再吃 LM 的末位 hidden。
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUMERIC_DIM = 8          # wp dx/dz + topo 3 + body vx/vz + mode（与记忆卡同源）
VIS_DIM = 1024           # 视觉塔输出维度（Qwen3.5-0.8B-S3）
VIS_TOKENS = 112         # 448×256，两帧 temporal merge 后的一条序列

READOUT_KINDS = ("none", "vision_query", "additive", "gated")


class VisionQueryReadout(nn.Module):
    """4-query 单头注意力池化：112 视觉 token → 4×q_dim。

    刻意**不用多头**：263k 参数的小模块在小数据上多头的收益抵不过方差；
    单头 + 4 个 query 已经给了 4 条独立的「看哪里」通路。
    """

    def __init__(self, vis_dim: int = VIS_DIM, q_dim: int = 256, n_query: int = 4):
        super().__init__()
        self.vis_dim = int(vis_dim)
        self.q_dim = int(q_dim)
        self.n_query = int(n_query)
        self.vis_proj = nn.Linear(self.vis_dim, self.q_dim)
        # 小方差初始化：query 初始近似「均匀看全图」，再由梯度分化
        self.query = nn.Parameter(torch.randn(self.n_query, self.q_dim) * 0.02)
        self.norm = nn.LayerNorm(self.q_dim)
        self.out_dim = self.n_query * self.q_dim

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        """``vis``: (B, N, vis_dim) 或 (B*N, vis_dim)。返回 (B, out_dim)。"""
        if vis.dim() == 2:
            raise ValueError("VisionQueryReadout 需要 (B,N,D) 的视觉 token，收到 2 维；"
                             "请从 forward hook 捕获后先 reshape")
        kv = self.vis_proj(vis.float())                       # (B, N, q)
        q = self.query.unsqueeze(0).expand(kv.shape[0], -1, -1)   # (B, 4, q)
        att = torch.matmul(q, kv.transpose(1, 2)) / self.q_dim ** 0.5
        w = att.softmax(dim=-1)                               # (B, 4, N)
        out = torch.matmul(w, kv)                             # (B, 4, q)
        return self.norm(out).reshape(kv.shape[0], -1)


class AdditiveFusion(nn.Module):
    """视觉 / 数值 各自小编码器 → 相加（gated 时学门控）。

    任务卡与 A* 槽位留空（当前无数据），字段见 ``slots``；留空不等于喂 0，
    而是**不参与**，这样将来接上时只需新增编码器，不改变已有两路的语义。
    """

    def __init__(self, vis_dim: int = VIS_DIM, num_dim: int = NUMERIC_DIM,
                 out_dim: int = 256, gated: bool = False):
        super().__init__()
        self.gated = bool(gated)
        self.vis_enc = nn.Linear(int(vis_dim), int(out_dim))
        self.num_enc = nn.Sequential(
            nn.Linear(int(num_dim), 64), nn.SiLU(), nn.Linear(64, int(out_dim)))
        self.norm = nn.LayerNorm(int(out_dim))
        self.gate = nn.Linear(int(out_dim) * 2, 2) if self.gated else None
        self.out_dim = int(out_dim)
        # 尚不存在的两路（任务卡 / A*）。注册成占位名字，方便后续版本对齐。
        self.slots = {"task_card": None, "astar_path": None}

    def forward(self, vis: torch.Tensor, numeric: torch.Tensor | None = None) -> torch.Tensor:
        if vis.dim() == 3:
            vis = vis.float().mean(dim=1)      # 池化到 (B, vis_dim)；均值只是基线，
        else:                                  # 4-query 版本才学「看哪里」
            vis = vis.float()
        if numeric is None:
            raise ValueError("AdditiveFusion 需要 numeric（8 维）输入；"
                             "缺它就等于把数值通道静默置零")
        fv = self.vis_enc(vis)
        fn = self.num_enc(numeric.float())
        if self.gated:
            g = self.gate(torch.cat([fv, fn], dim=-1)).softmax(dim=-1)   # (B, 2)
            out = g[:, 0:1] * fv + g[:, 1:2] * fn
        else:
            out = fv + fn
        return self.norm(out)


def build_readout(kind: str, *, vis_dim: int = VIS_DIM, num_dim: int = NUMERIC_DIM,
                  q_dim: int = 256, n_query: int = 4,
                  out_dim: int = 256) -> nn.Module | None:
    """按 ``kind`` 造读出模块。``none`` 返回 None（= 旧路径，不进 extra）。"""
    k = str(kind or "none").strip().lower()
    if k in ("none", ""):
        return None
    if k == "vision_query":
        return VisionQueryReadout(vis_dim=vis_dim, q_dim=q_dim, n_query=n_query)
    if k in ("additive", "gated"):
        return AdditiveFusion(vis_dim=vis_dim, num_dim=num_dim, out_dim=out_dim,
                              gated=(k == "gated"))
    raise ValueError("未知读出类型 %r（可选 %s）" % (kind, READOUT_KINDS))


def needs_numeric(m: nn.Module | None) -> bool:
    return isinstance(m, AdditiveFusion)


def readout_kind(m: nn.Module | None) -> str:
    if isinstance(m, VisionQueryReadout):
        return "vision_query"
    if isinstance(m, AdditiveFusion):
        return "gated" if m.gated else "additive"
    return "none"


def param_count(m: nn.Module | None) -> int:
    return 0 if m is None else int(sum(p.numel() for p in m.parameters()))


def schema_fields(m: nn.Module | None, no_last_token: bool = False) -> dict:
    """写进 ``input_schema.json`` 的读出字段（旧 checkpoint 全部取默认 off）。"""
    return {
        "readout": readout_kind(m),
        "readout_dim": int(getattr(m, "out_dim", 0) or 0),
        "vision_query": int(getattr(m, "n_query", 0) or 0),
        "vision_query_dim": int(getattr(m, "q_dim", 0) or 0),
        "no_last_token": bool(no_last_token),
        "use_hidden": not bool(no_last_token),
    }
