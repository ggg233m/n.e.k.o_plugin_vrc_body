# -*- coding: utf-8 -*-
"""输入口径 manifest：训练时存下，推理时按它重建**同一张**输入卡。

为什么必须要有（2026-11-05）：
  训练侧有一堆会改变输入分布/格式的开关（drop_wp / quad_feats / prompt_format /
  vision_dropout / topo_dropout / act_dim / steps），但它们只存在 args.json 里，
  **checkpoint 目录本身不携带**。runtime 默认值与训练值不一致时，推理会静默
  喂进模型没见过的输入。已实测踩到两次：
    * eval_vision_ablation 默认 drop_wp=False，而 r8_nowp 训练时 --drop-wp
      -> 评测时喂了训练中从未出现的 wp；
    * policy_runtime 默认 act_dim=2，而训练是 act_dim=3 -> size mismatch。

契约：
  * 训练侧 save() 每个 checkpoint 目录写 input_schema.json；
  * 推理侧 PolicyRuntime 读它，构造和训练逐字一致的输入卡；
  * 缺失时**不猜**：用 input_schema.LEGACY_DEFAULTS 并打警告，
    因为旧 checkpoint 的训练口径无从得知，猜错比报错更危险。
"""
from __future__ import annotations

import json
from pathlib import Path

from policy_histframe import HISTORY_FRAME_S

SCHEMA_NAME = "input_schema.json"
SCHEMA_VERSION = 1

# 老 checkpoint（没有 input_schema.json）的兜底口径。
# 刻意选「训练侧最常见的默认值」而不是「最安全的值」：
# 猜错方向必须是可检测的（warnings 里明确报出来），不能悄悄变成另一种模型。
LEGACY_DEFAULTS = {
    "version": 0,
    "act_dim": 2,
    "steps": 12,
    "step_s": 0.1,
    "prompt_format": "legacy",
    "drop_wp": False,
    "path_in": False,
    "drop_topo": False,
    "quad_feats": False,
    "numeric_bypass": False,
    "traversability_head": False,
    "mode_dim": 1,
    "video_frames": 2,
    # 老 checkpoint 一律按当时的 0.4 s 口径；新训练由 from_args 显式写入 0.2。
    "history_frame_s": 0.4,
    # 老 checkpoint 一律按当时的单点 ZOH 口径，源频率记 0。
    "omega_source_hz": 0,
    # ---- S2 读出（2026-11-06）：全部默认「关」，旧 checkpoint 语义不变 ----
    # 老 ckpt 的 head 输入只有 LM 末位 hidden；新增开关若默认开，会静默改变
    # 已发布 checkpoint 的输入口径。默认关 + load() setdefault 保证向后兼容。
    "readout": "none",          # none | vision_query | additive | gated
    "readout_dim": 0,
    "vision_query": 0,
    "vision_query_dim": 0,
    "no_last_token": False,
    "use_hidden": True,
    "legacy": True,
}


def from_args(a) -> dict:
    """从训练 argparse 结果构造 manifest。"""
    g = lambda k, d=None: getattr(a, k, d)
    _vq = int(g("vision_query", 0) or 0)
    _fu = str(g("fusion", "none") or "none").strip().lower()
    _kind = "vision_query" if _vq > 0 else (_fu if _fu not in ("none", "") else "none")
    return {
        "version": SCHEMA_VERSION,
        "act_dim": int(g("act_dim", 3) or 3),
        "steps": 12,
        "step_s": 0.1,
        "prompt_format": str(g("prompt_format", "card")),
        # 训练时是否把 wp 置 0（= 部署时没有未来轨迹，必须一致）
        "drop_wp": bool(g("drop_wp", False)),
        # 训练时显式打开才允许在线消费 A* 路径形状；默认关闭，避免旧模型被
        # 运行时注入未训练过的路径 token。
        "path_in": bool(g("path_in", False)),
        # topo 三项是否整组置 0
        "drop_topo": bool(g("topo_mode", "full") == "drop"),
        "topo_mode": str(g("topo_mode", "full")),
        "quad_feats": bool(g("quad_feats", False)),
        "numeric_bypass": bool(g("numeric_bypass", False)),
        "traversability_head": bool(g("traversability_head", False)),
        "mode_dim": 1,
        "video_frames": 2,
        # 视觉历史帧间隔（秒）。2026-11-05 由 0.4 改为 0.2；
        # 推理侧必须与训练时一致，否则喂进去的「历史帧」时间跨度不对，
        # 而形状完全合法 —— 属于 RuntimeError 防不住的那一类静默错误。
        "history_frame_s": float(HISTORY_FRAME_S),
        # ω 标签的**源采样频率**（Hz）。与视觉观测 5 Hz、动作标签 10 Hz 是三件事。
        # 30 = 用 30 Hz 子网格因果聚合；0 = 单点 ZOH（旧行为）。
        "omega_source_hz": int(getattr(a, "omega_src_hz", 30) or 0),
        "omega_unit": "rad_s",
        "omega_norm": bool(g("omega_norm", True)),
        # ---- S2 读出（2026-11-06）----
        "readout": _kind,
        # readout_dim 由训练侧 save() 用 policy_readout.schema_fields() 覆盖成真实值；
        # 这里给 0 只表示「无读出模块」，绝不猜一个看似合理的数。
        "readout_dim": int(g("readout_dim", 0) or 0),
        "vision_query": _vq,
        "vision_query_dim": int(g("vision_query_dim", 256) or 256),
        "no_last_token": bool(g("no_last_token", False)),
        "use_hidden": not bool(g("no_last_token", False)),
        "legacy": False,
    }


def save(path, schema: dict) -> Path:
    p = Path(path) / SCHEMA_NAME
    p.write_text(json.dumps(schema, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load(run_dir, warn=None) -> dict:
    """读 manifest；缺失则返回 LEGACY_DEFAULTS 并警告。**绝不静默猜**。"""
    p = Path(run_dir) / SCHEMA_NAME
    if not p.exists():
        if warn is not None:
            warn("[schema] %s 没有 %s，回退 LEGACY_DEFAULTS（act_dim=%d, drop_wp=%s）。"
                 "该 checkpoint 的训练口径无从得知，请确认后再用于真机。"
                 % (run_dir, SCHEMA_NAME, LEGACY_DEFAULTS["act_dim"],
                    LEGACY_DEFAULTS["drop_wp"]))
        return dict(LEGACY_DEFAULTS)
    s = json.loads(p.read_text(encoding="utf-8"))
    for k, v in LEGACY_DEFAULTS.items():
        s.setdefault(k, v)
    return s


def describe(s: dict) -> str:
    """一行摘要，用于启动日志。"""
    keys = ("act_dim", "steps", "prompt_format", "drop_wp", "path_in", "drop_topo",
            "quad_feats", "numeric_bypass", "traversability_head",
            "readout", "vision_query", "no_last_token")
    return " ".join("%s=%s" % (k, s.get(k)) for k in keys)
