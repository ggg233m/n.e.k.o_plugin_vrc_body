# -*- coding: utf-8 -*-
"""双图运行时：把 ViT 和 LM 各自捕获成一张 XPUGraph，每拍只 replay 两次。

**为什么不抓整模型一张图**：整模型里 `Qwen3_5Model.forward` 的 glue 段
（`get_placeholder_mask` 的 `.sum()` 比较、`compute_3d_position_ids` 的 D2H）
仍有同步点，实测捕获直接报 `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`。而 ViT 和 LM
**各自单独捕获都能成**，所以拆成两张图 + CPU 上的 glue。

**正确性**：双图串联 vs eager，`cos = 1.00000000`（2026-11-01 实测）。
**代价**：glue 段（split / masked_scatter / position_ids）在 CPU/GPU 上每拍走一遍，
实测约 10–15 ms。
**等价性前提**：`_single_window_attention` 只对单窗口成立（`cu_seqlens=[0,N]`）。
换成分块窗口必须撤销补丁 —— `teardown()` 会恢复原始 forward。
"""
from __future__ import annotations

import time

import torch

from policy_graph_runtime import (
    _single_window_attention,
    precompute_vision_geometry,
    vision_split_sizes,
)


class DualGraphRuntime:
    """ViT 图 + LM 图 + CPU glue。

    用法::

        dual = DualGraphRuntime(model, max_len=192)
        dual.setup(inputs)            # 建图；失败则 enabled=False，调用方回退 eager
        out = dual.forward(inputs)    # 返回 last_hidden_state；失败返回 None
        dual.teardown()               # 撤销 ViT attention 补丁
    """

    def __init__(self, model, max_len: int = 192, enabled: bool = True):
        self.model = model                  # Qwen3_5Model（已 merge）
        self.visual = model.visual
        self.lm = model.language_model
        self.max_len = int(max_len)
        self.enabled = bool(enabled)
        self.ready = False
        self.replays = 0
        self.fallbacks = 0
        self.setup_error = None
        self.setup_seconds = 0.0
        self.geom = None
        self.split_sizes = None
        self._restore_attn = None
        self._gvit = None
        self._glm = None
        self._pv = None
        self._ids = None
        self._mask = None
        self._mtt = None
        self._emb = None
        self._pos = None
        self._vit_out = None
        self._lm_out = None

    # ------------------------------------------------------------ 建图
    def setup(self, b: dict) -> bool:
        """用一份真实输入 ``b`` 建两张图。失败则置 ``enabled=False``，不抛。

        ``b`` 是 `build_inputs()` 的输出（CPU 或 XPU 张量都行）。
        """
        if not self.enabled:
            return False
        if self.ready:
            return True
        started = time.perf_counter()
        try:
            dev = next(self.visual.parameters()).device
            pv = b["pixel_values_videos"].to(dev)
            ids = b["input_ids"].to(dev)
            mask = b["attention_mask"].to(dev)
            mtt = b.get("mm_token_type_ids")
            if mtt is not None:
                mtt = mtt.to(dev)
            grid = b["video_grid_thw"].to(dev)

            # --- ViT 补丁：两个 D2H 同步点都得在图外解决 ---
            attn_list = [blk.attn for blk in self.visual.blocks]
            _, restore = _single_window_attention(attn_list)
            self._restore_attn = restore
            self.geom = precompute_vision_geometry(self.visual, grid)
            self.split_sizes = vision_split_sizes(grid, self.visual.spatial_merge_size)

            self._pv = pv.detach().clone()
            self._ids = ids.detach().clone()
            self._mask = mask.detach().clone()
            self._mtt = mtt.detach().clone() if mtt is not None else None

            stream = torch.xpu.Stream()
            stream.wait_stream(torch.xpu.current_stream())

            # --- 图 1：ViT ---
            with torch.inference_mode(), torch.xpu.stream(stream):
                for _ in range(3):
                    self.visual(self._pv, grid_thw=grid, **self.geom)
            stream.synchronize()
            gvit = torch.xpu.XPUGraph()
            with torch.inference_mode(), torch.xpu.graph(gvit, stream=stream):
                self._vit_out = self.visual(self._pv, grid_thw=grid, **self.geom)
            torch.xpu.synchronize()
            # ⚠️ capture 后输出是垃圾，必须 replay 一次（§14.3）
            gvit.replay()
            torch.xpu.synchronize()
            self._gvit = gvit

            # --- glue：把视觉特征拼进 text embeds ---
            emb = self._glue(self._vit_out.pooler_output)
            self._emb = emb.detach().clone()
            with torch.inference_mode():
                pos = self.model.compute_3d_position_ids(
                    input_ids=self._ids, inputs_embeds=self._emb,
                    video_grid_thw=grid, attention_mask=self._mask,
                    mm_token_type_ids=self._mtt)
            self._pos = pos.detach().clone() if pos is not None else None

            # --- 图 2：LM ---
            with torch.inference_mode(), torch.xpu.stream(stream):
                for _ in range(3):
                    self.lm(inputs_embeds=self._emb, attention_mask=self._mask,
                            position_ids=self._pos, use_cache=False)
            stream.synchronize()
            glm = torch.xpu.XPUGraph()
            with torch.inference_mode(), torch.xpu.graph(glm, stream=stream):
                self._lm_out = self.lm(inputs_embeds=self._emb, attention_mask=self._mask,
                                       position_ids=self._pos, use_cache=False)
            torch.xpu.synchronize()
            glm.replay()
            torch.xpu.synchronize()
            if not bool(torch.isfinite(self._lm_out.last_hidden_state).all()):
                raise RuntimeError("LM 图输出含非有限值")
            self._glm = glm

            self.ready = True
            self.setup_seconds = time.perf_counter() - started
            return True
        except Exception as exc:  # noqa: BLE001 - 建图失败必须能退回 eager
            self.setup_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self.enabled = False
            self.ready = False
            self.teardown()
            self.setup_seconds = time.perf_counter() - started
            return False

    def _glue(self, pooler_output):
        """视觉输出 → text embeds。这一步原在 `Qwen3_5Model.forward` 里。"""
        feats = torch.split(pooler_output, list(self.split_sizes))
        ie = self.model.get_input_embeddings()(self._ids)
        _, vmask = self.model.get_placeholder_mask(
            self._ids, inputs_embeds=ie, video_features=feats[0])
        return ie.masked_scatter(vmask, feats[0])

    # ------------------------------------------------------------ 每拍
    def forward(self, b: dict):
        """返回 ``last_hidden_state``；未就绪或形状不符返回 ``None``（调用方回退）。"""
        if not self.ready or not self.enabled:
            self.fallbacks += 1
            return None
        try:
            dev = self._pv.device
            pv = b["pixel_values_videos"].to(dev)
            if tuple(pv.shape) != tuple(self._pv.shape):
                self.fallbacks += 1
                return None
            self._pv.copy_(pv)
            ids = b["input_ids"].to(dev)
            mask = b["attention_mask"].to(dev)
            if tuple(ids.shape) != tuple(self._ids.shape) or tuple(mask.shape) != tuple(self._mask.shape):
                self.fallbacks += 1
                return None
            self._ids.copy_(ids)
            self._mask.copy_(mask)
            if self._mtt is not None and b.get("mm_token_type_ids") is not None:
                self._mtt.copy_(b["mm_token_type_ids"].to(dev))

            self._gvit.replay()
            emb = self._glue(self._vit_out.pooler_output)
            self._emb.copy_(emb)
            # position_ids 与 embeds 无关（只依赖 input_ids/mask/grid），图外算一次就够；
            # 但 mask 可能变，所以每拍重算。
            with torch.inference_mode():
                pos = self.model.compute_3d_position_ids(
                    input_ids=self._ids, inputs_embeds=self._emb,
                    video_grid_thw=b["video_grid_thw"].to(dev), attention_mask=self._mask,
                    mm_token_type_ids=self._mtt)
            if pos is not None and self._pos is not None:
                self._pos.copy_(pos)
            self._glm.replay()
            self.replays += 1
            return self._lm_out.last_hidden_state.clone()
        except Exception as exc:  # noqa: BLE001 - 交互中绝不能抛进控制环
            self.setup_error = f"replay: {type(exc).__name__}: {str(exc)[:200]}"
            self.enabled = False
            self.ready = False
            self.fallbacks += 1
            return None

    def teardown(self) -> None:
        if self._restore_attn is not None:
            try:
                self._restore_attn()
            except Exception:  # noqa: BLE001
                pass
            self._restore_attn = None

    def stats(self) -> dict:
        return dict(enabled=self.enabled, ready=self.ready,
                    replays=self.replays, fallbacks=self.fallbacks,
                    setup_seconds=round(self.setup_seconds, 1),
                    setup_error=self.setup_error,
                    split_sizes=list(self.split_sizes) if self.split_sizes else None)