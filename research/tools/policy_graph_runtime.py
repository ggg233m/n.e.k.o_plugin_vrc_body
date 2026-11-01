# -*- coding: utf-8 -*-
"""Policy 的 XPU 图重放运行时：把 ViT 和 LM 两张静态图抓下来，每拍只搬输入、replay。

**为什么值得做**：真机实测（2026-11-01 晚，VRChat 同跑），`torch.compile` 的端到端是
318 ms（ViT 97.9 + LM 190 + 预处理 29.5）；而图重放的离线数据是 34.6 ms。
两者**差约 8 倍**，不是同一个量级 —— 想进几十毫秒只有图重放这一条路。

**前提**：必须先 `SYCL_UR_USE_LEVEL_ZERO_V2=1`（见 `policy_runtime.setup_oneapi(graph=True)`）。
V1 适配器对图支持查询硬编码返回 false，永远抓不到。

设计上照抄 ARDY 的 `runtime/graph/graph_runtime.py`：
  * 形状白名单 —— 未知规格**回退原地**，绝不在交互时新建图；
  * 每拍把新输入 `copy_` 进固定 buffer；
  * 返回 `clone()`，否则下一次 replay 会覆盖调用方还持有的张量；
  * 捕获失败就把 `enabled` 置 False，永久回退，不反复重试；
  * **capture 完必须立刻 replay 一次**，否则输出是垃圾（见 §14.3）。

ViT 图额外需要两个补丁，原因是原实现里有两处 GPU→CPU 同步：
  * `get_vision_*` 系列里的 `.tolist()`  → 用 `precompute_vision_geometry` 预先算好；
  * attention 里的 `lengths.tolist()`     → 用 `_single_window_attention` 换掉。
"""
from __future__ import annotations

import time

import torch


class GraphRunner:
    """通用图重放器：捕获一个 `fn(*args)`，之后每拍只搬固定 buffer。

    用法::

        g = GraphRunner("lm", lambda: lm(inputs_embeds=B.e, attention_mask=B.m, use_cache=False))
        g.warmup()                                     # 建图
        out = g.forward(embeds=embeds, attention_mask=mask)
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = bool(enabled)
        self.graph = None
        self.buffers = {}          # 名字 -> 固定张量
        self.static_out = None
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0
        self.capture_error = None
        self.capture_seconds = 0.0
        self._fn = None
        self._replay_verified = False

    # ------------------------------------------------------------ 建图
    def has_graph(self) -> bool:
        return self.graph is not None

    def warmup(self, fn, buffers: dict) -> bool:
        """捕获一次图。``buffers`` 里的张量会被 clone 成固定 buffer，内容无所谓。

        返回是否成功。失败时记录 ``capture_error`` 并永久回退，不抛。
        """
        if not self.enabled:
            return False
        if self.graph is not None:
            return True
        started = time.perf_counter()
        try:
            self._fn = fn
            self.buffers = {k: v.detach().clone() for k, v in buffers.items()}
            stream = torch.xpu.Stream()
            stream.wait_stream(torch.xpu.current_stream())
            with torch.inference_mode(), torch.xpu.stream(stream):
                # 先空跑 3 次，把 Triton/内核都预热掉；否则捕获会带进首次编译。
                for _ in range(3):
                    fn()
            stream.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.inference_mode(), torch.xpu.graph(graph, stream=stream):
                self.static_out = fn()
            torch.xpu.synchronize()
            # ⚠️ capture 后的输出是垃圾，必须 replay 一次才是真值（§14.3）
            graph.replay()
            torch.xpu.synchronize()
            self._check_output_finite()
            self.graph = graph
            self.captures += 1
            self._replay_verified = True
            self.capture_seconds = time.perf_counter() - started
            return True
        except Exception as exc:  # noqa: BLE001 - 捕获失败必须能退回 eager
            self.capture_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.enabled = False
            self.graph = None
            self.buffers = {}
            self.static_out = None
            self.capture_seconds = time.perf_counter() - started
            return False

    def _check_output_finite(self) -> None:
        """捕获后确认输出没有 NaN/Inf，否则宁可回退也不要喂垃圾给控制器。"""
        outs = self._flatten(self.static_out)
        for t in outs:
            if not bool(torch.isfinite(t).all()):
                raise RuntimeError(f"{self.name} 图捕获输出含非有限值")

    @staticmethod
    def _flatten(obj):
        if torch.is_tensor(obj):
            return [obj]
        if isinstance(obj, (list, tuple)):
            out = []
            for o in obj:
                out.extend(GraphRunner._flatten(o))
            return out
        if hasattr(obj, "__dict__"):
            out = []
            for k, v in vars(obj).items():
                if v is not None and not isinstance(v, (str, int, float, bool)):
                    out.extend(GraphRunner._flatten(v))
            return out
        return []

    # ------------------------------------------------------------ 每拍
    def forward(self, **inputs):
        """把新输入 copy 进固定 buffer 后 replay。形状不符或没建图就回退。"""
        if self.graph is None or not self.enabled:
            self.fallbacks += 1
            return None
        try:
            for k, v in inputs.items():
                buf = self.buffers.get(k)
                if buf is None or tuple(v.shape) != tuple(buf.shape):
                    self.fallbacks += 1
                    return None
                buf.copy_(v)
            self.graph.replay()
            self.replays += 1
            # 必须 clone：下一次 replay 会把 static_out 原地覆盖掉。
            return self._clone_out(self.static_out)
        except Exception as exc:  # noqa: BLE001 - 交互中绝不能抛进控制环
            self.capture_error = f"replay: {type(exc).__name__}: {str(exc)[:200]}"
            self.enabled = False
            self.graph = None
            self.fallbacks += 1
            return None

    def _clone_out(self, obj):
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._clone_out(o) for o in obj)
        if hasattr(obj, "__dict__"):
            import copy
            return copy.copy(obj)
        return obj

    def stats(self) -> dict:
        return dict(name=self.name, enabled=self.enabled, captured=self.graph is not None,
                    captures=self.captures, replays=self.replays,
                    fallbacks=self.fallbacks,
                    capture_seconds=round(self.capture_seconds, 1),
                    capture_error=self.capture_error,
                    verified=self._replay_verified)


# ---------------------------------------------------------------- ViT 补丁


def _single_window_attention(attn_list):
    """把视觉注意力换成「单窗口整段算」版本。

    原实现在 `eager_attention_forward` 分支里做::

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]

    `lengths.tolist()` 是 **GPU→CPU 同步**，图捕获期间直接报
    `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`。我们的输入恒为单窗口
    （`cu_seqlens = [0, N]`，实测 448x256 / 320x176 / 224x128 都是），
    所以整段一次 SDPA 与 split 后 cat 完全等价。

    **等价性前提**：只有一个 segment。换成分块窗口（`window_index`）后必须撤销。
    返回 `(saved, restore)`，`restore()` 用来撤回补丁。
    """
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None,
                max_seqlen=None, **kwargs):
        n = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=self.scaling)
        out = out.transpose(1, 2).reshape(n, -1).contiguous()
        return self.proj(out)

    saves = [(a, type(a).forward) for a in attn_list]
    for a in attn_list:
        type(a).forward = forward

    def restore():
        for a, orig in saves:
            type(a).forward = orig

    return saves, restore


def precompute_vision_geometry(visual, grid_thw):
    """预先算好 4 个几何量，捕获时直接喂进 `visual(...)` 的 kwargs。

    这些函数内部都有 `.tolist()` / D2H（`get_vision_attention_seqlens`、
    `get_vision_interpolation_indices_and_weights` 等），图外算没成本，
    图内算必挂。transformers 的这几个函数本身就支持从 kwargs 取预计算值。
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    cfg = visual.config
    kw = {}
    ii, iw = M.get_vision_interpolation_indices_and_weights(
        grid_thw, num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=cfg.spatial_merge_size, kwargs={})
    kw["interp_indices"] = ii
    kw["interp_weights"] = iw
    kw["position_ids"] = M.get_vision_position_ids(
        grid_thw, cfg.spatial_merge_size, kwargs={})
    cu, _ = M.get_vision_attention_seqlens(grid_thw, cfg, kwargs={})
    kw["cu_seqlens"] = cu
    return kw


def vision_split_sizes(grid_thw, spatial_merge_size: int):
    """`get_image_features` 里 `(grid.prod(-1) // merge^2).tolist()` 的图外版本。"""
    sizes = (grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
    return tuple(int(s) for s in sizes)