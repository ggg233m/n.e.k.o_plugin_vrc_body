# -*- coding: utf-8 -*-
"""policy 运行时（M0，2 Hz）：固定 192 token + torch.compile + 软引导兼容。

对外接口::

    rt = PolicyRuntime(model_dir, run_dir)          # 加载、compile、预热
    out = rt.act(cur_rgb, prev_rgb, wp=(dx,dz), body=(vx,vz))
    # out["block"] = 12 × (vx,vz) m/s；out["obs_t"] = perf_counter
    # out["stop"] / out["stop_source"]：唯一的停车判定（见 policy_modes）
    # 控制器按 §9.1 取步执行；now-obs_t > 0.8 s 必须停车

三条硬口径：
1. **固定长度**：input_ids/attention_mask/mm_token_type_ids 一律 pad 到 max_len（默认 192），
   否则 Inductor 每遇到新 token 数就重编译。文本里航点/速度数字变长不影响形状。
2. **图像**：原始全分辨率 RGB → ``to_policy_rgb``（与训练同一几何：1077 中心裁 → 448×252 → pad 448×256）。
3. **wp≈0 不是停车条件**（2026-11-06 修）：``|wp| < wp_stop`` 只表示「没有路径先验」，
   不得因此把动作整块置零。停车只能来自 ``policy_modes.decide_stop`` 的五种来源：
   显式导航状态 / 停止头 / 安全门 / 过期保护 / ESTOP。``wp_stop`` 参数已 deprecated，
   现在**只**用来给日志标注「这一拍有没有路径先验」。

自测（在 dataset 的 val 样本上跑，验证固定长度不重编译 + 延迟）::

    G:\\vrc_recording\\vrc_model\\venv_xpu\\Scripts\\python.exe research/tools/policy_runtime.py \\
        --run h:\\...\\.venv_xpu\\runs\\m0_s3_448\\best \\
        --out h:\\...\\.venv_xpu\\runs\\runtime_selftest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_policy_lora import K_STEPS, load_model, _extract_visual_feat  # noqa: E402
from policy_modes import decide_stop  # noqa: E402
from policy_text import build_guidance_text, build_policy_text  # noqa: E402


# 2026-11-05：`to_policy_rgb` 已搬到 `policy_resize.py`（那个模块不 import torch，
# `.venv` 主后端也能用）。这里保留 re-export，旧的 `from policy_runtime import
# to_policy_rgb` 调用点不受影响。**只有一份实现**，不会漂移。
from policy_resize import to_policy_rgb  # noqa: E402,F401

# 单点前视版本（**训练分布就是它**）：训练集每帧只喂一个 lookahead=0.8 的点
# （build_policy_dataset.waypoint_body）。不提路径时行为与训练逐字一致。
PROMPT = ("Body velocity vx=%.2f vz=%.2f m/s. Local goal dx=%.2f dz=%.2f m. "
          "Predict the next 12 body velocities at 0.1 s steps.")

# 带路径形状版本（2026-11-02 新增，**默认关闭**）。
#
# 为什么要加：只喂一个前视点时，模型拿到的信息与 pure-pursuit 完全相同，
# 轨迹必然是 A* 的时间重参数化 —— 「不会绕障碍」不是学得差，是输入里没有弯。
# 这里把沿路径前进的 k 个点也写进文本，模型至少能看见「前面要左拐」。
#
# ⚠️ **这会偏离训练分布**：训练集里没有这些 token，模型没学过怎么用它们。
# 所以默认关闭，只在 VRC_POLICY_PATH_IN=1 时启用，且必须做 A/B 对照。
# 真正要让它学会用，得按同一格式重录/重训（方案 §15.17 的 M1 项）。
PROMPT_PATH = ("Body velocity vx=%.2f vz=%.2f m/s. Local goal dx=%.2f dz=%.2f m. "
               "Path ahead (dx dz, right/fwd, every 0.8 m):"
               "%s "
               "Predict the next 12 body velocities at 0.1 s steps.")


def setup_oneapi(graph: bool = False):
    """Triton/Inductor 在 XPU 上需要 oneAPI 的 icpx/ocloc；没配就补上。

    顺带把编译缓存钉死到仓库下的 .tc_cache/ ：不设的话 Inductor 会在**当前工作目录**
    生成 torchinductor_xxxxxxxx/ 随机名目录，一次一个、几百 MB，而且会被 git 看见。

    ``graph=True`` 时切到 Level-Zero **V2** 适配器（``SYCL_UR_USE_LEVEL_ZERO_V2=1``）。
    这是 torch.xpu.XPUGraph 能用的**唯一**前提：V1 适配器对
    ``UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP`` 硬编码返回 false（见
    ``Docs/CPU派发开销剖析-2026-11-01.md`` §13.1 与 §14）。

    ⚠️ V2 有代价：**单算子派发比 V1 慢 57%（add_ 22.3→35.1 µs），整模型 eager 慢 4.6%**
    （318.6→333.1 ms）。所以**只在确实要用图捕获时才开**，默认关。
    """
    bin_dir = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
    if os.path.isdir(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("ONEAPI_ROOT", r"D:\Program Files (x86)\Intel\oneAPI")
        os.environ.setdefault("CC", "icpx")
        os.environ.setdefault("CXX", "icpx")

    cache_root = Path(__file__).resolve().parents[2] / ".tc_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))

    if graph:
        os.environ["SYCL_UR_USE_LEVEL_ZERO_V2"] = "1"
        # V2 的分配器在大块 attention mask 上会误报 OUT_OF_RESOURCES，这两个是兜底。
        os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")
        os.environ.setdefault("UR_L0_USE_RELAXED_ALLOCATION_LIMITS", "1")


class PolicyRuntime:
    def __init__(self, model_dir: str, run_dir: str, size=(448, 256), max_len: int = 192,
                 wp_stop: float = 0.02, compile: bool = True, device: str | None = None,
                 graph: bool = False, merge: bool = True):
        # graph=True 会切到 Level-Zero V2 适配器（XPUGraph 的唯一前提）。
        # 注意 setup_oneapi 必须在 import torch 之前设环境变量，所以这里是第一个调用点。
        setup_oneapi(graph=graph)
        self.graph_enabled = bool(graph)
        self.merged = bool(merge)
        self.size = (int(size[0]), int(size[1]))
        self.max_len = int(max_len)
        # deprecated（2026-11-06）：wp≈0 不再置零动作，这个值只定义
        # 「多近算没有路径先验」的日志注记阈值，不参与停车判定。
        self.wp_stop = float(wp_stop)
        self.device = device or ("xpu" if torch.xpu.is_available() else "cpu")

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_dir)
        # ⚠️ 输入口径必须从 checkpoint 的 input_schema.json 读，不能用本文件默认值。
        # 训练侧有 drop_wp / quad_feats / prompt_format 等一堆会改变输入分布与
        # token 序列的开关；默认值不一致时会静默喂进模型没见过的输入（已踩两次）。
        import input_schema
        self.schema = input_schema.load(
            run_dir, warn=lambda m: print(m, flush=True))
        self.act_dim = int(self.schema["act_dim"])
        self.drop_wp = bool(self.schema["drop_wp"])
        # 只有专门按 A* 软引导格式训练的 checkpoint 才消费路径形状。
        # 旧 checkpoint 即使收到 path，也必须静默丢弃，避免输入分布被在线配置改变。
        self.path_in = bool(self.schema.get("path_in", False))
        self.drop_topo = bool(self.schema["drop_topo"])
        self.quad_feats = bool(self.schema["quad_feats"])
        self.prompt_format = str(self.schema["prompt_format"])
        # 记忆卡 v2（2026-11-06）：无 [mode]。旧 ckpt 用 v1 卡训练，**不能用新卡评测**。
        # 这里只告警不硬失败：B3 等旧 checkpoint 仍需能加载用于非横向比较的场景，
        # 但任何与 v2 数字的对比都是无效的。
        import policy_text
        self.card_version = int(self.schema.get("card_version", 1) or 1)
        if self.card_version != policy_text.CARD_VERSION:
            print("[runtime] ⚠️ card_version=%d，但当前卡是 v%d（无 [mode]）。"
                  "旧 ckpt 的训练输入含 [mode]，**不得与新卡结果横向比较**。"
                  % (self.card_version, policy_text.CARD_VERSION), flush=True)
        # 纯读出实验（S2）：禁止 A* 路径 / 任务卡进入输入（数据仍为 0 个）。
        self.readout_only = bool(self.schema.get("readout_only", False))
        print("[runtime] input_schema: " + input_schema.describe(self.schema), flush=True)
        _head_sd = torch.load(Path(run_dir) / "head.pt", map_location="cpu")
        _out = int(_head_sd["fc.weight"].shape[0])
        if _out // K_STEPS != self.act_dim:
            raise RuntimeError(
                "head 输出 %d = K_STEPS%d × act_dim%d，与 manifest act_dim=%d 矛盾"
                % (_out, K_STEPS, _out // K_STEPS, self.act_dim))
        self.model, self.head, self.hidden = load_model(model_dir, 128, self.device,
                                                        act_dim=self.act_dim)
        # ---- S2 读出（2026-11-06）：按 manifest 重建，缺失就**报错**不静默 ------
        # 旧 ckpt 的 schema 没有这些字段（input_schema.load 用 LEGACY_DEFAULTS 补齐），
        # 全部落到 readout="none" / use_hidden=True ⇒ 与旧路径逐位一致。
        import policy_readout
        self.readout = policy_readout.build_readout(
            str(self.schema.get("readout", "none") or "none"),
            q_dim=int(self.schema.get("vision_query_dim", 0) or 256),
            n_query=int(self.schema.get("vision_query", 0) or 4),
            out_dim=int(self.schema.get("readout_dim", 0) or 256))
        self.readout_dim = int(getattr(self.readout, "out_dim", 0) or 0)
        # 数值 7 维是否直接拼进动作头（--fusion additive 的 S2 口径）
        self.readout_num_in = bool(self.schema.get("readout_num_in", False))
        # use_hidden=False = 动作头不吃 LM 末位 hidden（--no-last-token）
        self.use_hidden = bool(self.schema.get("use_hidden", True))
        _head_in = int(_head_sd["fc.weight"].shape[1])
        # 数值旁路维数按 manifest 走（v1 旧卡 8 维 / v2 新卡 7 维，均不含 mode 语义）。
        _numeric_dim = int(self.schema.get("numeric_dim", 7) or 7)
        _quad_dim = _numeric_dim + _numeric_dim + _numeric_dim * (_numeric_dim - 1) // 2
        _expect_extra = (self.readout_dim
                         + (_numeric_dim if self.readout_num_in else 0)
                         + (_quad_dim if self.quad_feats else 0))
        _expect_in = (self.hidden if self.use_hidden else 0) + _expect_extra
        if _head_in != _expect_in:
            raise RuntimeError(
                "head 输入 %d 与 manifest 推算 %d 不一致（hidden=%d use_hidden=%s "
                "readout=%r readout_dim=%d quad_feats=%s）。"
                "说明该 checkpoint 需要本运行时还不支持的额外输入（例如 "
                "inject-visual / numeric-bypass），或 schema 与 head.pt 不配套 —— "
                "拒绝静默按错误口径推理" %
                (_head_in, _expect_in, self.hidden, self.use_hidden,
                 self.schema.get("readout"), self.readout_dim, self.quad_feats))
        if _head_in != self.hidden:
            from train_policy_lora import ActionHead as _AH  # 同一份实现，避免两套
            self.head = _AH(self.hidden, out=K_STEPS * self.act_dim,
                            extra=_head_in - (self.hidden if self.use_hidden else 0),
                            use_hidden=self.use_hidden).to(self.device)
        self.head.load_state_dict(_head_sd)
        if self.readout is not None:
            _rd = Path(run_dir) / "readout.pt"
            if not _rd.exists():
                raise RuntimeError("manifest 声明 readout=%r，但 %s 不存在；"
                                   "读出权重缺失时动作头输入不可用"
                                   % (self.schema.get("readout"), _rd))
            self.readout.load_state_dict(torch.load(_rd, map_location="cpu"))
            self.readout.to(self.device).eval()
            print("[runtime] readout=%s dim=%d params=%d no_last_token=%s"
                  % (self.schema.get("readout"), self.readout_dim,
                     policy_readout.param_count(self.readout), not self.use_hidden),
                  flush=True)
        # stop_head（2026-11-06）：manifest 声明就加载；缺失权重**报错**不静默。
        self.stop_head = None
        self.stop_thresh = float(self.schema.get("stop_head_thresh", 0.5) or 0.5)
        if bool(self.schema.get("stop_head", False)):
            _sh = Path(run_dir) / "stop_head.pt"
            if not _sh.exists():
                raise RuntimeError("manifest 声明 stop_head=true，但 %s 不存在" % _sh)
            _sd = torch.load(_sh, map_location="cpu")
            _s_in = int(_sd["fc1.weight"].shape[1])
            if _s_in != _head_in:
                raise RuntimeError("stop_head 输入 %d 与动作头输入 %d 不一致"
                                   % (_s_in, _head_in))
            self.stop_head = policy_readout.StopHead(
                _s_in, hidden=int(_sd["fc1.weight"].shape[0])).to(self.device)
            self.stop_head.load_state_dict(_sd)
            self.stop_head.eval()
            print("[runtime] stop_head in_dim=%d hidden=%d thresh=%.2f（与 safety_gate 并联）"
                  % (_s_in, int(_sd["fc1.weight"].shape[0]), self.stop_thresh), flush=True)
        # 训练侧对**内层模型**做了 torch.compile，save_pretrained 会把 372 条 LoRA
        # 键名加上 ._orig_mod. 前缀；裸 load_adapter 会静默跳过，
        # 结果等于用未训练的 adapter 推理（实测 mse8 差 7 倍）。
        from peft_compat import load_adapter_compat
        _st = load_adapter_compat(self.model, Path(run_dir) / "adapter", name="trained")
        print("[runtime] adapter=%s" % (_st,), flush=True)
        if self.merged:
            # 合并 LoRA 进基座权重。**这是最大的一笔单项优化**：真机实测
            # 417.3 → 319.5 ms（−23.4%，省 97.8 ms）。不合并时每层要多走一次
            # LoRA 分支（两次 matmul + add），24 层 × 12 个 Linear 全部累加。
            # 合并后 PeftModelForCausalLM 变成 Qwen3_5ForConditionalGeneration，
            # 层级少一层（见 _inner_module）。
            self.model = self.model.merge_and_unload()
        self.model.eval(); self.head.eval()

        pad = getattr(self.processor.tokenizer, "pad_token_id", None)
        self.pad_id = int(pad) if pad is not None else 0

        self._inner = self._inner_module()
        self.dual = None
        self.graph_ready = False
        # ---- S2 读出需要在**同一次前向**里拿视觉 token ----------------------
        # 训练侧用 forward hook 捕视觉塔输出（policy-runtime 与训练必须同一份口径）；
        # XPUGraph 双图只返回 last_hidden_state，捕不到视觉 token ⇒ 有读出时关掉双图。
        self._vis_cap = {}
        if self.readout is not None:
            _vis = self._visual_module()
            if _vis is None:
                raise RuntimeError("manifest 声明 readout，但内层模型没有 visual 模块")
            _vis.register_forward_hook(
                lambda m, args, kwargs, out: self._vis_cap.__setitem__("feat", out),
                with_kwargs=True)
            if graph:
                print("[runtime] readout 需要视觉 token，关闭 XPUGraph 双图（走 eager）",
                      flush=True)
                graph = False
                self.graph_enabled = False
        if compile:
            print("[runtime] torch.compile inner model …", flush=True)
            self._inner = torch.compile(self._inner)
        self._warmup()
        if graph and self.merged and self.device == "xpu":
            self._setup_graph()


    def _visual_module(self):
        """拿视觉塔（兼容 torch.compile 包一层 ``_orig_mod``）。"""
        inner = getattr(self._inner, "_orig_mod", self._inner)
        return getattr(inner, "visual", None)


    def _inner_module(self):
        """拿到要编译/前向的 Qwen3_5Model，兼容合并前后两种层级。

        不合并：PeftModelForCausalLM > LoraModel > Qwen3_5ForConditionalGeneration > Qwen3_5Model
        合并后：Qwen3_5ForConditionalGeneration > Qwen3_5Model（在 base_model）
        """
        if self.merged:
            return self.model.base_model
        return self.model.base_model.model.model

    # ------------------------------------------------------------ 输入
    def build_inputs(self, cur_rgb: np.ndarray, prev_rgb: np.ndarray,
                     wp, body, preprocessed: bool = False, path=None,
                     topo=None, guide=None,
                     task_prompt: str | None = None):
        """``path``：沿路径前进的 k 个机体系点 [(dx,dz), …]；None 或空 = 用单点 prompt。

        ⚠️ 给 ``path`` 会改变 token 序列（插入 k 组数字），**偏离训练分布**。
        默认 None 时与训练逐字一致。

        ``mode`` 已从本函数删除（2026-11-06）：mode 只进 ``decide_stop``，不进模型输入。
        """
        if not preprocessed:
            cur_rgb = to_policy_rgb(cur_rgb, self.size)
            prev_rgb = to_policy_rgb(prev_rgb, self.size)
        # ⚠️ 按 manifest 抹掉训练时被抹掉的字段。这是 RuntimeError 防不住的那一类：
        # 训练时 --drop-wp 的模型，推理若喂真 wp，形状对得上、不报错，只是分布错。
        if self.drop_wp:
            wp = (0.0, 0.0)
        if self.drop_topo:
            topo = None
        video = np.stack([np.asarray(prev_rgb, np.uint8), np.asarray(cur_rgb, np.uint8)])
        # 记忆卡 v2（2026-11-06）：**不再传 mode**。与训练侧同一个函数（门 G8）。
        # 各写一套模板必然静默漂移，这条和 §5.2 的图像 resize 是同一类纪律。
        text = build_policy_text(wp, topo, body)
        # 纯读出实验（S2）硬门：A* 路径与任务卡当前数据为 0 个，禁止提前注入。
        if self.readout_only and (path or task_prompt):
            raise RuntimeError("readout_only 模型禁止 A* 路径 / 任务卡输入；"
                               "S2 是纯感知读出实验，条件输入数据尚未录制")
        if path and self.path_in:
            # 路径形状是**附加**输入（VRC_POLICY_PATH_IN，默认关）。
            # 训练集里没有这些 token，所以它不属于记忆卡格式，单独追加在卡后。
            pts = " ".join("%.2f %.2f" % (float(a), float(b)) for a, b in path)
            text = text + " Path ahead (dx dz, right/fwd, every 0.8 m):" + pts
            text = text + build_guidance_text(guide)
        elif self.path_in:
            # 空路径也是有效的训练/部署状态：表示 A* 当前失效或过期，不能静默
            # 伪造一条路径。质量字段让模型知道应该更多依赖画面。
            text = text + " Path ahead (dx dz, right/fwd, every 0.8 m):" + build_guidance_text(guide)
        # 这是推理实验入口，不是旧 checkpoint 已训练过的输入字段。
        # 通过后端设置的场次级提示只追加一次，禁止逐帧拼接主 LLM 输出。
        if task_prompt:
            cleaned = " ".join(str(task_prompt).replace("\x00", " ").split())[:96]
            if cleaned:
                text = text + " Task instruction: " + cleaned
        tmpl = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)
        # 数值通道（4-query/additive fusion 与 quad_feats 都要）：与记忆卡**同源**，
        # 且必须是**被 drop_wp / drop_topo 处理过**的值，否则线上口径与训练不一致。
        # 7 维（wp2 + topo3 + body2，**不含 mode**）：mode 只用于停止决策。
        _tp = tuple(float(x) for x in (topo or (0.0, 0.0, 0.0)))[:3]
        self._nums = torch.tensor([[float(wp[0]), float(wp[1]), _tp[0], _tp[1], _tp[2],
                                    float(body[0]), float(body[1])]],
                                  dtype=torch.float32)
        min_px = max_px = self.size[0] * self.size[1]
        inputs = self.processor(text=[tmpl], videos=[video], return_tensors="pt",
                                padding=True, min_pixels=min_px, max_pixels=max_px,
                                cap_pixels_per_frame=False, do_sample_frames=False)
        ids = inputs["input_ids"][0].tolist()
        if len(ids) > self.max_len:
            raise RuntimeError(f"prompt {len(ids)} tokens > max_len {self.max_len}")
        pad = self.max_len - len(ids)
        ids = ids + [self.pad_id] * pad
        mask = [1] * (self.max_len - pad) + [0] * pad
        out = dict(input_ids=torch.tensor([ids], dtype=torch.long),
                   attention_mask=torch.tensor([mask], dtype=torch.long),
                   pixel_values_videos=inputs["pixel_values_videos"],
                   video_grid_thw=inputs["video_grid_thw"])
        if "mm_token_type_ids" in inputs:
            mtt = inputs["mm_token_type_ids"][0].tolist() + [0] * pad
            out["mm_token_type_ids"] = torch.tensor([mtt], dtype=torch.long)
        return out

    def _warmup(self):
        z = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        b = self.build_inputs(z, z, (0.0, 0.0), (0.0, 0.0), preprocessed=True)
        with torch.no_grad():
            self._forward(b)
        if self.device == "xpu":
            torch.xpu.synchronize()

    def _setup_graph(self):
        """建双图。失败就永久回退 eager，不抛 —— 图是优化不是正确性前提。"""
        from policy_dual_graph import DualGraphRuntime
        print("[runtime] XPUGraph 双图捕获 …", flush=True)
        self.dual = DualGraphRuntime(self._inner_module(), max_len=self.max_len)
        z = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        b = self.build_inputs(z, z, (0.0, 0.0), (0.0, 0.0), preprocessed=True)
        ok = self.dual.setup(b)
        st = self.dual.stats()
        if ok:
            print("[runtime] 双图就绪：setup %.1fs, split_sizes=%s"
                  % (st["setup_seconds"], st["split_sizes"]), flush=True)
        else:
            print("[runtime] 双图失败，回退 eager：%s" % st["setup_error"], flush=True)
        self.graph_ready = bool(ok)

    def _apply_head(self, pooled):
        """拼额外输入后过动作头；``readout`` 用**本次前向**捕到的视觉 token。"""
        extras = []
        _nums = self._nums.to(self.device)
        if self.readout is not None:
            feat = _extract_visual_feat(self._vis_cap.get("feat"), pooled.shape[0])
            if feat is None:
                raise RuntimeError("manifest 声明 readout，但这次前向没捕到视觉 token"
                                   "（torch.compile / 双图路径可能吞掉了 hook）")
            if policy_readout.needs_numeric(self.readout):
                extras.append(self.readout(feat.float(), _nums))
            else:
                extras.append(self.readout(feat.float()))
                if self.readout_num_in:
                    extras.append(_nums)
        if self.quad_feats:
            extras.append(_nums)
        x = pooled.float() if self.use_hidden else None
        _head_in = torch.cat(([x] if x is not None else []) + extras, -1) if extras else x
        # stop_head（2026-11-06）：与动作头**同一特征向量**；只进 decide_stop，
        # 不改变喂给模型的输入，也不替代 safety_gate。
        self._stop_logit = (self.stop_head(_head_in)
                            if self.stop_head is not None else None)
        return self.head(_head_in) if extras else self.head(x)

    def _forward(self, b):
        if self.dual is not None and self.dual.ready:
            h = self.dual.forward(b)
            if h is not None:
                kw = {k: v.to(self.device) for k, v in b.items()}
                idx = kw["attention_mask"].sum(-1) - 1
                pooled = h[torch.arange(h.shape[0], device=h.device), idx]
                return self._apply_head(pooled).view(-1, K_STEPS, self.act_dim)
        kw = {k: v.to(self.device) for k, v in b.items()}
        self._vis_cap.clear()
        self._stop_logit = None
        out = self._inner(**kw, use_cache=False)
        h = out.last_hidden_state
        idx = kw["attention_mask"].sum(-1) - 1
        pooled = h[torch.arange(h.shape[0], device=h.device), idx]
        return self._apply_head(pooled).view(-1, K_STEPS, self.act_dim)

    # ------------------------------------------------------------ 对外
    def act(self, cur_rgb: np.ndarray, prev_rgb: np.ndarray, wp, body,
            preprocessed: bool = False, path=None, mode: int = 0,
            topo=None, guide=None, task_prompt: str | None = None,
            nav_state=None, stop_head=None, safety_veto: bool = False,
            localization_ok=None, localization_reason=None) -> dict:
        """``nav_state``：NavSession.mode 字符串（``goto/explore/idle/done/stop``）。

        ``stop_head``：停止头概率（0..1），当前 checkpoint 没有这个头 ⇒ None。
        ``safety_veto``：感知线程给的近距障碍否决（安全门来源）。
        ``localization_ok`` / ``localization_reason``：**独立的**定位安全门结果
        （由 nav_online 用 ``policy_modes.decide_localization`` 算好后传入）；
        ``None`` = 没有定位子系统，不适用。**与 wp 无关**：``wp=(0,0)`` 永远不停车。
        这些参数都只进入 ``decide_stop``，**不会**改变喂给模型的输入。

        ⚠️ estop 与 stale（过期保护）**不**在这里判：它们属于执行器视野
        （STOP 文件、``now - obs_t``），见 policy_executor.decide_stop 的合并。
        """
        t0 = time.perf_counter()
        # ``mode`` 只为调用方兼容保留（policy_runner 仍在传）；**不进模型输入**。
        _ = mode
        b = self.build_inputs(cur_rgb, prev_rgb, wp, body, preprocessed=preprocessed,
                              path=path, topo=topo, guide=guide,
                              task_prompt=task_prompt)
        with torch.no_grad():
            pred = self._forward(b)
        if self.device == "xpu":
            torch.xpu.synchronize()
        block = pred[0].float().cpu().numpy()
        # 2026-11-06：删掉 `if |wp| < wp_stop: block = zeros`。
        # wp≈0 = 「没有路径先验」（A* 不可用 / 未定位 / 自由探索），是**合法**的
        # 正常工作状态 —— 此时模型应更多依赖画面，而不是被外部强行停车。
        # 停机只认 decide_stop 的六个来源；定位坏了由**独立**的 localization 安全门
        # 负责（safety_gate_unlocalized），不看 wp。wp_near_zero 只进日志注记。
        try:
            wp_near = float(np.hypot(float(wp[0]), float(wp[1]))) < self.wp_stop
        except (TypeError, ValueError, IndexError):
            wp_near = None
        # stop_head 概率：调用方显式传入优先（联调/回放用），否则用本次前向算出的。
        _p = None
        if self._stop_logit is not None:
            _p = float(torch.sigmoid(self._stop_logit.float())[0])
        stop_head_p = stop_head if stop_head is not None else _p
        decision = decide_stop(safety_veto=bool(safety_veto), nav_state=nav_state,
                               stop_head_p=stop_head_p,
                               localization_ok=localization_ok,
                               localization_reason=localization_reason,
                               wp_near_zero=wp_near)
        stop = bool(decision["stop"])
        if stop:
            block = np.zeros_like(block)
        return dict(block=block.round(4).tolist(), stop=stop,
                    stop_source=decision["source"], stop_reason=decision["reason"],
                    stop_detail=decision["detail"],
                    # stop_head 概率（None = 该 checkpoint 没有停止头）
                    stop_head_p=stop_head_p,
                    # 兼容别名：旧调用方（policy_runner / executor / 日志）只读 gated。
                    gated=stop, obs_t=t0,
                    latency_ms=(time.perf_counter() - t0) * 1000)


def selftest(a) -> int:
    from train_policy_lora import PolicyDataset, build_index
    ds_samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"][:a.samples]
    # 只取图像与标签；运行时输入已预处理成 448×256，跳过 to_policy_rgb
    from PIL import Image
    rt = PolicyRuntime(a.model, a.run, size=(448, 256), max_len=a.max_len,
                       wp_stop=a.wp_stop, compile=not a.no_compile)
    rows = []
    for s in ds_samples:
        cur = np.asarray(Image.open(s["img"]).convert("RGB"))
        prev = np.asarray(Image.open(s["prev_img"]).convert("RGB"))
        out = rt.act(cur, prev, s["wp"], s["body"], preprocessed=True)
        rows.append(out["latency_ms"])
    res = dict(samples=len(rows), max_len=a.max_len, compiled=not a.no_compile,
               wp_stop=a.wp_stop,
               latency_ms=dict(p50=float(np.percentile(rows, 50)),
                               p95=float(np.percentile(rows, 95)),
                               max=float(max(rows)))
               )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3")
    ap.add_argument("--run", default=r"h:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.venv_xpu\runs\m0_s3_448\best")
    ap.add_argument("--dataset", default=r"G:\vrc_recording\vrc_model\dataset\m0_policy_448")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--wp-stop", type=float, default=0.02,
                    help="**deprecated**（2026-11-06）：不再置零动作，只定义"
                         "「|wp| 多近算没有路径先验」的日志注记阈值")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return selftest(a)


if __name__ == "__main__":
    raise SystemExit(main())
