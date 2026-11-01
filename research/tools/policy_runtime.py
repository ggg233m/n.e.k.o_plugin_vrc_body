# -*- coding: utf-8 -*-
"""policy 运行时（M0，2 Hz）：固定 192 token + torch.compile + 航点门控。

对外接口::

    rt = PolicyRuntime(model_dir, run_dir)          # 加载、compile、预热
    out = rt.act(cur_rgb, prev_rgb, wp=(dx,dz), body=(vx,vz))
    # out["block"] = 12 × (vx,vz) m/s；out["obs_t"] = perf_counter
    # 控制器按 §9.1 取步执行；now-obs_t > 0.8 s 必须停车

三条硬口径：
1. **固定长度**：input_ids/attention_mask/mm_token_type_ids 一律 pad 到 max_len（默认 192），
   否则 Inductor 每遇到新 token 数就重编译。文本里航点/速度数字变长不影响形状。
2. **图像**：原始全分辨率 RGB → ``to_policy_rgb``（与训练同一几何：1077 中心裁 → 448×252 → pad 448×256）。
3. **停住门控**：``|wp| < wp_stop``（默认 0.02 m）时 24 维输出整体置 0（方案 §5.2 的契约）。

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
from train_policy_lora import K_STEPS, load_model  # noqa: E402


def to_policy_rgb(rgb: np.ndarray, size=(448, 256)) -> np.ndarray:
    """同一几何：中心裁 16:9（1077×606）→ 缩到 W×内容高 → 对称补黑边到 W×H。

    XPU venv 没有 cv2；PIL 的 resize 在下采样时按缩放系数放大滤波核（抗锯齿），
    与训练侧 ffmpeg 的 area 缩放同属面积平均族，几何完全一致。
    """
    w, h = int(size[0]), int(size[1])
    content_h = min(h, int(round(w * 9 / 16)))
    content_h -= content_h % 2
    top = (h - content_h) // 2
    src_h, src_w = rgb.shape[:2]
    crop_w = min(int(round(src_h * 16 / 9)), src_w)
    left = (src_w - crop_w) // 2
    crop = rgb[:, left:left + crop_w]
    im = Image.fromarray(crop).resize((w, content_h), Image.BILINEAR)
    out = np.zeros((h, w, 3), np.uint8)
    out[top:top + content_h] = np.asarray(im)
    return out

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
        self.wp_stop = float(wp_stop)
        self.device = device or ("xpu" if torch.xpu.is_available() else "cpu")

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model, self.head, _ = load_model(model_dir, 128, self.device)
        self.model.load_adapter(str(Path(run_dir) / "adapter"), adapter_name="trained", is_trainable=False)
        self.model.set_adapter("trained")
        if self.merged:
            # 合并 LoRA 进基座权重。**这是最大的一笔单项优化**：真机实测
            # 417.3 → 319.5 ms（−23.4%，省 97.8 ms）。不合并时每层要多走一次
            # LoRA 分支（两次 matmul + add），24 层 × 12 个 Linear 全部累加。
            # 合并后 PeftModelForCausalLM 变成 Qwen3_5ForConditionalGeneration，
            # 层级少一层（见 _inner_module）。
            self.model = self.model.merge_and_unload()
        self.head.load_state_dict(torch.load(Path(run_dir) / "head.pt", map_location=self.device))
        self.model.eval(); self.head.eval()

        pad = getattr(self.processor.tokenizer, "pad_token_id", None)
        self.pad_id = int(pad) if pad is not None else 0

        self._inner = self._inner_module()
        self.dual = None
        self.graph_ready = False
        if compile:
            print("[runtime] torch.compile inner model …", flush=True)
            self._inner = torch.compile(self._inner)
        self._warmup()
        if graph and self.merged and self.device == "xpu":
            self._setup_graph()


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
                     wp, body, preprocessed: bool = False, path=None):
        """``path``：沿路径前进的 k 个机体系点 [(dx,dz), …]；None 或空 = 用单点 prompt。

        ⚠️ 给 ``path`` 会改变 token 序列（插入 k 组数字），**偏离训练分布**。
        默认 None 时与训练逐字一致。
        """
        if not preprocessed:
            cur_rgb = to_policy_rgb(cur_rgb, self.size)
            prev_rgb = to_policy_rgb(prev_rgb, self.size)
        video = np.stack([np.asarray(prev_rgb, np.uint8), np.asarray(cur_rgb, np.uint8)])
        if path:
            pts = " ".join("%.2f %.2f" % (float(a), float(b)) for a, b in path)
            text = PROMPT_PATH % (float(body[0]), float(body[1]),
                                  float(wp[0]), float(wp[1]), pts)
        else:
            text = PROMPT % (float(body[0]), float(body[1]), float(wp[0]), float(wp[1]))
        tmpl = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)
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

    def _forward(self, b):
        if self.dual is not None and self.dual.ready:
            h = self.dual.forward(b)
            if h is not None:
                kw = {k: v.to(self.device) for k, v in b.items()}
                idx = kw["attention_mask"].sum(-1) - 1
                pooled = h[torch.arange(h.shape[0], device=h.device), idx]
                return self.head(pooled.float()).view(-1, K_STEPS, 2)
        kw = {k: v.to(self.device) for k, v in b.items()}
        out = self._inner(**kw, use_cache=False)
        h = out.last_hidden_state
        idx = kw["attention_mask"].sum(-1) - 1
        pooled = h[torch.arange(h.shape[0], device=h.device), idx]
        return self.head(pooled.float()).view(-1, K_STEPS, 2)

    # ------------------------------------------------------------ 对外
    def act(self, cur_rgb: np.ndarray, prev_rgb: np.ndarray, wp, body,
            preprocessed: bool = False, path=None) -> dict:
        t0 = time.perf_counter()
        b = self.build_inputs(cur_rgb, prev_rgb, wp, body, preprocessed=preprocessed,
                              path=path)
        with torch.no_grad():
            pred = self._forward(b)
        if self.device == "xpu":
            torch.xpu.synchronize()
        block = pred[0].float().cpu().numpy()
        gated = float(np.hypot(wp[0], wp[1])) < self.wp_stop
        if gated:
            block = np.zeros_like(block)
        return dict(block=block.round(4).tolist(), gated=gated, obs_t=t0,
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
    ap.add_argument("--wp-stop", type=float, default=0.02)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return selftest(a)


if __name__ == "__main__":
    raise SystemExit(main())
