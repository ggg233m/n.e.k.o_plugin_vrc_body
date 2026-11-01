# -*- coding: utf-8 -*-
"""实验 A：整模型单图（几何量走闭包，不污染 kwargs）。"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT / "research" / "tools"))
from policy_runtime import setup_oneapi
setup_oneapi(graph=True)

import torch
from transformers import AutoProcessor
from train_policy_lora import load_model
from policy_runtime import to_policy_rgb, PROMPT
import policy_graph_runtime as pgr

CK  = r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3"
RUN = r"G:\vrc_recording\vrc_model\runs\m0_s3_448\best"
DEV = "xpu"; SIZE=(448,256); MAXLEN=192

print("[1] loading", flush=True)
proc = AutoProcessor.from_pretrained(CK)
pad_id = int(getattr(proc.tokenizer, "pad_token_id", 0) or 0)
model, head, _ = load_model(CK, 128, DEV)
model.load_adapter(RUN + r"\adapter", adapter_name="trained", is_trainable=False)
model.set_adapter("trained")
model = model.merge_and_unload()
model.eval(); head.eval()
inner = model.base_model
visual = inner.visual
print("[2] ready", flush=True)

rng = np.random.default_rng(0)
cur = rng.integers(0, 255, (606, 1077, 3), dtype=np.uint8)
prev = rng.integers(0, 255, (606, 1077, 3), dtype=np.uint8)

def build(c, p, wp, body):
    c2 = to_policy_rgb(c, SIZE); p2 = to_policy_rgb(p, SIZE)
    text = PROMPT % (float(body[0]), float(body[1]), float(wp[0]), float(wp[1]))
    tmpl = proc.apply_chat_template([{"role":"user","content":[{"type":"video"},{"type":"text","text":text}]}],
                                    tokenize=False, add_generation_prompt=True)
    mp = SIZE[0]*SIZE[1]
    inp = proc(text=[tmpl], videos=[np.stack([p2, c2])], return_tensors="pt", padding=True,
               min_pixels=mp, max_pixels=mp, cap_pixels_per_frame=False, do_sample_frames=False)
    ids = inp["input_ids"][0].tolist(); n = len(ids)
    ids = ids + [pad_id]*(MAXLEN-n); mask = [1]*n + [0]*(MAXLEN-n)
    out = dict(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]),
               pixel_values_videos=inp["pixel_values_videos"], video_grid_thw=inp["video_grid_thw"])
    if "mm_token_type_ids" in inp:
        out["mm_token_type_ids"] = torch.tensor([inp["mm_token_type_ids"][0].tolist()+[0]*(MAXLEN-n)])
    return {k: v.to(DEV) for k, v in out.items()}

b = build(cur, prev, (0.0, 0.8), (0.0, 0.0))
res = {"grid_thw": b["video_grid_thw"].tolist(), "ntok": int(b["attention_mask"].sum())}

with torch.inference_mode():
    ref = inner(**b, use_cache=False).last_hidden_state.clone()
    torch.xpu.synchronize()
print("[3] ref ok", flush=True)

# ---------- 打补丁：几何量走闭包 ----------
attn_list = [blk.attn for blk in visual.blocks]
saved, restore_attn = pgr._single_window_attention(attn_list)
geom = pgr.precompute_vision_geometry(visual, b["video_grid_thw"])
split_sizes = pgr.vision_split_sizes(b["video_grid_thw"], visual.spatial_merge_size)
print("[4] geom ready, split_sizes=%s" % (split_sizes,), flush=True)

_orig_gif = type(inner).get_image_features
def _patched_gif(self, pixel_values, image_grid_thw=None, **kwargs):
    pixel_values = pixel_values.type(self.visual.dtype)
    vo = self.visual(pixel_values, grid_thw=image_grid_thw, **geom)
    vo.pooler_output = torch.split(vo.pooler_output, split_sizes)
    return vo
type(inner).get_image_features = _patched_gif

out = dict(res)
try:
    stream = torch.xpu.Stream(); stream.wait_stream(torch.xpu.current_stream())
    with torch.inference_mode(), torch.xpu.stream(stream):
        for _ in range(3):
            inner(**b, use_cache=False)
    stream.synchronize()
    print("[A] warmup ok", flush=True)
    g = torch.xpu.XPUGraph()
    with torch.inference_mode(), torch.xpu.graph(g, stream=stream):
        static = inner(**b, use_cache=False)
    torch.xpu.synchronize()
    g.replay(); torch.xpu.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        static.last_hidden_state.flatten().float(), ref.flatten().float(), dim=0).item()
    out["A_whole_graph"] = {"ok": True, "cos_vs_eager": round(cos, 8)}
    print("[A] CAPTURE OK cos=%.8f" % cos, flush=True)
except Exception as exc:
    out["A_whole_graph"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
    print("[A] FAIL %s" % out["A_whole_graph"]["error"], flush=True)
finally:
    restore_attn()
    type(inner).get_image_features = _orig_gif

print("RESULT " + json.dumps(out, ensure_ascii=False), flush=True)
(ROOT / ".tmp" / "dryrun_20261101" / "graph_expA.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")