# -*- coding: utf-8 -*-
"""定位：patch 之后还有哪些同步会触发 D2H。"""
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

proc = AutoProcessor.from_pretrained(CK)
pad_id = int(getattr(proc.tokenizer, "pad_token_id", 0) or 0)
model, head, _ = load_model(CK, 128, DEV)
model.load_adapter(RUN + r"\adapter", adapter_name="trained", is_trainable=False)
model.set_adapter("trained")
model = model.merge_and_unload()
model.eval(); head.eval()
inner = model.base_model
visual = inner.visual

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

attn_list = [blk.attn for blk in visual.blocks]
saved, restore_attn = pgr._single_window_attention(attn_list)
geom = pgr.precompute_vision_geometry(visual, b["video_grid_thw"])
split_sizes = pgr.vision_split_sizes(b["video_grid_thw"], visual.spatial_merge_size)
_orig_gif = type(inner).get_image_features
def _patched_gif(self, pixel_values, image_grid_thw=None, **kwargs):
    pixel_values = pixel_values.type(self.visual.dtype)
    vo = self.visual(pixel_values, grid_thw=image_grid_thw, **geom)
    vo.pooler_output = torch.split(vo.pooler_output, split_sizes)
    return vo
type(inner).get_image_features = _patched_gif

res = {}
try:
    with torch.inference_mode():
        inner(**b, use_cache=False)
        torch.xpu.synchronize()
    print("[1] forward ok", flush=True)

    # 只测 ViT 能不能捕获
    stream = torch.xpu.Stream(); stream.wait_stream(torch.xpu.current_stream())
    with torch.inference_mode(), torch.xpu.stream(stream):
        visual(b["pixel_values_videos"], grid_thw=b["video_grid_thw"], **geom)
    stream.synchronize()
    print("[2] visual alone ok", flush=True)
    try:
        g1 = torch.xpu.XPUGraph()
        with torch.inference_mode(), torch.xpu.graph(g1, stream=stream):
            visual(b["pixel_values_videos"], grid_thw=b["video_grid_thw"], **geom)
        torch.xpu.synchronize(); g1.replay(); torch.xpu.synchronize()
        res["vit_graph"] = {"ok": True}
        print("[3] ViT GRAPH OK", flush=True)
    except Exception as e:
        res["vit_graph"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
        print("[3] ViT GRAPH FAIL %s" % res["vit_graph"]["error"], flush=True)

    # 只测 LM 能不能捕获
    X = torch.zeros((1, MAXLEN, 1024), dtype=torch.bfloat16, device=DEV)
    AM = b["attention_mask"].clone()
    lm = inner.language_model
    try:
        g2 = torch.xpu.XPUGraph()
        with torch.inference_mode(), torch.xpu.graph(g2, stream=stream):
            lm(inputs_embeds=X, attention_mask=AM, use_cache=False)
        torch.xpu.synchronize(); g2.replay(); torch.xpu.synchronize()
        res["lm_graph"] = {"ok": True}
        print("[4] LM GRAPH OK", flush=True)
    except Exception as e:
        res["lm_graph"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
        print("[4] LM GRAPH FAIL %s" % res["lm_graph"]["error"], flush=True)
finally:
    restore_attn()
    type(inner).get_image_features = _orig_gif

print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
(ROOT / ".tmp" / "dryrun_20261101" / "graph_diag.json").write_text(
    json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")