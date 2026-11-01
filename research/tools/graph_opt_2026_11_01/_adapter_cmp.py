# -*- coding: utf-8 -*-
"""V1 vs V2 适配器下图重放的耗时对比（同一模型、同一输入）。"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT / "research" / "tools"))
TAG = sys.argv[1]

from policy_runtime import setup_oneapi
setup_oneapi(graph=True)
print("[env] SYCL_UR_USE_LEVEL_ZERO_V2=%s" % os.environ.get("SYCL_UR_USE_LEVEL_ZERO_V2"), flush=True)

import torch
print("[dev] %s" % torch.xpu.get_device_name(0), flush=True)
try:
    print("[dev] backend=%s" % torch.xpu.get_device_properties(0).driver_version, flush=True)
except Exception:
    pass

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
lm = inner.language_model

rng = np.random.default_rng(0)
cur = rng.integers(0, 255, (606, 1077, 3), dtype=np.uint8)

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

b = build(cur, cur, (0.0, 0.8), (0.0, 0.0))
attn_list = [blk.attn for blk in visual.blocks]
saved, restore_attn = pgr._single_window_attention(attn_list)
geom = pgr.precompute_vision_geometry(visual, b["video_grid_thw"])
split_sizes = pgr.vision_split_sizes(b["video_grid_thw"], visual.spatial_merge_size)

res = {"tag": TAG}
try:
    stream = torch.xpu.Stream(); stream.wait_stream(torch.xpu.current_stream())
    pv = b["pixel_values_videos"].clone()
    with torch.inference_mode(), torch.xpu.stream(stream):
        for _ in range(3):
            visual(pv, grid_thw=b["video_grid_thw"], **geom)
    stream.synchronize()
    gvit = torch.xpu.XPUGraph()
    with torch.inference_mode(), torch.xpu.graph(gvit, stream=stream):
        vit_out = visual(pv, grid_thw=b["video_grid_thw"], **geom)
    torch.xpu.synchronize(); gvit.replay(); torch.xpu.synchronize()

    ts = []
    for _ in range(30):
        torch.xpu.synchronize(); t = time.perf_counter()
        gvit.replay(); torch.xpu.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    res["vit_ms"] = dict(p50=round(ts[len(ts)//2], 2), min=round(ts[0], 2))

    # LM
    ve = torch.split(vit_out.pooler_output, list(split_sizes))
    ie = inner.get_input_embeddings()(b["input_ids"])
    _, vmask = inner.get_placeholder_mask(b["input_ids"], inputs_embeds=ie, video_features=ve[0])
    emb = ie.masked_scatter(vmask, ve[0])
    with torch.inference_mode():
        pos = inner.compute_3d_position_ids(input_ids=b["input_ids"], inputs_embeds=emb,
            video_grid_thw=b["video_grid_thw"], attention_mask=b["attention_mask"],
            mm_token_type_ids=b["mm_token_type_ids"])
    E = emb.detach().clone(); AM = b["attention_mask"].clone(); P = pos.detach().clone()
    with torch.inference_mode(), torch.xpu.stream(stream):
        for _ in range(3):
            lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
    stream.synchronize()
    glm = torch.xpu.XPUGraph()
    with torch.inference_mode(), torch.xpu.graph(glm, stream=stream):
        lm_out = lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
    torch.xpu.synchronize(); glm.replay(); torch.xpu.synchronize()
    ts = []
    for _ in range(30):
        torch.xpu.synchronize(); t = time.perf_counter()
        glm.replay(); torch.xpu.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    res["lm_ms"] = dict(p50=round(ts[len(ts)//2], 2), min=round(ts[0], 2))
    res["ok"] = True
except Exception as exc:
    res["ok"] = False
    res["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
finally:
    restore_attn()

print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
(ROOT / ".tmp" / "dryrun_20261101" / ("adapter_" + TAG + ".json")).write_text(
    json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")