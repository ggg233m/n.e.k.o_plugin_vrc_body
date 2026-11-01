# -*- coding: utf-8 -*-
"""replay 期间 GPU 侧到底在算什么：kernel 数、总时长、按类别分。"""
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

res = {}
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

    from torch.profiler import profile, ProfilerActivity
    def prof_one(g, label):
        g.replay(); torch.xpu.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as p:
            g.replay(); torch.xpu.synchronize()
        ka = p.key_averages()
        tot = sum(e.self_device_time_total for e in ka) / 1000.0
        cnt = sum(e.count for e in ka)
        top = sorted([(e.key[:70], e.count, round(e.self_device_time_total/1000.0, 2))
                      for e in ka if e.self_device_time_total > 0],
                     key=lambda x: -x[2])[:8]
        return {"label": label, "device_total_ms": round(tot, 2), "kernel_count": cnt, "top": top}

    with torch.inference_mode():
        res["vit"] = prof_one(gvit, "vit")
        res["lm"] = prof_one(glm, "lm")

    # 也用分析器看融合情况
    res["lm_top"] = res["lm"]["top"]
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
except Exception as exc:
    print("FAIL %s: %s" % (type(exc).__name__, str(exc)[:300]), flush=True)
    import traceback; traceback.print_exc()
finally:
    restore_attn()