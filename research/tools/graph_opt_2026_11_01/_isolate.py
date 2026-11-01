# -*- coding: utf-8 -*-
"""隔离测：ViT-only / LM-only 的 replay 时间，以及 LM 随 token 数怎么变。

目的：判断 77ms 是「计算量」还是「固定开销」。
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT / "research" / "tools"))
from policy_runtime import setup_oneapi
setup_oneapi(graph=True)

import torch, time
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

print("[cfg] layers=%d hidden=%d heads=%d" % (
    inner.config.text_config.num_hidden_layers if hasattr(inner.config, "text_config") else -1,
    inner.config.text_config.hidden_size if hasattr(inner.config, "text_config") else -1,
    inner.config.text_config.num_attention_heads if hasattr(inner.config, "text_config") else -1), flush=True)

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

def bench_replay(g, n=200):
    ts = []
    for _ in range(n):
        torch.xpu.synchronize(); t = time.perf_counter()
        g.replay(); torch.xpu.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return dict(p50=round(ts[len(ts)//2], 2), p10=round(ts[int(len(ts)*0.1)], 2), min=round(ts[0], 2))

res = {}
try:
    stream = torch.xpu.Stream(); stream.wait_stream(torch.xpu.current_stream())

    # ---- ViT 图 ----
    pv = b["pixel_values_videos"].clone()
    with torch.inference_mode(), torch.xpu.stream(stream):
        for _ in range(3): visual(pv, grid_thw=b["video_grid_thw"], **geom)
    stream.synchronize()
    gvit = torch.xpu.XPUGraph()
    with torch.inference_mode(), torch.xpu.graph(gvit, stream=stream):
        vit_out = visual(pv, grid_thw=b["video_grid_thw"], **geom)
    torch.xpu.synchronize(); gvit.replay(); torch.xpu.synchronize()
    res["vit_only"] = bench_replay(gvit)
    print("[1] ViT-only: %s" % res["vit_only"], flush=True)

    # ---- glue 成本 ----
    ve = torch.split(vit_out.pooler_output, list(split_sizes))
    ie = inner.get_input_embeddings()(b["input_ids"])
    _, vmask = inner.get_placeholder_mask(b["input_ids"], inputs_embeds=ie, video_features=ve[0])
    emb = ie.masked_scatter(vmask, ve[0])
    with torch.inference_mode():
        pos = inner.compute_3d_position_ids(input_ids=b["input_ids"], inputs_embeds=emb,
            video_grid_thw=b["video_grid_thw"], attention_mask=b["attention_mask"],
            mm_token_type_ids=b["mm_token_type_ids"])

    def glue():
        v = torch.split(vit_out.pooler_output, list(split_sizes))
        e = inner.get_input_embeddings()(b["input_ids"])
        _, m = inner.get_placeholder_mask(b["input_ids"], inputs_embeds=e, video_features=v[0])
        x = e.masked_scatter(m, v[0])
        p = inner.compute_3d_position_ids(input_ids=b["input_ids"], inputs_embeds=x,
            video_grid_thw=b["video_grid_thw"], attention_mask=b["attention_mask"],
            mm_token_type_ids=b["mm_token_type_ids"])
        return x, p
    gts = []
    for _ in range(50):
        torch.xpu.synchronize(); t = time.perf_counter()
        glue(); torch.xpu.synchronize()
        gts.append((time.perf_counter() - t) * 1e3)
    gts.sort()
    res["glue"] = dict(p50=round(gts[len(gts)//2], 2), min=round(gts[0], 2))
    print("[2] glue: %s" % res["glue"], flush=True)

    # ---- LM 图（真实 192 token）----
    E = emb.detach().clone(); AM = b["attention_mask"].clone(); P = pos.detach().clone()
    with torch.inference_mode(), torch.xpu.stream(stream):
        for _ in range(3): lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
    stream.synchronize()
    glm = torch.xpu.XPUGraph()
    with torch.inference_mode(), torch.xpu.graph(glm, stream=stream):
        lm_out = lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
    torch.xpu.synchronize(); glm.replay(); torch.xpu.synchronize()
    res["lm_192"] = bench_replay(glm)
    print("[3] LM 192tok: %s" % res["lm_192"], flush=True)

    # ---- LM 随 token 数变化（判断计算瓶颈 vs 固定开销）----
    for n in (48, 96, 192):
        e2 = torch.zeros((1, n, E.shape[-1]), dtype=E.dtype, device=DEV)
        a2 = torch.ones((1, n), dtype=AM.dtype, device=DEV)
        p2 = torch.zeros((3, 1, n), dtype=P.dtype, device=DEV)
        with torch.inference_mode(), torch.xpu.stream(stream):
            for _ in range(3): lm(inputs_embeds=e2, attention_mask=a2, position_ids=p2, use_cache=False)
        stream.synchronize()
        gg = torch.xpu.XPUGraph()
        with torch.inference_mode(), torch.xpu.graph(gg, stream=stream):
            lm(inputs_embeds=e2, attention_mask=a2, position_ids=p2, use_cache=False)
        torch.xpu.synchronize(); gg.replay(); torch.xpu.synchronize()
        res["lm_%d" % n] = bench_replay(gg, n=100)
        print("[4] LM %dtok: %s" % (n, res["lm_%d" % n]), flush=True)

    res["ok"] = True
except Exception as exc:
    res["ok"] = False
    res["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
    import traceback; traceback.print_exc()
finally:
    restore_attn()

print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)