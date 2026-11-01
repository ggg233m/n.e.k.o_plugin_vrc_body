# -*- coding: utf-8 -*-
"""双图串联可行性：ViT 图 + LM 图，中间 glue 在 CPU 上做。"""
import json, sys, time
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

res = {}
try:
    # ---------- 参考 ----------
    with torch.inference_mode():
        ref = inner(**b, use_cache=False)
        torch.xpu.synchronize()
    ref_last = ref.last_hidden_state.clone()
    print("[1] ref ok", flush=True)

    # ---------- ViT 图 ----------
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
    t0 = time.perf_counter()
    gvit.replay(); torch.xpu.synchronize()
    vit_ms = (time.perf_counter() - t0) * 1e3
    res["vit_replay_ms"] = round(vit_ms, 2)
    print("[2] ViT graph ok, replay=%.2f ms" % vit_ms, flush=True)

    # ---------- glue：把 vision 特征拼进 embeds ----------
    def glue_from(vit_hidden):
        ve = torch.split(vit_hidden, list(split_sizes))
        inputs_embeds = inner.get_input_embeddings()(b["input_ids"])
        _, vmask = inner.get_placeholder_mask(b["input_ids"], inputs_embeds=inputs_embeds, video_features=ve[0])
        return inputs_embeds.masked_scatter(vmask, ve[0])

    emb = glue_from(vit_out.pooler_output)
    print("[3] glue ok, embeds=%s" % (tuple(emb.shape),), flush=True)

    # position_ids 也要图外算
    with torch.inference_mode():
        pos = inner.compute_3d_position_ids(
            input_ids=b["input_ids"], inputs_embeds=emb,
            video_grid_thw=b["video_grid_thw"], attention_mask=b["attention_mask"],
            mm_token_type_ids=b["mm_token_type_ids"])
    print("[4] position_ids=%s" % (tuple(pos.shape) if pos is not None else None,), flush=True)

    # ---------- LM 图 ----------
    E = emb.detach().clone()
    AM = b["attention_mask"].clone()
    P  = pos.detach().clone() if pos is not None else None
    with torch.inference_mode(), torch.xpu.stream(stream):
        for _ in range(3):
            lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
    stream.synchronize()
    glm = torch.xpu.XPUGraph()
    with torch.inference_mode(), torch.xpu.graph(glm, stream=stream):
        lm_out = lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
    torch.xpu.synchronize(); glm.replay(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    glm.replay(); torch.xpu.synchronize()
    lm_ms = (time.perf_counter() - t0) * 1e3
    res["lm_replay_ms"] = round(lm_ms, 2)
    print("[5] LM graph ok, replay=%.2f ms" % lm_ms, flush=True)

    # ---------- 正确性 ----------
    cos = torch.nn.functional.cosine_similarity(
        lm_out.last_hidden_state.flatten().float(), ref_last.flatten().float(), dim=0).item()
    res["cos_vs_eager"] = round(cos, 8)
    print("[6] cos=%.8f" % cos, flush=True)

    # 端到端计时（两次 replay + glue）
    def full():
        gvit.replay()
        e = glue_from(vit_out.pooler_output)
        E.copy_(e)
        glm.replay()
        return lm_out.last_hidden_state
    ts = []
    for _ in range(20):
        torch.xpu.synchronize(); t = time.perf_counter()
        full(); torch.xpu.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    res["dual_graph_ms"] = dict(p50=round(ts[len(ts)//2], 1), p95=round(ts[int(len(ts)*0.95)], 1))
    print("[7] dual graph p50=%.1f ms" % ts[len(ts)//2], flush=True)

    res["ok"] = True
except Exception as exc:
    res["ok"] = False
    res["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:400])
    import traceback; traceback.print_exc()
finally:
    restore_attn()

print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
(ROOT / ".tmp" / "dryrun_20261101" / "dual_graph.json").write_text(
    json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")