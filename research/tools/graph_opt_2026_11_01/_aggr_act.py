# -*- coding: utf-8 -*-
import torch, time, sys, numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
sys.path.insert(0, r'H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\research\tools')
from policy_runtime import to_policy_rgb, PROMPT
import torch._inductor.config as ic
ic.aggressive_fusion = True
ic.max_fusion_size = 256
CK  = r'G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3'
RUN = r'G:\vrc_recording\vrc_model\runs\m0_s3_448\best'
dev = torch.device('xpu'); DT = torch.bfloat16
SIZE=(448,256); MAXLEN=192
proc = AutoProcessor.from_pretrained(CK)
pad_id = int(getattr(proc.tokenizer,'pad_token_id',0) or 0)
rng = np.random.default_rng(0)
cur_raw  = rng.integers(0,255,(606,1077,3),dtype=np.uint8)
prev_raw = rng.integers(0,255,(606,1077,3),dtype=np.uint8)
base = AutoModelForImageTextToText.from_pretrained(CK, dtype=DT).to(dev).eval()
p = PeftModel.from_pretrained(base, RUN+r'\adapter', adapter_name='trained')
p.set_adapter('trained'); m = p.merge_and_unload(); m.eval()
H = m.config.text_config.hidden_size
run = torch.load(RUN + r'\head.pt', map_location=dev)
head = torch.nn.Linear(H, 24).to(dev, torch.float32)
head.load_state_dict({'weight': run['fc.weight'], 'bias': run['fc.bias']}); head.eval()
inner = torch.compile(m.model)
def to_pi(cr, pr, wp, body):
    c = to_policy_rgb(cr, SIZE); pv = to_policy_rgb(pr, SIZE)
    text = PROMPT % (float(body[0]), float(body[1]), float(wp[0]), float(wp[1]))
    tmpl = proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}], tokenize=False, add_generation_prompt=True)
    mp = SIZE[0]*SIZE[1]
    inp = proc(text=[tmpl], videos=[np.stack([pv, c])], return_tensors='pt', padding=True, min_pixels=mp, max_pixels=mp, cap_pixels_per_frame=False, do_sample_frames=False)
    ids = inp['input_ids'][0].tolist(); n=len(ids)
    ids = ids+[pad_id]*(MAXLEN-n); mask=[1]*n+[0]*(MAXLEN-n)
    out = dict(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]), pixel_values_videos=inp['pixel_values_videos'], video_grid_thw=inp['video_grid_thw'])
    if 'mm_token_type_ids' in inp:
        out['mm_token_type_ids']=torch.tensor([inp['mm_token_type_ids'][0].tolist()+[0]*(MAXLEN-n)])
    return {k:v.to(dev) for k,v in out.items()}
def act():
    kw = to_pi(cur_raw, prev_raw, (0.3,0.5), (0.0,0.0))
    o = inner(**kw, use_cache=False)
    idx = kw['attention_mask'].sum(-1)-1
    return head(o.last_hidden_state[torch.arange(1,device=dev), idx].float())
with torch.inference_mode():
    act(); torch.xpu.synchronize()
    tot=[]
    for _ in range(25):
        torch.xpu.synchronize(); t=time.perf_counter(); act(); torch.xpu.synchronize()
        tot.append((time.perf_counter()-t)*1e3)
    tot.sort()
    print('AGGRFUZE_ACT p50=%.1f p95=%.1f p99=%.1f' % (tot[len(tot)//2], tot[int(len(tot)*0.95)], tot[min(len(tot)-1,int(len(tot)*0.99))]))