# SPDX-License-Identifier: Apache-2.0
"""Core8 XPU FP32 整步编译。条件仅在单个窗口内保留，不修改全局类。"""
from typing import List, Optional, Tuple, Union
from threading import RLock
from types import MethodType
from tqdm.auto import tqdm
from ardy.model.ardy_model import get_three_mask_from_len, translate_normalized_root_motion
from ardy.model.diffusion import DDIMSampler
from ardy.model.backbone import PositionalEncodingNegativeIndex

"""试验用窗口条件预计算；所有缓存只在当前窗口有效。"""
import torch
import torch.nn.functional as F
from ardy.model.backbone import pad_x_and_mask_to_fixed_size

def prepare_backbone(owner,text,text_mask,heading,index,pad_mask):
    if owner.num_text_tokens is not None:
        text,text_mask=pad_x_and_mask_to_fixed_size(text,text_mask,owner.num_text_tokens)
    embedded=owner.embed_text(text.to(owner.embed_text.weight.dtype))
    if not owner.use_text_mask:text_mask=torch.ones_like(text_mask,dtype=torch.bool)
    masks=[text_mask,torch.ones((text.shape[0],1),device=text.device,dtype=torch.bool)]
    angle=None
    if owner.input_first_heading_angle:
        angle=owner.linear_first_heading_angle(torch.stack([torch.cos(heading),torch.sin(heading)],-1))[:,None]
        masks.append(torch.ones((text.shape[0],1),device=text.device,dtype=torch.bool))
    prefix_len=embedded.shape[1]+1+int(angle is not None)
    assert prefix_len<=owner.learned_prefix_embedding.max_len
    prefix_pe=owner.learned_prefix_embedding.embedding(torch.arange(prefix_len,device=text.device,dtype=torch.int32)[None])
    pe=owner.motion_token_embedding
    safe_index=torch.where(index>=0,index,index+pe.pe.shape[0])
    motion_pe=F.embedding(safe_index.int(),pe.pe)
    padding=~torch.cat([*masks,pad_mask],dim=1)
    mask=torch.zeros((text.shape[0],1,1,padding.shape[1]),device=text.device,dtype=embedded.dtype).masked_fill(padding[:,None,None,:],float('-inf'))
    return embedded,angle,prefix_len,prefix_pe,motion_pe,mask

def prepare_window(model,x,history_len,generation_len,future_len,history_mask,generation_mask,future_mask,history_token_mask,generation_token_mask,future_token_mask,text_feat,text_pad_mask,first_heading_angle,cur_motion_mask,cur_observed_motion,unconstrained):
    cfg=model.denoiser
    two=model._pipeline_two_pass and unconstrained and cur_motion_mask is None and cur_observed_motion is None
    count=2 if two else 3
    repeat=lambda v:torch.cat([v]*count,dim=0)
    inputs=dict(history_len=history_len,generation_len=generation_len,future_len=future_len,
                history_mask=history_mask,generation_mask=generation_mask,future_mask=future_mask,
                history_token_mask=history_token_mask,generation_token_mask=generation_token_mask)
    static={name:repeat(inputs[name]) for name in ('history_len','generation_len','future_len','history_mask','generation_mask','future_mask','history_token_mask','generation_token_mask')}
    static['future_token_mask']=torch.cat([0*future_token_mask,0*future_token_mask] if two else [0*future_token_mask,future_token_mask,0*future_token_mask])
    static['text_feat']=torch.cat([text_feat,0*text_feat] if two else [text_feat,0*text_feat,0*text_feat])
    static['text_feat_pad_mask']=torch.cat([text_pad_mask,0*text_pad_mask] if two else [text_pad_mask,0*text_pad_mask,0*text_pad_mask])
    static['first_heading_angle']=repeat(first_heading_angle) if first_heading_angle is not None else None
    for name,value in (('motion_mask',cur_motion_mask),('observed_motion',cur_observed_motion)):
        static[name]=torch.cat([0*value,value,0*value]) if value is not None else None
    for name in ('history_mask','generation_mask','future_mask','history_token_mask','generation_token_mask','future_token_mask','text_feat_pad_mask'):
        static[name]=static[name]>0.5
    inner=cfg.model
    index=torch.arange(x.shape[1],device=x.device)[None].expand(count*x.shape[0],-1)-static['history_len'][:,None]//inner.num_frames_per_token
    pad=static['history_token_mask']|static['generation_token_mask']
    if inner.motion_mask_mode=='concat':pad=pad|static['future_token_mask']
    cfg._conditions=(count,static)
    for owner in (inner.root_model,inner.body_model):
        owner._conditions=prepare_backbone(owner,static['text_feat'],static['text_feat_pad_mask'],static['first_heading_angle'],index,pad)

def cached_cfg(self,cfg_weight_text,cfg_weight_cstr,x,history_len,generation_len,future_len,history_mask,generation_mask,future_mask,history_token_mask,generation_token_mask,future_token_mask,text_feat,text_feat_pad_mask,timesteps,first_heading_angle,motion_mask,observed_motion):
    count,static=self._conditions
    out=self.model(x=torch.cat([x]*count),timesteps=torch.cat([timesteps]*count),**static)
    if count==2:
        text,unconditional=out.chunk(2)
        return unconditional+cfg_weight_text*(text-unconditional)
    text,constraint,unconditional=out.chunk(3)
    return unconditional+cfg_weight_text*(text-unconditional)+cfg_weight_cstr*(constraint-unconditional)

def cached_backbone(self,x,x_pad_mask,text_feat,text_feat_pad_mask,timesteps,first_heading_angle=None,token_index=None):
    text,heading,prefix_len,prefix_pe,motion_pe,mask=self._conditions
    time=self.embed_timestep(timesteps)
    prefix=torch.cat([text,time] if heading is None else [text,time,heading],dim=1)
    prefix=self.learned_prefix_embedding.dropout(prefix+prefix_pe)
    motion=self.motion_token_embedding.dropout(self.input_linear(x)+motion_pe)
    value=torch.cat([prefix,motion],dim=1)
    for layer in self.seqTransEncoder.layers:
        if layer.norm_first:
            value=value+layer._sa_block(layer.norm1(value),mask)
            value=value+layer._ff_block(layer.norm2(value))
        else:
            value=layer.norm1(value+layer._sa_block(value,mask))
            value=layer.norm2(value+layer._ff_block(value))
    if self.seqTransEncoder.norm is not None:value=self.seqTransEncoder.norm(value)
    return self.output_linear(value[:,prefix_len:])

def denoising_step(self, x: torch.Tensor, history_len: torch.Tensor, generation_len: torch.Tensor, future_len: torch.Tensor, history_mask: torch.Tensor, generation_mask: torch.Tensor, future_mask: torch.Tensor, history_token_mask: torch.Tensor, generation_token_mask: torch.Tensor, future_token_mask: torch.Tensor, text_feat: torch.Tensor, text_pad_mask: torch.Tensor, t: torch.Tensor, first_heading_angle: Optional[torch.Tensor], motion_mask: torch.Tensor, observed_motion: torch.Tensor, num_denoising_steps: torch.Tensor, cfg_weight: Union[float, Tuple[float, float]], target_motion: Optional[torch.Tensor]=None, cfg_type: Optional[str]=None, prepared_sampling: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None, generation_token_range: Optional[Tuple[int, int]]=None) -> torch.Tensor:
    if prepared_sampling is None:
        num_denoising_steps = num_denoising_steps[0]
        use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps)
        self.diffusion.calc_diffusion_vars(use_timesteps)
        t_map = map_tensor[t]
        if isinstance(cfg_weight, (tuple, list)):
            w_text, w_cstr = cfg_weight
        else:
            w_text, w_cstr = (cfg_weight, 0.0)
        cfg_weight_text = torch.tensor([w_text], device=x.device, dtype=torch.float32)
        cfg_weight_cstr = torch.tensor([w_cstr], device=x.device, dtype=torch.float32)
    else:
        map_tensor, cfg_weight_text, cfg_weight_cstr = prepared_sampling
        t_map = map_tensor[t]
    with torch.no_grad():
        token_seq_pred_clean = self.denoiser(cfg_weight_text, cfg_weight_cstr, x, history_len, generation_len, future_len, history_mask, generation_mask, future_mask, history_token_mask, generation_token_mask, future_token_mask, text_feat, text_pad_mask, t_map, first_heading_angle, motion_mask, observed_motion)
    batch_size, num_token_dim = (x.shape[0], x.shape[2])
    num_generation_tokens = self.gen_horizon_len // self.num_frames_per_token
    if generation_token_range is None:
        generation_token_t = x[generation_token_mask].reshape(batch_size, num_generation_tokens, num_token_dim)
        generation_token_clean = token_seq_pred_clean[generation_token_mask].reshape(batch_size, num_generation_tokens, num_token_dim)
    else:
        start, end = generation_token_range
        generation_token_t = x[:, start:end].contiguous()
        generation_token_clean = token_seq_pred_clean[:, start:end].contiguous()
    generation_token_tm1 = self.sampler(generation_token_t, generation_token_clean, t)
    xm1 = x.clone()
    if generation_token_range is None:
        xm1[generation_token_mask] = generation_token_tm1.reshape(-1, num_token_dim)
    else:
        xm1[:, start:end] = generation_token_tm1
    return xm1

def _generate_window(self, history_sequence: Optional[torch.Tensor], global_transl: torch.Tensor, history_start_frame: int, history_end_frame: int, total_frames: int, text_feat: torch.Tensor, text_pad_mask: torch.Tensor, first_heading_angle: Optional[torch.Tensor], motion_mask: Optional[torch.Tensor], observed_motion: Optional[torch.Tensor], num_denoising_steps: torch.Tensor, cfg_weight: Union[float, Tuple[float, float]], indices: List[int], progress_bar=tqdm, target_motion: Optional[torch.Tensor]=None, cfg_type: Optional[str]=None) -> torch.Tensor:
    device = self.device
    batch_size = text_feat.shape[0]
    num_frames_per_token = self.num_frames_per_token
    checked_bounds((total_frames - history_start_frame) // num_frames_per_token, (history_end_frame - history_start_frame) // num_frames_per_token, self._pipeline_index_limits)
    latent_embedding_dim = self.denoiser.latent_embedding_dim
    nframe_root_dim = self.denoiser.nframe_root_dim
    gen_horizon_len = self.gen_horizon_len
    num_generation_tokens = gen_horizon_len // num_frames_per_token
    generation_len = torch.ones(batch_size, device=device, dtype=torch.long) * gen_horizon_len
    history_token_end = (history_end_frame - history_start_frame) // num_frames_per_token
    generation_token_end = history_token_end + num_generation_tokens
    history_len = torch.ones(batch_size, device=device, dtype=torch.long) * history_token_end * num_frames_per_token
    future_len = torch.ones(batch_size, device=device, dtype=torch.long) * (total_frames - gen_horizon_len - history_end_frame)
    history_mask, generation_mask, future_mask = get_three_mask_from_len(history_len, generation_len, future_len, total_frames - history_start_frame, device)
    if motion_mask is not None:
        cur_motion_mask = motion_mask[:, history_start_frame:] * ~history_mask[:, :, None]
        cur_observed_root_motion = self.motion_rep.extract_root(observed_motion[:, history_start_frame:])
        cur_observed_body_motion = self.motion_rep.extract_body(observed_motion[:, history_start_frame:])
        translated_observed_root_motion = translate_normalized_root_motion(cur_observed_root_motion, -global_transl, self.motion_rep)
        cur_observed_motion = self.motion_rep.concat_root_body(translated_observed_root_motion, cur_observed_body_motion)
        cur_observed_motion = cur_observed_motion * cur_motion_mask
    else:
        cur_motion_mask = None
        cur_observed_motion = None
    history_token_mask, generation_token_mask, future_token_mask = self.hybrid.convert_frame_mask_to_token_mask(history_mask, generation_mask, future_mask, cur_motion_mask)
    shape = (batch_size, num_generation_tokens, nframe_root_dim + latent_embedding_dim)
    x_t = torch.randn(shape, device=device)
    x = torch.zeros(batch_size, (total_frames - history_start_frame) // num_frames_per_token, nframe_root_dim + latent_embedding_dim, device=device)
    if history_token_end > 0:
        x[:, :history_token_end] = history_sequence[:, -history_token_end:]
    x[:, history_token_end:generation_token_end] = x_t
    prepared_sampling = None
    generation_token_range = None
    if torch.device(device).type == 'xpu' and getattr(self, 'xpu_eager_optimizations', True):
        use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps[0])
        self.diffusion.calc_diffusion_vars(use_timesteps)
        w_text, w_cstr = cfg_weight if isinstance(cfg_weight, (tuple, list)) else (cfg_weight, 0.0)
        weights = torch.tensor([w_text, w_cstr], device=device, dtype=torch.float32)
        prepared_sampling = (map_tensor, weights[:1], weights[1:])
        generation_token_range = (history_token_end, generation_token_end)
    from ardy.model.eager_window import sampling_window
    unconstrained = cur_motion_mask is None and total_frames == history_end_frame + gen_horizon_len
    # 仅补齐带约束窗口；新增位置在全部有效位置掩码中均为 False。
    if not unconstrained:
        tokens = x.shape[1]
        padded_tokens = ((tokens + 15) // 16) * 16
        # 原窗口有效时，补齐不能越过位置编码的上界。
        padded_tokens = min(padded_tokens, min(self._pipeline_index_limits) + history_token_end)
        checked_bounds(padded_tokens, history_token_end, self._pipeline_index_limits)
        extra_tokens = padded_tokens - tokens
        extra_frames = extra_tokens * num_frames_per_token
        if extra_tokens:
            x = F.pad(x, (0, 0, 0, extra_tokens))
            history_mask = F.pad(history_mask, (0, extra_frames), value=False)
            generation_mask = F.pad(generation_mask, (0, extra_frames), value=False)
            future_mask = F.pad(future_mask, (0, extra_frames), value=False)
            history_token_mask = F.pad(history_token_mask, (0, extra_tokens), value=False)
            generation_token_mask = F.pad(generation_token_mask, (0, extra_tokens), value=False)
            future_token_mask = F.pad(future_token_mask, (0, extra_tokens), value=False)
            if cur_motion_mask is not None:
                cur_motion_mask = F.pad(cur_motion_mask, (0, 0, 0, extra_frames))
                cur_observed_motion = F.pad(cur_observed_motion, (0, 0, 0, extra_frames))
    self._pipeline_prepare(self, x, history_len, generation_len, future_len, history_mask, generation_mask, future_mask, history_token_mask, generation_token_mask, future_token_mask, text_feat, text_pad_mask, first_heading_angle, cur_motion_mask, cur_observed_motion, unconstrained)
    with sampling_window(self, unconstrained):
        for i in progress_bar(indices):
            t = torch.tensor([i] * x_t.size(0), device=device)
            with torch.no_grad():
                x = self.denoising_step(x, history_len, generation_len, future_len, history_mask, generation_mask, future_mask, history_token_mask, generation_token_mask, future_token_mask, text_feat, text_pad_mask, t, first_heading_angle, cur_motion_mask, cur_observed_motion, num_denoising_steps, cfg_weight, target_motion, cfg_type=cfg_type, prepared_sampling=prepared_sampling, generation_token_range=generation_token_range)
    if history_sequence is None:
        history_sequence = x[:, :generation_token_end]
    else:
        history_sequence = torch.cat([history_sequence, x[:, history_token_end:generation_token_end]], dim=1)
    return history_sequence

def checked_bounds(total_tokens, history_tokens, limits):
    # 非稀疏窗口的索引为连续整数，检查两端等价于逐元素检查。
    if total_tokens <= 0 or not 0 <= history_tokens <= total_tokens:
        raise ValueError('无效生成窗口')
    if any(max(history_tokens, abs(total_tokens-1-history_tokens)) >= n for n in limits):
        raise ValueError('位置编码索引越界')

class FinalStepSafeSampler(DDIMSampler):
    def __call__(self, x_t, pred_xstart, t):
        # t=0 数学上严格等于干净预测，避开融合编译时的零点异常。
        value = super().__call__(x_t, pred_xstart, t)
        return torch.where(t[:, None, None] == 0, pred_xstart, value)

def enable_core8_pipeline(model):
    if getattr(model, '_core8_pipeline_enabled', False):
        return True
    cfg = getattr(model.denoiser, '_orig_mod', model.denoiser)
    inner = getattr(cfg, 'model', None)
    if (model.gen_horizon_len != 8 or torch.device(model.device).type != 'xpu'
            or cfg.training or next(model.parameters()).dtype != torch.float32
            or type(cfg).__name__ != 'AutoLatentClassifierFreeGuidedModelSeparated'
            or not getattr(inner, 'trt_compatible', False)
            or getattr(inner, 'sparsify_token_seq', True)
            or inner.motion_mask_mode != 'concat' or type(model.sampler) != DDIMSampler):
        return False
    backbones = (inner.root_model, inner.body_model)
    if any(b.positional_encoding_mode != 'learned_prefix_zero_at_first_generation' or b.training for b in backbones):
        return False
    limits = [m.max_len for m in inner.modules() if isinstance(m, PositionalEncodingNegativeIndex)]
    if not limits:
        return False
    # 只修改本模型实例。保留全部原方法，关闭时原样恢复。
    saved = (model.denoiser, model.sampler, model.denoising_step, model._generate_window,
             model.autoregressive_step, cfg.forward, tuple(b.forward for b in backbones),
             getattr(model, 'xpu_window_cache_enabled', True))
    lock = RLock()
    def clear_conditions():
        for owner in (cfg, *backbones):
            if hasattr(owner, '_conditions'):
                delattr(owner, '_conditions')
    def window_call(*args, **kwargs):
        try:
            return _generate_window(model, *args, **kwargs)
        finally:
            clear_conditions()
    def autoregressive_call(*args, **kwargs):
        # 大图禁止嵌套 inference_mode；同时序列化同一实例的条件和扩散缓冲更新。
        with lock, torch.inference_mode(False), torch.no_grad():
            result = saved[4](*args, **kwargs)
            model._core8_pipeline_calls += 1
            return result
    def disable():
        with lock:
            clear_conditions()
            model.denoiser, model.sampler, model.denoising_step, model._generate_window, model.autoregressive_step = saved[:5]
            cfg.forward = saved[5]
            for b, original in zip(backbones, saved[6]):
                b.forward = original
            model.xpu_window_cache_enabled = saved[7]
            model._core8_pipeline_enabled = False
            model._pipeline_prepare = None
    model.denoiser = cfg
    model.sampler = FinalStepSafeSampler(model.diffusion)
    cfg.forward = MethodType(cached_cfg, cfg)
    for b in backbones:
        b.forward = MethodType(cached_backbone, b)
    model._pipeline_index_limits = limits
    model._pipeline_two_pass = True
    model.xpu_window_cache_enabled = False
    compiled_prepare = torch.compile(prepare_window, backend='inductor', dynamic=False, fullgraph=True)
    def safe_prepare(*args, **kwargs):
        # 无历史首窗只执行一次，直接使用普通预处理，避免多编译一种形状。
        if args[1].shape[1] == model.gen_horizon_len // model.num_frames_per_token:
            return prepare_window(*args, **kwargs)
        if not model._pipeline_prepare_compiled:
            return prepare_window(*args, **kwargs)
        try:
            return compiled_prepare(*args, **kwargs)
        except AssertionError as exc:
            if 'Autotuned launcher config does not match any compile result' not in str(exc):
                raise
            # 预处理无随机数，完整重算覆盖部分条件是安全的；设备异常不吞掉。
            model._pipeline_prepare_compiled = False
            print('[Core8] 条件预处理调优不匹配，改用普通预处理；整步编译保留。', flush=True)
            return prepare_window(*args, **kwargs)
    model._pipeline_prepare_compiled = True
    model._pipeline_prepare = safe_prepare
    model.denoising_step = torch.compile(MethodType(denoising_step, model), backend='inductor', dynamic=False)
    model._generate_window = window_call
    model.autoregressive_step = autoregressive_call
    model.disable_core8_pipeline = disable
    model._core8_pipeline_calls = 0
    model._core8_pipeline_enabled = True
    print('[Core8] 已启用整步编译、窗口条件预处理和两路无约束 CFG。', flush=True)
    return True

