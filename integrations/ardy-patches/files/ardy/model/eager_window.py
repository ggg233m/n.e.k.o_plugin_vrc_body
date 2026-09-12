# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XPU eager 采样窗口缓存；不跨请求保留张量。

仅供 _generate_window 使用：窗口内条件、形状和权重不变，变化量为噪声 x 和时间步。
编译、训练、稀疏 token 和其他 CFG 类型沿用原路径。
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os

import torch
import torch.nn.functional as F

_WINDOW = ContextVar('ardy_eager_window', default=None)


def current_window():
    return _WINDOW.get()


@contextmanager
def sampling_window(model, unconstrained):
    denoiser = model.denoiser
    enabled = (
        torch.device(model.device).type == 'xpu'
        and getattr(model, 'xpu_window_cache_enabled', True)
        and os.environ.get('ARDY_XPU_WINDOW_CACHE', '1') != '0'
        and not denoiser.training
        and type(denoiser).__name__ == 'AutoLatentClassifierFreeGuidedModelSeparated'
        and not getattr(denoiser.model, 'sparsify_token_seq', False)
    )
    value = {'cfg': denoiser, 'unconstrained': unconstrained, 'entries': {}} if enabled else None
    token = _WINDOW.set(value)
    try:
        yield
    finally:
        # 正常返回与异常都立即释放缓存，多个线程/客户端不共享条件。
        _WINDOW.reset(token)


def cfg_forward(owner, window, args):
    cache = window['entries'].get(owner)
    if cache is None:
        two_pass = window['unconstrained'] and args['motion_mask'] is None and args['observed_motion'] is None
        count = 2 if two_pass else 3
        repeat = lambda value: torch.cat([value] * count, dim=0)
        static = {name: repeat(args[name]) for name in (
            'history_len', 'generation_len', 'future_len', 'history_mask',
            'generation_mask', 'future_mask', 'history_token_mask', 'generation_token_mask')}
        future = args['future_token_mask']
        text = args['text_feat']
        text_mask = args['text_feat_pad_mask']
        static['future_token_mask'] = torch.cat([0 * future, 0 * future] if two_pass else [0 * future, future, 0 * future])
        static['text_feat'] = torch.cat([text, 0 * text] if two_pass else [text, 0 * text, 0 * text])
        static['text_feat_pad_mask'] = torch.cat([text_mask, 0 * text_mask] if two_pass else [text_mask, 0 * text_mask, 0 * text_mask])
        heading = args['first_heading_angle']
        static['first_heading_angle'] = repeat(heading) if heading is not None else None
        for name in ('motion_mask', 'observed_motion'):
            value = args[name]
            static[name] = torch.cat([0 * value, value, 0 * value]) if value is not None else None
        for name in ('history_mask', 'generation_mask', 'future_mask', 'history_token_mask',
                     'generation_token_mask', 'future_token_mask', 'text_feat_pad_mask'):
            static[name] = static[name] > 0.5
        cache = (count, static)
        window['entries'][owner] = cache
    count, static = cache
    output = owner.model(x=torch.cat([args['x']] * count),
                         timesteps=torch.cat([args['timesteps']] * count), **static)
    if count == 2:
        text, unconditional = output.chunk(2)
        return unconditional + args['cfg_weight_text'] * (text - unconditional)
    text, constraint, unconditional = output.chunk(3)
    return (unconditional + args['cfg_weight_text'] * (text - unconditional)
            + args['cfg_weight_cstr'] * (constraint - unconditional))


def backbone_forward(owner, window, args):
    # 仅在 eval 且位置编码实现匹配时进入本路径。
    from ardy.model.backbone import pad_x_and_mask_to_fixed_size

    x = owner.input_linear(args['x'])
    cache = window['entries'].get(owner)
    if cache is None:
        text, text_mask = args['text_feat'], args['text_feat_pad_mask']
        if owner.num_text_tokens is not None:
            text, text_mask = pad_x_and_mask_to_fixed_size(text, text_mask, owner.num_text_tokens)
        embedded_text = owner.embed_text(text.to(owner.embed_text.weight.dtype))
        if not owner.use_text_mask:
            text_mask = torch.ones(embedded_text.shape[:2], dtype=torch.bool, device=x.device)
        prefix_mask = [text_mask, torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)]
        heading = None
        if owner.input_first_heading_angle:
            angle = args['first_heading_angle']
            assert angle is not None
            heading = owner.linear_first_heading_angle(torch.stack([torch.cos(angle), torch.sin(angle)], -1))[:, None]
            prefix_mask.append(torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device))
        prefix_len = embedded_text.shape[1] + 1 + int(heading is not None)
        assert prefix_len <= owner.learned_prefix_embedding.max_len
        positions = torch.arange(prefix_len, device=x.device, dtype=torch.int32)[None]
        prefix_pe = owner.learned_prefix_embedding.embedding(positions)
        index = args['token_index']
        pe = owner.motion_token_embedding
        assert index.abs().max() < pe.max_len, '位置编码索引越界'
        safe_index = torch.where(index >= 0, index, index + pe.pe.shape[0])
        motion_pe = F.embedding(safe_index.int(), pe.pe)
        padding = ~torch.cat([*prefix_mask, args['x_pad_mask']], dim=1)
        # 各层构造的加性 mask 相同，在窗口内共享。
        attention_mask = torch.zeros((x.shape[0], 1, 1, padding.shape[1]), device=x.device, dtype=x.dtype)
        attention_mask = attention_mask.masked_fill(padding[:, None, None, :], float('-inf'))
        cache = (embedded_text, heading, prefix_len, prefix_pe, motion_pe, attention_mask)
        window['entries'][owner] = cache
    text, heading, prefix_len, prefix_pe, motion_pe, attention_mask = cache
    time = owner.embed_timestep(args['timesteps'])
    prefix = torch.cat([text, time] if heading is None else [text, time, heading], dim=1)
    prefix = owner.learned_prefix_embedding.dropout(prefix + prefix_pe)
    motion = owner.motion_token_embedding.dropout(x + motion_pe)
    value = torch.cat([prefix, motion], dim=1)
    for layer in owner.seqTransEncoder.layers:
        if layer.norm_first:
            value = value + layer._sa_block(layer.norm1(value), attention_mask)
            value = value + layer._ff_block(layer.norm2(value))
        else:
            value = layer.norm1(value + layer._sa_block(value, attention_mask))
            value = layer.norm2(value + layer._ff_block(value))
    if owner.seqTransEncoder.norm is not None:
        value = owner.seqTransEncoder.norm(value)
    return owner.output_linear(value[:, prefix_len:])
