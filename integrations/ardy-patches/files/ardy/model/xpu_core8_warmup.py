"""在接收交互前预热普通续帧及常用路径点窗口。"""
import time
from contextlib import nullcontext
import torch


def warmup_core8_interactive(model):
    if not getattr(model, '_core8_pipeline_enabled', False):
        return False
    if getattr(model, '_core8_interactive_ready', False):
        return True
    device = torch.device(model.device)
    index = device.index if device.index is not None else torch.xpu.current_device()
    rep = model.motion_rep
    started = time.perf_counter()
    print('[Core8] 正在预热普通窗口和 64 帧路径点窗口，完成后再接收交互。', flush=True)
    # 预热不消耗用户生成时的随机状态，也不写入 Demo 历史。
    with torch.random.fork_rng(devices=[index], device_type='xpu'), torch.inference_mode(False), torch.no_grad():
        torch.manual_seed(51)
        text = torch.zeros(1, 1, 4096, device=device)
        pad = torch.ones(1, 1, device=device, dtype=torch.bool)
        history = None
        for frames, constrained in ((8, False), (12, False), (64, True)):
            # 首窗仅使用一次，避免为它加载整组编译内核。
            with torch.compiler.set_stance('force_eager') if frames == 8 else nullcontext():
                mask = obs = None
                if constrained:
                    mask = torch.zeros(1, frames, rep.motion_rep_dim, device=device)
                    obs = torch.zeros_like(mask)
                    mask[:, -1, rep.slice_dict['root_pos']] = 1
                value = model.autoregressive_step(
                    num_frames=frames, num_denoising_steps=10,
                    motion_mask=mask, observed_motion=obs, cfg_weight=(2., 2.),
                    text_feat=text, text_pad_mask=pad, init_history_sequence=history,
                )
                assert torch.isfinite(value).all(), '预热出现非有限动作'
                rep.inverse(rep.unnormalize(value), is_normalized=False)
                if history is None:
                    history = value[:, -4:].clone()
        torch.xpu.synchronize()
    model._core8_interactive_ready = True
    print(f'[Core8] 交互预热完成，用时 {time.perf_counter()-started:.1f} 秒；新形状禁止在线重编译。', flush=True)
    return True
