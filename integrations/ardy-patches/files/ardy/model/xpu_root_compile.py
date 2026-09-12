"""可选的 XPU FP32 根节点转换编译；不编译注意力或整个去噪模型。"""
import copy
import time
import warnings

import torch

from ardy.motion_rep.stats import Stats


def enable_root_compile(model):
    """预热完成后才安装局部函数；失败时保留原路径。"""
    rep = model.motion_rep
    if hasattr(model.denoiser, '_orig_mod'):
        print('[XPU root] 已启用完整 compile，跳过根节点局部编译。', flush=True)
        return False
    device = next(model.denoiser.parameters()).device
    if device.type != 'xpu' or getattr(rep, '_xpu_root_compile_installed', False):
        return False
    if rep.global_root_dim != 5 or rep.local_root_dim != 4:
        return False
    original = rep.global_root_to_local_root
    proxy = copy.copy(rep)
    # 编译函数使用独立的小常量副本，CPU 后处理继续使用原统计量。
    for name in ('global_root_stats', 'local_root_stats'):
        source = getattr(rep, name)
        target = Stats(load=False, eps=source.eps)
        target.register_from_tensors(source.mean.detach().clone(), source.std.detach().clone())
        setattr(proxy, name, target.to(device=device, dtype=torch.float32))
    runner = torch.compile(proxy.global_root_to_local_root, fullgraph=True, dynamic=True,
                           options={'max_autotune': False, 'autotune_local_cache': False})
    from torch._dynamo.exc import BackendCompilerFailed, Unsupported

    def recoverable(exc):
        if any(marker in str(exc).lower() for marker in ('device_lost', 'device lost', 'ur_result_error', 'access violation')):
            return False
        return (isinstance(exc, (BackendCompilerFailed, Unsupported))
                or (isinstance(exc, AssertionError)
                    and 'Autotuned launcher config does not match any compile result' in str(exc)))

    started = time.perf_counter()
    print('[XPU root] 预热 FP32 根节点转换；首次运行需要编译。', flush=True)
    try:
        with torch.inference_mode():
            for batch, frames in ((2, 12), (3, 60), (3, 44), (2, 8)):
                data = torch.zeros((batch, frames, 5), device=device)
                lengths = torch.full((batch,), min(frames, 12), device=device, dtype=torch.long)
                output = runner(data, normalized=True, lengths=lengths)
                if not bool(torch.isfinite(output).all()):
                    raise ValueError('根节点编译预热输出包含非有限值')
            torch.xpu.synchronize()
    except Exception as exc:
        if not recoverable(exc):
            raise
        warnings.warn(f'根节点局部编译不可用，保留原路径：{exc}')
        return False

    enabled = True

    def converted(root_features, normalized, lengths):
        nonlocal enabled
        eligible = (enabled and getattr(rep, '_xpu_root_compile_enabled', True)
                    and not torch.is_grad_enabled() and normalized
                    and root_features.device == device and root_features.dtype == torch.float32
                    and root_features.ndim == 3 and root_features.shape[0] in (2, 3)
                    and root_features.shape[1] >= 2 and root_features.shape[-1] == 5
                    and lengths is not None and lengths.device == device and lengths.dtype == torch.long
                    and lengths.shape == root_features.shape[:1])
        if eligible:
            try:
                rep._xpu_root_compile_calls += 1
                return runner(root_features, normalized=True, lengths=lengths)
            except Exception as exc:
                if not recoverable(exc):
                    raise
                # 本函数没有随机采样，可直接回退；设备丢失等运行错误继续抛出。
                enabled = False
                warnings.warn(f'根节点局部编译失败，本模型改回原路径：{exc}')
        return original(root_features, normalized=normalized, lengths=lengths)

    rep.global_root_to_local_root = converted
    rep._xpu_root_compile_calls = 0
    rep._xpu_root_compile_installed = True
    print(f'[XPU root] 局部编译就绪，预热 {time.perf_counter()-started:.1f} 秒；注意力与采样仍为原路径。', flush=True)
    return True
