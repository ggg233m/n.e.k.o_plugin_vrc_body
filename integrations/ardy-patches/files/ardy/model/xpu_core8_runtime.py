"""Core8 编解码与采样计划优化，按模型实例启用，可完整恢复。"""
import inspect
from threading import RLock, local
import torch


def enable_core8_runtime(model, *, encode=False, decode=False, schedule=True):
    if getattr(model, '_core8_runtime_enabled', False):
        return True
    if not getattr(model, '_core8_pipeline_enabled', False):
        return False
    diffusion=model.diffusion
    original_call=model.autoregressive_step
    original_encode=model._encode_init_history
    original_decode=model.hybrid.get_explicit_motion_from_hybrid
    original_space=diffusion.space_timesteps
    original_calc=diffusion.calc_diffusion_vars
    original_disable=model.disable_core8_pipeline
    # 从原类型获取签名，不依赖已有包装函数的 *args 形参。
    signature=inspect.signature(type(model).autoregressive_step)
    lock=RLock();scope=local();cached=None;cached_key=None;buffer_signature=None

    def fingerprint():
        return tuple((name,id(t),t._version,str(t.device),t.dtype)
                     for name,t in diffusion._buffers.items() if t is not None)

    def space(steps):
        if getattr(scope,'active',False):
            return cached
        return original_space(steps)

    def calc(timesteps):
        if getattr(scope,'active',False) and timesteps is cached[0]:
            return
        return original_calc(timesteps)

    def call(*args,**kwargs):
        nonlocal cached,cached_key,buffer_signature
        with lock,torch.inference_mode(False),torch.no_grad():
            if schedule:
                values=signature.bind(model,*args,**kwargs).arguments
                steps=values['num_denoising_steps']
                if not isinstance(steps,int) or isinstance(steps,bool):
                    return original_call(*args,**kwargs)
                key=(steps,diffusion.num_base_steps,str(diffusion.device))
                if key!=cached_key or fingerprint()!=buffer_signature:
                    # 与原实现一样使用设备上的步数标量，避免舍入路径变化。
                    scalar=torch.tensor([steps],device=diffusion.device)[0]
                    cached=original_space(scalar)
                    original_calc(cached[0])
                    cached_key=key
                    buffer_signature=fingerprint()
                    model._core8_schedule_builds+=1
                scope.active=True
            try:
                result=original_call(*args,**kwargs)
                model._core8_runtime_calls+=1
                return result
            finally:
                scope.active=False

    def disable():
        nonlocal cached
        with lock:
            model.autoregressive_step=original_call
            model._encode_init_history=original_encode
            model.hybrid.get_explicit_motion_from_hybrid=original_decode
            diffusion.space_timesteps=original_space
            diffusion.calc_diffusion_vars=original_calc
            model.disable_core8_pipeline=original_disable
            model._core8_runtime_enabled=False
            cached=None

    def disable_pipeline():
        disable()
        original_disable()

    if encode:model._encode_init_history=torch.compile(original_encode,backend='inductor',dynamic=False)
    if decode:model.hybrid.get_explicit_motion_from_hybrid=torch.compile(original_decode,backend='inductor',dynamic=False)
    if schedule:
        diffusion.space_timesteps=space
        diffusion.calc_diffusion_vars=calc
    model.autoregressive_step=call
    model.disable_core8_runtime=disable
    model.disable_core8_pipeline=disable_pipeline
    model._core8_runtime_enabled=True
    model._core8_runtime_calls=0
    model._core8_schedule_builds=0
    print(f'[Core{model.gen_horizon_len}] 编码编译={encode}，解码编译={decode}，采样计划缓存={schedule}',flush=True)
    return True
