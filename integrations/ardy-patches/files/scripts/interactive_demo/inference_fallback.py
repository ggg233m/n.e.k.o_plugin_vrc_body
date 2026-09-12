"""交互推理中已知 XPU 编译错误的窄范围回退。"""


def autoregressive_step_with_fallback(model, **kwargs):
    try:
        return model.autoregressive_step(**kwargs)
    except AssertionError as exc:
        # 仅处理已确认的调优配置不匹配，其他模型错误保留原始异常。
        original = getattr(model.denoiser, "_orig_mod", None)
        if original is None or "Autotuned launcher config does not match any compile result" not in str(exc):
            raise
        model.denoiser = original
        print("[XPU 编译回退] Triton 调优配置不匹配，已切回 eager 并重试当前动作段。", flush=True)
        # 重试失败时直接向上传播，避免无限重试掩盖真实故障。
        return model.autoregressive_step(**kwargs)
