"""统一 CUDA、Intel XPU 与 CPU 的设备选择。"""
import os
import torch


def resolve_device(device=None):
    """自动优先使用 CUDA、XPU；明确指定的不可用设备直接报错。"""
    if device is None or str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    result = torch.device(device)
    if result.type not in ("cpu", "cuda", "xpu"):
        raise ValueError(f"不支持的设备：{result}")
    if result.type != "cpu":
        backend = getattr(torch, result.type)
        if not backend.is_available() or (result.index or 0) >= backend.device_count():
            raise RuntimeError(f"设备不可用：{result}")
    return str(result)


def resolve_text_device(device=None, motion_device=None):
    """XPU 上默认将 8B 文本编码器放在 CPU，显式参数优先于环境变量。"""
    requested = device if device is not None else os.environ.get("TEXT_ENCODER_DEVICE")
    if requested is not None:
        return resolve_device(requested)
    target = resolve_device(motion_device)
    return "cpu" if torch.device(target).type == "xpu" else target
