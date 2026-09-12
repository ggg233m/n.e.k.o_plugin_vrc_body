"""逐层预量化缓存；只存张量和元数据，不反量化后再次量化。"""
import importlib.metadata
import os
from pathlib import Path

import torch
from torch import nn


def cache_file(source, name, scheme, threshold=6.):
    root = os.environ.get("TEXT_ENCODER_LAYER_CACHE")
    if not root:
        return None
    version = importlib.metadata.version("bitsandbytes")
    suffix = "int8-t0" if scheme == "int8" and threshold == 0 else scheme
    return Path(root) / f"v1-bnb-{version}" / source.identity / f"{name}.{suffix}.pt"


def read_cache(source, name, scheme, threshold=6.):
    path = cache_file(source, name, scheme, threshold=threshold)
    reused = False
    if path is not None and not path.exists() and scheme == "int8" and threshold == 0:
        # 阈值只控制运行时激活离群值；旧的权重整数与缩放系数可原样复用。
        path = cache_file(source, name, scheme, threshold=6.)
        reused = True
    if path is None or not path.exists():
        return None
    item = torch.load(path, map_location="cpu", weights_only=True)
    if item["source"] != source.identity or item["name"] != name or item["scheme"] != scheme:
        raise ValueError(f"逐层缓存身份不符：{path}")
    if reused:
        if item.get("threshold", 6.) != 6.:
            raise ValueError("旧 INT8 缓存阈值不符合预期。")
        item = {**item, "threshold": 0.}
    if scheme == "int8" and item.get("threshold", 6.) != threshold:
        raise ValueError("INT8 缓存阈值不匹配。")
    return item


def write_cache(source, name, scheme, value, threshold=6.):
    path = cache_file(source, name, scheme, threshold=threshold)
    if path is None:
        return
    state = value.state_dict() if isinstance(value, nn.Module) else {"weight": value}
    state = {key: tensor.detach().as_subclass(torch.Tensor).cpu().contiguous() for key, tensor in state.items()}
    shape = [value.out_features, value.in_features] if isinstance(value, nn.Module) else list(value.shape)
    item = {"source": source.identity, "name": name, "scheme": scheme, "shape": shape, "state": state, "threshold": threshold}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(item, temporary)
    temporary.replace(path)


def restore_linear(item, device):
    import bitsandbytes as bnb
    scheme, state = item["scheme"], item["state"]
    out_features, in_features = item["shape"]
    if scheme == "nf4":
        with torch.device("meta"):
            module = bnb.nn.Linear4bit(in_features, out_features, bias=False, compute_dtype=torch.bfloat16,
                                      compress_statistics=True, quant_type="nf4")
        module.weight = bnb.nn.Params4bit.from_prequantized(
            state["weight"], {key: value for key, value in state.items() if key != "weight"},
            requires_grad=False, device=device, module=module)
    elif scheme == "int8":
        if state["weight"].dtype != torch.int8 or int(state["weight_format"]) != 0:
            raise ValueError("INT8 缓存必须为已量化的行主序权重。")
        with torch.device("meta"):
            module = bnb.nn.Linear8bitLt(in_features, out_features, bias=False, has_fp16_weights=False, threshold=item.get("threshold", 6.))
        packed = state["weight"].to(device)
        module.weight = bnb.nn.Int8Params(packed, requires_grad=False, has_fp16_weights=False,
                                         CB=packed, SCB=state["SCB"].to(device))
    elif scheme == "bf16":
        with torch.device("meta"):
            module = nn.Linear(in_features, out_features, bias=False)
        module.weight = nn.Parameter(state["weight"].to(device), requires_grad=False)
    else:
        raise ValueError(f"未知的缓存精度：{scheme}")
    return module.eval()
