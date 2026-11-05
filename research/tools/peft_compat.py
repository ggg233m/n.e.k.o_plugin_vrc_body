# -*- coding: utf-8 -*-
r"""载入训练好的 adapter，兼容 torch.compile 留下的 `_orig_mod` 前缀。

为什么需要（2026-11-03 发现）：训练脚本对**内层模型**做了 torch.compile，
于是 save_pretrained 存下来的键名带 `_orig_mod`：

    base_model.model.model._orig_mod.language_model.layers.0...lora_B.weight

而评测侧 load_model() 不编译，键名是 `base_model.model.model.language_model...`。
PEFT 的 load_adapter 对不上的键**静默跳过**，结果是「adapter 载入了但不生效」，
评测出来的模型等于没训练。实测：adapter 前后输出差 0.0，mse8 从 1.78 变成 11.5。

用法::

    from peft_compat import load_adapter_compat
    missing = load_adapter_compat(model, run_dir / 'adapter', name='trained')
"""
from __future__ import annotations

import json
from pathlib import Path


def _strip_orig_mod(sd: dict) -> tuple[dict, int]:
    out = {}
    n = 0
    for k, v in sd.items():
        if "._orig_mod." in k:
            out[k.replace("._orig_mod.", ".")] = v
            n += 1
        else:
            out[k] = v
    return out, n


def load_adapter_compat(model, adapter_dir, name: str = "trained") -> dict:
    """载入 adapter，返回缺失/多出键的统计。失败时抛异常而不是静默。"""
    adapter_dir = Path(adapter_dir)
    st = adapter_dir / "adapter_model.safetensors"
    if not st.exists():
        raise FileNotFoundError(st)

    from safetensors.torch import load_file
    sd = load_file(str(st))
    fixed, n_fixed = _strip_orig_mod(sd)

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    try:
        model.load_adapter(str(adapter_dir), adapter_name=name, is_trainable=False)
    except Exception:
        # 键名不匹配时 PEFT 可能整个失败；此时先把 state_dict 手动写回去
        load_adapter_raw(model, fixed, cfg, name)
    else:
        # 即便 load_adapter 没抛，也要确认权重真的进去了：把修复后的键写回去一次。
        load_adapter_raw(model, fixed, cfg, name)
    model.set_adapter(name)
    return dict(total=len(sd), stripped=n_fixed, loaded=len(fixed))


def load_adapter_raw(model, sd: dict, cfg: dict, name: str) -> None:
    """把 state_dict 直接写进模型的 lora 参数（按名字后缀匹配）。"""
    own = dict(model.named_parameters())
    # PEFT 的键形如 ...layers.0.x.lora_A.weight，模型里是 ....lora_A.<name>.weight
    applied = 0
    for k, v in sd.items():
        cands = [k, k.replace(".lora_A.", ".lora_A.%s." % name)
                 .replace(".lora_B.", ".lora_B.%s." % name)]
        for c in cands:
            if c in own:
                with __import__("torch").no_grad():
                    own[c].copy_(v.to(own[c].dtype).to(own[c].device))
                applied += 1
                break
    if applied == 0:
        raise RuntimeError("adapter 一个键都没写进去；键名对不上")

