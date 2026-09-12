"""显式双向编码前向；共享于分层参考与量化模型。"""
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from .models.bidirectional_llama import LlamaBiModel


def attention_arguments(model, hidden, mask):
    """不使用三角因果掩码；单条无填充输入不产生 masked-SDPA 内核。"""
    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    attention = None
    if mask is not None and not bool(mask.all()):
        attention = torch.zeros((hidden.shape[0], 1, hidden.shape[1], hidden.shape[1]),
                                device=hidden.device, dtype=hidden.dtype)
        attention.masked_fill_(~mask[:, None, None, :].bool(), torch.finfo(hidden.dtype).min)
    return {"attention_mask": attention, "position_ids": positions,
            "position_embeddings": model.rotary_emb(hidden, positions), "use_cache": False}


class EncodingLlamaModel(LlamaBiModel):
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("文本编码不接受历史 KV cache。")
        if inputs_embeds is None:
            # 通用入口保持可用；单条快速入口已在 CPU 完成查表。
            hidden = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
            hidden = hidden.to(self.norm.weight.device)
        else:
            hidden = inputs_embeds
        args = attention_arguments(self, hidden, attention_mask)
        # XPU 带掩码时使用 MATH；避免历史上已出现的融合 BF16 masked-SDPA 故障。
        from contextlib import nullcontext
        from torch.nn.attention import SDPBackend, sdpa_kernel
        guard = sdpa_kernel(SDPBackend.MATH) if hidden.device.type == "xpu" and args["attention_mask"] is not None else nullcontext()
        with guard:
            for layer in self.layers:
                hidden = layer(hidden, **args)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden))


def make_model(config):
    config.use_cache = False
    return EncodingLlamaModel(config)
