"""单条提示词编码：在 CPU 决定掩码和池化范围，避免设备同步。"""
import torch


def encode_single(encoder, text, device):
    formatted = encoder.prepare_for_tokenization(encoder._convert_to_str("", text))
    features = encoder.tokenize([formatted])
    if encoder.pooling_mode != "mean":
        # 其他池化方式维持原始实现，不假设它们与均值池化等价。
        features = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in features.items()}
        return encoder.forward(features)
    attention = features["attention_mask"]
    pooling = features["embed_mask"] if encoder.skip_instruction else attention
    length = int(pooling[0].sum())
    no_padding = bool(attention.all())
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in features.items() if key not in ("embed_mask", "attention_mask")
              and not (key == "input_ids" and encoder.model.embed_tokens.weight.device.type == "cpu")}
    inputs["attention_mask"] = None if no_padding else attention.to(device)
    if encoder.model.embed_tokens.weight.device.type == "cpu":
        # 分词结果留在 CPU；只传输本条提示词实际使用的词向量。
        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = encoder.model.embed_tokens(features["input_ids"]).to(device)
    inputs["use_cache"] = False
    hidden = encoder.model(**inputs).last_hidden_state
    # 与原实现保持相同的尾部切片和 BF16 均值，包含 length=0 的原有语义。
    return hidden[0, -length:, :].mean(dim=0, keepdim=True)
