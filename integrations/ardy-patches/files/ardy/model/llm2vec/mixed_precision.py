"""LLM2Vec 的逐层 LoRA 合并、混合精度加载和可验证配置。"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoTokenizer

SCHEMES = {"nf4", "int8", "bf16"}
FORMAT_VERSION = 1


class _ExpectedInt8Cast(logging.Filter):
    def filter(self, record):
        # 此固定配置的 BF16→FP16 量化转换是预期行为，不应每层每次写日志。
        return not str(record.msg).startswith("MatMul8bitLt: inputs will be cast from")


logging.getLogger("bitsandbytes.autograd._functions").addFilter(_ExpectedInt8Cast())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_identity(*directories):
    """以配置内容及权重文件元数据隔离本机缓存，不把路径当作模型版本。"""
    result = []
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            result.append({"remote": str(directory)})
            continue
        files = []
        for file in sorted(root.iterdir()):
            if file.suffix not in {".json", ".safetensors", ".bin"}:
                continue
            stat = file.stat()
            files.append([file.name, stat.st_size, stat.st_mtime_ns,
                          hashlib.sha256(file.read_bytes()).hexdigest() if file.suffix == ".json" else None])
        result.append(files)
    return canonical_hash(result)


class MergedSource:
    """一次只构造一份 FP32 权重，原始检查点始终只读。"""

    def __init__(self, mntp_dir, supervised_dir):
        self.mntp_dir = Path(mntp_dir)
        self.supervised_dir = Path(supervised_dir)
        self.adapters = []
        self.configs = []
        for directory in (self.mntp_dir, self.supervised_dir):
            cfg = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
            if cfg.get("bias", "none") != "none" or cfg.get("modules_to_save") or cfg.get("use_dora"):
                raise ValueError("逐层合并仅接受无额外保存模块的普通 LoRA。")
            if cfg.get("fan_in_fan_out") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
                raise ValueError("当前检查点之外的 LoRA 布局需要单独验证，不能静默合并。")
            with safe_open(str(directory / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
                self.adapters.append({key: f.get_tensor(key) for key in f.keys()})
            self.configs.append(cfg)
        self.base_dir = Path(self.configs[0]["base_model_name_or_path"])
        if not self.base_dir.is_dir():
            raise ValueError("混合模式需要本地原始 Llama 权重。")
        index = json.loads((self.base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.index = index["weight_map"]
        self._reader = None
        self._reader_name = None
        self.identity = source_identity(self.base_dir, self.mntp_dir, self.supervised_dir)
        self.config = AutoConfig.from_pretrained(str(self.base_dir), local_files_only=True)
        self.config.use_cache = False
        # 此标识参与原版 LLM2Vec 的提示词格式选择，不能改为合并产物路径。
        self.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.config._attn_implementation = "sdpa"
        self.linear_names = [key.removeprefix("model.").removesuffix(".weight")
                             for key in self.index if key.startswith("model.layers.")
                             and key.endswith("_proj.weight")]
        self.linear_names.sort()

    def weight(self, name):
        key = "model." + name if not name.startswith("model.") else name
        shard = self.index[key]
        # 保持当前分片的映射，避免 Windows 上每层重新映射大文件；最多保留一个分片。
        if self._reader_name != shard:
            self.close()
            self._reader = safe_open(str(self.base_dir / shard), framework="pt", device="cpu")
            self._reader.__enter__()
            self._reader_name = shard
        value = self._reader.get_tensor(key).to(dtype=torch.float32, copy=True)
        local_name = key.removeprefix("model.").removesuffix(".weight")
        adapter_key = "base_model.model." + local_name
        for adapter, cfg in zip(self.adapters, self.configs):
            a = adapter.get(adapter_key + ".lora_A.weight")
            b = adapter.get(adapter_key + ".lora_B.weight")
            if (a is None) != (b is None):
                raise ValueError(f"LoRA 张量不完整：{key}")
            if local_name in self.linear_names and a is None:
                raise ValueError(f"目标线性层缺少 LoRA，禁止静默跳过：{key}")
            if a is not None:
                rank = cfg["r"]
                scale = cfg["lora_alpha"] / (rank ** 0.5 if cfg.get("use_rslora") else rank)
                value.addmm_(b.float(), a.float(), beta=1, alpha=scale)
        return value

    def close(self):
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.__exit__(None, None, None)
        self._reader = None
        self._reader_name = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def manifest(self, default="int8", overrides=None):
        if default not in SCHEMES:
            raise ValueError(default)
        layers = dict.fromkeys(self.linear_names, default)
        layers.update(overrides or {})
        result = {"format_version": FORMAT_VERSION, "source_identity": self.identity,
                  "layers": layers, "int8_threshold": 6.0, "compute_dtype": "bfloat16"}
        validate_manifest(result, self)
        return result


def validate_manifest(manifest, source):
    if manifest.get("format_version") not in (1, 2):
        raise ValueError("精度清单版本不支持。")
    if manifest.get("source_identity") != source.identity:
        raise ValueError("精度清单与底座或 LoRA 版本不一致。")
    if set(manifest.get("layers", {})) != set(source.linear_names):
        raise ValueError("精度清单必须恰好覆盖所有线性层，不能遗漏或包含未知层。")
    if not set(manifest["layers"].values()) <= SCHEMES:
        raise ValueError("精度清单仅支持 nf4 / int8 / bf16。")
    if manifest.get("compute_dtype") != "bfloat16" or manifest.get("int8_threshold") != 6.0:
        raise ValueError("当前已验证接口要求 BF16 和 INT8 阈值 6.0。")
    if manifest.get("format_version") == 1:
        if "embedding_device" in manifest or "int8_thresholds" in manifest:
            raise ValueError("CPU 嵌入表和逐层阈值需要版本 2 清单。")
    else:
        if manifest.get("embedding_device") not in ("cpu", "model"):
            raise ValueError("嵌入表位置必须为 cpu 或 model。")
        thresholds = manifest.get("int8_thresholds", {})
        expected = {n for n, s in manifest["layers"].items() if s == "int8"}
        if set(thresholds) != expected or any(type(v) not in (int, float) or v not in (0., 6.) for v in thresholds.values()):
            raise ValueError("逐层阈值必须恰好覆盖 INT8 层，且只能为 0 或 6。")
    return manifest


def make_linear(weight, scheme, device, threshold=6.):
    """仅当前线性层的浮点副本进入设备，量化后不保留整模型副本。"""
    import bitsandbytes as bnb
    out_features, in_features = weight.shape
    if scheme == "nf4":
        with torch.device("meta"):
            module = bnb.nn.Linear4bit(in_features, out_features, bias=False,
                                      compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4")
        module.weight = bnb.nn.Params4bit(weight.to(torch.bfloat16), requires_grad=False,
                                         compress_statistics=True, quant_type="nf4", module=module)
    elif scheme == "int8":
        with torch.device("meta"):
            module = bnb.nn.Linear8bitLt(in_features, out_features, bias=False,
                                        has_fp16_weights=False, threshold=threshold)
        module.weight = bnb.nn.Int8Params(weight.to(torch.bfloat16), requires_grad=False,
                                         has_fp16_weights=False)
    elif scheme == "bf16":
        with torch.device("meta"):
            module = nn.Linear(in_features, out_features, bias=False)
        module.weight = nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)
    else:
        raise ValueError(scheme)
    return module.to(device).eval()


def load_mixed(source, manifest, device, progress=True):
    from .bidirectional_runtime import make_model
    from .llm2vec import LLM2Vec
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from .layer_cache import read_cache, write_cache, restore_linear
    validate_manifest(manifest, source)
    with torch.device("meta"):
        model = make_model(source.config)
    parameter_names = [name for name, _ in model.named_parameters()]
    merge_seconds = 0.0
    device_seconds = 0.0
    for index, name in enumerate(parameter_names):
        module_name = name.removesuffix(".weight")
        scheme = manifest["layers"].get(module_name, "float")
        started = time.perf_counter()
        threshold = manifest.get("int8_thresholds", {}).get(module_name, 6.)
        cached = read_cache(source, name, scheme, threshold=threshold)
        value = source.weight(name) if cached is None else None
        merge_seconds += time.perf_counter() - started
        started = time.perf_counter()
        if module_name in manifest["layers"]:
            layer = restore_linear(cached, device) if cached is not None else make_linear(value, scheme, device, threshold=threshold)
            model.set_submodule(module_name, layer)
            if cached is None:
                write_cache(source, name, scheme, layer, threshold=threshold)
        else:
            parent, _, leaf = name.rpartition(".")
            target = "cpu" if name == "embed_tokens.weight" and manifest.get("embedding_device") == "cpu" else device
            tensor = (cached["state"]["weight"] if cached is not None else value).to(device=target, dtype=torch.bfloat16)
            model.get_submodule(parent).register_parameter(leaf, nn.Parameter(tensor, requires_grad=False))
            if cached is None:
                write_cache(source, name, scheme, tensor)
            del tensor
        del value, cached
        device_seconds += time.perf_counter() - started
        if progress and (index % 32 == 0 or index == len(parameter_names) - 1):
            print(f"[混合精度加载] {index + 1}/{len(parameter_names)} {name} "
                  f"CPU读取合并={merge_seconds:.1f}s 设备构建={device_seconds:.1f}s", flush=True)
    model.rotary_emb = LlamaRotaryEmbedding(source.config).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source.mntp_dir), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config_path = source.supervised_dir / "llm2vec_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    encoder = LLM2Vec(model, tokenizer, **config)
    return encoder

