# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for ARDY text conditioning."""

import os
import json
import threading
from pathlib import Path

import numpy as np
import torch

from ardy.device import resolve_text_device

from .llm2vec import LLM2Vec
from .encoding_path import encode_single


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
        device: str = "auto",
        quantization: str | None = None,
        precision_map: str | None = None,
    ) -> None:
        device = resolve_text_device(None if device == "auto" else device)
        self.quantization = (quantization or os.environ.get("TEXT_ENCODER_QUANTIZATION", "none")).lower()
        if self.quantization not in ("none", "nf4", "int8", "merged_nf4", "mixed"):
            raise ValueError("量化模式仅支持 none / nf4 / int8 / merged_nf4 / mixed。")
        self.fast_encoding = self.quantization in ("int8", "merged_nf4", "mixed") or os.environ.get("TEXT_ENCODER_FAST_PATH") == "1"
        self._encode_lock = threading.Lock()
        torch_dtype = getattr(torch, dtype)
        self._compute_dtype = torch_dtype
        load_kwargs = {}
        if self.quantization == "nf4":
            if torch.device(device).type not in ("xpu", "cuda") or torch_dtype != torch.bfloat16:
                raise ValueError("NF4 模式要求 XPU/CUDA 设备和 bfloat16 计算精度。")
            from transformers import BitsAndBytesConfig
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                ),
                device_map={"": device},
            )
            print(f"文本编码器：NF4 / BF16，加载到 {device}", flush=True)
        self.llm_dim = llm_dim

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path)
            peft_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path)

        from .mixed_precision import MergedSource, canonical_hash, load_mixed, source_identity
        manifest = None
        if self.quantization in ("int8", "merged_nf4", "mixed"):
            if torch.device(device).type not in ("xpu", "cuda") or torch_dtype != torch.bfloat16:
                raise ValueError("混合量化需要 XPU/CUDA 和 BF16 计算。")
            source = MergedSource(base_model_name_or_path, peft_model_name_or_path)
            if self.quantization == "mixed":
                map_path = precision_map or os.environ.get("TEXT_ENCODER_PRECISION_MAP")
                if not map_path:
                    raise ValueError("mixed 模式必须提供 TEXT_ENCODER_PRECISION_MAP。")
                manifest = json.loads(Path(map_path).read_text(encoding="utf-8"))
            else:
                manifest = source.manifest("int8" if self.quantization == "int8" else "nf4")
            self.model = load_mixed(source, manifest, device)
            identity = source.identity
        else:
            self.model = LLM2Vec.from_pretrained(
                base_model_name_or_path=base_model_name_or_path,
                peft_model_name_or_path=peft_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                **load_kwargs,
            )
            from peft import PeftConfig
            base_id = PeftConfig.from_pretrained(base_model_name_or_path).base_model_name_or_path
            identity = source_identity(base_id, base_model_name_or_path, peft_model_name_or_path)
        self._cache_identity = {"version": 2, "weights": identity, "quantization": self.quantization,
                                "encoding_path": "single-v2" if self.fast_encoding else "legacy",
                                "manifest": manifest, "pooling": self.model.pooling_mode,
                                "max_length": self.model.max_length, "doc_max_length": self.model.doc_max_length,
                                "skip_instruction": self.model.skip_instruction, "torch": torch.__version__,
                                "transformers": __import__("transformers").__version__}
        self.cache_fingerprint = canonical_hash({**self._cache_identity, "dtype": str(torch_dtype)})
        if self.fast_encoding:
            self.model.model.config.use_cache = False

        self._device = device
        if self.quantization == "none":
            self.model = self.model.to(device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        if self.quantization != "none":
            # 已量化参数的设备及计算类型由加载时决定，防止界面再次转换权重。
            current = torch.device(self._device)
            requested = torch.device(device) if device is not None else current
            same_device = (current.type == requested.type and (current.index or 0) == (requested.index or 0))
            if not same_device or (dtype is not None and dtype != self._compute_dtype):
                raise ValueError("量化编码器切换设备或精度需要重新加载。")
            return self
        if device is not None and dtype is not None:
            self.model = self.model.to(device=device, dtype=dtype)
        elif device is not None:
            self.model = self.model.to(device)
        elif dtype is not None:
            self.model = self.model.to(dtype=dtype)
        if device is not None:
            self._device = str(device) if not isinstance(device, str) else device
        if dtype is not None:
            from .mixed_precision import canonical_hash
            self._compute_dtype = dtype
            self.cache_fingerprint = canonical_hash({**self._cache_identity, "dtype": str(dtype)})
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return torch.device(self._device)

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        if not self.fast_encoding and text:
            # 原 NF4 默认保留旧编码路径，未通过完整验收时仍可原样回退。
            with self._encode_lock, torch.no_grad():
                encoded_text = self.model.encode(text, batch_size=1, show_progress_bar=False, device=self._device)
        else:
            # 仍逐条计算，直接保留设备端结果；防止预热线程与交互线程重复占用显存。
            with self._encode_lock, torch.inference_mode():
                outputs = []
                for item in text:
                    outputs.append(encode_single(self.model, item, self._device).float())
                encoded_text = torch.cat(outputs) if outputs else torch.empty((0, self.llm_dim), device=self._device)

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.as_tensor(encoded_text).to(self._device)
        return encoded_text, lengths
