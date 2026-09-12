# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feature normalization statistics (mean/std) for motion representations."""

import logging
import os
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class Stats(torch.nn.Module):
    """Utility module for feature normalization statistics.

    Normalization follows:
    ``(data - mean) / sqrt(std**2 + eps)``
    """

    def __init__(
        self,
        folder: Optional[str] = None,
        load: bool = True,
        load_as_fp32: bool = True,
        eps=1e-05,
    ):
        super().__init__()
        self.folder = folder
        self.eps = eps
        self.load_as_fp32 = load_as_fp32
        if folder is not None and load:
            self.load()

    def sliced(self, indices):
        """Return a new ``Stats`` object containing selected feature indices."""
        new_stats = Stats(folder=self.folder, load=False, eps=self.eps)
        new_stats.register_from_tensors(
            self.mean[..., indices].clone(),
            self.std[..., indices].clone(),
        )
        return new_stats

    def load(self):
        """Load ``mean.npy`` and ``std.npy`` from ``self.folder``."""
        mean_path = os.path.join(self.folder, "mean.npy")
        std_path = os.path.join(self.folder, "std.npy")
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(
                f"Missing stats files in '{self.folder}'. Expected:\n"
                f"  - {mean_path}\n"
                f"  - {std_path}\n\n"
                "These files ship alongside the model checkpoint. Re-download the "
                "checkpoint, or check that the checkpoint directory you passed is "
                "complete."
            )

        mean = torch.from_numpy(np.load(mean_path))
        std = torch.from_numpy(np.load(std_path))
        if self.load_as_fp32:
            mean = mean.to(dtype=torch.float32)
            std = std.to(dtype=torch.float32)
        self.register_from_tensors(mean, std)

    def register_from_tensors(self, mean: torch.Tensor, std: torch.Tensor):
        """Register mean/std tensors as non-persistent buffers."""
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        # Precomputed eps-adjusted denominator so normalize/unnormalize skip
        # the per-call sqrt.
        self.register_buffer("std_eps", torch.sqrt(std**2 + self.eps), persistent=False)
        self._xpu_stats_cache = None

    def _stats_for(self, data):
        """只缓存 XPU 推理常量，保留 CPU 原件及训练、编译时的原有行为。"""
        mean, scale = self.mean, self.std_eps
        if (
            data.device.type == "xpu"
            and not torch.compiler.is_compiling()
            and not torch.is_grad_enabled()
            and getattr(self, "xpu_stats_cache_enabled", True)
            and not mean.is_inference() and not scale.is_inference()
        ):
            # 只保留最近一种设备/精度；权重替换或原地更新后立即失效。
            key = (id(mean), mean._version, id(scale), scale._version, data.device, data.dtype)
            cached = getattr(self, "_xpu_stats_cache", None)
            if cached is not None and cached[0] == key:
                return cached[1], cached[2]
            converted = (mean.to(device=data.device, dtype=data.dtype),
                         scale.to(device=data.device, dtype=data.dtype))
            self._xpu_stats_cache = (key, *converted, mean, scale)
            return converted
        return (mean.to(device=data.device, dtype=data.dtype),
                scale.to(device=data.device, dtype=data.dtype))

    def _apply(self, fn, recurse=True):
        # 显式迁移/转换 Stats 时释放旧设备副本。
        self._xpu_stats_cache = None
        return super()._apply(fn, recurse=recurse)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data using the stored statistics."""
        mean, std_eps = self._stats_for(data)
        return (data - mean) / std_eps

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Undo normalization using the stored statistics."""
        mean, std_eps = self._stats_for(data)
        return data * std_eps + mean

    def is_loaded(self):
        """Return whether statistics are currently available."""
        return hasattr(self, "mean")

    def get_dim(self):
        """Return feature dimensionality."""
        return self.mean.shape[0]

    def save(
        self,
        folder: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        """Save statistics to ``folder`` as ``mean.npy`` and ``std.npy``."""
        if folder is None:
            folder = self.folder
            if folder is None:
                raise ValueError("No folder to save stats")

        if mean is None and std is None:
            try:
                mean = self.mean.cpu().numpy()
                std = self.std.cpu().numpy()
            except AttributeError:
                raise ValueError("Stats were not loaded")

        # don't override stats folder
        os.makedirs(folder, exist_ok=False)

        np.save(os.path.join(folder, "mean.npy"), mean)
        np.save(os.path.join(folder, "std.npy"), std)

    def __eq__(self, other):
        return (self.mean.cpu() == other.mean.cpu()).all() and (self.std.cpu() == other.std.cpu()).all()

    # should define a hash value for pytorch, as we defined __eq__
    def __hash__(self):
        # Convert mean and std to bytes for a consistent hash value
        mean_hash = hash(self.mean.detach().cpu().numpy().tobytes())
        std_hash = hash(self.std.detach().cpu().numpy().tobytes())
        return hash((mean_hash, std_hash))

    def __repr__(self):
        return f'Stats(folder="{self.folder}")'
