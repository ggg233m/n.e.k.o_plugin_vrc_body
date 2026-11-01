# -*- coding: utf-8 -*-
"""双图运行时（DualGraphRuntime）的纯逻辑单测：不加载模型、不碰 XPU。

这里锁的是**失败必须优雅回退**这条契约：图建不起来、形状不符、replay 抛错，
都必须返回 None 让调用方退回 eager，绝不能把异常带进控制环。
"""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))


def _fake_dual():
    """造一个不依赖 torch/XPU 的 DualGraphRuntime 实例。"""
    import policy_dual_graph as m
    obj = m.DualGraphRuntime.__new__(m.DualGraphRuntime)
    obj.model = mock.MagicMock()
    obj.visual = mock.MagicMock()
    obj.lm = mock.MagicMock()
    obj.max_len = 192
    obj.enabled = True
    obj.ready = False
    obj.replays = 0
    obj.fallbacks = 0
    obj.setup_error = None
    obj.setup_seconds = 0.0
    obj.geom = None
    obj.split_sizes = None
    obj._restore_attn = None
    obj._gvit = None
    obj._glm = None
    obj._pv = None
    obj._ids = None
    obj._mask = None
    obj._mtt = None
    obj._emb = None
    obj._pos = None
    obj._vit_out = None
    obj._lm_out = None
    return obj


class DualGraphFallbackTests(unittest.TestCase):
    def test_forward_without_setup_returns_none(self) -> None:
        """没建图时必须回退，而不是抛 AttributeError。"""
        obj = _fake_dual()
        out = obj.forward({"pixel_values_videos": None})
        self.assertIsNone(out)
        self.assertEqual(obj.fallbacks, 1)

    def test_disabled_returns_none(self) -> None:
        obj = _fake_dual()
        obj.enabled = False
        self.assertIsNone(obj.forward({}))

    def test_shape_mismatch_falls_back(self) -> None:
        """形状不符必须回退 —— 绝不在交互时新建/重建图。"""
        import torch
        obj = _fake_dual()
        obj.ready = True
        obj._pv = torch.zeros((448, 1536))
        obj._ids = torch.zeros((1, 192), dtype=torch.long)
        obj._mask = torch.zeros((1, 192), dtype=torch.long)
        b = {"pixel_values_videos": torch.zeros((320, 1536)),
             "input_ids": torch.zeros((1, 192), dtype=torch.long),
             "attention_mask": torch.zeros((1, 192), dtype=torch.long)}
        self.assertIsNone(obj.forward(b))
        self.assertEqual(obj.fallbacks, 1)

    def test_replay_error_disables_permanently(self) -> None:
        """replay 抛错要置 enabled=False 并永久回退，不反复重试。"""
        import torch
        obj = _fake_dual()
        obj.ready = True
        obj._pv = torch.zeros((448, 1536))
        obj._ids = torch.zeros((1, 192), dtype=torch.long)
        obj._mask = torch.zeros((1, 192), dtype=torch.long)
        obj._gvit = mock.MagicMock()
        obj._gvit.replay.side_effect = RuntimeError("device lost")
        b = {"pixel_values_videos": torch.zeros((448, 1536)),
             "input_ids": torch.zeros((1, 192), dtype=torch.long),
             "attention_mask": torch.zeros((1, 192), dtype=torch.long)}
        self.assertIsNone(obj.forward(b))
        self.assertFalse(obj.enabled)
        self.assertFalse(obj.ready)
        self.assertIn("device lost", obj.setup_error)

    def test_teardown_restores_attention_patch(self) -> None:
        """撤销补丁必须可重入且幂等。"""
        obj = _fake_dual()
        called = []
        obj._restore_attn = lambda: called.append(1)
        obj.teardown()
        obj.teardown()
        self.assertEqual(called, [1])
        self.assertIsNone(obj._restore_attn)


class PolicyRuntimeInnerTests(unittest.TestCase):
    """`_inner_module()` 必须兼容合并前后的两种层级。"""

    def test_merged_and_unmerged_paths(self) -> None:
        import policy_runtime as pr
        obj = pr.PolicyRuntime.__new__(pr.PolicyRuntime)
        sentinel_merged = object()
        obj.merged = True
        obj.model = types.SimpleNamespace(base_model=sentinel_merged)
        self.assertIs(obj._inner_module(), sentinel_merged)

        lora = types.SimpleNamespace(model=types.SimpleNamespace(model="Qwen3_5Model"))
        obj.merged = False
        obj.model = types.SimpleNamespace(base_model=lora)
        self.assertEqual(obj._inner_module(), "Qwen3_5Model")


if __name__ == "__main__":
    unittest.main()