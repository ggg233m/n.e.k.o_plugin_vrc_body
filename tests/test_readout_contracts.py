# -*- coding: utf-8 -*-
"""S2 读出契约（2026-11-06）：4-query 视觉读出 / additive-gated fusion / 动作头输入口径。

为什么要有这组测试
------------------
* 读出模块会改变**动作头的输入维度**。旧 checkpoint 的 head.pt 是 1024 维；
  新增开关若默认打开、或推理侧按默认值重建，加载会「形状对得上但语义错了」
  或者干脆炸在 load_state_dict —— 两种都不能接受。
* ``--no-last-token`` 是「绕过 LM」的实质：head **只吃**读出输出。
  一旦 extra 为空，ActionHead 必须在 0 维 matmul 之前报错，而不是静默输出常数。
* 读出必须复用**同一次前向**里视觉塔输出（112×1024），不能另跑一次 visual。

torch 不存在时整组 skip（``tests/`` 下其它 numpy 级契约测试仍可在 .venv 跑）。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

try:  # .venv 没有 torch 是正常情况
    import torch  # noqa: F401
    import policy_readout  # noqa: E402
    _TORCH_OK = True
except Exception:  # noqa: BLE001
    policy_readout = None
    _TORCH_OK = False

import input_schema  # noqa: E402


@unittest.skipUnless(_TORCH_OK, "需要 torch（用 .venv_xpu 跑）")
class VisionQueryReadoutTests(unittest.TestCase):
    def test_param_count_about_263k(self):
        m = policy_readout.VisionQueryReadout(vis_dim=1024, q_dim=256, n_query=4)
        n = policy_readout.param_count(m)
        self.assertAlmostEqual(n, 263680, delta=2000,
                               msg="4-query 读出参数量应约 263k，实测 %d" % n)

    def test_forward_shape_and_out_dim(self):
        m = policy_readout.VisionQueryReadout(1024, 256, 4).eval()
        with torch.no_grad():
            y = m(torch.randn(3, 112, 1024))
        self.assertEqual(tuple(y.shape), (3, 1024))
        self.assertEqual(m.out_dim, 1024)

    def test_rejects_2d_tokens(self):
        m = policy_readout.VisionQueryReadout(1024, 256, 4)
        with self.assertRaises(ValueError):
            m(torch.randn(112, 1024))

    def test_attention_is_a_convex_combination(self):
        """池化输出必须是 token 的加权平均（proj 后），不能是任意数值。"""
        m = policy_readout.VisionQueryReadout(1024, 256, 4).eval()
        tok = torch.randn(2, 112, 1024)
        with torch.no_grad():
            y = m(tok)
            kv = m.vis_proj(tok)                     # (2,112,256)
            q = m.query[None].expand(2, -1, -1)
            w = (q @ kv.transpose(1, 2) / 256 ** 0.5).softmax(-1)
            ref = m.norm(w @ kv).reshape(2, -1)
        self.assertTrue(torch.allclose(y, ref, atol=1e-5))


@unittest.skipUnless(_TORCH_OK, "需要 torch（用 .venv_xpu 跑）")
class FusionTests(unittest.TestCase):
    def test_additive_sums_two_encoders(self):
        m = policy_readout.AdditiveFusion(1024, 8, 256, gated=False).eval()
        with torch.no_grad():
            y = m(torch.randn(4, 112, 1024), torch.randn(4, 8))
        self.assertEqual(tuple(y.shape), (4, 256))
        self.assertEqual(m.out_dim, 256)

    def test_gated_has_gate_and_different_params(self):
        g = policy_readout.AdditiveFusion(1024, 8, 256, gated=True)
        a = policy_readout.AdditiveFusion(1024, 8, 256, gated=False)
        self.assertIsNotNone(g.gate)
        self.assertIsNone(a.gate)
        self.assertGreater(policy_readout.param_count(g), policy_readout.param_count(a))

    def test_missing_numeric_is_loud(self):
        """缺数值通道不能静默当 0 —— 那等于把「缺数据」教成一个模态。"""
        m = policy_readout.AdditiveFusion(1024, 8, 256)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 112, 1024), None)

    def test_task_and_astar_slots_are_empty(self):
        m = policy_readout.AdditiveFusion(1024, 8, 256)
        self.assertIsNone(m.slots["task_card"])
        self.assertIsNone(m.slots["astar_path"])

    def test_build_readout_kinds(self):
        self.assertIsNone(policy_readout.build_readout("none"))
        self.assertIsInstance(policy_readout.build_readout("vision_query"),
                              policy_readout.VisionQueryReadout)
        self.assertIsInstance(policy_readout.build_readout("additive"),
                              policy_readout.AdditiveFusion)
        self.assertIsInstance(policy_readout.build_readout("gated"),
                              policy_readout.AdditiveFusion)
        with self.assertRaises(ValueError):
            policy_readout.build_readout("bogus")


@unittest.skipUnless(_TORCH_OK, "需要 torch（用 .venv_xpu 跑）")
class HeadInputContractTests(unittest.TestCase):
    def _head(self, **kw):
        import train_policy_lora as T
        return T.ActionHead(**kw)

    def test_default_is_hidden_only(self):
        h = self._head(hidden=1024, out=36, extra=0)
        self.assertEqual(h.in_dim, 1024)
        self.assertTrue(h.use_hidden)

    def test_extra_appends_dim(self):
        h = self._head(hidden=1024, out=36, extra=1024)
        self.assertEqual(h.in_dim, 2048)

    def test_no_last_token_uses_extra_only(self):
        h = self._head(hidden=1024, out=36, extra=1024, use_hidden=False)
        self.assertEqual(h.in_dim, 1024)
        with torch.no_grad():
            y = h(None, torch.randn(2, 1024))
        self.assertEqual(tuple(y.shape), (2, 36))

    def test_no_inputs_is_loud(self):
        h = self._head(hidden=1024, out=36, extra=0, use_hidden=False)
        self.assertEqual(h.in_dim, 0)
        with self.assertRaises(RuntimeError):
            h(None)

    def test_build_extra_heads_uses_readout_dim(self):
        import train_policy_lora as T
        ro = policy_readout.build_readout("vision_query", q_dim=256, n_query=4)
        inj, ev, extra, params = T.build_extra_heads(
            1024, "cpu", False, 0, (8, 14), False, 0.0, readout=ro)
        self.assertIsNone(inj)
        self.assertEqual(extra, 1024)
        self.assertEqual(sum(p.numel() for p in params), policy_readout.param_count(ro))

    def test_visual_injector_and_readout_are_mutually_exclusive(self):
        import train_policy_lora as T
        ro = policy_readout.build_readout("vision_query")
        with self.assertRaises(RuntimeError):
            T.build_extra_heads(1024, "cpu", True, 256, (8, 14), False, 0.0, readout=ro)


class StopHeadTests(unittest.TestCase):
    """stop_head 契约（2026-11-06）：cls==idle 监督、与 safety_gate **并联**。"""

    def test_param_count_is_small(self):
        m = policy_readout.StopHead(in_dim=1031, hidden=64)
        n = policy_readout.param_count(m)
        self.assertEqual(n, 1031 * 64 + 64 + 64 + 1)

    def test_forward_shape_and_range(self):
        m = policy_readout.StopHead(1031, 64).eval()
        with torch.no_grad():
            y = m(torch.randn(5, 1031))
        self.assertEqual(tuple(y.shape), (5,))
        p = policy_readout.stop_prob(y)
        self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))

    def test_rejects_wrong_input_dim(self):
        m = policy_readout.StopHead(1031, 64)
        with self.assertRaises(RuntimeError):
            m(torch.randn(2, 1024))

    def test_label_is_idle_only(self):
        self.assertEqual(policy_readout.stop_label("idle"), 1.0)
        for c in ("move", "trans", "none"):
            self.assertEqual(policy_readout.stop_label(c), 0.0)

    def test_disabled_returns_none(self):
        self.assertIsNone(policy_readout.build_stop_head(1031, enabled=False))

    def test_shares_head_feature_dim(self):
        """stop_head 必须吃与动作头**同一**特征向量（读出输出 + 数值 7）。"""
        readout = policy_readout.VisionQueryReadout(1024, 256, 4)
        stop = policy_readout.StopHead(readout.out_dim + 7, 64)
        self.assertEqual(stop.in_dim, readout.out_dim + 7)


class SchemaCompatibilityTests(unittest.TestCase):
    """新增开关不得让旧 checkpoint 的语义漂移。"""

    def test_legacy_defaults_are_off(self):
        for k, v in (("readout", "none"), ("readout_dim", 0), ("vision_query", 0),
                     ("no_last_token", False), ("use_hidden", True),
                     ("readout_only", False), ("card_version", 1),
                     ("numeric_dim", 8)):
            self.assertEqual(input_schema.LEGACY_DEFAULTS.get(k), v, k)

    def test_old_schema_load_gets_off_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            old = {"version": input_schema.SCHEMA_VERSION, "act_dim": 3,
                   "steps": 12, "step_s": 0.1, "prompt_format": "card"}
            (Path(td) / "input_schema.json").write_text(
                json.dumps(old), encoding="utf-8")
            s = input_schema.load(td)
        self.assertEqual(s["readout"], "none")
        self.assertEqual(s["readout_dim"], 0)
        self.assertFalse(s["no_last_token"])
        self.assertTrue(s["use_hidden"])
        self.assertEqual(s["card_version"], 1)
        self.assertEqual(s["numeric_dim"], 8)
        self.assertFalse(s["readout_only"])

    def test_schema_fields_roundtrip(self):
        if not _TORCH_OK:
            self.skipTest("需要 torch")
        m = policy_readout.VisionQueryReadout(1024, 256, 4)
        f = policy_readout.schema_fields(m, no_last_token=True)
        self.assertEqual(f["readout"], "vision_query")
        self.assertEqual(f["readout_dim"], m.out_dim)
        self.assertEqual(f["vision_query"], 4)
        self.assertEqual(f["vision_query_dim"], 256)
        self.assertTrue(f["no_last_token"])
        self.assertFalse(f["use_hidden"])


if __name__ == "__main__":
    unittest.main()
