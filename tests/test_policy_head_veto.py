# -*- coding: utf-8 -*-
"""PolicyHead 的近距障碍否决：接线正确性（不加载模型）。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PolicyHeadVetoTests(unittest.TestCase):
    """`_on_block` 在近距障碍时必须整块置零，且不吞掉 SafetyGate 的职责。"""

    def _head(self, near=None, dry_run=False):
        import importlib.util
        spec = importlib.util.spec_from_file_location('policy_head_mod', ROOT / 'backend' / 'policy_head.py')
        mod = importlib.util.module_from_spec(spec)
        # 绕开 PolicyRuntime 的真实构造（它要 torch + 权重）
        fake_exec = mock.MagicMock()
        with mock.patch.dict(sys.modules):
            for name, stub in (
                ('policy_executor', mock.MagicMock(SafetyGate=mock.MagicMock)),
                ('policy_runner', mock.MagicMock()),
                ('policy_runtime', mock.MagicMock()),
            ):
                sys.modules[name] = stub
            spec.loader.exec_module(mod)
        head = mod.PolicyHead.__new__(mod.PolicyHead)
        head.logger = None
        head.dry_run = dry_run
        head._log_dir = ROOT / '.tmp'
        head._send = None
        head._near_obstacle = near
        head.executor = fake_exec
        head.rt = SimpleNamespace(graph_enabled=False)
        return head, fake_exec

    def test_no_veto_when_path_is_clear(self) -> None:
        head, ex = self._head(near=lambda: (False, 3))
        block = [[0.1, 0.5]] * 12
        head._on_block(block, 1.0, {'gated': False})
        args, _ = ex.submit.call_args
        self.assertEqual(args[0], block)
        self.assertNotIn('vetoed', args[2] or {})

    def test_veto_zeroes_block_when_obstacle_is_near(self) -> None:
        head, ex = self._head(near=lambda: (True, 91))
        block = [[0.3, 1.1]] * 12
        head._on_block(block, 1.0, {})
        args, _ = ex.submit.call_args
        self.assertTrue(all(row == [0.0, 0.0] for row in args[0]))
        self.assertTrue(args[2]['vetoed'])
        self.assertEqual(args[2]['near_obstacle'], 91)

    def test_missing_provider_is_a_noop(self) -> None:
        head, ex = self._head(near=None)
        block = [[0.2, 0.8]] * 12
        head._on_block(block, 1.0, None)
        args, _ = ex.submit.call_args
        self.assertEqual(args[0], block)

    def test_provider_error_does_not_block_policy(self) -> None:
        """读不到感知数据不能变成「策略不动」—— 那是 executor stale 阈值管的事。"""
        def boom():
            raise RuntimeError('perception down')
        head, ex = self._head(near=boom)
        block = [[0.2, 0.8]] * 12
        head._on_block(block, 1.0, {})
        args, _ = ex.submit.call_args
        self.assertEqual(args[0], block)
        self.assertFalse((args[2] or {}).get('vetoed', False))


if __name__ == '__main__':
    unittest.main()