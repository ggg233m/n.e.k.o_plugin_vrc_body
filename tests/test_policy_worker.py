# -*- coding: utf-8 -*-
"""策略 worker 跨解释器接入的测试（2026-11-05）。

背景：主后端跑 `.venv`（无 torch），PolicyRuntime 必须 `.venv_xpu`。
架构沿用 stereo_backend 的「独立解释器子进程」先例。

本文件**不加载模型、不启动 VRChat**，只测协议与降级逻辑；
真实端到端（要 ~50 s 载模型）由 `VRC_POLICY_E2E=1` 显式开启。
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / "Scripts" / "python.exe"
XPU = ROOT / ".venv_xpu" / "Scripts" / "python.exe"

sys.path.insert(0, str(ROOT / "research" / "tools"))
sys.path.insert(0, str(ROOT))


class ResizeSingleSourceTests(unittest.TestCase):
    """`to_policy_rgb` 只能有一份实现（复制会导致几何静默漂移）。"""

    def test_runtime_reexports_same_function(self):
        import policy_resize
        import policy_runtime
        self.assertIs(policy_runtime.to_policy_rgb, policy_resize.to_policy_rgb,
                      "policy_runtime 没有 re-export policy_resize 的实现，可能出现两份")

    def test_resize_geometry(self):
        import numpy as np
        from policy_resize import to_policy_rgb
        a = np.full((180, 320, 3), 7, dtype=np.uint8)
        out = to_policy_rgb(a, (448, 256))
        self.assertEqual(out.shape, (256, 448, 3))
        self.assertEqual(out.dtype, np.uint8)
        # 内容高 = round(448*9/16)=252，上下各补 2 行黑边
        self.assertTrue((out[0] == 0).all(), "顶部应有黑边")
        self.assertTrue((out[-1] == 0).all(), "底部应有黑边")


class InterpreterLookupTests(unittest.TestCase):
    def test_finds_venv_xpu(self):
        from backend.policy_backend import find_worker_python
        py = find_worker_python()
        self.assertIsNotNone(py, "找不到 .venv_xpu 解释器")
        self.assertTrue(Path(py).exists())

    def test_env_override_wins(self):
        from backend import policy_backend as B
        old = os.environ.get("VRC_POLICY_PYTHON")
        try:
            os.environ["VRC_POLICY_PYTHON"] = r"C:\fake\python.exe"
            self.assertEqual(B.find_worker_python(), r"C:\fake\python.exe")
        finally:
            if old is None:
                os.environ.pop("VRC_POLICY_PYTHON", None)
            else:
                os.environ["VRC_POLICY_PYTHON"] = old


class MainEnvHasNoTorchTests(unittest.TestCase):
    """主后端环境（.venv）必须没有 torch —— 这是本设计的**前提**。

    如果哪天有人把 torch 装进 .venv，本设计就失去意义（也就不必降级）。
    """

    def test_venv_has_no_torch(self):
        if not VENV.exists():
            self.skipTest(".venv 不存在")
        p = subprocess.run([str(VENV), "-c", "import torch"], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0, ".venv 竟然有 torch（设计前提已变）")

    def test_venv_can_import_policy_head(self):
        """关键：policy_head 在 .venv 里必须能 import（此前模块级 import torch 会炸）。"""
        if not VENV.exists():
            self.skipTest(".venv 不存在")
        code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); "
                "import backend.policy_head as H; print('OK', hasattr(H, 'make_runtime'))")
        code = code % (str(ROOT / "research" / "tools"), str(ROOT))
        p = subprocess.run([str(VENV), "-c", code], capture_output=True, text=True, timeout=180)
        self.assertEqual(p.returncode, 0,
                         "policy_head 在 .venv 里 import 失败：%s" % (p.stderr[-800:],))
        self.assertIn("OK True", p.stdout)


class WorkerProtocolTests(unittest.TestCase):
    """worker 的协议健壮性（不载模型，只测握手前的错误路径）。"""

    def test_worker_rejects_bad_json(self):
        if not XPU.exists():
            self.skipTest(".venv_xpu 不存在")
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env.pop("PYTHONPATH", None)
        p = subprocess.run(
            [str(XPU), "-m", "backend.policy_worker"],
            cwd=str(ROOT), env=env, input="not json\n{\"op\":\"stop\"}\n",
            capture_output=True, text=True, timeout=120)
        out = [l for l in (p.stdout or "").splitlines() if l.strip()]
        self.assertTrue(out, "worker 没有任何协议输出")
        first = json.loads(out[0])
        self.assertFalse(first.get("ok"))
        self.assertIn("bad json", first.get("error", ""))

    def test_worker_protocol_channel_clean(self):
        """worker 的 stdout 必须**只有 JSON**（库的 print 要落到 stderr）。"""
        if not XPU.exists():
            self.skipTest(".venv_xpu 不存在")
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env.pop("PYTHONPATH", None)
        p = subprocess.run(
            [str(XPU), "-m", "backend.policy_worker"],
            cwd=str(ROOT), env=env, input="{\"op\":\"stop\"}\n",
            capture_output=True, text=True, timeout=120)
        for line in (p.stdout or "").splitlines():
            if not line.strip():
                continue
            json.loads(line)  # 非 JSON 会抛，测试即失败


class RemoteRuntimeNoPythonTests(unittest.TestCase):
    def test_raises_when_no_interpreter(self):
        from backend.policy_backend import RemotePolicyRuntime, PolicyWorkerError
        old = os.environ.get("VRC_POLICY_PYTHON")
        try:
            os.environ["VRC_POLICY_PYTHON"] = r"C:\definitely\missing\python.exe"
            # 显式传 python 会绕过查找，直接尝试启动 → 应抛 OSError/PolicyWorkerError
            with self.assertRaises(Exception):
                RemotePolicyRuntime("x", "y", python=r"C:\definitely\missing\python.exe")
        finally:
            if old is None:
                os.environ.pop("VRC_POLICY_PYTHON", None)
            else:
                os.environ["VRC_POLICY_PYTHON"] = old


if __name__ == "__main__":
    unittest.main()
