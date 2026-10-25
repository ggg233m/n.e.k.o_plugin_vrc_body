"""视差后端选择：SGBM 适配器行为不变；LAS2 载不起来时会话退回 SGBM、不丢双目。

这些用例在 `.venv`（无 torch）里跑，所以 LAS2 只测「缺依赖 ⇒ 退回」这一侧；
LAS2 本体的数值与编译在 `.venv_las` 里另测（见 LAS2 doc）。
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import threading
import time
from dataclasses import replace
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body import config
from neko_anyadance_body.backend import nav_online, stereo_backend
from neko_anyadance_body.backend.nav_mapping import make_sgbm, stereo_disparity
from neko_anyadance_body.backend.nav_online import OnlineNavConfig
from tests.test_nav_online import Harness, ground_and_wall

ROOT = Path(__file__).resolve().parents[1]


def _pair(h: int = 64, w: int = 128, shift: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    right = (rng.random((h, w + shift)) * 255).astype(np.uint8)
    left = np.ascontiguousarray(right[:, shift:])
    return left, np.ascontiguousarray(right[:, :w])


class SGBMAdapterTest(unittest.TestCase):
    def test_matches_direct_sgbm(self) -> None:
        left, right = _pair()
        m = stereo_backend.make_stereo_backend("sgbm")
        np.testing.assert_array_equal(m.compute(left, right), stereo_disparity(left, right, make_sgbm(64)))

    def test_num_disparities_scale_with_fx(self) -> None:
        for fx, nd in ((202.5, 64), (303.75, 96), (405.0, 128)):
            d = stereo_backend.make_stereo_backend("sgbm", fx=fx).describe()
            self.assertEqual(d, {"kind": "sgbm", "num_disparities": nd})

    def test_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            stereo_backend.make_stereo_backend("crestereo")


class ImportIsolationTest(unittest.TestCase):
    def test_torch_only_reached_through_las2_branch(self) -> None:
        """发行包没有 torch：backend 里只有 las2_model 可以在模块顶层 import torch/timm，
        且 las2_model 只能在函数体里被引用。"""
        heavy = {"torch", "timm"}
        for p in (ROOT / "backend").rglob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in tree.body:
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    if p.stem != "las2_model":
                        self.assertNotIn(n.split(".")[0], heavy, f"{p.name}: top-level import {n}")
                        self.assertNotEqual(n.rsplit(".", 1)[-1], "las2_model", f"{p.name}: top-level import {n}")


class ConfigTest(unittest.TestCase):
    def test_default_and_override(self) -> None:
        self.assertEqual(OnlineNavConfig().stereo_backend, "las2")
        nav = config.PluginConfig.from_mapping({"navmesh": {"stereo_backend": " LAS2 "}}).navmesh
        self.assertEqual(OnlineNavConfig.from_plugin(nav).stereo_backend, "las2")

    def test_rejects_unknown(self) -> None:
        for bad in ("crestereo", "", 3):
            with self.assertRaises(ValueError, msg=repr(bad)):
                config.PluginConfig.from_mapping({"navmesh": {"stereo_backend": bad}})


class FallbackTest(unittest.TestCase):
    def _run(self, kind: str, decoupled: bool = True) -> dict:
        h = Harness(armed=False, vz=0.5)
        h.nav.cfg.stereo_backend = kind
        h.nav.cfg.decoupled = decoupled
        real = stereo_backend.make_stereo_backend

        def make(k: str, **kw):
            if k == "las2":
                raise ModuleNotFoundError("No module named 'torch'")
            return real(k, **kw)

        sgbm_out = np.zeros((4, 4), np.float32)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)), \
                mock.patch.object(nav_online, "make_stereo_backend", side_effect=make), \
                mock.patch.object(stereo_backend._SGBMAdapter, "compute", return_value=sgbm_out):
            h.nav.start()
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    st = h.nav.status()
                    if st["stereo_backend"]["active"] is not None and st["stereo_ms"] is not None:
                        break
                    time.sleep(0.05)
            finally:
                h.nav.stop()
        return st

    def test_las2_unavailable_falls_back_to_sgbm(self) -> None:
        st = self._run("las2")
        self.assertEqual(st["stereo_backend"]["configured"], "las2")
        self.assertEqual(st["stereo_backend"]["active"], "sgbm")
        self.assertIsNotNone(st["stereo_ms"])          # 双目仍在出帧
        self.assertTrue(any(e.startswith("stereo_backend: ModuleNotFoundError") for e in st["errors"]),
                        st["errors"])

    def test_las2_refused_on_inline_path(self) -> None:
        st = self._run("las2", decoupled=False)
        self.assertEqual(st["stereo_backend"]["active"], "sgbm")
        self.assertTrue(any("requires decoupled=True" in e for e in st["errors"]), st["errors"])

    def test_sgbm_default_reports_no_error(self) -> None:
        st = self._run("sgbm")
        self.assertEqual(st["stereo_backend"], {"configured": "sgbm", "active": "sgbm", "num_disparities": 64})
        self.assertFalse(any(e.startswith("stereo_backend") for e in st["errors"]), st["errors"])

    def test_stage_timing_reported(self) -> None:
        st = self._run("sgbm")
        sg = st["stereo_stages"]
        self.assertGreaterEqual(sg["frames"], 1)
        for k in ("disp", "points", "near", "total", "disp_cpu", "points_cpu", "near_cpu", "n_pts"):
            med, p90 = sg[k]
            self.assertLessEqual(med, p90, k)
        self.assertNotIn("las2_sync", sg)                 # SGBM 没有 last_timing
        self.assertGreater(sg["switchinterval_ms"], 0.0)



class _FakeLAS2:
    """预热用的假适配器：记下首帧尺寸，可选地卡住直到放行（模拟编译耗时）。"""
    def __init__(self, compiled: bool = True, gate: "threading.Event | None" = None) -> None:
        self.shapes: list[tuple] = []
        self.compiled, self.gate = compiled, gate

    def compute(self, l, r):
        if self.gate is not None:
            self.gate.wait(5.0)
        self.shapes.append(l.shape)
        return np.zeros(l.shape[:2], np.float32)

    def describe(self) -> dict:
        return {"kind": "las2", "compiled": self.compiled, "compile_error": None if self.compiled else "no cl"}


class PrewarmTest(unittest.TestCase):
    def setUp(self) -> None:
        stereo_backend._warm.clear()
        stereo_backend._warm["state"] = "idle"
        self.addCleanup(stereo_backend._warm.clear)
        self.addCleanup(stereo_backend._warm.update, state="idle")

    def _wait(self, *states: str) -> dict:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            st = stereo_backend.las2_prewarm_status()
            if st["state"] in states:
                return st
            time.sleep(0.01)
        self.fail(f"prewarm stuck at {stereo_backend.las2_prewarm_status()}")

    def test_compiles_once_in_background_and_reports_progress(self) -> None:
        gate = threading.Event()
        fake = _FakeLAS2(gate=gate)
        with mock.patch.object(stereo_backend, "make_stereo_backend", return_value=fake) as mk:
            first = stereo_backend.las2_prewarm((606, 1080))
            self.assertIn(first["state"], ("loading", "compiling"))       # 立即返回，不等编译
            self._wait("compiling")
            again = stereo_backend.las2_prewarm((606, 1080))              # 重复调用不另起线程
            self.assertEqual(again["state"], "compiling")
            gate.set()
            st = self._wait("ready")
        self.assertEqual(mk.call_count, 1)
        self.assertEqual(fake.shapes, [(606, 1080, 3)])                   # 三通道：LAS2 吃 RGB
        self.assertEqual(st["shape"], [606, 1080])
        self.assertIsNone(st["error"])
        self.assertGreaterEqual(st["elapsed_s"], 0.0)

    def test_eager_fallback_is_reported(self) -> None:
        with mock.patch.object(stereo_backend, "make_stereo_backend", return_value=_FakeLAS2(compiled=False)):
            stereo_backend.las2_prewarm((405, 720))
            st = self._wait("eager")
        self.assertEqual(st["error"], "no cl")

    def test_load_failure_is_reported_and_retried(self) -> None:
        with mock.patch.object(stereo_backend, "make_stereo_backend",
                               side_effect=ModuleNotFoundError("No module named 'torch'")):
            stereo_backend.las2_prewarm((606, 1080))
            st = self._wait("failed")
        self.assertIn("ModuleNotFoundError", st["error"])
        with mock.patch.object(stereo_backend, "make_stereo_backend", return_value=_FakeLAS2()):
            stereo_backend.las2_prewarm((606, 1080))                      # failed 允许重试
            self._wait("ready")

    def test_navigator_gate_and_status(self) -> None:
        h = Harness(armed=False)
        with mock.patch.object(nav_online, "las2_prewarm", return_value={"state": "loading"}) as pw:
            h.nav.cfg = h.nav._cfg_user = replace(h.nav.cfg, stereo_backend="sgbm")
            self.assertIsNone(h.nav.prewarm())
            h.nav.cfg = h.nav._cfg_user = replace(h.nav.cfg, stereo_backend="las2", stereo_prewarm=False)
            self.assertIsNone(h.nav.prewarm())
            h.nav.cfg = h.nav._cfg_user = replace(h.nav.cfg, stereo_prewarm=True, decoupled=False)
            self.assertIsNone(h.nav.prewarm())
            h.nav.cfg = h.nav._cfg_user = replace(h.nav.cfg, decoupled=True)
            self.assertEqual(h.nav.prewarm(), {"state": "loading"})
        pw.assert_called_once()                                           # 现行驱动 1080 档的采集尺寸
        self.assertEqual(pw.call_args.args[0], (606, 1080), pw.call_args)  # 形状是第一个位置参数
        self.assertEqual(h.nav.stereo_backend_status()["prewarm"]["state"], "idle")

    def test_prewarm_shape_matches_las2_padding_bucket(self) -> None:
        # 编译缓存按补到 32 倍数后的尺寸命中：猜的尺寸必须和真实采集尺寸落同一档。
        real = {720: (405, 720), 1080: (606, 1080), 1440: (810, 1440)}
        h = Harness(armed=False)
        for w, (rh, rw) in real.items():
            h.nav._cfg_user = replace(h.nav.cfg, capture_width=w)
            gh, gw = h.nav.prewarm_shape()
            self.assertEqual((gh + (-gh) % 32, gw + (-gw) % 32), (rh + (-rh) % 32, rw + (-rw) % 32), w)

    def test_config_key(self) -> None:
        self.assertTrue(OnlineNavConfig().stereo_prewarm)
        nav = config.PluginConfig.from_mapping({"navmesh": {"stereo_prewarm": False}}).navmesh
        self.assertFalse(OnlineNavConfig.from_plugin(nav).stereo_prewarm)

class SubprocessBackendTest(unittest.TestCase):
    """子进程模式的两条**父进程侧**契约：起不来立刻报错、prewarm 状态走 loading→compiling→ready。

    这里 mock 掉真进程：测的是「状态机与失败面」，不是 IPC 本体
    （IPC 见 tests/test_stereo_worker_ipc.py，那边真起子进程、假引擎逐像素验）。
    """

    def setUp(self) -> None:
        # ⚠️ 子进程适配器是**进程级单例**（stereo_backend._sub）：同形状同引擎会直接复用。
        # 不先关掉，上一个用例留下的活 worker 会让「起不来」用例根本走不到 spawn。
        stereo_backend.close_las2()
        self.addCleanup(stereo_backend.close_las2)
        self.addCleanup(stereo_backend._warm.clear)
        self.addCleanup(stereo_backend._warm.update, state="idle")
        stereo_backend._warm.clear()
        stereo_backend._warm["state"] = "idle"

    def test_unusable_interpreter_raises_stereo_worker_error(self) -> None:
        """找不到带 torch 的解释器 ⇒ StereoWorkerError（不是 ModuleNotFoundError 漏出去）。

        release 包装的是没有 torch 的 py3.11；子进程模式必须把「起不来 / 没 torch」
        统一收敛成 StereoWorkerError，会话才能按一条路径退回 SGBM。
        这里用 NEKO_LAS_PYTHON 指向当前解释器 + 假的 torch 包来构造该失败面，
        不依赖本机 .venv_las 装没装 torch。
        """
        with tempfile.TemporaryDirectory() as tmp:
            # 子进程真的执行 python -m backend.stereo_worker（cwd = 仓库根，模块找得到），
            # 但 PYTHONPATH 里塞一个抛 ImportError 的假 torch，保证「没有 torch」与宿主机无关。
            fake = Path(tmp) / "torch"
            fake.mkdir()
            (fake / "__init__.py").write_text("raise ImportError('no torch here')\n", encoding="utf-8")
            env = {"NEKO_LAS_PYTHON": sys.executable, "PYTHONPATH": tmp}
            with mock.patch.dict(os.environ, env):
                with self.assertRaises(RuntimeError) as ctx:
                    stereo_backend.make_stereo_backend("las2", subprocess=True, shape=(64, 128))
        self.assertIsInstance(ctx.exception, stereo_backend.StereoWorkerError)

    def test_prewarm_state_runs_loading_compiling_ready(self) -> None:
        """子进程模式：state 必须被观察到 loading → compiling → ready（父进程轮询的真实序列）。"""
        seen: list[str] = []
        released = threading.Event()
        calls: list[tuple] = []

        class _FakeAdapter:
            """假 worker 适配器：compute 卡住到用例放行，模拟子进程正在编译。"""

            def compute(self, left, right):
                calls.append(tuple(left.shape))
                released.wait(5.0)
                return np.zeros(left.shape[:2], np.float32)

            def describe(self) -> dict:
                return {"kind": "las2", "subprocess": True, "alive": True, "compiled": True,
                        "compile_error": None, "pid": 4242, "frames": 1, "timeouts": 0, "last_ms": 1.0}

            def prewarm_state(self) -> dict:
                return {"state": "ready", "error": None}

            def close(self) -> None:
                pass

        def fake_make(kind: str, **kw):
            self.assertEqual(kind, "las2")
            self.assertTrue(kw.get("subprocess"), f"子进程模式没传下去：{kw}")
            return _FakeAdapter()

        def pump() -> dict:
            st = stereo_backend.las2_prewarm_status()
            seen.append(st["state"])
            return st

        with mock.patch.object(stereo_backend, "make_stereo_backend", side_effect=fake_make):
            first = stereo_backend.las2_prewarm((606, 1080), subprocess=True)
            seen.append(first["state"])                    # 立即返回：loading（或已跳到 compiling）
            deadline = time.monotonic() + 5.0
            st = first
            while time.monotonic() < deadline and "compiling" not in seen:
                time.sleep(0.01)
                st = pump()
            self.assertIn("compiling", seen, f"没观察到 compiling：{seen}")
            released.set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                st = pump()
                if st["state"] == "ready":
                    break
                time.sleep(0.01)

        self.assertEqual(st["state"], "ready", st)
        self.assertEqual(tuple(st["shape"]), (606, 1080), st)
        self.assertTrue(calls and calls[0] == (606, 1080, 3), "子进程模式同样喂三通道 RGB")
        # 序列必须是「先 loading、再 compiling、最后 ready」，不许跳步或回退
        order = [s for s in seen if s in ("loading", "compiling", "ready")]
        deduped = [s for i, s in enumerate(order) if i == 0 or s != order[i - 1]]
        self.assertEqual(deduped[0], "loading", seen)
        self.assertIn("compiling", deduped, seen)
        self.assertEqual(deduped[-1], "ready", seen)
        self.assertLess(deduped.index("compiling"), deduped.index("ready"), seen)

if __name__ == "__main__":
    unittest.main()

