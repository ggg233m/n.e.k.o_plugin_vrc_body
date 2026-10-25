"""视差后端选择：SGBM 适配器行为不变；LAS2 载不起来时会话退回 SGBM、不丢双目。

这些用例在 `.venv`（无 torch）里跑，所以 LAS2 只测「缺依赖 ⇒ 退回」这一侧；
LAS2 本体的数值与编译在 `.venv_las` 里另测（见 LAS2 doc）。
"""
from __future__ import annotations

import ast
import time
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


if __name__ == "__main__":
    unittest.main()
