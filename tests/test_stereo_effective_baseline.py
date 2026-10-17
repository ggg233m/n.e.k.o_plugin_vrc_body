# -*- coding: utf-8 -*-
"""把「生效基线探针」的判据钉住 —— 尤其是那一步俯仰修正。

为什么值得写测试
----------------
这个工具的量（rho = B/h）被用来裁「IPD 到底有没有被 avatar 缩放」，
而它**很容易被写成错的形式**：2026-10-18 的草稿版漏了 `1/cos(theta)`，
得出的比值是 1.134 而不是 1.001 —— 方向相反。所以这里不只测「能跑」，
而是专门测**少了修正就会失败**（`test_pitch_correction_is_load_bearing`）。

⚠️ 边界：这里的「真值」是**解析地板模型**造出来的视差场，不是物理真值。
它验的是「估计器能不能从地板视差场里把 rho 解出来」，**不验**「渲染确实用了这个 B」。
后者只能靠实拍 + 外部锚，见 Docs/图像视差实测-生效基线随头像缩放（2026-10-18）.md §五。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "research" / "tools" / "stereo_effective_baseline.py"


def _load_tool():
    """按路径加载工具模块（它不在包里，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location("_seb", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_seb"] = mod
    spec.loader.exec_module(mod)
    return mod


def _called_attrs(path: Path) -> set[str]:
    """AST 扫出文件里**实际调用的** `a.b` 形式属性名（不含注释与 docstring）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            val = getattr(node.func.value, "id", None)
            if val:
                out.add("%s.%s" % (val, node.func.attr))
    return out


SEB = _load_tool()


def synth_floor(B: float, h: float, pitch_deg: float, *, shape=(405, 720),
                noise_px: float = 0.0, seed: int = 0,
                wall_frac: float = 0.0) -> np.ndarray:
    """按解析模型造一张地板视差图（可选噪声 + 上半幅假墙面）。

    d(v) = (B*cos(t)/h)*(v-cy) + fx*B*sin(t)/h
    """
    hh, ww = shape
    cy = hh / 2.0
    t = np.radians(pitch_deg)
    v = np.arange(hh, dtype=np.float64)[:, None]
    d = (B * np.cos(t) / h) * (v - cy) + (SEB.FX * B * np.sin(t) / h)
    d = np.repeat(d, ww, axis=1)
    rng = np.random.default_rng(seed)
    if noise_px:
        d = d + rng.normal(0.0, noise_px, d.shape)
    if wall_frac:
        # 远端墙面：视差恒定（不是地板），用来测 RANSAC 会不会被带偏
        w = int(hh * wall_frac)
        d[:w, :] = 2.0
    return d.astype(np.float32)


class FloorSlopeEstimatorTests(unittest.TestCase):
    def test_recovers_rho_on_synthetic_ground_truth(self) -> None:
        """已知 B/h，噪声 0.1 px —— 必须解回同一个 rho。"""
        for B, h in ((0.126, 1.76), (0.140, 1.40), (0.063, 1.10)):
            d = synth_floor(B, h, 0.0, noise_px=0.1, seed=3)
            r = SEB.fit_floor_rho(d, d.shape[0] / 2.0, SEB.FX)
            self.assertIsNotNone(r, "解析地板都拟合不出来")
            self.assertAlmostEqual(r["rho"], B / h, delta=0.02 * B / h)

    def test_rho_is_scale_invariant(self) -> None:
        """把 B 和 h 同乘 k，rho 必须不动 —— 这是跨头像比较的全部依据。"""
        base = SEB.fit_floor_rho(synth_floor(0.126, 1.76, 12.0, noise_px=0.05, seed=1),
                                 405 / 2.0, SEB.FX)["rho"]
        for k in (0.5, 1.113, 2.0, 4.0):
            d = synth_floor(0.126 * k, 1.76 * k, 12.0, noise_px=0.05, seed=1)
            got = SEB.fit_floor_rho(d, 405 / 2.0, SEB.FX)["rho"]
            self.assertAlmostEqual(got / base, 1.0, delta=0.02,
                                   msg="k=%.3f 时 rho 变了 ⇒ 尺度无关性被破坏" % k)

    def test_pitch_correction_is_load_bearing(self) -> None:
        """**核心**：少了 1/cos(theta) 就会偏 —— 证明那一步不是装饰。

        不修正时读数是 `s`（而非 `s/cos t`）；预期偏差 `cos t - 1`。
        在 t=18° 时偏差约 5%，不修正的比值 1.134 就是这么来的（那边 t≈14°、且
        还有第二个偏差源叠加）。这条测试保证**谁把修正删了都会红**。
        """
        for pitch in (14.0, 18.0):
            d = synth_floor(0.126, 1.76, pitch, noise_px=0.05, seed=5)
            r = SEB.fit_floor_rho(d, 405 / 2.0, SEB.FX)
            true = 0.126 / 1.76
            naive = r["slope"]                       # 未修正的读数
            self.assertAlmostEqual(r["rho"], true, delta=0.02 * true,
                                   msg="修正后的 rho 应当准")
            rel = abs(naive - true) / true
            self.assertGreater(rel, 0.02,
                               msg="俯仰 %.0f° 下未修正的读数居然没偏 ⇒ 本判据没有分辨力" % pitch)

    def test_robust_to_far_wall_contamination(self) -> None:
        """上半幅有恒定视差的墙面时，仍要解对（RANSAC 的用途）。"""
        clean = SEB.fit_floor_rho(synth_floor(0.126, 1.76, 12.0, noise_px=0.05, seed=2),
                                  405 / 2.0, SEB.FX)["rho"]
        dirty = SEB.fit_floor_rho(
            synth_floor(0.126, 1.76, 12.0, noise_px=0.05, seed=2, wall_frac=0.45),
            405 / 2.0, SEB.FX)["rho"]
        self.assertAlmostEqual(dirty / clean, 1.0, delta=0.02,
                               msg="远端墙面把地板拟合带偏了")

    def test_rejects_a_frame_with_no_floor(self) -> None:
        """整幅都是常值视差（没有地板）时必须返回 None，不能瞎给一个 rho。"""
        d = np.full((405, 720), 3.0, np.float32)
        self.assertIsNone(SEB.fit_floor_rho(d, 405 / 2.0, SEB.FX))
        d2 = np.zeros((405, 720), np.float32)          # 全无效
        self.assertIsNone(SEB.fit_floor_rho(d2, 405 / 2.0, SEB.FX))


class GuardrailTests(unittest.TestCase):
    def test_reads_non_ascii_paths(self) -> None:
        """代码里**不得有** cv2.imread 调用 —— 它在非 ASCII 路径上静默返回 None（踩过）。

        ⚠️ 只查**真调用**（走 AST），不扫文本：docstring 里正当地提到了这个坑，
        纯文本断言会把它当成违规（第一版就是这么红的）。
        """
        called = _called_attrs(_TOOL)
        self.assertNotIn("cv2.imread", called,
                         "cv2.imread 在非 ASCII 路径上静默返回 None —— 见模块 docstring")
        self.assertIn("np.fromfile", called, "含中文路径的录制会读不到")
        self.assertIn("cv2.imdecode", called)

    def test_declared_baseline_matches_the_recorded_value(self) -> None:
        """工具里写死的声明基线必须等于录制里实际记下来的那个数。

        它 15/15 场逐位相同；写错的话 h_track = B/rho 会整体偏。
        """
        self.assertEqual(SEB.B_DECLARED, 0.126)
        # float32(0.126) 的两个常见写法都应落在这个值附近，误差不该超过 1e-6
        self.assertAlmostEqual(SEB.B_DECLARED, 2 * float(np.float32(0.063)), places=6)

    def test_ratio_sigma_uses_a_ratio_bootstrap(self) -> None:
        """比值的不确定度必须**直接自举比值**，不能把两场半宽拼起来。

        2026-10-18 的草稿把两场的**上侧**半宽 hypot 起来，
        而本仓自举 CI 是不对称的（下侧更远）⇒ sigma 被放大近 2 倍。
        这里钉住：比较函数里出现的是比值分布，而不是半个宽度的组合。
        """
        src = _TOOL.read_text(encoding="utf-8")
        head = src[src.index("def report_compare"):]
        head = head[:head.index("def report_absolute")]
        self.assertIn("b_rho", head, "比值自举不见了")
        self.assertIn("rng.choice(b_rho", head, "必须重抽样逐帧 rho 直接构造比值分布")
        self.assertNotIn("_rel_ci(base)", head,
                         "又回到了「拼两个半宽」的老算法（会把 sigma 放大近 2 倍）")


if __name__ == "__main__":
    unittest.main()