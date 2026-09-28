from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import nav_loop
from neko_anyadance_body.backend.nav_loop import (M_OPT, KeyframeFeatures, LoopCloser, LoopConfig,
                                                  relative_pose)

S = 0.755
K = np.array([[202.5, 0.0, 360.0], [0.0, 202.5, 202.5], [0.0, 0.0, 1.0]])
HALF_B = 0.063


def yaw_R(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose(x: float, y: float, yaw_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = yaw_R(yaw_deg)
    T[:2, 3] = (x, y)
    return T


def project(P_head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """头部 base 系点 → 左眼像素；返回 (uv, 在图内的掩码)。"""
    X = (P_head - np.array([0.0, HALF_B, 0.0])) @ M_OPT      # 行向量：X_opt = M_OPTᵀ (P − e)
    uvw = X @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    ok = (X[:, 2] > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < 720) & (uv[:, 1] >= 0) & (uv[:, 1] < 405)
    return uv, ok


def scene_pair(R_ab: np.ndarray, t_ab: np.ndarray, n: int = 600, seed: int = 0):
    """a 看到的 3D 点 + 同一批点在 b 里的像素，描述子一致。"""
    rng = np.random.default_rng(seed)
    P = np.column_stack([rng.uniform(1.5, 6.0, n), rng.uniform(-3.0, 3.0, n), rng.uniform(-1.7, 1.0, n)])
    uv_a, ok_a = project(P)
    Pb = (P - t_ab) @ R_ab                                    # b 头部系：R_abᵀ (P − t_ab)
    uv_b, ok_b = project(Pb)
    keep = ok_a & ok_b
    P, uv_a, uv_b = P[keep], uv_a[keep], uv_b[keep]
    des = rng.integers(0, 256, (len(P), 32), dtype=np.uint8)
    a = KeyframeFeatures(uv_a.astype(np.float32), des, P.astype(np.float32), des, K, (720, 405), HALF_B)
    b = KeyframeFeatures(uv_b.astype(np.float32), des, P.astype(np.float32), des, K, (720, 405), HALF_B)
    return a, b


class RelativePoseTest(unittest.TestCase):
    def test_recovers_known_offset_including_eye_lever_arm(self) -> None:
        R_ab, t_ab = yaw_R(20.0), np.array([0.8, -0.4, 0.05])
        a, b = scene_pair(R_ab, t_ab)
        rel, why = relative_pose(a, b, LoopConfig())
        self.assertEqual(why, "ok")
        np.testing.assert_allclose(rel["t_ab"], t_ab, atol=0.01)
        np.testing.assert_allclose(rel["R_ab"], R_ab, atol=1e-3)
        self.assertGreaterEqual(rel["inliers"], 50)

    def test_unrelated_descriptors_rejected(self) -> None:
        a, b = scene_pair(np.eye(3), np.zeros(3))
        b.des = np.random.default_rng(9).integers(0, 256, b.des.shape, dtype=np.uint8)
        rel, why = relative_pose(a, b, LoopConfig())
        self.assertIsNone(rel)
        self.assertEqual(why, "few_matches")


def feat() -> KeyframeFeatures:
    return KeyframeFeatures(np.zeros((100, 2), np.float32), np.zeros((100, 32), np.uint8),
                            np.zeros((100, 3), np.float32), np.zeros((100, 32), np.uint8), K, (720, 405), HALF_B)


OK_REL = {"inliers": 80, "coverage": 0.3, "reproj_px": 0.8}


def walk(lc: LoopCloser, pts_true, drift_frac: float, rel_fn):
    """沿真值折线走，航位推算每米向 +y 漂 drift_frac。``rel_fn(a, b)`` 代替 relative_pose，
    a、b 是关键帧序号（按特征对象反查）。"""
    feats: list[KeyframeFeatures] = []

    def rel(map_kf, query, cfg):
        return rel_fn(next(i for i, f in enumerate(feats) if f is map_kf),
                      next(i for i, f in enumerate(feats) if f is query))

    dist, dr = 0.0, np.array(pts_true[0], float) / S
    found = []
    with mock.patch.object(nav_loop, "relative_pose", side_effect=rel):
        for k, p in enumerate(pts_true):
            if k:
                step = (np.array(p) - np.array(pts_true[k - 1])) / S
                seg = float(np.hypot(*step)) * S
                dist += seg
                dr = dr + step + np.array([0.0, drift_frac * seg]) / S
            feats.append(feat())
            found += lc.add_keyframe(k, pose(*dr), dist, feats[-1])
    return found


def square_loop(n_side: int = 8, side_m: float = 4.0):
    pts = []
    for (x0, y0), (x1, y1) in zip([(0, 0), (side_m, 0), (side_m, side_m), (0, side_m)],
                                  [(side_m, 0), (side_m, side_m), (0, side_m), (0, 0)]):
        for i in range(n_side):
            f = i / n_side
            pts.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    pts.append((0.0, 0.0))
    return pts


class LoopCloserTest(unittest.TestCase):
    def test_revisit_pulls_drifted_pose_back(self) -> None:
        pts = square_loop()
        last = len(pts) - 1
        lc = LoopCloser(LoopConfig())

        def rel(a, b):
            # 只有最后一帧与第 0 帧对上：同一地点，相对位姿为零。
            if (a, b) == (0, last):
                return {"R_ab": np.eye(3), "t_ab": np.zeros(3), **OK_REL}, "ok"
            return None, "few_matches"

        found = walk(lc, pts, 0.05, rel)
        self.assertEqual([(f["a"], f["b"]) for f in found], [(0, last)])
        drift = 0.05 * 16.0                                   # 16 m 路程
        self.assertAlmostEqual(found[0]["correction_m"], drift, delta=0.02)
        # 回环 σ 0.10 m 对里程计链的累计 σ 约 0.25 m：残差留约 13%。
        self.assertLess(float(np.hypot(*(lc.pose(last)[:2, 3] * S))), 0.15)
        np.testing.assert_allclose(lc.offset() * S, [0.0, -drift], atol=0.15)
        mid = lc.pose(16)[:2, 3] * S                          # 对角 (4, 4)：修掉约一半
        self.assertLess(abs(mid[1] - 4.0), 0.3 * drift)

    def test_rotation_mismatch_rejected(self) -> None:
        lc = LoopCloser(LoopConfig())

        def rel(a, b):
            return {"R_ab": yaw_R(15.0), "t_ab": np.zeros(3), **OK_REL}, "ok"
        found = walk(lc, square_loop(), 0.05, rel)
        self.assertEqual(found, [])
        self.assertGreater(lc.rejects.get("rotation_mismatch", 0), 0)
        self.assertEqual(float(np.hypot(*lc.offset())), 0.0)
        checks = lc.status()["yaw_checks"]
        self.assertTrue(checks)
        for chk in checks:
            self.assertFalse(chk["accepted"])
            self.assertAlmostEqual(abs(chk["yaw_err_deg"]), 15.0, delta=0.2)

    def test_far_jump_outside_drift_radius_rejected(self) -> None:
        lc = LoopCloser(LoopConfig())
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 0.5), (2.0, 0.5), (1.0, 0.5), (0.0, 0.5)]

        def rel(a, b):
            # 相似纹理：真值相对位置再横移 2 m（仍在 3 m 可视距离内），超过漂移半径（约 1.45 m）。
            d = (np.array(pts[b]) - np.array(pts[a]) + np.array([0.0, -2.0])) / S
            return {"R_ab": np.eye(3), "t_ab": np.array([d[0], d[1], 0.0]), **OK_REL}, "ok"
        found = walk(lc, pts, 0.0, rel)
        self.assertEqual(found, [])
        self.assertGreater(lc.rejects.get("outside_drift_radius", 0), 0)

    def test_no_loops_keeps_dead_reckoning(self) -> None:
        lc = LoopCloser(LoopConfig())
        walk(lc, square_loop(), 0.05, lambda a, b: (None, "few_matches"))
        self.assertEqual(lc.loops, [])
        self.assertAlmostEqual(lc.optimize(), 0.0, places=9)
        T = pose(1.0, 2.0)
        np.testing.assert_allclose(lc.correct(T), T)

    def test_ids_must_increase(self) -> None:
        lc = LoopCloser(LoopConfig())
        lc.add_keyframe(3, pose(0, 0), 0.0, None)
        with self.assertRaises(ValueError):
            lc.add_keyframe(3, pose(0, 0), 0.0, None)


if __name__ == "__main__":
    unittest.main()
