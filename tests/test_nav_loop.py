from __future__ import annotations

import math
import os
import tempfile
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

    def test_status_reports_loop_freshness(self) -> None:
        # 健康度：全程 0 回环时 = 从起点累计，单调增长 —— 这就是"尾部长期 0 回环"的报警器
        # （此前只能事后翻录制才发现，见 Docs/停顿后地图错位-根因诊断（2026-10-01）.md §七.3）。
        lc = LoopCloser(LoopConfig())
        pts = square_loop()
        walk(lc, pts, 0.05, lambda a, b: (None, "few_matches"))
        st = lc.status()
        self.assertIsNone(st["last_loop_keyframe"])
        self.assertEqual(st["kf_since_last_loop"], len(pts))
        self.assertAlmostEqual(st["path_since_last_loop_m"], 16.0, delta=0.2)

        # 回环发生那一刻：两个计数归零。
        lc2 = LoopCloser(LoopConfig())
        last = len(pts) - 1

        def rel(a, b):
            if (a, b) == (0, last):
                return {"R_ab": np.eye(3), "t_ab": np.zeros(3), **OK_REL}, "ok"
            return None, "few_matches"

        walk(lc2, pts, 0.05, rel)
        st2 = lc2.status()
        self.assertEqual(st2["last_loop_keyframe"], last)
        self.assertEqual(st2["kf_since_last_loop"], 0)
        self.assertAlmostEqual(st2["path_since_last_loop_m"], 0.0, places=1)

    def test_ids_must_increase(self) -> None:
        lc = LoopCloser(LoopConfig())
        lc.add_keyframe(3, pose(0, 0), 0.0, None)
        with self.assertRaises(ValueError):
            lc.add_keyframe(3, pose(0, 0), 0.0, None)


class BowCandidateTest(unittest.TestCase):
    """外观（词袋）候选的接线：默认关、缺词汇树不致命、候选必须过 min_path_m。

    对应 `Docs/停顿后地图错位-根因诊断（2026-10-01）.md` §10 的两条硬要求：
      ① `_verify` 不查 `min_path_m`，所以外观候选必须**自己过**；
      ② 必须**先过门再截断**，否则长时间停顿时名额被重复帧占满。
    """

    @staticmethod
    def _vocab(tmpdir: str) -> str:
        from neko_anyadance_body.backend import nav_bow

        rng = np.random.default_rng(0)
        pool = rng.integers(0, 256, (4000, 32), dtype=np.uint8)
        # 词数不能太少：16 词的粗直方图会让**随机**描述子也拿到满分 1.0 排到最前
        # （实测 top8 里混进 4 个只有 20 个描述子的填充帧），测试就失去判别力。
        v = nav_bow.BowVocabulary(branching=8, depth=3, seed=0).train(pool, iters=3)
        p = os.path.join(tmpdir, "bow_vocab_test.npz")
        v.save(p)
        return p

    @staticmethod
    def _filler(seed: int) -> KeyframeFeatures:
        """不合格的填充特征：des3d < min_inliers ⇒ 永远当不了候选。"""
        rng = np.random.default_rng(seed)
        n = 20
        P = np.column_stack([rng.uniform(2.0, 5.0, n), rng.uniform(-2.0, 2.0, n),
                             rng.uniform(-1.0, 0.5, n)])
        uv = rng.uniform(0, 700, (n, 2)).astype(np.float32)
        des = rng.integers(0, 256, (n, 32), dtype=np.uint8)
        return KeyframeFeatures(uv, des, P.astype(np.float32), des, K, (720, 405), HALF_B)

    def test_off_by_default(self) -> None:
        self.assertEqual(LoopConfig().bow_candidates, 0)
        st = LoopCloser(LoopConfig()).status()
        self.assertFalse(st["bow_ready"])
        self.assertEqual(st["bow_reason"], "disabled")

    def test_missing_vocab_is_not_fatal(self) -> None:
        cfg = LoopConfig(bow_candidates=4, bow_vocab="no/such/vocab.npz")
        lc = LoopCloser(cfg)
        st = lc.status()
        self.assertFalse(st["bow_ready"])
        self.assertEqual(st["bow_reason"], "vocab_missing")
        # 回环照常跑：外观只是候选来源之一，没有它不能崩。
        for i in range(3):
            lc.add_keyframe(i, pose(float(i) * 0.5, 0.0), float(i) * 0.4, None)
        self.assertEqual(len(lc), 3)

    def test_appearance_candidate_found_and_gated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._vocab(td)
            # 0 号与 8 号是同一处（描述子一致、相对位姿已知），中间夹 7 帧填充。
            fa, fb = scene_pair(yaw_R(5.0), np.array([0.25, 0.10, 0.0]), n=600, seed=1)
            cfg = LoopConfig(bow_candidates=4, bow_shortlist=32, bow_vocab=path)
            lc = LoopCloser(cfg)
            self.assertTrue(lc.status()["bow_ready"])
            feats = [fa] + [self._filler(100 + i) for i in range(7)] + [fb]
            yaws = [0.0] * 8 + [5.0]
            for i in range(9):
                lc.add_keyframe(i, pose(float(i) * 0.25, 0.0, yaws[i]), float(i) * 1.0, feats[i])

            # ① 外观分支确实把 0 号找了出来（填充帧因 des3d 太少被挡）
            cand = lc._bow_candidates(8, lc._nodes[8])
            self.assertIn(0, cand)
            self.assertNotIn(8, cand)                      # 永不自荐
            # ② 索引是 query 之后才收当前帧的 ⇒ 关键帧数 == 索引文档数，且无自回环
            self.assertEqual(lc.status()["bow_docs"], 9)
            self.assertEqual([(L.a, L.b) for L in lc.loops if L.a == L.b], [])
            # ③ 每条收下的回环都满足 min_path_m —— `_verify` 不查这个门，全靠候选分支自己过
            for L in lc.loops:
                self.assertGreaterEqual(lc._nodes[L.b].dist_m - lc._nodes[L.a].dist_m,
                                        cfg.min_path_m)

    def test_gate_before_truncate(self) -> None:
        """先过门再截断：短名单全被 min_path 挡掉时，也不能退化成"一个候选都没有"。"""
        with tempfile.TemporaryDirectory() as td:
            path = self._vocab(td)
            fa, fb = scene_pair(yaw_R(5.0), np.array([0.25, 0.10, 0.0]), n=600, seed=2)
            # 真实重访不会像素级相同：把 0 号 600 个描述子里的 250 个换掉 ⇒ 外观分低于
            # "和 61 号一模一样"的停顿帧，排名落在它们后面，这样"先截断"才会真把它挤掉。
            rng = np.random.default_rng(7)
            des0 = fa.des.copy()
            pick = rng.choice(len(des0), 250, replace=False)
            des0[pick] = rng.integers(0, 256, (len(pick), 32), dtype=np.uint8)
            fa = KeyframeFeatures(fa.uv, des0, fa.xyz, des0, fa.K, fa.size, fa.eye_y)
            cfg = LoopConfig(bow_candidates=4, bow_shortlist=32, bow_vocab=path)
            lc = LoopCloser(cfg)
            # 走到 10 m：0 号是出发点，1..30 号在走（路程增长）
            lc.add_keyframe(0, pose(0.0, 0.0), 0.0, fa)
            for i in range(1, 31):
                lc.add_keyframe(i, pose(float(i) * 0.1, 0.0), float(i) * 0.33, self._filler(200 + i))
            # 然后**原地停 20 帧**：路程全停在 10.0，且画面和 51 号一模一样（外观最像）
            for i in range(31, 51):
                lc.add_keyframe(i, pose(3.0, 0.0), 10.0, fb)
            # 51 号：仍停在 10.0，但这里是 0 号那处的重访
            lc.add_keyframe(51, pose(0.25, 0.10, 5.0), 10.0, fb)

            # 先证明"病灶"存在：外观排名前 4 全是停顿帧，先截断的话一个真候选都不剩
            raw = lc._bow.query(lc._nodes[51].feat.des, top_n=32)
            top4 = [a for a, _s in raw[:4]]
            self.assertTrue(top4)
            self.assertTrue(all(lc._nodes[a].dist_m >= 9.9 for a in top4))
            # 再证明修法有效：先过 min_path 再截断，真候选 0 号活下来了
            cand = lc._bow_candidates(51, lc._nodes[51])
            self.assertIn(0, cand)
            self.assertTrue(all(lc._nodes[a].dist_m <= 10.0 - cfg.min_path_m for a in cand))

    def test_shortlist_depth_is_a_hard_bound(self) -> None:
        """"先过门再截断"只在短名单深度内有效 —— 停顿长过 shortlist 时照样找不到。

        这不是 bug，是有界性：真候选必须**先出现在 top-N 里**才有机会过门。
        实测（同等构造、shortlist=32）：停 20 帧时 0 号排第 23；停 30 帧时排第 33 ⇒ 掉出短名单。
        把这条写成断言，是为了防止有人以为"先过门"能救任意长的停顿。
        """
        with tempfile.TemporaryDirectory() as td:
            path = self._vocab(td)
            fa, fb = scene_pair(yaw_R(5.0), np.array([0.25, 0.10, 0.0]), n=600, seed=2)
            rng = np.random.default_rng(7)
            des0 = fa.des.copy()
            pick = rng.choice(len(des0), 250, replace=False)
            des0[pick] = rng.integers(0, 256, (len(pick), 32), dtype=np.uint8)
            fa = KeyframeFeatures(fa.uv, des0, fa.xyz, des0, fa.K, fa.size, fa.eye_y)
            cfg = LoopConfig(bow_candidates=4, bow_shortlist=32, bow_vocab=path)
            lc = LoopCloser(cfg)
            lc.add_keyframe(0, pose(0.0, 0.0), 0.0, fa)
            for i in range(1, 31):
                lc.add_keyframe(i, pose(float(i) * 0.1, 0.0), float(i) * 0.33,
                                self._filler(200 + i))
            for i in range(31, 61):                       # 停 30 帧 > 短名单里给它留的位置
                lc.add_keyframe(i, pose(3.0, 0.0), 10.0, fb)
            lc.add_keyframe(61, pose(0.25, 0.10, 5.0), 10.0, fb)

            raw = lc._bow.query(lc._nodes[61].feat.des, top_n=64)
            rank0 = [r for r, (a, _s) in enumerate(raw) if a == 0]
            self.assertTrue(rank0)
            self.assertGreaterEqual(rank0[0], cfg.bow_shortlist)   # 已在短名单外
            self.assertEqual(lc._bow_candidates(61, lc._nodes[61]), [])
            # 把短名单抬到能覆盖它 ⇒ 立刻找得回来（证明瓶颈确实是深度，不是门）
            cfg.bow_shortlist = 64
            self.assertIn(0, lc._bow_candidates(61, lc._nodes[61]))


if __name__ == "__main__":
    unittest.main()
