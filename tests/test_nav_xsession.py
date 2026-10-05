from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.nav_bow import BowVocabulary
from neko_anyadance_body.backend.nav_loop import M_OPT, KeyframeFeatures, LoopConfig
from neko_anyadance_body.backend.nav_xsession import (BowIndex, XSessionConfig, XSessionTracker,
                                                      align_into, align_into_auto,
                                                      align_world_tree, build_index,
                                                      estimate_gauge, load_index,
                                                      session_pose_table)

K = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])
SIZE = (320, 240)


def make_vocab(path: Path, seed: int = 0) -> BowVocabulary:
    rng = np.random.default_rng(seed)
    des = rng.integers(0, 255, (4000, 32), dtype=np.uint8)
    v = BowVocabulary(branching=4, depth=2, seed=seed).train(des, iters=3)
    v.save(str(path))
    return v


def project(xyz: np.ndarray) -> np.ndarray:
    uv = xyz[:, :2] / xyz[:, 2:3]
    return (uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])).astype(np.float32)


def make_place(rng: np.random.Generator, center: np.ndarray, spread: float, n: int):
    """一个"地点"= 一组物理点 + 各自的描述子（绑定：同描述子 ⇒ 同物理点）。"""
    W = center + rng.uniform(-spread, spread, (n, 3))
    des = rng.integers(0, 255, (n, 32), dtype=np.uint8)
    return W, des


def view_of(W: np.ndarray, des: np.ndarray, cam_t: np.ndarray):
    """同一地点从 cam_t 处看：P = base 系 3D 点，uv = 光学位投影（X_opt = P_base @ M_OPT）。"""
    P = W - cam_t
    X = P @ M_OPT
    return P.astype(np.float32), project(X), des


def feat_from(xyz: np.ndarray, uv: np.ndarray, des: np.ndarray) -> KeyframeFeatures:
    return KeyframeFeatures(uv, des, xyz, des, K, SIZE, 0.03)


def write_session(wdir: Path, sid: str, kfs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                  T_maps: list[np.ndarray], dists: list[float], status: str = "complete") -> Path:
    """按 nav_memory 的会话落盘格式写一个会话目录（kf npz 字段同 encode_features）。"""
    sdir = wdir / "sessions" / sid
    (sdir / "kf").mkdir(parents=True, exist_ok=True)
    ids, has = [], []
    for kf_id, xyz, _uv, des in kfs:
        np.savez_compressed(sdir / "kf" / f"{kf_id:06d}.npz",
                            xyz=xyz.astype(np.float16), des3d=np.asarray(des, np.uint8),
                            K=K, size=np.asarray(SIZE, np.int32), eye_y=np.float32(0.03),
                            n_kp=np.int32(len(des)))
        ids.append(kf_id)
        has.append(True)
    n = len(kfs)
    np.savez_compressed(sdir / "poses.npz", ids=np.asarray(ids, np.int32),
                        T_map=np.asarray(T_maps, np.float32),
                        T_dr=np.zeros((n, 4, 4), np.float32),
                        dist_m=np.asarray(dists, np.float32),
                        t_s=np.zeros(n, np.float32), has_feat=np.asarray(has, bool))
    (sdir / "session.json").write_text(json.dumps(
        {"session_id": sid, "status": status, "features": n}), encoding="utf-8")
    return sdir


def eye(n: int) -> list[np.ndarray]:
    return [np.eye(4) for _ in range(n)]


class XSessionTestBase(unittest.TestCase):
    """两个历史会话（A/B，各含 place1 视角）+ 一个 tiny 词汇树。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.wdir = self.root / "wrld_test-abc"
        self.vpath = self.root / "vocab.npz"
        make_vocab(self.vpath)
        rng = np.random.default_rng(7)
        # place1 = 一组绑定 (物理点, 描述子)；三个会话从不同相机位置看同一批点。
        W1, self.des1 = make_place(rng, np.array([3.0, 0.0, 0.5]), 1.2, 90)
        W2, des2 = make_place(rng, np.array([13.0, 0.0, 0.5]), 1.2, 90)
        W3, des3 = make_place(rng, np.array([33.0, 0.0, 0.5]), 1.2, 90)
        a0 = view_of(W1, self.des1, np.zeros(3))
        a1 = view_of(W2, des2, np.array([10.0, 0.0, 0.0]))
        b0 = view_of(W1, self.des1, np.array([0.3, 0.0, 0.0]))
        b1 = view_of(W3, des3, np.array([30.0, 0.0, 0.0]))
        self.W1 = W1
        self.a0 = a0
        write_session(self.wdir, "A", [(0, *a0), (1, *a1)], eye(2), [0.0, 10.0])
        write_session(self.wdir, "B", [(0, *b0), (1, *b1)], eye(2), [0.0, 10.0])
        self.cfg = XSessionConfig(
            vocab=str(self.vpath), query_top=4, verify_max=2, min_session_kf=2,
            reconfirm_path_m=1.5,
            loop=LoopConfig(min_inliers=50, max_offset_m=6.0, yaw_tol_deg=6.0, min_coverage=0.12))

    def tearDown(self) -> None:
        # 后台装载线程可能还在写索引缓存，join 掉再清临时目录（Windows 下目录非空会炸 cleanup）。
        tr = getattr(self, "tr", None)
        if tr is not None:
            tr.join(timeout=30.0)
        self._tmp.cleanup()


class TestIndexBuildLoad(XSessionTestBase):
    def test_build_and_query_finds_revisit(self) -> None:
        idx, info = build_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertIsNone(info.get("reason"))
        self.assertEqual(idx.n_docs, 4)
        self.assertEqual(sorted(idx.sessions()), ["A", "B"])
        # place1 的一个新视角（同一批物理点、相机挪 0.1 m）→ 应排到 A kf0 / B kf0 前列
        q = view_of(self.W1, self.des1, np.array([0.1, 0.0, 0.0]))
        with_boot = feat_from(*q)
        words = idx.vocab.transform(with_boot.des3d)
        hits = idx.query(words, top_n=4)
        top2 = {idx.sids[i] for i, _s in hits[:2]}
        self.assertTrue(top2 <= {"A", "B"}, f"top2={hits}")

    def test_load_cache_matches_build(self) -> None:
        idx0, _ = build_index(self.cfg, self.wdir, exclude_sid=None)
        idx1, info = load_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertEqual(info.get("source"), "cache")
        self.assertEqual(idx0.n_docs, idx1.n_docs)
        q = view_of(self.W1, self.des1, np.array([0.1, 0.0, 0.0]))
        words = idx1.vocab.transform(q[2])
        self.assertEqual(idx0.query(words, 4), idx1.query(words, 4))

    def test_new_session_triggers_rebuild(self) -> None:
        build_index(self.cfg, self.wdir, exclude_sid=None)
        Wc, desc = make_place(np.random.default_rng(21), np.array([50.0, 0.0, 0.5]), 1.2, 90)
        c0 = view_of(Wc, desc, np.array([48.0, 0.0, 0.0]))
        c1 = view_of(Wc, desc, np.array([47.0, 0.0, 0.0]))
        write_session(self.wdir, "C", [(0, *c0), (1, *c1)], eye(2), [0.0, 5.0])   # ≥ min_session_kf
        idx, info = load_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertEqual(info.get("source"), "rebuild")
        self.assertIn("C", idx.sessions())

    def test_min_kf_filter(self) -> None:
        Wc, desc = make_place(np.random.default_rng(21), np.array([50.0, 0.0, 0.5]), 1.2, 90)
        c0 = view_of(Wc, desc, np.zeros(3))
        write_session(self.wdir, "tiny", [(0, *c0)], eye(1), [0.0])   # features=1 < min 2
        idx, _ = build_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertNotIn("tiny", idx.sessions())


class TestBowAddWords(unittest.TestCase):
    def test_add_words_matches_add(self) -> None:
        rng = np.random.default_rng(3)
        des = rng.integers(0, 255, (500, 32), dtype=np.uint8)
        v = BowVocabulary(branching=4, depth=2, seed=0).train(des, iters=3)
        q = rng.integers(0, 255, (300, 32), dtype=np.uint8)
        b1 = BowIndex(v)
        for i in range(5):
            b1.add(i, des[i * 100:(i + 1) * 100])
        b2 = BowIndex(v)
        for i in range(5):
            b2.add_words(i, v.transform(des[i * 100:(i + 1) * 100]))
        self.assertEqual(b1.query(q, 5), b2.query(q, 5))

    def test_l1_norm_stops_short_doc_domination(self) -> None:
        """跨会话打分回归（2026-10-02 wrld_home 025013 零确认的根因之一）：

        "min" 口径下，词数悬殊时（查询 400 词 vs 文档 20 词）短文档即使内容完全无关
        也趋近满分霸榜；"l1" 对称口径必须把它压下去、让真匹配排前。
        """
        rng = np.random.default_rng(5)
        des = rng.integers(0, 255, (4000, 32), dtype=np.uint8)
        v = BowVocabulary(branching=4, depth=2, seed=0).train(des, iters=3)
        idx = BowIndex(v, cap_docs=64)
        # doc0：真匹配（与查询同分布的 400 词）；doc1..9：无关短文档（各 20 词）
        q_des = rng.integers(0, 255, (400, 32), dtype=np.uint8)
        idx.add(0, q_des)
        for i in range(1, 10):
            idx.add(i, rng.integers(0, 255, (20, 32), dtype=np.uint8))
        words = v.transform(q_des)
        s_min = dict(idx.query_words(words, top_n=10, norm="min"))
        s_l1 = dict(idx.query_words(words, top_n=10, norm="l1"))
        # "min"：短文档接近满分（复现霸榜）；"l1"：doc0 必须第一，且短文档分数不得高于真匹配
        self.assertGreater(min(s_min[i] for i in range(1, 10)), 0.9)
        self.assertEqual(max(s_l1, key=s_l1.get), 0)
        best_short = max(s_l1[i] for i in range(1, 10))
        self.assertGreater(s_l1[0], best_short)
        # 默认口径不变（会话内回环路径依赖）
        self.assertEqual(idx.query(q_des, 10), idx.query_words(words, top_n=10, norm="min"))


class TestTracker(XSessionTestBase):
    def test_confirm_reconfirm_yawgate_writeback(self) -> None:
        tr = self.tr = XSessionTracker(self.cfg, self.wdir, "C")
        self.assertTrue(tr.wait_ready(30.0) and tr.ready)
        # kf0：place1 新视角（同一批物理点，相机挪 0.15 m）→ 跨会话确认
        v0 = view_of(self.W1, self.des1, np.array([0.15, 0.0, 0.0]))
        rec = tr.on_keyframe(0, feat_from(*v0), 0.0, np.eye(3), False)
        self.assertTrue(rec)
        self.assertTrue(all(r["old_sid"] in ("A", "B") for r in rec))
        self.assertTrue(all(r["inliers"] >= 50 and r["rot_err_deg"] <= 6.0 for r in rec))
        # kf1：原地不动（路程差 < reconfirm_path_m）→ 同 doc 不再确认
        rec1 = tr.on_keyframe(1, feat_from(*v0), 0.3, np.eye(3), False)
        self.assertEqual(rec1, [])
        # kf2：yaw 门 —— R_map 偏 10° > 6° → 拒绝
        th = np.deg2rad(10.0)
        R_yaw = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
        rec2 = tr.on_keyframe(2, feat_from(*v0), 10.0, R_yaw, False)
        self.assertEqual(rec2, [])
        st = tr.status()
        self.assertGreaterEqual(st["rejects"].get("rotation_mismatch", 0), 1)
        # 约束落盘
        lines = (self.wdir / "xsession" / "constraints.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), len(rec))
        first = json.loads(lines[0])
        for k in ("new_sid", "new_kf", "old_sid", "old_kf", "inliers", "t_ab", "R_ab"):
            self.assertIn(k, first)
        # 写回：位姿整表 + words → 索引含 C（生产环境会话目录由 SessionWriter 落盘，这里补齐）
        write_session(self.wdir, "C",
                      [(0, *view_of(self.W1, self.des1, np.array([0.15, 0.0, 0.0]))),
                       (1, *view_of(self.W1, self.des1, np.array([0.20, 0.0, 0.0]))),
                       (2, *view_of(self.W1, self.des1, np.array([0.25, 0.0, 0.0])))],
                      eye(3), [0.0, 0.3, 10.0])
        T = np.eye(4)
        table = {"ids": np.array([0, 1, 2]), "T_map": np.asarray([T, T, T]), "dist_m": np.array([0.0, 0.3, 10.0])}
        wb = tr.write_back(table)
        self.assertTrue(wb["written"], wb)
        idx, info = load_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertIn("C", idx.sessions())
        self.assertEqual(info.get("source"), "cache")
        # 双重写回防护
        self.assertEqual(tr.write_back(table).get("reason"), "already_written")

    def test_too_few_keyframes_skips_writeback(self) -> None:
        tr = self.tr = XSessionTracker(self.cfg, self.wdir, "D")
        v0 = view_of(self.W1, self.des1, np.zeros(3))
        tr.on_keyframe(0, feat_from(*v0), 0.0, np.eye(3), False)   # 只有 1 帧 < min_session_kf=2
        wb = tr.write_back({"ids": np.array([0]), "T_map": np.asarray([np.eye(4)]), "dist_m": np.array([0.0])})
        self.assertFalse(wb["written"])
        self.assertEqual(wb["reason"], "too_few_keyframes")

    def test_missing_vocab_inert(self) -> None:
        cfg = XSessionConfig(vocab=str(self.root / "nope.npz"), enabled=True)
        tr = XSessionTracker(cfg, self.wdir, "E")
        self.assertFalse(tr.status()["active"])
        self.assertEqual(tr.status()["reason"], "vocab_missing")
        self.assertEqual(tr.on_keyframe(0, None, 0.0, np.eye(3), False), [])


def _rand_rot(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """绕 z 轴（+微倾）的随机旋转——贴合实况（map 系 z 向上，倾斜极小）。"""
    a = np.radians(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tilt = rng.uniform(-0.01, 0.01, 2)
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt[0]), -np.sin(tilt[0])], [0, np.sin(tilt[0]), np.cos(tilt[0])]])
    Ry = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])], [0, 1, 0], [-np.sin(tilt[1]), 0, np.cos(tilt[1])]])
    return Rx @ Ry @ R


class TestEstimateGauge(unittest.TestCase):
    @staticmethod
    def _synth(n: int, seed: int, noise_pos: float = 0.0, noise_rot_deg: float = 0.0):
        rng = np.random.default_rng(seed)
        # 真 gauge：小 yaw + 米级平移（两会话 yaw 基准同源的实况）
        R_G = _rand_rot(rng, 2.0)
        t_G = rng.uniform(-3.0, 3.0, 3)
        R_old = np.array([_rand_rot(rng, 170.0) for _ in range(n)])
        p_old = rng.uniform(-15.0, 15.0, (n, 3))
        p_old[:, 2] = 1.5
        R_new = np.array([_rand_rot(rng, 170.0) for _ in range(n)])
        p_new = rng.uniform(-15.0, 15.0, (n, 3))
        p_new[:, 2] = 1.5
        # 约束观测：p_pred = R_old t_ab + p_old = R_G p_new + t_G（+噪声）
        p_pred = p_new @ R_G.T + t_G
        R_pred = np.einsum("ij,njk->nik", R_G, R_new)
        if noise_pos:
            p_pred = p_pred + rng.normal(0, noise_pos, (n, 3))
        if noise_rot_deg:
            for i in range(n):
                dR = _rand_rot(rng, noise_rot_deg)
                R_pred[i] = dR @ R_pred[i]
        R_ab = np.einsum("nji,njk->nik", R_old, R_pred)   # R_ab = R_old^T @ R_pred
        t_ab = np.einsum("nji,nj->ni", R_old, p_pred - p_old)
        return R_old, p_old, R_ab, t_ab, R_new, p_new, R_G, t_G

    def test_exact_recovery(self) -> None:
        R_old, p_old, R_ab, t_ab, R_new, p_new, R_G_true, t_G_true = self._synth(30, seed=1)
        out = estimate_gauge(R_old, p_old, R_ab, t_ab, R_new, p_new)
        self.assertIsNotNone(out)
        self.assertEqual(out["n_inlier"], 30)
        self.assertLess(np.abs(out["R_G"] - R_G_true).max(), 1e-9)
        self.assertLess(np.abs(out["t_G"] - t_G_true).max(), 1e-8)
        self.assertLess(out["pos_max"], 1e-6)
        self.assertLess(out["R_dev_deg"], 2.5)

    def test_noise_and_outliers(self) -> None:
        args = list(self._synth(60, seed=2, noise_pos=0.03, noise_rot_deg=0.5))
        rng = np.random.default_rng(9)
        n_out, n = 8, 60
        for i in rng.choice(n, n_out, replace=False):     # 伪造外点：平移错 ~3 m
            args[3][i] += rng.normal(0, 1.7, 3)
        out = estimate_gauge(*args[:6])
        self.assertIsNotNone(out)
        self.assertGreaterEqual(out["n_inlier"], n - n_out - 2)
        self.assertLess(out["pos_med"], 0.10)
        self.assertLess(out["rot_med"], 1.0)
        # 最大残差的 n_out 条应被踢出内点集
        order = np.argsort(out["e_pos"])[::-1]
        self.assertTrue(np.all(~out["inlier"][order[:n_out]]))

    def test_too_few_returns_none(self) -> None:
        args = list(self._synth(2, seed=3))
        self.assertIsNone(estimate_gauge(*args[:6]))
        out = estimate_gauge(*list(self._synth(10, seed=3))[:6])
        self.assertIsNotNone(out)


def _rz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _t4(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def build_align_fixture(wdir: Path):
    """合成"旧会话 + 带渐进尾漂的新会话 + 25 条跨会话约束"（视觉=真值 ⇒ R_ab=I/t_ab=0）。

    返回 (p_old, p_new_true, cons_n)。供 TestAlignInto 与 test_nav_online 的接线测试复用。
    """
    rng = np.random.default_rng(0)
    R_G, t_G = _rz(4.0), np.array([0.5, 0.3, 0.01])
    n = 30
    p_old = np.column_stack([0.5 * np.arange(n), np.zeros(n), np.zeros(n)])
    R_old = np.stack([np.eye(3)] * n)
    drift = np.zeros(n)                                # 尾半段渐进漂到 5 m（存档系 x 方向）
    drift[n // 2:] = np.linspace(0.0, 5.0, n - n // 2)
    R_new = np.stack([R_G.T @ np.eye(3)] * n)          # 新帧旋转（存档系）
    p_new_true = (p_old - t_G) @ R_G                   # 无漂移真值（重合旧轨迹）
    p_new_arc = p_new_true.copy()
    p_new_arc[:, 0] += drift                           # 存档位姿带漂移
    d_des = rng.integers(1, 255, (n, 4, 32), dtype=np.uint8)
    kfs = [(r, np.zeros((1, 3), np.float32), np.zeros((1, 2), np.float32), d_des[r])
           for r in range(n)]
    write_session(wdir, "OLD", kfs, [_t4(R_old[r], p_old[r]) for r in range(n)],
                  [0.5 * r for r in range(n)])
    write_session(wdir, "NEW", kfs, [_t4(R_new[r], p_new_arc[r]) for r in range(n)],
                  [0.5 * r for r in range(n)])
    # 约束 25 条（头 10 无漂移 + 尾 15 带漂移锚，密度贴近真实渐进漂移）
    cons = []
    for r in list(range(10)) + list(range(n - 15, n)):
        cons.append({"new_sid": "NEW", "new_kf": r, "old_sid": "OLD", "old_kf": r,
                     "inliers": 100, "rot_err_deg": 0.5, "offset_m": 0.1,
                     "t_ab": [0.0, 0.0, 0.0], "R_ab": np.eye(3).tolist()})
    (wdir / "xsession").mkdir(parents=True, exist_ok=True)
    with open(wdir / "xsession" / "constraints.jsonl", "w", encoding="utf-8") as f:
        for c in cons:
            f.write(json.dumps(c) + "\n")
    return p_old, p_new_true, len(cons)


class TestAlignInto(unittest.TestCase):
    """P0.3b：align_into / align_into_auto（合成 G + 尾部漂移恢复）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wdir = Path(self._tmp.name) / "wrld_x-abc"
        self.p_old, self.p_new_true, self.cons_n = build_align_fixture(self.wdir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_align_recovers_gauge_and_pulls_tail_back(self) -> None:
        out = align_into(self.wdir, "NEW", "OLD", tail_m=8.0)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_anchors"], self.cons_n)
        # gauge 恢复：R_G 偏离 ≈ 4°，t_G ≈ (0.5, 0.3)（漂移帧被 IRLS 剔除后）
        self.assertAlmostEqual(out["gauge"]["R_dev_deg"], 4.0, delta=1.0)
        self.assertLess(out["gauge"]["pos_med_m"], 0.3)
        # 优化后：尾段被拉回（留出锚 med <1 m），merged 表尾帧接近真值。
        # 已知局限：最尾帧只有单侧链（无后继锚），插值修正约一半（真实 044153 留出 p90 4.95 m 同源）。
        self.assertLess(out["holdout"]["med_m"], 1.0)
        with np.load(Path(out["merged"])) as z:
            T = np.asarray(z["T_map"], np.float64)
            self.assertEqual(z["base_sid"], "OLD")
        self.assertLess(float(np.linalg.norm(T[-1][:2, 3] - self.p_new_true[-1][:2])), 3.0)
        # align json 落盘 + inlier 掩码（掩码只对训练锚，留出锚无残差不参与）
        rec = json.loads(Path(out["align"]).read_text(encoding="utf-8"))
        self.assertEqual(len(rec["inlier"]), out["train"]["n"])
        self.assertTrue(any(rec["inlier"]))

    def test_align_into_auto_picks_best_and_gates(self) -> None:
        cfg = XSessionConfig(align_min_constraints=8)
        # 只有 3 条 → 门槛拒绝并报 best_old_sid
        (self.wdir / "xsession" / "constraints.jsonl").write_text(
            "\n".join(json.dumps({"new_sid": "NEW", "old_sid": "OLD", "new_kf": r, "old_kf": r,
                                  "inliers": 100, "R_ab": np.eye(3).tolist(), "t_ab": [0, 0, 0]})
                      for r in range(3)) + "\n", encoding="utf-8")
        out = align_into_auto(self.wdir, "NEW", cfg)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "too_few_constraints")
        self.assertEqual(out["best_old_sid"], "OLD")
        # 两组候选（5 vs 12 条）→ 选多的那组（OLD），并走到真对齐
        cons = ([{"new_sid": "NEW", "old_sid": "OTHER", "new_kf": r, "old_kf": r, "inliers": 100,
                  "R_ab": np.eye(3).tolist(), "t_ab": [0, 0, 0]} for r in range(5)]
                + [{"new_sid": "NEW", "old_sid": "OLD", "new_kf": r, "old_kf": r, "inliers": 100,
                    "R_ab": np.eye(3).tolist(), "t_ab": [0, 0, 0]} for r in range(12)])
        (self.wdir / "xsession" / "constraints.jsonl").write_text(
            "\n".join(json.dumps(c) for c in cons) + "\n", encoding="utf-8")
        out = align_into_auto(self.wdir, "NEW", cfg)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["old_sid"], "OLD")

    def test_align_missing_constraints(self) -> None:
        (self.wdir / "xsession" / "constraints.jsonl").unlink()
        out = align_into(self.wdir, "NEW", "OLD")
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "no_constraints")


class TestMergedConsumption(XSessionTestBase):
    """P0 收尾：已采纳的 merged 表被**下游消费**（索引读它；换表让缓存失效）。"""

    def _write_merged(self, sid: str, base: str, dx: float, ids_override=None) -> Path:
        xdir = self.wdir / "xsession"
        xdir.mkdir(parents=True, exist_ok=True)
        with np.load(self.wdir / "sessions" / sid / "poses.npz") as z:
            ids = np.asarray(z["ids"], np.int64)
            T = np.asarray(z["T_map"], np.float64)
            dist = np.asarray(z["dist_m"], np.float64)
        T = T.copy()
        T[:, 0, 3] += dx
        p = xdir / f"merged_{sid}_into_{base}.npz"
        np.savez_compressed(p, ids=np.asarray(ids if ids_override is None else ids_override),
                            T_map=T, dist_m=np.asarray(dist, np.float64), base_sid=base)
        return p

    @staticmethod
    def _xy_of(idx, sid: str) -> np.ndarray:
        rows = [i for i, s in enumerate(idx.sids) if s == sid]
        return np.asarray([idx.xy[i] for i in rows], np.float64)

    def test_index_prefers_merged_and_invalidates_cache(self) -> None:
        idx0, _ = build_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertTrue(np.allclose(self._xy_of(idx0, "B"), 0.0, atol=1e-6))
        self._write_merged("B", "A", dx=0.3)
        idx1, info = load_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertEqual(info.get("source"), "rebuild", "换了表必须让索引缓存失效")
        self.assertTrue(np.allclose(self._xy_of(idx1, "B")[:, 0], 0.3, atol=1e-6))
        self.assertEqual(info["tables"]["B"], "merged", "info 要如实报实际用到的表")

    def test_bad_merged_falls_back_to_raw(self) -> None:
        self._write_merged("B", "A", dx=0.3, ids_override=[98, 99])
        idx, info = build_index(self.cfg, self.wdir, exclude_sid=None)
        self.assertTrue(np.allclose(self._xy_of(idx, "B"), 0.0, atol=1e-6),
                        "ids 对不上必须退回原表（宁可旧坐标，不要错误坐标）")
        self.assertEqual(info["tables"]["B"], "raw")


class TestAlignBaseComposition(unittest.TestCase):
    """旧会话自己已被采纳时：再对齐按 ``base_sid`` **链式往上传**（C→B、B→A ⇒ C 落 A 系）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wdir = Path(self._tmp.name) / "wrld_x-abc"
        self.p_old, self.p_new_true, _ = build_align_fixture(self.wdir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_align_reads_merged_old_and_propagates_base(self) -> None:
        delta = 1.0                                   # OLD 早已被采纳进 BASE：整体平移 1 m
        with np.load(self.wdir / "sessions" / "OLD" / "poses.npz") as z:
            ids = np.asarray(z["ids"], np.int64)
            T = np.asarray(z["T_map"], np.float64)
            dist = np.asarray(z["dist_m"], np.float64)
        T = T.copy()
        T[:, 0, 3] += delta
        np.savez_compressed(self.wdir / "xsession" / "merged_OLD_into_BASE.npz",
                            ids=ids, T_map=T, dist_m=dist, base_sid="BASE")
        out = align_into(self.wdir, "NEW", "OLD", tail_m=8.0)
        self.assertTrue(out["ok"], out)
        with np.load(Path(out["merged"])) as z:
            self.assertEqual(str(z["base_sid"]), "BASE", "最终基准要往上传，而不是停在中间会话")
            T_out = np.asarray(z["T_map"], np.float64)
        # NEW 对齐到「OLD 被平移后的系」⇒ 尾帧应接近 真值 + δ
        err = np.linalg.norm(T_out[-1][:2, 3] - (self.p_new_true[-1][:2] + delta))
        self.assertLess(float(err), 3.0, "merged 基准没被消费：尾帧没跟着平移")


class TestIndexGainAfterAdoption(unittest.TestCase):
    """端到端收益（合成夹具带真值）：采纳后**索引里** NEW 的坐标被拉回真值。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wdir = Path(self._tmp.name) / "wrld_x-abc"
        self.p_old, self.p_new_true, _ = build_align_fixture(self.wdir)
        self.vpath = Path(self._tmp.name) / "vocab.npz"
        make_vocab(self.vpath)
        self.cfg = XSessionConfig(vocab=str(self.vpath), min_session_kf=2)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _new_xy(idx) -> np.ndarray:
        rows = [i for i, s in enumerate(idx.sids) if s == "NEW"]
        order = np.argsort([idx.kfs[i] for i in rows])
        return np.asarray([idx.xy[rows[j]] for j in order], np.float64)

    def test_tail_error_drops_after_adoption(self) -> None:
        idx0, _ = build_index(self.cfg, self.wdir, exclude_sid=None)
        err0 = np.linalg.norm(self._new_xy(idx0)[20:] - self.p_new_true[20:, :2], axis=1).mean()
        out = align_into(self.wdir, "NEW", "OLD", tail_m=8.0)
        self.assertTrue(out["ok"], out)
        idx1, info = build_index(self.cfg, self.wdir, exclude_sid=None)
        err1 = np.linalg.norm(self._new_xy(idx1)[20:] - self.p_new_true[20:, :2], axis=1).mean()
        self.assertLess(err1, 0.6 * err0, f"索引尾部误差 {err0:.2f} → {err1:.2f}")
        self.assertEqual(info["tables"]["NEW"], "merged")


def _chain_fixture(wdir: Path, *, rows_b: list[int] | None = None,
                   rows_c: list[int] | None = None) -> dict[str, Any]:
    """三段同一条走廊的会话链 A ← B ← C（视觉=真值 ⇒ t_ab=0/R_ab=I）。

    B 的存档系比 A 差一个 gauge（R_ba/t_ba）、C 又比 B 差一个：align_into(B,A) 与
    align_into(C,B) 应把它俩**先后**收回 A 系（链式）。真值轨迹 p_a 供几何断言。
    """
    n = 30
    R_ba, t_ba = _rz(3.0), np.array([0.4, 0.2, 0.01])
    R_cb, t_cb = _rz(-2.5), np.array([-0.3, 0.1, 0.02])
    p_a = np.column_stack([0.5 * np.arange(n), np.zeros(n), np.zeros(n)])
    p_b = (p_a - t_ba) @ R_ba
    p_c = (p_b - t_cb) @ R_cb
    R_a = np.stack([np.eye(3)] * n)
    R_b = np.stack([R_ba.T] * n)
    R_c = np.stack([R_cb.T @ R_ba.T] * n)
    rng = np.random.default_rng(11)
    des = {k: rng.integers(1, 255, (n, 4, 32), dtype=np.uint8) for k in "abc"}
    for sid, (p, R, dk) in {"A": (p_a, R_a, "a"), "B": (p_b, R_b, "b"),
                            "C": (p_c, R_c, "c")}.items():
        kfs = [(r, np.zeros((1, 3), np.float32), np.zeros((1, 2), np.float32), des[dk][r])
               for r in range(n)]
        write_session(wdir, sid, kfs, [_t4(R[r], p[r]) for r in range(n)],
                      [0.5 * r for r in range(n)])
    cons = ([{"new_sid": "B", "new_kf": r, "old_sid": "A", "old_kf": r, "inliers": 100,
              "R_ab": np.eye(3).tolist(), "t_ab": [0, 0, 0]}
             for r in (rows_b if rows_b is not None else range(n))]
            + [{"new_sid": "C", "new_kf": r, "old_sid": "B", "old_kf": r, "inliers": 100,
                "R_ab": np.eye(3).tolist(), "t_ab": [0, 0, 0]}
               for r in (rows_c if rows_c is not None else range(n))])
    (wdir / "xsession").mkdir(parents=True, exist_ok=True)
    (wdir / "xsession" / "constraints.jsonl").write_text(
        "\n".join(json.dumps(c) for c in cons) + "\n", encoding="utf-8")
    return {"p_a": p_a}


class TestWorldTree(unittest.TestCase):
    """P0 §洞4：世界树 pass —— 合成链 A←B←C：链式采纳进 A 系、幂等、门槛与干跑。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wdir = Path(self._tmp.name) / "wrld_x-abc"
        # tail-m=8 让 30 帧 ×0.5 m 的会话有尾段可留出（质量闸要留出锚 ≥2）
        self.cfg = XSessionConfig(min_session_kf=2, align_tail_m=8.0, align_holdout_frac=0.5)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_chain_aligns_into_one_gauge_and_is_idempotent(self) -> None:
        fx = _chain_fixture(self.wdir)
        out = align_world_tree(self.wdir, self.cfg, write=True)
        self.assertTrue(out["ok"], out)
        self.assertEqual(len(out["components"]), 1, "三个会话一条链 ⇒ 一个分量")
        comp = out["components"][0]
        self.assertEqual(comp["root"], "A")
        self.assertEqual(comp["coverage"], "3/3")
        self.assertEqual([o["status"] for o in comp["ops"]], ["aligned", "aligned"])
        self.assertEqual([(o["sid"], o["parent"]) for o in comp["ops"]],
                         [("B", "A"), ("C", "B")])
        # 链式上传：C 的基准应是 A（中间会话 B 只当跳板）而不是 B
        tab = session_pose_table(self.wdir, "C")
        self.assertEqual(tab["source"], "merged")
        self.assertEqual(tab["base_sid"], "A")
        with np.load(self.wdir / "xsession" / "merged_C_into_B.npz") as z:
            T_c = np.asarray(z["T_map"], np.float64)
        err = np.linalg.norm(T_c[-1][:2, 3] - fx["p_a"][-1][:2])
        self.assertLess(float(err), 0.5, "C 必须被收回 A 系的走廊上")
        self.assertTrue((self.wdir / "xsession" / "world_tree.json").is_file())
        # 幂等：第二遍全走 already，不重算不重写
        out2 = align_world_tree(self.wdir, self.cfg, write=True)
        st2 = [o["status"] for c in out2["components"] for o in c["ops"]]
        self.assertTrue(st2 and all(s == "already" for s in st2), st2)

    def test_below_threshold_stays_apart_until_bridge_min(self) -> None:
        # C→B 只 7 条 < 8：默认门槛下 C 自成分量；门槛降到 5 才被桥进来
        _chain_fixture(self.wdir, rows_c=[4, 8, 12, 18, 22, 26, 28])
        out = align_world_tree(self.wdir, self.cfg, write=True)
        self.assertEqual(sorted(c["root"] for c in out["components"]), ["A", "C"])
        comp_c = next(c for c in out["components"] if c["root"] == "C")
        self.assertFalse(comp_c["bridge"]["meets_min"])
        self.assertEqual(comp_c["bridge"]["count"], 7)
        self.assertFalse((self.wdir / "xsession" / "merged_C_into_B.npz").exists())
        out2 = align_world_tree(self.wdir, self.cfg, bridge_min=5, write=True)
        comp = out2["components"][0]
        self.assertEqual((comp["root"], comp["coverage"]), ("A", "3/3"))
        # 第一遍已把 B 采纳进 A ⇒ 这里 B 是 already，只有 C 是新桥
        self.assertEqual({o["sid"]: o["status"] for o in comp["ops"]},
                         {"B": "already", "C": "aligned"})
        self.assertTrue((self.wdir / "xsession" / "merged_C_into_B.npz").exists())

    def test_dry_run_writes_no_merged(self) -> None:
        _chain_fixture(self.wdir)
        out = align_world_tree(self.wdir, self.cfg, write=False)
        st = [o["status"] for c in out["components"] for o in c["ops"]]
        # B 能预演（父=A 已在系）；C 的父 B 在干跑里没真落位 ⇒ 只给计划不预演
        self.assertEqual(st, ["would_align", "planned"])
        self.assertFalse((self.wdir / "xsession" / "merged_B_into_A.npz").exists())
        self.assertFalse((self.wdir / "xsession" / "merged_C_into_B.npz").exists())


class TestYawConsensusGate(XSessionTestBase):
    """2026-10-06：绝对 6° yaw 门实测吞真重合（wrld_home，见 nav_xsession docstring）⇒
    ``yaw_consensus`` 改用"相对该会话对带符号 yaw 残差中位数"的窗；硬顶仍拦大偏置。"""

    @staticmethod
    def _cfg(vpath: Path, **kw: Any) -> XSessionConfig:
        return XSessionConfig(vocab=str(vpath), query_top=4, verify_max=2, min_session_kf=2,
                              loop=LoopConfig(min_inliers=50, max_offset_m=6.0,
                                              yaw_tol_deg=6.0, min_coverage=0.12), **kw)

    def _feed(self, tr: XSessionTracker, rot_deg: float, n: int = 12) -> None:
        v = view_of(self.W1, self.des1, np.array([0.15, 0.0, 0.0]))
        feat = feat_from(*v)
        R = _rz(rot_deg)
        for k in range(n):
            tr.on_keyframe(k, feat, 10.0 + 2.0 * k, R, False)   # 路程 2 m/帧：绕开 reconfirm 门

    def test_offset_9deg_absolute_rejects_consensus_admits(self) -> None:
        tr = self.tr = XSessionTracker(self._cfg(self.vpath, yaw_consensus=False), self.wdir, "C")
        self.assertTrue(tr.wait_ready(30.0) and tr.ready)
        self._feed(tr, 9.0)
        st = tr.status()
        self.assertEqual(st["confirmed"], 0, "9° > 绝对门 6°：旧判据全拒")
        self.assertGreaterEqual(st["rejects"].get("rotation_mismatch", 0), 4)
        tr.join(30.0)
        tr2 = self.tr = XSessionTracker(self._cfg(self.vpath, yaw_consensus=True,
                                                  yaw_consensus_min=4), self.wdir, "C2")
        self.assertTrue(tr2.wait_ready(30.0) and tr2.ready)
        self._feed(tr2, 9.0)
        self.assertGreaterEqual(tr2.status()["confirmed"], 1, "池稳到 +9° 后应放行")
        tr2.join(30.0)
        recs = [json.loads(l) for l in
                (self.wdir / "xsession" / "constraints.jsonl").read_text(encoding="utf-8")
                .strip().splitlines() if l.strip()]
        self.assertEqual([r["new_sid"] for r in recs], ["C2"] * len(recs))
        self.assertTrue(any(r["rot_err_deg"] > 6.0 for r in recs),
                        "共识门要能放行 >6° 的真匹配（rot_err 留痕应超过绝对门）")

    def test_hard_cap_still_blocks_large_offset(self) -> None:
        cfg = self._cfg(self.vpath, yaw_consensus=True, yaw_consensus_min=4, yaw_cap_deg=20.0)
        tr = self.tr = XSessionTracker(cfg, self.wdir, "C3")
        self.assertTrue(tr.wait_ready(30.0) and tr.ready)
        self._feed(tr, 40.0)
        st = tr.status()
        self.assertEqual(st["confirmed"], 0, "40° 超硬顶：共识窗再宽也不放")
        self.assertGreaterEqual(st["rejects"].get("rotation_mismatch", 0), 4)


if __name__ == "__main__":
    unittest.main()
