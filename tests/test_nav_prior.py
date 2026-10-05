from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.nav_grid import FREE, OCC, UNK, GridMeta, NavGrid
from neko_anyadance_body.backend.nav_mapping import KeyframeGridMapper, NavSession
from neko_anyadance_body.backend.nav_prior import (L_FREE, L_OCC, L_UNK, PriorConfig,
                                                   PriorConsumer, PriorMap, load_prior,
                                                   prior_dir, prior_meta, project_prior, write_prior)


def yaw(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_grid(shape=(40, 40), res=0.1, origin=(-2.0, -2.0), world_scale=1.0) -> NavGrid:
    """全 UNK 的三态图，默认与 ``sample_prior`` 同域（这样两边坐标能直接对着手算）。

    注意：``apply_into`` 收的是**追踪米**（与会话栅格同系），所以 world_scale 取 1 时
    世界米 == 追踪米，测试里的手算坐标不用再乘系数。
    """
    return NavGrid(np.full(shape, UNK, np.uint8), GridMeta(res, origin, world_scale))


def cell_of(ng: NavGrid, x: float, y: float) -> tuple[int, int]:
    """追踪米 → 格下标（与 ``PriorOverlay.apply_into`` 同一套取整，测试专用）。"""
    col = int(np.floor((x - ng.meta.origin_xy_m[0]) / ng.meta.resolution_m))
    rfb = int(np.floor((y - ng.meta.origin_xy_m[1]) / ng.meta.resolution_m))
    return ng.grid.shape[0] - 1 - rfb, col


#: ``sample_prior`` 里非 UNK 格的格心（追踪米，与世界系同值）—— 顺序 = ``np.argwhere`` 行序。
SAMPLE_XY = np.array([[-0.95, -0.95], [-0.85, -0.95], [-0.95, -0.85], [-0.85, -0.85], [0.05, -0.45]])


def sample_prior(labels: np.ndarray | None = None) -> PriorMap:
    """一张合成先验：origin (−2, −2)、res 0.1、行 = y、列 = x。"""
    if labels is None:
        labels = np.zeros((40, 40), np.uint8)
        labels[10:12, 10:12] = L_FREE      # 2×2 free
        labels[15, 20] = L_OCC             # 一个障碍格
    return PriorMap(labels=labels, origin_xy=np.array([-2.0, -2.0]), res_m=0.1, world_scale=1.0,
                    base_sid="ROOT", meta={"built_wall": 1.0}, path=Path("mem://prior.npz"))


class TestMemorySummaryExposesPriors(unittest.TestCase):
    """``/worldmodel/navmesh/memory`` 要带 ``priors``：面板/管理页靠它区分
    "没设世界身份" 与 "设了但那个世界还没固化先验"（两件事、两个动作）。"""

    def test_priors_map_present_and_reads_json_only(self) -> None:
        import json
        import types
        from neko_anyadance_body.backend.nav_memory import MemoryConfig, NavMemoryStore, world_dir_name
        from neko_anyadance_body.backend.service import BackendService

        with tempfile.TemporaryDirectory() as tmp:
            store = NavMemoryStore(Path(tmp), MemoryConfig())
            wdir = store.root / world_dir_name("wrld_x")
            wdir.mkdir(parents=True)
            (wdir / "world.json").write_text(json.dumps({"world_key": "wrld_x"}), encoding="utf-8")
            write_prior(wdir, np.array([[L_FREE, L_OCC]], np.uint8), (0.0, 0.0), 0.1,
                        world_scale=0.755, base_sid="ROOT")
            (wdir / "prior" / "prior.npz").unlink()          # 只剩 json 也要报得出来

            svc = BackendService.__new__(BackendService)      # 只测这一段组合，不启整套运行时
            svc.navmesh_memory = store
            svc.world_model = types.SimpleNamespace(identity=lambda: {"world_key": None})
            out = svc.navmesh_memory_summary()
        self.assertIn("priors", out)
        got = out["priors"][wdir.name]
        self.assertEqual((got["free_cells"], got["occ_cells"]), (1, 1))
        self.assertEqual(got["base_sid"], "ROOT")


class TestWriteLoad(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wdir = Path(self._tmp.name) / "wrld_x"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_roundtrip_and_counts(self) -> None:
        lab = np.zeros((20, 30), np.uint8)
        lab[2:5, 3:6] = L_FREE
        lab[10, 10] = L_OCC
        path = write_prior(self.wdir, lab, (-1.0, -2.0), 0.1, world_scale=0.755, base_sid="BASE",
                           meta={"tool": "t"})
        self.assertTrue(path.is_file())
        p, why = load_prior(self.wdir)
        self.assertEqual(why, "")
        assert p is not None
        self.assertEqual(p.labels.shape, (20, 30))
        self.assertTrue(np.array_equal(p.labels, lab))
        self.assertAlmostEqual(p.res_m, 0.1)
        self.assertAlmostEqual(p.world_scale, 0.755)
        self.assertEqual(p.base_sid, "BASE")
        self.assertEqual(p.meta["tool"], "t")
        self.assertEqual(p.meta["counts"], {"free": 9, "occ": 1, "unknown": 590})
        # 临时文件不残留（原子写的中间态只在 replace 前存在）
        names = sorted(f.name for f in prior_dir(self.wdir).iterdir())
        self.assertEqual(names, ["prior.json", "prior.npz"])

    def test_missing_is_no_prior(self) -> None:
        self.assertEqual(load_prior(self.wdir), (None, "no_prior"))
        prior_dir(self.wdir).mkdir(parents=True)
        (prior_dir(self.wdir) / "prior.npz").write_bytes(b"")
        self.assertEqual(load_prior(self.wdir), (None, "no_prior"))   # json 缺 ⇒ 成对不成立

    def test_prior_meta_reads_json_only(self) -> None:
        """``prior_meta``：只读 prior.json 摘要（给管理页/面板说"这个世界的先验在不在"）。

        它必须**不碰 npz** —— 所以 npz 故意删掉后仍要能报出格子数与固化时间。
        """
        self.assertIsNone(prior_meta(self.wdir), "没固化过 ⇒ 没有")
        write_prior(self.wdir, np.array([[L_FREE, L_OCC]], np.uint8), (-1.0, -2.0), 0.1,
                    world_scale=0.755, base_sid="BASE",
                    meta={"sessions": [{"sid": "S1"}, {"sid": "S2"}]})
        (prior_dir(self.wdir) / "prior.npz").unlink()               # 只剩 json
        got = prior_meta(self.wdir)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual((got["free_cells"], got["occ_cells"]), (1, 1))
        self.assertEqual(got["base_sid"], "BASE")
        self.assertEqual(got["sessions"], ["S1", "S2"])
        self.assertTrue(got["built_at"], "要给「哪天固化的」，否则没法判断新旧")
        (prior_dir(self.wdir) / "prior.json").write_text('{"schema": 99}', encoding="utf-8")
        self.assertIsNone(prior_meta(self.wdir), "schema 不认识 ⇒ 说没有，不猜")

    def test_pair_mismatch_refused(self) -> None:
        lab = np.zeros((4, 4), np.uint8)
        write_prior(self.wdir, lab, (0.0, 0.0), 0.1, world_scale=1.0, base_sid="B")
        jp = prior_dir(self.wdir) / "prior.json"
        doc = json.loads(jp.read_text(encoding="utf-8"))
        doc["built_wall"] = float(doc["built_wall"]) + 1.0     # 模拟读到"新 npz + 旧 json"
        jp.write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(load_prior(self.wdir), (None, "pair_mismatch"))

    def test_bad_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_prior(self.wdir, np.full((3, 3), 7, np.uint8), (0, 0), 0.1, world_scale=1.0, base_sid="B")
        with self.assertRaises(ValueError):
            write_prior(self.wdir, np.zeros((3, 3), np.uint8), (0, 0), 0.0, world_scale=1.0, base_sid="B")
        with self.assertRaises(ValueError):
            write_prior(self.wdir, np.zeros((3,), np.uint8), (0, 0), 0.1, world_scale=1.0, base_sid="B")


class TestProjectOverlay(unittest.TestCase):
    def test_identity_overlay_and_live_wins(self) -> None:
        prior = sample_prior()
        ovl = project_prior(prior, np.eye(2), np.zeros(2))
        ng = make_grid()
        # live 已看的格：先验不许动（把先验唯一的 OCC 格先画成 live FREE）
        r, c = cell_of(ng, *SAMPLE_XY[4])
        ng.grid[r, c] = FREE
        st = ovl.apply_into(ng)
        self.assertEqual(st["cells"], 5)
        self.assertEqual(st["in_crop"], 5)
        self.assertEqual(st["out_crop"], 0)
        self.assertEqual(st["applied_free"], 4)
        self.assertEqual(st["applied_occ"], 0)    # 唯一的 OCC 格被 live FREE 挡住
        self.assertEqual(st["skipped_known"], 1)
        self.assertEqual(ng.grid[r, c], FREE)
        self.assertEqual(int((ng.grid == FREE).sum()), 5)
        self.assertEqual(int((ng.grid == OCC).sum()), 0)
        self.assertEqual(int((ng.grid == UNK).sum()), ng.grid.size - 5)
        # applied_mask = "真写进栅格的那几格"，是 `prior` 图层区分"先验补的/live 自己的"的判据
        self.assertEqual(ovl.applied_mask.shape, ovl.labels.shape)
        self.assertEqual(int(ovl.applied_mask.sum()), 4)
        self.assertFalse(bool(ovl.applied_mask[4]), "被 live 挡下的那格不许记成已采纳")

    def test_rotation_places_prior_in_session_frame(self) -> None:
        # 世界系里一个障碍格：格心 (−0.95, 0.05)。gauge = 绕 z 转 +90°（p_world = R2·p_sess）：
        # 会话帧里它应在 R2.T @ (−0.95, 0.05) = (0.05, 0.95)（R2.T = [[0,1],[−1,0]]）。
        lab = np.zeros((40, 40), np.uint8)
        lab[20, 10] = L_OCC
        prior = sample_prior(lab)
        r2 = yaw(90.0)[:2, :2]
        ovl = project_prior(prior, r2, np.zeros(2))
        self.assertTrue(np.allclose(ovl.xy_session[0], r2.T @ np.array([-0.95, 0.05]), atol=1e-9))
        ng = make_grid()
        ovl.apply_into(ng)
        r, c = cell_of(ng, 0.05, 0.95)
        self.assertEqual(ng.grid[r, c], OCC)
        self.assertEqual(int((ng.grid == OCC).sum()), 1)

    def test_walked_corridor_blocks_prior_obstacle(self) -> None:
        lab = np.zeros((40, 40), np.uint8)
        lab[15, 20] = L_OCC
        prior = sample_prior(lab)
        ovl = project_prior(prior, np.eye(2), np.zeros(2))
        ng = make_grid()
        # 身体走线穿过该格（追踪米折线；本测试 world_scale=1）
        x, y = SAMPLE_XY[4]
        walked = [np.array([[x - 0.5, y], [x + 0.5, y]])]
        st = ovl.apply_into(ng, walked)
        self.assertEqual(st["blocked_walked"], 1)
        self.assertEqual(st["applied_occ"], 0)
        self.assertEqual(int((ng.grid == OCC).sum()), 0)

    def test_out_of_crop_counted(self) -> None:
        lab = np.zeros((4, 4), np.uint8)
        lab[0, 0] = L_OCC
        prior = sample_prior(lab)
        ovl = project_prior(prior, np.eye(2), np.zeros(2))
        ng = make_grid(shape=(5, 5), origin=(10.0, 10.0))       # 图离先验很远
        st = ovl.apply_into(ng)
        self.assertEqual(st["out_crop"], 1)
        self.assertEqual(st["applied_occ"], 0)


THETA = 0.0          # 先验测试用的真值 gauge：会话 → 世界（保持纯平移，便于手算）
T_TRUE = np.array([1.5, -0.7])


def synth(n: int, *, noise: float = 0.0, seed: int = 3, theta: float = THETA,
          t_true: np.ndarray = T_TRUE):
    """构造与真值 gauge 一致的约束集（noise=0 ⇒ ``estimate_gauge`` 应逐位恢复）。

    ``theta``：该批约束的真值 yaw（弧度）。**同一场会话里前后两半给不同的 theta 就是在模拟
    "会话帧随时间长转"** —— 2026-10-06 live 那场就是这么坏掉的（见 ``_cross_check``）。

    返回 (recs, poses_new, poses_old)：new 位姿在会话帧、old 位姿在"旧会话表所在系"
    （即先验所在世界系）。
    """
    rng = np.random.default_rng(seed)
    g3 = yaw(np.degrees(theta))
    r2 = g3[:2, :2]
    poses_new: dict[int, np.ndarray] = {}
    poses_old: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    recs = []
    for i in range(n):
        p_n = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3)])
        r_n = yaw(rng.uniform(-180, 180))
        p_o = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3)])
        r_o = yaw(rng.uniform(-180, 180))
        p_pred = r2 @ p_n + np.asarray(t_true) + np.array([noise * (1 if i % 2 else -1), 0.0])
        T = np.eye(4)
        T[:3, :3], T[:2, 3] = r_n, p_n
        poses_new[i] = T
        poses_old[("OLD", i)] = (r_o, p_o)
        recs.append({"new_sid": "NEW", "new_kf": i, "old_sid": "OLD", "old_kf": i,
                     "inliers": 40,
                     "t_ab": (r_o.T @ np.array([p_pred[0], p_pred[1], 0.0])
                              - r_o.T @ np.array([p_o[0], p_o[1], 0.0])).tolist(),
                     "R_ab": (r_o.T @ g3 @ r_n).tolist()})
    return recs, poses_new, poses_old


class TestPriorConsumer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wdir = Path(self._tmp.name) / "wrld_x"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_sample(self, *, world_scale: float = 0.755, base_sid: str = "ROOT") -> None:
        write_prior(self.wdir, sample_prior().labels, (-2.0, -2.0), 0.1,
                    world_scale=world_scale, base_sid=base_sid)

    def consumer(self, poses_new, poses_old, *, bases: dict | None = None,
                 world_scale: float = 0.755) -> PriorConsumer:
        return PriorConsumer(
            PriorConfig(), self.wdir, world_scale=world_scale,
            pose_of=lambda k: poses_new.get(int(k)),
            pose_of_old=lambda sid, kf: poses_old.get((str(sid), int(kf))),
            table_base=lambda sid: (bases or {}).get(str(sid), "ROOT"))

    def test_no_prior_file(self) -> None:
        recs, pn, po = synth(12)
        c = self.consumer(pn, po)
        c.add_constraints(recs)
        self.assertIsNone(c.overlay())
        self.assertEqual(c.status()["state"], "no_prior")
        self.assertIsNotNone(c.status()["gauge"])                 # gauge 照算，只是没图可注入

    def test_world_scale_mismatch_refused(self) -> None:
        self.write_sample(world_scale=0.9)
        recs, pn, po = synth(12)
        c = self.consumer(pn, po, world_scale=0.755)
        c.add_constraints(recs)
        self.assertIsNone(c.overlay())
        self.assertEqual(c.status()["state"], "world_scale_mismatch")

    def test_too_few_constraints(self) -> None:
        self.write_sample()
        recs, pn, po = synth(5)
        c = self.consumer(pn, po)
        c.add_constraints(recs)
        self.assertIsNone(c.overlay())
        self.assertEqual(c.status()["state"], "gauge_too_few")

    def test_ok_projects_with_recovered_gauge(self) -> None:
        self.write_sample()
        recs, pn, po = synth(12)
        c = self.consumer(pn, po)
        c.add_constraints(recs)
        ovl = c.overlay()
        self.assertIsNotNone(ovl)
        st = c.status()
        self.assertEqual(st["state"], "ok")
        self.assertEqual(st["gauge"]["n_inlier"], 12)
        self.assertLess(st["gauge"]["pos_med_m"], 1e-6)
        self.assertEqual(st["prior"]["base_sid"], "ROOT")
        # 投影用恢复出的 gauge：格心 = 世界格心 − t_G（θ=0）
        assert ovl is not None
        self.assertTrue(np.allclose(ovl.xy_session, SAMPLE_XY - T_TRUE, atol=1e-9))
        ng = make_grid(shape=(60, 60), origin=(-4.0, -4.0))
        info = ovl.apply_into(ng)
        self.assertEqual(info["applied_free"], 4)
        self.assertEqual(info["applied_occ"], 1)
        self.assertEqual((c.status()["applied"] or {})["applied_occ"], 1)

    def test_pos_residual_gate(self) -> None:
        self.write_sample()
        recs, pn, po = synth(12, noise=2.0)     # ±2 m 交替散点：IRLS 剔不完，内点残差 1.6 > 0.5
        c = self.consumer(pn, po)
        c.add_constraints(recs)
        self.assertIsNone(c.overlay())
        self.assertEqual(c.status()["state"], "gauge_pos_residual")

    def test_time_drifting_frame_is_refused(self) -> None:
        """**live 那个 bug 的回归**：会话帧随时间在转时，前后半各估的 gauge 对不上 ⇒ 不许注入。

        2026-10-06 实况：16 条约束时池化残差全过闸（pos 0.39 m / rot 1.8°，两组**交错**子集
        也一致 0.375 m / 2.3°），但那个 gauge 的 yaw 是 2.4°、全场 161 条给的是 8.6° ——
        先验按错 6° 的变换画了满屏废图，约束攒多后闸门改判拒绝、地图自己"恢复"。
        """
        self.write_sample()
        a, pn_a, po_a = synth(10, theta=0.0, seed=11)
        b, pn_b, po_b = synth(10, theta=math.radians(6.0), seed=12)   # 后半段整体转了 6°
        pn = {**pn_a, **{k + 100: v for k, v in pn_b.items()}}
        po = {**po_a, **{(s, k + 100): v for (s, k), v in po_b.items()}}
        recs = a + [{**c, "new_kf": c["new_kf"] + 100, "old_kf": c["old_kf"] + 100} for c in b]
        c = self.consumer(pn, po)
        c.add_constraints(recs)
        self.assertIsNone(c.overlay(), "会话帧在漂移时不许注入先验")
        st = c.status()
        self.assertIn(st["state"], {"gauge_unstable", "gauge_inlier_frac", "gauge_pos_residual"}, st)
        self.assertIsNotNone(st["split"], "交叉验证的数字要进 status（现场可核对）")
        assert st["split"] is not None
        self.assertGreater(st["split"]["yaw_deg"], 2.5, f"前后半 yaw 差应被看见：{st['split']}")

    def test_cross_check_passes_when_frame_is_consistent(self) -> None:
        self.write_sample()
        recs, pn, po = synth(20)
        c = self.consumer(pn, po)
        c.add_constraints(recs)
        self.assertIsNotNone(c.overlay())
        st = c.status()
        self.assertEqual(st["state"], "ok")
        self.assertLess(st["split"]["m"], 0.05, st["split"])
        self.assertLess(st["split"]["yaw_deg"], 0.5, st["split"])

    def test_frame_root_mismatch_refused(self) -> None:
        self.write_sample(base_sid="ROOT")
        recs, pn, po = synth(12)
        c = self.consumer(pn, po, bases={"OLD": "OTHER"})     # 旧会话表在另一棵树 ⇒ 绝不混 gauge
        c.add_constraints(recs)
        self.assertIsNone(c.overlay())
        self.assertEqual(c.status()["state"], "frame_mismatch")
        self.assertEqual(c.status()["n_constraints"], 12)

    def test_map_moved_reestimates_gauge(self) -> None:
        self.write_sample()
        recs, pn, po = synth(12)
        c = self.consumer(pn, po)
        c.add_constraints(recs)
        ovl0 = c.overlay()
        assert ovl0 is not None
        before = ovl0.xy_session.copy()
        delta = np.array([0.5, 0.2])
        for T in pn.values():                                # 模拟回环：会话帧全表平移
            T[:2, 3] = T[:2, 3] + delta
        c.note_map_moved()
        ovl1 = c.overlay()
        assert ovl1 is not None
        self.assertTrue(np.allclose(ovl1.xy_session, before + delta, atol=1e-9))
        self.assertEqual(c.status()["state"], "ok")


class TestSessionHook(unittest.TestCase):
    def test_session_consumes_overlay_before_build(self) -> None:
        """NavSession：先验在 build 之前叠进栅格，且先验覆盖区被并进裁剪范围（扩边）。"""
        mapper = KeyframeGridMapper()
        rng = np.random.default_rng(5)
        pts = np.column_stack([rng.uniform(-0.3, 0.3, 400), rng.uniform(-0.3, 0.3, 400),
                               -1.73 + rng.uniform(-0.02, 0.02, 400)]).astype(np.float32)
        mapper.add_keyframe(0, pts, np.eye(4))
        lab = np.zeros((40, 40), np.uint8)
        lab[15, 20] = L_OCC                 # 世界 (0.05, −0.45)：live 视距内
        lab[35, 35] = L_FREE                # 世界 (1.55, 1.55)：live 视距之外，靠扩边进图
        prior = sample_prior(lab)
        ovl = project_prior(prior, np.eye(2), np.zeros(2))
        s = NavSession(mapper, prior=lambda: ovl)
        res = s.compute({"state": "localized", "xy": (0.0, 0.0)}, s.snapshot())
        ng = res["ng"]
        self.assertIsNotNone(res["info"].get("prior"))
        self.assertEqual(res["info"]["prior"]["applied_occ"], 1)
        r, c = cell_of(ng, 1.55, 1.55)
        self.assertTrue(ng.inside((r, c)), "先验四角没并进裁剪范围")
        self.assertEqual(ng.grid[r, c], FREE)
        r, c = cell_of(ng, *SAMPLE_XY[4])
        self.assertEqual(ng.grid[r, c], OCC)
        # build 照常跑（先验障碍格不许进可走区）
        self.assertIn("walkable_center_m2", res["info"]["grid"])
        self.assertFalse(ng.center[r, c])

    def test_session_without_prior_unchanged(self) -> None:
        mapper = KeyframeGridMapper()
        rng = np.random.default_rng(6)
        pts = np.column_stack([rng.uniform(-0.3, 0.3, 400), rng.uniform(-0.3, 0.3, 400),
                               -1.73 + rng.uniform(-0.02, 0.02, 400)]).astype(np.float32)
        mapper.add_keyframe(0, pts, np.eye(4))
        s = NavSession(mapper)
        res = s.compute({"state": "localized", "xy": (0.0, 0.0)}, s.snapshot())
        self.assertNotIn("prior", res["info"])
        self.assertFalse((res["ng"].grid == OCC).any())


if __name__ == "__main__":
    unittest.main()