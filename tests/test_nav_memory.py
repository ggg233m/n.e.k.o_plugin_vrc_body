from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body import config
from neko_anyadance_body.backend.nav_memory import (MemoryConfig, NavMemoryStore, config_from_plugin,
                                                    load_features, world_dir_name)

WORLD = {"world_key": "wrld_abc:inst/1", "world_name": "泳池", "world_source": "manual_id"}


def feat(n: int = 50) -> SimpleNamespace:
    rng = np.random.default_rng(n)
    return SimpleNamespace(xyz=rng.normal(size=(n, 3)).astype(np.float32),
                           des3d=rng.integers(0, 255, (n, 32), dtype=np.uint8),
                           K=np.eye(3), size=(720, 405), eye_y=0.0, uv=np.zeros((n, 2)))


def table(n: int) -> dict:
    T = np.repeat(np.eye(4)[None], n, 0)
    T[:, 0, 3] = np.arange(n)
    return {"ids": np.arange(n), "T_map": T, "T_dr": T, "dist_m": np.arange(n, dtype=float),
            "t_s": np.arange(n, dtype=float)}


class NavMemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = NavMemoryStore(self.root, MemoryConfig(min_session_keyframes=3))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _session(self, n: int = 5, refresh_last: bool = False) -> str:
        w, reason = self.store.begin(WORLD, {"world_scale": 0.755})
        self.assertIsNone(reason)
        thumb = np.zeros((405, 720, 3), np.uint8)
        for k in range(n):
            w.keyframe(k, feat(), thumb[:180, :320], refresh=refresh_last and k == n - 1)
        self.store.end(table(n))
        return w.session_id

    def test_unknown_world_or_disabled_never_writes(self) -> None:
        self.assertEqual(self.store.begin({"world_key": None}, {})[1], "world_unknown")
        off = NavMemoryStore(self.root, MemoryConfig(enabled=False))
        self.assertEqual(off.begin(WORLD, {})[1], "memory_disabled")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_list_worlds_can_skip_the_size_walk(self) -> None:
        """``sizes=False`` 必须**真的不扫目录**：体积是纯展示值，但 ``_dir_bytes`` 递归 stat
        上万个文件（现役 10957 个 / 126 MB），会把只想要"有哪些世界"的调用方超时打爆
        （2026-10-06 实况：栅格页读世界列表 8 s 超时 → 面板报 abort）。"""
        from neko_anyadance_body.backend import nav_memory
        self._session(5)
        calls: list[int] = []
        real = nav_memory._dir_bytes

        def spy(path):
            calls.append(1)
            return real(path)

        nav_memory._dir_bytes = spy
        try:
            light = self.store.list_worlds(sizes=False)
            self.assertEqual(calls, [], "sizes=False 时一次都不该扫")
            self.assertIsNone(light[0]["mb"], "没缓存过就如实报 None，不假装知道体积")
            full = self.store.list_worlds()
            self.assertEqual(len(calls), 1, "要体积时扫一次")
            self.assertGreater(full[0]["mb"], 0)
            self.store.list_worlds()                      # 命中 TTL 缓存
            self.assertEqual(len(calls), 1, "60 s 内不该重复扫")
            again = self.store.list_worlds(sizes=False)
            self.assertIsNotNone(again[0]["mb"], "缓存过的体积可以白给")
            self.assertIsInstance(self.store.summary(sizes=False), dict)
        finally:
            nav_memory._dir_bytes = real

    def test_session_roundtrip_and_refresh_replaces_previous(self) -> None:
        sid = self._session(5, refresh_last=True)
        wid = world_dir_name(WORLD["world_key"])
        self.assertNotIn("/", wid)
        info = self.store.session(wid, sid)
        self.assertEqual(info["status"], "complete")
        self.assertEqual(info["keyframes"], 5)
        self.assertEqual(info["features"], 4)          # kf 3 让位给原地补帧 kf 4
        self.assertEqual(info["thumbnails"], [0, 1, 2, 4])
        poses = self.store.load_poses(wid, sid)
        self.assertEqual(poses["has_feat"].tolist(), [True, True, True, False, True])
        f = load_features(self.root / wid / "sessions" / sid / "kf" / "000004.npz")
        self.assertEqual(f["des3d"].shape, (50, 32))
        self.assertTrue(self.store.thumbnail(wid, sid, 0).startswith(b"\xff\xd8"))
        self.assertEqual(self.store.list_worlds()[0]["world_key"], WORLD["world_key"])

    def test_short_session_is_discarded(self) -> None:
        self._session(2)
        self.assertEqual(self.store.list_sessions(world_dir_name(WORLD["world_key"])), [])

    def test_active_session_is_protected(self) -> None:
        w, _ = self.store.begin(WORLD, {})
        wid = world_dir_name(WORLD["world_key"])
        self.assertEqual(self.store.begin(WORLD, {})[1], "session_already_active")
        with self.assertRaises(PermissionError):
            self.store.delete_session(wid, w.session_id)
        with self.assertRaises(PermissionError):
            self.store.delete_world(wid)
        self.assertEqual(self.store.prune()["deleted"], [])
        self.store.end(None)

    def test_ids_from_http_cannot_escape_root(self) -> None:
        sid = self._session()
        for bad in ("..", "../x", "a/b", "", "c:\\x"):
            with self.assertRaises((ValueError, KeyError)):
                self.store.list_sessions(bad)
        wid = world_dir_name(WORLD["world_key"])
        for bad in ("..", "../../x", "a/b"):
            with self.assertRaises(ValueError):
                self.store.delete_session(wid, bad)
        self.assertTrue((self.root / wid / "sessions" / sid).exists())

    def test_prune_keeps_pinned_and_deletes_oldest(self) -> None:
        store = NavMemoryStore(self.root, MemoryConfig(min_session_keyframes=1, max_sessions_per_world=2))
        self.store = store
        sids = []
        for _ in range(3):
            sids.append(self._session(3))
            time.sleep(1.05)                           # sid 精度是秒
        wid = world_dir_name(WORLD["world_key"])
        # prune 在 end 里自动跑：第三个结束时删掉最旧的。
        self.assertEqual([s["session_id"] for s in store.list_sessions(wid)], sids[:0:-1])
        store.update_session(wid, sids[1], pinned=True, label="  出生点旁  ")
        sids.append(self._session(3))
        left = {s["session_id"]: s for s in store.list_sessions(wid)}
        self.assertEqual(set(left), {sids[1], sids[3]})
        self.assertEqual(left[sids[1]]["label"], "出生点旁")
        with self.assertRaises(ValueError):
            store.update_session(wid, sids[1], pinned="yes")

    def test_recover_marks_crashed_session(self) -> None:
        w, _ = self.store.begin(WORLD, {})
        w.poses(table(3))
        time.sleep(0.2)
        meta_path = w.dir / "session.json"
        fresh = NavMemoryStore(self.root, MemoryConfig())      # 模拟重启：新进程不知道旧 writer
        self.assertEqual(fresh.recover(), 1)
        self.assertEqual(json.loads(meta_path.read_text(encoding="utf-8"))["status"], "interrupted")
        self.store.end(None)

    def test_plugin_config_maps_field_for_field(self) -> None:
        cfg = config.PluginConfig.from_mapping({"navmesh": {"memory": {"max_total_mb": 100, "enabled": False}}})
        mem = config_from_plugin(cfg.navmesh.memory)
        self.assertEqual((mem.max_total_mb, mem.enabled), (100.0, False))
        with self.assertRaises(ValueError):
            config.PluginConfig.from_mapping({"navmesh": {"memory": {"max_totl_mb": 1}}})


if __name__ == "__main__":
    unittest.main()
