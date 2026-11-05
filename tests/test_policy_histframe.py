# -*- coding: utf-8 -*-
"""视觉历史帧间隔 0.4 s → 0.2 s 的回归测试（2026-11-05）。

锁死五条，全部不依赖真实视频 / GPU / VRChat：
  1. 共享常量 policy_histframe.HISTORY_FRAME_S == 0.2，且在 5 Hz 网格上是 1 步；
  2. 训练侧 build_index 给样本选的 prev_img 是 t−0.2 s 那一帧；
  3. PolicyRunner._prev 按 t−0.2 s 取帧，取不到返回 None（调用方回退当前帧）；
  4. nav_online.policy_frame_before 默认参数是 HISTORY_FRAME_S，缺历史返回 None；
  5. service._policy_prev_frame 传的是 HISTORY_FRAME_S（不是字面量 0.4）。
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

import policy_histframe as H  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402


class SharedConstantTests(unittest.TestCase):
    def test_interval_is_0_2s(self):
        self.assertAlmostEqual(H.HISTORY_FRAME_S, 0.2, places=9)

    def test_aligned_with_5hz_grid(self):
        # 0.2 / 0.2 = 1 步，必须是整数步，否则「历史帧」会落在两个观测点之间
        self.assertAlmostEqual(H.OBS_GRID_S, 0.2, places=9)
        self.assertEqual(H.HISTORY_FRAME_STEPS, 1)

    def test_offset_matches_constant(self):
        self.assertAlmostEqual(H.history_frame_offset_s(), H.HISTORY_FRAME_S, places=12)


class TrainIndexPrevImgTests(unittest.TestCase):
    """训练侧：build_index 必须给样本选 t−0.2 s 的那一帧，不是 t−0.4 s。"""

    def _make_dataset(self):
        tmp = Path(tempfile.mkdtemp())
        sess = tmp / "sessA"
        sess.mkdir()
        # 造 6 个 5 Hz 样本：t = 0,0.2,0.4,0.6,0.8,1.0
        ts = [0.2 * i for i in range(6)]
        obs, lab = [], []
        for i, t in enumerate(ts):
            obs.append(dict(t=t, t_rel=t, i=i, img_L=f"images/{i:04d}.jpg"));
            lab.append(dict(t=t, act=[[0.0, 1.0, 0.0]] * 12, mask=[1] * 12,
                            cls="move", wp=[0.0, 0.0], split="train",
                            mode=0, topo=[0.0, 0.0, 0.0]))
        (sess / "policy_obs.jsonl").write_text(
            "\n".join(json.dumps(o) for o in obs), encoding="utf-8")
        (sess / "policy_labels.jsonl").write_text(
            "\n".join(json.dumps(l) for l in lab), encoding="utf-8")
        return tmp, ts

    def test_prev_img_is_one_step_back(self):
        import train_policy_lora as T
        tmp, ts = self._make_dataset()
        samples = T.build_index(tmp, None)
        by_t = {round(float(s["t"]), 6): s for s in samples}
        # 取中间某点，向前一个样本（0.2 s）
        t = round(ts[4], 6)
        self.assertIn(t, by_t)
        prev_name = Path(by_t[t]["prev_img"]).name
        self.assertEqual(prev_name, "0003.jpg",
                         "prev_img 应为 t−0.2 s（索引 3），实际 %s" % prev_name)

    def test_prev_img_not_two_steps_back(self):
        import train_policy_lora as T
        tmp, ts = self._make_dataset()
        samples = T.build_index(tmp, None)
        by_t = {round(float(s["t"]), 6): s for s in samples}
        t = round(ts[4], 6)
        prev_name = Path(by_t[t]["prev_img"]).name
        self.assertNotEqual(prev_name, "0002.jpg",
                            "prev_img 仍是 t−0.4 s（旧行为），改造未生效")


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def act(self, cur, prev, wp, body, preprocessed=False):
        self.calls.append(dict(cur=cur, prev=prev))
        return dict(block=[[0.0, 0.0]] * 12, gated=False, obs_t=0.0, latency_ms=1.0)


def _frame(v):
    return np.full((4, 4, 3), v, np.uint8)


class RunnerPrevTests(unittest.TestCase):
    """runner 的 _prev 必须按 0.2 s 取，并且取不到时返回 None（不抛、不乱取）。"""

    def _mk(self):
        rt = _FakeRuntime()
        r = PolicyRunner(rt, lambda: _frame(9), lambda: (0.0, 1.0), lambda: (0.0, 1.0),
                         on_block=lambda *a: None, dry_run=True)
        return r, rt

    def test_prev_at_0_2s(self):
        r, _ = self._mk();
        t = 100.0
        r._push(t - 0.2, _frame(7))   # 正好 0.2 s 前
        r._push(t - 0.1, _frame(8))
        got = r._prev(t)
        self.assertIsNotNone(got)
        self.assertEqual(int(got[0, 0, 0]), 7)

    def test_missing_returns_none(self):
        r, _ = self._mk();
        t = 100.0
        r._push(t - 0.9, _frame(3))   # 只有很远的历史，超出 0.15 s 容差
        self.assertIsNone(r._prev(t), "取不到 0.2 s 历史时必须返回 None，由调用方回退当前帧")

    def test_prev_not_0_4(self):
        r, _ = self._mk();
        t = 100.0
        r._push(t - 0.4, _frame(5))   # 只有 0.4 s 前那一帧
        self.assertIsNone(r._prev(t),
                          "0.4 s 前的帧不应再被当作历史帧（间隔已改 0.2 s）")


class ServiceHistoryFrameTests(unittest.TestCase):
    """服务端：policy_frame_before 默认取 HISTORY_FRAME_S，缺历史返回 None。"""

    def _nav(self, with_hist=True):
        from backend.nav_online import OnlineNavigator
        # 只测方法本身，不构造整个导航器：用最小替身挂上所需属性
        class _Stub:
            pass
        s = _Stub()
        s._state_lock = __import__("threading").Lock()
        s._clock = lambda: 100.0
        s._policy_hist = __import__("collections").deque(maxlen=12)
        if with_hist:
            s._policy_hist.append((99.8, _frame(4)))   # 0.2 s 前
        s.policy_frame_before = OnlineNavigator.policy_frame_before.__get__(s)
        return s

    def test_default_arg_is_shared_constant(self):
        import inspect
        from backend.nav_online import OnlineNavigator
        sig = inspect.signature(OnlineNavigator.policy_frame_before)
        default = sig.parameters["seconds"].default
        self.assertAlmostEqual(float(default), float(H.HISTORY_FRAME_S), places=9)

    def test_returns_frame_within_tolerance(self):
        s = self._nav(with_hist=True)
        got = s.policy_frame_before()
        self.assertIsNotNone(got)
        rgb, at = got
        self.assertEqual(int(rgb[0, 0, 0]), 4)
        self.assertAlmostEqual(at, 99.8, places=6)

    def test_empty_history_returns_none(self):
        s = self._nav(with_hist=False)
        self.assertIsNone(s.policy_frame_before(), "无历史必须返回 None，调用方回退当前帧")


class ManifestHistoryFrameTests(unittest.TestCase):
    """需求 7 第 5 条：manifest 必须记录历史帧间隔，且新旧口径可区分。

    为什么这条必须测：推理侧靠 manifest 判断「训练时历史帧是 0.2 还是 0.4」。
    若 manifest 不记或记错，推理会喂进时间跨度不同的历史帧——
    形状完全合法，RuntimeError 防不住，属于静默错误。
    """

    def test_new_schema_records_shared_constant(self):
        import input_schema as S
        fake = type("A", (), dict(drop_wp=True, quad_feats=False,
                                  prompt_format="card", act_dim=3))
        s = S.from_args(fake)
        self.assertIn("history_frame_s", s)
        self.assertAlmostEqual(float(s["history_frame_s"]), float(H.HISTORY_FRAME_S), places=9)
        self.assertAlmostEqual(float(s["history_frame_s"]), 0.2, places=9)

    def test_legacy_defaults_keep_old_caliber(self):
        # 旧 checkpoint（无 manifest）按当时的 0.4 s 口径回退，不能被新常量覆盖
        import input_schema as S
        self.assertAlmostEqual(float(S.LEGACY_DEFAULTS["history_frame_s"]), 0.4, places=9)

    def test_manifest_roundtrip_preserves_interval(self):
        import input_schema as S
        tmp = Path(tempfile.mkdtemp())
        fake = type("A", (), dict(drop_wp=True, act_dim=3))
        S.save(tmp, S.from_args(fake))
        back = S.load(tmp)
        self.assertAlmostEqual(float(back["history_frame_s"]), 0.2, places=9)
        self.assertFalse(bool(back["legacy"]), "新 manifest 不应被判成 legacy")

if __name__ == "__main__":
    unittest.main()