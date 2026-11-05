# -*- coding: utf-8 -*-
"""真驱动 policy head 启动时必须确保 body 输出 enable（剖析 §15.18）。

body disabled 时摇杆 overlay 不随帧发出：轴提交"成功"却没人发出去，
表现为"只转头（play-space yaw 走 elif 分支照发）不移动"。
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_service():
    """service.py 是包内模块（相对导入 ..osc），按包路径加载。"""
    from tests import _bootstrap  # noqa: F401  造 neko_anyadance_body 包别名
    import neko_anyadance_body.backend.service as mod
    return mod


class BodyAutoEnableTests(unittest.TestCase):
    """_start_policy_head 在 dry_run=False 且 body=disabled 时必须 submit enable。"""

    def _run_start(self, scheduler_state, dry_run):
        mod = _load_service()
        svc = mod.BackendService.__new__(mod.BackendService)
        svc.scheduler = SimpleNamespace(snapshot=lambda: {"state": scheduler_state})
        submits = []

        def fake_submit(kind, params=None, **kw):
            submits.append(kind)
            return {"accepted": True, "state": "idle"}

        svc.submit = fake_submit
        svc.logger = None
        svc._started = True
        svc.state_dir = Path(".tmp")
        svc.navmesh = SimpleNamespace(drive_enabled=True, policy_hold=False)
        svc._policy_frame_source = lambda: None
        svc._policy_prev_frame = lambda: None
        svc._policy_waypoint = lambda: (0.0, 0.0)
        svc._policy_body = lambda: (0.0, 0.0)
        svc._policy_send_axes = lambda x, y, d: True
        svc._policy_near_obstacle = lambda: (False, 0)

        head_calls = []
        fake_head = mock.MagicMock()
        fake_head.start = lambda: head_calls.append(1)
        fake_head_cls = mock.MagicMock(return_value=fake_head)

        env = {"VRC_POLICY_HEAD": "1", "VRC_POLICY_MODEL": "M", "VRC_POLICY_RUN": "R",
               "VRC_POLICY_DRYRUN": "0" if not dry_run else "1"}
        # _start_policy_head 里是惰性导入 from .policy_head import PolicyHead，
        # 所以要 patch 源模块 policy_head，而不是 service 的命名空间。
        import neko_anyadance_body.backend.policy_head as ph_mod
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(ph_mod, "PolicyHead", fake_head_cls):
            svc._start_policy_head()
        return head_calls, submits, fake_head_cls.call_args.kwargs

    def test_real_drive_enables_disabled_body(self):
        head_calls, submits, _ = self._run_start("disabled", dry_run=False)
        self.assertIn("enable", submits)
        self.assertEqual(head_calls, [1])

    def test_real_drive_skips_enable_when_already_enabled(self):
        _, submits, _ = self._run_start("idle", dry_run=False)
        self.assertNotIn("enable", submits)

    def test_dry_run_never_touches_body(self):
        _, submits, _ = self._run_start("disabled", dry_run=True)
        self.assertNotIn("enable", submits)

    def test_astar_is_soft_guidance_input(self):
        _, _, kwargs = self._run_start("idle", dry_run=True)
        self.assertIsNone(kwargs["wp_provider"])
        self.assertIsNotNone(kwargs["guidance_provider"])
        self.assertIsNone(kwargs["hold_provider"])
        self.assertIsNone(kwargs["floor_provider"])
        self.assertEqual(kwargs["floor_frac"], 0.0)
        self.assertEqual(kwargs["wp_stop"], 0.0)


if __name__ == "__main__":
    unittest.main()
