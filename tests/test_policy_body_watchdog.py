# -*- coding: utf-8 -*-
"""body 看门狗：闩锁 + 活跃任务 → 自动 reset/enable（剖析 §15.21）。

外部急停 / 宿主驱动 / 调度器故障都会把 body 打进 stopped_latched —— 摇杆静默
而转向帧照发，表现为"点了没反应"。有活跃导航任务时看门狗必须自动恢复；
无任务时保持闩锁（尊重急停语义）。
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import _bootstrap  # noqa: F401,E402
import neko_anyadance_body.backend.service as mod  # noqa: E402


class BodyWatchdogTests(unittest.TestCase):
    def _svc(self, body_state, nav_mode, drive_block=None):
        svc = mod.BackendService.__new__(mod.BackendService)
        svc.scheduler = SimpleNamespace(
            snapshot=lambda: {"state": body_state})
        calls = []

        def fake_submit(kind, params=None, **kw):
            calls.append(kind)
            state = "idle" if kind == "reset" else "idle"
            return {"accepted": True, "state": state}
        svc.submit = fake_submit
        svc.logger = None
        svc.navmesh = SimpleNamespace(status=lambda: {"mode": nav_mode})
        svc._drive_block_reason = lambda: drive_block
        return svc, calls

    def _loop_once(self, svc):
        svc._policy_body_watchdog_stop = SimpleNamespace(wait=lambda t: False)
        # patch Event.wait to run exactly one iteration
        iterations = {"n": 0}
        def wait(_t):
            iterations["n"] += 1
            return iterations["n"] > 1
        svc._policy_body_watchdog_stop = SimpleNamespace(wait=wait)
        svc._policy_body_watchdog_loop()

    def test_recovers_latched_with_active_task(self):
        svc, calls = self._svc("stopped_latched", "goto", drive_block=None)
        self._loop_once(svc)
        self.assertIn("reset", calls)

    def test_ignores_latched_without_task(self):
        svc, calls = self._svc("stopped_latched", "idle", drive_block=None)
        self._loop_once(svc)
        self.assertNotIn("reset", calls)

    def test_ignores_latched_with_drive_block(self):
        svc, calls = self._svc("stopped_latched", "goto",
                               drive_block="autonomy_not_armed")
        self._loop_once(svc)
        self.assertNotIn("reset", calls)

    def test_enables_disabled_with_active_task(self):
        svc, calls = self._svc("disabled", "goto", drive_block=None)
        self._loop_once(svc)
        self.assertIn("enable", calls)

    def test_healthy_state_is_noop(self):
        svc, calls = self._svc("idle", "goto", drive_block=None)
        self._loop_once(svc)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
