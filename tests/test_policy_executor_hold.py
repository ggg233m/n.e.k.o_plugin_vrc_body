# -*- coding: utf-8 -*-
"""PolicyExecutor 的转向代管 hold 门：hold_provider 置位时按住轴，块不过期。"""
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

from policy_executor import PolicyExecutor  # noqa: E402


class HoldGateTests(unittest.TestCase):
    def _exec(self, hold=False, dry_run=True):
        sent = []
        ex = PolicyExecutor(lambda x, y, d: sent.append((x, y)),
                            stale_s=1.3, dry_run=dry_run,
                            hold_provider=lambda: hold)
        ex.stop()          # 不跑线程，手动步进 _loop 循环体
        return ex, sent

    def _step_once(self, ex):
        # 直接调一次循环体等价逻辑：不开线程（daemon 线程手动步进难以只走一轮）
        import threading
        ex._stop.clear()
        th = threading.Thread(target=ex._loop, daemon=True)
        th.start()
        time.sleep(0.05)
        ex._stop.set()
        th.join(timeout=1.0)

    def test_hold_blocks_axes_but_keeps_block(self):
        ex, sent = self._exec(hold=True)
        block = [[0.0, 0.5]] * 12
        ex.submit(block, time.perf_counter(), {})
        self._step_once(ex)
        self.assertEqual(sent, [])
        self.assertGreaterEqual(ex.n_holds, 1)

    def test_hold_release_resumes(self):
        flag = {"hold": True}
        ex, sent = self._exec(hold=False)
        ex._hold_provider = lambda: flag["hold"]
        block = [[0.0, 0.5]] * 12
        ex.submit(block, time.perf_counter(), {})
        self._step_once(ex)
        n = ex.n_holds
        self.assertGreaterEqual(n, 1)
        flag["hold"] = False                # 转到位
        self._step_once(ex)
        self.assertGreater(ex.n_sends, 0)
        # dry_run 不真发轴，但 last_send 记录了「本该发的」摇杆量（vz=0.5 → y=0.2125，
        # last_send 按 3 位小数存 0.213）
        self.assertIsNotNone(ex.last_send)
        self.assertEqual(ex.last_send[0], 0.0)
        self.assertEqual(ex.last_send[1], 0.213)
        self.assertEqual(sent, [])

    def test_hold_provider_error_is_no_hold(self):
        def boom():
            raise RuntimeError("navmesh down")
        ex, sent = self._exec(hold=False)
        ex._hold_provider = boom
        ex.submit([[0.0, 0.5]] * 12, time.perf_counter(), {})
        self._step_once(ex)
        self.assertEqual(ex.n_holds, 0)
        self.assertGreaterEqual(ex.n_sends, 1)

    def test_dry_run_hold_records_no_send(self):
        ex, sent = self._exec(hold=True, dry_run=True)
        ex.submit([[0.2, 0.8]] * 12, time.perf_counter(), {})
        self._step_once(ex)
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
