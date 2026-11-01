# -*- coding: utf-8 -*-
"""决定性诊断：复现 PolicyHead 的调用链，看每一环返回什么。"""
import sys, time
from pathlib import Path
ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "tools"))

from policy_runner import PolicyRunner

# 1) frame_source：模拟 service._policy_frame_source
class FakeNav:
    def __init__(self):
        self._hist = []
    def policy_frame(self):
        return None   # ← 关键：如果这里是 None，runner 永远 no_frame
    def policy_frame_before(self, s):
        return None
    def policy_waypoint(self):
        return (0.3, 0.5)
    def _near(self): return False

sent = []
rt_calls = []

class FakeRT:
    graph_enabled = False
    def act(self, cur, prev, wp, body, preprocessed=False):
        rt_calls.append((wp, body))
        return dict(block=[[0.0, 0.5]]*12, gated=False, obs_t=time.perf_counter(), latency_ms=100.0)

nav = FakeNav()

def frame_source():
    latest = nav.policy_frame()
    return latest

runner = PolicyRunner(FakeRT(), frame_source, lambda: (0.3, 0.5), lambda: (0.0, 0.0),
                      on_block=lambda b, t, m: sent.append((b, m)),
                      hz=2.0, dry_run=False, stale_s=1.3)
runner.start()
time.sleep(3)
runner.stop()
print("3 秒内 runner 循环数:", len(rt_calls) + (3*2 - len(rt_calls)))
print("rt.act 调用数:", len(rt_calls))
print("on_block 调用数:", len(sent))
if not rt_calls:
    print("=> frame_source 一直返回 None —— runner 全程 no_frame，on_block 永远不会被调")
    print("=> 这就是 action_count=0 的根因")