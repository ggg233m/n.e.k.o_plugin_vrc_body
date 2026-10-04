# -*- coding: utf-8 -*-
r"""``ThreadsTest.test_session_is_persisted_to_world_memory`` 的取证器。

    python research/tools/_flaky_nav_online.py            # 跑一轮，复发就抓
    python research/tools/_flaky_nav_online.py --loops 5  # 连跑 5 轮

为什么需要它
------------
这条用例是**全量跑偶发、单跑必过**的。``.tmp/_run_tests.py`` 是权威 runner，
但它把失败消息截到 160 字符，Windows 控制台又是 GBK——
2026-10 那次复发留在屏幕上的只有 ``+ complete``，
**真实断言一行都没露出来**，等于什么都没说。

本脚本做三件事：

1. **不截断**：完整 traceback 直出。
2. **不改导入顺序**：模块仍然是**按名字排序、在循环里逐个 ``__import__``**。
   这一条是复现的前提——之前两次手工复现都因为在文件顶部提前 import 了
   ``tests.test_nav_online``，导致模块加载顺序变了，于是**一次都没复现**，
   差点得出"测不出来"的错误结论。
3. **附现场**：复发时一并打出前后线程数与目标模块的执行时长，
   用来区分"线程泄漏"和"单次卡顿"。

已知排除项（2026-10 实测，别再重走）
------------------------------------
* **不是时间余量**：隔离跑整条约 0.8s，预算 5.0s，6 倍余量。
* **不是 CPU 争用**：6 进程 / 4 进程并发均 0 失败。
* **不是线程泄漏**：每个模块跑完做线程普查，全程零泄漏。
* **不是 join 超时后提前封会话**：``nav_online.py:942-964`` 的写入顺序是
  缓冲 → 落盘 → 跨会话索引 → 录制落盘 → **最后**才 ``_keyframes += 1``，
  所以 ``status()["keyframes"] >= 3`` 蕴含第 3 帧已完整落盘。
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "test_session_is_persisted_to_world_memory"
TARGET_MODULE = "test_nav_online"
# 与 .tmp/_run_tests.py 保持一致：需要真实采集设备的模块跳过。
SKIP = {"test_local_perception"}

_hits: list[str] = []


class _Spy(unittest.TextTestRunner):
    def run(self, test):                       # noqa: D102 - 覆写只为拿完整 traceback
        res = super().run(test)
        for t, tb in (res.failures + res.errors):
            if TARGET in str(t):
                live = [x.name for x in threading.enumerate() if x is not threading.current_thread()]
                _hits.append(tb)
                print("\n" + "=" * 78, file=sys.stderr)
                print(f"FLAKY REPRODUCED  live_threads={len(live)} {live}", file=sys.stderr)
                print("=" * 78, file=sys.stderr)
                print(tb, file=sys.stderr)          # ← 不截断
                print("=" * 78, file=sys.stderr)
        return res


def run_once() -> int:
    for name in [k for k in sys.modules if k == "unittest"]:
        pass
    unittest.TextTestRunner = _Spy
    mods = [os.path.basename(p)[:-3]
            for p in sorted(glob.glob(str(ROOT / "tests" / "test_*.py")))]
    total = failed = 0
    for name in mods:
        if name in SKIP:
            continue
        t0 = time.perf_counter()
        try:
            # 逐个惰性导入——顺序本身就是复现条件的一部分，不要"优化"成一次性 import
            mod = __import__("tests." + name, fromlist=["*"])
        except Exception:
            continue
        suite = unittest.TestLoader().loadTestsFromModule(mod)
        res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        total += res.testsRun
        failed += len(res.failures) + len(res.errors)
        if name == TARGET_MODULE:
            print(f"[{name}] run={res.testsRun} fail={len(res.failures) + len(res.errors)} "
                  f"({time.perf_counter() - t0:.1f}s)", file=sys.stderr)
    print(f"TOTAL_RUN={total} TOTAL_FAIL={failed}", file=sys.stderr)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=1)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    for i in range(1, max(1, args.loops) + 1):
        if args.loops > 1:
            print(f"\n########## 第 {i}/{args.loops} 轮 ##########", file=sys.stderr)
        run_once()
        if _hits:
            print(f"\n抓到 {len(_hits)} 次。完整 traceback 见上。", file=sys.stderr)
            return 1
    print("\n本轮未复现。这不说明问题不存在——它是低频事件，继续跑或换个时间再来。",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
