# -*- coding: utf-8 -*-
"""Inductor 并行编译 A/B —— 验证 Windows 上能不能靠 env 绕开 pass_fds。

背景（全部在已安装的 torch 源码里核过）：
  config.py:1467            win32 -> compile_threads = 1（只是默认值）
  config.py:1464            TORCHINDUCTOR_COMPILE_THREADS 的 env 检查在 win32 判断**之前**，能覆盖
  subproc_pool.py:221       Popen(pass_fds=...) —— Windows 不支持，所以覆盖后会崩
  async_compile.py:329      只有 worker_start_method=="subprocess" 才走 SubprocPool
  async_compile.py:334-344  其它值走 TrackedProcessPoolExecutor(mp_context=...) —— 不碰 pass_fds
  config.py:1176            允许 "subprocess" / "fork" / "spawn"，Windows 用 spawn

=> 假设：TORCHINDUCTOR_WORKER_START=spawn + COMPILE_THREADS=N 可在 Windows 上并行编译。

测法：12 个不同形状，每个形状触发一组新 Triton kernel；每组独立 cache 目录，
保证是真编译而不是读缓存。

用法::

    $env:TORCHINDUCTOR_COMPILE_THREADS="8"; $env:TORCHINDUCTOR_WORKER_START="spawn"
    python research/tools/probe_compile_threads.py --tag spawn8
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shapes", type=int, default=12)
    ap.add_argument("--static", action="store_true",
                    help="dynamic=False：每个形状各自特化，强制编译 N 组 kernel")
    a = ap.parse_args()

    import torch
    import torch._inductor.config as ic

    print("[cfg] tag=%s compile_threads=%r worker_start=%r pid=%d"
          % (a.tag, ic.compile_threads, ic.worker_start_method, os.getpid()), flush=True)
    print("[cfg] cache=%s" % os.environ.get("TORCHINDUCTOR_CACHE_DIR"), flush=True)

    dev = "xpu" if torch.xpu.is_available() else "cpu"

    class Blend(torch.nn.Module):
        def forward(self, x, y):
            for _ in range(5):
                x = torch.sin(x) * 0.5 + torch.cos(y) * 0.25 + x
            return x

    m = torch.compile(Blend(), dynamic=(not a.static))
    base = [(1024, 1024), (2048, 512), (512, 2048), (4096, 128), (128, 4096), (1000, 1000),
            (777, 333), (1500, 1500), (900, 1100), (1234, 567), (640, 960), (320, 1920)]
    shapes = base[: a.shapes]

    t0 = time.perf_counter()
    ok = 0
    for i, (r, c) in enumerate(shapes):
        x = torch.randn(r, c, device=dev)
        y = torch.randn(r, c, device=dev)
        ts = time.perf_counter()
        with torch.no_grad():
            m(x, y)
        if dev == "xpu":
            torch.xpu.synchronize()
        ok += 1
        print("  shape %2d %-12s %6.1f s (累计 %.0f s)"
              % (i, str((r, c)), time.perf_counter() - ts, time.perf_counter() - t0), flush=True)

    total = time.perf_counter() - t0
    print("[done] tag=%s shapes=%d total=%.1f s (%.1f s/shape)"
          % (a.tag, ok, total, total / max(ok, 1)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
