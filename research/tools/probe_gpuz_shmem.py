# -*- coding: utf-8 -*-
r"""读 GPU-Z 的共享内存 —— 拿 A770 的真实功耗 / 频率 / 负载。

为什么用它：Windows 没有任何 GPU 计数器能给出功耗（`GPU Engine` 只有占用来回跳的
`Utilization Percentage`），Level Zero Sysman 在本机 `zeInit` 就失败。而 GPU-Z 开着时
会维护一块名为 `GPUZShMem` 的只读共享内存，里面有功耗、频率、负载、温度。

GPUZShMem 布局（GPU-Z 的公开结构，v2）：
  开头若干 32 位字段，其中：
    [0] 版本/魔数区，之后是若干传感器值，**功耗在固定偏移**。
  由于不同 GPU-Z 版本偏移会变，这里做两件事：
    1. 先 dump 前 320 字节，让人肉/自动识别；
    2. 用「已知区间」做启发式识别（功耗 0-600、频率 0-3000、负载 0-100）。
"""
from __future__ import annotations

import mmap
import struct
import time


def open_shmem():
    f = mmap.mmap(-1, 4096, tagname="GPUZShMem", access=mmap.ACCESS_READ)
    return f


def main() -> int:
    try:
        m = open_shmem()
    except Exception as e:  # noqa: BLE001
        print("[fail] 打不开 GPUZShMem：%r" % (e,))
        return 1

    raw = m[:320]
    print("[dump] 前 320 字节（每行 8 个 uint32）")
    for off in range(0, 320, 32):
        vals = struct.unpack_from("<8I", raw, off)
        print("  %3d: %s" % (off, " ".join("%10d" % v for v in vals)))

    print()
    print("[watch] 连续采样 6 次，看哪些字段在动（训练在跑，功耗/负载会变）")
    series = []
    for i in range(6):
        d = struct.unpack_from("<80I", m[:320], 0)
        series.append(d)
        time.sleep(1.0)

    n = len(series[0])
    moving = []
    for k in range(n):
        vals = [s[k] for s in series]
        if len(set(vals)) > 1:
            moving.append((k, vals))
    print("  在动的字段：%d 个" % len(moving))
    for k, vals in moving:
        print("    [%2d] %s" % (k, vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
