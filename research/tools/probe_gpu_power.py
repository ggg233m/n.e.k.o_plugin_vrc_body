# -*- coding: utf-8 -*-
r"""用 Level Zero Sysman 读 A770 的真实功耗 / 频率 / 引擎占用。

Windows 的 `GPU Engine` 计数器只给「某个 PID 占某个引擎的百分比」，而且实例数在变，
跨引擎求和也是错的（仓库 §15.13① 已记）。用户给的功耗（100 W / 220 W 墙 ≈ 45%）
比任何计数器都可信。

这里直接用 Level Zero 的 Sysman API（zesInit + zesDeviceEnum + 各 metric），
拿三样：
  1. GPU 功耗（W）与功耗墙；
  2. 核心频率实际值 vs 最大值（降频说明不是功耗受限）；
  3. 各引擎（compute / 3d / copy）的 **活跃时间百分比**（这是硬件自己的计数）。

⚠️ 需要 libze_loader.dll；oneAPI 装了就有。
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from pathlib import Path


def find_loader() -> str | None:
    cands = []
    for env in ("LEVEL_ZERO_V1_SDK_PATH", "ZE_PATH"):
        v = os.environ.get(env)
        if v:
            cands.append(os.path.join(v, "bin", "ze_loader.dll"))
            cands.append(os.path.join(v, "lib", "ze_loader.dll"))
    cands += [
        r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin\ze_loader.dll",
        r"C:\Windows\System32\ze_loader.dll",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def main() -> int:
    dll = find_loader()
    if not dll:
        print("[fail] 找不到 ze_loader.dll")
        return 1
    print("[lib] %s" % dll)
    ze = ctypes.WinDLL(dll)

    # 1) 初始化 + 枚举驱动与设备
    r = ze.zeInit(1 << 4)          # ZE_INIT_FLAG_GPU_ONLY
    print("[ze] zeInit = %d" % r)
    n_drv = ctypes.c_uint32(0)
    ze.zeDriverGet(ctypes.byref(n_drv), None)
    if n_drv.value == 0:
        print("[fail] 没有驱动")
        return 1
    drv = (ctypes.c_void_p * n_drv.value)()
    ze.zeDriverGet(ctypes.byref(n_drv), drv)
    n_dev = ctypes.c_uint32(0)
    ze.zeDeviceGet(drv[0], ctypes.byref(n_dev), None)
    dev = (ctypes.c_void_p * n_dev.value)()
    ze.zeDeviceGet(drv[0], ctypes.byref(n_dev), dev)
    print("[ze] drivers=%d devices=%d" % (n_drv.value, n_dev.value))

    d = dev[0]
    # 2) Sysman 枚举
    n_sub = ctypes.c_uint32(0)
    r = ze.zesDeviceEnumPowerDomains(d, ctypes.byref(n_sub), None)
    print("[sysman] power domains = %d (r=%d)" % (n_sub.value, r))
    pwr = (ctypes.c_void_p * max(n_sub.value, 1))()
    if n_sub.value:
        ze.zesDeviceEnumPowerDomains(d, ctypes.byref(n_sub), pwr)

    n_frq = ctypes.c_uint32(0)
    ze.zesDeviceEnumFrequencyDomains(d, ctypes.byref(n_frq), None)
    print("[sysman] freq domains  = %d" % n_frq.value)
    frq = (ctypes.c_void_p * max(n_frq.value, 1))()
    if n_frq.value:
        ze.zesDeviceEnumFrequencyDomains(d, ctypes.byref(n_frq), frq)

    n_eng = ctypes.c_uint32(0)
    ze.zesDeviceEnumEngineGroups(d, ctypes.byref(n_eng), None)
    print("[sysman] engine groups = %d" % n_eng.value)
    eng = (ctypes.c_void_p * max(n_eng.value, 1))()
    if n_eng.value:
        ze.zesDeviceEnumEngineGroups(d, ctypes.byref(n_eng), eng)

    # 3) 采样
    print()
    print("  %-10s %-10s %-12s %-12s" % ("sample", "power(W)", "core( MHz)", "compute(%)"))
    for i in range(6):
        pw = ctypes.c_double(0)
        ts = ctypes.c_uint64(0)
        if pwr[0]:
            ze.zesPowerGetEnergyCounter(pwr[0], ctypes.byref(pw), ctypes.byref(ts))
        # 频率
        f_cur = ctypes.c_uint32(0)
        if frq[0]:
            ze.zesFrequencyGetState(frq[0], ctypes.byref(f_cur))
        # 引擎活跃时间（累计 µs）
        act = ctypes.c_uint64(0)
        if eng[0]:
            ze.zesEngineGetActivity(eng[0], ctypes.byref(act))
        print("  %-10d %-10.1f %-12d %-12d" % (i, pw.value, f_cur.value, act.value))
        time.sleep(1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
