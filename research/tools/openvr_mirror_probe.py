# -*- coding: utf-8 -*-
"""SteamVR 合成器镜像纹理取帧 + 双眼视差实测（只读探针，不接实时链路）。

为什么要它
----------
``Headset Window`` 改 ``eye_mode=both`` 实测无效：桌面扩展模式下那个窗口显示的
是合成器自己的镜像输出，``GetEyeOutputViewport`` 决定不了它画什么。唯一能拿到
**真正两只眼**画面的官方途径是 ``IVRCompositor::GetMirrorTextureD3D11``。

已经实测确认的前提（2026-09-25）：
  - ``VRApplication_Background`` 即可拿到 compositor，不抢 VRChat 的 scene 位；
  - 眼间变换 ``tx = ∓0.0315``，即 63 mm 基线，由驱动的
    ``Prop_UserIpdMeters_Float`` 推导而来。

本探针验证剩下的那一关：纹理到底**取不取得到**，取到的两只眼是否构成真实立体对。

pyopenvr 1.26 封装的缺陷
------------------------
函数表声明 ``getMirrorTextureD3D11(EVREye, void*, void**)``，C API 要的是设备
指针本身；封装却传 ``byref(device)``（指针的地址）。``releaseMirrorTextureD3D11``
同理。所以这里**绕过封装直接调函数表**。

两只眼可能返回同一张纹理
----------------------
文档说 "for each eye"，但不保证两个 SRV 指向不同资源。若两眼纹理规格相同、像素
也逐字节相同，就是同一张图（可能是左右并排的整张合成图）。探针会显式报出来，
不会把同一张图当成立体对。

用法
----
  .venv/Scripts/python.exe -W ignore research/tools/openvr_mirror_probe.py --frames 3 --save .tmp/mirror
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

# 取帧实现已移入 backend（在线建图要用，backend 不能依赖 tools/）；这里只留探针。
from backend.openvr_mirror import MirrorEye, create_device, read_stereo  # noqa: F401
from backend.wgc_capture import _release
from research.tools.stereo_baseline_probe import match_disparity, verdict



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--features", type=int, default=1200)
    parser.add_argument("--save", default="", help="保存左右眼 PNG 的目录")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    import openvr

    system = openvr.init(openvr.VRApplication_Background)
    device = context = None
    eyes: list[MirrorEye] = []
    try:
        adapter = system.getDXGIOutputInfo()
        print(f"SteamVR 适配器索引 = {adapter}")
        if adapter not in (0, -1):
            # 默认设备建在适配器 0 上；共享纹理跨适配器打不开。先报出来，不静默继续。
            print("警告：SteamVR 不在默认适配器上，镜像纹理可能打不开")
        baseline = [system.getEyeToHeadTransform(e)[0][3] for e in (openvr.Eye_Left, openvr.Eye_Right)]
        print(f"眼间平移 L={baseline[0]:+.4f} R={baseline[1]:+.4f}  基线={baseline[1] - baseline[0]:.4f} m")

        device, context = create_device()
        compositor = openvr.VRCompositor()
        for eye in (openvr.Eye_Left, openvr.Eye_Right):
            eyes.append(MirrorEye(compositor, device, context, eye))
        for mirror, label in zip(eyes, "LR"):
            print(f"[{label}] texture=0x{mirror.texture_ptr:x} {mirror.desc['Width']}x{mirror.desc['Height']} "
                  f"format={mirror.desc['Format']} mips={mirror.desc['MipLevels']} array={mirror.desc['ArraySize']}")
        same_texture = eyes[0].texture_ptr == eyes[1].texture_ptr
        print(f"两眼同一纹理指针：{same_texture}")

        per_frame: list[dict[str, Any]] = []
        save_dir = Path(args.save) if args.save else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        for index in range(max(1, int(args.frames))):
            started = time.perf_counter()
            left, right = read_stereo(eyes[0], eyes[1])
            read_ms = (time.perf_counter() - started) * 1000.0
            if left is None or right is None:
                print(f"[{index}] Map 失败，跳过")
                continue
            identical = bool(left.shape == right.shape and np.array_equal(left, right))
            gray_std = float(left.mean(axis=2).std())
            stats = match_disparity(left, right, n_features=args.features) if left.shape == right.shape \
                else {"matches": 0, "reason": "eye_shape_mismatch"}
            decision = verdict(identical, stats)
            per_frame.append({"frame": index, "read_ms": round(read_ms, 1), "identical": identical,
                              "gray_std": round(gray_std, 2), **stats, **decision})
            print(f"[{index}] read={read_ms:.1f}ms std={gray_std:.1f} identical={identical} "
                  f"matches={stats.get('matches', 0)} dx_med={stats.get('dx_median')} "
                  f"|dy|_med={stats.get('abs_dy_median')} => {decision['verdict']}")
            print(f"     {decision['detail']}")
            if save_dir:
                import cv2
                cv2.imwrite(str(save_dir / f"L_{index}.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_dir / f"R_{index}.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
            time.sleep(max(0.0, float(args.interval)))

        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"adapter": adapter, "baseline_m": baseline[1] - baseline[0],
                                       "same_texture": same_texture,
                                       "eyes": [m.desc for m in eyes], "frames": per_frame},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
        usable = sum(1 for item in per_frame if item["stereo_usable"])
        print(f"\n裁决：{usable}/{len(per_frame)} 帧支持双目。")
        return 0 if per_frame and usable == len(per_frame) else 1
    finally:
        for mirror in eyes:
            mirror.close()
        _release(context)
        _release(device)
        openvr.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
