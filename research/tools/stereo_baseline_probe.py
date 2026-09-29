# -*- coding: utf-8 -*-
"""SBS 双眼窗口的**视差是否真实存在**（只读探针，不接实时链路，不改驱动）。

为什么要它
----------
AnyaDance 驱动已经是完整的 ``IVRDisplayComponent`` HMD，且 ``eye_mode`` 支持
``both``（左右半屏 side-by-side，见 ``src/driver/virtual_device.cpp:539``），
``Prop_UserIpdMeters_Float`` 也声明了 0.063 m。看起来双目测距只差一个配置。

但**声明不等于实现**：该驱动的 ``GetProjectionRaw`` 把 ``EVREye`` 参数注掉了，
两眼返回完全相同的视锥。如果眼位变换也退化成单位矩阵，左右半屏就是像素级相同
的两张图，视差恒为零，双目在这条路上结构性不成立。这一条只能实测，不能从
0.063 这个声明值推断出来。

> **2026-09-29 更正**：AnyaDance 驱动**已加上 `EVREye`**，两眼不再返回相同视锥。
> 上面这条历史记录保留，但**已不成立**，不要再据此认为虚拟驱动不能用。
> 旁证：`research/tools/openvr_mirror_probe.py` 已实测 `tx = ∓0.0315`（63 mm 基线）、
> `|dy| ≈ 0` ⇒ 镜像确为真实校正立体对。

判据（必须能自证，单报一个视差中位数是不够的）
--------------------------------------------
一帧 SBS 画面切成左右两半后做 ORB 匹配，三种结果含义完全不同：

1. **两半像素级相同** ⇒ 眼位退化，视差恒 0。双目不成立，不必再调 FOV。
2. **|dy| 小而 dx 单向非零** ⇒ 真正的校正后立体对。双目成立。
   附带产出：dx 的分布就是深度分布，dx 越大的物体越近。
3. **|dy| 大且散乱** ⇒ 抓到的不是 SBS。最可能是 ``eye_mode`` 仍为 ``left``，
   窗口里只有单眼，切两半得到的是同一张图的不同区域，匹配上的是重复纹理。

所以本探针同时报 dx 和 dy，并显式给出上述三档裁决；只有第 2 档才允许下一步。

用法
----
  .venv/Scripts/python.exe research/tools/stereo_baseline_probe.py --frames 5

注意：``eye_mode`` 在驱动构造时就被缓存（``DisplayComponent`` 的
``m_settings``），改 ``default.vrsettings`` 必须重启 SteamVR 才生效。在切到
``both`` 之前跑本探针，预期就是第 3 档——那是对照组，不是故障。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

DEFAULT_TITLE = "Headset Window"


def find_window(title: str) -> int | None:
    """按标题精确定位窗口；复用 ``backend/vision.py`` 的 FindWindowW 口径。"""
    user32 = ctypes.WinDLL("user32")
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    hwnd = user32.FindWindowW(None, str(title)[:256])
    if not hwnd:
        return None
    if user32.IsIconic(wintypes.HWND(hwnd)):
        # 最小化时 DWM 不再合成该窗口，WGC 拿不到内容。如实报错，不返回空帧。
        raise RuntimeError(f"窗口已最小化，无法采集：{title}")
    return int(hwnd)


def capture(hwnd: int, count: int) -> list[Any]:
    from backend.wgc_capture import WgcSession, wgc_supported

    if not wgc_supported():
        raise RuntimeError("本机不支持 Windows.Graphics.Capture")
    session = WgcSession(hwnd)
    frames: list[Any] = []
    try:
        deadline = count * 200
        waited = 0
        while len(frames) < count and waited < deadline:
            frame = session.read()
            if frame is None:
                import time

                time.sleep(0.02)
                waited += 20
                continue
            frames.append(np.array(frame, copy=True))
    finally:
        session.close()
    if not frames:
        raise RuntimeError("采集不到任何一帧")
    return frames


def split_halves(frame: Any) -> tuple[Any, Any]:
    width = frame.shape[1]
    half = width // 2
    return frame[:, :half], frame[:, half : half + half]


def match_disparity(left: Any, right: Any, *, n_features: int) -> dict[str, Any]:
    """ORB + ratio test，返回水平/垂直视差分布。

    ratio test 是必须的：左右半屏若来自同一张图，重复纹理会造出大量高分但
    错误的匹配，只看匹配数会误判成"立体对成立"。
    """
    import cv2

    gray_l = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=int(n_features))
    kp_l, des_l = orb.detectAndCompute(gray_l, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        return {"matches": 0, "reason": "insufficient_features",
                "keypoints_left": len(kp_l or []), "keypoints_right": len(kp_r or [])}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_l, des_r, k=2)
    dx: list[float] = []
    dy: list[float] = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance > 0.75 * second.distance:
            continue
        p = kp_l[best.queryIdx].pt
        q = kp_r[best.trainIdx].pt
        dx.append(float(p[0] - q[0]))
        dy.append(float(p[1] - q[1]))
    if not dx:
        return {"matches": 0, "reason": "no_confident_match",
                "keypoints_left": len(kp_l), "keypoints_right": len(kp_r)}
    dx_arr = np.asarray(dx)
    dy_arr = np.asarray(dy)
    return {
        "matches": int(dx_arr.size),
        "keypoints_left": len(kp_l),
        "keypoints_right": len(kp_r),
        "dx_median": float(np.median(dx_arr)),
        "dx_p10": float(np.percentile(dx_arr, 10)),
        "dx_p90": float(np.percentile(dx_arr, 90)),
        "abs_dy_median": float(np.median(np.abs(dy_arr))),
        "abs_dy_p90": float(np.percentile(np.abs(dy_arr), 90)),
        "same_sign_fraction": float(np.mean(dx_arr > 0) if np.median(dx_arr) > 0
                                    else np.mean(dx_arr < 0)),
    }


def verdict(identical: bool, stats: dict[str, Any]) -> dict[str, Any]:
    """三档裁决。判不出来时说"判不出来"，不猜一个倾向。"""
    if identical:
        return {"verdict": "degenerate_zero_baseline",
                "stereo_usable": False,
                "detail": "左右半屏像素级相同 ⇒ 眼位退化，视差恒为零，双目不成立"}
    if stats.get("matches", 0) < 30:
        return {"verdict": "inconclusive", "stereo_usable": False,
                "detail": f"可信匹配仅 {stats.get('matches', 0)} 个，不足以裁决"
                          f"（{stats.get('reason', 'too_few_matches')}）"}
    abs_dy = stats["abs_dy_median"]
    dx_median = abs(stats["dx_median"])
    same_sign = stats["same_sign_fraction"]
    # 校正后的立体对：同名点只在水平方向错开。dy 中位数超过 2 px 就说明这两半
    # 不是同一时刻同一朝向的两只眼。
    if abs_dy <= 2.0 and dx_median >= 1.0 and same_sign >= 0.8:
        return {"verdict": "real_stereo", "stereo_usable": True,
                "detail": f"|dy| 中位数 {abs_dy:.2f} px（小）、dx 中位数 "
                          f"{stats['dx_median']:.2f} px 且 {same_sign:.0%} 同号 ⇒ 真实立体对"}
    if abs_dy > 2.0:
        return {"verdict": "not_side_by_side", "stereo_usable": False,
                "detail": f"|dy| 中位数 {abs_dy:.2f} px 过大 ⇒ 两半不是同步双眼；"
                          f"最可能 eye_mode 仍为 left（窗口只有单眼）"}
    return {"verdict": "inconclusive", "stereo_usable": False,
            "detail": f"|dy|={abs_dy:.2f} px 小但 dx 中位数仅 {stats['dx_median']:.2f} px"
                      f"、同号率 {same_sign:.0%} ⇒ 视差过弱，无法确认基线"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--features", type=int, default=1200)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    hwnd = find_window(args.title)
    if hwnd is None:
        print(f"找不到窗口：{args.title!r}")
        return 2
    frames = capture(hwnd, max(1, int(args.frames)))
    height, width = frames[0].shape[:2]
    print(f"窗口 {args.title!r} hwnd={hwnd} 采到 {len(frames)} 帧，{width}x{height}")

    per_frame: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        left, right = split_halves(frame)
        identical = bool(np.array_equal(left, right))
        stats = match_disparity(left, right, n_features=args.features)
        decision = verdict(identical, stats)
        per_frame.append({"frame": index, "identical_halves": identical,
                          **stats, **decision})
        print(f"[{index}] identical={identical} matches={stats.get('matches', 0)} "
              f"dx_med={stats.get('dx_median')} |dy|_med={stats.get('abs_dy_median')} "
              f"=> {decision['verdict']}")
        print(f"     {decision['detail']}")

    usable = sum(1 for item in per_frame if item["stereo_usable"])
    print(f"\n裁决：{usable}/{len(per_frame)} 帧支持双目。", end=" ")
    if usable == len(per_frame):
        print("双目成立，可以进入 FOV/分辨率重标定。")
    elif usable == 0:
        print("双目不成立；上面的 detail 说明了卡在哪一档。")
    else:
        print("结果不一致，不能据此下结论——需要更多帧或更稳的画面。")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"hwnd": hwnd, "width": width, "height": height,
                                   "frames": per_frame}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"已写入 {out}")
    return 0 if usable == len(per_frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
