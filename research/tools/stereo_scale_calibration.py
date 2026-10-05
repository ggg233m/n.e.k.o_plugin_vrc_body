# -*- coding: utf-8 -*-
"""镜像双目的**世界尺度**标定（离线工具，不接实时链路）。

为什么要它
----------
``research/tools/openvr_mirror_probe.py`` 已实测两眼构成真实立体对（|dy|≈0，dx 随深度
变化），基线取自 ``getEyeToHeadTransform`` = 0.063 m。但这 0.063 是**追踪空间**
的眼距。VRChat 按 avatar 缩放世界，世界里的有效眼距是 ``0.063 × s``，``s`` 未知。
直接拿 0.063 算深度，得到的是"追踪米"，不是世界米——那就又是拿声明值冒充测量值。

标定方法
--------
同一进程里同时做两件事，共用 ``time.perf_counter`` 时钟（不经后端，不存在跨进程
时间对齐问题）：

1. 以 ~4 Hz 从合成器镜像取左右眼，ORB + 行约束做立体匹配，再做亚像素视差细化，
   按 B=0.063 三角化出每帧的 3D 点（单位：追踪米）。
2. 直接监听 VRChat 的 OSC 回传（``127.0.0.1:9001``），记录 VelocityX/Y/Z。

分析时，对时间间隔 0.5–1.2 s 的帧对 (i, j)（窗口过宽则行走中视角变化太大、匹配不上）：
  - 帧 i 的 3D 点 + 帧 j 左眼 2D 点做 PnP，得相机位移 ``d_stereo``（追踪米）；
  - 对 OSC 速度按**零阶保持**积分，得世界位移 ``d_osc``（世界米）。
    ZOH 是协议语义，不是猜测：VelocityX/Z 是变化驱动参数，静默 ⟺ 速度没变
    （见 ``backend/nav_online.py`` 的航位推算）。
  - ``s = d_osc / d_stereo``。

``d_osc`` 是**路径长度**，``d_stereo`` 是**弦长**，只有直线运动时两者相等，所以
PnP 解出的旋转超过 ``--max-rot-deg`` 的帧对直接丢弃，不靠人工假设"走得够直"。

判据：每对给出一个 s，报中位数与 MAD。若 s 随场景深度系统性变化，说明焦距/基线
模型本身有错，该结果不可用——工具会把按深度分档的 s 一并报出来。

用法
----
  # 1) 录制：后端必须停着（要占用 9001）。录制期间在 VRChat 里直线走动。
  .venv/Scripts/python.exe -W ignore research/tools/stereo_scale_calibration.py record \
      --seconds 60 --out .tmp/scale_cal/run1.npz
  # 2) 分析（可改参数反复跑，不用重录）
  .venv/Scripts/python.exe -W ignore research/tools/stereo_scale_calibration.py analyze \
      .tmp/scale_cal/run1.npz
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 仓库根目录名 ``n.e.k.o_plugin_vrc_body`` 不是合法标识符，所以根级模块（osc.py）
# 只能作为 ``neko_anyadance_body`` 的子模块导入——它内部写的是 ``from .config``。
# 这里复用 ``tests/_bootstrap.py`` 的同一招，且刻意不 import 根 ``__init__``
# （那会拉进整个 N.E.K.O SDK 插件入口，本工具不需要）。
if "neko_anyadance_body" not in sys.modules:
    _package = types.ModuleType("neko_anyadance_body")
    _package.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
    sys.modules["neko_anyadance_body"] = _package

import numpy as np

_VELOCITY_NAMES = ("VelocityX", "VelocityY", "VelocityZ")
_OSC_PREFIX = "/avatar/parameters/"


# ---------------------------------------------------------------------------
# 录制
# ---------------------------------------------------------------------------

class OscRecorder:
    """在独立线程里收 OSC：速度三轴进 ``samples``，**其余 avatar 参数**进 ``params``。

    🔑 为什么要把所有参数一起收（2026-10-05）：VRChat 的标准参数里有 ``EyeHeightAsMeters``
    （模型眼高，**米**）与 ``ScaleFactor`` / ``ScaleFactorInverse``——它们给出「世界米 ↔ 追踪米」
    这个比值的外部真值，而那双目链与 OSC 链**各自都自证不了自己的绝对米制**
    （见 `Docs/漂移形态诊断（2026-10-05）.md` §十）。原先只收速度三轴，等于每次录制都把
    这份真值丢掉了。
    ``params`` 只记**变化**（同名同值不重复记），否则 Voice/Viseme 这类高频参数会把文件撑爆。
    """

    def __init__(self, host: str, port: int) -> None:
        from neko_anyadance_body.osc import decode_osc_packet

        self._decode = decode_osc_packet
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind((host, port))
        except OSError as exc:
            self._sock.close()
            raise RuntimeError(
                f"无法监听 {host}:{port}（{exc}）。后端运行时会占用这个端口，录制前请先停掉后端。"
            ) from exc
        self._sock.settimeout(0.2)
        self.samples: list[tuple[float, int, float]] = []
        self.params: list[tuple[float, str, float]] = []   # 非速度参数（含 EyeHeightAsMeters）
        self._param_last: dict[str, float] = {}
        self.packets = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            received = time.perf_counter()
            self.packets += 1
            try:
                messages = self._decode(packet)
            except Exception:
                continue
            for address, arguments in messages:
                if not address.startswith(_OSC_PREFIX) or not arguments:
                    continue
                name = address[len(_OSC_PREFIX):]
                value = arguments[0]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if not math.isfinite(float(value)):
                    continue
                if name in _VELOCITY_NAMES:
                    self.samples.append((received, _VELOCITY_NAMES.index(name), float(value)))
                elif self._param_last.get(name) != float(value):
                    self._param_last[name] = float(value)
                    self.params.append((received, name, float(value)))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()


def refine_disparity(left: Any, right: Any, xl: float, y: float, xr: float,
                     half: int = 5, search: int = 3) -> float | None:
    """在右眼同一行上做 SSD 模板搜索 + 抛物线插值，给出亚像素视差。

    ORB 关键点只有整像素（高金字塔层还更粗），7–15 px 的视差上 0.5 px 误差就是
    3–7% 的深度误差，所以细化是必须的，不是锦上添花。
    """
    import cv2

    xi, yi, xri = int(round(xl)), int(round(y)), int(round(xr))
    h, w = left.shape
    if yi - half < 0 or yi + half >= h or xi - half < 0 or xi + half >= w:
        return None
    x0, x1 = xri - search - half, xri + search + half + 1
    if x0 < 0 or x1 > w:
        return None
    patch = left[yi - half:yi + half + 1, xi - half:xi + half + 1]
    strip = right[yi - half:yi + half + 1, x0:x1]
    score = cv2.matchTemplate(strip, patch, cv2.TM_SQDIFF_NORMED)[0]
    k = int(np.argmin(score))
    if k == 0 or k == len(score) - 1:
        # 极小值落在搜索窗边缘：真正的匹配可能在窗外，不能插值。
        return None
    a, b, c = float(score[k - 1]), float(score[k]), float(score[k + 1])
    denom = a - 2.0 * b + c
    offset = 0.5 * (a - c) / denom if denom > 1e-12 else 0.0
    matched_x = x0 + half + k + offset
    return float(xi - matched_x)


def stereo_frame(left_rgb: Any, right_rgb: Any, orb: Any, *, max_dy: float,
                 min_disp: float) -> dict[str, Any]:
    import cv2

    gl = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    kl, dl = orb.detectAndCompute(gl, None)
    kr, dr = orb.detectAndCompute(gr, None)
    n = len(kl)
    xy = np.array([k.pt for k in kl], dtype=np.float32).reshape(-1, 2)
    disparity = np.full(n, np.nan, dtype=np.float32)
    if dl is None or dr is None or n == 0 or len(kr) == 0:
        return {"xy": xy, "desc": np.zeros((n, 32), np.uint8) if dl is None else dl,
                "disp": disparity}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dl, dr, k=2)
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance > 0.75 * second.distance:
            continue
        pl, pr = kl[best.queryIdx].pt, kr[best.trainIdx].pt
        # 已校正的平行立体：同名点必须在同一行、且左眼 x 更大。
        if abs(pl[1] - pr[1]) > max_dy or pl[0] - pr[0] < min_disp - 1.0:
            continue
        refined = refine_disparity(gl, gr, pl[0], pl[1], pr[0])
        if refined is not None and refined >= min_disp:
            disparity[best.queryIdx] = refined
    return {"xy": xy, "desc": dl, "disp": disparity}


def record(args: argparse.Namespace) -> int:
    import cv2
    import openvr

    from research.tools.openvr_mirror_probe import MirrorEye, create_device, read_stereo
    from backend.wgc_capture import _release

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    osc = OscRecorder(args.osc_host, args.osc_port)
    system = openvr.init(openvr.VRApplication_Background)
    device = context = None
    eyes: list[Any] = []
    frames: list[dict[str, Any]] = []
    try:
        eye_x = [system.getEyeToHeadTransform(e)[0][3] for e in (openvr.Eye_Left, openvr.Eye_Right)]
        baseline = float(eye_x[1] - eye_x[0])
        device, context = create_device()
        compositor = openvr.VRCompositor()
        for eye in (openvr.Eye_Left, openvr.Eye_Right):
            eyes.append(MirrorEye(compositor, device, context, eye))
        width, height = eyes[0].desc["Width"], eyes[0].desc["Height"]
        left_b, right_b, top_b, bottom_b = system.getProjectionRaw(openvr.Eye_Left)
        fx = width / (right_b - left_b)
        fy = height / (bottom_b - top_b)
        cx = width * (-left_b) / (right_b - left_b)
        cy = height * (-top_b) / (bottom_b - top_b)
        print(f"基线 {baseline:.4f} m，每眼 {width}x{height}，fx={fx:.1f} fy={fy:.1f} "
              f"cx={cx:.1f} cy={cy:.1f}")
        print(f"录制 {args.seconds:.0f} s。请在 VRChat 里：站定 2 s → 直线前进 3–5 s → 停 → "
              f"转身 → 直线走回 → 重复。面向有纹理的场景，别贴着空白墙。")

        orb = cv2.ORB_create(nfeatures=int(args.features))
        started = time.perf_counter()
        last_report = started
        period = 1.0 / float(args.fps)
        next_due = started
        while time.perf_counter() - started < float(args.seconds):
            now = time.perf_counter()
            if now < next_due:
                time.sleep(min(0.01, next_due - now))
                continue
            next_due += period
            t0 = time.perf_counter()
            left, right = read_stereo(eyes[0], eyes[1])
            t1 = time.perf_counter()
            if left is None or right is None:
                continue
            frame = stereo_frame(left, right, orb, max_dy=args.max_dy, min_disp=args.min_disp)
            # 取帧时刻记为两次回读的中点；立体计算不影响画面时刻。
            frame["t"] = 0.5 * (t0 + t1)
            frames.append(frame)
            if len(frames) == 1 and args.preview:
                cv2.imwrite(str(out.with_suffix(".first_left.png")),
                            cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            if t1 - last_report >= 2.0:
                last_report = t1
                valid = int(np.isfinite(frame["disp"]).sum())
                print(f"  t={t1 - started:5.1f}s 帧={len(frames):4d} 本帧立体点={valid:4d} "
                      f"OSC包={osc.packets:5d} 速度样本={len(osc.samples):5d}")
        elapsed = time.perf_counter() - started
    finally:
        osc.close()
        for mirror in eyes:
            mirror.close()
        _release(context)
        _release(device)
        openvr.shutdown()

    if not frames:
        print("没有录到任何帧")
        return 2
    counts = np.array([len(f["xy"]) for f in frames], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez_compressed(
        out,
        t=np.array([f["t"] for f in frames], dtype=np.float64),
        offsets=offsets,
        xy=np.concatenate([f["xy"] for f in frames]).astype(np.float32),
        desc=np.concatenate([f["desc"] for f in frames]).astype(np.uint8),
        disp=np.concatenate([f["disp"] for f in frames]).astype(np.float32),
        osc=np.array(osc.samples, dtype=np.float64).reshape(-1, 3),
        meta=json.dumps({"baseline_m": baseline, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                         "width": width, "height": height, "elapsed_s": elapsed,
                         "osc_packets": osc.packets, "fps_target": args.fps}),
    )
    print(f"\n已写入 {out}：{len(frames)} 帧（实际 {len(frames) / elapsed:.2f} Hz），"
          f"OSC 速度样本 {len(osc.samples)} 条")
    if not osc.samples:
        print("警告：一条速度样本都没收到——没法标定。检查 VRChat OSC 是否开启、是否在走动。")
        return 1
    return 0


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------

def osc_displacement(osc: Any, t0: float, t1: float) -> float | None:
    """零阶保持积分 [t0, t1] 内的三维速度模长，返回路径长度（世界米）。

    t0 之前任一轴从未收到过值时返回 ``None``：起点速度未知，不能当成 0。
    """
    if osc.size == 0:
        return None
    current = [None, None, None]
    events = osc[osc[:, 0] <= t1]
    before = events[events[:, 0] <= t0]
    for _, axis, value in before:
        current[int(axis)] = value
    # VelocityY 只在跳跃/落地时变化，从未收到就按地面行走的 0 处理；
    # 水平两轴缺失则起点速度未知，拒绝积分。
    if current[0] is None or current[2] is None:
        return None
    if current[1] is None:
        current[1] = 0.0
    inside = events[events[:, 0] > t0]
    distance = 0.0
    cursor = t0
    for stamp, axis, value in inside:
        distance += math.sqrt(sum(v * v for v in current)) * (stamp - cursor)
        current[int(axis)] = value
        cursor = stamp
    distance += math.sqrt(sum(v * v for v in current)) * (t1 - cursor)
    return distance


def load_arrays(data: Any) -> dict[str, Any]:
    """把 npz 里的数组一次性解压进内存，之后按帧切片。

    ``np.load`` 返回的 ``NpzFile`` 每次下标访问都会**重新解压整个数组**，而
    ``frame_slice`` 每个帧对要访问 6 次。实测单帧对 523 ms，其中大部分花在反复
    解压 718000×32 的描述子上；预加载后降到约 137 ms。

    描述子另外转成连续 uint8：``knnMatch`` 在切片视图上 38.2 ms，连续内存上
    8.1 ms。返回的 dict 与 ``NpzFile`` 同样支持下标访问，可直接喂给
    ``frame_slice`` / ``stereo_displacement``。
    """
    return {
        "offsets": np.asarray(data["offsets"]),
        "xy": np.ascontiguousarray(data["xy"], dtype=np.float32),
        "desc": np.ascontiguousarray(data["desc"], dtype=np.uint8),
        "disp": np.ascontiguousarray(data["disp"], dtype=np.float32),
    }


def frame_slice(data: Any, index: int) -> tuple[Any, Any, Any]:
    a, b = int(data["offsets"][index]), int(data["offsets"][index + 1])
    return data["xy"][a:b], data["desc"][a:b], data["disp"][a:b]


def stereo_displacement(data: Any, meta: dict[str, Any], i: int, j: int, *,
                        min_inliers: int) -> dict[str, Any] | None:
    """帧 i 的立体 3D 点 + 帧 j 左眼 2D 点做 PnP，返回相机位移（追踪米）。"""
    import cv2

    xy_i, desc_i, disp_i = frame_slice(data, i)
    xy_j, desc_j, _ = frame_slice(data, j)
    has_depth = np.isfinite(disp_i)
    if has_depth.sum() < min_inliers or len(desc_j) < min_inliers:
        return None
    idx_i = np.nonzero(has_depth)[0]
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_i[idx_i], desc_j, k=2)
    obj, img, depth = [], [], []
    fx, fy, cx, cy, baseline = (meta[k] for k in ("fx", "fy", "cx", "cy", "baseline_m"))
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance > 0.75 * second.distance:
            continue
        k = idx_i[best.queryIdx]
        x, y = xy_i[k]
        z = fx * baseline / float(disp_i[k])
        obj.append(((x - cx) * z / fx, (y - cy) * z / fy, z))
        img.append(xy_j[best.trainIdx])
        depth.append(z)
    if len(obj) < min_inliers:
        return None
    obj_arr = np.asarray(obj, dtype=np.float64)
    img_arr = np.asarray(img, dtype=np.float64)
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, camera, None, iterationsCount=300, reprojectionError=2.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    inl = inliers.ravel()
    # EPnP 只作初值，再用全部内点做迭代精化。
    ok, rvec, tvec = cv2.solvePnP(obj_arr[inl], img_arr[inl], camera, None, rvec, tvec,
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    # 相机中心位移 = -Rᵀt；模长与 |t| 相同，这里显式算出以便日后取方向。
    center = -rotation.T @ tvec.ravel()
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0))))
    projected, _ = cv2.projectPoints(obj_arr[inl], rvec, tvec, camera, None)
    residual = np.linalg.norm(projected.reshape(-1, 2) - img_arr[inl], axis=1)
    return {"d_stereo": float(np.linalg.norm(center)), "rot_deg": angle,
            "inliers": int(len(inl)), "matches": int(len(obj)),
            "median_depth": float(np.median(np.asarray(depth)[inl])),
            "reproj_px": float(np.median(residual))}


def analyze(args: argparse.Namespace) -> int:
    npz = np.load(args.recording, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    times = np.asarray(npz["t"])
    osc = np.asarray(npz["osc"])
    data = load_arrays(npz)
    print(f"录制：{len(times)} 帧 / {meta['elapsed_s']:.1f} s，OSC 速度样本 {len(osc)} 条，"
          f"基线 {meta['baseline_m']:.4f} m，fx={meta['fx']:.1f}", flush=True)
    if osc.size:
        osc = osc[np.argsort(osc[:, 0], kind="stable")]

    results: list[dict[str, Any]] = []
    rejected = {"osc_unknown": 0, "too_little_motion": 0, "pnp_failed": 0,
                "rotation": 0, "stereo_too_small": 0}
    # 先用便宜的 OSC 积分筛掉静止帧对，只对真在动的帧对做匹配 + PnP。
    candidates: list[tuple[int, int, float, float]] = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            dt = times[j] - times[i]
            if dt < args.min_dt:
                continue
            if dt > args.max_dt:
                break
            d_osc = osc_displacement(osc, float(times[i]) + args.osc_offset,
                                     float(times[j]) + args.osc_offset)
            if d_osc is None:
                rejected["osc_unknown"] += 1
                continue
            if d_osc < args.min_osc_m:
                rejected["too_little_motion"] += 1
                continue
            candidates.append((i, j, float(dt), d_osc))
    print(f"OSC 筛选后待解帧对 {len(candidates)}", flush=True)
    last_report = time.perf_counter()
    for n, (i, j, dt, d_osc) in enumerate(candidates, 1):
        if time.perf_counter() - last_report >= 5.0:
            last_report = time.perf_counter()
            print(f"  进度 {n}/{len(candidates)}，已接受 {len(results)}", flush=True)
        stereo = stereo_displacement(data, meta, i, j, min_inliers=args.min_inliers)
        if stereo is None:
            rejected["pnp_failed"] += 1
            continue
        if stereo["rot_deg"] > args.max_rot_deg:
            rejected["rotation"] += 1
            continue
        if stereo["d_stereo"] < 0.05:
            rejected["stereo_too_small"] += 1
            continue
        results.append({"i": i, "j": j, "dt": dt, "d_osc": d_osc,
                        **stereo, "s": d_osc / stereo["d_stereo"]})
    print(f"帧对：接受 {len(results)}，拒绝 {rejected}")
    if not results:
        print("没有可用帧对，无法给出尺度。最常见原因：录制期间没走动，或 OSC 没收到。")
        return 1

    # 同一段运动会产生大量高度重叠的帧对，按起点帧去重，避免一段路在统计里占满。
    by_start: dict[int, dict[str, Any]] = {}
    for item in results:
        keep = by_start.get(item["i"])
        if keep is None or item["inliers"] > keep["inliers"]:
            by_start[item["i"]] = item
    picked = list(by_start.values())
    scales = np.array([r["s"] for r in picked])
    median = float(np.median(scales))
    mad = float(np.median(np.abs(scales - median)))
    print(f"\n独立帧对 {len(picked)}：s 中位数 = {median:.4f}，MAD = {mad:.4f} "
          f"（相对 {mad / median:.1%}），p10–p90 = {np.percentile(scales, 10):.4f}–"
          f"{np.percentile(scales, 90):.4f}")
    print(f"⇒ 世界有效基线 ≈ {meta['baseline_m'] * median:.4f} m；世界深度 = s × fx·0.063/dx")

    # 模型检验：s 若随场景深度系统性漂移，说明焦距或基线模型本身有误。
    depths = np.array([r["median_depth"] for r in picked])
    edges = np.percentile(depths, [0, 33, 67, 100])
    print("\n按场景深度（追踪米）分档的 s：")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (depths >= lo) & (depths <= hi)
        if mask.sum():
            print(f"  深度 {lo:5.2f}–{hi:5.2f}: n={int(mask.sum()):3d} s 中位数={np.median(scales[mask]):.4f}")
    distances = np.array([r["d_osc"] for r in picked])
    print("按运动距离（世界米）分档的 s：")
    for lo, hi in zip(*(lambda e: (e[:-1], e[1:]))(np.percentile(distances, [0, 33, 67, 100]))):
        mask = (distances >= lo) & (distances <= hi)
        if mask.sum():
            print(f"  距离 {lo:5.2f}–{hi:5.2f}: n={int(mask.sum()):3d} s 中位数={np.median(scales[mask]):.4f}")
    reproj = np.array([r["reproj_px"] for r in picked])
    print(f"PnP 重投影中位数：{np.median(reproj):.2f} px；内点中位数："
          f"{int(np.median([r['inliers'] for r in picked]))}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": meta, "rejected": rejected, "s_median": median,
                                   "s_mad": mad, "pairs": picked}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"已写入 {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record")
    rec.add_argument("--seconds", type=float, default=60.0)
    rec.add_argument("--fps", type=float, default=4.0)
    rec.add_argument("--features", type=int, default=2000)
    rec.add_argument("--max-dy", type=float, default=1.5)
    rec.add_argument("--min-disp", type=float, default=3.0,
                     help="视差下限（px），低于它的远点深度误差过大")
    rec.add_argument("--osc-host", default="127.0.0.1")
    rec.add_argument("--osc-port", type=int, default=9001)
    rec.add_argument("--preview", action="store_true")
    rec.add_argument("--out", required=True)
    rec.set_defaults(func=record)

    ana = sub.add_parser("analyze")
    ana.add_argument("recording")
    # 窗口依据 run1 实测：102° FOV 下以行走速度移动 1.5 s，前后两帧已基本匹配不上
    # （1.5–4.0 s 窗口 0 对通过）；0.3–1.5 s 内三个子窗口给出的 s 相互一致。
    ana.add_argument("--min-dt", type=float, default=0.5)
    ana.add_argument("--max-dt", type=float, default=1.2)
    ana.add_argument("--min-osc-m", type=float, default=0.5)
    ana.add_argument("--max-rot-deg", type=float, default=8.0)
    ana.add_argument("--min-inliers", type=int, default=40)
    # 画面 t 时刻对应的运动，OSC 在 t + offset 才报出来。run3 扫描 -0.30…+0.30 s：
    # +0.12～0.15 s 时 MAD 从 6.4% 降到 2.7%，短位移档从 0.53 拉平到与其余两档一致；
    # 中长位移档的 s 对 offset 基本不敏感（0.749–0.762）。
    ana.add_argument("--osc-offset", type=float, default=0.13)
    ana.add_argument("--json", default="")
    ana.set_defaults(func=analyze)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
