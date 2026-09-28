# -*- coding: utf-8 -*-
"""录一段合成器镜像双目序列，写成 ORB-SLAM3 ``stereo_euroc`` 可读的 EuRoC 目录。

为什么要它
----------
路线 A：用真版 ORB-SLAM3 双目模式作**研究参照**，回答"双目在 VRChat 镜像画面上能不能
初始化、跟住、尺度准不准"。ORB-SLAM3 是 GPLv3，只在 ``.slam_probe/`` 离线跑，不接插件。

输出布局（``stereo_euroc`` 硬编码的路径）::

    <out>/mav0/cam0/data/<ns>.png   左眼
    <out>/mav0/cam1/data/<ns>.png   右眼
    <out>/times.txt                 每行一个 <ns>
    <out>/VRChatStereo.yaml         Camera.type=Rectified
    <out>/osc.npy                   (t, axis, value) —— 与 stereo_scale_calibration 同格式
    <out>/meta.json
    <out>/osc.jsonl、hmd_frames.jsonl、run.json
                                    离线路线（recorder/run_motion.py）的遥测格式，
                                    供 research/tools/stereo_seq_ground_truth.py 做 OSC+HMD 航位推算

内参与基线
----------
镜像每眼 2880×1620、fx=fy=810（``getProjectionRaw``），两眼已校正平行
（``research/tools/openvr_mirror_probe.py`` 实测 |dy|≈0）。

输出分辨率**不写死**：GPU 端沿 mip 链整数折半，取仍不小于 ``--target-width``
的最深一级，内参按实际比例缩放。2880 宽 ⇒ mip 2 = 720×405、fx=fy=202.5。
换了 SteamVR 渲染分辨率不用改参数。

为什么是 720×405 而不是 640×360（2026-09-26 实测）：720×405 正好落在 mip 上，
省掉 CPU 那次 1.125:1 非整数缩放 —— 取帧 6.5→1.8 ms，锐度（Laplacian var）
831→1411，fx 还大 12.5%；ORB 成本涨 15%，合计耗时 36.3→36.1 ms 持平。
即更准且不更慢。

``Stereo.b`` 填**追踪空间**的 0.063 m（渲染器实际用的眼距），不填世界基线：这样
ORB-SLAM3 输出是追踪米，世界尺度 s（``research/tools/stereo_scale_calibration.py``，run3 得
0.755）只在分析时乘一次，两个量互不耦合。

时间戳
------
``stereo_euroc`` 会按时间戳间隔 usleep 回放，所以时间戳必须是真实采集时刻（ns，
``perf_counter`` 起点归零），不能按帧号伪造。

HMD 位姿在等下一帧的空转里顺手读（``getDeviceToAbsoluteTrackingPose``，约 200 Hz），
与图像、OSC 同一把 ``perf_counter``。不开线程：IVRSystem 不保证线程安全。

用法
----
  .venv/Scripts/python.exe -W ignore research/tools/record_stereo_euroc.py --seconds 60 \
      --out .slam_probe/stereo_seq/run1
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from research.tools.stereo_scale_calibration import _OSC_PREFIX, _VELOCITY_NAMES, OscRecorder

_YAML = """%YAML:1.0
#--------------------------------------------------------------------------------------------
# N.E.K.O：SteamVR 合成器镜像双目（GetMirrorTextureD3D11），由
# research/tools/record_stereo_euroc.py 生成。两眼已是校正好的平行立体对（|dy|≈0），无畸变。
# 内参从镜像的 {src_w}x{src_h} / fx={src_fx:.1f} 按比例缩放而来。
# Stereo.b 是追踪空间眼距（getEyeToHeadTransform），不是世界基线——
# 世界尺度 s 在分析时单独乘。
#--------------------------------------------------------------------------------------------
File.version: "1.0"

Camera.type: "Rectified"

Camera1.fx: {fx:.4f}
Camera1.fy: {fy:.4f}
Camera1.cx: {cx:.4f}
Camera1.cy: {cy:.4f}

Stereo.b: {baseline:.6f}

Camera.width: {width}
Camera.height: {height}

Camera.fps: {fps}

# 写盘的是 OpenCV BGR PNG。
Camera.RGB: 0

# 近点阈值，单位为基线倍数。{th_depth} × {baseline:.4f} m ≈ {th_depth_m:.2f} 追踪米。
Stereo.ThDepth: {th_depth}

ORBextractor.nFeatures: {features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


def _write_telemetry(out: Path, started: float, osc_samples, hmd_samples) -> int:
    """按 recorder/run_motion.py 的格式写遥测，让 OscPath / HmdYaw 直接读。

    run_motion 的时间轴是 ``t - obs_start_monotonic``；这里两者都用 perf_counter，
    锚点取录制起点，于是遥测秒与图像 ns/1e9 同一零点。
    """
    from scipy.spatial.transform import Rotation

    with (out / "osc.jsonl").open("w", encoding="utf-8") as f:
        for t, axis, value in osc_samples:
            addr = _OSC_PREFIX + _VELOCITY_NAMES[int(axis)]
            f.write(json.dumps({"t": t, "addr": addr, "args": [value]}) + "\n")
    written = 0
    with (out / "hmd_frames.jsonl").open("w", encoding="utf-8") as f:
        for t, m in hmd_samples:
            m = np.asarray(m, dtype=np.float64)
            q = Rotation.from_matrix(m[:, :3]).as_quat()  # x, y, z, w
            f.write(json.dumps({"t": t, "hmd": {"rotation_xyzw": q.tolist(),
                                                "position_xyz": m[:, 3].tolist()}}) + "\n")
            written += 1
    (out / "run.json").write_text(json.dumps({
        "source": "research/tools/record_stereo_euroc.py",
        "clock": "time.perf_counter（run_motion 只用差值，时钟名不影响）",
        "events": [{"obs_start_monotonic": started}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=15.0,
                        help="目标采集率；抓帧改为映射内存上直接缩放后双目约 26 ms，"
                             "PNG 写盘在后台线程，实测上限约 15 Hz 以上")
    parser.add_argument("--target-width", type=int, default=720,
                        help="目标宽度下限；实际输出取 mip 链上仍不小于它的最深一级，"
                             "所以会随 SteamVR 渲染分辨率自动变化，不写死像素数")
    parser.add_argument("--features", type=int, default=1200)
    parser.add_argument("--th-depth", type=float, default=40.0)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9001)
    parser.add_argument("--no-osc", action="store_true",
                        help="不监听 OSC（后端在跑、占着 9001 时用）；此时没有米制真值")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    import openvr

    from backend.wgc_capture import _release
    from research.tools.openvr_mirror_probe import MirrorEye, create_device, read_stereo

    out = Path(args.out)
    cam0 = out / "mav0" / "cam0" / "data"
    cam1 = out / "mav0" / "cam1" / "data"
    if cam0.exists() and any(cam0.iterdir()):
        print(f"{cam0} 已有图像，拒绝覆盖。换一个 --out。")
        return 2
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)

    osc = None if args.no_osc else OscRecorder(args.osc_host, args.osc_port)
    # PNG 编码（两眼约 34 ms）放到后台线程：否则它和抓帧串行，直接吃掉采集率。
    # 队列有界，写盘跟不上时抓帧会被 put() 阻塞，而不是无限堆内存。
    writes: queue.Queue = queue.Queue(maxsize=64)
    write_errors: list[str] = []

    def _writer() -> None:
        while True:
            item = writes.get()
            if item is None:
                return
            path, img = item
            if not cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
                write_errors.append(str(path))

    writer = threading.Thread(target=_writer, name="png-writer", daemon=True)
    writer.start()
    system = openvr.init(openvr.VRApplication_Background)
    device = context = None
    eyes: list = []
    stamps: list[int] = []
    grab_ms: list[float] = []
    blank_frames = 0
    hmd_samples: list[tuple[float, list]] = []
    poses = (openvr.TrackedDevicePose_t * 1)()

    def _sample_hmd() -> None:
        system.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0, poses)
        t = time.perf_counter()
        pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
        if pose.bPoseIsValid:
            m = pose.mDeviceToAbsoluteTracking
            hmd_samples.append((t, [[m[i][j] for j in range(4)] for i in range(3)]))

    try:
        eye_x = [system.getEyeToHeadTransform(e)[0][3] for e in (openvr.Eye_Left, openvr.Eye_Right)]
        baseline = float(eye_x[1] - eye_x[0])
        device, context = create_device()
        compositor = openvr.VRCompositor()
        # 目标高度传 1：mip 链靠宽度下限决定深度，这样换了渲染分辨率也不用改参数。
        for eye in (openvr.Eye_Left, openvr.Eye_Right):
            eyes.append(MirrorEye(compositor, device, context, eye,
                                  gpu_downscale=(args.target_width, 1)))
        src_w, src_h = eyes[0].desc["Width"], eyes[0].desc["Height"]
        # mip 链只能整数折半，所以实际尺寸由源分辨率决定；没有可用 mip 时就原样回读。
        out_w, out_h = eyes[0].gpu_size or (src_w, src_h)
        left_b, right_b, top_b, bottom_b = system.getProjectionRaw(openvr.Eye_Left)
        src_fx = src_w / (right_b - left_b)
        src_fy = src_h / (bottom_b - top_b)
        src_cx = src_w * (-left_b) / (right_b - left_b)
        src_cy = src_h * (-top_b) / (bottom_b - top_b)
        sx, sy = out_w / src_w, out_h / src_h
        fx, fy, cx, cy = src_fx * sx, src_fy * sy, src_cx * sx, src_cy * sy
        print(f"镜像 {src_w}x{src_h} fx={src_fx:.1f} → {out_w}x{out_h}（mip {eyes[0]._mip}，"
              f"宽度下限 {args.target_width}）fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}，"
              f"基线 {baseline:.4f} m")

        started = time.perf_counter()
        period = 1.0 / float(args.fps)
        next_due = started
        last_report = started
        size = (out_w, out_h)
        while time.perf_counter() - started < float(args.seconds):
            _sample_hmd()
            now = time.perf_counter()
            if now < next_due:
                time.sleep(min(0.005, next_due - now))
                continue
            # 落后时不补帧：丢掉欠账，从现在重新排期，避免连拍把时间戳挤在一起。
            next_due = max(next_due + period, now)
            t0 = time.perf_counter()
            left, right = read_stereo(eyes[0], eyes[1], size)
            t1 = time.perf_counter()
            # run5 首帧右眼整张全黑（镜像纹理刚创建、还没被合成器写过），RTAB-Map
            # 拿它初始化会拒掉全部双目对应、下一帧段错误。任一眼全黑就丢。
            if left is None or right is None or not left.any() or not right.any():
                blank_frames += 1
                continue
            ns = int(round((0.5 * (t0 + t1) - started) * 1e9))
            if stamps and ns <= stamps[-1]:
                continue
            for img, folder in ((left, cam0), (right, cam1)):
                writes.put((folder / f"{ns}.png", img))
            stamps.append(ns)
            grab_ms.append((t1 - t0) * 1000.0)
            if t1 - last_report >= 2.0:
                last_report = t1
                extra = "" if osc is None else f" OSC速度样本={len(osc.samples):5d}"
                print(f"  t={t1 - started:5.1f}s 帧={len(stamps):4d}{extra}", flush=True)
        elapsed = time.perf_counter() - started
    finally:
        writes.put(None)
        writer.join()
        if osc is not None:
            osc.close()
        for mirror in eyes:
            mirror.close()
        _release(context)
        _release(device)
        openvr.shutdown()

    if not stamps:
        print("没有录到任何帧")
        return 2
    if write_errors:
        # 缺图的序列 stereo_euroc 会在读到那一帧时直接退出，不能当成完整数据交出去。
        print(f"{len(write_errors)} 张 PNG 写盘失败，例如 {write_errors[0]}")
        return 2
    rate = len(stamps) / elapsed
    (out / "times.txt").write_text("".join(f"{ns}\n" for ns in stamps), encoding="ascii")
    (out / "VRChatStereo.yaml").write_text(_YAML.format(
        src_w=src_w, src_h=src_h, src_fx=src_fx, fx=fx, fy=fy, cx=cx, cy=cy,
        baseline=baseline, width=out_w, height=out_h,
        fps=max(1, int(round(rate))), th_depth=args.th_depth,
        th_depth_m=args.th_depth * baseline, features=args.features), encoding="utf-8")
    osc_arr = np.zeros((0, 3)) if osc is None else np.array(osc.samples, dtype=np.float64).reshape(-1, 3)
    if osc is not None and osc_arr.size:
        # OSC 用 perf_counter 绝对时刻，这里换到与图像同一个零点（秒）。
        osc_arr[:, 0] -= started
    np.save(out / "osc.npy", osc_arr)
    hmd_count = _write_telemetry(out, started, [] if osc is None else osc.samples, hmd_samples)
    hmd_t = np.array([t for t, _ in hmd_samples]) - started
    hmd_gap = float(np.max(np.diff(hmd_t))) if len(hmd_t) > 1 else None
    (out / "meta.json").write_text(json.dumps({
        "frames": len(stamps), "elapsed_s": elapsed, "rate_hz": rate,
        "grab_ms_median": float(np.median(grab_ms)),
        "blank_frames_dropped": blank_frames,
        "baseline_tracking_m": baseline, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": out_w, "height": out_h, "mip": eyes[0]._mip,
        "target_width": args.target_width,
        "source": {"width": src_w, "height": src_h, "fx": src_fx, "fy": src_fy},
        "osc_samples": int(len(osc_arr)), "osc_time_origin": "same as image ns/1e9",
        "hmd_samples": hmd_count, "hmd_max_gap_s": hmd_gap,
        "hmd_universe": "TrackingUniverseStanding",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}：{len(stamps)} 帧（{rate:.2f} Hz，grab 中位 {np.median(grab_ms):.1f} ms），"
          f"丢弃全黑帧 {blank_frames}，OSC 速度样本 {len(osc_arr)} 条，HMD 位姿 {hmd_count} 条"
          + ("" if hmd_gap is None else f"（最大间隔 {hmd_gap * 1000:.0f} ms）"))
    if hmd_count == 0:
        print("⚠ 没有有效的 HMD 位姿：这段序列只能验尺度，算不出航位推算真值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
