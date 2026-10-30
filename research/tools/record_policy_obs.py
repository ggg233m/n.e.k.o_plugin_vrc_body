# -*- coding: utf-8 -*-
"""录 policy_obs 的原始双目流：AV1 硬件编码定频 + 遥测，纯采集模式（后端不跑）。

为什么需要它
------------
``kf_img`` 只在关键帧写、间隔不等距（中位 0.16–0.45 s），而头 A 要的是定频图 + 12 步动作块。
方案 v3 §5.2 原先打算在线写 224² JPEG，但点云与占据图可以离线重建，所以在线只存原始双目流
与时间戳，深度/占据图留给离线（LAS2 在 A770 上跑 78k 帧约 1–2 h）。

编码选型（2026-10-30 实测，1080×606）
------------------------------------
AV1 gq16：33.8 KB/帧、``av1_qsv`` 33–57 fps、90 min 约 9.3 GB；对 LAS2 深度引入 1.17%
中位相对误差（JPEG q90 是 0.49%、体积 4 倍；HEVC 在同体积下更差 1.88%）。因此这里只写 AV1
一条流。要更准的深度源另录 JPEG，不在本脚本范围。

复用
----
* 取帧与标定：``backend.openvr_mirror``（MirrorEye / read_stereo / projection_intrinsics）
* OSC：``research.tools.stereo_scale_calibration.OscRecorder``（速度三轴 + 全部 avatar 参数）
* 遥测落盘：``research.tools.record_stereo_euroc._write_telemetry``

输出
----
::

    <out>/L.mp4、R.mp4        AV1(av1_qsv)，CFR；帧序号 = frames.jsonl 的行号
    <out>/frames.jsonl        {"i": 序号, "t": 相对秒, "ns": 相对纳秒}（均相对 run.json 锚点）
    <out>/osc.jsonl           速度三轴（ZOH 动作标签源；t 是 perf_counter 绝对秒）
    <out>/avatar_params.jsonl EyeHeightAsMeters / ScaleFactor 等（世界米↔追踪米真值）
    <out>/hmd_frames.jsonl    HMD 位姿（每个采集节拍采一次；t 是 perf_counter 绝对秒）
    <out>/run.json、meta.json
    <out>/ffmpeg_{L,R}.log    编码器 stderr，失败时先看这里

三条口径
--------
1. **视频的 CFR 时间轴不可信**：``-fps_mode passthrough`` 只是按到达顺序写帧，采集抖动不会进
   容器。逐帧真实时刻以 ``frames.jsonl`` 为准，配对姿势/OSC 一律用它。
2. **帧号/时刻分属两套时间轴，锚点在 run.json**：``frames.jsonl`` 的 ``t``/``ns`` 相对
   ``run.json`` 的 ``events[].obs_start_monotonic``（录制循环起点）；``osc.jsonl`` /
   ``hmd_frames.jsonl`` 写 perf_counter **绝对秒**。对齐一律
   ``abs = obs_start_monotonic + frames.t``。录制循环起点在打开 OpenVR/镜像/编码器**之前**
   取，启动期没有帧，所以首帧 ``t`` 通常是 1–4 s，不是 0。
3. **丢帧必须成对**：两路编码器各自独立，只有保证同一帧号在 L/R 都存在，帧号 ↔ 时间戳才成立。
   队列满或整帧全黑时丢的是**整对**。

用法
----
::

  .venv/Scripts/python.exe research/tools/record_policy_obs.py --seconds 480 \
      --out navmesh_recordings/20261030_policy_obs
  # 验编码管线、不需要 SteamVR：
  .venv/Scripts/python.exe research/tools/record_policy_obs.py --selftest --seconds 5 \
      --out .tmp/pobs_selftest
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.openvr_mirror import MirrorEye, _release, create_device, projection_intrinsics, read_stereo
from research.tools.record_stereo_euroc import _write_telemetry
from research.tools.stereo_scale_calibration import OscRecorder


def _ffmpeg_cmd(ff: str, out: Path, size: tuple[int, int], fps: float, gq: int,
                preset: str, encoder: str) -> list[str]:
    """原始 rgb24 从 stdin 进，硬件编码到 mp4。``-fps_mode passthrough`` 不做重复/丢帧。"""
    return [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
        "-r", f"{fps:g}", "-i", "-",
        "-fps_mode", "passthrough",
        "-c:v", encoder, "-preset", preset, "-global_quality", str(gq),
        "-pix_fmt", "nv12", str(out),
    ]


class _Encoder:
    """一条 ffmpeg 子进程 + stderr 日志。失败原因只会出现在 log 里，不阻塞采集。"""

    def __init__(self, ff: str, out: Path, size: tuple[int, int], fps: float,
                 gq: int, preset: str, encoder: str, tag: str) -> None:
        self.log = out.parent / f"ffmpeg_{tag}.log"
        self.cmd = _ffmpeg_cmd(ff, out, size, fps, gq, preset, encoder)
        self._stderr = self.log.open("wb")
        # bufsize=0：管道不做额外缓冲，帧直接进 ffmpeg，代价是每次 write 可能阻塞。
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stderr=self._stderr, bufsize=0)

    def feed(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> int:
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001 - 编码器已经死了就无所谓
            pass
        code = self.proc.wait()
        self._stderr.close()
        return code


def _pump(enc_l: _Encoder, enc_r: _Encoder, q: "queue.Queue", errors: list[str]) -> None:
    """写线程：成对写入两路 stdin。镜像通道每帧 ~1.96 MB，放在采集线程会直接吃掉帧率。"""
    while True:
        item = q.get()
        if item is None:
            return
        left, right = item
        try:
            enc_l.feed(left)
            enc_r.feed(right)
        except Exception as exc:  # noqa: BLE001 - 编码器崩了就把原因带回去
            errors.append(f"{type(exc).__name__}: {exc}")
            return


def _synthetic(size: tuple[int, int], i: int) -> np.ndarray:
    """自测用的合成图：横向渐变 + 平移竖条，保证两路编码器有真实内容可编。"""
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:, :, 0] = np.arange(w, dtype=np.uint16)[None] % 256
    img[:, :, 1] = np.arange(h, dtype=np.uint16)[:, None] % 256
    img[:, :, 2] = 128
    x = (i * 7) % w
    img[:, x:x + 24, 2] = 255
    return img


def _sample_hmd(openvr: object, system: object, poses: object, out: list) -> None:
    """读一次 HMD 位姿（3×4 矩阵，追踪站立系）。与 record_stereo_euroc 同一口径。"""
    system.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0, poses)
    t = time.perf_counter()
    pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    if pose.bPoseIsValid:
        m = pose.mDeviceToAbsoluteTracking
        out.append((t, [[m[i][j] for j in range(4)] for i in range(3)]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=30.0, help="目标采集率；实测编码侧上限见 selftest")
    ap.add_argument("--target-width", type=int, default=1080,
                    help="GPU mip 宽度下限；实际尺寸取 mip 链上仍不小于它的最深一级（本项目 1080→1080×606）")
    ap.add_argument("--gq", type=int, default=16, help="av1_qsv 的 global_quality（越小越好，实测 16 是拐点）")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--encoder", default="av1_qsv")
    ap.add_argument("--ffmpeg", default="", help="默认取 PATH 上的 ffmpeg")
    ap.add_argument("--queue", type=int, default=24,
                    help="写线程队列深度（帧对）；满了就丢整对。启动前 1–2 s 编码器在预热，"
                         "队列深一点能把这段吸收掉（实测 8→32 时启动期丢帧 55→28）")
    ap.add_argument("--osc-host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=9001)
    ap.add_argument("--no-osc", action="store_true", help="后端在跑、9001 被占时用：此时没有米制真值")
    ap.add_argument("--selftest", action="store_true", help="不开 SteamVR，用合成图验编码管线")
    # 归类与远程停止：给录制 UI（research/tools/record_ui.py）用的三个口子。
    ap.add_argument("--world-key", default="", help="世界标识（VRChat 世界或自定义名），写进 meta.json")
    ap.add_argument("--world-label", default="", help="世界显示名（可留空，默认同 world-key）")
    ap.add_argument("--stop-file", default="", help="该文件出现即正常收尾（给 UI 的「停止」用，不用杀进程）")
    args = ap.parse_args()

    ff = args.ffmpeg or shutil.which("ffmpeg") or ""
    if not ff:
        print("找不到 ffmpeg：装一个或传 --ffmpeg <路径>（需要含 av1_qsv 的构建）")
        return 2

    out = Path(args.out)
    if (out / "L.mp4").exists():
        print(f"{out}/L.mp4 已存在，拒绝覆盖。换一个 --out。")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    osc = None if (args.no_osc or args.selftest) else OscRecorder(args.osc_host, args.osc_port)
    hmd_samples: list = []
    frames_written: list[tuple[int, float, int]] = []
    grab_ms: list[float] = []
    dropped_blank = 0
    dropped_queue = 0
    q: queue.Queue = queue.Queue(maxsize=max(2, args.queue))
    write_errors: list[str] = []

    system = device = context = None
    eyes: list[MirrorEye] = []
    encoders: list[_Encoder] = []
    writer = None
    started = time.perf_counter()

    try:
        if args.selftest:
            size = (args.target_width if args.target_width % 2 == 0 else args.target_width + 1, 606)
            baseline = 0.063
            intr = (303.75, 303.75, size[0] / 2, size[1] / 2)
            src = size
            mip = 0
        else:
            import openvr

            openvr.init(openvr.VRApplication_Background)
            system = openvr.VRSystem()
            eye_x = [system.getEyeToHeadTransform(e)[0][3] for e in (openvr.Eye_Left, openvr.Eye_Right)]
            baseline = float(eye_x[1] - eye_x[0])
            device, context = create_device()
            compositor = openvr.VRCompositor()
            # 高度传 1：mip 深度只由宽度下限决定，换渲染分辨率不用改参数（同 record_stereo_euroc）。
            for eye in (openvr.Eye_Left, openvr.Eye_Right):
                eyes.append(MirrorEye(compositor, device, context, eye,
                                      gpu_downscale=(args.target_width, 1)))
            src = (eyes[0].desc["Width"], eyes[0].desc["Height"])
            size = eyes[0].gpu_size or src
            intr = projection_intrinsics(system, openvr.Eye_Left, size, src)
            mip = eyes[0]._mip
            print(f"镜像 {src[0]}x{src[1]} → {size[0]}x{size[1]}（mip {mip}），"
                  f"fx={intr[0]:.2f} cx={intr[2]:.2f}，基线 {baseline:.4f} m")

        # 编码器在采集前先起，启动失败能立刻返回而不是录一堆空帧。
        for tag in ("L", "R"):
            encoders.append(_Encoder(ff, out / f"{tag}.mp4", size, args.fps, args.gq,
                                     args.preset, args.encoder, tag))
        for e in encoders:
            if e.proc.poll() is not None:
                print(f"编码器 {e.cmd[0]} 启动即退出，看 {e.log}")
                return 2
        writer = threading.Thread(target=_pump, args=(encoders[0], encoders[1], q, write_errors),
                                  name="av1-writer", daemon=True)
        writer.start()

        frames_path = out / "frames.jsonl"
        fh = frames_path.open("w", encoding="utf-8")
        poses = None
        if not args.selftest:
            poses = (openvr.TrackedDevicePose_t * 1)()

        period = 1.0 / float(args.fps)
        next_due = started
        last_report = started
        i = 0
        while True:
            now = time.perf_counter()
            if now - started >= args.seconds:
                break
            # UI 的「停止」写这个文件：走正常收尾路径，ffmpeg 能干净收流、meta 也会落盘。
            if args.stop_file and os.path.exists(args.stop_file):
                break
            if poses is not None:
                _sample_hmd(openvr, system, poses, hmd_samples)
                now = time.perf_counter()
            if now < next_due:
                time.sleep(min(0.004, next_due - now))
                continue
            # 落后不补帧：把欠账丢掉重新排期，否则连拍会把几帧挤在同一毫秒。
            next_due = max(next_due + period, now)

            t0 = time.perf_counter()
            if args.selftest:
                left = _synthetic(size, i)
                right = _synthetic(size, i + 3)
            else:
                left, right = read_stereo(eyes[0], eyes[1], size)
            t1 = time.perf_counter()

            # 首帧右眼整张全黑（镜像纹理刚创建还没被写过）会让下游误判，整对丢掉。
            if left is None or right is None or not left.any() or not right.any():
                dropped_blank += 1
                continue

            ns = int(round((0.5 * (t0 + t1) - started) * 1e9))
            if frames_written and ns <= frames_written[-1][2]:
                continue
            try:
                q.put_nowait((left, right))
            except queue.Full:
                dropped_queue += 1
                continue

            i += 1
            frames_written.append((i, (0.5 * (t0 + t1) - started), ns))
            fh.write(json.dumps({"i": i, "t": round(0.5 * (t0 + t1) - started, 6), "ns": ns}) + "\n")
            grab_ms.append((t1 - t0) * 1000.0)
            if t1 - last_report >= 2.0:
                last_report = t1
                recent = np.median(grab_ms[-40:])
                print(f"  t={t1 - started:5.1f}s 帧={i:4d} 抓帧={recent:.1f}ms "
                      f"队列={q.qsize()} 丢(空/慢)={dropped_blank}/{dropped_queue}", flush=True)
                # 机器可读进度：录制 UI 解析这一行，不改上面的给人看的那行。
                print("[status] " + json.dumps({
                    "t": round(t1 - started, 2), "frames": i, "rate": round(i / max(1e-6, t1 - started), 2),
                    "grab_ms": round(float(recent), 2), "queue": q.qsize(),
                    "dropped_blank": dropped_blank, "dropped_queue": dropped_queue,
                }), flush=True)
                # 每 2 s 落一次盘：进程被强杀时最多丢这 2 s 的时间戳（图仍在 mp4 里，帧号可重建）。
                fh.flush()
        fh.close()
    finally:
        elapsed = time.perf_counter() - started
        q.put(None)
        if writer is not None:
            writer.join(timeout=30.0)
        codes = [e.close() for e in encoders]
        if osc is not None:
            osc.close()
        for mirror in eyes:
            mirror.close()
        if args.selftest:
            pass
        else:
            if context is not None:
                _release(context)
            if device is not None:
                _release(device)
        if not args.selftest:
            import openvr

            openvr.shutdown()

    if not frames_written:
        print("没有录到任何帧")
        return 2
    if write_errors:
        print(f"写线程中断：{write_errors[0]}")
        return 2
    if any(codes):
        print(f"编码器退出码 {codes}，看 {out}/ffmpeg_*.log")
        return 2

    # 遥测与图像同一零点（perf_counter），由 _write_telemetry 负责换算。
    hmd_count = _write_telemetry(
        out, started,
        [] if osc is None else osc.samples,
        [] if osc is None else osc.params,
        hmd_samples,
    )
    hmd_t = np.array([t for t, _ in hmd_samples]) - started
    hmd_gap = float(np.max(np.diff(hmd_t))) if len(hmd_t) > 1 else None
    rate = len(frames_written) / elapsed
    meta = {
        "source": "research/tools/record_policy_obs.py",
        "world_key": args.world_key, "world_label": args.world_label or args.world_key,
        "frames": len(frames_written), "elapsed_s": round(elapsed, 2), "rate_hz": round(rate, 3),
        "size": list(size), "source_size": list(src), "mip": int(mip),
        "fx": round(intr[0], 4), "fy": round(intr[1], 4),
        "cx": round(intr[2], 4), "cy": round(intr[3], 4),
        "baseline_tracking_m": round(baseline, 6),
        "grab_ms_median": round(float(np.median(grab_ms)), 2),
        "dropped_blank": dropped_blank, "dropped_queue": dropped_queue,
        "encoder": {"name": args.encoder, "preset": args.preset, "global_quality": args.gq,
                    "pix_fmt_in": "rgb24", "pix_fmt_out": "nv12", "fps_mode": "passthrough"},
        "time_base": "frames.jsonl 的 t 是相对 started 的秒；hmd_frames/osc 的 t 是 perf_counter "
                     "绝对秒。对齐：abs = started + frames.t，started 见 meta.started_monotonic "
                     "与 run.json events[].obs_start_monotonic；mp4 的 CFR 时间轴不可信",
        "started_monotonic": started,
        "hmd_samples": hmd_count, "hmd_max_gap_s": None if hmd_gap is None else round(hmd_gap, 4),
        "osc_samples": 0 if osc is None else len(osc.samples),
        "clock": "perf_counter",
        "selftest": bool(args.selftest),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    # _write_telemetry 已写 run.json（含 events[].obs_start_monotonic，是 frames/hmd/osc 的唯一锚点），
    # 这里只补本脚本的口径，不能整份覆盖——覆盖会把锚点丢掉（G:\vrc_recording 那批 10 场就是这样）。
    run_path = out / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    run.update({"source": meta["source"], "clock": "time.perf_counter", "frames": len(frames_written),
                "world_key": args.world_key, "world_label": meta["world_label"],
                "encoder": meta["encoder"]})
    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    mb = sum(p.stat().st_size for p in (out / "L.mp4", out / "R.mp4")) / 1e6
    print(f"\n已写入 {out}：{len(frames_written)} 对帧（{rate:.2f} Hz，抓帧中位 "
          f"{np.median(grab_ms):.1f} ms），视频 {mb:.1f} MB（{mb * 1000 / max(1, len(frames_written)):.1f} KB/对），"
          f"丢帧 空 {dropped_blank} / 慢 {dropped_queue}，HMD {hmd_count} 条")
    print("[done] " + json.dumps({
        "out": str(out), "frames": len(frames_written), "rate": round(rate, 2),
        "mb": round(mb, 2), "dropped_blank": dropped_blank, "dropped_queue": dropped_queue,
        "world_key": args.world_key, "world_label": meta["world_label"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
