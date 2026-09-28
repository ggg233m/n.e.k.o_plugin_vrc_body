# -*- coding: utf-8 -*-
"""room-scale 行走探针：只推虚拟 HMD 的位置，看 VRChat 怎么对待"头自己走过去"。

要回答的三件事（见 2026-09-27 OSC 与 room-scale 对比）：
1. 撞墙：朝墙走时视角会不会穿进墙里（看 ``snap_*.png``），还是被挡住/平移 playspace。
2. 别人看到的动作：VRChat 回报的 ``VelocityX/Z`` 是否跟着动。这组参数驱动 avatar
   的行走动画，全程为 0 基本说明别人看到的是滑行；最终还要对着镜子或用另一个账号确认。
3. 尺度：回报速度 ÷ 下发的追踪速度，应接近世界尺度 s≈0.755。

做法：从 HMD **当前**位姿出发（不跳），沿当前水平朝向匀速走 ``--distance`` 追踪米，
停住，原路走回，最后停在起点——驱动会保持最后一包，所以脚本结束后头就留在原处。
只发 HMD 部分帧（``encode_head_frame``），身体和手柄保持驱动里的上一次位姿。

后端必须停着：它会同时推 HMD（两个发送方互相覆盖），而且占着 OSC 9001。
脚本启动时会听 1.5 s 驱动日志，发现别的发送方就拒绝运行。

用法
----
  .venv/Scripts/python.exe -W ignore research/tools/roomscale_probe.py --out .slam_probe/roomscale/wall1
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from research.tools.stereo_scale_calibration import _OSC_PREFIX, OscRecorder  # noqa: F401  （也负责包路径引导）
from neko_anyadance_body.config import DriverLogConfig, SafetyConfig
from neko_anyadance_body.driver_log import DriverLogListener
from neko_anyadance_body.model import DeviceState
from neko_anyadance_body.osc import decode_osc_packet
from neko_anyadance_body.protocol import encode_head_frame

S_WORLD = 0.755
DRIVER_ADDR = ("127.0.0.1", 39570)


class AllOscRecorder(OscRecorder):
    """OscRecorder 只留速度三轴；这里把所有 avatar 参数都记下来（Grounded、Upright 等）。"""

    def _run(self) -> None:  # noqa: D401
        self.all: list[tuple[float, str, float]] = []
        while not self._stop.is_set():
            try:
                packet, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            t = time.perf_counter()
            try:
                messages = decode_osc_packet(packet)
            except Exception:
                continue
            for address, args in messages:
                if not address.startswith(_OSC_PREFIX) or not args:
                    continue
                value = args[0]
                if isinstance(value, (bool, int, float)):
                    self.all.append((t, address[len(_OSC_PREFIX):], float(value)))


def _other_senders(own_port: int, seconds: float) -> list[str]:
    listener = DriverLogListener(DriverLogConfig())
    listener.start()
    try:
        time.sleep(seconds)
        senders = listener.snapshot().get("senders") or []
    finally:
        listener.stop()
    return [s for s in senders if not str(s).endswith(f":{own_port}")]


def _raw_hmd(system, openvr) -> np.ndarray | None:
    poses = (openvr.TrackedDevicePose_t * 1)()
    system.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseRawAndUncalibrated, 0.0, poses)
    pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    if not pose.bPoseIsValid:
        return None
    m = pose.mDeviceToAbsoluteTracking
    return np.array([[m[i][j] for j in range(4)] for i in range(3)], dtype=np.float64)


def _plan(start: np.ndarray, forward: np.ndarray, distance: float, speed: float,
          hold: float, rate: float) -> list[tuple[str, np.ndarray]]:
    """起点停 hold → 前进 distance → 停 hold → 退回 → 停 1 s。匀速，端点不做加减速。"""
    steps = max(1, int(round(distance / speed * rate)))
    out: list[tuple[str, np.ndarray]] = []
    out += [("hold_start", start)] * int(hold * rate)
    out += [("forward", start + forward * distance * k / steps) for k in range(1, steps + 1)]
    far = start + forward * distance
    out += [("hold_far", far)] * int(hold * rate)
    out += [("back", far - forward * distance * k / steps) for k in range(1, steps + 1)]
    out += [("hold_end", start)] * int(rate)
    return out


def _summarize(out: Path, sent: list, osc_all: list, readback: list, speed: float) -> dict:
    """按阶段汇总：VRChat 回报的水平速度模长，和下发速度比。"""
    t_sent = np.array([t for t, _, _ in sent])
    phases = [p for _, p, _ in sent]
    vel = {"VelocityX": [], "VelocityZ": []}
    for t, name, v in osc_all:
        if name in vel:
            vel[name].append((t, v))
    summary: dict = {"commanded_speed_tracking_mps": speed, "phases": {}}
    for phase in dict.fromkeys(phases):
        idx = [i for i, p in enumerate(phases) if p == phase]
        t0, t1 = t_sent[idx[0]] + 0.13, t_sent[idx[-1]] + 0.13   # OSC 比画面晚约 0.13 s
        vx = [v for t, v in vel["VelocityX"] if t0 <= t <= t1]
        vz = [v for t, v in vel["VelocityZ"] if t0 <= t <= t1]
        n = min(len(vx), len(vz))
        mag = np.hypot(vx[:n], vz[:n]) if n else np.array([])
        summary["phases"][phase] = {
            "duration_s": float(t1 - t0),
            "osc_velocity_samples": int(n),
            "osc_speed_median_world_mps": float(np.median(mag)) if n else None,
            "osc_speed_max_world_mps": float(mag.max()) if n else None,
        }
    fwd = summary["phases"].get("forward", {}).get("osc_speed_median_world_mps")
    summary["ratio_osc_over_commanded"] = None if not fwd else fwd / speed
    summary["expected_ratio_if_world_scale"] = S_WORLD
    if readback:
        err = [float(np.linalg.norm(cmd - got)) for _, cmd, got in readback]
        summary["driver_readback_err_m"] = {"median": float(np.median(err)), "max": float(max(err))}
    params = sorted({name for _, name, _ in osc_all})
    summary["osc_parameters_seen"] = params
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--out", required=True)
    ap.add_argument("--distance", type=float, default=1.5, help="前进距离（追踪米）；朝墙测时设得比到墙的距离大")
    ap.add_argument("--speed", type=float, default=0.5, help="追踪米/秒")
    ap.add_argument("--hold", type=float, default=2.0, help="起点和远端各停几秒")
    ap.add_argument("--rate", type=float, default=60.0, help="发包频率，与 AnyaDance kStreamRateHz 一致")
    ap.add_argument("--snap-every", type=float, default=0.5, help="每隔几秒存一张左眼画面，0 为不存")
    ap.add_argument("--no-osc", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只读当前 HMD、打印计划并做边界检查，不发包")
    a = ap.parse_args()

    import cv2
    import openvr
    from scipy.spatial.transform import Rotation

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} 非空，换一个 --out。")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    safety = SafetyConfig()
    system = openvr.init(openvr.VRApplication_Background)
    eyes: list = []
    device = context = None
    osc = None
    sock = None
    sent: list[tuple[float, str, np.ndarray]] = []
    readback: list[tuple[float, np.ndarray, np.ndarray]] = []
    try:
        m = _raw_hmd(system, openvr)
        if m is None:
            print("HMD 位姿无效，SteamVR 是否在跑？")
            return 2
        start = m[:, 3].copy()
        rot = m[:, :3]
        quat = tuple(float(q) for q in Rotation.from_matrix(rot).as_quat())
        fwd = -rot[:, 2]
        fwd[1] = 0.0
        if np.linalg.norm(fwd) < 1e-3:
            print("头几乎竖直朝上/下，定不了水平朝向。")
            return 2
        fwd /= np.linalg.norm(fwd)
        plan = _plan(start, fwd, a.distance, a.speed, a.hold, a.rate)
        far = start + fwd * a.distance
        print(f"起点 {np.round(start, 3)}  朝向 {np.round(fwd, 3)}  远端 {np.round(far, 3)}  "
              f"共 {len(plan)} 包 / {len(plan) / a.rate:.1f} s")
        for _, pos in plan:   # 发第一包之前整条轨迹都要过插件安全边界
            encode_head_frame(DeviceState(position=tuple(pos), rotation=quat), safety)
        if a.dry_run:
            print("dry-run：边界检查通过，未发包。")
            return 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        others = _other_senders(sock.getsockname()[1], 1.5)
        if others:
            print(f"驱动还在收别的发送方 {others}：先停后端或 AnyaDance UI 的推流。")
            return 2
        osc = None if a.no_osc else AllOscRecorder("127.0.0.1", 9001)

        if a.snap_every > 0:
            from backend.wgc_capture import _release  # noqa: F401
            from research.tools.openvr_mirror_probe import MirrorEye, create_device, read_stereo
            device, context = create_device()
            compositor = openvr.VRCompositor()
            for eye in (openvr.Eye_Left, openvr.Eye_Right):
                eyes.append(MirrorEye(compositor, device, context, eye, gpu_downscale=(720, 1)))
            size = eyes[0].gpu_size or (eyes[0].desc["Width"], eyes[0].desc["Height"])

        period = 1.0 / a.rate
        t_start = time.perf_counter()
        next_snap = t_start
        for k, (phase, pos) in enumerate(plan):
            due = t_start + k * period
            while (now := time.perf_counter()) < due:
                time.sleep(min(0.002, due - now))
            sock.sendto(encode_head_frame(DeviceState(position=tuple(pos), rotation=quat), safety), DRIVER_ADDR)
            sent.append((now, phase, pos))
            if k % 6 == 0:
                got = _raw_hmd(system, openvr)
                if got is not None:
                    readback.append((now, pos, got[:, 3]))
            if eyes and now >= next_snap:
                next_snap += a.snap_every
                left, _ = read_stereo(eyes[0], eyes[1], size)
                if left is not None and left.any():
                    cv2.imwrite(str(out / f"snap_{now - t_start:05.2f}_{phase}.png"),
                                cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        time.sleep(0.5)   # 让最后一段 OSC 回报到齐
    finally:
        if osc is not None:
            osc.close()
        if sock is not None:
            sock.close()
        for mirror in eyes:
            mirror.close()
        if context is not None:
            from backend.wgc_capture import _release
            _release(context)
            _release(device)
        openvr.shutdown()

    t0 = sent[0][0]
    osc_all = [] if osc is None else osc.all
    with (out / "sent.jsonl").open("w", encoding="utf-8") as f:
        for t, phase, pos in sent:
            f.write(json.dumps({"t": t - t0, "phase": phase, "pos": pos.tolist()}) + "\n")
    with (out / "osc_all.jsonl").open("w", encoding="utf-8") as f:
        for t, name, v in osc_all:
            f.write(json.dumps({"t": t - t0, "name": name, "value": v}, ensure_ascii=False) + "\n")
    s = _summarize(out, sent, osc_all, readback, a.speed)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"\n画面在 {out}/snap_*.png。撞墙看 hold_far 那几张；动画要对着镜子或另一个账号目测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
