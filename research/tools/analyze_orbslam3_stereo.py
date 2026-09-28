# -*- coding: utf-8 -*-
"""分析 ORB-SLAM3 双目在镜像序列上的结果（路线 A：研究参照，不接插件）。

输入：``research/tools/record_stereo_euroc.py`` 录的目录 + ``stereo_euroc`` 跑出的
``f_<name>.txt``（逐帧）与 ``kf_<name>.txt``（关键帧），EuRoC 格式：
``timestamp_ns tx ty tz qx qy qz qw``，单位是**追踪米**（yaml 里 Stereo.b=0.063）。

回答三件事：
1. **覆盖**：逐帧轨迹覆盖了多少帧 / 多长时间。ORB-SLAM3 的 SaveTrajectory 只写
   当前地图（``Atlas::GetAllKeyFrames`` 只返回当前地图），重置过的前段会整段缺失——
   覆盖率本身就是"跟不跟得住"的直接证据。日志里的 reset / new map 次数一并解析。
2. **尺度**：双目给的是绝对尺度（追踪米），乘世界尺度 s 后应与 OSC 零阶保持积分的
   世界位移一致。对 dt∈[min,max] 的帧对算 ``s_eff = d_osc / d_orb``：若双目尺度
   正确，``s_eff`` 应≈独立标定的 s（run3 = 0.755）且不随深度/距离漂移。
   ``d_osc`` 是路径长、``d_orb`` 是弦长，所以同样只用旋转小的帧对。
3. **轨迹形状**：输出轨迹俯视图 PNG，便于肉眼看是否"乱飘"。

OSC 时间：录制器已把 OSC 时间换到与图像 ns/1e9 同一零点；再叠加标定得到的
``--osc-offset``（OSC 比画面晚约 0.13 s）。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from research.tools.stereo_scale_calibration import osc_displacement


def load_euroc(path: Path) -> np.ndarray:
    """返回 (N, 8)：t_s, tx, ty, tz, qx, qy, qz, qw。空文件返回 (0, 8)。"""
    rows = []
    if not path.is_file():
        return np.zeros((0, 8))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 8:
            continue
        try:
            values = [float(v) for v in parts]
        except ValueError:
            continue
        # stereo_euroc 写的是 ns（SaveTrajectoryEuRoC 乘了 1e9）；小于 1e6 视为秒。
        values[0] = values[0] / 1e9 if values[0] > 1e6 else values[0]
        rows.append(values)
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    # Tracking.cc:2313 在当前帧没有位姿（RECENTLY_LOST）时把上一帧的时间戳和
    # 位姿原样再压一次，mlbLost 却是 false，SaveTrajectoryEuRoC 不会跳过它们。
    # 这些重复行不是追踪结果，按时间戳去重，只留第一次出现的那一行。
    if len(arr):
        _, first = np.unique(arr[:, 0], return_index=True)
        arr = arr[np.sort(first)]
    return arr


def quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    dot = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def parse_log(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    # 模式取自 ORB-SLAM3 源码里真实的输出字符串（Tracking.cc / Atlas.cc /
    # LoopClosing.cc）。System.cc:240 把 Verbose 设成 QUIET，所以
    # "Track Lost..." / "System Reseting" 这类 NORMAL 级消息不会出现——
    # 丢追只能从直接 cout 的 "Fail to track local map!" 和
    # "TRACK_REF_KF: Less than 15 matches" 间接看出。
    patterns = {
        "fail_local_map": r"Fail to track local map!",
        "ref_kf_lt15": r"TRACK_REF_KF: Less than 15 matches",
        "new_map": r"Creation of new map with id:",
        "stored_map": r"Stored map with ID",
        "stereo_init": r"New Map created with \d+ points",
        "relocalized": r"Relocalized!!",
        "loop": r"\*Loop detected|PR: Loop detected",
        "merge": r"Merge finished!",
        "reset": r"Reseting active map|Reseting current map in Local Mapping",
    }
    return {k: len(re.findall(p, text)) for k, p in patterns.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("sequence", help="record_stereo_euroc.py 的输出目录")
    parser.add_argument("--traj", required=True, help="f_<name>.txt 逐帧轨迹")
    parser.add_argument("--kf", default="", help="kf_<name>.txt 关键帧轨迹")
    parser.add_argument("--log", default="", help="stereo_euroc 的 stdout 日志")
    parser.add_argument("--world-scale", type=float, default=0.755,
                        help="追踪米→世界米，来自 stereo_scale_calibration run3")
    parser.add_argument("--osc-offset", type=float, default=0.13)
    parser.add_argument("--min-dt", type=float, default=0.5)
    parser.add_argument("--max-dt", type=float, default=1.5)
    parser.add_argument("--min-osc-m", type=float, default=0.5)
    parser.add_argument("--max-rot-deg", type=float, default=8.0)
    parser.add_argument("--png", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    seq = Path(args.sequence)
    meta = json.loads((seq / "meta.json").read_text(encoding="utf-8"))
    frame_ns = [int(line) for line in (seq / "times.txt").read_text().split()]
    frame_t = np.asarray(frame_ns, dtype=np.float64) / 1e9
    osc = np.load(seq / "osc.npy")
    if osc.size:
        osc = osc[np.argsort(osc[:, 0], kind="stable")]

    traj = load_euroc(Path(args.traj))
    kf = load_euroc(Path(args.kf)) if args.kf else np.zeros((0, 8))
    log = parse_log(Path(args.log) if args.log else None)

    duration = float(frame_t[-1] - frame_t[0]) if len(frame_t) > 1 else 0.0
    print(f"序列：{len(frame_t)} 帧 / {duration:.1f} s（{meta['rate_hz']:.2f} Hz），"
          f"OSC 速度样本 {len(osc)} 条")
    report: dict[str, Any] = {"frames": len(frame_t), "duration_s": duration, "log": log}

    # ---- 1. 覆盖 ----------------------------------------------------------
    if traj.size == 0:
        print("逐帧轨迹为空：ORB-SLAM3 没有留下任何当前地图帧。")
        report["coverage"] = 0.0
    else:
        span = float(traj[-1, 0] - traj[0, 0])
        coverage = len(traj) / len(frame_t)
        path_len = float(np.sum(np.linalg.norm(np.diff(traj[:, 1:4], axis=0), axis=1)))
        print(f"逐帧轨迹：{len(traj)} 帧（覆盖 {coverage:.1%}），时间跨度 "
              f"[{traj[0, 0]:.2f}, {traj[-1, 0]:.2f}] s = {span:.1f} s，"
              f"路径 {path_len:.2f} 追踪米 ≈ {path_len * args.world_scale:.2f} 世界米")
        report.update(coverage=coverage, traj_frames=len(traj), traj_span_s=span,
                      path_tracking_m=path_len)
    print(f"关键帧：{len(kf)}")
    if log:
        print("日志计数：" + "，".join(f"{k}={v}" for k, v in log.items()))

    # ---- 1b. 丢追空档与转速 -----------------------------------------------
    # 轨迹里缺掉的帧就是丢追：逐帧轨迹本该每帧一行，相邻行的时间差超过一个采样
    # 周期就说明中间那些帧没有位姿。把空档和"空档前最后一段的转速"放在一起，
    # 才能判断丢追是不是甩头引起的——这是 run1 的结论要验证的那一点。
    gaps: list[dict[str, Any]] = []
    if len(traj) > 2 and len(frame_t) > 1:
        period = float(np.median(np.diff(frame_t)))
        dts = np.diff(traj[:, 0])
        for idx in np.flatnonzero(dts > 2.5 * period):
            t0, t1 = float(traj[idx, 0]), float(traj[idx + 1, 0])
            missing = int(np.count_nonzero((frame_t > t0) & (frame_t < t1)))
            # 空档前 0.5 s 的偏航速度：用四元数相邻角差／dt，取最大值。
            window = [k for k in range(max(1, idx - int(round(0.5 / period))), idx + 1)]
            rate = max((quat_angle_deg(traj[k - 1, 4:8], traj[k, 4:8])
                        / max(1e-6, traj[k, 0] - traj[k - 1, 0]) for k in window), default=0.0)
            d_osc = osc_displacement(osc, t0 + args.osc_offset, t1 + args.osc_offset) if osc.size else None
            gaps.append({"t0": t0, "t1": t1, "dur_s": t1 - t0, "missing_frames": missing,
                         "turn_deg_s_before": rate, "osc_m_during": d_osc})
        lost_frames = sum(g["missing_frames"] for g in gaps)
        print(f"丢追空档：{len(gaps)} 段，缺 {lost_frames} 帧"
              + ("" if not gaps else "；" + "，".join(
                  f"[{g['t0']:.1f}→{g['t1']:.1f}]s {g['dur_s']:.1f}s "
                  f"转速{g['turn_deg_s_before']:.0f}°/s" for g in gaps[:6])))
        report.update(gap_count=len(gaps), lost_frames=lost_frames, gaps=gaps)

    # ---- 2. 尺度 ----------------------------------------------------------
    results = []
    rejected = {"osc_unknown": 0, "too_little_motion": 0, "rotation": 0, "orb_too_small": 0}
    if traj.size and osc.size:
        t = traj[:, 0]
        for i in range(len(traj)):
            for j in range(i + 1, len(traj)):
                dt = t[j] - t[i]
                if dt < args.min_dt:
                    continue
                if dt > args.max_dt:
                    break
                d_osc = osc_displacement(osc, t[i] + args.osc_offset, t[j] + args.osc_offset)
                if d_osc is None:
                    rejected["osc_unknown"] += 1
                    continue
                if d_osc < args.min_osc_m:
                    rejected["too_little_motion"] += 1
                    continue
                rot = quat_angle_deg(traj[i, 4:8], traj[j, 4:8])
                if rot > args.max_rot_deg:
                    rejected["rotation"] += 1
                    continue
                d_orb = float(np.linalg.norm(traj[j, 1:4] - traj[i, 1:4]))
                if d_orb < 0.05:
                    rejected["orb_too_small"] += 1
                    continue
                results.append({"i": i, "j": j, "dt": float(dt), "d_osc": d_osc,
                                "d_orb_tracking": d_orb, "rot_deg": rot,
                                "s_eff": d_osc / d_orb})
    print(f"\n尺度帧对：接受 {len(results)}，拒绝 {rejected}")
    if results:
        # 与标定工具同一去重口径：每个起点只留一对（取 dt 最接近区间中点的）。
        mid = 0.5 * (args.min_dt + args.max_dt)
        by_start: dict[int, dict[str, Any]] = {}
        for item in results:
            keep = by_start.get(item["i"])
            if keep is None or abs(item["dt"] - mid) < abs(keep["dt"] - mid):
                by_start[item["i"]] = item
        picked = list(by_start.values())
        s = np.array([r["s_eff"] for r in picked])
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        print(f"独立帧对 {len(picked)}：s_eff 中位 = {med:.4f}，MAD = {mad:.4f}（{mad / med:.1%}），"
              f"p10–p90 = {np.percentile(s, 10):.4f}–{np.percentile(s, 90):.4f}")
        print(f"对照独立标定 s = {args.world_scale:.4f}：偏差 {med / args.world_scale - 1.0:+.1%}")
        dist = np.array([r["d_osc"] for r in picked])
        if len(picked) >= 9:
            edges = np.percentile(dist, [0, 33, 67, 100])
            print("按运动距离（世界米）分档：" + " / ".join(
                f"{lo:.2f}–{hi:.2f}: {np.median(s[(dist >= lo) & (dist <= hi)]):.4f}"
                for lo, hi in zip(edges[:-1], edges[1:])))
        report.update(scale_pairs=len(picked), s_eff_median=med, s_eff_mad=mad,
                      s_ref=args.world_scale, rejected=rejected)
    else:
        report["rejected"] = rejected

    # ---- 3. 轨迹图 --------------------------------------------------------
    if args.png and traj.size:
        import cv2

        xz = traj[:, [1, 3]] * args.world_scale
        span = max(1e-3, float(np.ptp(xz, axis=0).max()))
        size, margin = 800, 40
        scale = (size - 2 * margin) / span
        origin = xz.min(axis=0)
        canvas = np.full((size, size, 3), 255, np.uint8)
        pts = ((xz - origin) * scale + margin).astype(np.int32)
        pts[:, 1] = size - pts[:, 1]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), (200, 80, 0), 2)
        cv2.circle(canvas, tuple(int(v) for v in pts[0]), 6, (0, 160, 0), -1)
        cv2.circle(canvas, tuple(int(v) for v in pts[-1]), 6, (0, 0, 200), -1)
        cv2.putText(canvas, f"ORB-SLAM3 stereo, top view (x,z) world m, span {span:.2f} m",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.png, canvas)
        print(f"轨迹图：{args.png}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
