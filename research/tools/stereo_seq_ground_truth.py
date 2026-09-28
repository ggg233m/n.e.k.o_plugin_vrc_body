# -*- coding: utf-8 -*-
"""双目序列的位置真值：OSC 本地速度 + HMD 朝向做航位推算，再给一条 SLAM 轨迹打分。

真值口径与 ``research/tools/slam_metric_retest.py::build_gt`` 同式：
- 位移：``recorder/run_motion.OscPath.disp``（**矢量**，不是标量路程），默认 ``zoh``；
- 朝向：``recorder/run_motion.HmdYaw``（四元数 yaw 解卷绕）；
- 旋转：唯一实现 ``backend/pose_math.advance``，本文件不写三角函数。

输入是 ``research/tools/record_stereo_euroc.py`` 写的 ``osc.jsonl / hmd_frames.jsonl / run.json``，
时间与图像 ``ns/1e9`` 同一零点。OSC 比画面晚约 0.13 s（``--osc-offset``），HMD 位姿
与镜像纹理同一时刻读取，不加偏移。

yaw 符号
--------
离线路线的 ``-1`` 是 AnyaDance 驱动回执里的四元数，这里读的是 SteamVR 的
``TrackingUniverseStanding``，**不能直接沿用**。没给 ``--yaw-sign`` 时，用被测轨迹
自己的相机 yaw 与 HMD yaw 的**变化率相关系数**定符号，并把相关系数一并报告。
这只借用一个二值约定，不借用精度；|r| 太小就拒绝判定。

打分（俯视 2D，x–z 平面）
------------------------
- ``ate_sim``：Umeyama 相似变换对齐后的 ATE（尺度自由）；
- ``ate_fixed``：尺度钉在 ``--scale``（追踪米 → 世界米，默认 0.755），只对齐旋转平移；
- ``rpe``：每 ``--rpe-window`` 秒一段的相对位移误差，占真值段长的百分比 —— 这是
  "漂移有多快"，不受丢追后重置的影响。

用法
----
  .venv/Scripts/python.exe -W ignore research/tools/stereo_seq_ground_truth.py <seq> --traj f_odom.txt
  .venv/Scripts/python.exe -W ignore research/tools/stereo_seq_ground_truth.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from backend.pose_math import advance, yaw_radians  # noqa: E402

_RM_PATH = REPO / "research" / "recorder" / "run_motion.py"
_spec = importlib.util.spec_from_file_location("neko_run_motion", _RM_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("无法加载 run_motion：%s" % _RM_PATH)
_rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rm)
OscPath, HmdYaw = _rm.OscPath, _rm.HmdYaw

#: 相机 yaw 与 HMD yaw 变化率的 |相关系数| 低于它就不判符号。
MIN_SIGN_CORR = 0.5


def build_ground_truth(seq: Path, frame_t: np.ndarray, yaw_sign: float,
                       osc_offset: float, policy: str = "zoh") -> dict:
    """在帧时刻栅格上做航位推算，返回世界米下的 (x, z)、yaw 与有效掩码。

    OSC 比画面晚 ``osc_offset`` 秒：帧区间 [a, b] 的位移取 OSC 的 [a+off, b+off]；
    旋进世界用区间起点的 HMD yaw（与 ``slam_metric_retest.build_gt`` 同为"上一步 yaw"）。
    """
    osc = OscPath(str(seq))
    hmd = HmdYaw(str(seq))
    yaw_deg = np.interp(frame_t, hmd.t, hmd.yaw)
    yaw = np.array([yaw_radians(v, yaw_sign=yaw_sign) for v in yaw_deg])
    n = len(frame_t)
    x = np.zeros(n)
    z = np.zeros(n)
    for k in range(1, n):
        dvx, dvz = osc.disp(frame_t[k - 1] + osc_offset, frame_t[k] + osc_offset, policy)
        x[k], z[k] = advance(x[k - 1], z[k - 1], dvx, dvz, yaw[k - 1])
    # 两端都得落在遥测覆盖内，否则 interp 会把积分夹住，看上去像站着不动。
    lo = max(float(osc.t[0]) - osc_offset, float(hmd.t[0]))
    hi = min(float(osc.t[-1]) - osc_offset, float(hmd.t[-1]))
    return {"x": x, "z": z, "yaw": yaw, "valid": (frame_t >= lo) & (frame_t <= hi),
            "window": [lo, hi], "hmd_yaw_deg": yaw_deg,
            "osc_packets": int(len(osc.t)), "osc_max_gap_s": osc.max_gap}


def camera_top_view(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EuRoC 行 (t, tx, ty, tz, qx, qy, qz, qw)，左目光学系（x 右、y 下、z 前）→
    俯视位置 (x, z) 与 yaw（z 轴在 x–z 平面的方向，右转为正，与 pose_math 同号）。"""
    from scipy.spatial.transform import Rotation

    fwd = Rotation.from_quat(traj[:, 4:8]).apply([0.0, 0.0, 1.0])
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    return traj[:, [1, 3]], yaw


def contiguous(t: np.ndarray, max_dt: float) -> np.ndarray:
    """每帧所属的连续段编号：相邻帧间隔超过 max_dt 就算断开（丢追）。"""
    return np.concatenate([[0], np.cumsum(np.diff(t) > max_dt)]) if len(t) else np.zeros(0, int)


def infer_yaw_sign(cam_t, cam_yaw, hmd_t, hmd_yaw_deg, seg) -> tuple[float | None, float]:
    """相机 yaw 与 HMD yaw 在同一连续段内逐帧变化量的相关系数定符号。"""
    same = seg[1:] == seg[:-1]
    d_cam = np.diff(cam_yaw)[same]
    h = np.radians(np.interp(cam_t, hmd_t, hmd_yaw_deg))
    d_hmd = np.diff(h)[same]
    if len(d_cam) < 10 or np.std(d_cam) == 0 or np.std(d_hmd) == 0:
        return None, 0.0
    r = float(np.corrcoef(d_cam, d_hmd)[0, 1])
    if abs(r) < MIN_SIGN_CORR:
        return None, r
    return (1.0 if r > 0 else -1.0), r


def umeyama2d(P: np.ndarray, Q: np.ndarray, with_scale: bool):
    """求 (s, R, t) 使 ||s R P + t − Q|| 最小（R 为真旋转，不允许镜像）。"""
    mu_p, mu_q = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - mu_p, Q - mu_q
    U, D, Vt = np.linalg.svd(Qc.T @ Pc / len(P))
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_p = float((Pc ** 2).sum()) / len(P)
    s = float((D * np.diag(S)).sum() / var_p) if with_scale and var_p > 1e-12 else 1.0
    return s, R, mu_q - s * (R @ mu_p)


def err_stats(e: np.ndarray) -> dict:
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "rmse": float(np.sqrt((e ** 2).mean())),
            "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "max": float(e.max())}


def load_traj(path: Path) -> np.ndarray:
    """EuRoC/TUM 轨迹 → (t_s, tx, ty, tz, qx, qy, qz, qw)，按时间戳去重（保留首条）。

    去重理由同 analyze_orbslam3_stereo：RECENTLY_LOST 时会把上一帧位姿再压一次。
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 8 or parts[0].startswith("#"):
            continue
        rows.append([float(v) for v in parts[:8]])
    a = np.array(rows, dtype=np.float64).reshape(-1, 8)
    if len(a) and a[0, 0] > 1e6:  # ns → s
        a[:, 0] /= 1e9
    a = a[np.argsort(a[:, 0], kind="stable")]
    _, first = np.unique(a[:, 0], return_index=True)
    return a[np.sort(first)]


def score(seq: Path, traj: np.ndarray, *, scale: float, yaw_sign: float | None,
          osc_offset: float, rpe_window: float, policy: str = "zoh") -> dict:
    frame_t = np.loadtxt(seq / "times.txt", dtype=np.float64).reshape(-1) / 1e9
    period = float(np.median(np.diff(frame_t)))
    seg = contiguous(traj[:, 0], 2.5 * period)
    cam_xz, cam_yaw = camera_top_view(traj)

    hmd = HmdYaw(str(seq))
    sign_r = None
    if yaw_sign is None:
        yaw_sign, sign_r = infer_yaw_sign(traj[:, 0], cam_yaw, hmd.t, hmd.yaw, seg)
        if yaw_sign is None:
            raise SystemExit(f"yaw 符号判不出（相机/HMD 转速相关 r={sign_r:.2f}，"
                             f"|r|<{MIN_SIGN_CORR}）。请用 --yaw-sign 显式给出。")

    gt = build_ground_truth(seq, frame_t, yaw_sign, osc_offset, policy)
    ok = (traj[:, 0] >= gt["window"][0]) & (traj[:, 0] <= gt["window"][1])
    t = traj[ok, 0]
    P = cam_xz[ok]
    Q = np.column_stack([np.interp(t, frame_t, gt["x"]), np.interp(t, frame_t, gt["z"])])
    seg = seg[ok]

    s, R, tr = umeyama2d(P, Q, with_scale=True)
    e_sim = np.linalg.norm((s * (R @ P.T)).T + tr - Q, axis=1)
    _, R1, t1 = umeyama2d(P * scale, Q, with_scale=False)
    e_fix = np.linalg.norm((R1 @ (P * scale).T).T + t1 - Q, axis=1)

    # RPE：同一连续段内，每 rpe_window 秒一对（起点每 rpe_window/2 秒滑一次）。
    rel, lens = [], []
    for sid in np.unique(seg):
        idx = np.flatnonzero(seg == sid)
        ts = t[idx]
        start = ts[0]
        while start + rpe_window <= ts[-1]:
            i = idx[np.searchsorted(ts, start)]
            j = idx[min(len(idx) - 1, np.searchsorted(ts, start + rpe_window))]
            d_gt = Q[j] - Q[i]
            d_est = R1 @ ((P[j] - P[i]) * scale)
            if np.linalg.norm(d_gt) > 0.3:  # 站着不动的窗口没有"百分比"可言
                rel.append(float(np.linalg.norm(d_est - d_gt)))
                lens.append(float(np.linalg.norm(d_gt)))
            start += rpe_window / 2
    rel_a, len_a = np.array(rel), np.array(lens)

    path_gt = float(np.linalg.norm(np.diff(np.column_stack([gt["x"], gt["z"]])[gt["valid"]], axis=0),
                                   axis=1).sum())
    return {
        "yaw_sign": yaw_sign, "yaw_sign_corr": sign_r, "osc_offset": osc_offset, "policy": policy,
        "gt_window_s": gt["window"], "gt_path_m": path_gt,
        "gt_extent_m": float(np.ptp(np.column_stack([gt["x"], gt["z"]])[gt["valid"]], axis=0).max()),
        "osc_packets": gt["osc_packets"], "osc_max_gap_s": gt["osc_max_gap_s"],
        "scored_frames": int(ok.sum()), "segments": int(len(np.unique(seg))),
        "sim_scale": s, "ate_sim": err_stats(e_sim),
        "fixed_scale": scale, "ate_fixed": err_stats(e_fix),
        "rpe_window_s": rpe_window, "rpe_pairs": int(len(rel_a)),
        "rpe_median_m": float(np.median(rel_a)) if len(rel_a) else None,
        "rpe_pct_median": float(np.median(rel_a / len_a)) if len(rel_a) else None,
        "_plot": {"t": t, "est": (R1 @ (P * scale).T).T + t1, "gt": Q,
                  "gt_full": np.column_stack([gt["x"], gt["z"]])[gt["valid"]]},
    }


def plot(res: dict, path: Path, size: int = 900) -> None:
    """俯视图：灰=完整真值，绿=被测时段真值，红=被测轨迹（固定尺度对齐后）。"""
    import cv2

    p = res["_plot"]
    pts = np.vstack([p["gt_full"], p["est"]])
    lo, span = pts.min(axis=0), max(1e-3, float(np.ptp(pts, axis=0).max()))
    k = (size - 60) / span

    def px(a):
        return np.round((a - lo) * k + 30).astype(np.int32).reshape(-1, 1, 2)

    img = np.full((size, size, 3), 255, np.uint8)
    cv2.polylines(img, [px(p["gt_full"])], False, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(img, [px(p["gt"])], False, (40, 160, 40), 2, cv2.LINE_AA)
    cv2.polylines(img, [px(p["est"])], False, (40, 40, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"ATE fixed-scale {res['ate_fixed']['rmse']:.2f} m  "
                f"(sim {res['ate_sim']['rmse']:.2f} m, s={res['sim_scale']:.3f})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def _selftest() -> int:
    """合成一段"走—原地转 90°—带横移走—丢追—再走"，验证真值、对齐、符号判定。

    负对照：强行给错 yaw 符号，ATE 必须变大 —— 否则说明打分在空转。
    """
    import tempfile

    from scipy.spatial.transform import Rotation

    fps, off, true_scale, phi = 15.0, 0.13, 0.755, 0.7
    frame_t = np.arange(0, 16.0, 1 / fps)

    def theta(t):  # HMD 原始 yaw（度）：4–6 s 转 +90°，10–11 s 再转 -45°
        return np.interp(t, [0, 4, 6, 10, 11, 16], [0, 0, 90, 90, 45, 45])

    def vel(t):  # avatar 本地速度 (vx, vz)，转身时为 0
        if t < 4:
            return 0.0, 1.0
        if t < 6 or 10 <= t < 11:
            return 0.0, 0.0
        return (0.3, 1.0) if t < 10 else (-0.2, 0.8)

    # 世界 yaw = -theta（即 yaw_sign = -1），由 selftest 自己判出来。
    dt = 1e-3
    fine = np.arange(0, 16.0 + dt, dt)
    X = np.zeros((len(fine), 2))
    for i in range(1, len(fine)):
        vx, vz = vel(fine[i - 1])
        psi = -np.radians(theta(fine[i - 1]))
        X[i] = X[i - 1] + dt * np.array([vx * np.cos(psi) + vz * np.sin(psi),
                                         -vx * np.sin(psi) + vz * np.cos(psi)])
    gx, gz = np.interp(frame_t, fine, X[:, 0]), np.interp(frame_t, fine, X[:, 1])

    with tempfile.TemporaryDirectory() as tmp:
        seq = Path(tmp)
        np.savetxt(seq / "times.txt", np.round(frame_t * 1e9).astype(np.int64), fmt="%d")
        anchor = 1000.0
        (seq / "run.json").write_text(json.dumps({"events": [{"obs_start_monotonic": anchor}]}))
        with (seq / "osc.jsonl").open("w") as f:
            for tau in np.arange(0, 16.0 + off, 0.05):  # OSC 晚 off 秒到
                vx, vz = vel(max(0.0, tau - off))
                for name, v in (("VelocityX", vx), ("VelocityZ", vz)):
                    f.write(json.dumps({"t": anchor + tau, "addr": "/avatar/parameters/" + name,
                                        "args": [v]}) + "\n")
        with (seq / "hmd_frames.jsonl").open("w") as f:
            for tau in np.arange(0, 16.0, 0.01):
                q = Rotation.from_euler("y", theta(tau), degrees=True).as_quat()
                f.write(json.dumps({"t": anchor + tau, "hmd": {"rotation_xyzw": q.tolist()}}) + "\n")

        # 被测轨迹：追踪单位 = 世界米 / true_scale，再整体绕竖轴转 phi；7.0–7.6 s 丢追。
        c, s = np.cos(phi), np.sin(phi)
        px_, pz_ = (c * gx + s * gz) / true_scale, (-s * gx + c * gz) / true_scale
        psi = -np.radians(theta(frame_t)) + phi
        keep = ~((frame_t > 7.0) & (frame_t < 7.6))
        q = Rotation.from_euler("y", psi[keep][:, None]).as_quat()
        traj = np.column_stack([frame_t[keep], px_[keep], np.zeros(keep.sum()), pz_[keep], q])

        res = score(seq, traj, scale=true_scale, yaw_sign=None, osc_offset=off, rpe_window=2.0)
        bad = score(seq, traj, scale=true_scale, yaw_sign=+1.0, osc_offset=off, rpe_window=2.0)

    checks = [
        ("符号判为 -1", res["yaw_sign"] == -1.0),
        ("符号相关 r < -0.9", res["yaw_sign_corr"] is not None and res["yaw_sign_corr"] < -0.9),
        ("相似变换尺度 ≈ 0.755（±0.5%）", abs(res["sim_scale"] / true_scale - 1) < 0.005),
        ("固定尺度 ATE < 2 cm", res["ate_fixed"]["rmse"] < 0.02),
        ("RPE 中位 < 1%", res["rpe_pct_median"] is not None and res["rpe_pct_median"] < 0.01),
        ("丢追切出 2 段", res["segments"] == 2),
        ("负对照：错符号 ATE > 0.5 m", bad["ate_fixed"]["rmse"] > 0.5),
    ]
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"  ate_fixed={res['ate_fixed']['rmse']:.4f} m  sim_scale={res['sim_scale']:.4f}  "
          f"r={res['yaw_sign_corr']:.3f}  错符号 ate={bad['ate_fixed']['rmse']:.2f} m")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq", nargs="?", help="record_stereo_euroc.py 的输出目录")
    ap.add_argument("--traj", help="被测轨迹（EuRoC/TUM，左目光学系）")
    ap.add_argument("--scale", type=float, default=0.755, help="追踪米 → 世界米（双目标定值）")
    ap.add_argument("--yaw-sign", type=float, choices=(-1.0, 1.0), default=None,
                    help="HMD yaw 符号；不给就用被测轨迹自己的转速相关判定")
    ap.add_argument("--osc-offset", type=float, default=0.13, help="OSC 比画面晚多少秒")
    ap.add_argument("--policy", choices=("zoh", "stop"), default="zoh")
    ap.add_argument("--rpe-window", type=float, default=2.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.seq or not args.traj:
        ap.error("需要 seq 和 --traj（或 --selftest）")
    seq = Path(args.seq)
    for name in ("hmd_frames.jsonl", "osc.jsonl", "run.json"):
        if not (seq / name).is_file():
            raise SystemExit(f"{seq} 缺少 {name}：这段序列录于 HMD 采样加入之前，没有位置真值，"
                             f"只能用 analyze_orbslam3_stereo.py 验尺度。")
    res = score(seq, load_traj(Path(args.traj)), scale=args.scale, yaw_sign=args.yaw_sign,
                osc_offset=args.osc_offset, rpe_window=args.rpe_window, policy=args.policy)
    if args.plot:
        plot(res, Path(args.plot))
    res.pop("_plot")
    sign_src = "给定" if res["yaw_sign_corr"] is None else f"判定，r={res['yaw_sign_corr']:+.2f}"
    print(f"真值：路程 {res['gt_path_m']:.1f} m，范围 {res['gt_extent_m']:.1f} m，"
          f"yaw_sign={res['yaw_sign']:+.0f}（{sign_src}），OSC 最大空档 {res['osc_max_gap_s']:.2f} s")
    print(f"被测：{res['scored_frames']} 帧，{res['segments']} 段")
    for key, title in (("ate_fixed", f"ATE（尺度钉 {args.scale}）"), ("ate_sim", "ATE（相似变换）")):
        e = res[key]
        print(f"  {title:18s} rmse {e['rmse']:.3f}  中位 {e['median']:.3f}  "
              f"p95 {e['p95']:.3f}  max {e['max']:.3f} m")
    print(f"  相似变换尺度 {res['sim_scale']:.3f}（标定 {args.scale}）")
    if res["rpe_pairs"]:
        print(f"  RPE({args.rpe_window:g}s) 中位 {res['rpe_median_m']:.3f} m = "
              f"{res['rpe_pct_median']:.1%} 段长（{res['rpe_pairs']} 对）")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
