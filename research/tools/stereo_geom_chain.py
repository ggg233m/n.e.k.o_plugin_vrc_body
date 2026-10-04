# -*- coding: utf-8 -*-
r"""自建几何链：SGBM 视差 + ORB + PnP，量出**独立于 OSC** 的米制长度。

    # 受控双目序列（EuRoC 格式，要读图跑 SGBM）
    python research/tools/stereo_geom_chain.py .slam_probe/stereo_seq/run5_ipd126
    # 真实游玩录制（kf/*.npz 里已存 KeyframeFeatures + T_dr，**不用图像、不用重录**）
    python research/tools/stereo_geom_chain.py navmesh_recordings/20261001_044153 --skip 1,3

结论（2026-10-05，详见 `Docs/漂移形态诊断（2026-10-05）.md` §十）
------------------------------------------------------------
三场里样本最多的那档 s 都**显著 > 1**：045615 **1.0776**、044153 **1.0593**、
run5_ipd126 **1.0332**（CI 全不含 1）⇒ **DR/OSC 链比几何链长 3–8%，方向一致**。
但它**不是常量**：随基线变长而增大、在 044153 会话内还单调上升（1.021→1.103），
run5 上却随基线递减 ⇒ **不能拿去改 `world_scale` 常数**，该做的是**按会话自标定**
（`world_scale` 本就"随 avatar 变"）。

为什么要它
----------
`Docs/漂移形态诊断（2026-10-05）.md` §八 判定「判我们的链准不准不能拿 RTAB-Map 当尺子」
（该 run 的 RTAB-Map 轨迹不是直线：奇异值 161.9/51.9/7.5、横向展开 8.17 m，与文档记的
ATE 0.312 m 矛盾）。所以需要一个**不依赖任何第三方 SLAM** 的几何尺子：直接用双目视差
给深度（基线 0.126 由 `meta.json` 记录），用 PnP 量两帧之间的位移，与同时段的 OSC
航位推算逐段相比。

口径
----
* 几何位移（几何单位=基线所在的单位）：``nav_loop.relative_pose`` 用前一帧的双目 3D 点 +
  当前帧 2D 点做 PnP RANSAC，解出当前帧头部在前一帧头部 base 系（x 前 y 左 z 上）的位姿，
  ``t_ab[:2]`` 就是这一步的 (前, 左) 位移。
* DR 位移：``stereo_seq_ground_truth.build_ground_truth`` 给的是**世界米**，按
  ``world_scale``（世界米/几何单位，标定值 0.755）折到同一单位，再换到 (前, 左)。
* 两条链在**同一个抽帧网格**上取值，抽帧带来的弦-弧偏差对两者相同。

⚠️ 主判据是**已知对应的相似变换尺度 `s`**（Umeyama，几何步 → DR 步）：它同时吸收两链之间
的常量朝向差，且对 `yaw_sign` 不变。`s = 1` 即两链同尺；`world_scale` 的本次反推值 = 0.755 × s。

⚠️ 但**单个 s 不能直接当结论**：s 会随"弦多长""双目点留多远"摆动几个百分点（见扫描表）。
所以本工具默认跑一组 (子步长, 深度上限) 敏感性组合，只有这些组合**同向且都显著**时
才能说"两链不同尺"；摆动大就说明这个差是分析参数噪声，不是尺度误差。

⚠️ 弦长比（`d_dr/d_geo`）会被**侧向噪声**稀释，只作参考：本 run HMD 朝向不变、OSC 侧向恒 0，
几何的侧向分量是纯噪声（与 DR 相关 ≈ 0），混进弦长就把弦抬大、把比值压低。

⚠️ 头部在转的会话（run6/run7）**也能跑**：每步的几何位移会先用 `pose_math.advance` 按
该帧的 HMD yaw 旋进世界系（与 DR **同一个**旋转），再与 DR 比 —— 所以「前进分量」那条
轴分解只在头部不转时才有意义（那时 DR 侧向恒 0），转头的会话只看 **s 相似**。

⚠️ 但三场跑下来，**每步绝对残差都是 0.19–0.25 m，与步长基本无关**（PnP 的固有噪声地板）。
能不能读出尺度只取决于**步子够不够大**：run5_ipd126 步 RMS 0.79 m ⇒ 信噪比 0.27，s 读到
几个百分点；run6/run7 以原地转身为主（yaw 扫过 450–610°、步 RMS 0.46/0.49 m）⇒ 0.42/0.46，
CI 宽到 ±15–20%，只能算「与 1 不可区分」。工具末尾的质量闸就是这条。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "research" / "tools"), str(ROOT / "research" / "recorder")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402

from backend.nav_loop import (M_OPT, KeyframeFeatures, LoopConfig,  # noqa: E402
                              relative_pose, rotation_angle_deg)
from backend.nav_mapping import make_sgbm, stereo_disparity, stereo_points  # noqa: E402
from backend.pose_math import advance, yaw_radians                          # noqa: E402
from run_motion import HmdYaw                                               # noqa: E402
from stereo_seq_ground_truth import build_ground_truth, umeyama2d           # noqa: E402

#: 世界米 / 几何单位，与 mapper / nav_loop 的 world_scale 同值（stereo_scale_calibration 标定）。
WORLD_SCALE = 0.755

#: 默认敏感性扫描：子步长（×基础抽帧）× 双目点深度上限（米）。
DEFAULT_SWEEP = "1:8,2:8,3:8,5:8,1:5,2:5"


# ---------------------------------------------------------------------------
# 体检
# ---------------------------------------------------------------------------

def head_motion(seq: Path) -> dict:
    """HMD 遥测：相对首样本的最大转角、高度中位/极差。"""
    q, pos = [], []
    for line in (seq / "hmd_frames.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h = json.loads(line).get("hmd") or {}
        if h.get("rotation_xyzw") and h.get("position_xyz"):
            q.append(h["rotation_xyzw"])
            pos.append(h["position_xyz"])
    if len(q) < 2:
        return {"n": len(q), "max_angle_deg": 0.0, "y": 0.0, "y_spread_m": 0.0}
    q = np.asarray(q, np.float64)
    pos = np.asarray(pos, np.float64)
    from scipy.spatial.transform import Rotation
    r0 = Rotation.from_quat(q[0])
    ang = np.degrees((r0.inv() * Rotation.from_quat(q)).magnitude())
    return {"n": len(q), "max_angle_deg": float(ang.max()), "y": float(np.median(pos[:, 1])),
            "y_spread_m": float(np.ptp(pos[:, 1]))}


def hmd_yaw_at(seq: Path, frame_t: np.ndarray) -> np.ndarray:
    """HMD 解卷绕 yaw（度），按帧时刻取值。头部在转的会话要靠它把每步旋进同一帧。"""
    y = HmdYaw(str(seq))
    return np.interp(frame_t, y.t, y.yaw)


def camh_from_points(pts: np.ndarray, band: tuple[float, float] = (0.8, 2.4),
                     dmin: float = 1.2, dmax: float = 4.5) -> float | None:
    """一帧双目点里量"地面在相机下方多远"：地面是横贯的一片，给 1 cm 直方图的峰。

    墙是竖的，会摊到整条带上；地面给出尖锐的峰。
    """
    if len(pts) < 200:
        return None
    h = -pts[:, 2]                      # base z 向上，地面在相机下方 ⇒ −z 就是离地高
    d = np.hypot(pts[:, 0], pts[:, 1])
    sel = (h > band[0]) & (h < band[1]) & (d > dmin) & (d < dmax)
    if int(sel.sum()) < 100:
        return None
    edges = np.arange(band[0], band[1] + 1e-9, 0.01)
    hist, _ = np.histogram(h[sel], bins=edges)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def load_sequence(seq: Path) -> tuple[np.ndarray, list[Path], list[Path]]:
    """times.txt（每行一个 ns）→ (帧秒, cam0 路径, cam1 路径)。文件名就是那个 ns。"""
    stamps = [int(s) for s in (seq / "times.txt").read_text(encoding="ascii").split()]
    cam0 = seq / "mav0" / "cam0" / "data"
    cam1 = seq / "mav0" / "cam1" / "data"
    return (np.array(stamps, np.float64) / 1e9,
            [cam0 / f"{s}.png" for s in stamps],
            [cam1 / f"{s}.png" for s in stamps])


# ---------------------------------------------------------------------------
# 特征：ORB 只提一次，深度上限在装配时才用（扫描要反复换它）
# ---------------------------------------------------------------------------

def extract_orb_depth(left_gray: np.ndarray, disp: np.ndarray, orb, baseline_m: float) -> dict:
    """左目 ORB + 关键点处的视差（**这一步与深度上限无关**，所以可以只算一次）。"""
    h, w = left_gray.shape[:2]
    kps, des = orb.detectAndCompute(left_gray, None)
    if des is None or not kps:
        return {"uv": np.zeros((0, 2), np.float32), "des": np.zeros((0, 32), np.uint8),
                "d": np.zeros(0, np.float32), "size": (w, h), "eye_y": baseline_m / 2.0}
    uv = np.array([k.pt for k in kps], np.float32)
    ui = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    return {"uv": uv, "des": des, "d": disp[vi, ui].astype(np.float32),
            "size": (w, h), "eye_y": baseline_m / 2.0}


def build_features(rec: dict, *, fx: float, cx: float, cy: float, baseline_m: float,
                   max_depth_m: float) -> KeyframeFeatures:
    """按 ``max_depth_m`` 装配 ``KeyframeFeatures``。

    这里是 ``nav_loop.extract_features`` 的 xyz 那半段的**同式复刻**（它把 ORB 与深度上限
    绑在一起算了，重跑一次要重算 ORB + SGBM）。刻意不改生产代码：复刻 6 行比给生产函数
    加一个只给探针用的分叉更安全。改动那 6 行时**必须同步这里**。
    """
    uv, des = rec["uv"], rec["des"]
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    ok = rec["d"] > 1.0
    z = np.where(ok, fx * baseline_m / np.maximum(rec["d"], 1e-6), np.inf)
    ok = ok & (z <= max_depth_m)
    X = np.column_stack([(uv[ok, 0] - cx) * z[ok] / fx, (uv[ok, 1] - cy) * z[ok] / fx, z[ok]])
    P = X @ M_OPT.T + np.array([0.0, baseline_m / 2.0, 0.0])
    return KeyframeFeatures(uv, des, P.astype(np.float32), des[ok], K, rec["size"], rec["eye_y"])


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def pca_range(P: np.ndarray) -> tuple[float, float, np.ndarray]:
    """二维点集的主轴极差、次轴极差、主轴单位向量。直线运动时次轴极差 ≈ 0。"""
    C = P - P.mean(axis=0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    return float(np.ptp(C @ vt[0])), float(np.ptp(C @ vt[1])), vt[0]


def slope_ci(y: np.ndarray, x: np.ndarray, seed: int = 20261005, n: int = 2000
             ) -> tuple[float, float]:
    """最小二乘斜率的 bootstrap 95% CI（重采样"段"）。"""
    rng = np.random.default_rng(seed)
    got = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(x), len(x))
        got[i] = np.polyfit(x[j], y[j], 1)[0]
    return float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5))


def analyse(kfs: list[KeyframeFeatures], idx: list[int], cfg: LoopConfig,
            frame_t: np.ndarray, dr_x: np.ndarray, dr_z: np.ndarray,
            valid: np.ndarray, yaw_deg: np.ndarray, yaw_sign: float,
            rot_tol: float) -> dict:
    """一趟逐段 PnP + 统计。返回 dict（含 rows，供主配置细节展示与落盘）。

    ``dr_x``/``dr_z`` 是**已按 world_scale 折到几何单位**的世界坐标。
    """
    rows: list[dict] = []
    rejects: dict[str, int] = {}
    for m in range(len(idx) - 1):
        a, b = idx[m], idx[m + 1]
        # OSC 覆盖窗口之外的帧，DR 位姿被 interp 夹住（看起来站着不动），不能进统计。
        if not (valid[a] and valid[b]):
            rejects["dr_out_of_window"] = rejects.get("dr_out_of_window", 0) + 1
            continue
        rel, why = relative_pose(kfs[m], kfs[m + 1], cfg)
        d_dr = float(np.hypot(dr_x[b] - dr_x[a], dr_z[b] - dr_z[a]))
        if rel is None:
            rejects[why] = rejects.get(why, 0) + 1
            rows.append({"i": a, "j": b, "ok": False, "why": why})
            continue
        rot_deg = rotation_angle_deg(rel["R_ab"])
        # PnP 的相对旋转必须与 HMD 的相对 yaw 一致（HMD 朝向不漂，nav_loop._verify 的同一道门）。
        # 头部不转的会话 Δyaw ≈ 0，退回成"R 必须是恒等"，与旧行为逐位相同。
        d_yaw = abs(float(yaw_deg[b] - yaw_deg[a]))
        if abs(rot_deg - d_yaw) > rot_tol:
            rejects["rotation"] = rejects.get("rotation", 0) + 1
            rows.append({"i": a, "j": b, "ok": False, "why": "rotation",
                         "rot_deg": rot_deg, "d_yaw_deg": d_yaw})
            continue
        rows.append({"i": a, "j": b, "t": float(frame_t[b]),
                     "dt": float(frame_t[b] - frame_t[a]), "ok": True,
                     "d_geo_m": float(np.linalg.norm(rel["t_ab"])), "d_dr_m": d_dr,
                     "rot_deg": rot_deg, "d_yaw_deg": d_yaw, "inliers": rel["inliers"],
                     "coverage": rel["coverage"], "reproj_px": rel["reproj_px"],
                     "t_ab": [float(v) for v in rel["t_ab"]],
                     "R_ab": [[float(v) for v in r] for r in rel["R_ab"]]})

    ok = [r for r in rows if r["ok"]]
    res: dict = {"n_ok": len(ok), "n_reject": len(rows) - len(ok), "rejects": rejects,
                 "rows": rows}
    if len(ok) < 8:
        return res
    # 几何步矢量在 base 系（x 前, y 左）；DR 世界步矢量（x_w, z_w，几何单位）。
    gf = np.array([r["t_ab"][0] for r in ok])
    gl = np.array([r["t_ab"][1] for r in ok])
    df = np.array([dr_z[r["j"]] - dr_z[r["i"]] for r in ok])
    dl = np.array([-(dr_x[r["j"]] - dr_x[r["i"]]) for r in ok])
    # ★ 主判据：已知对应的相似变换（Umeyama）。输入是**旋进世界系之后**的几何步：
    #   base 本地位移 (右, 前) = (−t_ab[1], t_ab[0])，用与 DR 完全相同的 yaw 旋（同一个
    #   pose_math.advance）。头部在转的会话靠这一步才可比；yaw 恒 0 时会退化成原样。
    #   ⇒ yaw_sign 取正负对**两条链是同一个旋转**，所以判据对符号不变。
    gw = np.array([advance(0.0, 0.0, -gl[n], gf[n],
                           yaw_radians(yaw_deg[r["i"]], yaw_sign=yaw_sign))
                   for n, r in enumerate(ok)])
    Dw = np.column_stack([[dr_x[r["j"]] - dr_x[r["i"]] for r in ok],
                          [dr_z[r["j"]] - dr_z[r["i"]] for r in ok]])
    s_sim, R_sim, t_sim = umeyama2d(gw, Dw, True)
    pred = (s_sim * (R_sim @ gw.T)).T + t_sim
    res_sim = float(np.sqrt(((pred - Dw) ** 2).sum(axis=1).mean()))
    res0 = float(np.sqrt(((Dw - Dw.mean(axis=0)) ** 2).sum(axis=1).mean()))
    # 几何步自身的 RMS —— 信噪比的**分母**。用它而不是 DR 轨迹的尺度：原地转身多的会话
    # DR 步很小，拿轨迹尺度当分母会把"走得少"误判成"形状对不上"。
    g_rms = float(np.sqrt((gw ** 2).sum(axis=1).mean()))
    rot_deg = float(np.degrees(np.arctan2(R_sim[1, 0], R_sim[0, 0])))
    rng = np.random.default_rng(20261005)
    bs = np.empty(2000)
    for i in range(2000):
        j = rng.integers(0, len(gw), len(gw))
        bs[i] = umeyama2d(gw[j], Dw[j], True)[0]
    s_lo, s_hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    res.update({"s_sim": float(s_sim), "s_sim_ci95": [s_lo, s_hi],
                "sim_rot_deg": rot_deg, "sim_resid_m": res_sim, "sim_resid_norm_m": res0,
                "geo_step_rms_m": g_rms})
    # 分量回归：只是"头部不转"会话上的诊断量（那时 DR 侧向恒 0，左移应当是纯噪声）。
    kf, cf = np.polyfit(gf, df, 1)
    kl, cl = np.polyfit(gl, dl, 1)
    rf = float(np.corrcoef(gf, df)[0, 1])
    rl = float(np.corrcoef(gl, dl)[0, 1])
    resf = df - (kf * gf + cf)
    lo, hi = slope_ci(df, gf)
    still = np.abs(df) < 0.1
    res.update({"k": float(kf), "k_ci95": [lo, hi], "c": float(cf), "r": rf,
                "k_lat": float(kl), "r_lat": rl, "resid_std_m": float(resf.std()),
                "resid_max_m": float(np.abs(resf).max()),
                "n_static": int(still.sum()),
                "static_geo_mean": float(gf[still].mean()) if still.any() else 0.0,
                "geo_path_m": float(sum(r["d_geo_m"] for r in ok)),
                "dr_path_m": float(sum(r["d_dr_m"] for r in ok))})
    return res


# ---------------------------------------------------------------------------
# 生产录制（navmesh_recordings/<id>/kf/*.npz）：**真实随机行走**上的同一判据
# ---------------------------------------------------------------------------

def load_recorded(rec: Path, skip: int) -> tuple[list[KeyframeFeatures], np.ndarray, np.ndarray]:
    """``kf/*.npz`` 每 ``skip`` 个取一个 → (KeyframeFeatures, T_dr 4×4, dist_m)。

    npz 存的正是 ``KeyframeFeatures`` 的全部字段 + ``T_dr`` + ``dist_m``（见 `loop_replay.py`），
    而 ``T_dr[:3,:3]`` 是**该关键帧的 HMD 朝向（地图系）**、``T_dr[:2,3]`` 是航位推算平面位置
    （追踪米）—— 于是不需要图像、不需要重录，直接在真实游玩录制上跑同一套判据。
    """
    files = sorted((rec / "kf").glob("*.npz"), key=lambda p: int(p.stem))[::skip]
    kfs, mats, dist = [], [], []
    for f in files:
        z = np.load(f)
        kfs.append(KeyframeFeatures(z["uv"], z["des"], z["xyz"], z["des3d"], z["K"],
                                    (int(z["size"][0]), int(z["size"][1])), float(z["eye_y"])))
        mats.append(np.asarray(z["T_dr"], np.float64))
        dist.append(float(z["dist_m"]))
    return kfs, np.array(mats), np.array(dist)


def analyse_recorded(kfs: list[KeyframeFeatures], T: np.ndarray, cfg: LoopConfig,
                     rot_tol: float) -> dict:
    """真实录制：几何步 = ``R_a·t_ab``（地图系），DR 步 = ``T_dr`` 平面差，两者同系同单位。"""
    rows: list[dict] = []
    rejects: dict[str, int] = {}
    yaw = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
    for m in range(len(kfs) - 1):
        rel, why = relative_pose(kfs[m], kfs[m + 1], cfg)
        if rel is None:
            rejects[why] = rejects.get(why, 0) + 1
            rows.append({"i": m, "j": m + 1, "ok": False, "why": why})
            continue
        rot_deg = rotation_angle_deg(rel["R_ab"])
        d_yaw = abs(float(np.degrees(yaw[m + 1] - yaw[m])))
        if abs(rot_deg - d_yaw) > rot_tol:
            rejects["rotation"] = rejects.get("rotation", 0) + 1
            rows.append({"i": m, "j": m + 1, "ok": False, "why": "rotation"})
            continue
        gw = (T[m, :3, :3] @ np.asarray(rel["t_ab"], np.float64))[:2]
        dw = T[m + 1, :2, 3] - T[m, :2, 3]
        rows.append({"i": m, "j": m + 1, "ok": True,
                     "d_geo_m": float(np.linalg.norm(rel["t_ab"])),
                     "d_dr_m": float(np.linalg.norm(dw)),
                     "rot_deg": rot_deg, "d_yaw_deg": d_yaw,
                     "inliers": rel["inliers"], "reproj_px": rel["reproj_px"],
                     "gw": [float(v) for v in gw], "dw": [float(v) for v in dw]})
    ok = [r for r in rows if r["ok"]]
    res: dict = {"n_ok": len(ok), "n_reject": len(rows) - len(ok), "rejects": rejects,
                 "rows": rows}
    if len(ok) < 8:
        return res
    gw = np.array([r["gw"] for r in ok])
    Dw = np.array([r["dw"] for r in ok])
    s_sim, R_sim, _ = umeyama2d(gw, Dw, True)
    pred = (s_sim * (R_sim @ gw.T)).T
    res_sim = float(np.sqrt(((pred - Dw) ** 2).sum(axis=1).mean()))
    rng = np.random.default_rng(20261005)
    bs = np.array([umeyama2d(gw[j], Dw[j], True)[0]
                   for j in (rng.integers(0, len(gw), len(gw)) for _ in range(2000))])
    s_lo, s_hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    # 每帧都是**真实位移**，不像 run5 有大量静止段，所以用全部段算信噪比。
    g_rms = float(np.sqrt((gw ** 2).sum(axis=1).mean()))
    # 分段 s：按时间顺序切 4 段各算一个 s。**稳定 ⇒ 系统性的尺度偏差；乱跳 ⇒ 噪声。**
    # 这是"要不要去改 world_scale"的关键判据：尺度偏差必须在会话内稳定才值得修。
    n = len(gw)
    seg_s = []
    for i in range(4):
        lo_i, hi_i = i * n // 4, (i + 1) * n // 4
        if hi_i - lo_i >= 5:
            seg_s.append(umeyama2d(gw[lo_i:hi_i], Dw[lo_i:hi_i], True)[0])
    res.update({"s_sim": float(s_sim), "s_sim_ci95": [s_lo, s_hi],
                "sim_rot_deg": float(np.degrees(np.arctan2(R_sim[1, 0], R_sim[0, 0]))),
                "sim_resid_m": res_sim, "geo_step_rms_m": g_rms, "seg_s": seg_s,
                "geo_path_m": float(sum(r["d_geo_m"] for r in ok)),
                "dr_path_m": float(sum(r["d_dr_m"] for r in ok))})
    return res


def chain(traj_rows: list[dict], use_pnp_rot: bool) -> tuple[float, float, float]:
    """串链轨迹 → (主轴极差, 次轴极差, 直线偏离 rms)。累积误差会污染，只作参考。"""
    p, R = np.zeros(3), np.eye(3)
    pts = [p.copy()]
    for r in traj_rows:
        if r["ok"]:
            p = p + R @ np.array(r["t_ab"])
            if use_pnp_rot:
                R = R @ np.array(r["R_ab"])
        pts.append(p.copy())
    tr = np.array(pts)[:, [0, 1]]
    gr, gm, _ = pca_range(tr)
    C = tr - tr.mean(axis=0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    return gr, gm, float(np.sqrt(np.mean((C @ vt[1]) ** 2)))


# ---------------------------------------------------------------------------

def main_recorded(seq: Path, args: argparse.Namespace) -> int:
    """生产录制模式：``navmesh_recordings/<id>/kf/*.npz``，**真实随机行走**，不需要图像。"""
    names = sorted(int(p.stem) for p in (seq / "kf").glob("*.npz"))
    meta = json.loads((seq / "meta.json").read_text(encoding="utf-8")) if \
        (seq / "meta.json").exists() else {}
    print(f"生产录制 {seq.name}：{len(names)} 个关键帧"
          + (f"，world_scale={meta['config']['mapper']['world_scale']:g}"
             if meta.get("config", {}).get("mapper", {}).get("world_scale") else ""))
    print("  几何步 = R_a·t_ab（R_a 取自 T_dr 的 HMD 朝向，不漂）vs DR 步 = T_dr 平面差；"
          "两者同为追踪米，所以 s=1 即「world_scale 取值正确」。")
    print("  ⚠️ 跳帧只放大两条链共同的弦，不引入漂移：HMD 朝向不漂，DR 只在这一段内积分。")

    def run(skip: int) -> dict:
        kfs, T, dist = load_recorded(seq, skip)
        r = analyse_recorded(kfs, T, LoopConfig(min_inliers=int(args.min_inliers)),
                             args.rot_tol_deg)
        step = np.linalg.norm(np.diff(T[:, :2, 3], axis=0), axis=1)
        r.update({"skip": skip, "n_kf": len(kfs),
                  "path_m": float(dist[-1] - dist[0]) if len(dist) else 0.0,
                  "step_med_m": float(np.median(step)) if step.size else 0.0,
                  "step_rms_m": float(np.sqrt((step ** 2).mean())) if step.size else 0.0,
                  "step_p10_m": float(np.percentile(step, 10)) if step.size else 0.0,
                  "step_p90_m": float(np.percentile(step, 90)) if step.size else 0.0})
        return r

    plans = [int(s) for s in args.skip.split(",")] if args.skip else [1]
    results = [run(s) for s in plans]
    prim = results[0]
    print(f"\n=== 生产录制扫描（真实随机行走；s = 几何→DR 的相似变换尺度）===")
    print(f"  {'跳帧':>5} {'关键帧':>6} {'接受':>5} {'DR路程m':>8} {'DR步 p10/中位/p90':>19} {'步RMSm':>7} "
          f"{'s 相似':>8} {'95% CI':>18} {'含1?':>5} {'残差m':>7} {'信噪比':>7}")
    for r in results:
        if "s_sim" not in r:
            print(f"  {r['skip']:5d} {r['n_kf']:6d} {r['n_ok']:5d}   样本不足")
            continue
        lo, hi = r["s_sim_ci95"]
        snr = r["sim_resid_m"] / max(r["geo_step_rms_m"], 1e-9)
        print(f"  {r['skip']:5d} {r['n_kf']:6d} {r['n_ok']:5d} {r['path_m']:8.1f} "
              f"{r['step_p10_m']:6.2f}/{r['step_med_m']:5.2f}/{r['step_p90_m']:5.2f}   "
              f"{r['step_rms_m']:7.2f} {r['s_sim']:8.4f} "
              f"[{lo:7.4f}, {hi:7.4f}] {'是' if lo <= 1.0 <= hi else '**否**':>5} "
              f"{r['sim_resid_m']:7.3f} {snr:7.3f}")
    ss = [r["s_sim"] for r in results if "s_sim" in r]
    if ss:
        print(f"  s 的范围 {min(ss):.4f}–{max(ss):.4f}"
              + ("；全部含 1 ⇒ 与「两链同尺」不可区分" if all(
                  r["s_sim_ci95"][0] <= 1.0 <= r["s_sim_ci95"][1]
                  for r in results if "s_sim" in r) else "；**有配置显著偏离 1**"))
    for r in results:
        if "s_sim" in r:
            g = r["sim_resid_m"] / max(r["geo_step_rms_m"], 1e-9)
            print(f"  跳帧 {r['skip']}：拒因 {r['rejects']}；信噪比 {g:.3f}"
                  + ("" if g < 0.4 else "  ⚠️ ≥0.4，步子太小"))
    for r in results:
        if r.get("seg_s"):
            seg = np.array(r["seg_s"])
            print(f"  跳帧 {r['skip']} 分段 s（按关键帧顺序切 {len(seg)} 段）："
                  + "  ".join(f"{v:.4f}" for v in seg)
                  + f"   极差 {seg.max() - seg.min():.4f}"
                  + ("  ⇒ 稳定，是**系统性尺度偏差**" if seg.max() - seg.min() < 0.06
                     else "  ⇒ 段间摆动大，更像噪声"))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"rec": seq.name, "mode": "recorded",
             "sweep": [{"skip": r["skip"], "n_kf": r["n_kf"], "n_ok": r["n_ok"],
                        "path_m": r["path_m"], "step_med_m": r["step_med_m"],
                        "step_rms_m": r["step_rms_m"], "rejects": r["rejects"],
                        **{k: r[k] for k in ("s_sim", "s_sim_ci95", "seg_s", "sim_resid_m",
                                             "geo_step_rms_m", "sim_rot_deg",
                                             "geo_path_m", "dr_path_m") if k in r}}
                       for r in results]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {args.json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("seq", type=Path)
    ap.add_argument("--stride", type=int, default=3,
                    help="基础抽帧间隔（ORB/SGBM 只在这一层算一次）")
    ap.add_argument("--sweep", default=DEFAULT_SWEEP,
                    help="敏感性扫描：逗号分隔的 子步长:深度上限 组合；第一个是主配置")
    ap.add_argument("--skip", default="",
                    help="**生产录制模式**的扫描：逗号分隔的跳帧数（如 1,2,5,10,20）")
    ap.add_argument("--max-frames", type=int, default=0, help="只跑前 N 个基础网格点（调试）")
    ap.add_argument("--yaw-sign", type=float, default=+1.0,
                    help="build_ground_truth 的 HMD yaw 符号。判据对它是**不变的**（两条链用"
                         "同一个旋转），它只影响 DR 轨迹自身的形状；0 = 两个都跑一遍做对照")
    ap.add_argument("--osc-offset", type=float, default=0.13)
    ap.add_argument("--world-scale", type=float, default=WORLD_SCALE)
    ap.add_argument("--min-inliers", type=int, default=50)
    ap.add_argument("--rot-tol-deg", type=float, default=6.0)
    ap.add_argument("--min-chord-m", type=float, default=0.5,
                    help="弦长分档表只统计几何弦 ≥ 这个值的段")
    ap.add_argument("--camh-every", type=int, default=10)
    ap.add_argument("--verbose", action="store_true", help="逐段打印主配置")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    seq = args.seq
    if (seq / "kf").is_dir():                 # 生产录制：真实随机行走，无需图像
        return main_recorded(seq, args)
    meta = json.loads((seq / "meta.json").read_text(encoding="utf-8"))
    fx = float(meta["fx"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    baseline = float(meta["baseline_tracking_m"])
    frame_t, cam0, cam1 = load_sequence(seq)
    n = len(frame_t)
    print(f"序列 {seq.name}：{n} 帧 / {frame_t[-1] - frame_t[0]:.1f} s，"
          f"fx={fx:g} 基线={baseline:g}，world_scale={args.world_scale:g}")

    # ---- 体检①：头部朝向是否真的不变（(前, 左) 分解的前提）----
    hm = head_motion(seq)
    print(f"HMD：{hm['n']} 样本，相对首样本最大转角 {hm['max_angle_deg']:.2f}°；"
          f"y 中位 {hm['y']:.3f}（极差 {hm['y_spread_m']:.3f}）")
    head_ok = hm["max_angle_deg"] <= 2.0
    if not head_ok:
        print("  ⚠️ 头部在转：每步的几何位移要先按 yaw 旋进同一帧才可比（本工具会做），"
              "但「前进分量」那条**轴分解**不成立，只能看 s 相似列。")

    # ---- DR 链（世界米 → 折回几何单位）----
    # yaw 符号：判据对它是**不变的**（两条链用同一个旋转），所以默认只跑一个；
    # --yaw-sign 0 时两个都建，只为看 DR 轨迹形状的差别。
    signs = [float(args.yaw_sign)] if args.yaw_sign else [1.0, -1.0]
    drs: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sg in signs:
        g = build_ground_truth(seq, frame_t, sg, args.osc_offset)
        drs[sg] = (g["x"] / args.world_scale, g["z"] / args.world_scale,
                   np.asarray(g["valid"]))
    g0 = build_ground_truth(seq, frame_t, signs[0], args.osc_offset)
    print(f"DR：OSC 报文 {g0['osc_packets']} 条，有效窗口 "
          f"[{g0['window'][0]:.2f}, {g0['window'][1]:.2f}] s，"
          f"有效帧 {int(g0['valid'].sum())}/{n}")
    yaw_deg = hmd_yaw_at(seq, frame_t)
    print(f"yaw 序列：{yaw_deg.min():+.1f} ~ {yaw_deg.max():+.1f}°"
          f"（按帧时刻取值，与 build_ground_truth 同一把）")

    # ---- 抽帧：ORB + SGBM 只算一次 ----
    cfg = LoopConfig(min_inliers=int(args.min_inliers))
    orb = cv2.ORB_create(nfeatures=cfg.orb_features)
    matcher = make_sgbm()
    base_idx = list(range(0, n, max(1, args.stride)))
    if args.max_frames:
        base_idx = base_idx[:args.max_frames]
    print(f"基础网格 stride={args.stride} → {len(base_idx)} 帧，逐帧 ORB + SGBM 只算一次", flush=True)
    recs: list[dict] = []
    camh: list[float] = []
    blank = 0
    for c, k in enumerate(base_idx):
        img_l = cv2.imread(str(cam0[k]), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(cam1[k]), cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            raise SystemExit(f"读不到图像：{cam0[k]} / {cam1[k]}")
        if not img_l.any() or not img_r.any():
            blank += 1
        disp = stereo_disparity(img_l, img_r, matcher)
        recs.append(extract_orb_depth(img_l, disp, orb, baseline))
        if c % args.camh_every == 0:
            v = camh_from_points(stereo_points(img_l, img_r, fx=fx, cx=cx, cy=cy,
                                               baseline_m=baseline, max_range_m=6.0,
                                               step=2, disp=disp))
            if v is not None:
                camh.append(v)
        if c % 60 == 0:
            print(f"  抽帧 {c}/{len(base_idx)}", flush=True)
    print(f"抽帧完成（全黑帧 {blank}）", flush=True)

    # ---- 体检②：双目前置处理与生产链自洽（**不是米制真值**）----
    if camh:
        print(f"体检②：双目测得地面在相机下方 中位 {np.median(camh):.3f}（{len(camh)} 帧）"
              f" ⇒ 与 mapper 实测的 cam_h≈1.73 一致即说明前置处理自洽")
        print("        ⚠️ 不拿 HMD 遥测的 y 当地面真值：那是虚拟驱动声明的常量"
              f"（本 run 极差 {hm['y_spread_m']:.3f}），跟踪原点在不在世界地面没验证过。")

    sweep: list[tuple[int, float]] = []
    for item in args.sweep.split(","):
        mult, depth = item.split(":")
        sweep.append((int(mult), float(depth)))
    if not sweep:
        return 2

    def feats_for(mult: int, depth: float) -> tuple[list[KeyframeFeatures], list[int]]:
        kfs = [build_features(recs[i], fx=fx, cx=cx, cy=cy, baseline_m=baseline,
                              max_depth_m=depth) for i in range(0, len(recs), mult)]
        return kfs, base_idx[::mult]

    # ---- yaw 符号：判据本身对它是**不变的**（几何与 DR 用同一个 `advance(yaw)`）。
    # 默认只跑一个；`--yaw-sign 0` 时才两个都跑，只为看 DR 轨迹形状差多少。
    kfs0, idx0 = feats_for(*sweep[0])
    sign_trials = {sg: analyse(kfs0, idx0, cfg, frame_t, drs[sg][0], drs[sg][1],
                               drs[sg][2], yaw_deg, sg, args.rot_tol_deg) for sg in signs}
    def _norm_res(r: dict) -> float:
        return r.get("sim_resid_m", 1e9) / max(r.get("sim_resid_norm_m", 1e9), 1e-9)

    yaw_sign = min(signs, key=lambda s: _norm_res(sign_trials[s]))
    if len(signs) > 1:
        for sg in signs:
            t = sign_trials[sg]
            print(f"yaw 符号 {sg:+.0f}：相似变换残差 {t.get('sim_resid_m', float('nan')):.3f} m "
                  f"/ 轨迹尺度 {t.get('sim_resid_norm_m', float('nan')):.3f} m "
                  f"= {_norm_res(t):.3f}" + ("   ← 采用" if sg == yaw_sign else ""))
    dr_x, dr_z, valid = drs[yaw_sign]

    results: list[dict] = []
    for i, (mult, depth) in enumerate(sweep):
        if i == 0:
            r, idx = dict(sign_trials[yaw_sign]), idx0
        else:
            kfs, idx = feats_for(mult, depth)
            r = analyse(kfs, idx, cfg, frame_t, dr_x, dr_z, valid, yaw_deg, yaw_sign,
                        args.rot_tol_deg)
        r.update({"mult": mult, "depth_m": depth, "n_kf": len(idx),
                  "dt_s": float(np.median(np.diff(frame_t[idx]))) if len(idx) > 1 else 0.0})
        results.append(r)

    prim = results[0]
    print(f"\n=== 敏感性扫描（s = 几何→DR 的相似变换尺度；s=1 即两链同尺）===")
    print(f"  {'子步长':>6} {'帧间隔s':>8} {'深度上限':>9} {'关键帧':>6} {'接受':>5} "
          f"{'s 相似':>8} {'95% CI':>18} {'含1?':>5} {'残差m':>7} {'步RMS':>7} {'信噪比':>7} "
          f"{'k 前向':>8} {'r 前向':>7}")
    for r in results:
        if "s_sim" not in r:
            print(f"  {r['mult']:6d} {r['dt_s']:8.2f} {r['depth_m']:9.1f} {r['n_kf']:6d} "
                  f"{r['n_ok']:5d}   样本不足")
            continue
        lo, hi = r["s_sim_ci95"]
        snr = r["sim_resid_m"] / max(r["geo_step_rms_m"], 1e-9)
        print(f"  {r['mult']:6d} {r['dt_s']:8.2f} {r['depth_m']:9.1f} {r['n_kf']:6d} "
              f"{r['n_ok']:5d} {r['s_sim']:8.4f} [{lo:7.4f}, {hi:7.4f}] "
              f"{'是' if lo <= 1.0 <= hi else '**否**':>5} {r['sim_resid_m']:7.3f} "
              f"{r['geo_step_rms_m']:7.3f} {snr:7.3f} {r['k']:8.4f} {r['r']:+7.3f}")
    ss = [r["s_sim"] for r in results if "s_sim" in r]
    if len(ss) >= 2:
        print(f"  s 的极差 {max(ss) - min(ss):.4f}（{min(ss):.4f}–{max(ss):.4f}）"
              f" —— 摆动大 ⇒ 这个差是**分析参数噪声**，不是稳定的尺度误差")
    # 质量闸：每步残差 ÷ 几何步 RMS。跑过三场后实测：**绝对残差都在 0.19–0.25 m**，
    # 与步长基本无关（是 PnP 的固有噪声地板）。所以能不能读出尺度，取决于**步子够不够大**：
    # run5_ipd126 步 RMS 0.79 m ⇒ 信噪比 0.27；run6/run7 以原地转身为主、步 RMS 0.46/0.49 m
    # ⇒ 0.42/0.46，s 的 CI 立刻宽到 ±15–20%，只能算"与 1 不可区分"。
    q = prim.get("sim_resid_m", 0.0) / max(prim.get("geo_step_rms_m", 1e-9), 1e-9)
    print(f"  质量闸：每步残差 ÷ 几何步 RMS = {q:.3f}"
          + ("（<0.4，够读出几个百分点）" if q < 0.4 else
             " ⚠️ **≥0.4：步子太小、信噪比不足** ⇒ s 只能当「与 1 不可区分」，"
             "**不能给尺度界**（绝对残差 ~0.2 m 与步长无关，出行程短的会话必然如此）"))
    print(f"  挑中的 yaw 符号 {yaw_sign:+.0f}；头部最大转角 {hm['max_angle_deg']:.2f}°"
          + ("（k/r 两列只在朝向不变时有意义）" if head_ok else
             " ⇒ 头部在转，**只看 s 相似列**，k/r 两列作废"))

    if "s_sim" in prim:
        print(f"\n=== 主配置（子步长 {prim['mult']}、深度上限 {prim['depth_m']:g} m）细节 ===")
        print(f"  接受 {prim['n_ok']}/{prim['n_ok'] + prim['n_reject']}，拒绝 {prim['rejects']}")
        print(f"  ★ 相似变换：s = {prim['s_sim']:.4f} "
              f"CI [{prim['s_sim_ci95'][0]:.4f}, {prim['s_sim_ci95'][1]:.4f}]；"
              f"拟合旋转 {prim['sim_rot_deg']:+.2f}°；残差 {prim['sim_resid_m']:.3f} m "
              f"（无视几何链的基线残差 {prim['sim_resid_norm_m']:.3f} m）")
        print(f"     ⚠️ s 是**下界**：几何噪声会被 Umeyama 往 0 拉（收缩 ≈ r² 倍）")
        print(f"  前进分量（仅朝向不变时有意义）：r = {prim['r']:+.3f}  "
              f"DR = {prim['k']:.4f}·几何 {prim['c']:+.3f}  "
              f"CI [{prim['k_ci95'][0]:.4f}, {prim['k_ci95'][1]:.4f}]")
        print(f"  左移分量：r = {prim['r_lat']:+.3f}  DR = {prim['k_lat']:.4f}·几何"
              f"   ← 朝向不变时 DR 侧向恒 0，几何侧向是噪声（不进上面的斜率）")
        print(f"  前进残差 std {prim['resid_std_m']:.3f} m、最大 {prim['resid_max_m']:.3f} m；"
              f"DR 静止段 {prim['n_static']} 段上几何前进均值 {prim['static_geo_mean']:+.4f}（应为 0）")
        print(f"  若几何无偏，本 run 反推的 world_scale = {args.world_scale:g} × "
              f"{prim['s_sim']:.4f} = **{args.world_scale * prim['s_sim']:.4f}**")
        print(f"  弦长合计：几何 {prim['geo_path_m']:.2f}，DR {prim['dr_path_m']:.2f}，"
              f"比 {prim['dr_path_m'] / prim['geo_path_m']:.4f}")
        mov = [r for r in prim["rows"] if r["ok"] and r["d_geo_m"] >= args.min_chord_m]
        edges = [args.min_chord_m, 1.0, 2.0, 4.0, 1e9]
        print(f"  按几何弦长分档（只列 ≥{args.min_chord_m:g} m）")
        print(f"    {'几何弦':>14} {'段数':>5} {'s 中位':>9} {'IQR 宽':>8} {'内点中位':>9}")
        for a0, b0 in zip(edges[:-1], edges[1:]):
            m = [r for r in mov if a0 <= r["d_geo_m"] < b0]
            if len(m) < 5:
                continue
            sss = np.array([r["d_dr_m"] / r["d_geo_m"] for r in m])
            tag = f"{a0:4.1f}–{b0:5.1f} m" if b0 < 1e8 else f">{a0:4.1f} m"
            print(f"    {tag:>14} {len(m):5d} {np.median(sss):9.4f} "
                  f"{np.percentile(sss, 75) - np.percentile(sss, 25):8.4f} "
                  f"{np.median([r['inliers'] for r in m]):9.0f}")
        gr, gm, cres = chain(prim["rows"], True)
        gr2, gm2, cres2 = chain(prim["rows"], False)
        drv = np.array([k for k in base_idx if valid[k]])
        drR, drM, _ = pca_range(np.column_stack([dr_x[drv], dr_z[drv]]))
        print(f"  串链主轴极差（累积误差会污染，只作参考）：DR {drR:.2f}"
              f"（次轴 {drM:.2f}）；PnP 旋转串链 {gr:.2f}（次轴 {gm:.2f}，直线偏离 {cres:.3f}）；"
              f"恒等旋转串链 {gr2:.2f}（次轴 {gm2:.2f}，直线偏离 {cres2:.3f}）")

        if args.verbose:
            print("\n  逐段：")
            for r in prim["rows"]:
                if r["ok"]:
                    print(f"    {r['i']:4d}→{r['j']:4d} t={r['t']:6.1f} dt={r['dt']:.2f} "
                          f"几何 {r['d_geo_m']:6.3f} DR {r['d_dr_m']:6.3f} "
                          f"s={r['d_dr_m'] / r['d_geo_m']:8.4f} rot={r['rot_deg']:5.2f}° "
                          f"inl={r['inliers']:4d} cov={r['coverage']:.2f} rp={r['reproj_px']:.2f}")
                else:
                    print(f"    {r['i']:4d}→{r['j']:4d} 拒绝：{r['why']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "seq": seq.name, "stride": args.stride, "sweep": args.sweep,
            "fx": fx, "baseline_tracking_m": baseline, "world_scale": args.world_scale,
            "min_inliers": cfg.min_inliers, "rot_tol_deg": args.rot_tol_deg,
            "min_chord_m": args.min_chord_m, "yaw_sign": yaw_sign,
            "yaw_sign_trials": {f"{sg:+.0f}": {"sim_resid_m": sign_trials[sg].get("sim_resid_m"),
                                               "sim_resid_norm_m":
                                                   sign_trials[sg].get("sim_resid_norm_m")}
                                for sg in signs},
            "osc_offset": args.osc_offset, "blank_frames": blank,
            "head": hm, "head_ok": head_ok,
            "camh_median": float(np.median(camh)) if camh else None,
            "sweep_results": [{k: v for k, v in r.items() if k != "rows"} for r in results],
            "primary_rows": prim.get("rows", []),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
