# -*- coding: utf-8 -*-
"""从原始双目流构建 M0 训练集：5 Hz policy_obs（左目）+ policy_labels（12 步）。

口径全部按 ``Docs/VRChat具身agent工程方案v3（2026-10-27）.md`` §5.2 / §9.2：

* **时间基准**：一帧的真值时刻 = ``run.json.events[].obs_start_monotonic + frames.t``。
  老录制没有锚点时按 ``hmd_frames[0].t - frames[0].t`` 反推并标 ``anchor_source=inferred``。
* **5 Hz 网格**：从首帧 +0.5 s 起，步长 0.2 s；图像取最近的帧（|Δt| ≤ 0.1 s 才算有图）。
* **12 步标签**：第 k 步 = ``osc.jsonl`` 在 ``t_obs + osc_lag + k*0.1`` 的 ZOH 值，
  ``osc_lag_s = 0.18``。末尾超过最后一条 HMD 时刻的步 mask=0（不补零）；
  跨可疑空窗（最后一条速度 > 0.1 m/s 且之后 > 2.5 s 无新消息）的步 mask=0。
  ``t_obs + osc_lag`` 早于第一条 OSC 的整条丢弃。
* **挂机段**：OSC 静默 > 30 s **且** 10 s 内 HMD 姿态最大偏转 < 0.5°，整条丢弃。
* **cls**：``move``（第 0 步 > 0.1 m/s）/ ``idle``（全部有效步 ≤ 0.1）/ ``trans``（其余）。
* **wp**：沿真实轨迹前进 ``lookahead_m = 0.8`` 的点，转到 ``t_obs`` 时刻的机体系 ``(dx, dz)``。
  本录制没有 SLAM 位姿，轨迹用 OSC 速度 + HMD 朝向离线 DR 重建（局部 0.8 m 够用，不做回环）。
* **数据切分**：每场末 20% 标 ``val``（§5.3），其余 ``train``。

图像管线 ``to_policy_rgb()`` 是**训练/推理必须共用的那个函数**：中心裁到 16:9（1080×606 →
1077×606，只裁 3 px）→ 缩放到 W×内容高（16:9）→ 对称补黑边到 W×H。
默认 448×256：内容 448×252 + 上下各 2 行（黑边只占 1.6%）；ffmpeg 侧用
``crop exact=1 + scale flags=area + pad`` 完成同一套几何，Python 只做 JPEG 编码。
推理端必须 import 本函数，不要各写一套。

输出::

    <out>/<sid>/images/<frame_i>_L.jpg      左目 JPEG q90（默认 448×256，按帧号去重）
    <out>/<sid>/policy_obs.jsonl            {"t","t_rel","i","img_L"}
    <out>/<sid>/policy_labels.jsonl         {"t","act":[[vx,vz,omega]×12],"mask":[...],"cls","wp","path","guide"}
    <out>/<sid>/stats.json                  本场统计（含 G3 口径）
    <out>/manifest.json                    汇总

用法::

    .venv/Scripts/python.exe research/tools/build_policy_dataset.py \
        --root G:\\vrc_recording --out H:\\vrc_policy_m0
    # 使用录制时的 A* 软引导（每场需有 astar_guidance.jsonl）：
    .venv/Scripts/python.exe research/tools/build_policy_dataset.py \
        --root G:\\vrc_recording --out H:\\vrc_policy_hybrid --astar-guidance
    # 只验标签、不抽帧（很快）：
    .venv/Scripts/python.exe research/tools/build_policy_dataset.py \
        --root G:\\vrc_recording --out .tmp/policy_ds --no-images --sessions 20261031_020757
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_text import MODE_FOLLOW, MODE_STOP  # noqa: E402
from policy_causal import causal_topo  # noqa: E402  因果 topo（只用历史轨迹）

K_STEPS = 12
STEP_S = 0.1
GRID_S = 0.2
V_STILL = 0.1

# ---- ω（转向）标签 与 [wp] 去泄漏（2026-11-03）----
# ω 用 VRChat 上报的 AngularY：构念上它测的是**身体/胶囊体**，与 VelocityX/Z 同族；
# 不用 HMD 四元数（那是头的朝向，六点追踪下可以转头而不转身体，实测高频能量比 3.76-6.48）。
# 换算：probe_omega_source.py 实测 AngularY ≈ -K·(HMD 偏航角速度)，K 均值 1.434 单位/(度/秒)。
# 符号约定：**ω 正 = 右转**，与 vx（右正）一致；AngularY 与 Y-up 偏航率负相关，故取负。
# ⚠️ 单位不是 deg/s！2026-11-03 用「∫AngularY dt vs HMD 偏航角变化量」实测：
#   K = -0.008953 rad/unit（R² 0.79–0.96，6 个会话一致）
#   即 1 unit = 0.513 度/秒；满量程 ±1024 unit ≈ ±525 度/秒。
# 若单位是 deg/s，K 应为 π/180 = 0.017453 —— 实测只有它的一半。
# 官方文档只写「Angular velocity on the Y axis」，没给单位，所以这一条只能靠实测。
# 负号：AngularY 增大对应 HMD 向左转（Y-up 偏航角减小）。
# 我们要的是「ω 正 = 右转」，所以取负。
ANGULARY_RAD_PER_UNIT = -0.008953

# ω 标签的**源采样频率**（Hz）。与视觉观测 5 Hz、动作标签 10 Hz 是三件不同的事：
#   视觉观测 5 Hz   —— 每 0.2 s 一个关键帧（GRID_S）
#   动作标签 10 Hz  —— 每 0.1 s 一个 (vx, vz, ω) 三元组（K_STEPS x STEP_S）
#   ω 源频率 30 Hz  —— **只是计算 ω 时用的角速度子网格**，不改变上面两者
# 为什么是 30：AngularY 实测 32~50 Hz（10/10 场 >= 30 Hz），
# 而 5 Hz 单点 ZOH 只保留 35.2% 转向事件（.tmp/freq_events_probe.py）。
OMEGA_SOURCE_HZ = 30

# ω 标签的窗口聚合方式：
#   "mean" —— 窗口内 |AngularY| 的均值 × 方向（符号取带符号均值的符号）
#   "peak" —— 窗口内**绝对值最大**的那个采样，保留其自身符号
# 两者都用 30 Hz 子网格 + 因果窗口 [t0, t1)，不读未来。
OMEGA_AGG_MEAN = "mean"
OMEGA_AGG_PEAK = "peak"
OMEGA_AGG_MODES = (OMEGA_AGG_MEAN, OMEGA_AGG_PEAK)
OMEGA_AGG = OMEGA_AGG_MEAN
# 满量程约 1024 unit ≈ 9.17 rad/s；留一点余量，超出视为真正的坏帧。
ANGULARY_FULLSCALE_RAD = 1024 * abs(ANGULARY_RAD_PER_UNIT)
WP_MODE_NEAR = "near"   # 现状：1.2 s 行程内的点 —— 模长等于「未来走多远」，泄漏速度
WP_MODE_FAR = "far"     # 去泄漏：沿轨迹固定 WP_FAR_M 的目标，方向仍给、模长不再等于答案
WP_FAR_M = 4.0
# 因果模式（2026-11-05）：wp 恒 (0,0)，topo 换成只依赖历史的 past_prog/past_lateral。
# 这是部署时真正可得的输入形状 —— 旧 near/far 两个模式都由示教未来轨迹生成。
WP_MODE_CAUSAL = "causal"
GAP_SUSPECT_S = 2.5
IDLE_SILENCE_S = 30.0
HMD_STILL_S = 10.0
HMD_STILL_DEG = 0.5
DEFAULT_LAG_S = 0.18
DEFAULT_LOOKAHEAD_M = 0.8
JPEG_Q = 90
POLICY_SIZE = (448, 256)  # (W, H)：内容 448×252（16:9）+ 上下各 2 行黑边到 448×256


# ---------------------------------------------------------------- 图像口径
def policy_geometry(size: tuple[int, int]) -> tuple[int, int, int]:
    """(W, H) → (内容高, 上边距)：内容按 16:9 取，余下对称补黑。"""
    w, h = int(size[0]), int(size[1])
    content_h = min(h, int(round(w * 9 / 16)))
    content_h -= content_h % 2
    return content_h, (h - content_h) // 2


def to_policy_rgb(rgb: np.ndarray, size: tuple[int, int] = POLICY_SIZE) -> np.ndarray:
    """全分辨率 RGB (H×W×3, uint8) → policy 图（中心裁 16:9、缩放到 W×内容高、对称补黑边）。

    裁剪/缩放几何与 ``extract_images`` 里的 ffmpeg 链一致（ffmpeg 用 flags=area，
    这里用 cv2.INTER_AREA）。推理端必须复用本函数，不要各写一套。
    """
    w, h = int(size[0]), int(size[1])
    content_h, top = policy_geometry(size)
    src_h, src_w = rgb.shape[:2]
    crop_w = int(round(src_h * 16 / 9))
    crop_w = min(crop_w, src_w)
    left = (src_w - crop_w) // 2
    crop = rgb[:, left:left + crop_w]
    interp = cv2.INTER_AREA if src_h >= content_h else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (w, content_h), interpolation=interp)
    out = np.zeros((h, w, 3), np.uint8)
    out[top:top + content_h] = resized
    return out


def encode_jpeg(rgb: np.ndarray, q: int = JPEG_Q) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                          [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


# ---------------------------------------------------------------- 读数据
def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_astar_guidance(sess: Path) -> list[dict]:
    """读取录制时保存的 A* 软引导；旧录制没有该文件时返回空表。"""
    for name in ("astar_guidance.jsonl", "policy_guidance.jsonl"):
        path = sess / name
        if path.exists():
            rows = read_jsonl(path)
            return sorted(
                [row for row in rows if isinstance(row, dict) and row.get("t") is not None],
                key=lambda row: float(row["t"]),
            )
    return []


def guidance_at(rows: list[dict], t: float, max_age_s: float = 0.5) -> dict | None:
    """按 ZOH 取 t 时刻最近的规划引导。

    无效/过期引导也保留成 ``valid=false`` 的样本，而不是直接丢掉。混合 VLA
    必须见过“没有可靠 A*”的画面，才能在定位漂移时退回视觉避障；只有完全没有
    任何历史行（或尚未到第一行）才返回 ``None``。
    """
    if not rows:
        return None
    times = np.asarray([float(row["t"]) for row in rows], dtype=np.float64)
    j = int(np.searchsorted(times, float(t), side="right") - 1)
    if j < 0:
        return None
    row = rows[j]
    age = float(t) - float(times[j])
    waypoint = row.get("waypoint", row.get("wp"))
    if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
        waypoint = (0.0, 0.0)
    path = row.get("path")
    if isinstance(path, list):
        path = [[float(point[0]), float(point[1])] for point in path
                if isinstance(point, (list, tuple)) and len(point) >= 2]
    else:
        path = []
    valid = bool(row.get("valid", True)) and age <= float(max_age_s)
    reason = str(row.get("reason", "ok"))
    if age > float(max_age_s):
        reason = "guidance_stale"
    elif not bool(row.get("valid", True)):
        reason = reason if reason != "ok" else "guidance_invalid"
    elif not path:
        reason = "path_missing"
    return {
        "requested": bool(row.get("requested", True)),
        "control_mode": str(row.get("control_mode", "memory_revisit")),
        "valid": valid,
        "waypoint": [float(waypoint[0]), float(waypoint[1])] if valid else [0.0, 0.0],
        "path": path if valid else [],
        "mode": int(row.get("mode", 0) or 0),
        "confidence": float(row.get("confidence", 1.0) or 0.0),
        "pose_age_s": row.get("pose_age_s"),
        "sigma_m": row.get("sigma_m"),
        "source": str(row.get("source", "astar_navmesh")),
        "reason": reason,
        "age_s": max(0.0, age),
    }


def load_anchor(sess: Path) -> tuple[float, str]:
    run_path = sess / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        for ev in run.get("events") or []:
            if isinstance(ev, dict) and ev.get("obs_start_monotonic") is not None:
                return float(ev["obs_start_monotonic"]), "run.json"
    # 反推：首帧与首个 HMD 采样在同一拍，首帧相对秒就是启动耗时
    with (sess / "frames.jsonl").open("r", encoding="utf-8") as f:
        f0 = float(json.loads(f.readline())["t"])
    with (sess / "hmd_frames.jsonl").open("r", encoding="utf-8") as f:
        h0 = float(json.loads(f.readline())["t"])
    return h0 - f0, "inferred(hmd0-frames0)"


def load_streams(sess: Path):
    frames = read_jsonl(sess / "frames.jsonl")
    hmd = read_jsonl(sess / "hmd_frames.jsonl")
    osc = read_jsonl(sess / "osc.jsonl")

    f_t = np.array([r["t"] for r in frames], dtype=np.float64)
    f_i = np.array([r["i"] for r in frames], dtype=np.int64)
    h_t = np.array([r["t"] for r in hmd], dtype=np.float64)
    quat = np.array([r["hmd"]["rotation_xyzw"] for r in hmd], dtype=np.float64)

    vx_t, vx_v, vz_t, vz_v = [], [], [], []
    ay_t, ay_v = [], []
    for r in osc:
        val = float(r["args"][0])
        if r["addr"].endswith("VelocityX"):
            vx_t.append(r["t"]); vx_v.append(val)
        elif r["addr"].endswith("VelocityZ"):
            vz_t.append(r["t"]); vz_v.append(val)
        # ⚠️ AngularY **不在 osc.jsonl 里**（那里只有 VelocityX/Z）；
        # 它在 avatar_params.jsonl 里，字段是 {t, name, value}。见下方单独读取。
    vx_t = np.array(vx_t); vx_v = np.array(vx_v)
    vz_t = np.array(vz_t); vz_v = np.array(vz_v)
    # ω 标签的来源：avatar_params.jsonl 的 AngularY（构念上是身体/胶囊体角速度）
    ap_path = sess / "avatar_params.jsonl"
    if ap_path.exists():
        for r in read_jsonl(ap_path):
            if r.get("name") == "AngularY":
                ay_t.append(r["t"]); ay_v.append(float(r["value"]))
    ay_t = np.array(ay_t) if ay_t else np.zeros(0)
    ay_v = np.array(ay_v) if ay_v else np.zeros(0)
    if len(ay_t) == 0:
        print("⚠ %s: 没有 AngularY，ω 标签将全为 0" % sess.name, flush=True)

    # 合并成「任一路报文」的事件序列，两个轴的 ZOH 在同一时刻取样
    t_m = np.unique(np.concatenate([vx_t, vz_t]))
    ix = np.searchsorted(vx_t, t_m, side="right") - 1
    iz = np.searchsorted(vz_t, t_m, side="right") - 1
    vx_m = vx_v[np.clip(ix, 0, len(vx_v) - 1)]
    vz_m = vz_v[np.clip(iz, 0, len(vz_v) - 1)]
    return dict(f_t=f_t, f_i=f_i, h_t=h_t, quat=quat,
                t_m=t_m, vx_m=vx_m, vz_m=vz_m,
                vx_t=vx_t, vx_v=vx_v, vz_t=vz_t, vz_v=vz_v,
                ay_t=ay_t, ay_v=ay_v)


# ---------------------------------------------------------------- 姿态与轨迹
def hmd_forward(quat: np.ndarray) -> np.ndarray:
    """四元数 (x,y,z,w) → 水平前向单位向量 (fx, fz)。"""
    q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 2 * (x * z + w * y)
    fz = 1 - 2 * (x * x + y * y)
    norm = np.hypot(fx, fz)
    norm[norm < 1e-9] = 1.0
    return np.stack([fx / norm, fz / norm], axis=1)


def yaw_at(tq: np.ndarray | float, h_t: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """t 时刻的 HMD 航向（取最近一次采样，ZOH）。返回 (fx, fz) 或角度。"""
    tq_arr = np.atleast_1d(np.asarray(tq, dtype=np.float64))
    idx = np.searchsorted(h_t, tq_arr, side="right") - 1
    idx = np.clip(idx, 0, len(h_t) - 1)
    return fwd[idx]


def build_dr_path(streams: dict) -> dict:
    """OSC 速度 + HMD 朝向 → 机体系 DR 轨迹（世界米）。

    速度是角色局部系（vx 右 / vz 前），每段用区间起点的朝向把它转到世界系；
    可疑空窗（Δt > 2.5 s 且速度非零）按 0 积分（多半是丢了「停下」的包），避免轨迹跳飞。
    """
    t_m = streams["t_m"]
    vx, vz = streams["vx_m"], streams["vz_m"]
    fwd_m = yaw_at(t_m, streams["h_t"], streams["fwd"])
    p = np.zeros((len(t_m), 2), dtype=np.float64)
    for i in range(len(t_m) - 1):
        dt = float(t_m[i + 1] - t_m[i])
        vxi, vzi = float(vx[i]), float(vz[i])
        if dt > GAP_SUSPECT_S and np.hypot(vxi, vzi) > V_STILL:
            vxi = vzi = 0.0
        fx, fz = fwd_m[i]
        # 前向 (fx,fz)，右向 (fz,-fx)
        wx = vxi * fz + vzi * fx
        wz = -vxi * fx + vzi * fz
        p[i + 1] = p[i] + dt * np.array([wx, wz])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return dict(t=t_m, p=p, s=s)


def waypoint_body(t_obs: float, path: dict, fwd: np.ndarray,
                  lookahead_m: float, horizon_s: float = K_STEPS * STEP_S) -> list[float] | None:
    """t_obs 处沿轨迹前进 min(lookahead, 一个动作块内实际走过的距离) 的点，转到机体系。

    只看「动作块时长 1.2 s 内的真实轨迹」而不是整条未来轨迹：否则站着的人 0.8 m 外
    总有「以后会走到」的点，wp 恒 ≈ 0.8，§5.2 要求的「站着的窗口航点距离 ≈ 0」就没了。
    """
    t, p, s = path["t"], path["p"], path["s"]
    if len(t) < 2 or t_obs < t[0] or t_obs > t[-1]:
        return None

    def at(tq: float):
        j = int(np.searchsorted(t, tq, side="right") - 1)
        if j >= len(t) - 1:
            return p[-1], float(s[-1])
        frac = (tq - t[j]) / max(t[j + 1] - t[j], 1e-9)
        return p[j] + frac * (p[j + 1] - p[j]), float(s[j] + frac * (s[j + 1] - s[j]))

    p0, s0 = at(t_obs)
    _, s_end = at(min(t_obs + horizon_s, float(t[-1])))
    target_s = min(s0 + lookahead_m, s_end)
    if target_s <= s0 + 1e-9:
        target = p0
    else:
        m = int(np.searchsorted(s, target_s, side="right") - 1)
        m = max(0, min(m, len(s) - 2))
        ds = s[m + 1] - s[m]
        f = 0.0 if ds <= 1e-9 else (target_s - s[m]) / ds
        target = p[m] + f * (p[m + 1] - p[m])
    d = target - p0
    fx, fz = fwd
    dx = float(d[0] * fz + d[1] * -fx)   # 右向
    dz = float(d[0] * fx + d[1] * fz)    # 前向
    return [round(dx, 4), round(dz, 4)]


# ---------------------------------------------------------------- 标签
def zoh(streams: dict, tq: float):
    """返回 (vx, vz, 最近一次速度更新的年龄)。"""
    idx = int(np.searchsorted(streams["t_m"], tq, side="right") - 1)
    if idx < 0:
        return None
    return (float(streams["vx_m"][idx]), float(streams["vz_m"][idx]),
            float(tq - streams["t_m"][idx]))


def omega_rad_s_interval(streams: dict, t0: float, t1: float,
                        src_hz: int = OMEGA_SOURCE_HZ,
                        stale_s: float = 5.0,
                        agg: str = OMEGA_AGG_MEAN):
    """窗口 [t0, t1) 内的平均角速度（**rad/s，右转为正**），**只用过去信息**。

    为什么需要它（2026-11-05）：
      旧的 ``omega_rad_s_at`` 在 t_obs+k*0.1 处取**单个 ZOH 点**。
      实测（.tmp/freq_events_probe.py）5 Hz 观测下单个 ZOH 点只保留 35.2% 的转向事件，
      原因不是源数据不够——AngularY 实测 32~50 Hz（10/10 场 >= 30 Hz），
      而是**取点方式**把中间的事件跳过了。

    本函数改成：在窗口内按 ``src_hz`` 建 30 Hz 子网格，对每个子点做 ZOH，
    再取**绝对值加权平均**（保留方向：用带符号均值会正负抵消，把「先左后右」抹成 0）。

    因果性保证：
      * 窗口是 [t0, t1)，t1 是当前动作步的时刻，不含未来；
      * 每个子点的 ZOH 只回溯 ``t <= tq`` 的最后一条 AngularY，**不读未来**；
      * ``t1`` 由调用方给成 ``q + STEP_S``，即**当前步结束时刻**——动作步本身是
        「接下来 0.1 s 要执行什么」，所以用 [q, q+0.1) 是对标签的正当聚合，
        不是对未来决策的偷看。

    返回 None 表示该窗口完全没有 AngularY（调用方按契约写 0.0）。
    """
    ta, va = streams.get("ay_t"), streams.get("ay_v")
    if ta is None or len(ta) == 0:
        return None
    if t1 <= t0:
        return None
    n = max(1, int(round((t1 - t0) * float(src_hz))))
    edges = np.linspace(t0, t1, n + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    # 先把窗口内每个子网格点的 ZOH 原始值收齐（单位：AngularY 原始计数，非 rad/s）。
    # 一次遍历同时服务 mean 与 peak 两种聚合，避免两次查询引入分歧。
    vals = []
    for tq in mids:
        idx = int(np.searchsorted(ta, tq, side="right") - 1)
        if idx < 0 or idx >= len(ta):
            continue
        if tq - float(ta[idx]) > float(stale_s):
            continue
        vals.append(float(va[idx]))
    if not vals:
        return None

    unit_sign = -1.0 if ANGULARY_RAD_PER_UNIT < 0 else 1.0

    if agg == OMEGA_AGG_PEAK:
        # peak：取窗口内**绝对值最大**的那个采样，并**保留它自己的符号**。
        # 为什么不是「取最大值」：AngularY 左右转符号相反，取 max 会永远偏向一侧。
        # 为什么不把正负相加：那样「先急左再急右」会互相抵消，与 mean 的病一样。
        # 方向语义：保留峰值时刻的真实转向方向，这正是「这一窗里最剧烈的转向」。
        k = max(range(len(vals)), key=lambda i: abs(vals[i]))
        raw_peak = vals[k]
        return float(raw_peak * ANGULARY_RAD_PER_UNIT)

    # mean（默认，保持历史行为）：|AngularY| 均值 × 带符号均值的符号 × 单位符号
    acc = sum(abs(v) for v in vals)
    mag = acc / len(vals) * abs(ANGULARY_RAD_PER_UNIT)
    sacc = sum(vals)
    sgn = -1.0 if sacc < 0 else 1.0
    return float(sgn * unit_sign * mag)


def omega_rad_s_at(streams: dict, tq: float, stale_s: float = 5.0):
    """t 时刻的转向角速度（**弧度/秒，右转为正**），由 AngularY ZOH 换算。

    ⚠️ 单位必须 rad/s 而不是 deg/s：deg/s 是 ±120 量级、速度是 ±4，平方后差 3 个
    数量级，总损失会被 ω 独占（实测 mse_all=913 vs 速度口径 10）。

    ⚠️ AngularY 是**状态量**：VRChat 只在它变化时发事件，所以「很久没更新」
    **不等于缺失**——静止/匀速转时本来就不发。`stale_s` 太小会把「一直在转」
    误判成缺失，而缺失又按契约写成 0.0，等于教模型「没在转」。
    实测：0.5 s 容忍度下有 13.8%~25.8% 被判缺失；放宽到 2.0 s 后与 HMD 右转率
    的相关系数更高（见 probe 记录）。只有会话根本没录 AngularY 才返回 None。
    """
    ta, va = streams.get("ay_t"), streams.get("ay_v")
    if ta is None or len(ta) == 0:
        return None
    idx = int(np.searchsorted(ta, tq, side="right") - 1)
    if idx < 0 or idx >= len(ta):
        return None
    if tq - float(ta[idx]) > float(stale_s):   # 见 docstring：状态量，容忍度要宽
        return None
    return float(va[idx]) * ANGULARY_RAD_PER_UNIT


def hmd_static(tq: float, streams: dict) -> bool:
    h_t, quat = streams["h_t"], streams["quat"]
    lo = int(np.searchsorted(h_t, tq - HMD_STILL_S))
    hi = int(np.searchsorted(h_t, tq))
    if hi <= lo + 2:
        return False
    q0 = quat[lo]
    cos = np.abs(quat[lo:hi] @ q0)
    cos = np.clip(cos, 0.0, 1.0)
    max_deg = float(np.degrees(2.0 * np.arccos(cos.min())))
    return max_deg < HMD_STILL_DEG


def topo_at(tq: float, path: dict, topo_xyz: np.ndarray) -> tuple[float, float, float]:
    """tq 时刻的 (prog, lateral, remain)，世界米。取最近采样点（ZOH）。"""
    t = path["t"]
    if len(t) == 0:
        return (0.0, 0.0, 0.0)
    j = int(np.searchsorted(t, tq, side="right") - 1)
    j = max(0, min(j, len(t) - 1))
    return (float(topo_xyz[j, 0]), float(topo_xyz[j, 1]), float(topo_xyz[j, 2]))


def build_labels(streams: dict, frame_abs: np.ndarray, path: dict,
                 lag_s: float, lookahead_m: float, elapsed_s: float,
                 wp_mode: str = WP_MODE_NEAR,
                 omega_src_hz: int = OMEGA_SOURCE_HZ,
                 omega_agg: str = OMEGA_AGG,
                 guidance_rows: list[dict] | None = None,
                 use_astar_guidance: bool = False):
    """``omega_src_hz > 0`` 时用 30 Hz 源子网格聚合 ω；0 = 旧行为（单点 ZOH）。

    ``omega_agg`` 决定窗口内怎么聚：mean（默认）/ peak（取 |·| 最大的采样，保留符号）。
    """
    t_m = streams["t_m"]
    last_hmd = float(streams["h_t"][-1])
    grid = np.arange(frame_abs[0] + 0.5, frame_abs[-1] + 1e-9, GRID_S)
    vx_t0 = float(min(streams["vx_t"][0], streams["vz_t"][0]))
    fwd_m = yaw_at(grid, streams["h_t"], streams["fwd"])

    # [topo] 的离线合成：整条 DR 轨迹先算一遍，之后按 t 取（ZOH）。
    from policy_text import synthesize_topo
    topo_xyz = synthesize_topo(path["p"], path["s"])

    out = []
    stats = dict(grid=int(len(grid)), dropped_preosc=0, dropped_idle=0,
                 partial_mask=0, mask_all0=0, mask_all1=0,
                 guidance_available=0, guidance_invalid=0, guidance_missing=0)
    for n, t_obs in enumerate(grid):
        q0 = t_obs + lag_s
        if q0 < vx_t0:
            stats["dropped_preosc"] += 1
            continue
        z0 = zoh(streams, t_obs)
        if z0 is not None and z0[2] > IDLE_SILENCE_S and hmd_static(t_obs, streams):
            stats["dropped_idle"] += 1
            continue

        act, mask = [], []
        for k in range(K_STEPS):
            q = q0 + k * STEP_S
            z = zoh(streams, q)
            if q > last_hmd or z is None or (z[2] > GAP_SUSPECT_S and np.hypot(z[0], z[1]) > V_STILL):
                act.append([0.0, 0.0, 0.0]); mask.append(0)
            else:
                if omega_src_hz and omega_src_hz > 0:
                    # 30 Hz 源子网格 + 因果窗口聚合（2026-11-05）
                    # 窗口 [q, q+STEP_S)：q 是当前动作步的起点，q+0.1 是该步结束。
                    # 动作步的语义是「接下来 0.1 s 执行什么」，所以在这 0.1 s 内
                    # 做聚合是对标签的正当统计，不是偷看未来决策。
                    w = omega_rad_s_interval(streams, q, q + STEP_S,
                                             src_hz=omega_src_hz, agg=omega_agg)
                else:
                    w = omega_rad_s_at(streams, q)
                act.append([round(z[0], 4), round(z[1], 4),
                            0.0 if w is None else round(float(w), 2)])
                mask.append(1)

        valid = [k for k in range(K_STEPS) if mask[k]]
        if not valid:
            cls = "none"
        elif np.hypot(*act[0][:2]) > V_STILL and mask[0]:
            cls = "move"
        elif all(np.hypot(*act[k][:2]) <= V_STILL for k in valid):
            cls = "idle"
        else:
            cls = "trans"

        guide = guidance_at(guidance_rows or [], t_obs) if use_astar_guidance else None
        if use_astar_guidance:
            if guide is None:
                wp = [0.0, 0.0]
                stats["guidance_missing"] += 1
            else:
                wp = guide["waypoint"] if guide.get("valid") else [0.0, 0.0]
                if guide.get("valid"):
                    stats["guidance_available"] += 1
                else:
                    stats["guidance_invalid"] += 1
        elif wp_mode == WP_MODE_CAUSAL:
            # 因果口径：不留 wp（部署时没有未来轨迹，留 0 才是真相），
            # topo 换成只依赖历史的 (past_prog, past_lateral, 0)。
            # 见 policy_causal 模块头注释。
            wp = [0.0, 0.0]
        elif wp_mode == WP_MODE_FAR:
            # horizon 拉到无穷：目标 = 沿轨迹固定 WP_FAR_M 处（不足则取终点）
            wp = waypoint_body(t_obs, path, fwd_m[n], WP_FAR_M, horizon_s=1e9)
        else:
            wp = waypoint_body(t_obs, path, fwd_m[n], lookahead_m)
        t_rel = float(t_obs - frame_abs[0] + streams["f_t"][0])

        # ---- [mode] / [topo]（方案 §10.6，门 G8）----
        # M0 录制里没有地点图，也没有导航器状态机，所以：
        #   mode: 按 osc 零速判 stop(3)，其余 follow(0)。
        #         replan/explore 在 M0-M1 恒无（方案 §10.6 明写 explore 只在 M2 出现）。
        #   topo: 拿录制轨迹本身当「边」离线合成 prog/lateral/remain，
        #         格式在 M0 冻死，M3 换真实地点图时 token 位置不变。
        spd0 = float(np.hypot(act[0][0], act[0][1])) if mask[0] else 0.0
        mode = MODE_STOP if spd0 <= V_STILL else MODE_FOLLOW
        if use_astar_guidance and guide is not None and guide.get("valid"):
            mode = int(guide.get("mode", mode))
        if wp_mode == WP_MODE_CAUSAL:
            tp = causal_topo(t_obs, path, fwd_m[n])
        else:
            tp = topo_at(t_obs, path, topo_xyz)

        out.append(dict(
            t=round(float(t_obs), 6), t_rel=round(t_rel, 6),
            act=act, mask=mask, cls=cls,
            wp=[0.0, 0.0] if wp is None else wp,
            mode=int(mode),
            topo=[round(float(x), 4) for x in tp],
            path=None if guide is None else guide.get("path", []),
            guide=None if guide is None else {
                "requested": bool(guide.get("requested", False)),
                "control_mode": guide.get("control_mode", "unknown"),
                "valid": bool(guide.get("valid", False)),
                "confidence": guide.get("confidence"),
                "pose_age_s": guide.get("pose_age_s"),
                "sigma_m": guide.get("sigma_m"),
                "reason": guide.get("reason"),
                "age_s": guide.get("age_s"),
                "source": guide.get("source"),
            },
            split="val" if t_rel >= 0.8 * elapsed_s else "train",
        ))
        if sum(mask) == K_STEPS:
            stats["mask_all1"] += 1
        elif sum(mask) == 0:
            stats["mask_all0"] += 1
        else:
            stats["partial_mask"] += 1
    stats["samples"] = len(out)
    stats["mask_all1_pct"] = round(100.0 * stats["mask_all1"] / max(1, len(out)), 2)
    return out, stats


# ---------------------------------------------------------------- 抽帧
def extract_images(sess: Path, out_dir: Path, want: list[int],
                   size: tuple[int, int], q: int) -> tuple[int, int]:
    """一趟 ffmpeg 解码 + 裁剪/缩放/补边，只把 want 里的帧写成 JPEG。

    优化点（2026-10-31）：
    * 裁剪/缩放交给 ffmpeg（``crop ... exact=1`` + ``scale flags=area`` + ``pad``），
      Python 只做 JPEG 编码；管道里每帧从 1080×606 rgb24（1.96 MB）降到 448×256（0.34 MB），
      一个 8 min 场次从 ~27 GB 降到 ~4.8 GB。
    * ``crop`` 必须显式 ``exact=1``：默认 exact=0 对 4:2:0 会把 1077 悄悄改成 1076，
      帧字节数错 3 B/行、整条流错位。
    * ``-fps_mode passthrough`` 必须给：容器 30 fps CFR、实际 29.7 Hz，默认会丢/重复帧。
    返回 (写盘张数, 解码帧数)。
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("PATH 上没有 ffmpeg")
    meta = json.loads((sess / "meta.json").read_text(encoding="utf-8"))
    control_mode = str(meta.get("control_mode") or "legacy")
    src_w, src_h = meta["size"]
    w, h = int(size[0]), int(size[1])
    content_h, top = policy_geometry(size)
    crop_w = min(int(round(src_h * 16 / 9)), src_w)
    left = (src_w - crop_w) // 2
    vf = f"crop={crop_w}:{src_h}:{left}:0:exact=1,scale={w}:{content_h}:flags=area"
    if h > content_h:
        vf += f",pad={w}:{h}:0:{top}:black"
    vf += ",format=rgb24"
    cmd = [ff, "-v", "error", "-i", str(sess / "L.mp4"),
           "-fps_mode", "passthrough", "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    frame_bytes = w * h * 3
    want_set = set(int(x) for x in want)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            idx += 1  # frames.jsonl 的 i 从 1 开始
            if idx not in want_set:
                continue
            rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            (img_dir / f"{idx:06d}_L.jpg").write_bytes(encode_jpeg(rgb, q))
            written += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg 解码失败（exit {rc}）：{sess}/L.mp4")
    return written, idx


# ---------------------------------------------------------------- 单场
def build_session(sess: Path, out_root: Path, size: tuple[int, int], q: int, lag_s: float,
                  lookahead_m: float, no_images: bool, overwrite: bool,
                  wp_mode: str = WP_MODE_NEAR,
                  omega_src_hz: int = OMEGA_SOURCE_HZ,
                  omega_agg: str = OMEGA_AGG,
                  use_astar_guidance: bool = False) -> dict:
    sid = sess.name
    out_dir = out_root / sid
    if out_dir.exists() and not overwrite:
        raise SystemExit(f"{out_dir} 已存在；要重做加 --overwrite")
    meta = json.loads((sess / "meta.json").read_text(encoding="utf-8"))
    anchor, anchor_src = load_anchor(sess)
    streams = load_streams(sess)
    streams["fwd"] = hmd_forward(streams["quat"])
    frame_abs = streams["f_t"] + anchor
    path = build_dr_path(streams)
    guidance_rows = load_astar_guidance(sess) if use_astar_guidance else []
    guidance_required = use_astar_guidance and control_mode not in {"explore", "prompt_local"}
    if guidance_required and not guidance_rows:
        raise RuntimeError(
            f"{sid} 缺少 astar_guidance.jsonl/policy_guidance.jsonl，不能构建 hybrid 数据集")
    labels, stats = build_labels(streams, frame_abs, path, lag_s, lookahead_m,
                                 float(meta["elapsed_s"]), wp_mode=wp_mode,
                                 omega_src_hz=omega_src_hz, omega_agg=omega_agg,
                                 guidance_rows=guidance_rows,
                                 use_astar_guidance=use_astar_guidance)
    # 自由探索/局部命令没有 A* 是正常状态，不把它编码成“规划失败”。
    # 显式写出 requested=false，训练提示才能与 requested=true 的坏路径区分。
    for label in labels:
        label["control_mode"] = control_mode
        if use_astar_guidance and control_mode in {"explore", "prompt_local"} and not guidance_rows:
            label["path"] = []
            label["guide"] = {
                "requested": False,
                "control_mode": control_mode,
                "valid": False,
                "confidence": 0.0,
                "pose_age_s": None,
                "sigma_m": None,
                "reason": "astar_not_requested",
                "age_s": 0.0,
                "source": "astar_navmesh",
            }

    ts = np.array([r["t"] for r in labels])
    j = np.clip(np.searchsorted(frame_abs, ts), 1, len(frame_abs) - 1)
    cand = np.where(np.abs(frame_abs[j - 1] - ts) <= np.abs(frame_abs[j] - ts), j - 1, j)
    ok = np.abs(frame_abs[cand] - ts) <= 0.1

    kept, dropped_noimg = [], 0
    for r, k, good in zip(labels, cand, ok):
        if not good:
            dropped_noimg += 1
            continue
        kept.append(dict(label=r, obs_line=dict(t=r["t"], t_rel=r["t_rel"],
                                                 i=int(streams["f_i"][k]))))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_obs.jsonl").write_text("".join(
        json.dumps({"t": k["obs_line"]["t"], "t_rel": k["obs_line"]["t_rel"],
                    "i": k["obs_line"]["i"],
                    "img_L": f"images/{k['obs_line']['i']:06d}_L.jpg",
                    "sid": sid, "world": meta.get("world_key", ""),
                    "split": k["label"]["split"]}, ensure_ascii=False) + "\n"
        for k in kept), encoding="utf-8")
    (out_dir / "policy_labels.jsonl").write_text("".join(
        # ⚠️ 这里必须显式带上 mode/topo：build_labels 早就把它们算好了，
        # 但写盘用的是手写白名单，漏一个字段就会静默丢掉（实测踩过一次）。
        json.dumps({"t": k["label"]["t"], "act": k["label"]["act"], "mask": k["label"]["mask"],
                    "cls": k["label"]["cls"], "wp": k["label"]["wp"],
                    "mode": k["label"]["mode"], "topo": k["label"]["topo"],
                    "path": k["label"].get("path"), "guide": k["label"].get("guide"),
                    "control_mode": k["label"].get("control_mode", "legacy"),
                    "split": k["label"]["split"]}, ensure_ascii=False) + "\n"
        for k in kept), encoding="utf-8")

    n_img = 0
    if not no_images and kept:
        want = sorted({k["obs_line"]["i"] for k in kept})
        n_img, decoded = extract_images(sess, out_dir, want, size, q)
        if decoded != len(frame_abs):
            print(f"⚠ {sid}: mp4 解出 {decoded} 帧，frames.jsonl {len(frame_abs)} 帧", flush=True)

    cls_count = {}
    for k in kept:
        cls_count[k["label"]["cls"]] = cls_count.get(k["label"]["cls"], 0) + 1
    stats["mask_all1"] = int(sum(1 for k in kept if sum(k["label"]["mask"]) == K_STEPS))
    stats["mask_all1_pct"] = round(100.0 * stats["mask_all1"] / max(1, len(kept)), 2)
    stats.update(dict(
        sid=sid, world=meta.get("world_key", ""), elapsed_s=meta["elapsed_s"],
        control_mode=control_mode,
        frames=meta["frames"], anchor=anchor, anchor_source=anchor_src,
        samples_kept=len(kept), dropped_noimg=dropped_noimg, images=n_img,
        cls=cls_count, val=int(sum(1 for k in kept if k["label"]["split"] == "val")),
    ))
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("%s  grid=%d kept=%d (no-img %d) mask_all1=%.1f%% cls=%s images=%d"
          % (sid, stats["grid"], len(kept), dropped_noimg, stats["mask_all1_pct"],
             cls_count, n_img), flush=True)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构建 M0 policy 训练集（5 Hz 左目 + 12 步标签）")
    ap.add_argument("--root", default=r"G:\vrc_recording", help="原始录制根目录")
    ap.add_argument("--out", required=True, help="训练集输出目录")
    ap.add_argument("--sessions", nargs="*", default=None, help="只做这些场（默认全部）")
    ap.add_argument("--size", nargs=2, type=int, default=list(POLICY_SIZE),
                    metavar=("W", "H"), help="policy 输入尺寸（默认 448 256）")
    ap.add_argument("--q", type=int, default=JPEG_Q)
    ap.add_argument("--lag", type=float, default=DEFAULT_LAG_S, help="osc_lag_s")
    ap.add_argument("--lookahead", type=float, default=DEFAULT_LOOKAHEAD_M)
    ap.add_argument("--wp-mode", choices=[WP_MODE_NEAR, WP_MODE_FAR, WP_MODE_CAUSAL],
                    help="near=1.2 s 行程内的点（模长泄漏速度）；far=沿轨迹固定 4 m（去泄漏）；"
                         "causal=wp 恒 0 + topo 只用历史（部署口径，2026-11-05）")
    ap.add_argument("--astar-guidance", action="store_true",
                    help="读取每场 astar_guidance.jsonl，使用录制时的 A* 局部引导；"
                         "memory_revisit 缺文件直接失败；explore/prompt_local 允许无引导，"
                         "不回退到未来轨迹 wp")
    ap.add_argument("--omega-agg", choices=list(OMEGA_AGG_MODES), default=OMEGA_AGG,
                    help="ω 窗口聚合方式：mean=窗口内|AngularY|均值×方向（默认，兼容旧数据）；"
                         "peak=窗口内|AngularY|最大的采样，保留其自身符号")
    ap.add_argument("--omega-src-hz", type=int, default=OMEGA_SOURCE_HZ,
                    help="ω 标签的源采样频率（默认 30）。用该频率的子网格对每个 0.1 s "
                         "动作步做因果窗口聚合；0 = 退回单点 ZOH 的旧行为。"
                         "注意：只改 ω 标签的计算分辨率，不改视觉 5 Hz 与动作 12x0.1 s")
    ap.add_argument("--no-images", action="store_true", help="只生成标签，不抽帧")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4,
                    help="并行处理场次数（ffmpeg 解码是 CPU 密集，4 个左右最划算）")
    ap.add_argument("--manifest-only", action="store_true",
                    help="不重建场次，只按已有 stats.json 汇总 manifest.json")
    a = ap.parse_args(argv)
    size = (int(a.size[0]), int(a.size[1]))

    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # 根目录里可能混着 vrc_model 这类非录制目录：只认带 meta.json + frames.jsonl 的场次
    sess_dirs = sorted(p for p in root.iterdir()
                       if p.is_dir() and (p / "meta.json").exists() and (p / "frames.jsonl").exists())
    if a.sessions:
        sess_dirs = [p for p in sess_dirs if p.name in set(a.sessions)]
        if not sess_dirs:
            print("没有匹配的场次")
            return 2

    import sys as _sys
    # 完整构建命令行。2026-11-05 加：本轮两次出现「参数漏传导致静默错口径」
    # （先 omega_src_hz、后 omega_agg），而 manifest 恰好读的是 CLI 层对象，
    # 所以它「看起来」正确。把 argv 原样存下来，是唯一能事后自证的证据。
    build_cmd = " ".join(_sys.argv)
    manifest = dict(root=str(root), size=list(size), jpeg_q=a.q, grid_hz=1 / GRID_S,
                    steps=K_STEPS, step_s=STEP_S, osc_lag_s=a.lag,
                    lookahead_m=a.lookahead, sessions={},
                    astar_guidance=bool(a.astar_guidance),
                    build_cmd=build_cmd,
                    build_argv=list(_sys.argv),
                    build_py=_sys.version.split()[0],
                    # 2026-11-03：动作维从 2 扩到 3（vx, vz, ω），训练侧据此判定头维度
                    act_dim=3, wp_mode=a.wp_mode,
                    omega=dict(source="AngularY", unit="rad/s",
                               # 2026-11-05：源采样频率元数据。
                               # 30 = 用 30 Hz 子网格做因果窗口聚合（新）；
                               # 0  = 单点 ZOH 的旧行为。
                               source_hz=int(a.omega_src_hz),
                               aggregation=(
                                   "interval_abs_peak" if (a.omega_src_hz > 0 and a.omega_agg == OMEGA_AGG_PEAK)
                                   else ("interval_abs_mean" if a.omega_src_hz > 0 else "zoh_point")),
                               rad_per_unit=ANGULARY_RAD_PER_UNIT,
                               fullscale_rad=ANGULARY_FULLSCALE_RAD,
                               sign="right_positive", missing="0.0",
                               measured="2026-11-03 积分对齐 HMD，R2 0.79-0.96"))
    if a.manifest_only:
        for sess in sess_dirs:
            st_path = out / sess.name / "stats.json"
            if not st_path.exists():
                print(f"跳过 {sess.name}: 没有 stats.json")
                continue
            manifest["sessions"][sess.name] = json.loads(st_path.read_text(encoding="utf-8"))
    elif a.workers > 1 and len(sess_dirs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            # ⚠️ 并行分支必须与下面的串行分支传**同一组**关键字参数。
            # 2026-11-05 修：此前漏了 omega_src_hz，默认 30 时碰巧不出错，
            # 但 `--omega-src-hz 0`（旧单点 ZOH 口径）在多场并行构建时会被
            # 静默当成 30 —— 结果与串行分支不一致，且不报错。
            # ⚠️ 2026-11-05 二修：又漏了 omega_agg（与上一轮漏 omega_src_hz 同型）。
            # 后果：--omega-agg peak 只改了 manifest 里的字符串，标签仍按 mean 生成，
            # 而峰值与均值在数值上可以完全一致（实测 400/400 标签差 0.0000）——
            # 不报错、不崩、查不出来。所以两个分支都必须显式传全部 omega_* 参数。
            futs = {sess.name: ex.submit(build_session, sess, out, size, a.q, a.lag,
                                         a.lookahead, a.no_images, a.overwrite,
                                         wp_mode=a.wp_mode,
                                         omega_src_hz=a.omega_src_hz,
                                         omega_agg=a.omega_agg,
                                         use_astar_guidance=a.astar_guidance)
                    for sess in sess_dirs}
            for name, fut in futs.items():
                manifest["sessions"][name] = fut.result()
    else:
        for sess in sess_dirs:
            manifest["sessions"][sess.name] = build_session(
                sess, out, size, a.q, a.lag, a.lookahead, a.no_images, a.overwrite,
                wp_mode=a.wp_mode, omega_src_hz=a.omega_src_hz,
                omega_agg=a.omega_agg, use_astar_guidance=a.astar_guidance)

    all_cls = {}
    tot = dict(grid=0, kept=0, noimg=0, images=0)
    for st in manifest["sessions"].values():
        tot["grid"] += st["grid"]; tot["kept"] += st["samples_kept"]
        tot["noimg"] += st["dropped_noimg"]; tot["images"] += st["images"]
        for c, n in st["cls"].items():
            all_cls[c] = all_cls.get(c, 0) + n
    manifest["totals"] = dict(tot, cls=all_cls)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("总计 grid=%d kept=%d no-img=%d cls=%s" %
          (tot["grid"], tot["kept"], tot["noimg"], all_cls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
