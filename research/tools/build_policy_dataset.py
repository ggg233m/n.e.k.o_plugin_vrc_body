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
    <out>/<sid>/policy_labels.jsonl         {"t","act":[[vx,vz]×12],"mask":[...],"cls","wp":[dx,dz]}
    <out>/<sid>/stats.json                  本场统计（含 G3 口径）
    <out>/manifest.json                    汇总

用法::

    .venv/Scripts/python.exe research/tools/build_policy_dataset.py \
        --root G:\\vrc_recording --out H:\\vrc_policy_m0
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

K_STEPS = 12
STEP_S = 0.1
GRID_S = 0.2
V_STILL = 0.1
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
    for r in osc:
        val = float(r["args"][0])
        if r["addr"].endswith("VelocityX"):
            vx_t.append(r["t"]); vx_v.append(val)
        elif r["addr"].endswith("VelocityZ"):
            vz_t.append(r["t"]); vz_v.append(val)
    vx_t = np.array(vx_t); vx_v = np.array(vx_v)
    vz_t = np.array(vz_t); vz_v = np.array(vz_v)

    # 合并成「任一路报文」的事件序列，两个轴的 ZOH 在同一时刻取样
    t_m = np.unique(np.concatenate([vx_t, vz_t]))
    ix = np.searchsorted(vx_t, t_m, side="right") - 1
    iz = np.searchsorted(vz_t, t_m, side="right") - 1
    vx_m = vx_v[np.clip(ix, 0, len(vx_v) - 1)]
    vz_m = vz_v[np.clip(iz, 0, len(vz_v) - 1)]
    return dict(f_t=f_t, f_i=f_i, h_t=h_t, quat=quat,
                t_m=t_m, vx_m=vx_m, vz_m=vz_m,
                vx_t=vx_t, vx_v=vx_v, vz_t=vz_t, vz_v=vz_v)


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


def build_labels(streams: dict, frame_abs: np.ndarray, path: dict,
                 lag_s: float, lookahead_m: float, elapsed_s: float):
    t_m = streams["t_m"]
    last_hmd = float(streams["h_t"][-1])
    grid = np.arange(frame_abs[0] + 0.5, frame_abs[-1] + 1e-9, GRID_S)
    vx_t0 = float(min(streams["vx_t"][0], streams["vz_t"][0]))
    fwd_m = yaw_at(grid, streams["h_t"], streams["fwd"])

    out = []
    stats = dict(grid=int(len(grid)), dropped_preosc=0, dropped_idle=0,
                 partial_mask=0, mask_all0=0, mask_all1=0)
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
                act.append([0.0, 0.0]); mask.append(0)
            else:
                act.append([round(z[0], 4), round(z[1], 4)]); mask.append(1)

        valid = [k for k in range(K_STEPS) if mask[k]]
        if not valid:
            cls = "none"
        elif np.hypot(*act[0]) > V_STILL and mask[0]:
            cls = "move"
        elif all(np.hypot(*act[k]) <= V_STILL for k in valid):
            cls = "idle"
        else:
            cls = "trans"

        wp = waypoint_body(t_obs, path, fwd_m[n], lookahead_m)
        t_rel = float(t_obs - frame_abs[0] + streams["f_t"][0])
        out.append(dict(
            t=round(float(t_obs), 6), t_rel=round(t_rel, 6),
            act=act, mask=mask, cls=cls,
            wp=[0.0, 0.0] if wp is None else wp,
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
                  lookahead_m: float, no_images: bool, overwrite: bool) -> dict:
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
    labels, stats = build_labels(streams, frame_abs, path, lag_s, lookahead_m,
                                 float(meta["elapsed_s"]))

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
        json.dumps({"t": k["label"]["t"], "act": k["label"]["act"], "mask": k["label"]["mask"],
                    "cls": k["label"]["cls"], "wp": k["label"]["wp"],
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

    manifest = dict(root=str(root), size=list(size), jpeg_q=a.q, grid_hz=1 / GRID_S,
                    steps=K_STEPS, step_s=STEP_S, osc_lag_s=a.lag,
                    lookahead_m=a.lookahead, sessions={})
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
            futs = {sess.name: ex.submit(build_session, sess, out, size, a.q, a.lag,
                                         a.lookahead, a.no_images, a.overwrite)
                    for sess in sess_dirs}
            for name, fut in futs.items():
                manifest["sessions"][name] = fut.result()
    else:
        for sess in sess_dirs:
            manifest["sessions"][sess.name] = build_session(
                sess, out, size, a.q, a.lag, a.lookahead, a.no_images, a.overwrite)

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
