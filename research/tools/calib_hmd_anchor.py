# -*- coding: utf-8 -*-
"""用 **HMD 位姿差** 作为米制参照反解 cam_h —— 与 OSC 链、深度链都独立。

## 为什么需要它

`Docs/尺度标定门依赖-9%分歧（2026-09-24）.md` 查出：`scale_calib` 用「深度重投影位移」
对「OSC 航位位移」反解 cam_h，而这两条链在尺度上差 5–13%，无法判定谁对 ——
因为两个分量各有一份不确定度。

`scale_calib` 的位移源是 `dead_reckon`（= OSC 路程积分）。**本脚本把位移源换成 HMD 位姿差。**

依据（读 AnyaDance 驱动源码）：

- 位姿**逐字上报**、`qWorldFromDriverRotation` 为 identity（`virtual_device.cpp:619-621/712-718`）
  ⇒ 我们下发的 HMD 位姿就是 SteamVR 收到的位姿，**零失真回声**；
- 单位是**米**（`docs/protocol.md:411`）；
- HMD 的 Y 只被夹到 [0, 25] m、位置分量 ≤ 30 m（`constants.h:32/33/39`）⇒ 扫掠范围内不被夹取；
- `driver_log.py` 记录的是**驱动实际接受**的位姿（多播 `239.255.39.71:39571`）。

**因此**：avatar 静止时，相机的世界位移 **=** HMD 的局部位移。
这是一把自带米制的尺子，**不含 OSC，也不含深度模型** —— 用它反解 cam_h，
就能判定「深度链」与「OSC 链」究竟谁在米制上是对的。

## 与 scale_calib 的关系

求解式、门、统计、深度重投影完全沿用 `scale_calib`；**只换 `t_ij_cam` 的来源**：

    scale_calib : t_ij_cam = _R_wc(-yaw_j) @ (C_i - C_j)   # C 来自 dead_reckon（OSC）
    calib_hmd_anchor: t_ij_cam = _R_wc(-yaw_j) @ (p_i - p_j)  # p 来自 HMD 位姿

## 用法

    :: 合成自检（验证解算与位移源，不需要任何素材）
    .venv/Scripts/python.exe research/tools/calib_hmd_anchor.py --selftest

    :: 真实 run
    .venv/Scripts/python.exe research/tools/calib_hmd_anchor.py \
        --run .slam_probe/offline_probe/recorder/runs/<新run> \
        --cache .tmp/_depth_cache_hmd.npz --out .tmp/calib_hmd_anchor.json

> **前置**：需要一段「avatar 静止 + HMD 沿已知轨迹平移」的录制。
> 见 `Docs/标定录制SOP（2026-09-24）.md` 段 B。2026-09-20 那批 run 的 HMD 位置是常量
> `[0, 1.5, 0]`，本脚本会正确地报「HMD 位移不足」而不是硬凑一个数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_REC = REPO / "research" / "recorder"
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from scale_calib import _R_wc, _build_dr  # noqa: E402  同一套旋转/航位约定，不复制实现
from run_motion import OscPath  # noqa: E402


# ---------------------------------------------------------------------------
# HMD 位姿流
# ---------------------------------------------------------------------------

def load_hmd_poses(run_dir: Path, yaw_sign: int):
    """读 hmd_frames.jsonl -> (t, pos(N,3), yaw_rad(N,))；yaw 解卷绕后乘 yaw_sign。

    yaw 公式与 run_motion.HmdYaw 逐字一致，避免两套实现漂移。
    """
    t, pos, quat = [], [], []
    with (run_dir / "hmd_frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            hmd = r.get("hmd") or {}
            rot = hmd.get("rotation_xyzw")
            p = hmd.get("position")
            if not rot or not p:
                continue
            t.append(float(r["t"]))
            pos.append([float(p[0]), float(p[1]), float(p[2])])
            quat.append([float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])])
    if not t:
        raise ValueError("hmd_frames.jsonl 里没有可用位姿")
    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.degrees(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))
    u = np.zeros_like(yaw)
    u[0] = yaw[0]
    for i in range(1, len(yaw)):
        u[i] = u[i - 1] + ((yaw[i] - yaw[i - 1] + 180.0) % 360.0 - 180.0)
    order = np.argsort(t)
    t, pos, u = t[order], pos[order], u[order]
    # 去重时间戳（np.interp 要求严格递增）
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], pos[keep], np.radians(u[keep]) * float(yaw_sign)


# ---------------------------------------------------------------------------
# 单帧对解算（与 scale_calib 同一套式，只换 t_ij_cam 来源）
# ---------------------------------------------------------------------------

def solve_pair(d_i, d_j, Ki, vhi, Kj, vhj, fx, fy, cx, W, H,
               Rij, t_ij_cam, Uf, Vf, p):
    """解单帧对的 s。返回 (s, reason)。reason=='ok' 表示可用。"""
    ui0 = Uf.astype(int)
    vi0 = Vf.astype(int)
    Zi = Ki / np.maximum(d_i[vi0, ui0], 1e-6)
    good = np.isfinite(Zi) & (Zi >= p["near_min"]) & (Zi <= p["near_max"])
    if good.sum() < p["min_valid_pix"]:
        return None, "far"
    U = Uf[good]
    V = Vf[good]
    Zi = Zi[good]
    Pi = np.stack([(U - cx) * Zi / fx, (V - vhi) * Zi / fy, Zi], axis=1)

    s = 1.0
    base = Rij @ Pi.T
    ray = None
    valid = np.zeros(len(Zi), dtype=bool)
    for _ in range(p["n_fp_iter"]):
        Pj_star = (s * base).T + t_ij_cam
        Zj = Pj_star[:, 2]
        okf = Zj > 0.1
        uj = fx * Pj_star[:, 0] / np.where(okf, Zj, 1.0) + cx
        vj = fy * Pj_star[:, 1] / np.where(okf, Zj, 1.0) + vhj
        ui = np.clip(np.round(np.nan_to_num(uj, nan=0.0, posinf=0.0, neginf=0.0)).astype(int),
                     0, W - 1)
        vi = np.clip(np.round(np.nan_to_num(vj, nan=0.0, posinf=0.0, neginf=0.0)).astype(int),
                     0, H - 1)
        Dj = d_j[vi, ui]
        Zc_j = Kj / np.maximum(Dj, 1e-6)
        valid = (okf & np.isfinite(Dj) & (Zc_j >= p["near_min"]) & (Zc_j <= p["near_max"])
                 & (uj >= 1.0) & (uj <= W - 2.0) & (vj >= 1.0) & (vj <= H - 2.0))
        if valid.sum() < p["min_valid_pix"]:
            return None, "weak"
        Pj = np.stack([(uj - cx) * Zc_j / fx,
                       (vj - vhj) * Zc_j / fy,
                       Zc_j], axis=1)[valid]
        ray = Pj - (s * base).T[valid]
        den = float(np.sum(ray * ray))
        if den <= 1e-12:
            return None, "weak"
        s = float(np.sum(ray @ t_ij_cam)) / den
    if ray is None or valid.sum() < p["min_valid_pix"]:
        return None, "weak"
    resid = np.linalg.norm(ray - t_ij_cam[None, :], axis=1) / (np.linalg.norm(t_ij_cam) + 1e-9)
    if np.median(resid) > p["max_resid_per_trans"]:
        return None, "weak"
    return float(s), "ok"


DEFAULTS = dict(
    near_min=0.3, near_max=6.0,
    gap_min_frames=3, gap_max_frames=16,
    min_trans_m=0.4,
    max_dyaw_deg=10.0,      # 纯平移锚点：转向必须小，否则 Rij 的误差进 t_ij_cam
    seed_step_u=20, seed_step_v=20,
    n_fp_iter=3,
    max_resid_per_trans=0.15,
    min_valid_pix=12,
    trim_iqr_k=3.0,
    min_pairs=30,
    yaw_sign=-1,
    assumed_cam_h=1.5,
    # 仅在 --translation-source dr（复现 scale_calib）时使用
    min_cov=0.5, max_gap_s=0.30, osc_policy="zoh",
)


# ---------------------------------------------------------------------------
# 合成自检：平面地板上的已知 s 往返
# ---------------------------------------------------------------------------

def _synth_floor_depth(W, H, fx, fy, cx, cy, vh, k_cached, cam_x, s_true,
                       z_min=1.0, z_max=12.0):
    """合成「相机高 cam_h_true、俯仰 0、纯平移」下的**纯平面地板**深度图（缓存单位）。

    ⚠️ 纯平面地板对尺度是**退化**的：地板反深度只依赖像素行，且投影行
    `v = vh + fy*1.5*s_true/Z` 里 s 被约掉 ⇒ `s=1` 也是不动点，迭代从 1 出发就卡住。
    这个函数只用于演示该退化（见 selftest 的对照项），**不能**用来验证尺度反解。
    验证请用 `_synth_scene_depth`（地板 + 墙）。
    """
    VV = np.arange(H, dtype=float)[:, None] * np.ones((1, W))
    d = np.zeros((H, W), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = 1.5 * fy * s_true / (VV - vh)
    ok = (VV > vh + 1e-6) & np.isfinite(Z) & (Z >= z_min) & (Z <= z_max)
    d[ok] = (k_cached * s_true / Z[ok]).astype(np.float32)
    return d


def selftest() -> int:
    W, H = 448, 252
    fx = fy = 180.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    vh = cy
    k_cached = 1.5 * fy
    p = dict(DEFAULTS)
    Uf, Vf = np.meshgrid(np.arange(0, W, p["seed_step_u"], dtype=float),
                         np.arange(0, H, p["seed_step_v"], dtype=float))
    Uf, Vf = Uf.ravel(), Vf.ravel()

    print("=" * 78)
    print("合成自检 A：位移源（HMD 位姿差）的解析检查")
    print("=" * 78)
    ok_all = True
    cases = [
        # (说明, p_i, p_j, yaw_i_deg, yaw_j_deg, 期望的 |t| 与世界方向)
        ("纯横移 +0.4 m，无旋转", [0, 1.5, 0], [0.4, 1.5, 0], 0, 0, (-0.4, 0.0, 0.0)),
        ("纯横移 -0.5 m，无旋转", [0, 1.5, 0], [-0.5, 1.5, 0], 0, 0, (0.5, 0.0, 0.0)),
        ("纯前移 +0.6 m，无旋转", [0, 1.5, 0], [0, 1.5, 0.6], 0, 0, (0.0, 0.0, -0.6)),
        ("转 90° 后横移 +0.4 m", [0, 1.5, 0], [0.4, 1.5, 0], 0, 90, (0.0, 0.0, 0.4)),
        ("转 90° 后前移 +0.6 m", [0, 1.5, 0], [0, 1.5, 0.6], 0, 90, (-0.6, 0.0, 0.0)),
    ]
    for label, pi, pj, yi, yj, want in cases:
        yw_i = np.radians(float(yi) * DEFAULTS["yaw_sign"])
        yw_j = np.radians(float(yj) * DEFAULTS["yaw_sign"])
        t = _R_wc(-yw_j) @ (np.asarray(pi, float) - np.asarray(pj, float))
        good = np.allclose(t, np.asarray(want, float), atol=1e-9)
        ok_all = ok_all and good
        print("  %-26s t_ij_cam=(%+.3f,%+.3f,%+.3f)  期望=(%+.3f,%+.3f,%+.3f)  %s"
              % (label, t[0], t[1], t[2], want[0], want[1], want[2], "✅" if good else "❌"))

    print()
    print("=" * 78)
    print("合成自检 B：为什么**不用**合成场景验证尺度反解")
    print("=" * 78)
    W, H = 448, 252
    fx = fy = 180.0
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    k_cached = 1.5 * fy
    p = dict(DEFAULTS)
    Uf, Vf = np.meshgrid(np.arange(0, W, p["seed_step_u"], dtype=float),
                         np.arange(0, H, p["seed_step_v"], dtype=float))
    Uf, Vf = Uf.ravel(), Vf.ravel()
    for cam_h_true in (1.30, 1.42):
        s_true = cam_h_true / 1.5
        d_i = _synth_floor_depth(W, H, fx, fy, cx, cy, cy, k_cached, 0.0, s_true)
        d_j = _synth_floor_depth(W, H, fx, fy, cx, cy, cy, k_cached, 0.4, s_true)
        t_ij = _R_wc(0.0) @ (np.array([0.0, 1.5, 0.0]) - np.array([0.4, 1.5, 0.0]))
        s_est, reason = solve_pair(d_i, d_j, k_cached, cy, k_cached, cy,
                                   fx, fy, cx, W, H, _R_wc(0.0), t_ij, Uf, Vf, p)
        print("  纯平面地板，真实 cam_h=%.2f -> 解出 %.5f（恒为 s=1 不动点）"
              % (cam_h_true, 1.5 * s_est))
    print()
    print("  原因：平面场景的**反深度图只依赖像素行**，且投影行里的 s 被约掉，")
    print("        于是 d_i ≡ d_j、s=1 恒为不动点 ⇒ 平面场景对尺度不可辨识。")
    print("        真实场景靠**非平面结构**提供尺度信息，所以合成场景构造不出有效测试。")
    print("        ⇒ 尺度反解的正确性改用真实数据与 scale_calib 交叉验证：")
    print("           --translation-source dr 应复现 scale_calib 的 cam_h。")
    print()
    print("自检结论（位移源解析检查）：%s" % ("全部通过 ✅" if ok_all else "存在失败 ❌"))
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# 真实 run
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--yaw-sign", type=int, default=DEFAULTS["yaw_sign"])
    ap.add_argument("--translation-source", choices=["hmd", "dr"], default="hmd",
                    help="hmd = HMD 位姿差（独立锚点，本脚本的目的）；"
                         "dr = dead_reckon/OSC（复现 scale_calib，用于验证解算等价）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max-dyaw-deg", type=float, default=None,
                    help="相对转角上限（HMD 锚点默认 %g；复现 scale_calib 用 50）"
                         % DEFAULTS["max_dyaw_deg"])
    ap.add_argument("--max-resid-per-trans", type=float, default=None,
                    help="深度链 vs 位移源一致性门（默认 %g）" % DEFAULTS["max_resid_per_trans"])
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.run or not a.cache:
        ap.error("需要 --run 与 --cache（或用 --selftest）")

    run_dir = (REPO / a.run).resolve() if not Path(a.run).is_absolute() else Path(a.run)
    cache = (REPO / a.cache).resolve() if not Path(a.cache).is_absolute() else Path(a.cache)

    p = dict(DEFAULTS)
    p["yaw_sign"] = int(a.yaw_sign)
    if a.max_dyaw_deg is not None:
        p["max_dyaw_deg"] = float(a.max_dyaw_deg)
    if a.max_resid_per_trans is not None:
        p["max_resid_per_trans"] = float(a.max_resid_per_trans)

    f = np.load(cache)
    D16 = f["d16"].astype(np.float32)
    K = f["k"].astype(np.float64)
    VH = f["vh"].astype(np.float64)
    t_tele = f["t_tele"].astype(np.float64)
    fx, fy, cy = float(f["fx"]), float(f["fy"]), float(f["cy"])
    W, H = int(f["W"]), int(f["H"])
    cx = (W - 1) / 2.0
    N = D16.shape[0]
    f.close()

    hmd_t, hmd_pos, hmd_yaw = load_hmd_poses(run_dir, p["yaw_sign"])
    fr_pos = np.stack([np.interp(t_tele, hmd_t, hmd_pos[:, k]) for k in range(3)], axis=1)
    fr_yaw = np.interp(t_tele, hmd_t, hmd_yaw)

    src = a.translation_source
    osc = None
    if src == "dr":
        # 复现 scale_calib：位姿来自 dead_reckon（OSC 路程积分），位移源随之变成 OSC。
        dr = _build_dr(str(run_dir), p["yaw_sign"])
        fx_, fz_, fyaw_ = dr(t_tele)
        fr_pos = np.stack([fx_, np.full_like(fx_, p["assumed_cam_h"]), fz_], axis=1)
        fr_yaw = fyaw_
        osc = OscPath(str(run_dir))

    span = float(np.ptp(fr_pos[:, 0])), float(np.ptp(fr_pos[:, 2]))
    print("位移源 = %s   帧位姿跨度：X %.3f m，Z %.3f m（%d 帧，缓存 %d 帧）"
          % (src, span[0], span[1], len(hmd_t), N))
    if src == "hmd" and max(span) < p["min_trans_m"]:
        print()
        print("⇒ **HMD 位移不足**（X %.3f m / Z %.3f m，均 < min_trans_m %.2f m）："
              % (span[0], span[1], p["min_trans_m"]))
        print("  这段 run 里 HMD 没有平移，无法用 HMD 锚点反解 cam_h。")
        print("  这是**如实拒绝**，不是失败 —— 需要按")
        print("  Docs/标定录制SOP（2026-09-24）.md 的段 B 重录（avatar 静止 + HMD 扫掠）。")
        return 2

    Uf, Vf = np.meshgrid(np.arange(0, W, p["seed_step_u"], dtype=float),
                         np.arange(0, H, p["seed_step_v"], dtype=float))
    Uf, Vf = Uf.ravel(), Vf.ravel()

    cnt = dict(cand=0, lowtrans=0, pure_rot=0, far=0, weak=0, valid=0, osc_hole=0)
    s_all, s_trans = [], []
    t0 = time.time()
    for ci in range(N):
        lo, hi = ci + p["gap_min_frames"], min(N, ci + p["gap_max_frames"] + 1)
        for cj in range(lo, hi):
            cnt["cand"] += 1
            # dr 模式复现 scale_calib：先用 OSC 新鲜度门，再用 OSC 位移做门槛
            if osc is not None:
                fr = osc.freshness(t_tele[ci], t_tele[cj])
                if fr["coverage_frac"] < p["min_cov"] or fr["max_gap_s"] > p["max_gap_s"]:
                    cnt["osc_hole"] += 1
                    continue
                trans = (fr["dist_zoh_m"] if p["osc_policy"] == "zoh"
                         else fr["dist_stop_m"])
            else:
                trans = None
            dyaw = abs(fr_yaw[cj] - fr_yaw[ci])
            if np.degrees(dyaw) > p["max_dyaw_deg"]:
                cnt["pure_rot"] += 1
                continue
            t_ij_cam = _R_wc(-fr_yaw[cj]) @ (fr_pos[ci] - fr_pos[cj])
            if trans is None:
                trans = float(np.linalg.norm(t_ij_cam))
            if trans < p["min_trans_m"]:
                cnt["lowtrans"] += 1
                continue
            Rij = _R_wc(fr_yaw[ci] - fr_yaw[cj])
            s, reason = solve_pair(D16[ci], D16[cj], K[ci], VH[ci], K[cj], VH[cj],
                                   fx, fy, cx, W, H, Rij, t_ij_cam, Uf, Vf, p)
            if reason != "ok":
                cnt["far" if reason == "far" else "weak"] += 1
                continue
            cnt["valid"] += 1
            s_all.append(s)
            s_trans.append(trans)
        if (ci - 0) % 300 == 0 and ci:
            print("  frame %d/%d valid=%d elapsed=%.0fs" % (ci, N, cnt["valid"], time.time() - t0),
                  flush=True)

    if len(s_all) < p["min_pairs"]:
        print("⇒ 有效帧对 %d < min_pairs %d，不可观测" % (len(s_all), p["min_pairs"]))
        return 2

    s_all = np.asarray(s_all)
    med = float(np.median(s_all))
    iqr = float(np.percentile(s_all, 75) - np.percentile(s_all, 25))
    lo_t, hi_t = med - p["trim_iqr_k"] * iqr, med + p["trim_iqr_k"] * iqr
    st = s_all[(s_all >= lo_t) & (s_all <= hi_t)]
    cam_h = p["assumed_cam_h"] * float(np.median(st))
    res = {
        "schema": "neko.calib_hmd_anchor/v1",
        "run": str(run_dir), "cache": str(cache),
        "method": "t_ij_cam 来自 HMD 位姿差（与 OSC 链、深度链独立）",
        "cam_h_point_estimate": cam_h,
        "cam_h_iqr_lo": p["assumed_cam_h"] * float(np.percentile(st, 25)),
        "cam_h_iqr_hi": p["assumed_cam_h"] * float(np.percentile(st, 75)),
        "s_point_estimate": float(np.median(st)),
        "n_valid_pairs": int(len(s_all)), "n_trimmed_pairs": int(st.size),
        "failure_counts": cnt, "params": p,
        "hmd_span_m": {"x": span[0], "z": span[1]},
    }
    print()
    print("cam_h = %.4f m  (IQR %.4f..%.4f, n=%d)" % (
        cam_h, res["cam_h_iqr_lo"], res["cam_h_iqr_hi"], len(s_all)))
    print("对照 scale_calib：门 0.5 -> 1.3049 / 门 0.15 -> 1.4245")
    if a.out:
        outp = (REPO / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("写出：%s" % outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
