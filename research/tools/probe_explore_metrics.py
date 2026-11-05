# -*- coding: utf-8 -*-
"""探索/记忆型导航指标：用航向变化、曲率、分叉、回访替代横移比例。

为什么换指标（2026-11-05）：
  之前的 probe_data_coverage 用「横移 >=30%」判数据质量，这是**用摇杆式避障的
  标准去量自然探索数据**。VRChat 里人横向机动靠「转身+前进」，机体系横移天然接近 0。
  实测：机体系 |横移|/|前进| 中位 0.0000，但世界系航向每秒变 17 度、全场累计 1240 度。
  轨迹是弯的，只是弯曲发生在**航向**上而不是横移上。

本探针测量真正贴合「探索」的指标：
  turn_rate     航向变化率（deg/s）
  curvature     轨迹曲率（1/m，用航向变化/行进距离）
  fork_count    分叉选择（航向突变 >45 度且之前有停滞）
  stop_look     停止观察（速度<0.1 且 |omega|>0.3，即站定环顾）
  revisit       回访（世界系位置重访同一 1m 格子）
  coverage      新区域（首次进入的 1m 格子数 / 分钟）
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import build_policy_dataset as B

REC = Path("G:/vrc_recording")
GRID_M = 1.0

print("=" * 92)
print("  探索型导航指标（10 场自然录制）")
print("=" * 92)
print("  %-22s %8s %8s %7s %7s %8s %8s" % (
      "session", "turn/s", "curv", "fork", "look", "新格/分", "回访%"));

rows = []
for sess in sorted(p for p in REC.iterdir() if p.is_dir() and (p / "osc.jsonl").exists()):
    try:
        st = B.load_streams(sess)
        # build_dr_path 需要 streams["fwd"]，但它不在 load_streams 的返回值里，
        # 主流程是在 build_session 里单独注入的（build_policy_dataset.py:495）。
        st["fwd"] = B.hmd_forward(st["quat"])
        path = B.build_dr_path(st)
    except Exception as e:
        print("  %-22s 解析失败: %s" % (sess.name, e));
        continue
    t = path["t"]; p = path["p"]; s = path["s"]
    if len(t) < 200:
        continue
    ay_t = st.get("ay_t"); ay_v = st.get("ay_v")
    if ay_t is None or len(ay_t) == 0:
        continue
    w = ay_v * B.ANGULARY_RAD_PER_UNIT

    grid = np.arange(float(t[0]), float(t[-1]), 0.1)
    if grid.size < 100:
        continue
    j = np.clip(np.searchsorted(ay_t, grid, side="right") - 1, 0, len(w) - 1)
    wg = w[j]
    heading = np.concatenate([[0.0], np.cumsum(wg[:-1] * 0.1)])
    dur_min = float(t[-1] - t[0]) / 60.0

    # 曲率：航向变化 / 行进距离
    dhead = np.abs(np.diff(heading))
    dist = np.diff(s[:len(grid)]) if len(s) >= grid.size else np.diff(np.interp(grid, t, s))
    dist = np.maximum(dist, 1e-6)
    curv = float(np.median(dhead / dist))

    # 分叉：1 s 内航向变化 >45 度
    per_s = np.diff(heading[::10])
    forks = int((np.abs(per_s) > np.radians(45)).sum())

    # 停止观察：速度<0.1 且 |omega|>0.3
    vx = st["vx_m"]; vz = st["vz_m"]
    sp = np.hypot(vx, vz)
    sv_t = st["t_m"]
    sj = np.clip(np.searchsorted(sv_t, grid, side="right") - 1, 0, len(sp) - 1)
    spg = sp[sj]
    look = float(np.mean((spg < 0.1) & (np.abs(wg) > 0.3)))

    # 覆盖与回访：把世界系位置量化到 1 m 格子
    cell = np.floor(p / GRID_M).astype(np.int64)
    key = cell[:, 0] * 100000 + cell[:, 1]
    # ⚠️ 不能直接数「同一格子出现多次」：DR 轨迹每 20ms 一个点，站着不动也会
    # 在同一个格子里刷出上千个样本，回访率会恒等于 99%（第一版就这么错的）。
    # 正确做法是先把连续重复压掉，得到「格子访问序列」，再看序列里的重访。
    seq = [key[0]]
    for k in key[1:]:
        if k != seq[-1]:
            seq.append(k)
    seq = np.asarray(seq)
    uniq, counts = np.unique(seq, return_counts=True)
    new_cells = len(uniq)
    # 回访 = 访问序列里「非首次」的比例
    revisit = float(1.0 - len(uniq) / max(len(seq), 1))
    cells_per_min = new_cells / max(dur_min, 1e-6)

    rows.append((sess.name, np.degrees(np.mean(np.abs(wg))), curv, forks,
                 look, cells_per_min, revisit))
    print("  %-22s %7.1f %8.3f %7d %6.1f%% %8.1f %7.1f%%" % (
        sess.name, np.degrees(np.mean(np.abs(wg))), curv, forks,
        100 * look, cells_per_min, 100 * revisit))

if rows:
    a = np.array([r[1:] for r in rows], dtype=np.float64)
    print()
    print("  %-22s %7.1f %8.3f %7.1f %6.1f%% %8.1f %7.1f%%" % (
        "中位", np.median(a[:,0]), np.median(a[:,1]), np.median(a[:,2]),
        np.median(a[:,3]), np.median(a[:,4]), np.median(a[:,5])))
