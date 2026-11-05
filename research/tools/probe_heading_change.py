# -*- coding: utf-8 -*-
"""H3: 世界系航向变化 —— 机体系横移小不等于轨迹直。

复用 build_policy_dataset.load_streams 解析（OSC 是长格式事件流，
按 addr 分行，直接按 t 读会拿错字段）。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import build_policy_dataset as B

REC = Path("G:/vrc_recording")
dirs = sorted(p for p in REC.iterdir() if p.is_dir() and (p / "osc.jsonl").exists())
print("[%d 场]" % len(dirs))
print()
print("=" * 78)
print("  世界系航向变化（AngularY 积分）")
print("=" * 78)

rows = []
for sess in dirs:
    try:
        st = B.load_streams(sess)
    except Exception as e:
        print("  %-22s 解析失败: %s" % (sess.name, e))
        continue
    t = st["t_m"]
    if len(t) < 100:
        continue
    ay_t = st.get("ay_t")
    ay_v = st.get("ay_v")
    if ay_t is None or len(ay_t) == 0:
        print("  %-22s 无 AngularY 通道" % sess.name)
        continue
    w = ay_v * B.ANGULARY_RAD_PER_UNIT
    t0, t1 = float(t[0]), float(t[-1])
    grid = np.arange(t0, t1, 0.1)
    if len(grid) < 20:
        continue
    # ZOH 到 10 Hz 网格
    j = np.searchsorted(ay_t, grid, side="right") - 1
    j = np.clip(j, 0, len(w) - 1)
    wg = w[j]
    heading = np.concatenate([[0.0], np.cumsum(wg[:-1] * 0.1)])
    # 每 1 s 的航向变化绝对值
    per_s = np.abs(np.diff(heading[::10]))
    if per_s.size == 0:
        continue
    vx = st["vx_m"]; vz = st["vz_m"]
    sp = np.hypot(vx, vz)
    rows.append((sess.name, float(np.mean(sp > 0.2)),
                 float(np.degrees(np.median(per_s))),
                 float(np.degrees(per_s.max())),
                 float(np.degrees(np.abs(heading).max()))))

print("  %-22s %8s %12s %12s %12s" % ("session", "运动占比", "航向变化/s", "最大/s", "总转角"));
for n, mv, med, mx, tot in rows:
    print("  %-22s %7.1f%% %10.1f deg %10.1f deg %10.1f deg" % (n, 100*mv, med, mx, tot));

if rows:
    print()
    print("  中位: 航向变化/s = %.1f deg, 总转角 = %.0f deg"
          % (np.median([r[2] for r in rows]), np.median([r[4] for r in rows])));
    print()
    print("  判读：若航向变化/s 明显大于 0，说明轨迹是弯的；");
    print("        机体系横移 0.5%% 只说明「横向机动由转身实现」，不是没能力。");
