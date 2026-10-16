# -*- coding: utf-8 -*-
r"""同一录制、换位姿源导出稠密点云 HTML，肉眼看拖影（按时间着色：墙面分层成彩虹 = 拖影）。

位姿源同 grid_pose_ab.poses_for：dr / loop / <rec 下的 npz 名，如 vo_g6> / dr+N。
点云口径同 pointcloud_view.load_session（水平 range 闸、cam_h−gc 地面口径、体素质心）。

    .venv/Scripts/python.exe research/tools/smear_view.py 20261001_044153 --src dr loop vo_g6
输出 .tmp/pointcloud/smear_<rec>_<src>.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pointcloud_view as V                      # noqa: E402
from grid_pose_ab import poses_for               # noqa: E402

ROOT = V.ROOT


def cloud(rec: Path, pose_of: dict, cam_h: float, vox: float, z_min: float, z_max: float, range_m: float):
    chunks, tt = [], []
    for kf in sorted(rec.glob("kf/*.npz"), key=lambda p: int(p.stem)):
        k = int(kf.stem)
        T = pose_of.get(k)
        if T is None:
            continue
        with np.load(kf) as z:
            p = np.asarray(z["pts"], np.float32)
            dm = float(np.asarray(z["dist_m"]).reshape(-1)[0]) if "dist_m" in z.files else float(k)
        p = p[np.isfinite(p).all(1) & (np.hypot(p[:, 0], p[:, 1]) <= range_m)]
        if not len(p):
            continue
        gc = V.frame_ground_offset(p, T, cam_h)
        pw = p @ T[:3, :3].T
        xy = pw[:, :2] + T[:2, 3]
        zw = pw[:, 2] + cam_h - gc
        s = (zw > z_min) & (zw < z_max)
        chunks.append(np.column_stack([xy[s], zw[s]]))
        tt.append(np.full(int(s.sum()), dm, np.float32))
    P = np.concatenate(chunks)
    t = np.concatenate(tt)
    t = np.clip(t / max(float(t.max()), 1e-6), 0, 1)
    return V.voxelize(P, np.zeros(len(P), np.uint8), vox, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--src", nargs="+", default=["dr", "loop", "vo_g6"])
    ap.add_argument("--vox", type=float, default=0.04)
    ap.add_argument("--z-min", type=float, default=0.3, help="默认只看 0.3 m 以上（去地板，墙面分层更清楚）")
    ap.add_argument("--z-max", type=float, default=2.2)
    ap.add_argument("--range-m", type=float, default=5.0)
    a = ap.parse_args()
    rec = ROOT / "navmesh_recordings" / a.rec
    cam_h = V.estimate_cam_h(rec)
    out_dir = ROOT / ".tmp" / "pointcloud"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in a.src:
        P, _s, t = cloud(rec, poses_for(rec, src), cam_h, a.vox, a.z_min, a.z_max, a.range_m)
        sess = [{"name": f"{a.rec[-6:]} · 位姿={src}", "n": len(P)}]
        # 单场：会话通道全 0，时间通道 = 本场累积路程 → "按时间"才是这场的本意。
        html = V.build_html(P, np.zeros(len(P), np.uint8), sess, a.vox,
                            f"拖影目视 · {a.rec} · {src}", default_mode=2,
                            tval=(t * 254).astype(np.uint8))
        out = out_dir / f"smear_{a.rec}_{src}.html"
        out.write_text(html, encoding="utf-8")
        print(f"[out] {out}  {len(P):,} 点  {out.stat().st_size/1048576:.1f} MB  cam_h={cam_h:.3f}")


if __name__ == "__main__":
    main()
