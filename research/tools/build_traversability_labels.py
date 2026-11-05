# -*- coding: utf-8 -*-
"""从原始双目 policy 录制生成局部可通行性标签。

标签是每个策略观测的 ``8 扇区 × 4 距离环``，取值为：

* ``1``：该扇区/距离环有足够双目观测且没有身体高度范围内的障碍点；
* ``0``：观测到障碍点；
* ``-1``：观测不足，保持 unknown，不伪造 free。

这条工具只生成离线训练字段，不写导航地图、不改变在线 SafetyGate。
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_mapping import disparity_range_px, make_sgbm, stereo_disparity

SECTORS = 8
RINGS = ((0.25, 0.75), (0.75, 1.25), (1.25, 2.25), (2.25, 4.0))
LABEL_VERSION = "traversability_v1_8x4"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stream_cmd(ff: str, path: Path, width: int, height: int,
                src_w: int, src_h: int) -> list[str]:
    crop_w = min(src_w, int(round(src_h * 16 / 9)))
    left = (src_w - crop_w) // 2
    return [ff, "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
            "-vf", f"crop={crop_w}:{src_h}:{left}:0:exact=1,"
                    f"scale={width}:{height}:flags=area", "-f", "rawvideo",
            "-pix_fmt", "gray", "-"]


def _classify(points: np.ndarray, *, min_points: int = 6) -> tuple[list[float], list[int], float | None]:
    """把 base 系点云归入 8 扇区/4 距离环。``points=[前,左,上]``。"""
    labels = np.full((SECTORS, len(RINGS)), -1.0, dtype=np.float32)
    counts = np.zeros_like(labels, dtype=np.int32)
    if points.size == 0:
        return labels.reshape(-1).tolist(), counts.reshape(-1).tolist(), None
    p = np.asarray(points, dtype=np.float32)
    forward, lateral, height = p[:, 0], p[:, 1], p[:, 2]
    radius = np.hypot(forward, lateral)
    bearing = np.arctan2(lateral, np.maximum(forward, 1e-6))
    # 相机可见的前方 ±90° 划成 8 个扇区；视野之外自然保持 unknown。
    sector = np.floor((bearing + np.pi / 2) / np.pi * SECTORS).astype(np.int32)
    visible = (sector >= 0) & (sector < SECTORS) & (forward > 0.0)
    # 眼高坐标下排除地面和天花板，保留会挡住身体的障碍带。
    obstacle_band = (height > -1.45) & (height < 0.85)
    nearest = None
    for si in range(SECTORS):
        for ri, (lo, hi) in enumerate(RINGS):
            sel = visible & (sector == si) & (radius >= lo) & (radius < hi)
            n = int(sel.sum())
            counts[si, ri] = n
            if n < min_points:
                continue
            blocked = int((sel & obstacle_band).sum())
            labels[si, ri] = 0.0 if blocked > 0 else 1.0
            if blocked > 0:
                d = float(radius[sel & obstacle_band].min())
                nearest = d if nearest is None else min(nearest, d)
    return labels.reshape(-1).tolist(), counts.reshape(-1).tolist(), nearest


def _resize_pair(left: np.ndarray, right: np.ndarray, meta: dict, size: tuple[int, int]):
    """按 policy 图像同一中心裁剪缩放，同时返回缩放后的内参。"""
    w, h = size
    src_h, src_w = left.shape[:2]
    content_h = min(h, int(round(w * 9 / 16)))
    crop_w = min(src_w, int(round(src_h * 16 / 9)))
    left_x = (src_w - crop_w) // 2
    l = left[:, left_x:left_x + crop_w]
    r = right[:, left_x:left_x + crop_w]
    l = cv2.resize(l, (w, content_h), interpolation=cv2.INTER_AREA)
    r = cv2.resize(r, (w, content_h), interpolation=cv2.INTER_AREA)
    sx, sy = w / crop_w, content_h / src_h
    fx = float(meta["fx"]) * sx
    fy = float(meta.get("fy", meta["fx"])) * sy
    cx = (float(meta["cx"]) - left_x) * sx
    cy = float(meta["cy"]) * sy
    return l, r, fx, fy, cx, cy


def build_session(raw: Path, dataset_session: Path, *, size: tuple[int, int],
                  min_points: int, max_frames: int = 0) -> dict:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("PATH 上没有 ffmpeg")
    meta = json.loads((raw / "meta.json").read_text(encoding="utf-8"))
    obs = _read_jsonl(dataset_session / "policy_obs.jsonl")
    wanted = {int(row["i"]): row for row in obs if row.get("i") is not None}
    if max_frames:
        wanted = dict(list(sorted(wanted.items()))[:max_frames])
    if not wanted:
        return {"session": raw.name, "samples": 0, "reason": "no_policy_obs"}
    src_w, src_h = map(int, meta["size"])
    width, height = size
    # 先缩放到 policy 宽高对应的内容高，避免每帧传输原始 1080×606 灰度。
    content_h = min(height, int(round(width * 9 / 16)))
    pl = subprocess.Popen(_stream_cmd(ff, raw / "L.mp4", width, content_h, src_w, src_h),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pr = subprocess.Popen(_stream_cmd(ff, raw / "R.mp4", width, content_h, src_w, src_h),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = width * content_h
    matcher = None
    out_rows = []
    idx = 0
    try:
        while True:
            lb = pl.stdout.read(frame_bytes)
            rb = pr.stdout.read(frame_bytes)
            if not lb or not rb or len(lb) != frame_bytes or len(rb) != frame_bytes:
                break
            idx += 1
            if idx not in wanted:
                continue
            left = np.frombuffer(lb, np.uint8).reshape(content_h, width)
            right = np.frombuffer(rb, np.uint8).reshape(content_h, width)
            # ffmpeg 已完成中心裁剪/缩放；按同一几何换算 K。
            crop_w = min(src_w, int(round(src_h * 16 / 9)))
            left_x = (src_w - crop_w) // 2
            fx = float(meta["fx"]) * width / crop_w
            cx = (float(meta["cx"]) - left_x) * width / crop_w
            cy = float(meta["cy"]) * content_h / src_h
            baseline = float(meta["baseline_tracking_m"])
            if matcher is None:
                matcher = make_sgbm(disparity_range_px(fx))
            disp = stereo_disparity(left, right, matcher)
            d = disp[::2, ::2]
            v, u = np.mgrid[0:disp.shape[0]:2, 0:disp.shape[1]:2]
            ok = d > 1.0
            z = fx * baseline / np.maximum(d[ok], 1e-6)
            keep = z <= RINGS[-1][1]
            z = z[keep]
            x = (u[ok][keep] - cx) * z / fx
            y = (v[ok][keep] - cy) * z / fx
            points = np.column_stack([z, -x, -y]).astype(np.float32)
            labels, counts, nearest = _classify(points, min_points=min_points)
            row = wanted[idx]
            out_rows.append({"t": row.get("t"), "i": idx,
                             "traversability": labels, "traversability_counts": counts,
                             "nearest_obstacle_m": nearest,
                             "label_version": LABEL_VERSION})
    finally:
        for p in (pl, pr):
            try:
                if p.stdout is not None:
                    p.stdout.close()
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    out = dataset_session / "traversability.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows),
                   encoding="utf-8")
    return {"session": raw.name, "samples": len(out_rows), "out": str(out),
            "label_version": LABEL_VERSION, "size": list(size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="原始 policy 录制根目录")
    ap.add_argument("--dataset", required=True, help="build_policy_dataset 输出根目录")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--size", nargs=2, type=int, default=[448, 252])
    ap.add_argument("--min-points", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)
    root, dataset = Path(a.root), Path(a.dataset)
    names = set(a.sessions) if a.sessions else None
    reports = []
    for d in sorted(p for p in dataset.iterdir() if p.is_dir()):
        if names and d.name not in names:
            continue
        raw = root / d.name
        if not (raw / "L.mp4").exists() or not (raw / "R.mp4").exists():
            print("跳过 %s：缺少双目视频" % d.name, flush=True)
            continue
        reports.append(build_session(raw, d, size=(int(a.size[0]), int(a.size[1])),
                                     min_points=int(a.min_points), max_frames=int(a.max_frames)))
    print(json.dumps(reports, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
