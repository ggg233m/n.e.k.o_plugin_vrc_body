# -*- coding: utf-8 -*-
"""估计并回填「视频文件时间 → 遥测时间」的固定偏移。

原理（2026-09-21 定）：
  纯旋转下单应 H=K R K^-1；用标定 fx 从相邻帧对解出图像偏航角速度，
  与 HMD 四元数算出的偏航角速度做**带符号**滞后互相关，峰值即偏移。
  判据比旧的 MAD-vs-OSC 强得多（corr 0.6 vs 0.19），因为：
    - MAD 对转头、其他玩家、特效都敏感，不是速度代理；
    - 图像偏航角速度是同一物理量的直接测量。

## 抽帧走 ffmpeg，不要走 cv2.VideoCapture.read()
实测（1080p60，1200 帧窗口，本机 20 核，磁盘 205 MB/s）：
  OpenCV read() 全帧  71.6 s（59.7 ms/帧）   ← 原实现
  OpenCV grab() 全帧  64.4 s（53.6 ms/帧）
  ffmpeg 多线程        ~10× 更快（见 video_frames.py 文档：96.5 s → 10.2 s）
慢的不是 H.264 解码本身，而是 OpenCV 每帧构造 1920×1080 BGR Mat（6 MB）
的 swscale + 拷贝，而且它是**单线程**解码。所以：
  1. 抽帧统一走 recorder/video_frames.grab()（ffmpeg + .npy 缓存，跨脚本复用）；
  2. 每帧 ORB 只算一次（旧实现 prev 被重复计算，2× 浪费）。

用法：
  .venv/Scripts/python.exe research/tools/estimate_video_offset.py \
      --run .slam_probe/offline_probe/recorder/runs/<id> \
      --video "F:/obs/xxx.mkv" [--fx 385.6] [--stride 30] [--hw d3d11va] [--write]
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540  # 与 fx=385.6 的标定分辨率绑定，改动必须重新标定
RECORDER = Path(__file__).resolve().parent.parent / 'recorder'


def hmd_yaw_rate(run_dir: Path, anchor: float):
    """返回 (t_rel, yaw_rate_deg_s)；t_rel 相对 OBS 锚点（秒）。"""
    rows = []
    with open(run_dir / 'hmd_frames.jsonl', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    t = np.array([r['t'] for r in rows]) - anchor
    q = np.array([r['hmd']['rotation_xyzw'] for r in rows], dtype=float)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.degrees(np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z)))
    u = np.zeros_like(yaw)
    u[0] = yaw[0]
    for i in range(1, len(yaw)):
        d = (yaw[i] - yaw[i - 1] + 180.0) % 360.0 - 180.0
        u[i] = u[i - 1] + d
    dt = np.diff(t)
    ok = dt > 1e-6
    return t[1:][ok], (np.diff(u)[ok] / dt[ok])


def _load_video_frames():
    """按路径加载 recorder/video_frames.py（不在 tools 的 sys.path 上）。"""
    p = RECORDER / 'video_frames.py'
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location('_video_frames', p)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_frames_ffmpeg(video: str, sample_fps: float, hw: str | None, t_max: float | None):
    """ffmpeg 抽帧 + 缓存。返回 (frames[N,H,W] uint8, sample_fps, src_fps)。"""
    vf = _load_video_frames()
    if vf is None:
        raise RuntimeError('找不到 .slam_probe/offline_probe/recorder/video_frames.py')
    arr, meta = vf.grab(video, fps=sample_fps, size=(W, H), hw=hw, gray=True)
    if t_max:
        arr = arr[: int(t_max * sample_fps)]
    src_fps = float((meta.get('probe') or {}).get('fps') or sample_fps)
    return arr, sample_fps, src_fps


def load_frames_cv2(video: str, stride: int, t_max: float | None):
    """OpenCV 回退路径（ffmpeg 不可用时）。已改用 grab() 跳过未采样帧的像素拷贝。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('无法打开视频: %s' % video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if t_max:
        nfr = min(nfr, int(t_max * fps))
    out = []
    i = 0
    while i < nfr:
        if not cap.grab():        # grab 只解码，不做 swscale/BGR 拷贝
            break
        if i % stride == 0:
            ok, img = cap.retrieve()
            if ok:
                out.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (W, H)))
        i += 1
    cap.release()
    if not out:
        raise SystemExit('未取到任何帧')
    return np.stack(out), fps / stride, float(fps)


def yaw_from_pair(kpa, da, kpb, db, bf, Ki, K, min_inl=80, min_mask=60):
    """从两帧已算好的 ORB 特征解相对偏航角（度）。"""
    if da is None or db is None:
        return None
    m = bf.knnMatch(da, db, k=2)
    good = [p for p, q_ in (mm for mm in m if len(mm) == 2) if p.distance < 0.75 * q_.distance]
    if len(good) < min_inl:
        return None
    q1 = np.float32([kpa[p.queryIdx].pt for p in good])
    q2 = np.float32([kpb[p.trainIdx].pt for p in good])
    Hm, mask = cv2.findHomography(q1, q2, cv2.RANSAC, 3.0)
    if Hm is None or mask is None or mask.sum() < min_mask:
        return None
    M = Ki @ Hm @ K
    det = np.linalg.det(M)
    if det <= 0 or not np.isfinite(det):
        return None
    M = M / (det ** (1 / 3))
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return math.degrees(math.atan2(R[0, 2], R[0, 0]))


def image_yaw_rate(frames, sample_fps: float, fx: float, verbose=True):
    """相邻采样帧对的偏航角 → 角速度序列（度/秒）。每帧 ORB 只算一次。"""
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)

    t0 = time.perf_counter()
    feats = [orb.detectAndCompute(g, None) for g in frames]
    t_orb = time.perf_counter() - t0
    if verbose:
        print('      ORB %d 帧 %.1f s（%.1f ms/帧，每帧只算一次）'
              % (len(feats), t_orb, t_orb / max(len(feats), 1) * 1e3), flush=True)

    t0 = time.perf_counter()
    ts, rates = [], []
    dt = 1.0 / sample_fps
    for j in range(1, len(feats)):
        y = yaw_from_pair(feats[j - 1][0], feats[j - 1][1],
                          feats[j][0], feats[j][1], bf, Ki, K)
        if y is not None:
            ts.append((j - 0.5) * dt)
            rates.append(y / dt)
    if verbose:
        print('      配对 %d/%d 成功 %.1f s'
              % (len(ts), len(feats) - 1, time.perf_counter() - t0), flush=True)
    if not ts:
        raise SystemExit('所有帧对都解不出偏航角（素材太糊或 fx 不对）')
    return np.array(ts), np.array(rates)


def scan_lag(img_t, img_r, hmd_t, hmd_r, lo=-6.0, hi=6.0, step=0.05):
    lo_t = max(img_t[0], hmd_t[0])
    hi_t = min(img_t[-1], hmd_t[-1])
    grid = np.arange(lo_t, hi_t, 0.5)
    A = np.interp(grid, img_t, img_r)
    best, curve = None, []
    L = lo
    while L <= hi + 1e-9:
        B = np.interp(grid, hmd_t + L, hmd_r)
        c = float(np.corrcoef(A, B)[0, 1])
        curve.append((L, c))
        if best is None or c > best[1]:
            best = (L, c)
        L += step
    peak = best[1]
    band = [l for l, c in curve if c >= peak * 0.9]
    return best[0], peak, (min(band), max(band))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--fx', type=float, default=385.6, help='标定焦距(px @960x540)')
    ap.add_argument('--stride', type=int, default=15,
                    help='抽帧间隔(源帧)，60fps 下 15=4Hz（默认）。'
                         '注意 2Hz 会欠采样：实测同一 run 2Hz→+2.95/corr 0.493，'
                         '4Hz→+2.45/corr 0.758，快速转向被混叠 ⇒ 别调回 30')
    ap.add_argument('--sample-fps', type=float, default=None,
                    help='直接指定采样帧率，优先于 --stride')
    ap.add_argument('--hw', default=None, choices=[None, 'd3d11va', 'dxva2', 'qsv'],
                    help='ffmpeg 硬件解码（qsv 在本机报 -22，慎选）')
    ap.add_argument('--no-ffmpeg', action='store_true', help='强制走 OpenCV 回退路径')
    ap.add_argument('--t-max', type=float, default=None)
    ap.add_argument('--write', action='store_true', help='把结果写入 run.json')
    args = ap.parse_args()

    run_dir = Path(args.run)
    rj = json.load(open(run_dir / 'run.json', encoding='utf-8'))
    anchor = None
    for ev in rj.get('events', []) or []:
        if isinstance(ev, dict) and ev.get('obs_start_monotonic') is not None:
            anchor = float(ev['obs_start_monotonic'])
            break
    if anchor is None:
        anchor = rj.get('obs_start_monotonic')
    if anchor is None:
        raise SystemExit('run.json 缺少 obs_start_monotonic（events 里也没有）')

    T0 = time.perf_counter()
    print('[1/3] 抽帧 ...', flush=True)
    frames = sample_fps = src_fps = None
    if not args.no_ffmpeg:
        try:
            cap0 = cv2.VideoCapture(args.video)
            src = cap0.get(cv2.CAP_PROP_FPS)
            cap0.release()
            if args.sample_fps:
                sample_fps = args.sample_fps
            else:
                sample_fps = (src or 60.0) / args.stride
            frames, sample_fps, src_fps = load_frames_ffmpeg(
                args.video, sample_fps, args.hw, args.t_max)
            print('      ffmpeg: %d 帧 @ %.2f Hz（源 %.1f fps）'
                  % (len(frames), sample_fps, src_fps), flush=True)
        except Exception as exc:
            print('      ffmpeg 不可用（%s），回退 OpenCV（慢 ~7-10×）' % exc, flush=True)
            frames = None
    if frames is None:
        frames, sample_fps, src_fps = load_frames_cv2(args.video, args.stride, args.t_max)
        print('      OpenCV: %d 帧 @ %.2f Hz（源 %.1f fps）'
              % (len(frames), sample_fps, src_fps), flush=True)
    t_frames = time.perf_counter() - T0
    print('      抽帧耗时 %.1f s' % t_frames, flush=True)

    print('[2/3] 图像偏航角速度 ...', flush=True)
    it, ir = image_yaw_rate(frames, sample_fps, args.fx)
    print('      %d 点, %.1f~%.1f s' % (len(it), it[0], it[-1]), flush=True)

    print('[3/3] HMD 偏航角速度 + 滞后扫描 ...', flush=True)
    ht, hr = hmd_yaw_rate(run_dir, anchor)
    print('      HMD %d 点' % len(ht), flush=True)
    L, c, band = scan_lag(it, ir, ht, hr)

    # 约定：B(τ)=hmd(τ-L)，故 L<0 表示「文件时间 τ 对应遥测时间 τ+|L|」
    offset = -L
    print()
    print('最佳滞后 L = %+.2f s, corr = %.3f' % (L, c))
    print('=> 视频文件时间 τ  ≈  遥测时间 τ %+.2f s   (90%%峰值区间 %+.2f~%+.2f)'
          % (offset, -band[1], -band[0]))
    print('   lag=0 对照 corr = %.3f'
          % float(np.corrcoef(np.interp(np.arange(max(it[0], ht[0]), min(it[-1], ht[-1]), 0.5), it, ir),
                              np.interp(np.arange(max(it[0], ht[0]), min(it[-1], ht[-1]), 0.5), ht, hr))[0, 1]))
    print('   总耗时 %.1f s（抽帧 %.1f s）' % (time.perf_counter() - T0, t_frames))

    if args.write:
        rj['video_timebase'] = {
            'offset_s': round(float(offset), 2),
            'meaning': 'telemetry_time = video_file_time + offset_s',
            'method': 'image_yaw_rate x hmd_yaw_rate signed cross-correlation',
            'corr': round(float(c), 3),
            'band_s': [round(float(-band[1]), 2), round(float(-band[0]), 2)],
            'fx_used': args.fx,
            'sample_fps': round(float(sample_fps), 3),
            'frames_used': int(len(frames)),
        }
        json.dump(rj, open(run_dir / 'run.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print('已写入 %s -> video_timebase.offset_s = %+.2f' % (run_dir / 'run.json', offset))


if __name__ == '__main__':
    main()
