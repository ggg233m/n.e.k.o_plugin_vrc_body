#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Convert a navmesh recording (kf_img/*_L.png + *_R.png) into ORB-SLAM3
# stereo_euroc layout.
#
# Why: the earlier ORB-SLAM3 probe ran MONOCULAR at fx=320 on 640x360, and its
# trajectory collapsed to a 2.5 m box with 8 map resets (see
# .slam_probe/orbslam3_run/analysis.json). Monocular SLAM has no scale, so that
# failure says nothing about the algorithm. The data is stereo and has an
# absolute baseline, so stereo mode should work.
import json, sys
import numpy as np
from pathlib import Path
import cv2

ROOT = Path(r'H:/AI/neko-music/vrc/pc-vr/n.e.k.o_plugin_vrc_body')

_YAML = '''%YAML:1.0
# Converted from a navmesh recording by research/tools/make_orbslam3_stereo.py
# Images are the recorded stereo keyframes; both eyes are already rectified.
File.version: "1.0"

Camera.type: "Rectified"

Camera1.fx: {fx:.4f}
Camera1.fy: {fx:.4f}
Camera1.cx: {cx:.4f}
Camera1.cy: {cy:.4f}

Stereo.b: {baseline:.6f}

Camera.width: {width}
Camera.height: {height}

Camera.fps: {fps}

Camera.RGB: 0

Stereo.ThDepth: {th_depth}

ORBextractor.nFeatures: {features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
'''

def main():
    rec = sys.argv[1] if len(sys.argv) > 1 else '20261029_060258'
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = ROOT / '.slam_probe' / ('euroc_' + rec)
    R = ROOT / 'navmesh_recordings' / rec

    meta_p = R / 'meta.json'
    if not meta_p.exists():
        raise SystemExit('no meta.json')
    meta = json.loads(meta_p.read_text(encoding='utf-8'))
    cfg = meta.get('config', {})

    # 传感器参数从会话表里取（与录制一致）
    sj = ROOT / 'navmesh_memory' / 'wrld_home-7cf435ea' / 'sessions' / rec / 'session.json'
    fx = cx = cy = None; W = H = None; baseline = None
    if sj.exists():
        s = json.loads(sj.read_text(encoding='utf-8'))
        sen = s.get('sensors') or {}
        fx = sen.get('fx'); cx = sen.get('cx'); cy = sen.get('cy')
        size = sen.get('size') or []
        if len(size) == 2: W, H = int(size[0]), int(size[1])
        baseline = sen.get('baseline_m') or s.get('baseline_m')
    if fx is None:
        fx, cx, cy = 303.75, 540.0, 303.0
    if W is None:
        W, H = 1080, 606
    if baseline is None:
        baseline = 0.126
    print('[sensors] fx=%.2f cx=%.1f cy=%.1f %dx%d b=%.4f' % (fx, cx, cy, W, H, baseline))

    imgs = sorted((R / 'kf_img').glob('*_L.png'))
    imgs = imgs[::step]
    if not imgs:
        raise SystemExit('no kf_img')
    print('[imgs] %d frames (step=%d)' % (len(imgs), step))

    c0 = out / 'mav0' / 'cam0' / 'data'
    c1 = out / 'mav0' / 'cam1' / 'data'
    c0.mkdir(parents=True, exist_ok=True)
    c1.mkdir(parents=True, exist_ok=True)
    stamps = []
    for i, lp in enumerate(imgs):
        k = int(lp.stem.split('_')[0])
        rp = R / 'kf_img' / ('%06d_R.png' % k)
        if not rp.exists():
            continue
        L = cv2.imread(str(lp))
        Rr = cv2.imread(str(rp))
        if L is None or Rr is None:
            continue
        ns = int(k * 1e9 / 10.0)  # 关键帧名义 10 Hz
        stamps.append(ns)
        cv2.imwrite(str(c0 / ('%d.png' % ns)), L)
        cv2.imwrite(str(c1 / ('%d.png' % ns)), Rr)
        if (i + 1) % 200 == 0:
            print('  %d/%d' % (i + 1, len(imgs)), flush=True)
    (out / 'times.txt').write_text(''.join('%d\n' % s for s in stamps), encoding='ascii')
    y = _YAML.format(fx=fx, cx=cx, cy=cy, baseline=baseline, width=W, height=H,
                     fps=10.0, th_depth=40.0, features=1200)
    (out / 'VRChatStereo.yaml').write_text(y, encoding='utf-8')
    (out / 'meta.json').write_text(json.dumps({
        'rec': rec, 'n': len(stamps), 'fx': fx, 'cx': cx, 'cy': cy,
        'width': W, 'height': H, 'baseline_m': baseline,
    }, indent=2), encoding='utf-8')
    print('[out] %s  %d frames' % (out, len(stamps)))

if __name__ == '__main__':
    main()