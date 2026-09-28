# -*- coding: utf-8 -*-
"""把已录好的 EuRoC 双目序列整段降分辨率，用来做**受控**的分辨率对比。

为什么需要它
------------
两次独立录制的差异里，分辨率和"人当时怎么走的"是纠缠在一起的：run1（6.4 Hz、
640×360）覆盖 93.1%，run2_15hz（15 Hz、640×360）只有 31.1%，但后者丢追前的转速
是 313/483 °/s，前者是 33 °/s——运动本身就不是一回事，这种对比说明不了分辨率。

唯一干净的做法是**同一批帧**只改分辨率：时间戳、OSC、运动内容全部逐字节沿用，
只把图像缩小并按比例改内参。这样两次 ORB-SLAM3 的差异就只能来自分辨率。

注意缩放引入的模糊是**被测对象的一部分**，不是瑕疵：真实管线里 640×360 也正是从
2880×1620 经一次非整数重采样得到的，而 720×405 是 mip 2 整数折半。这里从 720 缩到
640 复现的就是那一次非整数重采样。

用法
----
  .venv/Scripts/python.exe -W ignore research/tools/downscale_stereo_seq.py \
      .slam_probe/stereo_seq/run3_720p .slam_probe/stereo_seq/run3_640 --width 640
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("src", help="源序列目录")
    parser.add_argument("dst", help="输出序列目录")
    parser.add_argument("--width", type=int, required=True, help="目标宽度")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    src_w, src_h = int(meta["width"]), int(meta["height"])
    if args.width >= src_w:
        print(f"目标宽度 {args.width} 不小于源宽度 {src_w}，只能降不能升。")
        return 2
    # 保持源的长宽比，避免 cx/cy 与 fx/fy 的比例关系被破坏。
    height = int(round(src_h * args.width / src_w))
    sx, sy = args.width / src_w, height / src_h

    cam0 = dst / "mav0" / "cam0" / "data"
    if cam0.exists() and any(cam0.iterdir()):
        print(f"{cam0} 已有图像，拒绝覆盖。换一个 dst。")
        return 2

    stamps = (src / "times.txt").read_text().split()
    size = (args.width, height)
    for cam in ("cam0", "cam1"):
        out_dir = dst / "mav0" / cam / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, ns in enumerate(stamps):
            img = cv2.imread(str(src / "mav0" / cam / "data" / f"{ns}.png"), cv2.IMREAD_COLOR)
            if img is None:
                print(f"读不到 {cam}/{ns}.png")
                return 2
            cv2.imwrite(str(out_dir / f"{ns}.png"),
                        cv2.resize(img, size, interpolation=cv2.INTER_AREA))
            if i % 200 == 0:
                print(f"  {cam} {i}/{len(stamps)}", flush=True)

    # 时间戳与 OSC 原样复制：运动内容必须逐字节一致，否则对比又不受控了。
    shutil.copy2(src / "times.txt", dst / "times.txt")
    shutil.copy2(src / "osc.npy", dst / "osc.npy")

    fx, fy = meta["fx"] * sx, meta["fy"] * sy
    cx, cy = meta["cx"] * sx, meta["cy"] * sy
    yaml = (src / "VRChatStereo.yaml").read_text(encoding="utf-8")
    for key, value in (("Camera1.fx", fx), ("Camera1.fy", fy),
                       ("Camera1.cx", cx), ("Camera1.cy", cy)):
        yaml = _replace(yaml, key, f"{value:.4f}")
    yaml = _replace(yaml, "Camera.width", str(args.width))
    yaml = _replace(yaml, "Camera.height", str(height))
    (dst / "VRChatStereo.yaml").write_text(yaml, encoding="utf-8")

    meta.update(width=args.width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
                downscaled_from={"dir": str(src), "width": src_w, "height": src_h})
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\n已写入 {dst}：{len(stamps)} 帧 {args.width}x{height}，"
          f"fx={fx:.2f} cx={cx:.2f} cy={cy:.2f}（源 {src_w}x{src_h} fx={fx / sx:.2f}）")
    return 0


def _replace(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    hit = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            lines[i] = f"{key}: {value}"
            hit = True
    if not hit:
        raise SystemExit(f"YAML 里找不到 {key}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
