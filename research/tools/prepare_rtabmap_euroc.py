# -*- coding: utf-8 -*-
"""把 record_stereo_euroc.py 录的序列改造成 ``rtabmap-euroc_dataset`` 能吃的布局。

RTAB-Map 自带的 EuRoC 工具（research/tools/EurocDataset/main.cpp，0.23.8）对输入有三处
硬假设，和我们的录制格式对不上：

1. **文件名即时间戳，前 10 位是秒**（CameraImages.cpp 的 ``_filenamesAreTimestamps``
   分支：长度 >10 时取前 10 位作秒、其余作小数）。我们的文件名是从录制起点算的 ns，
   只有 8–10 位，会被整个读成"秒"。这里统一加 ``OFFSET_NS = 1e18``（= 1e9 s），
   让每个名字都是 19 位；分析时再减回去。
2. **必须有 mav0/imu0/data.csv 且至少一行**，否则直接 ``return -1``。我们没有 IMU。
   工具只把满足 ``t_imu - start + 1 > 0`` 的样本喂给里程计，所以这里只写一行、时间戳
   比首帧早 100 s——它会被读到并丢弃，**不会有假 IMU 进里程计**。
3. **每个相机一个 sensor.yaml**（T_BS / rate_hz / resolution / intrinsics /
   distortion_coefficients）。两眼已是校正好的平行对、无畸变，cam1 相对 cam0 只有
   +x 方向的基线平移；基线沿用追踪米（0.063），世界尺度 s 仍在分析时单独乘。

T_BS 不是单位阵（2026-09-26 起）
------------------------------
工具里相机外参是 ``baseToImu · T_BS``（main.cpp，``baseToImu = (0,0,1 / 0,-1,0 / 1,0,0)``，
它的逆就是它自己）。T_BS 取单位阵时 base 的**竖直轴是 y**，而 RTAB-Map 的 2D 占据栅格
（Grid/MaxObstacleHeight、MinGroundHeight、投影平面）默认 z 朝上——在 run4 上直接画出
一条条假障碍。所以默认让 ``baseToImu · R_BS = opticalRotation``（光学 z 前→base x，
光学 x 右→base −y，光学 y 下→base −z），即 base 为 ROS 习惯的 x 前 y 左 z 上：

    R_BS = [[0,-1,0],[1,0,0],[0,0,1]]

双目外参只取决于 ``R_BSᵀ · t``，所以 cam1 的平移写成 ``R_BS · (b,0,0) = (0,b,0)``，
工具算出的基线仍是 b。``--legacy-y-up`` 保留旧的单位阵，便于复现旧结果。
``research/tools/rtabmap_poses_to_euroc.py`` 会读输出目录里的 rtabmap_calib_left.yaml 自动适配。

图像用**硬链接**，不拷贝也不改原序列（原目录的 times.txt / osc.npy / meta.json
原样复制过去，供分析脚本使用）。

用法
----
  .venv/Scripts/python.exe -W ignore research/tools/prepare_rtabmap_euroc.py \
      .slam_probe/stereo_seq/run3_640 .slam_probe/stereo_seq/run3_640_rtabmap
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

OFFSET_NS = 10**18

SENSOR_YAML = """\
# N.E.K.O：由 research/tools/prepare_rtabmap_euroc.py 从 {src} 生成，供 rtabmap-euroc_dataset 读取。
sensor_type: camera
comment: VRChat SteamVR mirror stereo ({eye}), rectified, no distortion
T_BS:
  cols: 4
  rows: 4
  data: [{r[0][0]:.1f}, {r[0][1]:.1f}, {r[0][2]:.1f}, {t[0]:.6f},
         {r[1][0]:.1f}, {r[1][1]:.1f}, {r[1][2]:.1f}, {t[1]:.6f},
         {r[2][0]:.1f}, {r[2][1]:.1f}, {r[2][2]:.1f}, {t[2]:.6f},
         0.0, 0.0, 0.0, 1.0]
rate_hz: {rate}
resolution: [{w}, {h}]
camera_model: pinhole
intrinsics: [{fx:.6f}, {fy:.6f}, {cx:.6f}, {cy:.6f}]
distortion_model: radial-tangential
distortion_coefficients: [0.0, 0.0, 0.0, 0.0]
"""

#: base 为 x 前、y 左、z 上时的 T_BS 旋转（推导见模块文档）。
R_BS_Z_UP = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
R_BS_LEGACY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _any_eye_blank(src: Path, ns: int) -> bool:
    import cv2

    for cam in ("cam0", "cam1"):
        img = cv2.imread(str(src / "mav0" / cam / "data" / f"{ns}.png"), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if img is None or not img.any():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("src", help="record_stereo_euroc.py 的输出目录")
    parser.add_argument("dst", help="输出目录（不能已存在）")
    parser.add_argument("--legacy-y-up", action="store_true",
                        help="T_BS 用单位阵（旧行为：base 竖直轴为 y，2D 栅格不可用）")
    args = parser.parse_args()
    r = R_BS_LEGACY if args.legacy_y_up else R_BS_Z_UP

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        parser.error(f"{dst} 已存在")
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    frame_ns = [int(line) for line in (src / "times.txt").read_text().split()]
    blank = [ns for ns in frame_ns if _any_eye_blank(src, ns)]
    if blank:
        # run5_ipd126 首帧右眼全黑：RTAB-Map 用它初始化会拒掉全部双目对应，下一帧段错误。
        print(f"跳过 {len(blank)} 帧任一眼全黑的图像：{blank[:5]}")
        frame_ns = [ns for ns in frame_ns if ns not in set(blank)]

    for cam, tx in (("cam0", 0.0), ("cam1", float(meta["baseline_tracking_m"]))):
        out = dst / "mav0" / cam / "data"
        out.mkdir(parents=True)
        for ns in frame_ns:
            os.link(src / "mav0" / cam / "data" / f"{ns}.png", out / f"{ns + OFFSET_NS}.png")
        (dst / "mav0" / cam / "sensor.yaml").write_text(SENSOR_YAML.format(
            src=src.as_posix(), eye=cam, r=r, t=[r[i][0] * tx for i in range(3)], rate=int(round(meta["rate_hz"])),
            w=meta["width"], h=meta["height"],
            fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"]), encoding="utf-8")

    imu = dst / "mav0" / "imu0"
    imu.mkdir(parents=True)
    dummy_ns = OFFSET_NS + frame_ns[0] - 100 * 10**9
    (imu / "data.csv").write_text(
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
        "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        f"{dummy_ns},0,0,0,0,0,9.81\n", encoding="utf-8")

    (dst / "times.txt").write_text("".join(f"{ns}\n" for ns in frame_ns), encoding="ascii")
    for name in ("osc.npy", "meta.json"):
        shutil.copy2(src / name, dst / name)
    meta_out = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    meta_out["rtabmap_adapted_from"] = {"dir": str(src), "offset_ns": OFFSET_NS,
                                        "blank_frames_skipped": blank}
    (dst / "meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{dst}: {len(frame_ns)} 帧 ×2（硬链接），sensor.yaml ×2，占位 imu0 1 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
