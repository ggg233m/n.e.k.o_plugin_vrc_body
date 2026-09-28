# -*- coding: utf-8 -*-
"""把 ``rtabmap-report --poses_raw`` 导出的 TUM 轨迹转成 analyze_orbslam3_stereo.py 的输入。

两处换算：
- **时间**：prepare_rtabmap_euroc.py 给文件名加了 1e18 ns（= 1e9 s）偏移，这里减回去，
  输出 ns，与原序列 times.txt 同一零点。
- **坐标系**：RTAB-Map 输出 base 系位姿，ORB-SLAM3 输出的是左目光学系（z 前、y 下）。
  分析里的尺度只用距离和转角，与坐标系无关；但轨迹俯视图取 (x, z)，所以按
  ``T_cam = L⁻¹ · T_base · L`` 换回光学系。L = base→cam0 光学系，EuRoC 工具里是
  ``baseToImu · T_BS``（main.cpp:283）。T_BS 的旋转从同目录的
  ``rtabmap_calib_left.yaml``（``local_transform`` 就是 T_BS）读取：新序列是 z 朝上的
  base（L = opticalRotation），旧序列 T_BS=I 时 base 的竖直轴是 **y**。
  找不到标定文件时按旧的单位阵处理并提示。

用法
----
  .venv/Scripts/python.exe -W ignore research/tools/rtabmap_poses_to_euroc.py \
      .tmp/rtabmap_run3_640/inter/rtabmap_odom.txt .tmp/rtabmap_run3_640/inter/f_odom.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OFFSET_S = 1e9
# baseToImu（main.cpp:283），自逆。
BASE_TO_IMU = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def load_t_bs_rotation(calib: Path) -> np.ndarray | None:
    """读 rtabmap_calib_left.yaml 的 local_transform（3×4）旋转部分；读不到返回 None。"""
    if not calib.is_file():
        return None
    import re

    # RTAB-Map 写的是不带 !!opencv-matrix 标签的 rows/cols/data，cv2.FileStorage 读不了。
    m = re.search(r"local_transform:.*?data:\s*\[([^\]]*)\]", calib.read_text(encoding="utf-8"), re.S)
    if not m:
        return None
    vals = [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
    return np.array(vals, dtype=np.float64).reshape(3, 4)[:, :3] if len(vals) == 12 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("src", help="rtabmap-report --poses_raw 导出的 *_odom.txt / *_slam.txt")
    parser.add_argument("dst", help="EuRoC 格式输出：t_ns tx ty tz qx qy qz qw")
    parser.add_argument("--calib", default="",
                        help="rtabmap_calib_left.yaml（默认取 src 同目录）")
    args = parser.parse_args()

    calib = Path(args.calib) if args.calib else Path(args.src).parent / "rtabmap_calib_left.yaml"
    r_bs = load_t_bs_rotation(calib)
    if r_bs is None:
        print(f"未读到 {calib}，按旧序列 T_BS=I 处理")
        r_bs = np.eye(3)
    L = BASE_TO_IMU @ r_bs

    rows = []
    for line in Path(args.src).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) < 8:
            continue
        t, x, y, z, qx, qy, qz, qw = (float(v) for v in parts[:8])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        R_cam = L.T @ R @ L
        p_cam = L.T @ np.array([x, y, z])
        q = Rotation.from_matrix(R_cam).as_quat()
        t_ns = int(round((t - OFFSET_S) * 1e9))
        rows.append(f"{t_ns} {p_cam[0]:.6f} {p_cam[1]:.6f} {p_cam[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    Path(args.dst).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{args.dst}: {len(rows)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
