# -*- coding: utf-8 -*-
"""航位推算的共享纯数学 —— 在线与离线的**唯一**实现。

## 为什么存在这个模块

旧的在线 ``online_pose.py`` 与离线 ``pose_graph.py``（均已删除）此前各写了一份
"把 avatar 本地位移按朝向旋进世界系"的公式。两份拷贝**已经漂移过一次**：

* 在线那份一直是对的（矢量位移）。
* 离线那份把 OSC 位移当成**标量路程**沿 yaw 前进 —— 等价于"把横移当成了
  前进"，只在 vx≈0 时正确。2026-09-24 实测该偏差造成终点差 1.419 m
  （当次横移报文仅占 2.9%）。

那次修正把离线对齐回在线，但**根因是"同一公式存了两份"**，没消除。本模块是
收敛的落地：公式只此一份。现在的调用方是 ``research/recorder/run_motion.py``、
``research/recorder/pose_graph.py``（离线航位推算）和
``research/tools/stereo_seq_ground_truth.py``（离线真值）。

## 坐标约定（勿改）

    local +Z = avatar 前进方向，local +X = avatar 右侧横移
    world_x =  dx_local * cos(y) + dz_local * sin(y)
    world_z = -dx_local * sin(y) + dz_local * cos(y)
    y = yaw_radians(yaw_deg, yaw_sign)

## yaw_sign 必须显式传入

在线与离线的 yaw 来源不同，符号**尚未统一**，所以本模块不设默认值 ——
强迫每个调用点写明它用的是哪一侧：

* ``OFFLINE_YAW_SIGN = -1.0`` —— 离线默认。由 ``yaw_sign_check.json`` 的画面
  证据标定（HMD 报的 yaw 方向与画面里实际转向相反）。
* ``ONLINE_YAW_SIGN = +1.0`` —— 在线当前行为，原样使用调用方 POST 的
  ``yaw_deg``。

⚠️ 在线的符号在本仓库内**既没有标定证据、也没有测试钉住**：
旧的 ``/worldmodel/navroute/feedback`` 由外部 POST ``yaw_deg``，正负约定从未在
本仓库钉住（该端点已删除）。在线 navmesh（``backend/nav_online.py``）不经过本
模块，直接用 HMD 朝向矩阵。

## 已知仍未收敛的副本（有意保留，勿照抄）

* ``research/recorder/scale_calib.py::_R_wc`` —— 同一旋转，
  但是 3×3 矩阵形式，用于相机→世界的三维落点，与二维航位推算形状不同。
* ``pointcloud_build.py`` / ``pointcloud_splat.py`` 里的 GL/JS 着色器旋转 ——
  是渲染坐标变换，不是航位推算。

## 边界

本模块是**纯数学**：不读文件、不依赖 numpy、不引入任何本项目其它模块。
这样在线（shipped 插件）与离线（``.slam_probe`` 开发树）都能用同一份，
且不会把任何一侧的依赖带进另一侧。
"""
from __future__ import annotations

import math

__all__ = [
    "OFFLINE_YAW_SIGN",
    "ONLINE_YAW_SIGN",
    "yaw_radians",
    "rotate_local_to_world",
    "advance",
]

# 离线侧默认符号：由 yaw_sign_check.json 的画面证据标定（见模块 docstring）。
OFFLINE_YAW_SIGN = -1.0

# 在线侧当前符号：原样使用外部 POST 的 yaw_deg，符号约定不在本仓库内。
# 这是未决项 —— 改它之前先补在线侧的符号证据。
ONLINE_YAW_SIGN = 1.0


def yaw_radians(yaw_deg: float, *, yaw_sign: float) -> float:
    """度 → 带符号的弧度。

    等价于 ``math.radians(yaw_deg) * yaw_sign``。``yaw_sign`` 为 ±1.0 时
    乘法是精确的，不引入舍入。
    """
    return math.radians(float(yaw_deg)) * float(yaw_sign)


def rotate_local_to_world(dx_local: float, dz_local: float, yaw_rad: float):
    """把 avatar 本地位移矢量旋进世界系，返回 ``(dx_world, dz_world)``。

    ``yaw_rad`` 必须是**已乘过 yaw_sign** 的弧度（用 :func:`yaw_radians` 得到）。

    本函数返回**位移**而不是就地累加，供"速度 × dt"这类调用方使用：
    在线 ``OnlinePose2D`` 传的是 ``(vx, vz)``，再自己乘 ``dt``。
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (dx_local * c + dz_local * s, -dx_local * s + dz_local * c)


def advance(x_m: float, z_m: float, dx_local: float, dz_local: float,
            yaw_rad: float):
    """从 ``(x_m, z_m)`` 走一步本地位移，返回新的 ``(x, z)``。

    与 :func:`rotate_local_to_world` 是同一公式，只是把累加放在内部，供
    "已经在网格上积好了本地位移"的调用方使用（离线 ``dead_reckon`` /
    ``build_edges`` 传的是 ``osc.disp()`` 的区间位移）。
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (x_m + dx_local * c + dz_local * s,
            z_m - dx_local * s + dz_local * c)
