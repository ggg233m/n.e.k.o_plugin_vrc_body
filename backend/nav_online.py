# -*- coding: utf-8 -*-
"""在线增量 navmesh：航位推算位姿 + 镜像双目关键帧 → NavSession → OSC 走位 / HMD 转向。

三个频率各管各的（别混成一个"1 Hz"）。**下面每一条都区分「目标」与「实测」**——
2026-10-18 实测 223033（720×405）之前，第 2 行那句"~10 Hz"一直没被核过：

* 位姿：目标 ``pose_hz = 20``，**实测 15.87 Hz**（每轮 ~63 ms，超出 50 ms 预算）。它是另外
  两条的**上游**——感知循环是单线程，立体块内联在里面。
* 双目：目标 ``stereo_period_s = 0.1`` ⇒ 10 Hz，**实测 7.94 Hz**。原因不是 SGBM 慢
  （20 ms/帧，余量 3 倍），而是门写在 ``tick >= next_stereo`` 上、被**向上量化到整数个
  位姿 tick**：0.1 / 0.063 = 1.59 ⇒ 只能等到第 2 个 tick = 0.126 s。既做近距急停，也按
  **距离/转角/时长**触发关键帧（``KeyframePolicy``：快转推迟、原地补帧替换）。
* 关键帧：**实测全程平均 1.86 Hz**，但它不是周期量——跑动时 46.1% 的间隔卡在双目台阶
  （7.94 Hz），原地时落到 0.3–0.7 Hz。
* 地图更新：有新关键帧才栅格化 + 重规划，``map_min_interval_s = 0.3`` ⇒ 上限 3.3 Hz
  （栅格化已是增量，853 帧约 50 ms）。
* 控制：目标 ``control_hz = 10``：跟随器吃最新位姿，每拍重发带 400 ms 自动过期的轴
  （线程死了人就停）。

``status()["rates"]`` 把上面三条的**实测值**与目标并排报出来（``stereo_ticks`` 是关键：
≈整数 ⇒ 撞在量化台阶上；明显非整数 ⇒ 才是真算不过来）。见 ``RateMeter``。

位姿源 = 航位推算 + 回环（``nav_loop.LoopCloser``：ORB+PnP，只修平移，朝向来自 HMD）。
回环只在重访时出现；两次回环之间仍是纯航位推算（实测漂移约 2.8% 路程）。status 如实报 ``pose_source``。

坐标：地图系 = SteamVR 站立系经 C 换轴（x 前 y 左 z 上），原点 = 会话起点，追踪米；
对外（goto / status）一律导航系世界米 = 追踪米 × world_scale。
"""
from __future__ import annotations

import base64
import json
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

# 共享节拍器。为什么需要：Windows 上 threading.Event.wait 被量化到 15.625 ms 的
# 系统时钟节拍，请求 49.7 ms 实测 62.26 ms；time.monotonic 本身也是 GetTickCount64()，
# 读数台阶 16.000 ms。两者叠加让 pose_hz=20 的循环实测只有 16.1 Hz（gap p50 62 ms）。
# Pacer 用绝对截止点 + time.sleep（高精度 waitable timer，不受 tick 影响）+ 1.5 ms
# 自旋收尾，实测 |err|p99 = 0.014 ms、spin_frac 2.32%。详见 backend/pacing.py。
from .nav_grid import FREE, OCC, UNK
from .nav_loop import LoopCloser, LoopConfig, extract_features
from .nav_memory import NavMemoryStore, SessionWriter, make_thumbnail
from .nav_prior import L_FREE, L_OCC, PriorConfig, PriorConsumer
from .nav_xsession import XSessionConfig, XSessionTracker, align_into_auto, align_world_tree
from .nav_mapping import KeyframeGridMapper, MapperConfig, NavSession, stereo_points
from .pacing import Gate, Pacer, sleep_until
from .stereo_backend import las2_prewarm, las2_prewarm_status, make_stereo_backend

# 列 = base 的 x/y/z 轴在 SteamVR 站立系（x 右 y 上 z 后）中的坐标。
C_BASE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def hmd_to_base_rotation(r_hmd: np.ndarray) -> np.ndarray:
    return C_BASE.T @ np.asarray(r_hmd, float).reshape(3, 3) @ C_BASE


@dataclass
class OnlineNavConfig:
    # 世界米 = 追踪米 × world_scale（随 avatar 变）。唯一真值源：__post_init__ 把它推给 mapper / loop，
    # 两边各写一份时以这里为准（改 mapper.world_scale 不会生效，别在那边改）。
    world_scale: float = 0.755
    # world_scale 跟随头像缩放：开会话时把 world_scale 乘 本场眼高 ÷ 这个参考眼高（0.755 标定时的眼高）。
    # 2026-10-14 同世界同路线 A/B（默认 1.2577 vs 1.40）：OSC 世界速度不随缩放、双目 IPD 随缩放 ⇒
    # 只有双目单位→米这一侧要乘；DR 的 k 不动。眼高取 EyeHeightAsMeters → /avatar/eyeheight →
    # ScaleFactor（最后这个只对标定用的 avatar 成立）；都没收到就按 1 不改。0 = 关闭（固定 world_scale）。
    avatar_ref_eye_height_m: float = 1.2577
    # 启动自检：实测双目基线与期望差超过 tol 就报 baseline_mismatch（不停机：深度按实测基线算，几何仍对，
    # 只是噪声变了；但跨会话记忆不能混用）。0.126 = 自编驱动；发行版驱动是 0.063。0 表示不检查。
    expected_baseline_m: float = 0.126
    baseline_tol_m: float = 0.01
    osc_lag_s: float = 0.18          # OSC 速度比画面晚到约 0.18 s（run5/run6 局部窗口回放扫描最优 0.175–0.2）
    # OSC VelocityX/Z 乘这个系数再积分（米 = k × ∫|OSC|）。
    # 1.0（2026-10-14 由 0.85 改回）：两种不依赖初值、不按结果筛选的视觉法都给 k≈1.0 ——
    # ORB des3d+RANSAC 六场逐对中位 0.93–1.08、前进/后退/横移/斜向各档无系统差；稠密光流去失败对 0.99–1.02。
    # 旧 0.85 的两条出处已不成立：墙 ICP 的 rv 被初值锚住；run8_ruler 尺子 1.16–1.25 是起停瞬态里
    # k/滞后/τ 简并的产物（稳速穿越 0.98）。栅格指标随 k 单调、按面积归一后也选不出 k。
    # 见 Docs/FACTS.md 与 Docs/OSC速度尺度与回访拖影（2026-10-12）.md §2.10、§六。
    osc_speed_scale: float = 1.0
    # 非零速度的 ZOH 外推上限：VelocityX/Z 无心跳，静默只在短时间内等于"速度没变"
    # （实测需要外推非零速度的空档全部 ≤ 2.03 s）。静默超过 osc_zoh_max_s 后在 osc_zoh_fade_s
    # 内线性淡出到 0，避免丢掉"停下"那个 0 包时把最后一个速度永远积分下去（静止漂移 + 路程虚涨）。
    # 任一项为 0 = 不封顶（回到旧行为，只在明知道自己在赌时才这么配）。
    osc_zoh_max_s: float = 2.5
    osc_zoh_fade_s: float = 1.0
    # 航位推算的朝向取样：True = 每个积分子段按**该段中点**的 HMD 朝向旋转；
    # False = 旧行为（整批样本共用本帧朝向，系统性"偏超前"，转身时把早先那段位移多转一点）。
    # VelocityX/Z 是角色局部系（实测 atan2(vx,vz) 尖峰钉在 0°/±90°/180°，与 HMD 航向圆相关 ≤0.19），
    # 所以必须乘**被测量那一刻**的朝向。解析量级 ½·v·ω·ΔT·δ ≈ 8 cm/0.5 s（跑步 + 转身）；
    # 离线 A/B 见 .tmp/dr_lag_sweep3.py 的 end vs exact 两列（方向差 14.0°→12.2°）。
    # 关掉它只为对照，不是修复方案。
    dr_heading_midpoint: bool = True
    pose_hz: float = 20.0
    # 每帧双目只看得到脚前 ~1.3 m（世界米）到 range_m 那一条带：3 Hz 时跑步一帧走 1.3 m，
    # 带与带之间就是空白。SGBM 720×405 约 26 ms，10 Hz 占一个核的 1/4，近距急停也跟着变快。
    stereo_period_s: float = 0.1
    # 双目**解耦**（2026-10-24）：把"算视差 + 点云 + 近距判定"从位姿线程挪到独立线程。
    # 为什么必须拆：块内联在位姿 tick 里，块耗时直接吃位姿预算。实测（A770）
    #   SGBM 720 ~21 ms / LAS2 编译版 720 ~51 ms，而位姿 tick 预算 50 ms（pose_hz=20）。
    #   内联 ⇒ 位姿被压到约 11 Hz；拆开 ⇒ 位姿不再被拖，实测 18.79 → 18.78 Hz（零损失）。
    # ⚠️ 采集（read_stereo）仍留在位姿线程：D3D11 立即上下文不是线程安全的，
    #    OpenVRSensors 的方法只能在感知线程里调（见 OpenVRSensors 文档串）。
    decoupled: bool = True
    # 视差后端（backend/stereo_backend.py）："las2"（LAS2-M，XPU）或 "sgbm"（CPU）。
    # las2 载不起来（发行包 / `.venv` 没有 torch、没有 XPU、缺权重）⇒ errors 里记 stereo_backend、退回 sgbm。
    # las2 首帧要编译 70–330 s，期间没有双目 ⇒ 控制按 stereo_stale 停车，不会盲走。
    # 实测（A770，部署块 p50）：720 档 51 vs SGBM 21 ms；1080 档 58.7 vs 64.6 ms；
    # 20261025_202630 真机（VRChat 同跑）SGBM 1080 档 85.9 ms，队列满丢了 16% 关键帧决策。
    # ⚠️ las2 编译需 MSVC cl + PYTHONUTF8=1（_tc_run314.bat）；纯 `.venv` 启动会退回 sgbm。
    stereo_backend: str = "las2"
    # 后端启动时在后台线程预热 LAS2（``prewarm``）：热缓存下首帧也要 ~80 s，冷编译 ~5 min。
    # 预热尺寸按 capture_width 猜（见 ``prewarm_shape``）；猜错档只是会话首帧照旧再编一次。
    stereo_prewarm: bool = True
    stereo_min_period_s: float = 0.02    # 解耦线程最小间隔：算得比这快也要让出 CPU 给位姿线程    # 双目采集宽度下限（像素）。走 mip 链：实际输出取仍不小于它的最深一级。
    # ⚠️ 实际分辨率由镜像纹理决定：镜像 = AnyaDance 驱动 render × 1.5（SteamVR 滑块不改它）。
    #   驱动 1440×810（现行）⇒ 镜像 2160×1212 ⇒ 1080 = mip1 1080×606 / fx 303.75；
    #   驱动 1920×1080（旧）⇒ 镜像 2880×1620 ⇒ 1080 拿到 mip1 1440×810 / fx 405、720 = mip2 720×405。
    # 深度尺度不随它变（fx 与视差同比例变）；SGBM 视差范围按 fx 自动缩放（720/1080/1440 档 nd 64/96/128）。
    # 1080 档部署块 p50：LAS2 58.7 ms、SGBM 64.6 ms（720 档 51 / 21 ms），都在 10 Hz 预算内。
    capture_width: int = 1080
    control_hz: float = 10.0
    map_min_interval_s: float = 0.3  # 栅格化已是增量（853 帧 ~50 ms），且不再占着导航锁
    kf_dist_m: float = 0.4           # 世界米；< 条带深度，快走也首尾相接
    kf_turn_deg: float = 25.0
    kf_max_age_s: float = 3.0        # 原地不动时也隔这么久补一帧（刷新动态物体、给射线清除新视线）
    # 原地不动（离上一关键帧 < still_m 且转角 < still_deg）补的帧**替换**上一帧的点云，不叠加：
    # 同一视角叠 N 份只会把深度噪声放大 N 倍。21 min 录制 3014 帧里约 43% 是这种帧。
    kf_still_m: float = 0.1
    kf_still_deg: float = 10.0
    # 快速转头时推迟关键帧：HMD 朝向与画面有时差，转得快点云就被甩歪。兜底：推迟超过 defer_max_s
    # 或期间走了 defer_max_m 就照取，原地连转不会整段没帧、边走边转不会断带。
    # 2026-09-30 回放（替换+推迟+β=1 合计）：21 min 录制路径上障碍格 251→13、可走 221→265 m²；
    # run5/run6 路径障碍仍为 0，可走面积持平（30°/s 阈值会让 run6 丢 5 m²，故取 45）。
    kf_defer_dps: float = 45.0
    kf_defer_max_s: float = 1.0
    kf_defer_max_m: float = 0.8
    trail_period_s: float = 0.2
    pose_stale_s: float = 0.5
    stereo_stale_s: float = 0.8
    dr_sigma_m: float = 0.2          # 相对**自身地图**的误差，不是全局误差
    move_hold_ms: int = 400
    stop_ahead_m: float = 0.7        # 世界米；近距急停检测区
    stop_min_pts: int = 40
    radius_m: float = 0.25
    yaw_jump_dps: float = 400.0      # 调度器转向上限 360°/s，留点余量
    yaw_jump_min_deg: float = 10.0
    mapper: MapperConfig = field(default_factory=MapperConfig)
    # 跨会话地点检索（P0）：世界索引装载/查询/约束写回。确认对只写约束不碰位姿图（隔离层）。
    xsession: XSessionConfig = field(default_factory=XSessionConfig)
    # 世界先验消费（P0.3b）：会话中攒够跨会话约束 ⇒ 在线估 gauge ⇒ 注入 <world>/prior/
    # 的三态先验（产出见 research/tools/offline_fusion.py --write-prior）。默认开，带质量闸；
    # 没有先验文件 / gauge 不够格时零注入（状态如实进 status()["prior"]）。
    prior: PriorConfig = field(default_factory=PriorConfig)
    loop_closure: bool = True
    loop: LoopConfig = field(default_factory=LoopConfig)
    record_max_mb: float = 2048.0    # 录制上限（每关键帧约 0.4 MB，20 分钟约 500 MB）；超了停录、导航照常
    # 会话末自动**并树**（P0 §洞4 的世界树 pass，跑在采纳线程里、逐对采纳之后）。
    # 为什么需要它：逐对采纳只把本场挂到"它看见最多的那一个"旧会话上，看不到**别的会话之间的桥** ——
    # 实测漏过：`050848→035644` 有 18 条够格约束，但 035644 自己从没被采纳 ⇒ 那 18 条因
    # "坐标系不同根"被丢（先验的 gauge 因此少了最有用的一批票）。世界树按整张约束图选根 + 链式采纳，
    # 顺手把孤立会话收进来。关掉它 = 回到"要手动跑 research/tools/xsession_align.py --world-tree"。
    xsession_world_tree: bool = True
    # 覆盖可视化（纯显示，不进三态/规划）：粗格边长（追踪米）、走过走廊的 hull 半径（世界米）、
    # "近看且密"的近看点数阈值（粗格）、趋势历史条数（≈5 min @2 Hz）。
    cov_coarse_m: float = 0.30
    cov_hull_m: float = 3.0
    cov_sparse_pts: float = 8.0
    cov_hist_len: int = 600

    def __post_init__(self) -> None:
        if self.mapper.world_scale != self.world_scale:
            self.mapper = replace(self.mapper, world_scale=self.world_scale)
        if self.loop.world_scale != self.world_scale:
            self.loop = replace(self.loop, world_scale=self.world_scale)

    @classmethod
    def from_plugin(cls, nav: Any) -> "OnlineNavConfig":
        """由 ``config.NavmeshConfig`` 构造：顶层字段 + online/mapper/loop/prior 四组覆盖项（键已在 config 校验）。"""
        mapper = replace(MapperConfig(), **dict(nav.mapper))
        loop = replace(LoopConfig(), **dict(nav.loop))
        prior = replace(PriorConfig(), **dict(getattr(nav, "prior", ())))
        return cls(world_scale=nav.world_scale, expected_baseline_m=nav.expected_baseline_m,
                   baseline_tol_m=nav.baseline_tol_m, loop_closure=nav.loop_closure,
                   mapper=mapper, loop=loop, prior=prior, **dict(nav.online))


def check_baseline(measured_m: float, cfg: OnlineNavConfig) -> dict[str, Any]:
    """实测基线对期望。``ok`` 为 None 表示不检查。"""
    exp = float(cfg.expected_baseline_m)
    if exp <= 0.0:
        return {"ok": None, "measured_m": round(float(measured_m), 4), "expected_m": None}
    ok = math.isfinite(measured_m) and abs(float(measured_m) - exp) <= cfg.baseline_tol_m
    return {"ok": bool(ok), "measured_m": round(float(measured_m), 4), "expected_m": exp,
            "tol_m": cfg.baseline_tol_m}


_SCALE_RANGE = (0.2, 5.0)      # 超出这个倍数的缩放读数当作坏值，不用
_RESCALE_TOL = 0.01            # 会话中缩放偏离开头读数超过 1% ⇒ 地图作废、重开会话
_RESCALE_SETTLE_S = 2.0        # 新读数要稳住（±0.5%）这么久才重开：拖动缩放手势是连续变的
_RESCALE_CHECK_S = 1.0


def avatar_scale_factor(snapshot: dict[str, Any], ref_eye_m: float) -> tuple[float, str]:
    """头像缩放倍数（相对 world_scale 标定时的眼高）及来源。拿不到 / 关闭时返回 ``(1.0, 原因)``。"""
    if not ref_eye_m or ref_eye_m <= 0:
        return 1.0, "disabled"
    for key, div in (("EyeHeightAsMeters", ref_eye_m), ("/avatar/eyeheight", ref_eye_m), ("ScaleFactor", 1.0)):
        try:
            v = float(snapshot.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and _SCALE_RANGE[0] <= v / div <= _SCALE_RANGE[1]:
            return v / div, key
    return 1.0, "none"


class RateMeter:
    """最近 N 次事件的**实际**频率。用来把「配置的频率」和「跑出来的频率」分开报。

    为什么必须外显：`stereo_period_s = 0.1` 意图 10 Hz，但门写的是
    `if tick >= next_stereo: next_stereo = tick + stereo_period_s`，而 `tick` 是
    **位姿循环的实际节拍**。位姿循环跑不满 `pose_hz` 时，这个门会被**向上量化到整数个
    tick** —— 0.1 s / 0.063 s = 1.59 ⇒ 只能等到第 2 个 tick，实际步长变成 2 tick =
    0.126 s = 7.94 Hz，比配置低 21%，而**此前没有任何字段报过这件事**。

    🔴 与 `baseline_m` 同类：声明值（配置/驱动）不等于生效值。凡是「按周期推进」的量，
    都要有一个实测频率跟它并排，否则缺口只能靠事后从日志反推。

    ⚠️ 读数只在**最近 N 次**上算，长会话里换速度档后它会跟着变；`samples` 说明窗口多满。
    """

    __slots__ = ("_t",)

    def __init__(self, n: int = 64) -> None:
        self._t: deque[float] = deque(maxlen=n)      # maxlen 就是窗口大小

    def mark(self, t: float) -> None:
        self._t.append(t)

    def hz(self) -> float | None:
        """窗口内 (n−1)/跨度。样本 < 2 时返回 None（不是 0 —— 那会被读成卡死）。"""
        t = self._t
        if len(t) < 2:
            return None
        span = t[-1] - t[0]
        return None if span <= 0.0 else (len(t) - 1) / span

    def gap_p50_s(self) -> float | None:
        """间隔中位。**这个数才看得出「量化台阶」**：它应当等于整数个位姿 tick。"""
        try:
            vals = list(self._t)
        except RuntimeError:          # 另一线程正在改写 deque
            return None
        if len(vals) < 3:
            return None
        return float(np.median(np.diff(vals)))

    def samples(self) -> int:
        return len(self._t)


class KeyframePolicy:
    """双目帧要不要成为关键帧。``decide`` 返回 None（不取）、``"new"``（追加）或
    ``"refresh"``（原地补帧：点云替换上一关键帧的，位姿/回环节点照留）。在线与离线回放共用。"""

    def __init__(self, cfg: OnlineNavConfig) -> None:
        self.cfg = cfg
        self.scale = cfg.world_scale
        self.xy: np.ndarray | None = None
        self.yaw = 0.0
        self.at = -math.inf
        self.defer_since: float | None = None
        self.deferred = 0
        self._prev: tuple | None = None          # 上一次 _take 之前的状态，给 undo 用

    def decide(self, t: float, xy: np.ndarray, yaw: float, yaw_rate_dps: float) -> str | None:
        c = self.cfg
        if self.xy is None:
            return self._take(t, xy, yaw, "new")
        d = float(np.hypot(*(np.asarray(xy[:2], float) - self.xy))) * self.scale
        dyaw = abs(math.degrees(_wrap(yaw - self.yaw)))
        moved = d >= c.kf_dist_m or dyaw >= c.kf_turn_deg
        if not moved and t - self.at < c.kf_max_age_s:
            return None
        if abs(yaw_rate_dps) > c.kf_defer_dps:
            if self.defer_since is None:
                self.defer_since = t
            if t - self.defer_since < c.kf_defer_max_s and d < c.kf_defer_max_m:
                self.deferred += 1
                return None
        # defer 一旦开过（defer_since 非 None，_take 才会清它），说明上一关键帧
        # 之后有过一段「快转中被推迟」的时间，本帧要么是超时兜底取到的、要么是
        # 转速刚回落时取到的——两者都还在快转的余波里。这种帧一律按 new 处理：
        #   * refresh 的代价远大于 new —— 它替换上一关键帧的点云（:925-927），
        #     并连带删掉那一帧的回环词袋（nav_xsession.on_keyframe）与磁盘上的
        #     kf npz / thumb jpg（nav_memory._write_kf），删掉的东西找不回来；
        #   * 判成 new 只是多一份噪声，判成 refresh 是拿模糊帧覆盖好帧并丢数据。
        # 只在原地（d、dyaw 都很小）时才可能走到 refresh，而 defer 恰恰只在
        # 净位移很小时才持续——原地快摆头/抖动是最典型的触发场景。
        if self.defer_since is not None:
            return self._take(t, xy, yaw, "new")
        kind = "refresh" if (not moved and d < c.kf_still_m and dyaw < c.kf_still_deg) else "new"
        return self._take(t, xy, yaw, kind)

    def _take(self, t: float, xy: np.ndarray, yaw: float, kind: str) -> str:
        self._prev = (self.xy, self.yaw, self.at, self.defer_since)
        self.xy, self.yaw, self.at, self.defer_since = np.asarray(xy[:2], float).copy(), yaw, t, None
        return kind

    def undo(self) -> None:
        """撤销上一次取帧：这一帧没送到双目线程（解耦队列满），等于没取过。

        不撤的话下一帧要从这个**没落盘**的位置再走满 ``kf_dist_m`` 才判 ⇒ 关键帧间距被拉稀
        （20261025_202630：丢了的间隔中位 1.16 m vs 0.56 m）。撤了下一拍照常再判。
        只撤一次：连续两次 undo 没有意义（中间必然有过一次 decide）。
        """
        if self._prev is not None:
            self.xy, self.yaw, self.at, self.defer_since = self._prev
            self._prev = None


class SessionRecorder:
    """默认不开。把一次在线会话的原始输入落盘，离线能按同样顺序重放建图/回环（长时间场景复现）。

    目录内容：``meta.json``（配置、传感器）、``hmd.jsonl``（每个位姿周期的 HMD 旋转，base 系）、
    ``osc.jsonl``（新到的 OSC 速度样本）、``vert.jsonl``（OSC VelocityY / Grounded）、
    ``params.jsonl``（内置运动/缩放参数逐条原始值 ``{"t","n","v"}``，含 Upright / AngularY / VelocityMagnitude /
    ScaleFactor / /avatar/eyeheight 等；开录时的缓存值另在 meta.json ``avatar``）、
    ``devices.jsonl``（关键帧拍上全部已连接 OpenVR 设备 ``{"t","d":[{"i","c","r","v","T"(3×4 追踪站立系),
    "serial"/"model"/"sys"/"ctype"}]}``，``t`` 与 events 里该关键帧的 ``t`` 相同；自身身体/手进点云的离线标注用）、
    ``events.jsonl``（关键帧/trail/回环，按建图线程处理顺序）、
    ``kf/<id>.npz``（点云 + 航位推算位姿 + 入图位姿 + ORB 特征）、``final_poses.npz``（停止时的回环位姿）；
    ``images=True`` 时另有 ``kf_img/<id>_{L,R}.png``（关键帧左右原图，采集分辨率；彩色按 OpenCV 惯例
    RGB→BGR 后写，PNG 颜色是对的，``cv2.imread`` 读回 BGR；只有灰度就存灰度。
    ⚠️ 2026-10-25 修之前的彩色录制（如 20261025_202630）没转，PNG 红蓝互换：``imread`` 的数组直接当 RGB 用）。

    感知线程只往内存缓冲里追加（``line``），所有磁盘写都在建图线程（``flush`` / ``keyframe``）里做。
    写失败或超出上限就停录并留痕，不影响导航。
    """

    def __init__(self, root: Path, max_bytes: float, meta: dict[str, Any], images: bool = False) -> None:
        self.dir = Path(root)
        # 关键帧左右原图另存 ``kf_img/<id>_{L,R}.png``（720 / 视差 64 档下每帧两图约 0.26 MB；
        # 彩色是同尺寸的 3 倍，约 0.78 MB，见 navmesh.html 的提示）。
        # 用途：离线重算视差 / 稠密重定位 / 跨经过对齐 / 给点云上色 —— kf/*.npz 只有体素点和稀疏 ORB。
        self.images = bool(images)
        (self.dir / "kf").mkdir(parents=True, exist_ok=True)
        self.max_bytes = float(max_bytes)
        self.bytes = 0
        self.keyframes = 0
        self.stopped: str | None = None
        self._lock = threading.Lock()
        self._buf: list[tuple[str, str]] = []
        self._files: dict[str, Any] = {}
        self._write("meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"dir": str(self.dir), "keyframes": self.keyframes, "mb": round(self.bytes / 1e6, 1),
                    "stopped": self.stopped, "images": self.images}

    def line(self, name: str, obj: dict[str, Any]) -> None:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        with self._lock:
            if self.stopped is None:
                self._buf.append((name, text))

    def flush(self) -> None:
        with self._lock:
            buf, self._buf = self._buf, []
            if self.stopped is not None:
                return
            try:
                for name, text in buf:
                    f = self._files.get(name)
                    if f is None:
                        f = self._files[name] = open(self.dir / f"{name}.jsonl", "a", encoding="utf-8")
                    f.write(text + "\n")
                    self.bytes += len(text) + 1
                for f in self._files.values():
                    f.flush()
            except OSError as exc:
                self._halt_locked(f"write_failed: {exc}")
            self._check_size_locked()

    def keyframe(self, k: int, pts: np.ndarray, T_dr: np.ndarray, T_map: np.ndarray, dist_m: float,
                 feat: Any, images: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        arrays: dict[str, np.ndarray] = {"pts": np.asarray(pts, np.float32), "T_dr": np.asarray(T_dr, float),
                                         "T_map": np.asarray(T_map, float), "dist_m": np.array(float(dist_m))}
        if feat is not None:
            arrays.update(uv=feat.uv, des=feat.des, xyz=feat.xyz, des3d=feat.des3d, K=feat.K,
                          size=np.array(feat.size), eye_y=np.array(feat.eye_y))
        with self._lock:
            if self.stopped is not None:
                return
            path = self.dir / "kf" / f"{int(k):06d}.npz"
            try:
                np.savez(path, **arrays)   # 不压缩：点云压不了多少，还占建图线程
                self.bytes += path.stat().st_size
                self.keyframes += 1
                if images is not None and self.images:
                    # PNG 无损（视差要亚像素，JPEG 的块效应会直接变成深度偏置）；压缩级 1 求快。
                    # 感知线程传来的是 RGB（OpenVR 镜像纹理格式，`openvr_mirror.py::read_stereo_rgb`），
                    # 但 ``cv2.imwrite`` 把数组当成 BGR 写 PNG，所以这里**必须转 BGR**——否则落盘红蓝反。
                    # 下游研究脚本 ``imread`` 读回来又是 BGR（OpenCV 惯例），颜色就对得上。灰度图不受影响。
                    (self.dir / "kf_img").mkdir(exist_ok=True)
                    for eye, img in zip("LR", images):
                        out = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        ip = self.dir / "kf_img" / f"{int(k):06d}_{eye}.png"
                        if not cv2.imwrite(str(ip), out, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                            raise OSError(f"imwrite failed: {ip.name}")
                        self.bytes += ip.stat().st_size
            except OSError as exc:
                self._halt_locked(f"write_failed: {exc}")
            self._check_size_locked()

    def close(self, final_poses: dict[int, np.ndarray] | None) -> None:
        self.flush()
        with self._lock:
            if final_poses and self.stopped is None:
                ids = sorted(final_poses)
                try:
                    np.savez_compressed(self.dir / "final_poses.npz", ids=np.array(ids),
                                        T=np.stack([np.asarray(final_poses[k], float) for k in ids]))
                except OSError as exc:
                    self._halt_locked(f"write_failed: {exc}")
            if self.stopped is None:
                self.stopped = "closed"
            self._close_files_locked()

    def _write(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text, encoding="utf-8")
        self.bytes += len(text.encode("utf-8"))

    def _check_size_locked(self) -> None:
        if self.stopped is None and self.bytes >= self.max_bytes:
            self._halt_locked("size_limit")

    def _halt_locked(self, reason: str) -> None:
        self.stopped = reason
        self._buf = []
        self._close_files_locked()

    def _close_files_locked(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except OSError:
                pass
        self._files = {}


def _rnd(v: float | None, nd: int = 3) -> float | None:
    """四舍五入，但 None 保持 None（别把「还没测到」写成 0）。"""
    return None if v is None else round(float(v), nd)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


class HeadingTrack:
    """近 ~1 s 的 (时刻, 展开 yaw)，给 ``DeadReckoner`` 按**积分区间中点**取朝向。

    为什么另起一份而不是复用感知循环里的 ``yaw_hist``（0.25 s，给转速用）：积分区间要
    覆盖到上一个位姿周期，跨静默时还可能更长；两处口径不同，各留各的，互不改对方的窗口。

    只记 (t, yaw)，不记整帧旋转：积分只要基向量 [:,0]，一个标量够。yaw 在这里**展开**
    （跨 ±π 累加），插值才不会在 ±π 附近跳变。
    """

    def __init__(self, span_s: float = 1.0) -> None:
        self.span = float(span_s)
        self._t: list[float] = []
        self._y: list[float] = []

    def push(self, t: float, yaw: float) -> None:
        t = float(t)
        if self._t and t <= self._t[-1]:
            # 时钟回退 / 重复 tick：只改最后一点，不制造非单调序列（np.interp 要求升序）。
            self._t[-1], self._y[-1] = t, float(yaw)
            return
        y = float(yaw)
        if self._y:
            y = self._y[-1] + _wrap(y - self._y[-1])
        self._t.append(t)
        self._y.append(y)
        while len(self._t) > 2 and self._t[-1] - self._t[0] > self.span:
            self._t.pop(0)
            self._y.pop(0)

    def at(self, t: float) -> float | None:
        """t 处的朝向（弧度）；落在已知区间之外返回 None（调用方退回当前朝向）。"""
        if len(self._t) < 2 or not (self._t[0] <= t <= self._t[-1]):
            return None
        return float(np.interp(t, self._t, self._y))

    def span_s(self) -> float:
        """当前覆盖时长（秒）。自检用：远小于一个位姿周期说明历史没在喂。"""
        return 0.0 if len(self._t) < 2 else self._t[-1] - self._t[0]


class DeadReckoner:
    """OSC 本地速度（ZOH 前值保持）× HMD 朝向 → 地图系位姿（追踪米）。

    积分到 now，超出最后一个样本的部分按前值外推（VRChat 速度是变化驱动的，静默 = 没变）。
    迟到样本（时间戳早于已积分时刻）只从现在起生效，不回改历史。

    ``osc_lag_s``：t 时刻身体的运动在 t+lag 才由 OSC 报出，积分结果是 lag 之前的位置，
    所以输出 = 积分位置 + 当前速度 × lag（只加在输出上，不写回积分）。旧实现把积分终点
    推到 now+lag，但迟到样本同样从 now+lag 起生效，两者相互抵消，等于没补。
    代价：起步/停下时输出沿运动方向跳 v×lag（跑步约 0.5 m）。

    ## 非零速度的 ZOH 外推必须封顶（2026-10 静止漂移修复）

    VelocityX/Z 是**变化驱动、没有心跳**的通道：静止不发包，匀速也不发包，所以
    "静默"在信息上等于"速度没变"——**但只在短时间内成立**。实测（Docs/SLAM米制复测报告
    2026-09-24 §3.1，以及 navmesh_recordings/20260929_045615 回放）需要拿非零速度跨空档
    外推的空档全部 ≤ 2.03 s；超过 1 s 的空档里前速度几乎全是 0（真的停住了）。

    旧实现对静默**无上限**外推。丢掉"停下"那个 0 包（UDP 丢包、Avatar 切换、追踪丢失、
    断网）时，最后一个非零速度会被永远积分下去：0.4 m/s 静默 1 小时 = 虚增 1.4 km 路程，
    位姿沿最后运动方向一路"飘"，关键帧与回环被这条假路程带着连发。这就是现场看到的
    "人没动、OSC 路程一直在涨"。

    所以给外推封顶：``zoh_max_s`` 内按满速外推（协议语义），之后在 ``zoh_fade_s`` 内线性
    淡出到 0，再往后完全停止积分。**被丢掉的那段路程照实计数**（``holdout_m`` /
    ``holdout_s``）并进 status()，既不静默改数，也不假装"本来就该是 0"。真正的匀速长走
    每 ~0.09 s 就有新报文（20260929 录制实测 ~11.7 Hz），碰不到这个上限。
    """

    def __init__(self, world_scale: float, osc_lag_s: float = 0.18, *,
                 zoh_max_s: float = 2.5, zoh_fade_s: float = 1.0, speed_scale: float = 1.0,
                 heading_at: Callable[[float], float | None] | None = None) -> None:
        self.s = float(world_scale)
        self.k = float(speed_scale)     # OSC 速度 → 实际速度 的系数（见 OnlineNavConfig.osc_speed_scale）
        self.lag = float(osc_lag_s)
        # 朝向取样器：传入则每个积分子段按**该段中点**的朝向旋转（见 _basis_at）。None = 旧行为。
        self._heading_at = heading_at
        self.heading_segments = 0       # 用历史朝向积分的段数（自检：接线掉了会一直是 0）
        self.heading_fallbacks = 0      # 退回当前朝向的段数（启动头几帧、位姿长空档）
        self.zoh_max = max(0.0, float(zoh_max_s))
        self.zoh_fade = max(0.0, float(zoh_fade_s))
        self.xy = np.zeros(2)
        self.dist_m = 0.0            # OSC 路程，世界米
        self.holdout_s = 0.0         # 因静默超上限而**没有**积分的时长（秒）
        self.holdout_m = 0.0         # 同上，按最后速度折算的**没算进去**的路程（世界米）
        self._v = (0.0, 0.0)         # (vx 右, vz 前)，世界米/秒
        self._v_at: float | None = None   # 最后一个速度样本的 OSC 时刻
        # 本会话里"被新报文证明当时确实在动"的静默段最长多长（两端速度都非零的空档）。
        # 它是 zoh_max_s 该取多少的**现场读数**：实测恒速走的中位间隔 16 ms、最长 0.16 s，
        # 若这个数逼近 osc_zoh_max_s，说明该 Avatar 的回传比录制时更慢，该调大上限。
        self.max_legit_gap_s = 0.0
        self._t: float | None = None
        self._last_ts = -math.inf
        self.samples = 0

    @property
    def last_sample_at(self) -> float | None:
        """最后一个被接受的 OSC 速度样本时刻（None = 一个都没有）。"""
        return self._v_at

    def update(self, t_now: float, r_ob: np.ndarray, osc: Sequence[dict[str, Any]],
               heading_at: Callable[[float], float | None] | None = None) -> np.ndarray:
        self._heading_at = heading_at if heading_at is not None else self._heading_at
        fwd = np.array([r_ob[0, 0], r_ob[1, 0]], float)
        n = float(np.hypot(*fwd))
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0])
        right = np.array([fwd[1], -fwd[0]])
        target = float(t_now)
        if self._t is None:
            self._t = target
        for item in osc:
            ts = float(item.get("timestamp", -math.inf))
            if ts <= self._last_ts:
                continue
            vx, vz = item.get("velocity_x"), item.get("velocity_z")
            if vx is None or vz is None or not (math.isfinite(vx) and math.isfinite(vz)):
                continue
            # 这段静默是被"新报文速度仍非零"证明当时确实在动的 → 一次合法的 ZOH 外推，
            # 记下来做封顶阈值的现场对照（不能说它一定全对：中途滞空也会这样）。
            if self._v_at is not None and math.hypot(vx, vz) > 0.05:
                self.max_legit_gap_s = max(self.max_legit_gap_s, ts - self._v_at)
            self._advance(min(max(ts, self._t), target), fwd, right)
            self._v = (float(vx) * self.k, float(vz) * self.k)
            self._v_at = ts
            self._last_ts = ts
            self.samples += 1
        self._advance(target, fwd, right)
        T = np.eye(4)
        T[:3, :3] = r_ob
        vx, vz = self._v
        T[:2, 3] = self.xy + (vz * fwd + vx * right) * self.lag / self.s
        return T

    def _hold_dt(self, a: float, b: float) -> float:
        """[a,b] 里"有报文支撑"的等效时长：静默 ≤ ``zoh_max_s`` 全给，之后线性淡出到 0。

        权重 w(τ)（τ = 距最后一个样本的时长）：≤M 取 1，M→M+F 之间 1→0 线性，之后取 0。
        返回 ∫w dτ 的闭式解，不是采样近似。
        """
        m, f = self.zoh_max, self.zoh_fade
        if self._v_at is None or m <= 0.0 or f <= 0.0:
            return b - a

        def _area(u: float) -> float:
            if u <= m:
                return u
            if u >= m + f:
                return m + 0.5 * f
            return m + ((m + f) * (u - m) - 0.5 * (u * u - m * m)) / f

        return min(max(_area(b - self._v_at) - _area(a - self._v_at), 0.0), b - a)

    def _basis_at(self, a: float, b: float, fwd: np.ndarray, right: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
        """积分区间 [a,b] 该用哪组基向量。

        VelocityX/Z 是**角色局部系**（实测：atan2(vx,vz) 的直方图尖峰钉在 0°/±90°/180°，
        与 HMD 航向的圆相关 ≤0.19 ⇒ 不是世界系）。局部系向量要落到地图里，必须乘
        **它被测量那一刻**的朝向，而不是处理它那一刻的朝向。旧实现整批样本共用 update()
        入口那一帧的朝向 —— 系统性地"偏超前"：转身时把早先那段位移多转了一点，
        解析量级 ½·v·ω·ΔT·δ（跑步 + 转身约 8 cm / 0.5 s）。取区间中点是一阶精度，
        与离线原型（.tmp/dr_lag_sweep3.py 的 exact 列）同口径。

        没有取样器（旧行为）或它返回 None（启动头几帧、位姿长空档）时退回传入的当前基，
        两种计数分开记，接线掉了表现为 heading_segments 一直是 0。
        """
        if self._heading_at is None:
            return fwd, right
        psi = self._heading_at(0.5 * (a + b))
        if psi is None:
            self.heading_fallbacks += 1
            return fwd, right
        c, s = math.cos(psi), math.sin(psi)
        # 与 update() 同约定：fwd = 基向量 [:,0] = (cos, sin)，right = [:,1] = (sin, -cos)。
        self.heading_segments += 1
        return np.array([c, s]), np.array([s, -c])

    def _advance(self, t: float, fwd: np.ndarray, right: np.ndarray) -> None:
        dt = t - float(self._t)
        if dt <= 0.0:
            return
        vx, vz = self._v
        speed = math.hypot(vx, vz)
        kept = dt if (speed <= 0.0 or self._v_at is None) else self._hold_dt(float(self._t), t)
        # 只在**真正积分的那一段**上取朝向：被封顶丢掉的尾部不该进中点。
        seg_fwd, seg_right = self._basis_at(float(self._t), float(self._t) + kept, fwd, right)
        self.xy = self.xy + (vz * seg_fwd + vx * seg_right) * kept / self.s
        self.dist_m += speed * kept
        dropped = dt - kept
        if dropped > 0.0:
            # 静默太久，后面这段"位移"没有任何证据支撑：只记账，不积分。
            self.holdout_s += dropped
            self.holdout_m += speed * dropped
        self._t = t


#: 打开 SteamVR 时的自检/报错话术。为什么需要：``openvr.init()`` 本身只检查 ``VR_InitInternal2``
#: 的错误码，失败时**抛异常没问题**；但"错误码是 OK、拿到的 IVRSystem 却是空指针"这条路径
#: 不会报错，直到第一次调用（``getEyeToHeadTransform``）才在原生层**读空指针崩掉**
#: （2026-10-06 实况：``sensors_open: OSError: exception: access violation reading 0x0000000000000000``）。
#: 所以先用两个**不需要初始化**的自检函数（官方注释写明它们就是为了"判断有没有可能初始化"）
#: 把"SteamVR 没开 / 头显没接"挡在 init 之前，再把任何原生崩翻译成人能照着做的话。
_STEAMVR_TIPS = ("先启动 SteamVR 并等头显就绪，再点「启动」；"
                 "若刚重启过 SteamVR，请把后端也重启一次——本进程持有的本地 VR 上下文可能已失效。")


def _steamvr_preflight(openvr: Any) -> str | None:
    """返回拒绝原因（None = 可以试 init）。只调**不需要 VR_Init 的安全函数**。"""
    try:
        installed = bool(openvr.isRuntimeInstalled())
    except Exception:                                     # noqa: BLE001 - 探测不了就别拦
        installed = True
    if not installed:
        return f"OpenVR runtime 未安装（isRuntimeInstalled=False）。{_STEAMVR_TIPS}"
    try:
        hmd = bool(openvr.isHmdPresent())
    except Exception:                                     # noqa: BLE001
        return None
    if not hmd:
        return (f"SteamVR 没在运行或头显未就绪（isHmdPresent=False，这个检查不需要初始化 VR）。"
                f"{_STEAMVR_TIPS}")
    return None


def _steamvr_failed(exc: BaseException | None) -> str:
    """原生层失败 → 人话（**保留原始文本**：它是现场证据，不许吞）。"""
    raw = "" if exc is None else f"{type(exc).__name__}: {exc}"
    if isinstance(exc, OSError):
        return (f"打开 SteamVR 时原生层崩了（{raw}）。这通常是 SteamVR 没运行/正在退出"
                f"（拿到了空指针）。{_STEAMVR_TIPS}")
    return f"打开 SteamVR 失败（{raw or '未知原因'}）。{_STEAMVR_TIPS}"


class OpenVRSensors:
    """同一个 OpenVR Background 会话里取 HMD 姿态和镜像双目。

    D3D11 立即上下文不是线程安全的，所以本类的方法只能在感知线程里调用。
    ``openvr`` 惰性导入：没装 / SteamVR 没开时 ``open()`` 抛出，不影响 backend 其他部分。
    """

    def __init__(self, target_width: int = 720) -> None:
        self.target_width = int(target_width)
        self._vr: Any = None
        self._system: Any = None
        self._inited = False          # init 成功过才允许 shutdown（见 close）
        self._device = self._context = None
        self._eyes: list[Any] = []
        self._poses: Any = None
        self._dev_props: dict[int, dict[str, Any]] = {}
        self.fx = self.cx = self.cy = 0.0
        self.baseline_m = 0.0
        self.size = (0, 0)

    def open(self) -> dict[str, Any]:
        import openvr

        from .openvr_mirror import MirrorEye, create_device, projection_intrinsics

        reason = _steamvr_preflight(openvr)
        if reason:
            raise RuntimeError(reason)
        self._vr = openvr
        try:
            self._system = openvr.init(openvr.VRApplication_Background)
        except Exception as exc:                          # noqa: BLE001 - openvr 的错误码异常
            raise RuntimeError(_steamvr_failed(exc)) from exc
        if self._system is None:                          # 空指针：下面第一句调用就会崩
            raise RuntimeError("openvr.init() 返回了空对象（SteamVR 没就绪/正在启停）；"
                               f"继续调用会在原生层读空指针。{_STEAMVR_TIPS}")
        self._inited = True
        try:
            eye_x = [self._system.getEyeToHeadTransform(e)[0][3] for e in (openvr.Eye_Left, openvr.Eye_Right)]
            self.baseline_m = float(eye_x[1] - eye_x[0])
            self._device, self._context = create_device()
            compositor = openvr.VRCompositor()
            for eye in (openvr.Eye_Left, openvr.Eye_Right):
                self._eyes.append(MirrorEye(compositor, self._device, self._context, eye,
                                            gpu_downscale=(self.target_width, 1)))
            src = (self._eyes[0].desc["Width"], self._eyes[0].desc["Height"])
            self.size = self._eyes[0].gpu_size or src
            self.fx, _fy, self.cx, self.cy = projection_intrinsics(self._system, openvr.Eye_Left, self.size, src)
            # 全部 64 个槽位：录制要手柄/追踪器位姿（自身身体进点云的离线标注）。
            # 原来只开 1 个槽 ⇒ 只拿得到 HMD。多出的只是一次原生 memcpy，每拍照样只读 HMD。
            self._poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        except OSError as exc:                            # 原生崩（空指针/dll 失效）走这里
            self.close()
            raise RuntimeError(_steamvr_failed(exc)) from exc
        except Exception:
            self.close()
            raise
        return {"size": list(self.size), "fx": round(self.fx, 2), "cx": round(self.cx, 2),
                "cy": round(self.cy, 2), "baseline_m": round(self.baseline_m, 4)}

    def hmd_rotation(self) -> np.ndarray | None:
        vr = self._vr
        self._system.getDeviceToAbsoluteTrackingPose(vr.TrackingUniverseStanding, 0.0, self._poses)
        pose = self._poses[vr.k_unTrackedDeviceIndex_Hmd]
        if not pose.bPoseIsValid:
            return None
        m = pose.mDeviceToAbsoluteTracking
        return np.array([[m[i][j] for j in range(3)] for i in range(3)], float)

    def read_stereo_rgb(self) -> tuple[np.ndarray, np.ndarray] | None:
        """两眼同帧 RGB（记忆缩略图要彩色）；``read_stereo`` 是它的灰度版。"""
        from .openvr_mirror import read_stereo

        left, right = read_stereo(self._eyes[0], self._eyes[1])
        if left is None or right is None:
            return None
        return left, right

    def read_stereo(self) -> tuple[np.ndarray, np.ndarray] | None:
        pair = self.read_stereo_rgb()
        if pair is None:
            return None
        return cv2.cvtColor(pair[0], cv2.COLOR_RGB2GRAY), cv2.cvtColor(pair[1], cv2.COLOR_RGB2GRAY)

    def snapshot_devices(self) -> list[dict[str, Any]]:
        """所有已连接 OpenVR 设备的位姿（追踪站立系，与 HMD 同一拍）+ 类别/角色/型号。

        不再调 ``getDeviceToAbsoluteTrackingPose``：读的是本拍 ``hmd_rotation()`` 刚填好的 ``_poses``，
        所以手柄/追踪器与 HMD、与紧随其后采的双目帧是**同一时刻**。只在录制的关键帧拍上调。
        ``T`` 是 3×4（OpenVR 原生 +y 上、−z 前，追踪米），离线用 ``inv(T_hmd) @ T_dev`` 换到头部系。
        型号等静态属性按 (槽位, 类别) 缓存：每个设备只问一次。
        """
        vr, sys_ = self._vr, self._system
        out = []
        for i in range(len(self._poses)):
            if not sys_.isTrackedDeviceConnected(i):
                continue
            cls = int(sys_.getTrackedDeviceClass(i))
            props = self._dev_props.get(i)
            if props is None or props["c"] != cls:
                props = {"c": cls}
                # 不是所有设备都有这些属性：取不到/为空就不记。
                for p, k in ((vr.Prop_SerialNumber_String, "serial"), (vr.Prop_ModelNumber_String, "model"),
                             (vr.Prop_TrackingSystemName_String, "sys"), (vr.Prop_ControllerType_String, "ctype")):
                    try:
                        v = sys_.getStringTrackedDeviceProperty(i, p)
                    except Exception:  # noqa: BLE001 - openvr 的属性错误码异常
                        continue
                    if v:
                        props[k] = v
                self._dev_props[i] = props
            pose = self._poses[i]
            m = pose.mDeviceToAbsoluteTracking
            # 角色每次都读：两只手柄会互换左右。
            out.append({"i": i, **props, "r": int(sys_.getControllerRoleForTrackedDeviceIndex(i)),
                        "v": bool(pose.bPoseIsValid),
                        "T": [[round(float(m[a][b]), 4) for b in range(4)] for a in range(3)]})
        return out

    def close(self) -> None:
        from .wgc_capture import _release

        for eye in self._eyes:
            try:
                eye.close()
            except Exception:  # noqa: BLE001 - 关闭路径尽力而为
                pass
        self._eyes = []
        if self._context is not None:
            _release(self._context)
        if self._device is not None:
            _release(self._device)
        self._device = self._context = None
        # 只有 init 成功过才 shutdown：失败路径上再调一次 VR_Shutdown，本身就是又一次
        # 读已失效上下文的机会（同一个 0x0 崩法）。
        if self._system is not None and self._inited and self._vr is not None:
            try:
                self._vr.shutdown()
            except Exception:                             # noqa: BLE001 - 关闭路径尽力而为
                pass
            self._inited = False
        self._system = None


def near_obstacle(points_base: np.ndarray, r_ob: np.ndarray, *, cam_h: float, scale: float,
                  ahead_m: float, half_width_m: float, ground_tol_m: float, top_m: float,
                  min_pts: int) -> tuple[bool, int]:
    """当前双目帧里，正前方 ahead_m × 2·half_width_m（世界米）内的障碍点数。"""
    if not len(points_base):
        return False, 0
    q = points_base @ np.asarray(r_ob, np.float32).T
    fwd = np.array([r_ob[0, 0], r_ob[1, 0]], np.float32)
    n = float(np.hypot(*fwd))
    if n < 1e-6:
        return False, 0
    fwd /= n
    along = (q[:, 0] * fwd[0] + q[:, 1] * fwd[1]) * scale
    across = (-q[:, 0] * fwd[1] + q[:, 1] * fwd[0]) * scale
    h = q[:, 2] + cam_h
    hit = (along > 0.0) & (along < ahead_m) & (np.abs(across) < half_width_m) & (h > ground_tol_m) & (h < top_m)
    count = int(hit.sum())
    return count >= min_pts, count


class OnlineNavigator:
    """三条线程：感知（位姿 20 Hz + 双目）、建图（排队的关键帧/轨迹 → 栅格化 + 重规划）、控制 10 Hz。

    感知线程只拿轻锁 ``_state_lock``：关键帧、trail 先排队，由建图线程在重锁 ``_nav_lock``
    下灌进 mapper。否则一次 ~130 ms 的栅格化会卡住 20 Hz 的航位推算。

    外部依赖全是注入的可调用对象，测试可以用假的：
    * ``motion_history()`` → OSC 速度样本（``timestamp`` 与 ``clock`` 同一单调时钟）；
    * ``vertical_history()``（可选）→ OSC VelocityY / Grounded 样本，同一时钟；只进录制；
    * ``send_move(forward, duration_ms)`` / ``send_turn(correction_deg)``（正数 = 左转 CCW）；
    * ``stop_motion()``；``drive_block_reason()`` → None 可驾驶，否则为不能动的原因。
    """

    def __init__(self, *, motion_history: Callable[[], Sequence[dict[str, Any]]],
                 send_move: Callable[[float, int], Any], send_turn: Callable[[float], Any],
                 stop_motion: Callable[[], Any], drive_block_reason: Callable[[], str | None],
                 sensors_factory: Callable[[], Any] | None = None,
                 cfg: OnlineNavConfig | None = None, clock: Callable[[], float] = time.perf_counter,
                 record_root: Path | None = None, memory: NavMemoryStore | None = None,
                 world_identity: Callable[[], dict[str, Any]] | None = None,
                 vertical_history: Callable[[], Sequence[dict[str, Any]]] | None = None,
                 param_history: Callable[[], Sequence[dict[str, Any]]] | None = None,
                 avatar_snapshot: Callable[[], dict[str, Any]] | None = None) -> None:
        self.cfg = cfg or OnlineNavConfig()
        # start() 把 cfg 换成按头像缩放修正过的那份（_cfg_eff）；_cfg_user 留着配置原值，重启不会叠乘。
        # 外部直接赋值 self.cfg（测试）时以新值为准。
        self._cfg_user = self._cfg_eff = self.cfg
        self._avatar_scale: dict[str, Any] | None = None
        self._start_args: dict[str, Any] = {}
        self._life_lock = threading.RLock()           # start/stop 与缩放重开互斥
        self._rescales: list[dict[str, Any]] = []     # 会话中缩放触发的重开（跨重开保留，status 里看）
        self._rescaling = False
        self._scale_pending: tuple[float, float] | None = None   # (首次看到的时刻, 读数)：等它稳住
        self._scale_view: dict[str, Any] | None = None           # 缩放线程最近一次读数
        # 持久记忆（按世界分区）。world_identity() → {"world_key", ...}；key 为空时本次会话不写记忆。
        self.memory = memory
        self._world_identity = world_identity
        self._mem: SessionWriter | None = None
        self._mem_reason: str | None = "not_started"
        self._record_root = None if record_root is None else Path(record_root)
        self.recorder: SessionRecorder | None = None
        self._motion_history = motion_history
        # VelocityY / Grounded：目前只录不用（多层/竖直漂移的离线评估数据）。
        self._vertical_history = vertical_history or (lambda: ())
        # 内置运动/缩放参数逐条原始样本 + 开录时的缓存快照：同样只录不用（缩放会改双目尺度和移动速度）。
        self._param_history = param_history or (lambda: ())
        self._avatar_snapshot = avatar_snapshot or (lambda: {})
        self._send_move = send_move
        self._send_turn = send_turn
        self._stop_motion = stop_motion
        self._drive_block_reason = drive_block_reason
        # 默认工厂吃 ``cfg.capture_width``（默认 1080；实际分辨率看镜像纹理，见 OnlineNavConfig）。
        # 测试注入的假传感器不受影响。
        self._sensors_factory = sensors_factory or (
            lambda: OpenVRSensors(target_width=self.cfg.capture_width))
        self._clock = clock
        self._state_lock = threading.Lock()
        self._nav_lock = threading.RLock()
        self._stop = threading.Event()
        self._map_event = threading.Event()
        # 会话代号：start 时 +1。stop 只等线程 2 s，LAS2 首次编译（70–330 s）会超时，
        # 旧双目线程编完那一帧时可能已经换了会话 ⇒ 它要据此丢掉结果，不能写进新会话的图。
        self._gen = 0
        self._threads: list[threading.Thread] = []
        self._reset()

    def _clock_domain_name(self) -> str:
        """注入时钟的名字，落盘用。测试注入假时钟时如实写 custom，不谎称 perf_counter。"""
        name = getattr(self._clock, "__name__", "") or "custom"
        return name if name in ("perf_counter", "monotonic") else "custom"

    def _reset(self) -> None:
        c = self.cfg
        self.mapper = KeyframeGridMapper(c.mapper)
        self.session = NavSession(self.mapper, radius_m=c.radius_m, request_update=self._request_map_update)
        self.heading_track = HeadingTrack()
        self.dr = DeadReckoner(c.world_scale, c.osc_lag_s,
                               zoh_max_s=c.osc_zoh_max_s, zoh_fade_s=c.osc_zoh_fade_s,
                               speed_scale=c.osc_speed_scale)
        self.loops = LoopCloser(c.loop) if c.loop_closure else None
        self._offset = np.zeros(2)          # 回环修正量（追踪米），加在原始航位推算上
        self._last_loop: dict[str, Any] | None = None
        self._loop_status: dict[str, Any] | None = None if self.loops is None else self.loops.status()
        self._cam_h = self.mapper.cam_h
        self._pose: np.ndarray | None = None
        self._pose_at: float | None = None
        self._stereo_at: float | None = None
        self._near = False
        self._near_pts = 0
        # 解耦双目：交接槽（maxsize=1，满则丢最新，绝不阻塞位姿线程）。
        self._stereo_q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        # 帧序号（仅用于日志/调试）与关键帧编号。解耦线程与位姿线程共享这些计数器，所以取号要在
        # _state_lock 下（否则两边都能分到同一个 kf_id）。
        self._stereo_seq = 0
        # 交接槽计数（见 _offer_stereo）：kf = 关键帧没送进去（已撤销策略，下一拍重判）；
        # frame = 普通帧让路给槽里的关键帧；replaced = 槽里的旧普通帧被新帧顶掉。
        self._stereo_drops = {"kf": 0, "frame": 0, "replaced": 0}
        self._kf_next_id = 0
        #: 最近一个**已入队**的关键帧号（-1 = 还没有）。trail 挂它。
        self._last_committed_kf = -1
        self._pending: list[tuple[str, Any]] = []
        self._keyframes = 0
        self._kf_refreshed = 0
        # 落盘的关键帧里**位置不重复**的那些（追踪米下与上一帧同点即视为重复）。
        # 为什么必须外显：`kf_max_age_s` 让原地不动也每 3 s 补一帧，于是**长停顿的会话**
        # 会把同一地点反复计成独立关键帧。实测 20260929_045615 的 3014 帧里 1250 帧
        # （41.5%）`T_dr` 冻结在同一点。它同时污染两件事——离线仿射/相似拟合把同一位置
        # 放大上千倍权重（按位置去重后该场尺度从 −9.91% 变到 −5.09%，其余场动 ≤0.01pp），
        # 以及位姿图里那串补帧边的 σ（path=0 ⇒ σ 取最小）。**只在建图线程写。**
        self._kf_unique = 0
        self._kf_last_xy: np.ndarray | None = None
        self._kf_policy: KeyframePolicy | None = None
        self._map_updates = 0
        self._map_ms: float | None = None
        self._stereo_ms: float | None = None
        # 解耦双目每帧分段（ms，墙钟 + 本线程 CPU）：只给诊断用，见 _stage_summary。
        self._stereo_stages: deque[dict[str, float]] = deque(maxlen=64)
        self._matcher: Any = None
        # 节拍器的实测遥测：声明值旁边必须并排给实测值（见 RateMeter 的教训）。
        # pose / control / stereo / trail 各一份，None 表示那一路还没起过。
        self._pacer_telemetry: dict[str, Any] = {}
        # 实测频率（与配置值并排报，见 RateMeter）。`_rate_pose` 记的是感知循环的
        # **节拍**，它是另外两个的上游：双目门被量化在它上面，关键帧又只能跟着双目走。
        self._rate_pose = RateMeter(int(self.cfg.pose_hz * 4))
        self._rate_stereo = RateMeter(64)
        self._rate_kf = RateMeter(64)
        self._sensor_info: dict[str, Any] | None = None
        self._baseline_check: dict[str, Any] | None = None
        self._errors: list[str] = []
        # 一个位姿周期里 HMD yaw 变化超过调度器转速上限能解释的量 = 有人在我们之外转了 play space
        # （SteamVR/VRChat 重置朝向、传送）。航位推算会从那一刻起朝错方向积分，必须留痕。
        self._yaw_jumps: list[dict[str, Any]] = []
        self._moving = False
        self._drive_block: str | None = "not_started"
        self._last_step: dict[str, Any] = {}
        self._started_at: float | None = None
        # 记忆位姿表：k → (T_dr, osc 路程, 相对启动秒)。T_map 在整表写时现取（回环会改）。
        self._mem_rows: dict[int, tuple[np.ndarray, float, float]] = {}
        self._mem_flushed_at = -math.inf
        # 覆盖可视化快照（mapping 线程整 dict 原子替换，HTTP 线程只读引用）+ 惰性 PNG 编码缓存。
        self._cov: dict[str, Any] | None = None
        self._cov_hist: deque = deque(maxlen=c.cov_hist_len)
        self._cov_png: tuple[int, str | None] | None = None
        # 跨会话地点检索（P0）：tracker 在 start() 里创建，mapping 线程喂数据。
        self._xs: XSessionTracker | None = None
        self._xs_reason = "not_started"
        # 世界先验消费（P0.3b）：constraints 喂给 consumer，由它在线估 gauge 后注入。
        self._prior: PriorConsumer | None = None
        # 会话末自动采纳（P0.3b）：write_back 成功后后台线程跑 align_into_auto。
        self._align_out: dict | None = None
        self._align_thread: threading.Thread | None = None

    # ---- 生命周期 ----
    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self, record: bool = False, record_images: bool = False) -> dict[str, Any]:
        """``record=True``：把这次会话的原始输入录到 ``record_root/<时间>/``（默认不录）。
        ``record_images=True``：录制时另存关键帧左右原图（只在 ``record`` 时生效）。"""
        with self._life_lock:
            return self._start(record, record_images)

    def _start(self, record: bool, record_images: bool) -> dict[str, Any]:
        if self.running:
            return self.status()
        self._stop.clear()
        self._start_args = {"record": bool(record), "record_images": bool(record_images)}
        try:
            avatar = dict(self._avatar_snapshot() or {})
        except Exception as exc:  # noqa: BLE001 - 快照拿不到不能挡住导航/录制，按不缩放走
            avatar = {"error": f"{type(exc).__name__}: {exc}"}
        self._apply_avatar_scale(avatar)
        self._reset()
        self._started_at = self._clock()
        self.recorder = None
        if record:
            if self._record_root is None:
                self._fail("recorder", RuntimeError("record_root_not_configured"))
            else:
                try:
                    self.recorder = SessionRecorder(
                        self._record_root / time.strftime("%Y%m%d_%H%M%S"), self.cfg.record_max_mb * 1e6,
                        # clock_domain 必须落盘：started_clock 是**绝对读数**，只有知道它来自
                        # 哪个时钟，离线分析才能判断能否与自己算出的时刻相减。2026-10-24 之前
                        # 是 monotonic，之后是 perf_counter —— 没这个字段就分不清新旧录制。
                        {"config": asdict(self.cfg), "started_wall": time.time(),
                         "started_clock": self._started_at,
                         "clock_domain": self._clock_domain_name(),
                         "record_images": bool(record_images), "avatar": avatar,
                         "avatar_scale": self._avatar_scale},
                        images=record_images)
                except (OSError, TypeError) as exc:
                    self._fail("recorder", exc)
        self._gen += 1
        self._mem_rows = {}
        self._mem_flushed_at = -math.inf
        self._begin_memory()
        self._begin_xsession()
        loops = [("perception", self._perception_loop),
                 ("mapping", self._mapping_loop),
                 ("control", self._control_loop),
                 ("scale", self._scale_loop)]
        if self.cfg.decoupled:
            # 双目解耦线程：位姿线程只交接采集帧，重活在这里做。
            loops.insert(1, ("stereo", self._stereo_worker_loop))
        self._threads = [threading.Thread(target=fn, name=f"navmesh-{name}", daemon=True)
                         for name, fn in loops]
        for t in self._threads:
            t.start()
        return self.status()

    def _apply_avatar_scale(self, avatar: dict[str, Any]) -> None:
        """会话开头定一次 world_scale。中途缩放改不了已建的图 ⇒ ``_watch_avatar_scale`` 重开会话。"""
        user = self._cfg_user if self.cfg is self._cfg_eff else self.cfg
        factor, source = avatar_scale_factor(avatar, user.avatar_ref_eye_height_m)
        eff = user if factor == 1.0 else replace(user, world_scale=user.world_scale * factor)
        self._cfg_user, self._cfg_eff, self.cfg = user, eff, eff
        self._scale_view = self._scale_pending = None
        self._avatar_scale = {"factor": round(factor, 4), "source": source,
                              "base_world_scale": user.world_scale, "world_scale": eff.world_scale}

    def _avatar_scale_status(self) -> dict[str, Any] | None:
        """status() 用：只读缩放线程最近一次的结果（快照可能走 HTTP，不在 status 路径上现取）。"""
        if self._avatar_scale is None:
            return None
        return {**(self._scale_view or self._avatar_scale), "restarts": list(self._rescales[-5:])}

    def _read_avatar_scale(self) -> dict[str, Any] | None:
        info = self._avatar_scale
        if info is None or info["source"] == "disabled":
            return info
        try:
            now, src = avatar_scale_factor(dict(self._avatar_snapshot() or {}), self.cfg.avatar_ref_eye_height_m)
        except Exception:  # noqa: BLE001 - 只是状态读数
            return info
        # 会话中改了缩放：地图仍按开头的 world_scale，DR 与几何从此差 now/factor 倍，要重开会话。
        changed = src != "none" and abs(now / info["factor"] - 1.0) > _RESCALE_TOL
        return {**info, "current_factor": round(now, 4), "current_source": src, "changed": changed}

    def _scale_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._watch_avatar_scale(self._clock())
            except Exception as exc:  # noqa: BLE001 - 缩放监视坏了导航照常
                self._fail("avatar_scale", exc)
            self._stop.wait(_RESCALE_CHECK_S)

    def _watch_avatar_scale(self, tick: float) -> None:
        """缩放线程每秒查一次；偏离开头读数 >1% 且稳住 2 s ⇒ 另起线程 stop()+start() 重开。

        不能在本线程里直接 stop()（它要 join 本线程）。重开后旧会话照常收尾
        （录制关闭、记忆落盘 complete），新会话按新眼高定 world_scale。"""
        if self._rescaling:
            return
        st = self._scale_view = self._read_avatar_scale()
        if not st or not st.get("changed"):
            self._scale_pending = None
            return
        cur, p = float(st["current_factor"]), self._scale_pending
        if p is None or abs(cur / p[1] - 1.0) > 0.005:
            self._scale_pending = (tick, cur)
            return
        if tick - p[0] >= _RESCALE_SETTLE_S:
            self._rescaling = True
            threading.Thread(target=self._rescale_restart, args=(st,), name="navmesh-rescale", daemon=True).start()

    def _rescale_restart(self, st: dict[str, Any]) -> None:
        try:
            with self._life_lock:
                if self._stop.is_set() or not self.running:      # 期间用户已经停了：不替他开
                    return
                args = dict(self._start_args)
                self.stop()
                self.start(**args)
                self._rescales = (self._rescales + [{
                    "wall": round(time.time(), 1), "from_factor": st["factor"],
                    "to_factor": (self._avatar_scale or {}).get("factor"),
                    "world_scale": self.cfg.world_scale}])[-20:]
        except Exception as exc:  # noqa: BLE001 - 重开失败要留痕
            self._fail("avatar_rescale", exc)
        finally:
            self._rescaling = False

    def stop(self, timeout_s: float = 2.0) -> dict[str, Any]:
        with self._life_lock:
            return self._stop_session(timeout_s)

    def _stop_session(self, timeout_s: float) -> dict[str, Any]:
        self._stop.set()
        self._map_event.set()
        threads, self._threads = self._threads, []
        for t in threads:
            t.join(timeout=timeout_s)
        self._halt()
        with self._nav_lock:
            self.session.cancel()
        # 停止前把 _pending 里剩下的关键帧写进录制（不过图、不进回环）：建图线程停在两帧之间时
        # 会把没处理的放回 _pending（见 _mapping_loop）。只写 kf/*.npz + events 行（``unmapped``），
        # 停止不再被积压拖几十秒，录制也不再在积压起点断掉。
        rec = self.recorder
        mapping_alive = any(t.name == "navmesh-mapping" and t.is_alive() for t in threads)
        if rec is not None and mapping_alive:
            # join 超时：建图线程还在跑（单帧卡住），这时再碰 loops/录制是竞争 ⇒ 不补写，记一笔。
            self._fail("record_flush", RuntimeError("mapping thread still running at stop"))
        elif rec is not None:
            with self._state_lock:
                pending, self._pending = self._pending, []
            for kind, item in pending:
                if kind == "kf":
                    k, pts, T, dist, feat, t_kf, refresh, thumb, imgs = item
                    # T_map 没进位姿图：按最后一个入图关键帧的修正量平移（与逐帧 trail 同口径）。
                    T_map = T if self.loops is None else self.loops.correct(T)
                    rec.keyframe(k, pts, T, T_map, dist, feat, images=imgs)
                    rec.line("events", {"kind": "kf", "k": k, "t": t_kf, "dist_m": dist,
                                        "loops": [], "refresh": refresh, "unmapped": True})
                elif kind == "trail":
                    k, T_dr, dist, t_tr = item
                    rec.line("events", {"kind": "trail", "k": k, "t": t_tr, "dist_m": dist, "T_dr": T_dr})
            rec.flush()
        if rec is not None:
            rec.close(None if self.loops is None else self.loops.poses())
        # 采纳线程要的 session_id **只有这里还取得到**：_end_memory 会把 _mem 置 None。
        # （2026-10-05 live 实测：线程里再读 self._mem 恒为 None ⇒ xsession_align 永远报
        # memory_gone，采纳从未真正跑过——旧测试直接调线程体，绕开了这个顺序。）
        mem = self._mem
        self._end_memory()
        self._end_xsession(None if mem is None else str(mem.session_id))
        return self.status()

    # ---- 跨会话地点检索（P0）----
    def _begin_xsession(self) -> None:
        """世界索引装载在 tracker 的后台线程里做，start() 不阻塞。依赖记忆会话（world_key + 目录）。"""
        self._xs = None
        mem = self._mem
        if mem is None:
            self._xs_reason = self._mem_reason or "memory_not_configured"
            return
        if not self.cfg.xsession.enabled:
            self._xs_reason = "disabled"
            return
        try:
            wdir = mem.dir.parent.parent          # <root>/<world>/sessions/<sid> → <root>/<world>
            self._xs = XSessionTracker(self.cfg.xsession, wdir, mem.session_id, session_dir=mem.dir,
                                       world_scale=self.cfg.world_scale)
            self._xs_reason = ""
        except Exception as exc:                  # noqa: BLE001 - 检索坏了导航照常
            self._xs, self._xs_reason = None, f"init_error:{type(exc).__name__}"
            self._fail("xsession", exc)
            return
        # 世界先验消费：约束（gauge 的原料）只能从 tracker 来 ⇒ 与 tracker 同生共死。
        # 位姿源：新帧 = mapper（回环修正后）；旧帧 = 索引快照（merged 优先，与其验证口径同源）。
        try:
            if self.cfg.prior.enabled:
                self._prior = PriorConsumer(
                    self.cfg.prior, wdir, world_scale=self.cfg.world_scale,
                    pose_of=self.mapper.pose, pose_of_old=self._xs.old_pose,
                    table_base=self._xs.table_base)
                self.session.prior = self._prior.overlay
        except Exception as exc:                  # noqa: BLE001 - 先验坏了导航照常
            self._prior = None
            self._fail("prior", exc)

    def _end_xsession(self, mem_sid: str | None = None) -> None:
        """会话末把本会话并入世界索引。在 ``_end_memory`` 之后调用（特征已落盘、线程已 join）。

        ``mem_sid`` 由 ``stop()`` 在 **_end_memory 之前**捕获后传入——到这一步内存会话
        对象已被释放（``self._mem is None``），线程里再去读只会拿到 None。
        """
        xs = self._xs
        if xs is None:
            return
        try:
            wb = xs.write_back(self._mem_table())
        except Exception as exc:                  # noqa: BLE001
            self._fail("xsession_end", exc)
            wb = {}
        # P0.3b：写回成功后后台自动采纳（对齐进历史世界系；对齐失败不拖垮 stop）。
        if not wb.get("written"):
            return
        self._align_out = None
        if mem_sid is None:
            self._align_out = {"ok": False, "reason": "memory_gone"}
            return
        try:
            # world_dir 也在此刻捕获：线程跑到一半用户重开一场时 _xs 会被换掉。
            self._align_thread = threading.Thread(
                target=self._align_xsession, args=(mem_sid, xs.world_dir),
                name="navmesh-xsession-align", daemon=True)
            self._align_thread.start()
        except Exception as exc:                  # noqa: BLE001
            self._fail("xsession_align", exc)

    def _align_xsession(self, mem_sid: str, world_dir: Path) -> None:
        """后台采纳线程体：挑约束最多的旧会话对齐 → 再跑**世界树**收孤立会话。

        两步都要：``align_into_auto`` 只把**本场**挂到它看见最多的那个旧会话上；世界树才看得见
        "别的会话之间的桥"（实测漏过 `050848→035644` 那 18 条够格约束，原因就是 035644 自己
        从没被采纳过）。世界树失败不影响逐对采纳的结果，各自进 status。
        """
        try:
            out = align_into_auto(world_dir, mem_sid, self.cfg.xsession)
        except Exception as exc:                  # noqa: BLE001 - 采纳只是锦上添花
            self._fail("xsession_align", exc)
            out = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        if self.cfg.xsession_world_tree:
            tree = self._world_tree_pass(world_dir)
            if isinstance(out, dict):
                out = {**out, "world_tree": tree}
        self._align_out = out

    def _world_tree_pass(self, world_dir: Path) -> dict[str, Any]:
        """世界树 pass（**后台线程**跑，代价是逐桥一次弹性优化，与导航线程无关）。

        只在会话末跑：它要读整张约束图，且会写 ``merged_*.npz``（原子写）与 ``world_tree.json``。
        对自己的会话是否已采纳**不敏感** —— 它收的是"别的会话之间的桥"（正是逐对采纳漏掉的那批）。
        """
        try:
            out = align_world_tree(world_dir, self.cfg.xsession)
        except Exception as exc:                  # noqa: BLE001 - 并树失败不影响本场记录
            self._fail("xsession_tree", exc)
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        comps = out.get("components") or []
        return {"ok": bool(out.get("ok")), "reason": out.get("reason"),
                "aligned": out.get("aligned"), "skipped": out.get("skipped"),
                "refused": out.get("refused"), "components": len(comps),
                "report": out.get("report")}

    # ---- 持久记忆 ----
    def _begin_memory(self) -> None:
        self._mem = None
        if self.memory is None:
            self._mem_reason = "memory_not_configured"
            return
        try:
            identity = {} if self._world_identity is None else dict(self._world_identity() or {})
            self._mem, self._mem_reason = self.memory.begin(identity, {
                "world_scale": self.cfg.world_scale, "expected_baseline_m": self.cfg.expected_baseline_m,
                "loop_closure": self.cfg.loop_closure})
        except Exception as exc:  # noqa: BLE001 - 记忆开不了照样导航
            self._mem, self._mem_reason = None, "memory_error"
            self._fail("memory", exc)

    def _mem_table(self) -> dict[str, np.ndarray] | None:
        rows = dict(self._mem_rows)
        if not rows:
            return None
        ids = sorted(rows)
        loops = self.loops
        T_map = [loops.pose(k) if loops is not None and k in loops else rows[k][0] for k in ids]
        return {"ids": np.array(ids), "T_map": np.array(T_map), "T_dr": np.array([rows[k][0] for k in ids]),
                "dist_m": np.array([rows[k][1] for k in ids]), "t_s": np.array([rows[k][2] for k in ids])}

    def _end_memory(self) -> None:
        mem, self._mem = self._mem, None
        if mem is None or self.memory is None:
            return
        extra: dict[str, Any] = {"odometry_m": round(self.dr.dist_m, 2)}
        if self._loop_status is not None:
            extra["loops"] = self._loop_status.get("loops")
        mem.annotate(**extra)
        try:
            # 线程都已 join，loops 不会再变。
            self.memory.end(self._mem_table(), "complete_with_errors" if self._errors else "complete")
            self._mem_reason = "ended"
        except Exception as exc:  # noqa: BLE001
            self._fail("memory_end", exc)

    def _make_matcher(self, fx: float) -> Any:
        """按 ``cfg.stereo_backend`` 建视差后端；LAS2 载不起来就记错误并退回 SGBM。"""
        kind = self.cfg.stereo_backend
        try:
            if kind != "sgbm" and not self.cfg.decoupled:
                # 内联路径上首帧编译会把位姿线程卡住几分钟。
                raise RuntimeError(f"stereo_backend={kind!r} requires decoupled=True")
            m = make_stereo_backend(kind, fx=fx)
        except Exception as exc:  # noqa: BLE001 - 缺 torch / XPU / 权重：退回 SGBM，不能丢掉双目
            if kind == "sgbm":
                raise
            self._fail("stereo_backend", exc)
            m = make_stereo_backend("sgbm", fx=fx)
        self._matcher = m
        return m

    def prewarm_shape(self) -> tuple[int, int]:
        """按 ``capture_width`` 猜采集尺寸 (H, W)：现行驱动 1440×810 下 1080 档 = 1080×606。

        LAS2 的编译缓存按补到 32 倍数后的尺寸命中，16:9 附近这个猜法和真实尺寸落同一档
        （720/1080/1440 ⇒ 416×736 / 608×1088 / 832×1440）。真实尺寸是 mip 链上不小于 capture_width 的
        最深一级（驱动 1920×1080 时 1080 档实际拿到 1440×810），猜不中就由会话首帧再编。
        """
        w = int(self._cfg_user.capture_width)
        return round(w * 606 / 1080), w

    def prewarm(self) -> dict[str, Any] | None:
        """后端启动时调：配置是 las2（且走解耦路径）就在后台线程预热，立即返回。否则 None。"""
        c = self._cfg_user
        if c.stereo_backend != "las2" or not c.decoupled or not c.stereo_prewarm:
            return None
        return las2_prewarm(self.prewarm_shape())

    def _stage_summary(self) -> dict[str, Any] | None:
        """最近 ≤64 帧解耦双目的分段 [中位, p90]（ms；``*_cpu`` 是 [均值, p90]）。

        墙钟 vs ``*_cpu``（本线程 CPU 时间）的差 = 这段里**没在跑**的时间：GIL 等待、
        GPU 同步（las2_sync）、被别的进程抢核。LAS2 的 disp 再拆成 upload/launch/sync：
        launch 远大于空载基准而 disp_cpu 不大 ⇒ 等 GIL；sync 大 ⇒ GPU 慢或被抢。
        """
        with self._state_lock:
            rows = list(self._stereo_stages)
        if not rows:
            return None
        out: dict[str, Any] = {"frames": len(rows), "switchinterval_ms": round(sys.getswitchinterval() * 1e3, 3)}
        for k in rows[-1]:
            v = np.array([r.get(k, math.nan) for r in rows], float)
            v = v[np.isfinite(v)]
            if v.size:
                # Windows 的 thread_time 是 15.625 ms 一格：*_cpu 的中位只会落在格点上，改给均值。
                c = float(v.mean()) if k.endswith("_cpu") else float(np.median(v))
                out[k] = [round(c, 1), round(float(np.percentile(v, 90)), 1)]
        return out

    def stereo_backend_status(self) -> dict[str, Any]:
        m = self._matcher
        out: dict[str, Any] = {"configured": self.cfg.stereo_backend}
        if self.cfg.stereo_backend == "las2":
            out["prewarm"] = las2_prewarm_status()
        if m is None:
            out["active"] = None          # 还没收到第一帧
        else:
            d = dict(m.describe())
            out["active"] = d.pop("kind", None)
            out.update(d)
        return out

    def _fail(self, where: str, exc: BaseException) -> None:
        with self._state_lock:
            self._errors = (self._errors + [f"{where}: {type(exc).__name__}: {exc}"[:300]])[-5:]

    def _halt(self) -> None:
        if self._moving:
            self._moving = False
            try:
                self._stop_motion()
            except Exception as exc:  # noqa: BLE001 - 停车失败要留痕，但不能让线程死掉
                self._fail("stop_motion", exc)

    # ---- 位姿 ----
    def _est(self) -> dict[str, Any]:
        with self._state_lock:
            T, at, off = self._pose, self._pose_at, self._offset
        # ⚠️ 这里的 max(0.0, ...) **不提供安全性**：max(0.0, 负数) 仍是 0.0，而 0.0 > 阈值
        # 依旧为假 —— 跨域负 age 照样永远判“新鲜”。它只让**报告出来的** age 非负。
        # 真正的防线是两端同域（_pose_at 与 _clock() 都是 perf_counter，见 __init__ 的 clock=
        # 与 _perception_loop 里 tick = _tk.actual）。若将来有人把某一侧换回 monotonic，
        # 这条守卫不会拦住，只能靠 tests/test_pacing_regression.py 的 `mocks_are_effective`
        # 与 status().rates.pacer 的实测值发现。
        if T is None or at is None or max(0.0, self._clock() - at) > self.cfg.pose_stale_s:
            return {"state": "unknown", "reason": "pose_stale"}
        s = self.cfg.world_scale
        x, y = T[0, 3] + off[0], T[1, 3] + off[1]
        return {"state": "localized", "xy": (float(x * s), float(y * s)),
                "theta": math.atan2(T[1, 0], T[0, 0]), "sigma_m": self.cfg.dr_sigma_m}

    # ---- 感知线程 ----
    def _perception_loop(self) -> None:
        c = self.cfg
        s = c.world_scale
        try:
            sensors = self._sensors_factory()
            info = sensors.open()
        except Exception as exc:  # noqa: BLE001 - SteamVR 没开 / openvr 没装：如实报，不崩 backend
            self._fail("sensors_open", exc)
            return
        bl = check_baseline(float(sensors.baseline_m), c)
        with self._state_lock:
            self._sensor_info = info
            self._baseline_check = bl
        mem = self._mem
        if mem is not None:
            # 基线不对的会话也记，但带实测基线：跨会话重定位只拿同基线的记忆比。
            mem.annotate(sensors=info, baseline_m=round(float(sensors.baseline_m), 4), baseline_check=bl)
        # 需要彩色的有三处：记忆缩略图（要彩色）、录制存档（要与采集端同颜色空间）、
        # LAS2 后端（RGB 比 gray3 精度高 5.5–9.5% EPE）。
        # `read_stereo_rgb` 与 `read_stereo` 是**同一次采集**（只差一次 cvtColor），成本差可忽略。
        # 曾经这里只看 mem，于是“没有世界记忆”的会话录下的原图**永远只有灰度**，下游无法上色。
        need_rgb = (mem is not None or (self.recorder is not None and self.recorder.images)
                    or c.stereo_backend == "las2")
        read_rgb = getattr(sensors, "read_stereo_rgb", None) if need_rgb else None
        if bl["ok"] is False:
            self._fail("baseline_mismatch", RuntimeError(
                f"measured {bl['measured_m']} m, expected {bl['expected_m']}±{bl['tol_m']} m "
                "(SteamVR 注册的 AnyaDance 驱动不是自编版？)"))
        rec = self.recorder
        # 关键帧拍上的全部设备位姿 → devices.jsonl（只在录制时）。必须在感知线程读：
        # `_poses` 是本拍 hmd_rotation() 刚填的，与同拍双目帧同一时刻；交给双目线程就晚了。
        snap_devices = getattr(sensors, "snapshot_devices", None) if rec is not None else None
        if rec is not None:
            rec.line("events", {"kind": "sensors", "t": self._clock(), "info": info, "fx": sensors.fx,
                                "cx": sensors.cx, "cy": sensors.cy, "baseline_m": sensors.baseline_m})
        osc_seen = -math.inf
        vert_seen = -math.inf
        param_seen = 0
        matcher = None
        orb = None
        kf_id = -1
        policy = self._kf_policy = KeyframePolicy(c)
        yaw_hist: list[tuple[float, float]] = []    # 近 ~0.25 s 的 (时刻, 展开 yaw)，算转速
        # 节拍器：绝对截止点，tick 取自 Tick.actual（perf_counter 域，与 self._clock 同域）。
        # 为什么不用 while + Event.wait：那会回到 62 ms（Event.wait 向上取整到 15.625 ms tick）。
        pacer = Pacer(hz=c.pose_hz, clock=self._clock, stop=self._stop, name="pose")
        pacer.reset()
        # 双目门：绝对累加。相对累加（tick + period）在以实际 tick 为锚时，49.9 ms 的
        # 周期会把 0.1 s 向上取整成 3 个 tick = 6.73 Hz；绝对累加与 tick 无关，实测恒定 10.0 Hz。
        gate = Gate(period_s=c.stereo_period_s)
        gate.reset(self._clock())
        # trail 门同理：相对累加在 tick 与网格不同步时整拍取整，实测 798/800 个间隔
        # 是 0.25 s 而不是 0.2 s（4 Hz vs 5 Hz，与双目门同样的 20% 误差）。
        trail_gate = Gate(period_s=c.trail_period_s)
        trail_gate.reset(self._clock())
        period = 1.0 / c.pose_hz
        prev_yaw: float | None = None
        prev_tick = -math.inf
        try:
            for _tk in pacer.ticks():
                if _tk.stopped:
                    break
                tick = _tk.actual
                self._rate_pose.mark(tick)
                # 每 20 拍刷新一次遥测：status() 要在运行中就能读到，不能只在退出时写。
                if _tk.index % 20 == 0:
                    with self._state_lock:
                        self._pacer_telemetry["pose"] = pacer.telemetry()
                        self._pacer_telemetry["stereo_gate"] = gate.telemetry()
                        self._pacer_telemetry["trail_gate"] = trail_gate.telemetry()
                r = sensors.hmd_rotation()
                if r is not None:
                    r_ob = hmd_to_base_rotation(r)
                    hist = self._motion_history()
                    if rec is not None:
                        for o in hist:
                            ts = float(o.get("timestamp", -math.inf))
                            if ts > osc_seen:
                                osc_seen = ts
                                rec.line("osc", {"t": ts, "vx": o.get("velocity_x"), "vz": o.get("velocity_z")})
                        for o in self._vertical_history():
                            ts = float(o.get("timestamp", -math.inf))
                            if ts > vert_seen:
                                vert_seen = ts
                                rec.line("vert", {"t": ts, "src": o.get("source"), "vy": o.get("velocity_y"),
                                                  "g": o.get("grounded")})
                        # 同一 bundle 的参数共用接收时刻，按 seq 去重。
                        for o in self._param_history():
                            seq = int(o.get("seq", 0))
                            if seq > param_seen:
                                param_seen = seq
                                rec.line("params", {"t": o.get("timestamp"), "n": o.get("name"), "v": o.get("value")})
                        rec.line("hmd", {"t": tick, "R": np.round(r_ob, 6)})
                    # 先喂朝向历史再积分：本帧 yaw 入历史后，区间中点一定能插到值
                    # （否则每帧都会退回"当前朝向"，等于没改）。
                    self.heading_track.push(tick, math.atan2(r_ob[1, 0], r_ob[0, 0]))
                    head = (self.heading_track.at if c.dr_heading_midpoint else None)
                    T = self.dr.update(tick, r_ob, hist, head)
                    yaw_now = math.atan2(T[1, 0], T[0, 0])
                    if prev_yaw is not None:
                        dyaw = math.degrees(_wrap(yaw_now - prev_yaw))
                        if abs(dyaw) > c.yaw_jump_dps * max(tick - prev_tick, period) + c.yaw_jump_min_deg:
                            with self._state_lock:
                                self._yaw_jumps = (self._yaw_jumps + [{
                                    "t_s": round(tick - (self._started_at or tick), 2),
                                    # 用已入队的号：局部 kf_id 在解耦分支永不递增，会永远记 -1
                            # （2026-10-24 审查发现，与 trail 是同一个根因）。
                            "delta_deg": round(dyaw, 1),
                            "keyframe": self._last_committed_kf,
                                    "odometry_m": round(self.dr.dist_m, 2),
                                    # 精确单位旋转 = 驱动收到了别的发送端的中立帧，不是 VRChat 转的。
                                    "neutral": bool(np.allclose(r_ob, np.eye(3), atol=1e-6))}])[-10:]
                    yaw_hist.append((tick, yaw_now if not yaw_hist else
                                     yaw_hist[-1][1] + _wrap(yaw_now - yaw_hist[-1][1])))
                    while len(yaw_hist) > 2 and tick - yaw_hist[1][0] >= 0.25:
                        yaw_hist.pop(0)
                    prev_yaw, prev_tick = yaw_now, tick
                    with self._state_lock:
                        self._pose, self._pose_at = T, tick
                    # trail 挂到「最近一个**已入队**的关键帧」上（_last_committed_kf 由
                    # _commit_keyframe 在同一临界区里发布）。
                    # ⚠️ 两个坑都踩过（2026-10-24）：
                    #   ① 用局部 kf_id —— 解耦分支在下面 continue 掉了，它永远 -1 ⇒ 一次都不写；
                    #   ② 用 _kf_next_id-1 —— 那是**正在提交**的那帧，mapper 里还没有，
                    #      add_trail 的 `if k not in self._pose: return` 会静默丢弃；
                    #      且 inline 路径根本不递增 _kf_next_id，kid 恒为 -1 ⇒ 一条都不写。
                    if trail_gate.due(tick):
                        with self._state_lock:
                            kid = self._last_committed_kf
                            if kid >= 0:
                                self._pending.append(("trail", (kid, T.copy(), self.dr.dist_m, tick)))
                    if gate.due(tick):
                        t0 = time.perf_counter()
                        left_rgb = None
                        # ⚠️ `rgb` 必须每轮先置 None 再赋：下面存档原图要读它，
                        # 而在 `read_rgb is None` 的路径上原来根本不绑定这个名字 ——
                        # 直接用就是 NameError，还被感知线程的兜底 except 吃掉，
                        # 表现成“关键帧一个都不落”。
                        rgb = None
                        if read_rgb is not None:
                            rgb = read_rgb()
                            pair = None if rgb is None else (cv2.cvtColor(rgb[0], cv2.COLOR_RGB2GRAY),
                                                             cv2.cvtColor(rgb[1], cv2.COLOR_RGB2GRAY))
                            left_rgb = None if rgb is None else rgb[0]
                        else:
                            pair = sensors.read_stereo()
                        if pair is not None:
                            if c.decoupled:
                                # 只交接，不在这里算视差（见 _stereo_worker_loop）。
                                # 关键帧判定也在这里做：它按 tick 算位移/转角/时长，
                                # 必须在位姿线程的时钟上。切点一并算好递过去。
                                yaw = math.atan2(T[1, 0], T[0, 0])
                                (t_a, y_a), (t_b, y_b) = yaw_hist[0], yaw_hist[-1]
                                rate = math.degrees(y_b - y_a) / (t_b - t_a) if t_b > t_a else 0.0
                                kf_kind = policy.decide(tick, T[:2, 3], yaw, rate)
                                self._stereo_seq += 1
                                job = (self._stereo_seq, tick, pair[0], pair[1],
                                       cv2.cvtColor(rgb[0], cv2.COLOR_RGB2GRAY) if rgb is not None else None,
                                       rgb, r_ob, sensors.fx, sensors.cx, sensors.cy,
                                       sensors.baseline_m, kf_kind, T.copy(), self.dr.dist_m)
                                if self._offer_stereo(job, kf_kind is not None):
                                    # devices 行只给真送进双目线程的关键帧写：被丢的拍不留孤行。
                                    if kf_kind is not None and snap_devices is not None:
                                        snap_devices = self._record_devices(rec, snap_devices, tick)
                                elif kf_kind is not None:
                                    policy.undo()     # 没送到 ⇒ 当没取过，下一拍再判
                                self._rate_stereo.mark(tick)
                                continue
                            if matcher is None:
                                matcher = self._make_matcher(sensors.fx)
                            # 内联路径实际只跑 SGBM（_make_matcher 不许 las2 走内联），分支留着防配置漂移。
                            if rgb is not None and matcher.describe()["kind"] == "las2":
                                disp = matcher.compute(rgb[0], rgb[1])
                            else:
                                disp = matcher.compute(pair[0], pair[1])
                            # 深度上限 = 建图视距：水平半径 ≥ 前向深度，再远的点建图也会丢。
                            pts = stereo_points(pair[0], pair[1], fx=sensors.fx, cx=sensors.cx, cy=sensors.cy,
                                                baseline_m=sensors.baseline_m, disp=disp,
                                                max_range_m=c.mapper.range_m)
                            near, n = near_obstacle(
                                pts, r_ob, cam_h=self._cam_h, scale=s, ahead_m=c.stop_ahead_m,
                                half_width_m=c.radius_m, ground_tol_m=c.mapper.ground_tol_m,
                                top_m=c.mapper.obst_top_m, min_pts=c.stop_min_pts)
                            yaw = math.atan2(T[1, 0], T[0, 0])
                            (t_a, y_a), (t_b, y_b) = yaw_hist[0], yaw_hist[-1]
                            rate = math.degrees(y_b - y_a) / (t_b - t_a) if t_b > t_a else 0.0
                            kf_kind = policy.decide(tick, T[:2, 3], yaw, rate)
                            new_kf = kf_kind is not None
                            if new_kf and snap_devices is not None:
                                snap_devices = self._record_devices(rec, snap_devices, tick)
                            feat = None
                            if new_kf and (self.loops is not None or mem is not None):
                                # 同一张视差图给 ORB 点深度；只在关键帧上做，约 6 ms。
                                if orb is None:
                                    orb = cv2.ORB_create(nfeatures=c.loop.orb_features)
                                feat = extract_features(pair[0], disp, fx=sensors.fx, cx=sensors.cx,
                                                        cy=sensors.cy, baseline_m=sensors.baseline_m,
                                                        orb=orb, max_depth_m=c.loop.max_depth_m)
                            near_state = (near, n, tick, (time.perf_counter() - t0) * 1000.0)
                            self._rate_stereo.mark(tick)
                            with self._state_lock:
                                self._near, self._near_pts = near, n
                                self._stereo_at = tick
                                self._stereo_ms = near_state[3]
                            if new_kf:
                                self._commit_keyframe(kf_id, pts, T, feat, tick, kf_kind,
                                                      mem, left_rgb, rgb, pair)
                                kf_id += 1
        except OSError as exc:  # noqa: BLE001 - 原生层崩（SteamVR 被关掉/退出）走这里
            # 本地 VR 上下文已失效，本场不可能自愈：把话说到"该怎么办"，并保留原始错误文本。
            self._fail("perception", RuntimeError(_steamvr_failed(exc)))
        except Exception as exc:  # noqa: BLE001 - 线程死了控制线程会因位姿过期而停车
            self._fail("perception", exc)
        finally:
            with self._state_lock:
                self._pacer_telemetry["pose"] = pacer.telemetry()
                self._pacer_telemetry["stereo_gate"] = gate.telemetry()
                self._pacer_telemetry["trail_gate"] = trail_gate.telemetry()
            try:
                sensors.close()
            except Exception as exc:  # noqa: BLE001
                self._fail("sensors_close", exc)

    def _record_devices(self, rec: "SessionRecorder", snap: Any, tick: float) -> Any:
        """写一行 devices.jsonl；``t`` 与该关键帧 events.jsonl 的 ``t`` 逐位相同（同一个 tick），按它配对。

        解耦路径的关键帧号要到双目线程才分配，这里拿不到，所以只靠 ``t`` 配对；
        只在关键帧**真进了双目队列**之后才写（见 ``_offer_stereo``），所以不再有孤行；
        2026-10-25 之前的录制（如 20261025_202630）丢帧会留孤行，离线按 ``t`` 配不上就跳过。
        读失败只停这一路（返回 None），不影响导航和其它录制。
        """
        try:
            rec.line("devices", {"t": tick, "d": snap()})
            return snap
        except Exception as exc:  # noqa: BLE001
            self._fail("record_devices", exc)
            return None

    def _commit_keyframe(self, kf_id: int, pts: Any, T: np.ndarray, feat: Any, tick: float,
                         kf_kind: str, mem: Any, left_rgb: Any, rgb: Any, pair: Any) -> None:
        """关键帧入队。内联与解耦两条路径共用，保证行为逐位相同。"""
        with self._state_lock:
            thumb = None
            if mem is not None:
                src = left_rgb if left_rgb is not None else pair[0]
                thumb = make_thumbnail(src, mem.cfg.thumb_width)
            imgs = None
            if self.recorder is not None and self.recorder.images:
                imgs = ((rgb[0].copy(), rgb[1].copy()) if rgb is not None
                        else (pair[0].copy(), pair[1].copy()))
            self._pending.append(("kf", (kf_id, pts, T.copy(), self.dr.dist_m, feat, tick,
                                         kf_kind == "refresh", thumb, imgs)))
            # 发布「已入队」的关键帧号：它与上面的 append 在同一个临界区里，且排在任何
            # 后续 trail 入队之前 ⇒ 建图线程按 _pending 顺序必先见到 kf、再见 trail。
            # trail 必须挂这个号，不能挂 _kf_next_id-1（那是**正在提交**的那一帧，
            # mapper 里还没有它，add_trail 会静默丢弃），也不能挂局部 kf_id（解耦路径
            # 永远停在 -1，inline 路径又根本不递增 _kf_next_id）。
            self._last_committed_kf = int(kf_id)
        self._map_event.set()

    def _offer_stereo(self, job: tuple, is_keyframe: bool) -> bool:
        """非阻塞地把一帧交给双目线程；返回这一帧是否进了队列。

        槽满时按「关键帧优先、其余取最新」处理：槽里是普通帧 ⇒ 换成这一帧（更新鲜，近距判定
        也更及时）；槽里是关键帧 ⇒ 它不能丢（devices 行已写、策略状态已推进），这一帧让路。
        只有感知线程一个生产者，所以 get_nowait/put_nowait 之间不会有别人往里放。
        """
        try:
            self._stereo_q.put_nowait(job)
            return True
        except queue.Full:
            pass
        try:
            held = self._stereo_q.get_nowait()
        except queue.Empty:                       # 双目线程刚好取走了
            self._stereo_q.put_nowait(job)
            return True
        if held[11] is not None:                  # 槽里是关键帧：放回去，这一帧丢
            self._stereo_q.put_nowait(held)
            self._stereo_drops["kf" if is_keyframe else "frame"] += 1
            return False
        self._stereo_q.put_nowait(job)
        self._stereo_drops["replaced"] += 1
        return True

    # ---- 双目解耦线程 ----
    def _stereo_worker_loop(self) -> None:
        """独立线程：视差 → 点云 → 近距判定 → 关键帧。

        **只做重活，不碰采集**：采集留在感知线程，因为 D3D11 立即上下文
        不是线程安全的（见 OpenVRSensors 文档串）。这里拿到的都是已复制好的 numpy 数组。

        关键帧判定（KeyframePolicy）仍在感知线程做 —— 它按 tick 算位移/转角/时长，
        必须在位姿镜的时钟上；这里只消费已经算好的 kf_kind。
        """
        c = self.cfg
        s = c.world_scale
        gen = self._gen
        matcher = None
        orb = None
        while not self._stop.is_set():
            try:
                job = self._stereo_q.get(timeout=0.1)
            except queue.Empty:
                continue
            (seq, tick, left, right, left_gray, rgb, r_ob,
             fx, cx, cy, baseline_m, kf_kind, T, dist_m) = job
            t0 = time.perf_counter()
            c0 = time.thread_time()
            try:
                if matcher is None:
                    matcher = self._make_matcher(fx)
                # LAS2 喂 RGB（比 gray3 EPE 低 5.5–9.5%）；传感器没给 RGB 就退回灰度。SGBM 只吃灰度。
                if rgb is not None and matcher.describe()["kind"] == "las2":
                    disp = matcher.compute(rgb[0], rgb[1])
                else:
                    disp = matcher.compute(left, right)
                t1, c1 = time.perf_counter(), time.thread_time()
                if gen != self._gen:
                    return                        # 编译期间会话已换（见 _gen）：这一帧不属于新会话
                pts = stereo_points(left, right, fx=fx, cx=cx, cy=cy, baseline_m=baseline_m,
                                    disp=disp, max_range_m=c.mapper.range_m)
                t2, c2 = time.perf_counter(), time.thread_time()
                near, n = near_obstacle(
                    pts, r_ob, cam_h=self._cam_h, scale=s, ahead_m=c.stop_ahead_m,
                    half_width_m=c.radius_m, ground_tol_m=c.mapper.ground_tol_m,
                    top_m=c.mapper.obst_top_m, min_pts=c.stop_min_pts)
                t3, c3 = time.perf_counter(), time.thread_time()
                stage = {"disp": t1 - t0, "points": t2 - t1, "near": t3 - t2, "total": t3 - t0,
                         "disp_cpu": c1 - c0, "points_cpu": c2 - c1, "near_cpu": c3 - c2,
                         # tick 之后到这里开算：采集复制 + 排队。只有时钟是 perf_counter 时才可比。
                         "since_tick": (t0 - tick) if self._clock is time.perf_counter else math.nan}
                stage = {k: v * 1000.0 for k, v in stage.items()}
                lt = getattr(matcher, "last_timing", None)
                if lt:
                    stage.update({f"las2_{k}": v for k, v in lt.items()})
                stage["n_pts"] = float(len(pts))
                with self._state_lock:
                    self._near, self._near_pts = near, n
                    self._stereo_at = tick
                    self._stereo_ms = (t3 - t0) * 1000.0
                    self._stereo_stages.append(stage)
                if kf_kind is None:
                    continue
                feat = None
                if self.loops is not None or self._mem is not None:
                    if orb is None:
                        orb = cv2.ORB_create(nfeatures=c.loop.orb_features)
                    feat = extract_features(left, disp, fx=fx, cx=cx, cy=cy,
                                            baseline_m=baseline_m, orb=orb,
                                            max_depth_m=c.loop.max_depth_m)
                with self._state_lock:
                    kid = self._kf_next_id
                    self._kf_next_id += 1
                self._commit_keyframe(kid, pts, T, feat, tick, kf_kind, self._mem,
                                      rgb[0] if rgb is not None else None, rgb, (left, right))
            except Exception as exc:  # noqa: BLE001 - 单帧失败不能带走整个线程
                self._fail("stereo", exc)
            finally:
                # 这个 worker 的两端本来就用 perf_counter；旧写法 Event.wait(20 ms) 实测 30.95 ms
                # （向上取整到 2 个 15.625 ms tick）。sleep_until 不受 tick 影响。
                deadline = t0 + c.stereo_min_period_s
                if deadline > time.perf_counter():
                    sleep_until(deadline, clock=time.perf_counter, stop=self._stop, spin_s=0.0)

    # ---- 建图线程 ----
    def _mapping_loop(self) -> None:
        c = self.cfg
        dirty = False
        last_update = -math.inf
        timeout = c.map_min_interval_s
        rec = self.recorder
        while not self._stop.is_set():
            self._map_event.wait(timeout=timeout)
            self._map_event.clear()
            if self._stop.is_set():
                return
            timeout = c.map_min_interval_s
            with self._state_lock:
                pending, self._pending = self._pending, []
            try:
                # mapper / loops 只有本线程写，灌数据和栅格化都不拿 _nav_lock；
                # 控制和 status 只在取意图快照、换结果时和这里抢一下锁。
                for i, (kind, item) in enumerate(pending):
                    if self._stop.is_set():
                        # 停止：没处理的原样放回 _pending 队首，交给 _stop_session 只写录制。
                        # 积压时一批能有几十帧（20261025_202630 末尾 28 s），整批跑完 join 会超时。
                        with self._state_lock:
                            self._pending[:0] = pending[i:]
                        return
                    if kind == "kf":
                        k, pts, T, dist, feat, t_kf, refresh, thumb, imgs = item
                        dirty = True
                        found = []
                        if self.loops is None:
                            self.mapper.add_keyframe(k, pts, T, osc_dist_m=dist, t_s=t_kf)
                        else:
                            found = self.loops.add_keyframe(k, T, dist, feat)
                            # 关键帧按修正后的位姿入图；回环后所有关键帧整体换位姿。
                            self.mapper.add_keyframe(k, pts, self.loops.pose(k), osc_dist_m=dist, t_s=t_kf)
                        if refresh:
                            # 原地补帧：上一帧的点云让给这帧；它的位姿、轨迹、回环节点都留着。
                            self.mapper.drop_points(k - 1)
                            self._kf_refreshed += 1
                        # 位置去重只对**真正落盘**的帧计数（补帧判在下面，但它不新占一个地点）。
                        # 只在建图线程写 `_kf_unique` / `_kf_last_xy`，感知线程读不到，无竞争。
                        xy2 = np.asarray(T, float)[:2, 3]
                        if self._kf_last_xy is None or not np.allclose(xy2, self._kf_last_xy, atol=1e-6):
                            self._kf_unique += 1
                            self._kf_last_xy = xy2
                        mem = self._mem
                        if mem is not None:
                            self._mem_rows[k] = (T.copy(), float(dist), t_kf - (self._started_at or t_kf))
                            mem.keyframe(k, feat, thumb, refresh)
                        if self._xs is not None:
                            try:
                                # 跨会话查询+验证；T 的旋转来自 HMD（yaw 门的外部真值）。
                                # 返回值 = 本帧确认的约束 —— 先验消费拿它当 gauge 的原料。
                                confirmed = self._xs.on_keyframe(k, feat, float(dist), T[:3, :3], refresh)
                                if confirmed and self._prior is not None:
                                    self._prior.add_constraints(confirmed)
                            except Exception as exc:  # noqa: BLE001 - 检索出错禁用自身，不杀建图线程
                                self._fail("xsession", exc)
                                self._xs = None
                        if rec is not None:
                            rec.keyframe(k, pts, T, T if self.loops is None else self.loops.pose(k), dist, feat,
                                         images=imgs)
                            rec.line("events", {"kind": "kf", "k": k, "t": t_kf, "dist_m": dist, "loops": found,
                                                "refresh": refresh})
                        if self.loops is not None:
                            if found:
                                moved = self.mapper.update_poses(self.loops.poses())
                                self._last_loop = {"keyframe": k, "loops": found,
                                                   "map_shift_m": round(moved * c.world_scale, 3)}
                                if self._prior is not None:
                                    # 回环改了全表位姿 ⇒ 会话帧内容变了，先验的 gauge 必须重估。
                                    self._prior.note_map_moved()
                            loop_status = self.loops.status()
                            with self._state_lock:
                                self._offset = self.loops.offset()
                                self._loop_status = loop_status
                        self._rate_kf.mark(self._clock())
                        self._keyframes += 1
                    elif kind == "trail":
                        k, T, dist, t_tr = item
                        if rec is not None:
                            rec.line("events", {"kind": "trail", "k": k, "t": t_tr, "dist_m": dist, "T_dr": T})
                        if self.loops is not None and k in self.loops:
                            # trail 存成相对关键帧的位姿，用该关键帧的修正量换系即可。
                            T = self.loops.correct_at(k, T)
                        self.mapper.add_trail(k, T, osc_dist_m=dist)
                    else:                   # kick：换了目标，立刻重规划
                        dirty = True
                if rec is not None:
                    rec.flush()
                mem = self._mem
                # ⚠️ 录制**自带** `kf/<id>.npz`（且是权威副本：位置去重、终态位姿都从它算），
                # 而会话侧的位姿表会覆盖同一批文件。有录制器时这个周期转存跳过：
                # 会话内的消费方（xsession 的 `write_back`）读的是内存里的 `_mem_rows`，
                # 不读这个文件，所以跳过一个信息损失都没有，省掉的是每 `pose_flush_s`
                # 一次的全表重写（`kf_max_age_s` 的长停顿会话里 frames 极多）。
                if mem is not None and self._clock() - self._mem_flushed_at >= mem.cfg.pose_flush_s:
                    if rec is not None:
                        self._mem_flushed_at = self._clock()
                    else:
                        # 位姿整表写：回环会改之前所有帧的位姿，逐帧写的会过期。
                        table = self._mem_table()
                        if table is not None:
                            mem.poses(table)
                            self._mem_flushed_at = self._clock()
                if dirty and len(self.mapper) and not self._stop.is_set():
                    wait = c.map_min_interval_s - (self._clock() - last_update)
                    if wait > 0.0:
                        # 节流中：不等下一个关键帧，时间一到就补这次更新。
                        timeout = wait
                    else:
                        t0 = time.perf_counter()
                        with self._nav_lock:
                            snap = self.session.snapshot()
                        res = self.session.compute(self._est(), snap)
                        with self._nav_lock:
                            self.session.apply(res)
                        self._cam_h = self.mapper.cam_h
                        self._map_ms = (time.perf_counter() - t0) * 1000.0
                        self._map_updates += 1
                        last_update = self._clock()
                        dirty = False
                        self._update_coverage(res)
            except Exception as exc:  # noqa: BLE001
                self._fail("mapping", exc)

    def _request_map_update(self) -> None:
        with self._state_lock:
            self._pending.append(("kick", None))
        self._map_event.set()

    # ---- 控制线程 ----
    def _control_loop(self) -> None:
        c = self.cfg
        period = 1.0 / c.control_hz
        # 与感知循环同一个道理：Event.wait 被 15.625 ms tick 量化，100 ms 的周期实测 109 ms
        # （9.17 Hz）。Pacer 绝对截止点 + sleep，实测 10.00 Hz。
        # spin_s=0：控制环只要不晚于 100 ms 就够，纯 sleep 的 p99 是 0.79 ms，远超需求。
        # 为什么刻意关掉自旋：两个循环同时自旋时（switchinterval=5 ms），位姿的
        # late_p99 会从 0.02 ms 涨到 4.16 ms（双峰，最差 8.10 ms）—— 另一个线程的
        # 自旋抢 GIL。控制环不值得为 0.79 ms 的精度付这个代价。
        pacer = Pacer(hz=c.control_hz, clock=self._clock, stop=self._stop, name="control", spin_s=0.0)
        pacer.reset()
        for _tk in pacer.ticks():
            if _tk.stopped:
                break
            tick = _tk.actual
            try:
                self._control_tick(tick)
            except Exception as exc:  # noqa: BLE001 - 控制出错先停车
                self._fail("control", exc)
                self._halt()
            # 运行中也要能读到（旧写法只在 finally 写 ⇒ 活着的会话里 pacer.control 永远缺失）。
            if _tk.index % 10 == 0:
                with self._state_lock:
                    self._pacer_telemetry["control"] = pacer.telemetry()
        with self._state_lock:
            self._pacer_telemetry["control"] = pacer.telemetry()
        self._halt()

    def _control_tick(self, tick: float) -> None:
        c = self.cfg
        with self._nav_lock:
            mode = self.session.mode
        if mode in ("idle", "done"):
            self._drive_block = None
            self._halt()
            return
        block = self._drive_block_reason()
        est = self._est()
        with self._state_lock:
            stereo_at, near = self._stereo_at, self._near
        if block is None and est.get("state") != "localized":
            block = est.get("reason", "pose_stale")
        # 同 _est()：max(0.0, ...) 只保证报告值为非负，**不**能让跨域错误触发出停车。
        # 安全性来自 tick 与 _stereo_at 同域（都是 Pacer 的 Tick.actual / perf_counter）。
        if block is None and (stereo_at is None or max(0.0, tick - stereo_at) > c.stereo_stale_s):
            # 双目断了就没有近距急停，不能盲走。
            block = "stereo_stale"
        self._drive_block = block
        if block is not None:
            self._halt()
            return
        with self._nav_lock:
            out = self.session.step(est, local_stop=near)
        self._last_step = {k: out.get(k) for k in ("state", "reason", "forward", "turn_rate")}
        forward = float(out.get("forward") or 0.0)
        turn_rate = float(out.get("turn_rate") or 0.0)
        if forward <= 0.0 and turn_rate == 0.0:
            self._halt()
            return
        self._moving = True
        if turn_rate != 0.0:
            # correction_deg 相对**当前实际** yaw 算目标，每拍重发不会累加超调。
            self._send_turn(math.degrees(turn_rate) / c.control_hz)
        if forward > 0.0:
            self._send_move(min(1.0, forward), c.move_hold_ms)

    # ---- 对外 API（导航系世界米）----
    def goto(self, x: float, y: float) -> dict[str, Any]:
        if not (math.isfinite(x) and math.isfinite(y)):
            return {"ok": False, "reason": "target_not_finite"}
        if not self.running:
            return {"ok": False, "reason": "navmesh_not_running"}
        with self._nav_lock:
            if not len(self.mapper):
                return {"ok": False, "reason": "no_keyframes_yet"}
            self.session.goto((float(x), float(y)))
        self._request_map_update()
        return {"ok": True, "mode": "goto", "goal_xy_m": [float(x), float(y)]}

    def explore(self) -> dict[str, Any]:
        if not self.running:
            return {"ok": False, "reason": "navmesh_not_running"}
        with self._nav_lock:
            self.session.explore()
        self._request_map_update()
        return {"ok": True, "mode": "explore"}

    def cancel(self) -> dict[str, Any]:
        with self._nav_lock:
            self.session.cancel()
        self._halt()
        return {"ok": True, "mode": "idle"}

    def _rate_summary(self) -> dict[str, Any]:
        """实测频率 —— 与配置值**并排**报，缺口自己会说话。

        `stereo_hz` 低于 `stereo_hz_target` 时**先看 `stereo_ticks`**：

        * ≈ 整数 ⇒ 门撞在**量化台阶**上（门按位姿 tick 推进，0.1 s 只能等到第 2 个 tick）
          —— 这时调 `stereo_period_s` 没用，要先把位姿循环喂饱；
        * 明显非整数 ⇒ 才真的是 SGBM 算不过来。

        2026-10-18 实测 223033：pose 15.87 Hz（目标 20）、stereo 7.94 Hz（目标 10）、
        `stereo_ticks = 2.00`，关键帧全程平均 1.86 Hz。**该症状已于 2026-10-24 根治**：
        根因是 Windows 定时器 —— `Event.wait` 被量化到 15.625 ms tick、`time.monotonic`
        本身就是 `GetTickCount64()`（读数台阶 16 ms），叠加后 50 ms 的周期实测 62 ms。
        改用 `backend/pacing.py` 的 Pacer（perf_counter + sleep + 自旋）与 Gate（绝对累加）后，
        真机 SteamVR 实测 pose 20.00 Hz / stereo 10.00 Hz。⇒ **`stereo_ticks` 若再出现
        非整数且远低于 target，先看 `pacer` 里的 `late_p99_ms` / `skipped_total`。**
        """
        pose_gap = self._rate_pose.gap_p50_s()
        stereo_gap = self._rate_stereo.gap_p50_s()
        return {
            "pose_hz": _rnd(self._rate_pose.hz(), 2),
            "pose_hz_target": self.cfg.pose_hz,
            "pose_gap_p50_s": _rnd(pose_gap, 4),
            "stereo_hz": _rnd(self._rate_stereo.hz(), 2),
            "stereo_hz_target": _rnd(1.0 / self.cfg.stereo_period_s, 2),
            "stereo_gap_p50_s": _rnd(stereo_gap, 4),
            "stereo_ticks": (_rnd(stereo_gap / pose_gap, 2)
                             if pose_gap and stereo_gap else None),
            # 节拍器自报的实测（perf_counter 域，不受 15.6 ms 量化污染）：
            # late_p99_ms = 抖动、skipped_total = 掉拍、spin_frac = 真正自旋的比例。
            # 这三个是「声明 vs 实测」里实测的那一半，别只信上面 RateMeter 的 hz。
            "pacer": dict(self._pacer_telemetry),
            # 关键帧不是周期量（位移/转角/时长三选一触发），所以它没有 target。
            # 它会随走路速度在「卡在 stereo 台阶上」和「原地 3 s 补一帧」之间摆。
            "keyframe_hz": _rnd(self._rate_kf.hz(), 2),
            "stereo_drops": dict(self._stereo_drops),
            "keyframe_gap_p50_s": _rnd(self._rate_kf.gap_p50_s(), 4),
            "samples": {"pose": self._rate_pose.samples(),
                        "stereo": self._rate_stereo.samples(),
                        "keyframe": self._rate_kf.samples()},
        }

    def status(self) -> dict[str, Any]:
        est = self._est()
        with self._state_lock:
            near, near_pts, stereo_at = self._near, self._near_pts, self._stereo_at
            errors, info, stereo_ms = list(self._errors), self._sensor_info, self._stereo_ms
            baseline_check = self._baseline_check
            loop_status, last_loop = self._loop_status, self._last_loop
            yaw_jumps = list(self._yaw_jumps)
        with self._nav_lock:
            goal = self.session.goal_xy()
            last = dict(self.session.last)
            mode = self.session.mode
        cam_h = self._cam_h
        loop = None if loop_status is None else {**loop_status, "last": last_loop}
        now = self._clock()
        return {
            "running": self.running,
            "mode": mode,
            # 高度分带（旁路产物，不进导航）：obst_top_m 之上的几何量。044153 实测 28.7% 的
            # 双目点落在那儿，此前被整段丢弃——多层建筑/天桥/天花板对地面层是不可见的。
            "height_bands": self._band_summary(),
            # 回环只修平移（朝向来自 HMD），修正发生在关键帧上；关掉时地图与位姿共用一条漂移链。
            "pose_source": "dead_reckoning" if self.loops is None else "dead_reckoning+loop_closure",
            "loop_closure": loop,
            "frame": "nav_map_xy_world_m",
            "pose": None if est.get("state") != "localized" else {
                "xy_m": [round(v, 3) for v in est["xy"]], "theta_rad": round(est["theta"], 4),
                "sigma_m": est["sigma_m"]},
            "pose_state": est.get("state"),
            "hmd_yaw_jumps": yaw_jumps,
            "odometry_distance_m": round(self.dr.dist_m, 2),
            "osc_samples": self.dr.samples,
            # 朝向取样的现场读数：segments 应随路程持续增长；fallbacks 只在启动头几帧、
            # 位姿长空档里出现。segments 恒为 0 = 取样器没接上（dr_heading_midpoint 被关或历史没喂）。
            "dr_heading": {"midpoint": bool(self.cfg.dr_heading_midpoint),
                            "segments": self.dr.heading_segments,
                            "fallbacks": self.dr.heading_fallbacks,
                            "track_span_s": round(self.heading_track.span_s(), 2)},
            # 静默封顶的现场证据：丢包丢掉"停下"那个 0 包时，位姿不再被最后一个速度带着跑。
            # dropped_m 是**没算进** odometry_distance_m 的那部分，不是误差条。
            "odometry_holdout": {
                "zoh_max_s": self.cfg.osc_zoh_max_s,
                "zoh_fade_s": self.cfg.osc_zoh_fade_s,
                "last_sample_age_s": (None if self.dr.last_sample_at is None
                                      else round(max(0.0, now - self.dr.last_sample_at), 2)),
                # 本会话最长"两端都非零"的静默段：与 zoh_max_s 对照看，逼近它就说明该调大。
                "max_legit_gap_s": round(self.dr.max_legit_gap_s, 2),
                "dropped_s": round(self.dr.holdout_s, 1),
                "dropped_m": round(self.dr.holdout_m, 2),
            },
            "goal_xy_m": None if goal is None else [round(goal[0], 3), round(goal[1], 3)],
            "keyframes": self._keyframes,
            # 原地补帧（点云替换上一帧）与快转推迟的次数；地图里有点云的关键帧 = keyframes − refreshed。
            "keyframes_refreshed": self._kf_refreshed,
            # 位置去重后的关键帧数，以及由此得到的重复率。**长停顿的会话会很高**（实测
            # 045615 有 41.5%），而它同时是离线拟合的权重要紧事与位姿图 σ 的要紧事
            # ⇒ 不做成事后翻盘才能发现的量。reason 让"为什么高"当场可读。
            "keyframes_unique": self._kf_unique,
            "keyframes_duplicate_frac": (round(1.0 - self._kf_unique / self._keyframes, 4)
                                          if self._keyframes else None),
            "keyframe_dedup_reason": (
                None if self._keyframes < 2 else
                "ok" if self._kf_unique >= 0.98 * self._keyframes else
                "standing_still_refresh" if self._kf_refreshed >= 0.1 * (self._keyframes - self._kf_unique)
                else "refresh_and_repeat"),
            "keyframes_deferred": 0 if self._kf_policy is None else self._kf_policy.deferred,
            "map_updates": self._map_updates,
            "map_update_ms": None if self._map_ms is None else round(self._map_ms, 1),
            "stereo_ms": None if stereo_ms is None else round(stereo_ms, 1),
            "stereo_stages": self._stage_summary(),
            "stereo_backend": self.stereo_backend_status(),
            "stereo_age_s": None if stereo_at is None else round(now - stereo_at, 2),
            "rates": self._rate_summary(),
            # cam_h 是**追踪米**（双目单位），而 status 的对外契约是导航系世界米（见模块
            # docstring）。两个口径都给出、键名自带单位，避免"读到 1.7 就当成世界米"——
            # 那正是 C9 六个 cam_h 值的来源（1.73 追踪米 ≡ 1.306 世界米）。
            "camera_height_m": round(cam_h * self.cfg.world_scale, 3),
            "camera_height_track_m": round(cam_h, 3),
            "near_obstacle": {"stop": near, "points": near_pts, "ahead_m": self.cfg.stop_ahead_m},
            "drive_block": self._drive_block,
            "last_step": dict(self._last_step),
            "last_plan": last,
            "sensors": info,
            "baseline_check": baseline_check,
            "world_scale": self.cfg.world_scale,
            "avatar_scale": self._avatar_scale_status(),
            "errors": errors,
            "uptime_s": None if self._started_at is None else round(now - self._started_at, 1),
            "recording": None if self.recorder is None else self.recorder.status(),
            "memory": self._mem.status() if self._mem is not None else {"active": False, "reason": self._mem_reason},
            "xsession": self._xs.status() if self._xs is not None else {"active": False, "reason": self._xs_reason},
            # 世界先验消费（P0.3b）：state = ok / gauge_*（质量闸）/ no_prior / ...；applied = 最近一次
            # 叠进栅格的格数（in_crop / applied_free / applied_occ / blocked_walked）。
            "prior": (self._prior.status() if self._prior is not None else
                      {"enabled": bool(self.cfg.prior.enabled),
                       "state": "no_xsession" if self._xs is None else "not_started"}),
            # P0.3b 会话末自动采纳：None=还没跑到，running=后台线程在算，其余为结果/跳过原因
            "xsession_align": ({"state": "running"} if self._align_out is None and self._align_thread is not None
                               and self._align_thread.is_alive() else
                               (self._align_out if self._align_out is not None
                                else {"state": "pending"})),
        }

    def grid_png(self) -> str | None:
        view = self.grid_view()
        return None if view is None else view["png_base64"]

    # ---- 覆盖可视化（纯显示）----
    def _update_coverage(self, res: dict[str, Any]) -> None:
        """建图线程内（apply 之后、_nav_lock 之外）算覆盖伴生快照：粗格分类 + 走过 hull + 诚实指标。

        分类规则（粗格，0.30 m 追踪格）：occ=黑（障碍叠加）；hull 内 total==0 = 信息洞·未观测（亮紫红）；
        total>0 且 near==0 = 信息洞·只远看（橙）；0<near<cov_sparse_pts = 融合洞提示（黄）；
        near≥cov_sparse_pts = 高质量（绿）；hull 外未观测 = 深灰（不算洞，不把没走过的世界报成欠账）。
        纯显示：这里出任何异常只记进快照，不许拖垮建图线程。"""
        t0 = time.perf_counter()
        try:
            cc = self.mapper.coverage_counts()
            ng = res["ng"]
            if cc is None:
                return
            c = self.cfg
            g = ng.grid                       # 行 0 = +y 最大（显示布局）
            h, w = g.shape
            m = ng.meta
            res_m, s = m.resolution_m, m.world_scale
            f = max(1, int(round(c.cov_coarse_m / res_m)))
            Hc, Wc = -(-h // f), -(-w // f)
            # 累加器布局 → 主格布局窗口裁剪（与 rasterize 同一套对齐；主格行 0 = min y）。
            near = np.zeros((h, w))
            far = np.zeros((h, w))
            (x0, y0), (Ha, Wa) = cc["lo"], cc["near"].shape
            lx, ly = int(round(m.origin_xy_m[0] / res_m)), int(round(m.origin_xy_m[1] / res_m))
            ox0, oy0 = max(lx, x0), max(ly, y0)
            ox1, oy1 = min(lx + w, x0 + Wa), min(ly + h, y0 + Ha)
            if ox1 > ox0 and oy1 > oy0:
                near[oy0 - ly:oy1 - ly, ox0 - lx:ox1 - lx] = cc["near"][oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
                far[oy0 - ly:oy1 - ly, ox0 - lx:ox1 - lx] = cc["far"][oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]

            def pool(a: np.ndarray) -> np.ndarray:
                # 细格 → 粗格求和池化（观测是加性量）；补零到 f 的倍数。
                pad_h, pad_w = Hc * f - h, Wc * f - w
                if pad_h or pad_w:
                    a = np.pad(a, ((0, pad_h), (0, pad_w)))
                return a.reshape(Hc, f, Wc, f).sum(axis=(1, 3))

            # 转到显示布局（行 0 = +y 最大），与 g 一致。
            near_c, far_c = pool(near)[::-1], pool(far)[::-1]
            occ_c = pool((g[::-1] == OCC).astype(np.float64))[::-1] > 0.5
            # hull = 走过折线（OSC 门控，walked() 按 _ver 缓存）按 cov_hull_m 半径膨胀的粗格掩码。
            cw = c.cov_coarse_m * s           # 粗格世界米
            hull = np.zeros((Hc, Wc), np.uint8)
            thick = max(1, int(round(2 * c.cov_hull_m / cw)))
            for poly in self.mapper.walked():
                p = np.asarray(poly, float)
                cols = np.floor((p[:, 0] / s - m.origin_xy_m[0]) / c.cov_coarse_m).astype(np.int32)
                rows = (Hc - 1
                        - np.floor((p[:, 1] / s - m.origin_xy_m[1]) / c.cov_coarse_m).astype(np.int32))
                pts = np.column_stack([cols, rows]).astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(hull, [pts.reshape(-1, 1, 2)], False, 1, thick)
            hullb = hull > 0
            total_c = near_c + far_c
            cls = np.zeros((Hc, Wc), np.uint8)
            cls[total_c > 0] = 1                                  # 只远看
            cls[(near_c > 0) & (near_c < c.cov_sparse_pts)] = 2   # 近看但稀
            cls[near_c >= c.cov_sparse_pts] = 3                   # 高质量
            cls[(total_c == 0) & hullb] = 4                       # 信息洞·未观测
            hull_cells = int(hullb.sum())
            observed_cells = int((hullb & (total_c > 0)).sum())
            coverage_ratio = observed_cells / hull_cells if hull_cells else 0.0
            near_ratio = float((hullb & (near_c > 0)).sum()) / hull_cells if hull_cells else 0.0
            # frontier 边界（细格、模式无关）：观测 free 紧挨 unknown。
            touch = cv2.dilate((g == UNK).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            frontier_cells = int(((g == FREE) & touch).sum())
            self._cov_hist.append({
                "t_s": round(self._clock() - (self._started_at if self._started_at is not None else self._clock()), 1),
                "coverage_ratio": round(coverage_ratio, 3), "near_ratio": round(near_ratio, 3),
                "frontier_cells": frontier_cells})
            img = np.full((Hc, Wc, 3), 60, np.uint8)              # hull 外未观测 = 深灰
            for v, col in ((1, (0, 140, 255)), (2, (0, 215, 255)), (3, (60, 180, 75)), (4, (255, 0, 255))):
                img[cls == v] = col
            img[occ_c] = (0, 0, 0)
            # 走过中心线（浅蓝）+ 当前位姿（红）+ 目标（绿），与 grid_view 同色约定。
            for poly in self.mapper.walked():
                p = np.asarray(poly, float)
                cols = np.floor((p[:, 0] / s - m.origin_xy_m[0]) / c.cov_coarse_m).astype(np.int32)
                rows = (Hc - 1
                        - np.floor((p[:, 1] / s - m.origin_xy_m[1]) / c.cov_coarse_m).astype(np.int32))
                pts = np.column_stack([cols, rows]).astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 190, 120), 1)
            est = self._est()
            if est.get("state") == "localized":
                cc_r, cc_c = self._world_to_cov_cell(est["xy"], m, c.cov_coarse_m, Hc)
                cv2.circle(img, (cc_c, cc_r), 2, (0, 0, 255), -1)
            goal = self.session.goal_xy()
            if goal is not None:
                gg_r, gg_c = self._world_to_cov_cell(goal, m, c.cov_coarse_m, Hc)
                cv2.circle(img, (gg_c, gg_r), 2, (0, 200, 0), -1)
            self._cov = {
                "available": True, "seq": self._map_updates,
                "metrics": {
                    "hull_cells": hull_cells, "observed_cells": observed_cells,
                    "coverage_ratio": round(coverage_ratio, 3), "near_ratio": round(near_ratio, 3),
                    "hull_m2": round(hull_cells * cw * cw, 1), "observed_m2": round(observed_cells * cw * cw, 1),
                    "hole_cells": int((hullb & (total_c == 0)).sum()),
                    "far_only_cells": int((hullb & (total_c > 0) & (near_c == 0)).sum()),
                    "frontier_cells": frontier_cells,
                    "frontier_goals": res["info"].get("frontiers"),
                    "keyframes": self._keyframes, "map_updates": self._map_updates,
                    "near_m": float(cc["cov_near_m"]), "coarse_m": float(c.cov_coarse_m),
                    "cov_ms": round((time.perf_counter() - t0) * 1000.0, 2)},
                "grid": {"rows": int(Hc), "cols": int(Wc),
                         "origin_xy_track_m": [float(m.origin_xy_m[0]), float(m.origin_xy_m[1])],
                         "resolution_track_m": float(c.cov_coarse_m), "world_scale": float(s),
                         "frame": "nav_map_xy_world_m"},
                "trend": list(self._cov_hist),
                "img": img}
        except Exception as exc:  # noqa: BLE001 - 显示功能不出现在建图线程的错误里
            self._cov = {"available": False, "seq": self._map_updates,
                         "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _world_to_cov_cell(xy_world: Sequence[float], meta: Any, coarse_m: float,
                           rows: int) -> tuple[int, int]:
        x = xy_world[0] / meta.world_scale - meta.origin_xy_m[0]
        y = xy_world[1] / meta.world_scale - meta.origin_xy_m[1]
        return rows - 1 - int(math.floor(y / coarse_m)), int(math.floor(x / coarse_m))

    def _band_summary(self) -> dict[str, Any] | None:
        """高度分带的**摘要**（给 status 看；原始 (4,h,w) 数组走 ``mapper.band_grid()``）。

        只报"每条带有多少个格见过东西、总共多少点"——不替上层下"这是障碍还是楼板"的结论，
        那个要带内高度聚类，目前还没做。cells 是格子数、points 是原始点计数。
        """
        bc = self.mapper.band_counts()
        if bc is None:
            return None
        edges = bc["edges_m"]
        out: dict[str, Any] = {"edges_m": edges, "res_m": bc["res_m"], "bands": {}}
        for name in ("below", "lo", "mid", "hi"):
            a = bc[name]
            out["bands"][name] = {"cells": int((a > 0.5).sum()), "points": int(a.sum())}
        out["obst_top_m"] = float(self.mapper.cfg.obst_top_m)
        return out

    def coverage_view(self) -> dict[str, Any] | None:
        """覆盖快照（HTTP 线程只读 mapping 线程产出的整 dict 引用）；PNG 按快照代数惰性编码缓存。"""
        cov = self._cov
        if cov is None:
            return None
        out = {k: v for k, v in cov.items() if k != "img"}
        img = cov.get("img")
        if isinstance(img, np.ndarray):
            seq = cov["seq"]
            if self._cov_png is None or self._cov_png[0] != seq:
                ok, buf = cv2.imencode(".png", img)
                png = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
                self._cov_png = (seq, png)
            out["grid"]["png_base64"] = self._cov_png[1]
        return out

    def grid_view(self, layer: str = "tristate") -> dict[str, Any] | None:
        """当前栅格的 base64 PNG（一格一像素，第 0 行 = 北/+y 最大）+ 像素↔导航系世界米的换算。

        ``layer`` 选画什么底图（``_GRID_LAYERS``）：

        * ``tristate`` 白 = 可走中心区，浅灰 = 观测 free，深灰 = unknown，黑 = 障碍
        * ``prior``    世界先验在本场会话帧的投影：**绿/红 = 先验真补进去的格**，
                        暗绿/暗红 = 先验有意见但 live 已有证据（没被采纳），其余为 live 灰阶
        * ``surface``  头顶 ``obst_top_m`` 之上那张**面**的离地高（热力图，0–``hi_band_top_m`` m）
        * ``bands``    四条高度带里**点数最多**的那一条（离散色，见 ``_BAND_COLORS``）
        * ``clearance`` 到最近非可走格的距离（热力图，0–1 m）

        路径 / 位姿 / 目标在**所有**图层上都画 —— 否则没法把地形和"实际走哪儿"对起来。
        配色范围是**固定**的（不从数据取分位），这样两帧之间颜色可直接比较；
        每层都随 ``ramp`` 返回自己的取值范围，UI 照着画色条。
        """
        with self._nav_lock:
            ng = self.session.ng
            if ng is None:
                return None
            g = ng.grid
            if layer == "tristate":
                img = np.full(g.shape + (3,), 90, np.uint8)
                img[g == FREE] = (170, 170, 170)
                img[g == OCC] = (0, 0, 0)
                img[ng.center] = (255, 255, 255)
                ramp: dict[str, Any] = {}
            elif layer == "prior":
                img, ramp = self._prior_layer(ng, g)
            elif layer == "surface":
                img, ramp = self._surface_layer(ng, g)
            elif layer == "bands":
                img, ramp = self._band_layer(ng, g)
            elif layer == "clearance":
                img, ramp = _ramp_layer(ng.clearance, float(np.nanmax(ng.clearance) or 0.0),
                                        lambda v: np.isfinite(v), 0.0, 1.0, "m")
                img[~ng.center] = (90, 90, 90)
            else:
                raise ValueError(f"layer 只能是 {_GRID_LAYERS}，收到 {layer!r}")
            contract = self.session.contract
            goal = self.session.goal_xy()
        if contract is not None and contract.accepted and len(contract.waypoints_xy_m) >= 2:
            pts = np.array([ng.to_cell(p)[::-1] for p in contract.waypoints_xy_m], np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 120, 0), 1)
        if goal is not None:
            r, cc = ng.to_cell(goal)
            cv2.circle(img, (cc, r), 2, (0, 200, 0), -1)
        est = self._est()
        if est.get("state") == "localized":
            r, cc = ng.to_cell(est["xy"])
            cv2.circle(img, (cc, r), 2, (0, 0, 255), -1)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return None
        m = ng.meta
        return {"png_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "rows": int(g.shape[0]), "cols": int(g.shape[1]),
                "layer": layer, "ramp": ramp,
                # 列 c、行 r 的格中心：x = (ox + (c+0.5)·res)·s，y = (oy + (rows−1−r+0.5)·res)·s
                "origin_xy_track_m": [float(m.origin_xy_m[0]), float(m.origin_xy_m[1])],
                "resolution_track_m": float(m.resolution_m), "world_scale": float(m.world_scale),
                "frame": "nav_map_xy_world_m"}

    # ---- 深度/高度相关图层（只读旁路，不参与任何判定）----
    def _prior_layer(self, ng, g) -> tuple[np.ndarray, dict[str, Any]]:
        """世界先验（P0.3b）在本场会话帧的投影。**只显示，不参与任何判定**。

        配色只说一件事："这些格是先验补的，还是 live 自己看的"——
        亮绿/亮红 = 先验**真写进去了**（当时那格是 unknown，见 ``PriorOverlay.applied_mask``）；
        暗绿/暗红 = 先验有意见但 live 已有证据（或走廊规则挡下）⇒ 没被采纳。
        底图与三态同色，切回 tristate 就能看纯 live，两层对一眼就知道先验补得对不对。
        """
        img = np.full(g.shape + (3,), 90, np.uint8)
        img[g == FREE] = (170, 170, 170)
        img[g == OCC] = (0, 0, 0)
        info: dict[str, Any] = {"available": False, "reason": "no_prior"}
        if self._prior is None:
            info["reason"] = "disabled"
            return img, info
        ovl, pst = self._prior.view()
        if ovl is None:
            info["reason"] = str(pst.get("state") or "no_prior")
            return img, info
        row, col, inb = ovl.cells_of(ng)
        applied = ovl.applied_mask
        if applied is None or len(applied) != len(ovl.labels):
            # 还没跑过 apply_into（正常路径不会发生：compute 先叠加再 build）⇒ 只画位置不置色。
            applied = np.zeros(len(ovl.labels), bool)
        sel = np.flatnonzero(inb)
        rs, cs = row[inb], col[inb]
        lab, app = ovl.labels[inb], applied[inb]
        for is_occ in (False, True):
            want = (lab == (L_OCC if is_occ else L_FREE))
            img[rs[want & app], cs[want & app]] = _PRIOR_COLORS[("occ" if is_occ else "free")]
            img[rs[want & ~app], cs[want & ~app]] = _PRIOR_COLORS[("occ_ignored" if is_occ
                                                                   else "free_ignored")]
        info = {"available": True, "cells_in_crop": int(len(sel)),
                "applied_free": int((lab[app] == L_FREE).sum()),
                "applied_occ": int((lab[app] == L_OCC).sum()),
                "not_applied": int((~app).sum()),
                "state": str(pst.get("state") or ""),
                "base_sid": (pst.get("prior") or {}).get("base_sid")}
        return img, info

    def _surface_layer(self, ng, g) -> tuple[np.ndarray, dict[str, Any]]:
        """头顶那张**面**的离地高。用 ``surface_grid`` 的均值 —— 注意它系统性偏高约 0.1 m
        （见 ``Docs/建图实测能力边界（2026-10-05）.md` §二），**只用于肉眼找结构，不当米制**。"""
        sg = self.mapper.surface_grid()
        hi = float(self.mapper.cfg.hi_band_top_m)
        if sg is None:
            return np.full(g.shape + (3,), 90, np.uint8), {"available": False,
                                                          "reason": "hi_bands_disabled"}
        mean_h = sg["mean_h"]
        n = sg["n"]
        img, ramp = _ramp_layer(mean_h, hi, lambda v: np.isfinite(v) & (v > -1.0),
                                0.0, hi, "m")
        img[n <= 0.5] = (60, 60, 60)          # 没观测到面
        ramp.update({"available": True, "field": "surface_mean_h", "above_m": float(self.mapper.cfg.obst_top_m)})
        return img, ramp

    def _band_layer(self, ng, g) -> tuple[np.ndarray, dict[str, Any]]:
        """四条高度带里点数最多的那条。**不替上层下"这是障碍还是楼板"的结论** ——
        那个要带内高度聚类（``surface_counts`` docstring 原话），尚未做。"""
        bg = self.mapper.band_grid()
        if bg is None:
            return np.full(g.shape + (3,), 90, np.uint8), {"available": False,
                                                          "reason": "hi_bands_disabled"}
        stacked = np.stack(bg)                       # (4, h, w)：below / lo / mid / hi
        dom = np.argmax(stacked, axis=0)
        any_pts = stacked.max(axis=0) > 0.5
        img = np.full(g.shape + (3,), 55, np.uint8)
        for i, col in enumerate(_BAND_COLORS):
            img[(dom == i) & any_pts] = col
        edges = self.mapper.band_counts()["edges_m"] if self.mapper.band_counts() else []
        return img, {"available": True, "field": "dominant_height_band",
                     # ⚠️ 图例要用 **RGB**（前端写 background:rgb(...)），图里要用 BGR ——
                     # 同一份意图导出两套，见 _BAND_COLORS_RGB 的注释。
                     "bands": [{"name": n, "rgb": list(c)}
                               for n, c in zip(("below", "lo", "mid", "hi"), _BAND_COLORS_RGB)],
                     "edges_m": [float(v) for v in edges]}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---- 栅格底图图层 ----
# 为什么不加"地面高度"层：在线 mapper 每格只留 7 个数（4 个分带计数 + 3 个矩），
# **没有逐格的地面高度**。要它就得给热路径加累加器（`_kf_base` 每帧多两次 bincount），
# 而本轮改动先走"零热路径"路线。⇒ 现有四层里，深度信息最直接的是 `surface`
# （头顶 obst_top_m 之上那张面的绝对高度）—— 那是天花板/楼板的高度，不是脚下。
_GRID_LAYERS = ("tristate", "prior", "surface", "bands", "clearance")

# 四条高度带（below=地面以下 / lo=2~hi_band / mid / hi）的配色，顺序与
# ``MapperConfig`` 里 hb 的赋值顺序一致（nav_mapping.py:568-574）。
# below 用**冷色**：它代表"地面以下有东西"，是坑的候选，必须一眼能挑出来。
#
# ⚠️ 颜色有**两个口径**，写成一份必错（2026-10-06 修）：
#   * 图里的像素走 ``np 数组 → cv2.imencode`` ⇒ OpenCV 认 **BGR**；
#   * 图例（``_band_layer`` 回给前端的 ``rgb``）走 CSS ``background:rgb(...)`` ⇒ 认 **RGB**。
# 原来只有一份 RGB 意图，被当成 BGR 画进图里 ⇒ "below 冷色"实际显示成红，而且**图例与图不一致**。
# 现在意图只写一份（RGB），画图时取反：改动只此一处。
_BAND_COLORS_RGB = ((60, 60, 255), (40, 190, 255), (60, 210, 90), (40, 70, 240))
_BAND_COLORS = tuple(c[::-1] for c in _BAND_COLORS_RGB)      # BGR，只给 cv2 用

#: ``prior`` 图层的四色。⚠️ **BGR**：整条链是 ``np 数组 → cv2.imencode``，OpenCV 认的是 BGR，
#: 写成 RGB 会把红画成蓝（灰阶色调掩盖了这个坑，彩色的那几层才碰得到）。
#: UI 图例写 ``#rrggbb``（石绿/红/暗绿/暗红），两边必须成对改。
_PRIOR_COLORS = {
    "free": (150, 225, 150),          # #96e196 先验补 free（真写进去了）
    "occ": (60, 60, 225),             # #e13c3c 先验补 障碍
    "free_ignored": (110, 160, 110),  # #6ea06e 先验说 free，但 live 已有证据 ⇒ 没采纳
    "occ_ignored": (50, 50, 130),     # #823232 先验说障碍，但 live 已有证据 ⇒ 没采纳
}


def _ramp_layer(field: np.ndarray, vmax: float, valid, vmin: float, top: float,
                unit: str) -> tuple[np.ndarray, dict[str, Any]]:
    """标量场 → 热力图。范围**固定**（不从数据取分位），这样帧与帧之间颜色可比。

    ⚠️ NaN/Inf 必须在**转 uint8 之前**清掉：``(NaN*255).astype(uint8)`` 在 numpy 里是
    未定义行为（实测会打 ``invalid value encountered in cast`` 警告并写出垃圾像素）。
    做法是先按掩码把无效格压到 ``vmin``，再 clip、再转 —— 这样后面那句
    ``img[~ok] = 灰`` 只是"把无效格显式涂灰"，不再承担修正垃圾的责任。
    """
    f = np.asarray(field, np.float32)
    ok = np.asarray(valid(f), bool)
    safe = np.where(ok, f, np.float32(vmin))
    t = np.clip((safe - vmin) / max(top - vmin, 1e-9), 0, 1)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~ok] = (70, 70, 70)
    return img, {"available": True, "vmin": vmin, "vmax": top, "unit": unit,
                 "data_max": (round(float(np.max(np.where(ok, f, -np.inf))), 3)
                              if ok.any() else None),
                 "valid_cells": int(ok.sum())}
