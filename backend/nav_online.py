# -*- coding: utf-8 -*-
"""在线增量 navmesh：航位推算位姿 + 镜像双目关键帧 → NavSession → OSC 走位 / HMD 转向。

三个频率各管各的（别混成一个"1 Hz"）：
* 位姿 20 Hz：OSC 速度（ZOH，前值保持）× HMD 朝向，与离线 ``dr_odom_db.py`` 同一口径；
* 双目 ~10 Hz：SGBM 约 21 ms/帧，既做近距急停，也按**距离/转角/时长**触发关键帧（``KeyframePolicy``：快转推迟、原地补帧替换）；
* 地图更新：有新关键帧才栅格化 + 重规划（~130 ms），最快 2 Hz；
* 控制 10 Hz：跟随器吃最新位姿，每拍重发带 400 ms 自动过期的轴（线程死了人就停）。

位姿源 = 航位推算 + 回环（``nav_loop.LoopCloser``：ORB+PnP，只修平移，朝向来自 HMD）。
回环只在重访时出现；两次回环之间仍是纯航位推算（实测漂移约 2.8% 路程）。status 如实报 ``pose_source``。

坐标：地图系 = SteamVR 站立系经 C 换轴（x 前 y 左 z 上），原点 = 会话起点，追踪米；
对外（goto / status）一律导航系世界米 = 追踪米 × world_scale。
"""
from __future__ import annotations

import base64
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .nav_grid import FREE, OCC, UNK
from .nav_loop import LoopCloser, LoopConfig, extract_features
from .nav_memory import NavMemoryStore, SessionWriter, make_thumbnail
from .nav_xsession import XSessionConfig, XSessionTracker, align_into_auto
from .nav_mapping import (KeyframeGridMapper, MapperConfig, NavSession, disparity_range_px,
                          make_sgbm, stereo_disparity, stereo_points)

# 列 = base 的 x/y/z 轴在 SteamVR 站立系（x 右 y 上 z 后）中的坐标。
C_BASE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def hmd_to_base_rotation(r_hmd: np.ndarray) -> np.ndarray:
    return C_BASE.T @ np.asarray(r_hmd, float).reshape(3, 3) @ C_BASE


@dataclass
class OnlineNavConfig:
    # 世界米 = 追踪米 × world_scale（随 avatar 变）。唯一真值源：__post_init__ 把它推给 mapper / loop，
    # 两边各写一份时以这里为准（改 mapper.world_scale 不会生效，别在那边改）。
    world_scale: float = 0.755
    # 启动自检：实测双目基线与期望差超过 tol 就报 baseline_mismatch（不停机：深度按实测基线算，几何仍对，
    # 只是噪声变了；但跨会话记忆不能混用）。0.126 = 自编驱动；发行版驱动是 0.063。0 表示不检查。
    expected_baseline_m: float = 0.126
    baseline_tol_m: float = 0.01
    osc_lag_s: float = 0.18          # OSC 速度比画面晚到约 0.18 s（run5/run6 局部窗口回放扫描最优 0.175–0.2）
    # 非零速度的 ZOH 外推上限：VelocityX/Z 无心跳，静默只在短时间内等于"速度没变"
    # （实测需要外推非零速度的空档全部 ≤ 2.03 s）。静默超过 osc_zoh_max_s 后在 osc_zoh_fade_s
    # 内线性淡出到 0，避免丢掉"停下"那个 0 包时把最后一个速度永远积分下去（静止漂移 + 路程虚涨）。
    # 任一项为 0 = 不封顶（回到旧行为，只在明知道自己在赌时才这么配）。
    osc_zoh_max_s: float = 2.5
    osc_zoh_fade_s: float = 1.0
    pose_hz: float = 20.0
    # 每帧双目只看得到脚前 ~1.3 m（世界米）到 range_m 那一条带：3 Hz 时跑步一帧走 1.3 m，
    # 带与带之间就是空白。SGBM 720×405 约 26 ms，10 Hz 占一个核的 1/4，近距急停也跟着变快。
    stereo_period_s: float = 0.1
    # 双目采集宽度下限（像素）。走 mip 链：实际输出取仍不小于它的最深一级 ⇒ 720（默认）
    # = mip2 720×405 / fx 202.5（现役行为）、1440 = mip1 / fx 405、2880 = 全分辨率 / fx 810。
    # 深度尺度不随它变（fx 与视差同比例变）；SGBM 视差范围按 fx 自动缩放（见 make_sgbm 调用点）。
    capture_width: int = 720
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
    loop_closure: bool = True
    loop: LoopConfig = field(default_factory=LoopConfig)
    record_max_mb: float = 2048.0    # 录制上限（每关键帧约 0.4 MB，20 分钟约 500 MB）；超了停录、导航照常
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
        """由 ``config.NavmeshConfig`` 构造：顶层字段 + online/mapper/loop 三组覆盖项（键已在 config 校验）。"""
        mapper = replace(MapperConfig(), **dict(nav.mapper))
        loop = replace(LoopConfig(), **dict(nav.loop))
        return cls(world_scale=nav.world_scale, expected_baseline_m=nav.expected_baseline_m,
                   baseline_tol_m=nav.baseline_tol_m, loop_closure=nav.loop_closure,
                   mapper=mapper, loop=loop, **dict(nav.online))


def check_baseline(measured_m: float, cfg: OnlineNavConfig) -> dict[str, Any]:
    """实测基线对期望。``ok`` 为 None 表示不检查。"""
    exp = float(cfg.expected_baseline_m)
    if exp <= 0.0:
        return {"ok": None, "measured_m": round(float(measured_m), 4), "expected_m": None}
    ok = math.isfinite(measured_m) and abs(float(measured_m) - exp) <= cfg.baseline_tol_m
    return {"ok": bool(ok), "measured_m": round(float(measured_m), 4), "expected_m": exp,
            "tol_m": cfg.baseline_tol_m}


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
        self.xy, self.yaw, self.at, self.defer_since = np.asarray(xy[:2], float).copy(), yaw, t, None
        return kind


class SessionRecorder:
    """默认不开。把一次在线会话的原始输入落盘，离线能按同样顺序重放建图/回环（长时间场景复现）。

    目录内容：``meta.json``（配置、传感器）、``hmd.jsonl``（每个位姿周期的 HMD 旋转，base 系）、
    ``osc.jsonl``（新到的 OSC 速度样本）、``events.jsonl``（关键帧/trail/回环，按建图线程处理顺序）、
    ``kf/<id>.npz``（点云 + 航位推算位姿 + 入图位姿 + ORB 特征）、``final_poses.npz``（停止时的回环位姿）。

    感知线程只往内存缓冲里追加（``line``），所有磁盘写都在建图线程（``flush`` / ``keyframe``）里做。
    写失败或超出上限就停录并留痕，不影响导航。
    """

    def __init__(self, root: Path, max_bytes: float, meta: dict[str, Any]) -> None:
        self.dir = Path(root)
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
                    "stopped": self.stopped}

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
                 feat: Any) -> None:
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


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


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
                 zoh_max_s: float = 2.5, zoh_fade_s: float = 1.0) -> None:
        self.s = float(world_scale)
        self.lag = float(osc_lag_s)
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

    def update(self, t_now: float, r_ob: np.ndarray, osc: Sequence[dict[str, Any]]) -> np.ndarray:
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
            self._v = (float(vx), float(vz))
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

    def _advance(self, t: float, fwd: np.ndarray, right: np.ndarray) -> None:
        dt = t - float(self._t)
        if dt <= 0.0:
            return
        vx, vz = self._v
        speed = math.hypot(vx, vz)
        kept = dt if (speed <= 0.0 or self._v_at is None) else self._hold_dt(float(self._t), t)
        self.xy = self.xy + (vz * fwd + vx * right) * kept / self.s
        self.dist_m += speed * kept
        dropped = dt - kept
        if dropped > 0.0:
            # 静默太久，后面这段"位移"没有任何证据支撑：只记账，不积分。
            self.holdout_s += dropped
            self.holdout_m += speed * dropped
        self._t = t


class OpenVRSensors:
    """同一个 OpenVR Background 会话里取 HMD 姿态和镜像双目。

    D3D11 立即上下文不是线程安全的，所以本类的方法只能在感知线程里调用。
    ``openvr`` 惰性导入：没装 / SteamVR 没开时 ``open()`` 抛出，不影响 backend 其他部分。
    """

    def __init__(self, target_width: int = 720) -> None:
        self.target_width = int(target_width)
        self._vr: Any = None
        self._system: Any = None
        self._device = self._context = None
        self._eyes: list[Any] = []
        self._poses: Any = None
        self.fx = self.cx = self.cy = 0.0
        self.baseline_m = 0.0
        self.size = (0, 0)

    def open(self) -> dict[str, Any]:
        import openvr

        from .openvr_mirror import MirrorEye, create_device, projection_intrinsics

        self._vr = openvr
        self._system = openvr.init(openvr.VRApplication_Background)
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
            self._poses = (openvr.TrackedDevicePose_t * 1)()
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
        if self._system is not None:
            self._vr.shutdown()
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
    * ``send_move(forward, duration_ms)`` / ``send_turn(correction_deg)``（正数 = 左转 CCW）；
    * ``stop_motion()``；``drive_block_reason()`` → None 可驾驶，否则为不能动的原因。
    """

    def __init__(self, *, motion_history: Callable[[], Sequence[dict[str, Any]]],
                 send_move: Callable[[float, int], Any], send_turn: Callable[[float], Any],
                 stop_motion: Callable[[], Any], drive_block_reason: Callable[[], str | None],
                 sensors_factory: Callable[[], Any] | None = None,
                 cfg: OnlineNavConfig | None = None, clock: Callable[[], float] = time.monotonic,
                 record_root: Path | None = None, memory: NavMemoryStore | None = None,
                 world_identity: Callable[[], dict[str, Any]] | None = None) -> None:
        self.cfg = cfg or OnlineNavConfig()
        # 持久记忆（按世界分区）。world_identity() → {"world_key", ...}；key 为空时本次会话不写记忆。
        self.memory = memory
        self._world_identity = world_identity
        self._mem: SessionWriter | None = None
        self._mem_reason: str | None = "not_started"
        self._record_root = None if record_root is None else Path(record_root)
        self.recorder: SessionRecorder | None = None
        self._motion_history = motion_history
        self._send_move = send_move
        self._send_turn = send_turn
        self._stop_motion = stop_motion
        self._drive_block_reason = drive_block_reason
        # 默认工厂吃 ``cfg.capture_width``（720 = 现役 mip2；1440 / 2880 = 高分辨率录制实验）。
        # 测试注入的假传感器不受影响。
        self._sensors_factory = sensors_factory or (
            lambda: OpenVRSensors(target_width=self.cfg.capture_width))
        self._clock = clock
        self._state_lock = threading.Lock()
        self._nav_lock = threading.RLock()
        self._stop = threading.Event()
        self._map_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._reset()

    def _reset(self) -> None:
        c = self.cfg
        self.mapper = KeyframeGridMapper(c.mapper)
        self.session = NavSession(self.mapper, radius_m=c.radius_m, request_update=self._request_map_update)
        self.dr = DeadReckoner(c.world_scale, c.osc_lag_s,
                               zoh_max_s=c.osc_zoh_max_s, zoh_fade_s=c.osc_zoh_fade_s)
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
        self._pending: list[tuple[str, Any]] = []
        self._keyframes = 0
        self._kf_refreshed = 0
        self._kf_policy: KeyframePolicy | None = None
        self._map_updates = 0
        self._map_ms: float | None = None
        self._stereo_ms: float | None = None
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
        # 会话末自动采纳（P0.3b）：write_back 成功后后台线程跑 align_into_auto。
        self._align_out: dict | None = None
        self._align_thread: threading.Thread | None = None

    # ---- 生命周期 ----
    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self, record: bool = False) -> dict[str, Any]:
        """``record=True``：把这次会话的原始输入录到 ``record_root/<时间>/``（默认不录）。"""
        if self.running:
            return self.status()
        self._stop.clear()
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
                        {"config": asdict(self.cfg), "started_wall": time.time(), "started_clock": self._started_at})
                except (OSError, TypeError) as exc:
                    self._fail("recorder", exc)
        self._mem_rows = {}
        self._mem_flushed_at = -math.inf
        self._begin_memory()
        self._begin_xsession()
        self._threads = [threading.Thread(target=fn, name=f"navmesh-{name}", daemon=True)
                         for name, fn in (("perception", self._perception_loop),
                                          ("mapping", self._mapping_loop),
                                          ("control", self._control_loop))]
        for t in self._threads:
            t.start()
        return self.status()

    def stop(self, timeout_s: float = 2.0) -> dict[str, Any]:
        self._stop.set()
        self._map_event.set()
        for t in self._threads:
            t.join(timeout=timeout_s)
        self._threads = []
        self._halt()
        with self._nav_lock:
            self.session.cancel()
        if self.recorder is not None:
            self.recorder.close(None if self.loops is None else self.loops.poses())
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
            self._xs = XSessionTracker(self.cfg.xsession, wdir, mem.session_id, session_dir=mem.dir)
            self._xs_reason = ""
        except Exception as exc:                  # noqa: BLE001 - 检索坏了导航照常
            self._xs, self._xs_reason = None, f"init_error:{type(exc).__name__}"
            self._fail("xsession", exc)

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
        """后台采纳线程体：挑约束最多的旧会话对齐，结果进 status()["xsession_align"]。"""
        try:
            out = align_into_auto(world_dir, mem_sid, self.cfg.xsession)
        except Exception as exc:                  # noqa: BLE001 - 采纳只是锦上添花
            self._fail("xsession_align", exc)
            out = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        self._align_out = out

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
        if T is None or at is None or self._clock() - at > self.cfg.pose_stale_s:
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
        read_rgb = getattr(sensors, "read_stereo_rgb", None) if mem is not None else None
        if bl["ok"] is False:
            self._fail("baseline_mismatch", RuntimeError(
                f"measured {bl['measured_m']} m, expected {bl['expected_m']}±{bl['tol_m']} m "
                "(SteamVR 注册的 AnyaDance 驱动不是自编版？)"))
        rec = self.recorder
        if rec is not None:
            rec.line("events", {"kind": "sensors", "t": self._clock(), "info": info, "fx": sensors.fx,
                                "cx": sensors.cx, "cy": sensors.cy, "baseline_m": sensors.baseline_m})
        osc_seen = -math.inf
        matcher = None
        orb = None
        kf_id = -1
        policy = self._kf_policy = KeyframePolicy(c)
        yaw_hist: list[tuple[float, float]] = []    # 近 ~0.25 s 的 (时刻, 展开 yaw)，算转速
        next_stereo = next_trail = self._clock()
        period = 1.0 / c.pose_hz
        prev_yaw: float | None = None
        prev_tick = -math.inf
        try:
            while not self._stop.is_set():
                tick = self._clock()
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
                        rec.line("hmd", {"t": tick, "R": np.round(r_ob, 6)})
                    T = self.dr.update(tick, r_ob, hist)
                    yaw_now = math.atan2(T[1, 0], T[0, 0])
                    if prev_yaw is not None:
                        dyaw = math.degrees(_wrap(yaw_now - prev_yaw))
                        if abs(dyaw) > c.yaw_jump_dps * max(tick - prev_tick, period) + c.yaw_jump_min_deg:
                            with self._state_lock:
                                self._yaw_jumps = (self._yaw_jumps + [{
                                    "t_s": round(tick - (self._started_at or tick), 2),
                                    "delta_deg": round(dyaw, 1), "keyframe": kf_id,
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
                    if kf_id >= 0 and tick >= next_trail:
                        next_trail = tick + c.trail_period_s
                        with self._state_lock:
                            self._pending.append(("trail", (kf_id, T.copy(), self.dr.dist_m, tick)))
                    if tick >= next_stereo:
                        next_stereo = tick + c.stereo_period_s
                        t0 = time.perf_counter()
                        left_rgb = None
                        if read_rgb is not None:
                            rgb = read_rgb()
                            pair = None if rgb is None else (cv2.cvtColor(rgb[0], cv2.COLOR_RGB2GRAY),
                                                             cv2.cvtColor(rgb[1], cv2.COLOR_RGB2GRAY))
                            left_rgb = None if rgb is None else rgb[0]
                        else:
                            pair = sensors.read_stereo()
                        if pair is not None:
                            if matcher is None:
                                # 视差范围随 fx 缩放：换采集宽度（fx 202.5→405→810）时不缩
                                # 会把近场裁掉；默认 720 下仍是 64，行为逐位不变。
                                matcher = make_sgbm(disparity_range_px(sensors.fx))
                            disp = stereo_disparity(pair[0], pair[1], matcher)
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
                            feat = None
                            if new_kf and (self.loops is not None or mem is not None):
                                # 同一张视差图给 ORB 点深度；只在关键帧上做，约 6 ms。
                                if orb is None:
                                    orb = cv2.ORB_create(nfeatures=c.loop.orb_features)
                                feat = extract_features(pair[0], disp, fx=sensors.fx, cx=sensors.cx,
                                                        cy=sensors.cy, baseline_m=sensors.baseline_m,
                                                        orb=orb, max_depth_m=c.loop.max_depth_m)
                            with self._state_lock:
                                self._near, self._near_pts = near, n
                                self._stereo_at = tick
                                self._stereo_ms = (time.perf_counter() - t0) * 1000.0
                                if new_kf:
                                    kf_id += 1
                                    thumb = None
                                    if mem is not None:
                                        src = left_rgb if left_rgb is not None else pair[0]
                                        thumb = make_thumbnail(src, mem.cfg.thumb_width)
                                    self._pending.append(("kf", (kf_id, pts, T.copy(), self.dr.dist_m, feat, tick,
                                                                 kf_kind == "refresh", thumb)))
                            if new_kf:
                                self._map_event.set()
                self._stop.wait(max(0.0, period - (self._clock() - tick)))
        except Exception as exc:  # noqa: BLE001 - 线程死了控制线程会因位姿过期而停车
            self._fail("perception", exc)
        finally:
            try:
                sensors.close()
            except Exception as exc:  # noqa: BLE001
                self._fail("sensors_close", exc)

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
                for kind, item in pending:
                    if kind == "kf":
                        k, pts, T, dist, feat, t_kf, refresh, thumb = item
                        dirty = True
                        found = []
                        if self.loops is None:
                            self.mapper.add_keyframe(k, pts, T, osc_dist_m=dist)
                        else:
                            found = self.loops.add_keyframe(k, T, dist, feat)
                            # 关键帧按修正后的位姿入图；回环后所有关键帧整体换位姿。
                            self.mapper.add_keyframe(k, pts, self.loops.pose(k), osc_dist_m=dist)
                        if refresh:
                            # 原地补帧：上一帧的点云让给这帧；它的位姿、轨迹、回环节点都留着。
                            self.mapper.drop_points(k - 1)
                            self._kf_refreshed += 1
                        mem = self._mem
                        if mem is not None:
                            self._mem_rows[k] = (T.copy(), float(dist), t_kf - (self._started_at or t_kf))
                            mem.keyframe(k, feat, thumb, refresh)
                        if self._xs is not None:
                            try:
                                # 跨会话查询+验证；T 的旋转来自 HMD（yaw 门的外部真值）。
                                self._xs.on_keyframe(k, feat, float(dist), T[:3, :3], refresh)
                            except Exception as exc:  # noqa: BLE001 - 检索出错禁用自身，不杀建图线程
                                self._fail("xsession", exc)
                                self._xs = None
                        if rec is not None:
                            rec.keyframe(k, pts, T, T if self.loops is None else self.loops.pose(k), dist, feat)
                            rec.line("events", {"kind": "kf", "k": k, "t": t_kf, "dist_m": dist, "loops": found,
                                                "refresh": refresh})
                        if self.loops is not None:
                            if found:
                                moved = self.mapper.update_poses(self.loops.poses())
                                self._last_loop = {"keyframe": k, "loops": found,
                                                   "map_shift_m": round(moved * c.world_scale, 3)}
                            loop_status = self.loops.status()
                            with self._state_lock:
                                self._offset = self.loops.offset()
                                self._loop_status = loop_status
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
                if mem is not None and self._clock() - self._mem_flushed_at >= mem.cfg.pose_flush_s:
                    # 位姿整表写：回环会改之前所有帧的位姿，逐帧写的会过期。
                    table = self._mem_table()
                    if table is not None:
                        mem.poses(table)
                        self._mem_flushed_at = self._clock()
                if dirty and len(self.mapper):
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
        while not self._stop.is_set():
            tick = self._clock()
            try:
                self._control_tick(tick)
            except Exception as exc:  # noqa: BLE001 - 控制出错先停车
                self._fail("control", exc)
                self._halt()
            self._stop.wait(max(0.0, period - (self._clock() - tick)))
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
        if block is None and (stereo_at is None or tick - stereo_at > c.stereo_stale_s):
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
            "keyframes_deferred": 0 if self._kf_policy is None else self._kf_policy.deferred,
            "map_updates": self._map_updates,
            "map_update_ms": None if self._map_ms is None else round(self._map_ms, 1),
            "stereo_ms": None if stereo_ms is None else round(stereo_ms, 1),
            "stereo_age_s": None if stereo_at is None else round(now - stereo_at, 2),
            "camera_height_m": round(cam_h, 3),
            "near_obstacle": {"stop": near, "points": near_pts, "ahead_m": self.cfg.stop_ahead_m},
            "drive_block": self._drive_block,
            "last_step": dict(self._last_step),
            "last_plan": last,
            "sensors": info,
            "baseline_check": baseline_check,
            "world_scale": self.cfg.world_scale,
            "errors": errors,
            "uptime_s": None if self._started_at is None else round(now - self._started_at, 1),
            "recording": None if self.recorder is None else self.recorder.status(),
            "memory": self._mem.status() if self._mem is not None else {"active": False, "reason": self._mem_reason},
            "xsession": self._xs.status() if self._xs is not None else {"active": False, "reason": self._xs_reason},
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
                     "bands": [{"name": n, "rgb": list(c)}
                               for n, c in zip(("below", "lo", "mid", "hi"), _BAND_COLORS)],
                     "edges_m": [float(v) for v in edges]}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---- 栅格底图图层 ----
# 为什么不加"地面高度"层：在线 mapper 每格只留 7 个数（4 个分带计数 + 3 个矩），
# **没有逐格的地面高度**。要它就得给热路径加累加器（`_kf_base` 每帧多两次 bincount），
# 而本轮改动先走"零热路径"路线。⇒ 现有四层里，深度信息最直接的是 `surface`
# （头顶 obst_top_m 之上那张面的绝对高度）—— 那是天花板/楼板的高度，不是脚下。
_GRID_LAYERS = ("tristate", "surface", "bands", "clearance")

# 四条高度带（below=地面以下 / lo=2~hi_band / mid / hi）的固定配色，顺序与
# ``MapperConfig`` 里 hb 的赋值顺序一致（nav_mapping.py:568-574）。
# below 用**冷色**：它代表"地面以下有东西"，是坑的候选，必须一眼能挑出来。
_BAND_COLORS = ((60, 60, 255), (40, 190, 255), (60, 210, 90), (40, 70, 240))


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
