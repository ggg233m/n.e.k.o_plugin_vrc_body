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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .nav_grid import FREE, OCC
from .nav_loop import LoopCloser, LoopConfig, extract_features
from .nav_mapping import KeyframeGridMapper, MapperConfig, NavSession, make_sgbm, stereo_disparity, stereo_points

# 列 = base 的 x/y/z 轴在 SteamVR 站立系（x 右 y 上 z 后）中的坐标。
C_BASE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def hmd_to_base_rotation(r_hmd: np.ndarray) -> np.ndarray:
    return C_BASE.T @ np.asarray(r_hmd, float).reshape(3, 3) @ C_BASE


@dataclass
class OnlineNavConfig:
    osc_lag_s: float = 0.18          # OSC 速度比画面晚到约 0.18 s（run5/run6 局部窗口回放扫描最优 0.175–0.2）
    pose_hz: float = 20.0
    # 每帧双目只看得到脚前 ~1.3 m（世界米）到 range_m 那一条带：3 Hz 时跑步一帧走 1.3 m，
    # 带与带之间就是空白。SGBM 720×405 约 26 ms，10 Hz 占一个核的 1/4，近距急停也跟着变快。
    stereo_period_s: float = 0.1
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
    loop_closure: bool = True
    loop: LoopConfig = field(default_factory=LoopConfig)
    record_max_mb: float = 2048.0    # 录制上限（每关键帧约 0.4 MB，20 分钟约 500 MB）；超了停录、导航照常


class KeyframePolicy:
    """双目帧要不要成为关键帧。``decide`` 返回 None（不取）、``"new"``（追加）或
    ``"refresh"``（原地补帧：点云替换上一关键帧的，位姿/回环节点照留）。在线与离线回放共用。"""

    def __init__(self, cfg: OnlineNavConfig) -> None:
        self.cfg = cfg
        self.scale = cfg.mapper.world_scale
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
    """

    def __init__(self, world_scale: float, osc_lag_s: float = 0.18) -> None:
        self.s = float(world_scale)
        self.lag = float(osc_lag_s)
        self.xy = np.zeros(2)
        self.dist_m = 0.0            # OSC 路程，世界米
        self._v = (0.0, 0.0)         # (vx 右, vz 前)，世界米/秒
        self._t: float | None = None
        self._last_ts = -math.inf
        self.samples = 0

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
            self._advance(min(max(ts, self._t), target), fwd, right)
            self._v = (float(vx), float(vz))
            self._last_ts = ts
            self.samples += 1
        self._advance(target, fwd, right)
        T = np.eye(4)
        T[:3, :3] = r_ob
        vx, vz = self._v
        T[:2, 3] = self.xy + (vz * fwd + vx * right) * self.lag / self.s
        return T

    def _advance(self, t: float, fwd: np.ndarray, right: np.ndarray) -> None:
        dt = t - float(self._t)
        if dt <= 0.0:
            return
        vx, vz = self._v
        self.xy = self.xy + (vz * fwd + vx * right) * dt / self.s
        self.dist_m += math.hypot(vx, vz) * dt
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

    def read_stereo(self) -> tuple[np.ndarray, np.ndarray] | None:
        from .openvr_mirror import read_stereo

        left, right = read_stereo(self._eyes[0], self._eyes[1])
        if left is None or right is None:
            return None
        return cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

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
                 sensors_factory: Callable[[], Any] = OpenVRSensors,
                 cfg: OnlineNavConfig | None = None, clock: Callable[[], float] = time.monotonic,
                 record_root: Path | None = None) -> None:
        self.cfg = cfg or OnlineNavConfig()
        self._record_root = None if record_root is None else Path(record_root)
        self.recorder: SessionRecorder | None = None
        self._motion_history = motion_history
        self._send_move = send_move
        self._send_turn = send_turn
        self._stop_motion = stop_motion
        self._drive_block_reason = drive_block_reason
        self._sensors_factory = sensors_factory
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
        self.dr = DeadReckoner(c.mapper.world_scale, c.osc_lag_s)
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
        self._errors: list[str] = []
        # 一个位姿周期里 HMD yaw 变化超过调度器转速上限能解释的量 = 有人在我们之外转了 play space
        # （SteamVR/VRChat 重置朝向、传送）。航位推算会从那一刻起朝错方向积分，必须留痕。
        self._yaw_jumps: list[dict[str, Any]] = []
        self._moving = False
        self._drive_block: str | None = "not_started"
        self._last_step: dict[str, Any] = {}
        self._started_at: float | None = None

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
        return self.status()

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
        s = self.cfg.mapper.world_scale
        x, y = T[0, 3] + off[0], T[1, 3] + off[1]
        return {"state": "localized", "xy": (float(x * s), float(y * s)),
                "theta": math.atan2(T[1, 0], T[0, 0]), "sigma_m": self.cfg.dr_sigma_m}

    # ---- 感知线程 ----
    def _perception_loop(self) -> None:
        c = self.cfg
        s = c.mapper.world_scale
        try:
            sensors = self._sensors_factory()
            info = sensors.open()
        except Exception as exc:  # noqa: BLE001 - SteamVR 没开 / openvr 没装：如实报，不崩 backend
            self._fail("sensors_open", exc)
            return
        with self._state_lock:
            self._sensor_info = info
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
                                    "odometry_m": round(self.dr.dist_m, 2)}])[-10:]
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
                        pair = sensors.read_stereo()
                        if pair is not None:
                            if matcher is None:
                                matcher = make_sgbm()
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
                            if new_kf and self.loops is not None:
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
                                    self._pending.append(("kf", (kf_id, pts, T.copy(), self.dr.dist_m, feat, tick,
                                                                 kf_kind == "refresh")))
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
                        k, pts, T, dist, feat, t_kf, refresh = item
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
                        if rec is not None:
                            rec.keyframe(k, pts, T, T if self.loops is None else self.loops.pose(k), dist, feat)
                            rec.line("events", {"kind": "kf", "k": k, "t": t_kf, "dist_m": dist, "loops": found,
                                                "refresh": refresh})
                        if self.loops is not None:
                            if found:
                                moved = self.mapper.update_poses(self.loops.poses())
                                self._last_loop = {"keyframe": k, "loops": found,
                                                   "map_shift_m": round(moved * c.mapper.world_scale, 3)}
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
            "errors": errors,
            "uptime_s": None if self._started_at is None else round(now - self._started_at, 1),
            "recording": None if self.recorder is None else self.recorder.status(),
        }

    def grid_png(self) -> str | None:
        view = self.grid_view()
        return None if view is None else view["png_base64"]

    def grid_view(self) -> dict[str, Any] | None:
        """当前栅格的 base64 PNG（一格一像素，第 0 行 = 北/+y 最大）+ 像素↔导航系世界米的换算。
        白 = 可走中心区，浅灰 = 观测 free，深灰 = unknown，黑 = 障碍；
        蓝线 = 规划路径，红点 = 当前位姿，绿点 = 目标。"""
        with self._nav_lock:
            ng = self.session.ng
            if ng is None:
                return None
            g = ng.grid
            img = np.full(g.shape + (3,), 90, np.uint8)
            img[g == FREE] = (170, 170, 170)
            img[g == OCC] = (0, 0, 0)
            img[ng.center] = (255, 255, 255)
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
                # 列 c、行 r 的格中心：x = (ox + (c+0.5)·res)·s，y = (oy + (rows−1−r+0.5)·res)·s
                "origin_xy_track_m": [float(m.origin_xy_m[0]), float(m.origin_xy_m[1])],
                "resolution_track_m": float(m.resolution_m), "world_scale": float(m.world_scale),
                "frame": "nav_map_xy_world_m"}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
