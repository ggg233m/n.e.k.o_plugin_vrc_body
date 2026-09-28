"""v2 路线驱动：HMD 指南针 + 横移绕行 + 动作日志。

设计依据：报告 §22.19、录制手册 §7（09-20 决议）。

通道纪律（硬约束）：
  - **转向只走虚拟 HMD yaw**（零失真：推多少转多少）；同一 run 禁用 OSC
    ``LookHorizontal``——两通道叠加且其一失真，混合即污染
    （与「AngularY 不参与朝向积分」同一条纪律）。
  - 平移走 OSC ``/input/Vertical``；路程由离线 ``hypot(VelocityX, VelocityZ)``
    时间积分得出（本模块只负责把速度如实记进行）。
  - 撞墙 → **横移绕行**（``/input/Horizontal``，不改朝向，故不依赖转角精度）；
    连续 ``max_detours`` 次绕行仍撞才判段结束。
  - 动作日志经 ``driver_log.ActionLogRecorder`` 落盘（格式单一来源）；
    **未锚定 ⇒ 不写、如实降级**，绝不伪造时间基准。

HMD 推帧协议（AnyaDance docs/protocol.md）：UDP 127.0.0.1:39570，UTF-8 JSON，
``{"version":1,"devices":{"hmd":{...}}}``；第三方发送端可以只发部分设备，
但至少要有一个有效设备条目——这里只推 HMD。
⚠️ 推帧前必须关闭 AnyaDance.exe 伴随 UI，否则两个发送端争抢 HMD（驱动逐包
接受后到者生效，HMD 会高频抖动）。

坐标约定：position 米；rotation_xyzw 四元数 XYZW 顺序。绕 +Y 的 yaw 四元数为
``[0, sin(θ/2), 0, cos(θ/2)]``。**正 yaw 对应 VRChat 里的实际转向（左/右）未
预先标定**——由路线开头的 90° 指南针验证实验用视频实证，本模块只如实记录
指令值，绝不臆断方向。
"""

from __future__ import annotations

import json
import math
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

# ---- driver_log 引入：复用 tests/_bootstrap 的包别名模式 -------------------
# 录制器必须零 SDK 依赖运行；neko_anyadance_body 包名在这里只是给
# driver_log 的相对导入（from .config import ...）一个可解析的宿主。
_PKG_ROOT = Path(__file__).resolve().parents[2]
if "neko_anyadance_body" not in sys.modules:
    import types

    _pkg = types.ModuleType("neko_anyadance_body")
    _pkg.__path__ = [str(_PKG_ROOT)]  # type: ignore[attr-defined]
    sys.modules["neko_anyadance_body"] = _pkg

from neko_anyadance_body.driver_log import (  # noqa: E402
    ActionLogRecorder,
    VideoTimebase,
    load_action_timeline,
)

HMD_HOST = "127.0.0.1"
HMD_PORT = 39570
HMD_STREAM_HZ_DEFAULT = 60.0
HMD_HEIGHT_DEFAULT = 1.5  # 站姿头部高度（米），与 AnyaDance 示例一致

ROUTE_OPS = ("settle", "hold", "forward", "turn", "strafe", "beacon",
             "compass_check")
_OPS_NEED_SEC = ("settle", "hold", "forward", "strafe")


def yaw_quat_xyzw(yaw_rad: float) -> list[float]:
    """绕 +Y 轴的 yaw 四元数（XYZW 顺序）。"""
    return [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)]


class HmdPusher:
    """后台线程按固定速率推虚拟 HMD 位姿（fire-and-forget）。

    只推 HMD 一个设备条目；其余五个设备保持驱动内当前状态（驱动契约：
    设备一旦有效就持续有效）。退出时**保持最后姿态**、不做归零——归零是
    一个我们并没有被要求的"动作"，按不伪造原则不做。
    """

    def __init__(self, *, hz: float = HMD_STREAM_HZ_DEFAULT,
                 height: float = HMD_HEIGHT_DEFAULT,
                 host: str = HMD_HOST, port: int = HMD_PORT,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._hz = max(1.0, float(hz))
        self._height = float(height)
        self._host, self._port = host, port
        self._clock = clock
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._yaw_rad = 0.0
        self._stop = threading.Event()
        self.sent = 0
        self.errors = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def set_yaw_deg(self, yaw_deg: float) -> None:
        with self._lock:
            self._yaw_rad = math.radians(float(yaw_deg))

    @property
    def yaw_deg(self) -> float:
        with self._lock:
            return math.degrees(self._yaw_rad)

    def _run(self) -> None:
        period = 1.0 / self._hz
        next_t = self._clock()
        while not self._stop.is_set():
            now = self._clock()
            if now < next_t:
                time.sleep(min(next_t - now, 0.002))
                continue
            next_t = now + period
            with self._lock:
                frame = {
                    "version": 1,
                    "devices": {
                        "hmd": {
                            "valid": True,
                            "connected": True,
                            "pose": {
                                "position": [0.0, self._height, 0.0],
                                "rotation_xyzw": yaw_quat_xyzw(self._yaw_rad),
                            },
                        },
                    },
                }
            try:
                self._sock.sendto(
                    json.dumps(frame, separators=(",", ":")).encode("utf-8"),
                    (self._host, self._port))
                self.sent += 1
            except OSError:
                self.errors += 1
                time.sleep(0.05)  # 别在错误路径上空转烧 CPU

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self._sock.close()
        except OSError:
            pass


def load_route(path: str | Path) -> dict[str, Any]:
    """解析并**前置**校验路线文件——坏路线必须在开录前失败，不能录到一半炸。"""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise ValueError("route file must be a JSON object with version == 1")
    legs = doc.get("legs")
    if not isinstance(legs, list) or not legs:
        raise ValueError("route file must contain a non-empty 'legs' array")
    for i, spec in enumerate(legs):
        if not isinstance(spec, dict):
            raise ValueError(f"legs[{i}] must be an object")
        op = spec.get("op")
        if op not in ROUTE_OPS:
            raise ValueError(f"legs[{i}]: unknown op {op!r} (known: {ROUTE_OPS})")
        if op in _OPS_NEED_SEC and not isinstance(spec.get("sec"), (int, float)):
            raise ValueError(f"legs[{i}]: op {op!r} requires numeric 'sec'")
        if op == "turn" and not isinstance(spec.get("deg"), (int, float)):
            raise ValueError(f"legs[{i}]: op 'turn' requires numeric 'deg'")
    return doc


class RouteDriver:
    """执行 v2 路线：每个控制拍 = OSC 轴 + HMD yaw + 一行动作日志。

    ``hmd``/``recorder`` 允许为 None（降级运行）：
    - recorder=None（未锚定）：不写动作日志，其余照跑；
    - hmd=None（--no-hmd）：不推 HMD，turn/beacon 只记账不转向（仅供干跑）。
    """

    def __init__(self, *, axes, cache, events: list[dict], recorder=None,
                 hmd: HmdPusher | None = None, vertical: float = 0.30,
                 hz: float = 20.0, stall_speed: float = 0.15,
                 stall_ticks: int = 6, strafe_value: float = 0.5,
                 strafe_s: float = 1.0, max_detours: int = 2,
                 hmd_turn_rate: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.axes = axes
        self.cache = cache
        self.events = events
        self.recorder = recorder
        self.hmd = hmd
        self.vertical = float(vertical)
        self.hz = max(1.0, float(hz))
        self.stall_speed = float(stall_speed)
        self.stall_ticks = int(stall_ticks)
        self.strafe_value = float(strafe_value)
        self.strafe_s = float(strafe_s)
        self.max_detours = int(max_detours)
        self.hmd_turn_rate = float(hmd_turn_rate)
        self._clock = clock
        self._sleep = sleep
        self._seq = 0
        self._prev_yaw_deg = 0.0
        self._dyaw_deg = 0.0
        self._episode = ""

    # ------------------------------------------------------------------ 基础

    def _step(self, forward: float = 0.0, horizontal: float = 0.0,
              yaw_deg: float | None = None) -> None:
        """一个控制拍：发轴 + 推 yaw + 记一行动作日志 + 等下一拍。"""
        now = self._clock()
        if yaw_deg is not None:
            self._dyaw_deg = float(yaw_deg) - self._prev_yaw_deg
            self._prev_yaw_deg = float(yaw_deg)
            if self.hmd is not None:
                self.hmd.set_yaw_deg(self._prev_yaw_deg)
        else:
            self._dyaw_deg = 0.0
        before = self.axes.send_errors
        self.axes.set(forward, 0.0, horizontal)   # look 恒 0：v2 禁用 LookHorizontal
        self.axes.tick()
        sent = self.axes.send_errors == before
        if self.recorder is not None:
            vx, vz = self.cache.velocity_last_known()
            action = {
                "goal_id": "route_v2",
                "episode_id": self._episode,
                "input_command": {"forward": round(forward, 4),
                                  "horizontal": round(horizontal, 4)},
                "actual_send_result": "sent" if sent else "send_failed",
                "driver_ack": "none",
                "osc_velocity": {"vx": 0.0 if vx is None else round(vx, 4),
                                 "vy": 0.0,
                                 "vz": 0.0 if vz is None else round(vz, 4)},
                "turn_intent": {"yaw_delta": round(self._dyaw_deg, 4),
                                "hmd_yaw_deg": round(self._prev_yaw_deg, 4),
                                "source": "hmd_pose"},
            }
            self._seq += 1
            self.recorder.record(action=action, sequence=self._seq,
                                 monotonic_time=now)
        self._sleep(1.0 / self.hz)

    def _ramp_yaw(self, target_deg: float, rate_dps: float,
                  op_index: int, event_name: str) -> tuple[float, float]:
        """把 yaw 以 rate_dps 匀速推到 target；返回 (t_start, t_end)。"""
        t0 = self._clock()
        cur = self._prev_yaw_deg
        delta = float(target_deg) - cur
        if abs(delta) < 1e-9:
            return t0, t0
        steps = max(1, int(math.ceil(abs(delta) / max(1.0, rate_dps)
                                       * self.hz)))
        step_deg = delta / steps
        for _ in range(steps):
            self._step(forward=0.0, horizontal=0.0,
                       yaw_deg=self._prev_yaw_deg + step_deg)
        t1 = self._clock()
        self.events.append({"event": event_name, "op_index": op_index,
                            "yaw_cmd_deg": round(delta, 2),
                            "rate_dps": rate_dps,
                            "t_start": round(t0, 6), "t_end": round(t1, 6),
                            "yaw_after_deg": round(self._prev_yaw_deg, 2)})
        return t0, t1

    def _hold(self, sec: float, *, forward: float = 0.0,
              horizontal: float = 0.0, note_vel: bool = False) -> None:
        t0 = self._clock()
        while self._clock() - t0 < sec:
            self._step(forward=forward, horizontal=horizontal)
        _ = note_vel

    # ------------------------------------------------------------------ 各 op

    def _op_forward(self, spec: dict, index: int) -> None:
        total = float(spec["sec"])
        remaining = total
        detours = 0
        stalls = 0
        speeds: list[float] = []
        low = 0
        t0 = self._clock()
        abandoned = False
        while remaining > (1.0 / self.hz) / 2.0:
            t_seg = self._clock()
            stalled = False
            while self._clock() - t_seg < remaining:
                sp = self.cache.horizontal_speed()
                if sp is not None:
                    speeds.append(sp)
                    low = low + 1 if sp < self.stall_speed else 0
                    if low >= self.stall_ticks:
                        stalled = True
                        break
                self._step(forward=self.vertical, horizontal=0.0,
                           yaw_deg=self._prev_yaw_deg)
            elapsed = self._clock() - t_seg
            remaining -= elapsed
            if not stalled:
                break
            stalls += 1
            detours += 1
            self.events.append({"event": "stall", "op_index": index,
                                "detour_index": detours,
                                "at_elapsed_s": round(total - remaining, 2)})
            if detours > self.max_detours:
                self.events.append({"event": "leg_abandoned",
                                    "op_index": index,
                                    "detours": detours})
                abandoned = True
                break
            # ---- 横移绕行：不改朝向 ----
            t_det = self._clock()
            while self._clock() - t_det < self.strafe_s:
                self._step(forward=0.0, horizontal=self.strafe_value,
                           yaw_deg=self._prev_yaw_deg)
            self._hold(0.3)  # 让速度方向恢复再继续前进
            self.events.append({"event": "detour", "op_index": index,
                                "n": detours,
                                "strafe_value": self.strafe_value,
                                "strafe_s": round(self._clock() - t_det, 2)})
            low = 0
        dur = self._clock() - t0
        avg = (sum(speeds) / len(speeds)) if speeds else None
        self.events.append({"event": "forward", "op_index": index,
                            "duration_s": round(dur, 2), "stalls": stalls,
                            "detours": detours, "abandoned": abandoned,
                            "avg_speed": None if avg is None else round(avg, 3),
                            "samples": len(speeds)})

    def _op_turn(self, spec: dict, index: int) -> None:
        deg = float(spec["deg"])
        rate = float(spec.get("rate", self.hmd_turn_rate))
        t0, t1 = self._ramp_yaw(self._prev_yaw_deg + deg, rate, index,
                                "turn_hmd")
        self.events.append({"event": "turn", "op_index": index,
                            "duration_s": round(t1 - t0, 2),
                            "deg": deg, "channel": "hmd"})

    def _op_beacon(self, index: int) -> None:
        """扫视信标：+45 → -45 → 0，快速。视频侧是显著视觉事件。"""
        t0 = self._clock()
        for target in (45.0, -45.0, 0.0):
            self._ramp_yaw(target, 180.0, index, "beacon_sweep")
        t1 = self._clock()
        self.events.append({"event": "beacon", "op_index": index,
                            "t_start": round(t0, 6), "t_end": round(t1, 6),
                            "peak_deg": 45.0})

    def _op_compass_check(self, spec: dict, index: int) -> None:
        """90° 指南针验证：推 yaw +deg → 前进 fwd_sec 秒。

        验证在离线侧做（视频视角恰转 deg、VelocityZ>0、VelocityX≈0）；
        这里只负责精确记录指令窗口。
        """
        deg = float(spec.get("deg", 90.0))
        fwd_sec = float(spec.get("fwd_sec", 2.0))
        t_turn0, t_turn1 = self._ramp_yaw(self._prev_yaw_deg + deg,
                                          self.hmd_turn_rate, index,
                                          "compass_turn")
        t_fwd0 = self._clock()
        self._hold(fwd_sec, forward=self.vertical)
        t_fwd1 = self._clock()
        self.events.append({"event": "compass_check", "op_index": index,
                            "yaw_cmd_deg": deg,
                            "turn_t": [round(t_turn0, 6), round(t_turn1, 6)],
                            "fwd_t": [round(t_fwd0, 6), round(t_fwd1, 6)],
                            "fwd_sec": fwd_sec})

    # ------------------------------------------------------------------ 主入口

    def run(self, route: dict[str, Any]) -> None:
        for index, spec in enumerate(route["legs"]):
            op = spec["op"]
            self._episode = f"{op}:{index}"
            if op in ("settle", "hold"):
                t0 = self._clock()
                self._hold(float(spec["sec"]))
                self.events.append({"event": op, "op_index": index,
                                    "duration_s": round(self._clock() - t0, 2)})
            elif op == "forward":
                self._op_forward(spec, index)
            elif op == "turn":
                self._op_turn(spec, index)
            elif op == "strafe":
                t0 = self._clock()
                self._hold(float(spec["sec"]),
                           horizontal=float(spec.get("value",
                                                     self.strafe_value)))
                self.events.append({"event": "strafe", "op_index": index,
                                    "duration_s": round(self._clock() - t0, 2),
                                    "value": float(spec.get(
                                        "value", self.strafe_value))})
            elif op == "beacon":
                self._op_beacon(index)
            elif op == "compass_check":
                self._op_compass_check(spec, index)


def readback_summary(path: str | Path) -> dict[str, Any]:
    """落盘后自检：用**真正的加载器**读回，证明格式没漂移。"""
    try:
        timebase, rows = load_action_timeline(path)
        return {
            "readback_rows": len(rows),
            "readback_timebase_ok": bool(timebase is not None
                                         and timebase.anchor is not None),
            "readback_fps": None if timebase is None else timebase.fps,
        }
    except Exception as exc:  # 读回失败必须暴露，不能静默
        return {"readback_rows": -1, "readback_timebase_ok": False,
                "readback_error": repr(exc)[:200]}


# ---------------------------------------------------------------------------
# v2.1 手动 HMD 模式：被动捕获驱动命令回执（09-20 22:39 用户定调）
#
# 不推帧（推帧会与手动发送端争抢 HMD），改为**监听**驱动多播
# 239.255.39.71:39571：驱动对每条"发生变化"的命令发一份 command_processed
# 报告，其中 command.payload 是它实际接受的完整位姿帧。于是手动操作
# （AnyaDance.exe 鼠标操控）产生的 HMD 轨迹，可以与 OSC 共用同一个
# time.monotonic 时钟被如实录下来。
# 注意：报告是**变化驱动**的——静止/匀速时没有新事件，分析侧按零阶保持处理。
# 排序/去重以 sequence 为准（驱动契约），到达时间只回答"何时"。

DRIVER_LOG_GROUP = "239.255.39.71"
DRIVER_LOG_PORT = 39571
DRIVER_LOG_IFACE = "127.0.0.1"


def parse_command_event(data: bytes) -> dict[str, Any] | None:
    """一条驱动多播数据报 → 捕获行；不相关/坏行/被拒命令返回 None。"""
    try:
        event = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(event, dict) or event.get("version") != 1:
        return None
    if event.get("event") != "command_processed":
        return None
    command = event.get("command") or {}
    if not command.get("accepted"):
        return None  # 被拒数据报没有可信 payload（驱动契约原文）
    payload_raw = command.get("payload")
    if not isinstance(payload_raw, str):
        return None
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None
    devices = payload.get("devices") or {}
    hmd_pose = ((devices.get("hmd") or {}).get("pose") or {})
    if not hmd_pose:
        return None
    return {
        "seq": event.get("sequence"),
        "driver_ts_ms": event.get("timestamp_ms"),
        "devices_seen": command.get("devices"),
        "hmd": {"position": hmd_pose.get("position"),
                "rotation_xyzw": hmd_pose.get("rotation_xyzw")},
        "all_device_poses": {name: (dev or {}).get("pose")
                             for name, dev in devices.items()},
    }


class HmdCaptureListener:
    """监听驱动命令回执多播，把实际接受的 HMD/设备位姿按到达时刻落 JSONL。

    每行：{"t": <time.monotonic 到达时刻>, "seq", "driver_ts_ms",
           "hmd": {position, rotation_xyzw}, "all_device_poses", ...}
    去重按 sequence（重复数据报丢弃）；乱序不在此纠正——落盘带 seq，
    离线分析排序。socket 四步顺序遵循驱动契约：REUSEADDR→bind 0.0.0.0→
    回环接口加组，任何一步颠倒都会静默收不到。
    """

    def __init__(self, path: str | Path, *,
                 group: str = DRIVER_LOG_GROUP,
                 port: int = DRIVER_LOG_PORT,
                 iface: str = DRIVER_LOG_IFACE,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.path = Path(path)
        self._group, self._port, self._iface = group, int(port), iface
        self._clock = clock
        self._stop = threading.Event()
        self._handle = None
        self._thread: threading.Thread | None = None
        self.received = 0      # 收到的全部数据报（含无关事件）
        self.accepted = 0      # 其中 accepted 的 command_processed
        self.hmd_frames = 0    # 实际写盘的捕获行
        self.duplicates = 0    # 按 sequence 判定的重复数据报
        self.last_seq: int | None = None
        self.last_error: str | None = None

    # -- 事件入口（测试注入点：不经 socket 直接种行）--------------------
    def _ingest(self, parsed: dict[str, Any] | None) -> bool:
        if parsed is None:
            return False
        self.accepted += 1
        seq = parsed.get("seq")
        if seq is not None and seq == self.last_seq:
            self.duplicates += 1
            return False  # 重复数据报 = 重复事件，丢弃（驱动契约）
        if seq is not None:
            self.last_seq = seq
        if self._handle is None:
            return False
        row = {"t": round(self._clock(), 6), **parsed}
        try:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._handle.flush()
        except OSError as exc:
            self.last_error = f"hmd capture write failed: {exc}"
            return False
        self.hmd_frames += 1
        return True

    def start(self) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 先于 bind
            sock.bind(("", self._port))                                  # 绑 0.0.0.0
            sock.setsockopt(                                            # 回环接口加组
                socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                struct.pack("=4s4s", socket.inet_aton(self._group),
                            socket.inet_aton(self._iface)))
            sock.settimeout(0.2)
        except OSError as exc:
            self.last_error = f"hmd capture socket setup failed: {exc}"
            sock.close()
            return False
        self._sock = sock
        try:
            self._handle = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            self.last_error = f"hmd capture open failed: {exc}"
            self._handle = None
            sock.close()
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        sock = self._sock
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(65507)
            except socket.timeout:
                continue
            except OSError:
                break
            self.received += 1
            try:
                self._ingest(parse_command_event(data))
            except Exception as exc:  # 单包坏不得拖垮整个监听
                self.last_error = repr(exc)[:200]

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        sock = getattr(self, "_sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
