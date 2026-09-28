"""第一阶段录制：OSC 时间戳记录 + 方形走位（撞墙自适应）。

这个脚本同时做两件事，所以两路数据天然共用一个单调时钟、零对齐误差：

  1. **记录**：监听 UDP 9001，把 VRChat 每条回传按 ``time.monotonic()`` 落 JSONL。
     这是现有素材缺的那一半 —— 没有它就没有米制尺度源。
  2. **走位**：按状态机向 UDP 9000 发移动轴。直线段是纯平移（角速度 0），
     转向集中到短事件里，总转向锁定 360° 使其闭合。

为什么撞墙自适应不需要视觉、不需要地图：
    本脚本本来就在读 ``VelocityX/Z``，速度塌陷就是撞墙。这一判据项目里
    已实测过：空地 4.0 m/s vs 顶墙 0.08 m/s，差 50 倍，阈值很硬。
    所以「撞了就立刻转段」是白捡的闭环信号。

=== 使用方式 ===
  1. VRChat 里开 OSC（Action Menu > Options > OSC > Enabled）
  2. 让角色站在房间中较空的位置，朝向房间最长的一边
  3. 启动 OBS 录制（1080p60 / H.264 / 约 20 Mbps）
  4. 运行本脚本，按提示回车开始

  脚本会自动走：静止 3 秒（对齐标记）→ [直线 3 秒 → 转 90°] × 4 → 静止 3 秒

=== 注意 ===
  - 本脚本**直接向 9000 发 UDP**，不经过插件后端，因此不受后端 autonomy
    闩锁约束。录制是用户主动发起的一次性操作，且只发移动轴、不发姿态。
    中途 Ctrl+C 会先归零所有轴再退出。
  - 轴需要持续重发：VRChat 的移动轴有超时归零，所以按 --hz 周期重发。
"""

import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time

SEND_HOST = "127.0.0.1"
SEND_PORT = 9000
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 9001

ADDR_MOVE_VERTICAL = "/input/Vertical"
ADDR_LOOK_HORIZONTAL = "/input/LookHorizontal"
ADDR_MOVE_HORIZONTAL = "/input/Horizontal"   # v2 横移（绕行用）；v1 恒发 0
PARAM_VX = "/avatar/parameters/VelocityX"
PARAM_VZ = "/avatar/parameters/VelocityZ"


# ---------------------------------------------------------------- OSC 编解码
# 自包含实现（只支持本脚本用到的类型），避免对插件模块的运行时依赖。

def _osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    while len(raw) % 4:
        raw += b"\x00"
    return raw


def encode_float(address: str, value: float) -> bytes:
    return _osc_string(address) + _osc_string(",f") + struct.pack(">f", float(value))


def encode_int(address: str, value: int) -> bytes:
    """按钮类地址（/input/LookRight 等）用 int：1=按下，0=释放。"""
    return _osc_string(address) + _osc_string(",i") + struct.pack(">i", int(value))


def _read_string(packet: bytes, offset: int):
    end = packet.find(b"\x00", offset)
    if end < 0:
        raise ValueError("unterminated OSC string")
    value = packet[offset:end].decode("utf-8", "replace")
    return value, (end + 4) & ~3


def decode_packet(packet: bytes):
    """解析 OSC 消息或 bundle，返回 [(address, [args...]), ...]。"""
    if packet.startswith(b"#bundle\x00"):
        out, offset = [], 16
        while offset + 4 <= len(packet):
            (length,) = struct.unpack_from(">i", packet, offset)
            offset += 4
            if length <= 0 or offset + length > len(packet):
                break
            out.extend(decode_packet(packet[offset:offset + length]))
            offset += length
        return out

    address, offset = _read_string(packet, 0)
    tags, offset = _read_string(packet, offset)
    if not tags.startswith(","):
        return []
    args = []
    for tag in tags[1:]:
        if tag == "f":
            args.append(struct.unpack_from(">f", packet, offset)[0])
            offset += 4
        elif tag == "i":
            args.append(struct.unpack_from(">i", packet, offset)[0])
            offset += 4
        elif tag == "s":
            value, offset = _read_string(packet, offset)
            args.append(value)
        elif tag in ("T", "F"):
            args.append(tag == "T")
        else:
            break
    return [(address, args)]


# ---------------------------------------------------------------- 参数缓存

class ParamCache:
    """记录每个参数的最新值与到达时刻，供主线程读实时速度。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._values: dict[str, tuple[float, float]] = {}
        self._packets = 0
        self._first_packet_t: float | None = None
        # 最近一次收到的速度（零阶保持）：动作日志每拍都要记速度，
        # 而 VRChat 是变化驱动——静止/匀速时没有新包，不能记成"没速度"。
        self._last_vx: float | None = None
        self._last_vz: float | None = None

    def update(self, address: str, value: float) -> None:
        with self._lock:
            if self._first_packet_t is None:
                self._first_packet_t = time.monotonic()
            self._values[address] = (time.monotonic(), float(value))
            self._packets += 1
            if address == PARAM_VX:
                self._last_vx = float(value)
            elif address == PARAM_VZ:
                self._last_vz = float(value)

    def velocity_last_known(self):
        """(vx, vz) 最近已知值，零阶保持；还没收到过时为 (None, None)。"""
        with self._lock:
            return self._last_vx, self._last_vz

    def first_packet_monotonic(self):
        """第一条包到达时的 monotonic 时刻。

        ⚠️ 必须记录它，不能事后推断：OSC 的时间轴相对「第一条包」，
        而阶段标记相对「脚本启动」；开头静止不发包时两者会差好几秒。
        本项目已因此出错四次（文件名当结束时刻 / probe 窗口当整段 /
        对齐偏移 / calib 的阶段边界），所以这里直接把它写进产物。
        """
        with self._lock:
            return self._first_packet_t

    def fresh(self, address: str, max_age_ms: float = 500.0):
        with self._lock:
            item = self._values.get(address)
        if item is None:
            return None
        stamp, value = item
        if (time.monotonic() - stamp) * 1000.0 > max_age_ms:
            return None
        return value

    def any_seen(self) -> bool:
        """是否收到过任何包。用来判断链路是否活着。

        不能只看 VelocityX：VRChat 参数是变化驱动的，站着不动时根本不发速度包，
        等它会把 probe 白等满 5 秒（第一版录制实测浪费了 5 秒静止）。
        """
        with self._lock:
            return bool(self._values)

    def packet_count(self) -> int:
        with self._lock:
            return self._packets

    def horizontal_speed(self):
        """水平速度，用 hypot 而非三维模长——贴墙下滑不该被读成前进。"""
        vx = self.fresh(PARAM_VX)
        vz = self.fresh(PARAM_VZ)
        if vx is None or vz is None:
            return None
        return math.hypot(vx, vz)


# ---------------------------------------------------------------- 录制器

def receiver(sock: socket.socket, cache: ParamCache, writer, stop: threading.Event):
    sock.settimeout(0.2)
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        stamp = time.monotonic()
        try:
            messages = decode_packet(data)
        except Exception:
            continue
        for address, args in messages:
            if args and isinstance(args[0], (int, float)) and not isinstance(args[0], bool):
                cache.update(address, args[0])
            writer.write(json.dumps({"t": round(stamp, 6), "addr": address,
                                     "args": args}, ensure_ascii=False) + "\n")


class Axes:
    """持续重发轴值。VRChat 的移动轴会超时归零，所以不能只发一次。"""

    def __init__(self, sock: socket.socket, hz: float):
        self._sock = sock
        self._period = 1.0 / max(1.0, hz)
        self._next = 0.0
        self._vertical = 0.0
        self._look = 0.0
        self._horizontal = 0.0
        self.send_errors = 0   # v2 动作日志用它判定 actual_send_result

    def set(self, vertical: float, look: float, horizontal: float = 0.0) -> None:
        self._vertical = float(vertical)
        self._look = float(look)
        self._horizontal = float(horizontal)

    def tick(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next:
            return
        self._next = now + self._period
        try:
            self._sock.sendto(encode_float(ADDR_MOVE_VERTICAL, self._vertical),
                              (SEND_HOST, SEND_PORT))
            self._sock.sendto(encode_float(ADDR_LOOK_HORIZONTAL, self._look),
                              (SEND_HOST, SEND_PORT))
            self._sock.sendto(encode_float(ADDR_MOVE_HORIZONTAL, self._horizontal),
                              (SEND_HOST, SEND_PORT))
        except OSError:
            self.send_errors += 1

    def release(self) -> None:
        self.set(0.0, 0.0, 0.0)
        for _ in range(3):
            self.tick(force=True)
            time.sleep(0.05)


# ---------------------------------------------------------------- 锚点（OSC↔视频时间绑定）
# 纯函数：锚点计算与时钟注入解耦，便于用假时钟单元化测试。

def wait_obs_first_frame(obs_session, clock, timeout=10.0, poll=0.05, sleep=time.sleep):
    """等 OBS 真正写出第一帧，返回那一刻的 monotonic（**不是**发命令那一刻）。

    背景（实测，2026-09-21）：`run.json` 里 `obs_start_monotonic` 取的是
    `StartRecord` 被**确认**的瞬间，而 OBS 真正落第一帧要比它晚 2.25~2.85 s
    （两个 run 互相关测得 +2.85 / +2.25，corr 0.61 / 0.65）。拿命令时刻当
    视频帧 0，会把整条视觉时间轴往前推两秒多 —— 曾据此误判
    「HMD yaw 与画面脱钩」（结论已撤销）。

    判据只用**可观测信号**：`GetRecordStatus().outputDuration > 0`（毫秒），
    即 OBS 已经输出了非零时长的画面。等不到就返回 (None, "first_frame_timeout")，
    **绝不回退到命令时刻冒充首帧时刻**。
    """
    if obs_session is None:
        return None, "no_obs_session"
    deadline = clock() + timeout
    while True:
        try:
            st = obs_session.request("GetRecordStatus")
        except Exception:
            return None, "record_status_failed"
        # OBS 5.x：outputDuration 单位为毫秒；字段缺失时不猜。
        dur = st.get("outputDuration", None)
        if isinstance(dur, (int, float)) and dur > 0:
            return round(float(clock()), 6), None
        if clock() >= deadline:
            return None, "first_frame_timeout"
        sleep(poll)


def build_obs_start_event(ok, clock, reason=None,
                          first_frame_monotonic=None, first_frame_reason=None):
    """构造 obs_start 事件，把 OSC 的 time.monotonic() 时钟绑到 OBS 起录确认时刻。

    - ok=True 且 reason=None：已锚定。写入 StartRecord 被确认那一瞬的
      time.monotonic() 与时钟来源，下游可据此把 OSC 包时间轴对齐到视频帧。
    - 否则（reason 非空）：代码无法从可观测信号确定 OBS 真正起录的时刻，
      obs_start_monotonic 为 null 并附带机器可读原因，下游据此区分
      「已锚定」与「未锚定」，绝不凭空捏造时间戳。
    clock 生产路径传 time.monotonic，测试传假时钟。

    **两个时刻必须分开记**（2026-09-21 修正）：
      · obs_start_monotonic      = 发 StartRecord 并被确认的时刻（命令时刻）
      · obs_first_frame_monotonic= OBS 真正落第一帧的时刻（帧 0 时刻）
    二者相差实测 2.25~2.85 s。视频帧号 ↔ 时间的换算**只能**用后者；
    拿前者当帧 0 会让整条视觉时间轴整体前移。等不到首帧时后者为 null +
    obs_first_frame_reason，**不回退、不用命令时刻冒充**。
    """
    if ok and reason is None:
        ev = {
            "event": "obs_start",
            "ok": True,
            "obs_start_monotonic": round(float(clock()), 6),
            "obs_clock": "time.monotonic",
        }
        if first_frame_monotonic is not None:
            ev["obs_first_frame_monotonic"] = round(float(first_frame_monotonic), 6)
            ev["obs_start_offset_s"] = round(
                float(first_frame_monotonic) - float(clock()), 6)
        elif first_frame_reason:
            ev["obs_first_frame_monotonic"] = None
            ev["obs_first_frame_reason"] = first_frame_reason
        return ev
    return {
        "event": "obs_start",
        "ok": False,
        "obs_start_monotonic": None,
        "obs_clock": "time.monotonic",
        "obs_start_reason": reason or "not_started",
    }


def extract_obs_anchor(summary):
    """从 run.json 摘要抽取锚点，向前兼容旧格式（无锚点字段）。

    旧 run.json 的 obs_start 事件只有 {"event","ok"}。此时 anchored=False、
    obs_start_monotonic=None，下游降级为视觉代理而非捏造锚点。
    返回 {"anchored","obs_start_monotonic","obs_clock","reason",
          "frame0_monotonic","frame0_reason"}。

    **frame0_monotonic 是视频帧号↔时间的唯一基准**（有则用它）。
    缺失时（旧 run / 首帧超时）为 None，调用方必须显式降级，
    **不得拿 obs_start_monotonic（命令时刻）顶替** —— 二者实测差 2.25~2.85 s。
    """
    ev = None
    for e in summary.get("events", []):
        if e.get("event") == "obs_start":
            ev = e
            break
    if ev is None:
        return {"anchored": False, "obs_start_monotonic": None,
                "obs_clock": None, "reason": "no_obs_start_event",
                "frame0_monotonic": None, "frame0_reason": None}
    mono = ev.get("obs_start_monotonic", None)
    anchored = bool(ev.get("ok", False)) and mono is not None
    f0 = ev.get("obs_first_frame_monotonic", None)
    return {
        "anchored": anchored,
        "obs_start_monotonic": mono,
        "obs_clock": ev.get("obs_clock", None),
        "reason": None if anchored else ev.get("obs_start_reason", None),
        "frame0_monotonic": f0 if isinstance(f0, (int, float)) else None,
        "frame0_reason": None if isinstance(f0, (int, float))
                         else ev.get("obs_first_frame_reason", "not_recorded"),
    }


def resolve_outdir(runs_root: str, stamp: str, user_outdir: str | None) -> str:
    """run_id = 启动时间戳（YYYYMMDD-HHMMSS）。

    同一秒内重跑会撞目录——**绝不静默合并**两个 run 的产物，
    自动追加 -2/-3 后缀。--outdir 显式指定时按用户的来（含已存在目录）。
    """
    if user_outdir:
        return user_outdir
    outdir = os.path.join(runs_root, stamp)
    n = 2
    while os.path.exists(outdir):
        outdir = os.path.join(runs_root, "%s-%d" % (stamp, n))
        n += 1
    return outdir


def main() -> int:
    ap = argparse.ArgumentParser(description="第一阶段录制：OSC 记录 + 方形走位")
    ap.add_argument("--vertical", type=float, default=0.30,
                    help="前进轴值。实测 0.3 → 0.667 m/s，0.6 → 1.667 m/s（非线性）")
    ap.add_argument("--forward-s", type=float, default=4.0,
                    help="直线段时长（秒）。0.667 m/s × 4 s ≈ 2.7 m/边")
    ap.add_argument("--turn-s", type=float, default=1.1,
                    help="每段转向时长（秒）。1.2 s 实测转多了约 7%%，已收到 1.1")
    ap.add_argument("--turn-value", type=float, default=0.55,
                    help="转向轴值")
    ap.add_argument("--legs", type=int, default=4,
                    help="直线段数量。**4 段 = 360° = 一圈闭合**；不要随口加大，"
                         "总转向必须保持 360° 才闭合")
    ap.add_argument("--stall-speed", type=float, default=0.15,
                    help="低于此速度视为撞墙（实测 空地 4.0 vs 顶墙 0.08）")
    ap.add_argument("--stall-ticks", type=int, default=6,
                    help="连续多少拍低速才判撞墙")
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="开头/结尾静止时长。第一版用 3 s，加上 probe 5 s 让静止段"
                         "占到 70%%，把回环检测都淹了；现收紧")
    ap.add_argument("--probe-s", type=float, default=1.5,
                    help="链路探测时长上限（收到任何包即继续，不再等满）")
    ap.add_argument("--hz", type=float, default=20.0, help="轴重发频率")
    ap.add_argument("--no-move", action="store_true",
                    help="只记录 OSC、不发任何轴。想手动在 VRChat 里走位时用这个 —— "
                         "**手动走也必须记录 OSC**，否则没有米制尺度源，尺度锚定不成立")
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="配合 --no-move：自动结束的时长（秒）。0 = 直到 Ctrl+C")
    ap.add_argument("--obs", action="store_true",
                    help="自动控制 OBS 起停录制。需 OBS 已启用 WebSocket 服务器；"
                         "端口与密码默认从 OBS 自己的配置读取")
    ap.add_argument("--obs-host", default="127.0.0.1")
    ap.add_argument("--obs-port", type=int, default=None)
    ap.add_argument("--obs-password", default=None)
    ap.add_argument("--no-confirm", action="store_true",
                    help="跳过「按回车开始」，直接开录（便于无人值守）")
    ap.add_argument("--outdir", default=None)
    # ---- v2（HMD 指南针路线，报告 §22.19 / 手册 §7）----
    ap.add_argument("--route-file", default=None,
                    help="v2 路线 JSON（op: forward/turn/strafe/settle/hold/"
                         "beacon/compass_check）。给定后走 v2 驱动：转向全走"
                         "虚拟 HMD yaw，弃用 LookHorizontal")
    ap.add_argument("--no-hmd", action="store_true",
                    help="v2 路线下也不推 HMD 帧（干跑路线用）")
    ap.add_argument("--video-fps", type=float, default=60.0,
                    help="OBS 录像帧率——动作日志 frame_index 的时间基准")
    ap.add_argument("--first-frame-timeout-s", type=float, default=10.0,
                    help="等 OBS 落首帧（GetRecordStatus.outputDuration>0）的超时。"
                         "实测命令→首帧 2.25~2.85 s；超时则帧 0 时刻留空、不伪造")
    ap.add_argument("--hmd-rate", type=float, default=60.0,
                    help="HMD 推帧频率（AnyaDance UI 惯例 60 Hz）")
    ap.add_argument("--hmd-height", type=float, default=1.5,
                    help="虚拟 HMD 高度（米）")
    ap.add_argument("--hmd-turn-rate", type=float, default=60.0,
                    help="HMD yaw 默认转向速率（度/秒）。太快身体跟不上")
    ap.add_argument("--strafe-value", type=float, default=0.5,
                    help="绕行横移轴值（左右方向未标定，由视频实证）")
    ap.add_argument("--strafe-s", type=float, default=1.0,
                    help="每次绕行的横移时长（秒）")
    ap.add_argument("--max-detours", type=int, default=2,
                    help="一条腿最多绕行次数；仍撞则放弃该腿")
    # ---- v2.1 手动 HMD 模式（09-20 定调：HMD 手动操作，录制器只采集）----
    ap.add_argument("--hmd-capture", action="store_true",
                    help="监听驱动命令回执多播（39571），把 AnyaDance 驱动实际"
                         "接受的 HMD/设备位姿按到达时刻落 hmd_frames.jsonl。"
                         "配合手动操作（AnyaDance.exe 鼠标操控）使用；与推帧互斥")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    runs_root = os.path.join(
        repo_root, ".slam_probe", "offline_probe", "recorder", "runs")
    outdir = resolve_outdir(runs_root, stamp, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    run_id = os.path.basename(outdir)
    print("run_id：%s" % run_id)

    # v2 路线**前置**校验：坏路线必须在开录/OBS 起录之前失败，不能录到一半炸。
    route = None
    if a.route_file:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import route_v2
            route = route_v2.load_route(a.route_file)
        except Exception as exc:
            print("❌ 路线文件无效（%s）：%s" % (a.route_file, exc))
            return 2
        print("v2 路线：%d 个 op（转向走 HMD，弃用 LookHorizontal）"
              % len(route["legs"]))

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        recv_sock.bind((LISTEN_HOST, LISTEN_PORT))
    except OSError as exc:
        print("无法绑定 %s:%d —— 9001 可能已被插件后端占用。\n%s"
              % (LISTEN_HOST, LISTEN_PORT, exc))
        print("请先停掉后端（或改 LISTEN_PORT 并让 VRChat 转发）。")
        return 2

    cache = ParamCache()
    stop = threading.Event()
    jsonl_path = os.path.join(outdir, "osc.jsonl")
    jsonl = open(jsonl_path, "w", encoding="utf-8")
    thread = threading.Thread(target=receiver,
                              args=(recv_sock, cache, jsonl, stop), daemon=True)
    thread.start()

    axes = Axes(send_sock, a.hz)
    events = []
    obs_session = None
    obs_video = None
    print("输出目录：%s" % outdir)

    # ---- v2.1 手动 HMD 被动捕获：与推帧互斥（两个 HMD 发送端会争抢）----
    hmd_capture = None
    if a.hmd_capture:
        import route_v2 as _rv2c
        hmd_capture = _rv2c.HmdCaptureListener(
            os.path.join(outdir, "hmd_frames.jsonl"))
        if hmd_capture.start():
            print("HMD 捕获：监听 %s:%d（手动操作模式，推帧已禁用）"
                  % (_rv2c.DRIVER_LOG_GROUP, _rv2c.DRIVER_LOG_PORT))
        else:
            print("⚠️ HMD 捕获启动失败：%s（继续录制，只是没有 HMD 通道）"
                  % hmd_capture.last_error)
            hmd_capture = None

    # ---- OBS 联动：连接与起录 ----
    if a.obs:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import obs_ctl
            pw, port = a.obs_password, a.obs_port
            if pw is None or port is None:
                auto_pw, auto_port = obs_ctl.load_obs_config()
                if pw is None:
                    pw = auto_pw
                if port is None:
                    port = auto_port
            obs_session = obs_ctl.Obs(a.obs_host, int(port), pw or "")
            print("已连接 OBS（%s:%s）" % (a.obs_host, port))
        except Exception as exc:
            print("⚠️  OBS 连接失败：%s" % exc)
            print("    请手动开始录制，或检查 OBS「工具 > WebSocket 服务器设置」。")
            obs_session = None

    if not a.no_confirm:
        print("准备好后按回车开始录制……")
        try:
            input()
        except EOFError:
            pass

    obs_ok = False
    obs_reason = None
    if obs_session is not None:
        try:
            obs_session.request("StartRecord")
            obs_ok = True
            print("→ OBS 已开始录制")
        except Exception as exc:
            obs_reason = "obs_start_failed"
            print("⚠️  启动 OBS 录制失败：%s（请手动开始）" % exc)
    elif not a.obs:
        obs_reason = "no_obs_flag"
        print("未启用 --obs：请自行开始 OBS 录制。")
    else:
        # --obs 已传但连接失败（obs_session is None）：无法锚定
        obs_reason = "obs_connect_failed"
    # 锚点只取一次，obs_start 事件与动作日志时间基准共用同一时刻——
    # 两次各自 time.monotonic() 会差几十微秒，没必要引入这点漂移。
    anchor_t = time.monotonic()
    # 但 anchor_t 是**命令时刻**，不是视频帧 0 时刻。OBS 落第一帧要晚
    # 2.25~2.85 s（实测），所以额外等 outputDuration>0 拿真正的帧 0。
    frame0_t = frame0_reason = None
    if obs_ok:
        frame0_t, frame0_reason = wait_obs_first_frame(
            obs_session, time.monotonic, timeout=a.first_frame_timeout_s)
        if frame0_t is not None:
            print("→ OBS 首帧已落盘：命令→首帧 %.2f s" % (frame0_t - anchor_t))
        else:
            print("⚠️  等不到 OBS 首帧（%s）——不伪造帧 0 时刻，"
                  "视频时间基准留空。" % frame0_reason)
    events.append(build_obs_start_event(obs_ok, lambda: anchor_t,
                                        reason=obs_reason,
                                        first_frame_monotonic=frame0_t,
                                        first_frame_reason=frame0_reason))

    # ---- v2 动作日志：只在已锚定时启用（不伪造时间基准）----
    action_recorder = None
    timebase = None
    action_log_path = os.path.join(outdir, "action_timeline.jsonl")
    # 视频帧号 ↔ 时间的基准**只能**是首帧时刻 frame0_t；拿命令时刻 anchor_t
    # 会让整条动作时间轴前移 2 s 多（与视觉对不上）。拿不到首帧就降级。
    frame0_anchor = frame0_t if frame0_t is not None else anchor_t
    if route is not None:
        if obs_ok:
            import route_v2 as _rv2
            timebase = _rv2.VideoTimebase(a.video_fps, clock=lambda: frame0_anchor)
            timebase.begin()   # anchor = 首帧时刻（缺则退回命令时刻并已告警）
            action_recorder = _rv2.ActionLogRecorder(action_log_path, timebase)
            if action_recorder.start():
                print("动作日志：%s（%.0f fps 基准，锚点已共享）"
                      % (action_log_path, a.video_fps))
            else:
                print("⚠️ 动作日志打不开：%s" % action_recorder.last_error)
                action_recorder = None
        else:
            print("⚠️ OBS 未锚定（%s）——不写动作日志（不伪造时间基准），"
                  "仅视觉/OSC 通道可用。" % obs_reason)

    # 确认链路活着 —— 这是整条路线的命门。
    # 注意不能只等 VelocityX：静止时 VRChat 根本不发速度包，会白等满整段。
    # 收到「任何包」就说明 9001 链路通；VelocityX 单独在运动中复核。
    probe_deadline = time.monotonic() + a.probe_s
    vx_seen = any_seen = False
    while time.monotonic() < probe_deadline:
        if cache.fresh(PARAM_VX, max_age_ms=5000.0) is not None:
            vx_seen = any_seen = True
            break
        if cache.any_seen():
            any_seen = True
            break
        time.sleep(0.05)
    print("链路探测：%s（%.1fs）"
          % ("收到数据" if any_seen else "静止无数据",
             time.monotonic() - (probe_deadline - a.probe_s)))
    if not any_seen:
        print("  说明：静止时 VRChat 根本不发参数包（变化驱动），**这是正常的**。")
        print("        开始走位后下面会显示实时包数；若走起来仍是 0，再去查 OSC。")
    events.append({"event": "probe", "any_seen": any_seen, "velocity_seen": vx_seen})

    hmd = None
    try:
        if a.no_move:
            # 手动走位模式：不碰任何轴，只把 OSC 记下来。
            print("--no-move：不发任何轴，只记录 OSC。")
            print("  请手动在 VRChat 里走位：**慢速前进、少转头**，走一圈回到起点。")
            if a.duration_s > 0:
                print("  将在 %.0f 秒后自动结束。" % a.duration_s)
            else:
                print("  走完按 Ctrl+C 结束记录。")
            # 实时回报：走起来后这里必须出现不断增长的包数，
            # 否则说明 OSC 链路没通（而不是"什么都没发生"）。
            t_start = time.monotonic()
            last_report = 0.0
            while True:
                axes.tick()          # 保持 0，纯粹保证不残留输入
                time.sleep(0.1)
                now = time.monotonic()
                if now - t_start - last_report < 2.0:
                    continue
                last_report = now - t_start
                n = cache.packet_count()
                sp = cache.horizontal_speed()
                print("  [%5.0fs] 累计 %4d 包%s"
                      % (now - t_start, n,
                         "    当前水平速度 %.2f m/s" % sp if sp is not None
                         else "    （静止）"))
                if a.duration_s > 0 and now - t_start >= a.duration_s:
                    break
        elif route is not None:
            # ---- v2 路线驱动：HMD 指南针 + 横移绕行 + 动作日志 ----
            import route_v2
            if a.no_hmd or hmd_capture is not None:
                print("v2：HMD 推帧禁用（%s）"
                      % ("--no-hmd" if a.no_hmd else "手动捕获模式"))
            else:
                print("⚠️ 推 HMD 前确认 AnyaDance.exe 伴随 UI 已关闭，"
                      "否则两个发送端争抢 HMD。")
                hmd = route_v2.HmdPusher(hz=a.hmd_rate, height=a.hmd_height)
                hmd.start()
            driver = route_v2.RouteDriver(
                axes=axes, cache=cache, events=events,
                recorder=action_recorder, hmd=hmd,
                vertical=a.vertical, hz=a.hz,
                stall_speed=a.stall_speed, stall_ticks=a.stall_ticks,
                strafe_value=a.strafe_value, strafe_s=a.strafe_s,
                max_detours=a.max_detours, hmd_turn_rate=a.hmd_turn_rate)
            driver.run(route)
        else:
            # ---- 开头静止：对齐标记 ----
            print("[settle] 静止 %.1fs（对齐标记）" % a.settle_s)
            axes.set(0.0, 0.0)
            t_end = time.monotonic() + a.settle_s
            while time.monotonic() < t_end:
                axes.tick()
                time.sleep(0.02)

            for leg in range(a.legs):
                # ---- 直线段：纯平移 ----
                t0 = time.monotonic()
                low = 0
                stalls = 0
                speeds = []
                while True:
                    elapsed = time.monotonic() - t0
                    if elapsed >= a.forward_s:
                        break
                    axes.set(a.vertical, 0.0)      # look 恒为 0 -> 角速度 0
                    axes.tick()
                    sp = cache.horizontal_speed()
                    if sp is not None:
                        speeds.append(sp)
                        low = low + 1 if sp < a.stall_speed else 0
                        if low >= a.stall_ticks:
                            stalls += 1
                            print("  [forward %d/%d] 撞墙判定，提前转段 (%.2fs)"
                                  % (leg + 1, a.legs, elapsed))
                            break
                    time.sleep(1.0 / a.hz)
                fwd_s = time.monotonic() - t0
                avg = (sum(speeds) / len(speeds)) if speeds else None
                events.append({"event": "forward", "leg": leg + 1,
                               "duration_s": round(fwd_s, 2), "stalls": stalls,
                               "avg_speed": None if avg is None else round(avg, 3),
                               "samples": len(speeds)})
                print("[forward %d/%d] %.2fs  撞墙 %d  均速 %s"
                      % (leg + 1, a.legs, fwd_s, stalls,
                         "None" if avg is None else "%.2f" % avg))

                # ---- 转向段：短、集中，事后整段丢弃 ----
                t0 = time.monotonic()
                while time.monotonic() - t0 < a.turn_s:
                    axes.set(0.0, a.turn_value)
                    axes.tick()
                    time.sleep(1.0 / a.hz)
                events.append({"event": "turn", "leg": leg + 1,
                               "duration_s": round(time.monotonic() - t0, 2)})

            # ---- 结尾静止 ----
            print("[settle] 静止 %.1fs（判断闭合）" % a.settle_s)
            axes.set(0.0, 0.0)
            t_end = time.monotonic() + a.settle_s
            while time.monotonic() < t_end:
                axes.tick()
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n中断，归零所有轴")
    finally:
        axes.release()
        if hmd is not None:
            hmd.stop()   # 保持最后姿态退出，不做归零（不伪造）
        if hmd_capture is not None:
            hmd_capture.stop()
        if action_recorder is not None:
            action_recorder.stop()
        time.sleep(0.3)
        if obs_session is not None:
            try:
                data = obs_session.request("StopRecord")
                obs_video = data.get("outputPath")
                print("→ OBS 已停止录制：%s" % obs_video)
            except Exception as exc:
                print("⚠️  停止 OBS 录制失败：%s（请在 OBS 里手动停止）" % exc)
            finally:
                obs_session.close()
        stop.set()
        thread.join(timeout=1.0)
        jsonl.close()
        recv_sock.close()
        send_sock.close()

    total_packets = cache.packet_count()
    # v2 产物信息与读回自检：用真正的加载器读回，证明格式没漂移。
    action_info: dict = {"enabled": action_recorder is not None,
                         "path": action_log_path if route is not None else None}
    if action_recorder is not None:
        import route_v2
        action_info.update({
            "written": action_recorder.written,
            **route_v2.readback_summary(action_log_path),
        })
    elif route is not None:
        action_info["reason"] = obs_reason if not obs_ok else "recorder_start_failed"
    hmd_info = None
    if hmd is not None:
        hmd_info = {"enabled": True, "rate_hz": a.hmd_rate,
                    "height_m": a.hmd_height, "sent": hmd.sent,
                    "errors": hmd.errors,
                    "final_yaw_deg": round(hmd.yaw_deg, 2)}
    hmd_capture_info = None
    if a.hmd_capture:
        if hmd_capture is not None:
            hmd_capture_info = {
                "enabled": True,
                "path": os.path.join(outdir, "hmd_frames.jsonl"),
                "received": hmd_capture.received,
                "accepted": hmd_capture.accepted,
                "hmd_frames": hmd_capture.hmd_frames,
                "duplicates": hmd_capture.duplicates,
                "last_seq": hmd_capture.last_seq,
                "error": hmd_capture.last_error,
            }
        else:
            hmd_capture_info = {"enabled": True, "error": "start_failed"}
    summary = {
        "started_at": stamp,
        "run_id": run_id,
        "params": {k: getattr(a, k) for k in
                   ("vertical", "forward_s", "turn_s", "turn_value",
                    "legs", "stall_speed", "stall_ticks", "settle_s",
                    "probe_s", "hz", "route_file", "video_fps",
                    "hmd_rate", "hmd_height", "hmd_turn_rate",
                    "strafe_value", "strafe_s", "max_detours")},
        "total_packets": total_packets,
        "first_packet_monotonic": cache.first_packet_monotonic(),
        "first_packet_clock": "time.monotonic",
        "probe_any_seen": any_seen,
        "velocity_seen": vx_seen,
        "obs_started": obs_ok,
        "obs_video": obs_video,
        "events": events,
        "jsonl": jsonl_path,
        "action_timeline": action_info,
        "hmd": hmd_info,
        "hmd_capture": hmd_capture_info,
    }
    with open(os.path.join(outdir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print("\n=== 摘要 ===")
    legs = [e for e in events if e.get("event") == "forward"]
    moved = sum(e["duration_s"] for e in legs)
    turns = sum(e["duration_s"] for e in events if e.get("event") == "turn")
    print("直线段 %d 段，撞墙 %d 次，均速 %s"
          % (len(legs), sum(e["stalls"] for e in legs),
             "无数据" if not any(e["avg_speed"] for e in legs)
             else "%.2f" % (sum(e["avg_speed"] for e in legs if e["avg_speed"])
                            / max(1, sum(1 for e in legs if e["avg_speed"])))))
    print("运动段合计 %.1fs（直线 %.1fs + 转向 %.1fs），静止 %.1fs"
          % (moved + turns, moved, turns, a.settle_s * 2))
    print("OSC 记录：%s" % jsonl_path)
    print("OSC 合计 %d 包" % total_packets)
    if total_packets == 0:
        print("⚠️  整段一个 OSC 包都没收到 —— 尺度锚定不成立。")
    elif not vx_seen:
        print("提示：整段收到 %d 包，但探测窗口内未见到 VelocityX。"
              "以 osc.jsonl 里的实际地址统计为准（用 verify_run.py 看）。"
              % total_packets)
    else:
        print("✓ 命门通过：Velocity 有回传。")
    print("\n录完记得跑：python verify_run.py --run %s --video \"%s\""
          % (outdir, obs_video or "<OBS 输出>"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
