"""AnyaDance 控制器的有界闭环导航。

该导航器刻意设计为小型局部控制循环。它从不调用模型、等待 HTTP 请求，也不会在世界状态未知时凭空生成移动指令。高层自主目标会被转换为简短的、最新优先的控制器更新，且每次更新都必须以新鲜且可见的实体观测作为门控条件。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Mapping

from .direction_memory import DirectionMemory, SegmentOutcome, integrate_progress
from .explorer import ExplorerStateMachine
from .world_state import blocking_uncertainties


AxisSender = Callable[[str, float, float, int], bool]
TurnSender = Callable[[float], bool]
ReleaseSender = Callable[[str], None]
SnapshotProvider = Callable[[], Mapping[str, Any]]
TurnStateProvider = Callable[[], Mapping[str, Any]]
GoalCompleter = Callable[[str], None]

# 目标贴到画面上下边时表观高度会饱和，无法再作为距离的单调函数。此时按一个
# 更低的比例判定「已到达」，避免因为读数封顶而一直前进撞上对方。安全边界留在
# 代码里而不是配置项。
_CLIPPED_REACH_FACTOR = 0.6

# wander 是 LLM 根据当前画面规划的一条短路段，不是本地随机漫游。导航器只负责
# 执行、避撞和停车；每段完成后由宿主带一张新画面唤醒 LLM 决定下一段。
_WANDER_DEFAULT_STEP_S = 2.0
_WANDER_MAX_STEP_S = 3.0
# 自由移动的前进轴上限按行为分开：wander 是朝着可见空地巡航，depart 是背对观察点
# 后退，看不见身后，所以更慢。两个值只在这里定义，执行侧和终态摘要共用同一份解析，
# 避免回报给主 LLM 的 requested 与实际推的摇杆漂移。
_WANDER_MAX_FORWARD_AXIS = 0.45
_DEPART_MAX_FORWARD_AXIS = 0.35
# 闲逛时没有“目标方向”可以容忍斜走；forward_ratio=0.738 的实机贴墙滑行对
# 普通目标导航尚可接受，对自由巡航则应尽快绕开。这里只收紧 wander，不改变
# 已经用正负样本校准过的全局 0.55 阈值。
_WANDER_SLIP_FORWARD_RATIO = 0.85
_WANDER_SLIP_TICKS = 3

# 门控积分丢弃转向拍的判据。AngularY 只在移动期间回传，量级未标定，所以只用它
# 判「有没有在转」，不参与任何角度积分。
#
# 年龄门槛比一拍略宽：10Hz 下一拍 100ms，250ms 允许错开一两个包，又不至于让上
# 一次转向的旧包把整段直行都标成转弯。
_ANGULAR_RESTING = 0.05
_ANGULAR_FRESH_MS = 250.0


@dataclass(frozen=True)
class NavigatorConfig:
    """Safety limits for the local 10 Hz navigation loop."""

    tick_hz: float = 10.0
    pulse_ms: int = 220
    max_forward_axis: float = 0.60
    max_turn_axis: float = 0.35
    # 转向现在直接给角度（转虚拟 HMD），不再压成摇杆量。gain < 1 是刻意的：
    # bearing 来自上一帧观测，按整个偏差一次转到位会越过中心然后来回摆；
    # 留一点余量让 10 Hz 的闭环收敛。
    turn_gain: float = 0.8
    max_turn_deg: float = 45.0
    # scheduler 接收相对转角后还要异步推进 HMD 朝向，并额外保留约 250ms 的收尾
    # 帧。新视觉帧可能在转向落地前就到达，因此仅按 revision 去重仍会叠加多条
    # 相对转角。状态门控负责等待真实完成；该冷却同时覆盖 submit 到 scheduler
    # 出队之间 turning 尚未变成 true 的短暂竞态。
    turn_cooldown_s: float = 0.20
    max_observation_age_ms: int = 2500
    min_confidence: float = 0.40
    # 检测置信度偶发掉一帧或短暂漏检时，继续使用最后一条可靠观测一小段时间。
    # 只覆盖视觉抖动；世界不确定、观测整体过期和目标切换仍会立即停车。
    target_grace_s: float = 0.30
    bearing_ema_alpha: float = 0.65
    range_ema_alpha: float = 0.50
    bearing_deadband_deg: float = 8.0
    target_distance_m: float = 1.25
    # 表观高度（归一化检测框高度）是二维检测器唯一能实测的接近度指标。它随
    # avatar 体型等比缩放，因此停止距离也随体型缩放，符合「个人空间随体型
    # 变化」的直觉；而按固定身高反算的米制距离对矮/高 avatar 会差出数倍。
    target_apparent_height: float = 0.55
    max_goal_age_s: float = 60.0
    # 「过去看看」不是永久跟随。到达后先停稳，再在较宽的注视死区内观察一小段
    # 时间，随后自动完成目标。整个过程只用本地视觉闭环，不调用模型。
    behavior_settle_s: float = 0.6
    behavior_observe_s: float = 1.8
    behavior_observe_deadband_deg: float = 18.0
    # 短暂漏检继续用 target_grace_s；持续看不见才把有限行为判为目标丢失。
    behavior_reacquire_s: float = 2.5
    # 顶着墙推摇杆的判据。检测器只看画面，永远不会告诉你「前面有堵墙」；
    # VRChat 内置 Velocity 参数是唯一能说明「命令发了但人没动」的回传。
    #
    # 已实机校准：y=0.30 实测 0.8889 m/s、角色满速 2.6667 m/s。巡航使用
    # y=0.60，最后 20% 接近区间才开始减速，同时依靠新鲜视觉及时刹车。
    # 收不到内置参数时整个判据自动失效（不阻断移动），这是刻意的——没数据不
    # 等于没动。注意 0.15 这个阈值只在命令过得了死区时才有意义，见
    # min_forward_axis。
    stall_speed_mps: float = 0.15
    # 能力已确认后的沉默或低速连续 4 tick 才算失速。加上 450 ms 起步宽限，
    # 正面顶墙约 0.85 s 进入绕行；仍保留足够去抖，不因单个漏包误停。
    stall_ticks: int = 4
    # VRChat 的起步死区。摇杆量低于它时 VRChat 完全不驱动 avatar，而导航器接近
    # 目标时会把前进轴按比例缩小，于是最后一段距离命令照发、人不动，失速守卫
    # 读到真实的 0 并误判撞墙。
    #
    # 实测（10 Hz 按住 220ms 脉冲，每档后接同方向 y=0.28 控制组证明前方是通的）：
    #   y=0.076 -> 0.0      y=0.10 -> 0.0     y=0.13 -> 0.1333
    #   y=0.15  -> 0.2222   y=0.20 -> 0.4444  y=0.28 -> 0.8
    # 用户连续实机评估认为 0.15 和 0.20 都过慢。最低档提高到 0.25，但它只在
    # 即将达到停止尺寸时使用；中远距离直接使用巡航档。
    min_forward_axis: float = 0.25
    # VelocityX/Z 只有角色实际移动时才回传。发出第一条前进命令后先给角色控制器
    # 一段起步时间；在此之前没有新速度样本是正常现象，不能计作卡墙。
    motion_start_grace_s: float = 0.45
    # 失速过的目标要记多久「够不着」。闩锁只停住当前这次尝试，解不开「换个目标
    # 再选中同一个实体」的死循环：镜面倒影是画面里置信度最高的 person，重提目标
    # 后 _select_target 照样挑它，于是又会连续多个 tick 顶墙。
    #
    # 不做永久：挡在中间的可能是会走开的人，自己转过身之后墙也不在正前方了。
    # 45 s 足够 LLM 换个方向做别的事，又不至于把一次偶然的堵塞记成永久地形。
    unreachable_ttl_s: float = 45.0
    # 斜撞墙判据。VRChat 的角色控制器把移动投影到墙面上，所以贴着墙斜着走时
    # 速度模长可能还在 stall_speed_mps 之上，而前进分量已经塌了——纯速度判据
    # 看不见这种撞墙，会一直沿着墙滑到天涯海角。
    #
    # forward_ratio = velocity_z / horizontal_speed（avatar 本地系，已实机验证）。
    # 低于该阈值即认为前进被挡住。
    #
    # 实机实测（2026-08-23，海滩地图斜角推墙）：
    #   斜着走畅通:  |h|=3.255  fwd=0.887  slip=+0.461
    #   斜撞墙滑行:  |h|=0.170  fwd=0.507  slip=-0.862   <- 连续 23 个 tick
    #
    # 注意撞墙时速度 0.170 **高于** stall_speed_mps=0.15，所以纯速度判据永远
    # 不会触发——这正是本判据存在的理由。同时 0.507 贴着 0.55 很近：阈值再低
    # 一点（比如 0.5）这次就漏了，不要为了"更保守"往下调。
    slip_forward_ratio: float = 0.55
    # 滑行要连续多少 tick 才算数。比 stall_ticks 短：滑行时人确实在动，误判的
    # 代价只是多转一次身，而卡在墙上白滑几秒的代价更大。
    slip_ticks: int = 5
    # 撞墙后自动绕行的预算。闩锁只停车、等 LLM 重提目标，往返一轮要好几秒；
    # 绕行本身不需要语义判断——滑行方向就是可通行方向，几何上已经给出答案。
    #
    # 有限次数是刻意的：真正的死胡同里怎么绕都出不去，转够 auto_recover_limit
    # 次仍然撞墙就必须把决策权交还 LLM，让它换个目标或者放弃。0 表示关闭自动
    # 绕行，退回原来的纯闩锁行为。
    auto_recover_limit: int = 3
    # 单次绕行的转身角度。正面墙没有滑行方向可用，只能盲转；取一个足够大、能
    # 明显换个朝向，又不至于原地打转的角度。
    auto_recover_turn_deg: float = 55.0
    # 正面墙先后退再转，避免贴着墙转身时蹭着墙角出不去。
    auto_recover_back_axis: float = 0.22

    def __post_init__(self) -> None:
        if not math.isfinite(self.tick_hz) or not 2.0 <= self.tick_hz <= 30.0:
            raise ValueError("navigator.tick_hz must be between 2 and 30")
        if not 100 <= int(self.pulse_ms) <= 1000:
            raise ValueError("navigator.pulse_ms must be between 100 and 1000")
        if not 0.05 <= float(self.max_forward_axis) <= 0.6:
            raise ValueError("navigator.max_forward_axis must be between 0.05 and 0.6")
        if not 0.05 <= float(self.max_turn_axis) <= 0.8:
            raise ValueError("navigator.max_turn_axis must be between 0.05 and 0.8")
        if not 0.1 <= float(self.turn_gain) <= 1.0:
            raise ValueError("navigator.turn_gain must be between 0.1 and 1")
        if not 5.0 <= float(self.max_turn_deg) <= 180.0:
            raise ValueError("navigator.max_turn_deg must be between 5 and 180")
        if not 0.0 <= float(self.turn_cooldown_s) <= 2.0:
            raise ValueError("navigator.turn_cooldown_s must be between 0 and 2")
        if not 100 <= int(self.max_observation_age_ms) <= 5000:
            raise ValueError("navigator.max_observation_age_ms must be between 100 and 5000")
        if not 0.1 <= float(self.min_confidence) <= 1.0:
            raise ValueError("navigator.min_confidence must be between 0.1 and 1")
        if not 0.0 <= float(self.target_grace_s) <= 1.0:
            raise ValueError("navigator.target_grace_s must be between 0 and 1")
        if not 0.05 <= float(self.bearing_ema_alpha) <= 1.0:
            raise ValueError("navigator.bearing_ema_alpha must be between 0.05 and 1")
        if not 0.05 <= float(self.range_ema_alpha) <= 1.0:
            raise ValueError("navigator.range_ema_alpha must be between 0.05 and 1")
        if not 1.0 <= float(self.bearing_deadband_deg) <= 30.0:
            raise ValueError("navigator.bearing_deadband_deg must be between 1 and 30")
        if not 0.25 <= float(self.target_distance_m) <= 5.0:
            raise ValueError("navigator.target_distance_m must be between 0.25 and 5")
        if not 0.05 <= float(self.target_apparent_height) <= 0.95:
            raise ValueError("navigator.target_apparent_height must be between 0.05 and 0.95")
        if not 5.0 <= float(self.max_goal_age_s) <= 600.0:
            raise ValueError("navigator.max_goal_age_s must be between 5 and 600")
        if not 0.2 <= float(self.behavior_settle_s) <= 3.0:
            raise ValueError("navigator.behavior_settle_s must be between 0.2 and 3")
        if not 0.5 <= float(self.behavior_observe_s) <= 10.0:
            raise ValueError("navigator.behavior_observe_s must be between 0.5 and 10")
        if not float(self.bearing_deadband_deg) <= float(self.behavior_observe_deadband_deg) <= 30.0:
            raise ValueError(
                "navigator.behavior_observe_deadband_deg must be between bearing_deadband_deg and 30"
            )
        if not 0.5 <= float(self.behavior_reacquire_s) <= 10.0:
            raise ValueError("navigator.behavior_reacquire_s must be between 0.5 and 10")
        if not 0.01 <= float(self.stall_speed_mps) <= 1.0:
            raise ValueError("navigator.stall_speed_mps must be between 0.01 and 1")
        if not 2 <= int(self.stall_ticks) <= 100:
            raise ValueError("navigator.stall_ticks must be between 2 and 100")
        if not 0.05 <= float(self.min_forward_axis) <= 0.6:
            raise ValueError("navigator.min_forward_axis must be between 0.05 and 0.6")
        if float(self.min_forward_axis) > float(self.max_forward_axis):
            raise ValueError("navigator.min_forward_axis must not exceed max_forward_axis")
        if not 0.0 <= float(self.motion_start_grace_s) <= 3.0:
            raise ValueError("navigator.motion_start_grace_s must be between 0 and 3")
        if not 0.0 <= float(self.unreachable_ttl_s) <= 600.0:
            raise ValueError("navigator.unreachable_ttl_s must be between 0 and 600")
        if not 0.0 <= float(self.slip_forward_ratio) <= 1.0:
            raise ValueError("navigator.slip_forward_ratio must be between 0 and 1")
        if not 2 <= int(self.slip_ticks) <= 100:
            raise ValueError("navigator.slip_ticks must be between 2 and 100")
        if not 0 <= int(self.auto_recover_limit) <= 20:
            raise ValueError("navigator.auto_recover_limit must be between 0 and 20")
        if not 5.0 <= float(self.auto_recover_turn_deg) <= 180.0:
            raise ValueError("navigator.auto_recover_turn_deg must be between 5 and 180")
        if not 0.0 <= float(self.auto_recover_back_axis) <= 0.6:
            raise ValueError("navigator.auto_recover_back_axis must be between 0 and 0.6")


@dataclass(frozen=True)
class NavigationDecision:
    state: str
    reason: str
    target_id: str | None = None
    bearing_deg: float | None = None
    distance_m: float | None = None
    side: str | None = None
    x: float = 0.0
    y: float = 0.0
    pulse_ms: int = 0
    revision: int = 0
    observed_age_ms: float | None = None
    turn_deg: float = 0.0
    observation_mode: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "target_id": self.target_id,
            "bearing_deg": None if self.bearing_deg is None else round(self.bearing_deg, 2),
            "distance_m": None if self.distance_m is None else round(self.distance_m, 3),
            "side": self.side,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "pulse_ms": self.pulse_ms,
            "revision": self.revision,
            "observed_age_ms": None if self.observed_age_ms is None else round(self.observed_age_ms, 1),
            "turn_deg": round(self.turn_deg, 2),
            "observation_mode": self.observation_mode,
        }


@dataclass(frozen=True)
class _TargetObservation:
    """同一个目标最近一条经过平滑的可靠视觉观测。"""

    target_id: str
    bearing_deg: float
    distance_m: float | None
    apparent_height: float | None
    apparent_clipped: bool
    revision: int
    observed_at: float


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _spatial_hint(entity: Mapping[str, Any]) -> tuple[float | None, float | None, float | None, bool]:
    """Read a detector-neutral bearing/distance/apparent-size hint.

    Detectors should publish ``attributes.bearing_deg`` plus either
    ``attributes.apparent_height`` (normalized bbox height, the only approach
    metric a 2-D detector can actually measure) or ``attributes.distance_m``
    when a depth adapter supplies real metric depth.  Bboxes and relative
    positions are accepted as conservative fallbacks so a new detector can be
    integrated incrementally.

    Returns ``(bearing_deg, distance_m, apparent_height, apparent_clipped)``.
    """

    attributes = _mapping(entity.get("attributes")) or {}
    bearing = None
    for key in ("bearing_deg", "bearing", "angle_deg"):
        bearing = _finite(attributes.get(key))
        if bearing is not None:
            break
    if bearing is None:
        bearing_rad = _finite(attributes.get("bearing_rad"))
        if bearing_rad is not None:
            bearing = math.degrees(bearing_rad)

    distance = None
    for key in ("distance_m", "distance"):
        distance = _finite(attributes.get(key))
        if distance is not None:
            break

    position = attributes.get("relative_position", attributes.get("position"))
    if isinstance(position, Mapping):
        px = _finite(position.get("x"))
        pz = _finite(position.get("z"))
        if bearing is None and px is not None and pz is not None:
            bearing = math.degrees(math.atan2(px, -pz if abs(pz) > 1e-6 else 1e-6))
        if distance is None and px is not None and pz is not None:
            distance = math.sqrt(px * px + pz * pz)
    elif isinstance(position, (list, tuple)) and len(position) >= 3:
        px = _finite(position[0])
        pz = _finite(position[2])
        if bearing is None and px is not None and pz is not None:
            bearing = math.degrees(math.atan2(px, -pz if abs(pz) > 1e-6 else 1e-6))
        if distance is None and px is not None and pz is not None:
            distance = math.sqrt(px * px + pz * pz)

    if bearing is None:
        bbox = entity.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            left, _top, right, _bottom = (_finite(item) for item in bbox)
            if left is not None and right is not None and right >= left:
                center = (left + right) / 2.0
                # A bbox normally uses normalized coordinates; reject pixels
                # rather than turning wildly from an uncalibrated source.
                if -0.25 <= center <= 1.25:
                    bearing = (center - 0.5) * 90.0

    apparent = _finite(attributes.get("apparent_height"))
    clipped = bool(attributes.get("apparent_height_clipped"))
    if apparent is None:
        # Fall back to the bbox itself so a detector that only publishes boxes
        # can still close the approach loop.  Pixel-scale boxes are rejected
        # rather than guessed at.
        bbox = entity.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            _left, top, _right, bottom = (_finite(item) for item in bbox)
            if top is not None and bottom is not None and bottom >= top and bottom <= 1.25:
                apparent = bottom - top
                if not clipped:
                    # 0.97/0.03: NPC 坐在地上时摄像机在站立视高，bbox 底边实测
                    # 稳定在 0.991，永远够不到 0.999 的阈值。放宽到 0.97 让接近
                    # 到足够近时的夹帧触发 clipped，从而走 _CLIPPED_REACH_FACTOR
                    # 分支提前判定 reached，而不是让 apparent 一路缩到 0 也不停。
                    clipped = top <= 0.03 or bottom >= 0.97
    if apparent is not None and not 0.0 < apparent <= 1.0:
        apparent = None

    if bearing is not None:
        bearing = _clamp(bearing, -180.0, 180.0)
    if distance is not None and (distance < 0.0 or distance > 100.0):
        distance = None
    return bearing, distance, apparent, clipped


class LocalNavigator:
    """A bounded local controller driven by fresh world observations."""

    def __init__(
        self,
        *,
        world_provider: SnapshotProvider,
        goal_provider: SnapshotProvider,
        send_axes: AxisSender,
        send_turn: TurnSender,
        release_inputs: ReleaseSender,
        motion_provider: SnapshotProvider | None = None,
        turn_state_provider: TurnStateProvider | None = None,
        complete_goal: GoalCompleter | None = None,
        direction_memory: "DirectionMemory | None" = None,
        turn_retarget_supported: bool = False,
        config: NavigatorConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or NavigatorConfig()
        self._world_provider = world_provider
        self._goal_provider = goal_provider
        self._send_axes = send_axes
        self._send_turn = send_turn
        self._release_inputs = release_inputs
        # 可选：VRChat 内置 Velocity 参数。没有它时导航器行为与之前完全一致，
        # 只是无法察觉卡墙——这是降级，不是故障。
        self._motion_provider = motion_provider
        # 可选：scheduler 的 heading 状态。没有该回传时仍用本地冷却限速，
        # 保持独立测试和第三方适配器可用。
        self._turn_state_provider = turn_state_provider
        self._complete_goal = complete_goal
        # true 表示发送器会把每次相对修正换算成「基于当前朝向的绝对目标」。这种
        # 发送器可以在上一段平滑转向仍在执行时安全重定向，不会把相对角度累加超调。
        self._turn_retarget_supported = bool(turn_retarget_supported)
        self._clock = clock
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_side: str | None = None
        self._last_decision = NavigationDecision("idle", "not_started")
        self._last_error: str | None = None
        self._tick_count = 0
        self._command_count = 0
        self._stop_count = 0
        self._stall_ticks = 0
        self._stall_count = 0
        self._stalled = False
        self._stall_goal_key: tuple[str, str, str] | None = None
        self._stall_goal_age: float | None = None
        self._completion_notified_goal_key: tuple[str, str, str] | None = None
        # 有限行为导演只保存极小的内存状态，不写逐帧日志：主 LLM 给一次意图，
        # 本地依次完成获取、朝向、接近、停稳和观察。
        self._behavior_phase = "idle"
        self._behavior_phase_started_at: float | None = None
        self._behavior_target_lost_at: float | None = None
        self._behavior_outcome_sequence = 0
        self._behavior_last_outcome: dict[str, Any] | None = None
        self._behavior_outcome_notified: tuple[tuple[str, str, str], str] | None = None
        # wander 只执行 LLM 已经选定的一条短路段，不保存轨迹、不写磁盘，也不
        # 自己挑下一条路线。
        self._wander_turn_sent = False
        self._wander_segment_started_at: float | None = None
        # 每条 wander 只保留一个有界的内存执行摘要。它记录主模型请求、实际提交
        # 给调度器的转向，以及调度器虚拟 HMD 输出朝向的变化；后者仍不是 VRChat
        # 世界实测朝向，终态字段会明确标注来源。不会保存逐帧图片或写硬盘。
        self._wander_trace_sequence = 0
        self._wander_trace: dict[str, Any] | None = None
        # 方向记忆：只记「刚朝哪个方向试过、撞没撞」，短 TTL 自然过期。扇区锚在
        # 调度器虚拟 HMD yaw 上，所以换世界必须 reset()——见 _reset_direction_memory。
        self._direction_memory = direction_memory or DirectionMemory()
        # 每条 wander 轨迹只允许记账一次。movement_stalled 会先经 _record_behavior_outcome
        # 产生终态，紧接着 goal_expired 或下一次失速判定还会再来——重复 record()
        # 会把一次撞墙攒成 confident_block，凭一段轨迹就把方向永久封死。
        #
        # 用 dict 当有序集合：裁剪必须按插入顺序丢最老的。set 没有顺序，
        # list(set)[-32:] 可能正好把刚加进去的 segment_id 丢掉，于是同一次碰撞
        # 会被第二次记账——正是这个守卫要防的事。
        self._direction_recorded_segments: dict[str, None] = {}
        # target_id -> 记录到期的时刻。刻意不随换目标清空：「连续顶着它推摇杆
        # 一动不动」是关于那个实体的实测事实，换一句目标文本不会让墙消失。
        self._unreachable: dict[str, float] = {}
        self._last_motion: dict[str, Any] | None = None
        self._forward_started_at: float | None = None
        self._motion_feedback_usable = False
        self._motion_feedback_state = "idle_not_required"
        # usable 表示这一 tick 有可用于判定的事实；confirmed 表示当前 Avatar 已
        # 证明移动时会回传 X/Z，因此命令期间的沉默也能成为“速度为零”的事实。
        self._motion_feedback_capability_confirmed = False
        # 最近一次前进轴指令有没有真的发出去。None = 还没发过。
        # 由 _apply 写、_stall_guard 读，所以读到的是「上一 tick 那条」——这正是
        # 失速判据要的：这一 tick 的速度反映的就是上一条指令的结果。
        self._axis_send_ok: bool | None = None
        # 斜撞墙：连续多少 tick 前进分量被压住。与 _stall_ticks 分开计，因为
        # 这时人确实在动，纯速度判据永远不会累加。
        self._slip_ticks = 0
        self._slip_count = 0
        # 自动绕行预算。撞墙后先自己转身重试，转够 auto_recover_limit 次仍然
        # 撞墙才闩锁交还 LLM。换目标时随闩锁一起清零。
        self._recover_attempts = 0
        self._recover_count = 0
        # 最近一次绕行往哪边转（+1 右 / -1 左）。滑行时跟着滑行方向走，正面墙
        # 没有方向可用就沿用上次的，避免左右横跳原地打转。
        self._recover_sign = 1.0
        # 已经为哪一次观测发过转向。故意不随目标切换重置：它描述的是「这一帧
        # 世界我已经转过了」，是观测的属性而不是目标的属性。换目标时观测没变，
        # 照样不该再转一次。
        self._last_turn_revision: int | None = None
        self._last_turn_sent_at: float | None = None
        self._turn_suppressed_count = 0
        self._turn_settling_suppressed_count = 0
        self._turn_cooldown_suppressed_count = 0
        self._target_observation: _TargetObservation | None = None
        self._target_grace_count = 0
        self._explorer = ExplorerStateMachine(
            default_max_duration_s=min(90.0, self.config.max_goal_age_s),
            default_forward_axis=min(0.45, self.config.max_forward_axis),
        )

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="neko-local-navigator", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        self._safe_release()

    @property
    def thread_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def tick(self) -> NavigationDecision:
        now = self._clock()
        with self._lock:
            self._tick_count += 1
        try:
            goal_state = self._goal_provider()
            world = self._world_provider()
            goal = _mapping((goal_state or {}).get("goal")) or {}
            goal_kind = str(goal.get("kind") or "").strip().lower()
            with self._lock:
                trace_active = self._wander_trace is not None
            turn_state_before = (
                self._sample_turn_output_state()
                if goal_kind == "wander" or trace_active else None
            )
            self._pre_tick_goal_check(
                goal_state,
                now=now,
                turn_state=turn_state_before,
            )
            self._record_wander_output_heading(turn_state_before)
            base_decision = self._decide(goal_state, world, now)
            directed_decision = self._direct_behavior(base_decision, goal_state, now)
            decision = self._stall_guard(directed_decision, goal_state, now)
            applied = self._apply(decision, now)
            self._explorer.record_applied(decision.reason, applied)
            if applied:
                with self._lock:
                    if decision.reason == "wander_llm_turn":
                        self._wander_turn_sent = True
                    elif decision.reason == "wander_forward" and self._wander_segment_started_at is None:
                        self._wander_segment_started_at = now
                    elif decision.state == "recover" and goal_kind == "wander":
                        # 绕墙转向已经改变了路线；从转向完成后重新给这一段完整预算，
                        # 避免刚绕开墙就被旧路段计时器立刻要求再转一次。
                        self._wander_segment_started_at = now
            turn_state_after = (
                self._sample_turn_output_state()
                if goal_kind == "wander" or trace_active else None
            )
            self._record_wander_execution(
                decision,
                applied,
                now,
                goal_kind=goal_kind,
                turn_state=turn_state_after,
            )
            with self._lock:
                self._last_error = None
                self._last_decision = decision
            self._record_behavior_outcome(decision, now)
            self._notify_goal_complete(decision)
            return decision
        except Exception as exc:
            self._safe_release()
            decision = NavigationDecision("degraded", f"navigator_error:{type(exc).__name__}")
            with self._lock:
                self._last_error = str(exc)[:240]
                self._last_decision = decision
            return decision

    def _pre_tick_goal_check(
        self,
        goal_state: Mapping[str, Any] | None,
        *,
        now: float | None = None,
        turn_state: Mapping[str, Any] | None = None,
    ) -> None:
        """在 _decide 之前处理目标切换，保证 _decide 拿到的 skip_ids 已经反映了
        当前目标的新鲜度。

        分离到这里的原因：_decide 的求值顺序早于 _stall_guard（Python 先对内层
        表达式求值再传给外层），所以如果目标清单的更新只在 _stall_guard 里做，
        _decide 这一 tick 永远拿到的是上一 tick 的过期 skip_ids。把更新提前到
        _decide 之前就可以消除这个一拍延迟。
        """
        goal = _mapping((goal_state or {}).get("goal")) or {}
        goal_key = (
            str(goal.get("kind") or ""),
            str(goal.get("text") or ""),
            str(goal.get("target_id") or ""),
        )
        goal_age = _finite(goal.get("age_seconds"))
        with self._lock:
            previous_key = self._stall_goal_key
            previous_age = self._stall_goal_age
            renewed = goal_key != previous_key or (
                goal_age is not None and previous_age is not None and goal_age < previous_age
            )
            if renewed:
                self._stall_ticks = 0
                self._stalled = False
                self._slip_ticks = 0
                # 绕行预算随闩锁一起归零：LLM 换了目标就是新的一次尝试，不该
                # 背着上一个目标用掉的次数。
                self._recover_attempts = 0
                # 绕行方向在一个路段内锁定，但不跨路段：新目标面对的地形不同，
                # 应重新由第一次滑行取样决定，而不是继承上一段的偏向。
                self._recover_sign = 1.0
                self._completion_notified_goal_key = None
                self._behavior_target_lost_at = None
                self._behavior_outcome_notified = None
                next_kind = goal_key[0].strip().lower()
                self._behavior_phase = "acquire" if next_kind == "approach_observe" else "idle"
                self._behavior_phase_started_at = None
                constraints = _mapping(goal.get("constraints")) or {}
                turn_deg = _finite(constraints.get("turn_deg"))
                # turn_deg=0 明确表示 LLM 选择直行，不需要伪造一条零度转向命令。
                self._wander_turn_sent = bool(turn_deg is not None and abs(turn_deg) <= 0.5)
                self._wander_segment_started_at = None
                if next_kind == "wander":
                    self._start_wander_trace_locked(
                        goal,
                        self._clock() if now is None else now,
                        turn_state,
                    )
                else:
                    self._wander_trace = None
                # 目标文本真的变了才清空拉黑列表，给每个实体一次重试机会。
                # 同一句目标重提（age 倒退）= 「再试一次」，地形没变，不重置；
                # 镜面倒影之类的会在新目标文本后立刻再次触发失速并重新记账。
                if goal_key != previous_key:
                    self._unreachable.clear()
                    self._target_observation = None
            self._stall_goal_key = goal_key
            self._stall_goal_age = goal_age

    @staticmethod
    def _turn_direction_by_contract(turn_deg: float) -> str:
        """按 wander 对外契约解释方向；这不是实测世界旋转方向。"""
        if turn_deg > 0.5:
            return "left"
        if turn_deg < -0.5:
            return "right"
        return "straight"

    def _sample_turn_output_state(self) -> dict[str, Any] | None:
        """读取调度器虚拟 HMD 朝向；读取失败不影响导航，只让摘要降级。"""
        provider = self._turn_state_provider
        if provider is None:
            return None
        try:
            state = provider()
        except Exception:
            return None
        if not isinstance(state, Mapping) or state.get("available") is False:
            return None
        yaw = _finite(state.get("yaw_deg"))
        if yaw is None:
            return None
        commands = _finite(state.get("turn_commands"))
        return {
            "yaw_deg": yaw % 360.0,
            "turning": bool(state.get("turning")),
            "turn_commands": None if commands is None else int(commands),
        }

    def _free_roam_params(
        self,
        goal: Mapping[str, Any],
        kind: str,
    ) -> dict[str, Any]:
        """解析一次自由移动的请求参数。

        执行侧和 wander 终态摘要都只能走这里。两边各自夹取过一次，任何一侧改了
        上限都会让回报给主 LLM 的 requested 偏离实际推的摇杆，而这种偏差恰好是
        execution_summary 想消除的东西。

        ``duration`` 对 depart 保留原始请求值，因为 depart 用它和 age_seconds
        直接比较判完成；wander 的时长在这里就夹进 [1, _WANDER_MAX_STEP_S]。
        """
        constraints = _mapping(goal.get("constraints")) or {}
        is_depart = kind == "depart"
        duration = _finite(constraints.get("max_duration_s"))
        if duration is None:
            duration = 2.0 if is_depart else _WANDER_DEFAULT_STEP_S
        if not is_depart:
            duration = min(_WANDER_MAX_STEP_S, max(1.0, duration))
        max_axis = _finite(constraints.get("max_forward_axis"))
        axis = min(
            self.config.max_forward_axis,
            _DEPART_MAX_FORWARD_AXIS if is_depart else _WANDER_MAX_FORWARD_AXIS,
            self.config.max_forward_axis if max_axis is None else max_axis,
        )
        return {
            "turn_deg": _finite(constraints.get("turn_deg")),
            "duration_s": duration,
            "forward_axis": axis,
        }

    def _start_wander_trace_locked(
        self,
        goal: Mapping[str, Any],
        now: float,
        turn_state: Mapping[str, Any] | None,
    ) -> None:
        """开始一条有界内存轨迹；调用方必须持有 ``_lock``。"""
        params = self._free_roam_params(goal, "wander")
        requested_turn = params["turn_deg"] or 0.0
        requested_duration = params["duration_s"]
        requested_axis = params["forward_axis"]
        yaw = _finite((turn_state or {}).get("yaw_deg"))
        commands = _finite((turn_state or {}).get("turn_commands"))
        self._wander_trace_sequence += 1
        self._wander_trace = {
            "segment_id": f"wander:{self._wander_trace_sequence}",
            "started_at_monotonic": now,
            "requested_turn_deg": requested_turn,
            "requested_duration_s": requested_duration,
            "requested_forward_axis": requested_axis,
            "initial_turn_required": abs(requested_turn) > 0.5,
            "initial_turn_submitted": abs(requested_turn) <= 0.5,
            "submitted_turn_delta_deg": 0.0,
            "recoveries": [],
            "segment_timer_reset_count": 0,
            "forward_command_count": 0,
            "output_heading_delta_deg": 0.0,
            "output_heading_last_yaw_deg": None if yaw is None else yaw % 360.0,
            "output_heading_sample_count": 0 if yaw is None else 1,
            # 方向记忆的锚点：这一段**起始**时的虚拟 HMD yaw。绕行会改变朝向，
            # 用终点 yaw 反算会把证据记到隔壁扇区，所以必须在这里就固定下来。
            "anchor_yaw_deg": yaw,
            "turn_commands_start": None if commands is None else int(commands),
            "turn_commands_end": None if commands is None else int(commands),
            "motion_feedback_observed": False,
            "wall_slide_detected": False,
            "stall_detected": False,
            "speed_sample_count": 0,
            "last_horizontal_speed_mps": None,
            "last_forward_ratio": None,
            # 门控积分的输入。每拍一条，上限 240（4Hz 下约 60 秒），够覆盖一段
            # wander 又不会无界增长。
            "progress_samples": [],
            "progress_last_sample_at": None,
            # 只允许本段真正发送成功之后的速度样本进入里程积分。否则 body
            # 未 enable 时，上一段缓存的速度也可能被误算成当前段的推进。
            "forward_command_started_at": None,
            "last_angular_age_ms": None,
        }

    def _record_wander_output_heading(
        self,
        turn_state: Mapping[str, Any] | None,
    ) -> None:
        """累计调度器输出朝向变化；用最短圆周差处理 0/360 度回绕。"""
        if not isinstance(turn_state, Mapping):
            return
        yaw = _finite(turn_state.get("yaw_deg"))
        commands = _finite(turn_state.get("turn_commands"))
        if yaw is None:
            return
        yaw %= 360.0
        with self._lock:
            trace = self._wander_trace
            if trace is None:
                return
            previous = _finite(trace.get("output_heading_last_yaw_deg"))
            if previous is not None:
                delta = (yaw - previous + 180.0) % 360.0 - 180.0
                trace["output_heading_delta_deg"] = (
                    float(trace.get("output_heading_delta_deg", 0.0)) + delta
                )
            trace["output_heading_last_yaw_deg"] = yaw
            trace["output_heading_sample_count"] = int(
                trace.get("output_heading_sample_count", 0)
            ) + 1
            if commands is not None:
                trace["turn_commands_end"] = int(commands)

    def _record_wander_execution(
        self,
        decision: NavigationDecision,
        applied: bool,
        now: float,
        *,
        goal_kind: str,
        turn_state: Mapping[str, Any] | None,
    ) -> None:
        """记录本段命令与恢复事件；所有列表都有固定上限且只驻留内存。"""
        self._record_wander_output_heading(turn_state)
        with self._lock:
            trace = self._wander_trace
            if trace is None or goal_kind != "wander":
                return
            if decision.reason == "wander_llm_turn":
                trace["initial_turn_submitted"] = bool(applied)
                if applied:
                    trace["submitted_turn_delta_deg"] = (
                        float(trace.get("submitted_turn_delta_deg", 0.0))
                        + float(decision.turn_deg)
                    )
            elif decision.state == "recover":
                recoveries = trace.get("recoveries")
                if not isinstance(recoveries, list):
                    recoveries = []
                    trace["recoveries"] = recoveries
                trigger = (
                    "wall_slide"
                    if decision.reason.startswith("auto_recover_slide")
                    else "forward_stall"
                )
                if len(recoveries) < 20:
                    recoveries.append({
                        "sequence": len(recoveries) + 1,
                        "trigger": trigger,
                        "turn_deg": round(float(decision.turn_deg), 2),
                        "direction_by_contract": self._turn_direction_by_contract(
                            float(decision.turn_deg)
                        ),
                        "turn_submitted": bool(applied),
                        "backed_up": bool(decision.y < -1e-6),
                    })
                if trigger == "wall_slide":
                    trace["wall_slide_detected"] = True
                else:
                    trace["stall_detected"] = True
                if applied:
                    trace["submitted_turn_delta_deg"] = (
                        float(trace.get("submitted_turn_delta_deg", 0.0))
                        + float(decision.turn_deg)
                    )
                    trace["segment_timer_reset_count"] = int(
                        trace.get("segment_timer_reset_count", 0)
                    ) + 1
                    # recover 会改变朝向；下一次前进要从新的命令起点重新验收速度，
                    # 不能把绕行前缓存的速度当成绕行后的推进。
                    trace["forward_command_started_at"] = None
                    trace["progress_last_sample_at"] = now
            elif decision.reason == "wander_forward" and applied:
                trace["forward_command_count"] = int(
                    trace.get("forward_command_count", 0)
                ) + 1
                if trace.get("forward_command_started_at") is None:
                    trace["forward_command_started_at"] = now

            motion = self._last_motion
            if (
                isinstance(motion, Mapping)
                and decision.state in {"advance", "recover"}
                and applied
            ):
                # ``motion_feedback`` 返回的是带年龄的最新缓存值；本段刚开始时
                # 这里很可能仍是上一段的包。它不能证明本段已移动，也不能进入
                # progress_samples。没有年龄字段的第三方 provider 保持兼容，
                # 交给原有的 evidence/progress 门槛处理。
                sample_age_ms = _finite(motion.get("value_age_ms"))
                segment_start = _finite(trace.get("forward_command_started_at"))
                if (
                    sample_age_ms is not None
                    and segment_start is not None
                    and now - sample_age_ms / 1000.0 < segment_start - 0.01
                ):
                    return
                if motion.get("available"):
                    trace["motion_feedback_observed"] = True
                speed = _finite(motion.get("horizontal_speed_mps"))
                ratio = _finite(motion.get("forward_ratio"))
                if speed is not None:
                    trace["speed_sample_count"] = int(
                        trace.get("speed_sample_count", 0)
                    ) + 1
                    trace["last_horizontal_speed_mps"] = speed
                if ratio is not None:
                    trace["last_forward_ratio"] = ratio
                self._append_progress_sample_locked(
                    trace,
                    now,
                    motion=motion,
                    speed=speed,
                    ratio=ratio,
                    turning=bool(decision.state == "recover"),
                )
            if decision.reason == "movement_stalled":
                trace["stall_detected"] = True

    def _append_progress_sample_locked(
        self,
        trace: dict[str, Any],
        now: float,
        *,
        motion: Mapping[str, Any],
        speed: float | None,
        ratio: float | None,
        turning: bool,
    ) -> None:
        """给门控积分攒一拍样本；调用方必须持有 ``_lock``。

        ``dt`` 用相邻两拍的实际间隔，不是标称 tick 周期：导航循环会被视觉和锁拖慢，
        用标称值积出来的距离会系统性偏大，而这个距离正是 verified_free 的门槛。

        ``turned`` 有两个来源，任一为真就丢弃这拍：绕行决策（我们自己发的转向），
        以及新鲜的 ``AngularY``（VRChat 说人在转）。角速度必须看年龄——它和
        VelocityX/Z 各有各的回传时机，旧包会把整段都标成「一直在转」。
        """
        samples = trace.get("progress_samples")
        if not isinstance(samples, list):
            samples = []
            trace["progress_samples"] = samples
        previous_at = _finite(trace.get("progress_last_sample_at"))
        trace["progress_last_sample_at"] = now
        if previous_at is None:
            # 第一拍没有区间可积。只记时间戳，让下一拍有 dt 可用。
            return
        dt = max(0.0, now - previous_at)
        angular_age_ms = _finite(motion.get("angular_age_ms"))
        angular_speed = _finite(motion.get("angular_speed"))
        trace["last_angular_age_ms"] = angular_age_ms
        fresh_angular = (
            angular_speed is not None
            and abs(angular_speed) > _ANGULAR_RESTING
            and angular_age_ms is not None
            and angular_age_ms <= _ANGULAR_FRESH_MS
        )
        if len(samples) < 240:
            samples.append({
                "speed": speed,
                "forward_ratio": ratio,
                "dt": dt,
                "turned": bool(turning or fresh_angular),
            })

    def _wander_execution_summary_locked(
        self,
        decision: NavigationDecision,
        now: float,
    ) -> dict[str, Any] | None:
        """生成供主 LLM 消费的紧凑终态；调用方必须持有 ``_lock``。"""
        trace = self._wander_trace
        if trace is None:
            return None
        requested_turn = float(trace.get("requested_turn_deg", 0.0))
        submitted_turn = float(trace.get("submitted_turn_delta_deg", 0.0))
        recoveries = [
            dict(item)
            for item in (trace.get("recoveries") or ())[:20]
            if isinstance(item, Mapping)
        ]
        submitted_recoveries = sum(
            1 for item in recoveries if item.get("turn_submitted") is True
        )
        if decision.reason == "wander_step_complete":
            completion = (
                "completed_after_recovery" if submitted_recoveries else "completed_clean"
            )
        elif decision.reason == "movement_stalled":
            completion = (
                "blocked_after_recovery" if submitted_recoveries else "blocked"
            )
        elif decision.reason == "goal_expired":
            completion = "expired"
        else:
            completion = "stopped"
        start = _finite(trace.get("started_at_monotonic"))
        if start is None:
            start = now
        final_leg_ms = (
            None
            if self._wander_segment_started_at is None
            else round(max(0.0, now - self._wander_segment_started_at) * 1000.0, 1)
        )
        heading_samples = int(trace.get("output_heading_sample_count", 0))
        heading_delta = (
            round(float(trace.get("output_heading_delta_deg", 0.0)), 2)
            if heading_samples >= 2 else None
        )
        start_commands = trace.get("turn_commands_start")
        end_commands = trace.get("turn_commands_end")
        command_count_delta = (
            int(end_commands) - int(start_commands)
            if isinstance(start_commands, int) and isinstance(end_commands, int)
            else None
        )
        return {
            "segment_id": str(trace.get("segment_id") or "wander:unknown")[:64],
            "completion": completion,
            "requested": {
                "turn_deg": round(requested_turn, 2),
                "duration_s": round(float(trace.get("requested_duration_s", 0.0)), 2),
                "forward_axis": round(float(trace.get("requested_forward_axis", 0.0)), 3),
            },
            "execution": {
                # submitted 是调用方返回成功的命令累计，不保证 VRChat 已呈现同样角度。
                "submitted_turn_delta_deg": round(submitted_turn, 2),
                "submitted_deviation_from_request_deg": round(
                    submitted_turn - requested_turn, 2
                ),
                # output 来自调度器虚拟 HMD 内部 yaw，比 submit 更接近输出，但仍不是
                # VRChat 世界回传；字段名和 verified 标志禁止上层把它当作实测真值。
                "output_heading_delta_deg": heading_delta,
                "output_heading_source": "scheduler_virtual_hmd",
                "world_observation_verified": False,
                "turn_command_count_delta": command_count_delta,
                "initial_turn_required": bool(trace.get("initial_turn_required")),
                "initial_turn_submitted": bool(trace.get("initial_turn_submitted")),
                "total_elapsed_ms": round(max(0.0, now - start) * 1000.0, 1),
                "final_leg_elapsed_ms": final_leg_ms,
                "segment_timer_reset_count": int(
                    trace.get("segment_timer_reset_count", 0)
                ),
                "forward_command_count": int(trace.get("forward_command_count", 0)),
            },
            "recoveries": recoveries,
            "direction_memory": self._record_direction_memory_locked(
                trace,
                completion=completion,
                now=now,
            ),
            "motion_evidence": {
                "velocity_feedback_observed": bool(
                    trace.get("motion_feedback_observed")
                ),
                "wall_slide_detected": bool(trace.get("wall_slide_detected")),
                "stall_detected": bool(trace.get("stall_detected")),
                "speed_sample_count": int(trace.get("speed_sample_count", 0)),
                "last_horizontal_speed_mps": (
                    None
                    if _finite(trace.get("last_horizontal_speed_mps")) is None
                    else round(float(trace["last_horizontal_speed_mps"]), 4)
                ),
                "last_forward_ratio": (
                    None
                    if _finite(trace.get("last_forward_ratio")) is None
                    else round(float(trace["last_forward_ratio"]), 4)
                ),
                # 门控积分出的推进距离下界：转向拍和贴墙拍都已丢弃。它是
                # verified_free 的门槛，所以必须和摘要一起回报，让主 LLM 能自己
                # 判断「记成走通」这个结论有多少证据支撑。
                "gated_progress_m": round(
                    integrate_progress(trace.get("progress_samples") or ())[0], 2
                ),
                "progress_sample_count": len(trace.get("progress_samples") or ()),
                "angular_age_ms": _finite(trace.get("last_angular_age_ms")),
            },
            "turn_sign_convention": "positive_left_negative_right",
        }

    def _record_direction_memory_locked(
        self,
        trace: Mapping[str, Any],
        *,
        completion: str,
        now: float,
    ) -> dict[str, Any]:
        """把这一段的实测结果记进方向记忆，并回传给主 LLM 的建议摘要。

        调用方必须持有 ``_lock``。每个 ``segment_id`` 只记一次：``movement_stalled``
        产生终态后，``goal_expired`` 或下一 tick 的失速判定还会再进来一次，重复
        record() 会把一次撞墙攒成 ``confident_block``，凭一段轨迹就把整个扇区封死。

        bearing 用 LLM 请求的 ``turn_deg``，不是绕行后的实际朝向：被记住、之后可能
        被拒绝的是「主模型选的那个方向」。绕行净转角另存在 recoveries 里。锚点用
        这一段**起始**时的虚拟 HMD yaw，绕行改变的朝向不该把证据挪到隔壁扇区。
        """
        segment_id = str(trace.get("segment_id") or "wander:unknown")[:64]
        bearing = float(trace.get("requested_turn_deg", 0.0))
        anchor = _finite(trace.get("anchor_yaw_deg"))
        already_recorded = segment_id in self._direction_recorded_segments
        progress_m, turned = integrate_progress(
            trace.get("progress_samples") or ()
        )
        if not already_recorded:
            blocked = completion in ("blocked", "blocked_after_recovery") or bool(
                trace.get("stall_detected")
            ) or bool(trace.get("wall_slide_detected"))
            # 速度证据只认真的取到过样本。velocity_feedback_observed 为假时，
            # 「没撞上」和「压根没动」无法区分，只能记 unknown。
            #
            # 但有样本也不等于走通了：顶着墙时 VRChat 照样回传速度包，值可能就是
            # 0.0。真正的门槛是 progress_m——SegmentOutcome.cleared 会检查它。
            evidence = int(trace.get("speed_sample_count", 0)) > 0 and bool(
                trace.get("motion_feedback_observed")
            )
            self._direction_memory.record(
                SegmentOutcome(
                    bearing_deg=bearing,
                    blocked=blocked,
                    heading_deg=0.0 if anchor is None else anchor,
                    progress_m=progress_m,
                    turned=turned,
                    evidence_available=evidence,
                ),
                now,
            )
            self._direction_recorded_segments[segment_id] = None
            while len(self._direction_recorded_segments) > 64:
                # dict 保序，popitem(last=False) 的等价写法：丢最早插入的那条。
                # 顺序在这里是正确性问题，不是整洁问题——丢掉刚加进去的 id 就等于
                # 允许同一次碰撞被第二次记账。
                self._direction_recorded_segments.pop(
                    next(iter(self._direction_recorded_segments))
                )
        # 记录必须用路段起始锚点，否则 recover 改变朝向后会把这段证据记到
        # 邻近扇区；但给主 LLM 的终态摘要必须按**当前**虚拟 yaw 汇报，才能让
        # 下一次 turn_deg 直接复用这些相对角度。两者不能共用一个 heading。
        record_heading = 0.0 if anchor is None else anchor
        report_heading = self._current_heading_deg()
        advice = self._direction_memory.advice(now, heading_deg=report_heading)
        advice["recorded_bearing_deg"] = round(bearing, 2)
        advice["recorded_state"] = self._direction_memory.state_of(
            bearing, now, heading_deg=record_heading
        )
        # 同一段被重复问到时明确标出来，避免上层把它当成第二次独立实测。
        advice["recorded_this_segment"] = not already_recorded
        advice["recorded_progress_m"] = round(progress_m, 2)
        advice["recorded_turned_during_segment"] = turned
        advice["heading_anchor_deg"] = None if anchor is None else round(anchor, 2)
        advice["heading_report_deg"] = round(report_heading, 2)
        return advice

    def _current_heading_deg(self) -> float:
        """当前虚拟 HMD yaw，读不到时退回 0。

        退回 0 会让记忆退化成「相对会话起点」，仍然自相一致（写入和查询用同一个
        锚点），只是转身之后精度下降。这比让方向记忆整体失效要好。
        """
        state = self._sample_turn_output_state()
        if not isinstance(state, Mapping):
            return 0.0
        yaw = _finite(state.get("yaw_deg"))
        return 0.0 if yaw is None else yaw

    def direction_advice(self) -> dict[str, Any]:
        """当前方向记忆摘要，供宿主在非终态场合查询。"""
        return self._direction_memory.advice(
            self._clock(), heading_deg=self._current_heading_deg()
        )

    def update_direction_scores(
        self, scores: Mapping[Any, Any]
    ) -> dict[str, Any]:
        """记录主 LLM 对当前画面方向的偏好；绝不在这里选择执行方向。"""
        heading = self._current_heading_deg()
        now = self._clock()
        with self._lock:
            # 新一帧的偏好替换旧预测，但保留 empirical_state 和撞墙历史。
            self._direction_memory.clear_predictions()
            self._direction_memory.predict(scores, now, heading_deg=heading)
            advice = self._direction_memory.advice(now, heading_deg=heading)
        advice["prediction_heading_deg"] = round(heading, 2)
        return advice

    def should_refuse_bearing(self, bearing_deg: float) -> bool:
        """这个方向是否已确认封死。真值表示应当拒绝并把选择权还给主 LLM。

        ``bearing_deg`` 是相对**当前**朝向的转角，和 wander 的 turn_deg 同源；这里
        用当前 yaw 换算到记忆系，所以转身之后同一个 +25° 不会再命中旧扇区。
        """
        return self._direction_memory.should_refuse(
            bearing_deg, self._clock(), heading_deg=self._current_heading_deg()
        )

    def reset_direction_memory(self, reason: str = "world_changed") -> None:
        """换世界时清空方向记忆。

        锚点是虚拟 HMD yaw，只在一个世界里连续。换了世界之后所有扇区指向的都是
        上个世界里的墙，留着只会拒绝本来能走的方向。
        """
        with self._lock:
            self._direction_memory.reset()
            self._direction_recorded_segments.clear()

    def _notify_goal_complete(self, decision: NavigationDecision) -> None:
        """有限行为完成或到期后只通知一次，并释放上层目标。"""
        terminal_reasons = {
            "approach_observe_complete",
            "approach_observe_target_lost",
            "explore_duration_exhausted",
            "depart_complete",
            "wander_step_complete",
            "goal_expired",
        }
        if decision.reason not in terminal_reasons or self._complete_goal is None:
            return
        with self._lock:
            goal_key = self._stall_goal_key
            if goal_key is None or goal_key == self._completion_notified_goal_key:
                return
            self._completion_notified_goal_key = goal_key
        try:
            self._complete_goal(decision.reason)
        except Exception as exc:
            # 完成回调失败不能把已经停车的决策改成导航器故障；状态里保留诊断。
            with self._lock:
                self._last_error = f"goal_complete_callback:{type(exc).__name__}: {exc}"[:240]

    @staticmethod
    def _stationary_decision(
        source: NavigationDecision,
        state: str,
        reason: str,
    ) -> NavigationDecision:
        """保留产生决定的观测证据，但明确清零所有控制器输出。"""
        return NavigationDecision(
            state,
            reason,
            source.target_id,
            source.bearing_deg,
            source.distance_m,
            revision=source.revision,
            observed_age_ms=source.observed_age_ms,
            observation_mode=source.observation_mode,
        )

    def _set_behavior_phase(self, phase: str, now: float) -> None:
        with self._lock:
            if self._behavior_phase != phase or self._behavior_phase_started_at is None:
                self._behavior_phase = phase
                self._behavior_phase_started_at = now

    @staticmethod
    def _behavior_seconds(
        goal: Mapping[str, Any],
        name: str,
        default: float,
    ) -> float:
        constraints = _mapping(goal.get("constraints")) or {}
        value = _finite(constraints.get(name))
        return default if value is None else value

    def _direct_behavior(
        self,
        decision: NavigationDecision,
        goal_state: Mapping[str, Any] | None,
        now: float,
    ) -> NavigationDecision:
        """把一次高层意图导演成有限的本地行为，不把逐步决策退回给 LLM。"""
        goal = _mapping((goal_state or {}).get("goal")) or {}
        if str(goal.get("kind") or "").strip().lower() != "approach_observe":
            return decision

        missing_reasons = {
            "target_not_visible", "target_low_confidence", "target_bearing_unknown",
        }
        if decision.reason in missing_reasons:
            with self._lock:
                if self._behavior_target_lost_at is None:
                    self._behavior_target_lost_at = now
                lost_for = now - self._behavior_target_lost_at
            self._set_behavior_phase("acquire", now)
            if lost_for < self.config.behavior_reacquire_s:
                return self._stationary_decision(
                    decision, "acquire", "behavior_reacquiring_target"
                )
            self._set_behavior_phase("failed", now)
            return self._stationary_decision(
                decision, "complete", "approach_observe_target_lost"
            )

        # 只要重新拿到目标观测，就取消丢失计时；世界未知等安全停车不伪装成
        # “目标丢失”，仍沿用原决定等待感知恢复。
        if decision.target_id:
            with self._lock:
                self._behavior_target_lost_at = None

        if decision.state == "turn":
            self._set_behavior_phase("orient", now)
            return decision
        if decision.state == "advance":
            self._set_behavior_phase("approach", now)
            return decision
        if decision.state != "reached":
            return decision

        with self._lock:
            phase = self._behavior_phase
            phase_started_at = self._behavior_phase_started_at
        if phase == "complete":
            return self._stationary_decision(
                decision, "complete", "approach_observe_complete"
            )
        if phase not in {"settle", "observe"}:
            self._set_behavior_phase("settle", now)
            phase = "settle"
            phase_started_at = now

        if phase == "settle":
            settle_s = self._behavior_seconds(
                goal, "settle_seconds", self.config.behavior_settle_s
            )
            if phase_started_at is None or now - phase_started_at < settle_s:
                return self._stationary_decision(decision, "settle", "behavior_settling")
            self._set_behavior_phase("observe", now)
            phase_started_at = now

        observe_s = self._behavior_seconds(
            goal, "observe_seconds", self.config.behavior_observe_s
        )
        if phase_started_at is None or now - phase_started_at < observe_s:
            return self._stationary_decision(decision, "observe", "behavior_observing")
        self._set_behavior_phase("complete", now)
        return self._stationary_decision(
            decision, "complete", "approach_observe_complete"
        )

    def _record_behavior_outcome(self, decision: NavigationDecision, now: float) -> None:
        """只记录离散结果，供宿主低频通知；不会把逐帧运动写入硬盘或 LLM。"""
        outcome_reasons = {
            "approach_observe_complete",
            "approach_observe_target_lost",
            "depart_complete",
            "wander_step_complete",
            "movement_stalled",
            "target_unreachable",
            "goal_expired",
        }
        if decision.reason not in outcome_reasons:
            return
        with self._lock:
            goal_key = self._stall_goal_key
            if goal_key is None:
                return
            outcome_key = (goal_key, decision.reason)
            if outcome_key == self._behavior_outcome_notified:
                return
            self._behavior_outcome_notified = outcome_key
            self._behavior_outcome_sequence += 1
            outcome = {
                "sequence": self._behavior_outcome_sequence,
                "reason": decision.reason,
                "state": decision.state,
                "target_id": decision.target_id,
                "revision": decision.revision,
                "observed_age_ms": decision.observed_age_ms,
                "occurred_at_monotonic": now,
            }
            if goal_key[0].strip().lower() == "wander":
                execution_summary = self._wander_execution_summary_locked(decision, now)
                if execution_summary is not None:
                    outcome["execution_summary"] = execution_summary
            self._behavior_last_outcome = outcome

    def _stall_guard(
        self,
        decision: NavigationDecision,
        goal_state: Mapping[str, Any] | None,
        now: float,
    ) -> NavigationDecision:
        """把「命令发了但人没动」变成一次显式停车。

        检测器只看画面，永远不会报告「前面有堵墙」；VRChat 内置 Velocity 参数是
        唯一能区分「正在前进」和「顶着墙推摇杆」的回传。当前 Avatar 从未证明
        支持这个回传时，本守卫整体失效并放行；能力已经确认后，前进期间超过
        起步宽限的沉默则正是“没有移动”的信号。

        判定会闩锁：停下之后速度当然还是 0，靠速度自己是解不开的。闩锁由
        _pre_tick_goal_check 在 _decide 之前处理；本方法只读取结果。
        """
        with self._lock:
            stalled = self._stalled

        goal = _mapping((goal_state or {}).get("goal")) or {}
        goal_kind = str(goal.get("kind") or "").strip().lower()
        slip_forward_ratio = (
            max(self.config.slip_forward_ratio, _WANDER_SLIP_FORWARD_RATIO)
            if goal_kind == "wander"
            else self.config.slip_forward_ratio
        )
        slip_threshold_ticks = (
            min(self.config.slip_ticks, _WANDER_SLIP_TICKS)
            if goal_kind == "wander"
            else self.config.slip_ticks
        )

        if decision.state != "advance":
            # 转身和停车都不算尝试前进；不清零计数，以免「转一下再撞」反复重置。
            # 但有一个例外：如果已经闩锁了，此时 _decide 看到 skip_ids 没有候选
            # 就会返回 target_unreachable，_stall_guard 如果只过 advance 就永远
            # 输出 target_unreachable 而不是 movement_stalled，让上游 LLM 以为是
            # 「视野里没有人」而不是「顶着墙卡住了」。
            if stalled and decision.reason == "target_unreachable":
                return NavigationDecision(
                    "stop",
                    "movement_stalled",
                    decision.target_id,
                    decision.bearing_deg,
                    decision.distance_m,
                    revision=decision.revision,
                    observed_age_ms=decision.observed_age_ms,
                )
            return decision
        if stalled:
            return NavigationDecision(
                "stop",
                "movement_stalled",
                decision.target_id,
                decision.bearing_deg,
                decision.distance_m,
                revision=decision.revision,
                observed_age_ms=decision.observed_age_ms,
            )

        with self._lock:
            axis_send_ok = self._axis_send_ok
        if axis_send_ok is False:
            # 上一条前进指令被下游拒收（scheduler 拒收，多半是 body output 还没
            # enable）。速度当然是 0，但那是「没发命令」而不是「顶着墙」。
            # 重置计数而不是累加，确保 body enable 后立刻从头开始计。
            # axis_send_ok is None（从没发过）时正常走速度判据——这是首次。
            with self._lock:
                self._stall_ticks = 0
                self._motion_feedback_usable = False
                self._motion_feedback_state = "command_rejected"
            return decision

        motion = self._sample_motion()
        motion_available = bool(motion is not None and motion.get("available"))
        explicit_capability = (
            motion.get("horizontal_feedback_confirmed")
            if isinstance(motion, Mapping)
            else None
        )
        with self._lock:
            if isinstance(explicit_capability, bool):
                # OSC 桥在 Avatar 切换时清空参数并显式返回 false，旧能力不能继承。
                self._motion_feedback_capability_confirmed = explicit_capability
            elif motion_available:
                # 兼容没有新字段的第三方 provider：给过可用水平速度就证明支持。
                self._motion_feedback_capability_confirmed = True
            capability_confirmed = self._motion_feedback_capability_confirmed

        silence_as_zero = False
        if not motion_available:
            with self._lock:
                started_at = self._forward_started_at
                self._motion_feedback_usable = False
                if started_at is None:
                    self._motion_feedback_state = "awaiting_first_command"
                elif now - started_at < self.config.motion_start_grace_s:
                    self._motion_feedback_state = "awaiting_motion"
                elif (
                    capability_confirmed
                    and isinstance(motion, Mapping)
                    and motion.get("reason") == "velocity_feedback_quiet"
                ):
                    # 当前 Avatar 已证明“移动时回传、静止时沉默”。命令已过起步
                    # 宽限仍无新 X/Z，等价于本次速度为零，可累计正面失速。
                    self._motion_feedback_usable = True
                    self._motion_feedback_state = "confirmed_silence_zero"
                    silence_as_zero = True
                else:
                    # 当前 Avatar 从未证明支持 X/Z；没数据仍然不等于没动。
                    self._motion_feedback_state = "unavailable_while_commanding"
            if not silence_as_zero:
                return decision

        if silence_as_zero:
            speed = 0.0
        else:
            assert motion is not None
            value_age_ms = _finite(motion.get("value_age_ms"))
            with self._lock:
                started_at = self._forward_started_at
            if value_age_ms is not None:
                if started_at is None:
                    # _stall_guard 在 _apply 之前运行；第一 tick 看到的速度必然属于旧动作。
                    with self._lock:
                        self._motion_feedback_usable = False
                        self._motion_feedback_state = "awaiting_first_command"
                    return decision
                command_age_ms = max(0.0, now - started_at) * 1000.0
                # 允许 5ms 的收包/时钟舍入误差。比本次命令年龄更老的样本来自静止前
                # 或上一次移动，绝不能拿来证明这条命令正在生效。
                if value_age_ms > command_age_ms + 5.0:
                    with self._lock:
                        self._motion_feedback_usable = False
                        self._motion_feedback_state = "sample_predates_command"
                    return decision
                if now - started_at < self.config.motion_start_grace_s:
                    with self._lock:
                        self._motion_feedback_usable = False
                        self._motion_feedback_state = "starting"
                    return decision
            speed = _finite(motion.get("horizontal_speed_mps"))
            if speed is None:
                with self._lock:
                    self._motion_feedback_usable = False
                    self._motion_feedback_state = "invalid_motion_sample"
                return decision
            with self._lock:
                self._motion_feedback_usable = True
                self._motion_feedback_state = "active"

        # 斜撞墙：人在动，但动的方向不是前方。角色控制器把移动投影到墙面上，
        # 所以速度模长过得了 stall_speed_mps，纯速度判据永远不会触发，会一直
        # 贴着墙滑下去。forward_ratio 是 None 表示水平速度低于静止阈值，那属于
        # 下面的失速判据管的范围，这里不插手。
        forward_ratio = None if silence_as_zero else _finite(motion.get("forward_ratio"))
        slip_ratio = None if silence_as_zero else _finite(motion.get("slip_ratio"))
        sliding = False
        if forward_ratio is not None and speed >= self.config.stall_speed_mps:
            with self._lock:
                if forward_ratio < slip_forward_ratio:
                    self._slip_ticks += 1
                    sliding = self._slip_ticks >= slip_threshold_ticks
                else:
                    self._slip_ticks = 0

        with self._lock:
            if speed >= self.config.stall_speed_mps and not sliding:
                self._stall_ticks = 0
                return decision
            if not sliding:
                self._stall_ticks += 1
                if self._stall_ticks < self.config.stall_ticks:
                    return decision

            # 到这里已经确认撞墙了，区别只在正面还是斜着。先花掉绕行预算，
            # 预算用尽才闩锁交还 LLM。
            if sliding:
                self._slip_count += 1
            budget_left = self._recover_attempts < self.config.auto_recover_limit
            if budget_left:
                self._recover_attempts += 1
                self._recover_count += 1
                self._stall_ticks = 0
                self._slip_ticks = 0
                attempt = self._recover_attempts
                if slip_ratio is not None and abs(slip_ratio) > 1e-3:
                    # 正贴着墙滑，滑行方向就是几何上可通行的方向：跟着它转。
                    #
                    # 但只有本路段的第一次绕行才允许由滑行方向定调。转开之后往往
                    # 会蹭上邻墙，那一面的 slip_ratio 符号相反；若每次都重新取样，
                    # 就会 -55/+55/-55 原地对撞，把绕行预算耗在摆动上而不是脱困
                    # （实机复现：三次绕行净转向 -30°，最终仍 movement_stalled）。
                    # 方向一旦选定就锁到路段结束，让连续几次绕行朝同一侧累积。
                    if attempt <= 1:
                        self._recover_sign = 1.0 if slip_ratio > 0.0 else -1.0
                    reason = "auto_recover_slide"
                    back_axis = 0.0
                else:
                    # 正面墙：没有滑行方向可用，沿用上次的转向并先退一步，
                    # 免得贴着墙角转身蹭不出去。
                    reason = "auto_recover_blocked"
                    back_axis = -abs(self.config.auto_recover_back_axis)
                sign = self._recover_sign
            else:
                self._stalled = True
                self._stall_count += 1
                self._mark_unreachable_locked(decision.target_id)

        if budget_left:
            return NavigationDecision(
                "recover",
                f"{reason}:{attempt}/{self.config.auto_recover_limit}",
                decision.target_id,
                decision.bearing_deg,
                decision.distance_m,
                y=back_axis,
                pulse_ms=self.config.pulse_ms,
                revision=decision.revision,
                observed_age_ms=decision.observed_age_ms,
                turn_deg=sign * self.config.auto_recover_turn_deg,
            )
        return NavigationDecision(
            "stop",
            "movement_stalled",
            decision.target_id,
            decision.bearing_deg,
            decision.distance_m,
            revision=decision.revision,
            observed_age_ms=decision.observed_age_ms,
        )

    def _mark_unreachable_locked(self, target_id: str | None) -> None:
        """记下「朝这个实体推过摇杆，人没动」。调用方必须已持有 _lock。

        ttl <= 0 表示关掉这个机制，此时不记账，行为退回纯闩锁。
        """
        if not target_id or self.config.unreachable_ttl_s <= 0.0:
            return
        self._unreachable[target_id] = self._clock() + self.config.unreachable_ttl_s
        if len(self._unreachable) > 64:
            # track id 会随检测跳变而不断新增，不清理就是一个只涨不跌的字典。
            now = self._clock()
            self._unreachable = {
                key: expiry for key, expiry in self._unreachable.items() if expiry > now
            }

    def _unreachable_ids(self) -> set[str]:
        now = self._clock()
        with self._lock:
            expired = [key for key, expiry in self._unreachable.items() if expiry <= now]
            for key in expired:
                del self._unreachable[key]
            return set(self._unreachable)

    def _sample_motion(self) -> Mapping[str, Any] | None:
        provider = self._motion_provider
        if provider is None:
            return None
        motion = provider()
        if not isinstance(motion, Mapping):
            return None
        with self._lock:
            self._last_motion = dict(motion)
        return motion

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            decision = self._last_decision.to_dict()
            return {
                "enabled": True,
                "running": self.thread_alive,
                "tick_hz": self.config.tick_hz,
                "tick_count": self._tick_count,
                "command_count": self._command_count,
                "stop_count": self._stop_count,
                "active_side": self._active_side,
                "last_decision": decision,
                "last_error": self._last_error,
                "behavior": {
                    "policy": "acquire_orient_approach_settle_observe_or_bounded_free_roam",
                    "phase": self._behavior_phase,
                    "phase_age_ms": (
                        None
                        if self._behavior_phase_started_at is None
                        else round(max(0.0, now - self._behavior_phase_started_at) * 1000.0, 1)
                    ),
                    "llm_calls_in_loop": 0,
                    "outcome_sequence": self._behavior_outcome_sequence,
                    "last_outcome": (
                        None
                        if self._behavior_last_outcome is None
                        else dict(self._behavior_last_outcome)
                    ),
                    "wander_turn_sent": self._wander_turn_sent,
                    "wander_segment_age_ms": (
                        None
                        if self._wander_segment_started_at is None
                        else round(max(0.0, now - self._wander_segment_started_at) * 1000.0, 1)
                    ),
                    "wander_step_max_ms": round(_WANDER_MAX_STEP_S * 1000.0, 1),
                },
                "turn": {
                    # 转向是相对位移，同一次观测重发会累加。suppressed 高于
                    # sent 是正常的（tick 频率本就高于出帧频率）；两者相等则说明
                    # 去重没生效，转向正在按 tick/观测 的倍数超调。
                    "last_revision": self._last_turn_revision,
                    "last_sent_age_ms": (
                        None
                        if self._last_turn_sent_at is None
                        else round(max(0.0, now - self._last_turn_sent_at) * 1000.0, 1)
                    ),
                    "cooldown_ms": round(self.config.turn_cooldown_s * 1000.0, 1),
                    "suppressed_count": self._turn_suppressed_count,
                    "settling_suppressed_count": self._turn_settling_suppressed_count,
                    "cooldown_suppressed_count": self._turn_cooldown_suppressed_count,
                    "continuous_retarget": self._turn_retarget_supported,
                },
                "stall": {
                    # detectable=false 表示收不到 VRChat 内置 Velocity 参数，
                    # 「卡墙」这件事在本次会话里根本无法被观测到——不是「没卡」。
                    "detectable": self._motion_feedback_usable,
                    "feedback_required": self._forward_started_at is not None,
                    "feedback_state": self._motion_feedback_state,
                    "capability_confirmed": self._motion_feedback_capability_confirmed,
                    "motion_start_grace_ms": round(self.config.motion_start_grace_s * 1000.0, 1),
                    # 上一条前进指令是否被下游接受。false 表示命令被 scheduler
                    # 拒了（多半是 body output 还没 enable），此时失速判据整体
                    # 失效——速度为零是因为没发命令，不是因为撞墙。
                    "axis_send_ok": self._axis_send_ok,
                    "stalled": self._stalled,
                    "consecutive_ticks": self._stall_ticks,
                    "threshold_ticks": self.config.stall_ticks,
                    "speed_threshold_mps": self.config.stall_speed_mps,
                    "stall_count": self._stall_count,
                    # 斜撞墙：人在动但前进分量被墙压住。纯速度判据看不见这种撞墙，
                    # 因为速度模长还在阈值之上。
                    "slip_ticks": self._slip_ticks,
                    "slip_threshold_ticks": (
                        min(self.config.slip_ticks, _WANDER_SLIP_TICKS)
                        if self._stall_goal_key and self._stall_goal_key[0] == "wander"
                        else self.config.slip_ticks
                    ),
                    "slip_forward_ratio": (
                        max(self.config.slip_forward_ratio, _WANDER_SLIP_FORWARD_RATIO)
                        if self._stall_goal_key and self._stall_goal_key[0] == "wander"
                        else self.config.slip_forward_ratio
                    ),
                    "slip_count": self._slip_count,
                    # 自动绕行：撞墙后先自己转身重试，预算用尽才闩锁交还 LLM。
                    # attempts 达到 limit 且 stalled=true 表示「怎么绕都出不去」，
                    # 这时才真的需要 LLM 换目标。
                    "recover_attempts": self._recover_attempts,
                    "recover_limit": self.config.auto_recover_limit,
                    "recover_count": self._recover_count,
                    "last_motion": None if self._last_motion is None else dict(self._last_motion),
                    # 已实测「推了摇杆但没动」的目标。换目标能解开闩锁，但解不开
                    # 这一份——不然又会选中同一个镜面倒影。
                    "unreachable_targets": sorted(
                        key for key, expiry in self._unreachable.items() if expiry > now
                    ),
                    "unreachable_ttl_s": self.config.unreachable_ttl_s,
                },
                "target_filter": {
                    "grace_ms": round(self.config.target_grace_s * 1000.0, 1),
                    "grace_forward_axis": self.config.min_forward_axis,
                    "bearing_ema_alpha": self.config.bearing_ema_alpha,
                    "range_ema_alpha": self.config.range_ema_alpha,
                    "grace_count": self._target_grace_count,
                    "target_id": (
                        None if self._target_observation is None else self._target_observation.target_id
                    ),
                },
                "explorer": self._explorer.snapshot(),
            }

    def _run(self) -> None:
        period = 1.0 / self.config.tick_hz
        deadline = self._clock()
        while not self._stop_event.is_set():
            now = self._clock()
            if now < deadline:
                self._stop_event.wait(min(period, deadline - now))
                continue
            deadline = now + period
            self.tick()
        self._safe_release()

    def _record_target_observation(
        self,
        *,
        target_id: str,
        bearing: float,
        distance: float | None,
        apparent: float | None,
        apparent_clipped: bool,
        revision: int,
        observed_age_ms: float,
        now: float,
    ) -> _TargetObservation:
        """记录可靠目标，并对新视觉帧做轻量指数平滑。"""
        with self._lock:
            previous = self._target_observation
            def smooth(old: float | None, new: float | None, alpha: float) -> float | None:
                if new is None:
                    return None
                if old is None:
                    return new
                return old + alpha * (new - old)

            same_target = previous is not None and previous.target_id == target_id
            old_distance = previous.distance_m if same_target and previous is not None else None
            old_apparent = previous.apparent_height if same_target and previous is not None else None
            # 接近停止边界时只允许滤波延迟“目标变远”，不延迟“目标已经更近”。
            # 否则表观高度突然越过停止线时还会多走几帧，平滑反而损害安全。
            filtered_distance = (
                distance
                if distance is not None and old_distance is not None and distance < old_distance
                else smooth(old_distance, distance, self.config.range_ema_alpha)
            )
            filtered_apparent = (
                apparent
                if apparent is not None and old_apparent is not None and apparent > old_apparent
                else smooth(old_apparent, apparent, self.config.range_ema_alpha)
            )
            old_bearing = previous.bearing_deg if same_target and previous is not None else None
            # 明显跨过画面中心时立即采用新方向，不能让 EMA 的惯性再朝旧方向转一帧；
            # 同侧的小幅检测框抖动才做平滑。
            filtered_bearing = (
                bearing
                if old_bearing is not None
                and old_bearing * bearing < 0.0
                and abs(bearing) > self.config.bearing_deadband_deg
                else smooth(old_bearing, bearing, self.config.bearing_ema_alpha)
            )
            observation = _TargetObservation(
                target_id=target_id,
                bearing_deg=float(filtered_bearing),
                distance_m=filtered_distance,
                apparent_height=filtered_apparent,
                apparent_clipped=bool(apparent_clipped),
                revision=revision,
                observed_at=now - max(0.0, observed_age_ms) / 1000.0,
            )
            self._target_observation = observation
            return observation

    def _cached_target_for_goal(
        self,
        goal: Mapping[str, Any],
        skip_ids: set[str],
        now: float,
    ) -> _TargetObservation | None:
        """只在很短的视觉抖动窗口内复用同一目标的可靠观测。"""
        with self._lock:
            observation = self._target_observation
            if observation is None:
                return None
            goal_target = str(goal.get("target_id") or "").strip().lower()
            if goal_target and observation.target_id.lower() != goal_target:
                return None
            if observation.target_id in skip_ids:
                return None
            if now - observation.observed_at > self.config.target_grace_s:
                return None
            self._target_grace_count += 1
            return observation

    def _decide(
        self,
        goal_state: Mapping[str, Any] | None,
        world: Mapping[str, Any] | None,
        now: float,
    ) -> NavigationDecision:
        if not isinstance(goal_state, Mapping) or goal_state.get("state") != "armed":
            return NavigationDecision("idle", "autonomy_not_armed")
        goal = _mapping(goal_state.get("goal"))
        if not goal:
            return NavigationDecision("idle", "no_goal")
        age_s = _finite(goal.get("age_seconds"))
        if age_s is not None and age_s > self.config.max_goal_age_s:
            return NavigationDecision("stop", "goal_expired")
        goal_kind = str(goal.get("kind") or "").strip().lower()
        if goal_kind in {"depart", "wander"}:
            return self._free_roam_decision(goal, world, now)
        target_id = str(goal.get("target_id") or "").strip()
        if not target_id:
            if str(goal.get("kind") or "").strip().lower() != "explore":
                return NavigationDecision("stop", "target_id_required")
            if not isinstance(goal.get("selector"), Mapping):
                # 自由文本仍然不能驱动本地自动选人；只有经过 schema 校验的明确
                # selector 才能启用 Explorer。
                return NavigationDecision("stop", "target_id_required")
            if not isinstance(world, Mapping) or world.get("capture_active") is False:
                return NavigationDecision("stop", "world_unknown")
            status = _mapping(world.get("status")) or {}
            revision = int(_finite(status.get("revision")) or 0)
            observed_age = _finite(status.get("last_observation_age_ms"))
            if observed_age is None or observed_age > self.config.max_observation_age_ms:
                return NavigationDecision(
                    "stop", "observation_stale", revision=revision, observed_age_ms=observed_age
                )
            # 空画面会带 no_recent_visual_observation，但 fresh status 证明这一帧
            # 确实刚被处理过；搜索时允许它表示“当前方向没候选”。其他不确定性
            # （遮挡、世界切换、后端错误）仍然立即停车。
            uncertainties = [
                item for item in blocking_uncertainties(world.get("uncertainties"))
                if item != "no_recent_visual_observation"
            ]
            if uncertainties:
                return NavigationDecision(
                    "stop", "world_uncertain", revision=revision, observed_age_ms=observed_age
                )
            directive = self._explorer.decide(
                goal,
                world,
                max_turn_deg=self.config.max_turn_deg,
                turn_gain=self.config.turn_gain,
                bearing_deadband_deg=self.config.bearing_deadband_deg,
                navigator_max_forward_axis=self.config.max_forward_axis,
                skip_ids=self._unreachable_ids(),
            )
            if directive.state == "turn":
                # 搜索扫描是固定 45° 的离散步，不是视觉 bearing 修正。必须等上一
                # 步真实落地后再记一次预算；传 revision=0 会走 turning/cooldown
                # 门控，避免 continuous-retarget 模式在 0.8 秒里“账面转满一圈”。
                turn_revision = (
                    0
                    if directive.reason in {"explore_scan", "explore_reacquire_scan"}
                    else revision
                )
                return NavigationDecision(
                    "turn",
                    directive.reason,
                    directive.target_id,
                    directive.bearing_deg,
                    revision=turn_revision,
                    observed_age_ms=observed_age,
                    turn_deg=directive.turn_deg,
                )
            if directive.state == "advance":
                return NavigationDecision(
                    "advance",
                    directive.reason,
                    side="left",
                    y=directive.forward_axis,
                    pulse_ms=self.config.pulse_ms,
                    revision=revision,
                    observed_age_ms=observed_age,
                )
            return NavigationDecision(
                "reached" if directive.state == "found" else "stop",
                directive.reason,
                directive.target_id,
                directive.bearing_deg,
                revision=revision,
                observed_age_ms=observed_age,
            )
        if not isinstance(world, Mapping):
            return NavigationDecision("stop", "world_unknown")
        if not bool(world.get("available")):
            status = _mapping(world.get("status")) or {}
            revision = int(_finite(status.get("revision")) or 0)
            observed_age = _finite(status.get("last_observation_age_ms"))
            uncertainties = set(blocking_uncertainties(world.get("uncertainties")))
            # 一张新鲜空画面是“这次没看到锁定目标”，不是“世界未知”。有限行为
            # 需要靠它累计重捕获窗口；否则目标一离开画面，available=false 会先把
            # 导航器卡在 world_unknown，永远产不出 target_lost 终态。
            fresh_empty_target = (
                str(goal.get("kind") or "").strip().lower() == "approach_observe"
                and world.get("capture_active") is not False
                and observed_age is not None
                and observed_age <= self.config.max_observation_age_ms
                and not (uncertainties - {"no_recent_visual_observation"})
            )
            if fresh_empty_target:
                return NavigationDecision(
                    "stop",
                    "target_not_visible",
                    target_id=str(goal.get("target_id") or "")[:96] or None,
                    revision=revision,
                    observed_age_ms=observed_age,
                )
            return NavigationDecision("stop", "world_unknown")
        uncertainties = blocking_uncertainties(world.get("uncertainties"))
        if uncertainties:
            return NavigationDecision("stop", "world_uncertain")
        status = _mapping(world.get("status")) or {}
        revision = int(_finite(status.get("revision")) or 0)
        observed_age = _finite(status.get("last_observation_age_ms"))
        if observed_age is None or observed_age > self.config.max_observation_age_ms:
            return NavigationDecision("stop", "observation_stale", revision=revision, observed_age_ms=observed_age)

        skip = self._unreachable_ids()
        entity = self._select_target(goal, world.get("entities"), skip)
        observation: _TargetObservation | None = None
        failure_reason = "target_not_visible"
        failure_target_id = str(goal.get("target_id") or "")[:96] or None
        if entity is not None:
            failure_target_id = str(entity.get("id") or "")[:96] or None
            confidence = _finite(entity.get("confidence")) or 0.0
            if entity.get("visible") is False or confidence < self.config.min_confidence:
                failure_reason = "target_low_confidence"
            else:
                bearing, distance, apparent, apparent_clipped = _spatial_hint(entity)
                if bearing is None:
                    failure_reason = "target_bearing_unknown"
                else:
                    observation = self._record_target_observation(
                        target_id=failure_target_id or "",
                        bearing=bearing,
                        distance=distance,
                        apparent=apparent,
                        apparent_clipped=apparent_clipped,
                        revision=revision,
                        observed_age_ms=observed_age,
                        now=now,
                    )

        if observation is None:
            # 区分「没看到匹配的东西」和「看到了但刚证明够不着」。不可达是硬状态，
            # 不能被视觉宽限覆盖；普通漏检/低置信才允许短暂复用最后可靠观测。
            if skip and self._select_target(goal, world.get("entities"), frozenset()) is not None:
                return NavigationDecision(
                    "stop", "target_unreachable", revision=revision, observed_age_ms=observed_age
                )
            observation = self._cached_target_for_goal(goal, skip, now)
            if observation is None:
                return NavigationDecision(
                    "stop",
                    failure_reason,
                    target_id=failure_target_id,
                    revision=revision,
                    observed_age_ms=observed_age,
                )
            observation_mode = "grace"
            revision = observation.revision
            observed_age = max(observed_age, max(0.0, now - observation.observed_at) * 1000.0)
        else:
            observation_mode = "live"

        target_id = observation.target_id or None
        bearing = observation.bearing_deg
        distance = observation.distance_m
        apparent = observation.apparent_height
        apparent_clipped = observation.apparent_clipped
        # 观察阶段不用把人钉死在视野正中央。较宽的注视死区更像自然地站在旁边
        # 看一会儿；目标真正走远时，后面的距离判断仍会重新进入接近阶段。
        bearing_deadband = self.config.bearing_deadband_deg
        if (
            str(goal.get("kind") or "").strip().lower() == "approach_observe"
            and self._behavior_phase == "observe"
        ):
            bearing_deadband = self.config.behavior_observe_deadband_deg
        if abs(bearing) > bearing_deadband:
            # 转向直接给角度，不再经摇杆。bearing 本身就是「要转多少度才能把目标
            # 转到画面中央」，压成摇杆量再靠脉冲时长积分只会丢精度；而且 VR 模式
            # 下右摇杆根本不产生转向。
            #
            # 符号：bearing>0 表示目标在画面右侧（local_perception.py 的
            # (center_x - 0.5) * fov），而 +yaw 是左转（实测 tmp/turn_sign.py：
            # 转 +20° 画面内容右移 154 px），所以要取反才能转向目标。
            turn_deg = _clamp(
                -bearing * self.config.turn_gain,
                -self.config.max_turn_deg,
                self.config.max_turn_deg,
            )
            return NavigationDecision(
                "turn",
                "target_off_center",
                target_id,
                bearing,
                distance,
                None,
                0.0,
                0.0,
                0,
                revision,
                observed_age,
                turn_deg=turn_deg,
                observation_mode=observation_mode,
            )
        # 优先使用表观高度：这是二维检测器唯一实测得到的接近度指标。必须排在
        # ``distance is None`` 之前，否则不带深度适配器的检测器永远走不到这里。
        if apparent is not None:
            target_apparent = self.config.target_apparent_height
            reach_apparent = (
                target_apparent * _CLIPPED_REACH_FACTOR if apparent_clipped else target_apparent
            )
            if apparent >= reach_apparent:
                return NavigationDecision(
                    "reached",
                    "target_in_interaction_range",
                    target_id,
                    bearing,
                    distance,
                    revision=revision,
                    observed_age_ms=observed_age,
                    observation_mode=observation_mode,
                )
            # 目标框达到停止尺寸的 80% 前保持巡航；最后 20% 才线性降到最低档。
            # 旧线性/平方根曲线都过早减速，实机大部分时间只有 0.22~0.89 m/s。
            brake_start = target_apparent * 0.80
            if apparent <= brake_start:
                forward = self.config.max_forward_axis
            else:
                brake_span = max(1e-6, target_apparent - brake_start)
                brake_ratio = _clamp(
                    (target_apparent - apparent) / brake_span,
                    0.0,
                    1.0,
                )
                forward = self.config.min_forward_axis + brake_ratio * (
                    self.config.max_forward_axis - self.config.min_forward_axis
                )
            if observation_mode == "grace":
                # 0.60 巡航速度下继续沿旧画面走满 300 ms，实测可能盲走约 0.67 m。
                # 宽限只用于跨过一两帧检测抖动，立即降到最低档可把最坏距离压到
                # 约 0.2 m；重新看到目标后下一次 10 Hz tick 会恢复正常速度。
                forward = self.config.min_forward_axis
            return NavigationDecision(
                "advance",
                "target_grace_slow" if observation_mode == "grace" else "target_centered",
                target_id,
                bearing,
                distance,
                "left",
                0.0,
                forward,
                self.config.pulse_ms,
                revision,
                observed_age,
                observation_mode=observation_mode,
            )
        if distance is None:
            return NavigationDecision(
                "stop",
                "target_distance_unknown",
                target_id,
                bearing,
                None,
                revision=revision,
                observed_age_ms=observed_age,
                observation_mode=observation_mode,
            )
        if distance <= self.config.target_distance_m:
            return NavigationDecision(
                "reached",
                "target_in_interaction_range",
                target_id,
                bearing,
                distance,
                revision=revision,
                observed_age_ms=observed_age,
                observation_mode=observation_mode,
            )
        # 有米制深度时也只在目标距离外最后 50% 区间减速。
        brake_span = self.config.target_distance_m * 0.50
        if distance >= self.config.target_distance_m + brake_span:
            forward = self.config.max_forward_axis
        else:
            brake_ratio = _clamp(
                (distance - self.config.target_distance_m) / max(brake_span, 1e-6),
                0.0,
                1.0,
            )
            forward = self.config.min_forward_axis + brake_ratio * (
                self.config.max_forward_axis - self.config.min_forward_axis
            )
        if observation_mode == "grace":
            forward = self.config.min_forward_axis
        return NavigationDecision(
            "advance",
            "target_grace_slow" if observation_mode == "grace" else "target_centered",
            target_id,
            bearing,
            distance,
            "left",
            0.0,
            forward,
            self.config.pulse_ms,
            revision,
            observed_age,
            observation_mode=observation_mode,
        )

    def _free_roam_decision(
        self,
        goal: Mapping[str, Any],
        world: Mapping[str, Any] | None,
        now: float,
    ) -> NavigationDecision:
        """执行有限后退或一条由 LLM 规划的闲逛路段，不伪造导航对象。"""
        kind = str(goal.get("kind") or "").strip().lower()
        age_s = _finite(goal.get("age_seconds")) or 0.0
        params = self._free_roam_params(goal, kind)
        duration = params["duration_s"]
        axis = params["forward_axis"]
        if kind == "depart" and age_s >= duration:
            return NavigationDecision("stop", "depart_complete")

        if kind == "depart":
            # 短时后退只用于离开当前观察点，不依赖人物框，也不把倒退速度套进
            # “向前分量”失速判据。
            return NavigationDecision(
                "retreat",
                "depart_back_away",
                side="left",
                y=-axis,
                pulse_ms=self.config.pulse_ms,
            )

        if not isinstance(world, Mapping) or world.get("capture_active") is False:
            return NavigationDecision("stop", "world_unknown")
        status = _mapping(world.get("status")) or {}
        revision = int(_finite(status.get("revision")) or 0)
        observed_age = _finite(status.get("last_observation_age_ms"))
        if observed_age is None or observed_age > self.config.max_observation_age_ms:
            return NavigationDecision(
                "stop", "observation_stale", revision=revision, observed_age_ms=observed_age
            )
        uncertainties = [
            item for item in blocking_uncertainties(world.get("uncertainties"))
            if item != "no_recent_visual_observation"
        ]
        if uncertainties:
            return NavigationDecision(
                "stop", "world_uncertain", revision=revision, observed_age_ms=observed_age
            )
        turn_deg = params["turn_deg"]
        if turn_deg is None:
            # 正常入口会在 AutonomyRuntime 拒绝这种目标；这里仍做最后一道防御，
            # 保证第三方 provider 不能让导航器自行决定路线。
            return NavigationDecision(
                "stop",
                "wander_direction_required",
                revision=revision,
                observed_age_ms=observed_age,
            )
        with self._lock:
            turn_sent = self._wander_turn_sent
            segment_started_at = self._wander_segment_started_at
        if not turn_sent:
            return NavigationDecision(
                "turn",
                "wander_llm_turn",
                revision=0,
                observed_age_ms=observed_age,
                turn_deg=turn_deg,
            )
        if self._turn_gate_reason(now, None) is not None:
            return NavigationDecision(
                "stop",
                "wander_turn_settling",
                revision=revision,
                observed_age_ms=observed_age,
            )
        if (
            segment_started_at is not None
            and now - segment_started_at >= duration
        ):
            # 一条路段完成就停车并释放目标。下一条方向必须由 LLM 看新画面后提交。
            return NavigationDecision(
                "stop",
                "wander_step_complete",
                revision=revision,
                observed_age_ms=observed_age,
            )
        return NavigationDecision(
            "advance",
            "wander_forward",
            side="left",
            y=axis,
            pulse_ms=self.config.pulse_ms,
            revision=revision,
            observed_age_ms=observed_age,
        )

    def _select_target(
        self,
        goal: Mapping[str, Any],
        raw_entities: Any,
        skip_ids: frozenset[str] | set[str] = frozenset(),
    ) -> Mapping[str, Any] | None:
        if not isinstance(raw_entities, (list, tuple)):
            return None
        target_id = str(goal.get("target_id") or "").strip().lower()
        if not target_id:
            return None
        candidates: list[tuple[float, Mapping[str, Any]]] = []
        for raw in raw_entities:
            if not isinstance(raw, Mapping):
                continue
            entity_id = str(raw.get("id") or "").strip().lower()
            # 这里只做精确 ID 查找。目标暂时消失应停车等待，而不是根据 goal 文本、
            # label 或检测置信度把海报、镜像、门或另一个玩家选成新目标。
            if entity_id != target_id:
                continue
            if raw.get("visible") is False:
                continue
            # 与 _decide 里 target_id 的取法保持一致，否则记账和跳过对不上。
            if skip_ids and str(raw.get("id") or "")[:96] in skip_ids:
                continue
            confidence = _finite(raw.get("confidence")) or 0.0
            candidates.append((confidence, raw))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _turn_already_sent(self, revision: int) -> bool:
        """同一次观测只转一次。

        转向和前进的命令语义不一样：前进轴是**按住的状态**，重发只是覆盖，
        幂等；转向是**相对位移**，重发会累加。而导航器的 tick 频率远高于检测
        器的出帧频率（实测 9.14 Hz 对 1.75 Hz，每观测 5.2 tick），于是同一个
        bearing 会被当成 5 条独立的转向命令发出去。

        实测 bearing -11.47 度算出 +9.18 度，连发 5 次就是 +45.9 度，对着 11.47
        度的偏差转过去 4 倍，冲过中心后符号翻转，再朝反方向超调——闭环第一次
        实跑就是这样从 -11 发散到 -18，最后摄像机停在海面上。

        去重之后转向按观测频率发（1.75 Hz），单次仍是 -bearing * turn_gain。
        实测发 10 度实际转 5.13 度，等效增益约 0.5，欠阻尼收敛，不会震荡。

        revision <= 0 表示这个世界源根本不发版本号，观测之间无从区分。此时不
        去重——「一条都不转」比「多转几度」坏得多，会直接废掉转向。
        """
        if revision <= 0:
            return False
        with self._lock:
            return self._last_turn_revision == revision

    def _turn_gate_reason(self, now: float, revision: int | None) -> str | None:
        """判断当前发送器能否安全接收下一条视觉转向修正。"""
        if self._turn_retarget_supported and revision is not None and revision > 0:
            # 每个视觉 revision 最多发送一次，而发送器把修正换算成基于当前 yaw 的
            # 绝对目标；上一段还没结束也可以直接重定向，不会发生相对角度累加。
            return None
        provider = self._turn_state_provider
        if provider is not None:
            try:
                state = provider()
            except Exception:
                state = {}
            if isinstance(state, Mapping) and bool(state.get("turning")):
                return "settling"
        with self._lock:
            last_sent_at = self._last_turn_sent_at
        if (
            last_sent_at is not None
            and now - last_sent_at < self.config.turn_cooldown_s
        ):
            return "cooldown"
        return None

    def _record_turn_suppressed(self, reason: str) -> None:
        with self._lock:
            self._turn_suppressed_count += 1
            if reason == "settling":
                self._turn_settling_suppressed_count += 1
            elif reason == "cooldown":
                self._turn_cooldown_suppressed_count += 1

    def _send_gated_turn(
        self,
        delta_deg: float,
        now: float,
        *,
        revision: int | None = None,
    ) -> bool:
        gate_reason = self._turn_gate_reason(now, revision)
        if gate_reason is not None:
            self._record_turn_suppressed(gate_reason)
            return False
        if not self._send_turn(delta_deg):
            return False
        with self._lock:
            self._command_count += 1
            self._last_turn_sent_at = now
            if revision is not None:
                self._last_turn_revision = revision
        return True

    def _apply(self, decision: NavigationDecision, now: float) -> bool:
        if decision.state == "turn":
            # 转向不占摇杆，所以不进 _active_side：它是 play space 的朝向，不是
            # 需要按住再松开的输入。但前进摇杆必须先松——边走边转会让 bearing
            # 在观测间隔里持续漂移，转向永远收敛不了。
            if self._active_side is not None:
                self._safe_release()
            if self._turn_already_sent(decision.revision):
                self._record_turn_suppressed("revision")
                return False
            return self._send_gated_turn(
                decision.turn_deg,
                now,
                revision=decision.revision,
            )
        if decision.state == "retreat" and decision.side:
            if self._active_side == "right":
                self._release_inputs("right")
            sent = self._send_axes("left", decision.x, decision.y, decision.pulse_ms)
            with self._lock:
                self._axis_send_ok = bool(sent)
                self._forward_started_at = None
                self._motion_feedback_usable = False
                self._motion_feedback_state = "retreat_not_evaluated"
                if sent:
                    self._active_side = "left"
                    self._command_count += 1
            return bool(sent)
        if decision.state == "advance" and decision.side:
            if self._active_side == "right":
                self._release_inputs("right")
            sent = self._send_axes("left", decision.x, decision.y, decision.pulse_ms)
            with self._lock:
                # 记下「这条前进指令到底发出去了吗」。失速守卫必须能区分
                # 「发了但人没动」（真卡墙）和「压根没发」（body 未 enable）。
                self._axis_send_ok = bool(sent)
                if sent and self._forward_started_at is None:
                    self._forward_started_at = now
                    self._motion_feedback_usable = False
                    self._motion_feedback_state = "awaiting_motion"
                elif not sent:
                    self._forward_started_at = None
                    self._motion_feedback_usable = False
                    self._motion_feedback_state = "command_rejected"
            if sent:
                with self._lock:
                    self._active_side = "left"
                    self._command_count += 1
            return bool(sent)
        if decision.state == "recover":
            # 绕行是「先退开、再转身」。后退轴可选：斜撞墙时人还在动，直接转
            # 就行；正面墙才需要先退一步，免得贴着墙角转身蹭不出去。
            #
            # 转向不做 revision 去重。去重是为了避免同一次观测被重复转，而绕行
            # 本就与观测无关——它由速度回传触发，每次都是新的一步。
            if decision.y:
                self._send_axes("left", 0.0, decision.y, decision.pulse_ms)
            else:
                self._safe_release()
            # recover 决策已经在失速守卫里消耗了一次有限预算，若在这里静默抑制，
            # 会出现“预算用掉了但实际没转”的假恢复。它由速度反馈触发且不是视觉
            # revision 的重复命令，因此保持一次决策对应一次真实提交。
            turn_sent = self._send_turn(decision.turn_deg)
            if turn_sent:
                with self._lock:
                    self._command_count += 1
                    # 让下一 tick 也经过转向冷却门控。否则 scheduler 尚未来得及把
                    # turning 置为 true 时，导航器可能立刻再次压下前进轴。
                    self._last_turn_sent_at = now
            return bool(turn_sent)
        self._safe_release()
        return False

    def _safe_release(self) -> None:
        with self._lock:
            active = self._active_side
            self._active_side = None
            self._forward_started_at = None
            self._motion_feedback_usable = False
            self._motion_feedback_state = "idle_not_required"
        if active is not None:
            try:
                self._release_inputs("all")
            finally:
                with self._lock:
                    self._stop_count += 1


__all__ = ["LocalNavigator", "NavigationDecision", "NavigatorConfig"]
