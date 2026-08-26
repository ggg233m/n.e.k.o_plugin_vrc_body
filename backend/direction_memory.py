"""方向性死路记忆：只记「最近朝哪个方向试过、结果如何」。

这不是地图，也不是里程计。VRChat 不回传绝对朝向（``AngularY`` 只在移动期间产生
新样本，量级不确定且停止后不再更新），所以任何全局坐标积分都会旋转漂移且无法闭
环校正。这里刻意退一步：不建坐标系，只在方向扇区上记账，并给每条记录一个短
TTL，超时就忘。

**扇区锚在调度器虚拟 HMD 的 yaw 上，不是「相对当前朝向」。** 存的是 yaw 加请求
转角得到的记忆系角度；对外汇报时再换算回相对当前朝向。两者混用是个真实的 bug：
撞墙时记下 ``+45°``，绕行或转身之后同一个 ``+45°`` 已经指向另一堵墙，拿旧记录去
拒绝会把可走的方向封死。虚拟 HMD yaw 是全仓库唯一一个连续、单调、我们自己驱动
的朝向量，所以它是这里唯一可用的公共锚点——但它只在一个世界里有意义，换世界必须
``reset()``。

它回答的唯一问题是：*从现在这个位置出发，哪些方向刚试过、撞了没有*。这足以消除
实测到最痛的行为——反复把同一堵墙选成目标——而不需要假装知道自己在哪。

预测和实测**分开保存**，不合并成单值状态。主 LLM 看图给出的方向偏好是预测，走一段
撞没撞是实测；两者冲突本身就是要交给主 LLM 的信息（「你说左边能走 0.8，实际撞了」）。
如果压进一个 ``state`` 字段，这个分歧就被抹掉了。

实测态只有三个，没有 ``predicted_free``——预测不是一种实测结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Literal, Mapping

# 实测态：只描述「实际走过之后发生了什么」。
EmpiricalState = Literal["unknown", "verified_free", "blocked"]

# 预测兑现情况。只在同一扇区既有预测又有实测时才有值。
PredictionOutcome = Literal["confirmed", "false_positive", "false_negative"]

# 方向扇区宽度。45° 足够粗，能把 wander 的 ±25° 和 recover 的 ±55° 合并成同一个
# 「左前方」——两者撞到的通常就是同一堵墙，分开记会让它攒不够 confident_block。
# 同时它又足够细，能区分左右和前后，不会把「前面是墙」扩散成「四周都是墙」。
_SECTOR_DEG = 45.0
_SECTOR_COUNT = 8

# 实测记忆窗口。地形不会变，但「我在哪」会——没有绝对定位，走远之后旧记录就不再描述
# 当前位置的周边。宁可忘掉重试一次，也不要拿着过期证据把可走的方向永久封死。
_DEFAULT_TTL_S = 45.0

# 预测 TTL 远短于实测。预测来自一帧画面，只对「站在拍那一帧的位置」有效；一旦走出
# 那个位置，画面里看到的通道可能已经不在那个方向上了。5 秒约等于一段 wander。
_DEFAULT_PREDICTION_TTL_S = 5.0

# 同一扇区反复受阻才升级为高置信。单次可能是擦到一个装饰物或另一个玩家挡了一下。
#
# 2026-08-26 实机确认：这个计数在自由导航里几乎攒不起来，而这是**设计的结果**，
# 不是没调好的阈值。扇区键是绝对方向 `anchor_yaw + turn_deg`，`anchor_yaw` 取每段
# 提交时的虚拟 HMD yaw；第一次撞墙后 recovery 会把朝向转开（实测 55°），下一段除非
# 主 LLM 反向补偿才会落回同一扇区。实测四段连撞分别落在 sector 7/5/0/…，各自
# blocked_count=1，confident_block 一次没触发。
#
# 这是正确的：两段指向两个不同的绝对方向，就是两堵墙，合并计数才是错的。代价是
# should_refuse() 实际很少生效，这份记忆主要作为给主 LLM 的只读历史证据存在。
# 「别再往那撞」的实时职责本应在光流安全门（navigator._traversability_guard_reason），
# 但 2026-08-26 同一轮实测发现**那道门也从不触发**：整段撞墙里三个扇区一直报
# predicted_clear，前进中 expansion_rate 实测 max 0.1863 而阈值是 0.55。所以当前
# 两个机制都不拦，见 ROADMAP 的 P2.5 校准条目。
# 即便如此也不要把这个计数改小到 1：单次擦碰就封死一个 45° 扇区是另一个方向的错，
# 而且封的是绝对方向、TTL 45s，代价比漏拦更难察觉。缺的拦路能力应该在光流门补。
_CONFIDENT_BLOCK_COUNT = 2

# 判定预测兑现的阈值。中间带（0.35~0.65）算「没表态」，实测结果不论好坏都记
# confirmed——把模棱两可的分数算成预测失败，会让 disagreement 摘要噪声太大。
_PREDICTION_LEAN_FREE = 0.65
_PREDICTION_LEAN_BLOCKED = 0.35

# 算「真的走通了」所需的最小门控推进距离。0.37m 是 2026-08-26 实测巡航速度
# 1.1111 m/s（vertical=0.35，多段一致）下三拍（10Hz）的位移，比一次起步抖动大，
# 又远小于一段正常 wander。
#
# 原值 0.35 基于 1.05 m/s 的估计巡航速度，偏低约 6%，会让门槛系统性偏松——刚够
# 三拍的贴墙滑行也可能凑到 0.35。按实测速度重算后一并收紧。
#
# 光有速度样本不够：VRChat 顶着墙时照样回传速度包，值可能就是 0.0，而 0.0 是个
# 有限数——只看「收到过样本」会把撞墙记成 verified_free，正好和这套记忆想解决的
# 问题相反。所以必须要求门控积分出正向距离。
#
# 实测参考：贴墙滑行 forward_ratio 0.628、水平速度 0.6976 m/s；三拍只能积出约
# 0.21m，低于本门槛，不会被误记成 verified_free。
_MIN_PROGRESS_M = 0.37

# 主 LLM 用的三个方向标签。角度沿用项目约定：正左负右，与 wander 的 turn_deg 同源。
_NAMED_BEARINGS = {"left": 45.0, "forward": 0.0, "right": -45.0}


def _bearing_of_key(key: Any) -> float | None:
    """把 direction_scores 的键解析成角度。无法识别的键跳过而不是报错——这份分数
    来自 LLM，多一个拼错的键不该让整段闲逛失败。"""
    if isinstance(key, (int, float)) and not isinstance(key, bool):
        value = float(key)
        return value if math.isfinite(value) else None
    if isinstance(key, str):
        named = _NAMED_BEARINGS.get(key.strip().lower())
        if named is not None:
            return named
        try:
            value = float(key)
        except ValueError:
            return None
        return value if math.isfinite(value) else None
    return None


def normalize_bearing(deg: float) -> float:
    """把任意角度折进 (-180, 180]，正左负右（沿用项目既有约定）。"""
    wrapped = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def sector_of(deg: float) -> int:
    """把相对朝向映射到扇区索引。0 = 正前，向左递增。"""
    return int(((normalize_bearing(deg) + _SECTOR_DEG / 2.0) % 360.0) // _SECTOR_DEG)


def sector_center_deg(sector: int) -> float:
    """扇区代表方向，用于回报给上层。"""
    return normalize_bearing((sector % _SECTOR_COUNT) * _SECTOR_DEG)


@dataclass
class DirectionEvidence:
    """一个方向扇区上的记录：实测证据 + 主 LLM 的方向偏好，两者分开存。"""

    sector: int
    empirical_state: EmpiricalState = "unknown"
    blocked_count: int = 0
    cleared_count: int = 0
    # 主 LLM 看图给的方向偏好，0~1。刻意不叫「通行概率」：它是偏好排序的依据，不是
    # 校准过的概率，也没有几何依据。None = 这个扇区没有有效预测。
    predicted_score: float | None = None
    prediction_expires_at: float = 0.0
    # 预测下过之后第一次实测的兑现情况。保留到该扇区记录本身过期，这样主 LLM 即使
    # 在预测早已失效之后，仍能看到自己上次判断错了。
    prediction_outcome: PredictionOutcome | None = None
    # 附加代价，不进可通行判断。检测器看到人只说明那边可能有动态障碍或是开放区域，
    # 说明不了有没有墙——所以它单独一条，不写进 predicted_score。
    dynamic_obstacle: bool = False
    # 最近一次「有效推进距离」的粗估：直行段速度积分。只在 forward_ratio 足够高
    # 时累计，所以它不是位移真值，只是「这个方向至少走通了多远」的下界。
    last_progress_m: float | None = None
    updated_at: float = 0.0
    expires_at: float = 0.0
    # 这一段移动期间是否观测到转向。没有绝对朝向，所以只记「转过」这个事实，
    # 让上层知道该扇区的方向标签本身有多少不确定性。
    turned_during_segment: bool = False

    @property
    def confident_block(self) -> bool:
        return (
            self.empirical_state == "blocked"
            and self.blocked_count >= _CONFIDENT_BLOCK_COUNT
        )

    def live_prediction(self, now: float) -> float | None:
        """未过期的预测分；过期返回 None（但 prediction_outcome 仍保留）。"""
        if self.predicted_score is None or self.prediction_expires_at <= now:
            return None
        return self.predicted_score

    def snapshot(self, now: float) -> dict[str, Any]:
        return {
            "sector": self.sector,
            "bearing_deg": sector_center_deg(self.sector),
            "empirical_state": self.empirical_state,
            "blocked_count": self.blocked_count,
            "cleared_count": self.cleared_count,
            "confident_block": self.confident_block,
            "predicted_score": self.live_prediction(now),
            "prediction_outcome": self.prediction_outcome,
            "dynamic_obstacle": self.dynamic_obstacle,
            "last_progress_m": (
                None if self.last_progress_m is None else round(self.last_progress_m, 2)
            ),
            "age_ms": round(max(0.0, now - self.updated_at) * 1000.0, 1),
            "ttl_remaining_ms": round(max(0.0, self.expires_at - now) * 1000.0, 1),
            "turned_during_segment": self.turned_during_segment,
        }


@dataclass
class SegmentOutcome:
    """一段移动结束后交给记忆的实测事实。

    全部字段都来自已经在跑的信号：``bearing_deg`` 是提交给调度器的相对转角，
    ``heading_deg`` 是这一段**起始时**的调度器虚拟 HMD yaw（记忆锚点），
    ``progress_m`` 是门控积分的结果，``blocked`` 来自现有失速/贴墙判据，
    ``turned`` 来自这一段期间是否收到新鲜 AngularY 样本。
    """

    bearing_deg: float
    blocked: bool
    heading_deg: float = 0.0
    progress_m: float | None = None
    turned: bool = False
    # 这一段有没有拿到能证明「确实走了」的速度证据。为假时不受阻也**不算** verified_free：
    # 到时了但一个速度样本都没有，可能是 Avatar 不回传、也可能是压根没动起来，
    # 把它记成走通会让下一次闲逛拿着一条从未验证的通道当既有结论。
    evidence_available: bool = True

    @property
    def cleared(self) -> bool:
        """是否够条件记成 ``verified_free``。

        三个条件缺一不可：没受阻、拿到过速度证据、门控积分出了实际推进距离。
        ``progress_m is None`` 表示调用方没接积分（例如只有失速判据的旧路径），
        此时退回「有速度证据即可」——但它只应出现在离线回放里，实机路径必须传值。
        """
        if self.blocked or not self.evidence_available:
            return False
        if self.progress_m is None:
            return True
        return float(self.progress_m) >= _MIN_PROGRESS_M


@dataclass
class DirectionMemory:
    """短期方向记忆。纯内存、无 IO，可离线回放验证。

    所有公开方法都接受 ``heading_deg``——调用时的调度器虚拟 HMD yaw。扇区键存在
    「记忆系」里（``heading + bearing``），查询和汇报时再换算回相对当前朝向。不带
    锚点混用相对角是个真实的 bug：撞墙时记 ``+45°``，转身之后同一个 ``+45°`` 指向
    另一个地方，旧记录会把可走的方向封死。

    刻意**不提供**「选一个方向」的方法。闲逛路线必须由主 LLM 决定，后端只保存证据、
    汇报证据、以及拒绝已确认封死的方向；一旦这里出现 ``best_direction()``，导航器就
    又变成了替 LLM 做闲逛决策的那一层。
    """

    ttl_s: float = _DEFAULT_TTL_S
    prediction_ttl_s: float = _DEFAULT_PREDICTION_TTL_S
    _sectors: dict[int, DirectionEvidence] = field(default_factory=dict)
    _sequence: int = 0

    def _entry(self, sector: int) -> DirectionEvidence:
        entry = self._sectors.get(sector)
        if entry is None:
            entry = DirectionEvidence(sector=sector)
            self._sectors[sector] = entry
        return entry

    @staticmethod
    def _memory_sector(bearing_deg: float, heading_deg: float) -> int:
        """把「相对当前朝向的方向」换成记忆系扇区。"""
        return sector_of(float(heading_deg) + float(bearing_deg))

    @staticmethod
    def _relative_bearing(sector: int, heading_deg: float) -> float:
        """把记忆系扇区换回「相对当前朝向」的角度，供上层直接使用。"""
        return normalize_bearing(sector_center_deg(sector) - float(heading_deg))

    def predict(
        self,
        scores: Mapping[str, float] | Mapping[float, float],
        now: float,
        *,
        heading_deg: float = 0.0,
    ) -> list[DirectionEvidence]:
        """写入主 LLM 的方向偏好。键可以是 left/forward/right，也可以是角度。

        分数被夹进 0~1。这里只存不判断：预测不会创建实测态，也不会覆盖已有实测态。
        """
        self._prune(now)
        touched: list[DirectionEvidence] = []
        for key, raw in scores.items():
            bearing = _bearing_of_key(key)
            if bearing is None:
                continue
            score = min(1.0, max(0.0, float(raw)))
            entry = self._entry(self._memory_sector(bearing, heading_deg))
            entry.predicted_score = score
            entry.prediction_expires_at = now + self.prediction_ttl_s
            # 换了新预测就重新开始记兑现情况，旧的结论不该套在新判断上。
            entry.prediction_outcome = None
            # 预测本身也要让记录活着，否则 TTL 到点会把还没验证的预测清掉。
            entry.expires_at = max(entry.expires_at, now + self.prediction_ttl_s)
            touched.append(entry)
        return touched

    def mark_dynamic_obstacle(
        self,
        bearings_deg: Iterable[float],
        now: float,
        *,
        present: bool = True,
        heading_deg: float = 0.0,
    ) -> None:
        """标记「那个方向有人」。附加代价，不参与可通行判断。"""
        self._prune(now)
        for bearing in bearings_deg:
            entry = self._entry(self._memory_sector(bearing, heading_deg))
            entry.dynamic_obstacle = bool(present)
            entry.expires_at = max(entry.expires_at, now + self.prediction_ttl_s)

    def record(self, outcome: SegmentOutcome, now: float) -> DirectionEvidence:
        """把一段移动的结果记进对应扇区，并结算预测兑现情况。

        锚点用 ``outcome.heading_deg``（该段**起始**时的 yaw），不是当前 yaw：这一段
        跑完时人可能已经被绕行转开了，用终点朝向反算会把证据记到隔壁扇区。
        """
        self._prune(now)
        entry = self._entry(
            self._memory_sector(outcome.bearing_deg, outcome.heading_deg)
        )

        cleared = outcome.cleared
        predicted = entry.live_prediction(now)
        if predicted is not None and (outcome.blocked or cleared):
            # 没有实测结论就不结算预测。把「不知道」算成 confirmed 会让
            # prediction_disagreements 反过来给主 LLM 的看图判断背书。
            leaned_free = predicted >= _PREDICTION_LEAN_FREE
            if outcome.blocked and leaned_free:
                entry.prediction_outcome = "false_positive"
            elif not outcome.blocked and predicted <= _PREDICTION_LEAN_BLOCKED:
                entry.prediction_outcome = "false_negative"
            else:
                entry.prediction_outcome = "confirmed"

        if outcome.blocked:
            entry.blocked_count += 1
            entry.empirical_state = "blocked"
        elif not cleared:
            # 没受阻，但也没证明走通（没速度证据，或门控积分不足）：留在 unknown，
            # 只刷新时间戳让记录活着。既不累计 cleared_count，也不覆盖这个扇区上
            # 已有的 blocked——没走出距离的一段不该把撞过两次的墙洗成可走。
            pass
        else:
            entry.cleared_count += 1
            # 走通过的方向不会因为一次受阻就永久降级，但也不能抹掉受阻历史：
            # blocked_count 保留，供上层判断这个方向有多不稳。
            entry.empirical_state = "verified_free"
        if outcome.progress_m is not None:
            entry.last_progress_m = max(0.0, float(outcome.progress_m))
        entry.turned_during_segment = bool(outcome.turned)
        entry.updated_at = now
        entry.expires_at = now + self.ttl_s
        self._sequence += 1
        return entry

    def state_of(
        self, bearing_deg: float, now: float, *, heading_deg: float = 0.0
    ) -> EmpiricalState:
        """查一个方向的**实测**状态；过期视为 unknown。预测不影响这个返回值。"""
        self._prune(now)
        entry = self._sectors.get(self._memory_sector(bearing_deg, heading_deg))
        return "unknown" if entry is None else entry.empirical_state

    def blocked_sectors(self, now: float) -> list[int]:
        self._prune(now)
        return sorted(
            s for s, e in self._sectors.items() if e.empirical_state == "blocked"
        )

    def should_refuse(
        self, bearing_deg: float, now: float, *, heading_deg: float = 0.0
    ) -> bool:
        """这个方向是否已确认封死，值得拒绝执行。

        只对 ``confident_block``（同一扇区撞过两次）返回真。单次受阻可能是被人挡了
        一下，拒绝会显得莫名其妙；而拒绝**不是**改方向——上层收到拒绝后应该把理由回给
        主 LLM 让它重选，不能静默换成别的方向。
        """
        self._prune(now)
        entry = self._sectors.get(self._memory_sector(bearing_deg, heading_deg))
        return entry is not None and entry.confident_block

    def clear_predictions(self) -> None:
        """清空预测，保留实测。视觉重启/目标明显变化时调用。

        实测记忆照旧靠 TTL 自然过期：撞过的墙不会因为主 LLM 换了个目标就消失。
        换世界要用 ``reset()``——那时连锚点本身都不再有意义。
        """
        for entry in self._sectors.values():
            entry.predicted_score = None
            entry.prediction_expires_at = 0.0
            entry.prediction_outcome = None
            entry.dynamic_obstacle = False

    def advice(self, now: float, *, heading_deg: float = 0.0) -> dict[str, Any]:
        """给主 LLM 的紧凑摘要。所有角度都换算成相对 ``heading_deg`` 的方向。

        刻意只报实测过的扇区：没记录不等于可走，让上层自己决定要不要试。把 unknown
        也列出来会让摘要看起来像一张地图，而这里没有地图。
        """
        self._prune(now)
        ordered = sorted(
            self._sectors,
            key=lambda s: self._relative_bearing(s, heading_deg),
        )
        entries = []
        for sector in ordered:
            snapshot = self._sectors[sector].snapshot(now)
            snapshot["bearing_deg"] = self._relative_bearing(sector, heading_deg)
            entries.append(snapshot)
        mispredicted = [
            {
                "bearing_deg": e["bearing_deg"],
                "predicted_score": self._sectors[e["sector"]].predicted_score,
                "empirical_state": e["empirical_state"],
                "prediction_outcome": e["prediction_outcome"],
            }
            for e in entries
            if e["prediction_outcome"] in ("false_positive", "false_negative")
        ]
        return {
            "sector_width_deg": _SECTOR_DEG,
            "ttl_ms": round(self.ttl_s * 1000.0, 1),
            "prediction_ttl_ms": round(self.prediction_ttl_s * 1000.0, 1),
            "turn_sign_convention": "positive_left_negative_right",
            # 扇区锚在调度器虚拟 HMD yaw 上，汇报时换算回相对当前朝向。上层看到的
            # 角度可以直接当 turn_deg 用，不需要自己补偿转过多少。
            "position_reference": "relative_to_current_heading",
            "heading_anchor_source": "scheduler_virtual_hmd",
            "world_localization_available": False,
            "route_choice_owner": "main_llm",
            "score_semantics": "direction_preference_not_traversability_probability",
            "min_progress_m": _MIN_PROGRESS_M,
            "records": entries,
            "blocked_bearings_deg": [
                self._relative_bearing(s, heading_deg)
                for s in self.blocked_sectors(now)
            ],
            "confident_block_bearings_deg": [
                e["bearing_deg"] for e in entries if e["confident_block"]
            ],
            "crowded_bearings_deg": [
                e["bearing_deg"] for e in entries if e["dynamic_obstacle"]
            ],
            # 预测和实测冲突的方向。主 LLM 用这个校准自己的看图判断。
            "prediction_disagreements": mispredicted,
        }

    def clear(self) -> None:
        self._sectors.clear()

    def reset(self) -> None:
        """换世界时清空。

        和 ``clear()`` 同义，保留这个名字是为了让调用点读起来说明「为什么清」：
        锚点是虚拟 HMD yaw，只在一个世界里连续；换了世界之后所有记录指向的都是
        上个世界里的墙。
        """
        self._sectors.clear()

    def _prune(self, now: float) -> None:
        expired = [s for s, e in self._sectors.items() if e.expires_at <= now]
        for sector in expired:
            del self._sectors[sector]


def integrate_progress(
    samples: Iterable[Mapping[str, Any]],
    *,
    min_forward_ratio: float = 0.9,
) -> tuple[float, bool]:
    """门控积分：只累计「确实在直行」的样本，返回 (米, 期间是否转向)。

    实机实测 ``forward_ratio`` 在贴墙滑行时会掉到 0.675 甚至更低，那段位移里有很大
    一部分是沿墙横移，算进前进距离会高估。转弯、贴墙、recover 和速度缺失的样本全部
    丢弃——宁可低估走了多远，也不要把横移记成推进。

    每个样本需要 ``speed``（m/s）、``forward_ratio``、``dt``（秒）；``turned`` 为真表示
    这一拍收到了新鲜的角速度样本。速度缺失不当作零速度，直接跳过该样本。
    """
    total = 0.0
    turned = False
    for sample in samples:
        if sample.get("turned"):
            # 转向拍直接丢弃，不看 forward_ratio。AngularY 只在移动期间更新，所以
            # 边走边转的样本 forward_ratio 仍可能很高，积分下去会把转弯弧长记成直行
            # 里程——而扇区标签本来就是按段起始朝向打的，那段位移不属于这个方向。
            turned = True
            continue
        speed = sample.get("speed")
        ratio = sample.get("forward_ratio")
        dt = sample.get("dt")
        if not isinstance(speed, (int, float)) or not isinstance(dt, (int, float)):
            continue
        if not isinstance(ratio, (int, float)) or ratio < min_forward_ratio:
            continue
        if dt <= 0.0:
            continue
        total += float(speed) * float(dt)
    return total, turned
