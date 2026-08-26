"""基于连续画面的局部可通行性预测。

这个模块只回答一个几何问题：*沿当前视线继续走，近处是否出现了明显的
光流发散？* 它不识别人、不创建世界实体，也不把未知画面当成空地。

实现优先使用 OpenCV 的稠密 Farneback 光流；当前独立后端环境可能没有 cv2，
所以保留一个只依赖 numpy 的小型 Lucas--Kanade 后备实现。两条路径都只保存
160x90 的灰度副本，调用方停止 worker 时即可随对象一起释放，不写磁盘。
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import time
from typing import Any, Callable, Mapping


# 扇区范围受采集 FOV 限制，不能随意加方向。``horizontal_fov_deg=90`` 下画面最
# 边缘一列对应的方位角只有 ±45°，而扇区按最近邻分配——±60° 需要方位角 > 45°，
# 那样的像素**不存在**。2026-08-26 实测：曾经的 ±60° 两个扇区分到 0 列，
# numpy LK(step=8) 与 cv2 Farneback(step=4) 都是 0，与取样步长无关，
# 于是它们恒为 unknown。改 FOV 或宽高比后此处需重算。
_DEFAULT_SECTORS = (-30.0, 0.0, 30.0)


@dataclass(frozen=True)
class TraversabilityConfig:
    """光流估计的有界参数；数值是相对风险阈值，不是米制距离。"""

    width: int = 160
    height: int = 90
    horizontal_fov_deg: float = 90.0
    roi_top_ratio: float = 0.18
    roi_bottom_ratio: float = 0.96
    min_feature_count: int = 8
    min_texture: float = 18.0
    max_flow_px: float = 24.0
    # 每秒归一化发散率（1/s）。判定必须用每秒量：采集间隔并不恒定（目标
    # 100ms，实测天花板约 6.5 帧/秒，掉帧时更长），用「每帧」位移做阈值会让
    # 同一面墙在流畅时判 clear、卡顿时判 blocked。数值仍是相对风险刻度，
    # 不能解释成米制距离；倒数量级上约等于「多少秒后接触」。
    blocked_expansion_rate_per_s: float = 0.55
    clear_expansion_rate_per_s: float = 0.14
    min_confidence_for_state: float = 0.55
    max_frame_gap_s: float = 1.0


@dataclass(frozen=True)
class GroundExtentConfig:
    """单帧地面可见范围的有界参数。

    输出是**序数**的：只回答「哪个方向的地面延伸得更远」，不换算成米。要换算
    需要相机俯仰角和眼高，前者我们只知道自己发出的指令、不知道 VRChat 实际
    呈现的角度，后者随 avatar 变；两个都不可靠，相乘之后更不可靠。
    """

    width: int = 160
    height: int = 90
    # 必须与 TraversabilityConfig.horizontal_fov_deg 一致：两者读的是同一个
    # 采集区域，真实 FOV 只有一个值。此前这里写 120 是为了让 ±60° 扇区收到列，
    # 那是**为迁就扇区表而假造 FOV**，代价是每一列的 bearing_deg 都偏大 1/3
    # ——扇区收缩到 ±30 之后这条理由不再存在。
    horizontal_fov_deg: float = 90.0
    # 只扫画面下半部分。地平线以上不可能是脚下的地面，扫进去只会让天空、
    # 远景墙面这些大片同质区域伪装成「很远的地板」。
    scan_top_ratio: float = 0.52
    # 最底部几行通常是 avatar 自己的身体或 HUD，不是地面。
    scan_bottom_ratio: float = 0.97
    # 同质性阈值：相邻扫描行与地面参考色的最大灰度差。超过就认为地面在这里
    # 结束。VRChat 世界的地板材质千差万别，所以这是个粗判据，只用于比较
    # 扇区之间的相对远近，不做绝对判定。
    max_ground_deviation: float = 26.0
    # 参考色取样行数：从底部往上取这么多行的中位数作为「地面长什么样」。
    reference_rows: int = 4
    # 一个扇区里至少要被分到这么多列，才给出读数。注意它比的是**几何分配**的
    # 列数（每列都会产出一个 extent），不是「成功找到边界」的列数——后者由
    # confidence 的 edge_located_columns 单独表达。
    min_columns_per_sector: int = 3
    # 地面纹理过于均匀时（纯色地板+纯色墙），边界判据会失效。参考区域的
    # 灰度标准差低于这个值就报 unknown，而不是报「地面延伸到天边」。
    min_reference_texture: float = 2.5


def _unknown(
    *,
    now: float,
    captured_at: float | None,
    reason: str,
    backend: str = "unavailable",
    turning: bool = False,
    moving: bool | None = None,
    sectors: tuple[float, ...] = _DEFAULT_SECTORS,
) -> dict[str, Any]:
    """构造保守结果：所有方向都未知，绝不把缺数据写成畅通。"""
    return {
        "available": False,
        "source": "optical_flow",
        "backend": backend,
        "state": "unknown",
        "reason": reason,
        "captured_at_monotonic": captured_at,
        "age_ms": None if captured_at is None else round(max(0.0, now - captured_at) * 1000.0, 1),
        "turning": bool(turning),
        "moving": moving,
        "feature_count": 0,
        "sectors": [
            {
                "bearing_deg": float(bearing),
                "state": "unknown",
                "free_score": None,
                "confidence": 0.0,
                "feature_count": 0,
                "expansion_rate_per_s": None,
                "contact_time_s": None,
            }
            for bearing in sectors
        ],
        "time_semantics": "monocular_optical_flow_ttc_estimate",
    }


def _as_gray(frame: Any, width: int, height: int) -> Any | None:
    """把 ndarray/PIL 风格帧复制成固定大小 uint8 灰度图。"""
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        array = np.asarray(frame)
    except Exception:
        return None
    if array.ndim == 0 or array.size == 0:
        return None
    if array.ndim == 3:
        if array.shape[2] < 3:
            array = array[..., 0]
        else:
            # RGB/BGR 的差别对光流几何影响很小，使用标准亮度组合即可。
            array = (
                array[..., 0].astype(np.float32) * 0.299
                + array[..., 1].astype(np.float32) * 0.587
                + array[..., 2].astype(np.float32) * 0.114
            )
    elif array.ndim != 2:
        return None
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] < 2 or array.shape[1] < 2:
        return None
    finite = np.isfinite(array)
    if not finite.all():
        array = np.where(finite, array, 0.0)
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        return np.zeros((height, width), dtype=np.uint8)
    if low < 0.0 or high > 255.0:
        array = (array - low) * (255.0 / max(high - low, 1e-6))
    # 最近邻取样不保存原图，也不会在 worker 队列外留住采集对象。
    ys = np.linspace(0, array.shape[0] - 1, max(1, int(height))).astype(np.int32)
    xs = np.linspace(0, array.shape[1] - 1, max(1, int(width))).astype(np.int32)
    return np.clip(array[ys[:, None], xs[None, :]], 0.0, 255.0).astype(np.uint8)


def _lucas_kanade_flow(previous: Any, current: Any, config: TraversabilityConfig) -> list[tuple[float, float, float, float, float]]:
    """只依赖 numpy 的稀疏块 Lucas--Kanade 光流。

    返回 ``(x, y, u, v, texture)``。纹理不足的块直接跳过；这让白墙结果变成
    unknown，而不是被错误解释成没有障碍。
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return []
    previous_f = previous.astype(np.float32, copy=False)
    current_f = current.astype(np.float32, copy=False)
    gy, gx = np.gradient(previous_f)
    it = current_f - previous_f
    height, width = previous_f.shape
    result: list[tuple[float, float, float, float, float]] = []
    radius = 3
    step = 8
    sample_count = float((2 * radius + 1) ** 2)
    top = max(radius + 1, int(height * config.roi_top_ratio))
    bottom = min(height - radius - 1, int(height * config.roi_bottom_ratio))
    for y in range(top, bottom + 1, step):
        for x in range(radius + 1, width - radius - 1, step):
            ys = slice(y - radius, y + radius + 1)
            xs = slice(x - radius, x + radius + 1)
            ix = gx[ys, xs].reshape(-1)
            iy = gy[ys, xs].reshape(-1)
            dt = it[ys, xs].reshape(-1)
            sxx = float(np.dot(ix, ix))
            syy = float(np.dot(iy, iy))
            sxy = float(np.dot(ix, iy))
            # 除以样本数才和 _opencv_flow 的局部方差同量纲。不归一化的话
            # sxx+syy 会大出两个数量级，白墙保护在 numpy 后备上等于没有——
            # 而没有 cv2 时走的正是这条路径。
            texture = (sxx + syy) / sample_count
            determinant = sxx * syy - sxy * sxy
            if texture < config.min_texture:
                continue
            # determinant 是梯度四次方量纲，不能复用 min_texture。这里只要求
            # 结构张量可逆：沿单一边缘的块（孔径问题）det 会塌到 0，此时解出
            # 的位移只有法向分量，不能当作可信光流。
            if determinant <= max(1e-6, texture * texture * 1e-3):
                continue
            sx_t = float(np.dot(ix, dt))
            sy_t = float(np.dot(iy, dt))
            # [Ix Iy] [u v] = -It
            u = (-syy * sx_t + sxy * sy_t) / determinant
            v = (sxy * sx_t - sxx * sy_t) / determinant
            if not math.isfinite(u) or not math.isfinite(v):
                continue
            if abs(u) > config.max_flow_px or abs(v) > config.max_flow_px:
                continue
            result.append((float(x), float(y), float(u), float(v), texture))
    return result


def _opencv_flow(previous: Any, current: Any, config: TraversabilityConfig) -> list[tuple[float, float, float, float, float]]:
    """OpenCV 稠密光流适配；不可用或失败时交给 numpy 后备。"""
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            2,
            15,
            2,
            5,
            1.1,
            0,
        )
        height, width = previous.shape
        result: list[tuple[float, float, float, float, float]] = []
        top = max(2, int(height * config.roi_top_ratio))
        bottom = min(height - 2, int(height * config.roi_bottom_ratio))
        for y in range(top, bottom, 4):
            for x in range(2, width - 2, 4):
                u, v = flow[y, x]
                magnitude = float(np.hypot(u, v))
                if not math.isfinite(magnitude) or magnitude > config.max_flow_px:
                    continue
                # 局部梯度能量只是“这点有多少可跟踪纹理”，不是障碍置信度。
                patch = previous[max(0, y - 2):y + 3, max(0, x - 2):x + 3].astype(np.float32)
                texture = float(np.var(patch))
                if texture < config.min_texture:
                    continue
                result.append((float(x), float(y), float(u), float(v), texture))
        return result
    except Exception:
        return []


class OpticalFlowTraversability:
    """连续帧局部可通行性预测器。

    预测器不选择方向、不发送控制器输入。``turning`` 和 ``moving`` 由外部
    反馈门控；无法判断时保守返回 unknown。
    """

    def __init__(
        self,
        config: TraversabilityConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sectors: tuple[float, ...] = _DEFAULT_SECTORS,
    ) -> None:
        self.config = config or TraversabilityConfig()
        self._clock = clock
        self._sectors = tuple(float(item) for item in sectors) or _DEFAULT_SECTORS
        self._previous_gray: Any | None = None
        self._previous_at: float | None = None
        self._last_backend = "numpy_lucas_kanade"

    @staticmethod
    def dependency_available() -> dict[str, bool]:
        """报告可选实现，不在 import 时加载 OpenCV。"""
        try:
            numpy_available = importlib.util.find_spec("numpy") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            numpy_available = False
        try:
            opencv_available = importlib.util.find_spec("cv2") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            opencv_available = False
        return {"numpy": numpy_available, "opencv": opencv_available}

    def reset(self) -> None:
        """场景切换、遮挡或时间跳跃后丢弃上一帧参考。"""
        self._previous_gray = None
        self._previous_at = None

    def estimate(
        self,
        frame: Any,
        *,
        captured_at: float | None = None,
        moving: bool | None = None,
        turning: bool = False,
        now: float | None = None,
    ) -> dict[str, Any]:
        current_at = self._clock() if captured_at is None else float(captured_at)
        current_now = self._clock() if now is None else float(now)
        gray = _as_gray(frame, self.config.width, self.config.height)
        if gray is None:
            self.reset()
            return _unknown(now=current_now, captured_at=captured_at, reason="frame_unreadable", sectors=self._sectors)

        previous = self._previous_gray
        previous_at = self._previous_at
        self._previous_gray = gray
        self._previous_at = current_at
        if previous is None or previous_at is None:
            return _unknown(
                now=current_now,
                captured_at=current_at,
                reason="warmup",
                moving=moving,
                sectors=self._sectors,
            )
        gap = current_at - previous_at
        if gap <= 0.0 or gap > self.config.max_frame_gap_s:
            return _unknown(
                now=current_now,
                captured_at=current_at,
                reason="frame_gap",
                moving=moving,
                sectors=self._sectors,
            )
        if turning:
            return _unknown(now=current_now, captured_at=current_at, reason="turning", turning=True, moving=moving, sectors=self._sectors)
        if moving is not True:
            return _unknown(now=current_now, captured_at=current_at, reason="motion_gate_unknown", moving=moving, sectors=self._sectors)

        flows = _opencv_flow(previous, gray, self.config)
        if flows:
            self._last_backend = "opencv_farneback"
        else:
            flows = _lucas_kanade_flow(previous, gray, self.config)
            self._last_backend = "numpy_lucas_kanade"
        if len(flows) < self.config.min_feature_count:
            return _unknown(now=current_now, captured_at=current_at, reason="insufficient_features", backend=self._last_backend, moving=True, sectors=self._sectors)

        cx = (self.config.width - 1.0) / 2.0
        cy = self.config.height * 0.52
        grouped: dict[float, list[float]] = {bearing: [] for bearing in self._sectors}
        for x, y, u, v, _texture in flows:
            dx = x - cx
            dy = y - cy
            radius_sq = dx * dx + dy * dy
            if radius_sq < 25.0:
                continue
            # 正值表示从视野中心向外扩张；以“每帧比例”表达，避免伪造米制。
            expansion = (u * dx + v * dy) / radius_sq
            bearing = (cx - x) / max(cx, 1.0) * (self.config.horizontal_fov_deg / 2.0)
            target = min(self._sectors, key=lambda item: abs(item - bearing))
            if math.isfinite(expansion):
                grouped[target].append(float(expansion))

        total_count = sum(len(values) for values in grouped.values())
        if total_count < self.config.min_feature_count:
            return _unknown(now=current_now, captured_at=current_at, reason="insufficient_radial_features", backend=self._last_backend, moving=True, sectors=self._sectors)

        sector_results: list[dict[str, Any]] = []
        blocked_sector_count = 0
        for bearing in self._sectors:
            values = grouped[bearing]
            if not values:
                sector_results.append({
                    "bearing_deg": bearing,
                    "state": "unknown",
                    "free_score": None,
                    "confidence": 0.0,
                    "feature_count": 0,
                    "expansion_rate_per_s": None,
                    "contact_time_s": None,
                })
                continue
            # 先除以帧间隔换算成每秒量，之后的一致性、阈值和置信度都在同一个
            # 与帧率无关的刻度上比较。
            values = sorted(value / gap for value in values)
            middle = len(values) // 2
            median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
            deviations = sorted(abs(value - median) for value in values)
            mad = deviations[len(deviations) // 2]
            count_confidence = min(1.0, len(values) / max(float(self.config.min_feature_count * 2), 1.0))
            # 离散度容许量同样取每秒刻度，与 blocked 阈值挂钩而不是写死常数。
            consistency = max(0.0, min(1.0, 1.0 - mad / max(self.config.blocked_expansion_rate_per_s, 1e-6)))
            confidence = round(count_confidence * consistency, 3)
            risk = max(0.0, median) / max(self.config.blocked_expansion_rate_per_s, 1e-6)
            risk = max(0.0, min(1.0, risk))
            free_score = round(max(0.0, min(1.0, 1.0 - risk)), 3)
            if confidence < self.config.min_confidence_for_state:
                state = "unknown"
            elif median >= self.config.blocked_expansion_rate_per_s:
                state = "predicted_blocked"
                blocked_sector_count += 1
            elif median <= self.config.clear_expansion_rate_per_s:
                state = "predicted_clear"
            else:
                state = "unknown"
            contact_time = None
            if median > 1e-5:
                contact_time = round(min(30.0, 1.0 / median), 3)
            sector_results.append({
                "bearing_deg": bearing,
                "state": state,
                "free_score": free_score,
                "confidence": confidence,
                "feature_count": len(values),
                "expansion_rate_per_s": round(median, 6),
                "contact_time_s": contact_time,
            })

        overall_state = "predicted_blocked" if blocked_sector_count else "predicted_clear"
        return {
            "available": True,
            "source": "optical_flow",
            "backend": self._last_backend,
            "state": overall_state,
            "reason": "ok",
            "captured_at_monotonic": current_at,
            "age_ms": round(max(0.0, current_now - current_at) * 1000.0, 1),
            "turning": False,
            "moving": True,
            "feature_count": total_count,
            "sectors": sector_results,
            "time_semantics": "monocular_optical_flow_ttc_estimate",
        }


class GroundExtentEstimator:
    """单帧地面可见范围估计。补光流的盲区：站着不动时也有输出。

    光流需要 ``moving=True``，而**选方向的那一刻恰好站着不动**——这个类填的
    就是那个空档。它同样不选方向、不发指令。

    与光流的关键区别：这条**不接安全门**。光流的每秒发散率经过标定
    （3m 处 0.393/s 贴合理论 1/TTC），可以用来停车；地面边界是单帧启发式，
    失效模式多得多（地板与墙同色、强光溢出、地毯花纹、栏杆下方透视到远处
    地面），只能作为给主 LLM 的序数参考。把它接进停车判据会让 agent 在
    花纹地毯上莫名停住。
    """

    def __init__(
        self,
        config: GroundExtentConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sectors: tuple[float, ...] = _DEFAULT_SECTORS,
    ) -> None:
        self.config = config or GroundExtentConfig()
        self._clock = clock
        self._sectors = tuple(float(item) for item in sectors) or _DEFAULT_SECTORS

    def _unknown(self, reason: str, captured_at: float | None, now: float) -> dict[str, Any]:
        return {
            "available": False,
            "source": "ground_extent",
            "state": "unknown",
            "reason": reason,
            "captured_at_monotonic": captured_at,
            "age_ms": None if captured_at is None else round(max(0.0, now - captured_at) * 1000.0, 1),
            "sectors": [
                {
                    "bearing_deg": float(bearing),
                    "state": "unknown",
                    "extent_ratio": None,
                    "confidence": 0.0,
                    "column_count": 0,
                    "edge_located_columns": 0,
                }
                for bearing in self._sectors
            ],
            "extent_semantics": "ordinal_visible_floor_span_not_metric_distance",
        }

    def estimate(
        self,
        frame: Any,
        *,
        captured_at: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """估计各扇区的地面可见范围。

        不需要 ``moving``/``turning`` 门控：单帧判据，转向中和静止时同样有效。
        """
        current_at = self._clock() if captured_at is None else float(captured_at)
        current_now = self._clock() if now is None else float(now)
        try:
            import numpy as np  # type: ignore[import-not-found]
        except Exception:
            return self._unknown("numpy_unavailable", captured_at, current_now)

        gray = _as_gray(frame, self.config.width, self.config.height)
        if gray is None:
            return self._unknown("frame_unreadable", captured_at, current_now)

        height, width = gray.shape
        top = max(0, int(height * self.config.scan_top_ratio))
        bottom = min(height - 1, int(height * self.config.scan_bottom_ratio))
        reference_top = max(top, bottom - self.config.reference_rows + 1)
        if bottom - top < 4:
            return self._unknown("scan_band_too_thin", captured_at, current_now)

        reference_band = gray[reference_top : bottom + 1, :].astype(np.float32)
        if reference_band.size == 0:
            return self._unknown("scan_band_too_thin", captured_at, current_now)
        # 纹理过低说明地板和其它表面无法区分；此时边界判据会把整个画面读成
        # 地面，必须报 unknown 而不是「一路畅通」。
        if float(np.std(reference_band)) < self.config.min_reference_texture:
            return self._unknown("insufficient_ground_texture", captured_at, current_now)

        span = float(bottom - top)
        grouped: dict[float, list[float]] = {bearing: [] for bearing in self._sectors}
        # 每扇区里**在扫描带内真正定位到边界**的列数。confidence 由它算，不用
        # 边界离散度：离散度测的是「这个方向的几何有多平坦」，与读数可信度无关，
        # 而且方向正好相反。2026-08-26 实测——延伸的木地板走道因透视收敛的板缝
        # 离散度高，得 0.415；正前方齐平的广告墙边界一致，得 0.902。extent_ratio
        # 的排序全对（0.85 vs 0.22），confidence 却把最该信的方向标成最不可信。
        #
        # 为什么不是「参考色与脚边地板匹配的列数占比」（本来想这么写）：每列的
        # 参考色取自**该列自己**的底部，脚边参考色也来自同一批底部像素。广告墙
        # 填满画面下沿时两边都是墙漆，完美一致，占比恒为 1.0——它在唯一该报警的
        # 场景里自证正确。已实测确认该判据恒 1.0，不可用。
        #
        # 饱和的 extent=1.0 意味着一路扫到带顶都没遇到边界，那个 1.0 是**扫描带
        # 上限**而不是测量值，所以记 0 分。代价说明白：开阔地板与均匀墙面都是
        # 无边界的，两者同样得低分。纯颜色判据确实分不开这两者，低分是诚实的读法，
        # 不是缺陷——要分开需要深度，而检测器没有深度头。
        located: dict[float, int] = {bearing: 0 for bearing in self._sectors}
        half_fov = self.config.horizontal_fov_deg / 2.0
        cx = (width - 1.0) / 2.0
        for x in range(width):
            reference = float(np.median(reference_band[:, x]))
            boundary_row = top
            edge_found = False
            for y in range(bottom, top - 1, -1):
                if abs(float(gray[y, x]) - reference) > self.config.max_ground_deviation:
                    boundary_row = y
                    edge_found = True
                    break
                boundary_row = y
            # 归一化成 0~1 的序数刻度：1 表示地面一直延伸到扫描带顶端，
            # 0 表示脚边就断了。这是画面上的跨度比例，不是距离。
            extent = (bottom - boundary_row) / max(span, 1.0)
            bearing = (cx - x) / max(cx, 1.0) * half_fov
            target = min(self._sectors, key=lambda item: abs(item - bearing))
            grouped[target].append(max(0.0, min(1.0, extent)))
            if edge_found:
                located[target] += 1

        sector_results: list[dict[str, Any]] = []
        usable = 0
        for bearing in self._sectors:
            values = grouped[bearing]
            if len(values) < self.config.min_columns_per_sector:
                sector_results.append({
                    "bearing_deg": bearing,
                    "state": "unknown",
                    "extent_ratio": None,
                    "confidence": 0.0,
                    "column_count": len(values),
                    "edge_located_columns": located[bearing],
                })
                continue
            values.sort()
            middle = len(values) // 2
            median = (
                values[middle]
                if len(values) % 2
                else (values[middle - 1] + values[middle]) / 2.0
            )
            # 这个读数由多少列支撑：在带内真正找到边界的列占比。低分意味着
            # 「大多数列没看到任何边界」，此时 extent_ratio 只是扫描带上限。
            confidence = round(max(0.0, min(1.0, located[bearing] / len(values))), 3)
            usable += 1
            sector_results.append({
                "bearing_deg": bearing,
                "state": "measured",
                "extent_ratio": round(median, 4),
                "confidence": confidence,
                "column_count": len(values),
                "edge_located_columns": located[bearing],
            })

        if not usable:
            return self._unknown("no_usable_sector", captured_at, current_now)

        return {
            "available": True,
            "source": "ground_extent",
            "state": "measured",
            "reason": "ok",
            "captured_at_monotonic": current_at,
            "age_ms": round(max(0.0, current_now - current_at) * 1000.0, 1),
            "sectors": sector_results,
            # 刻意不给 state=clear/blocked：这条不做通行判定，只排序。
            "extent_semantics": "ordinal_visible_floor_span_not_metric_distance",
            "advisory_only": True,
        }


__all__ = [
    "GroundExtentConfig",
    "GroundExtentEstimator",
    "OpticalFlowTraversability",
    "TraversabilityConfig",
]
