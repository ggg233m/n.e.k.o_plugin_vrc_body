"""把 OSC 速度样本对齐到 WGC 视觉帧时刻。

WGC 的 ``captured_at`` 和 OSC 接收时间都必须来自 ``time.monotonic()`` 同一时钟域。缺样本、跨越过大的间隔或旧样本不会被
补成零速度；调用方得到 ``available=false`` 后应保持安全暂停。
"""
from __future__ import annotations

from collections import deque
import math
import threading
from typing import Any


class TimeAlignmentBuffer:
    """保存少量带时间戳的速度，并按视觉帧时刻取样。"""

    def __init__(self, *, max_samples: int = 512, max_age_s: float = 0.25,
                 max_gap_s: float = 0.25) -> None:
        self.max_age_s = max(0.01, float(max_age_s))
        self.max_gap_s = max(0.01, float(max_gap_s))
        self._motions: deque[dict[str, float]] = deque(maxlen=max(16, int(max_samples)))
        self._lock = threading.RLock()

    @staticmethod
    def _finite(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _insert(cls, queue: deque[dict[str, float]], item: dict[str, float]) -> None:
        # 正常情况下样本单调到达；允许少量乱序，避免把离序样本静默接到末尾。
        if not queue or item["timestamp"] >= queue[-1]["timestamp"]:
            queue.append(item)
            return
        values = list(queue)
        values.append(item)
        values.sort(key=lambda entry: entry["timestamp"])
        queue.clear()
        queue.extend(values[-queue.maxlen:])

    def add_motion(self, timestamp: Any, velocity_x: Any, velocity_z: Any) -> bool:
        t = self._finite(timestamp)
        vx = self._finite(velocity_x)
        vz = self._finite(velocity_z)
        if t is None or vx is None or vz is None:
            return False
        with self._lock:
            self._insert(self._motions, {"timestamp": t, "velocity_x": vx, "velocity_z": vz})
        return True

    def _sample(self, queue: deque[dict[str, float]], captured_at: Any,
                fields: tuple[str, ...]) -> dict[str, Any]:
        t = self._finite(captured_at)
        if t is None:
            return {"available": False, "reason": "invalid_capture_timestamp", "method": "none"}
        with self._lock:
            values = list(queue)
        if not values:
            return {"available": False, "reason": "no_timestamped_samples", "method": "none"}
        before = None
        after = None
        for item in values:
            if item["timestamp"] <= t:
                before = item
            elif after is None:
                after = item
                break
        method = "none"
        gap = None
        if before is not None and after is not None:
            gap = after["timestamp"] - before["timestamp"]
            if gap <= self.max_gap_s:
                ratio = (t - before["timestamp"]) / gap if gap > 0 else 0.0
                result = {name: before[name] + (after[name] - before[name]) * ratio for name in fields}
                method = "interpolated"
            else:
                return {"available": False, "reason": "timestamp_gap", "method": "none",
                        "max_gap_s": round(gap, 6)}
        elif before is not None:
            age = t - before["timestamp"]
            if age < 0.0 or age > self.max_age_s:
                return {"available": False, "reason": "sample_stale", "method": "none",
                        "sample_age_s": round(max(0.0, age), 6)}
            result = {name: before[name] for name in fields}
            method = "held"
            gap = age
        else:
            return {"available": False, "reason": "capture_before_samples", "method": "none"}
        age = max(0.0, t - (before["timestamp"] if before else t))
        return {
            "available": True,
            "method": method,
            "timestamp": t,
            "sample_age_s": round(age, 6),
            "max_gap_s": round(float(gap or 0.0), 6),
            **result,
        }

    def sample_motion(self, captured_at: Any) -> dict[str, Any]:
        result = self._sample(self._motions, captured_at, ("velocity_x", "velocity_z"))
        if result.get("available"):
            result["value_age_ms"] = round(float(result.get("sample_age_s", 0.0)) * 1000.0, 1)
            result["horizontal_speed_mps"] = math.hypot(result["velocity_x"], result["velocity_z"])
            result["time_aligned"] = True
            result["clock_domain"] = "monotonic"
        else:
            result.update({"time_aligned": False, "clock_domain": "monotonic"})
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "clock_domain": "monotonic",
                "motion_samples": len(self._motions),
                "max_age_s": self.max_age_s,
                "max_gap_s": self.max_gap_s,
            }


__all__ = ["TimeAlignmentBuffer"]
