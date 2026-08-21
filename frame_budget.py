"""拉图的滑动窗口预算。

单独成模块有两个理由。一是它必须能被测到：窗口边界的 off-by-one 是这类计数器
的经典 bug，而 ``__init__.py`` 需要 SDK 才能导入，测试里进不去。二是它要挡的
东西很具体——agent 每回合都拉一张图。一张 960 px 的 JPEG 进上下文大约十万字符
量级的 base64，循环里拉几轮就能把会话挤爆，成本也跟着走。

刻意不做令牌桶：桶允许攒额度，于是「安静十分钟」之后能一口气连拉十张，恰好是
最该拦住的那种突发。滑动窗口在任何一分钟内都只放行固定张数。
"""

from __future__ import annotations

from collections import deque
from typing import Any


__all__ = ["FrameBudget"]

_WINDOW_S = 60.0


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


class FrameBudget:
    """每 ``window_s`` 秒最多放行 ``limit`` 次。

    调用方自带时钟（``now`` 由外部传入）：这样测试不用 sleep，也不用给整个插件
    塞一个可注入的时钟。``limit <= 0`` 表示完全关闭拉图，不是「不限量」——把
    上限写成 0 却得到无限次，是这类开关最容易被误读的方向。
    """

    def __init__(self, limit: int = 10, *, window_s: float = _WINDOW_S) -> None:
        self.limit = max(0, _positive_int(limit, 10))
        try:
            self.window_s = max(1.0, float(window_s))
        except (TypeError, ValueError, OverflowError):
            self.window_s = _WINDOW_S
        self._calls: deque[float] = deque()

    def _evict(self, now: float) -> None:
        horizon = now - self.window_s
        while self._calls and self._calls[0] <= horizon:
            self._calls.popleft()

    def check(self, now: float) -> dict[str, Any]:
        """只看不取：返回当前是否放行，以及还要等多久。"""
        if self.limit <= 0:
            return {
                "allowed": False,
                "used": 0,
                "limit": 0,
                "retry_after_ms": None,
                "reason": "frame_pull_disabled",
            }
        self._evict(now)
        used = len(self._calls)
        if used < self.limit:
            return {"allowed": True, "used": used, "limit": self.limit, "retry_after_ms": 0}
        # 最早那次调用滑出窗口的时刻，就是下一次能放行的时刻。
        wait_s = max(0.0, self._calls[0] + self.window_s - now)
        return {
            "allowed": False,
            "used": used,
            "limit": self.limit,
            "retry_after_ms": int(round(wait_s * 1000.0)),
            "reason": "frame_rate_limited",
        }

    def consume(self, now: float) -> dict[str, Any]:
        """放行时记账并返回结果；被拒时不记账。

        被拒的调用不能计数，否则反复重试会把窗口无限续期，形成一个永远解不开的
        限流——本意只是让 agent 慢下来，不是把它锁死。
        """
        verdict = self.check(now)
        if verdict["allowed"]:
            self._calls.append(now)
            verdict = dict(verdict)
            verdict["used"] = len(self._calls)
        return verdict

    def status(self, now: float) -> dict[str, Any]:
        if self.limit > 0:
            self._evict(now)
        return {
            "limit": self.limit,
            "used_last_window": len(self._calls),
            "window_s": self.window_s,
        }
