"""聊天链路的脱敏结构化日志；不持有消息正文和请求载荷。"""

from __future__ import annotations

from collections import deque
import json
import os
import re
import threading
import time
from typing import Any


class PipelineDiagnostics:
    """通过固定字段白名单记录阶段；重复状态最多每三十秒输出一次。"""

    _NUMBER_FIELDS = frozenset({
        "session", "event_session", "player_slot", "submit_seq", "log_seq",
        "reply_serial", "transfer_sequence", "display_seconds", "pages",
        "chars", "latency_ms", "attempt", "reply_baseline_serial",
    })
    _BOOL_FIELDS = frozenset({
        "discovery_ready", "world_ui_ready", "reply_end", "ready",
        "preserve_midi", "midi_open", "manual_disconnect", "engagement_applied",
    })
    _CODE_FIELDS = frozenset({"reason", "error", "status", "source", "control_state"})

    def __init__(self, logger: Any, *, clock=time.monotonic) -> None:
        self._logger = logger
        self._clock = clock
        self._lock = threading.RLock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=64)
        self._states: dict[str, tuple[str, float, int]] = {}
        self._runtime_id = f"{os.getpid()}-{time.time_ns():x}"

    def emit(self, event: str, *, deduplicate: bool = False, **fields: Any) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_.]{0,63}", event) is None:
            return
        safe: dict[str, Any] = {}
        for key, value in fields.items():
            if key in self._NUMBER_FIELDS and type(value) is int:
                safe[key] = value
            elif key in self._BOOL_FIELDS and type(value) is bool:
                safe[key] = value
            elif key in self._CODE_FIELDS and isinstance(value, str):
                # 错误只接受短代码，异常正文、URL、Authorization 等一律丢弃。
                safe[key] = (
                    value if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value)
                    else "redacted"
                )
        with self._lock:
            now = self._clock()
            signature = json.dumps(safe, sort_keys=True)
            previous = self._states.get(event)
            suppressed = 0
            if deduplicate:
                if previous is not None and previous[0] == signature:
                    if now - previous[1] < 30:
                        self._states[event] = (signature, previous[1], previous[2] + 1)
                        return
                    suppressed = previous[2]
                self._states[event] = (signature, now, 0)
            record = {
                "event": event, "runtime_id": self._runtime_id,
                "at_unix_ms": time.time_ns() // 1_000_000, **safe,
            }
            if suppressed:
                record["suppressed"] = suppressed
            self._recent.append(record)
        try:
            if self._logger is not None:
                self._logger.info("YUI_DIAG %s", json.dumps(record, sort_keys=True))
        except Exception:
            # 日志设备故障不能阻止玩家消息进入宿主。
            pass

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._recent]
