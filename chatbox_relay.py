"""把 N.E.K.O 宿主对话轮转成 VRChat 聊天框文本。

宿主通过 ``ctx.bus.conversations`` 把「她收到的指令」和「她说出口的回复」两类
记录抄给插件，本模块只转后者。这个区分不是可选的：

- ``proactive_instruction`` 是**输入**，不是发言。它承载的是用户的原话，转发
  它等于把一段私聊广播给 VRChat 里周围所有玩家。
- ``proactive_reply`` 是宿主已经确认提交的那句，也就是她真正说出口的话。

轮询而不是订阅，因为 conversations 是宿主独占写入的 store，宿主只提供按
``since_ts`` / ``limit`` 拉取的历史查询，没有推送回调。代价是必须自己维护
游标，且游标只前进不回补。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping

from .config import ChatboxRelayConfig


# 只有角色真正说出口的记录才允许进聊天框。宿主的 proactive 链路一次性写两条：
# 收到指令时写 ``proactive_instruction``，回复提交后写 ``proactive_reply``。
# 两者共用同一个 conversation_id，成对出现。
_SPEAKER_TURN_TYPES = frozenset({"proactive_reply"})


@dataclass(frozen=True)
class ChatboxLine:
    """一条已通过过滤、准备发送的聊天框文本。"""

    text: str
    record_id: str | None
    turn_type: str | None
    lanlan_name: str | None
    timestamp: float | None


def _record_field(record: Any, name: str) -> Any:
    """同时读 dataclass 字段与原始 dict。

    宿主 SDK 在不同版本里可能给 ``ConversationRecord`` 也可能给原始映射，这里
    两种都认，避免版本差异把整条链路变成静默丢弃。
    """
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _normalize_text(value: Any) -> str:
    """把记录正文压成单行。

    VRChat 的聊天框不渲染换行——多行文本会被折成一长条，而且换行符照样吃掉
    144 字符的额度。合并空白比让它们原样发出去更接近「她想说一句话」。
    """
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", "").split())


class ChatboxRelay:
    """轮询宿主对话轮并把角色发言转发到 VRChat 聊天框。"""

    def __init__(
        self,
        config: ChatboxRelayConfig,
        *,
        source: Callable[[int, float | None], Any],
        send: Callable[[str], tuple[bool, str | None]],
        logger: Any = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        # source(max_count, since_ts) -> 记录列表。注入而不是直接拿 ctx，
        # 这样转发器可以在没有宿主（独立运行）时被整体替换成空实现，
        # 也让它能在测试里不依赖 RPC。
        self._source = source
        # send(text) -> (accepted, reason)。真实实现走 OSC bridge /chatbox/input。
        self._send = send
        self.logger = logger
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # 已转发过的记录 id。游标之外再存一份是因为宿主的 since_ts 用的是记录
        # 自己的时间戳，而同一秒内可能落进多条记录；只靠时间会漏。
        self._seen_ids: deque[str] = deque(maxlen=512)
        self._seen_id_set: set[str] = set()
        self._last_seen_ts: float | None = None
        # 本次开启的水位：早于它的记录一律不转。它是「重新开启不回补历史」的
        # 最后一道闸——宿主对 since_ts 的过滤是尽力而为，不能只靠它。
        self._enabled_since_ts: float | None = None
        self._last_forwarded_at: float | None = None
        self._last_text: str | None = None
        self._poll_count = 0
        self._forwarded_count = 0
        self._skipped_count = 0
        self._truncated_count = 0
        self._send_failure_count = 0
        self._last_error: str | None = None
        self._last_source_error: str | None = None
        # 转发开关。配置给的是启动默认值；面板开关在运行期改这个字段。
        self._enabled = bool(config.enabled)
        # 构造即开启时水位也要立起来，否则第一次轮询会把 conversations store
        # 里已有的整段历史当成「新记录」发出去。
        if self._enabled:
            self._enabled_since_ts = self._wall_clock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """开关转发。

        每次从「关」切到「开」都把水位抬到此刻，这样关闭期间攒下的历史不会被
        补发——那会在她重新开口的瞬间一次性刷屏，而且其中可能有用户已经翻篇
        的内容。

        水位用记录自己的 timestamp 来比，而不是本机 wall clock：两者可能差一个
        时区或一次校时，拿本机的值当基准会让刚写下的记录因为时间戳更旧而被永久
        跳过。水位取本机时间与已见记录时间的较大者，避免宿主时钟略慢于本机时
        把新记录误判成旧记录。
        """
        with self._lock:
            was_enabled = self._enabled
            self._enabled = bool(enabled)
            if not self._enabled or was_enabled:
                return
            # 首次开启、或从关闭切回开启：立新水位。
            newest = self._last_seen_ts if self._last_seen_ts is not None else 0.0
            self._enabled_since_ts = max(float(newest), self._wall_clock())

    @property
    def thread_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        if self.thread_alive:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="neko-chatbox-relay", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    # ── 轮询与转发 ──────────────────────────────────────────────────

    def _remember(self, record_id: str) -> bool:
        """记录 id 首次出现时返回 True。"""
        with self._lock:
            if record_id in self._seen_id_set:
                return False
            if len(self._seen_ids) == self._seen_ids.maxlen:
                # deque 满时最旧的一项会被挤掉，集合必须跟着删，否则集合无限增长
                # 而且挤掉的那个 id 再也挡不住重复转发。
                oldest = self._seen_ids[0]
                self._seen_id_set.discard(oldest)
            self._seen_ids.append(record_id)
            self._seen_id_set.add(record_id)
            return True

    def _format(self, line: ChatboxLine) -> str:
        text = line.text
        if self.config.include_speaker and line.lanlan_name:
            text = f"{line.lanlan_name}: {text}"
        limit = self.config.max_chars
        if len(text) > limit:
            with self._lock:
                self._truncated_count += 1
            # 截断而不是丢弃：一句被砍掉尾巴的话仍然传达了「她在说话」，而完全
            # 不发等于这次对话在 VRChat 侧不存在。
            text = text[: max(0, limit - 1)] + "…"
        return text

    def _extract(self, record: Any) -> ChatboxLine | None:
        turn_type = _record_field(record, "turn_type")
        if turn_type not in _SPEAKER_TURN_TYPES:
            return None
        text = _normalize_text(_record_field(record, "content"))
        if not text:
            return None
        record_id = _record_field(record, "message_id") or _record_field(record, "id")
        return ChatboxLine(
            text=text,
            record_id=str(record_id) if record_id else None,
            turn_type=str(turn_type),
            lanlan_name=_record_field(record, "lanlan_name"),
            timestamp=_record_field(record, "timestamp"),
        )

    def _poll_once(self) -> None:
        with self._lock:
            if not self._enabled:
                return
            since_ts = self._last_seen_ts
        try:
            records = self._source(self.config.max_count, since_ts)
        except Exception as exc:
            # 宿主不可用（未连上 / RPC 超时）不是错误状态，下一轮重试即可。
            with self._lock:
                self._last_source_error = str(exc)[:500]
            return
        with self._lock:
            self._last_source_error = None
            self._poll_count += 1
        if not records:
            return
        for record in records:
            line = self._extract(record)
            timestamp = _record_field(record, "timestamp")
            is_newer = isinstance(timestamp, (int, float))
            with self._lock:
                if is_newer and (self._last_seen_ts is None or timestamp > self._last_seen_ts):
                    self._last_seen_ts = float(timestamp)
                # 水位之外再挡一道：宿主对 since_ts 的过滤是尽力而为，重新开启
                # 或重连后可能把整段历史一起送回来。不在这里挡住，那些旧话会在
                # 她重新开口的瞬间被一次性倾泻进聊天框。
                stale = (
                    is_newer
                    and self._enabled_since_ts is not None
                    and float(timestamp) < self._enabled_since_ts
                )
            if stale:
                with self._lock:
                    self._skipped_count += 1
                continue
            if line is None:
                with self._lock:
                    self._skipped_count += 1
                continue
            # 没有 id 就无法去重，只能按内容+时间判重；这类记录直接跳过，
            # 否则同一句话会在每个轮询周期里重复刷屏。
            if line.record_id is None:
                with self._lock:
                    self._skipped_count += 1
                continue
            if not self._remember(line.record_id):
                continue
            if not self.enabled:
                # 本轮中途被关掉，剩下的不再发。
                return
            payload = self._format(line)
            accepted, reason = self._send(payload)
            if accepted:
                with self._lock:
                    self._forwarded_count += 1
                    self._last_forwarded_at = self._wall_clock()
                    self._last_text = payload
                    self._last_error = None
            else:
                with self._lock:
                    self._send_failure_count += 1
                    self._last_error = str(reason or "send failed")[:500]

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            # 用 wait 而不是 sleep，停止时不必等满一个轮询周期。
            self._stop_event.wait(max(0.05, float(self.config.poll_interval_s)))

    # ── 诊断 ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "thread_alive": self.thread_alive,
                "poll_interval_s": self.config.poll_interval_s,
                "poll_count": self._poll_count,
                "forwarded_count": self._forwarded_count,
                "skipped_count": self._skipped_count,
                "truncated_count": self._truncated_count,
                "send_failure_count": self._send_failure_count,
                "last_forwarded_at_unix": self._last_forwarded_at,
                "last_text": self._last_text,
                "last_error": self._last_error,
                # 宿主不可用与发送失败是两件事：前者说明对话总线没连上，后者
                # 说明 OSC 没发出去。混成一个字段会让排查时选错方向。
                "last_source_error": self._last_source_error,
            }
