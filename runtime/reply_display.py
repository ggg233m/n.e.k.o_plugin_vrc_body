"""把 N.E.K.O 已完成的主对话回答安全投影到 YUI NPC 头顶。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
import hashlib
import math
import threading
import time
from typing import Any

from .chat_context import RecentChatContextProvider
from .config import YuiChatBridgeConfig


DisplayCallback = Callable[[str, int], Mapping[str, Any]]
ConversationFetcher = Callable[[], object]


class MainReplyDisplayBridge:
    """轮询只读回复源；只做显示，不产生新回复或 TTS。

    普通用户回合从 ``recent.json`` 读取。插件 ``respond`` 产生的主动回合不会
    立即写入该文件，因此额外读取宿主已经存在的 ``proactive_reply`` 记录。
    这里没有扩展总线协议，也不会把聊天内容写回任何存储。
    """

    _PAGE_BODY_BYTES = 370
    _READING_CHARS_PER_SECOND = 6
    _POST_READ_SECONDS = 6
    _MAX_ADAPTIVE_DISPLAY_SECONDS = 30

    def __init__(
        self,
        provider: RecentChatContextProvider,
        config: YuiChatBridgeConfig,
        display_callback: DisplayCallback,
        *,
        conversation_fetcher: ConversationFetcher | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.provider = provider
        self.config = config
        self._display_callback = display_callback
        self._conversation_fetcher = conversation_fetcher
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_ready = False
        self._seen_revision: str | None = None
        self._pending_pages: list[tuple[str, bool]] = []
        self._next_page_at = 0.0
        self._displayed_replies = 0
        self._displayed_pages = 0
        self._last_display_seconds: int | None = None
        self._last_result = "idle"
        self._last_error: str | None = None
        self._bus_baseline_at = self._wall_clock()
        self._bus_baseline_ready = conversation_fetcher is None
        self._bus_seen_order: deque[str] = deque()
        self._bus_seen: set[str] = set()
        self._bus_last_error: str | None = None
        self._bus_last_timestamp: float | None = None
        self._bus_records_seen = 0
        self._recent_content_hashes: dict[str, float] = {}

    @staticmethod
    def _take_utf8_prefix(text: str, maximum_bytes: int) -> tuple[str, str]:
        used = 0
        end = 0
        for index, char in enumerate(text):
            size = len(char.encode("utf-8"))
            if used + size > maximum_bytes:
                break
            used += size
            end = index + 1
        return text[:end], text[end:]

    @classmethod
    def paginate(cls, text: str, max_pages: int) -> list[str]:
        remaining = text.strip()
        if not remaining:
            return []
        bodies: list[str] = []
        while remaining and len(bodies) < max_pages:
            body, remaining = cls._take_utf8_prefix(remaining, cls._PAGE_BODY_BYTES)
            if not body:
                break
            bodies.append(body)
        if remaining and bodies:
            body, _ = cls._take_utf8_prefix(
                bodies[-1], cls._PAGE_BODY_BYTES - len("…".encode("utf-8")),
            )
            bodies[-1] = body.rstrip() + "…"
        if len(bodies) <= 1:
            return bodies
        total = len(bodies)
        return [f"({index}/{total}) {body}" for index, body in enumerate(bodies, 1)]

    @classmethod
    def display_duration(cls, text: str, minimum_seconds: int) -> int:
        """按可见字符估算阅读时间，同时保留配置的最短显示时长。"""

        visible_characters = sum(1 for char in text if not char.isspace())
        adaptive = (
            math.ceil(visible_characters / cls._READING_CHARS_PER_SECOND)
            + cls._POST_READ_SECONDS
        )
        return max(
            minimum_seconds,
            min(cls._MAX_ADAPTIVE_DISPLAY_SECONDS, adaptive),
        )

    def start(self) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                name="yui-main-reply-display",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            self._thread = None
            self._pending_pages.clear()

    @staticmethod
    def _field(record: object, name: str) -> object:
        if isinstance(record, Mapping):
            direct = record.get(name)
            metadata = record.get("metadata")
        else:
            direct = getattr(record, name, None)
            metadata = getattr(record, "metadata", None)
        if direct is not None:
            return direct
        return metadata.get(name) if isinstance(metadata, Mapping) else None

    @staticmethod
    def _records(value: object) -> list[object]:
        if value is None:
            return []
        dump_records = getattr(value, "dump_records", None)
        if callable(dump_records):
            dumped = dump_records()
            return list(dumped) if isinstance(dumped, Iterable) else []
        if isinstance(value, Iterable) and not isinstance(
            value, (str, bytes, bytearray, Mapping)
        ):
            return list(value)
        return []

    @classmethod
    def _proactive_replies(
        cls,
        value: object,
        *,
        character: str | None,
    ) -> list[tuple[float | None, str, str]]:
        replies: list[tuple[float | None, str, str]] = []
        for record in cls._records(value):
            if cls._field(record, "source") != "proactive":
                continue
            if cls._field(record, "turn_type") != "proactive_reply":
                continue
            lanlan_name = cls._field(record, "lanlan_name")
            if not character or lanlan_name != character:
                continue
            conversation_id = cls._field(record, "conversation_id")
            content = cls._field(record, "content")
            if not isinstance(conversation_id, str) or not conversation_id.strip():
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            raw_timestamp = cls._field(record, "timestamp")
            timestamp = (
                float(raw_timestamp)
                if isinstance(raw_timestamp, (int, float))
                and not isinstance(raw_timestamp, bool)
                else None
            )
            replies.append((timestamp, conversation_id.strip(), content))
        replies.sort(key=lambda item: (item[0] is None, item[0] or 0.0, item[1]))
        return replies

    def _remember_bus_key(self, key: str) -> None:
        if key in self._bus_seen:
            return
        self._bus_seen.add(key)
        self._bus_seen_order.append(key)
        while len(self._bus_seen_order) > 256:
            self._bus_seen.discard(self._bus_seen_order.popleft())

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _queue_reply(self, text: str, *, source: str) -> bool:
        now = self._wall_clock()
        # 主动回复可能在下一次普通回合结算时才补进 recent.json。仅在短窗口内
        # 去掉这类跨源重复，不妨碍玩家稍后再次得到相同的简短回答。
        digest = self._content_hash(text)
        previous = self._recent_content_hashes.get(digest)
        self._recent_content_hashes = {
            key: seen_at
            for key, seen_at in self._recent_content_hashes.items()
            if now - seen_at <= 30.0
        }
        if previous is not None and now - previous <= 30.0:
            return False
        pages = self.paginate(text, self.config.max_pages)
        if not pages:
            return False
        self._recent_content_hashes[digest] = now
        self._pending_pages.extend(
            (page, index == len(pages) - 1)
            for index, page in enumerate(pages)
        )
        if len(self._pending_pages) == len(pages):
            self._next_page_at = self._clock()
        self._last_result = f"queued_{source}"
        self._last_error = None
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self.config.poll_interval_s)

    def tick(self) -> None:
        update = self.provider.poll()
        context = self.provider.context()
        turns = context.get("turns")
        latest = turns[-1] if isinstance(turns, list) and turns else None
        assistant = latest.get("assistant") if isinstance(latest, Mapping) else None
        provider_status = self.provider.status()
        current_character = provider_status.get("current_character")
        if not isinstance(current_character, str) or not current_character.strip():
            current_character = None

        bus_snapshot: object | None = None
        bus_query_ok = self._conversation_fetcher is None
        if self._conversation_fetcher is not None:
            try:
                bus_snapshot = self._conversation_fetcher()
                bus_query_ok = True
            except Exception as exc:
                # 不记录异常正文，防止底层请求把载荷或连接细节带入状态。
                self._bus_last_error = f"query_{type(exc).__name__}"[:80]

        with self._lock:
            if not self._baseline_ready or update.character_changed:
                self._baseline_ready = update.revision is not None
                self._seen_revision = update.revision
                self._pending_pages.clear()
                self._recent_content_hashes.clear()
                if update.character_changed:
                    self._bus_baseline_at = self._wall_clock()
                    self._bus_baseline_ready = self._conversation_fetcher is None
                    self._bus_seen.clear()
                    self._bus_seen_order.clear()
                self._last_result = "baseline"
            if bus_query_ok and self._conversation_fetcher is not None:
                replies = self._proactive_replies(
                    bus_snapshot,
                    character=current_character,
                )
                if not self._bus_baseline_ready:
                    for timestamp, key, text in replies:
                        self._remember_bus_key(key)
                        # 首次成功查询时只接收桥启动之后产生的回合，避免重放旧聊天。
                        if timestamp is not None and timestamp > self._bus_baseline_at:
                            if self._queue_reply(text, source="proactive_bus"):
                                self._bus_records_seen += 1
                        if timestamp is not None:
                            self._bus_last_timestamp = timestamp
                    self._bus_baseline_ready = True
                else:
                    for timestamp, key, text in replies:
                        if key in self._bus_seen:
                            continue
                        self._remember_bus_key(key)
                        if self._queue_reply(text, source="proactive_bus"):
                            self._bus_records_seen += 1
                        if timestamp is not None:
                            self._bus_last_timestamp = timestamp
                self._bus_last_error = None
            if update.revision is not None and update.revision != self._seen_revision:
                self._seen_revision = update.revision
                queued = (
                    self._queue_reply(assistant, source="recent_file")
                    if isinstance(assistant, str)
                    else False
                )
                if not queued and not self._pending_pages:
                    self._last_result = "empty"
            if not self._pending_pages or self._clock() < self._next_page_at:
                return
            page, reply_end = self._pending_pages[0]

        display_seconds = self.display_duration(page, self.config.display_seconds)
        try:
            result = self._display_callback(page, display_seconds)
            status = result.get("status") if isinstance(result, Mapping) else None
        except Exception:
            status = None
            result = {"error": "display_callback_failed"}

        with self._lock:
            if status in {"accepted", "succeeded"}:
                self._pending_pages.pop(0)
                self._displayed_pages += 1
                if reply_end:
                    self._displayed_replies += 1
                if not self._pending_pages:
                    self._last_result = "displayed"
                else:
                    self._last_result = "displaying"
                self._last_error = None
                self._last_display_seconds = display_seconds
                self._next_page_at = self._clock() + display_seconds
            else:
                error = result.get("error") if isinstance(result, Mapping) else None
                self._last_result = "waiting"
                self._last_error = str(error or "display_failed")[:80]
                self._next_page_at = self._clock() + 1.0

    def status(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            return {
                "enabled": self.config.enabled,
                "source": self.config.source,
                "supplemental_source": (
                    "existing_conversations_bus"
                    if self._conversation_fetcher is not None
                    else None
                ),
                "worker_running": bool(thread is not None and thread.is_alive()),
                "baseline_ready": self._baseline_ready,
                "queued_pages": len(self._pending_pages),
                "displayed_replies": self._displayed_replies,
                "displayed_pages": self._displayed_pages,
                "last_display_seconds": self._last_display_seconds,
                "last_result": self._last_result,
                "last_error": self._last_error,
                "revision": (
                    None if self._seen_revision is None else self._seen_revision[:16]
                ),
                "memory": self.provider.status(),
                "proactive_bus": {
                    "available": self._conversation_fetcher is not None,
                    "baseline_ready": self._bus_baseline_ready,
                    "records_seen": self._bus_records_seen,
                    "last_timestamp": self._bus_last_timestamp,
                    "last_error": self._bus_last_error,
                },
                "player_chat_input": {
                    "available": True,
                    "source": "world_custom_input",
                    "activation": "local_follow_hotkey_T",
                },
            }


__all__ = ["MainReplyDisplayBridge"]
