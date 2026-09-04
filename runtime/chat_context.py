"""只读 N.E.K.O 硬盘近期聊天，为独立动作模型提供有限上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping

from .config import YuiChatContextConfig


_TEXT_PART_TYPES = frozenset({None, "text", "input_text", "output_text"})
_READ_RETRIES = 3


@dataclass(frozen=True, slots=True)
class ChatContextUpdate:
    """一次轮询的脱敏变化结果。"""

    changed: bool
    character_changed: bool
    revision: str | None


def _absolute_path(value: object) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return None
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _standard_runtime_root(environ: Mapping[str, str]) -> Path:
    if sys.platform == "win32":
        base = environ.get("LOCALAPPDATA") or environ.get("APPDATA")
        if base:
            return Path(base).expanduser().resolve(strict=False) / "N.E.K.O"
        return Path.home() / "AppData" / "Local" / "N.E.K.O"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "N.E.K.O"
    base = environ.get("XDG_DATA_HOME")
    return (
        Path(base).expanduser().resolve(strict=False)
        if base
        else Path.home() / ".local" / "share"
    ) / "N.E.K.O"


def resolve_neko_runtime_root(
    *,
    environ: Mapping[str, str] | None = None,
    explicit_root: Path | str | None = None,
) -> Path:
    """按 N.E.K.O 选定根目录环境、存储策略、标准目录的顺序解析。"""

    if explicit_root is not None:
        resolved = _absolute_path(explicit_root)
        if resolved is None:
            raise ValueError("runtime_root_invalid")
        return resolved
    values = os.environ if environ is None else environ
    selected = _absolute_path(values.get("NEKO_STORAGE_SELECTED_ROOT"))
    if selected is not None:
        return selected
    anchor = (
        _absolute_path(values.get("NEKO_STORAGE_ANCHOR_ROOT"))
        or _standard_runtime_root(values)
    )
    policy_path = anchor / "state" / "storage_policy.json"
    try:
        if policy_path.is_symlink():
            raise OSError("policy_symlink")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy_selected = (
            _absolute_path(policy.get("selected_root"))
            if isinstance(policy, Mapping)
            else None
        )
        if policy_selected is not None:
            return policy_selected
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    return anchor


class RecentChatContextProvider:
    """轮询当前角色的 recent.json；从不锁定、写入或导入宿主私有模块。"""

    def __init__(
        self,
        config: YuiChatContextConfig,
        *,
        runtime_root: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._environ = os.environ if environ is None else environ
        self._runtime_root_override = runtime_root
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleeper = sleeper
        self._lock = threading.RLock()
        self._last_poll_at = float("-inf")
        self._character: str | None = None
        self._turns: tuple[dict[str, str], ...] = ()
        self._revision: str | None = None
        self._modified_at: float | None = None
        self._last_refresh_at: float | None = None
        self._file_state = "disabled" if not config.enabled else "not_polled"
        self._last_error: str | None = None

    @staticmethod
    def _valid_character_name(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        name = value.strip()
        if (
            not name
            or name in {".", ".."}
            or len(name) > 128
            or "\x00" in name
            or Path(name).name != name
            or any(separator in name for separator in ("/", "\\"))
        ):
            return None
        return name

    def _runtime_root(self) -> Path:
        return resolve_neko_runtime_root(
            environ=self._environ,
            explicit_root=self._runtime_root_override,
        )

    def _read_stable_bytes(self, path: Path, *, maximum: int) -> tuple[bytes, os.stat_result]:
        if path.is_symlink():
            raise OSError("symlink_rejected")
        last_error: Exception | None = None
        for attempt in range(_READ_RETRIES):
            try:
                before = path.stat()
                if not path.is_file():
                    raise OSError("not_file")
                if before.st_size > maximum:
                    raise ValueError("file_too_large")
                payload = path.read_bytes()
                after = path.stat()
                if (
                    before.st_size == after.st_size == len(payload)
                    and before.st_mtime_ns == after.st_mtime_ns
                ):
                    return payload, after
                last_error = OSError("file_changed")
            except (OSError, ValueError) as exc:
                last_error = exc
                if isinstance(exc, ValueError):
                    break
            if attempt + 1 < _READ_RETRIES:
                self._sleeper(0.02)
        if last_error is None:
            last_error = OSError("read_failed")
        raise last_error

    @staticmethod
    def _text_content(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if not isinstance(value, list):
            return ""
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item
            elif isinstance(item, Mapping) and item.get("type") in _TEXT_PART_TYPES:
                text = item.get("text")
            else:
                continue
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)

    @classmethod
    def _complete_turns(cls, payload: Any) -> list[dict[str, str]]:
        if not isinstance(payload, list):
            raise ValueError("invalid_root")
        turns: list[dict[str, str]] = []
        users: list[str] = []
        assistants: list[str] = []

        def finish() -> None:
            if users and assistants:
                turns.append({
                    "user": "\n".join(users),
                    "assistant": "\n".join(assistants),
                })

        for row in payload:
            if not isinstance(row, Mapping) or row.get("type") not in {"human", "ai"}:
                continue
            data = row.get("data")
            if not isinstance(data, Mapping):
                continue
            text = cls._text_content(data.get("content"))
            if not text:
                continue
            if row.get("type") == "human":
                if users:
                    finish()
                    users = []
                    assistants = []
                users.append(text)
            elif users:
                assistants.append(text)
        finish()
        return turns

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        if limit <= 1:
            return text[:limit]
        left = max(1, (limit - 1) // 2)
        return text[:left] + "…" + text[-(limit - left - 1):]

    def _bounded_turns(self, turns: list[dict[str, str]]) -> tuple[dict[str, str], ...]:
        selected = [dict(item) for item in turns[-self.config.max_turns:]]
        while len(selected) > 1 and sum(
            len(item["user"]) + len(item["assistant"]) for item in selected
        ) > self.config.max_chars:
            selected.pop(0)
        if selected:
            total = len(selected[0]["user"]) + len(selected[0]["assistant"])
            if total > self.config.max_chars:
                user_budget = min(
                    len(selected[0]["user"]),
                    max(1, self.config.max_chars // 2),
                )
                assistant_budget = max(1, self.config.max_chars - user_budget)
                selected[0]["user"] = self._shorten(selected[0]["user"], user_budget)
                selected[0]["assistant"] = self._shorten(
                    selected[0]["assistant"], assistant_budget
                )
        return tuple(selected)

    @staticmethod
    def _revision_for(character: str, turns: tuple[dict[str, str], ...]) -> str:
        body = json.dumps(
            {"character": character, "turns": turns},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    def _set_error(self, state: str, error: str) -> None:
        with self._lock:
            self._file_state = state
            self._last_error = error
            self._last_refresh_at = self._wall_clock()

    def poll(self, *, force: bool = False) -> ChatContextUpdate:
        now = self._clock()
        with self._lock:
            if not self.config.enabled:
                return ChatContextUpdate(False, False, self._revision)
            if not force and now - self._last_poll_at < self.config.poll_interval_s:
                return ChatContextUpdate(False, False, self._revision)
            self._last_poll_at = now
            previous_character = self._character
            previous_revision = self._revision

        candidate_character: str | None = None
        try:
            root = self._runtime_root()
            characters_path = root / "config" / "characters.json"
            raw_characters, _ = self._read_stable_bytes(
                characters_path,
                maximum=self.config.max_file_bytes,
            )
            characters = json.loads(raw_characters.decode("utf-8-sig"))
            character = self._valid_character_name(
                characters.get("当前猫娘") if isinstance(characters, Mapping) else None
            )
            if character is None:
                raise ValueError("invalid_current_character")
            candidate_character = character

            memory_root = (root / "memory").resolve(strict=False)
            character_dir = memory_root / character
            recent_path = character_dir / "recent.json"
            resolved_recent = recent_path.resolve(strict=False)
            if resolved_recent.parent.parent != memory_root:
                raise ValueError("path_outside_memory")
            if character_dir.is_symlink() or recent_path.is_symlink():
                raise ValueError("symlink_rejected")
            raw_recent, recent_stat = self._read_stable_bytes(
                recent_path,
                maximum=self.config.max_file_bytes,
            )
            decoded = json.loads(raw_recent.decode("utf-8-sig"))
            complete_turns = self._complete_turns(decoded)
            turns = self._bounded_turns(complete_turns)
            # 修订必须覆盖文件内的完整规范化轮次，而不能只覆盖准备注入的末尾
            # N 轮。否则用户连续发送相同内容、模型返回相同回答时，截断后的
            # 最后一轮完全一致，头顶显示桥会误判为“没有新回复”。上下文仍然
            # 只暴露受 max_turns/max_chars 限制的 turns，不扩大注入范围。
            revision = self._revision_for(character, tuple(complete_turns))
        except FileNotFoundError:
            # 已识别出新角色但尚无历史时立即清空旧角色内容；配置文件自身被
            # 原子替换的短暂缺失则沿用上一次有效快照。
            with self._lock:
                character_changed = bool(
                    candidate_character is not None
                    and previous_character is not None
                    and candidate_character != previous_character
                )
                if candidate_character is not None:
                    self._character = candidate_character
                    self._turns = ()
                    self._revision = None
                    self._modified_at = None
                self._file_state = "missing"
                self._last_error = "file_missing"
                self._last_refresh_at = self._wall_clock()
            return ChatContextUpdate(
                bool(candidate_character is not None and previous_revision is not None),
                character_changed,
                self._revision,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            code = str(exc) if str(exc) in {
                "file_too_large",
                "invalid_current_character",
                "invalid_root",
                "path_outside_memory",
                "symlink_rejected",
            } else "read_failed"
            character_changed = bool(
                candidate_character is not None
                and previous_character is not None
                and candidate_character != previous_character
            )
            if character_changed:
                with self._lock:
                    self._character = candidate_character
                    self._turns = ()
                    self._revision = None
                    self._modified_at = None
            self._set_error("unreadable", code)
            return ChatContextUpdate(
                character_changed,
                character_changed,
                None if character_changed else previous_revision,
            )

        character_changed = previous_character is not None and previous_character != character
        changed = revision != previous_revision or character_changed
        with self._lock:
            self._character = character
            self._turns = turns
            self._revision = revision
            self._modified_at = recent_stat.st_mtime
            self._last_refresh_at = self._wall_clock()
            self._file_state = "available"
            self._last_error = None
        return ChatContextUpdate(changed, character_changed, revision)

    def context(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source": "recent_file",
                "untrusted": True,
                "turns": [dict(item) for item in self._turns],
            }

    @staticmethod
    def _iso(timestamp: float | None) -> str | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = (
                None
                if self._modified_at is None
                else max(0.0, self._wall_clock() - self._modified_at)
            )
            return {
                "enabled": self.config.enabled,
                "source": self.config.source,
                "current_character": self._character,
                "file_state": self._file_state,
                "turn_count": len(self._turns),
                "revision": None if self._revision is None else self._revision[:16],
                "revision_time": self._iso(self._modified_at),
                "revision_age_s": None if age is None else round(age, 1),
                "last_refresh_at": self._iso(self._last_refresh_at),
                "last_error": self._last_error,
            }


__all__ = [
    "ChatContextUpdate",
    "RecentChatContextProvider",
    "resolve_neko_runtime_root",
]
