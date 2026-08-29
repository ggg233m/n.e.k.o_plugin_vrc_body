"""独立 YUI NPC 插件使用的 VRChat ``output_log`` 只读跟随器。

日志路径会在轮转后重新发现；只有带 ``[NEKO]`` 的行进入会话投影，其他 VRChat
诊断保持原样跳过。读取错误会进入状态快照，不会被伪装成“世界没有事件”。
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any

from .yui_session import YuiSessionState


def default_vrchat_log_directory() -> Path:
    user_profile = os.environ.get("USERPROFILE")
    if not user_profile:
        raise RuntimeError("无法从 USERPROFILE 定位 VRChat output_log")
    return Path(user_profile) / "AppData" / "LocalLow" / "VRChat" / "VRChat"


def newest_vrchat_output_log(directory: str | Path | None = None) -> Path | None:
    root = Path(directory) if directory is not None else default_vrchat_log_directory()
    if not root.is_dir():
        return None
    candidates = [
        path
        for pattern in ("output_log_*.txt", "output_log.txt")
        for path in root.glob(pattern)
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


class YuiOutputLogTailer:
    """后台跟随 VRChat 输出日志，并把协议事件交给 ``YuiSessionState``。"""

    def __init__(
        self,
        session: YuiSessionState,
        *,
        log_path: str | Path | None = None,
        log_directory: str | Path | None = None,
        from_end: bool = True,
        poll_interval_s: float = 0.1,
    ) -> None:
        if log_path is not None and log_directory is not None:
            raise ValueError("log_path 与 log_directory 只能设置一个")
        self.session = session
        self.log_path = Path(log_path) if log_path is not None else None
        self.log_directory = Path(log_directory) if log_directory is not None else None
        self.from_end = bool(from_end)
        self.poll_interval_s = max(0.02, float(poll_interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._active_path: Path | None = None
        self._lines_read = 0
        self._events_read = 0
        self._decode_errors = 0
        self._last_error: str | None = None

    def _resolve_path(self) -> Path | None:
        if self.log_path is not None:
            return self.log_path if self.log_path.is_file() else None
        return newest_vrchat_output_log(self.log_directory)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="yui-output-log", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        handle = None
        opened_path: Path | None = None
        first_open = True
        try:
            while not self._stop.is_set():
                target = self._resolve_path()
                if target is None:
                    self._stop.wait(self.poll_interval_s)
                    continue
                if handle is None or opened_path != target:
                    if handle is not None:
                        handle.close()
                    handle = target.open("rb")
                    opened_path = target
                    with self._lock:
                        self._active_path = target
                    if first_open and self.from_end:
                        handle.seek(0, os.SEEK_END)
                    first_open = False
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    try:
                        size = target.stat().st_size
                    except OSError:
                        size = -1
                    if 0 <= size < handle.tell():
                        # 同一路径被截断时从新文件开头继续。
                        handle.seek(0)
                    else:
                        latest = self._resolve_path()
                        if latest is not None and latest != opened_path:
                            continue
                        self._stop.wait(self.poll_interval_s)
                    continue
                if not line.endswith(b"\n"):
                    # 普通文件的 readline 会把当前 EOF 前的残片当成一行并前移。
                    # 回退到本行开头，等 VRChat 补齐换行后再原样读取，避免唯一
                    # 上行事件被拆成两段并永久丢失。
                    handle.seek(line_start)
                    self._stop.wait(self.poll_interval_s)
                    continue
                with self._lock:
                    self._lines_read += 1
                if b"[NEKO]" not in line:
                    continue
                try:
                    event = self.session.ingest_line(line)
                except Exception as exc:
                    with self._lock:
                        self._decode_errors += 1
                        self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                else:
                    if event is not None:
                        with self._lock:
                            self._events_read += 1
                            self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            if handle is not None:
                handle.close()
            with self._lock:
                self._active_path = None

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, self.poll_interval_s * 4.0))
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            return {
                "running": bool(thread is not None and thread.is_alive()),
                "path": str(self._active_path) if self._active_path is not None else None,
                "lines_read": self._lines_read,
                "events_read": self._events_read,
                "decode_errors": self._decode_errors,
                "last_error": self._last_error,
            }

    def close(self) -> None:
        self.stop()


__all__ = [
    "YuiOutputLogTailer",
    "default_vrchat_log_directory",
    "newest_vrchat_output_log",
]
