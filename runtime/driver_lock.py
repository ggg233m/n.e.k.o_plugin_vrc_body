"""YUI 单驱动者进程锁。

VRChat 世界端仍以 session/driver/ownership 为最终权威；这个锁只负责在本机提前
阻止 N.E.K.O 与独立 MCP 同时打开同一个 MIDI 端口。
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO


class YuiDriverLeaseError(RuntimeError):
    """同一 MIDI 端口已经被另一个本地 YUI 控制器占用。"""


def default_driver_lock_path(midi_port: str) -> Path:
    root_text = os.environ.get("LOCALAPPDATA")
    root = Path(root_text) if root_text else Path(tempfile.gettempdir())
    safe_port = re.sub(r"[^A-Za-z0-9_.-]+", "_", midi_port.strip()) or "NEKO_MIDI"
    return root / "N.E.K.O" / "yui_npc_controller" / f"{safe_port}.lock"


class YuiDriverLease:
    """跨进程的独占租约；释放句柄即可恢复。"""

    def __init__(self, midi_port: str, *, lock_path: str | Path | None = None) -> None:
        self.midi_port = midi_port
        self.path = Path(lock_path) if lock_path is not None else default_driver_lock_path(midi_port)
        self._handle: BinaryIO | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> "YuiDriverLease":
        if self._handle is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle: BinaryIO | None = None
        try:
            handle = self.path.open("a+b")
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            if handle is not None:
                handle.close()
            raise YuiDriverLeaseError(
                f"MIDI 端口 {self.midi_port!r} 已由另一个 YUI 控制器占用"
            ) from exc
        assert handle is not None
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "YuiDriverLease":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()


__all__ = [
    "YuiDriverLease",
    "YuiDriverLeaseError",
    "default_driver_lock_path",
]
