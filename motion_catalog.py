"""Validated metadata and deterministic semantic selection for motion clips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
from typing import Any


@dataclass(frozen=True)
class MotionMetadata:
    name: str
    label: str
    description: str
    intents: tuple[str, ...]
    tags: tuple[str, ...]
    source_kind: str
    source_name: str
    body_scope: str
    side: str
    intensity: float
    recommended_speed: float
    transition_ms: int
    restore_after: bool
    loop_count: int

    def public(self) -> dict[str, Any]:
        result = asdict(self)
        result["intents"] = list(self.intents)
        result["tags"] = list(self.tags)
        return result


class MotionCatalog:
    """Load ``motions/catalog.json`` and select clips without parsing .nya payloads."""

    def __init__(self, directory: Path, filename: str = "catalog.json") -> None:
        self.directory = directory
        self.path = directory / filename
        self._lock = threading.RLock()
        self._signature: tuple[int, int] | None = None
        self._entries: dict[str, MotionMetadata] = {}
        self._errors: list[str] = []

    @staticmethod
    def _text(value: Any, label: str, *, maximum: int = 256) -> str:
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
            raise ValueError(f"{label} must be a non-empty string of at most {maximum} characters")
        return value.strip()

    @staticmethod
    def _strings(value: Any, label: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label} must be a non-empty string list")
        result = tuple(MotionCatalog._text(item, f"{label}[]", maximum=64) for item in value)
        if len(set(result)) != len(result):
            raise ValueError(f"{label} must not contain duplicates")
        return result

    @staticmethod
    def _number(value: Any, label: str, low: float, high: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number")
        result = float(value)
        if not math.isfinite(result) or not low <= result <= high:
            raise ValueError(f"{label} must be between {low:g} and {high:g}")
        return result

    @classmethod
    def _integer(cls, value: Any, label: str, low: int, high: int) -> int:
        result = cls._number(value, label, low, high)
        if not result.is_integer():
            raise ValueError(f"{label} must be an integer")
        return int(result)

    @classmethod
    def _parse_entry(cls, value: Any, index: int) -> MotionMetadata:
        if not isinstance(value, dict):
            raise ValueError(f"motions[{index}] must be an object")
        prefix = f"motions[{index}]"
        name = cls._text(value.get("name"), f"{prefix}.name", maximum=64)
        if name != Path(name).name or any(char in name for char in '<>:"/\\|?*'):
            raise ValueError(f"{prefix}.name must be a safe clip name")
        body_scope = cls._text(value.get("body_scope", "full_body"), f"{prefix}.body_scope", maximum=32)
        if body_scope not in {"full_body", "upper_body", "head"}:
            raise ValueError(f"{prefix}.body_scope is unsupported")
        side = cls._text(value.get("side", "both"), f"{prefix}.side", maximum=16)
        if side not in {"left", "right", "both", "neutral"}:
            raise ValueError(f"{prefix}.side is unsupported")
        transition_ms = cls._integer(value.get("transition_ms", 400), f"{prefix}.transition_ms", 0, 5000)
        loop_count = cls._integer(value.get("loop_count", 1), f"{prefix}.loop_count", 1, 10)
        restore_after = value.get("restore_after", True)
        if not isinstance(restore_after, bool):
            raise ValueError(f"{prefix}.restore_after must be a boolean")
        return MotionMetadata(
            name=name,
            label=cls._text(value.get("label", name), f"{prefix}.label", maximum=128),
            description=cls._text(value.get("description", name), f"{prefix}.description", maximum=512),
            intents=cls._strings(value.get("intents"), f"{prefix}.intents"),
            tags=cls._strings(value.get("tags", ["motion"]), f"{prefix}.tags"),
            source_kind=cls._text(value.get("source_kind", "vmd_bake"), f"{prefix}.source_kind", maximum=32),
            source_name=cls._text(value.get("source_name", name), f"{prefix}.source_name", maximum=128),
            body_scope=body_scope,
            side=side,
            intensity=cls._number(value.get("intensity", 0.5), f"{prefix}.intensity", 0.0, 1.0),
            recommended_speed=cls._number(value.get("recommended_speed", 1.0), f"{prefix}.recommended_speed", 0.25, 3.0),
            transition_ms=transition_ms,
            restore_after=restore_after,
            loop_count=loop_count,
        )

    def _reload(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            signature = None
        else:
            signature = stat.st_mtime_ns, stat.st_size
        with self._lock:
            if signature == self._signature:
                return
        entries: dict[str, MotionMetadata] = {}
        errors: list[str] = []
        if signature is not None:
            try:
                root = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(root, dict) or root.get("version", 1) != 1:
                    raise ValueError("catalog root/version is invalid")
                raw_entries = root.get("motions")
                if not isinstance(raw_entries, list):
                    raise ValueError("catalog motions must be a list")
                for index, raw in enumerate(raw_entries):
                    try:
                        entry = self._parse_entry(raw, index)
                        if entry.name in entries:
                            raise ValueError(f"duplicate motion name: {entry.name}")
                        entries[entry.name] = entry
                    except ValueError as exc:
                        errors.append(str(exc))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(str(exc))
        with self._lock:
            self._signature = signature
            self._entries = entries
            self._errors = errors

    def metadata(self, name: str) -> dict[str, Any] | None:
        self._reload()
        with self._lock:
            entry = self._entries.get(name)
            return entry.public() if entry is not None else None

    def select(
        self,
        intent: str,
        *,
        side: str = "auto",
        intensity: float | None = None,
        sequence_index: int = 0,
    ) -> dict[str, Any] | None:
        self._reload()
        with self._lock:
            candidates = [
                entry
                for entry in self._entries.values()
                if intent in entry.intents and (self.directory / f"{entry.name}.nya").is_file()
            ]
        if side != "auto":
            exact = [entry for entry in candidates if entry.side in {side, "neutral"}]
            if exact:
                candidates = exact
            else:
                return None
        if not candidates:
            return None
        if intensity is None:
            eligible = sorted(candidates, key=lambda item: item.name)
        else:
            ranked = sorted(candidates, key=lambda item: (abs(item.intensity - intensity), item.name))
            best_delta = abs(ranked[0].intensity - intensity)
            eligible = [item for item in ranked if abs(item.intensity - intensity) <= best_delta + 0.15]
        return eligible[sequence_index % len(eligible)].public()

    def summary(self) -> dict[str, Any]:
        self._reload()
        with self._lock:
            entries = [entry.public() for entry in sorted(self._entries.values(), key=lambda item: item.name)]
            errors = list(self._errors)
        missing = [entry["name"] for entry in entries if not (self.directory / f"{entry['name']}.nya").is_file()]
        return {
            "version": 1,
            "path": str(self.path),
            "entries": entries,
            "errors": errors,
            "missing_clips": missing,
        }


__all__ = ["MotionCatalog", "MotionMetadata"]
