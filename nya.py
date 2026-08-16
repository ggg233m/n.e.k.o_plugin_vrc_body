"""Safe AnyaDance .nya clip loading, cataloguing, and sampling."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from .config import PluginConfig
from .model import CONTROLLER_IDS, DEVICE_IDS, ControllerState, DeviceState, FrameState, normalized_quat
from .motion_catalog import MotionCatalog
from .motion import interpolate_frame
from .protocol import validate_frame

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
WINDOWS_INVALID_NAME_CHARS = set('<>:"/\\|?*')


@dataclass(frozen=True)
class NyaKeyframe:
    time_s: float
    frame: FrameState
    has_fingers: bool


@dataclass(frozen=True)
class NyaClip:
    name: str
    loop_hint: bool
    fps: float
    model: str
    frames: tuple[NyaKeyframe, ...]
    duration_s: float
    times: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", tuple(frame.time_s for frame in self.frames))

    @property
    def is_pose(self) -> bool:
        return len(self.frames) == 1

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_s": round(self.duration_s, 4),
            "frame_count": len(self.frames),
            "fps": self.fps,
            "loop_hint": self.loop_hint,
            "model": self.model,
            "is_pose": self.is_pose,
        }


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def parse_nya(text: str, *, name: str, config: PluginConfig) -> NyaClip:
    try:
        root = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"clip is not valid JSON: {exc}") from exc
    if not isinstance(root, dict) or root.get("format") != "anyadance_nya":
        raise ValueError("clip is not an AnyaDance .nya document")
    version = root.get("version", 1)
    if version != 1:
        raise ValueError(f"unsupported .nya version: {version}")
    fps = _number(root.get("fps", 60.0), "fps")
    if not 1.0 <= fps <= 240.0:
        raise ValueError("fps must be between 1 and 240")
    loop_hint = root.get("loop", True)
    if not isinstance(loop_hint, bool):
        raise ValueError("loop must be a boolean")
    model = root.get("model", "")
    if not isinstance(model, str) or len(model) > 256:
        raise ValueError("model must be a string of at most 256 characters")
    raw_frames = root.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("clip must contain at least one frame")
    if len(raw_frames) > config.clip_max_frames:
        raise ValueError(f"clip exceeds the {config.clip_max_frames} frame limit")

    parsed: list[tuple[float, FrameState, bool]] = []
    previous_time: float | None = None
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            raise ValueError(f"frames[{index}] must be an object")
        time_s = _number(raw_frame.get("t", index / fps), f"frames[{index}].t")
        if time_s < 0.0:
            raise ValueError(f"frames[{index}].t must be non-negative")
        if previous_time is not None and time_s <= previous_time:
            raise ValueError("frame times must be strictly increasing")
        previous_time = time_s
        devices = raw_frame.get("devices")
        if not isinstance(devices, dict):
            raise ValueError(f"frames[{index}] is missing devices")
        parsed_devices: dict[str, DeviceState] = {}
        for device_name in DEVICE_IDS:
            raw_device = devices.get(device_name)
            if not isinstance(raw_device, dict):
                raise ValueError(f"frames[{index}] has a bad pose for {device_name}")
            position = _vector(raw_device.get("p"), 3, f"frames[{index}].devices.{device_name}.p")
            # Match AnyaDance's native .nya loader: values slightly above the
            # driver ceiling are accepted and clamped instead of rejecting an
            # otherwise valid exported dance.
            position = (position[0], min(position[1], config.safety.max_y_m), position[2])
            rotation = normalized_quat(_vector(raw_device.get("q"), 4, f"frames[{index}].devices.{device_name}.q"))
            parsed_devices[device_name] = DeviceState(position, rotation)  # type: ignore[arg-type]

        controllers = {controller_name: ControllerState() for controller_name in CONTROLLER_IDS}
        has_fingers = False
        raw_fingers = raw_frame.get("fingers")
        if raw_fingers is not None:
            if not isinstance(raw_fingers, dict):
                raise ValueError(f"frames[{index}].fingers must be an object")
            for side, controller_name in (("left", "left_controller"), ("right", "right_controller")):
                values = _vector(raw_fingers.get(side), 5, f"frames[{index}].fingers.{side}")
                if any(not 0.0 <= value <= 1.0 for value in values):
                    raise ValueError(f"frames[{index}].fingers.{side} values must be in [0, 1]")
                controller = controllers[controller_name]
                controller.finger_bends = dict(zip(FINGER_NAMES, values))
                fist = all(value >= 0.95 for value in values)
                controller.grip_click = fist
                controller.grip_value = 1.0 if fist else 0.0
            has_fingers = True
        frame = FrameState(parsed_devices, controllers)
        validate_frame(frame, config.safety)
        parsed.append((time_s, frame, has_fingers))

    first_time = parsed[0][0]
    keyframes = tuple(NyaKeyframe(time_s - first_time, frame, fingers) for time_s, frame, fingers in parsed)
    duration_s = keyframes[-1].time_s
    if duration_s > config.clip_max_duration_seconds:
        raise ValueError(f"clip exceeds the {config.clip_max_duration_seconds:g} second duration limit")
    return NyaClip(name=name, loop_hint=loop_hint, fps=fps, model=model, frames=keyframes, duration_s=duration_s)


def _prepared_frame(keyframe: NyaKeyframe, base: FrameState, offset_x: float, offset_z: float) -> FrameState:
    result = keyframe.frame.clone()
    for device in result.devices.values():
        device.position = (device.position[0] + offset_x, device.position[1], device.position[2] + offset_z)
    result.controllers = copy.deepcopy(base.controllers)
    if keyframe.has_fingers:
        for controller_name in CONTROLLER_IDS:
            source = keyframe.frame.controllers[controller_name]
            target = result.controllers[controller_name]
            target.finger_bends = dict(source.finger_bends)
            target.grip_click = source.grip_click
            target.grip_value = source.grip_value
    return result


def sample_clip(clip: NyaClip, time_s: float, *, base: FrameState, offset_x: float, offset_z: float) -> FrameState:
    if clip.is_pose or time_s <= 0.0:
        return _prepared_frame(clip.frames[0], base, offset_x, offset_z)
    if time_s >= clip.duration_s:
        return _prepared_frame(clip.frames[-1], base, offset_x, offset_z)
    upper = bisect_right(clip.times, time_s)
    lower = max(0, upper - 1)
    before = clip.frames[lower]
    after = clip.frames[min(upper, len(clip.frames) - 1)]
    span = max(after.time_s - before.time_s, 1e-9)
    progress = (time_s - before.time_s) / span
    return interpolate_frame(
        _prepared_frame(before, base, offset_x, offset_z),
        _prepared_frame(after, base, offset_x, offset_z),
        progress,
    )


class ClipLibrary:
    def __init__(self, directory: Path, config: PluginConfig) -> None:
        self.directory = directory
        self.config = config
        self._cache_lock = threading.RLock()
        self._load_locks: dict[str, threading.Lock] = {}
        self._summaries: dict[str, tuple[tuple[int, int], dict[str, Any] | None, str | None]] = {}
        self._loaded: OrderedDict[str, tuple[tuple[int, int], NyaClip]] = OrderedDict()
        self._max_resident_clips = 2
        self._parse_count = 0
        self._cache_hits = 0
        self._catalog_calls = 0
        self._last_parse_ms = 0.0
        self._last_catalog_ms = 0.0
        self.motion_catalog = MotionCatalog(directory)

    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or not name or len(name) > 64:
            raise ValueError("clip_name must contain between 1 and 64 characters")
        if name != name.strip() or name in {".", ".."} or name.endswith((".", " ")):
            raise ValueError("clip_name must not contain leading/trailing spaces or dot path components")
        if any(ord(char) < 32 or char in WINDOWS_INVALID_NAME_CHARS for char in name):
            raise ValueError("clip_name contains a path separator, control character, or invalid filename character")
        root = self.directory.resolve()
        candidate = (root / f"{name}.nya").resolve()
        if candidate.parent != root:
            raise ValueError("clip path escapes the configured motions directory")
        return candidate

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _metrics(self) -> dict[str, Any]:
        with self._cache_lock:
            return {
                "parse_count": self._parse_count,
                "cache_hits": self._cache_hits,
                "catalog_calls": self._catalog_calls,
                "resident_clips": list(self._loaded),
                "last_parse_ms": round(self._last_parse_ms, 3),
                "last_catalog_ms": round(self._last_catalog_ms, 3),
            }

    def _indexed_summary(self, clip: NyaClip, *, file_size_bytes: int) -> dict[str, Any]:
        result = {
            **clip.summary(),
            "file_size_bytes": file_size_bytes,
            "indexed": True,
        }
        metadata = self.motion_catalog.metadata(clip.name)
        if metadata is not None:
            result["metadata"] = metadata
        return result

    def _unindexed_summary(self, path: Path, *, file_size_bytes: int) -> dict[str, Any]:
        result = {
            "name": path.stem,
            "duration_s": None,
            "frame_count": None,
            "fps": None,
            "loop_hint": None,
            "model": "",
            "is_pose": False,
            "file_size_bytes": file_size_bytes,
            "indexed": False,
        }
        metadata = self.motion_catalog.metadata(path.stem)
        if metadata is not None:
            result["metadata"] = metadata
        return result

    def select_for_intent(
        self,
        intent: str,
        *,
        side: str = "auto",
        intensity: float | None = None,
        sequence_index: int = 0,
    ) -> dict[str, Any] | None:
        return self.motion_catalog.select(
            intent,
            side=side,
            intensity=intensity,
            sequence_index=sequence_index,
        )

    def load(self, name: str) -> NyaClip:
        path = self._path(name)
        if not path.is_file():
            raise ValueError(f"unknown preset clip: {name}")
        signature = self._signature(path)
        size = signature[1]
        if size > self.config.clip_max_file_bytes:
            raise ValueError(f"clip exceeds the {self.config.clip_max_file_bytes} byte file limit")

        with self._cache_lock:
            load_lock = self._load_locks.setdefault(name, threading.Lock())
        with load_lock:
            # Recheck after waiting for another caller that may have parsed it.
            signature = self._signature(path)
            size = signature[1]
            if size > self.config.clip_max_file_bytes:
                raise ValueError(f"clip exceeds the {self.config.clip_max_file_bytes} byte file limit")
            with self._cache_lock:
                resident = self._loaded.get(name)
                if resident is not None and resident[0] == signature:
                    self._loaded.move_to_end(name)
                    self._cache_hits += 1
                    return resident[1]
                cached = self._summaries.get(name)
                if cached is not None and cached[0] == signature and cached[2] is not None:
                    self._cache_hits += 1
                    raise ValueError(cached[2])

            started = time.perf_counter()
            try:
                clip = parse_nya(path.read_text(encoding="utf-8"), name=name, config=self.config)
                if self._signature(path) != signature:
                    raise ValueError("clip changed while it was being loaded; retry the command")
            except Exception as exc:
                error = str(exc)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with self._cache_lock:
                    self._summaries[name] = (signature, None, error)
                    self._loaded.pop(name, None)
                    self._parse_count += 1
                    self._last_parse_ms = elapsed_ms
                if isinstance(exc, ValueError):
                    raise
                raise ValueError(error) from exc

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            summary = self._indexed_summary(clip, file_size_bytes=size)
            with self._cache_lock:
                self._summaries[name] = (signature, summary, None)
                self._loaded[name] = (signature, clip)
                self._loaded.move_to_end(name)
                while len(self._loaded) > self._max_resident_clips:
                    self._loaded.popitem(last=False)
                self._parse_count += 1
                self._last_parse_ms = elapsed_ms
            return clip

    def catalog(self) -> dict[str, Any]:
        """Return a fast directory catalog without parsing new clip payloads."""
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        clips: list[dict[str, Any]] = []
        invalid: list[dict[str, str]] = []
        present: set[str] = set()
        indexed_count = 0
        for path in sorted(self.directory.glob("*.nya"), key=lambda item: item.name.lower()):
            name = path.stem
            present.add(name)
            try:
                signature = self._signature(path)
            except OSError as exc:
                invalid.append({"name": name, "error": str(exc)})
                continue
            if signature[1] > self.config.clip_max_file_bytes:
                error = f"clip exceeds the {self.config.clip_max_file_bytes} byte file limit"
                with self._cache_lock:
                    self._summaries[name] = (signature, None, error)
                    self._loaded.pop(name, None)
                invalid.append({"name": name, "error": error})
                continue
            with self._cache_lock:
                cached = self._summaries.get(name)
                if cached is not None and cached[0] != signature:
                    self._summaries.pop(name, None)
                    self._loaded.pop(name, None)
                    cached = None
            if cached is not None and cached[2] is not None:
                invalid.append({"name": name, "error": cached[2]})
            elif cached is not None and cached[1] is not None:
                clips.append(dict(cached[1]))
                indexed_count += 1
            else:
                clips.append(self._unindexed_summary(path, file_size_bytes=signature[1]))

        with self._cache_lock:
            removed = set(self._summaries) - present
            for name in removed:
                self._summaries.pop(name, None)
                self._loaded.pop(name, None)
            self._catalog_calls += 1
            self._last_catalog_ms = (time.perf_counter() - started) * 1000.0
        return {
            "clips": clips,
            "invalid_clips": invalid,
            "directory": str(self.directory),
            "indexed_count": indexed_count,
            "unindexed_count": len(clips) - indexed_count,
            "cache": self._metrics(),
            "motion_catalog": self.motion_catalog.summary(),
        }

    def list(self) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        clips: list[dict[str, Any]] = []
        invalid: list[dict[str, str]] = []
        for path in sorted(self.directory.glob("*.nya"), key=lambda item: item.name.lower()):
            try:
                clip = self.load(path.stem)
            except Exception as exc:
                invalid.append({"name": path.stem, "error": str(exc)})
            else:
                signature = self._signature(path)
                clips.append(self._indexed_summary(clip, file_size_bytes=signature[1]))
        return {
            "clips": clips,
            "invalid_clips": invalid,
            "directory": str(self.directory),
            "indexed_count": len(clips),
            "unindexed_count": 0,
            "cache": self._metrics(),
            "motion_catalog": self.motion_catalog.summary(),
        }
