"""独立后端 Web 控制台的配置存储与静态资源入口。

普通运行配置和凭据刻意分开：面板只把经过 ``PluginConfig`` 校验的普通配置
写入 JSON 覆盖文件，API key 始终来自进程环境变量，也不会通过读取接口回显。
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping

from .adapters import PluginConfig


UI_DIRECTORY = Path(__file__).with_name("standalone_ui")


_EDITABLE_FIELDS = {
    "anyadance": frozenset({"host", "port", "rate_hz"}),
    "vrchat_osc": frozenset({
        "enabled", "send_host", "send_port", "listen_host", "listen_port",
        "allowed_sender", "input_pulse_ms", "parameter_cache_size",
        "awareness_parameters",
    }),
    "input": frozenset({
        "primary", "rate_hz", "osc_fallback", "max_hold_ms",
        "emergency_release_ms",
    }),
    "autonomy": frozenset({"manual_arm", "session_ttl_minutes"}),
    "world_memory": frozenset({"persist_world", "persist_players"}),
    "vision": frozenset({
        "enabled", "source", "capture", "local_backend", "model_path",
        "labels_path", "device", "onnxruntime_cuda",
        "onnxruntime_cuda_device_id", "fallback_backend",
        "confidence_threshold", "input_width", "input_height",
        "horizontal_fov_deg", "max_detections", "min_box_ratio",
        "min_box_width_ratio", "min_box_height_ratio",
        "identity_reid_enabled", "identity_reid_similarity",
        "identity_reid_margin", "identity_reid_retention_s",
        "identity_reid_max_identities", "semantic_backend",
        "semantic_endpoint", "semantic_model", "semantic_max_per_minute",
        "semantic_main_llm_min_interval_s",
        "frame_cache_interval_s", "frame_max_width", "frame_jpeg_quality",
        "frame_max_per_minute", "monitor_index", "dxcam_device_idx",
        "dxcam_output_idx", "dxcam_backend", "interval_ms", "queue_size",
        "detector_threads", "detector_interval_ms",
        "detector_accelerator_interval_ms", "lifecycle_watermark_limit",
        "window_title", "window_track_interval_ms",
    }),
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并普通配置映射，不修改调用方传入的对象。"""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_settings_file(path: Path) -> dict[str, Any]:
    """读取面板生成的覆盖文件；不存在时返回空覆盖。"""
    if not path.exists():
        return {}
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("standalone settings file is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("standalone settings must be a JSON object")
    config = value.get("config", value)
    if not isinstance(config, Mapping):
        raise ValueError("standalone settings config must be an object")
    # 手工编辑的文件也走与 Web API 相同的白名单，不能借重启偷偷注入凭据字段。
    return _validate_update(config)


def _canonical_config(config: PluginConfig) -> dict[str, Any]:
    """把内部 dataclass 还原成面板可编辑、可再次解析的配置段。"""
    return {
        "anyadance": {
            "host": config.host,
            "port": config.port,
            "rate_hz": config.rate_hz,
        },
        "vrchat_osc": asdict(config.vrchat_osc),
        "input": asdict(config.input),
        "autonomy": asdict(config.autonomy),
        "world_memory": asdict(config.world_memory),
        "vision": asdict(config.vision),
    }


def _validate_update(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("config must be an object")
    if len(value) > len(_EDITABLE_FIELDS):
        raise ValueError("config contains too many sections")
    result: dict[str, Any] = {}
    for raw_section, raw_values in value.items():
        section = str(raw_section)
        allowed = _EDITABLE_FIELDS.get(section)
        if allowed is None:
            raise ValueError(f"config section is not editable: {section}")
        if not isinstance(raw_values, Mapping):
            raise ValueError(f"config.{section} must be an object")
        unknown = set(map(str, raw_values)) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"config.{section} contains unsupported fields: {names}")
        result[section] = copy.deepcopy(dict(raw_values))
    return result


class StandaloneConfigStore:
    """保存独立运行覆盖配置，并报告哪些改动需要重启。"""

    def __init__(
        self,
        config_data: Mapping[str, Any],
        *,
        settings_path: Path | None,
        editable: bool,
        mode: str,
        source: str,
        offline: bool,
    ) -> None:
        parsed = PluginConfig.from_mapping(config_data)
        self._active_raw = copy.deepcopy(dict(config_data))
        self._pending_raw = copy.deepcopy(dict(config_data))
        self._active = _canonical_config(parsed)
        self._pending = copy.deepcopy(self._active)
        self.settings_path = settings_path
        self.editable = bool(editable and settings_path is not None)
        self.mode = str(mode)
        self.source = str(source)
        self.offline = bool(offline)
        self._lock = threading.RLock()

    @staticmethod
    def _environment_status() -> dict[str, bool]:
        # 只返回是否存在，绝不把凭据或 endpoint 内容送到浏览器。
        return {
            "vlm_endpoint": bool(os.getenv("VRC_VLM_ENDPOINT") or os.getenv("OPENAI_BASE_URL")),
            "vlm_model": bool(os.getenv("VRC_VLM_MODEL") or os.getenv("OPENAI_VLM_MODEL")),
            "vlm_api_key": bool(os.getenv("VRC_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")),
            "openvino_model": bool(os.getenv("VRC_OPENVINO_MODEL")),
            "openvino_labels": bool(os.getenv("VRC_OPENVINO_LABELS")),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            settings_path = None if self.settings_path is None else str(self.settings_path)
            return {
                "config": copy.deepcopy(self._pending),
                "active_config": copy.deepcopy(self._active),
                "restart_required": self._pending != self._active,
                "editable": self.editable,
                "mode": self.mode,
                "source": self.source,
                "offline": self.offline,
                "settings_path": settings_path,
                "settings_exists": bool(self.settings_path and self.settings_path.exists()),
                "secrets": self._environment_status(),
                "secret_policy": "environment_only",
            }

    def save(self, value: Any) -> dict[str, Any]:
        if not self.editable or self.settings_path is None:
            raise PermissionError("configuration is managed by the N.E.K.O host")
        update = _validate_update(value)
        with self._lock:
            candidate_raw = deep_merge(self._pending_raw, update)
            parsed = PluginConfig.from_mapping(candidate_raw)
            candidate = _canonical_config(parsed)
            self._write(candidate)
            self._pending_raw = candidate_raw
            self._pending = candidate
            return self.snapshot()

    def _write(self, config: Mapping[str, Any]) -> None:
        path = self.settings_path
        if path is None:
            raise PermissionError("standalone settings path is not configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "notice": "API keys are intentionally not stored here; use VRC_VLM_API_KEY or OPENAI_API_KEY.",
            "config": config,
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = [
    "StandaloneConfigStore",
    "UI_DIRECTORY",
    "deep_merge",
    "load_settings_file",
]
