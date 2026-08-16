"""Configuration parsing and validation for the AnyaDance body plugin."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any, default: float, *, minimum: float, maximum: float, name: str) -> float:
    if value is None:
        # Some bounds are derived from other config sections, so the built-in
        # default can fall outside the effective range.
        return min(maximum, max(minimum, default))
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int, name: str) -> int:
    if value is None:
        return min(maximum, max(minimum, default))
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    parsed = int(numeric)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _boolean(value: Any, default: bool, *, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class BodyProfile:
    height_m: float = 1.50
    shoulder_width_m: float = 0.36
    shoulder_drop_m: float = 0.18
    arm_length_m: float = 0.58


@dataclass(frozen=True)
class SafetyConfig:
    max_position_abs_m: float = 3.0
    max_y_m: float = 2.0
    max_linear_speed_mps: float = 2.0
    max_angular_speed_dps: float = 360.0
    max_action_duration_ms: int = 5000


@dataclass(frozen=True)
class BehaviorConfig:
    default_crossfade_ms: int = 400
    protect_full_body_motion: bool = True
    prefer_vmd_expressions: bool = True
    transition_history_size: int = 16


@dataclass(frozen=True)
class VmcIdleConfig:
    enabled: bool = True
    listen_host: str = "127.0.0.1"
    listen_port: int = 39539
    allowed_sender: str = "127.0.0.1"
    stale_after_ms: int = 500
    manage_host_output: bool = True
    host_api_url: str = "http://127.0.0.1:48911"
    host_api_timeout_seconds: float = 3.0
    host_output_host: str = "127.0.0.1"
    host_send_rate_hz: int = 60


@dataclass(frozen=True)
class VrchatOscConfig:
    enabled: bool = True
    send_host: str = "127.0.0.1"
    send_port: int = 9000
    listen_host: str = "127.0.0.1"
    listen_port: int = 9001
    allowed_sender: str = "127.0.0.1"
    input_pulse_ms: int = 100
    parameter_cache_size: int = 256
    awareness_parameters: tuple[str, ...] = (
        "NEKO_Action",
        "NEKO_ActionActive",
        "NEKO_ActionPhase",
        "NEKO_Holding",
    )


@dataclass(frozen=True)
class PluginConfig:
    host: str = "127.0.0.1"
    port: int = 39570
    rate_hz: int = 60
    default_duration_ms: int = 600
    max_queue_size: int = 8
    clip_directory: str = "motions"
    clip_max_file_bytes: int = 64 * 1024 * 1024
    clip_max_frames: int = 18000
    clip_max_duration_seconds: float = 300.0
    behavior: BehaviorConfig = BehaviorConfig()
    vmc_idle: VmcIdleConfig = VmcIdleConfig()
    vrchat_osc: VrchatOscConfig = VrchatOscConfig()
    profile: BodyProfile = BodyProfile()
    safety: SafetyConfig = SafetyConfig()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PluginConfig":
        root = data or {}
        anyadance = _section(root, "anyadance")
        profile = _section(root, "body_profile")
        motion = _section(root, "motion")
        clips = _section(root, "clips")
        behavior = _section(root, "behavior")
        vmc_idle = _section(root, "vmc_idle")
        vrchat_osc = _section(root, "vrchat_osc")
        safety = _section(root, "safety")

        host = str(anyadance.get("host", "127.0.0.1")).strip()
        if not host:
            raise ValueError("anyadance.host must not be empty")

        body_profile = BodyProfile(
            height_m=_finite_float(profile.get("height_m"), 1.50, minimum=0.8, maximum=2.0, name="body_profile.height_m"),
            shoulder_width_m=_finite_float(profile.get("shoulder_width_m"), 0.36, minimum=0.15, maximum=0.8, name="body_profile.shoulder_width_m"),
            shoulder_drop_m=_finite_float(profile.get("shoulder_drop_m"), 0.18, minimum=0.05, maximum=0.5, name="body_profile.shoulder_drop_m"),
            arm_length_m=_finite_float(profile.get("arm_length_m"), 0.58, minimum=0.25, maximum=1.0, name="body_profile.arm_length_m"),
        )
        safety_config = SafetyConfig(
            max_position_abs_m=_finite_float(safety.get("max_position_abs_m"), 3.0, minimum=0.5, maximum=10.0, name="safety.max_position_abs_m"),
            max_y_m=_finite_float(safety.get("max_y_m"), 2.0, minimum=1.0, maximum=2.0, name="safety.max_y_m"),
            max_linear_speed_mps=_finite_float(safety.get("max_linear_speed_mps"), 2.0, minimum=0.1, maximum=10.0, name="safety.max_linear_speed_mps"),
            max_angular_speed_dps=_finite_float(safety.get("max_angular_speed_dps"), 360.0, minimum=30.0, maximum=1440.0, name="safety.max_angular_speed_dps"),
            max_action_duration_ms=_bounded_int(safety.get("max_action_duration_ms"), 5000, minimum=100, maximum=30000, name="safety.max_action_duration_ms"),
        )
        clip_directory = str(clips.get("directory", "motions")).strip()
        if not clip_directory or any(char in clip_directory for char in ("/", "\\", ":")) or clip_directory in {".", ".."}:
            raise ValueError("clips.directory must be a single relative directory name")

        def osc_host(key: str, default: str) -> str:
            value = str(vrchat_osc.get(key, default)).strip()
            if not value or "\x00" in value:
                raise ValueError(f"vrchat_osc.{key} must not be empty")
            return value

        raw_awareness = vrchat_osc.get("awareness_parameters", VrchatOscConfig.awareness_parameters)
        if not isinstance(raw_awareness, (list, tuple)) or len(raw_awareness) > 32:
            raise ValueError("vrchat_osc.awareness_parameters must be an array with at most 32 names")
        awareness_parameters: list[str] = []
        for raw_name in raw_awareness:
            name = str(raw_name).strip()
            if not name or len(name) > 128 or "/" in name or "\x00" in name:
                raise ValueError("vrchat_osc.awareness_parameters contains an invalid parameter name")
            if name not in awareness_parameters:
                awareness_parameters.append(name)
        osc_config = VrchatOscConfig(
            enabled=_boolean(vrchat_osc.get("enabled"), True, name="vrchat_osc.enabled"),
            send_host=osc_host("send_host", "127.0.0.1"),
            send_port=_bounded_int(vrchat_osc.get("send_port"), 9000, minimum=1, maximum=65535, name="vrchat_osc.send_port"),
            listen_host=osc_host("listen_host", "127.0.0.1"),
            listen_port=_bounded_int(vrchat_osc.get("listen_port"), 9001, minimum=1, maximum=65535, name="vrchat_osc.listen_port"),
            allowed_sender=osc_host("allowed_sender", "127.0.0.1"),
            input_pulse_ms=_bounded_int(vrchat_osc.get("input_pulse_ms"), 100, minimum=20, maximum=1000, name="vrchat_osc.input_pulse_ms"),
            parameter_cache_size=_bounded_int(vrchat_osc.get("parameter_cache_size"), 256, minimum=16, maximum=2048, name="vrchat_osc.parameter_cache_size"),
            awareness_parameters=tuple(awareness_parameters),
        )
        behavior_config = BehaviorConfig(
            default_crossfade_ms=_bounded_int(
                behavior.get("default_crossfade_ms"),
                400,
                minimum=0,
                maximum=5000,
                name="behavior.default_crossfade_ms",
            ),
            protect_full_body_motion=_boolean(
                behavior.get("protect_full_body_motion"),
                True,
                name="behavior.protect_full_body_motion",
            ),
            prefer_vmd_expressions=_boolean(
                behavior.get("prefer_vmd_expressions"),
                True,
                name="behavior.prefer_vmd_expressions",
            ),
            transition_history_size=_bounded_int(
                behavior.get("transition_history_size"),
                16,
                minimum=4,
                maximum=64,
                name="behavior.transition_history_size",
            ),
        )
        vmc_idle_config = VmcIdleConfig(
            enabled=_boolean(vmc_idle.get("enabled"), True, name="vmc_idle.enabled"),
            listen_host=str(vmc_idle.get("listen_host", "127.0.0.1")).strip(),
            listen_port=_bounded_int(
                vmc_idle.get("listen_port"),
                39539,
                minimum=1,
                maximum=65535,
                name="vmc_idle.listen_port",
            ),
            allowed_sender=str(vmc_idle.get("allowed_sender", "127.0.0.1")).strip(),
            stale_after_ms=_bounded_int(
                vmc_idle.get("stale_after_ms"),
                500,
                minimum=100,
                maximum=5000,
                name="vmc_idle.stale_after_ms",
            ),
            manage_host_output=_boolean(
                vmc_idle.get("manage_host_output"),
                True,
                name="vmc_idle.manage_host_output",
            ),
            host_api_url=str(
                vmc_idle.get("host_api_url", "http://127.0.0.1:48911")
            ).strip(),
            host_api_timeout_seconds=_finite_float(
                vmc_idle.get("host_api_timeout_seconds"),
                3.0,
                minimum=0.2,
                maximum=30.0,
                name="vmc_idle.host_api_timeout_seconds",
            ),
            host_output_host=str(
                vmc_idle.get("host_output_host", "127.0.0.1")
            ).strip(),
            host_send_rate_hz=_bounded_int(
                vmc_idle.get("host_send_rate_hz"),
                60,
                minimum=1,
                maximum=120,
                name="vmc_idle.host_send_rate_hz",
            ),
        )
        if not vmc_idle_config.listen_host or "\x00" in vmc_idle_config.listen_host:
            raise ValueError("vmc_idle.listen_host must not be empty")
        if not vmc_idle_config.allowed_sender or "\x00" in vmc_idle_config.allowed_sender:
            raise ValueError("vmc_idle.allowed_sender must not be empty")
        if not vmc_idle_config.host_api_url or "\x00" in vmc_idle_config.host_api_url:
            raise ValueError("vmc_idle.host_api_url must not be empty")
        if not vmc_idle_config.host_output_host or "\x00" in vmc_idle_config.host_output_host:
            raise ValueError("vmc_idle.host_output_host must not be empty")

        return cls(
            host=host,
            port=_bounded_int(anyadance.get("port"), 39570, minimum=1, maximum=65535, name="anyadance.port"),
            rate_hz=_bounded_int(anyadance.get("rate_hz"), 60, minimum=10, maximum=120, name="anyadance.rate_hz"),
            default_duration_ms=_bounded_int(motion.get("default_duration_ms"), 600, minimum=100, maximum=safety_config.max_action_duration_ms, name="motion.default_duration_ms"),
            max_queue_size=_bounded_int(motion.get("max_queue_size"), 8, minimum=1, maximum=64, name="motion.max_queue_size"),
            clip_directory=clip_directory,
            clip_max_file_bytes=_bounded_int(clips.get("max_file_bytes"), 64 * 1024 * 1024, minimum=1024, maximum=256 * 1024 * 1024, name="clips.max_file_bytes"),
            clip_max_frames=_bounded_int(clips.get("max_frames"), 18000, minimum=1, maximum=100000, name="clips.max_frames"),
            clip_max_duration_seconds=_finite_float(clips.get("max_duration_seconds"), 300.0, minimum=0.1, maximum=3600.0, name="clips.max_duration_seconds"),
            behavior=behavior_config,
            vmc_idle=vmc_idle_config,
            vrchat_osc=osc_config,
            profile=body_profile,
            safety=safety_config,
        )
