"""N.E.K.O 配置到 YUI 运行时配置的严格投影。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _finite_number(value: Any, *, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值")
    number = float(value)
    if number < minimum or number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} 必须是不小于 {minimum} 的有限数值")
    return number


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


@dataclass(frozen=True, slots=True)
class YuiPluginConfig:
    """只保存 YUI 世界原生 NPC 方案需要的配置。"""

    midi_port: str = "NEKO_MIDI"
    claim_code: int = 0
    log_path: str | None = None
    log_directory: str | None = None
    log_from_end: bool = True
    log_poll_interval_s: float = 0.1
    ack_timeout_s: float = 2.0
    command_deadline_s: float = 5.0
    heartbeat_interval_s: float = 1.0
    free_coordinate_navigation: bool = False
    include_player_names: bool = False
    host_arm_authorized: bool = False
    enable_wander_tool: bool = False

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "YuiPluginConfig":
        defaults = cls()
        data = dict(source or {})
        midi_port = data.get("midi_port", defaults.midi_port)
        if not isinstance(midi_port, str) or not midi_port.strip():
            raise ValueError("midi_port 必须是非空字符串")

        claim_code = data.get("claim_code", defaults.claim_code)
        if isinstance(claim_code, bool) or not isinstance(claim_code, int) or not 0 <= claim_code <= 16383:
            raise ValueError("claim_code 必须是 0..16383 的整数")

        log_path = data.get("log_path") or None
        log_directory = data.get("log_directory") or None
        for name, value in (("log_path", log_path), ("log_directory", log_directory)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} 必须是字符串")
        if log_path is not None and log_directory is not None:
            raise ValueError("log_path 与 log_directory 只能配置一个")

        ack_timeout_s = _finite_number(
            data.get("ack_timeout_s", defaults.ack_timeout_s),
            name="ack_timeout_s",
            minimum=0.01,
        )
        command_deadline_s = _finite_number(
            data.get("command_deadline_s", defaults.command_deadline_s),
            name="command_deadline_s",
            minimum=ack_timeout_s,
        )

        return cls(
            midi_port=midi_port.strip(),
            claim_code=claim_code,
            log_path=log_path.strip() if isinstance(log_path, str) and log_path.strip() else None,
            log_directory=(
                log_directory.strip()
                if isinstance(log_directory, str) and log_directory.strip()
                else None
            ),
            log_from_end=_boolean(
                data.get("log_from_end", defaults.log_from_end),
                name="log_from_end",
            ),
            log_poll_interval_s=_finite_number(
                data.get("log_poll_interval_s", defaults.log_poll_interval_s),
                name="log_poll_interval_s",
                minimum=0.02,
            ),
            ack_timeout_s=ack_timeout_s,
            command_deadline_s=command_deadline_s,
            heartbeat_interval_s=_finite_number(
                data.get("heartbeat_interval_s", defaults.heartbeat_interval_s),
                name="heartbeat_interval_s",
                minimum=0.05,
            ),
            free_coordinate_navigation=_boolean(
                data.get("free_coordinate_navigation", defaults.free_coordinate_navigation),
                name="free_coordinate_navigation",
            ),
            include_player_names=_boolean(
                data.get("include_player_names", defaults.include_player_names),
                name="include_player_names",
            ),
            host_arm_authorized=_boolean(
                data.get("host_arm_authorized", defaults.host_arm_authorized),
                name="host_arm_authorized",
            ),
            enable_wander_tool=_boolean(
                data.get("enable_wander_tool", defaults.enable_wander_tool),
                name="enable_wander_tool",
            ),
        )


__all__ = ["YuiPluginConfig"]
