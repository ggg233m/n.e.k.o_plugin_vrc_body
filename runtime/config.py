"""N.E.K.O 配置到 YUI 运行时配置的严格投影。"""

from __future__ import annotations

from dataclasses import dataclass, field
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


def _string(
    value: Any,
    *,
    name: str,
    allow_empty: bool = False,
    maximum_length: int = 2048,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是字符串")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"{name} 不能为空")
    if len(normalized) > maximum_length:
        raise ValueError(f"{name} 不能超过 {maximum_length} 个字符")
    return normalized


def _bounded_number(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    number = _finite_number(value, name=name, minimum=minimum)
    if number > maximum:
        raise ValueError(f"{name} 必须不大于 {maximum}")
    return number


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} 必须是 {minimum}..{maximum} 的整数")
    return value


def _number_range(
    value: Any,
    *,
    name: str,
    minimum: float,
) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise ValueError(f"{name} 必须是包含两个数值的数组")
    lower = _finite_number(value[0], name=f"{name}[0]", minimum=minimum)
    upper = _finite_number(value[1], name=f"{name}[1]", minimum=lower)
    return lower, upper


@dataclass(frozen=True, slots=True)
class YuiChatContextConfig:
    """动作模型使用的只读 N.E.K.O 近期聊天文件配置。"""

    enabled: bool = False
    source: str = "recent_file"
    max_turns: int = 6
    max_chars: int = 6000
    poll_interval_s: float = 1.0
    max_file_bytes: int = 2 * 1024 * 1024

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "YuiChatContextConfig":
        defaults = cls()
        data = dict(source or {})
        source_name = _string(
            data.get("source", defaults.source),
            name="autonomy.intent_model.chat_context.source",
            maximum_length=32,
        )
        if source_name != "recent_file":
            raise ValueError(
                "autonomy.intent_model.chat_context.source 只能是 recent_file"
            )
        return cls(
            enabled=_boolean(
                data.get("enabled", defaults.enabled),
                name="autonomy.intent_model.chat_context.enabled",
            ),
            source=source_name,
            max_turns=_integer(
                data.get("max_turns", defaults.max_turns),
                name="autonomy.intent_model.chat_context.max_turns",
                minimum=1,
                maximum=20,
            ),
            max_chars=_integer(
                data.get("max_chars", defaults.max_chars),
                name="autonomy.intent_model.chat_context.max_chars",
                minimum=256,
                maximum=32_000,
            ),
            poll_interval_s=_bounded_number(
                data.get("poll_interval_s", defaults.poll_interval_s),
                name="autonomy.intent_model.chat_context.poll_interval_s",
                minimum=0.1,
                maximum=60.0,
            ),
            max_file_bytes=_integer(
                data.get("max_file_bytes", defaults.max_file_bytes),
                name="autonomy.intent_model.chat_context.max_file_bytes",
                minimum=1024,
                maximum=16 * 1024 * 1024,
            ),
        )


@dataclass(frozen=True, slots=True)
class YuiChatBridgeConfig:
    """主对话输出到 NPC 头顶的只读磁盘桥接配置。"""

    enabled: bool = True
    source: str = "recent_file"
    poll_interval_s: float = 0.5
    # 每页的最短保留时间；桥接层会按可见字符数自动延长，避免长回答来不及阅读。
    display_seconds: int = 15
    max_pages: int = 4
    max_file_bytes: int = 2 * 1024 * 1024

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "YuiChatBridgeConfig":
        defaults = cls()
        data = dict(source or {})
        source_name = _string(
            data.get("source", defaults.source),
            name="chat_bridge.source",
            maximum_length=32,
        )
        if source_name != "recent_file":
            raise ValueError("chat_bridge.source 只能是 recent_file")
        return cls(
            enabled=_boolean(
                data.get("enabled", defaults.enabled),
                name="chat_bridge.enabled",
            ),
            source=source_name,
            poll_interval_s=_bounded_number(
                data.get("poll_interval_s", defaults.poll_interval_s),
                name="chat_bridge.poll_interval_s",
                minimum=0.1,
                maximum=60.0,
            ),
            display_seconds=_integer(
                data.get("display_seconds", defaults.display_seconds),
                name="chat_bridge.display_seconds",
                minimum=1,
                maximum=127,
            ),
            max_pages=_integer(
                data.get("max_pages", defaults.max_pages),
                name="chat_bridge.max_pages",
                minimum=1,
                maximum=8,
            ),
            max_file_bytes=_integer(
                data.get("max_file_bytes", defaults.max_file_bytes),
                name="chat_bridge.max_file_bytes",
                minimum=1024,
                maximum=16 * 1024 * 1024,
            ),
        )


@dataclass(frozen=True, slots=True)
class YuiPlayerChatConfig:
    """世界内自定义输入框到宿主主对话的显式消息桥。"""

    enabled: bool = True
    max_chars: int = 144
    cooldown_s: float = 2.0

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "YuiPlayerChatConfig":
        defaults = cls()
        data = dict(source or {})
        return cls(
            enabled=_boolean(
                data.get("enabled", defaults.enabled),
                name="player_chat.enabled",
            ),
            max_chars=_integer(
                data.get("max_chars", defaults.max_chars),
                name="player_chat.max_chars",
                minimum=1,
                maximum=144,
            ),
            cooldown_s=_bounded_number(
                data.get("cooldown_s", defaults.cooldown_s),
                name="player_chat.cooldown_s",
                minimum=0.5,
                maximum=60.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class YuiIntentModelConfig:
    """与宿主聊天彻底隔离的 OpenAI 兼容意图模型。"""

    enabled: bool = False
    endpoint: str = ""
    model: str = "gemini-3.7-flash"
    api_key_env: str = "TEST_API"
    persona_prompt: str = "你是友好、好奇且会自然安排生活片段的世界 NPC。"
    timeout_s: float = 20.0
    min_interval_s: float = 30.0
    temperature: float = 0.7
    max_output_tokens: int = 700
    chat_context: YuiChatContextConfig = field(default_factory=YuiChatContextConfig)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "YuiIntentModelConfig":
        defaults = cls()
        data = dict(source or {})
        # 留空表示不配置中转站；此时意图模型不会被启用，规则循环照常运行。
        endpoint = _string(
            data.get("endpoint", defaults.endpoint),
            name="autonomy.intent_model.endpoint",
            allow_empty=True,
        )
        if endpoint and not endpoint.startswith("https://"):
            raise ValueError("autonomy.intent_model.endpoint 必须使用 https://")
        api_key_env = _string(
            data.get("api_key_env", defaults.api_key_env),
            name="autonomy.intent_model.api_key_env",
            allow_empty=True,
            maximum_length=128,
        )
        if api_key_env and not api_key_env.replace("_", "").isalnum():
            raise ValueError("autonomy.intent_model.api_key_env 只能包含字母、数字和下划线")
        return cls(
            enabled=_boolean(
                data.get("enabled", defaults.enabled),
                name="autonomy.intent_model.enabled",
            ),
            endpoint=endpoint,
            model=_string(
                data.get("model", defaults.model),
                name="autonomy.intent_model.model",
                maximum_length=160,
            ),
            api_key_env=api_key_env,
            persona_prompt=_string(
                data.get("persona_prompt", defaults.persona_prompt),
                name="autonomy.intent_model.persona_prompt",
                maximum_length=1000,
            ),
            timeout_s=_bounded_number(
                data.get("timeout_s", defaults.timeout_s),
                name="autonomy.intent_model.timeout_s",
                minimum=1.0,
                maximum=120.0,
            ),
            min_interval_s=_bounded_number(
                data.get("min_interval_s", defaults.min_interval_s),
                name="autonomy.intent_model.min_interval_s",
                minimum=1.0,
                maximum=3600.0,
            ),
            temperature=_bounded_number(
                data.get("temperature", defaults.temperature),
                name="autonomy.intent_model.temperature",
                minimum=0.0,
                maximum=2.0,
            ),
            max_output_tokens=_integer(
                data.get("max_output_tokens", defaults.max_output_tokens),
                name="autonomy.intent_model.max_output_tokens",
                minimum=128,
                maximum=4096,
            ),
            chat_context=YuiChatContextConfig.from_mapping(
                data.get("chat_context")
                if isinstance(data.get("chat_context"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class YuiChatEngagementConfig:
    """玩家世界聊天期间的近距离陪伴配置。"""

    enabled: bool = True
    near_distance_m: float = 2.5
    follow_trigger_m: float = 3.0
    approach_distance_m: float = 1.5
    post_reply_hold_s: float = 15.0
    no_reply_timeout_s: float = 90.0
    approach_retry_s: float = 10.0

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "YuiChatEngagementConfig":
        defaults = cls()
        data = dict(source or {})
        value = cls(
            enabled=_boolean(
                data.get("enabled", defaults.enabled),
                name="autonomy.chat_engagement.enabled",
            ),
            near_distance_m=_bounded_number(
                data.get("near_distance_m", defaults.near_distance_m),
                name="autonomy.chat_engagement.near_distance_m",
                minimum=0.5,
                maximum=8.0,
            ),
            follow_trigger_m=_bounded_number(
                data.get("follow_trigger_m", defaults.follow_trigger_m),
                name="autonomy.chat_engagement.follow_trigger_m",
                minimum=0.5,
                maximum=10.0,
            ),
            approach_distance_m=_bounded_number(
                data.get("approach_distance_m", defaults.approach_distance_m),
                name="autonomy.chat_engagement.approach_distance_m",
                minimum=0.5,
                maximum=5.0,
            ),
            post_reply_hold_s=_bounded_number(
                data.get("post_reply_hold_s", defaults.post_reply_hold_s),
                name="autonomy.chat_engagement.post_reply_hold_s",
                minimum=0.0,
                maximum=300.0,
            ),
            no_reply_timeout_s=_bounded_number(
                data.get("no_reply_timeout_s", defaults.no_reply_timeout_s),
                name="autonomy.chat_engagement.no_reply_timeout_s",
                minimum=5.0,
                maximum=600.0,
            ),
            approach_retry_s=_bounded_number(
                data.get("approach_retry_s", defaults.approach_retry_s),
                name="autonomy.chat_engagement.approach_retry_s",
                minimum=1.0,
                maximum=120.0,
            ),
        )
        if not (
            value.approach_distance_m
            < value.near_distance_m
            <= value.follow_trigger_m
        ):
            raise ValueError(
                "autonomy.chat_engagement 距离必须满足 "
                "approach_distance_m < near_distance_m <= follow_trigger_m"
            )
        return value


@dataclass(frozen=True, slots=True)
class YuiAutonomyConfig:
    """宿主常驻自主循环配置；默认关闭，避免改变通用配置行为。"""

    enabled: bool = False
    auto_connect: bool = False
    decision_interval_s: float = 1.0
    resume_delay_s: float = 8.0
    # 自主移动默认保持在 Animator 的 Walk 区间；显式工具仍可单独请求更高速度。
    walk_speed_mps: float = 1.0
    dwell_range_s: tuple[float, float] = (8.0, 20.0)
    explore_range_s: tuple[float, float] = (15.0, 35.0)
    social_cooldown_s: float = 60.0
    llm_inspiration_range_s: tuple[float, float] = (180.0, 360.0)
    chat_engagement: YuiChatEngagementConfig = field(
        default_factory=YuiChatEngagementConfig
    )
    intent_model: YuiIntentModelConfig = field(default_factory=YuiIntentModelConfig)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "YuiAutonomyConfig":
        defaults = cls()
        data = dict(source or {})
        return cls(
            enabled=_boolean(data.get("enabled", defaults.enabled), name="autonomy.enabled"),
            auto_connect=_boolean(
                data.get("auto_connect", defaults.auto_connect),
                name="autonomy.auto_connect",
            ),
            decision_interval_s=_finite_number(
                data.get("decision_interval_s", defaults.decision_interval_s),
                name="autonomy.decision_interval_s",
                minimum=0.1,
            ),
            resume_delay_s=_finite_number(
                data.get("resume_delay_s", defaults.resume_delay_s),
                name="autonomy.resume_delay_s",
                minimum=0.0,
            ),
            walk_speed_mps=_finite_number(
                data.get("walk_speed_mps", defaults.walk_speed_mps),
                name="autonomy.walk_speed_mps",
                minimum=0.1,
            ),
            dwell_range_s=_number_range(
                data.get("dwell_range_s", defaults.dwell_range_s),
                name="autonomy.dwell_range_s",
                minimum=1.0,
            ),
            explore_range_s=_number_range(
                data.get("explore_range_s", defaults.explore_range_s),
                name="autonomy.explore_range_s",
                minimum=1.0,
            ),
            social_cooldown_s=_finite_number(
                data.get("social_cooldown_s", defaults.social_cooldown_s),
                name="autonomy.social_cooldown_s",
                minimum=0.0,
            ),
            llm_inspiration_range_s=_number_range(
                data.get(
                    "llm_inspiration_range_s",
                    defaults.llm_inspiration_range_s,
                ),
                name="autonomy.llm_inspiration_range_s",
                minimum=1.0,
            ),
            chat_engagement=YuiChatEngagementConfig.from_mapping(
                data.get("chat_engagement")
                if isinstance(data.get("chat_engagement"), Mapping)
                else {}
            ),
            intent_model=YuiIntentModelConfig.from_mapping(
                data.get("intent_model")
                if isinstance(data.get("intent_model"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class YuiPluginConfig:
    """只保存 YUI 世界原生 NPC 方案需要的配置。"""

    midi_port: str = "NEKO_MIDI"
    claim_code: int = 0
    log_path: str | None = None
    log_directory: str | None = None
    log_from_end: bool = True
    log_poll_interval_s: float = 0.05
    ack_timeout_s: float = 2.0
    command_deadline_s: float = 5.0
    heartbeat_interval_s: float = 1.0
    free_coordinate_navigation: bool = False
    include_player_names: bool = False
    enable_wander_tool: bool = False
    chat_bridge: YuiChatBridgeConfig = field(default_factory=YuiChatBridgeConfig)
    player_chat: YuiPlayerChatConfig = field(default_factory=YuiPlayerChatConfig)
    autonomy: YuiAutonomyConfig = field(default_factory=YuiAutonomyConfig)

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
            enable_wander_tool=_boolean(
                data.get("enable_wander_tool", defaults.enable_wander_tool),
                name="enable_wander_tool",
            ),
            chat_bridge=YuiChatBridgeConfig.from_mapping(
                data.get("chat_bridge")
                if isinstance(data.get("chat_bridge"), Mapping)
                else {}
            ),
            player_chat=YuiPlayerChatConfig.from_mapping(
                data.get("player_chat")
                if isinstance(data.get("player_chat"), Mapping)
                else {}
            ),
            autonomy=YuiAutonomyConfig.from_mapping(
                data.get("autonomy") if isinstance(data.get("autonomy"), Mapping) else {}
            ),
        )


__all__ = [
    "YuiAutonomyConfig",
    "YuiChatEngagementConfig",
    "YuiChatBridgeConfig",
    "YuiChatContextConfig",
    "YuiIntentModelConfig",
    "YuiPlayerChatConfig",
    "YuiPluginConfig",
]
