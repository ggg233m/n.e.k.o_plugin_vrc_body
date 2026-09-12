"""人工面板配置白名单；不向浏览器传递密钥或聊天内容。"""
import math
import os
import tempfile
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import tomli_w

from .config import YuiPluginConfig


SECRET = "autonomy.intent_model.api_key"
CLEAR_SECRET = "autonomy.intent_model.clear_api_key"

FIELDS = {
    "midi_port": ("MIDI 输出端口", str, None, None),
    "claim_code": ("世界控制码", int, 0, 16383),
    "log_path": ("VRChat / Unity 日志文件", str, None, None),
    "log_directory": ("VRChat 日志目录", str, None, None),
    "ardy.endpoint": ("ARDY 本机服务地址", str, None, None),
    "autonomy.intent_model.endpoint": ("模型 API 地址", str, None, None),
    "autonomy.intent_model.model": ("模型名称", str, None, None),
    SECRET: ("模型 API Key", str, None, None),
    CLEAR_SECRET: ("清除已保存的 API Key", bool, None, None),
    "autonomy.intent_model.api_key_env": ("备用密钥环境变量", str, None, None),
    "autonomy.intent_model.persona_prompt": ("动作模型角色提示词", str, None, None),
    "autonomy.intent_model.timeout_s": ("模型超时（秒）", float, 1, 120),
    "autonomy.intent_model.min_interval_s": ("自主意图最短间隔（秒）", float, 1, 3600),
    "autonomy.intent_model.max_output_tokens": ("模型输出 Token 上限", int, 128, 4096),
    "ardy.enabled": ("ARDY 动作后端（需要世界执行端）", bool, None, None),
    "chat_bridge.enabled": ("角色字幕", bool, None, None),
    "chat_bridge.display_seconds": ("字幕最短秒数", int, 10, 25),
    "chat_bridge.max_pages": ("回复最多页数", int, 1, 4),
    "player_chat.enabled": ("世界聊天触发回复", bool, None, None),
    "autonomy.enabled": ("自主生活", bool, None, None),
    "autonomy.auto_connect": ("自动连接世界", bool, None, None),
    "autonomy.chat_engagement.enabled": ("对话陪伴锁", bool, None, None),
    "autonomy.proactive_chat_enabled": ("主动找玩家搭话", bool, None, None),
    "autonomy.intent_model.enabled": ("独立意图模型", bool, None, None),
    "autonomy.intent_model.chat_context.enabled": ("聊天记忆辅助动作", bool, None, None),
}


def settings_view(config):
    result = {}
    for path in FIELDS:
        if path in {SECRET, CLEAR_SECRET}:
            result[path] = "" if path == SECRET else False
            continue
        fallback = "" if FIELDS[path][1] is str else False
        value = config
        for key in path.split("."):
            value = value.get(key, fallback) if isinstance(value, Mapping) else getattr(value, key)
        result[path] = value if value is not None else fallback
    return result


def action_progress_view(autonomy, plan):
    """动作概览只读取已发布状态，不把模型动机或世界对白带入进度卡片。"""
    chat = autonomy.get("chat_engagement") or {}
    return {
        "state": autonomy.get("state", "not_initialized"),
        "activity": autonomy.get("panel_activity"),
        "plan": plan,
        "chat": {key: chat.get(key) for key in ("active", "phase", "player_slot", "remaining_s", "retry_in_s", "input_supported")},
        "proactive": autonomy.get("proactive_chat"),
        "pending_intent": bool(autonomy.get("pending_intent")),
        "preference": autonomy.get("current_preference"),
    }


def merge_profile_patch(base, patch):
    """按 SDK 的配置档语义递归合并，不修改调用方传入对象。"""
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(merged.get(key), dict) and isinstance(value, Mapping):
            merged[key] = merge_profile_patch(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def persist_profile_compat(profile_name, config, config_roots):
    """兼容缺少配置档写入通道的宿主，只改已有且已登记的插件配置档。"""
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("配置档名称无效")
    if not isinstance(config, Mapping) or "plugin" in config:
        raise ValueError("配置档内容无效")

    checked_roots = set()
    for root_value in config_roots:
        try:
            root = Path(root_value).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        root_key = os.path.normcase(str(root))
        if root_key in checked_roots:
            continue
        checked_roots.add(root_key)

        profiles_path = root / "profiles.toml"
        if not profiles_path.is_file():
            continue
        with profiles_path.open("rb") as stream:
            profiles_data = tomllib.load(stream)
        profiles_cfg = profiles_data.get("config_profiles")
        files = profiles_cfg.get("files") if isinstance(profiles_cfg, Mapping) else None
        raw_path = files.get(profile_name) if isinstance(files, Mapping) else None
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        expanded = os.path.expandvars(os.path.expanduser(raw_path))
        target = Path(expanded)
        if not target.is_absolute():
            target = root / target
        target = target.resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("配置档路径超出插件配置目录") from exc
        if target == profiles_path or target.suffix.lower() != ".toml" or not target.parent.is_dir():
            raise ValueError("配置档路径无效")

        payload = tomli_w.dumps(dict(config)).encode("utf-8")
        temp_fd, temp_name = tempfile.mkstemp(prefix=".yui_profile_", suffix=".toml", dir=str(target.parent))
        temp_path = Path(temp_name)
        try:
            with os.fdopen(temp_fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return

    raise ValueError("未找到当前配置档的受限写入路径")


def validated_patch(raw, changes):
    if not isinstance(changes, dict) or not changes:
        raise ValueError("没有可保存的修改")
    if set(changes) - FIELDS.keys():
        raise ValueError("提交中包含面板不支持的设置")
    candidate = deepcopy(raw.get("yui", {}))
    patch = {}
    if changes.get(CLEAR_SECRET) is True and changes.get(SECRET):
        raise ValueError("不能同时设置和清除密钥")
    for path, value in changes.items():
        label, kind, low, high = FIELDS[path]
        valid = type(value) is kind
        if kind is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"{label}的数据类型无效")
        if kind in (int, float) and not low <= value <= high:
            raise ValueError(f"{label}必须在 {low}..{high} 范围内")
        if kind is str and (len(value) > 4096 or "\0" in value):
            raise ValueError(f"{label}包含无效内容或长度超过限制")
        if path == SECRET and not value.strip():
            continue  # 密码框留空表示不修改。
        if path == CLEAR_SECRET:
            if not value:
                continue
            path, value = SECRET, ""
        keys = path.split(".")
        for root in (candidate, patch):
            node = root
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
    try:
        YuiPluginConfig.from_mapping(candidate)
    except ValueError as exc:
        # 配置校验器只返回字段规则，不包含字段值；去掉换行并限制长度，
        # 避免把底层对象或未来新增的敏感信息透传到浏览器。
        detail = " ".join(str(exc).split())[:200]
        raise ValueError(detail or "设置值未通过配置校验") from None
    return {"yui": patch}
