"""人工面板配置白名单；不向浏览器传递密钥或聊天内容。"""
from copy import deepcopy
from .config import YuiPluginConfig


FIELDS = {
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
        value = config
        for key in path.split("."):
            value = getattr(value, key)
        result[path] = value
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


def validated_patch(raw, changes):
    if not isinstance(changes, dict) or not changes or set(changes) - FIELDS.keys():
        raise ValueError("invalid_fields")
    candidate = deepcopy(raw.get("yui", {}))
    patch = {}
    for path, value in changes.items():
        _, kind, low, high = FIELDS[path]
        if type(value) is not kind or (kind is int and not low <= value <= high):
            raise ValueError("invalid_value")
        keys = path.split(".")
        for root in (candidate, patch):
            node = root
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
    YuiPluginConfig.from_mapping(candidate)
    return {"yui": patch}
