"""基础语义动作的确定性编译；精确路径与接触由执行层提供。"""
from __future__ import annotations

BASE_PROMPTS = {
    "idle": "A person stands relaxed in place, subtly shifts weight, arms resting naturally.",
    "talk": "A person stands in place and talks with restrained, gentle hand gestures.",
    "walk": "A person walks naturally toward a target and comes to a balanced stop.",
    "observe": "A person stands in place and gently turns their head to observe a target.",
}


def compile_motion_intent(activity, *, explicit_stop=False, chat_engaged=False):
    """状态约束优先于自由描述；输出语义边界，不让模型生成坐标。"""
    kind = activity.get("kind", "linger")
    stationary = explicit_stop or chat_engaged or kind in {"linger", "observe", "perform"}
    stationary = stationary or (kind == "local_roam" and activity.get("style") == "stay_and_look")
    base = "idle" if explicit_stop else "talk" if chat_engaged else "observe" if kind == "observe" else "idle" if stationary else "walk"
    description = activity.get("motion_description")
    if description is not None and (not isinstance(description, str) or not 1 <= len(description.strip()) <= 320):
        raise ValueError("motion_description 必须为 1..320 字符")
    style = activity.get("motion_style", "natural")
    if style not in {"natural", "relaxed", "gentle", "lively"}:
        raise ValueError("motion_style 无效")
    prompt = BASE_PROMPTS[base] if explicit_stop or chat_engaged or not description else description.strip()
    return {"version": 1, "prompt": prompt, "style": style,
            "movement": "blocked" if stationary else "planned_only",
            "target_key": activity.get("target_key"), "player_slot": activity.get("player_slot"),
            "duration_s": activity.get("duration_s", 10), "action_key": activity.get("action_key")}
