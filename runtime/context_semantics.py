"""环境语义投影：量化距离，避免时间戳和连续位置淹没主模型。"""
import json
import math


def semantic_projection(value):
    ignored = {"timestamp", "updated_at", "age_ms", "d", "brg", "position", "velocity", "x", "y", "z", "yaw"}
    if isinstance(value, dict):
        result = {key: semantic_projection(item) for key, item in value.items() if key not in ignored}
        distance = value.get("d")
        if type(distance) in (int, float) and math.isfinite(distance) and distance >= 0:
            result["distance_band"] = "within_reach" if distance < 1 else "near" if distance < 3 else "far"
        return result
    if isinstance(value, list):
        return sorted((semantic_projection(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return round(value, 1) if isinstance(value, float) else value
