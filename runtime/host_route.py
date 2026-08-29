"""宿主/测试专用的连续坐标路线编排。

本模块不会注册为 LLM 工具。它只使用冻结 v1.1 的 GOTO_XZ：在当前目标进入
预切半径时发送下一条 GOTO_XZ，让 Unity 保留 NavMeshAgent 速度；中间操作按协议
进入 cancelled/replaced，只有最后一段必须完成。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .yui_adapter import YuiSemanticAdapter
from .yui_session import YuiSessionState


_VALID_OPERATION_STATUSES = {
    "accepted",
    "running",
    "succeeded",
    "cancelled",
    "failed",
    "unknown",
}


class YuiContinuousRouteRunner:
    """在协议单移动通道上编排无停车的多点路线。"""

    def __init__(self, adapter: YuiSemanticAdapter, session: YuiSessionState) -> None:
        self.adapter = adapter
        self.session = session

    @staticmethod
    def _point(value: Mapping[str, Any], index: int) -> dict[str, float | None]:
        if not isinstance(value, Mapping):
            raise ValueError(f"points[{index}] 必须是对象")
        unknown = set(value) - {"x", "z", "yaw"}
        if unknown:
            raise ValueError(f"points[{index}] 包含未知字段: {', '.join(sorted(unknown))}")
        if "x" not in value or "z" not in value:
            raise ValueError(f"points[{index}] 必须包含 x/z")
        x = float(value["x"])
        z = float(value["z"])
        yaw = None if value.get("yaw") is None else float(value["yaw"])
        if not math.isfinite(x) or not math.isfinite(z) or (yaw is not None and not math.isfinite(yaw)):
            raise ValueError(f"points[{index}] 坐标必须是有限数")
        return {"x": x, "z": z, "yaw": yaw}

    @staticmethod
    def _failed(error: str, detail: str, **extra: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "failed",
            "error": error,
            "detail": detail,
            "midi_sent": False,
        }
        result.update(extra)
        return result

    def run(
        self,
        points: Sequence[Mapping[str, Any]],
        *,
        speed_mps: float | None = None,
        handoff_distance_m: float = 1.5,
        minimum_handoff_speed_mps: float = 0.1,
        handoff_timeout_s: float = 30.0,
        final_timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """执行连续路线，并用 npc.state.speed 验证中间交接没有停车。"""
        try:
            route = [self._point(item, index) for index, item in enumerate(points)]
            handoff_distance = float(handoff_distance_m)
            minimum_speed = float(minimum_handoff_speed_mps)
            if not route:
                raise ValueError("points 至少包含一个目标")
            if not math.isfinite(handoff_distance) or handoff_distance <= 0.0:
                raise ValueError("handoff_distance_m 必须是正有限数")
            if not math.isfinite(minimum_speed) or minimum_speed < 0.0:
                raise ValueError("minimum_handoff_speed_mps 必须是非负有限数")
        except (TypeError, ValueError) as exc:
            return self._failed("invalid_param", str(exc))

        skipped_leading_points = 0
        observe = getattr(self.session, "observe", None)
        if callable(observe) and len(route) > 1:
            npc = observe().get("npc", {})
            position = npc.get("pos") if isinstance(npc, Mapping) else None
            if isinstance(position, list) and len(position) == 3:
                current_x = float(position[0])
                current_z = float(position[2])
                while len(route) > 1:
                    dx = float(route[0]["x"]) - current_x
                    dz = float(route[0]["z"]) - current_z
                    if dx * dx + dz * dz > handoff_distance * handoff_distance:
                        break
                    route.pop(0)
                    skipped_leading_points += 1

        segments: list[dict[str, Any]] = []
        handoff_speeds: list[float] = []
        current = self.adapter.go_to_xyz(**route[0], speed_mps=speed_mps)
        segments.append(dict(current))
        if current.get("status") != "accepted" or not isinstance(current.get("op_id"), str):
            result = dict(current)
            result["segments"] = segments
            return result

        for index in range(len(route) - 1):
            operation_id = str(current["op_id"])
            target = route[index]
            state = self.session.wait_for_npc_near(
                float(target["x"]),
                float(target["z"]),
                handoff_distance,
                handoff_timeout_s,
                operation_id=operation_id,
            )
            if state is None:
                operation = self.session.wait_for_operation(operation_id, 0.0)
                return {
                    "status": "unknown" if operation is None else str(operation["status"]),
                    "error": "handoff_timeout" if operation is None else "handoff_missed",
                    "detail": "未在当前段结束前取得可靠的预切位置，未继续发送路线",
                    "midi_sent": True,
                    "snapshot_required": operation is None,
                    "segments": segments,
                    "operation": operation,
                    "handoff_speeds_mps": handoff_speeds,
                }
            speed = float(state.get("speed", 0.0))
            handoff_speeds.append(speed)
            current = self.adapter.go_to_xyz(**route[index + 1], speed_mps=speed_mps)
            segments.append(dict(current))
            if current.get("status") != "accepted" or not isinstance(current.get("op_id"), str):
                result = dict(current)
                result.update({"segments": segments, "handoff_speeds_mps": handoff_speeds})
                return result
            replaced = self.session.wait_for_operation(operation_id, 2.0)
            if replaced is None or replaced.get("status") != "cancelled" or replaced.get("reason") != "replaced":
                return {
                    "status": "unknown" if replaced is None else "failed",
                    "error": "replacement_unconfirmed",
                    "detail": "中间 GOTO 未确认 cancelled/replaced，路线生命周期不完整",
                    "midi_sent": True,
                    "snapshot_required": replaced is None,
                    "segments": segments,
                    "operation": replaced,
                    "handoff_speeds_mps": handoff_speeds,
                }

        final_id = str(current["op_id"])
        final_operation = self.session.wait_for_operation(final_id, final_timeout_s)
        if final_operation is None:
            return {
                "status": "unknown",
                "error": "operation_timeout",
                "detail": "最终导航操作超时；必须先快照取证，不能视为成功",
                "midi_sent": True,
                "snapshot_required": True,
                "segments": segments,
                "handoff_speeds_mps": handoff_speeds,
                "final_op_id": final_id,
            }
        status = str(final_operation.get("status", "unknown"))
        if status not in _VALID_OPERATION_STATUSES:
            status = "unknown"
        seamless = all(speed >= minimum_speed for speed in handoff_speeds)
        if status == "succeeded" and not seamless:
            status = "failed"
        return {
            "status": status,
            "error": None if status == "succeeded" else (
                "visual_continuity_lost" if final_operation.get("status") == "succeeded" else final_operation.get("reason")
            ),
            "midi_sent": True,
            "segments": segments,
            "handoff_speeds_mps": handoff_speeds,
            "minimum_handoff_speed_mps": min(handoff_speeds) if handoff_speeds else None,
            "seamless": seamless,
            "skipped_leading_points": skipped_leading_points,
            "final_operation": final_operation,
        }


__all__ = ["YuiContinuousRouteRunner"]
