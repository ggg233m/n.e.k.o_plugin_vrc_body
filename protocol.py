"""AnyaDance UDP protocol v1 encoding and safety validation."""

from __future__ import annotations

import json
from typing import Any, Dict

from .config import SafetyConfig
from .model import CONTROLLER_IDS, DEVICE_IDS, FrameState, all_finite, quat_norm_sq

PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 8192


def validate_frame(frame: FrameState, safety: SafetyConfig) -> None:
    if tuple(frame.devices.keys()) != DEVICE_IDS and set(frame.devices) != set(DEVICE_IDS):
        raise ValueError("frame must contain exactly the six AnyaDance devices")
    if set(frame.controllers) != set(CONTROLLER_IDS):
        raise ValueError("frame must contain both controller input states")

    for name in DEVICE_IDS:
        device = frame.devices[name]
        if not all_finite(device.position) or not all_finite(device.rotation):
            raise ValueError(f"{name} pose contains NaN or Infinity")
        if any(abs(value) > safety.max_position_abs_m for value in device.position):
            raise ValueError(f"{name} position exceeds plugin safety bounds")
        if device.position[1] > safety.max_y_m:
            raise ValueError(f"{name} Y position exceeds plugin safety bounds")
        norm_sq = quat_norm_sq(device.rotation)
        if not 0.5 <= norm_sq <= 1.5:
            raise ValueError(f"{name} quaternion length is invalid")

    for name in CONTROLLER_IDS:
        controller = frame.controllers[name]
        scalar_values = (
            controller.trigger_value,
            controller.grip_value,
            controller.joystick_x,
            controller.joystick_y,
            controller.trackpad_x,
            controller.trackpad_y,
            *controller.finger_bends.values(),
        )
        if not all_finite(scalar_values):
            raise ValueError(f"{name} inputs contain NaN or Infinity")
        if not 0.0 <= controller.trigger_value <= 1.0 or not 0.0 <= controller.grip_value <= 1.0:
            raise ValueError(f"{name} trigger/grip value is outside [0, 1]")
        if any(not -1.0 <= value <= 1.0 for value in (
            controller.joystick_x,
            controller.joystick_y,
            controller.trackpad_x,
            controller.trackpad_y,
        )):
            raise ValueError(f"{name} axis is outside [-1, 1]")
        if set(controller.finger_bends) != {"thumb", "index", "middle", "ring", "pinky"}:
            raise ValueError(f"{name} finger_bends must contain all five fingers")
        if any(not 0.0 <= value <= 1.0 for value in controller.finger_bends.values()):
            raise ValueError(f"{name} finger bend is outside [0, 1]")


def frame_payload(frame: FrameState) -> Dict[str, Any]:
    devices: Dict[str, Any] = {}
    for name in DEVICE_IDS:
        device = frame.devices[name]
        devices[name] = {
            "valid": bool(device.valid),
            "connected": bool(device.connected),
            "pose": {
                "position": list(device.position),
                "rotation_xyzw": list(device.rotation),
            },
        }

    inputs: Dict[str, Any] = {}
    for name in CONTROLLER_IDS:
        controller = frame.controllers[name]
        inputs[name] = {
            "trigger_click": controller.trigger_click,
            "trigger_value": controller.trigger_value,
            "menu_click": controller.menu_click,
            "system_click": controller.system_click,
            "a_click": controller.a_click,
            "b_click": controller.b_click,
            "grip_click": controller.grip_click,
            "grip_value": controller.grip_value,
            "joystick_x": controller.joystick_x,
            "joystick_y": controller.joystick_y,
            "trackpad_x": controller.trackpad_x,
            "trackpad_y": controller.trackpad_y,
            "finger_bends": dict(controller.finger_bends),
        }
    return {"version": PROTOCOL_VERSION, "devices": devices, "inputs": inputs}


def encode_frame(frame: FrameState, safety: SafetyConfig) -> bytes:
    validate_frame(frame, safety)
    encoded = json.dumps(frame_payload(frame), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) >= MAX_PACKET_BYTES:
        raise ValueError(f"serialized UDP packet is {len(encoded)} bytes; limit is < {MAX_PACKET_BYTES}")
    return encoded

