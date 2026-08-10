"""Device, controller, and frame models shared by the scheduler and protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from typing import Dict, Iterable, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

DEVICE_IDS = (
    "hmd",
    "left_controller",
    "right_controller",
    "hip",
    "left_foot",
    "right_foot",
)
CONTROLLER_IDS = ("left_controller", "right_controller")
IDENTITY_QUAT: Quat = (0.0, 0.0, 0.0, 1.0)
LEFT_CANONICAL_QUAT: Quat = (0.0, 0.0, -0.7071067811865475, 0.7071067811865475)
RIGHT_CANONICAL_QUAT: Quat = (0.0, 0.0, 0.7071067811865475, 0.7071067811865475)


@dataclass
class DeviceState:
    position: Vec3
    rotation: Quat = IDENTITY_QUAT
    valid: bool = True
    connected: bool = True


@dataclass
class ControllerState:
    trigger_click: bool = False
    trigger_value: float = 0.0
    menu_click: bool = False
    system_click: bool = False
    a_click: bool = False
    b_click: bool = False
    grip_click: bool = False
    grip_value: float = 0.0
    joystick_x: float = 0.0
    joystick_y: float = 0.0
    trackpad_x: float = 0.0
    trackpad_y: float = 0.0
    finger_bends: Dict[str, float] = field(default_factory=lambda: {
        "thumb": 0.0,
        "index": 0.0,
        "middle": 0.0,
        "ring": 0.0,
        "pinky": 0.0,
    })


@dataclass
class FrameState:
    devices: Dict[str, DeviceState]
    controllers: Dict[str, ControllerState]

    def clone(self) -> "FrameState":
        return copy.deepcopy(self)


def neutral_frame() -> FrameState:
    """Return AnyaDance's canonical reset T-pose from the current C++ implementation."""
    return FrameState(
        devices={
            "hmd": DeviceState((0.0, 1.50, 0.0), IDENTITY_QUAT),
            "left_controller": DeviceState((-0.62, 1.33, -0.10), LEFT_CANONICAL_QUAT),
            "right_controller": DeviceState((0.62, 1.33, -0.10), RIGHT_CANONICAL_QUAT),
            "hip": DeviceState((0.0, 1.07, -0.05), IDENTITY_QUAT),
            "left_foot": DeviceState((-0.09, 0.26, 0.10), IDENTITY_QUAT),
            "right_foot": DeviceState((0.09, 0.26, 0.10), IDENTITY_QUAT),
        },
        controllers={name: ControllerState() for name in CONTROLLER_IDS},
    )


def neutralize_inputs(frame: FrameState) -> None:
    frame.controllers = {name: ControllerState() for name in CONTROLLER_IDS}


def quat_norm_sq(quat: Quat) -> float:
    return sum(component * component for component in quat)


def normalized_quat(quat: Quat) -> Quat:
    length_sq = quat_norm_sq(quat)
    if not math.isfinite(length_sq) or length_sq < 1e-12:
        raise ValueError("quaternion is not normalizable")
    scale = 1.0 / math.sqrt(length_sq)
    return tuple(component * scale for component in quat)  # type: ignore[return-value]


def all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)

