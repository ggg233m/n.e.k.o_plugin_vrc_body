"""Geometry, interpolation, and procedural body motions."""

from __future__ import annotations

import math
from typing import Iterable

from .config import BodyProfile
from .model import (
    CONTROLLER_IDS,
    IDENTITY_QUAT,
    LEFT_CANONICAL_QUAT,
    RIGHT_CANONICAL_QUAT,
    ControllerState,
    FrameState,
    Quat,
    Vec3,
    normalized_quat,
)


GESTURE_DURATIONS = {
    "wave": 1.4,
    "nod": 1.0,
    "bow": 1.5,
    "shake_head": 1.15,
    "shrug": 1.45,
    "think": 1.55,
    "point": 1.25,
    "beckon": 1.55,
    "clap": 1.65,
    "surprise": 1.25,
    "comfort": 1.4,
    "sigh": 1.45,
}
GESTURE_NAMES = frozenset(GESTURE_DURATIONS)


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def smoothstep(value: float) -> float:
    value = clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def vec_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def quat_multiply(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return normalized_quat((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ))


def axis_angle(axis: Vec3, degrees: float) -> Quat:
    length = math.sqrt(sum(value * value for value in axis))
    if length < 1e-12:
        return IDENTITY_QUAT
    half = math.radians(degrees) * 0.5
    scale = math.sin(half) / length
    return normalized_quat((axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half)))


def quat_between_vectors(source: Vec3, target: Vec3) -> Quat:
    """Return the shortest rotation that maps ``source`` onto ``target``."""
    source_length = math.sqrt(sum(value * value for value in source))
    target_length = math.sqrt(sum(value * value for value in target))
    if source_length < 1e-12 or target_length < 1e-12:
        return IDENTITY_QUAT
    a = tuple(value / source_length for value in source)
    b = tuple(value / target_length for value in target)
    dot = min(1.0, max(-1.0, sum(left * right for left, right in zip(a, b))))
    if dot < -0.999999:
        candidate = (0.0, 1.0, 0.0) if abs(a[1]) < 0.9 else (0.0, 0.0, 1.0)
        axis = (
            a[1] * candidate[2] - a[2] * candidate[1],
            a[2] * candidate[0] - a[0] * candidate[2],
            a[0] * candidate[1] - a[1] * candidate[0],
        )
        return axis_angle(axis, 180.0)
    cross = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    return normalized_quat((cross[0], cross[1], cross[2], 1.0 + dot))


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    a = normalized_quat(a)
    b = normalized_quat(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalized_quat(tuple(lerp(x, y, t) for x, y in zip(a, b)))  # type: ignore[arg-type]
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    left = math.sin((1.0 - t) * angle) / sin_angle
    right = math.sin(t * angle) / sin_angle
    return normalized_quat(tuple(left * x + right * y for x, y in zip(a, b)))  # type: ignore[arg-type]


def interpolate_frame(start: FrameState, target: FrameState, progress: float) -> FrameState:
    t = smoothstep(progress)
    result = start.clone()
    for name, device in result.devices.items():
        device.position = vec_lerp(start.devices[name].position, target.devices[name].position, t)
        device.rotation = quat_slerp(start.devices[name].rotation, target.devices[name].rotation, t)
        device.valid = target.devices[name].valid if progress >= 1.0 else start.devices[name].valid
        device.connected = target.devices[name].connected if progress >= 1.0 else start.devices[name].connected
    for name in CONTROLLER_IDS:
        source = start.controllers[name]
        destination = target.controllers[name]
        controller = result.controllers[name]
        for attr in ("trigger_value", "grip_value", "joystick_x", "joystick_y", "trackpad_x", "trackpad_y"):
            setattr(controller, attr, lerp(getattr(source, attr), getattr(destination, attr), t))
        for finger in controller.finger_bends:
            controller.finger_bends[finger] = lerp(source.finger_bends[finger], destination.finger_bends[finger], t)
        for attr in ("trigger_click", "menu_click", "system_click", "a_click", "b_click", "grip_click"):
            setattr(controller, attr, getattr(destination if progress >= 0.95 else source, attr))
    return result


def _side_names(side: str) -> Iterable[str]:
    if side == "both":
        return ("left", "right")
    return (side,)


def palm_rotation(side: str, palm: str) -> Quat:
    canonical = LEFT_CANONICAL_QUAT if side == "left" else RIGHT_CANONICAL_QUAT
    if palm in ("neutral", "forward"):
        return canonical
    if palm == "down":
        return quat_multiply(axis_angle((1.0, 0.0, 0.0), -90.0), canonical)
    if palm == "inward":
        yaw = -70.0 if side == "left" else 70.0
        return quat_multiply(axis_angle((0.0, 1.0, 0.0), yaw), canonical)
    raise ValueError(f"unknown palm orientation: {palm}")


def euler_rotation(*, pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> Quat:
    """Build a body-local yaw/pitch/roll rotation using the AnyaDance axes."""
    yaw = axis_angle((0.0, 1.0, 0.0), yaw_deg)
    pitch = axis_angle((1.0, 0.0, 0.0), pitch_deg)
    roll = axis_angle((0.0, 0.0, 1.0), roll_deg)
    return quat_multiply(yaw, quat_multiply(pitch, roll))


def wrist_rotation(
    side: str,
    palm: str,
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> Quat:
    return quat_multiply(
        euler_rotation(pitch_deg=pitch_deg, yaw_deg=yaw_deg, roll_deg=roll_deg),
        palm_rotation(side, palm),
    )


def directed_wrist_rotation(
    side: str,
    direction: Vec3,
    palm: str,
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> Quat:
    """Follow the arm direction, then apply palm and wrist-local offsets."""
    reference = (-1.0, 0.0, 0.0) if side == "left" else (1.0, 0.0, 0.0)
    return quat_multiply(
        quat_between_vectors(reference, direction),
        wrist_rotation(
            side,
            palm,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
        ),
    )


def arm_pose_target(
    frame: FrameState,
    *,
    side: str,
    elevation_deg: float,
    plane: str | None = None,
    azimuth_deg: float | None = None,
    reach: float,
    palm: str,
    wrist_pitch_deg: float = 0.0,
    wrist_yaw_deg: float = 0.0,
    wrist_roll_deg: float = 0.0,
    profile: BodyProfile,
) -> FrameState:
    result = frame.clone()
    hmd = frame.devices["hmd"].position
    theta = math.radians(elevation_deg)
    vertical = -math.cos(theta)
    horizontal = math.sin(theta)
    distance = profile.arm_length_m * reach
    for current_side in _side_names(side):
        sign = -1.0 if current_side == "left" else 1.0
        shoulder = (hmd[0] + sign * profile.shoulder_width_m * 0.5, hmd[1] - profile.shoulder_drop_m, hmd[2])
        if azimuth_deg is None:
            if plane in (None, "front"):
                current_azimuth = 0.0
            elif plane == "side":
                current_azimuth = sign * 90.0
            else:
                raise ValueError(f"unknown arm plane: {plane}")
        else:
            current_azimuth = azimuth_deg
        azimuth = math.radians(current_azimuth)
        direction = (
            math.sin(azimuth) * horizontal,
            vertical,
            -math.cos(azimuth) * horizontal,
        )
        controller_name = f"{current_side}_controller"
        result.devices[controller_name].position = (
            shoulder[0] + direction[0] * distance,
            shoulder[1] + direction[1] * distance,
            shoulder[2] + direction[2] * distance,
        )
        result.devices[controller_name].rotation = directed_wrist_rotation(
            current_side,
            direction,
            palm,
            pitch_deg=wrist_pitch_deg,
            yaw_deg=wrist_yaw_deg,
            roll_deg=wrist_roll_deg,
        )
    return result


def move_hand_target(
    frame: FrameState,
    *,
    side: str,
    relative_to: str,
    x_m: float,
    y_m: float,
    z_m: float,
    palm: str,
    wrist_pitch_deg: float = 0.0,
    wrist_yaw_deg: float = 0.0,
    wrist_roll_deg: float = 0.0,
    profile: BodyProfile | None = None,
) -> FrameState:
    """Move a controller to a position relative to a stable body anchor."""
    result = frame.clone()
    hmd = frame.devices["hmd"].position
    hip = frame.devices["hip"].position
    anchors = {
        "hmd": hmd,
        "chest": (hmd[0], hmd[1] - 0.35, hmd[2]),
        "hip": hip,
    }
    anchor = anchors[relative_to]
    controller_name = f"{side}_controller"
    target = (anchor[0] + x_m, anchor[1] + y_m, anchor[2] + z_m)
    result.devices[controller_name].position = target
    body_profile = profile or BodyProfile()
    sign = -1.0 if side == "left" else 1.0
    shoulder = (
        hmd[0] + sign * body_profile.shoulder_width_m * 0.5,
        hmd[1] - body_profile.shoulder_drop_m,
        hmd[2],
    )
    direction: Vec3 = tuple(target[index] - shoulder[index] for index in range(3))  # type: ignore[assignment]
    result.devices[controller_name].rotation = directed_wrist_rotation(
        side,
        direction,
        palm,
        pitch_deg=wrist_pitch_deg,
        yaw_deg=wrist_yaw_deg,
        roll_deg=wrist_roll_deg,
    )
    return result


def apply_hand_pose(frame: FrameState, *, side: str, pose: str, strength: float) -> FrameState:
    result = frame.clone()
    for current_side in _side_names(side):
        controller = result.controllers[f"{current_side}_controller"]
        if pose == "open":
            bends = {finger: 0.0 for finger in controller.finger_bends}
        elif pose in ("fist", "grip"):
            bends = {finger: strength for finger in controller.finger_bends}
        elif pose == "point":
            bends = {
                "thumb": strength * 0.35,
                "index": 0.0,
                "middle": strength,
                "ring": strength,
                "pinky": strength,
            }
        else:
            raise ValueError(f"unknown hand pose: {pose}")
        controller.finger_bends = bends
        controller.grip_click = pose == "grip" and strength >= 0.5
        controller.grip_value = strength if pose == "grip" else 0.0
    return result


def reach_target(
    frame: FrameState,
    *,
    side: str,
    height: str,
    direction: str,
    distance_m: float,
    profile: BodyProfile,
) -> FrameState:
    result = frame.clone()
    hmd = frame.devices["hmd"].position
    sign = -1.0 if side == "left" else 1.0
    shoulder_x = hmd[0] + sign * profile.shoulder_width_m * 0.5
    y_offsets = {"waist": -0.65, "chest": -0.35, "head": -0.05}
    lateral = {"forward": 0.0, "inward": -sign * 0.12, "outward": sign * 0.12}[direction]
    controller_name = f"{side}_controller"
    result.devices[controller_name].position = (
        shoulder_x + lateral,
        hmd[1] + y_offsets[height],
        hmd[2] - distance_m,
    )
    target = result.devices[controller_name].position
    shoulder = (shoulder_x, hmd[1] - profile.shoulder_drop_m, hmd[2])
    direction_to_target: Vec3 = tuple(
        target[index] - shoulder[index] for index in range(3)
    )  # type: ignore[assignment]
    result.devices[controller_name].rotation = directed_wrist_rotation(
        side,
        direction_to_target,
        "neutral",
    )
    return apply_hand_pose(result, side=side, pose="open", strength=0.0)


def gesture_frame(
    start: FrameState,
    *,
    name: str,
    side: str,
    intensity: float,
    progress: float,
    profile: BodyProfile,
) -> FrameState:
    p = clamp01(progress)

    def held(target: FrameState, *, attack: float = 0.25, release: float = 0.75) -> FrameState:
        if p < attack:
            return interpolate_frame(start, target, p / attack)
        if p > release:
            return interpolate_frame(target, start, (p - release) / (1.0 - release))
        return target

    if name == "wave":
        raised = arm_pose_target(
            start,
            side=side,
            elevation_deg=145.0,
            plane="front",
            reach=0.9,
            palm="forward",
            profile=profile,
        )
        if p < 0.2:
            return interpolate_frame(start, raised, p / 0.2)
        if p > 0.8:
            return interpolate_frame(raised, start, (p - 0.8) / 0.2)
        result = raised.clone()
        oscillation = math.sin(((p - 0.2) / 0.6) * math.pi * 4.0) * 0.09 * intensity
        for current_side in _side_names(side):
            controller = result.devices[f"{current_side}_controller"]
            controller.position = (controller.position[0] + oscillation, controller.position[1], controller.position[2])
        return result

    if name == "nod":
        result = start.clone()
        pitch = 24.0 * intensity * (math.sin(p * math.pi * 2.0) ** 2)
        result.devices["hmd"].rotation = quat_multiply(axis_angle((1.0, 0.0, 0.0), pitch), start.devices["hmd"].rotation)
        return result

    if name == "bow":
        target = start.clone()
        hmd = target.devices["hmd"]
        hmd.position = (hmd.position[0], hmd.position[1] - 0.16 * intensity, hmd.position[2] - 0.10 * intensity)
        hmd.rotation = quat_multiply(axis_angle((1.0, 0.0, 0.0), 35.0 * intensity), hmd.rotation)
        target.devices["hip"].rotation = quat_multiply(axis_angle((1.0, 0.0, 0.0), 22.0 * intensity), target.devices["hip"].rotation)
        if p < 0.35:
            return interpolate_frame(start, target, p / 0.35)
        if p > 0.65:
            return interpolate_frame(target, start, (p - 0.65) / 0.35)
        return target
    if name == "shake_head":
        result = start.clone()
        envelope = math.sin(p * math.pi)
        yaw = math.sin(p * math.pi * 4.0) * 20.0 * intensity * envelope
        result.devices["hmd"].rotation = quat_multiply(
            axis_angle((0.0, 1.0, 0.0), yaw),
            start.devices["hmd"].rotation,
        )
        return result
    if name == "shrug":
        target = start.clone()
        target = move_hand_target(
            target, side="left", relative_to="chest", x_m=-0.34, y_m=0.08,
            z_m=-0.18, palm="forward", wrist_roll_deg=-18.0 * intensity,
        )
        target = move_hand_target(
            target, side="right", relative_to="chest", x_m=0.34, y_m=0.08,
            z_m=-0.18, palm="forward", wrist_roll_deg=18.0 * intensity,
        )
        return held(target)
    if name == "think":
        current_side = "right" if side == "both" else side
        sign = -1.0 if current_side == "left" else 1.0
        target = move_hand_target(
            start, side=current_side, relative_to="hmd", x_m=sign * 0.18,
            y_m=-0.11, z_m=-0.09, palm="inward", wrist_pitch_deg=-12.0,
        )
        target = apply_hand_pose(target, side=current_side, pose="point", strength=0.65)
        return held(target, attack=0.3, release=0.72)
    if name == "point":
        current_side = "right" if side == "both" else side
        sign = -1.0 if current_side == "left" else 1.0
        target = move_hand_target(
            start, side=current_side, relative_to="chest", x_m=sign * 0.12,
            y_m=0.10, z_m=-0.48, palm="neutral", wrist_pitch_deg=-8.0,
        )
        target = apply_hand_pose(target, side=current_side, pose="point", strength=0.9)
        return held(target, attack=0.22, release=0.74)
    if name == "beckon":
        current_side = "right" if side == "both" else side
        sign = -1.0 if current_side == "left" else 1.0
        target = move_hand_target(
            start, side=current_side, relative_to="hmd", x_m=sign * 0.25,
            y_m=-0.06, z_m=-0.24, palm="forward", wrist_pitch_deg=-10.0,
        )
        result = held(target, attack=0.22, release=0.8)
        if 0.22 <= p <= 0.8:
            bend = (0.25 + 0.55 * (math.sin((p - 0.22) / 0.58 * math.pi * 4.0) ** 2)) * intensity
            controller = result.controllers[f"{current_side}_controller"]
            for finger in ("index", "middle", "ring", "pinky"):
                controller.finger_bends[finger] = bend
        return result
    if name == "clap":
        target = start.clone()
        target = move_hand_target(
            target, side="left", relative_to="chest", x_m=-0.07, y_m=0.12,
            z_m=-0.27, palm="inward", wrist_roll_deg=-20.0,
        )
        target = move_hand_target(
            target, side="right", relative_to="chest", x_m=0.07, y_m=0.12,
            z_m=-0.27, palm="inward", wrist_roll_deg=20.0,
        )
        result = held(target, attack=0.2, release=0.82)
        if 0.2 <= p <= 0.82:
            separation = 0.025 + 0.055 * abs(math.sin((p - 0.2) / 0.62 * math.pi * 5.0))
            chest_x = start.devices["hmd"].position[0]
            left = result.devices["left_controller"]
            right = result.devices["right_controller"]
            left.position = (chest_x - separation, left.position[1], left.position[2])
            right.position = (chest_x + separation, right.position[1], right.position[2])
        return result
    if name == "surprise":
        target = start.clone()
        target = move_hand_target(
            target, side="left", relative_to="hmd", x_m=-0.24, y_m=-0.10,
            z_m=-0.11, palm="forward", wrist_roll_deg=-12.0,
        )
        target = move_hand_target(
            target, side="right", relative_to="hmd", x_m=0.24, y_m=-0.10,
            z_m=-0.11, palm="forward", wrist_roll_deg=12.0,
        )
        return held(target, attack=0.16, release=0.72)
    if name == "comfort":
        current_side = "right" if side == "both" else side
        sign = -1.0 if current_side == "left" else 1.0
        target = move_hand_target(
            start, side=current_side, relative_to="chest", x_m=sign * 0.10,
            y_m=0.07, z_m=-0.05, palm="inward", wrist_pitch_deg=-18.0,
        )
        return held(target, attack=0.28, release=0.74)
    if name == "sigh":
        target = start.clone()
        hmd = target.devices["hmd"]
        hmd.position = (hmd.position[0], hmd.position[1] - 0.06 * intensity, hmd.position[2])
        hmd.rotation = quat_multiply(
            axis_angle((1.0, 0.0, 0.0), 18.0 * intensity),
            hmd.rotation,
        )
        return held(target, attack=0.32, release=0.7)
    raise ValueError(f"unknown gesture: {name}")
