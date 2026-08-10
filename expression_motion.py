"""Low-priority semantic expression overlays for body animation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import BodyProfile
from .model import CONTROLLER_IDS, IDENTITY_QUAT, FrameState, Quat, normalized_quat
from .motion import (
    apply_hand_pose,
    axis_angle,
    interpolate_frame,
    move_hand_target,
    quat_multiply,
    quat_slerp,
)


EXPRESSION_GESTURES = frozenset({
    "nod", "deny", "wave", "offer", "explain", "celebrate", "tilt",
    "shrug", "think", "point", "beckon", "clap", "surprise", "comfort",
    "apologize", "sigh", "laugh",
})


@dataclass
class ExpressionOverlay:
    action_id: str
    gesture: str
    side: str
    energy: float
    started_at: float
    duration_s: float
    reference: FrameState
    source: str = "llm_intent"
    intent: str | None = None
    cancelled_at: float | None = None
    cancel_fade_s: float = 0.25

    def progress(self, now: float) -> float:
        return min(1.0, max(0.0, (now - self.started_at) / max(self.duration_s, 1e-6)))

    def cancel_weight(self, now: float) -> float:
        if self.cancelled_at is None:
            return 1.0
        return max(0.0, 1.0 - (now - self.cancelled_at) / max(self.cancel_fade_s, 1e-6))

    def expired(self, now: float) -> bool:
        return self.progress(now) >= 1.0 or self.cancel_weight(now) <= 0.0


def _quat_inverse(quat: Quat) -> Quat:
    x, y, z, w = normalized_quat(quat)
    return (-x, -y, -z, w)


def _phase_frames(reference: FrameState, gesture: str, side: str, energy: float, profile: BodyProfile) -> tuple[FrameState, FrameState]:
    """Return anticipation and stroke frames relative to the pose at gesture start."""
    anticipation = reference.clone()
    stroke = reference.clone()
    selected = ("left", "right") if side == "both" else (side,)

    if gesture == "nod":
        anticipation.devices["hmd"].rotation = quat_multiply(axis_angle((1.0, 0.0, 0.0), -4.0 * energy), reference.devices["hmd"].rotation)
        stroke.devices["hmd"].rotation = quat_multiply(axis_angle((1.0, 0.0, 0.0), 22.0 * energy), reference.devices["hmd"].rotation)
        return anticipation, stroke
    if gesture == "deny":
        anticipation.devices["hmd"].rotation = quat_multiply(axis_angle((0.0, 1.0, 0.0), -8.0 * energy), reference.devices["hmd"].rotation)
        stroke.devices["hmd"].rotation = quat_multiply(axis_angle((0.0, 1.0, 0.0), 20.0 * energy), reference.devices["hmd"].rotation)
        return anticipation, stroke
    if gesture == "tilt":
        anticipation.devices["hmd"].rotation = quat_multiply(axis_angle((0.0, 0.0, 1.0), -2.0 * energy), reference.devices["hmd"].rotation)
        stroke.devices["hmd"].rotation = quat_multiply(axis_angle((0.0, 0.0, 1.0), 15.0 * energy), reference.devices["hmd"].rotation)
        return anticipation, stroke
    if gesture in {"sigh", "laugh"}:
        anticipation = reference.clone()
        stroke = reference.clone()
        pitch = 16.0 * energy if gesture == "sigh" else 10.0 * energy
        stroke.devices["hmd"].rotation = quat_multiply(
            axis_angle((1.0, 0.0, 0.0), pitch), reference.devices["hmd"].rotation
        )
        if gesture == "sigh":
            hmd = stroke.devices["hmd"]
            hmd.position = (hmd.position[0], hmd.position[1] - 0.05 * energy, hmd.position[2])
        return anticipation, stroke

    for current_side in selected:
        name = f"{current_side}_controller"
        hand = anticipation.devices[name]
        sign = -1.0 if current_side == "left" else 1.0
        hand.position = (
            hand.position[0] - sign * 0.025 * energy,
            hand.position[1] - 0.035 * energy,
            hand.position[2] + 0.025 * energy,
        )

    if gesture == "wave":
        stroke = move_hand_target(
            reference,
            side=side if side != "both" else "right",
            relative_to="hmd",
            x_m=0.27 if side != "left" else -0.27,
            y_m=0.08,
            z_m=-0.10,
            palm="forward",
        )
    elif gesture == "offer":
        current_side = selected[0]
        stroke = move_hand_target(
            reference,
            side=current_side,
            relative_to="chest",
            x_m=0.24 if current_side == "right" else -0.24,
            y_m=0.05,
            z_m=-0.36,
            palm="forward",
            wrist_pitch_deg=-18.0 * energy,
        )
    elif gesture == "explain":
        current_side = selected[0]
        stroke = move_hand_target(
            reference,
            side=current_side,
            relative_to="chest",
            x_m=0.34 if current_side == "right" else -0.34,
            y_m=0.10,
            z_m=-0.23,
            palm="inward",
            wrist_roll_deg=(18.0 if current_side == "right" else -18.0) * energy,
        )
    elif gesture == "celebrate":
        stroke = reference.clone()
        for current_side in ("left", "right"):
            sign = -1.0 if current_side == "left" else 1.0
            name = f"{current_side}_controller"
            hmd = reference.devices["hmd"].position
            stroke.devices[name].position = (hmd[0] + sign * 0.34, min(1.96, hmd[1] + 0.34), hmd[2] - 0.12)
    elif gesture == "shrug":
        for current_side in ("left", "right"):
            sign = -1.0 if current_side == "left" else 1.0
            stroke = move_hand_target(
                stroke, side=current_side, relative_to="chest", x_m=sign * 0.34,
                y_m=0.08, z_m=-0.18, palm="forward",
                wrist_roll_deg=sign * 18.0 * energy,
            )
    elif gesture == "think":
        current_side = selected[0]
        sign = -1.0 if current_side == "left" else 1.0
        stroke = move_hand_target(
            reference, side=current_side, relative_to="hmd", x_m=sign * 0.18,
            y_m=-0.11, z_m=-0.09, palm="inward", wrist_pitch_deg=-12.0,
        )
        stroke = apply_hand_pose(stroke, side=current_side, pose="point", strength=0.6)
    elif gesture == "point":
        current_side = selected[0]
        sign = -1.0 if current_side == "left" else 1.0
        stroke = move_hand_target(
            reference, side=current_side, relative_to="chest", x_m=sign * 0.12,
            y_m=0.10, z_m=-0.46, palm="neutral", wrist_pitch_deg=-8.0,
        )
        stroke = apply_hand_pose(stroke, side=current_side, pose="point", strength=0.88)
    elif gesture == "beckon":
        current_side = selected[0]
        sign = -1.0 if current_side == "left" else 1.0
        stroke = move_hand_target(
            reference, side=current_side, relative_to="hmd", x_m=sign * 0.25,
            y_m=-0.06, z_m=-0.24, palm="forward", wrist_pitch_deg=-10.0,
        )
    elif gesture == "clap":
        for current_side in ("left", "right"):
            sign = -1.0 if current_side == "left" else 1.0
            stroke = move_hand_target(
                stroke, side=current_side, relative_to="chest", x_m=sign * 0.07,
                y_m=0.12, z_m=-0.27, palm="inward",
                wrist_roll_deg=sign * 20.0,
            )
    elif gesture == "surprise":
        for current_side in ("left", "right"):
            sign = -1.0 if current_side == "left" else 1.0
            stroke = move_hand_target(
                stroke, side=current_side, relative_to="hmd", x_m=sign * 0.24,
                y_m=-0.10, z_m=-0.11, palm="forward",
                wrist_roll_deg=sign * 12.0,
            )
    elif gesture == "comfort":
        current_side = selected[0]
        sign = -1.0 if current_side == "left" else 1.0
        stroke = move_hand_target(
            reference, side=current_side, relative_to="chest", x_m=sign * 0.10,
            y_m=0.07, z_m=-0.05, palm="inward", wrist_pitch_deg=-18.0,
        )
    elif gesture == "apologize":
        stroke = move_hand_target(
            reference, side="right", relative_to="chest", x_m=0.10,
            y_m=0.06, z_m=-0.05, palm="inward", wrist_pitch_deg=-18.0,
        )
        hmd = stroke.devices["hmd"]
        hmd.position = (hmd.position[0], hmd.position[1] - 0.08 * energy, hmd.position[2] - 0.03 * energy)
        hmd.rotation = quat_multiply(
            axis_angle((1.0, 0.0, 0.0), 22.0 * energy), hmd.rotation
        )
    return anticipation, stroke


def sample_expression(overlay: ExpressionOverlay, now: float, profile: BodyProfile) -> tuple[FrameState, tuple[str, ...], float]:
    p = overlay.progress(now)
    anticipation, stroke = _phase_frames(overlay.reference, overlay.gesture, overlay.side, overlay.energy, profile)
    if p < 0.14:
        sampled = interpolate_frame(overlay.reference, anticipation, p / 0.14)
    elif p < 0.48:
        sampled = interpolate_frame(anticipation, stroke, (p - 0.14) / 0.34)
    elif p < 0.70:
        sampled = stroke.clone()
        if overlay.gesture == "wave":
            swing = math.sin((p - 0.48) / 0.22 * math.pi * 2.0) * 0.045 * overlay.energy
            for current_side in (("left", "right") if overlay.side == "both" else (overlay.side,)):
                device = sampled.devices[f"{current_side}_controller"]
                device.position = (device.position[0] + swing, device.position[1], device.position[2])
        elif overlay.gesture == "deny":
            yaw = math.sin((p - 0.48) / 0.22 * math.pi * 2.0) * 14.0 * overlay.energy
            sampled.devices["hmd"].rotation = quat_multiply(axis_angle((0.0, 1.0, 0.0), yaw), stroke.devices["hmd"].rotation)
        elif overlay.gesture == "beckon":
            for current_side in (("left", "right") if overlay.side == "both" else (overlay.side,)):
                bend = (0.25 + 0.55 * (math.sin((p - 0.48) / 0.22 * math.pi * 2.0) ** 2)) * overlay.energy
                controller = sampled.controllers[f"{current_side}_controller"]
                for finger in ("index", "middle", "ring", "pinky"):
                    controller.finger_bends[finger] = bend
        elif overlay.gesture == "clap":
            center_x = sampled.devices["hmd"].position[0]
            separation = 0.025 + 0.05 * abs(math.sin((p - 0.48) / 0.22 * math.pi * 3.0))
            left = sampled.devices["left_controller"]
            right = sampled.devices["right_controller"]
            left.position = (center_x - separation, left.position[1], left.position[2])
            right.position = (center_x + separation, right.position[1], right.position[2])
        elif overlay.gesture == "laugh":
            pitch = math.sin((p - 0.48) / 0.22 * math.pi * 3.0) * 7.0 * overlay.energy
            sampled.devices["hmd"].rotation = quat_multiply(
                axis_angle((1.0, 0.0, 0.0), pitch), stroke.devices["hmd"].rotation
            )
    else:
        sampled = interpolate_frame(stroke, overlay.reference, (p - 0.70) / 0.30)
    if overlay.gesture in {"nod", "deny", "tilt", "sigh", "laugh"}:
        channels = ("hmd",)
    elif overlay.gesture == "apologize":
        channels = ("hmd", "right_controller")
    elif overlay.side == "both" or overlay.gesture == "celebrate":
        channels = ("left_controller", "right_controller")
    else:
        channels = (f"{overlay.side}_controller",)
    return sampled, channels, overlay.cancel_weight(now)


def apply_expression_overlay(base: FrameState, reference: FrameState, sampled: FrameState, channels: tuple[str, ...], weight: float) -> FrameState:
    """Apply sampled motion as a delta so the current base pose remains authoritative."""
    result = base.clone()
    weight = min(1.0, max(0.0, weight))
    for name in channels:
        source = reference.devices[name]
        gesture = sampled.devices[name]
        target = result.devices[name]
        delta_position = tuple(gesture.position[index] - source.position[index] for index in range(3))
        target.position = tuple(target.position[index] + delta_position[index] * weight for index in range(3))
        delta_rotation = quat_multiply(gesture.rotation, _quat_inverse(source.rotation))
        weighted_delta = quat_slerp(IDENTITY_QUAT, delta_rotation, weight)
        target.rotation = quat_multiply(weighted_delta, target.rotation)
        if name in CONTROLLER_IDS:
            # Expression overlays may shape fingers but never synthesize controller clicks.
            source_controller = reference.controllers[name]
            gesture_controller = sampled.controllers[name]
            target_controller = result.controllers[name]
            for finger in target_controller.finger_bends:
                delta = gesture_controller.finger_bends[finger] - source_controller.finger_bends[finger]
                target_controller.finger_bends[finger] = min(1.0, max(0.0, target_controller.finger_bends[finger] + delta * weight))
    return result
