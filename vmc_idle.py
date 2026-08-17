"""Relay N.E.K.O's public VMC OSC stream into AnyaDance idle frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import threading
import time
from typing import Any, Iterable

from .config import BodyProfile, VmcIdleConfig
from .model import (
    IDENTITY_QUAT,
    LEFT_CANONICAL_QUAT,
    RIGHT_CANONICAL_QUAT,
    ControllerState,
    DeviceState,
    FrameState,
    Quat,
    Vec3,
    normalized_quat,
)
from .osc import MAX_OSC_PACKET_BYTES, OscProtocolError, decode_osc_packet


@dataclass(frozen=True)
class Transform:
    position: Vec3
    rotation: Quat


_ROOT = Transform((0.0, 0.0, 0.0), IDENTITY_QUAT)
_REQUIRED_BONES = frozenset({
    "Hips",
    "Spine", "Chest", "UpperChest", "Neck",
    "Head",
    "LeftShoulder", "RightShoulder",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
})
_PARENT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Spine": ("Hips",),
    "Chest": ("Spine", "Hips"),
    "UpperChest": ("Chest", "Spine", "Hips"),
    "Neck": ("UpperChest", "Chest", "Spine", "Hips"),
    "Head": ("Neck", "UpperChest", "Chest", "Spine", "Hips"),
    "LeftShoulder": ("UpperChest", "Chest", "Spine", "Hips"),
    "LeftUpperArm": ("LeftShoulder", "UpperChest", "Chest", "Spine", "Hips"),
    "LeftLowerArm": ("LeftUpperArm",),
    "LeftHand": ("LeftLowerArm",),
    "RightShoulder": ("UpperChest", "Chest", "Spine", "Hips"),
    "RightUpperArm": ("RightShoulder", "UpperChest", "Chest", "Spine", "Hips"),
    "RightLowerArm": ("RightUpperArm",),
    "RightHand": ("RightLowerArm",),
    "LeftUpperLeg": ("Hips",),
    "LeftLowerLeg": ("LeftUpperLeg",),
    "LeftFoot": ("LeftLowerLeg",),
    "LeftToes": ("LeftFoot",),
    "RightUpperLeg": ("Hips",),
    "RightLowerLeg": ("RightUpperLeg",),
    "RightFoot": ("RightLowerLeg",),
    "RightToes": ("RightFoot",),
}
_DEVICE_BONES = {
    "hmd": "Head",
    "left_controller": "LeftHand",
    "right_controller": "RightHand",
    "hip": "Hips",
    "left_foot": "LeftFoot",
    "right_foot": "RightFoot",
}
_CANONICAL_HEIGHT_M = 1.50
# AnyaDance's six devices represent physical tracking mounts, not the VRM
# humanoid joint origins.  These are the driver's canonical T-pose mounts at
# 1.50 m; calibration maps the current VRM's joint anchors onto them.
_CANONICAL_DEVICE_POSITIONS: dict[str, Vec3] = {
    "hmd": (0.0, 1.50, 0.0),
    "left_controller": (-0.68, 1.33, -0.10),
    "right_controller": (0.68, 1.33, -0.10),
    "hip": (0.0, 1.07, -0.05),
    "left_foot": (-0.09, 0.26, 0.10),
    "right_foot": (0.09, 0.26, 0.10),
}
_FINGER_CHAINS: dict[str, dict[str, tuple[tuple[str, float], ...]]] = {
    "left_controller": {
        "thumb": (("LeftThumbProximal", 5.0), ("LeftThumbIntermediate", 90.0), ("LeftThumbDistal", 90.0)),
        "index": (("LeftIndexProximal", 90.0), ("LeftIndexIntermediate", 80.0), ("LeftIndexDistal", 80.0)),
        "middle": (("LeftMiddleProximal", 90.0), ("LeftMiddleIntermediate", 80.0), ("LeftMiddleDistal", 80.0)),
        "ring": (("LeftRingProximal", 90.0), ("LeftRingIntermediate", 80.0), ("LeftRingDistal", 80.0)),
        "pinky": (("LeftLittleProximal", 90.0), ("LeftLittleIntermediate", 80.0), ("LeftLittleDistal", 80.0)),
    },
    "right_controller": {
        "thumb": (("RightThumbProximal", 5.0), ("RightThumbIntermediate", 90.0), ("RightThumbDistal", 90.0)),
        "index": (("RightIndexProximal", 90.0), ("RightIndexIntermediate", 80.0), ("RightIndexDistal", 80.0)),
        "middle": (("RightMiddleProximal", 90.0), ("RightMiddleIntermediate", 80.0), ("RightMiddleDistal", 80.0)),
        "ring": (("RightRingProximal", 90.0), ("RightRingIntermediate", 80.0), ("RightRingDistal", 80.0)),
        "pinky": (("RightLittleProximal", 90.0), ("RightLittleIntermediate", 80.0), ("RightLittleDistal", 80.0)),
    },
}
_FINGER_BONES = frozenset(
    bone
    for fingers in _FINGER_CHAINS.values()
    for chain in fingers.values()
    for bone, _ in chain
)


def _v_add(left: Vec3, right: Vec3) -> Vec3:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _v_sub(left: Vec3, right: Vec3) -> Vec3:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _v_scale(value: Vec3, scale: float) -> Vec3:
    return value[0] * scale, value[1] * scale, value[2] * scale


def _quat_conjugate(value: Quat) -> Quat:
    return -value[0], -value[1], -value[2], value[3]


def _quat_multiply(left: Quat, right: Quat) -> Quat:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return normalized_quat((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ))


def _quat_rotate(rotation: Quat, value: Vec3) -> Vec3:
    qx, qy, qz, qw = rotation
    vx, vy, vz = value
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _quat_angle(left: Quat, right: Quat) -> float:
    """Shortest angular distance between two normalized quaternions."""
    delta = _quat_multiply(left, _quat_conjugate(right))
    return 2.0 * math.acos(min(1.0, max(0.0, abs(delta[3]))))


def _compose(parent: Transform, local: Transform) -> Transform:
    return Transform(
        _v_add(parent.position, _quat_rotate(parent.rotation, local.position)),
        _quat_multiply(parent.rotation, local.rotation),
    )


def _vmc_transform(arguments: tuple[Any, ...]) -> tuple[str, Transform] | None:
    if len(arguments) < 8 or not isinstance(arguments[0], str):
        return None
    try:
        values = tuple(float(value) for value in arguments[1:8])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    px, py, pz, qx, qy, qz, qw = values
    try:
        # N.E.K.O emits Unity/VMC left-handed transforms. Reflect them back to
        # the right-handed coordinates used by its VRM scene and AnyaDance.
        rotation = normalized_quat((-qx, -qy, qz, qw))
    except ValueError:
        return None
    return arguments[0], Transform((px, py, -pz), rotation)


class VmcIdleRelay:
    """Receive VMC frames, run humanoid FK, and expose the latest six-point pose."""

    def __init__(
        self,
        config: VmcIdleConfig,
        profile: BodyProfile,
        *,
        logger: Any = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.config = config
        self.profile = profile
        self.logger = logger
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._root = _ROOT
        self._pending_bones: dict[str, Transform] = {}
        self._latest_frame: FrameState | None = None
        self._last_packet_at: float | None = None
        self._last_frame_at: float | None = None
        self._source_available = False
        self._receiver_listening = False
        self._received_packets = 0
        self._accepted_frames = 0
        self._rejected_packets = 0
        self._incomplete_frames = 0
        self._last_error: str | None = None
        self._origin: Vec3 | None = None
        self._scale = 1.0
        self._rest_rotations: dict[str, Quat] = {}
        self._position_mount_offsets: dict[str, Vec3] = {}
        self._rest_finger_rotations: dict[str, Quat] = {}
        self._calibration_generation = 0
        self._calibration_reason = "initial_frame"
        self._calibrated_at: float | None = None
        self._calibration_held = False
        self._require_t_pose_frame = False
        self._rejected_calibration_frames = 0

    def _clear_calibration_locked(self) -> None:
        self._root = _ROOT
        self._pending_bones.clear()
        self._latest_frame = None
        self._last_frame_at = None
        self._origin = None
        self._scale = 1.0
        self._rest_rotations.clear()
        self._position_mount_offsets.clear()
        self._rest_finger_rotations.clear()
        self._calibrated_at = None

    def hold_calibration(self, *, reason: str = "waiting_for_rest_pose") -> None:
        """Reject ordinary frames until an authoritative rest pose is ready."""
        with self._lock:
            self._clear_calibration_locked()
            self._calibration_reason = str(reason or "waiting_for_rest_pose")[:64]
            self._calibration_held = True
            self._require_t_pose_frame = False

    def reset_calibration(self, *, reason: str = "manual") -> None:
        """Discard pose-dependent zero points so the next complete frame becomes rest.

        N.E.K.O's VMC stream contains local humanoid transforms, but it does not
        carry the avatar's raw rest pose in every normal frame.  Callers use the
        documented host T-pose request before invoking this method so wrist
        orientation and finger curls are calibrated against the actual VRM rest
        pose instead of whichever animation happened to be playing at startup.
        """
        with self._lock:
            self._clear_calibration_locked()
            self._calibration_generation += 1
            self._calibration_reason = str(reason or "manual")[:64]
            self._calibration_held = False
            self._require_t_pose_frame = self._calibration_reason == "host_t_pose"

    def start(self) -> None:
        if not self.config.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        receiver: socket.socket | None = None
        try:
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            receiver.settimeout(0.2)
            receiver.bind((self.config.listen_host, self.config.listen_port))
        except OSError as exc:
            if receiver is not None:
                try:
                    receiver.close()
                except OSError:
                    pass
            with self._lock:
                self._last_error = f"VMC listen failed: {exc}"
                self._receiver_listening = False
            if self.logger:
                self.logger.warning("N.E.K.O VMC idle relay could not listen: %s", exc)
            return
        self._socket = receiver
        with self._lock:
            self._receiver_listening = True
            self._last_error = None
        self._thread = threading.Thread(target=self._run, name="neko-vmc-idle-relay", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        receiver = self._socket
        self._socket = None
        if receiver is not None:
            try:
                receiver.close()
            except OSError:
                pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        with self._lock:
            self._receiver_listening = False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            receiver = self._socket
            if receiver is None:
                break
            try:
                packet, sender = receiver.recvfrom(MAX_OSC_PACKET_BYTES)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    with self._lock:
                        self._last_error = f"VMC receive failed: {exc}"
                break
            self.ingest_packet(packet, sender=sender, now=self._clock())
        with self._lock:
            self._receiver_listening = False

    def ingest_packet(
        self,
        packet: bytes,
        *,
        sender: tuple[str, int] = ("127.0.0.1", 0),
        now: float | None = None,
    ) -> bool:
        """Decode one UDP packet. Public for deterministic protocol tests."""
        timestamp = self._clock() if now is None else now
        if sender[0] != self.config.allowed_sender:
            with self._lock:
                self._rejected_packets += 1
                self._last_error = f"rejected VMC sender {sender[0]}"
            return False
        try:
            messages = decode_osc_packet(packet)
        except OscProtocolError as exc:
            with self._lock:
                self._rejected_packets += 1
                self._last_error = f"invalid VMC OSC packet: {exc}"
            return False
        self.ingest_messages(messages, now=timestamp)
        with self._lock:
            self._received_packets += 1
            self._last_packet_at = timestamp
        return True

    def ingest_messages(
        self,
        messages: Iterable[tuple[str, tuple[Any, ...]]],
        *,
        now: float | None = None,
    ) -> None:
        timestamp = self._clock() if now is None else now
        with self._lock:
            for address, arguments in messages:
                if address == "/VMC/Ext/OK" and arguments:
                    available = bool(arguments[0])
                    self._source_available = available
                    if not available:
                        self._clear_calibration_locked()
                    continue
                if address == "/VMC/Ext/T":
                    self._finalize_pending_locked(timestamp)
                    self._pending_bones = {}
                    continue
                if address == "/VMC/Ext/Root/Pos":
                    parsed = _vmc_transform(arguments)
                    if parsed is not None:
                        self._root = parsed[1]
                    continue
                if address == "/VMC/Ext/Bone/Pos":
                    parsed = _vmc_transform(arguments)
                    if parsed is not None and len(parsed[0]) <= 64:
                        self._pending_bones[parsed[0]] = parsed[1]

    def _world_transforms_locked(self) -> dict[str, Transform] | None:
        if not _REQUIRED_BONES.issubset(self._pending_bones):
            return None
        result: dict[str, Transform] = {}
        visiting: set[str] = set()

        def resolve(name: str) -> Transform | None:
            if name in result:
                return result[name]
            local = self._pending_bones.get(name)
            if local is None or name in visiting:
                return None
            visiting.add(name)
            parent = self._root
            candidates = _PARENT_CANDIDATES.get(name, ())
            if candidates:
                parent_name = next(
                    (candidate for candidate in candidates if candidate in self._pending_bones),
                    None,
                )
                if parent_name is None:
                    visiting.discard(name)
                    return None
                resolved_parent = resolve(parent_name)
                if resolved_parent is None:
                    visiting.discard(name)
                    return None
                parent = resolved_parent
            visiting.discard(name)
            result[name] = _compose(parent, local)
            return result[name]

        for name in self._pending_bones:
            resolve(name)
        return result

    def _finalize_pending_locked(self, now: float) -> None:
        if not self._pending_bones:
            return
        if self._calibration_held:
            return
        world = self._world_transforms_locked()
        if world is None:
            self._incomplete_frames += 1
            return
        if self._require_t_pose_frame and not self._is_t_pose_candidate_locked(world):
            # The host toggles its T-pose status at a packet boundary, while
            # OSC bones are separate UDP datagrams.  A transition can therefore
            # leave one ordinary or mixed frame at the receiver.  Never let
            # that frame become the long-lived wrist/finger/position baseline.
            self._rejected_calibration_frames += 1
            return
        try:
            frame = self._frame_from_world_locked(world)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            self._incomplete_frames += 1
            self._last_error = f"VMC frame conversion failed: {exc}"
            return
        self._latest_frame = frame
        self._last_frame_at = now
        self._accepted_frames += 1
        self._last_error = None
        self._require_t_pose_frame = False

    def _is_t_pose_candidate_locked(self, world: dict[str, Transform]) -> bool:
        """Return whether a complete frame has a credible humanoid rest pose.

        N.E.K.O's documented T-pose stream has no OSC-level marker, so the
        relay validates anatomy instead: hands must be spread to opposite
        sides and paired limbs must be approximately symmetric.  Thresholds
        are proportional to source height and deliberately accept both T and
        moderately sloped A poses.
        """
        hips = world["Hips"].position
        head = world["Head"].position
        left_hand = world["LeftHand"].position
        right_hand = world["RightHand"].position
        left_foot = world["LeftFoot"].position
        right_foot = world["RightFoot"].position
        toe_heights = [
            world[name].position[1]
            for name in ("LeftToes", "RightToes")
            if name in world
        ]
        floor_y = min(toe_heights) if toe_heights else min(left_foot[1], right_foot[1])
        source_height = head[1] - floor_y + 0.12
        if not math.isfinite(source_height) or source_height < 0.3:
            return False

        hand_span = right_hand[0] - left_hand[0]
        foot_span = right_foot[0] - left_foot[0]
        hand_mid = _v_scale(_v_add(left_hand, right_hand), 0.5)
        foot_mid = _v_scale(_v_add(left_foot, right_foot), 0.5)
        return bool(
            left_hand[0] < hips[0]
            and right_hand[0] > hips[0]
            and hand_span >= 0.35 * source_height
            and abs(hand_mid[0] - hips[0]) <= 0.08 * source_height
            and abs(left_hand[1] - right_hand[1]) <= 0.12 * source_height
            and abs(left_hand[2] - right_hand[2]) <= 0.12 * source_height
            and left_foot[0] < hips[0]
            and right_foot[0] > hips[0]
            and foot_span >= 0.04 * source_height
            and abs(foot_mid[0] - hips[0]) <= 0.06 * source_height
            and abs(left_foot[1] - right_foot[1]) <= 0.06 * source_height
            and abs(left_foot[2] - right_foot[2]) <= 0.10 * source_height
        )

    def _frame_from_world_locked(self, world: dict[str, Transform]) -> FrameState:
        hips = world["Hips"]
        head = world["Head"]
        left_foot = world["LeftFoot"]
        right_foot = world["RightFoot"]
        if self._origin is None:
            toe_heights = [
                world[name].position[1]
                for name in ("LeftToes", "RightToes")
                if name in world
            ]
            floor_y = min(toe_heights) if toe_heights else min(
                left_foot.position[1], right_foot.position[1]
            ) - 0.08
            self._origin = hips.position[0], floor_y, hips.position[2]
            source_height = max(0.3, head.position[1] - floor_y + 0.12)
            self._scale = self.profile.height_m / source_height
            self._rest_rotations = {
                device: world[bone].rotation for device, bone in _DEVICE_BONES.items()
            }
            self._rest_finger_rotations = {
                bone: transform.rotation
                for bone, transform in self._pending_bones.items()
                if bone in _FINGER_BONES
            }
            self._calibrated_at = self._clock()

        origin = self._origin
        assert origin is not None

        def scaled_position(value: Vec3) -> Vec3:
            return (
                (value[0] - origin[0]) * self._scale,
                (value[1] - origin[1]) * self._scale,
                (value[2] - origin[2]) * self._scale,
            )

        joint_positions = {
            device: scaled_position(world[bone].position)
            for device, bone in _DEVICE_BONES.items()
        }
        for side in ("left", "right"):
            hand = world[f"{side.title()}Hand"].position
            elbow = world[f"{side.title()}LowerArm"].position
            palm = _v_add(hand, _v_scale(_v_sub(hand, elbow), 0.21))
            joint_positions[f"{side}_controller"] = scaled_position(palm)

        rotations: dict[str, Quat] = {}
        rotation_deltas: dict[str, Quat] = {}
        for device, bone in _DEVICE_BONES.items():
            delta = _quat_multiply(
                world[bone].rotation,
                _quat_conjugate(self._rest_rotations[device]),
            )
            rotation_deltas[device] = delta
            if device == "left_controller":
                delta = _quat_multiply(delta, LEFT_CANONICAL_QUAT)
            elif device == "right_controller":
                delta = _quat_multiply(delta, RIGHT_CANONICAL_QUAT)
            rotations[device] = delta

        # The VMC bones are anatomical joint origins, whereas OpenVR expects
        # HMD/controller/tracker mount positions.  Capture that difference from
        # the authoritative host T pose, then rotate each mount offset with its
        # driving bone.  This keeps the calibrated six-point rig aligned while
        # the head, wrists, pelvis, and feet turn.
        if not self._position_mount_offsets:
            profile_scale = self.profile.height_m / _CANONICAL_HEIGHT_M
            self._position_mount_offsets = {
                device: _v_sub(
                    _v_scale(_CANONICAL_DEVICE_POSITIONS[device], profile_scale),
                    joint_positions[device],
                )
                for device in _DEVICE_BONES
            }
        positions = {
            device: _v_add(
                joint_positions[device],
                _quat_rotate(rotation_deltas[device], self._position_mount_offsets[device]),
            )
            for device in _DEVICE_BONES
        }

        finger_bends: dict[str, dict[str, float]] = {}
        for controller, fingers in _FINGER_CHAINS.items():
            finger_bends[controller] = {}
            for finger, chain in fingers.items():
                angle = 0.0
                full_curl = 0.0
                for bone, full_curl_degrees in chain:
                    current = self._pending_bones.get(bone)
                    rest = self._rest_finger_rotations.get(bone)
                    if current is None:
                        continue
                    if rest is None:
                        self._rest_finger_rotations[bone] = current.rotation
                        rest = current.rotation
                    angle += _quat_angle(current.rotation, rest)
                    full_curl += math.radians(full_curl_degrees)
                bend = angle / full_curl if full_curl > 0.0 else 0.0
                # Suppress tiny rest-pose jitter while retaining the complete
                # normalized range expected by AnyaDance's finger synthesizer.
                finger_bends[controller][finger] = 0.0 if bend < 0.015 else min(1.0, bend)

        return FrameState(
            devices={
                name: DeviceState(positions[name], rotations[name])
                for name in _DEVICE_BONES
            },
            controllers={
                name: ControllerState(finger_bends=finger_bends[name])
                for name in ("left_controller", "right_controller")
            },
        )

    def latest_frame(self) -> FrameState | None:
        now = self._clock()
        with self._lock:
            fresh = (
                self._source_available
                and self._latest_frame is not None
                and self._last_frame_at is not None
                and now - self._last_frame_at <= self.config.stale_after_ms / 1000.0
            )
            return self._latest_frame.clone() if fresh and self._latest_frame is not None else None

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            age_ms = (
                max(0.0, (now - self._last_frame_at) * 1000.0)
                if self._last_frame_at is not None
                else None
            )
            fresh = bool(
                self._source_available
                and age_ms is not None
                and age_ms <= self.config.stale_after_ms
            )
            if self._last_error and not self._receiver_listening:
                connection = "error"
            elif fresh:
                connection = "detected"
            elif self._last_packet_at is not None:
                connection = "stale"
            elif self._receiver_listening:
                connection = "listening"
            else:
                connection = "unknown"
            return {
                "enabled": self.config.enabled,
                "listen_address": f"{self.config.listen_host}:{self.config.listen_port}",
                "allowed_sender": self.config.allowed_sender,
                "receiver_listening": self._receiver_listening,
                "connection": connection,
                "source_available": fresh,
                "stale_after_ms": self.config.stale_after_ms,
                "last_frame_age_ms": round(age_ms, 1) if age_ms is not None else None,
                "last_packet_at_monotonic": self._last_packet_at,
                "last_frame_at_monotonic": self._last_frame_at,
                "received_packets": self._received_packets,
                "accepted_frames": self._accepted_frames,
                "rejected_packets": self._rejected_packets,
                "incomplete_frames": self._incomplete_frames,
                "last_error": self._last_error,
                "calibration": {
                    "generation": self._calibration_generation,
                    "reason": self._calibration_reason,
                    "held": self._calibration_held,
                    "calibrated": self._calibrated_at is not None,
                    "calibrated_at_monotonic": self._calibrated_at,
                    "waiting_for_t_pose_frame": self._require_t_pose_frame,
                    "rejected_frames": self._rejected_calibration_frames,
                    "position_mount_offsets_m": {
                        device: [round(value, 4) for value in offset]
                        for device, offset in self._position_mount_offsets.items()
                    },
                },
            }


__all__ = ["Transform", "VmcIdleRelay"]
