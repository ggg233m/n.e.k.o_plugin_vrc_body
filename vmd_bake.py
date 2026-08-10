"""Offline VMD -> AnyaDance .nya baking helpers.

The real skeleton solve is intentionally delegated to Blender + MMD Tools via
AnyaDance's ``blender_export_mmd.py``.  This module only performs AnyaDance's
deterministic solved-joint -> six-device retarget and writes the resulting
``.nya`` clip.  It uses the Python standard library and is never imported by the
60 Hz scheduler.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

JOINTS = (
    "pelvis",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
    "left_toe",
    "right_toe",
)
DEVICE_JOINTS = {
    "hmd": "head",
    "left_controller": "left_wrist",
    "right_controller": "right_wrist",
    "hip": "pelvis",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}

_BODY_FORWARD: Vec3 = (0.0, 0.0, -1.0)
_BODY_UP: Vec3 = (0.0, 1.0, 0.0)
_BODY_RIGHT: Vec3 = (1.0, 0.0, 0.0)
_CONTROLLER_LOCAL_FORWARD: Vec3 = (0.0, 0.0, -1.0)
_NEUTRAL_INDEX_LEFT: Vec3 = (0.11569004, -0.51338429, -0.85032487)
_NEUTRAL_INDEX_RIGHT: Vec3 = (-0.11569004, -0.51338429, -0.85032487)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _v_sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _v_scale(value: Vec3, scalar: float) -> Vec3:
    return value[0] * scalar, value[1] * scalar, value[2] * scalar


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize_vec(value: Vec3, fallback: Vec3) -> Vec3:
    length = math.sqrt(_dot(value, value))
    if length <= 0.0 or not math.isfinite(length):
        return fallback
    return value[0] / length, value[1] / length, value[2] / length


def _normalize_quat(value: Quat) -> Quat:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 0.0 or not math.isfinite(length):
        return 0.0, 0.0, 0.0, 1.0
    return tuple(component / length for component in value)  # type: ignore[return-value]


def _quat_conjugate(value: Quat) -> Quat:
    return -value[0], -value[1], -value[2], value[3]


def _quat_multiply(lhs: Quat, rhs: Quat) -> Quat:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_rotate(rotation: Quat, value: Vec3) -> Vec3:
    qv: Quat = value[0], value[1], value[2], 0.0
    result = _quat_multiply(_quat_multiply(rotation, qv), _quat_conjugate(rotation))
    return result[0], result[1], result[2]


def _project_perpendicular(reference: Vec3, normal: Vec3, minimum_length_squared: float) -> Vec3 | None:
    projected = _v_sub(reference, _v_scale(normal, _dot(reference, normal)))
    if _dot(projected, projected) < minimum_length_squared:
        return None
    return _normalize_vec(projected, _BODY_FORWARD)


def _best_perpendicular(candidates: Iterable[Vec3], normal: Vec3) -> Vec3:
    best = _BODY_FORWARD
    best_length_squared = -1.0
    for candidate in candidates:
        projected = _v_sub(candidate, _v_scale(normal, _dot(candidate, normal)))
        length_squared = _dot(projected, projected)
        if length_squared > best_length_squared:
            best = projected
            best_length_squared = length_squared
    if best_length_squared < 1e-6:
        return _BODY_FORWARD
    return _normalize_vec(best, _BODY_FORWARD)


def _basis(primary: Vec3, secondary: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    c1 = _normalize_vec(primary, (0.0, -1.0, 0.0))
    c2 = _project_perpendicular(secondary, c1, 1e-8) or _BODY_FORWARD
    c0 = _normalize_vec(_cross(c1, c2), _BODY_RIGHT)
    c2 = _normalize_vec(_cross(c0, c1), c2)
    return c0, c1, c2


def _matrix_to_quat(matrix: Sequence[Sequence[float]]) -> Quat:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        result = (
            (matrix[2][1] - matrix[1][2]) / s,
            (matrix[0][2] - matrix[2][0]) / s,
            (matrix[1][0] - matrix[0][1]) / s,
            0.25 * s,
        )
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        result = (
            0.25 * s,
            (matrix[0][1] + matrix[1][0]) / s,
            (matrix[0][2] + matrix[2][0]) / s,
            (matrix[2][1] - matrix[1][2]) / s,
        )
    elif matrix[1][1] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        result = (
            (matrix[0][1] + matrix[1][0]) / s,
            0.25 * s,
            (matrix[1][2] + matrix[2][1]) / s,
            (matrix[0][2] - matrix[2][0]) / s,
        )
    else:
        s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        result = (
            (matrix[0][2] + matrix[2][0]) / s,
            (matrix[1][2] + matrix[2][1]) / s,
            0.25 * s,
            (matrix[1][0] - matrix[0][1]) / s,
        )
    return _normalize_quat(result)


def _basis_mapping(local_primary: Vec3, local_secondary: Vec3, world_primary: Vec3, world_secondary: Vec3) -> Quat:
    local = _basis(local_primary, local_secondary)
    world = _basis(world_primary, world_secondary)
    matrix = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            matrix[row][column] = sum(world[axis][row] * local[axis][column] for axis in range(3))
    return _matrix_to_quat(matrix)


def _controller_reference_twist(shoulder: Vec3, elbow: Vec3, wrist: Vec3) -> Vec3:
    finger_axis = _normalize_vec(_v_sub(wrist, elbow), (0.0, -1.0, 0.0))
    upper_to_shoulder = _normalize_vec(_v_sub(shoulder, elbow), _BODY_UP)
    projected = _project_perpendicular(upper_to_shoulder, finger_axis, 1e-4)
    if projected is not None:
        return projected
    return _best_perpendicular((_BODY_FORWARD, _BODY_UP, _BODY_RIGHT, _BODY_FORWARD, _BODY_UP), finger_axis)


def _controller_rotation(
    shoulder: Vec3,
    elbow: Vec3,
    wrist: Vec3,
    wrist_rotation: Quat,
    wrist_twist_local: Vec3,
    *,
    left: bool,
) -> Quat:
    finger_axis = _normalize_vec(_v_sub(wrist, elbow), (0.0, -1.0, 0.0))
    upper_to_shoulder = _normalize_vec(_v_sub(shoulder, elbow), _BODY_UP)
    hand_forward = _project_perpendicular(_quat_rotate(wrist_rotation, wrist_twist_local), finger_axis, 1e-4)
    if hand_forward is None:
        hand_forward = _project_perpendicular(upper_to_shoulder, finger_axis, 1e-4)
    if hand_forward is None:
        hand_forward = _best_perpendicular((_BODY_FORWARD, _BODY_UP, _BODY_RIGHT, _BODY_FORWARD, _BODY_UP), finger_axis)
    neutral_index = _NEUTRAL_INDEX_LEFT if left else _NEUTRAL_INDEX_RIGHT
    return _basis_mapping(neutral_index, _CONTROLLER_LOCAL_FORWARD, finger_axis, hand_forward)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _joint_set(value: Any, label: str) -> dict[str, tuple[Vec3, Quat]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result: dict[str, tuple[Vec3, Quat]] = {}
    for name in JOINTS:
        joint = value.get(name)
        if not isinstance(joint, Mapping):
            raise ValueError(f"{label}.{name} must be an object")
        position = _vector(joint.get("p"), 3, f"{label}.{name}.p")
        rotation = _normalize_quat(_vector(joint.get("q"), 4, f"{label}.{name}.q"))
        result[name] = (position, rotation)  # type: ignore[arg-type]
    return result


def _dance_local(
    joint: tuple[Vec3, Quat], *, root_x: float, root_z: float, floor_y: float
) -> tuple[Vec3, Quat]:
    position, rotation = joint
    return (position[0] - root_x, position[1] - floor_y, position[2] - root_z), _normalize_quat(rotation)


def retarget_solved_document(
    document: Mapping[str, Any],
    *,
    target_height_m: float = 1.5,
    hand_reach_scale: float = 1.22,
    body_width_scale: float = 1.0,
    leg_length_scale: float = 1.0,
    hip_height_offset_m: float = 0.0,
    head_mount_up_m: float = 0.10,
    floor_offset_m: float = 0.0,
    loop: bool = True,
) -> dict[str, Any]:
    """Convert an AnyaDance solved-motion document into a version-1 .nya document."""

    if document.get("format") != "anyadance_mmd_solved":
        raise ValueError("input is not an AnyaDance solved MMD document")
    if document.get("version", 1) != 1:
        raise ValueError(f"unsupported solved-motion version: {document.get('version')}")
    fps = _number(document.get("fps", 60.0), "fps")
    if not 1.0 <= fps <= 240.0:
        raise ValueError("fps must be between 1 and 240")
    target_height_m = _number(target_height_m, "target_height_m")
    hand_reach_scale = _number(hand_reach_scale, "hand_reach_scale")
    body_width_scale = _number(body_width_scale, "body_width_scale")
    leg_length_scale = _number(leg_length_scale, "leg_length_scale")
    hip_height_offset_m = _number(hip_height_offset_m, "hip_height_offset_m")
    head_mount_up_m = _number(head_mount_up_m, "head_mount_up_m")
    floor_offset_m = _number(floor_offset_m, "floor_offset_m")
    if not 0.5 <= target_height_m <= 2.0:
        raise ValueError("target_height_m must be between 0.5 and 2.0")
    if not 0.5 <= hand_reach_scale <= 2.0:
        raise ValueError("hand_reach_scale must be between 0.5 and 2.0")
    if not 0.5 <= body_width_scale <= 1.5:
        raise ValueError("body_width_scale must be between 0.5 and 1.5")
    if not 0.7 <= leg_length_scale <= 1.3:
        raise ValueError("leg_length_scale must be between 0.7 and 1.3")
    if not -0.25 <= hip_height_offset_m <= 0.25:
        raise ValueError("hip_height_offset_m must be between -0.25 and 0.25")

    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("solved motion has no frames")
    parsed_frames: list[tuple[float, dict[str, tuple[Vec3, Quat]], list[float] | None, list[float] | None]] = []
    previous_time: float | None = None
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise ValueError(f"frames[{index}] must be an object")
        timestamp = _number(raw_frame.get("t"), f"frames[{index}].t")
        if timestamp < 0.0 or (previous_time is not None and timestamp <= previous_time):
            raise ValueError("frame timestamps must be non-negative and strictly increasing")
        previous_time = timestamp
        joints = _joint_set(raw_frame.get("j"), f"frames[{index}].j")
        left_fingers = right_fingers = None
        if raw_frame.get("fl") is not None and raw_frame.get("fr") is not None:
            left_fingers = [min(1.0, max(0.0, value)) for value in _vector(raw_frame.get("fl"), 5, f"frames[{index}].fl")]
            right_fingers = [min(1.0, max(0.0, value)) for value in _vector(raw_frame.get("fr"), 5, f"frames[{index}].fr")]
        parsed_frames.append((timestamp, joints, left_fingers, right_fingers))

    rest = _joint_set(document.get("rest"), "rest") if document.get("rest") is not None else parsed_frames[0][1]
    root_x = rest["pelvis"][0][0]
    root_z = rest["pelvis"][0][2]
    contacts = [
        min(frame[1][name][0][1] for name in ("left_ankle", "right_ankle", "left_toe", "right_toe"))
        for frame in parsed_frames
    ]
    contacts.extend((rest["left_toe"][0][1], rest["right_toe"][0][1]))
    floor_y = _percentile(contacts, 0.05)
    rest_local = {
        name: _dance_local(joint, root_x=root_x, root_z=root_z, floor_y=floor_y)
        for name, joint in rest.items()
    }
    source_height = max(
        0.3,
        min(
            5.0,
            rest_local["head"][0][1]
            - min(rest_local[name][0][1] for name in ("left_toe", "right_toe", "left_ankle", "right_ankle"))
            + 0.12,
        ),
    )
    scale = target_height_m / max(1e-3, source_height)
    rest_leg_length = (
        rest_local["pelvis"][0][1]
        - (rest_local["left_ankle"][0][1] + rest_local["right_ankle"][0][1]) * 0.5
    ) * scale
    upper_body_lift = rest_leg_length * (leg_length_scale - 1.0) + hip_height_offset_m
    inverse_rest_rotation = {
        device: _quat_conjugate(rest_local[joint][1]) for device, joint in DEVICE_JOINTS.items()
    }

    def calibrate_twist(side: str) -> Vec3:
        reference = _controller_reference_twist(
            rest_local[f"{side}_shoulder"][0],
            rest_local[f"{side}_elbow"][0],
            rest_local[f"{side}_wrist"][0],
        )
        local = _quat_rotate(_quat_conjugate(rest_local[f"{side}_wrist"][1]), reference)
        return _normalize_vec(local, _BODY_FORWARD)

    twist = {"left": calibrate_twist("left"), "right": calibrate_twist("right")}
    output_frames: list[dict[str, Any]] = []
    for timestamp, joints, left_fingers, right_fingers in parsed_frames:
        local = {
            name: _dance_local(joint, root_x=root_x, root_z=root_z, floor_y=floor_y)
            for name, joint in joints.items()
        }

        def device_position(name: str) -> Vec3:
            position = local[DEVICE_JOINTS[name]][0]
            if name.endswith("_controller"):
                side = "left" if name.startswith("left") else "right"
                wrist = local[f"{side}_wrist"][0]
                elbow = local[f"{side}_elbow"][0]
                shoulder = local[f"{side}_shoulder"][0]
                palm = _v_add(wrist, _v_scale(_v_sub(wrist, elbow), 0.21))
                position = _v_add(shoulder, _v_scale(_v_sub(palm, shoulder), hand_reach_scale))
            scaled = _v_scale(position, scale)
            pelvis_scaled = _v_scale(local["pelvis"][0], scale)
            if name in {"left_controller", "right_controller", "left_foot", "right_foot"}:
                scaled = (
                    pelvis_scaled[0] + (scaled[0] - pelvis_scaled[0]) * body_width_scale,
                    scaled[1],
                    scaled[2],
                )
            if name in {"hmd", "hip", "left_controller", "right_controller"}:
                scaled = scaled[0], scaled[1] + upper_body_lift, scaled[2]
            lift = head_mount_up_m if name == "hmd" else 0.0
            return scaled[0], min(2.0, scaled[1] + lift + floor_offset_m), scaled[2]

        def device_rotation(name: str) -> Quat:
            if name.endswith("_controller"):
                side = "left" if name.startswith("left") else "right"
                return _controller_rotation(
                    local[f"{side}_shoulder"][0],
                    local[f"{side}_elbow"][0],
                    local[f"{side}_wrist"][0],
                    local[f"{side}_wrist"][1],
                    twist[side],
                    left=side == "left",
                )
            joint = local[DEVICE_JOINTS[name]]
            return _normalize_quat(_quat_multiply(joint[1], inverse_rest_rotation[name]))

        devices = {
            name: {"p": list(device_position(name)), "q": list(device_rotation(name))}
            for name in DEVICE_JOINTS
        }
        frame: dict[str, Any] = {"t": timestamp, "devices": devices}
        if left_fingers is not None and right_fingers is not None:
            frame["fingers"] = {"left": left_fingers, "right": right_fingers}
        output_frames.append(frame)

    model = document.get("model", "")
    if not isinstance(model, str):
        model = str(model)
    return {
        "format": "anyadance_nya",
        "version": 1,
        "loop": bool(loop),
        "fps": fps,
        "model": model,
        "bake_profile": {
            "target_height_m": target_height_m,
            "hand_reach_scale": hand_reach_scale,
            "body_width_scale": body_width_scale,
            "leg_length_scale": leg_length_scale,
            "hip_height_offset_m": hip_height_offset_m,
            "head_mount_up_m": head_mount_up_m,
            "floor_offset_m": floor_offset_m,
        },
        "frames": output_frames,
    }


def convert_solved_file(
    solved_path: Path,
    output_path: Path,
    *,
    loop: bool = True,
    target_height_m: float = 1.5,
    hand_reach_scale: float = 1.22,
    body_width_scale: float = 1.0,
    leg_length_scale: float = 1.0,
    hip_height_offset_m: float = 0.0,
) -> dict[str, Any]:
    document = json.loads(solved_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("solved-motion root must be an object")
    result = retarget_solved_document(
        document,
        loop=loop,
        target_height_m=target_height_m,
        hand_reach_scale=hand_reach_scale,
        body_width_scale=body_width_scale,
        leg_length_scale=leg_length_scale,
        hip_height_offset_m=hip_height_offset_m,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(output_path)
    return result


def bake_vmd_file(
    *,
    blender_path: Path,
    export_script_path: Path,
    model_path: Path,
    vmd_path: Path,
    output_path: Path,
    mmd_tools_path: Path | None = None,
    fps: float = 60.0,
    loop: bool = True,
    target_height_m: float = 1.5,
    hand_reach_scale: float = 1.22,
    body_width_scale: float = 1.0,
    leg_length_scale: float = 1.0,
    hip_height_offset_m: float = 0.0,
) -> dict[str, Any]:
    for label, path in (
        ("Blender", blender_path),
        ("AnyaDance export script", export_script_path),
        ("PMX/PMD model", model_path),
        ("VMD motion", vmd_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    with tempfile.TemporaryDirectory(prefix="neko-anyadance-vmd-") as directory:
        solved_path = Path(directory) / "solved.json"
        command = [
            str(blender_path),
            "--background",
            "--python",
            str(export_script_path),
            "--",
            "--model",
            str(model_path),
            "--vmd",
            str(vmd_path),
            "--output",
            str(solved_path),
            "--fps",
            f"{fps:g}",
        ]
        if mmd_tools_path is not None:
            command.extend(("--mmd-tools-path", str(mmd_tools_path)))
        completed = subprocess.run(command, check=False, capture_output=True, text=True, errors="replace")
        if completed.returncode != 0 or not solved_path.is_file():
            details = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-20:])
            raise RuntimeError(f"Blender MMD solve failed ({completed.returncode}):\n{details}")
        return convert_solved_file(
            solved_path,
            output_path,
            loop=loop,
            target_height_m=target_height_m,
            hand_reach_scale=hand_reach_scale,
            body_width_scale=body_width_scale,
            leg_length_scale=leg_length_scale,
            hip_height_offset_m=hip_height_offset_m,
        )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Bake a real VMD motion into an AnyaDance .nya clip")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--solved", type=Path, help="existing anyadance_mmd_solved JSON")
    source.add_argument("--vmd", type=Path, help="VMD motion to solve with Blender")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--export-script", type=Path)
    parser.add_argument("--mmd-tools", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--target-height", type=float, default=1.5)
    parser.add_argument("--hand-reach-scale", type=float, default=1.22)
    parser.add_argument("--body-width-scale", type=float, default=1.0)
    parser.add_argument("--leg-length-scale", type=float, default=1.0)
    parser.add_argument("--hip-height-offset", type=float, default=0.0)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()
    if args.solved is not None:
        result = convert_solved_file(
            args.solved,
            args.output,
            loop=not args.no_loop,
            target_height_m=args.target_height,
            hand_reach_scale=args.hand_reach_scale,
            body_width_scale=args.body_width_scale,
            leg_length_scale=args.leg_length_scale,
            hip_height_offset_m=args.hip_height_offset,
        )
    else:
        missing = [name for name in ("model", "blender", "export_script") if getattr(args, name) is None]
        if missing:
            parser.error("--vmd also requires " + ", ".join("--" + name.replace("_", "-") for name in missing))
        result = bake_vmd_file(
            blender_path=args.blender,
            export_script_path=args.export_script,
            model_path=args.model,
            vmd_path=args.vmd,
            output_path=args.output,
            mmd_tools_path=args.mmd_tools,
            fps=args.fps,
            loop=not args.no_loop,
            target_height_m=args.target_height,
            hand_reach_scale=args.hand_reach_scale,
            body_width_scale=args.body_width_scale,
            leg_length_scale=args.leg_length_scale,
            hip_height_offset_m=args.hip_height_offset,
        )
    frames = result["frames"]
    duration = frames[-1]["t"] if frames else 0.0
    print(f"Wrote {args.output} ({len(frames)} frames, {duration:.3f}s @ {result['fps']:g} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
