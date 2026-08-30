"""独立 YUI NPC 插件的 v1.1/v1.2/v1.3 Python 协议编解码。

本模块只处理确定性的 wire 事实：MIDI 帧、CRC、量化、UTF-8 载荷和
``[NEKO]`` 日志公共头。会话、重试和 LLM 语义适配放在更高一层，避免把
传输细节与业务状态混在一起。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


SPEC_VERSION = "1.1"
CURRENT_SPEC_VERSION = "1.3"
SUPPORTED_SPEC_VERSIONS = (SPEC_VERSION, "1.2", CURRENT_SPEC_VERSION)
WIRE_VERSION = 1
NPC_ID = "yui"
MIDI_PORT_NAME = "NEKO_MIDI"
NEKO_LOG_MARKER = "[NEKO]"

COMMAND_CHANNEL = 0
UPPER_BODY_CHANNEL = 1
TEXT_CHANNEL = 2
TEXT_PAYLOAD_CC = 29
UPPER_BODY_COMMIT_NOTE = 0x40
ESTOP_COMMAND_ID = 0x7F

SEQUENCE_MIN = 1
SEQUENCE_MAX = 127
SESSION_MIN = 1
SESSION_MAX = 268_435_455
TEXT_MAX_UTF8_BYTES = 384
MAX_LOG_JSON_UTF8_BYTES = 950

COMMAND_REGISTER_CCS = (20, 21, 22, 23, 24, 25, 26, 27, 28)

COMMAND_IDS: dict[str, int] = {
    "SET_MODE": 0x01,
    "GOTO_XZ": 0x02,
    "SET_SPEED": 0x03,
    "TURN_TO": 0x04,
    "LOOK_AT": 0x05,
    "PLAY_ANIM": 0x06,
    "STOP": 0x07,
    "TEXT_PRESET": 0x08,
    "RAY_SCAN": 0x09,
    "SET_RATE": 0x0A,
    "HEARTBEAT": 0x0B,
    "DISCOVER": 0x0C,
    "CLEAR_ESTOP": 0x0D,
    "STOP_ACTION": 0x0E,
    "SNAPSHOT_REQUEST": 0x0F,
    "SET_TARGET": 0x10,
    "LOOK_AT_XYZ": 0x11,
    "SET_EXPRESSION": 0x12,
    "TEXT_BEGIN": 0x13,
    "TEXT_COMMIT": 0x14,
    "SPEECH_CUE": 0x15,
    "SET_CONTROL_MODE": 0x16,
    "GOTO_ANCHOR": 0x17,
    "ORBIT_ENTITY": 0x18,
    "MOVE_RELATIVE": 0x19,
    "EXPLORE_REGION": 0x1A,
    "ESTOP": ESTOP_COMMAND_ID,
}
COMMAND_NAMES = {command_id: name for name, command_id in COMMAND_IDS.items()}

CAPABILITY_BITS: dict[str, int] = {
    "goto": 0,
    "follow": 1,
    "wander": 2,
    "actions": 3,
    "expressions": 4,
    "text_preset": 5,
    "text_utf8": 6,
    "upper_body_stream": 7,
    "ray_scan": 8,
    "touch": 9,
    "player_pose": 10,
    "voice_stream": 11,
    "snapshot": 12,
    "navmesh": 13,
    "social_signals": 14,
    "anchors": 15,
    "operation_lifecycle": 16,
    "world_map": 17,
    "semantic_navigation": 18,
    "region_localization": 19,
    "local_navigation": 20,
}

ERROR_CODES: dict[str, int] = {
    "unknown_cmd": 1,
    "not_handshaken": 3,
    "not_driver": 4,
    "not_owner": 5,
    "invalid_state": 6,
    "estop_latched": 7,
    "invalid_param": 8,
    "reserved_bits": 9,
    "target_out_of_bounds": 10,
    "target_not_on_navmesh": 11,
    "no_path": 12,
    "target_missing": 13,
    "slot_unknown": 14,
    "unsupported_capability": 15,
    "action_not_found": 16,
    "action_busy": 17,
    "expression_not_found": 18,
    "text_preset_not_found": 19,
    "transfer_busy": 20,
    "transfer_missing": 21,
    "transfer_seq_mismatch": 22,
    "text_too_long": 23,
    "length_mismatch": 24,
    "crc_mismatch": 25,
    "invalid_utf8": 26,
    "text_timeout": 27,
    "stream_incomplete": 28,
    "seq_conflict": 29,
    "voice_unavailable": 30,
    "ownership_failed": 31,
    "session_conflict": 32,
    "rate_limited": 33,
    "internal_error": 34,
    "driver_auth_failed": 35,
    "action_seq_conflict": 36,
    "speech_seq_conflict": 37,
    "catalog_invalid": 38,
}

UPPER_BODY_PARAMETERS: tuple[tuple[str, int, float, float, str], ...] = (
    ("head_yaw", 20, -60.0, 60.0, "centered"),
    ("head_pitch", 21, -35.0, 35.0, "centered"),
    ("torso_yaw", 22, -35.0, 35.0, "centered"),
    ("torso_pitch", 23, -20.0, 20.0, "centered"),
    ("left_arm_elevation", 24, 0.0, 160.0, "unsigned"),
    ("left_arm_azimuth", 25, -120.0, 120.0, "centered"),
    ("right_arm_elevation", 26, 0.0, 160.0, "unsigned"),
    ("right_arm_azimuth", 27, -120.0, 120.0, "centered"),
)
UPPER_BODY_NEUTRAL_Q = (64, 64, 64, 64, 0, 64, 0, 64)


class YuiProtocolError(ValueError):
    """输入不能表示为冻结的 YUI wire 数据。"""


class YuiLogDecodeError(YuiProtocolError):
    """一条带 ``[NEKO]`` 标记的日志不符合公共格式。"""


@dataclass(frozen=True)
class MidiEvent:
    """与测试向量字段一一对应的单条 MIDI 事件。"""

    type: Literal["cc", "note_on"]
    channel: int
    number: int
    value: int

    def as_dict(self, *, at_ms: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "channel": self.channel,
            "number": self.number,
        }
        result["value" if self.type == "cc" else "velocity"] = self.value
        if at_ms is not None:
            result["at_ms"] = int(at_ms)
        return result


@dataclass(frozen=True)
class CommandFrame:
    """一条可靠命令的规范寄存器快照和物理事件。"""

    command: str
    command_id: int
    sequence: int
    parameters: tuple[int, int, int, int, int, int]
    request_hash: str
    events: tuple[MidiEvent, ...]


@dataclass(frozen=True)
class TextTransaction:
    """一笔完整动态文本事务，BEGIN/载荷/COMMIT 可分阶段发送。"""

    transfer_sequence: int
    text: str
    utf8_bytes: bytes
    crc16: str
    begin: CommandFrame
    payload: tuple[MidiEvent, ...]
    commit: CommandFrame


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise YuiProtocolError(f"{name} 必须是 {minimum}..{maximum} 的整数")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise YuiProtocolError(f"{name} 必须是 {minimum}..{maximum} 的整数") from exc
    if parsed != value or not minimum <= parsed <= maximum:
        raise YuiProtocolError(f"{name} 必须是 {minimum}..{maximum} 的整数")
    return parsed


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise YuiProtocolError(f"{name} 必须是有限数值")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise YuiProtocolError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(parsed):
        raise YuiProtocolError(f"{name} 必须是有限数值")
    return parsed


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def split_u14(value: Any, name: str = "value") -> tuple[int, int]:
    parsed = _integer(value, name, 0, 16383)
    return parsed >> 7, parsed & 0x7F


def join_u14(high: Any, low: Any) -> int:
    return _integer(high, "high", 0, 127) * 128 + _integer(low, "low", 0, 127)


def crc16_ccitt_false(data: bytes | bytearray | memoryview) -> int:
    """计算 CRC-16/CCITT-FALSE（poly=0x1021、init=0xFFFF）。"""
    crc = 0xFFFF
    for byte in bytes(data):
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def command_request_hash(
    command_id: Any,
    sequence: Any,
    parameters: Sequence[Any],
) -> str:
    command_id_value = _integer(command_id, "command_id", 0, 127)
    sequence_value = _integer(sequence, "sequence", 0, 127)
    if len(parameters) != 6:
        raise YuiProtocolError("parameters 必须恰好包含 P0..P5 六项")
    p0 = _integer(parameters[0], "P0", 0, 16383)
    p1 = _integer(parameters[1], "P1", 0, 16383)
    p2 = _integer(parameters[2], "P2", 0, 16383)
    p3 = _integer(parameters[3], "P3", 0, 127)
    p4 = _integer(parameters[4], "P4", 0, 127)
    p5 = _integer(parameters[5], "P5", 0, 127)
    p0_hi, p0_lo = split_u14(p0, "P0")
    p1_hi, p1_lo = split_u14(p1, "P1")
    p2_hi, p2_lo = split_u14(p2, "P2")
    payload = bytes((
        command_id_value,
        sequence_value,
        p0_hi,
        p0_lo,
        p1_hi,
        p1_lo,
        p2_hi,
        p2_lo,
        p3,
        p4,
        p5,
    ))
    return f"{crc16_ccitt_false(payload):04X}"


_PARAMETER_INDEX = {f"P{index}": index for index in range(6)}


@lru_cache(maxsize=1)
def _frozen_command_constraints() -> Mapping[str, Mapping[str, Any]]:
    """缓存冻结事实源；编码热路径不重复读取 JSON。"""
    try:
        constants = load_frozen_constants(spec_version=CURRENT_SPEC_VERSION)
    except (OSError, json.JSONDecodeError) as exc:
        raise YuiProtocolError(f"无法加载冻结协议常量: {exc}") from exc
    commands = constants.get("commands")
    if not isinstance(commands, Mapping):
        raise YuiProtocolError("冻结常量缺少 commands 对象")
    return commands


def preload_command_constraints() -> None:
    """在构造期主动触发冻结常量加载；缺失文件提前失败而不是等首条命令。"""
    _frozen_command_constraints()


def _parameter_value(parameters: tuple[int, ...], name: str) -> int:
    try:
        return parameters[_PARAMETER_INDEX[name]]
    except KeyError as exc:
        raise YuiProtocolError(f"冻结常量包含未知参数名: {name}") from exc


def _constraint_range(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise YuiProtocolError(f"冻结常量 {label} 不是 [min,max]")
    minimum = _integer(value[0], f"{label}.min", 0, 268_435_455)
    maximum = _integer(value[1], f"{label}.max", 0, 268_435_455)
    if minimum > maximum:
        raise YuiProtocolError(f"冻结常量 {label} 满足 min > max")
    return minimum, maximum


def _validate_conditional_rules(command: str, parameters: tuple[int, ...]) -> None:
    """执行冻结常量中的少量条件规则；未知声明采取失败关闭。"""
    if command == "GOTO_XZ":
        if parameters[4] == 0 and parameters[2] != 0:
            raise YuiProtocolError("GOTO_XZ 在 P4.has_yaw=0 时 P2 必须为 0")
    elif command == "TEXT_PRESET":
        if parameters[3] == 127 and parameters[4] != 0:
            raise YuiProtocolError("TEXT_PRESET 清除气泡时 P4 必须为 0")
    elif command == "SET_EXPRESSION":
        if parameters[3] == 127 and (parameters[4] != 0 or parameters[5] != 0):
            raise YuiProtocolError("SET_EXPRESSION 清除表情时 P4/P5 必须为 0")
    elif command == "ORBIT_ENTITY":
        laps_minus_one = (parameters[5] >> 1) & 0x03
        if laps_minus_one > 2:
            raise YuiProtocolError("ORBIT_ENTITY P5.bits1..2 只能编码 1..3 圈")
    else:
        raise YuiProtocolError(f"{command} 含未实现的 conditional_rules，拒绝编码")


def validate_command_parameters(command: str, parameters: tuple[int, ...]) -> None:
    """按冻结常量校验一条已完成通用位宽解析的 P0..P5 快照。"""
    constraints = _frozen_command_constraints().get(command)
    if not isinstance(constraints, Mapping):
        raise YuiProtocolError(f"命令 {command} 未在冻结常量中声明")

    for name, raw_mask in constraints.get("reserved_zero_mask", {}).items():
        value = _parameter_value(parameters, str(name))
        mask = _integer(raw_mask, f"{command}.{name}.reserved_zero_mask", 0, 16383)
        if value & mask:
            raise YuiProtocolError(f"{command} {name} 的保留位必须为 0")

    for raw_name in constraints.get("zero", []):
        name = str(raw_name)
        if _parameter_value(parameters, name) != 0:
            raise YuiProtocolError(f"{command} {name} 必须为 0")

    for raw_name, raw_limits in constraints.get("range", {}).items():
        name = str(raw_name)
        minimum, maximum = _constraint_range(raw_limits, f"{command}.range.{name}")
        if name == "combined_session":
            value = parameters[0] + (parameters[1] << 14)
        else:
            value = _parameter_value(parameters, name)
        if not minimum <= value <= maximum:
            raise YuiProtocolError(f"{command} {name} 必须位于 {minimum}..{maximum}")

    for raw_name, raw_limits in constraints.get("semantic_range", {}).items():
        name = str(raw_name)
        minimum, maximum = _constraint_range(raw_limits, f"{command}.semantic_range.{name}")
        value = _parameter_value(parameters, name)
        if not minimum <= value <= maximum:
            raise YuiProtocolError(f"{command} {name} 语义范围必须位于 {minimum}..{maximum}")

    for raw_name, raw_ranges in constraints.get("allowed_ranges", {}).items():
        name = str(raw_name)
        value = _parameter_value(parameters, name)
        if not isinstance(raw_ranges, (list, tuple)):
            raise YuiProtocolError(f"冻结常量 {command}.allowed_ranges.{name} 不是范围数组")
        ranges = tuple(
            _constraint_range(item, f"{command}.allowed_ranges.{name}")
            for item in raw_ranges
        )
        if not any(minimum <= value <= maximum for minimum, maximum in ranges):
            display = ",".join(f"{minimum}..{maximum}" for minimum, maximum in ranges)
            raise YuiProtocolError(f"{command} {name} 必须位于允许范围 {display}")

    if constraints.get("conditional_rules"):
        _validate_conditional_rules(command, parameters)


def encode_command(
    command: str | int,
    sequence: Any,
    parameters: Sequence[Any] = (0, 0, 0, 0, 0, 0),
    *,
    estop_channel: int = COMMAND_CHANNEL,
) -> CommandFrame:
    """编码一条可靠命令；普通命令固定写 9 个 CC 后提交 NoteOn。"""
    if isinstance(command, str):
        normalized = command.strip().upper()
        if normalized not in COMMAND_IDS:
            raise YuiProtocolError(f"未知命令名: {command}")
        command_id = COMMAND_IDS[normalized]
        command_name = normalized
    else:
        command_id = _integer(command, "command_id", 0, 127)
        command_name = COMMAND_NAMES.get(command_id, "UNKNOWN")

    sequence_value = _integer(sequence, "sequence", 0, 127)
    if command_id != ESTOP_COMMAND_ID and sequence_value == 0:
        raise YuiProtocolError("非 ESTOP 命令的 sequence 必须是 1..127")
    if len(parameters) != 6:
        raise YuiProtocolError("parameters 必须恰好包含 P0..P5 六项")
    parsed = (
        _integer(parameters[0], "P0", 0, 16383),
        _integer(parameters[1], "P1", 0, 16383),
        _integer(parameters[2], "P2", 0, 16383),
        _integer(parameters[3], "P3", 0, 127),
        _integer(parameters[4], "P4", 0, 127),
        _integer(parameters[5], "P5", 0, 127),
    )
    validate_command_parameters(command_name, parsed)
    request_hash = command_request_hash(command_id, sequence_value, parsed)

    if command_id == ESTOP_COMMAND_ID:
        channel = _integer(estop_channel, "estop_channel", 0, 15)
        events = (MidiEvent("note_on", channel, ESTOP_COMMAND_ID, sequence_value),)
    else:
        p0_hi, p0_lo = split_u14(parsed[0], "P0")
        p1_hi, p1_lo = split_u14(parsed[1], "P1")
        p2_hi, p2_lo = split_u14(parsed[2], "P2")
        values = (p0_hi, p0_lo, p1_hi, p1_lo, p2_hi, p2_lo, parsed[3], parsed[4], parsed[5])
        events = tuple(
            MidiEvent("cc", COMMAND_CHANNEL, cc, value)
            for cc, value in zip(COMMAND_REGISTER_CCS, values, strict=True)
        ) + (MidiEvent("note_on", COMMAND_CHANNEL, command_id, sequence_value),)

    return CommandFrame(
        command=command_name,
        command_id=command_id,
        sequence=sequence_value,
        parameters=parsed,
        request_hash=request_hash,
        events=events,
    )


def encode_position_q14(value: Any, minimum: Any, maximum: Any) -> int:
    numeric = _finite(value, "position")
    low = _finite(minimum, "minimum")
    high = _finite(maximum, "maximum")
    if not low < high:
        raise YuiProtocolError("position bounds 必须满足 minimum < maximum")
    if not low <= numeric <= high:
        raise YuiProtocolError("position 超出 wire bounds")
    return _round_half_up(((numeric - low) / (high - low)) * 16383.0)


def decode_position_q14(value: Any, minimum: Any, maximum: Any) -> float:
    quantized = _integer(value, "position_q14", 0, 16383)
    low = _finite(minimum, "minimum")
    high = _finite(maximum, "maximum")
    if not low < high:
        raise YuiProtocolError("position bounds 必须满足 minimum < maximum")
    return low + quantized / 16383.0 * (high - low)


def wrap_yaw(degrees: Any) -> float:
    return _finite(degrees, "yaw") % 360.0


def normalize_bearing(degrees: Any) -> float:
    wrapped = (float(_finite(degrees, "bearing")) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def absolute_yaw_from_bearing(npc_yaw_degrees: Any, bearing_degrees: Any) -> float:
    """协议约定 brg 右正，因此必须相加后回绕。"""
    return wrap_yaw(_finite(npc_yaw_degrees, "npc_yaw") + _finite(bearing_degrees, "bearing"))


def encode_yaw_q14(degrees: Any) -> int:
    return _round_half_up(wrap_yaw(degrees) / 360.0 * 16384.0) % 16384


def decode_yaw_q14(value: Any) -> float:
    return _integer(value, "yaw_q14", 0, 16383) * 360.0 / 16384.0


def encode_speed_q7(speed_mps: Any, maximum_mps: Any) -> int:
    speed = _finite(speed_mps, "speed_mps")
    maximum = _finite(maximum_mps, "maximum_mps")
    if maximum <= 0.0 or not 0.0 <= speed <= maximum:
        raise YuiProtocolError("speed_mps 必须位于 0..maximum_mps")
    return _round_half_up(speed / maximum * 127.0)


def decode_speed_q7(value: Any, maximum_mps: Any) -> float:
    maximum = _finite(maximum_mps, "maximum_mps")
    if maximum <= 0.0:
        raise YuiProtocolError("maximum_mps 必须大于 0")
    return _integer(value, "speed_q7", 0, 127) * maximum / 127.0


def encode_centered_q7(value: Any, minimum: Any, maximum: Any) -> int:
    numeric = _finite(value, "value")
    low = _finite(minimum, "minimum")
    high = _finite(maximum, "maximum")
    if not low < 0.0 < high or not low <= numeric <= high:
        raise YuiProtocolError("centered_q7 要求 minimum < 0 < maximum 且 value 在范围内")
    if numeric <= 0.0:
        return _round_half_up(64.0 + numeric / abs(low) * 64.0)
    return 64 + _round_half_up(numeric / high * 63.0)


def decode_centered_q7(value: Any, minimum: Any, maximum: Any) -> float:
    quantized = _integer(value, "centered_q7", 0, 127)
    low = _finite(minimum, "minimum")
    high = _finite(maximum, "maximum")
    if not low < 0.0 < high:
        raise YuiProtocolError("centered_q7 要求 minimum < 0 < maximum")
    if quantized <= 64:
        return (quantized - 64) / 64.0 * abs(low)
    return (quantized - 64) / 63.0 * high


def encode_unsigned_q7(value: Any, minimum: Any, maximum: Any) -> int:
    numeric = _finite(value, "value")
    low = _finite(minimum, "minimum")
    high = _finite(maximum, "maximum")
    if not low < high or not low <= numeric <= high:
        raise YuiProtocolError("unsigned_q7 的 value 必须位于有效范围")
    return _round_half_up((numeric - low) / (high - low) * 127.0)


def decode_unsigned_q7(value: Any, minimum: Any, maximum: Any) -> float:
    quantized = _integer(value, "unsigned_q7", 0, 127)
    low = _finite(minimum, "minimum")
    high = _finite(maximum, "maximum")
    if not low < high:
        raise YuiProtocolError("unsigned_q7 要求 minimum < maximum")
    return low + quantized / 127.0 * (high - low)


def encode_upper_body_frame(sequence: Any, values: Mapping[str, Any]) -> tuple[MidiEvent, ...]:
    sequence_value = _integer(sequence, "stream_sequence", 1, 127)
    encoded: list[MidiEvent] = []
    for key, cc, minimum, maximum, encoding in UPPER_BODY_PARAMETERS:
        if key not in values:
            raise YuiProtocolError(f"上身帧缺少 {key}")
        quantized = (
            encode_centered_q7(values[key], minimum, maximum)
            if encoding == "centered"
            else encode_unsigned_q7(values[key], minimum, maximum)
        )
        encoded.append(MidiEvent("cc", UPPER_BODY_CHANNEL, cc, quantized))
    encoded.append(MidiEvent("note_on", UPPER_BODY_CHANNEL, UPPER_BODY_COMMIT_NOTE, sequence_value))
    return tuple(encoded)


def decode_upper_body_frame(values: Sequence[Any]) -> dict[str, float]:
    if len(values) != len(UPPER_BODY_PARAMETERS):
        raise YuiProtocolError("上身量化帧必须包含 8 个值")
    decoded: dict[str, float] = {}
    for raw, (key, _cc, minimum, maximum, encoding) in zip(values, UPPER_BODY_PARAMETERS, strict=True):
        decoded[key] = (
            decode_centered_q7(raw, minimum, maximum)
            if encoding == "centered"
            else decode_unsigned_q7(raw, minimum, maximum)
        )
    return decoded


def pack_midi_7bit(raw: bytes | bytearray | memoryview) -> bytes:
    source = bytes(raw)
    packed = bytearray()
    for offset in range(0, len(source), 7):
        group = source[offset:offset + 7]
        msb = 0
        for index, byte in enumerate(group):
            msb |= ((byte >> 7) & 1) << index
        packed.append(msb)
        packed.extend(byte & 0x7F for byte in group)
    return bytes(packed)


def unpack_midi_7bit(packed: bytes | bytearray | memoryview, raw_length: Any) -> bytes:
    expected_length = _integer(raw_length, "raw_length", 0, TEXT_MAX_UTF8_BYTES)
    source = bytes(packed)
    packed_length = expected_length + math.ceil(expected_length / 7)
    if len(source) != packed_length:
        raise YuiProtocolError(f"packed 长度应为 {packed_length}，实际为 {len(source)}")
    raw = bytearray()
    cursor = 0
    while len(raw) < expected_length:
        msb = source[cursor]
        cursor += 1
        group_length = min(7, expected_length - len(raw))
        for index in range(group_length):
            low = source[cursor]
            cursor += 1
            if low > 0x7F:
                raise YuiProtocolError("packed 载荷包含非 7-bit 值")
            raw.append(low | (((msb >> index) & 1) << 7))
    return bytes(raw)


def encode_text_transaction(
    text: str,
    *,
    transfer_sequence: Any,
    begin_sequence: Any,
    commit_sequence: Any,
    display_seconds: Any = 0,
) -> TextTransaction:
    if not isinstance(text, str):
        raise YuiProtocolError("text 必须是字符串")
    raw = text.encode("utf-8", errors="strict")
    if not 1 <= len(raw) <= TEXT_MAX_UTF8_BYTES:
        raise YuiProtocolError("text 的 UTF-8 长度必须是 1..384 字节")
    transfer = _integer(transfer_sequence, "transfer_sequence", 1, 16383)
    duration = _integer(display_seconds, "display_seconds", 0, 127)
    crc = crc16_ccitt_false(raw)
    parameters = (
        transfer,
        len(raw),
        crc & 0x3FFF,
        (crc >> 14) & 0x03,
        duration,
        0,
    )
    begin = encode_command("TEXT_BEGIN", begin_sequence, parameters)
    commit = encode_command("TEXT_COMMIT", commit_sequence, parameters)
    payload = tuple(MidiEvent("cc", TEXT_CHANNEL, TEXT_PAYLOAD_CC, value) for value in pack_midi_7bit(raw))
    return TextTransaction(
        transfer_sequence=transfer,
        text=text,
        utf8_bytes=raw,
        crc16=f"{crc:04X}",
        begin=begin,
        payload=payload,
        commit=commit,
    )


def capability_bits(capabilities: Iterable[str]) -> int:
    result = 0
    for capability in capabilities:
        if capability not in CAPABILITY_BITS:
            raise YuiProtocolError(f"未知 capability: {capability}")
        result |= 1 << CAPABILITY_BITS[capability]
    return result


def _reject_json_constant(value: str) -> None:
    raise YuiLogDecodeError(f"日志包含非标准 JSON 数值 {value}")


def parse_neko_log_line(line: str | bytes, *, allow_compatible_minor: bool = True) -> dict[str, Any] | None:
    """解析一行 VRChat output log；无标记时返回 ``None``。"""
    if isinstance(line, bytes):
        try:
            text = line.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise YuiLogDecodeError("日志行不是严格 UTF-8") from exc
    elif isinstance(line, str):
        text = line
    else:
        raise TypeError("line 必须是 str 或 bytes")

    marker_index = text.find(NEKO_LOG_MARKER)
    if marker_index < 0:
        return None
    payload = text[marker_index + len(NEKO_LOG_MARKER):].rstrip("\r\n")
    if not payload:
        raise YuiLogDecodeError("[NEKO] 后没有 JSON")
    if len(payload.encode("utf-8")) > MAX_LOG_JSON_UTF8_BYTES:
        raise YuiLogDecodeError("业务 JSON 超过 950 UTF-8 字节")
    try:
        decoded = json.loads(payload, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, YuiLogDecodeError) as exc:
        if isinstance(exc, YuiLogDecodeError):
            raise
        raise YuiLogDecodeError(f"业务 JSON 无法解析: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise YuiLogDecodeError("业务 JSON 顶层必须是对象")

    required = ("v", "spec", "session", "world_id", "npc", "log_seq", "t", "type")
    missing = [key for key in required if key not in decoded]
    if missing:
        raise YuiLogDecodeError(f"业务 JSON 缺少公共字段: {', '.join(missing)}")
    if decoded["v"] != WIRE_VERSION:
        raise YuiLogDecodeError(f"不支持的 wire 版本: {decoded['v']!r}")
    spec = decoded["spec"]
    if not isinstance(spec, str):
        raise YuiLogDecodeError("spec 必须是字符串")
    if allow_compatible_minor:
        if spec not in SUPPORTED_SPEC_VERSIONS:
            raise YuiLogDecodeError(f"不兼容的 spec 版本: {spec}")
    elif spec != SPEC_VERSION:
        raise YuiLogDecodeError(f"期望 spec {SPEC_VERSION}，实际为 {spec}")
    _integer(decoded["session"], "session", 0, SESSION_MAX)
    _integer(decoded["log_seq"], "log_seq", 1, 2_147_483_647)
    if not isinstance(decoded["world_id"], str) or not 1 <= len(decoded["world_id"]) <= 64:
        raise YuiLogDecodeError("world_id 必须是 1..64 字符的字符串")
    if decoded["npc"] != NPC_ID:
        raise YuiLogDecodeError(f"npc 必须是 {NPC_ID!r}")
    timestamp = _finite(decoded["t"], "t")
    if timestamp < 0.0:
        raise YuiLogDecodeError("t 不得为负数")
    if not isinstance(decoded["type"], str) or not decoded["type"]:
        raise YuiLogDecodeError("type 必须是非空字符串")
    return decoded


def _constants_path(spec_version: str) -> Path:
    if spec_version not in SUPPORTED_SPEC_VERSIONS:
        raise YuiProtocolError(f"不支持的冻结常量版本: {spec_version}")
    return (
        Path(__file__).resolve().parents[1]
        / "Docs"
        / "Protocols"
        / f"YUI_NPC_ProtocolConstants_v{spec_version}.json"
    )


def load_frozen_constants(
    path: str | Path | None = None,
    *,
    spec_version: str = SPEC_VERSION,
) -> dict[str, Any]:
    """读取冻结机器事实源；扩展版本按 ``base_spec`` 逐级只增不改地合并。"""
    source = Path(path) if path is not None else _constants_path(spec_version)
    with source.open("r", encoding="utf-8") as handle:
        decoded = json.load(handle, parse_constant=_reject_json_constant)
    if not isinstance(decoded, dict):
        raise YuiProtocolError("常量 JSON 顶层必须是对象")
    actual_spec = decoded.get("spec")
    if decoded.get("wire_version") != WIRE_VERSION or actual_spec not in SUPPORTED_SPEC_VERSIONS:
        raise YuiProtocolError("常量 JSON 版本与 Python 实现不一致")
    if path is None and actual_spec != spec_version:
        raise YuiProtocolError(f"期望常量 spec {spec_version}，实际为 {actual_spec}")
    if decoded.get("status") != "frozen":
        raise YuiProtocolError("只允许加载已冻结的协议常量")
    if actual_spec != SPEC_VERSION and "commands_add" in decoded:
        base_spec = decoded.get("base_spec")
        if base_spec not in SUPPORTED_SPEC_VERSIONS or base_spec == actual_spec:
            raise YuiProtocolError(f"v{actual_spec} base_spec 无效: {base_spec!r}")
        base = load_frozen_constants(spec_version=str(base_spec))
        merged = deepcopy(base)
        merged["document"] = decoded.get("document", f"YUI NPC Protocol Constants v{actual_spec}")
        merged["spec"] = actual_spec
        merged["released_at"] = decoded.get("released_at")
        for addition_key, target_key in (
            ("commands_add", "commands"),
            ("command_states_add", "command_state_permissions"),
            ("capabilities_add", "capabilities"),
            ("capability_contracts_add", "capability_contracts"),
        ):
            additions = decoded.get(addition_key, {})
            if not isinstance(additions, Mapping):
                raise YuiProtocolError(f"v{actual_spec} {addition_key} 必须是对象")
            target = merged.get(target_key)
            if not isinstance(target, dict):
                raise YuiProtocolError(f"v{base_spec} 基线缺少 {target_key}")
            for key, value in additions.items():
                if key in target:
                    raise YuiProtocolError(f"v{actual_spec} 不得覆盖 v{base_spec} {target_key}.{key}")
                target[key] = deepcopy(value)

        catalog_additions = decoded.get("catalog_kinds_add", {})
        if catalog_additions:
            if not isinstance(catalog_additions, Mapping):
                raise YuiProtocolError(f"v{actual_spec} catalog_kinds_add 必须是对象")
            target_catalogs = merged.setdefault("catalog_kinds_add", {})
            for key, value in catalog_additions.items():
                if key in target_catalogs:
                    raise YuiProtocolError(f"v{actual_spec} 不得覆盖目录类型 {key}")
                target_catalogs[key] = deepcopy(value)

        if "behavior_graph" in decoded:
            if "behavior_graph" in merged:
                raise YuiProtocolError(f"v{actual_spec} 不得覆盖 v{base_spec} behavior_graph")
            if not isinstance(decoded["behavior_graph"], Mapping):
                raise YuiProtocolError(f"v{actual_spec} behavior_graph 必须是对象")
            merged["behavior_graph"] = deepcopy(decoded["behavior_graph"])

        catalog_fields = decoded.get("catalog_fields_add", {})
        if catalog_fields:
            if not isinstance(catalog_fields, Mapping):
                raise YuiProtocolError(f"v{actual_spec} catalog_fields_add 必须是对象")
            target_catalogs = merged.setdefault("catalog_kinds_add", {})
            for kind, raw_fields in catalog_fields.items():
                target_fields = target_catalogs.get(kind)
                if not isinstance(target_fields, list) or not isinstance(raw_fields, list):
                    raise YuiProtocolError(f"v{actual_spec} 目录字段扩展 {kind} 无效")
                for field in raw_fields:
                    if field in target_fields:
                        raise YuiProtocolError(f"v{actual_spec} 目录字段重复: {kind}.{field}")
                    target_fields.append(deepcopy(field))

        behavior_additions = decoded.get("behavior_graph_add", {})
        if behavior_additions:
            if not isinstance(behavior_additions, Mapping):
                raise YuiProtocolError(f"v{actual_spec} behavior_graph_add 必须是对象")
            target_graph = merged.get("behavior_graph")
            if not isinstance(target_graph, dict):
                raise YuiProtocolError(f"v{base_spec} 基线缺少 behavior_graph")
            for key, raw_values in behavior_additions.items():
                target_values = target_graph.get(key)
                if not isinstance(target_values, list) or not isinstance(raw_values, list):
                    raise YuiProtocolError(f"v{actual_spec} 行为图字段扩展 {key} 无效")
                for value in raw_values:
                    if value in target_values:
                        raise YuiProtocolError(f"v{actual_spec} 行为图值重复: {key}.{value}")
                    target_values.append(deepcopy(value))

        tools_additions = decoded.get("agent_tools_add", [])
        if tools_additions:
            if not isinstance(tools_additions, list):
                raise YuiProtocolError(f"v{actual_spec} agent_tools_add 必须是数组")
            target_tools = merged.setdefault("agent_tools_add", [])
            for tool in tools_additions:
                if tool in target_tools:
                    raise YuiProtocolError(f"v{actual_spec} Agent 工具重复: {tool}")
                target_tools.append(deepcopy(tool))

        for key in (
            "base_spec",
            "base_file",
            "catalog_fields_add",
            "behavior_graph_add",
            "state_projection_add",
            "event_types_add",
            "region_volume_rules",
        ):
            if key in decoded:
                merged[key] = deepcopy(decoded[key])
        return merged
    return decoded


__all__ = [
    "CAPABILITY_BITS",
    "COMMAND_IDS",
    "COMMAND_NAMES",
    "CURRENT_SPEC_VERSION",
    "CommandFrame",
    "ERROR_CODES",
    "ESTOP_COMMAND_ID",
    "MIDI_PORT_NAME",
    "MidiEvent",
    "NEKO_LOG_MARKER",
    "NPC_ID",
    "SPEC_VERSION",
    "SUPPORTED_SPEC_VERSIONS",
    "TextTransaction",
    "UPPER_BODY_NEUTRAL_Q",
    "UPPER_BODY_PARAMETERS",
    "WIRE_VERSION",
    "YuiLogDecodeError",
    "YuiProtocolError",
    "absolute_yaw_from_bearing",
    "capability_bits",
    "command_request_hash",
    "crc16_ccitt_false",
    "decode_centered_q7",
    "decode_position_q14",
    "decode_speed_q7",
    "decode_unsigned_q7",
    "decode_upper_body_frame",
    "decode_yaw_q14",
    "encode_centered_q7",
    "encode_command",
    "encode_position_q14",
    "encode_speed_q7",
    "encode_text_transaction",
    "encode_unsigned_q7",
    "encode_upper_body_frame",
    "encode_yaw_q14",
    "join_u14",
    "load_frozen_constants",
    "normalize_bearing",
    "pack_midi_7bit",
    "parse_neko_log_line",
    "split_u14",
    "unpack_midi_7bit",
    "preload_command_constraints",
    "validate_command_parameters",
    "wrap_yaw",
]
