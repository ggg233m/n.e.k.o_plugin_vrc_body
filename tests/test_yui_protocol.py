"""YUI 插件协议层测试。"""

from __future__ import annotations

from collections import defaultdict, deque
import ast
import json
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.yui_protocol import (
    CAPABILITY_BITS,
    COMMAND_IDS,
    ERROR_CODES,
    UPPER_BODY_NEUTRAL_Q,
    YuiLogDecodeError,
    YuiProtocolError,
    absolute_yaw_from_bearing,
    command_request_hash,
    crc16_ccitt_false,
    decode_position_q14,
    decode_speed_q7,
    decode_upper_body_frame,
    decode_yaw_q14,
    encode_command,
    encode_text_transaction,
    encode_yaw_q14,
    join_u14,
    load_frozen_constants,
    pack_midi_7bit,
    parse_neko_log_line,
    unpack_midi_7bit,
)


ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "Docs" / "Protocols" / "YUI_NPC_TestVectors_v1.1.json"


def _load_vectors() -> dict:
    with VECTORS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class YuiProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.constants = load_frozen_constants()
        cls.constants_v12 = load_frozen_constants(spec_version="1.2")
        cls.vector_document = _load_vectors()
        cls.vectors = {
            vector["id"]: vector
            for vector in cls.vector_document["vectors"]
        }

    def test_python_constant_tables_match_frozen_json(self) -> None:
        self.assertEqual(
            {name: item["id"] for name, item in self.constants_v12["commands"].items()},
            COMMAND_IDS,
        )
        self.assertEqual(self.constants_v12["capabilities"], CAPABILITY_BITS)
        self.assertEqual(self.constants["error_codes"], ERROR_CODES)
        self.assertEqual(self.constants["reserved_error_codes"], [2])
        self.assertNotIn("GOTO_ANCHOR", self.constants["commands"])
        self.assertNotIn("world_map", self.constants["capabilities"])

    def test_yui_modules_are_isolated_from_anydance_and_yolo_runtime(self) -> None:
        """YUI 只能依赖标准库、可选 MIDI 库和同组 yui 模块。"""
        allowed_relative_modules = {
            "behavior_plan",
            "yui_protocol",
            "yui_session",
            "yui_transport",
        }
        for path in sorted((ROOT / "runtime").glob("yui_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    self.assertIn(node.module, allowed_relative_modules, path.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name.startswith(("backend", "neko_anyadance_body")),
                            f"{path.name} 不得导入 AnyDance/YOLO 运行栈",
                        )

    def test_all_vector_command_hashes_match_expected_ack(self) -> None:
        comparisons = 0
        for vector in self.vector_document["vectors"]:
            expected: dict[tuple[int, int], deque[str]] = defaultdict(deque)
            for event in vector.get("expected_logs", []):
                if event.get("type") == "npc.ack":
                    expected[(event["cmd_id"], event["seq"])].append(event["request_hash"])

            registers: dict[int, int] = {}
            for event in vector.get("raw_midi_events", []):
                if event["type"] == "cc" and event["channel"] == 0 and 20 <= event["number"] <= 28:
                    registers[event["number"]] = event["value"]
                    continue
                if event["type"] != "note_on":
                    continue
                command_id = event["number"]
                sequence = event["velocity"]
                if command_id == 0x7F:
                    parameters = (0, 0, 0, 0, 0, 0)
                elif event["channel"] == 0 and sequence != 0 and all(cc in registers for cc in range(20, 29)):
                    parameters = (
                        join_u14(registers[20], registers[21]),
                        join_u14(registers[22], registers[23]),
                        join_u14(registers[24], registers[25]),
                        registers[26],
                        registers[27],
                        registers[28],
                    )
                    registers.clear()
                else:
                    continue
                candidates = expected[(command_id, sequence)]
                if not candidates:
                    continue
                actual = command_request_hash(command_id, sequence, parameters)
                wanted = candidates.popleft()
                self.assertEqual(actual, wanted, vector["id"])
                comparisons += 1
        self.assertGreaterEqual(comparisons, 70)

    def test_goto_quantization_vectors_decode_exactly(self) -> None:
        fixture = self.vector_document["fixture"]
        wire = fixture["wire_bounds"]
        for vector_id in ("goto_minimum", "goto_midpoint", "goto_maximum", "goto_without_yaw"):
            vector = self.vectors[vector_id]
            decoded = vector["udon_decode"]
            p = decoded["P"]
            self.assertAlmostEqual(decode_position_q14(p[0], wire[0], wire[3]), decoded["target_x"])
            self.assertAlmostEqual(decode_position_q14(p[1], wire[2], wire[5]), decoded["target_z"])
            self.assertAlmostEqual(decode_speed_q7(p[3], fixture["max_speed"]), decoded["speed"])
            if decoded["has_yaw"]:
                self.assertAlmostEqual(decode_yaw_q14(p[2]), decoded["yaw"])
            else:
                self.assertEqual(p[2], 0)
                self.assertIsNone(decoded["yaw"])

    def test_bearing_right_positive_vector_uses_addition_and_wrap(self) -> None:
        vector = self.vectors["bearing_right_positive_to_turn_q14"]
        semantic = vector["semantic_input"]
        yaw = absolute_yaw_from_bearing(
            semantic["observation"]["npc_yaw"],
            semantic["observation"]["target_brg"],
        )
        self.assertEqual(yaw, 10.0)
        self.assertEqual(encode_yaw_q14(yaw), 455)
        self.assertEqual(vector["udon_decode"]["P"][0], 455)

    def test_upper_body_boundary_vectors(self) -> None:
        cases = {
            "upper_body_minimum_frame": (
                -60.0, -35.0, -35.0, -20.0, 0.0, -120.0, 0.0, -120.0,
            ),
            "upper_body_neutral_frame": (0.0,) * 8,
            "upper_body_maximum_frame": (
                60.0, 35.0, 35.0, 20.0, 160.0, 120.0, 160.0, 120.0,
            ),
        }
        for vector_id, expected in cases.items():
            vector = self.vectors[vector_id]
            decoded = decode_upper_body_frame(vector["semantic_input"]["q"])
            self.assertEqual(tuple(decoded.values()), expected)
        self.assertEqual(
            tuple(self.vectors["upper_body_neutral_frame"]["semantic_input"]["q"]),
            UPPER_BODY_NEUTRAL_Q,
        )

    def test_utf8_transaction_matches_vector_byte_for_byte(self) -> None:
        vector = self.vectors["utf8_text_nihao_yui_wave"]
        semantic = vector["semantic_input"]
        decoded = vector["udon_decode"]
        transaction = encode_text_transaction(
            semantic["text"],
            transfer_sequence=semantic["transfer_seq"],
            begin_sequence=decoded["begin"]["seq"],
            commit_sequence=decoded["commit"]["seq"],
            display_seconds=semantic["display_seconds"],
        )
        self.assertEqual(transaction.utf8_bytes.hex(" "), semantic["utf8_hex"])
        self.assertEqual(transaction.crc16, semantic["crc16"])
        self.assertEqual(transaction.begin.parameters, tuple(decoded["begin"]["P"]))
        self.assertEqual(transaction.begin.request_hash, decoded["begin"]["request_hash"])
        self.assertEqual(transaction.commit.parameters, tuple(decoded["commit"]["P"]))
        self.assertEqual(transaction.commit.request_hash, decoded["commit"]["request_hash"])
        packed = bytes(event.value for event in transaction.payload)
        self.assertEqual(list(packed), decoded["packed_values"])
        self.assertEqual(pack_midi_7bit(transaction.utf8_bytes), packed)
        self.assertEqual(unpack_midi_7bit(packed, len(transaction.utf8_bytes)), transaction.utf8_bytes)
        self.assertEqual(crc16_ccitt_false(transaction.utf8_bytes), int(semantic["crc16"], 16))

    def test_command_encoder_writes_all_registers_in_order(self) -> None:
        vector = self.vectors["goto_midpoint"]
        decoded = vector["udon_decode"]
        frame = encode_command(decoded["cmd_id"], decoded["seq"], decoded["P"])
        actual = [event.as_dict(at_ms=index) for index, event in enumerate(frame.events)]
        self.assertEqual(actual, vector["raw_midi_events"])

    def test_command_encoder_enforces_frozen_per_command_constraints(self) -> None:
        invalid_cases = (
            ("SET_TARGET", (0, 0, 0, 64, 0, 0)),
            ("PLAY_ANIM", (1, 0, 0, 127, 0, 0)),
            ("PLAY_ANIM", (0, 0, 0, 1, 0, 0)),
            ("PLAY_ANIM", (1, 0, 0, 1, 2, 0)),
            ("STOP", (5, 0, 0, 0, 0, 0)),
        )
        for command, parameters in invalid_cases:
            with self.subTest(command=command, parameters=parameters):
                with self.assertRaises(YuiProtocolError):
                    encode_command(command, 1, parameters)

    def test_command_encoder_enforces_semantic_and_combined_ranges(self) -> None:
        with self.assertRaises(YuiProtocolError):
            encode_command("TEXT_BEGIN", 1, (1, 0, 0, 0, 0, 0))
        with self.assertRaises(YuiProtocolError):
            encode_command("TEXT_COMMIT", 1, (1, 385, 0, 0, 0, 0))
        with self.assertRaises(YuiProtocolError):
            encode_command("DISCOVER", 1, (0, 0, 1, 0, 0, 0))

    def test_every_expected_log_round_trips_through_parser(self) -> None:
        parsed = 0
        for vector in self.vector_document["vectors"]:
            for event in vector.get("expected_logs", []):
                payload = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                line = f"2026.08.29 00:00:00 Log - [NEKO]{payload}\n"
                self.assertEqual(parse_neko_log_line(line), event, vector["id"])
                parsed += 1
        self.assertGreater(parsed, 150)

    def test_log_parser_ignores_unmarked_lines_and_rejects_invalid_json(self) -> None:
        self.assertIsNone(parse_neko_log_line("ordinary VRChat log"))
        with self.assertRaises(YuiLogDecodeError):
            parse_neko_log_line('[NEKO]{"v":1,"spec":"1.1","session":0,"world_id":"x","npc":"yui","log_seq":1,"t":NaN,"type":"sys.boot"}')
        with self.assertRaises(YuiLogDecodeError):
            parse_neko_log_line("[NEKO]{}")


if __name__ == "__main__":
    unittest.main()
