"""YUI v1.3 冻结扩展向量与逐级继承测试。"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.yui_protocol import (
    CAPABILITY_BITS,
    COMMAND_IDS,
    YuiProtocolError,
    encode_command,
    load_frozen_constants,
    parse_neko_log_line,
)
from yui_npc_controller.runtime.yui_transport import YuiReliableTransport


ROOT = Path(__file__).resolve().parents[1]


class ProtocolV13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "Docs" / "Protocols" / "YUI_NPC_TestVectors_v1.3.json"
        with path.open("r", encoding="utf-8") as handle:
            cls.vectors = json.load(handle)

    def test_positive_vectors_are_exact(self) -> None:
        for vector in self.vectors["vectors"]:
            with self.subTest(vector=vector["id"]):
                frame = encode_command(vector["command"], vector["seq"], vector["parameters"])
                self.assertEqual(frame.request_hash, vector["request_hash"])
                self.assertEqual(list(frame.parameters), vector["parameters"])

    def test_negative_vectors_are_rejected_locally(self) -> None:
        for index, vector in enumerate(self.vectors["negative_vectors"], start=1):
            with self.subTest(vector=vector["id"]), self.assertRaisesRegex(YuiProtocolError, vector["error"]):
                encode_command(vector["command"], index, vector["parameters"])

    def test_v13_extends_v12_without_overwriting_it(self) -> None:
        v12 = load_frozen_constants(spec_version="1.2")
        v13 = load_frozen_constants(spec_version="1.3")
        for name, command in v12["commands"].items():
            self.assertEqual(v13["commands"][name], command)
        for name, bit in v12["capabilities"].items():
            self.assertEqual(v13["capabilities"][name], bit)
        self.assertEqual(v13["commands"]["MOVE_RELATIVE"]["id"], 25)
        self.assertEqual(v13["commands"]["EXPLORE_REGION"]["id"], 26)
        self.assertEqual(v13["capabilities"]["region_localization"], 19)
        self.assertEqual(v13["capabilities"]["local_navigation"], 20)
        self.assertIn("explorable", v13["catalog_kinds_add"]["region"])
        self.assertIn("move_relative", v13["behavior_graph"]["leaf_nodes"])
        self.assertIn("npc.move_relative", v13["agent_tools_add"])
        self.assertIn("npc.operation_failed", v13["event_types_add"])

    def test_public_numeric_tables_match_frozen_constants(self) -> None:
        self.assertEqual(COMMAND_IDS["MOVE_RELATIVE"], 0x19)
        self.assertEqual(COMMAND_IDS["EXPLORE_REGION"], 0x1A)
        self.assertEqual(CAPABILITY_BITS["region_localization"], 19)
        self.assertEqual(CAPABILITY_BITS["local_navigation"], 20)

    def test_parser_accepts_v13(self) -> None:
        payload = {
            "v": 1,
            "spec": "1.3",
            "session": 1,
            "world_id": "wrld_test",
            "npc": "yui",
            "log_seq": 1,
            "t": 0.0,
            "type": "sys.boot",
        }
        line = "[NEKO]" + json.dumps(payload, separators=(",", ":"))
        self.assertEqual(parse_neko_log_line(line)["spec"], "1.3")

    def test_v13_movement_commands_require_operation_terminal_evidence(self) -> None:
        relative = encode_command("MOVE_RELATIVE", 1, [1000, 0, 0, 0, 64, 3])
        explore = encode_command("EXPLORE_REGION", 2, [200, 0, 0, 0, 64, 0])
        self.assertEqual(YuiReliableTransport._operation_kind(relative), "move_relative")
        self.assertEqual(YuiReliableTransport._operation_kind(explore), "explore")


if __name__ == "__main__":
    unittest.main()
