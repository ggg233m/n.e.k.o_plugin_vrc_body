"""YUI v1.2 冻结扩展向量与版本兼容测试。"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.yui_protocol import (
    YuiProtocolError,
    encode_command,
    load_frozen_constants,
    parse_neko_log_line,
)


ROOT = Path(__file__).resolve().parents[1]


class ProtocolV12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "Docs" / "Protocols" / "YUI_NPC_TestVectors_v1.2.json").open("r", encoding="utf-8") as handle:
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

    def test_v12_extends_without_overwriting_v11(self) -> None:
        v11 = load_frozen_constants()
        v12 = load_frozen_constants(spec_version="1.2")
        for name, command in v11["commands"].items():
            self.assertEqual(v12["commands"][name], command)
        for name, bit in v11["capabilities"].items():
            self.assertEqual(v12["capabilities"][name], bit)
        self.assertEqual(v12["commands"]["GOTO_ANCHOR"]["id"], 23)
        self.assertEqual(v12["commands"]["ORBIT_ENTITY"]["id"], 24)

    def test_parser_accepts_only_supported_minor_versions(self) -> None:
        common = {
            "v": 1, "session": 1, "world_id": "wrld_test", "npc": "yui",
            "log_seq": 1, "t": 0.0, "type": "sys.boot",
        }
        for spec in ("1.1", "1.2"):
            line = "[NEKO]" + json.dumps({**common, "spec": spec}, separators=(",", ":"))
            self.assertEqual(parse_neko_log_line(line)["spec"], spec)
        bad = "[NEKO]" + json.dumps({**common, "spec": "1.9"}, separators=(",", ":"))
        with self.assertRaisesRegex(Exception, "不兼容"):
            parse_neko_log_line(bad)


if __name__ == "__main__":
    unittest.main()
