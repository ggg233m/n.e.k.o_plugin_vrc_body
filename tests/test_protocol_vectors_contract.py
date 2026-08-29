"""对冻结的 82 条向量执行结构、日志预算和禁用能力一致性验证。"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.yui_protocol import MAX_LOG_JSON_UTF8_BYTES, parse_neko_log_line


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "Docs" / "Protocols"


class FrozenVectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.constants = json.loads(
            (PROTOCOLS / "YUI_NPC_ProtocolConstants_v1.1.json").read_text(encoding="utf-8")
        )
        cls.document = json.loads(
            (PROTOCOLS / "YUI_NPC_TestVectors_v1.1.json").read_text(encoding="utf-8")
        )
        cls.vectors = cls.document["vectors"]

    def test_all_82_vectors_are_unique_and_executable_documents(self) -> None:
        self.assertEqual(len(self.vectors), 82)
        ids = [vector.get("id") for vector in self.vectors]
        self.assertEqual(len(set(ids)), 82)
        allowed_assertions = set(self.document["assertion_schema"]["kinds"])
        for vector in self.vectors:
            with self.subTest(vector=vector["id"]):
                self.assertIsInstance(vector.get("purpose"), str)
                self.assertIsInstance(vector.get("initial_state"), dict)
                self.assertIsInstance(vector.get("semantic_input"), dict)
                self.assertIsInstance(vector.get("raw_midi_events"), list)
                self.assertIsInstance(vector.get("expected_logs"), list)
                for assertion in vector.get("assertions", []):
                    self.assertIn(assertion.get("kind"), allowed_assertions)

    def test_all_expected_logs_round_trip_and_obey_wire_budget(self) -> None:
        for vector in self.vectors:
            window: deque[float] = deque()
            for event in vector.get("expected_logs", []):
                payload = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                with self.subTest(vector=vector["id"], event=event.get("type")):
                    self.assertLessEqual(len(payload.encode("utf-8")), MAX_LOG_JSON_UTF8_BYTES)
                    self.assertEqual(parse_neko_log_line(f"[NEKO]{payload}\n"), event)
                timestamp = float(event["t"])
                while window and timestamp - window[0] >= 1.0:
                    window.popleft()
                window.append(timestamp)
                self.assertLessEqual(len(window), 20, vector["id"])

    def test_all_17_disabled_capability_vectors_exist(self) -> None:
        expected = {
            contract["disabled_test_vector"]
            for contract in self.constants["capability_contracts"].values()
        }
        self.assertEqual(len(expected), 17)
        actual = {vector["id"] for vector in self.vectors}
        self.assertTrue(expected <= actual)

if __name__ == "__main__":
    unittest.main()
