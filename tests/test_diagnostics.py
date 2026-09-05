"""验证诊断日志脱敏、限频和日志失败时的业务隔离。"""

import json
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.diagnostics import PipelineDiagnostics


class PipelineDiagnosticsTests(unittest.TestCase):
    def test_payload_and_exception_body_never_enter_log_or_status(self):
        logger = Mock()
        diag = PipelineDiagnostics(logger)
        diag.emit(
            "chat.rejected", session=0, event_session=61,
            player_slot={"name": "隐私姓名"}, text="私密正文",
            Authorization="Bearer secret", error="request failed secret /v1",
        )
        output = json.dumps(diag.snapshot(), ensure_ascii=False) + str(logger.mock_calls)
        for private in ("隐私姓名", "私密正文", "secret", "Authorization", "/v1"):
            self.assertNotIn(private, output)
        self.assertEqual(diag.snapshot()[0]["error"], "redacted")
        self.assertEqual(diag.snapshot()[0]["event_session"], 61)

    def test_repeated_state_throttled_and_recovery_emitted_immediately(self):
        now = [0.0]
        diag = PipelineDiagnostics(Mock(), clock=lambda: now[0])
        for _ in range(20):
            diag.emit("connection.result", deduplicate=True, error="ack_timeout")
        self.assertEqual(len(diag.snapshot()), 1)
        now[0] = 31
        diag.emit("connection.result", deduplicate=True, error="ack_timeout")
        self.assertEqual(diag.snapshot()[-1]["suppressed"], 19)
        diag.emit("connection.result", deduplicate=True, status="succeeded")
        self.assertEqual(len(diag.snapshot()), 3)

    def test_logger_failure_and_bounded_history(self):
        logger = Mock()
        logger.info.side_effect = RuntimeError("日志写入失败")
        diag = PipelineDiagnostics(logger)
        for index in range(100):
            diag.emit("chat.received", submit_seq=index)
        self.assertEqual(len(diag.snapshot()), 64)
        self.assertEqual(diag.snapshot()[-1]["submit_seq"], 99)

