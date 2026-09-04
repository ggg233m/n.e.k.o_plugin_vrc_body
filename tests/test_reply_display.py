"""主 LLM 回答头顶显示桥接测试。"""

from __future__ import annotations

from collections import deque
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.chat_context import ChatContextUpdate
from yui_npc_controller.runtime.config import YuiChatBridgeConfig
from yui_npc_controller.runtime.reply_display import MainReplyDisplayBridge


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeProvider:
    def __init__(self) -> None:
        self.updates = deque()
        self.turns: list[dict[str, str]] = []
        self.character = "然然"

    def poll(self):
        return self.updates.popleft()

    def context(self):
        return {"source": "recent_file", "untrusted": True, "turns": self.turns}

    def status(self):
        return {
            "file_state": "available",
            "turn_count": len(self.turns),
            "current_character": self.character,
        }


class MainReplyDisplayBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.provider = FakeProvider()
        self.sent: list[tuple[str, int]] = []
        self.bridge = MainReplyDisplayBridge(
            self.provider,  # type: ignore[arg-type]
            YuiChatBridgeConfig(enabled=True, display_seconds=15, max_pages=4),
            self._display,
            clock=self.clock,
        )

    def _display(self, text: str, seconds: int):
        self.sent.append((text, seconds))
        return {"status": "succeeded", "error": None}

    def test_first_snapshot_is_baseline_and_new_reply_is_displayed(self) -> None:
        self.provider.turns = [{"user": "旧问题", "assistant": "旧回答"}]
        self.provider.updates.append(ChatContextUpdate(True, False, "r1"))
        self.bridge.tick()
        self.assertEqual(self.sent, [])

        self.provider.turns = [{"user": "新问题", "assistant": "新回答"}]
        self.provider.updates.append(ChatContextUpdate(True, False, "r2"))
        self.bridge.tick()
        self.assertEqual(self.sent, [("新回答", 15)])
        status = self.bridge.status()
        self.assertEqual(status["displayed_replies"], 1)
        self.assertNotIn("新回答", str(status))

    def test_display_duration_keeps_short_text_and_extends_long_text(self) -> None:
        self.assertEqual(MainReplyDisplayBridge.display_duration("短回答", 15), 15)
        self.assertEqual(MainReplyDisplayBridge.display_duration("猫" * 72, 15), 18)
        self.assertEqual(MainReplyDisplayBridge.display_duration("猫" * 500, 15), 30)

    def test_utf8_pages_stay_inside_wire_limit_and_are_bounded(self) -> None:
        pages = MainReplyDisplayBridge.paginate("猫" * 1000, 4)
        self.assertEqual(len(pages), 4)
        self.assertTrue(pages[-1].endswith("…"))
        self.assertTrue(all(len(page.encode("utf-8")) <= 384 for page in pages))

    def test_failed_send_is_retried_without_losing_page(self) -> None:
        attempts = []

        def fail_once(text: str, seconds: int):
            attempts.append((text, seconds))
            if len(attempts) == 1:
                return {"status": "failed", "error": "not_connected"}
            return {"status": "succeeded", "error": None}

        bridge = MainReplyDisplayBridge(
            self.provider,  # type: ignore[arg-type]
            YuiChatBridgeConfig(enabled=True),
            fail_once,
            clock=self.clock,
        )
        self.provider.turns = [{"user": "旧", "assistant": "旧"}]
        self.provider.updates.append(ChatContextUpdate(True, False, "r1"))
        bridge.tick()
        self.provider.turns = [{"user": "新", "assistant": "待显示"}]
        self.provider.updates.append(ChatContextUpdate(True, False, "r2"))
        bridge.tick()
        self.assertEqual(bridge.status()["queued_pages"], 1)
        self.clock.advance(1.0)
        self.provider.updates.append(ChatContextUpdate(False, False, "r2"))
        bridge.tick()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(bridge.status()["queued_pages"], 0)

    def test_character_switch_does_not_replay_history(self) -> None:
        self.provider.turns = [{"user": "甲", "assistant": "回答甲"}]
        self.provider.updates.append(ChatContextUpdate(True, False, "r1"))
        self.bridge.tick()
        self.provider.turns = [{"user": "乙", "assistant": "角色乙旧回答"}]
        self.provider.updates.append(ChatContextUpdate(True, True, "r2"))
        self.bridge.tick()
        self.assertEqual(self.sent, [])

    def test_existing_proactive_reply_bus_supplies_world_chat_reply(self) -> None:
        self.clock.value = 100.0
        records: list[dict[str, object]] = [{
            "source": "proactive",
            "turn_type": "proactive_reply",
            "conversation_id": "old",
            "lanlan_name": "然然",
            "timestamp": 99.0,
            "content": "旧主动回答",
            "message_count": 2,
        }]
        bridge = MainReplyDisplayBridge(
            self.provider,  # type: ignore[arg-type]
            YuiChatBridgeConfig(enabled=True, display_seconds=15),
            self._display,
            conversation_fetcher=lambda: list(records),
            clock=self.clock,
            wall_clock=self.clock,
        )
        self.provider.updates.append(ChatContextUpdate(True, False, "r1"))
        bridge.tick()
        self.assertEqual(self.sent, [])

        records.append({
            "source": "proactive",
            "turn_type": "proactive_reply",
            "conversation_id": "new",
            "lanlan_name": "然然",
            "timestamp": 101.0,
            "content": "世界输入对应回答",
            "message_count": 2,
        })
        self.clock.advance(1.0)
        self.provider.updates.append(ChatContextUpdate(False, False, "r1"))
        bridge.tick()

        self.assertEqual(self.sent, [("世界输入对应回答", 15)])
        status = bridge.status()
        self.assertEqual(status["displayed_replies"], 1)
        self.assertEqual(status["proactive_bus"]["records_seen"], 1)
        self.assertNotIn("世界输入对应回答", str(status))

    def test_proactive_bus_filters_character_and_non_reply_records(self) -> None:
        self.clock.value = 20.0
        records: list[dict[str, object]] = []
        bridge = MainReplyDisplayBridge(
            self.provider,  # type: ignore[arg-type]
            YuiChatBridgeConfig(enabled=True),
            self._display,
            conversation_fetcher=lambda: list(records),
            clock=self.clock,
            wall_clock=self.clock,
        )
        self.provider.updates.append(ChatContextUpdate(True, False, "r1"))
        bridge.tick()
        records.extend([
            {
                "source": "proactive",
                "turn_type": "proactive_instruction",
                "conversation_id": "instruction",
                "lanlan_name": "然然",
                "timestamp": 21.0,
                "content": "模型指令",
            },
            {
                "source": "proactive",
                "turn_type": "proactive_reply",
                "conversation_id": "other-character",
                "lanlan_name": "别的角色",
                "timestamp": 21.0,
                "content": "错误角色回答",
            },
        ])
        self.clock.advance(1.0)
        self.provider.updates.append(ChatContextUpdate(False, False, "r1"))
        bridge.tick()
        self.assertEqual(self.sent, [])

    def test_bus_query_failure_keeps_recent_file_fallback_working(self) -> None:
        def unavailable():
            raise TimeoutError("不得进入状态的细节")

        bridge = MainReplyDisplayBridge(
            self.provider,  # type: ignore[arg-type]
            YuiChatBridgeConfig(enabled=True),
            self._display,
            conversation_fetcher=unavailable,
            clock=self.clock,
            wall_clock=self.clock,
        )
        self.provider.turns = [{"user": "旧", "assistant": "旧"}]
        self.provider.updates.append(ChatContextUpdate(True, False, "r1"))
        bridge.tick()
        self.provider.turns = [{"user": "新", "assistant": "磁盘回答"}]
        self.provider.updates.append(ChatContextUpdate(True, False, "r2"))
        bridge.tick()

        self.assertEqual(self.sent, [("磁盘回答", 15)])
        status = bridge.status()
        self.assertEqual(status["proactive_bus"]["last_error"], "query_TimeoutError")
        self.assertNotIn("不得进入状态的细节", str(status))


if __name__ == "__main__":
    unittest.main()
