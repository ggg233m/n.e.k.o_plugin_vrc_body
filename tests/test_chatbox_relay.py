"""对话轮 → VRChat 聊天框 转发的过滤、去重与计数。"""

from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.chatbox_relay import ChatboxRelay
from neko_anyadance_body.config import ChatboxRelayConfig


class _Record:
    """宿主 ConversationRecord 的最小替身。"""

    def __init__(
        self,
        *,
        content: str = "",
        turn_type: str = "proactive_reply",
        message_id: str | None = "turn-1",
        lanlan_name: str | None = "Lanlan",
        timestamp: float = 1001.0,
    ) -> None:
        self.content = content
        self.turn_type = turn_type
        self.message_id = message_id
        self.lanlan_name = lanlan_name
        self.timestamp = timestamp


class _Harness:
    """喂给转发器一批记录，并收集它真正发出去的东西。

    ``wall_clock`` 固定为 ``_WALL``，记录时间戳默认比它晚，模拟「转发器起来
    之后她才开口」这条正常路径。
    """

    _WALL = 1000.0

    def __init__(self, records: list[_Record], enabled: bool = True, **config_overrides: object) -> None:
        self.records = records
        self.sent: list[str] = []
        self.send_failures: list[str] = []
        self.source_calls: list[tuple[int, float | None]] = []
        options = {"poll_interval_s": 0.01, "enabled": enabled}
        options.update(config_overrides)
        config = ChatboxRelayConfig(**options)  # type: ignore[arg-type]
        self.relay = ChatboxRelay(
            config,
            source=self._source,
            send=self._send,
            clock=lambda: 0.0,
            wall_clock=lambda: self._WALL,
        )

    def _source(self, max_count: int, since_ts: float | None):
        self.source_calls.append((max_count, since_ts))
        return list(self.records)

    def _send(self, text: str):
        if text in self.send_failures:
            return False, "send rejected"
        self.sent.append(text)
        return True, None


class ChatboxRelayFilterTests(unittest.TestCase):
    def test_only_character_replies_are_forwarded(self) -> None:
        """指令承载的是用户原话，绝不能出现在聊天框里。"""
        harness = _Harness([
            _Record(content="用户说了一句私密的话", turn_type="proactive_instruction", message_id="i1"),
            _Record(content="她回答了一句", turn_type="proactive_reply", message_id="r1"),
        ])
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["她回答了一句"])

    def test_unknown_turn_type_is_skipped(self) -> None:
        harness = _Harness([_Record(content="未知类型", turn_type="unknown")])
        harness.relay._poll_once()
        self.assertEqual(harness.sent, [])

    def test_blank_content_is_skipped(self) -> None:
        harness = _Harness([
            _Record(content="   ", message_id="a"),
            _Record(content="\x00\x00", message_id="b"),
        ])
        harness.relay._poll_once()
        self.assertEqual(harness.sent, [])

    def test_whitespace_is_collapsed_to_one_line(self) -> None:
        """聊天框不渲染换行，多行会被折成一长条且照样吃掉字符额度。"""
        harness = _Harness([_Record(content="第一行\n第二行\t  第三行")])
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["第一行 第二行 第三行"])

    def test_record_without_id_is_skipped(self) -> None:
        """没有 id 就无法去重，发出去会在每个轮询周期重复刷屏。"""
        harness = _Harness([_Record(content="没有 id", message_id=None)])
        harness.relay._poll_once()
        self.assertEqual(harness.sent, [])
        self.assertEqual(harness.relay.snapshot()["skipped_count"], 1)


class ChatboxRelayDedupTests(unittest.TestCase):
    def test_the_same_record_is_forwarded_once_across_polls(self) -> None:
        harness = _Harness([_Record(content="只说一次", message_id="dup")])
        harness.relay._poll_once()
        harness.relay._poll_once()
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["只说一次"])

    def test_new_records_are_still_forwarded_after_a_repeat(self) -> None:
        harness = _Harness([_Record(content="第一句", message_id="a")])
        harness.relay._poll_once()
        harness.records = [_Record(content="第二句", message_id="b", timestamp=1001.0)]
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["第一句", "第二句"])

    def test_cursor_advances_so_the_next_poll_asks_for_newer_records(self) -> None:
        harness = _Harness([_Record(content="推进游标", timestamp=1234.0)])
        harness.relay._poll_once()
        harness.relay._poll_once()
        self.assertIsNone(harness.source_calls[0][1])
        self.assertEqual(harness.source_calls[1][1], 1234.0)


class ChatboxRelayTruncationTests(unittest.TestCase):
    def test_long_text_is_truncated_to_the_limit(self) -> None:
        harness = _Harness([_Record(content="字" * 200)], max_chars=144)
        harness.relay._poll_once()
        self.assertEqual(len(harness.sent[0]), 144)
        self.assertTrue(harness.sent[0].endswith("…"))
        self.assertEqual(harness.relay.snapshot()["truncated_count"], 1)

    def test_short_text_is_untouched(self) -> None:
        harness = _Harness([_Record(content="短句")], max_chars=144)
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["短句"])
        self.assertEqual(harness.relay.snapshot()["truncated_count"], 0)

    def test_speaker_prefix_is_opt_in(self) -> None:
        harness = _Harness([_Record(content="你好", lanlan_name="Lanlan")], include_speaker=True)
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["Lanlan: 你好"])

    def test_speaker_prefix_is_absent_by_default(self) -> None:
        harness = _Harness([_Record(content="你好", lanlan_name="Lanlan")])
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["你好"])


class ChatboxRelaySwitchTests(unittest.TestCase):
    def test_disabled_relay_sends_nothing(self) -> None:
        harness = _Harness([_Record(content="不该发出去")], enabled=False)
        harness.relay._poll_once()
        self.assertEqual(harness.sent, [])

    def test_disabling_at_runtime_stops_forwarding(self) -> None:
        harness = _Harness([_Record(content="先开着", message_id="a")])
        harness.relay._poll_once()
        harness.relay.set_enabled(False)
        harness.records = [_Record(content="关掉之后", message_id="b", timestamp=1002.0)]
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["先开着"])

    def test_re_enabling_does_not_backfill_what_was_missed(self) -> None:
        """关闭期间攒下的历史不该在她重新开口时被一次性倾泻进聊天框。

        重新开启会把水位抬到当下（wall_clock = 1000.0），所以一条时间戳更早的
        记录必须被挡住——宿主把 since_ts 当尽力而为的过滤，不能只靠它。
        """
        harness = _Harness([_Record(content="关闭期间的旧话", timestamp=900.0, message_id="old")])
        harness.relay.set_enabled(False)
        harness.relay.set_enabled(True)
        harness.relay._poll_once()
        self.assertEqual(harness.sent, [])

    def test_cold_start_does_not_replay_existing_history(self) -> None:
        """构造即开启时，store 里已有的整段历史不能被当成新记录发出去。"""
        harness = _Harness([_Record(content="很久以前的对话", timestamp=100.0, message_id="ancient")])
        harness.relay._poll_once()
        self.assertEqual(harness.sent, [])

    def test_enabling_after_a_disable_still_forwards_new_lines(self) -> None:
        harness = _Harness([_Record(content="旧话", message_id="old")])
        harness.relay.set_enabled(False)
        harness.relay.set_enabled(True)
        harness.records = [_Record(content="新话", timestamp=1002.0, message_id="new")]
        harness.relay._poll_once()
        self.assertEqual(harness.sent, ["新话"])


class ChatboxRelayFailureTests(unittest.TestCase):
    def test_source_error_is_recorded_without_raising(self) -> None:
        """宿主不可用是常态，必须重试而不是让轮询线程退出。"""
        relay = ChatboxRelay(
            ChatboxRelayConfig(poll_interval_s=0.01),
            source=lambda _count, _since: (_ for _ in ()).throw(RuntimeError("bus down")),
            send=lambda _text: (True, None),
            clock=lambda: 0.0,
            wall_clock=lambda: 0.0,
        )
        relay._poll_once()
        snapshot = relay.snapshot()
        self.assertIn("bus down", str(snapshot["last_source_error"]))
        self.assertEqual(snapshot["forwarded_count"], 0)

    def test_send_failure_is_counted_and_reported(self) -> None:
        harness = _Harness([_Record(content="发不出去")])
        harness.send_failures.append("发不出去")
        harness.relay._poll_once()
        snapshot = harness.relay.snapshot()
        self.assertEqual(harness.sent, [])
        self.assertEqual(snapshot["send_failure_count"], 1)
        self.assertEqual(snapshot["last_error"], "send rejected")

    def test_source_error_clears_after_a_successful_poll(self) -> None:
        harness = _Harness([_Record(content="恢复了")])
        relay = harness.relay
        relay._source = lambda _count, _since: (_ for _ in ()).throw(RuntimeError("bus down"))
        relay._poll_once()
        self.assertIsNotNone(relay.snapshot()["last_source_error"])
        relay._source = harness._source
        relay._poll_once()
        self.assertIsNone(relay.snapshot()["last_source_error"])


class ChatboxRelayThreadTests(unittest.TestCase):
    def test_start_and_stop_manage_the_worker_thread(self) -> None:
        harness = _Harness([])
        relay = harness.relay
        self.assertFalse(relay.thread_alive)
        relay.start()
        self.assertTrue(relay.thread_alive)
        relay.stop()
        self.assertFalse(relay.thread_alive)

    def test_start_is_idempotent(self) -> None:
        harness = _Harness([])
        relay = harness.relay
        relay.start()
        first = relay._thread
        relay.start()
        self.assertIs(relay._thread, first)
        relay.stop()

    def test_a_running_relay_forwards_through_the_background_thread(self) -> None:
        import time

        harness = _Harness([_Record(content="线程转发", message_id="t1")])
        harness.relay.start()
        try:
            deadline = time.monotonic() + 2.0
            while not harness.sent and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            harness.relay.stop()
        self.assertEqual(harness.sent, ["线程转发"])


if __name__ == "__main__":
    unittest.main()
