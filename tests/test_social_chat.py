"""输入租约与主动搭话必须由玩家身份和真实执行终态驱动。"""
import unittest

import _bootstrap  # noqa: F401
import test_autonomy as fixtures


class SocialChatTests(unittest.TestCase):
    def setUp(self):
        fixtures.AutonomyTests.setUp(self)
        self.session.players[2].update(pid=22, region_key="a", reachable=True)
        self.openings = []
        self.director._opening_callback = self.openings.append

    tearDown = fixtures.AutonomyTests.tearDown

    def activity(self, seq, state="active", slot=2, pid=22, idle_ms=0, session=9):
        return self.director.note_chat_input({"session": session, "slot": slot, "pid": pid,
            "input_seq": seq, "state": state, "idle_ms": idle_ms})

    def arrive(self):
        self.clock.advance(31)
        self.assertTrue(self.director._try_proactive_chat(self.clock()))
        self.director._service_chat_engagement(self.clock())
        self.assertEqual(self.openings, [])
        self.adapter.plan_manager.statuses[self.director._active.plan_id] = "succeeded"
        self.director._update_active(self.clock())
        self.director._service_chat_engagement(self.clock())
        self.assertEqual(len(self.openings), 1)
        return self.openings[0]

    def test_open_cancels_roaming_and_keepalive_preserves_typing(self):
        self.clock.advance(1)
        self.director._tick()
        self.assertIsNotNone(self.director._active)
        self.assertTrue(self.activity(1, "open"))
        self.assertIn(("autonomy", "chat_engagement", True), self.adapter.plan_manager.cancelled)
        for sequence in range(2, 21):
            self.clock.advance(5)
            self.assertTrue(self.activity(sequence))
            self.director._tick()
            self.assertEqual(self.director._chat_engagement.phase, "waiting_input")
        self.clock.advance(15.1)
        self.director._tick()
        self.assertIsNone(self.director._chat_engagement)

    def test_new_session_can_restart_input_sequence(self):
        self.assertTrue(self.activity(10, "open"))
        self.session.session = 10
        self.assertFalse(self.activity(11, session=9))
        self.assertTrue(self.activity(1, "open", session=10))
        self.director._on_session_event({"type": "sys.chat_input_ready", "session": 9, "activity_version": 0})
        self.assertTrue(self.director._input_supported)

    def test_keepalive_does_not_reset_120_second_input_idle(self):
        self.activity(1, "open")
        for sequence in range(2, 25):
            self.clock.advance(5)
            self.activity(sequence, idle_ms=int(self.clock() * 1000))
            self.director._tick()
        self.clock.advance(5)
        self.assertFalse(self.activity(25, idle_ms=120000))
        self.director._tick()
        self.assertIsNone(self.director._chat_engagement)

    def test_order_identity_and_other_player_open_cannot_steal(self):
        self.assertFalse(self.activity(1, "open", session=8))
        self.assertFalse(self.activity(1, "open", pid=23))
        self.assertTrue(self.activity(2, "open"))
        self.assertFalse(self.activity(1, "closed"))
        self.session.players[3] = {"pid": 33, "d": 2}
        self.assertFalse(self.activity(1, "open", slot=3, pid=33))
        self.assertEqual(self.director._chat_engagement.player_slot, 2)
        self.assertTrue(self.director.begin_chat_engagement(3))
        self.assertEqual(self.director._chat_engagement.player_slot, 3)

    def test_submit_close_keeps_reply_and_manual_close_has_grace(self):
        self.activity(1, "open")
        self.activity(2, "submitted")
        self.director.begin_chat_engagement(2)
        self.activity(3, "submitted")
        self.clock.advance(20)
        self.director._tick()
        self.assertEqual(self.director._chat_engagement.phase, "waiting_reply")
        self.director._finish_chat_engagement("test", request_intent=False)
        self.activity(4, "open")
        self.activity(5, "closed")
        self.clock.advance(9.9)
        self.director._tick()
        self.assertIsNotNone(self.director._chat_engagement)
        self.clock.advance(0.2)
        self.director._tick()
        self.assertIsNone(self.director._chat_engagement)

    def test_reply_window_is_extended_while_player_prepares_next_turn(self):
        self.director.begin_chat_engagement(2)
        self.director.note_reply_page({"reply_serial": 1, "reply_end": True,
            "display_seconds": 10, "transfer_sequence": 4})
        self.session.emit({"type": "npc.text_cleared", "reason": "expired", "transfer_seq": 4})
        self.clock.advance(55)
        self.activity(1, "open")
        self.clock.advance(10)
        self.director._tick()
        self.assertIsNotNone(self.director._chat_engagement)

    def test_arrival_is_required_and_opening_is_sent_once(self):
        request = self.arrive()
        sent = []
        self.assertTrue(self.director.dispatch_proactive_opening(request, lambda: sent.append(1) or True, 7))
        self.assertFalse(self.director.dispatch_proactive_opening(request, lambda: sent.append(2) or True, 7))
        self.assertEqual(sent, [1])
        self.assertFalse(self.director.note_reply_page({"reply_serial": 7, "reply_end": True, "display_seconds": 10}))
        self.assertTrue(self.director.note_reply_page({"reply_serial": 8, "reply_end": True, "display_seconds": 10}))
        self.clock.advance(70.1)
        self.director._tick()
        self.assertIsNone(self.director._chat_engagement)
        self.assertGreaterEqual(self.director._player_chat_next[22], self.clock() + 599)

    def test_player_input_cancels_queued_opening(self):
        request = self.arrive()
        self.activity(1, "open")
        sent = []
        self.assertFalse(self.director.dispatch_proactive_opening(request, lambda: sent.append(1) or True, 0))
        self.assertEqual(sent, [])
        self.assertEqual(self.director._chat_engagement.phase, "waiting_input")

    def test_arrival_look_does_not_block_opening_but_external_operation_does(self):
        self.clock.advance(31)
        self.director._try_proactive_chat(self.clock())
        self.adapter.plan_manager.statuses[self.director._active.plan_id] = "succeeded"
        self.session.operations["owned-look"] = {"kind": "look", "status": "running"}
        self.session.npc_state["active_ops"] = ["owned-look"]
        self.director._update_active(self.clock())
        self.session.npc_state["active_ops"].append("external-move")
        self.director._service_chat_engagement(self.clock())
        self.assertEqual(self.openings, [])
        self.session.npc_state["active_ops"] = ["owned-look"]
        self.director._service_chat_engagement(self.clock())
        self.director._service_chat_engagement(self.clock())
        self.assertEqual(len(self.openings), 1)
        self.assertTrue(self.director._chat_engagement.look_owned)

    def test_nearby_player_cooldown_has_its_own_reason_and_countdown(self):
        self.clock.advance(31)
        self.director._player_chat_next[22] = self.clock() + 75
        self.assertFalse(self.director._try_proactive_chat(self.clock()))
        self.assertEqual(self.director.status()["proactive_chat"]["reason"], "player_cooldown")
        self.assertEqual(self.director.status()["proactive_chat"]["retry_in_s"], 75)

    def test_failure_pause_and_slot_reuse_do_not_speak(self):
        self.clock.advance(31)
        self.director._try_proactive_chat(self.clock())
        self.adapter.plan_manager.statuses[self.director._active.plan_id] = "failed"
        self.director._update_active(self.clock())
        self.assertEqual(self.openings, [])
        self.assertIsNone(self.director._chat_engagement)
        self.director._player_chat_next.clear()
        self.clock.advance(120)
        request = self.arrive()
        self.session.players[2]["pid"] = 23
        self.assertFalse(self.director.dispatch_proactive_opening(request, lambda: True, 0))
        self.director.pause()
        self.assertFalse(self.director.dispatch_proactive_opening(request, lambda: True, 0))

    def test_eligibility_and_cooldown_do_not_use_unverified_players(self):
        self.clock.advance(31)
        for field, invalid in (("reachable", False), ("region_key", "b"), ("d", 8.1)):
            original = self.session.players[2][field]
            self.session.players[2][field] = invalid
            self.assertFalse(self.director._try_proactive_chat(self.clock()))
            self.session.players[2][field] = original
        self.director._reply_busy = lambda: True
        self.assertFalse(self.director._try_proactive_chat(self.clock()))
        self.director._reply_busy = lambda: False
        self.assertTrue(self.director._try_proactive_chat(self.clock()))
        self.director._finish_chat_engagement("test", request_intent=False)
        self.assertFalse(self.director._try_proactive_chat(self.clock()))


if __name__ == "__main__":
    unittest.main()
