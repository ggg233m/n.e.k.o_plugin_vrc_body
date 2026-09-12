"""聊天身体意图的即时采纳、旧结果隔离与回退，不模拟为世界验收。"""
import unittest
import _bootstrap  # noqa: F401
import test_autonomy as fixtures
from yui_npc_controller.runtime.intent import validate_intent, IntentModelError


class ChatMotionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AutonomyTests()
        self.fixture.setUp()
        self.director = self.fixture.director
        self.clock = self.fixture.clock
        self.requests = []
        self.director._inspiration_callback = self.requests.append
        self.assertTrue(self.director.begin_chat_engagement(2))

    def tearDown(self):
        self.fixture.tearDown()

    def offer(self, prompt="A person gently nods and opens their hands while standing in place.", token=None):
        request = self.requests[-1]
        value = {"motivation":"认真回应", "mood":"social", "activities":[
            {"kind":"linger","duration_s":10,"motion_description":prompt},
            {"kind":"linger","duration_s":10,"motion_description":"A person stands calmly and listens."}],
            "ttl_s":60,"interests":[],"avoid_targets":[]}
        value = validate_intent(value, request['context'])
        return self.director.offer_intent(value, token or request['request_token'])

    def test_chat_result_immediate_stationary_and_not_navigation(self):
        self.assertEqual(self.requests[-1]['context']['mode'], 'chat_motion')
        before = len(self.fixture.adapter.plan_manager.submissions)
        self.assertTrue(self.offer())
        motion = self.director.motion_intent_snapshot()
        self.assertIn('nods', motion['prompt'])
        self.assertEqual(motion['movement'], 'blocked')
        self.assertIsNone(motion['target_key'])
        self.assertEqual(len(self.fixture.adapter.plan_manager.submissions), before)

    def test_new_chat_supersedes_inflight_and_exit_invalidates(self):
        old = self.requests[-1]['request_token']
        self.director._request_intent('chat_updated')
        self.assertFalse(self.offer(token=old))
        self.assertTrue(self.offer())
        current = self.requests[-1]['request_token']
        self.director._finish_chat_engagement('test', request_intent=False)
        self.assertFalse(self.offer(token=current))
        self.assertNotIn('nods', self.director.motion_intent_snapshot()['prompt'])

    def test_pages_do_not_end_or_replace_motion(self):
        self.assertTrue(self.offer())
        before = self.director.motion_intent_snapshot()
        self.director.note_reply_page({'reply_serial':1,'reply_end':True,'display_seconds':10,'transfer_sequence':1})
        self.assertEqual(self.director.motion_intent_snapshot(),before)
        self.clock.advance(31)
        self.assertNotIn('nods',self.director.motion_intent_snapshot()['prompt'])

    def test_chat_validation_rejects_navigation_and_missing_motion(self):
        value = {"motivation":"回应", "mood":"social", "activities":[
            {"kind":"linger","duration_s":10}, {"kind":"linger","duration_s":10}],
            "ttl_s":60,"interests":[],"avoid_targets":[]}
        with self.assertRaises(IntentModelError):
            validate_intent(value,self.requests[-1]['context'])
        for activity in value['activities']:
            activity.update(kind='local_roam',style='meander',motion_description='Walk around.')
        with self.assertRaises(IntentModelError):
            validate_intent(value,self.requests[-1]['context'])

    def test_pause_rejects_late_result(self):
        token = self.requests[-1]['request_token']
        self.director.pause('manual_pause')
        self.assertFalse(self.offer(token=token))
        self.assertEqual(self.director.motion_intent_snapshot()['movement'],'blocked')

    def test_periodic_refresh_does_not_supersede_inflight_each_tick(self):
        self.assertTrue(self.offer())
        self.director._service_chat_engagement = lambda now: True
        self.director._update_active = lambda now: None
        self.director._startup_intent_needed = False
        self.clock.advance(10)
        self.director._tick()
        count = len(self.requests)
        self.assertEqual(self.requests[-1]['reason'], 'chat_motion_refresh')
        self.clock.advance(10)
        self.director._tick()
        self.assertEqual(len(self.requests),count)


if __name__ == '__main__':
    unittest.main()
