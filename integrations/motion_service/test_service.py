"""协议与竞态验证：使用有限假生成器，不把模拟回执视为世界证据。"""
import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from .core import MotionService, compile_constraints
from .server import make_server


class FakeGenerator:
    ready = True
    frames = 8
    fps = 20
    def generate(self, *args):
        return {"root_positions": [[0, 1, 0]] * 8}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.service = MotionService(FakeGenerator(), clock=lambda: self.now)
        self.world = {"mode": "isolated", "session": 1, "revision": 1, "root": [0, 0],
                      "paths": {"desk": [[0, 1], [1, 1]]}, "max_speed": 1}
        self.service.bind(self.world)

    def task(self, **kwargs):
        return dict(session=1, instance=self.service.instance, request_id="one", prompt="Walk naturally", duration_s=.4,
                    epoch=self.service.epoch, **kwargs)

    def test_health_never_claims_real_world_ready(self):
        self.assertTrue(self.service.health()["ready"])
        self.assertFalse(self.service.health()["execution_ready"])

    def test_base_intent_does_not_preempt_explicit_task(self):
        task = self.service.submit(self.task())
        with self.assertRaisesRegex(ValueError, "explicit_task_owns_motion"):
            self.service.submit_intent({"session": 1, "instance": self.service.instance, "epoch": self.service.epoch, "request_id": "base",
                "intent": {"version": 1, "prompt": "Stand relaxed", "movement": "blocked"}})
        self.assertEqual(self.service.active["op_id"], task["op_id"])

    def test_base_stationary_intent_cannot_supply_movement_target(self):
        self.service.submit_intent({"session": 1, "instance": self.service.instance, "epoch": self.service.epoch, "request_id": "base",
            "intent": {"version": 1, "prompt": "Stand relaxed", "movement": "blocked", "target_key": "desk"}})
        self.assertIsNone(self.service.active["target_key"])

    def test_root_deviation_rejected_before_buffering(self):
        self.service.generator.generate = lambda *args: {"root_positions": [[2, 1, 0]] * 8}
        task = self.service.submit(self.task())
        self.assertFalse(self.service.step())
        self.assertEqual(len(self.service.buffer), 0)
        self.assertEqual(self.service.status(task["op_id"])["status"], "failed")
        self.assertEqual(self.service.status(task["op_id"])["error"], "root_constraint_violation")
        self.assertTrue(self.service.health()["ready"])

    def test_pull_must_match_task_id_not_only_session_and_epoch(self):
        task=self.service.submit(self.task())
        self.service.step()
        with self.assertRaisesRegex(ValueError,"stale_task"):
            self.service.pull(dict(task,op_id="other"))
        self.assertIsNotNone(self.service.pull(task)["chunk"])

    def test_old_cancel_cannot_cancel_replacement(self):
        old=self.service.submit(self.task())
        self.service.cancel(old)
        new=self.service.submit(dict(self.task(),request_id="two"))
        with self.assertRaisesRegex(ValueError,"stale_task_cancel"):
            self.service.cancel(old)
        self.assertEqual(self.service.active["op_id"],new["op_id"])

    def test_task_deadline_expires_even_with_live_world_lease(self):
        task = self.service.submit(self.task())
        for index in range(1, 12):
            self.now = index
            self.service.bind(self.world)
        self.assertEqual(self.service.status(task["op_id"])["error"], "task_timed_out")

    def test_premature_terminal_ack_does_not_consume_chunk(self):
        task = self.service.submit(dict(self.task(), duration_s=2))
        self.service.step()
        self.service.pull(task)
        with self.assertRaises(ValueError):
            self.service.ack(dict(task, sequence=0, terminal=True))
        self.assertEqual(len(self.service.buffer), 1)

    def test_idempotency_and_conflict(self):
        data = self.task()
        first = self.service.submit(data)
        self.assertEqual(first, self.service.submit(data))
        self.assertEqual(first, self.service.request_status({"session": 1, "request_id": "one"}))
        with self.assertRaises(ValueError):
            self.service.submit(dict(data, prompt="other"))
        self.service.cancel({"session": 1})
        self.assertEqual(self.service.submit(data)["status"], "cancelled")

    def test_delayed_request_after_cancel_rejected(self):
        data = self.task()
        self.service.cancel({"session": 1})
        with self.assertRaisesRegex(ValueError, "stale_epoch"):
            self.service.submit(data)

    def test_request_from_previous_process_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale_service_instance"):
            self.service.submit(dict(self.task(), instance="previous-process"))

    def test_inflight_generation_discarded_after_cancel(self):
        entered, release = threading.Event(), threading.Event()
        def generate(*args):
            entered.set()
            release.wait(2)
            return {}
        self.service.generator.generate = generate
        task = self.service.submit(self.task())
        thread = threading.Thread(target=self.service.step)
        thread.start()
        self.assertTrue(entered.wait(1))
        self.service.cancel({"session": 1})
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(self.service.buffer), 0)
        self.assertEqual(self.service.dropped, 1)
        self.assertEqual(self.service.status(task["op_id"])["status"], "cancelled")

    def test_no_completion_without_matching_final_ack(self):
        task = self.service.submit(self.task())
        with self.assertRaises(ValueError):
            self.service.ack(dict(task, sequence=0, terminal=True))
        self.service.step()
        self.assertEqual(self.service.status(task["op_id"])["status"], "awaiting_world")
        first = self.service.pull(task)
        self.assertEqual(first, self.service.pull(task))
        result = self.service.ack(dict(task, sequence=0, terminal=True))
        self.assertEqual(result["status"], "isolated_completed")

    def test_bounded_buffer_and_world_lease(self):
        task = self.service.submit(dict(self.task(), duration_s=30))
        for _ in range(20):
            self.service.step()
        self.assertEqual(len(self.service.buffer), 3)
        self.now = 3
        self.assertIsNone(self.service.health()["world_session"])
        self.assertEqual(len(self.service.buffer), 0)
        self.assertEqual(self.service.status(task["op_id"])["status"], "cancelled")

    def test_world_revision_invalidates_active_task(self):
        task = self.service.submit(self.task())
        self.service.bind(dict(self.world, revision=2))
        self.assertEqual(self.service.status(task["op_id"])["status"], "cancelled")
        with self.assertRaises(ValueError):
            self.service.bind(self.world)

    def test_path_sampling_and_missing_target(self):
        world = self.service.world
        result = compile_constraints(world, {"target_key": "desk"}, .8, 8, 20)
        self.assertAlmostEqual(result["root_xz"][-1][0], .2)
        self.assertEqual(result["root_xz"][-1][1], 1)
        stationary = compile_constraints(world, {}, 5, 8, 20)
        self.assertTrue(all(p == [0, 0] for p in stationary["root_xz"]))
        with self.assertRaises(ValueError):
            compile_constraints(world, {"target_key": "missing"}, 0, 8, 20)

    def test_http_auth_health_and_roundtrip(self):
        token = "test-token-" * 4
        server = make_server(self.service, token, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)
        def call(path, data=None, auth=True, origin=False):
            headers = {"Authorization": "Bearer " + token} if auth else {}
            if origin:
                headers["Origin"] = "http://example.test"
            request = Request(base + path, data=None if data is None else json.dumps(data).encode(), headers=headers)
            with urlopen(request, timeout=2) as response:
                return json.load(response)
        try:
            self.assertFalse(call("/health", auth=False)["execution_ready"])
            for auth, origin in ((False, False), (True, True)):
                with self.assertRaises(HTTPError):
                    call("/tasks", self.task(), auth, origin)
            task = call("/tasks", self.task())
            self.assertEqual(call("/tasks/" + task["op_id"]), task)
            self.assertEqual(call("/cancel", {"session": 1})["status"], "cancelled")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
