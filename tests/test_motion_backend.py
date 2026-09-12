"""后端失效、能力切换与语义边界回归，不连接真实世界。"""
import time
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.motion_backend import MotionBackend, MotionBackendConfig
from yui_npc_controller.runtime.motion_intent import compile_motion_intent
from yui_npc_controller.runtime.tool_surface import YuiToolSurface


class MotionBackendTests(unittest.TestCase):
    def test_backend_reconfigure_updates_existing_tool_surface(self):
        from test_plugin_config import _plugin_class
        plugin=_plugin_class()(None)
        plugin._surface=Mock()
        previous=plugin._motion_backend
        plugin._configure_motion_backend()
        self.assertIs(plugin._surface.motion_backend,plugin._motion_backend)
        self.assertIsNot(plugin._motion_backend,previous)
        plugin._motion_backend.close()

    def test_accepted_task_is_dispatched_and_dispatch_failure_is_cancelled(self):
        backend=MotionBackend(MotionBackendConfig(enabled=True))
        backend._health={"instance":"service","epoch":3}
        backend.world_ready=Mock(return_value=True)
        task=dict(status="accepted",session=1,op_id="task",epoch=4,prompt_version=4)
        backend._request=Mock(return_value=task)
        backend._execution_dispatch=Mock()
        session=SimpleNamespace(session=1)
        self.assertEqual(backend.perform(session,prompt="Stand naturally"),task)
        backend._execution_dispatch.start.assert_called_once_with(session,task)
        backend._execution_dispatch.start.side_effect=RuntimeError("unavailable")
        self.assertEqual(backend.perform(session,prompt="Stand naturally")["status"],"failed")
        self.assertEqual(backend._request.call_args.args,("/cancel",dict(session=1,op_id="task",epoch=4)))

    def test_disabled_backend_never_starts_or_performs(self):
        backend = MotionBackend(MotionBackendConfig())
        backend.start()
        self.assertIsNone(backend._thread)
        self.assertFalse(backend.snapshot()["ready"])
        self.assertEqual(backend.perform(SimpleNamespace(session=1))["error"], "backend_unavailable")

    def test_demo_health_without_world_binding_is_not_execution_ready(self):
        backend = MotionBackend(MotionBackendConfig(enabled=True))
        backend._seen = time.monotonic()
        backend._health = {"ready": True, "protocol": "yui-motion/1"}
        session = SimpleNamespace(session=3, control_state="external", capabilities=["pose_stream_v1"])
        self.assertFalse(backend.world_ready(session))
        backend._health.update(world_session=3, execution_ready=True)
        self.assertFalse(backend.world_ready(session))
        factory=Mock()
        factory.ready.return_value=True
        backend.bind_execution(factory)
        self.assertTrue(backend.world_ready(session))
        session.session = 4
        self.assertFalse(backend.world_ready(session))
        session.session = 3
        backend._seen -= 10
        self.assertFalse(backend.world_ready(session))
        backend.close()

    def test_remote_endpoint_and_nonfinite_timeout_rejected(self):
        for value in ({"endpoint": "http://example.org"}, {"timeout_s": float("nan")},
                      {"endpoint": "http://127.0.0.1@evil.test"}, {"enabled": "yes"},
                      {"endpoint": "http://127.0.0.1:0"}, {"endpoint": "http://127.0.0.1:99999"}):
            with self.assertRaises(ValueError):
                MotionBackendConfig.from_mapping(value)

    def test_world_binding_change_notifies_without_model_disconnect(self):
        backend = MotionBackend(MotionBackendConfig(enabled=True, poll_s=.2))
        backend._seen = time.monotonic()
        backend._health = {"ready": True, "protocol": "yui-motion/1",
                           "world_session": 1, "execution_ready": False}
        backend._request = Mock(return_value={"ready": True, "protocol": "yui-motion/1",
                                             "world_session": 1, "execution_ready": True})
        changes = []
        def changed():
            changes.append(backend._health.copy())
            backend._stop.set()
        backend._changed = changed
        backend._poll()
        self.assertEqual(len(changes), 1)
        self.assertTrue(changes[0]["execution_ready"])
        self.assertEqual(backend._generation, 1)

    def test_failed_health_does_not_block_start_and_thread_stops(self):
        backend = MotionBackend(MotionBackendConfig(enabled=True, poll_s=.2))
        backend._request = Mock(side_effect=OSError("offline"))
        backend.start()
        backend.close()
        self.assertFalse(backend._thread.is_alive())
        self.assertFalse(backend.snapshot()["ready"])

    def test_old_perform_call_rejected_after_backend_loss(self):
        session = SimpleNamespace(session=1, control_state="external", capabilities=[], players={}, catalogs={})
        surface = YuiToolSurface(Mock(), session)
        surface.definitions = lambda: []
        backend = Mock()
        backend.world_ready.return_value = True
        surface.motion_backend = backend
        self.assertEqual({x.name for x in surface.host_definitions()}, {"npc.stop", "npc.perform"})
        backend.world_ready.return_value = False
        result = surface.call_host("npc.perform", {"prompt": "Walk forward"})
        self.assertEqual(result["error"], "tool_unavailable")
        backend.perform.assert_not_called()
        self.assertEqual([x.name for x in surface.host_definitions()], ["npc.stop"])

    def test_stop_does_not_depend_on_motion_backend(self):
        adapter = Mock()
        surface = YuiToolSurface(adapter, SimpleNamespace(session=0))
        surface.call_host("npc.stop", {"immediate": True})
        adapter.estop.assert_called_once()
        surface.call_host("npc.stop", {})
        adapter.stop.assert_called_once_with("all")

    def test_stationary_constraints_override_prompt(self):
        activity = {"kind": "visit", "motion_description": "Run far away", "duration_s": 10}
        stopped = compile_motion_intent(activity, explicit_stop=True)
        self.assertEqual(stopped["movement"], "blocked")
        self.assertNotIn("Run", stopped["prompt"])
        chatting = compile_motion_intent(activity, chat_engaged=True)
        self.assertEqual(chatting["movement"], "blocked")

    def test_cancel_timeout_latches_backend_unavailable(self):
        backend = MotionBackend(MotionBackendConfig(enabled=True))
        backend._seen = time.monotonic()
        backend._health = {"ready": True, "protocol": "yui-motion/1", "world_session": 1,
                           "execution_ready": True, "epoch": 2}
        backend._request = Mock(side_effect=TimeoutError())
        session = SimpleNamespace(session=1, control_state="external", capabilities=["pose_stream_v1"])
        backend.cancel(session)
        self.assertFalse(backend.world_ready(session))

    def test_expired_context_cannot_execute_stale_tool(self):
        import asyncio
        from test_plugin_config import _plugin_class
        plugin = _plugin_class()(None)
        plugin._last_world_event_at = time.monotonic() - 11
        plugin._surface = Mock()
        result = asyncio.run(plugin._make_tool_handler("npc.execute_plan")())
        self.assertEqual(result["error"], "world_snapshot_expired")
        plugin._surface.call_host.assert_not_called()

    def test_refresh_worker_retries_without_asyncio_loop(self):
        from yui_npc_controller.runtime.refresh_worker import RefreshWorker
        succeeded = threading.Event()
        count = []
        def callback():
            count.append(1)
            if len(count) == 1:
                raise OSError("temporary")
            succeeded.set()
        worker = RefreshWorker(callback)
        worker.start()
        worker.wake.set()
        try:
            self.assertTrue(succeeded.wait(2))
        finally:
            worker.close()
        self.assertFalse(worker.thread.is_alive())
