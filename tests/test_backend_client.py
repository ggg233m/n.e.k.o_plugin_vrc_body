from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.client import BackendClient, BackendUnavailable, RemoteScheduler
from neko_anyadance_body.backend.service import BackendService


class BackendClientTests(unittest.TestCase):
    def test_remote_osc_config_invalid_values_fall_back_safely(self) -> None:
        client = BackendClient({"vrchat_osc": {"enabled": "no", "input_pulse_ms": "bad"}}, Path.cwd())
        self.assertTrue(client.osc_config.enabled)
        self.assertEqual(client.osc_config.input_pulse_ms, 100)

    def test_scheduler_snapshot_contains_safe_awareness_when_backend_is_down(self) -> None:
        class DownClient:
            def request(self, *_args, **_kwargs):
                raise BackendUnavailable("offline")

        snapshot = RemoteScheduler(DownClient()).snapshot()
        self.assertEqual(snapshot["state"], "backend_unavailable")
        self.assertIn("awareness", snapshot)

    def test_expanded_clip_duration_limit_is_enforced_in_backend(self) -> None:
        service = BackendService({}, Path.cwd())
        service.config = replace(service.config, clip_max_duration_seconds=3.0)

        class FakeScheduler:
            def snapshot(self):
                return {"state": "idle", "safety_state": "normal"}

            def submit(self, *_args, **_kwargs):
                raise AssertionError("overlong clip must be rejected before scheduling")

        service.scheduler = FakeScheduler()
        clip = SimpleNamespace(is_pose=False, duration_s=2.0, name="long")
        result = service.submit(
            "play_clip",
            {"clip_name": "long", "speed": 1.0, "loop_count": 2, "_clip": clip},
        )
        self.assertFalse(result["accepted"])
        self.assertIn("must not exceed", result["reason"])

    def test_standalone_backend_process_health_and_shutdown(self) -> None:
        root = Path(__file__).resolve().parents[1]
        client = BackendClient(
            {
                "anyadance": {"port": 39570, "rate_hz": 60},
                "vmc_idle": {"enabled": False, "manage_host_output": False},
                "vrchat_osc": {"enabled": False},
                "driver_log": {"enabled": False},
            },
            root,
        )
        try:
            client.start(timeout_s=8.0)
            self.assertTrue(client.request("GET", "/health")["ok"])
            self.assertEqual(client.scheduler.snapshot()["state"], "disabled")
            self.assertFalse(client.vision.snapshot()["available"])
            world = client.request(
                "POST",
                "/world/ingest",
                {
                    "source": "test_detector",
                    "entities": [{"id": "button", "label": "button", "confidence": 0.9}],
                },
            )
            self.assertTrue(world["available"])
            self.assertEqual(world["entities"][0]["id"], "button")
            rejected = client.scheduler.submit("play_clip", {"clip_name": "does-not-exist"})
            self.assertFalse(rejected["accepted"])
            self.assertIn("unknown preset clip", rejected["reason"])
        finally:
            client.stop()
        self.assertIsNone(client.process)


if __name__ == "__main__":
    unittest.main()
