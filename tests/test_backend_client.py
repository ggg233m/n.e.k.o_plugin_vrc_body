from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import time
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

    def test_backend_submit_gates_declared_world_preconditions(self) -> None:
        service = BackendService({}, Path.cwd())

        class FakeScheduler:
            def __init__(self):
                self.submissions = []

            def snapshot(self):
                return {"state": "idle", "safety_state": "normal"}

            def submit(self, kind, params):
                self.submissions.append((kind, params))
                return {
                    "accepted": True,
                    "action_id": "a-1",
                    "state": "active",
                    "normalized_params": params,
                    "reason": None,
                    "safety_state": "normal",
                }

        scheduler = FakeScheduler()
        service.scheduler = scheduler
        condition = [{
            "kind": "entity_visible",
            "entity_id": "yolo:cup:7",
            "source": "yolo",
            "min_confidence": 0.8,
            "max_age_ms": 500,
        }]
        rejected = service.submit(
            "reach_and_grab",
            {"side": "right"},
            preconditions=condition,
        )
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["reason_code"], "world_precondition_failed")
        self.assertTrue(rejected["replan_required"])
        self.assertEqual(
            rejected["precondition_check"]["failures"][0]["code"],
            "entity_not_visible",
        )
        self.assertEqual(scheduler.submissions, [])
        feedback = service.cognition.snapshot()["feedback"][-1]
        self.assertEqual(feedback["reason_code"], "world_precondition_failed")
        self.assertEqual(
            feedback["precondition_check"]["failures"][0]["code"],
            "entity_not_visible",
        )

        conflicting_alias = service.submit(
            "reach_and_grab",
            {"side": "right", "_world_preconditions": []},
            preconditions=condition,
        )
        self.assertFalse(conflicting_alias["accepted"])
        self.assertEqual(
            conflicting_alias["precondition_check"]["failures"][0]["code"],
            "invalid_world_precondition",
        )
        self.assertEqual(scheduler.submissions, [])

        stopped = service.submit("stop", {}, preconditions=condition)
        self.assertTrue(stopped["accepted"])
        self.assertTrue(stopped["precondition_check"]["bypassed"])
        self.assertEqual(len(scheduler.submissions), 1)

        service.ingest_world({
            "source": "yolo",
            "entities": [{
                "id": "yolo:cup:7",
                "label": "cup",
                "confidence": 0.95,
                "source": ["yolo"],
                "ttl_s": 2.0,
            }],
        })
        accepted = service.submit(
            "reach_and_grab",
            {"side": "right"},
            preconditions=condition,
        )
        self.assertTrue(accepted["accepted"])
        self.assertTrue(accepted["precondition_check"]["passed"])
        self.assertEqual(len(scheduler.submissions), 2)

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
                    "entities": [
                        {"id": "button", "label": "button", "confidence": 0.9},
                        {"id": "avatar", "label": "avatar"},
                    ],
                },
            )
            self.assertTrue(world["available"])
            self.assertEqual(world["entities"][0]["id"], "button")
            gated = client.scheduler.submit(
                "arm_pose",
                {"side": "right", "elevation_deg": 90},
                preconditions=[{
                    "kind": "entity_visible",
                    "entity_id": "missing-target",
                    "min_confidence": 0.8,
                }],
            )
            self.assertFalse(gated["accepted"])
            self.assertEqual(gated["reason_code"], "world_precondition_failed")
            malformed_gate = client.scheduler.submit(
                "arm_pose",
                {"side": "right", "elevation_deg": 90},
                preconditions=42,
            )
            self.assertFalse(malformed_gate["accepted"])
            self.assertEqual(
                malformed_gate["precondition_check"]["failures"][0]["code"],
                "invalid_world_precondition",
            )
            plan = client.request(
                "POST",
                "/cognition/plan",
                {
                    "goal": "raise hand",
                    "action": "arm_pose",
                    "params": {"side": "right"},
                    "preconditions": [{
                        "kind": "entity_visible",
                        "entity_id": "button",
                        "min_confidence": 0.8,
                    }],
                },
            )
            self.assertEqual(plan["status"], "planned")
            self.assertTrue(plan["precondition_check"]["passed"])
            cognition = client.request("GET", "/cognition")
            self.assertEqual(cognition["plan"]["id"], plan["id"])
            self.assertEqual(cognition["state"]["mode"], "nominal")
            self.assertGreaterEqual(
                cognition["state"]["sources"]["world"]["runtime"]["entity_count"],
                1,
            )
            feedback = client.request(
                "POST",
                "/cognition/feedback",
                {"type": "world_changed", "data": {"entity": "button"}},
            )
            self.assertTrue(feedback["replan_required"])
            self.assertEqual(feedback["replan_reason"], "world_changed")
            rejected = client.scheduler.submit("play_clip", {"clip_name": "does-not-exist"})
            self.assertFalse(rejected["accepted"])
            self.assertIn("unknown preset clip", rejected["reason"])
        finally:
            client.stop()
        self.assertIsNone(client.process)

    def test_process_and_debug_cli_run_without_plugin_sdk(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "backend.toml"
            config_file.write_text(
                "[vmc_idle]\nenabled=false\nmanage_host_output=false\n"
                "[vrchat_osc]\nenabled=false\n[driver_log]\nenabled=false\n",
                encoding="utf-8",
            )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                port = int(probe.getsockname()[1])
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "backend" / "process.py"),
                    "--config-file", str(config_file),
                    "--config-dir", temp_dir,
                    "--offline",
                    "--port", str(port),
                    "--token", "dev",
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
            try:
                health = None
                for _ in range(60):
                    if process.poll() is not None:
                        break
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(root / "backend" / "debug_cli.py"),
                            "--port", str(port),
                            "--token", "dev",
                            "health",
                        ],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        creationflags=creationflags,
                    )
                    if result.returncode == 0:
                        health = json.loads(result.stdout)
                        break
                    time.sleep(0.1)
                self.assertIsNotNone(health)
                self.assertTrue(health["ok"])
                action = subprocess.run(
                    [
                        sys.executable,
                        str(root / "backend" / "debug_cli.py"),
                        "--port", str(port),
                        "--token", "dev",
                        "action", "--kind", "enable", "--json", "{}",
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                )
                self.assertEqual(action.returncode, 0, action.stderr)
                snapshot = subprocess.run(
                    [
                        sys.executable,
                        str(root / "backend" / "debug_cli.py"),
                        "--port", str(port),
                        "--token", "dev",
                        "snapshot",
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                )
                self.assertEqual(snapshot.returncode, 0, snapshot.stderr)
                snapshot_data = json.loads(snapshot.stdout)
                self.assertTrue(snapshot_data["backend"]["dry_run"])
                self.assertIsNone(snapshot_data["body"]["udp"]["local_port"])
            finally:
                subprocess.run(
                    [
                        sys.executable,
                        str(root / "backend" / "debug_cli.py"),
                        "--port", str(port),
                        "--token", "dev",
                        "shutdown",
                    ],
                    cwd=root,
                    capture_output=True,
                    creationflags=creationflags,
                )
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
