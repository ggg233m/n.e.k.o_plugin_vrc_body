from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.autonomy import AutonomyRuntime
from neko_anyadance_body.backend.vision import OpenAICompatibleSemanticBackend
from neko_anyadance_body.backend.world_state import WorldStateStore
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.scheduler import BodyScheduler


class _Transport:
    local_port = 39501

    def __init__(self) -> None:
        self.packets: list[bytes] = []
        self.lock = threading.Lock()

    def send(self, payload: bytes, _target: tuple[str, int]) -> None:
        with self.lock:
            self.packets.append(payload)

    def close(self) -> None:
        return


def _wait(predicate, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


class ControllerInputTests(unittest.TestCase):
    def test_overlay_is_latest_wins_and_auto_releases(self) -> None:
        transport = _Transport()
        scheduler = BodyScheduler(PluginConfig(), transport=transport)
        scheduler.start()
        try:
            self.assertTrue(scheduler.submit("enable")["accepted"])
            _wait(lambda: scheduler.snapshot()["state"] == "idle")
            self.assertTrue(scheduler.submit("input_axes", {
                "side": "left", "x": 0.75, "y": 0.25, "duration_ms": 500,
            })["accepted"])
            self.assertTrue(scheduler.submit("input_button", {
                "side": "right", "button": "trigger", "pressed": True,
                "value": 1.0, "hold_ms": 100,
            })["accepted"])
            _wait(lambda: scheduler.snapshot()["controller_input"]["axes"].get("left", {}).get("x") == 0.75)
            with transport.lock:
                frame = json.loads(transport.packets[-1])
            self.assertAlmostEqual(frame["inputs"]["left_controller"]["joystick_x"], 0.75)
            self.assertTrue(frame["inputs"]["right_controller"]["trigger_click"])
            _wait(lambda: "right:trigger" not in scheduler.snapshot()["controller_input"]["buttons"])
            self.assertIn("left", scheduler.snapshot()["controller_input"]["axes"])
            self.assertTrue(scheduler.submit("input_release", {"side": "all"})["accepted"])
            _wait(lambda: not scheduler.snapshot()["controller_input"]["axes"])
        finally:
            scheduler.shutdown()


class AutonomyAndWorldTests(unittest.TestCase):
    def test_autonomy_requires_arm_and_ttl_releases(self) -> None:
        now = [0.0]
        released: list[str] = []
        runtime = AutonomyRuntime(
            world_provider=lambda: {},
            release_inputs=lambda: released.append("release"),
            clock=lambda: now[0],
            session_ttl_s=60.0,
        )
        self.assertFalse(runtime.submit_goal("find a portal")["accepted"])
        self.assertTrue(runtime.arm()["armed"])
        self.assertTrue(runtime.submit_goal("find a portal")["accepted"])
        now[0] = 61.0
        self.assertFalse(runtime.snapshot()["armed"])
        self.assertIn("release", released)

    def test_world_delta_revision_and_player_memory_boundary(self) -> None:
        store = WorldStateStore(clock=lambda: 1.0)
        first = store.delta(wait_ms=0)
        self.assertFalse(first["changed"])
        store.ingest(
            entities=[
                {"id": "world:cup", "label": "cup", "source": "vision"},
                {"id": "vrchat:player:user", "label": "player", "source": "vrchat_log"},
            ],
            events=[{"type": "chat_message", "data": {"text": "secret"}}],
            source="vision",
        )
        delta = store.delta(first["revision"], wait_ms=0)
        self.assertTrue(delta["changed"])
        self.assertGreater(delta["revision"], first["revision"])
        self.assertIn("world:cup", [item["id"] for item in delta["changes"]["entities"]])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.json"
            persisted = WorldStateStore(
                clock=lambda: 1.0,
                persistence_path=path,
                persist_world=True,
                persist_players=False,
            )
            persisted.ingest(
                entities=[
                    {"id": "world:cup", "label": "cup", "source": "vision"},
                    {"id": "vrchat:player:user", "label": "player", "source": "vrchat_log"},
                ],
                events=[{"type": "chat_message", "data": {"text": "secret"}}],
                source="vision",
            )
            loaded = WorldStateStore(
                clock=lambda: 1.0,
                persistence_path=path,
                persist_world=True,
                persist_players=False,
            ).snapshot()
            self.assertEqual([item["id"] for item in loaded["entities"]], ["world:cup"])
            self.assertEqual(loaded["events"], [])

    def test_semantic_backend_structures_json_and_enforces_rate_limit(self) -> None:
        class _Response:
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": json.dumps({
                        "entities": [{"id": "world:door", "label": "door", "confidence": 0.8}],
                        "events": [],
                        "uncertainties": ["depth_unknown"],
                    })}}],
                }).encode("utf-8")

        backend = OpenAICompatibleSemanticBackend(
            endpoint="http://127.0.0.1/v1/chat/completions",
            model="test",
            max_per_minute=1,
            request_fn=lambda _request, _timeout: _Response(),
        )
        observation = backend.observe(b"jpeg", world={}, now=1.0)
        self.assertEqual(observation.entities[0]["id"], "world:door")
        self.assertEqual(observation.uncertainties, ("depth_unknown",))
        with self.assertRaises(RuntimeError):
            backend.observe(b"jpeg", world={}, now=1.1)


if __name__ == "__main__":
    unittest.main()
