from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import debug_cli


class DebugCliTests(unittest.TestCase):
    def test_scalar_arg_preserves_json_types(self) -> None:
        self.assertIs(debug_cli._scalar_arg("true"), True)
        self.assertEqual(debug_cli._scalar_arg("17"), 17)
        self.assertEqual(debug_cli._scalar_arg("0.5"), 0.5)
        self.assertEqual(debug_cli._scalar_arg("not-json"), "not-json")
        with self.assertRaisesRegex(ValueError, "scalar"):
            debug_cli._scalar_arg("[1, 2]")

    def test_locomotion_command_dispatches_to_osc_endpoint(self) -> None:
        calls: list[tuple[object, ...]] = []

        def fake_request(*args, **kwargs):
            calls.append((args, kwargs))
            return {"accepted": True}

        argv = [
            "debug_cli.py",
            "--token",
            "dev",
            "locomotion",
            "--vertical",
            "0.5",
            "--horizontal",
            "-0.25",
            "--duration-ms",
            "800",
        ]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(debug_cli, "request", fake_request):
            with redirect_stdout(output):
                self.assertEqual(debug_cli.main(), 0)

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[:5], ("127.0.0.1", 48912, "dev", "POST", "/osc/locomotion"))
        self.assertEqual(
            args[5],
            {"vertical": 0.5, "horizontal": -0.25, "duration_ms": 800},
        )
        self.assertIn('"accepted": true', output.getvalue())

    def test_deferred_chatbox_sets_immediate_false(self) -> None:
        calls: list[tuple[object, ...]] = []

        def fake_request(*args, **kwargs):
            calls.append((args, kwargs))
            return {"accepted": True}

        argv = [
            "debug_cli.py",
            "--token",
            "dev",
            "chatbox",
            "--text",
            "hello",
            "--deferred",
        ]
        with patch.object(sys, "argv", argv), patch.object(debug_cli, "request", fake_request):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(debug_cli.main(), 0)

        self.assertEqual(calls[0][0][4], "/osc/chatbox")
        self.assertEqual(calls[0][0][5], {"text": "hello", "immediate": False})

    def test_persistent_shell_reuses_one_session(self) -> None:
        class FakeClient:
            def __init__(self, host, port, token):
                self.args = (host, port, token)
                self.calls = []
                self.closed = False

            def request(self, method, path, payload):
                self.calls.append((method, path, payload))
                return {"accepted": True}

            def close(self):
                self.closed = True

        fake = FakeClient("127.0.0.1", 48912, "dev")
        stdin = io.StringIO(
            '{"path":"/osc/locomotion","payload":{"vertical":0.2}}\n'
            '{"path":"/osc/stop_movement","payload":{}}\n'
            "quit\n"
        )
        with patch.object(debug_cli, "_PersistentHttpClient", return_value=fake), patch.object(
            sys, "stdin", stdin
        ):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(debug_cli._run_shell("127.0.0.1", 48912, "dev"), 0)

        self.assertEqual(
            fake.calls,
            [
                ("POST", "/osc/locomotion", {"vertical": 0.2}),
                ("POST", "/osc/stop_movement", {}),
            ],
        )
        self.assertTrue(fake.closed)
        self.assertIn('"ready": true', output.getvalue())


if __name__ == "__main__":
    unittest.main()
