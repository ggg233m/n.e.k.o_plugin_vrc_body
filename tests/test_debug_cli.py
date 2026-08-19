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


if __name__ == "__main__":
    unittest.main()
