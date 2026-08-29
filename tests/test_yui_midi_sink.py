"""Mido 输出端口名称解析的回归测试。"""

from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.yui_transport import MidoOutputSink


class _FakePort:
    def close(self) -> None:
        pass


def _fake_mido(names: list[str]):
    opened: list[str] = []

    def open_output(name: str):
        opened.append(name)
        return _FakePort()

    fake = SimpleNamespace(get_output_names=lambda: list(names), open_output=open_output)
    return fake, opened


class MidoOutputSinkTests(unittest.TestCase):
    def test_prefers_exact_port_name(self) -> None:
        fake, opened = _fake_mido(["NEKO_MIDI 1", "NEKO_MIDI"])

        with patch.dict(sys.modules, {"mido": fake}):
            sink = MidoOutputSink("NEKO_MIDI")

        self.assertEqual(sink.requested_port_name, "NEKO_MIDI")
        self.assertEqual(sink.port_name, "NEKO_MIDI")
        self.assertEqual(opened, ["NEKO_MIDI"])

    def test_resolves_unique_rtmidi_numeric_suffix(self) -> None:
        fake, opened = _fake_mido(
            ["Microsoft GS Wavetable Synth 0", "NEKO_MIDI 1"]
        )

        with patch.dict(sys.modules, {"mido": fake}):
            sink = MidoOutputSink("NEKO_MIDI")

        self.assertEqual(sink.port_name, "NEKO_MIDI 1")
        self.assertEqual(opened, ["NEKO_MIDI 1"])

    def test_rejects_ambiguous_rtmidi_ports(self) -> None:
        fake, opened = _fake_mido(["NEKO_MIDI 1", "NEKO_MIDI 2"])

        with patch.dict(sys.modules, {"mido": fake}):
            with self.assertRaisesRegex(RuntimeError, "对应多个 RtMidi 端口"):
                MidoOutputSink("NEKO_MIDI")

        self.assertEqual(opened, [])

    def test_does_not_match_unrelated_prefix(self) -> None:
        fake, opened = _fake_mido(["NEKO_MIDI Monitor", "Other 1"])

        with patch.dict(sys.modules, {"mido": fake}):
            with self.assertRaisesRegex(OSError, "找不到 MIDI 输出端口"):
                MidoOutputSink("NEKO_MIDI")

        self.assertEqual(opened, [])
