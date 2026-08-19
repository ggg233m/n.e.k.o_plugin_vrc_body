from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import vision
from neko_anyadance_body.backend.vision import DxcamFrameSource, optional_dependency_status


class _FakeDxcam:
    def enum_dxgi_adapters(self):
        return [object()]

    def output_info(self):
        return "Device[0] Output[0]"


class _LegacyDxcam:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if "backend" in kwargs:
            raise TypeError("backend is not supported by this old DXcam")
        return "legacy-camera"


class VisionCaptureTests(unittest.TestCase):
    def test_optional_dependency_status_reports_winrt_capability(self) -> None:
        status = optional_dependency_status()
        for key in ("winrt", "winrt_graphics_capture", "dxcam_winrt"):
            self.assertIn(key, status)
            self.assertIsInstance(status[key], bool)
        self.assertEqual(
            status["dxcam_winrt"],
            status["dxcam"] and status["winrt_graphics_capture"],
        )

    def test_auto_candidates_include_winrt_only_when_projection_is_available(self) -> None:
        source = object.__new__(DxcamFrameSource)
        source._requested_device_idx = -1
        source._requested_output_idx = -1
        source._requested_backend = "auto"
        source._winrt_available = True
        candidates = source._build_candidates(_FakeDxcam())
        self.assertIn((0, None, "dxgi"), candidates)
        self.assertIn((0, None, "winrt"), candidates)

        source._winrt_available = False
        candidates_without_winrt = source._build_candidates(_FakeDxcam())
        self.assertTrue(candidates_without_winrt)
        self.assertTrue(all(item[2] == "dxgi" for item in candidates_without_winrt))

    def test_explicit_winrt_does_not_silently_fallback_to_legacy_dxgi(self) -> None:
        source = object.__new__(DxcamFrameSource)
        source._dxcam = _LegacyDxcam()
        with self.assertRaises(TypeError):
            source._create_camera((0, None, "winrt"))
        self.assertEqual(len(source._dxcam.calls), 1)
        self.assertEqual(source._dxcam.calls[0]["backend"], "winrt")

    def test_legacy_dxcam_can_still_use_dxgi_fallback(self) -> None:
        source = object.__new__(DxcamFrameSource)
        source._dxcam = _LegacyDxcam()
        self.assertEqual(source._create_camera((0, None, "dxgi")), "legacy-camera")
        self.assertEqual(len(source._dxcam.calls), 2)
        self.assertNotIn("backend", source._dxcam.calls[-1])


if __name__ == "__main__":
    unittest.main()
