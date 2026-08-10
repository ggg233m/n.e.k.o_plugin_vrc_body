from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import VmcIdleConfig
from neko_anyadance_body.host_vmc import HostVmcController, _normalize_base_url


class FakeRequester:
    def __init__(self, status: dict[str, object]) -> None:
        self.status = {"success": True, **status}
        self.calls: list[tuple[str, str, dict[str, object] | None, str | None]] = []

    def __call__(self, method, path, payload, token):
        self.calls.append((method, path, payload, token))
        if path == "/api/config/page_config":
            return {"autostart_csrf_token": "test-token"}
        if path == "/api/vmc/status":
            return dict(self.status)
        if path == "/api/vmc/enable":
            self.status.update({"enabled": True, **(payload or {})})
            return dict(self.status)
        if path == "/api/vmc/disable":
            self.status["enabled"] = False
            return dict(self.status)
        if path == "/api/vmc/t_pose":
            self.status.update({"t_pose_requested": True, **(payload or {})})
            return dict(self.status)
        raise AssertionError(path)


class HostVmcControllerTests(unittest.TestCase):
    def test_start_enables_target_and_stop_restores_disabled_state(self) -> None:
        requester = FakeRequester({"enabled": False, "host": "127.0.0.1", "port": 40000, "send_rate_hz": 30})
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())
        self.assertTrue(controller.snapshot()["active"])
        self.assertEqual(requester.status["port"], 39539)
        self.assertTrue(controller.stop())
        self.assertFalse(requester.status["enabled"])
        self.assertEqual(requester.status["port"], 40000)
        self.assertEqual(requester.status["send_rate_hz"], 30)
        paths = [call[1] for call in requester.calls]
        self.assertEqual(paths.count("/api/vmc/enable"), 2)
        self.assertEqual(paths.count("/api/vmc/disable"), 1)

    def test_matching_pre_enabled_sender_is_not_owned_or_disabled(self) -> None:
        requester = FakeRequester({"enabled": True, "host": "127.0.0.1", "port": 39539, "send_rate_hz": 60})
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())
        self.assertFalse(controller.snapshot()["changed_by_plugin"])
        self.assertTrue(controller.stop())
        self.assertEqual([call[1] for call in requester.calls], ["/api/vmc/status"])

    def test_stop_restores_pre_enabled_different_destination(self) -> None:
        requester = FakeRequester({"enabled": True, "host": "127.0.0.1", "port": 41000, "send_rate_hz": 45})
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())
        self.assertTrue(controller.stop())
        self.assertTrue(requester.status["enabled"])
        self.assertEqual(requester.status["port"], 41000)
        self.assertEqual(requester.status["send_rate_hz"], 45)

    def test_api_origin_is_restricted_to_loopback_http(self) -> None:
        self.assertEqual(_normalize_base_url("http://127.0.0.1:48911/"), "http://127.0.0.1:48911")
        with self.assertRaisesRegex(ValueError, "loopback"):
            _normalize_base_url("https://example.com:48911")

    def test_rest_calibration_waits_for_host_t_pose_before_reset(self) -> None:
        requester = FakeRequester({
            "enabled": True,
            "host": "127.0.0.1",
            "port": 39539,
            "send_rate_hz": 60,
        })
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())
        resets: list[str] = []

        original = requester.__call__
        status_reads = 0

        def request(method, path, payload, token):
            nonlocal status_reads
            result = original(method, path, payload, token)
            if path == "/api/vmc/status" and requester.status.get("t_pose_requested"):
                status_reads += 1
                if status_reads >= 2:
                    requester.status["t_pose_requested"] = False
                    result = dict(requester.status)
            return result

        controller._requester = request
        self.assertTrue(controller.calibrate_rest_pose(
            lambda: resets.append("reset"),
            poll_interval_seconds=0.0,
        ))
        self.assertEqual(resets, ["reset"])
        self.assertEqual(controller.snapshot()["calibration"]["state"], "calibrated")
        t_pose_call = next(call for call in requester.calls if call[1] == "/api/vmc/t_pose")
        self.assertEqual(t_pose_call[2], {"duration_sec": 2.0})


if __name__ == "__main__":
    unittest.main()
