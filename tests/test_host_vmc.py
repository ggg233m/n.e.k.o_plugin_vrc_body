from __future__ import annotations

import threading
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
            # 真实宿主：POST 的返回里带着本次受理后的 generation 计数；requested
            # 只是「此刻有一次请求在生效」，T Pose 播完（duration_sec）后自行复位
            # ——复位不代表「T Pose 刚开始」。
            self.status["t_pose_generation"] = int(self.status.get("t_pose_generation") or 0) + 1
            self.status["t_pose_requested"] = True
            self.status["t_pose_duration_sec"] = float((payload or {}).get("duration_sec") or 2.0)
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

    def test_rest_calibration_resets_as_soon_as_the_host_accepts_the_post(self) -> None:
        """POST 返回即受理；此时 T Pose 正在播，必须立刻去取帧。"""
        requester = FakeRequester({
            "enabled": True,
            "host": "127.0.0.1",
            "port": 39539,
            "send_rate_hz": 60,
            "t_pose_generation": 41,
        })
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())
        resets: list[str] = []
        self.assertTrue(controller.calibrate_rest_pose(lambda: resets.append("reset")))
        self.assertEqual(resets, ["reset"])
        self.assertEqual(controller.snapshot()["calibration"]["state"], "calibrated")
        t_pose_call = next(call for call in requester.calls if call[1] == "/api/vmc/t_pose")
        self.assertEqual(t_pose_call[2], {"duration_sec": 2.0})
        # 受理后 requested 仍为真（T Pose 还在播），不能因此判定失败。
        self.assertTrue(requester.status["t_pose_requested"])

    def test_rest_calibration_does_not_wait_for_the_requested_flag_to_clear(self) -> None:
        """旧实现在等 requested 变 false 才动手，那是 T Pose 已经播完的时刻。"""
        requester = FakeRequester({
            "enabled": True,
            "host": "127.0.0.1",
            "port": 39539,
            "send_rate_hz": 60,
        })
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())
        self.assertTrue(controller.calibrate_rest_pose(lambda: None))
        # 整个握手只允许一次 t_pose POST；不需要再轮询 status 等复位。
        status_reads = sum(1 for call in requester.calls if call[1] == "/api/vmc/status")
        self.assertEqual(status_reads, 1)  # start() 那一次
        self.assertTrue(requester.status["t_pose_requested"])

    def test_rest_calibration_reports_failure_when_the_post_is_rejected(self) -> None:
        """POST 被拒时不能谎报受理，否则中转会白等一个不会来的 T Pose。"""
        requester = FakeRequester({
            "enabled": True,
            "host": "127.0.0.1",
            "port": 39539,
            "send_rate_hz": 60,
        })
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())

        original = requester.__call__

        def request(method, path, payload, token):
            if path == "/api/vmc/t_pose":
                raise RuntimeError("N.E.K.O VMC API returned HTTP 403: forbidden")
            return original(method, path, payload, token)

        controller._requester = request
        self.assertFalse(controller.calibrate_rest_pose(
            lambda: self.fail("被拒绝的请求不能触发基准重置")
        ))
        self.assertEqual(controller.snapshot()["calibration"]["state"], "failed")

    def test_rest_calibration_cancels_before_posting_when_already_stopped(self) -> None:
        """已请求停止时不该再向宿主发 T Pose 请求。"""
        requester = FakeRequester({
            "enabled": True,
            "host": "127.0.0.1",
            "port": 39539,
            "send_rate_hz": 60,
        })
        controller = HostVmcController(VmcIdleConfig(), requester=requester)
        self.assertTrue(controller.start())
        stop_event = threading.Event()
        stop_event.set()
        self.assertFalse(controller.calibrate_rest_pose(
            lambda: self.fail("停止后不能再重置基准"),
            stop_event=stop_event,
        ))
        self.assertEqual(controller.snapshot()["calibration"]["state"], "cancelled")
        self.assertFalse([call for call in requester.calls if call[1] == "/api/vmc/t_pose"])


if __name__ == "__main__":
    unittest.main()
