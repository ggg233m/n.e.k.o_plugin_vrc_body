from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.obs_capture import (
    ObsVirtualCameraFrameSource,
    ObsWebSocketFrameSource,
)


class _FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.sequence = 0

    def isOpened(self):
        return self.opened and not self.released

    def set(self, *_args):
        return True

    def read(self):
        if self.released or not self.opened:
            return False, None
        self.sequence += 1
        return True, {"sequence": self.sequence}

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, *, opened_indices: set[int] | None = None) -> None:
        self.opened_indices = opened_indices if opened_indices is not None else {0}
        self.captures: list[_FakeCapture] = []

    def VideoCapture(self, index, *_args):
        capture = _FakeCapture(index in self.opened_indices)
        self.captures.append(capture)
        return capture


class _FakeSocket:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = [json.dumps(item) for item in messages]
        self.sent: list[str] = []
        self.closed = False

    def recv(self):
        if not self.messages:
            raise RuntimeError("no more fake websocket messages")
        return self.messages.pop(0)

    def send(self, payload: str):
        self.sent.append(payload)

    def settimeout(self, _value):
        return None

    def close(self):
        self.closed = True


class _FakeWebsocket:
    def __init__(self, socket: _FakeSocket) -> None:
        self.socket = socket
        self.url = None

    def create_connection(self, url, timeout):
        self.url = (url, timeout)
        return self.socket


class _FakeImage:
    def convert(self, mode):
        return ("decoded", mode)


class _FakePil:
    @staticmethod
    def open(_stream):
        return _FakeImage()


class ObsVirtualCameraTests(unittest.TestCase):
    def test_latest_frame_reader_does_not_queue_stale_frames(self) -> None:
        cv2 = _FakeCv2()
        source = ObsVirtualCameraFrameSource(
            camera_index=0,
            backend="dshow",
            clock=time.monotonic,
            cv2_module=cv2,
        )
        try:
            deadline = time.monotonic() + 1.0
            latest = None
            while time.monotonic() < deadline:
                latest = source.read()
                if source.status()["frames"] >= 3:
                    break
                time.sleep(0.005)
            status = source.status()
            self.assertTrue(status["available"])
            self.assertTrue(status["reader_running"])
            self.assertGreaterEqual(status["frames"], 3)
            self.assertIsNotNone(latest)
            # The producer is deliberately faster than the consumer; the
            # observed sequence should be near the newest frame, not a queue's
            # first stale item.
            self.assertEqual(latest["sequence"], status["frames"])
        finally:
            source.close()
        self.assertFalse(source.status()["available"])
        self.assertTrue(cv2.captures[0].released)

    def test_unavailable_camera_is_explicit_and_non_throwing(self) -> None:
        source = ObsVirtualCameraFrameSource(
            camera_index=-1,
            probe_count=2,
            cv2_module=_FakeCv2(opened_indices=set()),
        )
        try:
            status = source.status()
            self.assertFalse(status["available"])
            self.assertIn("unavailable", str(status["last_error"]).lower())
            self.assertIsNone(source.read())
        finally:
            source.close()


class ObsWebSocketTests(unittest.TestCase):
    def test_v5_authentication_matches_protocol(self) -> None:
        password, salt, challenge = "secret", "salt", "challenge"
        first = base64.b64encode(hashlib.sha256((password + salt).encode()).digest())
        expected = base64.b64encode(hashlib.sha256(first + challenge.encode()).digest()).decode()
        self.assertEqual(
            ObsWebSocketFrameSource._authentication(password, salt, challenge),
            expected,
        )

    def test_screenshot_bridge_handshakes_and_decodes_one_frame(self) -> None:
        socket = _FakeSocket([
            {
                "op": 0,
                "d": {
                    "rpcVersion": 1,
                    "authentication": {"salt": "salt", "challenge": "challenge"},
                },
            },
            {"op": 2, "d": {}},
            {
                "op": 7,
                "d": {
                    "requestType": "GetSourceScreenshot",
                    "requestId": "1",
                    "requestStatus": {"result": True},
                    "responseData": {
                        "imageData": "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode(),
                    },
                },
            },
        ])
        websocket = _FakeWebsocket(socket)
        source = ObsWebSocketFrameSource(
            source_name="VRChat Mirror",
            password="secret",
            websocket_module=websocket,
            cv2_module=None,
            pil_module=_FakePil,
        )
        frame = source.read()
        self.assertEqual(frame, ("decoded", "RGB"))
        self.assertTrue(source.status()["available"])
        self.assertEqual(source.status()["frames"], 1)
        self.assertEqual(websocket.url[0], "ws://127.0.0.1:4455")
        identify = json.loads(socket.sent[0])
        self.assertEqual(identify["op"], 1)
        self.assertIn("authentication", identify["d"])
        request = json.loads(socket.sent[1])
        self.assertEqual(request["d"]["requestType"], "GetSourceScreenshot")
        self.assertEqual(request["d"]["requestData"]["sourceName"], "VRChat Mirror")
        source.close()
        self.assertFalse(source.status()["available"])

    def test_missing_source_name_is_unavailable_without_connecting(self) -> None:
        websocket = _FakeWebsocket(_FakeSocket([]))
        source = ObsWebSocketFrameSource(source_name="", websocket_module=websocket)
        self.assertFalse(source.status()["available"])
        self.assertIsNone(source.read())
        self.assertIsNone(websocket.url)
        source.close()


if __name__ == "__main__":
    unittest.main()
