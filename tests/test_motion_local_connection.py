"""真实回环HTTP验证；不启动模型或世界。"""
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.motion_backend import MotionBackend, MotionBackendConfig
from integrations.motion_service.server import make_server


class LocalConnectionTests(unittest.TestCase):
    def setUp(self):
        class Service:
            def health(self):
                return {"protocol": "yui-motion/1", "ready": True}

            def status(self, op_id):
                return {"op_id": op_id, "status": "running"}

            def submit(self, data):
                return {"op_id": data["op_id"], "status": "accepted"}

            bind = submit_intent = cancel = pull = ack = request_status = submit

        self.server = make_server(Service(), port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.backend = MotionBackend(MotionBackendConfig(enabled=True, endpoint=self.base))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_host_without_environment_token_and_with_stale_token(self):
        for env in ({}, {"YUI_MOTION_TOKEN": "obsolete-token"}):
            with self.subTest(env_present=bool(env)), patch.dict("os.environ", env, clear=True):
                self.assertTrue(self.backend._request("/health")["ready"])
                result = self.backend._request("/tasks", {"op_id": "test"})
                self.assertEqual(result["status"], "accepted")
                self.assertEqual(self.backend._request("/tasks/test")["status"], "running")

    def test_reject_browser_and_rebound_host(self):
        opener = build_opener(ProxyHandler({}))
        for headers in ({"Origin": "https://example.test"}, {"Origin": "null"},
                        {"Origin": ""}, {"Sec-Fetch-Site": "cross-site"},
                        {"Sec-Fetch-Site": "none"}, {"Host": "example.test"}):
            for path, data in (("/health", None), ("/tasks", b'{"op_id":"test"}')):
                with self.subTest(headers=headers, path=path), self.assertRaises(HTTPError) as failure:
                    opener.open(Request(self.base + path, data=data, headers=headers), timeout=2)
                self.assertEqual(failure.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
