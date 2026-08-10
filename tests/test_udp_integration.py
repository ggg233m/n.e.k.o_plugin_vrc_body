from __future__ import annotations

import json
import socket
import time
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.scheduler import BodyScheduler


class UdpIntegrationTests(unittest.TestCase):
    def test_local_receiver_observes_complete_frames_near_60_hz(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1.0)
        port = receiver.getsockname()[1]
        scheduler = BodyScheduler(PluginConfig(port=port))
        scheduler.start()
        try:
            self.assertTrue(scheduler.submit("enable")["accepted"])
            first_packet, _ = receiver.recvfrom(8192)
            payload = json.loads(first_packet)
            self.assertEqual(payload["version"], 1)
            self.assertEqual(len(payload["devices"]), 6)
            self.assertEqual(len(payload["inputs"]), 2)

            count = 1
            started = time.perf_counter()
            while time.perf_counter() - started < 1.0:
                try:
                    packet, _ = receiver.recvfrom(8192)
                except TimeoutError:
                    break
                self.assertLess(len(packet), 8192)
                count += 1
            self.assertGreaterEqual(count, 55)
            self.assertLessEqual(count, 65)
        finally:
            scheduler.shutdown()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
