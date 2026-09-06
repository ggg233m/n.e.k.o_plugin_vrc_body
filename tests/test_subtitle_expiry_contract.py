"""字幕有符号时钟与本地到期隐藏的回归契约。"""
from pathlib import Path
import unittest


class SubtitleExpiryTests(unittest.TestCase):
    def test_signed_clock_and_wraparound(self):
        def signed(value):
            return (value + 2**31) % 2**32 - 2**31

        for start in (-200000, -6000, 0, 2**31 - 3000):
            deadline = signed(start + 6000)
            for elapsed, expired in ((0, False), (5999, False), (6000, True), (300000, True)):
                self.assertEqual(signed(signed(start + elapsed) - deadline) >= 0, expired)

    def test_owner_clear_and_remote_projection_share_expiry(self):
        source = (Path(__file__).resolve().parents[1] / "unity/Assets/NEKO/Npc/NekoNpcNameplate.cs").read_text(encoding="utf-8-sig")
        self.assertNotIn("_displayUntilServerMs > 0", source)
        self.assertIn("nowServerMs - _displayUntilServerMs >= 0", source)
        self.assertIn("_unitCount <= 0 || HasTextExpired(nowServerMs)", source)
        self.assertIn("Networking.IsOwner(gameObject) && HasTextExpired", source)
        self.assertIn('ClearBubbleWithReason("expired")', source)


if __name__ == "__main__":
    unittest.main()
