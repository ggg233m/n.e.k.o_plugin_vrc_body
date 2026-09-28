from __future__ import annotations

import math
import random
import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.nav_follow import (
    FollowConfig,
    FuserConfig,
    PathFollower,
    RelocFuser,
    _compose,
    _inverse,
    wrap,
)


class Se2Tests(unittest.TestCase):
    def test_inverse_compose_identity(self):
        a = (1.2, -0.4, 2.8)
        e = _compose(a, _inverse(a))
        for v in e:
            self.assertAlmostEqual(v, 0.0, places=9)


class FuserTests(unittest.TestCase):
    ALIGN = (3.0, -1.0, math.radians(40))   # 真 odom→map

    def reloc(self, f: RelocFuser, odom, dist, noise=(0.0, 0.0, 0.0)):
        m = _compose(self.ALIGN, odom)
        m = (m[0] + noise[0], m[1] + noise[1], m[2] + noise[2])
        f.predict(odom, dist)
        return f.add_reloc(m, odom, dist)

    def test_single_reloc_does_not_localize(self):
        f = RelocFuser()
        self.assertFalse(self.reloc(f, (0, 0, 0), 0.0))
        self.assertEqual(f.estimate()["state"], "unknown")

    def test_two_agreeing_relocs_localize(self):
        f = RelocFuser()
        self.reloc(f, (0, 0, 0), 0.0, (0.1, 0, 0))
        self.assertTrue(self.reloc(f, (1, 0, 0), 1.0, (-0.1, 0.05, 0.02)))
        est = f.estimate()
        true = _compose(self.ALIGN, (1, 0, 0))
        self.assertLess(math.hypot(est["xy"][0] - true[0], est["xy"][1] - true[1]), 0.15)

    def test_disagreeing_relocs_do_not_init(self):
        f = RelocFuser()
        self.reloc(f, (0, 0, 0), 0.0, (2.0, 0, 0))
        self.assertFalse(self.reloc(f, (1, 0, 0), 1.0))

    def test_outlier_rejected_after_init(self):
        f = RelocFuser()
        self.reloc(f, (0, 0, 0), 0.0)
        self.reloc(f, (1, 0, 0), 1.0)
        before = f.estimate()["xy"]
        self.assertFalse(self.reloc(f, (1, 0, 0), 1.0, (3.0, 0, 0)))
        self.assertEqual(f.rejected, 1)
        self.assertEqual(f.estimate()["xy"], before)

    def test_sigma_grows_with_distance_without_reloc(self):
        f = RelocFuser()
        self.reloc(f, (0, 0, 0), 0.0)
        self.reloc(f, (1, 0, 0), 1.0)
        s0 = f.estimate()["sigma_m"]
        f.predict((21, 0, 0), 21.0)
        self.assertAlmostEqual(f.estimate()["sigma_m"] - s0, 0.028 * 20, places=6)

    def test_noisy_stream_converges(self):
        rnd = random.Random(1)
        f = RelocFuser(FuserConfig())
        errs = []
        for k in range(60):
            odom = (0.3 * k, 0.1 * k, 0.02 * k)
            n = (rnd.gauss(0, 0.4), rnd.gauss(0, 0.4), rnd.gauss(0, 0.04))
            if rnd.random() < 0.3:
                n = (rnd.uniform(-3, 3), rnd.uniform(-3, 3), rnd.uniform(-0.5, 0.5))
            self.reloc(f, odom, 0.32 * k, n)
            e = f.estimate()
            if e["state"] == "localized" and k > 10:
                t = _compose(self.ALIGN, odom)
                errs.append(math.hypot(e["xy"][0] - t[0], e["xy"][1] - t[1]))
        errs.sort()
        self.assertLess(errs[len(errs) // 2], 0.35)


class FollowerTests(unittest.TestCase):
    def est(self, x, y, th, sigma=0.3):
        return {"state": "localized", "xy": (x, y), "theta": th, "sigma_m": sigma}

    def test_unknown_waits_and_scans(self):
        out = PathFollower([[0, 0], [5, 0]]).step({"state": "unknown"})
        self.assertEqual(out["state"], "wait_localization")
        self.assertEqual(out["forward"], 0.0)

    def test_local_stop_overrides(self):
        out = PathFollower([[0, 0], [5, 0]]).step(self.est(0, 0, 0), local_stop=True)
        self.assertEqual((out["state"], out["forward"]), ("stopped", 0.0))

    def test_high_sigma_stops(self):
        out = PathFollower([[0, 0], [5, 0]]).step(self.est(0, 0, 0, sigma=1.2))
        self.assertEqual(out["reason"], "localization_uncertain")

    def test_turn_in_place_when_facing_away(self):
        out = PathFollower([[0, 0], [5, 0]]).step(self.est(0, 0, math.pi))
        self.assertEqual(out["state"], "turning")
        self.assertEqual(out["forward"], 0.0)

    def test_steers_back_toward_path(self):
        out = PathFollower([[0, 0], [5, 0]]).step(self.est(1.0, 0.3, 0.0))
        self.assertEqual(out["state"], "following")
        self.assertLess(out["turn_rate"], 0.0)       # 在路径左侧 → 右转（顺时针）

    def test_off_path_requests_replan(self):
        out = PathFollower([[0, 0], [5, 0]]).step(self.est(2.0, 1.0, 0.0))
        self.assertEqual(out["state"], "replan")

    def test_arrives(self):
        out = PathFollower([[0, 0], [5, 0]]).step(self.est(4.85, 0.05, 0.0))
        self.assertEqual(out["state"], "arrived")

    def test_closed_loop_kinematic(self):
        path = [[0, 0], [4, 0], [4, 3], [0, 3]]
        fol = PathFollower(path, FollowConfig())
        x, y, th, dt, v = 0.0, 0.0, 0.0, 0.1, 1.0
        for _ in range(600):
            out = fol.step(self.est(x, y, th, 0.2))
            if out["state"] == "arrived":
                break
            self.assertNotIn(out["state"], ("replan", "wait_localization"))
            th = wrap(th + out["turn_rate"] * dt)
            x += out["forward"] * v * dt * math.cos(th)
            y += out["forward"] * v * dt * math.sin(th)
        self.assertEqual(out["state"], "arrived")


if __name__ == "__main__":
    unittest.main()
