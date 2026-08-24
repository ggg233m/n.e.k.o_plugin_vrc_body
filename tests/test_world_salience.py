from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.world_salience import (
    bearing_sector,
    classify,
    delta_signature,
    describe_entities,
    proximity_band,
)


def _person(entity_id: str, *, height: float, bearing: float, clipped: bool = False,
            label: str = "person") -> dict[str, object]:
    return {
        "id": entity_id,
        "label": label,
        "state": "visible",
        "confidence": 0.8,
        "attributes": {
            "apparent_height": height,
            "apparent_height_clipped": clipped,
            "bearing_deg": bearing,
        },
    }


class ProximityBandTests(unittest.TestCase):
    def test_bands_are_monotonic_in_apparent_height(self) -> None:
        self.assertEqual(proximity_band(0.60), "very_close")
        self.assertEqual(proximity_band(0.35), "close")
        self.assertEqual(proximity_band(0.20), "mid")
        self.assertEqual(proximity_band(0.10), "far")
        self.assertEqual(proximity_band(0.03), "very_far")

    def test_clipped_and_missing_heights_are_unknown_not_guessed(self) -> None:
        """贴边的框高度会饱和，缺失就是缺失——都不能编一个距离出来。"""
        self.assertEqual(proximity_band(0.9, clipped=True), "unknown")
        self.assertEqual(proximity_band(None), "unknown")
        self.assertEqual(proximity_band("nope"), "unknown")
        self.assertEqual(proximity_band(float("nan")), "unknown")
        self.assertEqual(proximity_band(0.0), "unknown")


class BearingSectorTests(unittest.TestCase):
    def test_sectors_carry_side(self) -> None:
        self.assertEqual(bearing_sector(0.0), "front")
        self.assertEqual(bearing_sector(-20.0), "front_slight_left")
        self.assertEqual(bearing_sector(45.0), "side_right")
        self.assertEqual(bearing_sector(-90.0), "wide_left")
        self.assertEqual(bearing_sector(150.0), "behind_right")

    def test_missing_bearing_is_unknown(self) -> None:
        self.assertEqual(bearing_sector(None), "unknown")
        self.assertEqual(bearing_sector(float("inf")), "unknown")


class DeltaSignatureTests(unittest.TestCase):
    def test_walking_closer_changes_the_signature(self) -> None:
        """回归：这正是旧签名漏掉的情形。

        id / label / state 全程不变，检测器又不发事件，于是「有人径直走过来」
        在旧实现下产生零次推送——距离和方位一个都不在签名里。
        """
        far = {"entities": [_person("p1", height=0.08, bearing=-5.0)]}
        near = {"entities": [_person("p1", height=0.50, bearing=-5.0)]}
        self.assertNotEqual(delta_signature(far), delta_signature(near))

    def test_walking_sideways_changes_the_signature(self) -> None:
        left = {"entities": [_person("p1", height=0.30, bearing=-60.0)]}
        right = {"entities": [_person("p1", height=0.30, bearing=60.0)]}
        self.assertNotEqual(delta_signature(left), delta_signature(right))

    def test_pixel_jitter_does_not_change_the_signature(self) -> None:
        """量化是刻意的：用原始浮点等于取消去重，每帧都会推一次。"""
        first = {"entities": [_person("p1", height=0.3000, bearing=-5.00)]}
        second = {"entities": [_person("p1", height=0.3004, bearing=-5.20)]}
        self.assertEqual(delta_signature(first), delta_signature(second))

    def test_signature_is_order_independent(self) -> None:
        a = _person("p1", height=0.3, bearing=-10.0)
        b = _person("p2", height=0.2, bearing=40.0)
        self.assertEqual(
            delta_signature({"entities": [a, b]}),
            delta_signature({"entities": [b, a]}),
        )

    def test_malformed_entries_are_skipped_not_fatal(self) -> None:
        signature = delta_signature({
            "entities": ["not a mapping", None, _person("p1", height=0.3, bearing=0.0)],
            "events": ["junk"],
            "removed_entity_ids": ["p9"],
        })
        self.assertIn("p1", signature)


class ClassifyTests(unittest.TestCase):
    def test_person_appearing_in_conversation_range_is_context_only(self) -> None:
        result = classify({"entities": [_person("p1", height=0.30, bearing=-20.0)]}, {})
        self.assertFalse(result["wake"])
        self.assertIn("出现", result["reasons"][0])
        self.assertEqual(result["entity_states"]["p1"], ("close", "front_slight_left"))

    def test_person_appearing_far_away_does_not_wake(self) -> None:
        """远处出现的人只进上下文；不然一个路过的人就能打断对话。"""
        result = classify({"entities": [_person("p1", height=0.07, bearing=-20.0)]}, {})
        self.assertFalse(result["wake"])
        # 但状态仍要记录，后续靠近才有得比。
        self.assertEqual(result["entity_states"]["p1"], ("far", "front_slight_left"))

    def test_approaching_from_far_to_close_is_context_only(self) -> None:
        history = {"p1": ("far", "front")}
        result = classify({"entities": [_person("p1", height=0.35, bearing=2.0)]}, history)
        self.assertFalse(result["wake"])
        self.assertIn("靠近", result["reasons"][0])

    def test_walking_away_does_not_wake(self) -> None:
        history = {"p1": ("very_close", "front")}
        result = classify({"entities": [_person("p1", height=0.08, bearing=2.0)]}, history)
        self.assertFalse(result["wake"])

    def test_staying_put_does_not_wake(self) -> None:
        history = {"p1": ("close", "front")}
        result = classify({"entities": [_person("p1", height=0.30, bearing=2.0)]}, history)
        self.assertFalse(result["wake"])

    def test_unknown_proximity_never_counts_as_approaching(self) -> None:
        """不确定不能当成靠近——否则贴边的框会反复触发唤醒。"""
        history = {"p1": ("far", "front")}
        result = classify(
            {"entities": [_person("p1", height=0.9, bearing=2.0, clipped=True)]},
            history,
        )
        self.assertFalse(result["wake"])
        self.assertEqual(result["entity_states"]["p1"], ("unknown", "front"))

    def test_non_person_entities_never_wake_the_agent(self) -> None:
        chair = _person("c1", height=0.40, bearing=0.0, label="chair")
        result = classify({"entities": [chair]}, {})
        self.assertFalse(result["wake"])

    def test_social_events_wake_the_agent(self) -> None:
        result = classify({"events": [{"kind": "wave", "target_id": "p1"}]}, {})
        self.assertTrue(result["wake"])
        self.assertIn("wave", result["reasons"][0])

    def test_unrelated_events_do_not_wake(self) -> None:
        result = classify({"events": [{"kind": "frame_decoded"}]}, {})
        self.assertFalse(result["wake"])

    def test_nearby_person_leaving_is_context_only(self) -> None:
        near_gone = classify(
            {"removed_entity_ids": ["p1"]},
            {"p1": ("close", "front")},
        )
        self.assertFalse(near_gone["wake"])
        self.assertIn("离开", near_gone["reasons"][0])
        far_gone = classify(
            {"removed_entity_ids": ["p1"]},
            {"p1": ("far", "front")},
        )
        self.assertFalse(far_gone["wake"])

    def test_empty_delta_is_quiet(self) -> None:
        result = classify({}, {})
        self.assertFalse(result["wake"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["entity_states"], {})

    def test_reasons_are_bounded(self) -> None:
        crowd = {"entities": [
            _person(f"p{index}", height=0.30, bearing=float(index))
            for index in range(20)
        ]}
        result = classify(crowd, {})
        self.assertLessEqual(len(result["reasons"]), 6)
        # 状态表不截断：被截掉的实体下一轮会变成「新出现」而反复唤醒。
        self.assertEqual(len(result["entity_states"]), 20)


class DescribeEntitiesTests(unittest.TestCase):
    def test_description_carries_direction_and_distance(self) -> None:
        described = describe_entities([_person("p1", height=0.50, bearing=-45.0)])
        self.assertEqual(len(described), 1)
        self.assertIn("side_left", described[0])
        self.assertIn("very_close", described[0])

    def test_description_is_bounded(self) -> None:
        entities = [_person(f"p{index}", height=0.3, bearing=0.0) for index in range(30)]
        self.assertEqual(len(describe_entities(entities)), 12)


if __name__ == "__main__":
    unittest.main()
