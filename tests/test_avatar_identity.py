from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy 是本地视觉的可选依赖
    np = None  # type: ignore[assignment]

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.avatar_identity import AvatarIdentityRegistry
from neko_anyadance_body.backend.local_perception import OpenVinoLocalDetector, _Detection


def _avatar_frame(
    bbox: tuple[float, float, float, float],
    *,
    top_color: tuple[int, int, int] = (220, 35, 45),
    bottom_color: tuple[int, int, int] = (35, 70, 220),
    band_side: str = "left",
    background_color: tuple[int, int, int] = (0, 0, 0),
):
    """生成位置可变、局部外观完全一致的测试 Avatar。"""

    assert np is not None
    frame = np.full((80, 120, 3), background_color, dtype=np.uint8)
    left, top, right, bottom = bbox
    x0, x1 = round(left * 120), round(right * 120)
    y0, y1 = round(top * 80), round(bottom * 80)
    middle = y0 + (y1 - y0) // 2
    frame[y0:middle, x0:x1] = top_color
    frame[middle:y1, x0:x1] = bottom_color
    # 加一条不对称饰带，让空间描述子不只依赖两块纯色。
    band_width = max(1, (x1 - x0) // 5)
    if band_side == "right":
        frame[y0:y1, x1 - band_width:x1] = (235, 220, 45)
    else:
        frame[y0:y1, x0:x0 + band_width] = (235, 220, 45)
    return frame


def _detection(track_id: int, bbox: tuple[float, float, float, float]) -> _Detection:
    return _Detection("person", 0.95, bbox, track_id=track_id)


@unittest.skipIf(np is None, "numpy 是本地视觉的可选依赖")
class AvatarIdentityRegistryTests(unittest.TestCase):
    def test_new_track_at_new_position_reuses_session_identity(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test", retention_s=60.0)
        first_box = (0.05, 0.20, 0.30, 0.80)
        second_box = (0.65, 0.20, 0.90, 0.80)

        first = registry.assign(
            [_detection(1, first_box)],
            _avatar_frame(first_box),
            now=1.0,
            source_name="openvino",
        )[1]
        second = registry.assign(
            [_detection(2, second_box)],
            _avatar_frame(second_box),
            now=3.0,
            source_name="openvino",
        )[2]

        self.assertEqual(first.identity_id, "avatar:session:test:1")
        self.assertEqual(second.identity_id, first.identity_id)
        self.assertEqual(second.method, "appearance_reid")
        self.assertAlmostEqual(second.similarity or 0.0, 1.0, places=5)
        self.assertEqual(registry.status()["reidentified_count"], 1)

    def test_different_appearance_gets_a_different_identity(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        first_box = (0.05, 0.20, 0.30, 0.80)
        second_box = (0.65, 0.20, 0.90, 0.80)
        first = registry.assign(
            [_detection(1, first_box)],
            _avatar_frame(first_box),
            now=1.0,
            source_name="openvino",
        )[1]
        second = registry.assign(
            [_detection(2, second_box)],
            _avatar_frame(
                second_box,
                top_color=(25, 210, 75),
                bottom_color=(230, 225, 225),
            ),
            now=2.0,
            source_name="openvino",
        )[2]

        self.assertNotEqual(second.identity_id, first.identity_id)
        self.assertEqual(second.method, "new_identity")

    def test_identical_avatars_in_one_frame_are_not_merged(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        left_box = (0.05, 0.20, 0.30, 0.80)
        right_box = (0.65, 0.20, 0.90, 0.80)
        frame = _avatar_frame(left_box)
        right = _avatar_frame(right_box)
        frame = np.maximum(frame, right)

        assignments = registry.assign(
            [_detection(1, left_box), _detection(2, right_box)],
            frame,
            now=1.0,
            source_name="openvino",
        )

        self.assertNotEqual(assignments[1].identity_id, assignments[2].identity_id)
        self.assertEqual(registry.status()["identity_count"], 2)

    def test_ambiguous_reappearance_uses_decisive_recent_geometry(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        left_box = (0.05, 0.20, 0.30, 0.80)
        right_box = (0.65, 0.20, 0.90, 0.80)
        frame = np.maximum(_avatar_frame(left_box), _avatar_frame(right_box))
        first = registry.assign(
            [_detection(1, left_box), _detection(2, right_box)],
            frame,
            now=1.0,
            source_name="openvino",
        )

        # 两个外观完全相同的目标曾同帧出现；新轨迹紧贴左侧旧位置时，几何证据
        # 足够明确，可以复用左侧身份而不继续制造第三个 ID。
        new_left_box = (0.07, 0.20, 0.32, 0.80)
        reappeared = registry.assign(
            [_detection(3, new_left_box)],
            _avatar_frame(new_left_box),
            now=12.0,
            source_name="openvino",
        )[3]
        self.assertEqual(reappeared.identity_id, first[1].identity_id)
        self.assertEqual(reappeared.method, "appearance_geometry_reid")
        status = registry.status()
        self.assertEqual(status["identity_count"], 2)
        self.assertEqual(status["ambiguous_reused_count"], 1)

    def test_ambiguous_reappearance_without_geometry_stays_separate(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        left_box = (0.05, 0.20, 0.30, 0.80)
        right_box = (0.65, 0.20, 0.90, 0.80)
        frame = np.maximum(_avatar_frame(left_box), _avatar_frame(right_box))
        registry.assign(
            [_detection(1, left_box), _detection(2, right_box)],
            frame,
            now=1.0,
            source_name="openvino",
        )

        # 正中位置对左右两个身份同样合理，继续分配新 ID；不能为了稳定而猜错人。
        middle_box = (0.375, 0.20, 0.625, 0.80)
        ambiguous = registry.assign(
            [_detection(3, middle_box)],
            _avatar_frame(middle_box),
            now=1.2,
            source_name="openvino",
        )[3]
        self.assertEqual(ambiguous.method, "new_identity")
        self.assertEqual(registry.status()["identity_count"], 3)

    def test_coexisting_twins_get_one_stable_bucket_instead_of_unbounded_ids(self) -> None:
        # 死锁回归场景：两个同款 Avatar 同帧出现过，之后单个目标反复以全新
        # track 重现。修复前 margin 判定永远失败，每次重现都新建身份（实测
        # 12 次转回增殖出 12 个新 ID）；修复后第一次重现建立"未知是哪一个"
        # 桶，后续重现全部落进同一个桶，身份总数封顶在 3。
        registry = AvatarIdentityRegistry(session_token="test")
        left_box = (0.05, 0.20, 0.30, 0.80)
        right_box = (0.65, 0.20, 0.90, 0.80)
        frame = np.maximum(_avatar_frame(left_box), _avatar_frame(right_box))
        registry.assign(
            [_detection(1, left_box), _detection(2, right_box)],
            frame,
            now=1.0,
            source_name="openvino",
        )

        boxes = [
            (0.375, 0.20, 0.625, 0.80),
            (0.15, 0.20, 0.40, 0.80),
            (0.55, 0.20, 0.80, 0.80),
            (0.30, 0.20, 0.55, 0.80),
        ]
        bucket_ids = []
        for index, box in enumerate(boxes):
            assignment = registry.assign(
                [_detection(10 + index, box)],
                _avatar_frame(box),
                # 超过 15 秒几何窗口，逼所有启发式失败、走重复兜底。
                now=20.0 * (index + 1),
                source_name="openvino",
            )[10 + index]
            bucket_ids.append(assignment.identity_id)

        # 首次重现建桶，其余全部复用同一个桶。
        self.assertEqual(len(set(bucket_ids)), 1)
        status = registry.status()
        self.assertEqual(status["identity_count"], 3)
        self.assertGreaterEqual(status["duplicate_reidentified_count"], len(boxes) - 1)
        # 两个真实身份的同帧共存对被记录，且从未被兜底互相顶替。
        self.assertGreaterEqual(status["distinct_pair_count"], 1)

    def test_serial_duplicates_are_absorbed_back_into_one_identity(self) -> None:
        # 单人场景：库里出现两个都认得同一外观的串行重复后，重复兜底应把
        # 它们吸并回一个身份，而不是留着第三个候选继续加深歧义。
        registry = AvatarIdentityRegistry(session_token="test")
        box_a = (0.10, 0.20, 0.35, 0.80)
        box_b = (0.60, 0.20, 0.85, 0.80)
        first = registry.assign(
            [_detection(1, box_a)],
            _avatar_frame(box_a),
            now=1.0,
            source_name="openvino",
        )[1]
        # 第二个身份以完全不同的外观建立（从未与 first 同帧共存）……
        different = _avatar_frame(
            box_b,
            top_color=(25, 210, 75),
            bottom_color=(230, 225, 225),
            band_side="right",
        )
        duplicate = registry.assign(
            [_detection(2, box_b)],
            different,
            now=20.0,
            source_name="openvino",
        )[2]
        self.assertEqual(duplicate.method, "new_identity")
        # ……随后在轨迹连续期间学到了与 first 相同的视角（换装/失配的典型
        # 后果），画廊从此互认，构成打平的串行重复。观测数压在 established
        # 启发式（>=8）之下，逼真地落进重复兜底。
        for index in range(6):
            registry.assign(
                [_detection(2, box_b)],
                _avatar_frame(box_b),
                now=20.1 + index * 0.1,
                source_name="openvino",
            )
        merged = registry.assign(
            [_detection(3, box_a)],
            _avatar_frame(box_a),
            now=60.0,
            source_name="openvino",
        )[3]
        status = registry.status()
        self.assertEqual(merged.method, "appearance_duplicate_reid")
        self.assertEqual(status["identity_count"], 1)
        self.assertGreaterEqual(status["merged_identity_count"], 1)
        self.assertIn(merged.identity_id, {first.identity_id, duplicate.identity_id})

    def test_scene_context_beats_misleading_screen_geometry(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        left_box = (0.05, 0.20, 0.30, 0.80)
        right_box = (0.65, 0.20, 0.90, 0.80)
        common = np.maximum(
            _avatar_frame(left_box, background_color=(25, 25, 25)),
            _avatar_frame(right_box, background_color=(25, 25, 25)),
        )
        first = registry.assign(
            [_detection(1, left_box), _detection(2, right_box)],
            common,
            now=1.0,
            source_name="openvino",
        )

        # 两个外观相同的身份后来分别出现在红、绿背景。重新进入红色场景时，
        # 即使检测框更靠近右侧身份的旧位置，也应由背景指纹找回左侧身份。
        registry.assign(
            [_detection(1, left_box)],
            _avatar_frame(left_box, background_color=(150, 20, 20)),
            now=1.1,
            source_name="openvino",
        )
        registry.assign(
            [_detection(2, right_box)],
            _avatar_frame(right_box, background_color=(20, 150, 20)),
            now=1.2,
            source_name="openvino",
        )
        near_right = (0.62, 0.20, 0.87, 0.80)
        reacquired = registry.assign(
            [_detection(3, near_right)],
            _avatar_frame(near_right, background_color=(150, 20, 20)),
            now=1.3,
            source_name="openvino",
        )[3]

        self.assertEqual(reacquired.identity_id, first[1].identity_id)
        self.assertEqual(reacquired.method, "appearance_context_reid")
        self.assertEqual(registry.status()["context_reidentified_count"], 1)

    def test_long_lived_identity_wins_over_one_frame_duplicate_when_geometry_is_ambiguous(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        left_box = (0.05, 0.20, 0.30, 0.80)
        right_box = (0.65, 0.20, 0.90, 0.80)
        frame = np.maximum(_avatar_frame(left_box), _avatar_frame(right_box))
        first = registry.assign(
            [_detection(1, left_box), _detection(2, right_box)],
            frame,
            now=1.0,
            source_name="openvino",
        )
        # 左侧身份持续八帧，右侧只出现过一次；这是“稳定轨迹 + 一帧重复框”的
        # 典型形态。视角移动到中间后几何打平，稳定度仍可阻止 ID 无限增殖。
        for index in range(8):
            registry.assign(
                [_detection(1, left_box)],
                _avatar_frame(left_box),
                now=1.1 + index * 0.1,
                source_name="openvino",
            )
        middle_box = (0.375, 0.20, 0.625, 0.80)
        reacquired = registry.assign(
            [_detection(3, middle_box)],
            _avatar_frame(middle_box),
            now=2.0,
            source_name="openvino",
        )[3]
        self.assertEqual(reacquired.identity_id, first[1].identity_id)
        self.assertEqual(reacquired.method, "appearance_established_reid")
        self.assertEqual(registry.status()["established_reidentified_count"], 1)

    def test_track_continuity_builds_a_bounded_multi_view_gallery(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        bbox = (0.10, 0.10, 0.40, 0.90)
        front = _avatar_frame(bbox)
        back = _avatar_frame(
            bbox,
            top_color=(40, 210, 210),
            bottom_color=(170, 45, 190),
            band_side="right",
        )
        identity = registry.assign(
            [_detection(1, bbox)],
            front,
            now=1.0,
            source_name="openvino",
        )[1]
        # 第 5 次轨迹观测会刷新描述子，把明显不同的背面视角加入原型库。
        for index in range(4):
            registry.assign(
                [_detection(1, bbox)],
                back,
                now=1.1 + index * 0.1,
                source_name="openvino",
            )
        status = registry.status()
        self.assertGreaterEqual(status["appearance_prototype_count"], 2)
        self.assertLessEqual(status["appearance_prototype_count"], status["max_appearance_prototypes"])

        moved_box = (0.60, 0.10, 0.90, 0.90)
        moved_back = _avatar_frame(
            moved_box,
            top_color=(40, 210, 210),
            bottom_color=(170, 45, 190),
            band_side="right",
        )
        reacquired = registry.assign(
            [_detection(2, moved_box)],
            moved_back,
            now=2.0,
            source_name="openvino",
        )[2]
        self.assertEqual(reacquired.identity_id, identity.identity_id)
        self.assertEqual(reacquired.method, "appearance_reid")

    def test_expired_identity_is_not_reused(self) -> None:
        registry = AvatarIdentityRegistry(
            session_token="test",
            retention_s=1.0,
        )
        first_box = (0.05, 0.20, 0.30, 0.80)
        second_box = (0.65, 0.20, 0.90, 0.80)
        first = registry.assign(
            [_detection(1, first_box)],
            _avatar_frame(first_box),
            now=1.0,
            source_name="openvino",
        )[1]
        second = registry.assign(
            [_detection(2, second_box)],
            _avatar_frame(second_box),
            now=2.1,
            source_name="openvino",
        )[2]

        self.assertNotEqual(second.identity_id, first.identity_id)
        self.assertEqual(second.method, "new_identity")

    def test_non_image_frame_safely_falls_back_without_assignment(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        assignments = registry.assign(
            [_detection(1, (0.1, 0.1, 0.5, 0.9))],
            object(),
            now=1.0,
            source_name="openvino",
        )
        self.assertEqual(assignments, {})

    def test_bound_track_keeps_identity_when_one_descriptor_is_unavailable(self) -> None:
        registry = AvatarIdentityRegistry(session_token="test")
        bbox = (0.10, 0.10, 0.40, 0.90)
        first = registry.assign(
            [_detection(1, bbox)],
            _avatar_frame(bbox),
            now=1.0,
            source_name="openvino",
        )[1]
        second = registry.assign(
            [_detection(1, bbox)],
            object(),
            now=1.1,
            source_name="openvino",
        )[1]

        self.assertEqual(second.identity_id, first.identity_id)
        self.assertEqual(second.method, "track_continuity")
        self.assertIsNone(second.similarity)


@unittest.skipIf(np is None, "numpy 是本地视觉的可选依赖")
class AvatarIdentityDetectorIntegrationTests(unittest.TestCase):
    def test_detector_publishes_same_entity_after_iou_track_changes(self) -> None:
        first_box = (0.05, 0.20, 0.30, 0.80)
        second_box = (0.65, 0.20, 0.90, 0.80)
        outputs = [
            {"detections": [{"label": "person", "confidence": 0.95, "bbox": first_box}]},
            {"detections": [{"label": "person", "confidence": 0.95, "bbox": second_box}]},
        ]
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: outputs.pop(0),
            track_ttl_s=0.5,
            identity_reid_retention_s=60.0,
        )

        first = detector.observe(_avatar_frame(first_box), now=1.0).entities[0]
        second = detector.observe(_avatar_frame(second_box), now=2.0).entities[0]

        self.assertEqual(first["id"], second["id"])
        self.assertTrue(first["id"].startswith("avatar:session:"))
        self.assertNotEqual(
            first["attributes"]["track_entity_id"],
            second["attributes"]["track_entity_id"],
        )
        self.assertEqual(second["attributes"]["identity_method"], "appearance_reid")
        self.assertEqual(detector.status()["identity_reid"]["reidentified_count"], 1)

    def test_disabled_reid_preserves_track_entity_id(self) -> None:
        bbox = (0.10, 0.10, 0.40, 0.90)
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [{"label": "person", "confidence": 0.95, "bbox": bbox}],
            },
            identity_reid_enabled=False,
        )

        entity = detector.observe(_avatar_frame(bbox), now=1.0).entities[0]

        self.assertEqual(entity["id"], "openvino:track:1")
        self.assertEqual(entity["attributes"]["identity_scope"], "track")

    def test_non_avatar_classes_keep_existing_track_id_contract(self) -> None:
        bbox = (0.10, 0.10, 0.40, 0.90)
        detector = OpenVinoLocalDetector(
            infer=lambda _frame: {
                "detections": [{"label": "cup", "confidence": 0.95, "bbox": bbox}],
            },
        )

        entity = detector.observe(_avatar_frame(bbox), now=1.0).entities[0]

        self.assertEqual(entity["id"], "openvino:track:1")
        self.assertEqual(entity["attributes"]["identity_method"], "appearance_unavailable")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
