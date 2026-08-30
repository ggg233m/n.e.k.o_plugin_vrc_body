"""Unity v1.3 路由、移动器与火柴盒生成器的静态契约测试。"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "unity" / "Assets" / "NEKO" / "Npc" / "NekoMidiRouter.cs"
LOCOMOTION = ROOT / "unity" / "Assets" / "NEKO" / "Npc" / "NekoNpcLocomotion.cs"
BUILDER = ROOT / "unity" / "Assets" / "NEKO" / "Editor" / "NekoYuiFullNpcBuilder.cs"


class UnityV13ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.router = ROUTER.read_text(encoding="utf-8-sig")
        cls.locomotion = LOCOMOTION.read_text(encoding="utf-8-sig")
        cls.builder = BUILDER.read_text(encoding="utf-8-sig")

    def test_new_commands_and_capabilities_are_published(self) -> None:
        self.assertIn("CMD_MOVE_RELATIVE = 0x19", self.router)
        self.assertIn("CMD_EXPLORE_REGION = 0x1A", self.router)
        self.assertIn("1 << 19", self.router)
        self.assertIn("1 << 20", self.router)
        self.assertIn('result += "\\\"region_localization\\\""', self.router)
        self.assertIn('result += "\\\"local_navigation\\\""', self.router)

    def test_location_projection_contains_semantics_but_no_coordinates(self) -> None:
        start = self.router.index("public string BuildLocationJson")
        end = self.router.index("public string ActiveOpsJson", start)
        method = self.router[start:end]
        for field in ("localized", "region_key", "floor_label", "nearest_anchor", "semantic_key", "d", "brg"):
            self.assertIn(field, method)
        self.assertNotIn('"pos"', method)
        self.assertNotIn('"center"', method)

    def test_region_overlap_order_and_rotation_validation_are_explicit(self) -> None:
        localized = self.router[self.router.index("private int LocalizedRegion"):]
        self.assertIn("priority > bestPriority", localized)
        self.assertIn("size < bestVolume", localized)
        self.assertIn("regionId < bestRegion", localized)
        self.assertIn("euler.x", self.router)
        self.assertIn("euler.z", self.router)

    def test_relative_navigation_keeps_bearing_and_shortens_only_by_policy(self) -> None:
        self.assertIn("public string StartRelative", self.locomotion)
        self.assertIn("requested * (1f - i * 0.2f)", self.locomotion)
        self.assertIn("lateral > 0.08f", self.locomotion)
        self.assertIn("path.status != NavMeshPathStatus.PathComplete", self.locomotion)

    def test_exploration_is_one_continuous_operation(self) -> None:
        self.assertIn("public string StartExplore", self.locomotion)
        self.assertIn("private const float ExploreSwitchDistance = 0.4f", self.locomotion)
        self.assertIn("navAgent.autoBraking = false", self.locomotion)
        self.assertIn("generated < 12", self.locomotion)
        self.assertIn("new Vector3[8]", self.locomotion)
        self.assertIn("_explorePendingStart = true", self.locomotion)
        self.assertIn("private void BeginExploreMovement", self.locomotion)
        self.assertIn('telemetry.Emit("npc.operation_failed"', self.router)

    def test_matchbox_publishes_three_regions_and_upper_patrol_points(self) -> None:
        self.assertIn('new[] { "ground_floor", "stairway", "upper_floor" }', self.builder)
        self.assertIn("new[] { true, false, true }", self.builder)
        self.assertIn('"stairway", "upper_floor", "upper_floor"', self.builder)
        self.assertIn("regionVolumePriorities = new[] { 0, 10, 0 }", self.builder)
        self.assertIn("ConfigureMatchboxV13Batch", self.builder)
        self.assertIn("ValidateOpenSceneV13Batch", self.builder)

    def test_yui_stack_has_no_anydance_or_yolo_dependency(self) -> None:
        combined = "\n".join((self.router, self.locomotion, self.builder)).lower()
        self.assertNotIn("using anydance", combined)
        self.assertNotIn("using yolo", combined)


if __name__ == "__main__":
    unittest.main()
