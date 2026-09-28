"""W1 世界身份子系统的单元测试。

只覆盖本期能确定实现的部分：world_key 归一化与可信度分级、memory_partition 的
「拒绝加载」语义、启停后台线程与状态字段。深度模型/地点识别/路径规划不在本期，
对应的接口留空，这里不测伪造状态。
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401  registers the neko_anyadance_body namespace
from neko_anyadance_body.backend import world_model


def _new_model(enabled: bool = True, persist: bool = True) -> tuple[world_model.WorldModel, Path]:
    state_dir = Path(tempfile.mkdtemp(prefix="world_model_test_"))
    model = world_model.WorldModel(enabled=enabled, persist=persist, state_dir=state_dir)
    return model, state_dir


class WorldModelNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.state_dir = _new_model()

    def tearDown(self) -> None:
        shutil.rmtree(self.state_dir, ignore_errors=True)

    def test_wrld_id_is_normalized_to_lowercase_manual_id(self) -> None:
        result = self.model.set_world("  WRLD_AbC123  ")
        self.assertEqual(result["world_key"], "wrld_abc123")
        self.assertEqual(result["world_source"], "manual_id")
        self.assertFalse(result["world_conflict_risk"])
        status = self.model.status()
        self.assertEqual(status["world_key"], "wrld_abc123")
        self.assertEqual(status["world_name"], None)
        self.assertFalse(status["world_conflict_risk"])

    def test_world_name_is_manual_name_with_conflict_risk(self) -> None:
        result = self.model.set_world("My Cool Club")
        self.assertEqual(result["world_key"], "My Cool Club")
        self.assertEqual(result["world_source"], "manual_name")
        self.assertTrue(result["world_conflict_risk"])
        status = self.model.status()
        self.assertEqual(status["world_name"], "My Cool Club")
        self.assertTrue(status["world_conflict_risk"])

    def test_name_with_explicit_display_name(self) -> None:
        result = self.model.set_world("wrld_xyz", "The Club")
        self.assertEqual(result["world_key"], "wrld_xyz")
        self.assertEqual(result["world_source"], "manual_id")
        self.assertEqual(self.model.status()["world_name"], "The Club")

    def test_empty_key_clears_identity(self) -> None:
        self.model.set_world("wrld_abc")
        result = self.model.set_world("")
        self.assertIsNone(result["world_key"])
        self.assertEqual(result["world_source"], "unknown")
        self.assertFalse(result["world_conflict_risk"])
        self.assertIsNone(self.model.status()["world_key"])
        self.assertIsNone(self.model.status()["world_name"])

    def test_none_key_clears_identity(self) -> None:
        self.model.set_world("Some World")
        result = self.model.set_world(None)
        self.assertIsNone(result["world_key"])
        self.assertEqual(result["world_source"], "unknown")


class WorldModelPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.state_dir = _new_model()

    def tearDown(self) -> None:
        shutil.rmtree(self.state_dir, ignore_errors=True)

    def test_partition_rejected_when_key_unknown(self) -> None:
        # 最关键的失效模式：没有 world_key 时绝不能返回任何分区路径，否则上层会
        # 拿别的世界的记忆凑合。
        self.assertIsNone(self.model.memory_partition())
        self.assertIsNone(self.model.status()["memory_partition"])

    def test_partition_path_for_id(self) -> None:
        self.model.set_world("wrld_abc")
        partition = self.model.memory_partition()
        self.assertIsNotNone(partition)
        self.assertTrue(partition.endswith("world_memory_wrld_abc.json"))
        self.assertEqual(Path(partition).parent, self.state_dir)

    def test_partition_path_sanitizes_names(self) -> None:
        self.model.set_world("Weird/Name\\With:Chars")
        partition = self.model.memory_partition()
        self.assertIsNotNone(partition)
        # 分隔符只来自所在目录；key 本身被规整成安全文件名段，不残留路径字符。
        filename = Path(partition).name
        self.assertNotIn("/", filename)
        self.assertNotIn("\\", filename)
        self.assertNotIn(":", filename)
        self.assertTrue(filename.startswith("world_memory_"))
        self.assertTrue(filename.endswith(".json"))


class WorldModelLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.state_dir = _new_model()

    def tearDown(self) -> None:
        shutil.rmtree(self.state_dir, ignore_errors=True)

    def test_disabled_status_does_not_throw(self) -> None:
        model, _ = _new_model(enabled=False)
        status = model.status()
        self.assertFalse(status["available"])
        self.assertFalse(status["running"])
        self.assertFalse(status["starting"])
        # 未运行时不伪造定位能力。
        self.assertEqual(status["localization"], "not_running")

    def test_start_when_disabled_is_rejected(self) -> None:
        model, _ = _new_model(enabled=False)
        result = model.start()
        self.assertFalse(result["accepted"])
        self.assertIn("disabled", result["reason"])

    def test_start_runs_in_background_and_sets_running(self) -> None:
        result = self.model.start()
        self.assertTrue(result["accepted"])
        # 启动立即返回，后台线程稍后把 running 置位；等它完成。
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.model.status()["running"]:
                break
            time.sleep(0.01)
        status = self.model.status()
        self.assertFalse(status["starting"])
        self.assertTrue(status["running"])
        self.assertEqual(status["localization"], "unknown")
        self.assertEqual(status["places"], 0)
        self.assertEqual(status["edges"], 0)

    def test_stop_is_idempotent(self) -> None:
        self.model.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not self.model.status()["running"]:
            time.sleep(0.01)
        stopped = self.model.stop(reason="test")
        self.assertTrue(stopped["accepted"])
        self.assertFalse(self.model.status()["running"])
        # 再次 stop 也不抛异常。
        again = self.model.stop()
        self.assertTrue(again["accepted"])

    def test_set_world_allowed_without_running(self) -> None:
        # 世界身份不依赖子系统运行；未启动时也应能设置。
        self.assertFalse(self.model.status()["running"])
        result = self.model.set_world("wrld_standalone")
        self.assertEqual(result["world_key"], "wrld_standalone")


if __name__ == "__main__":
    unittest.main()
