"""N.E.K.O 落盘近期聊天的安全只读解析测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.chat_context import (
    RecentChatContextProvider,
    resolve_neko_runtime_root,
)
from yui_npc_controller.runtime.config import YuiChatContextConfig


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp"


def _write_runtime(root: Path, character: str, rows: list[dict]) -> Path:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "memory" / character).mkdir(parents=True, exist_ok=True)
    (root / "config" / "characters.json").write_text(
        json.dumps({"当前猫娘": character}, ensure_ascii=False),
        encoding="utf-8",
    )
    recent = root / "memory" / character / "recent.json"
    recent.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return recent


class RecentChatContextTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT)
        self.root = Path(self.temporary.name).resolve()
        self.config = YuiChatContextConfig(
            enabled=True,
            max_turns=6,
            max_chars=6000,
            poll_interval_s=1.0,
            max_file_bytes=2 * 1024 * 1024,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_extracts_only_last_complete_human_ai_turns(self) -> None:
        secret = "聊天正文不应进入状态"
        rows = [
            {"type": "system", "data": {"content": "系统提示"}},
            {"type": "ai", "data": {"content": "主动但没有用户回应"}},
            {
                "type": "human",
                "data": {"content": [
                    {"type": "text", "text": secret},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                ]},
            },
            {"type": "ai", "data": {"content": "第一段"}},
            {"type": "ai", "data": {"content": [{"type": "output_text", "text": "第二段"}]}},
            {"type": "human", "data": {"content": "尚未完成"}},
        ]
        _write_runtime(self.root, "然然", rows)
        provider = RecentChatContextProvider(self.config, runtime_root=self.root)

        update = provider.poll(force=True)

        self.assertTrue(update.changed)
        self.assertEqual(provider.context()["turns"], [{
            "user": secret,
            "assistant": "第一段\n第二段",
        }])
        status_text = json.dumps(provider.status(), ensure_ascii=False)
        self.assertNotIn(secret, status_text)
        self.assertEqual(provider.status()["turn_count"], 1)

    def test_keeps_last_six_turns_and_bounds_injected_characters(self) -> None:
        rows = []
        for index in range(9):
            rows.extend([
                {"type": "human", "data": {"content": f"用户{index}-" + "甲" * 80}},
                {"type": "ai", "data": {"content": f"回答{index}-" + "乙" * 80}},
            ])
        _write_runtime(self.root, "然然", rows)
        provider = RecentChatContextProvider(
            YuiChatContextConfig(enabled=True, max_turns=6, max_chars=256),
            runtime_root=self.root,
        )

        provider.poll(force=True)
        turns = provider.context()["turns"]

        self.assertLessEqual(len(turns), 6)
        self.assertLessEqual(sum(len(v) for turn in turns for v in turn.values()), 256)
        self.assertTrue(turns[-1]["user"].startswith("用户8-"))

    def test_repeated_identical_last_turn_still_advances_revision(self) -> None:
        """相同问答再次发生也必须被头顶显示桥识别为新一轮。"""

        repeated = [
            {"type": "human", "data": {"content": "摸摸头"}},
            {"type": "ai", "data": {"content": "好呀。"}},
        ]
        recent = _write_runtime(self.root, "然然", list(repeated))
        provider = RecentChatContextProvider(
            YuiChatContextConfig(enabled=True, max_turns=1),
            runtime_root=self.root,
        )
        first = provider.poll(force=True)

        recent.write_text(
            json.dumps(repeated + repeated, ensure_ascii=False),
            encoding="utf-8",
        )
        second = provider.poll(force=True)

        self.assertTrue(first.changed)
        self.assertTrue(second.changed)
        self.assertNotEqual(first.revision, second.revision)
        self.assertEqual(provider.status()["turn_count"], 1)
        self.assertEqual(provider.context()["turns"], [{
            "user": "摸摸头",
            "assistant": "好呀。",
        }])

    def test_invalid_rewrite_preserves_last_valid_snapshot(self) -> None:
        recent = _write_runtime(self.root, "然然", [
            {"type": "human", "data": {"content": "喜欢窗边"}},
            {"type": "ai", "data": {"content": "记住了"}},
        ])
        provider = RecentChatContextProvider(self.config, runtime_root=self.root)
        provider.poll(force=True)
        valid = provider.context()

        recent.write_text("{half-written", encoding="utf-8")
        update = provider.poll(force=True)

        self.assertFalse(update.changed)
        self.assertEqual(provider.context(), valid)
        self.assertEqual(provider.status()["file_state"], "unreadable")
        self.assertEqual(provider.status()["last_error"], "read_failed")

    def test_character_switch_clears_old_context_even_when_new_file_missing(self) -> None:
        _write_runtime(self.root, "旧角色", [
            {"type": "human", "data": {"content": "旧秘密"}},
            {"type": "ai", "data": {"content": "旧回答"}},
        ])
        provider = RecentChatContextProvider(self.config, runtime_root=self.root)
        provider.poll(force=True)
        (self.root / "config" / "characters.json").write_text(
            json.dumps({"当前猫娘": "新角色"}, ensure_ascii=False),
            encoding="utf-8",
        )

        update = provider.poll(force=True)

        self.assertTrue(update.character_changed)
        self.assertEqual(provider.context()["turns"], [])
        self.assertEqual(provider.status()["current_character"], "新角色")
        self.assertEqual(provider.status()["file_state"], "missing")

    def test_invalid_character_and_oversized_file_are_rejected(self) -> None:
        _write_runtime(self.root, "然然", [])
        characters = self.root / "config" / "characters.json"
        characters.write_text(json.dumps({"当前猫娘": ".."}), encoding="utf-8")
        provider = RecentChatContextProvider(self.config, runtime_root=self.root)
        provider.poll(force=True)
        self.assertEqual(provider.status()["last_error"], "invalid_current_character")

        recent = _write_runtime(self.root, "然然", [])
        recent.write_bytes(b"x" * 1025)
        small = RecentChatContextProvider(
            YuiChatContextConfig(enabled=True, max_file_bytes=1024),
            runtime_root=self.root,
        )
        small.poll(force=True)
        self.assertEqual(small.status()["last_error"], "file_too_large")

    def test_symlink_character_directory_is_rejected(self) -> None:
        real = self.root / "outside"
        real.mkdir(parents=True)
        (real / "recent.json").write_text("[]", encoding="utf-8")
        (self.root / "config").mkdir(parents=True)
        (self.root / "memory").mkdir(parents=True)
        (self.root / "config" / "characters.json").write_text(
            json.dumps({"当前猫娘": "然然"}, ensure_ascii=False),
            encoding="utf-8",
        )
        link = self.root / "memory" / "然然"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"当前 Windows 环境不能创建符号链接：{exc}")
        provider = RecentChatContextProvider(self.config, runtime_root=self.root)

        provider.poll(force=True)

        self.assertIn(provider.status()["last_error"], {"path_outside_memory", "symlink_rejected"})


class RuntimeRootResolutionTests(unittest.TestCase):
    def test_selected_root_wins_then_storage_policy_then_anchor(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            base = Path(temporary).resolve()
            selected = base / "selected"
            policy_selected = base / "policy-selected"
            anchor = base / "anchor"
            (anchor / "state").mkdir(parents=True)
            (anchor / "state" / "storage_policy.json").write_text(
                json.dumps({"selected_root": str(policy_selected)}),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_neko_runtime_root(environ={
                    "NEKO_STORAGE_SELECTED_ROOT": str(selected),
                    "NEKO_STORAGE_ANCHOR_ROOT": str(anchor),
                }),
                selected,
            )
            self.assertEqual(
                resolve_neko_runtime_root(environ={"NEKO_STORAGE_ANCHOR_ROOT": str(anchor)}),
                policy_selected,
            )
            (anchor / "state" / "storage_policy.json").unlink()
            self.assertEqual(
                resolve_neko_runtime_root(environ={"NEKO_STORAGE_ANCHOR_ROOT": str(anchor)}),
                anchor,
            )


if __name__ == "__main__":
    unittest.main()
