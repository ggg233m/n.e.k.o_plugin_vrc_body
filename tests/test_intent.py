"""独立意图 API 的认证、结构化输出与脱敏契约。"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.config import YuiIntentModelConfig
from yui_npc_controller.runtime.intent import (
    AutonomyIntentProvider,
    IntentModelError,
    validate_intent,
)


def _context() -> dict:
    return {
        "reason": "startup",
        "location": {"region_key": "home", "nearest_anchor": "spawn"},
        "players": [{"slot": 2, "distance_m": 3.0, "bearing_deg": 10.0}],
        "catalog": {
            "anchor": [
                {
                    "semantic_key": "spawn",
                    "region_key": "home",
                    "description_zh": "出生点",
                    "tags": ["quiet"],
                },
                {
                    "semantic_key": "window",
                    "region_key": "home",
                    "description_zh": "窗边",
                    "tags": ["quiet", "view"],
                },
            ],
            "region": [
                {
                    "semantic_key": "home",
                    "description_zh": "家",
                    "tags": ["quiet"],
                    "explorable": True,
                }
            ],
            "entity": [{"semantic_key": "lamp", "description_zh": "落地灯"}],
            "action": [{"semantic_key": "nod", "description_zh": "点头"}],
        },
        "recent_targets": [],
        "recent_regions": [],
        "movement_ratio": 0.5,
    }


def _intent(**overrides) -> dict:
    value = {
        "motivation": "想去窗边安静看看，再留在原地休息一会儿。",
        "mood": "quiet",
        "activities": [
            {
                "kind": "visit",
                "target_key": "window",
                "duration_s": 8,
            },
            {
                "kind": "linger",
                "tags": ["quiet"],
                "duration_s": 10,
            },
        ],
        "avoid_targets": ["spawn"],
        "ttl_s": 240,
    }
    value.update(overrides)
    return value


def _envelope(intent: dict) -> bytes:
    return json.dumps({
        "choices": [{"message": {"content": json.dumps(intent, ensure_ascii=False)}}]
    }, ensure_ascii=False).encode("utf-8")


class IntentValidationTests(unittest.TestCase):
    def test_valid_intent_is_normalized(self) -> None:
        value = validate_intent(_intent(), _context())
        self.assertEqual(value["activities"][0]["target_key"], "window")
        self.assertEqual(value["ttl_s"], 240)

    def test_unknown_semantic_target_is_rejected(self) -> None:
        invalid = _intent()
        invalid["activities"][0]["target_key"] = "made_up_coordinate"
        with self.assertRaisesRegex(IntentModelError, "unknown_target"):
            validate_intent(invalid, _context())

    def test_unknown_fields_and_markdown_are_not_tolerated(self) -> None:
        invalid = _intent(debug_reasoning="hidden chain")
        with self.assertRaisesRegex(IntentModelError, "invalid_root_fields"):
            validate_intent(invalid, _context())

    def test_interests_observe_and_local_roam_are_strictly_normalized(self) -> None:
        value = validate_intent(_intent(
            activities=[
                {"kind": "observe", "target_key": "lamp", "duration_s": 12},
                {"kind": "local_roam", "style": "small_loop", "duration_s": 15},
            ],
            interests=[{"target_key": "window", "strength": 0.8, "ttl_s": 300}],
        ), _context())
        self.assertEqual(value["activities"][0]["target_key"], "lamp")
        self.assertEqual(value["activities"][1]["style"], "small_loop")
        self.assertEqual(value["interests"][0]["strength"], 0.8)

    def test_local_roam_and_interest_cannot_escape_semantic_contract(self) -> None:
        invalid_style = _intent(activities=[
            {"kind": "local_roam", "style": "turn_137_degrees", "duration_s": 10},
            {"kind": "linger", "duration_s": 10},
        ])
        with self.assertRaisesRegex(IntentModelError, "invalid_local_roam_style"):
            validate_intent(invalid_style, _context())
        with self.assertRaisesRegex(IntentModelError, "unknown_target"):
            validate_intent(_intent(
                interests=[{"target_key": "absolute:1,2,3", "strength": 1, "ttl_s": 120}],
            ), _context())


class IntentProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = YuiIntentModelConfig(
            enabled=True,
            endpoint="https://relay.example.com/v1/chat/completions",
        )

    def test_request_uses_test_api_without_exposing_it_in_status(self) -> None:
        calls = []

        def http_post(endpoint, headers, body, timeout_s, max_bytes):
            calls.append((endpoint, dict(headers), json.loads(body), timeout_s, max_bytes))
            return 200, _envelope(_intent())

        with patch.dict(os.environ, {"TEST_API": "super-secret-value"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=http_post)
            result = asyncio.run(provider.request(_context()))
            status = provider.status()

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(calls[0][0], "https://relay.example.com/v1/chat/completions")
        self.assertEqual(calls[0][1]["Authorization"], "Bearer super-secret-value")
        self.assertEqual(calls[0][2]["model"], "gemini-3.7-flash")
        self.assertEqual(calls[0][2]["temperature"], 0.7)
        self.assertEqual(calls[0][2]["response_format"]["type"], "json_schema")
        self.assertNotIn("super-secret-value", json.dumps(status, ensure_ascii=False))
        self.assertEqual(status["key_state"], "present")

    def test_recent_chat_is_sent_only_inside_untrusted_action_context(self) -> None:
        calls = []
        context = _context()
        context["recent_conversation"] = {
            "source": "recent_file",
            "untrusted": True,
            "turns": [{"user": "最近想看灯", "assistant": "可以去看看"}],
        }

        def http_post(_endpoint, _headers, body, _timeout_s, _max_bytes):
            calls.append(json.loads(body))
            return 200, _envelope(_intent())

        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=http_post)
            result = asyncio.run(provider.request(context))

        self.assertEqual(result["status"], "succeeded")
        sent_context = json.loads(calls[0]["messages"][1]["content"])
        self.assertTrue(sent_context["recent_conversation"]["untrusted"])
        self.assertEqual(sent_context["recent_conversation"]["turns"][0]["user"], "最近想看灯")
        self.assertIn('"strength":0.8', calls[0]["messages"][0]["content"])
        self.assertIn("没有明确兴趣时返回空数组", calls[0]["messages"][0]["content"])
        self.assertNotIn("最近想看灯", json.dumps(provider.status(), ensure_ascii=False))

    def test_schema_unsupported_retries_once_as_json_object(self) -> None:
        formats = []

        def http_post(_endpoint, _headers, body, _timeout_s, _max_bytes):
            request = json.loads(body)
            formats.append(request["response_format"]["type"])
            if len(formats) == 1:
                return 400, b'{"error":"response_format json_schema not supported"}'
            return 200, _envelope(_intent())

        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=http_post)
            result = asyncio.run(provider.request(_context()))

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["format"], "json_object")
        self.assertEqual(formats, ["json_schema", "json_object"])
        self.assertEqual(provider.status()["schema_fallbacks"], 1)

    def test_empty_endpoint_stays_unconfigured_without_network_call(self) -> None:
        called = []
        config = YuiIntentModelConfig(enabled=True, endpoint="")
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(
                config,
                http_post=lambda *_args: called.append(True) or (200, b"{}"),
            )
            result = asyncio.run(provider.request(_context()))
            status = provider.status()

        self.assertFalse(provider.configured())
        self.assertEqual(result, {"status": "failed", "error": "not_configured"})
        self.assertEqual(called, [])
        self.assertIsNone(status["endpoint_origin"])

    def test_missing_key_fails_without_network_call(self) -> None:
        called = []
        with patch.dict(os.environ, {}, clear=True):
            provider = AutonomyIntentProvider(
                self.config,
                http_post=lambda *_args: called.append(True) or (200, b"{}"),
            )
            result = asyncio.run(provider.request(_context()))

        self.assertEqual(result, {"status": "failed", "error": "missing_api_key"})
        self.assertEqual(called, [])
        self.assertEqual(provider.status()["key_state"], "missing")

    def test_markdown_response_is_rejected_without_returning_body(self) -> None:
        body = json.dumps({
            "choices": [{"message": {"content": "```json\n{}\n```"}}]
        }).encode("utf-8")
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(
                self.config,
                http_post=lambda *_args: (200, body),
            )
            result = asyncio.run(provider.request(_context()))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "invalid_json")
        self.assertNotIn("```", json.dumps(result))

    def test_concurrent_request_is_rejected_without_second_http_call(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def http_post(*_args):
            calls.append(True)
            entered.set()
            release.wait(timeout=2.0)
            return 200, _envelope(_intent())

        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=http_post)
            first_result = {}
            thread = threading.Thread(
                target=lambda: first_result.update(asyncio.run(provider.request(_context())))
            )
            thread.start()
            self.assertTrue(entered.wait(timeout=1.0))
            second = asyncio.run(provider.request(_context()))
            release.set()
            thread.join(timeout=2.0)

        self.assertEqual(second, {"status": "failed", "error": "request_busy"})
        self.assertEqual(first_result["status"], "succeeded")
        self.assertEqual(len(calls), 1)

    def test_socket_timeout_uses_stable_timeout_error(self) -> None:
        with (
            patch.dict(os.environ, {"TEST_API": "secret"}, clear=False),
            patch("yui_npc_controller.runtime.intent.urlopen", side_effect=TimeoutError),
        ):
            provider = AutonomyIntentProvider(self.config)
            result = asyncio.run(provider.request(_context()))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "timeout")


if __name__ == "__main__":
    unittest.main()
