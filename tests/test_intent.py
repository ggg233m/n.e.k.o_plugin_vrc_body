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
        self.assertEqual(provider.panel_status()["requests"], 1)
        self.assertEqual(provider.panel_status()["http_requests"], 2)
        self.assertEqual(provider.panel_status()["last_call"]["format"], "json_object")

    def test_panel_records_output_and_failure_without_leaking_to_status(self) -> None:
        responses = [(200, _envelope(_intent())), (503, b"private server error")]
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=lambda *_: responses.pop(0))
            asyncio.run(provider.request(_context()))
            panel = provider.panel_status()
            self.assertEqual(json.loads(panel["last_call"]["output"]), _intent())
            self.assertEqual(panel["last_call"]["status"], "succeeded")
            self.assertNotIn("last_call", provider.status())
            panel["last_call"]["output"] = "changed by caller"
            self.assertNotEqual(provider.panel_status()["last_call"]["output"], "changed by caller")
            asyncio.run(provider.request(_context()))
            panel = provider.panel_status()
        self.assertEqual((panel["requests"], panel["http_requests"], panel["successes"], panel["failures"]), (2, 2, 1, 1))
        self.assertEqual(panel["last_call"]["number"], 2)
        self.assertEqual(panel["last_call"]["error"], "http_503")
        self.assertIsNone(panel["last_call"]["output"])
        self.assertNotIn("private server error", json.dumps(panel))

    def test_panel_keeps_invalid_output_bounded_and_redacted(self) -> None:
        raw = "secret" + "x" * 9000
        body = json.dumps({"choices": [{"message": {"content": raw}}]}).encode()
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=lambda *_: (200, body))
            asyncio.run(provider.request(_context()))
            panel = provider.panel_status()
        call = panel["last_call"]
        self.assertEqual(call["error"], "invalid_json")
        self.assertTrue(call["truncated"])
        self.assertEqual(len(call["output"]), 8000)
        self.assertNotIn("secret", json.dumps(panel))
        self.assertNotIn("output", provider.status())

    def test_history_is_bounded_and_dispositions_follow_request_token(self) -> None:
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=lambda *_: (200, _envelope(_intent())))
            for number in range(23):
                asyncio.run(provider.request(_context(), request_token=f"token-{number}"))
            provider.record_disposition("token-3", "expired")
            provider.record_disposition("token-22", "queued")
            history = provider.panel_status()["history"]
        self.assertEqual(len(history), 20)
        self.assertEqual([item["number"] for item in history], list(range(23, 3, -1)))
        self.assertEqual(history[0]["disposition"], "queued")
        self.assertEqual(history[-1]["disposition"], "expired")
        self.assertNotIn("token-", json.dumps(history))
        history[0]["disposition"] = "changed"
        self.assertEqual(provider.panel_status()["last_call"]["disposition"], "queued")

    def test_usage_counts_failed_validation_and_preserves_missing_fields(self) -> None:
        responses = [
            {"usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}, "choices": [{"message": {"content": "invalid"}}]},
            {"choices": [{"message": {"content": "invalid again"}}]},
            json.loads(_envelope(_intent())),
            {**json.loads(_envelope(_intent())), "usage": {"prompt_tokens": 0, "completion_tokens": True, "total_tokens": -1}},
        ]
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=lambda *_: (200, json.dumps(responses.pop(0)).encode()))
            asyncio.run(provider.request(_context()))
            asyncio.run(provider.request(_context()))
            missing = provider.panel_status()["last_call"]["usage"]
            self.assertTrue(all(value is None for value in missing.values()))
            asyncio.run(provider.request(_context()))
            panel = provider.panel_status()
        self.assertEqual(panel["usage_totals"], {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15})
        self.assertEqual(panel["usage_reported_requests"], 2)
        self.assertEqual(panel["last_call"]["usage"], {"input_tokens": 0, "output_tokens": None, "total_tokens": None})
        self.assertEqual(panel["history"][-1]["status"], "failed")

    def test_reference_constraints_and_one_correction_keep_strict_validation(self) -> None:
        invalid = _intent(avoid_targets=["invented-place"])
        responses = [_envelope(invalid), _envelope(_intent())]
        bodies = []
        def post(_endpoint, _headers, body, *_args):
            bodies.append(json.loads(body))
            return 200, responses.pop(0)
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=post)
            result = asyncio.run(provider.request(_context()))
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(provider.status()["requests"], 1)
        self.assertEqual(provider.status()["http_requests"], 2)
        self.assertEqual(provider.status()["validation_retries"], 1)
        schema = bodies[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["activities"]["items"]["properties"]["target_key"]["enum"], ["home", "lamp", "spawn", "window"])
        correction = json.loads(bodies[1]["messages"][1]["content"])
        self.assertEqual(correction["validation_feedback"]["error"], "unknown_target")
        self.assertEqual(correction["reference_options"]["explorable_regions"], ["home"])
        self.assertNotIn("invented-place", json.dumps(correction))

    def test_player_reference_examples_and_targeted_correction(self) -> None:
        invalid = _intent(activities=[{"kind": "observe", "target_key": "player_slot_2", "duration_s": 10}, {"kind": "linger", "duration_s": 10}])
        corrected = _intent(activities=[{"kind": "observe", "player_slot": 2, "duration_s": 10}, {"kind": "socialize", "player_slot": 2, "duration_s": 10}])
        responses = [_envelope(invalid), _envelope(corrected)]
        bodies = []
        def post(_endpoint, _headers, body, *_args):
            bodies.append(json.loads(body))
            return 200, responses.pop(0)
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=post)
            result = asyncio.run(provider.request(_context()))
        self.assertEqual(result["status"], "succeeded")
        context = json.loads(bodies[1]["messages"][1]["content"])
        self.assertEqual(context["validation_feedback"]["paths"], ["activities[0].target_key"])
        self.assertIn("player_slot", context["validation_feedback"]["repair"])
        self.assertNotIn("player_slot_2", json.dumps(context))
        for example in context["activity_examples"]:
            validate_intent(_intent(activities=[example, {"kind": "linger", "duration_s": 10}]), _context())

    def test_player_slot_requires_integer_and_no_players_means_no_player_examples(self) -> None:
        for slot in (2.0, "2", True):
            with self.subTest(slot=slot), self.assertRaises(IntentModelError):
                validate_intent(_intent(activities=[{"kind": "observe", "player_slot": slot, "duration_s": 10}, {"kind": "linger", "duration_s": 10}]), _context())
        context = _context()
        context["players"] = []
        provider = AutonomyIntentProvider(self.config)
        body = json.loads(provider._request_body(context, response_format="json_object"))
        examples = json.loads(body["messages"][1]["content"])["activity_examples"]
        self.assertTrue(all("player_slot" not in example for example in examples))

    def test_schema_fallback_and_correction_share_original_timeout(self) -> None:
        now = [0.0]
        timeouts = []
        responses = [
            (400, b'{"error":"response_format json_schema unsupported"}'),
            (200, _envelope(_intent(avoid_targets=["invented-place"]))),
            (200, _envelope(_intent())),
        ]
        def post(_endpoint, _headers, _body, timeout, _limit):
            timeouts.append(timeout)
            now[0] += 1.0
            return responses.pop(0)
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=post, clock=lambda: now[0])
            result = asyncio.run(provider.request(_context()))
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(timeouts, [self.config.timeout_s - offset for offset in range(3)])
        self.assertEqual(provider.status()["schema_fallbacks"], 1)
        self.assertEqual(provider.status()["validation_retries"], 1)

    def test_second_invalid_target_is_not_silently_replaced(self) -> None:
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=lambda *_: (200, _envelope(_intent(avoid_targets=["invented-place"]))))
            result = asyncio.run(provider.request(_context()))
        self.assertEqual(result["error"], "unknown_target")
        self.assertEqual(provider.status()["http_requests"], 2)
        self.assertNotIn("intent", result)

    def test_usage_includes_schema_retry_response(self) -> None:
        responses = [
            (400, json.dumps({"error": "response_format json_schema unsupported", "usage": {"prompt_tokens": 5, "total_tokens": 5}}).encode()),
            (200, json.dumps({**json.loads(_envelope(_intent())), "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}).encode()),
        ]
        with patch.dict(os.environ, {"TEST_API": "secret"}, clear=False):
            provider = AutonomyIntentProvider(self.config, http_post=lambda *_: responses.pop(0))
            asyncio.run(provider.request(_context()))
            panel = provider.panel_status()
        self.assertEqual(panel["last_call"]["usage"], {"input_tokens": 13, "output_tokens": 2, "total_tokens": 15})
        self.assertEqual(panel["usage_reported_requests"], 2)

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
        self.assertEqual(provider.panel_status()["requests"], 0)
        self.assertEqual(provider.panel_status()["http_requests"], 0)
        self.assertIsNone(provider.panel_status()["last_call"])
        self.assertEqual(provider.panel_status()["configuration_error"], "not_configured")

    def test_missing_key_fails_without_network_call(self) -> None:
        called = []
        with patch.dict(os.environ, {}, clear=True):
            provider = AutonomyIntentProvider(
                self.config,
                http_post=lambda *_args: called.append(True) or (200, b"{}"),
            )
            result = asyncio.run(provider.request(_context()))
            panel = provider.panel_status()

        self.assertEqual(result, {"status": "failed", "error": "missing_api_key"})
        self.assertEqual(called, [])
        self.assertEqual(provider.status()["key_state"], "missing")
        self.assertEqual(panel["configuration_error"], "missing_api_key")

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
