from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.process import BackendHttpServer
from neko_anyadance_body.backend.webui import (
    StandaloneConfigStore,
    deep_merge,
    load_settings_file,
)


class StandaloneConfigStoreTests(unittest.TestCase):
    def test_settings_are_validated_persisted_and_never_contain_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backend.settings.json"
            store = StandaloneConfigStore(
                {
                    "vision": {"enabled": True},
                    "world_memory": {"persist_world": False},
                },
                settings_path=path,
                editable=True,
                mode="standalone",
                source="plugin.toml",
                offline=False,
            )
            with patch.dict(os.environ, {"VRC_VLM_API_KEY": "super-secret"}, clear=False):
                before = store.snapshot()
                self.assertTrue(before["secrets"]["vlm_api_key"])
                self.assertNotIn("super-secret", json.dumps(before))

                result = store.save({
                    "vision": {
                        "semantic_endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                        "semantic_model": "local-vlm",
                        "semantic_max_per_minute": 12,
                    },
                    "world_memory": {"persist_world": False, "persist_players": False},
                })

            self.assertTrue(result["restart_required"])
            self.assertTrue(path.exists())
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", raw)
            self.assertNotIn("api_key", json.dumps(load_settings_file(path)))
            self.assertEqual(
                load_settings_file(path)["vision"]["semantic_model"],
                "local-vlm",
            )

    def test_unknown_or_secret_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StandaloneConfigStore(
                {},
                settings_path=Path(directory) / "settings.json",
                editable=True,
                mode="standalone",
                source="defaults",
                offline=False,
            )
            with self.assertRaisesRegex(ValueError, "not editable"):
                store.save({"safety": {"max_y_m": 2.0}})
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                store.save({"vision": {"api_key": "must-not-be-stored"}})
            with self.assertRaisesRegex(ValueError, "http\(s\)"):
                store.save({"vision": {"semantic_endpoint": "file:///secret"}})

    def test_managed_mode_is_read_only(self) -> None:
        store = StandaloneConfigStore(
            {},
            settings_path=None,
            editable=False,
            mode="managed",
            source="config-json",
            offline=False,
        )
        self.assertFalse(store.snapshot()["editable"])
        with self.assertRaises(PermissionError):
            store.save({"vision": {"enabled": True}})

    def test_deep_merge_keeps_unrelated_base_sections(self) -> None:
        base = {"vision": {"enabled": True, "device": "GPU"}, "input": {"rate_hz": 120}}
        merged = deep_merge(base, {"vision": {"device": "CPU"}})
        self.assertEqual(merged["vision"], {"enabled": True, "device": "CPU"})
        self.assertEqual(merged["input"], {"rate_hz": 120})
        self.assertEqual(base["vision"]["device"], "GPU")


class StandaloneHttpUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        store = StandaloneConfigStore(
            {},
            settings_path=Path(self.temporary.name) / "settings.json",
            editable=True,
            mode="standalone",
            source="defaults",
            offline=True,
        )
        self.server = BackendHttpServer(
            ("127.0.0.1", 0),
            "test-token",
            object(),  # 静态 UI 与配置路由不需要构造完整运行时。
            config_store=store,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.temporary.cleanup()

    def test_static_ui_is_public_but_config_api_requires_token(self) -> None:
        html = urlopen(self.base + "/", timeout=2.0).read().decode("utf-8")
        self.assertIn("VRChat 独立后端", html)
        self.assertIn("/ui/app.js", html)

        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base + "/config", timeout=2.0)
        self.assertEqual(raised.exception.code, 401)

        request = Request(
            self.base + "/config",
            headers={"X-Neko-Backend-Token": "test-token"},
        )
        payload = json.loads(urlopen(request, timeout=2.0).read())
        self.assertTrue(payload["editable"])
        self.assertEqual(payload["secret_policy"], "environment_only")

    def test_config_post_validates_and_sets_restart_required(self) -> None:
        request = Request(
            self.base + "/config",
            method="POST",
            headers={
                "X-Neko-Backend-Token": "test-token",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "config": {
                    "vision": {
                        "semantic_endpoint": "https://example.invalid/v1/chat/completions",
                    }
                }
            }).encode("utf-8"),
        )
        payload = json.loads(urlopen(request, timeout=2.0).read())
        self.assertTrue(payload["restart_required"])
        self.assertEqual(
            payload["config"]["vision"]["semantic_endpoint"],
            "https://example.invalid/v1/chat/completions",
        )

    def test_main_llm_semantic_http_bridge_requires_token_and_forwards_revision(self) -> None:
        calls = []

        class Service:
            def main_llm_semantic_request(self, after_request_id=None):
                calls.append(("request", after_request_id))
                return {"available": True, "request_id": "semantic-request:test:2"}

            def main_llm_semantic_commit(self, request_id, frame_revision, entities):
                calls.append(("commit", request_id, frame_revision, entities))
                return {"accepted": True}

            def record_control_dispatch(self, operation, started_at):
                return 0.1

        self.server.service = Service()
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base + "/semantic/request", timeout=2.0)
        self.assertEqual(raised.exception.code, 401)

        get_request = Request(
            self.base + "/semantic/request?after_request_id=semantic-request%3Atest%3A1",
            headers={"X-Neko-Backend-Token": "test-token"},
        )
        payload = json.loads(urlopen(get_request, timeout=2.0).read())
        self.assertTrue(payload["available"])

        post_request = Request(
            self.base + "/semantic/commit",
            method="POST",
            headers={
                "X-Neko-Backend-Token": "test-token",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "request_id": "semantic-request:test:2",
                "frame_revision": 42,
                "entities": [],
            }).encode("utf-8"),
        )
        committed = json.loads(urlopen(post_request, timeout=2.0).read())
        self.assertTrue(committed["accepted"])
        self.assertEqual(calls[0], ("request", "semantic-request:test:1"))
        self.assertEqual(calls[1], ("commit", "semantic-request:test:2", 42, []))

    def test_navmesh_coverage_endpoint_requires_token_and_assets_served(self) -> None:
        class Service:
            def navmesh_coverage(self):
                return {"available": False, "reason": "no_map_yet"}

        self.server.service = Service()
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base + "/worldmodel/navmesh/coverage", timeout=2.0)
        self.assertEqual(raised.exception.code, 401)

        request = Request(self.base + "/worldmodel/navmesh/coverage",
                          headers={"X-Neko-Backend-Token": "test-token"})
        payload = json.loads(urlopen(request, timeout=2.0).read())
        self.assertFalse(payload["available"])
        self.assertEqual(payload["reason"], "no_map_yet")

        # 覆盖页三件套已注册（CSP style-src 'self' 禁内联，样式必须走文件）。
        for path, ctype in (("/coverage", "text/html"), ("/ui/coverage.js", "text/javascript"),
                            ("/ui/coverage.css", "text/css")):
            with urlopen(self.base + path, timeout=2.0) as resp:
                self.assertTrue(resp.headers["Content-Type"].startswith(ctype), path)


class UiAssetSyntaxTests(unittest.TestCase):
    """standalone_ui 的 JS 必须能解析。

    一个字符串里未转义的 ASCII 引号就让整份脚本不执行——页面看着正常，但所有按钮
    都没反应、状态永远停在 "—"（2026-10-05 实况：navmesh.js 的 LAYER_NOTE 里
    写了 ``分不出"障碍还是楼板"``）。node 可用时逐个跑 ``node --check``，否则跳过。
    """

    def test_js_files_parse(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node 不在 PATH，跳过 JS 语法校验")
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        for js in sorted(ui.glob("*.js")):
            proc = subprocess.run([node, "--check", str(js)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0,
                             f"{js.name} 语法错误：{proc.stderr.strip()[:300]}")


class ScriptModeImportTests(unittest.TestCase):
    """process.py 必须同时支持「宿主包导入」与「脚本直跑」（``python backend/process.py``）。

    函数里的相对导入在包导入时正常、脚本直跑时抛 ``ImportError`` —— 而且只炸那一个
    HTTP 端点（2026-10-05 实况：``/worldmodel/navmesh`` 空回复 ⇒ navmesh 页面全空）。
    用 AST 钉住：整份文件不允许出现相对导入；跨模块引用一律走模块级 importlib + PACKAGE_NAME。
    """

    def test_no_relative_imports_in_process_module(self) -> None:
        src = Path(__file__).resolve().parents[1] / "backend" / "process.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        bad = [node.lineno for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.level > 0]
        self.assertEqual(bad, [], f"process.py 出现相对导入（脚本直跑会 ImportError）：行 {bad}")


if __name__ == "__main__":
    unittest.main()
