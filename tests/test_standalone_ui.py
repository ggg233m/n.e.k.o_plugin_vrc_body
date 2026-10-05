from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import types
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
            # ⚠️ 必须是能挂属性的对象：do_POST 会给 service 写 `_http_client_addr`（turn 账本归因），
            # 用裸 `object()` 会抛 AttributeError 并把连接直接掐掉（客户端看到 RemoteDisconnected，
            # 看起来像"后端崩溃"，实际只是这个假对象不合格）。
            types.SimpleNamespace(),
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

    def test_navmesh_panel_exposes_every_status_block_it_renders(self) -> None:
        """面板行是**固定行**：`status()` 新加的字段没人往 HTML/JS 里写，就永远不会显示。

        2026-10-06 实况：先验消费（`status()["prior"]`）与跨会话检索（`status()["xsession"]`）
        都已接进后端，页面上却一行都没有 ⇒ 只能在 JSON 里看。这条按"后端有字段、前端必须有对应
        id 与读取"钉住，免得下次再漏。
        """
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        html = (ui / "navmesh.html").read_text(encoding="utf-8")
        js = (ui / "navmesh.js").read_text(encoding="utf-8")
        for el_id in ("xsession", "prior"):
            self.assertIn(f'id="{el_id}"', html, f"navmesh.html 少了 {el_id} 行")
            self.assertIn(f'$("{el_id}")', js, f"navmesh.js 没有渲染 {el_id}")
        # 未启用时甩内部码（world_unknown…）等于没说；world_unknown 与 memory_not_configured
        # 要人做的事不同（设世界身份 / 配记忆存储），必须译成人话。
        self.assertIn("world_unknown", js)
        self.assertIn("未设世界身份", js)
        # 采纳是**会话末**才跑的：state=pending 表示"还没轮到"，不是失败，不能画成 ✗
        self.assertIn('al.state === "pending"', js)
        self.assertIn("采纳未跑", js)

    def test_navmesh_layers_match_backend_whitelist(self) -> None:
        """图层下拉的每个 value 都必须是后端 ``_GRID_LAYERS`` 里的真图层。

        HTTP 层对未知 layer **静默回落到 tristate**（``process.py``），所以名字写错的表现是
        "选了新图层、看到的还是三态图"——不报错、查不出。这条把它钉在文本上。
        """
        import re
        from neko_anyadance_body.backend.nav_online import _GRID_LAYERS
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        html = (ui / "navmesh.html").read_text(encoding="utf-8")
        values = set(re.findall(r'<option value="([^"]+)"', html))
        self.assertTrue(values, "没找到图层下拉项")
        self.assertEqual(values - set(_GRID_LAYERS), set(),
                         f"下拉里有后端不认的图层：{sorted(values - set(_GRID_LAYERS))}")
        for layer in ("tristate", "prior"):
            self.assertIn(layer, values, f"下拉缺 {layer} 层")

    def test_band_colors_have_one_source_of_truth(self) -> None:
        """高度分带配色：图里用 BGR、图例回给前端用 RGB，**必须同源导出**。

        2026-10-06 实况：只有一份 RGB 意图被当成 BGR 画进图，于是"below 用冷色"实际显示成红，
        而且图例与图**互相矛盾**（图例 `background:rgb(...)` 用的是未取反的值）。
        """
        import cv2
        import numpy as np
        from neko_anyadance_body.backend import nav_online
        self.assertEqual(len(nav_online._BAND_COLORS), 4)
        self.assertEqual(nav_online._BAND_COLORS_RGB[0], (60, 60, 255), "below 的意图色写在这里")
        for rgb, bgr in zip(nav_online._BAND_COLORS_RGB, nav_online._BAND_COLORS):
            self.assertEqual(tuple(bgr), tuple(rgb[::-1]), "BGR 必须是 RGB 的取反")
        b, _g, r = nav_online._BAND_COLORS[0]
        self.assertGreater(b, r, "below 的蓝分量必须大于红分量（坑的候选要一眼能挑出来）")
        # 真正编一张 PNG 再解回来：显示出来的像素要是冷色（蓝 > 红）
        img = np.zeros((1, 1, 3), np.uint8)
        img[0, 0] = nav_online._BAND_COLORS[0]
        ok, buf = cv2.imencode(".png", img)
        self.assertTrue(ok)
        shown = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)[0, 0]
        b_val, _g_val, r_val = (int(v) for v in shown)               # 解回来仍是 BGR
        self.assertGreater(b_val, r_val, "PNG 里 below 显示出来必须是冷色（蓝 > 红）")
        self.assertEqual((r_val, b_val), (60, 255), "并把意图色的红/蓝对上")

    def test_main_page_links_to_navmesh_pages(self) -> None:
        """主页面（``/``）必须能点到导航三页 —— 否则 navmesh 页面只能用 URL 手输。

        并且这三页各自从 URL hash 读 token，链接必须带 token（``paintPageLinks``），
        否则点进去是"没有 token"的空壳页。
        """
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        html = (ui / "index.html").read_text(encoding="utf-8")
        app = (ui / "app.js").read_text(encoding="utf-8")
        for el_id, href in (("linkNavmesh", "/navmesh"), ("linkNavmeshMemory", "/navmesh/memory"),
                            ("linkNavmeshCoverage", "/navmesh/coverage")):
            self.assertIn(f'id="{el_id}"', html, f"主页面少了 {el_id}")
            self.assertRegex(html, rf'id="{el_id}"[^>]*href="{href}"', f"{el_id} 的 href 不对")
            self.assertIn(f'"{el_id}"', app, f"app.js 没把 token 挂到 {el_id}")
        self.assertIn("token=", app, "跨页链接要带 token")

    def test_navmesh_page_can_set_world_identity(self) -> None:
        """症状页（``/navmesh``）必须能直接设世界身份 —— 跨会话/先验全挂在它上面。

        2026-10-06 实况：这页只显示 `world_unknown`，修法却在另一个 tab 里，还要手抄 `wrld_` id。
        """
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        html = (ui / "navmesh.html").read_text(encoding="utf-8")
        js = (ui / "navmesh.js").read_text(encoding="utf-8")
        for el_id in ("worldPick", "worldKeyInput", "worldSet", "worldHint", "consoleLink"):
            self.assertIn(f'id="{el_id}"', html, f"navmesh.html 少了 {el_id}")
        # 世界下拉的数据源是记忆分区接口（里面有 world_list）；设置走 /worldmodel/world
        self.assertIn("/worldmodel/navmesh/memory", js)
        self.assertIn("/worldmodel/world", js)

    def test_world_picker_disambiguates_partitions(self) -> None:
        """同一物理世界可能被**按名字另存成第二个分区**（`home` vs `wrld_home`），
        而"自动识别"只在本分区里找历史 —— 选错 key 就等于换了一套记忆。

        2026-10-06 实况：第二次启动没有自动识别，因为身份被设成了名字 `home`
        （`home-e83249bd`：2 会话、无先验），而历史与 prior 都在 `wrld_home-7cf435ea`（7 会话）。
        所以下拉与提示必须：① 以 key 打头（不只显示名字）；② 标出"有先验"；
        ③ 当身份不是 `wrld_` 稳定 ID 时，提示"看起来是同一个世界的另一个分区"+ 该怎么做。
        """
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        js = (ui / "navmesh.js").read_text(encoding="utf-8")
        app = (ui / "app.js").read_text(encoding="utf-8")
        for text in (js, app):
            self.assertTrue(text.count("会话") > 0)
            self.assertIn("有先验", text, "下拉要标出哪个分区有先验")
        self.assertIn("看起来是同一个世界的另一个分区", js)
        self.assertIn('!curKey.startsWith("wrld_")', js, "只在身份不是 wrld_ 稳定 ID 时才提示")
        self.assertIn("world_source", js, "名字来源（manual_name）要触发警告")

    def test_light_callers_skip_the_memory_size_walk(self) -> None:
        """栅格页/主页面的世界下拉必须用 ``sizes=0``。

        `/worldmodel/navmesh/memory` 默认会递归 stat 每个世界的所有文件（现役 10957 个 / 126 MB，
        秒级），2026-10-06 实况把栅格页 8 s 超时打爆、面板报 `signal is aborted`。
        体积只有记忆管理页那一列需要。
        """
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        for name in ("navmesh.js", "app.js"):
            text = (ui / name).read_text(encoding="utf-8")
            self.assertIn("/worldmodel/navmesh/memory?sizes=0", text, f"{name} 没走轻档")
            self.assertNotIn('api("/worldmodel/navmesh/memory")', text, f"{name} 还有重档调用")
        # AbortError 要译成人话，别把 "signal is aborted without reason" 甩给用户
        self.assertIn("超时（", (ui / "navmesh.js").read_text(encoding="utf-8"))

    def test_world_identity_is_shown_as_restored(self) -> None:
        """身份落盘后，"沿用上次"必须在 UI 上可见 —— 否则用户会以为系统在猜世界。

        语义：`world_identity.json` 里只有**用户显式设过**的 key，恢复出来的身份带
        ``restored``；主页面横幅与栅格页提示都要说出来（含保存时间）。
        """
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        html = (ui / "index.html").read_text(encoding="utf-8")
        app = (ui / "app.js").read_text(encoding="utf-8")
        js = (ui / "navmesh.js").read_text(encoding="utf-8")
        self.assertIn('id="wmRestoredBanner"', html, "主页面少了「沿用上次」横幅")
        self.assertIn('byId("wmRestoredBanner")', app)
        self.assertIn("wm.restored", app, "要读 status 里的 restored 标志")
        self.assertIn("沿用上次", app)
        self.assertIn("沿用上次", js, "栅格页提示里也要说")
        self.assertIn("restored_wall", js, "带上保存时间，用户才知道这份是哪来的")

    def test_xsession_row_shows_who_was_recognized(self) -> None:
        """「确认 N」要能看出**认出了谁**。

        2026-10-06 实况：用户快走遍整张图、确认 44 条（含 18 条对上一场），却因为只看到一个
        总数而以为"没有重识别"。`status()["xsession"]["recent"]` 本来就带
        (old_sid, old_kf, inliers, offset_m)，把它按旧会话聚合显示即可 —— 位移小 = 确实认出了同一个地方。
        """
        ui = Path(__file__).resolve().parents[1] / "backend" / "standalone_ui"
        js = (ui / "navmesh.js").read_text(encoding="utf-8")
        self.assertIn("最近认出", js)
        self.assertIn("xs.recent", js, "用后端已经在报的 recent，别另造数据源")
        self.assertIn("位移", js, "位移是「认出同一个地方」的直接证据")


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
