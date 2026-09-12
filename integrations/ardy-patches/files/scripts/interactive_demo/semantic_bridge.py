"""本机语义事件桥；网络接收、文本编码和动作生成互不阻塞。"""

from __future__ import annotations

import atexit
import html
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LABELS = {"waiting": "等待新回复", "paused": "联动已暂停", "analyzing": "分析中",
    "ready": "已提交 demo", "failed": "处理失败", "expired": "已过期",
    "waiting_model": "等待模型加载", "applying": "更新提示中", "applied": "提示已应用",
    "generating": "生成动作中", "playing": "已生成并播放", "superseded": "已被新回复替代"}


def validate_event(value):
    if not isinstance(value, dict):
        raise ValueError("invalid_event")
    result = {}
    for key, limit in (("producer", 64), ("event_id", 64), ("character", 128), ("text", 16000), ("source", 160)):
        item = value.get(key, "")
        if not isinstance(item, str) or len(item) > limit:
            raise ValueError("invalid_" + key)
        result[key] = item
    if not result["producer"] or not result["event_id"]:
        raise ValueError("missing_id")
    sequence = value.get("sequence")
    if type(sequence) is not int or sequence < 1:
        raise ValueError("invalid_sequence")
    result["sequence"] = sequence
    for key in ("created_at", "expires_at"):
        item = value.get(key)
        if type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError("invalid_time")
        result[key] = item
    if not 0 < result["expires_at"] - result["created_at"] <= 300 or result["created_at"] > time.time() + 5:
        raise ValueError("invalid_lifetime")
    phase = value.get("phase")
    if phase not in {"waiting", "paused", "analyzing", "ready", "failed", "expired"}:
        raise ValueError("invalid_phase")
    result["phase"] = phase
    if phase == "ready":
        for key, limit in (("emotion", 80), ("intent", 120), ("action", 240), ("prompt", 600)):
            item = value.get(key)
            if not isinstance(item, str) or not item.strip() or len(item) > limit:
                raise ValueError("invalid_semantics")
            result[key] = item.strip()
        if not result["prompt"].startswith("A person "):
            raise ValueError("invalid_prompt")
    for key in ("analysis_ms",):
        if type(value.get(key)) in (int, float) and math.isfinite(value[key]):
            result[key] = value[key]
    return result


class SemanticBridge:
    def __init__(self, demo, port=2334):
        self.demo = demo
        self.lock = threading.RLock()
        self.event = None
        self.clients = {}
        self.widgets = {}
        self.retired = set()
        self.stop = threading.Event()
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                # 不把原文及 HTTP 载荷写入日志。
                pass

            def reply(self, status, payload):
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path not in {"/status", "/health"}:
                    return self.reply(404, {"error": "not_found"})
                self.reply(200, bridge.status())

            def do_POST(self):
                if self.path != "/events":
                    return self.reply(404, {"error": "not_found"})
                # 仅接受本机进程 JSON；禁止网页跨源提交动作。
                if self.headers.get("Origin") or self.headers.get_content_type() != "application/json":
                    return self.reply(403, {"error": "origin_or_content_type"})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 100000:
                        raise ValueError("invalid_size")
                    self.connection.settimeout(2)
                    event = validate_event(json.loads(self.rfile.read(size)))
                    self.reply(200, bridge.receive(event))
                except (ValueError, UnicodeError, TimeoutError):
                    self.reply(400, {"error": "invalid_event"})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True, name="semantic-http").start()
        threading.Thread(target=self.run, daemon=True, name="semantic-apply").start()
        atexit.register(self.close)

    def close(self):
        if not self.stop.is_set():
            self.stop.set()
            self.server.shutdown()
            self.server.server_close()

    def status(self):
        with self.lock:
            return {"state": "accepted" if self.event else "waiting", "event_id": (self.event or {}).get("event_id"),
                "phase": (self.event or {}).get("phase"), "clients": {str(k): dict(v) for k, v in self.clients.items()}}

    def receive(self, event):
        with self.lock:
            old = self.event
            if time.time() >= event["expires_at"]:
                return {"state": "expired", "event_id": event["event_id"]}
            if event["producer"] in self.retired:
                return {"state": "superseded", "event_id": event["event_id"]}
            if old:
                if old["producer"] != event["producer"]:
                    if event["created_at"] <= old["created_at"]:
                        return {"state": "superseded"}
                    self.retired.add(old["producer"])
                elif event["sequence"] < old["sequence"]:
                    return {"state": "superseded"}
                elif event["sequence"] == old["sequence"]:
                    if event["event_id"] != old["event_id"]:
                        return {"state": "superseded"}
                    if event["phase"] == old["phase"] or old["phase"] != "analyzing":
                        return self.status()
            self.event = event
            self.clients = {}
            return self.status()

    def add_gui(self, client):
        with client.gui.add_folder("NEKO 语义联动", expand_by_default=True):
            enabled = client.gui.add_checkbox("自动应用动作", initial_value=True)
            body = client.gui.add_markdown("等待 N.E.K.O 新回复。")
        with self.lock:
            self.widgets[client.client_id] = (enabled, body)

    def valid(self, event, client_id):
        return (not self.stop.is_set() and self.event is not None
            and self.event["event_id"] == event["event_id"] and self.event["phase"] == "ready"
            and time.time() < event["expires_at"] and self.widgets[client_id][0].value
            and self.demo.client_active(client_id))

    def render(self, client_id, event, state):
        # 转义 HTML 和 Markdown，LLM 文本只能作为展示内容。
        def safe(text):
            text = html.escape(str(text or ""))
            for char in "\\`*_{}[]()#+-.!|":
                text = text.replace(char, "\\" + char)
            return text
        parts = ["**状态：** " + LABELS.get(state, state)]
        if event:
            parts += ["**回复：** " + safe(event.get("text")),
                "**情绪：** " + safe(event.get("emotion", "—")) + "　**意图：** " + safe(event.get("intent", "—")),
                "**动作：** " + safe(event.get("action", "—")),
                "**英文提示：** " + safe(event.get("prompt", "—"))]
            if "analysis_ms" in event:
                parts.append(f"分析耗时：{event['analysis_ms']} ms")
        self.widgets[client_id][1].content = "\n\n".join(parts)

    def apply(self, event, client_id, session):
        start = time.monotonic()
        with self.demo._text_update_lock:
            # 编码慢时仍允许接收新事件；提交前再次核对最新事件。
            model = session.model
            feature, _ = model.text_encoder([event["prompt"]])
            with session.replan_lock:
                with self.lock:
                    if not self.valid(event, client_id) or session.model is not model:
                        return
                    self.demo.on_text_prompt_update(client_id, trigger_replan=False,
                        prepared_prompt=event["prompt"], prepared_feature=feature)
                    self.clients[client_id] = {"state": "generating", "applied_event_id": event["event_id"],
                        "apply_ms": round((time.monotonic() - start) * 1000)}
                    self.render(client_id, event, "generating")
                # 已在当前姿态提交的动作完成生成后，再处理最新待处理结果。
                self.demo._generate_step(client_id)
                with self.lock:
                    if self.valid(event, client_id):
                        if session.motion_tensor is None:
                            self.clients[client_id]["state"] = "failed"
                            return
                        session.playing = True
                        session.gui_elements.gui_play_pause_button.label = "Pause"
                        session.gui_elements.gui_next_frame_button.disabled = True
                        session.gui_elements.gui_prev_frame_button.disabled = True
                        self.clients[client_id].update(state="playing", generation_ms=round((time.monotonic() - start) * 1000))

    def run(self):
        while not self.stop.wait(0.25):
            with self.lock:
                event = dict(self.event) if self.event else None
                ids = list(self.widgets)
            for client_id in ids:
                session = self.demo.client_sessions.get(client_id)
                if session is None:
                    continue
                try:
                    with self.lock:
                        if event is None:
                            self.render(client_id, None, "waiting")
                            continue
                        if self.event["event_id"] != event["event_id"]:
                            break
                        phase = event["phase"]
                        state = self.clients.get(client_id, {}).get("state", phase)
                        applied = self.clients.get(client_id, {}).get("applied_event_id") == event["event_id"]
                        if time.time() >= event["expires_at"] and not applied:
                            state = "expired"
                        elif not self.widgets[client_id][0].value:
                            state = "paused"
                        elif applied and state == "paused":
                            state = "playing"
                        elif phase == "ready" and not applied and state not in {"playing", "failed"}:
                            state = "waiting_model" if session.model is None else "applying"
                        self.clients.setdefault(client_id, {})["state"] = state
                        self.render(client_id, event, state)
                    if state == "applying":
                        self.apply(event, client_id, session)
                except Exception:
                    with self.lock:
                        if self.event and event and self.event["event_id"] == event["event_id"]:
                            self.clients[client_id] = {"state": "failed"}
                            self.render(client_id, event, "failed")


def start_bridge(demo):
    if os.environ.get("ARDY_SEMANTIC_BRIDGE", "1") == "0":
        return None
    try:
        bridge = SemanticBridge(demo, port=int(os.environ.get("ARDY_SEMANTIC_PORT", "2334")))
        print(f"NEKO semantic bridge: http://127.0.0.1:{bridge.port}")
        return bridge
    except (OSError, ValueError) as exc:
        print(f"NEKO semantic bridge unavailable: {type(exc).__name__}")
        return None
