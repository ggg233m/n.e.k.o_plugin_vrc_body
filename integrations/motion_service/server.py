"""本机 HTTP 适配；默认无需令牌，仅允许本机程序访问。"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from urllib.parse import urlsplit


def make_server(service, token=None, port=2346):
    if token is not None and (not isinstance(token, str) or len(token) < 32):
        raise ValueError("服务令牌至少需要 32 字符")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, code, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
                # 轮询超时断开连接不会改变模型或任务状态。
                pass

        def local_request(self):
            # 固定回环监听，并拒绝网页请求与DNS重绑定；不增加动作协议负载。
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return (self.headers.get("Host", "").lower() in hosts
                    and "Origin" not in self.headers
                    and "Sec-Fetch-Site" not in self.headers)

        def authorized(self):
            # 显式令牌参数仅兼容已有隔离测试；正常启动不启用令牌。
            return self.local_request() and (token is None or hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token))

        def do_GET(self):
            if not self.local_request():
                return self.send_json(403, {"error": "local_only"})
            path = urlsplit(self.path).path
            if path == "/health":
                return self.send_json(200, service.health())
            if not self.authorized():
                return self.send_json(403, {"error": "unauthorized"})
            if path.startswith("/tasks/"):
                return self.send_json(200, service.status(path.rsplit("/", 1)[-1]))
            self.send_json(404, {"error": "not_found"})

        def do_POST(self):
            if not self.authorized():
                return self.send_json(403, {"error": "unauthorized"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("invalid_body_size")
                self.connection.settimeout(2)
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("invalid_body")
                routes = {"/world": service.bind, "/tasks": service.submit, "/intent": service.submit_intent, "/cancel": service.cancel,
                          "/chunks": service.pull, "/ack": service.ack, "/requests": service.request_status}
                if hasattr(service,"streams"):
                    routes.update({"/streams/open":service.open_stream,"/streams/arm":service.streams.arm,"/streams/world":service.streams.renew,
                                   "/streams/intent":service.streams.update,"/streams/pull":service.streams.pull,
                                   "/streams/exchange":service.streams.exchange,
                                   "/streams/ack":service.streams.ack,"/streams/received":service.streams.received,"/streams/status":service.streams.status,
                                   "/streams/close":service.streams.close})
                route = routes.get(urlsplit(self.path).path)
                if route is None:
                    return self.send_json(404, {"error": "not_found"})
                self.send_json(200, route(data))
            except (ValueError, TypeError, KeyError):
                self.send_json(400, {"status": "failed", "error": "invalid_or_stale_request"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
