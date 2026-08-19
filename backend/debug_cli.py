"""用于实时调试后端的小型命令行客户端，不依赖插件 SDK。"""

from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _json_arg(value: str | None, file_path: Path | None) -> dict:
    if value and file_path:
        raise ValueError("use either --json or --file")
    if file_path:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    elif value:
        payload = json.loads(value)
    else:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    return payload


def _scalar_arg(value: str) -> object:
    """Parse a JSON scalar used by an OSC parameter command."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    if isinstance(parsed, (dict, list)):
        raise ValueError("value must be a scalar")
    return parsed


class _PersistentHttpClient:
    """Small JSON-lines shell transport that reuses one loopback connection."""

    def __init__(self, host: str, port: int, token: str) -> None:
        self.host = host
        self.port = port
        self.token = token
        self.connection: http.client.HTTPConnection | None = None

    def close(self) -> None:
        connection = self.connection
        self.connection = None
        if connection is not None:
            connection.close()

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "X-Neko-Backend-Token": self.token,
        }
        for attempt in range(2):
            try:
                if self.connection is None:
                    self.connection = http.client.HTTPConnection(self.host, self.port, timeout=5.0)
                self.connection.request(method, path, body=body, headers=headers)
                response = self.connection.getresponse()
                raw = response.read()
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise RuntimeError("backend returned a non-object response")
                if response.status >= 400:
                    raise RuntimeError(str(value.get("error") or f"HTTP {response.status}"))
                return value
            except (OSError, TimeoutError, http.client.HTTPException, ValueError) as exc:
                self.close()
                if attempt:
                    raise RuntimeError(str(exc)) from exc
        raise RuntimeError("persistent request failed")


def _run_shell(host: str, port: int, token: str) -> int:
    """Run a persistent JSON-lines control session for high-frequency callers."""
    client = _PersistentHttpClient(host, port, token)
    print(json.dumps({"ready": True, "protocol": "jsonl"}, ensure_ascii=False), flush=True)
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                break
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise ValueError("command must be a JSON object")
                path = str(command.get("path") or "")
                if not path.startswith("/"):
                    raise ValueError("path must start with '/'")
                method = str(command.get("method") or "POST").upper()
                payload = command.get("payload")
                if payload is not None and not isinstance(payload, dict):
                    raise ValueError("payload must be a JSON object")
                result = client.request(method, path, payload)
            except (ValueError, RuntimeError, OSError) as exc:
                result = {"accepted": False, "error": str(exc)}
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    finally:
        client.close()
    return 0


def request(host: str, port: int, token: str, method: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        f"http://{host}:{port}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Neko-Backend-Token": token,
        },
    )
    try:
        with urlopen(req, timeout=5.0) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            message = detail.get("error") if isinstance(detail, dict) else None
        except (OSError, ValueError, UnicodeError):
            message = None
        raise RuntimeError(message or str(exc)) from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(result, dict):
        raise RuntimeError("backend returned a non-object response")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="live debug client for backend/process.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48912)
    parser.add_argument("--token", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("health", "snapshot", "awareness", "cognition", "shutdown"):
        sub.add_parser(name)
    action = sub.add_parser("action")
    action.add_argument("--kind", required=True)
    action.add_argument("--json", default=None)
    action.add_argument("--file", type=Path, default=None)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--json", default=None)
    ingest.add_argument("--file", type=Path, default=None)
    plan = sub.add_parser("plan")
    plan.add_argument("--json", default=None)
    plan.add_argument("--file", type=Path, default=None)
    feedback = sub.add_parser("feedback")
    feedback.add_argument("--json", default=None)
    feedback.add_argument("--file", type=Path, default=None)
    sub.add_parser(
        "shell",
        help="persistent JSON-lines control session (one HTTP connection)",
    )
    batch = sub.add_parser("batch", help="send a bounded batch of OSC commands")
    batch.add_argument("--json", default=None)
    batch.add_argument("--file", type=Path, default=None)
    parameter = sub.add_parser("parameter", help="send one VRChat Avatar parameter")
    parameter.add_argument("--name", required=True)
    parameter.add_argument("--value", required=True, help="JSON scalar such as true, 1, or 0.5")
    osc_input = sub.add_parser("input", help="pulse a VRChat Grab, Use, or Drop input")
    osc_input.add_argument("--action", choices=("grab", "use", "drop"), required=True)
    osc_input.add_argument("--side", choices=("left", "right"), required=True)
    osc_input.add_argument("--hold-ms", type=int, default=100)
    locomotion = sub.add_parser("locomotion", help="send a timed VRChat movement axis")
    locomotion.add_argument("--vertical", type=float, default=0.0)
    locomotion.add_argument("--horizontal", type=float, default=0.0)
    locomotion.add_argument("--duration-ms", type=int, default=1000)
    turn = sub.add_parser("turn", help="send a timed VRChat turn axis")
    turn.add_argument("--horizontal", type=float, required=True)
    turn.add_argument("--duration-ms", type=int, default=500)
    sub.add_parser("stop-movement", help="zero VRChat movement and turn axes")
    chatbox = sub.add_parser("chatbox", help="send text to the VRChat chatbox")
    chatbox.add_argument("--text", required=True)
    chatbox.add_argument(
        "--deferred",
        action="store_false",
        dest="immediate",
        help="show the message only while typing",
    )
    chatbox.set_defaults(immediate=True)
    sub.add_parser("cancel-inputs", help="cancel pending inputs and release buttons")
    args = parser.parse_args()
    if args.command == "shell":
        return _run_shell(args.host, args.port, args.token)
    try:
        if args.command == "health":
            result = request(args.host, args.port, args.token, "GET", "/health")
        elif args.command == "snapshot":
            result = request(args.host, args.port, args.token, "GET", "/snapshot")
        elif args.command == "awareness":
            result = request(args.host, args.port, args.token, "GET", "/awareness")
        elif args.command == "cognition":
            result = request(args.host, args.port, args.token, "GET", "/cognition")
        elif args.command == "shutdown":
            result = request(args.host, args.port, args.token, "POST", "/shutdown", {})
        elif args.command == "plan":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/cognition/plan",
                _json_arg(args.json, args.file),
            )
        elif args.command == "feedback":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/cognition/feedback",
                _json_arg(args.json, args.file),
            )
        elif args.command == "action":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/action",
                {"kind": args.kind, "params": _json_arg(args.json, args.file)},
            )
        elif args.command == "parameter":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/osc/parameter",
                {"name": args.name, "value": _scalar_arg(args.value)},
            )
        elif args.command == "batch":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/osc/batch",
                _json_arg(args.json, args.file),
            )
        elif args.command == "input":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/osc/input",
                {"action": args.action, "side": args.side, "hold_ms": args.hold_ms},
            )
        elif args.command == "locomotion":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/osc/locomotion",
                {
                    "vertical": args.vertical,
                    "horizontal": args.horizontal,
                    "duration_ms": args.duration_ms,
                },
            )
        elif args.command == "turn":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/osc/turn",
                {"horizontal": args.horizontal, "duration_ms": args.duration_ms},
            )
        elif args.command == "stop-movement":
            result = request(args.host, args.port, args.token, "POST", "/osc/stop_movement", {})
        elif args.command == "chatbox":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/osc/chatbox",
                {"text": args.text, "immediate": args.immediate},
            )
        elif args.command == "cancel-inputs":
            result = request(args.host, args.port, args.token, "POST", "/osc/cancel", {})
        else:
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/world/ingest",
                _json_arg(args.json, args.file),
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"debug request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
