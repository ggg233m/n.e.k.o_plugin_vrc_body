"""Small SDK-free command line client for live backend debugging."""

from __future__ import annotations

import argparse
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
    for name in ("health", "snapshot", "awareness", "shutdown"):
        sub.add_parser(name)
    action = sub.add_parser("action")
    action.add_argument("--kind", required=True)
    action.add_argument("--json", default=None)
    action.add_argument("--file", type=Path, default=None)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--json", default=None)
    ingest.add_argument("--file", type=Path, default=None)
    args = parser.parse_args()
    try:
        if args.command == "health":
            result = request(args.host, args.port, args.token, "GET", "/health")
        elif args.command == "snapshot":
            result = request(args.host, args.port, args.token, "GET", "/snapshot")
        elif args.command == "awareness":
            result = request(args.host, args.port, args.token, "GET", "/awareness")
        elif args.command == "shutdown":
            result = request(args.host, args.port, args.token, "POST", "/shutdown", {})
        elif args.command == "action":
            result = request(
                args.host,
                args.port,
                args.token,
                "POST",
                "/action",
                {"kind": args.kind, "params": _json_arg(args.json, args.file)},
            )
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
