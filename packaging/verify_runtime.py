"""用干净 Python 检查安装后的依赖、真实模型推理和后端服务。"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import runpy
import secrets
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plugin", type=Path)
    args = parser.parse_args()
    root = args.plugin.resolve()
    # 与实际子进程使用同一入口的 vendor 引导，禁止借用系统 site-packages。
    assert sys.flags.no_site and sys.flags.isolated, "请使用 python -I -S 启动"
    runpy.run_path(str(root / "backend/process.py"), run_name="package_probe")
    for name in ("numpy", "PIL", "onnxruntime", "cv2", "dxcam", "websocket", "winrt.windows.graphics.capture"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(root / "vendor"), name
    import numpy as np
    detector_class = importlib.import_module(f"{root.name}.backend.local_perception").OpenVinoLocalDetector
    detector = detector_class(
        model_path=str(root / "models/person_detect_v1.3_s/person_detect_v1.3_s_model.onnx"),
        labels_path=str(root / "models/person_detect_v1.3_s/person_detect_v1.3_s_labels.json"),
        onnxruntime_cuda="disabled",
    )
    assert detector.status()["available"], detector.status()
    detector.observe(np.zeros((640, 640, 3), dtype=np.uint8), now=time.monotonic())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(24)
    proc = subprocess.Popen([
        sys.executable, "-I", "-S", "-B", str(root / "backend/process.py"),
        "--standalone", "--offline", f"--port={port}", f"--token={token}",
    ], cwd=root.parent, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    def request(path: str, body: bytes | None = None):
        with urlopen(Request(f"http://127.0.0.1:{port}{path}", data=body,
                             headers={"X-Neko-Backend-Token": token, "Content-Type": "application/json"}), timeout=2) as response:
            return response.read()
    try:
        deadline = time.monotonic() + 25
        while True:
            try:
                assert json.loads(request("/health"))["ok"]
                break
            except OSError:
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("隔离后端启动失败；请检查依赖和端口")
                time.sleep(0.2)
        assert b"<html" in request("/").lower()
        assert request("/ui/app.js")
        assert request("/ui/styles.css")
        assert isinstance(json.loads(request("/snapshot")), dict)
        request("/shutdown", b"{}")
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    print(json.dumps({"isolated_vendor_imports": True, "model_inference": dict(detector.status()),
                      "backend_health_ui_snapshot_shutdown": True}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
