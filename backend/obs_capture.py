"""Optional OBS capture bridges.

This module deliberately stays outside :mod:`backend.vision`'s core capture
implementations.  OBS is useful on systems where DXGI desktop duplication or
GDI ``BitBlt`` cannot read the VRChat mirror, but it is an optional external
bridge rather than a permission bypass:

* :class:`ObsVirtualCameraFrameSource` reads the OBS Virtual Camera through
  OpenCV.  A small reader thread keeps only the newest frame, so a slow vision
  consumer cannot build a stale camera queue.
* :class:`ObsWebSocketFrameSource` uses obs-websocket v5 ``GetSourceScreenshot``
  as a compatibility fallback.  It is request/response based and therefore
  has more latency than the virtual camera; it is intentionally never used
  implicitly.

Neither class is instantiated unless the caller explicitly selects it.  No
  OBS package is a hard dependency of the plugin.
"""

from __future__ import annotations

import base64
from io import BytesIO
import hashlib
import importlib
import json
import os
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import quote


def _safe_error(exc: BaseException, limit: int = 320) -> str:
    return f"{type(exc).__name__}: {exc}"[:limit]


def _optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError):
        return None


class ObsVirtualCameraFrameSource:
    """Read OBS Virtual Camera frames with a latest-frame handoff.

    OpenCV is imported lazily.  ``camera_index=-1`` probes a bounded set of
    DirectShow/Media Foundation device indices and keeps the first opened
    device.  The reader thread continuously drains the camera while
    :meth:`read` returns the newest frame; this avoids the common OpenCV camera
    buffer lag when the detector runs more slowly than the camera.

    The source does not claim that a frame is available merely because OBS is
    installed.  ``status()['available']`` becomes false when OpenCV is absent,
    no camera can be opened, or the reader has encountered a persistent error.
    """

    name = "obs_virtual_camera"

    def __init__(
        self,
        *,
        camera_index: int = -1,
        backend: str = "auto",
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        probe_count: int = 8,
        reconnect_after_failures: int = 5,
        clock: Callable[[], float] = time.monotonic,
        cv2_module: Any | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._cv2 = cv2_module if cv2_module is not None else _optional_import("cv2")
        self._requested_index = int(camera_index)
        self._requested_backend = str(backend or "auto").strip().lower() or "auto"
        if self._requested_backend not in {"auto", "dshow", "msmf", "any"}:
            self._requested_backend = "auto"
        self._width = max(0, int(width))
        self._height = max(0, int(height))
        self._fps = max(0, int(fps))
        self._probe_count = max(1, min(32, int(probe_count)))
        self._reconnect_after = max(1, min(100, int(reconnect_after_failures)))
        self._capture: Any | None = None
        self._selected_index: int | None = None
        self._selected_backend: str | None = None
        self._latest: Any | None = None
        self._last_frame_at: float | None = None
        self._frames = 0
        self._read_failures = 0
        self._reconnects = 0
        self._last_error: str | None = None
        self._closed = False
        self._started = False
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._open_candidates: list[str] = []
        self._candidate_errors: dict[str, str] = {}
        self._open_initial()

    def _backend_candidates(self) -> list[tuple[str, int | None]]:
        cv2 = self._cv2
        if cv2 is None:
            return []
        # ``None`` means let OpenCV choose its default backend.  Constants are
        # intentionally looked up dynamically so tests and older OpenCV builds
        # without one of the backends remain compatible.
        dshow = getattr(cv2, "CAP_DSHOW", None)
        msmf = getattr(cv2, "CAP_MSMF", None)
        if self._requested_backend == "dshow":
            return [("dshow", dshow)] if dshow is not None else []
        if self._requested_backend == "msmf":
            return [("msmf", msmf)] if msmf is not None else []
        if self._requested_backend == "any":
            return [("default", None)]
        # DirectShow tends to expose OBS Virtual Camera more reliably on
        # Windows; default is retained as a final portable fallback.
        result: list[tuple[str, int | None]] = []
        if dshow is not None:
            result.append(("dshow", dshow))
        if msmf is not None:
            result.append(("msmf", msmf))
        result.append(("default", None))
        return result

    def _index_candidates(self) -> list[int]:
        if self._requested_index >= 0:
            return [self._requested_index]
        return list(range(self._probe_count))

    def _open_capture(self, index: int, label: str, api: int | None) -> Any | None:
        cv2 = self._cv2
        if cv2 is None:
            return None
        try:
            try:
                capture = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
            except TypeError:
                capture = cv2.VideoCapture(index)
            if capture is None or not bool(capture.isOpened()):
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass
                raise RuntimeError("device is not opened")
            # Keep the driver's queue shallow.  Unsupported properties are
            # harmless and are ignored by OpenCV.
            prop_buffer = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
            if prop_buffer is not None:
                try:
                    capture.set(prop_buffer, 1)
                except Exception:
                    pass
            prop_width = getattr(cv2, "CAP_PROP_FRAME_WIDTH", None)
            prop_height = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", None)
            prop_fps = getattr(cv2, "CAP_PROP_FPS", None)
            if self._width and prop_width is not None:
                try:
                    capture.set(prop_width, self._width)
                except Exception:
                    pass
            if self._height and prop_height is not None:
                try:
                    capture.set(prop_height, self._height)
                except Exception:
                    pass
            if self._fps and prop_fps is not None:
                try:
                    capture.set(prop_fps, self._fps)
                except Exception:
                    pass
            with self._lock:
                self._open_candidates.append(f"index={index},backend={label}")
            return capture
        except Exception as exc:
            key = f"index={index},backend={label}"
            with self._lock:
                self._candidate_errors[key] = _safe_error(exc)
            return None

    def _open_initial(self) -> None:
        if self._cv2 is None:
            self._last_error = "opencv (cv2) is not installed; enable OBS Virtual Camera or install opencv-python"
            return
        candidates = self._backend_candidates()
        if not candidates:
            self._last_error = "requested OpenCV capture backend is unavailable"
            return
        for index in self._index_candidates():
            for label, api in candidates:
                capture = self._open_capture(index, label, api)
                if capture is not None:
                    with self._lock:
                        self._capture = capture
                        self._selected_index = index
                        self._selected_backend = label
                        self._last_error = None
                    return
        self._last_error = "OBS Virtual Camera is unavailable; no OpenCV camera index could be opened"

    def _release_capture(self, capture: Any | None) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _ensure_reader(self) -> None:
        with self._lock:
            if self._started or self._closed or self._capture is None:
                return
            self._started = True
            self._stop.clear()
            self._reader = threading.Thread(
                target=self._reader_loop,
                name="neko-obs-camera",
                daemon=True,
            )
            self._reader.start()

    def _reconnect(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            old = self._capture
            index = self._selected_index
            backend = self._selected_backend
            self._capture = None
        self._release_capture(old)
        if index is None:
            return False
        # Prefer the selected device/backend, then probe all candidates in case
        # OBS recreated its virtual camera at a different index.
        candidates: list[tuple[int, str, int | None]] = []
        for label, api in self._backend_candidates():
            if label == backend:
                candidates.insert(0, (index, label, api))
            else:
                candidates.append((index, label, api))
        for candidate_index in self._index_candidates():
            for label, api in self._backend_candidates():
                item = (candidate_index, label, api)
                if item not in candidates:
                    candidates.append(item)
        for candidate_index, label, api in candidates:
            capture = self._open_capture(candidate_index, label, api)
            if capture is None:
                continue
            with self._lock:
                self._capture = capture
                self._selected_index = candidate_index
                self._selected_backend = label
                self._reconnects += 1
                self._last_error = None
            return True
        with self._lock:
            self._last_error = "OBS Virtual Camera reconnect failed; verify OBS virtual camera is running"
        return False

    def _reader_loop(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            with self._lock:
                capture = self._capture
                closed = self._closed
            if closed or capture is None:
                break
            try:
                ok, frame = capture.read()
                if ok and frame is not None:
                    now = self._clock()
                    with self._lock:
                        self._latest = frame
                        self._last_frame_at = now
                        self._frames += 1
                        self._read_failures = 0
                        self._last_error = None
                    consecutive_failures = 0
                    continue
                consecutive_failures += 1
                with self._lock:
                    self._read_failures += 1
                if consecutive_failures >= self._reconnect_after:
                    consecutive_failures = 0
                    self._reconnect()
                    self._stop.wait(0.05)
            except Exception as exc:
                consecutive_failures += 1
                with self._lock:
                    self._read_failures += 1
                    self._last_error = _safe_error(exc)
                if consecutive_failures >= self._reconnect_after:
                    consecutive_failures = 0
                    self._reconnect()
                    self._stop.wait(0.05)
                else:
                    self._stop.wait(0.01)

    def read(self) -> Any:
        self._ensure_reader()
        with self._lock:
            if self._closed:
                return None
            return self._latest

    def status(self) -> Mapping[str, Any]:
        now = self._clock()
        with self._lock:
            return {
                "available": bool(
                    self._capture is not None
                    and not self._closed
                    and self._last_error is None
                ),
                "name": self.name,
                "transport": "opencv",
                "requested_camera_index": self._requested_index,
                "camera_index": self._selected_index,
                "backend": self._selected_backend,
                "reader_running": bool(self._reader and self._reader.is_alive()),
                "frame_available": self._latest is not None,
                "frames": self._frames,
                "read_failures": self._read_failures,
                "reconnects": self._reconnects,
                "last_frame_age_ms": (
                    None
                    if self._last_frame_at is None
                    else round(max(0.0, now - self._last_frame_at) * 1000.0, 1)
                ),
                "candidate_errors": dict(self._candidate_errors),
                "last_error": self._last_error,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            capture = self._capture
            self._capture = None
            reader = self._reader
            self._reader = None
        self._release_capture(capture)
        if reader and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=1.5)


class ObsWebSocketFrameSource:
    """Compatibility capture through obs-websocket v5 screenshots.

    This is intentionally a pull-based fallback: each :meth:`read` sends one
    ``GetSourceScreenshot`` request and decodes the returned JPEG/PNG.  It is
    useful when OBS Virtual Camera cannot be enabled, but its latency is tied
    to the OBS websocket round trip and image encoding.  For responsive
    navigation prefer :class:`ObsVirtualCameraFrameSource`.
    """

    name = "obs_websocket"

    def __init__(
        self,
        *,
        source_name: str,
        host: str = "127.0.0.1",
        port: int = 4455,
        password: str | None = None,
        image_format: str = "jpg",
        image_width: int = 0,
        image_height: int = 0,
        timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        websocket_module: Any | None = None,
        cv2_module: Any | None = None,
        pil_module: Any | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._source_name = str(source_name).strip()
        self._host = str(host).strip() or "127.0.0.1"
        self._port = int(port)
        self._password = password if password is not None else os.getenv("OBS_WEBSOCKET_PASSWORD", "")
        self._image_format = str(image_format or "jpg").strip().lower().lstrip(".")
        if self._image_format not in {"jpg", "jpeg", "png", "webp"}:
            self._image_format = "jpg"
        self._image_width = max(0, int(image_width))
        self._image_height = max(0, int(image_height))
        self._timeout_s = min(30.0, max(0.1, float(timeout_s)))
        self._websocket = websocket_module if websocket_module is not None else _optional_import("websocket")
        self._cv2 = cv2_module if cv2_module is not None else _optional_import("cv2")
        self._pil = pil_module if pil_module is not None else _optional_import("PIL.Image")
        self._socket: Any | None = None
        self._closed = False
        self._request_id = 0
        self._requests = 0
        self._frames = 0
        self._last_latency_ms: float | None = None
        self._last_frame_at: float | None = None
        self._last_error: str | None = None

        if not self._source_name:
            self._last_error = "OBS source_name is required"
        elif self._websocket is None:
            self._last_error = "websocket-client is not installed; use OBS Virtual Camera or install websocket-client"

    @staticmethod
    def _authentication(password: str, salt: str, challenge: str) -> str:
        """Build obs-websocket v5's challenge-response authentication token."""
        secret = base64.b64encode(hashlib.sha256((password + salt).encode("utf-8")).digest())
        auth = base64.b64encode(hashlib.sha256(secret + challenge.encode("utf-8")).digest())
        return auth.decode("ascii")

    @staticmethod
    def _decode_message(raw: Any) -> Mapping[str, Any]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("OBS websocket message must be an object")
        return value

    def _connect(self) -> None:
        if self._websocket is None:
            raise RuntimeError(self._last_error or "websocket-client is unavailable")
        url = f"ws://{self._host}:{self._port}"
        socket = self._websocket.create_connection(url, timeout=self._timeout_s)
        try:
            hello = self._decode_message(socket.recv())
            if int(hello.get("op", -1)) != 0:
                raise RuntimeError("OBS websocket did not send Hello")
            data = hello.get("d") if isinstance(hello.get("d"), Mapping) else {}
            identify: dict[str, Any] = {
                "rpcVersion": int(data.get("rpcVersion", 1) or 1),
            }
            auth_data = data.get("authentication")
            if isinstance(auth_data, Mapping):
                salt = str(auth_data.get("salt") or "")
                challenge = str(auth_data.get("challenge") or "")
                if not self._password:
                    raise RuntimeError("OBS websocket password is required by the server")
                identify["authentication"] = self._authentication(self._password, salt, challenge)
            socket.send(json.dumps({"op": 1, "d": identify}, separators=(",", ":")))
            identified = self._decode_message(socket.recv())
            if int(identified.get("op", -1)) != 2:
                detail = identified.get("d")
                raise RuntimeError(f"OBS websocket Identify failed: {detail}")
        except Exception:
            try:
                socket.close()
            except Exception:
                pass
            raise
        with self._lock:
            self._socket = socket

    def _request_screenshot(self) -> bytes:
        with self._lock:
            socket = self._socket
            self._request_id += 1
            request_id = str(self._request_id)
        if socket is None:
            self._connect()
            with self._lock:
                socket = self._socket
        if socket is None:
            raise RuntimeError("OBS websocket connection is unavailable")
        request_data: dict[str, Any] = {
            "sourceName": self._source_name,
            "imageFormat": self._image_format,
        }
        if self._image_width:
            request_data["imageWidth"] = self._image_width
        if self._image_height:
            request_data["imageHeight"] = self._image_height
        socket.send(json.dumps({
            "op": 6,
            "d": {
                "requestType": "GetSourceScreenshot",
                "requestId": request_id,
                "requestData": request_data,
            },
        }, separators=(",", ":")))
        deadline = self._clock() + self._timeout_s
        while True:
            remaining = max(0.05, deadline - self._clock())
            try:
                if hasattr(socket, "settimeout"):
                    socket.settimeout(remaining)
                message = self._decode_message(socket.recv())
            except Exception:
                raise
            if int(message.get("op", -1)) != 7:
                # Events and other asynchronous messages can be interleaved.
                if self._clock() < deadline:
                    continue
                raise TimeoutError("timed out waiting for OBS screenshot response")
            data = message.get("d") if isinstance(message.get("d"), Mapping) else {}
            if str(data.get("requestId")) != request_id:
                if self._clock() < deadline:
                    continue
                raise TimeoutError("timed out waiting for matching OBS screenshot response")
            status = data.get("requestStatus") if isinstance(data.get("requestStatus"), Mapping) else {}
            if not bool(status.get("result")):
                raise RuntimeError(str(status.get("comment") or "OBS screenshot request failed"))
            response = data.get("responseData") if isinstance(data.get("responseData"), Mapping) else {}
            image_data = response.get("imageData")
            if not isinstance(image_data, str) or "," not in image_data:
                raise ValueError("OBS screenshot response did not contain imageData")
            encoded = image_data.split(",", 1)[1]
            return base64.b64decode(encoded, validate=False)

    def _decode_frame(self, payload: bytes) -> Any:
        if self._cv2 is not None:
            np = _optional_import("numpy")
            if np is not None:
                buffer = np.frombuffer(payload, dtype=np.uint8)
                flag = getattr(self._cv2, "IMREAD_COLOR", 1)
                frame = self._cv2.imdecode(buffer, flag)
                if frame is not None:
                    return frame
        if self._pil is not None:
            image = self._pil.open(BytesIO(payload))
            return image.convert("RGB")
        raise RuntimeError("OBS screenshot needs opencv-python or Pillow for image decoding")

    def read(self) -> Any:
        if self._closed or self._last_error == "OBS source_name is required":
            return None
        started = self._clock()
        try:
            payload = self._request_screenshot()
            frame = self._decode_frame(payload)
            now = self._clock()
            with self._lock:
                self._frames += 1
                self._requests += 1
                self._last_frame_at = now
                self._last_latency_ms = max(0.0, now - started) * 1000.0
                self._last_error = None
            return frame
        except Exception as exc:
            with self._lock:
                self._requests += 1
                self._last_latency_ms = max(0.0, self._clock() - started) * 1000.0
                self._last_error = _safe_error(exc)
                socket = self._socket
                self._socket = None
            if socket is not None:
                try:
                    socket.close()
                except Exception:
                    pass
            return None

    def status(self) -> Mapping[str, Any]:
        now = self._clock()
        with self._lock:
            return {
                "available": bool(
                    not self._closed
                    and self._socket is not None
                    and self._last_error is None
                ),
                "name": self.name,
                "transport": "obs-websocket-v5",
                "endpoint": f"ws://{self._host}:{self._port}",
                "source_name": self._source_name,
                "connected": self._socket is not None,
                "frames": self._frames,
                "requests": self._requests,
                "last_latency_ms": None if self._last_latency_ms is None else round(self._last_latency_ms, 1),
                "last_frame_age_ms": (
                    None
                    if self._last_frame_at is None
                    else round(max(0.0, now - self._last_frame_at) * 1000.0, 1)
                ),
                "last_error": self._last_error,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            socket = self._socket
            self._socket = None
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass


__all__ = ["ObsVirtualCameraFrameSource", "ObsWebSocketFrameSource"]
