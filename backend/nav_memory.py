# -*- coding: utf-8 -*-
"""navmesh 持久记忆：按世界分区存关键帧的位姿 + 地图侧 ORB 特征 + 彩色缩略图，并提供管理（列/删/钉/清理）。

目录（``root`` = ``<state_dir>/navmesh_memory``）::

    <world_dir>/world.json                       世界身份（原样 world_key，目录名只是它的安全编码）
    <world_dir>/sessions/<sid>/session.json      会话元数据：状态、传感器、计数、字节、pinned/label
    <world_dir>/sessions/<sid>/poses.npz         位姿整表（ids, T_map, T_dr, dist_m, t_s, has_feat）
    <world_dir>/sessions/<sid>/kf/<k>.npz        地图侧特征：xyz(f16, 头部 base 系追踪米) + des3d + K/size/eye_y
    <world_dir>/sessions/<sid>/thumb/<k>.jpg     左目彩色缩略图

约定（见 memory spatial-memory-base-feasibility）：
* 点云不进持久记忆（约 219 KB/帧），只存 PnP 重定位要用的地图侧特征（约 17 KB/帧）；
* 回环会整体改位姿，逐帧写的 T_map 会过期，所以位姿只整表写：周期性 + 会话结束；
* 原地补帧（refresh）同一视角，替换上一帧的特征/缩略图（与 mapper.drop_points 同口径），位姿行保留；
* 位姿系 = **该会话自己的地图系**（原点 = 会话起点，追踪米）；跨会话对齐是后续步骤，这里不假设共享原点；
* world_key 未知时不写：绝不把记忆记到猜的世界里。

写盘全在每个会话自己的后台线程里做；队列满了丢帧计数，不阻塞建图线程。IO 出错只让记忆停写，导航照常。
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .world_model import sanitize_world_key

SCHEMA = 1
_SID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")
_LABEL_MAX = 80


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_total_mb: float = 4096.0         # 全部世界合计；超了从最旧的未钉会话删起
    max_world_mb: float = 1024.0
    max_sessions_per_world: int = 30
    max_session_mb: float = 512.0        # 单会话上限：超了停写特征/缩略图，位姿照写
    min_session_keyframes: int = 20      # 结束时关键帧少于它的会话直接丢（误启动、没走动）
    pose_flush_s: float = 10.0           # 位姿整表周期写入间隔；崩溃最多丢这么久的位姿
    thumb_width: int = 320               # 720×405 → 320×180，q70 约 8 KB
    thumb_quality: int = 70
    queue_max: int = 64


def world_dir_name(world_key: str) -> str:
    """安全目录名 + 原 key 的短哈希：不同 key 规整后撞名也分得开。"""
    digest = hashlib.sha1(world_key.encode("utf-8")).hexdigest()[:8]
    return f"{sanitize_world_key(world_key)[:80]}-{digest}"


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    _write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


#: 世界目录体积的缓存 TTL（秒）。为什么需要它：`_dir_bytes` 是**递归 stat 每个文件**
#: （现役 10957 个文件 / 126 MB，实测单趟 2 s+），而 `list_worlds` 每次请求都对每个世界算一遍 ——
#: 管理页刷新一次就是好几秒；再加上正在写关键帧的导航会话，很容易把**调用方**的超时打爆
#:（2026-10-06 实况：栅格页读世界列表 8 s 超时，提示报 abort）。体积是纯展示用的近似值，
#: 允许陈旧几十秒。
_SIZE_TTL_S = 60.0


def _cached_dir_bytes(path: Path, cache: dict[str, tuple[float, int]]) -> int:
    key = str(path)
    hit = cache.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] <= _SIZE_TTL_S:
        return hit[1]
    n = _dir_bytes(path)
    cache[key] = (now, n)
    return n


def encode_features(feat: Any) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, xyz=np.asarray(feat.xyz, np.float16), des3d=np.asarray(feat.des3d, np.uint8),
                        K=np.asarray(feat.K, np.float32), size=np.asarray(feat.size, np.int32),
                        eye_y=np.float32(feat.eye_y), n_kp=np.int32(len(feat.uv)))
    return buf.getvalue()


def load_features(path: str | Path) -> dict[str, Any]:
    """读回一帧地图侧特征（xyz 转回 float32）。"""
    with np.load(path) as z:
        return {"xyz": z["xyz"].astype(np.float32), "des3d": z["des3d"], "K": z["K"].astype(np.float64),
                "size": (int(z["size"][0]), int(z["size"][1])), "eye_y": float(z["eye_y"]),
                "n_kp": int(z["n_kp"])}


def make_thumbnail(image: np.ndarray, width: int) -> np.ndarray:
    """在感知线程里做：只缩放（~0.2 ms），JPEG 编码留给写盘线程。输入 RGB 或灰度。"""
    h, w = image.shape[:2]
    if w <= width:
        return image.copy()
    return cv2.resize(image, (int(width), max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
class SessionWriter:
    """一次导航会话的记忆写入器。所有公开方法非阻塞、不抛异常（出错记进 status）。"""

    def __init__(self, session_dir: Path, meta: dict[str, Any], cfg: MemoryConfig) -> None:
        self.dir = Path(session_dir)
        self.cfg = cfg
        self._lock = threading.Lock()
        self._meta = {"schema": SCHEMA, "status": "recording", "keyframes": 0, "features": 0, "thumbs": 0,
                      "bytes": 0, "dropped": 0, "size_limited": False, "pinned": False, "label": "",
                      "frame": "session_map_track_m", "error": None, **meta}
        self._feat_ids: set[int] = set()
        self._q: queue.Queue = queue.Queue(maxsize=cfg.queue_max)
        self._closed = False
        (self.dir / "kf").mkdir(parents=True, exist_ok=True)
        (self.dir / "thumb").mkdir(exist_ok=True)
        self._save_meta()
        self._thread = threading.Thread(target=self._run, name="navmesh-memory", daemon=True)
        self._thread.start()

    @property
    def session_id(self) -> str:
        return str(self._meta["session_id"])

    def status(self) -> dict[str, Any]:
        with self._lock:
            m = self._meta
            return {k: m.get(k) for k in ("session_id", "world_key", "status", "keyframes", "features", "thumbs",
                                           "dropped", "size_limited", "error")} | {"mb": round(m["bytes"] / 1e6, 2)}

    def annotate(self, **fields: Any) -> None:
        """补充会话元数据（传感器、基线等启动后才知道的字段）；下次整表写位姿时落盘。"""
        with self._lock:
            self._meta.update(fields)

    # ---- 生产者（建图线程）----
    def keyframe(self, k: int, feat: Any, thumb: np.ndarray | None, refresh: bool) -> None:
        self._put(("kf", int(k), feat, thumb, bool(refresh)))

    def poses(self, table: dict[str, np.ndarray]) -> None:
        self._put(("poses", table))

    def close(self, table: dict[str, np.ndarray] | None, status: str = "complete",
              timeout_s: float = 5.0) -> dict[str, Any]:
        if not self._closed:
            self._closed = True
            self._q.put(("close", table, status))     # 阻塞放入：收尾那一条不能丢
            self._thread.join(timeout=timeout_s)
        return self.status()

    def _put(self, item: tuple) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._meta["dropped"] += 1

    # ---- 写盘线程 ----
    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item[0] == "kf":
                    self._write_kf(*item[1:])
                elif item[0] == "poses":
                    self._write_poses(item[1])
                else:
                    _, table, status = item
                    if table is not None:
                        self._write_poses(table)
                    self._finish(status)
                    return
            except Exception as exc:  # noqa: BLE001 - 记忆坏了不能拖垮导航
                with self._lock:
                    self._meta["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    self._meta["status"] = "error"
                if item[0] == "close":
                    self._save_meta_safe()
                    return

    def _add_bytes(self, n: int) -> None:
        with self._lock:
            self._meta["bytes"] += int(n)
            if self._meta["bytes"] > self.cfg.max_session_mb * 1e6:
                self._meta["size_limited"] = True

    def _write_kf(self, k: int, feat: Any, thumb: np.ndarray | None, refresh: bool) -> None:
        with self._lock:
            limited = self._meta["size_limited"] or self._meta["status"] == "error"
        has_new = not limited and feat is not None and len(feat.des3d) > 0
        if refresh and has_new and (k - 1) in self._feat_ids:
            # 原地补帧：同一视角只留最新一份（与 mapper.drop_points 同口径）。
            self._feat_ids.discard(k - 1)
            for p in (self.dir / "kf" / f"{k - 1:06d}.npz", self.dir / "thumb" / f"{k - 1:06d}.jpg"):
                try:
                    size = p.stat().st_size
                    p.unlink()
                    self._add_bytes(-size)
                except OSError:
                    pass
        if limited:
            return
        if has_new:
            data = encode_features(feat)
            _write_atomic(self.dir / "kf" / f"{k:06d}.npz", data)
            self._feat_ids.add(k)
            self._add_bytes(len(data))
        if thumb is not None:
            img = thumb if thumb.ndim == 2 else cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.thumb_quality)])
            if ok:
                _write_atomic(self.dir / "thumb" / f"{k:06d}.jpg", buf.tobytes())
                self._add_bytes(len(buf))
        with self._lock:
            self._meta["features"] = len(self._feat_ids)
            self._meta["thumbs"] = sum(1 for _ in (self.dir / "thumb").glob("*.jpg"))

    def _write_poses(self, table: dict[str, np.ndarray]) -> None:
        import io

        ids = np.asarray(table["ids"], np.int32)
        buf = io.BytesIO()
        np.savez_compressed(buf, ids=ids, T_map=np.asarray(table["T_map"], np.float32),
                            T_dr=np.asarray(table["T_dr"], np.float32),
                            dist_m=np.asarray(table["dist_m"], np.float32), t_s=np.asarray(table["t_s"], np.float32),
                            has_feat=np.array([int(k) in self._feat_ids for k in ids], bool))
        path = self.dir / "poses.npz"
        old = path.stat().st_size if path.exists() else 0
        _write_atomic(path, buf.getvalue())
        self._add_bytes(buf.tell() - old)
        with self._lock:
            self._meta["keyframes"] = int(len(ids))
            self._meta["poses_written_wall"] = time.time()
        self._save_meta()

    def _finish(self, status: str) -> None:
        with self._lock:
            if self._meta["status"] != "error":
                self._meta["status"] = status
            self._meta["ended_wall"] = time.time()
            too_short = self._meta["keyframes"] < self.cfg.min_session_keyframes
        if too_short:
            # 误启动 / 没怎么走：不留垃圾会话。
            shutil.rmtree(self.dir, ignore_errors=True)
            with self._lock:
                self._meta["status"] = "discarded"
            return
        self._save_meta()

    def _save_meta(self) -> None:
        with self._lock:
            meta = dict(self._meta)
        _write_json(self.dir / "session.json", meta)

    def _save_meta_safe(self) -> None:
        try:
            self._save_meta()
        except OSError:
            pass
class NavMemoryStore:
    """记忆根目录的管理面：开会话、列世界/会话、改标签/钉住、删除、配额清理、中断恢复。

    正在写的会话（``active``）不许改、不许删：它的 session.json 归写盘线程所有。
    线程安全：管理操作全在 ``_lock`` 下，``begin``/``end`` 也是。
    """

    def __init__(self, root: Path, cfg: MemoryConfig | None = None) -> None:
        self.root = Path(root)
        self.cfg = cfg or MemoryConfig()
        self._lock = threading.RLock()
        self._active: SessionWriter | None = None
        self._active_world: Path | None = None
        self._recovered = 0
        self._last_prune: dict[str, Any] | None = None
        self._size_cache: dict[str, tuple[float, int]] = {}

    # ---- 生命周期 ----
    def recover(self) -> int:
        """进程启动时调用：上次没正常结束（崩溃/杀进程）的会话标成 interrupted。位姿以最后一次周期写入为准。"""
        with self._lock:
            n = 0
            for wdir in self._world_dirs():
                for sdir in self._session_dirs(wdir):
                    meta = _read_json(sdir / "session.json")
                    if meta is not None and meta.get("status") == "recording" and self._active_dir() != sdir:
                        meta["status"] = "interrupted" if (sdir / "poses.npz").exists() else "empty"
                        _write_json(sdir / "session.json", meta)
                        n += 1
            self._recovered += n
            return n

    def begin(self, identity: dict[str, Any], meta: dict[str, Any]) -> tuple[SessionWriter | None, str | None]:
        """开一个会话。返回 (writer, None) 或 (None, 不写的原因)。"""
        if not self.cfg.enabled:
            return None, "memory_disabled"
        key = identity.get("world_key")
        if not key:
            return None, "world_unknown"
        with self._lock:
            if self._active is not None:
                return None, "session_already_active"
            wdir = self.root / world_dir_name(str(key))
            wdir.mkdir(parents=True, exist_ok=True)
            winfo = _read_json(wdir / "world.json") or {"world_key": key, "created_wall": time.time()}
            winfo.update({"world_key": key, "world_name": identity.get("world_name"),
                          "world_source": identity.get("world_source"), "last_used_wall": time.time()})
            _write_json(wdir / "world.json", winfo)
            sid = time.strftime("%Y%m%d_%H%M%S")
            n = 1
            while (wdir / "sessions" / sid).exists():
                n += 1
                sid = f"{time.strftime('%Y%m%d_%H%M%S')}_{n}"
            writer = SessionWriter(wdir / "sessions" / sid, {
                "session_id": sid, "world_key": key, "world_name": identity.get("world_name"),
                "world_source": identity.get("world_source"), "started_wall": time.time(), **meta}, self.cfg)
            self._active, self._active_world = writer, wdir
            return writer, None

    def end(self, table: dict[str, np.ndarray] | None, status: str = "complete") -> dict[str, Any] | None:
        with self._lock:
            writer, self._active = self._active, None
            self._active_world = None
        if writer is None:
            return None
        result = writer.close(table, status)
        self.prune()
        return result

    def active_status(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._active is None else self._active.status()

    # ---- 查询 ----
    def summary(self, *, sizes: bool = True) -> dict[str, Any]:
        with self._lock:
            worlds = self.list_worlds(sizes=sizes)
            mbs = [w["mb"] for w in worlds if w.get("mb") is not None]
            return {"root": str(self.root), "enabled": self.cfg.enabled, "worlds": len(worlds),
                    "sessions": sum(w["sessions"] for w in worlds),
                    "mb": round(sum(mbs), 2) if mbs else None, "max_total_mb": self.cfg.max_total_mb,
                    "active": self.active_status(), "recovered": self._recovered, "last_prune": self._last_prune}

    def list_worlds(self, *, sizes: bool = True) -> list[dict[str, Any]]:
        """列世界分区。``sizes=False`` 时**不扫目录算体积**（``mb`` 给上一次缓存的、没有就给 None）。

        体积是纯展示（管理页那一列），但要递归 stat 上万个文件 —— 只想要"有哪些世界 / 各几场会话"
        的调用方（栅格页、主页面下拉）必须能躲开它，否则会把它们的请求超时打爆。见 ``_SIZE_TTL_S``。
        """
        with self._lock:
            out = []
            for wdir in self._world_dirs():
                info = _read_json(wdir / "world.json") or {}
                sessions = self._session_dirs(wdir)
                hit = self._size_cache.get(str(wdir))
                if sizes:
                    mb = round(_cached_dir_bytes(wdir, self._size_cache) / 1e6, 2)
                else:
                    mb = round(hit[1] / 1e6, 2) if hit is not None else None
                out.append({"world_id": wdir.name, "world_key": info.get("world_key"),
                            "world_name": info.get("world_name"), "world_source": info.get("world_source"),
                            "last_used_wall": info.get("last_used_wall"), "sessions": len(sessions),
                            "mb": mb, "active": self._active_world == wdir})
            return sorted(out, key=lambda w: -(w["last_used_wall"] or 0))

    def list_sessions(self, world_id: str) -> list[dict[str, Any]]:
        with self._lock:
            wdir = self._world(world_id)
            return [self._session_info(s) for s in reversed(self._session_dirs(wdir))]

    def session(self, world_id: str, session_id: str) -> dict[str, Any]:
        with self._lock:
            sdir = self._session(world_id, session_id)
            info = self._session_info(sdir)
            info["thumbnails"] = sorted(int(p.stem) for p in (sdir / "thumb").glob("*.jpg"))
            return info

    def thumbnail(self, world_id: str, session_id: str, k: int) -> bytes:
        with self._lock:
            path = self._session(world_id, session_id) / "thumb" / f"{int(k):06d}.jpg"
        try:
            return path.read_bytes()
        except OSError:
            raise KeyError("thumbnail_not_found") from None

    def load_poses(self, world_id: str, session_id: str) -> dict[str, np.ndarray]:
        with self._lock:
            path = self._session(world_id, session_id) / "poses.npz"
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    # ---- 修改 ----
    def update_session(self, world_id: str, session_id: str, *, label: Any = None,
                       pinned: Any = None) -> dict[str, Any]:
        with self._lock:
            sdir = self._session(world_id, session_id)
            self._refuse_active(sdir)
            meta = _read_json(sdir / "session.json") or {}
            if label is not None:
                meta["label"] = str(label).strip()[:_LABEL_MAX]
            if pinned is not None:
                if not isinstance(pinned, bool):
                    raise ValueError("pinned must be a boolean")
                meta["pinned"] = pinned
            _write_json(sdir / "session.json", meta)
            return self._session_info(sdir)

    def delete_session(self, world_id: str, session_id: str) -> dict[str, Any]:
        with self._lock:
            sdir = self._session(world_id, session_id)
            self._refuse_active(sdir)
            freed = _dir_bytes(sdir)
            shutil.rmtree(sdir)
            return {"deleted": [session_id], "freed_mb": round(freed / 1e6, 2)}

    def delete_world(self, world_id: str) -> dict[str, Any]:
        with self._lock:
            wdir = self._world(world_id)
            if self._active_world == wdir:
                raise PermissionError("session_active")
            freed = _dir_bytes(wdir)
            shutil.rmtree(wdir)
            return {"deleted_world": world_id, "freed_mb": round(freed / 1e6, 2)}

    def prune(self) -> dict[str, Any]:
        """配额清理：每个世界超出会话数/字节就删最旧的未钉会话，然后全局超额同样处理。钉住和进行中的永不删。"""
        with self._lock:
            deleted: list[str] = []
            freed = 0
            for wdir in self._world_dirs():
                sessions = [(s, _dir_bytes(s)) for s in self._session_dirs(wdir)]
                total = sum(b for _, b in sessions)
                for sdir, b in list(sessions):
                    if (len(sessions) <= self.cfg.max_sessions_per_world
                            and total <= self.cfg.max_world_mb * 1e6):
                        break
                    if self._protected(sdir):
                        continue
                    shutil.rmtree(sdir, ignore_errors=True)
                    sessions.remove((sdir, b))
                    total -= b
                    freed += b
                    deleted.append(f"{wdir.name}/{sdir.name}")
            everything = sorted(((s, _dir_bytes(s)) for w in self._world_dirs() for s in self._session_dirs(w)),
                                key=lambda x: x[0].name)
            total = sum(b for _, b in everything)
            for sdir, b in everything:
                if total <= self.cfg.max_total_mb * 1e6:
                    break
                if self._protected(sdir):
                    continue
                shutil.rmtree(sdir, ignore_errors=True)
                total -= b
                freed += b
                deleted.append(f"{sdir.parent.parent.name}/{sdir.name}")
            self._last_prune = {"wall": time.time(), "deleted": deleted, "freed_mb": round(freed / 1e6, 2)}
            return dict(self._last_prune)

    # ---- 内部 ----
    def _active_dir(self) -> Path | None:
        return None if self._active is None else self._active.dir

    def _refuse_active(self, sdir: Path) -> None:
        if self._active_dir() == sdir:
            raise PermissionError("session_active")

    def _protected(self, sdir: Path) -> bool:
        return self._active_dir() == sdir or bool((_read_json(sdir / "session.json") or {}).get("pinned"))

    def _world_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if p.is_dir() and (p / "world.json").exists())

    @staticmethod
    def _session_dirs(wdir: Path) -> list[Path]:
        sd = wdir / "sessions"
        return sorted(p for p in sd.iterdir() if p.is_dir()) if sd.is_dir() else []

    def _world(self, world_id: str) -> Path:
        # world_id/session_id 来自 HTTP：只接受目录名本身，拒绝一切路径成分。
        if not isinstance(world_id, str) or not re.fullmatch(r"[0-9A-Za-z._-]{1,100}", world_id) \
                or world_id in {".", ".."}:
            raise ValueError("bad_world_id")
        wdir = self.root / world_id
        if not (wdir / "world.json").exists():
            raise KeyError("world_not_found")
        return wdir

    def _session(self, world_id: str, session_id: str) -> Path:
        wdir = self._world(world_id)
        if not isinstance(session_id, str) or not _SID_RE.fullmatch(session_id):
            raise ValueError("bad_session_id")
        sdir = wdir / "sessions" / session_id
        if not sdir.is_dir():
            raise KeyError("session_not_found")
        return sdir

    def _session_info(self, sdir: Path) -> dict[str, Any]:
        meta = _read_json(sdir / "session.json") or {"status": "corrupt"}
        keep = ("session_id", "status", "label", "pinned", "started_wall", "ended_wall", "keyframes", "features",
                "thumbs", "dropped", "size_limited", "error", "sensors", "baseline_m", "world_scale",
                "odometry_m", "loops", "frame")
        info = {k: meta.get(k) for k in keep}
        info["session_id"] = sdir.name
        info["mb"] = round(_dir_bytes(sdir) / 1e6, 2)
        info["active"] = self._active_dir() == sdir
        return info


def config_from_plugin(mem: Any) -> MemoryConfig:
    """由 ``config.NavmeshMemoryConfig`` 构造（字段同名）。"""
    return MemoryConfig(**{k: getattr(mem, k) for k in asdict(MemoryConfig()) if hasattr(mem, k)})
