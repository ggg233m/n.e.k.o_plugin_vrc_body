"""W1 世界身份子系统：只负责「当前在哪个世界」与启停骨架。

设计约束（按优先级，违反就是 bug）
==============================
1. 最危险的失效模式是把 A 世界的记忆用到 B 世界。它不会报错，只会让 agent
   自信地走错。所以 ``world_key`` 未知或未设置时，**必须拒绝加载任何持久化
   记忆**：``memory_partition()`` 返回 ``None`` 表示拒绝，绝不能「先用上一份
   记忆凑合」。

2. VRChat 的 OSC / OSCQuery 拿不到世界标识（官方地址表只有 ``/avatar/*``、
   ``/input/*``、``/chatbox/*``、``/tracking/*``）。``world_key`` **只能靠用户
   手动输入**。`backend/autonomy.py` 里监听 `world_changed` 的钩子当前全仓
   没有任何发布者，是空接缝——本模块不造假的发布者，也不依赖它。

3. 名字作 key 有同名冲突风险。这是已知且要暴露的风险，不假装它不存在：
   当 ``world_source == "manual_name"`` 时 ``world_conflict_risk`` 为 ``true``。

4. 重启后从 ``<state_dir>/world_identity.json`` **恢复上一次用户显式设置过的 key**
   （2026-10-06 用户裁决）。原来"绝不跨进程恢复"的理由是怕自动沿用错 key —— 但那份风险
   来自**推断**世界，不来自**记住用户亲手设过的值**：文件里只存 ``set_world`` 落下的东西，
   本模块从不从画面/环境/OSC 猜世界（见第 2 条）。恢复出来的身份带 ``restored=true``，
   UI 必须显示成"沿用上次"，用户随时可改；用户显式清空后文件即删，不会复活。
   ``persist=false`` 时整个机制关闭（不写不读）。

5. 不实现深度模型、地点识别、路径规划、日志解析。只留空接口与 TODO，绝不
   伪造 ``hypothesis`` / ``confirmed`` 这类需要后端能力才有的定位状态。

启用与状态语义
==============
- ``available`` 表示子系统是否在配置里启用（``[world_model].enabled``）。
- ``running`` 表示启动完成；``starting`` 是后台启动中间态。
- 现在没有重模型要加载，启动是轻量的；但 ``start()`` 已经实现成「请求立即返回、
  真正初始化在后台线程做」的形式，因为后面接深度模型时启动会包含 OpenVINO
  编译（可能几秒到几十秒），现在不做对，到时候就要返工。
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping

# VRChat 世界 ID 固定以 ``wrld_`` 开头（大小写不敏感）。把它和普通世界名区分开，
# 是因为只有它能作为稳定的世界 key；世界名随时可能重名。
_WORLD_ID_PREFIX = "wrld_"

# 身份落盘的文件名（在 ``<state_dir>`` 下）。只存用户显式设过的值，见模块 docstring 第 4 条。
IDENTITY_FILENAME = "world_identity.json"

# 文件名里只保留这些字符，避免 world_key 里的路径分隔符/空字节污染分区路径。
_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_KEY_LEN = 128


def sanitize_world_key(value: str) -> str:
    """把任意 world_key 规整成安全的单段文件名（navmesh 记忆分区目录也用它）。"""
    safe = _SAFE_KEY.sub("_", value).strip("_.-")
    if not safe:
        safe = "unknown"
    return safe[:_MAX_KEY_LEN]


class WorldModel:
    """线程安全的世界身份与启停骨架。

    W1 范围之外（深度模型、地点识别、路径规划）都以 TODO / 空接口形式存在，
    不在本期实现。
    """

    def __init__(
        self,
        *,
        enabled: bool,
        persist: bool,
        state_dir: Path,
    ) -> None:
        self._lock = threading.RLock()
        self._enabled = bool(enabled)
        self._persist = bool(persist)
        self._state_dir = Path(state_dir)
        # 当前世界身份。启动时**只**从自己落下的 world_identity.json 恢复用户显式设过的值
        # （见模块 docstring 第 4 条）；没有任何推断成分。
        self._world_key: str | None = None
        self._world_name: str | None = None
        self._world_source: str = "unknown"
        self._restored_wall: float | None = None       # 恢复来源文件的写入时刻（UI 用）
        self._restore_identity()
        # 启停状态。
        self._running = False
        self._starting = False
        self._error: str | None = None
        self._start_thread: threading.Thread | None = None

    # ---- 世界身份 ----------------------------------------------------------

    @property
    def identity_path(self) -> Path:
        """身份文件的完整路径（管理页/测试要看它；不落盘时也就没人读）。"""
        return self._state_dir / IDENTITY_FILENAME

    def _restore_identity(self) -> None:
        """启动时恢复上次**用户显式设置**的世界身份。只读自己写的文件，失败一律当没有。"""
        if not self._persist:
            return
        try:
            doc = json.loads(self.identity_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(doc, dict):
            return
        key = doc.get("world_key")
        if not isinstance(key, str) or not key.strip():
            return
        # 恢复也要走一遍归一化：文件可能是手改的，来源因此必须是 manual_*，不许是别的。
        normalized, source = self._normalize(key, doc.get("world_name"))
        if normalized is None:
            return
        name = doc.get("world_name")
        with self._lock:
            self._world_key = normalized
            self._world_source = source
            self._world_name = (str(name).strip() or None) if isinstance(name, str) else None
            if source == "manual_name" and not self._world_name:
                self._world_name = normalized
            self._restored_wall = float(doc.get("saved_wall") or 0.0) or None

    def _save_identity(self) -> None:
        """把当前身份原子写到 ``<state_dir>/world_identity.json``；清空则删文件。

        为什么清空要**删**而不是写 null：文件存在 = 有可沿用的身份。留一个 null 文件
        只会让"用户明确清过"和"从没设过"两种状态长得一样，下次启动的判断就要多一层。
        """
        if not self._persist:
            return
        path = self.identity_path
        if self._world_key is None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        doc = {"world_key": self._world_key, "world_name": self._world_name,
               "world_source": self._world_source, "saved_wall": time.time()}
        tmp = path.with_suffix(".json.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)                      # 原子写：不许读到半个文件
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _normalize(raw_key: Any, raw_name: Any) -> tuple[str | None, str]:
        """把用户输入归一化，返回 ``(归一化 key, 可信度来源)``。

        - 以 ``wrld_`` 开头（忽略大小写与首尾空白）→ 稳定 ID，``manual_id``，
          归一化为小写。
        - 否则视为世界名，``manual_name``（同名冲突风险，调用方负责暴露）。
        - 空输入或清除 → ``(None, "unknown")``。
        """
        if raw_key is None:
            return None, "unknown"
        text = str(raw_key).strip()
        if not text:
            return None, "unknown"
        lowered = text.lower()
        if lowered.startswith(_WORLD_ID_PREFIX):
            return lowered, "manual_id"
        return text, "manual_name"

    def set_world(
        self,
        world_key: Any,
        world_name: Any = None,
    ) -> dict[str, Any]:
        """手动设置当前世界标识。

        世界身份不依赖子系统运行：即使 ``enabled`` 为 false 或未 ``start()``，
        这里也接受设置。返回 POST ``/worldmodel/world`` 契约里需要的字段。
        """
        with self._lock:
            normalized_key, source = self._normalize(world_key, world_name)
            self._world_key = normalized_key
            self._world_source = source
            if normalized_key is None:
                self._world_name = None
            elif source == "manual_id":
                # ID 是稳定的，世界名只作可读展示；没给就用 None。
                name = str(world_name).strip() if isinstance(world_name, str) else None
                self._world_name = name or None
            else:  # manual_name：名字本身就是 key，展示优先用给出的 name。
                name = str(world_name).strip() if isinstance(world_name, str) else None
                self._world_name = name or normalized_key
            self._restored_wall = None                # 用户亲手设的，不再算"沿用上次"
            self._save_identity()
            return self._identity_payload()

    def _identity_payload(self) -> dict[str, Any]:
        """构造 POST ``/worldmodel/world`` 的响应体。调用方已持有锁。"""
        source = self._world_source
        return {
            "accepted": True,
            "world_key": self._world_key,
            "world_source": source,
            "world_conflict_risk": source == "manual_name",
            # 本次进程启动时是从文件恢复的（不是用户刚设的）：UI 必须显示成"沿用上次"，
            # 否则用户会以为是自己设过的、或者以为系统在猜世界。
            "restored": self._restored_wall is not None and self._world_key is not None,
        }

    def identity(self) -> dict[str, Any]:
        """当前世界身份快照：``{"world_key", "world_name", "world_source", "restored", "restored_wall"}``。

        ``restored=true`` 表示这一份是**启动时从 world_identity.json 恢复的**（用户上次显式设过），
        不是用户在本进程里刚设的 —— 也**不是**推断出来的。未知时 ``world_key`` 为 None。
        """
        with self._lock:
            return {"world_key": self._world_key, "world_name": self._world_name,
                    "world_source": self._world_source,
                    "restored": self._restored_wall is not None and self._world_key is not None,
                    "restored_wall": self._restored_wall}

    # ---- 记忆分区 ----------------------------------------------------------

    def memory_partition(self) -> str | None:
        """返回当前世界的持久化记忆分区路径。

        ``world_key`` 为 ``None`` 时返回 ``None`` —— 明确表示**拒绝加载任何
        持久化记忆**。调用方必须据此回退到 empty / unknown，绝不能用别的世界的
        分区凑合。

        注意：W1 阶段还没有可持久化的记忆（places/edges 恒为 0），``persist``
        只门控后续阶段才真正落盘的逻辑；本期仅计算并返回路径，不创建文件。
        """
        with self._lock:
            key = self._world_key
            if key is None:
                return None
            return str(self._state_dir / f"world_memory_{sanitize_world_key(key)}.json")

    # ---- 启停（后台线程，支持耗时启动） ------------------------------------

    def start(self) -> dict[str, Any]:
        """启动子系统。请求立即返回，真正初始化在后台线程完成。

        现在没有重模型要加载，但 ``starting`` 中间态与后台线程已经就位，后面
        接 OpenVINO 编译时无需返工。
        """
        with self._lock:
            if not self._enabled:
                return {
                    "accepted": False,
                    "state": self.status(),
                    "reason": "world_model is disabled in configuration",
                }
            if self._running and not self._starting:
                return {
                    "accepted": True,
                    "state": self.status(),
                    "reason": "already_running",
                }
            if self._starting:
                return {
                    "accepted": True,
                    "state": self.status(),
                    "reason": "already_starting",
                }
            self._starting = True
            self._error = None
            self._start_thread = threading.Thread(
                target=self._start_worker, name="world-model-start", daemon=True
            )
            self._start_thread.start()
            return {
                "accepted": True,
                "state": self.status(),
                "reason": "starting",
            }

    def _start_worker(self) -> None:
        # TODO(W2+): 接深度模型时这里会包含 OpenVINO 编译（可能几秒到几十秒）。
        # 现在没有重模型要加载，只把运行状态置位。
        try:
            with self._lock:
                self._running = True
                self._starting = False
                self._error = None
        except Exception as exc:  # pragma: no cover - 置位本身不应失败
            with self._lock:
                self._starting = False
                self._running = False
                self._error = f"{type(exc).__name__}: {exc}"[:500]

    def stop(self, reason: Any = "manual_stop") -> dict[str, Any]:
        """停止子系统（不停止后端进程）。"""
        with self._lock:
            was_running = self._running or self._starting
            self._running = False
            self._starting = False
            normalized_reason = str(reason or "manual_stop")
            return {
                "accepted": True,
                "reason": normalized_reason if was_running else "not_running",
            }

    # ---- 状态 --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """GET ``/worldmodel/status`` 的契约字段。永不抛异常。"""
        with self._lock:
            running = self._running and not self._starting
            if not running:
                # 未运行时定位能力完全不可用。
                localization = "not_running"
            else:
                # 本期没有地点识别，绝不以 hypothesis/confirmed 伪装定位能力。
                localization = "unknown"
            return {
                "available": self._enabled,
                "running": running,
                "starting": self._starting,
                "world_key": self._world_key,
                "world_name": self._world_name,
                "world_source": self._world_source,
                "world_conflict_risk": self._world_source == "manual_name",
                # 复用锁内的实现：RLock 允许同一线程重入。
                "memory_partition": self._memory_partition_locked(),
                "places": 0,
                "edges": 0,
                "localization": localization,
                "last_error": self._error,
            }

    def _memory_partition_locked(self) -> str | None:
        """``memory_partition()`` 的持锁版本，避免 status() 里重复解释 None。"""
        key = self._world_key
        if key is None:
            return None
        return str(self._state_dir / f"world_memory_{sanitize_world_key(key)}.json")


__all__ = ["WorldModel"]
