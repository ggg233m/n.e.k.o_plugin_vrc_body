"""可选动作后端契约；连接失败不得影响原导航与停止入口。"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
import json
import uuid


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("动作服务不允许重定向")


@dataclass(frozen=True)
class MotionBackendConfig:
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:2346"
    timeout_s: float = 0.5
    poll_s: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        data = dict(value or {})
        enabled = data.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("ardy.enabled 必须为布尔值")
        endpoint = data.get("endpoint", cls.endpoint)
        if not isinstance(endpoint, str):
            raise ValueError("ardy.endpoint 必须为字符串")
        url = urlsplit(endpoint)
        # 提前校验端口，避免错误配置进入后台重试循环。
        if url.port == 0:
            raise ValueError("ardy.endpoint 端口无效")
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("ardy.endpoint 必须为本机 HTTP 服务根地址")
        values = {}
        for key, default, low, high in (("timeout_s", .5, .05, 2.), ("poll_s", 1., .2, 30.)):
            number = data.get(key, default)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high:
                raise ValueError(f"ardy.{key} 超出范围")
            values[key] = float(number)
        return cls(enabled, endpoint.rstrip("/"), **values)


class MotionBackend:
    """后台健康检查；就绪必须同时满足服务和世界执行契约。"""

    def __init__(self, config: MotionBackendConfig, *, changed=None, clock=time.monotonic):
        self.config = config
        self._changed = changed
        self._clock = clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._health: dict[str, Any] = {}
        self._seen = float("-inf")
        self._generation = 0
        self._cancel_uncertain = False
        self._base_signature = None
        self._base_task = None
        self._base_last_check = float("-inf")
        self._execution_dispatch = None
        self._continuous = None
        self._continuous_environment = None
        self._continuous_instance = None
        self._continuous_recovery_pending = False
        self._continuous_retry_at=0.
        self._continuous_retry_count=0
        self._continuous_manage_lock = threading.Lock()
        self._submission_lock = threading.Lock()
        self._intent_lock = threading.RLock()
        self._resume_goal = None
        self._resume_result = None
        self._intent_revision = 0
        self._surface_retry_count = 0
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, path: str, data=None):
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        request = Request(self.config.endpoint + path, data=body, headers=headers)
        # Core40 的 27 骨骼矩阵分段约 220KB，控制响应仍保持原有小上限。
        limit = 524288 if path == "/chunks" else 65536
        with self._opener.open(request, timeout=self.config.timeout_s) as response:
            raw = response.read(limit+1)
        if len(raw) > limit:
            raise ValueError("动作服务响应过大")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("动作服务响应格式无效")
        return result

    def snapshot(self):
        with self._lock:
            fresh = self._clock() - self._seen <= self.config.poll_s * 2 + self.config.timeout_s
            ready = bool(self.config.enabled and not self._stop.is_set() and fresh
                         and self._health.get("protocol") == "yui-motion/1"
                         and self._health.get("ready") is True)
            return {"backend": "ardy" if ready else "legacy", "ready": ready,
                    "state": "ready" if ready else "unavailable" if self.config.enabled else "disabled",
                    "generation": self._generation}

    def world_ready(self, session):
        if self._continuous is not None:
            return self.snapshot()["ready"] and self._continuous.ready(session)
        with self._lock:
            return bool(self.snapshot()["ready"] and not self._cancel_uncertain and session.session > 0
                        and session.control_state in {"external", "moving", "action"}
                        and "pose_stream_v1" in session.capabilities
                        and ((self.manages_preparation()
                              and "neko-pose/1" in self._health.get("execution_protocols", []))
                             or (self._health.get("world_session") == session.session
                                 and self._health.get("execution_ready") is True))
                        and self._execution_dispatch is not None
                        and self._execution_dispatch.ready(session))

    def manages_preparation(self):
        return (self._execution_dispatch is not None
                and getattr(self._execution_dispatch.world_factory, "manages_preparation", False) is True)

    def bind_execution(self, world_factory):
        """安装经过世界握手的适配器工厂；仅在线模型不能使动作工具就绪。"""
        from .motion_dispatch import MotionDispatch
        if self._execution_dispatch:
            self._execution_dispatch.close()
        self._execution_dispatch = MotionDispatch(self, world_factory)
        if self._changed:
            self._changed()

    def execution_status(self):
        if self._continuous is not None:
            return dict(self._continuous.snapshot(),path_recovery=self._resume_result)
        return self._execution_dispatch.snapshot() if self._execution_dispatch else {"status":"unavailable"}

    def unbind_execution(self):
        self._discard_resume()
        self._continuous_environment = None
        if self._continuous is not None:
            self._continuous.close()
            self._continuous = None
        if self._execution_dispatch:
            self._execution_dispatch.close()
            self._execution_dispatch = None

    def configure_continuous(self, transport, session, *, expired=lambda: False):
        self._continuous_environment = (transport, session, expired)

    def refresh_continuous(self):
        if not self._continuous_manage_lock.acquire(blocking=False):
            return
        try:
            self._refresh_continuous_locked()
        finally:
            self._continuous_manage_lock.release()

    def _refresh_continuous_locked(self):
        """仅双方声明v2时接管；旧世界继续走既有适配器。"""
        environment = self._continuous_environment
        if environment is None:
            return
        transport, session, expired = environment
        backend_ready = self.config.enabled and self.snapshot()["ready"] and self._health.get("continuous_protocol") == "yui-motion-stream/2"
        if not backend_ready:
            # 只有真实失联/关闭再恢复的边沿允许同实例重新握手；普通健康轮询不复活失败任务。
            if self._continuous is not None:
                self._continuous_recovery_pending = True
                self._continuous.close()
            return
        if ("pose_stream_v2" not in session.capabilities or session.estop or expired()
            or getattr(transport,'motion_recovery_blocked',False) is True):
            self._discard_resume()
            return
        instance = (self._health.get("instance"), session.world_id, session.session)
        if self._continuous is not None and self._continuous.status=='running' and self._resume_goal:
            with self._intent_lock:
                pending=self._resume_goal
                self._resume_goal=None
                if pending and pending['instance']==instance and self._continuous.ready(session):
                    # 只提交一次；重新查询世界路径，未知响应不能自动重试。
                    result=self._continuous.perform(session,**pending['arguments'])
                    self._resume_result=dict(previous_op_id=pending['op_id'],**result)
        if self._continuous is not None:
            # 失败后的同一身份保持终态；不能通过重复健康检查复活已取消任务。
            if instance == self._continuous_instance and not self._continuous_recovery_pending:
                old=self._continuous
                if ('pose_link_recovery_v1' not in session.capabilities or old.status!='failed'
                    or not old.closed.is_set() or getattr(transport,'motion_recovery_blocked',False)):
                    return
                reason=getattr(getattr(old.sender,'core',None),'reason',None)
                surface_failure=old.error in {'foot_no_ground','unsupported_ground','obstacle','foot_correction_limit','foot_unreachable','pelvis_correction_limit'} and 'pose_surface_recovery_v1' in session.capabilities
                if reason=='explicit_stop' or (old.error not in {'pose_lease_lost','transport_fault','stream_underrun','stream_receive_timeout'}
                    and not surface_failure and reason not in {'receipt_timeout','transaction_expired','sender_stopped'}):return
                if self._clock()<self._continuous_retry_at:return
                # 故障期间低频退避探测；不复用旧任务、帧或模型历史。
                self._continuous_retry_count+=1
                self._continuous_retry_at=self._clock()+min(15.,2**min(self._continuous_retry_count,4))
                try:
                    recovery_revision=self._intent_revision
                    if not transport.recover_motion_link():return
                except Exception:return
                if session.estop or expired() or getattr(transport,'motion_recovery_blocked',False):return
                with self._intent_lock:
                    getter=getattr(old,'resumable_path_goal',None)
                    goal=getter() if getter else None
                    if surface_failure and isinstance(goal,dict):
                        if self._surface_retry_count>=3:
                            self._resume_result=dict(status='failed',error='surface_recovery_exhausted',previous_op_id=goal['op_id'])
                            self._resume_goal=None
                            goal=None
                        else:self._surface_retry_count+=1
                    if isinstance(goal,dict) and recovery_revision==self._intent_revision:
                        self._resume_goal=dict(goal,instance=instance)
                self._continuous_recovery_pending=True
            self._continuous.close()
            self._continuous = None
        if session.control_state != "external":
            return
        if self._execution_dispatch is not None:
            dispatch = self._execution_dispatch
            if dispatch.execution is not None or getattr(dispatch.world_factory,"sender",None) is not None:
                return
            dispatch.close()
            self._execution_dispatch = None
        from .continuous_execution import ContinuousExecution
        self._continuous = ContinuousExecution(self,transport,session,expired=expired)
        self._continuous_instance = instance
        self._continuous_recovery_pending = False
        self._continuous_retry_count=0
        self._base_signature = None
        self._base_task = None
        self._continuous.start()

    def _dispatch_task(self, session, task):
        if task.get("status") != "accepted":
            return task
        try:
            if not self._execution_dispatch:
                raise RuntimeError("execution_bridge_unavailable")
            self._execution_dispatch.start(session, task)
        except Exception:
            # 任务已经受理但桥未启动，取消只针对该任务，不影响更新的任务。
            try:
                self._request("/cancel",{k:task[k] for k in ("session","op_id","epoch")})
            except Exception:
                self._cancel_uncertain=True
            return {"status":"failed","error":"execution_dispatch_failed","op_id":task.get("op_id")}
        return task

    def execution_binding(self, session):
        with self._lock:
            return {"ready":self.world_ready(session),"instance":self._health.get("instance"),
                    "epoch":self._health.get("epoch")}

    def _availability_key(self):
        """世界绑定变化同样需要刷新工具，即使模型始终在线。"""
        with self._lock:
            return (self.snapshot()["ready"], self._health.get("instance"), self._health.get("world_session"),
                    self._health.get("execution_ready") is True, tuple(self._health.get("execution_protocols", [])))

    def cancel(self, session):
        """取消远端缓冲是附加操作；世界端停止必须先独立完成。"""
        self._discard_resume()
        if self._continuous is not None:
            if session.estop:
                self._continuous.stop()
            else:
                self._continuous.close()
        if self._execution_dispatch:
            self._execution_dispatch.stop()
        if not self.config.enabled:
            return
        with self._lock:
            self._cancel_uncertain = True
        try:
            result = self._request("/cancel", {"session": session.session})
            with self._lock:
                if result.get("status") == "cancelled" and type(result.get("epoch")) is int:
                    self._health["epoch"] = result["epoch"]
                    self._cancel_uncertain = False
        except Exception:
            pass

    def _poll(self):
        while not self._stop.is_set():
            before = self._availability_key()
            try:
                health = self._request("/health")
            except Exception:
                health = {}
            with self._lock:
                self._health = health
                self._seen = self._clock()
                after = self._availability_key()
                if before != after:
                    self._generation += 1
            if before != after and self._changed is not None:
                try:
                    self._changed()
                except Exception:
                    # 下轮继续检查；通知失败不能结束健康线程。
                    pass
            self.refresh_continuous()
            self._stop.wait(self.config.poll_s)

    def start(self):
        if not self.config.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="yui-ardy-health", daemon=True)
        self._thread.start()

    def close(self):
        self._discard_resume()
        self._stop.set()
        self._continuous_environment = None
        if self._continuous is not None:
            self._continuous.close()
            self._continuous = None
        if self._execution_dispatch:
            self._execution_dispatch.close()
            self._execution_dispatch=None
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(self.config.timeout_s + .5)
        with self._lock:
            self._health = {}

    def perform(self, session, **arguments):
        if self._continuous is not None:
            with self._intent_lock:
                self._discard_resume()
                return self._continuous.perform(session, **arguments)
        if self.manages_preparation():
            return self._perform_prepared(session, arguments)
        if not self.world_ready(session):
            return {"status": "failed", "error": "backend_unavailable", "midi_sent": False}
        # 请求 ID 由插件创建；超时只返回可追踪的未知状态，不盲目重发。
        request_id = uuid.uuid4().hex
        with self._lock:
            epoch = self._health.get("epoch")
        if type(epoch) is not int:
            return {"status": "failed", "error": "backend_protocol_incomplete", "midi_sent": False}
        try:
            result = self._request("/tasks", {**arguments, "session": session.session,
                                             "request_id": request_id, "epoch": epoch, "instance": self._health.get("instance")})
        except Exception:
            return {"status": "unknown", "error": "backend_request_uncertain", "request_id": request_id, "midi_sent": False}
        if (result.get("status") not in {"accepted", "failed", "cancelled"}
                or result.get("status") == "accepted" and (not result.get("op_id") or result.get("session") != session.session)):
            return {"status": "unknown", "error": "backend_invalid_receipt", "midi_sent": False}
        return self._dispatch_task(session,result)

    def _discard_resume(self):
        # 停止入口不等待正在进行的HTTP请求；世界停止授权仍独立生效。
        self._intent_revision+=1
        self._surface_retry_count=0
        self._resume_goal=None
        if self._continuous is not None:
            discard=getattr(self._continuous,'discard_path_goal',None)
            if discard:discard()

    def _perform_prepared(self, session, arguments):
        if not self.world_ready(session):
            return {"status": "failed", "error": "backend_unavailable", "midi_sent": False}
        if not self._submission_lock.acquire(blocking=False):
            return {"status": "failed", "error": "execution_busy", "midi_sent": False}
        request_id = uuid.uuid4().hex
        submitted = False
        try:
            instance = self._health.get("instance")
            health = self._execution_dispatch.prepare(session, arguments, instance)
            if not self._execution_dispatch.ready(session):
                raise RuntimeError("execution_stopped_during_prepare")
            submitted = True
            result = self._request("/tasks", {**arguments, "session": session.session, "request_id": request_id,
                                             "epoch": health["epoch"], "instance": instance})
            if result.get("status") != "accepted" or result.get("session") != session.session:
                raise RuntimeError("backend_invalid_receipt")
            return self._dispatch_task(session, result)
        except Exception as exc:
            # 准备已发送时未知请求不得重放；停止也覆盖尚未创建执行器的窗口。
            if submitted:
                self._execution_dispatch.stop()
            return {"status": "unknown" if submitted else "failed", "request_id": request_id,
                    "error": "backend_request_uncertain" if submitted else str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "world_prepare_failed"}
        finally:
            self._submission_lock.release()

    def update_base_intent(self, session, intent):
        if self._continuous is not None:
            if not self._continuous.ready(session) or intent.get("version") != 1:
                return
            signature = json.dumps(intent,sort_keys=True)
            if signature == self._base_signature:
                return
            self._base_signature = signature
            result = self._continuous.perform(session,prompt=intent.get("prompt"),source="base",
                duration_s=None,target_key=intent.get("target_key") if intent.get("movement")=="planned_only" else None,
                mode="path" if intent.get("movement")=="planned_only" and intent.get("target_key") else "idle")
            if result.get("status") == "deferred":
                self._base_signature = None
            return
        """只在完整执行就绪后消费语义快照；不调用意图模型，也不重发不确定请求。"""
        if not self.world_ready(session):
            return
        signature = json.dumps([session.session, self._health.get("instance"), intent], sort_keys=True)
        if signature == self._base_signature:
            if not self._base_task or self._clock() - self._base_last_check < 1:
                return
            self._base_last_check = self._clock()
            try:
                status = self._request("/tasks/" + self._base_task).get("status")
            except Exception:
                return
            if status not in {"succeeded", "isolated_completed"}:
                return
            # 只有已确认完成的基础动作可以续期；取消、失败与未知状态不自动复活。
        self._base_signature = signature
        try:
            if self.manages_preparation():
                if (intent.get("version") != 1 or intent.get("movement") not in {"blocked", "planned_only"}
                        or intent.get("player_slot") is not None or intent.get("action_key") is not None):
                    return
                result = self._perform_prepared(session, dict(prompt=intent.get("prompt"), source="base",
                    duration_s=intent.get("duration_s", 4),
                    target_key=intent.get("target_key") if intent["movement"] == "planned_only" else None))
            else:
                result = self._request("/intent", {"session": session.session, "epoch": self._health.get("epoch"),
                    "request_id": uuid.uuid4().hex, "intent": intent, "instance": self._health.get("instance")})
                result = self._dispatch_task(session,result)
            self._base_task = result.get("op_id") if result.get("status")=="accepted" else None
            if result.get('error') == 'world_busy':
                # 未发送的忙碌请求可在旧操作结束后重新尝试。
                self._base_signature = None
        except Exception:
            # 不确定请求仅等待下次语义变化；避免后台循环重复启动动作。
            self._base_task = None

    def request_status(self, session, request_id):
        """供内部诊断查询不确定请求，查询绝不重新启动任务。"""
        try:
            return self._request("/requests", {"session": session.session, "request_id": request_id})
        except Exception:
            return {"status": "unknown", "error": "backend_query_failed"}
