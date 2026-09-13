"""同一 MIDI 所有者完成路径准备、服务租约和动作武装。"""
import threading
import uuid
import logging

logger = logging.getLogger(__name__)

from .motion_space import MotionSpace
from .pose_codec import encode_motion_control, midi_events
from .session_pose_world import SessionPoseWorld
from .yui_protocol import MidiEvent


class SessionPoseFactory:
    manages_preparation = True

    def __init__(self, backend, transport, session, *, expired=lambda: False):
        self.backend, self.transport, self.session = backend, transport, session
        self.expired = expired
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stopped = threading.Event()
        self.sender = None
        self.identity = None
        self.pending = None
        self.binding = None
        self.header = None
        self.points = {}
        self.last_log = 0
        self.request_seq = 0
        self.session.add_event_listener(self.ingest)
        self.thread = threading.Thread(target=self._lease, name="yui-world-lease", daemon=True)
        self.thread.start()

    def ready(self, session):
        return (session is self.session and not self.stopped.is_set() and not self.expired()
                and bool(session.world_id) and session.session > 0 and not session.estop
                and session.control_state in {"external", "action", "moving"}
                and {"pose_stream_v1", "pose_operation_v1"} <= set(session.capabilities)
                and (self.identity is None or self.identity == (session.world_id, session.session))
                and (self.sender is None or not self.sender.closed.is_set()))

    def ingest(self, event):
        if "npc" in event:
            if "npc_id" in event and event["npc_id"] != event["npc"]:
                return
            event = dict(event, npc_id=event["npc"])
        with self.condition:
            self.condition.notify_all()
            if self.identity and (event.get("type") == "sys.boot" or not self.ready(self.session)):
                self.stop()
            if not self.pending or not self.ready(self.session):
                return
            if (event.get("world_id"), event.get("session"), event.get("npc_id")) != (*self.identity, "yui"):
                return
            log_seq = event.get("log_seq")
            if type(log_seq) is not int or log_seq <= self.last_log:
                return
            self.last_log = log_seq
            if (event.get("op_id") != self.pending or type(event.get("request_seq")) is not int
                    or event["request_seq"] != self.request_seq or type(event.get("pose_session")) is not int
                    or event["pose_session"] != self.session.session
                    or type(event.get("pose_epoch")) is not int or event["pose_epoch"] <= 0):
                return
            if event.get("type") == "npc.motion_prepared" and self.header is None:
                if type(event.get("path_count")) is int and 1 <= event["path_count"] <= 16:
                    self.header = dict(event)
            elif event.get("type") == "npc.motion_path" and self.header:
                index = event.get("index")
                if (event["pose_epoch"] == self.header["pose_epoch"] and type(index) is int
                        and 0 <= index < self.header["path_count"] and index not in self.points):
                    self.points[index] = event.get("point")
            self.condition.notify_all()

    def prepare(self, session, arguments, instance):
        """只使用新世界回执；不从目录坐标生成可行路径。"""
        with self.condition:
            # operation_completed 可能先于下一条 npc.state；有限等待旧导航退出。
            if self.ready(session) and session.control_state != "external":
                self.condition.wait_for(lambda: session.control_state == "external" or not self.ready(session), .5)
            if not self.ready(session) or session.control_state != "external":
                raise RuntimeError("world_prepare_unavailable")
            # 旧世界要求所有操作结束；注视也占操作槽，忙碌时不取得端口。
            if getattr(session, 'npc_state', {}).get('active_ops'):
                raise RuntimeError("world_busy")
            if arguments.get("player_slot") is not None:
                raise ValueError("interaction_bridge_not_implemented")
            target = arguments.get("target_key")
            anchor = session._target_anchor_id(target) if target else None
            if target and (type(anchor) is not int or not 0 <= anchor <= 126):
                raise ValueError("verified_anchor_unavailable")
            self.binding = None
            self.identity = (session.world_id, session.session)
            if self.sender is None:
                # 内部端口代次在会话内固定；线上的姿态代次由世界单独分配。
                self.sender = self.transport.attach_pose_sender(1)
            self.pending = uuid.uuid4().hex
            self.request_seq += 1
            self.header, self.points = None, {}
            self.last_log = session.last_log_sequence or 0
            try:
                ticket = self._send(4, 1, self.request_seq, self.pending, anchor)
                self.condition.wait_for(lambda: self.header is not None or not self.ready(session), .5)
                if not self.header or not self.ready(session):
                    raise RuntimeError("world_prepare_unconfirmed")
                self.sender.ack(ticket, session.session)
                self.condition.wait_for(lambda: len(self.points) == self.header["path_count"] or not self.ready(session), .5)
                if len(self.points) != self.header["path_count"] or not self.ready(session):
                    raise RuntimeError("world_path_incomplete")
                header = self.header
                space = MotionSpace(origin=header.get("origin"), yaw=header.get("yaw"), scale=header.get("scale"))
                path = [self.points[i] for i in range(header["path_count"])]
                # 原地动作也验证所有回传点，拒绝非有限值和非平地路径。
                converted = space.verified_paths({target or "idle": path}, max_speed=header.get("max_speed"))
                if not target:
                    converted["paths"] = {}
                binding = dict(converted, mode="neko", session=session.session, world_id=session.world_id,
                               revision=self.request_seq, pose_epoch=header["pose_epoch"],
                               op_id=self.pending, instance=instance)
                health = self.backend._request("/world", binding)
                if (health.get("instance") != instance or health.get("world_session") != session.session
                        or health.get("execution_ready") is not True or type(health.get("epoch")) is not int
                        or not self.ready(session)):
                    raise RuntimeError("service_binding_unconfirmed")
                self.binding = binding
                return health
            except Exception as exc:
                logger.warning('YUI_PREPARE_FAILED session=%s request_seq=%s error=%s active_ops=%s',
                               session.session, self.request_seq, type(exc).__name__,
                               len(getattr(session, 'npc_state', {}).get('active_ops', [])))
                self.stop()
                raise

    def _send(self, version, epoch, sequence, op_id, anchor=None):
        packet = encode_motion_control(version, self.session.session, epoch, sequence, op_id, anchor_id=anchor)
        return self.sender.send(tuple(MidiEvent(*e) for e in midi_events(packet)), "pose")

    def __call__(self, session, task):
        with self.lock:
            if (not self.ready(session) or not self.binding or task.get("op_id") != self.binding["op_id"]
                    or task.get("session") != self.binding["session"]):
                raise RuntimeError("world_preparation_mismatch")
            return SessionPoseWorld(session, task, self.sender, self.binding["pose_epoch"],
                                    lambda accepted, epoch: self._send(5, epoch, 1, accepted["op_id"]))

    def _lease(self):
        while not self.stopped.wait(.4):
            with self.lock:
                binding = self.binding
            if not binding:
                continue
            try:
                if not self.ready(self.session):
                    raise RuntimeError("world_binding_lost")
                # HTTP 等待不能占用日志接收锁，否则姿态 ACK 和 finish 回执都会排队。
                health = self.backend._request("/world", binding)
                with self.lock:
                    if self.binding is not binding or self.stopped.is_set():
                        continue
                    if health.get("instance") != binding["instance"] or health.get("execution_ready") is not True:
                        raise RuntimeError("service_binding_lost")
            except Exception:
                with self.lock:
                    if self.binding is binding:
                        self.stop()

    def stop(self):
        # 不等待路径准备锁或 HTTP；在途准备也不能逃过停止。
        self.stopped.set()
        if self.sender:
            self.sender.fault_stop()

    def close(self):
        self.stop()
        if self.thread is not threading.current_thread():
            self.thread.join(2)
        self.session.remove_event_listener(self.ingest)
        if self.sender:
            self.transport.detach_pose_sender()
