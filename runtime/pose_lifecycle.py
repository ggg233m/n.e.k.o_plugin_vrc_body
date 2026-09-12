"""宿主生命周期绑定；后端未就绪保留旧路径，曾中断的会话禁止自动复活。"""
from .pose_receipts import PoseReceipts
import threading


class PoseLifecycle:
    def __init__(self, transport, session):
        self.transport,self.session=transport,session
        self.receipts=None
        self.identity=None
        self.blocked_session=None
        self.lock=threading.RLock()
        self.closed=False
        self.session.add_event_listener(self.ingest)

    def update(self, binding):
        with self.lock:
            if not self.closed:
                self._update(binding)

    def _update(self, binding):
        identity=(self.session.world_id,self.session.session,binding.get("instance"),binding.get("epoch"))
        valid=(binding.get("ready") is True and "pose_stream_v1" in self.session.capabilities
               and self.session.control_state in {"external","moving","action"}
               and identity[0] and type(identity[1]) is int and identity[1]>0
               and isinstance(identity[2],str) and bool(identity[2])
               and type(identity[3]) is int and identity[3]>0)
        if self.receipts is not None:
            if not valid or identity!=self.identity or self.receipts.sender.closed.is_set():
                self.blocked_session=self.identity[:2]
                self.transport.detach_pose_sender()
                self.receipts=None
                self.identity=None
            return
        if valid and identity[:2]!=self.blocked_session:
            sender=self.transport.attach_pose_sender(identity[3])
            self.receipts=PoseReceipts(sender,world_id=identity[0],session=identity[1],epoch=identity[3])
            self.identity=identity

    def ingest(self,event):
        receipts=self.receipts
        if receipts is not None:
            if event.get("type")=="sys.boot" or self.session.session!=receipts.session or self.session.world_id!=receipts.world_id:
                receipts.stop(fault=True)
            else:
                receipts.ingest(event)

    def close(self):
        with self.lock:
            self.closed=True
            self.session.remove_event_listener(self.ingest)
            if self.receipts is not None:
                self.receipts.stop(fault=True)
                self.transport.detach_pose_sender()
                self.receipts=None
