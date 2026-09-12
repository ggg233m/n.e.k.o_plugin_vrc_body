"""候选全身流的单线程端口所有者；仅显式注入时接管旧传输。"""
from concurrent.futures import Future
from queue import Queue, Empty, Full
import threading
import time
from .pose_shared_port import SharedPort
from .yui_protocol import MidiEvent, encode_command


class PoseSender:
    def __init__(self, sink, session, epoch, *, pose_pipeline=False, fault_recovery=False):
        self.sink = sink
        self.core = SharedPort(sink, session, epoch,
            (encode_command("ESTOP",0).events[0], MidiEvent("cc",15,112,1)), pose_pipeline=pose_pipeline,
            fault_events=(MidiEvent('cc',15,113,1),) if fault_recovery else None)
        self._exit_lock=threading.Lock()
        self._manual_stop=False
        self.mailbox = Queue(32)
        self.graceful_requested = threading.Event()
        self.gracefully_closed = False
        self.stop_requested = threading.Event()
        self.closed = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._run,name="yui-pose-port",daemon=True)
        self.thread.start()

    def send(self, events, lane=None):
        events=tuple(events)
        if not events:
            raise ValueError("空事务")
        if lane is None:
            lane = "heartbeat" if events[-1].type=="note_on" and events[-1].number==11 else "control"
            if all(e.channel==2 for e in events):
                lane="text"
        future=Future()
        if self.stop_requested.is_set() or self.graceful_requested.is_set() or self.closed.is_set():
            raise RuntimeError("端口已停止")
        try:
            self.mailbox.put_nowait(("send",lane,events,future))
            self.wake.set()
        except Full:
            raise RuntimeError("端口事务队列已满") from None
        try:
            return future.result(timeout=1)
        except Exception:
            self.fault_stop()
            raise

    def ack(self, ticket, session):
        if ticket is None or self.closed.is_set():
            return
        try:
            self.mailbox.put_nowait(("ack",ticket,session,None))
            self.wake.set()
        except Full:
            self.fault_stop()

    def stop(self):
        # 运行时由唯一 worker 写入；结束后的人工急停在退出锁内仅补发一次。
        with self._exit_lock:
            self._manual_stop=True
            self.stop_requested.set()
            if self.closed.is_set():self.core.stop('explicit_stop')
        self.wake.set()

    def fault_stop(self):
        self.stop_requested.set()
        self.wake.set()

    def _run(self):
        waiting={}
        sequence=0
        try:
            while not self.stop_requested.is_set() and not self.core.stopped:
                now=time.perf_counter()
                try:
                    kind,a,b,future=self.mailbox.get_nowait()
                except Empty:
                    pass
                else:
                    if kind=="ack":
                        self.core.ack(a[0],b,self.core.epoch,a[1],now)
                    else:
                        sequence+=1
                        if self.core.submit(a,sequence,b,now):
                            waiting[(a,sequence)]=future
                        else:
                            future.set_exception(RuntimeError("端口信用不足或事务冲突"))
                self.core.step(now)
                # 正常关闭等待全部回执；异常退出按世界能力选择故障暂停或旧急停。
                if (self.graceful_requested.is_set() and not self.core.stopped
                    and not waiting and self.core.outstanding==0 and self.core.active is None
                    and not any(self.core.queues.values()) and self.mailbox.empty()):
                    self.gracefully_closed=True
                    break
                # 以实际 sink 完成作为 send 返回条件，业务 ACK 仍由旧会话验证。
                for key,future in list(waiting.items()):
                    queued=any(t.sequence==key[1] for q in self.core.queues.values() for t in q)
                    running=self.core.active is not None and self.core.active.sequence==key[1]
                    if not queued and not running:
                        if self.core.stopped or self.core.last_completed != key:
                            future.set_exception(RuntimeError("发送停止或结果不确定"))
                        else:
                            future.set_result(key)
                        del waiting[key]
                if self.core.active is None and not any(self.core.queues.values()) and self.mailbox.empty():
                    self.wake.wait(.01)
                    self.wake.clear()
                else:
                    time.sleep(0)
        except Exception as exc:
            for future in waiting.values():
                if not future.done():
                    future.set_exception(exc)
        finally:
            with self._exit_lock:
                if not self.gracefully_closed:
                    self.core.stop('explicit_stop' if self._manual_stop else 'sender_stopped')
                self.closed.set()
            for future in waiting.values():
                if not future.done():
                    future.set_exception(RuntimeError("端口停止"))
            while True:
                try:
                    kind,a,b,future=self.mailbox.get_nowait()
                except Empty:
                    break
                if future is not None and not future.done():
                    future.set_exception(RuntimeError("端口停止"))

    def close(self, *, close_sink=True, graceful=False):
        if graceful:
            self.graceful_requested.set()
            self.wake.set()
        else:
            self.fault_stop()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            self.fault_stop()
            raise RuntimeError("底层 MIDI 写入仍阻塞，未关闭端口")
        close=getattr(self.sink,"close",None)
        if close and close_sink:
            close()
