"""有界分段执行桥；世界适配器必须提供帧应用和动作终态，接收 ACK 不代表成功。"""
import threading
import time

from .pose_frames import compile_chunk


class MotionExecution:
    def __init__(self, backend, world, *, clock=time.monotonic):
        self.backend,self.world,self.clock=backend,world,clock
        self.lock=threading.RLock()
        self.cancelled=threading.Event()
        self.thread=None
        self.result={"status":"idle"}

    def start(self, task):
        if not isinstance(task,dict) or task.get("status")!="accepted" or any(not task.get(k) for k in ("op_id","epoch","session","prompt_version")):
            raise ValueError("invalid_accepted_task")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("execution_busy")
            if self.cancelled.is_set():
                raise RuntimeError("execution_stopped_requires_new_binding")
            self.result={"status":"running","op_id":task["op_id"]}
            self.thread=threading.Thread(target=self._run,args=(dict(task),),name="yui-motion-execution",daemon=True)
            self.thread.start()

    def stop(self):
        # 优先切断端口，不等待网络请求、推理线程或世界确认。
        self.cancelled.set()
        self.world.stop()

    def close(self):
        self.stop()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(2)

    def snapshot(self):
        with self.lock:
            return dict(self.result)

    def _check(self):
        if self.cancelled.is_set():
            raise RuntimeError("execution_cancelled")
        if not self.world.is_current():
            raise RuntimeError(getattr(self.world,"failure",None) or "world_binding_lost")

    def _run(self, task):
        armed=False
        started=self.clock()
        timing={"frames":[],"chunk_fetch_ms":0.0}
        try:
            chunk_sequence=0; frame_sequence=1
            startup_deadline=self.clock()+10
            next_frame=None
            while True:
                self._check()
                fetch_started=self.clock()
                reply=self.backend._request("/chunks",task)
                timing["chunk_fetch_ms"]+=(self.clock()-fetch_started)*1000
                self._check()
                chunk=reply.get("chunk")
                if chunk is None:
                    status=self.backend._request("/tasks/"+task["op_id"])
                    if status.get("status") in {"failed","cancelled","unknown"}:
                        raise RuntimeError("generation_unavailable")
                    if armed or self.clock()>startup_deadline:
                        raise RuntimeError("generation_underflow")
                    self.cancelled.wait(.02)
                    continue
                # 首段准备好后才武装，首段模型延迟不会耗尽世界的短接收租约。
                if not armed:
                    # 坏的首段不得先抢占世界控制权；线上的代次在武装后重新编码。
                    compile_chunk(chunk,task,expected_chunk=chunk_sequence,
                                  first_frame=frame_sequence,wire_epoch=task["epoch"])
                    wire_epoch=self.world.begin(task)
                    armed=True
                    next_frame=self.clock()
                    timing["startup_ms"]=(next_frame-started)*1000
                    playback_started=next_frame
                frames=compile_chunk(chunk,task,expected_chunk=chunk_sequence,
                                     first_frame=frame_sequence,wire_epoch=wire_epoch)
                for events in frames:
                    self._check()
                    self.cancelled.wait(max(0,next_frame-self.clock()))
                    self._check()
                    sent_at=self.clock()
                    self.world.receipts.send(frame_sequence,events)
                    send_finished=self.clock()
                    if not self.world.receipts.wait_received(frame_sequence,.5):
                        raise RuntimeError("pose_receipt_timeout")
                    self._check()
                    timing["frames"].append(dict(sequence=frame_sequence,
                        sent_ms=(sent_at-playback_started)*1000,
                        send_ms=(send_finished-sent_at)*1000,
                        ack_ms=(self.clock()-send_finished)*1000))
                    frame_sequence+=1
                    # 以本帧实际开始发送为基准，帧间至少100ms；延迟帧不追赶。
                    # ACK 超过帧周期时，不再额外叠加20ms空等；端口预算仍由共享发送器保证。
                    next_frame=sent_at+.1
                # 正式世界在 finish 后等待最后插值及双终态；中间分段 ACK 仅释放生成缓存。
                if (getattr(self.world,"completion_confirms_application",False) is not True
                        and not self.world.wait_applied(frame_sequence-1,.5)):
                    raise RuntimeError("pose_application_timeout")
                self._check()
                terminal=reply.get("status")=="awaiting_world" and bool(chunk.get("final"))
                completion=None
                if terminal:
                    completion=self.world.finish(task,frame_sequence-1,.5)
                    if (not isinstance(completion,dict) or completion.get("state")!="succeeded" or completion.get("op_id")!=task["op_id"]
                        or (completion.get("pose_session"),completion.get("pose_epoch"),completion.get("pose_sequence"))!=(task["session"],wire_epoch,frame_sequence)):
                        raise RuntimeError("world_completion_missing")
                    self._check()
                receipt=self.backend._request("/ack",dict(task,sequence=chunk_sequence,terminal=terminal,completion=completion))
                if any(receipt.get(k)!=task[k] for k in ("op_id","session","epoch")) or receipt.get("status") not in {"accepted","awaiting_world","isolated_completed","succeeded"}:
                    raise RuntimeError("service_ack_mismatch")
                if terminal:
                    expected = "succeeded" if completion.get("production_receipt") is True else "isolated_completed"
                    if receipt.get("status") != expected:
                        raise RuntimeError("service_completion_unconfirmed")
                    with self.lock:
                        timing["playback_ms"]=(self.clock()-playback_started)*1000
                        self.result={"status":"succeeded","op_id":task["op_id"],"world_receipt":completion,"timing":timing}
                    return
                chunk_sequence+=1
        except Exception as exc:
            was_cancelled=self.cancelled.is_set()
            self.stop()
            try:
                self.backend._request("/cancel",{k:task[k] for k in ("session","op_id","epoch")})
            except Exception:
                pass
            with self.lock:
                self.result={"status":"cancelled" if was_cancelled else "failed",
                             "op_id":task["op_id"],"timing":timing,"error":str(exc) if isinstance(exc,(RuntimeError,ValueError)) else type(exc).__name__}
