"""单客户端本地验收，走候选宿主适配器和正式世界日志。"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.driver_lock import YuiDriverLease
from runtime.motion_backend import MotionBackend, MotionBackendConfig
from runtime.session_pose_factory import SessionPoseFactory
from runtime.yui_adapter import YuiSemanticAdapter
from runtime.yui_log import YuiOutputLogTailer, newest_vrchat_output_log
from runtime.yui_session import YuiSessionState
from runtime.yui_transport import MidoOutputSink, YuiReliableTransport


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline-only", action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result={"source":"real_vrchat_local_world_bridge", "formal_world_verified":False, "cases":[]}
    session=YuiSessionState()
    events=deque(maxlen=512)
    session.add_event_listener(lambda e: events.append(e) if e.get("type", "").startswith(("npc.motion_", "npc.pose_", "npc.operation_", "sys.session", "sys.error")) else None)
    log_path=newest_vrchat_output_log()
    tailer=YuiOutputLogTailer(session, log_path=log_path, from_end=True, poll_interval_s=.02)
    lease=YuiDriverLease("NEKO_MIDI")
    backend=MotionBackend(MotionBackendConfig(enabled=not args.offline_only))
    request_backend=backend._request
    captured=set()
    def capture(path,data=None):
        reply=request_backend(path,data)
        chunk=reply.get("chunk") if path=="/chunks" else None
        if isinstance(chunk,dict):
            key=(chunk.get("op_id"),chunk.get("sequence"))
            if key not in captured:
                (args.output/("chunk-"+str(key[0])+"-"+str(key[1])+".json")).write_text(json.dumps(chunk),encoding="utf-8")
                captured.add(key)
        return reply
    backend._request=capture
    transport=None
    adapter=None
    sink=None
    try:
        # 未握手的世界只输出一次 boot；同步读完末尾记录，再跟随新增日志。
        # 这些记录只确认测试世界与急停状态，随后仍必须重新 DISCOVER。
        if log_path:
            with log_path.open("rb") as handle:
                handle.seek(max(0,log_path.stat().st_size-2*1024*1024))
                for line in handle.read().splitlines():
                    if b"[NEKO]" in line:
                        try: session.ingest_line(line)
                        except ValueError: pass
        tailer.start()
        deadline=time.monotonic()+45
        while (session.world_id!="wrld_neko_ardy_bridge_lab" or session.estop) and time.monotonic()<deadline: time.sleep(.05)
        if session.world_id!="wrld_neko_ardy_bridge_lab" or session.estop: raise RuntimeError("isolated_world_not_ready")
        lease.acquire()
        sink=MidoOutputSink("NEKO_MIDI")
        transport=YuiReliableTransport(sink, session)
        adapter=YuiSemanticAdapter(transport, session)
        connected=adapter.connect(0)
        result["connection"]=connected
        if connected.get("status")!="succeeded": raise RuntimeError("world_handshake_failed")
        result["capabilities"]=list(session.capabilities)
        # 先以完全禁用的动作后端验证原导航，必须拿到 operation_completed。
        offline=MotionBackend(MotionBackendConfig())
        result["offline_backend_ready"]=offline.world_ready(session)
        for target in ("forward", "home"):
            request=adapter.go_to(target, speed_mps=.5)
            operation=session.wait_for_operation(request.get("operation_id") or request.get("op_id"), 10)
            result["cases"].append(dict(case="legacy_"+target,request=request,operation=operation))
            if not operation or operation.get("status")!="succeeded": raise RuntimeError("legacy_navigation_unconfirmed")
        if not args.offline_only:
            backend.bind_execution(SessionPoseFactory(backend, transport, session))
            backend.start()
            deadline=time.monotonic()+5
            while not backend.world_ready(session) and time.monotonic()<deadline: time.sleep(.05)
            for name, target, prompt in (("idle",None,"A person stands naturally with relaxed arms and subtle breathing."),
                    ("walk","forward","A person walks forward naturally and comes to a balanced stop.")):
                task=backend.perform(session,prompt=prompt,target_key=target,duration_s=4)
                record=dict(case=name,task=task)
                result["cases"].append(record)
                if task.get("status")!="accepted": raise RuntimeError("motion_task_not_started")
                deadline=time.monotonic()+20
                while backend.execution_status().get("status")=="running" and time.monotonic()<deadline: time.sleep(.05)
                record["execution"]=backend.execution_status()
                record["service"]=backend._request("/tasks/"+task["op_id"])
                if record["execution"].get("status")!="succeeded" or record["service"].get("status")!="succeeded":
                    raise RuntimeError("motion_execution_unconfirmed")
                time.sleep(.3)
        if not args.offline_only:
            # 在同一会话和共享 MIDI 端口上重新导航，验证根位移控制权已归还。
            for target in ("home", "forward", "home"):
                request=adapter.go_to(target,speed_mps=.5)
                operation=session.wait_for_operation(request.get("operation_id") or request.get("op_id"),10)
                result["cases"].append(dict(case="legacy_after_motion_"+target,request=request,operation=operation))
                if not operation or operation.get("status")!="succeeded":
                    raise RuntimeError("navigation_recovery_unconfirmed")
            result["navigation_recovery_verified"]=True
        result["status"]="succeeded"
    except Exception as exc:
        result.update(status="failed", error=str(exc) if isinstance(exc,(RuntimeError,ValueError)) else type(exc).__name__)
    finally:
        dispatch=backend._execution_dispatch
        factory=dispatch.world_factory if dispatch else None
        sender=factory.sender if factory else None
        result["final_control_state"]=session.control_state
        if sender:
            result["port"]=dict(stopped=sender.core.stopped,reason=sender.core.reason,
                                outstanding=sender.core.outstanding)
        backend.close()
        if adapter: adapter.close()
        if transport:
            transport.stop_heartbeat()
            transport.close()
        if sink: sink.close()
        tailer.close()
        lease.release()
        (args.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        (args.output/"events.json").write_text(json.dumps(list(events),ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(dict(status=result["status"], error=result.get("error"), output=str(args.output)),ensure_ascii=False))
    if result["status"]!="succeeded": raise SystemExit(1)


if __name__=="__main__": main()
