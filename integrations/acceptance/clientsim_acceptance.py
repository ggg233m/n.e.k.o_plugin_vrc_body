"""主场景真实 MIDI 验收；动作模型由外部服务提供，接受不等于完成。"""
import json
import math
from pathlib import Path
import sys
import time
import argparse
import random
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.yui_session import YuiSessionState
from runtime.yui_log import YuiOutputLogTailer
from runtime.yui_transport import MidoOutputSink, YuiReliableTransport
from runtime.yui_adapter import YuiSemanticAdapter
from runtime.driver_lock import YuiDriverLease
from runtime.autonomy import AutonomyDirector
from runtime.config import YuiAutonomyConfig
from runtime.motion_backend import MotionBackend, MotionBackendConfig
from runtime.session_pose_factory import SessionPoseFactory

parser = argparse.ArgumentParser()
parser.add_argument('--skip-nav', action='store_true')
parser.add_argument('--autonomy-seconds', type=float, default=0)
parser.add_argument('--motion', action='store_true')
parser.add_argument('--motion-target')
parser.add_argument('--targets', nargs='+')
parser.add_argument('--verify-spatial-fallback', action='store_true')
parser.add_argument('--claim-file', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rejected-step-file', type=Path)
args = parser.parse_args()

base = Path(__file__).parent
output = args.output
output.mkdir()
claim_file = args.claim_file
claim = int(claim_file.read_text())
claim_file.unlink()
session = YuiSessionState()
events = deque(maxlen=4096)
session.add_event_listener(lambda e: events.append(e) if e.get('type', '').startswith(('npc.operation_', 'npc.pose_', 'npc.motion_', 'sys.error')) or (e.get('type') == 'npc.ack' and e.get('ok') is False) else None)
tail = YuiOutputLogTailer(session, log_path=Path.home()/'AppData/Local/Unity/Editor/Editor.log', from_end=True, poll_interval_s=.02)
lease = YuiDriverLease('NEKO_MIDI')
sink = transport = adapter = None
director = backend = None
result = dict(source='unity_clientsim_real_midi', ardy_requested=args.motion, status='running', cases=[])

def save():
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'events.json').write_text(json.dumps(list(events), ensure_ascii=False, indent=2), encoding='utf-8')

try:
    tail.start()
    lease.acquire()
    sink = MidoOutputSink('NEKO_MIDI')
    transport = YuiReliableTransport(sink, session)
    adapter = YuiSemanticAdapter(transport, session)
    result['connection'] = adapter.connect(claim)
    if result['connection'].get('status') != 'succeeded' or session.world_id != 'neko_home_world_formal':
        raise RuntimeError('formal_world_handshake_failed')
    result['catalogs'] = session.catalogs
    # ACK 可能先于排队的状态遥测到达，等真实 external 投影后才发动作。
    deadline = time.monotonic() + 5
    while (session.npc_state.get('state') != 'external' or not session.npc_state.get('pos')) and time.monotonic() < deadline:
        time.sleep(.05)
    if session.npc_state.get('state') != 'external':
        raise RuntimeError('control_state_not_confirmed')
    if args.verify_spatial_fallback:
        fixture = json.loads(args.rejected_step_file.read_text(encoding='utf-8'))
        if math.dist(session.npc_state['pos'], fixture['origin']) > .05:
            raise RuntimeError('spatial_fixture_origin_changed')
        angle = abs((session.npc_state['yaw'] - fixture['yaw'] + 180) % 360 - 180)
        if angle > 1:
            raise RuntimeError('spatial_fixture_heading_changed')
        graph = dict(entry='root', nodes=[
            dict(id='root', type='selector', children=['steps', 'rest'], recover_errors=['target_not_on_navmesh', 'no_path']),
            dict(id='steps', type='sequence', children=['rejected', 'must_not_run']),
            dict(id='rejected', type='move_relative', bearing_deg=fixture['bearing'], distance_m=fixture['distance'], allow_shorter=True, face_travel=True, speed_mps=.5),
            dict(id='must_not_run', type='turn_relative', delta_deg=30),
            dict(id='rest', type='wait', duration_ms=250),
        ])
        request = adapter.plan_manager.submit(graph, origin='autonomy')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            plan = adapter.plan_manager.status(request.get('plan_id'))
            if plan.get('status') in {'succeeded', 'failed', 'cancelled', 'unknown'}:
                break
            time.sleep(.05)
        record = dict(case='confirmed_spatial_rejection_fallback', plan=plan)
        result['cases'].append(record)
        nodes = plan.get('node_status', {})
        record['passed'] = (plan.get('status') == 'succeeded'
            and nodes.get('rejected', {}).get('error') in {'target_not_on_navmesh', 'no_path'}
            and nodes.get('rest', {}).get('status') == 'succeeded'
            and 'must_not_run' not in nodes)
        if not record['passed']:
            raise RuntimeError('spatial_fallback_unconfirmed')
    pending = [] if args.skip_nav else list(session.catalogs.get('anchor', {}).values())
    if args.targets:
        known = {item['semantic_key']: item for item in pending}
        if not set(args.targets) <= set(known):
            raise RuntimeError('test_target_unknown')
        pending = [known[key] for key in args.targets]
    while pending:
        pos = session.npc_state.get('pos')
        if not isinstance(pos, list) or len(pos) != 3:
            raise RuntimeError('npc_position_unknown')
        target = min(pending, key=lambda a: math.dist(pos, a['pos']))
        pending.remove(target)
        record = dict(target=target['semantic_key'], before=dict(session.npc_state))
        result['cases'].append(record)
        record['request'] = adapter.go_to(target['semantic_key'], speed_mps=1.0)
        op_id = record['request'].get('operation_id') or record['request'].get('op_id')
        record['operation'] = session.wait_for_operation(op_id, 60) if op_id else None
        time.sleep(.3)
        record['after'] = dict(session.npc_state)
        record['passed'] = bool(record['operation'] and record['operation'].get('status') == 'succeeded')
        save()
        print(json.dumps(dict(target=record['target'], passed=record['passed'], operation=record['operation']), ensure_ascii=True), flush=True)
        if not record['passed']:
            # 第一处真实失败即停止，保留现场，不自动重放失败动作。
            adapter.stop()
            raise RuntimeError('navigation_unconfirmed')
    if args.autonomy_seconds:
        # 使用主项目的真实自主控制器，测试器不指定访问目标，也不发送聊天。
        telemetry = []
        plans = {}
        submit = adapter.plan_manager.submit
        def capture_submit(*positional, **keywords):
            reply = submit(*positional, **keywords)
            plan_id = reply.get('plan_id')
            if plan_id:
                plans[plan_id] = reply
            return reply
        adapter.plan_manager.submit = capture_submit
        director = AutonomyDirector(adapter, session,
            YuiAutonomyConfig(enabled=True, proactive_chat_enabled=False),
            rng=random.Random(20260911), telemetry_callback=telemetry.append)
        result['autonomy_start'] = director.start()
        started = time.monotonic()
        while time.monotonic() - started < args.autonomy_seconds:
            time.sleep(1)
            result['autonomy'] = director.status()
            result['autonomy_plans'] = {key: adapter.plan_manager.status(key) for key in list(plans)}
            result['autonomy_telemetry'] = list(telemetry)
            result['autonomy_elapsed_s'] = time.monotonic() - started
            save()
            if not result['autonomy']['running'] or session.estop:
                raise RuntimeError('autonomy_stopped_unexpectedly')
        result['autonomy_telemetry'] = telemetry
        director.close()
        director = None
        if result['autonomy']['plans_completed'] < 2 or result['autonomy']['movement_seconds'] <= 0:
            raise RuntimeError('autonomy_coverage_insufficient')
        if result['autonomy']['plans_failed']:
            raise RuntimeError('autonomy_plan_failed')
    if args.motion:
        backend = MotionBackend(MotionBackendConfig(enabled=True))
        backend.bind_execution(SessionPoseFactory(backend, transport, session))
        backend.start()
        deadline = time.monotonic() + 5
        while not backend.world_ready(session) and time.monotonic() < deadline:
            time.sleep(.05)
        motion_cases = [('ardy_idle', None, 'A person stands naturally with relaxed arms and subtle breathing.')]
        if args.motion_target:
            motion_cases.append(('ardy_walk', args.motion_target, 'A person walks naturally toward the target and comes to a balanced stop.'))
        for name, target, prompt in motion_cases:
            task = backend.perform(session, prompt=prompt, target_key=target, duration_s=4)
            record = dict(case=name, task=task)
            result['cases'].append(record)
            if task.get('status') != 'accepted':
                raise RuntimeError('motion_task_not_started')
            deadline = time.monotonic() + 30
            while backend.execution_status().get('status') == 'running' and time.monotonic() < deadline:
                time.sleep(.05)
            record['execution'] = backend.execution_status()
            record['service'] = backend._request('/tasks/' + task['op_id'])
            record['passed'] = record['execution'].get('status') == 'succeeded' and record['service'].get('status') == 'succeeded'
            if not record['passed']:
                raise RuntimeError('motion_execution_unconfirmed')
            time.sleep(.8)
    result['status'] = 'succeeded'
except Exception as exc:
    result.update(status='failed', error=str(exc))
finally:
    if director:
        result['autonomy'] = director.status()
        director.close()
    if backend:
        dispatch = backend._execution_dispatch
        factory = dispatch.world_factory if dispatch else None
        sender = factory.sender if factory else None
        if sender:
            result['port'] = dict(stopped=sender.core.stopped, reason=sender.core.reason, outstanding=sender.core.outstanding)
        backend.close()
    if adapter:
        result['stop'] = adapter.stop()
        adapter.close()
    if transport:
        transport.close()
    elif sink:
        sink.close()
    tail.close()
    lease.release()
    result['tail'] = tail.snapshot()
    save()
    print(json.dumps(dict(status=result['status'], output=str(output)), ensure_ascii=True), flush=True)
if result['status'] != 'succeeded':
    raise SystemExit(1)
