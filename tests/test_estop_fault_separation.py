"""使用真实发送线程验证抖动、忙碌与人工急停，不连接实际 MIDI。"""
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import _bootstrap
from yui_npc_controller.runtime.pose_sender import PoseSender
from yui_npc_controller.runtime.pose_shared_port import SharedPort
from yui_npc_controller.runtime.yui_protocol import MidiEvent
from yui_npc_controller.runtime.yui_transport import YuiReliableTransport
from yui_npc_controller.runtime.yui_adapter import YuiSemanticAdapter
from yui_npc_controller.runtime.yui_session import YuiSessionState
from test_session_pose_factory import setup


def session_with_ack(delay=0):
    def wait(*args, **kwargs):
        time.sleep(delay)
        return SimpleNamespace(ok=True, session=7, error=None)
    return SimpleNamespace(session=7, ack_generation=0, wait_for_ack=wait,
                           estop=False, set_host_arm_authorized=lambda value: None)


def test_active_look_defers_preparation_without_taking_port_then_can_prepare():
    factory, session, transport, backend, sender, _ = setup()
    session.npc_state = {'active_ops': ['look-operation']}
    try:
        with pytest.raises(RuntimeError, match='world_busy'):
            factory.prepare(session, {}, 'service')
        transport.attach_pose_sender.assert_not_called()
        sender.stop.assert_not_called()
        sender.fault_stop.assert_not_called()
        assert factory.ready(session)
        session.npc_state['active_ops'] = []
        assert factory.prepare(session, {}, 'service')['execution_ready']
    finally:
        factory.close()


def test_delayed_heartbeat_ack_does_not_stop_real_sender():
    sent = []
    sink = sent.append
    sender = PoseSender(sink, 7, 1)
    transport = YuiReliableTransport(sink, session_with_ack(.65), shared_sender=sender)
    try:
        transport.send_heartbeat()
        sender.close(graceful=True)
        assert sender.gracefully_closed
        assert not any(e.type == 'note_on' and e.number == 127 for e in sent)
    finally:
        transport.close()


def test_fault_releases_port_for_heartbeat_without_replaying_pose_or_clear_estop():
    sent = []
    sink = sent.append
    sender = PoseSender(sink, 7, 1)
    transport = YuiReliableTransport(sink, session_with_ack(), shared_sender=sender)
    try:
        sender.send([MidiEvent('note_off', 15, 2, 3)], lane='pose')
        assert sender.closed.wait(1)
        assert sender.core.reason == 'receipt_timeout'
        transport.send_heartbeat()
        assert transport._shared_sender is None
        assert [e.number for e in sent if e.type == 'note_on'] == [11]
        assert sum(e.type == 'note_off' for e in sent) == 1
    finally:
        transport.close()


def test_manual_estop_is_not_released_until_human_clear_entry():
    sent = []
    sink = sent.append
    sender = PoseSender(sink, 7, 1)
    transport = YuiReliableTransport(sink, session_with_ack(), shared_sender=sender)
    try:
        transport.send_estop()
        assert sender.closed.wait(1)
        transport._release_faulted_sender()
        assert transport._shared_sender is sender
        assert transport.motion_recovery_blocked
        transport._send_prebuilt_normal = Mock(return_value=SimpleNamespace(status='succeeded'))
        transport.send_command('CLEAR_ESTOP')
        assert transport._shared_sender is None
        assert not transport.motion_recovery_blocked
        assert transport._send_prebuilt_normal.call_args.args[0].command == 'CLEAR_ESTOP'
    finally:
        transport.close()


def test_control_grace_does_not_extend_pose_deadline():
    sent = []
    port = SharedPort(sent.append, 7, 1, ('estop', 'pose_stop'), pose_pipeline=True)
    port.submit('heartbeat', 1, ['heartbeat'], 0)
    port.step(0)
    port.submit('pose', 1, ['pose'], .01)
    port.step(.01)
    port.step(.52)
    assert port.reason == 'receipt_timeout'
    assert sent == ['heartbeat', 'pose', 'pose_stop']
    port.stop('explicit_stop')
    assert sent[-1] == 'estop'


def test_heartbeat_timeout_is_recoverable_not_estop():
    sent = []
    sink = sent.append
    session = session_with_ack()
    session.wait_for_ack = lambda *args, **kwargs: None
    sender = PoseSender(sink, 7, 1)
    transport = YuiReliableTransport(sink, session, shared_sender=sender)
    try:
        with pytest.raises(TimeoutError):
            transport.send_heartbeat()
        assert sender.closed.wait(1)
        transport.send_heartbeat()
        assert transport._shared_sender is None
        assert not any(e.type == 'note_on' and e.number == 127 for e in sent)
    finally:
        transport.close()


def test_discovery_under_estop_keeps_session_for_manual_recovery():
    session = YuiSessionState()
    session.session = 7
    session.control_state = 'estop'
    session.wait_for_session = Mock(return_value=True)
    session.wait_for_discovery = Mock(return_value=True)
    transport = Mock()
    transport.command_deadline_s = 5
    transport.send_command.return_value = SimpleNamespace(status='succeeded')
    adapter = YuiSemanticAdapter(transport, session)
    result = adapter.connect(0, session=7)
    assert result['error'] == 'estop_latched' and result['already_connected']
    assert adapter._connected_session == 7
    assert [call.args[0] for call in transport.send_command.call_args_list] == ['DISCOVER']
