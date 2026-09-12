"""共享端口心跳必须消费真实会话 ACK，超时不重发。"""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import _bootstrap
from yui_npc_controller.runtime.yui_transport import YuiReliableTransport
from yui_npc_controller.runtime.pose_shared_port import SharedPort
from yui_npc_controller.runtime.yui_protocol import MidiEvent


@pytest.mark.parametrize("confirmed", [True, False])
def test_heartbeat_returns_credit_only_after_matching_ack(confirmed):
    sink=Mock()
    sender=Mock(sink=sink)
    sender.send.return_value=("heartbeat",23)
    session=SimpleNamespace(session=7,ack_generation=42,wait_for_ack=Mock(
        return_value=SimpleNamespace(ok=True,session=7) if confirmed else None))
    transport=YuiReliableTransport(sink,session,shared_sender=sender)
    if confirmed:
        frame=transport.send_heartbeat()
        session.wait_for_ack.assert_called_once_with(frame.sequence,frame.command_id,frame.request_hash,.4,
            session=7,after_arrival_index=42)
        sender.ack.assert_called_once_with(("heartbeat",23),7)
        sender.stop.assert_not_called()
    else:
        with pytest.raises(TimeoutError): transport.send_heartbeat()
        sender.ack.assert_not_called()
        sender.fault_stop.assert_called_once()
    assert sender.send.call_count==1
    sink.assert_not_called()


def test_heartbeat_ack_advances_actual_port_watermark():
    event=MidiEvent("cc",0,1,1)
    port=SharedPort(lambda e: None,7,1,[event,event])
    assert port.submit("heartbeat",23,[event]*10,0)
    for i in range(10): port.step(i*.002)
    assert port.outstanding==10
    assert not port.ack("heartbeat",8,1,23,.03)
    assert port.ack("heartbeat",7,1,23,.04)
    assert port.outstanding==0
    port.step(.6)
    assert not port.stopped


def test_shared_text_has_bounded_receipt_fences_without_subtitle_motion_coupling():
    sink=Mock()
    sender=Mock(sink=sink)
    transport=YuiReliableTransport(sink,SimpleNamespace(session=7),shared_sender=sender)
    pending=[]
    chunks=[]
    def sent(events): pending.extend(events)
    sender.send.side_effect=sent
    def confirmed():
        chunks.append(len(pending))
        pending.clear()
    transport.send_heartbeat=confirmed
    transport._send_text_payload([MidiEvent("note_on",2,1,1)]*100)
    assert chunks==[32,32,32,4]
    assert not pending
    sink.assert_not_called()


def test_shared_text_stops_payload_when_receipt_fence_fails():
    sink=Mock()
    sender=Mock(sink=sink)
    session=SimpleNamespace(session=7,ack_generation=0,wait_for_ack=Mock(return_value=None))
    transport=YuiReliableTransport(sink,session,shared_sender=sender)
    with pytest.raises(TimeoutError):
        transport._send_text_payload([MidiEvent("note_on",2,1,1)]*100)
    assert sender.send.call_count==33
    sender.fault_stop.assert_called_once()
