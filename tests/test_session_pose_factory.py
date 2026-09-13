"""路径准备使用模拟日志验证协议和竞态；不作为真实客户端验收。"""
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import _bootstrap
from yui_npc_controller.runtime.session_pose_factory import SessionPoseFactory


def setup(modify=lambda event: event):
    listeners = []
    session = SimpleNamespace(world_id="world", session=7, estop=False, control_state="external",
        capabilities={"pose_stream_v1", "pose_operation_v1"}, last_log_sequence=10,
        add_event_listener=listeners.append, remove_event_listener=listeners.remove,
        _target_anchor_id=lambda key: 3 if key == "desk" else None)
    sender = SimpleNamespace(closed=threading.Event(), ack=Mock(), stop=Mock(), fault_stop=Mock())
    transport = SimpleNamespace(attach_pose_sender=Mock(return_value=sender), detach_pose_sender=Mock())
    backend = SimpleNamespace(_request=Mock(return_value=dict(instance="service", world_session=7, execution_ready=True, epoch=51)))
    factory = SessionPoseFactory(backend, transport, session)
    def send(*args):
        header = dict(type="npc.motion_prepared", world_id="world", session=7, npc="yui", log_seq=11,
            op_id=factory.pending, pose_session=7, pose_epoch=19, request_seq=factory.request_seq,
            origin=[0,0,0], yaw=90, scale=1, max_speed=.5, path_count=2)
        events = [header, dict(header, type="npc.motion_path", log_seq=12, index=0, point=[0,0,0]),
                  dict(header, type="npc.motion_path", log_seq=13, index=1, point=[1,0,0])]
        for event in events:
            factory.ingest(modify(event))
        return ("pose", 1)
    sender.send = Mock(side_effect=send)
    return factory, session, transport, backend, sender, listeners


def test_preparation_uses_world_epoch_and_converted_path():
    factory, session, transport, backend, sender, listeners = setup()
    try:
        health = factory.prepare(session, {"target_key":"desk"}, "service")
        assert health["epoch"] == 51
        binding = backend._request.call_args.args[1]
        assert binding["pose_epoch"] == 19
        assert binding["paths"]["desk"][-1] == pytest.approx([0,1])
        assert binding["op_id"] == factory.pending
        assert binding["mode"] == "neko"
        sender.ack.assert_called_once_with(("pose",1),7)
        transport.attach_pose_sender.assert_called_once_with(1)
        world = factory(session, dict(session=7, op_id=factory.pending))
        assert world.wire_epoch == 19
        world.close()
    finally:
        factory.close()
    assert not listeners


def test_slow_lease_http_does_not_block_world_log_delivery():
    factory, session, _, backend, _, _ = setup()
    entered, release, delivered = threading.Event(), threading.Event(), threading.Event()
    try:
        factory.prepare(session, {}, "service")
        def slow(*args):
            entered.set()
            release.wait(2)
            return dict(instance="service", execution_ready=True)
        backend._request.side_effect = slow
        assert entered.wait(1)
        def deliver():
            factory.ingest(dict(type="npc.pose_ack"))
            delivered.set()
        thread = threading.Thread(target=deliver)
        thread.start()
        # 同步屏障保证续租已进入 HTTP；日志必须在 HTTP 释放前到达后续监听器。
        assert delivered.wait(.2)
        thread.join(1)
    finally:
        release.set()
        factory.close()


@pytest.mark.parametrize("field,value", [("world_id","old"),("request_seq",99),("pose_epoch",0),("session",8)])
def test_old_or_invalid_preparation_never_binds_service(field,value):
    factory, session, _, backend, sender, _ = setup(lambda event: dict(event, **{field:value}))
    try:
        with pytest.raises(RuntimeError, match="world_prepare_unconfirmed"):
            factory.prepare(session, {}, "service")
        backend._request.assert_not_called()
        sender.ack.assert_not_called()
        assert factory.stopped.is_set()
    finally:
        factory.close()


def test_invalid_height_stops_without_service_submission():
    factory, session, _, backend, sender, _ = setup(
        lambda event: dict(event, point=[1,1,0]) if event.get("index")==1 else event)
    try:
        with pytest.raises(ValueError, match="unsupported_path_height"):
            factory.prepare(session, {"target_key":"desk"}, "service")
        backend._request.assert_not_called()
        sender.fault_stop.assert_called()
        sender.stop.assert_not_called()
    finally:
        factory.close()


def test_stop_during_service_bind_does_not_start_execution():
    factory, session, _, backend, sender, _ = setup()
    def bind(*args):
        factory.stop()
        return dict(instance="service", world_session=7, execution_ready=True, epoch=51)
    backend._request.side_effect = bind
    try:
        with pytest.raises(RuntimeError, match="service_binding_unconfirmed"):
            factory.prepare(session, {}, "service")
        assert factory.binding is None
        assert not factory.ready(session)
    finally:
        factory.close()


def test_missing_capability_never_acquires_port():
    factory, session, transport, _, _, _ = setup()
    session.capabilities = {"pose_stream_v1"}
    try:
        with pytest.raises(RuntimeError): factory.prepare(session, {}, "service")
        transport.attach_pose_sender.assert_not_called()
    finally:
        factory.close()
