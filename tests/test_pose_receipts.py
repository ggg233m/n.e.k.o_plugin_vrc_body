import _bootstrap
from yui_npc_controller.runtime.pose_receipts import PoseReceipts


class Sender:
    def __init__(self):
        self.acks=[]
        self.stopped=False
        self.on_send=None
    def send(self,events,lane):
        if self.on_send:
            self.on_send()
        return (lane,99)
    def ack(self,ticket,session):
        self.acks.append((ticket,session))
    def stop(self):
        self.stopped=True


def event(**kwargs):
    return dict(dict(type="npc.pose_ack",state="received",npc_id="yui",world_id="world",
        session=7,pose_session=7,pose_epoch=3,pose_sequence=1,log_seq=50),**kwargs)


def test_wrong_identity_and_old_log_do_not_release_ticket():
    sender=Sender()
    receipts=PoseReceipts(sender,world_id="world",session=7,epoch=3)
    receipts.send(1,[1])
    for fields in (dict(world_id="other"),dict(pose_epoch=2),dict(npc_id="other"),dict(session=True),dict(pose_sequence=2)):
        assert not receipts.ingest(event(**fields))
    assert not sender.acks
    assert receipts.ingest(event())
    assert sender.acks==[(("pose",99),7)]
    receipts.send(2,[1])
    assert not receipts.ingest(event(pose_sequence=2))
    receipts.stop()
    assert not receipts.ingest(event(pose_sequence=2,log_seq=51))


def test_early_ack_is_bound_after_future_returns():
    sender=Sender()
    receipts=PoseReceipts(sender,world_id="world",session=7,epoch=3)
    sender.on_send=lambda: receipts.ingest(event())
    receipts.send(1,[1])
    assert sender.acks==[(("pose",99),7)]
    assert receipts.pending is None


def test_two_inflight_receipts_match_their_own_packet_before_releasing_credit():
    sender=Sender();receipts=PoseReceipts(sender,world_id='world',session=7,epoch=3,max_in_flight=2)
    receipts.send(1,[1]);receipts.send(2,[1])
    assert not receipts.wait_received(2,0)
    assert receipts.ingest(event())
    assert not receipts.wait_received(2,0)
    assert receipts.ingest(event(pose_sequence=2,log_seq=51))
    assert receipts.wait_received(2,0) and receipts.pending is None
