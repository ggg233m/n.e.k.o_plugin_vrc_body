"""正式绑定的任务身份与双终态验证，使用假生成器。"""
from copy import deepcopy
import pytest
from .core import MotionService
from .test_service import FakeGenerator


def setup():
    service = MotionService(FakeGenerator())
    binding = dict(mode="neko", session=7, revision=1, root=[0,0], paths={}, max_speed=.5,
                   world_id="world", pose_epoch=19, op_id="a"*32, instance=service.instance)
    service.bind(binding)
    task = service.submit(dict(session=7, epoch=service.epoch, instance=service.instance,
                               prompt="Stand naturally", request_id="request", duration_s=.4))
    service.step()
    service.pull(task)
    identity = dict(world_id="world", session=7, npc_id="yui", op_id=task["op_id"])
    completion = dict(identity, type="npc.pose_completed", state="succeeded", pose_session=7,
                      pose_epoch=19, pose_sequence=5, log_seq=40, production_receipt=True,
                      operation_receipt=dict(identity, type="npc.operation_completed", kind="motion",
                                             result="motion_completed", log_seq=41))
    return service, binding, task, completion


def test_production_completion_requires_same_prepared_operation():
    service, binding, task, completion = setup()
    assert service.health()["execution_ready"]
    assert task["op_id"] == binding["op_id"]
    result = service.ack(dict(task, sequence=0, terminal=True, completion=completion))
    assert result["status"] == "succeeded"
    with pytest.raises(ValueError, match="world_preparation_already_used"):
        service.submit(dict(session=7, epoch=service.epoch, instance=service.instance,
                            prompt="Stand", request_id="another"))


@pytest.mark.parametrize("field,value", [("pose_epoch",18),("pose_sequence",4),("world_id","old"),
    ("op_id","b"*32),("production_receipt",False),("operation_receipt",None),("log_seq",True)])
def test_invalid_terminal_does_not_consume_final_buffer(field,value):
    service, _, task, completion = setup()
    invalid = dict(completion, **{field:value})
    with pytest.raises(ValueError, match="world_completion"):
        service.ack(dict(task, sequence=0, terminal=True, completion=invalid))
    assert len(service.buffer)==1
    assert service.ack(dict(task, sequence=0, terminal=True, completion=completion))["status"]=="succeeded"


def test_old_operation_or_wrong_log_order_rejected():
    service, _, task, completion = setup()
    for changed in ({"op_id":"b"*32}, {"log_seq":39}, {"result":"arrived"}):
        invalid = deepcopy(completion)
        invalid["operation_receipt"].update(changed)
        with pytest.raises(ValueError, match="world_completion_mismatch"):
            service.ack(dict(task, sequence=0, terminal=True, completion=invalid))


def test_expired_binding_removes_execution_readiness():
    service, _, _, _ = setup()
    service.seen -= 3
    assert not service.health()["execution_ready"]
    assert service.health()["world_session"] is None
