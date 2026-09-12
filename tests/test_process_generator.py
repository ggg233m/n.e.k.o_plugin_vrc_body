"""真实spawn边界验证，不加载权重；CPU忙循环模拟持有GIL的原生推理。"""
import time
import threading
from urllib.request import urlopen
import pytest
import _bootstrap
from integrations.motion_service.process_generator import ProcessGenerator
from integrations.motion_service.server import make_server
from types import SimpleNamespace


class BusyGenerator:
    frames=40
    fps=20
    history_limit=20
    load_s=warmup_s=execution_warmup_s=0.
    def __init__(self):pass
    def generate(self,duration):
        deadline=time.monotonic()+duration
        while time.monotonic()<deadline:pass
        return {'finished':True}
    def reseed(self,seed):return seed


def test_busy_inference_does_not_block_health_and_owned_process_closes():
    generator=ProcessGenerator(factory=BusyGenerator)
    server=make_server(SimpleNamespace(health=lambda:{'ready':generator.ready}), 'x'*32, port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    results=[]
    worker=threading.Thread(target=lambda:results.append(generator.generate(.5)));worker.start()
    try:
        for _ in range(4):
            with urlopen('http://127.0.0.1:'+str(server.server_address[1])+'/health',timeout=.2) as response:
                assert response.status==200
        worker.join(2);assert results==[{'finished':True}]
    finally:
        server.shutdown();server.server_close();generator.close()
    assert not generator.ready and not generator.process.is_alive()


def test_inference_timeout_reclaims_child_without_revival():
    generator=ProcessGenerator(factory=BusyGenerator,request_timeout=.1)
    with pytest.raises(TimeoutError):generator.generate(5.)
    assert not generator.ready and not generator.process.is_alive()
    with pytest.raises(RuntimeError):generator.reseed(1)


def test_checkpoints_remain_independent_after_replacement():
    generator=object.__new__(ProcessGenerator);generator.frames=40;generator.history_limit=20
    sample=[[float(i)] for i in range(60)]
    checkpoint=generator.history_at(sample,4)
    assert checkpoint==[[float(i)] for i in range(4,24)]
    sample[4][0]=-1
    assert checkpoint[0]==[4.]
    with pytest.raises(ValueError):generator.history_at(sample,3)
