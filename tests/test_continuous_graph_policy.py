"""图重放对照不污染旧有限任务或启动预热的执行策略。"""
from types import SimpleNamespace
import pytest
import _bootstrap
from integrations.motion_service.ardy_generator import ArdyGenerator


def test_continuous_graph_policy_restores_on_failure_and_keeps_legacy_override():
    generator=object.__new__(ArdyGenerator)
    generator.ready=True;generator.continuous_graph=False;generator.graph=SimpleNamespace(enabled=True)
    with pytest.raises(ValueError):
        with generator._select_graph():
            assert not generator.graph.enabled
            raise ValueError('candidate_cancelled')
    assert generator.graph.enabled
    with generator._select_graph(True):assert generator.graph.enabled
    assert generator.graph.enabled
    generator.ready=False
    with generator._select_graph():assert generator.graph.enabled
