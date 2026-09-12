"""复用用户启动器的图重放组件，避免导入 Demo 及其全局补丁。"""
import importlib
from pathlib import Path
import sys
import time


def install(model, directory):
    import torch
    root = Path(directory).resolve()
    for filename in ("core_pipeline.py", "core_warmup.py", "graph_runtime.py", "candidates.py"):
        if not (root / filename).is_file():
            raise FileNotFoundError("缺少图重放组件：" + filename)
    sys.path.insert(0, str(root))
    pipeline = importlib.import_module("core_pipeline")
    warmup = importlib.import_module("core_warmup")
    graph = importlib.import_module("graph_runtime")
    candidates = importlib.import_module("candidates")
    from ardy.model.xpu_inverse import enable_fast_inverse
    from ardy.model.xpu_core8_runtime import enable_core8_runtime
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    enable_fast_inverse(model.motion_rep)
    if not pipeline.enable_core_pipeline(model):
        raise RuntimeError("图重放前置条件不满足")
    enable_core8_runtime(model)
    candidates.install_stats_candidate(model)
    warmup.warmup_core_interactive(model)
    count = 2 if model.gen_horizon_len == 8 else 5
    runtime = graph.GraphRuntime(model, max_entries=count)
    model._core8_interactive_ready = False
    try:
        warmup.warmup_core_interactive(model)
        if runtime.captures != count:
            raise RuntimeError("图捕获数量不完整")
    except Exception:
        runtime.restore()
        raise
    runtime.allow_capture = False
    return runtime, time.perf_counter() - started
