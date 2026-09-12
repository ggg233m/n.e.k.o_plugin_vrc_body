"""真实权重最小连续生成验证；结果不代表目标 Unity 骨架已经标定。"""
import json
from pathlib import Path
import time
import math
from .ardy_generator import ArdyGenerator
from .core import compile_constraints


def main():
    import os
    generator = ArdyGenerator(checkpoints=os.environ.get("CHECKPOINTS_DIR"),
        model_name=os.environ.get("YUI_MOTION_MODEL", "core8"), graph_dir=os.environ.get("YUI_MOTION_GRAPH_DIR"))
    report = {"model_load_s": generator.load_s, "warmup_s": generator.warmup_s,
              "profile": "graph" if generator.graph else "eager", "fps": generator.fps, "frames": generator.frames, "segments": []}
    world = {"root": [0, 0], "paths": {"forward": [[0, 2]]}, "max_speed": .6, "revision": 1}
    for task, prompt, target in (("idle", "A person stands relaxed in place.", None),
                                  ("walk", "A person walks forward naturally.", "forward")):
        for index in range(2):
            constraints = compile_constraints(world, {"target_key": target}, index * generator.frames / generator.fps,
                                              generator.frames, generator.fps)
            start = time.perf_counter()
            pose = generator.generate(task, prompt, constraints)
            json.dumps(pose, allow_nan=False)
            if os.environ.get("YUI_MOTION_SAVE_SAMPLES") == "1":
                sample_folder=Path(__file__).parent.parent/"unity_lab"/"motion-samples"
                sample_folder.mkdir(exist_ok=True)
                (sample_folder/f"{task}-{index}.json").write_text(json.dumps(pose,allow_nan=False),encoding="utf-8")
            report["segments"].append({"task": task, "index": index, "seconds": time.perf_counter() - start,
                "frames": len(pose["root_positions"]), "joints": len(pose["local_rot_mats"][0]),
                "first_root": pose["root_positions"][0], "last_root": pose["root_positions"][-1],
                "history_frames": generator.history.shape[1] if generator.history is not None else 0,
                "max_root_error_m": max(math.dist([actual[0], actual[2]], expected)
                    for actual, expected in zip(pose["root_positions"], constraints["root_xz"])),
                "graph_replays": generator.graph.replays if generator.graph else 0,
                "graph_fallbacks": generator.graph.fallbacks if generator.graph else 0})
            print(json.dumps(report["segments"][-1]), flush=True)
    # 在同一个驻留模型上验证 HTTP 受理、真实生成、拉取和隔离回执。
    import secrets
    import threading
    from urllib.request import Request, urlopen
    from .core import MotionService
    from .server import make_server
    service = MotionService(generator, lease_s=10)
    token = secrets.token_hex(32)
    server = make_server(service, token, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(path, data=None):
        request = Request("http://127.0.0.1:" + str(server.server_port) + path,
            data=None if data is None else json.dumps(data).encode(), headers={"Authorization": "Bearer " + token})
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    try:
        health = call("/world", {"mode": "isolated", "session": 77, "revision": 1, "root": [0, 0]})
        task = call("/intent", {"session": 77, "epoch": health["epoch"], "instance": health["instance"], "request_id": "http-smoke",
            "intent": {"version": 1, "prompt": "A person stands relaxed in place.", "movement": "blocked", "duration_s": .4}})
        service.step()
        status = call("/tasks/" + task["op_id"])
        report["http_smoke"] = {"after_generation": status["status"], "execution_ready": call("/health")["execution_ready"]}
        if status["status"] == "awaiting_world":
            chunk = call("/chunks", task)["chunk"]
            report["http_smoke"]["frames"] = len(chunk["pose"]["root_positions"])
            report["http_smoke"]["receipt"] = call("/ack", dict(task, sequence=chunk["sequence"], terminal=True))["status"]
        print(json.dumps(report["http_smoke"]), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(2)
    output = Path(__file__).with_name("smoke-ardy-graph-result.json" if generator.graph else "smoke-ardy-result.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
