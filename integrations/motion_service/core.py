"""有限任务状态机：生成完成不等于世界完成，取消使在途结果失效。"""
from collections import OrderedDict, deque
from .continuous import ContinuousService
from copy import deepcopy
import hashlib
import json
import math
import threading
import time
import uuid


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("invalid_number")
    return float(value)


def compile_constraints(world, task, elapsed, frames, fps):
    """仅沿执行桥提供的已验证路径采样，不推测场景碰撞或落脚点。"""
    origin = world["root"]
    target = task.get("target_key")
    path = world.get("paths", {}).get(target) if target else [origin]
    if not path:
        raise ValueError("verified_path_unavailable")
    points = [origin, *path]
    speed = world["max_speed"]
    positions = []
    for frame in range(frames):
        distance = (elapsed + (frame + 1) / fps) * speed if target else 0
        position = points[-1]
        for left, right in zip(points, points[1:]):
            length = math.dist(left, right)
            if length and distance <= length:
                ratio = distance / length
                position = [a + (b - a) * ratio for a, b in zip(left, right)]
                break
            distance -= length
        positions.append(position)
    return {"root_xz": positions, "movement": "planned_only" if target else "blocked",
            "origin_xz": origin, "coordinate_frame": "ardy_y_up_meters", "world_revision": world["revision"]}


class MotionService:
    def __init__(self, generator, *, clock=time.monotonic, lease_s=2., buffer_size=3):
        self.generator, self.clock = generator, clock
        self.streams = ContinuousService(generator,clock=clock)
        self.lease_s, self.buffer_size = lease_s, buffer_size
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.closed = False
        self.instance = uuid.uuid4().hex
        self.epoch = 0
        self.world = None
        self.seen = float("-inf")
        self.active = None
        self.records = OrderedDict()
        self.keys = OrderedDict()
        self.buffer = deque()
        self.dropped = 0
        self.error = None
        self.thread = None

    def _expire(self):
        if self.world and self.clock() - self.seen > self.lease_s:
            self._cancel("world_lease_expired")
            self.world = None
        if self.active and self.clock() - self.active["created"] > self.active["duration_s"] + 10:
            self._cancel("task_timed_out")

    def health(self):
        with self.lock:
            self._expire()
            ready = bool(self.generator.ready and not self.closed and not self.error)
            return {"protocol": "yui-motion/1", "instance": self.instance, "ready": ready,
                    "state": "failed" if self.error else "ready" if ready else "loading",
                    "fps": self.generator.fps, "segment_frames": self.generator.frames,
                    "world_session": self.world["session"] if self.world else None,
                    "execution_protocols": ["neko-pose/1"],
                    "continuous_protocol": "yui-motion-stream/2",
                    "execution_ready": bool(ready and self.world and self.world["mode"] == "neko"),
                    "mode": self.world["mode"] if self.world else "isolated", "epoch": self.epoch,
                    "buffered": len(self.buffer), "dropped_stale": self.dropped,
                    "error": self.error}

    def open_stream(self, data):
        with self.lock:
            if self.active and self.active["status"] not in {"succeeded","cancelled","failed","isolated_completed"}:
                raise ValueError("legacy_task_active")
            return self.streams.open(data)

    def bind(self, data):
        if self.streams.timeline and not self.streams.timeline.closed:
            raise ValueError("continuous_stream_active")
        session = data.get("session")
        if type(session) is not int or session <= 0 or data.get("mode") not in {"isolated", "neko"}:
            raise ValueError("isolated_session_required")
        revision = data.get("revision")
        if type(revision) is not int or revision < 0:
            raise ValueError("invalid_revision")
        def point(value):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("invalid_xz")
            return [number(x, -10000, 10000) for x in value]
        paths = data.get("paths", {})
        if not isinstance(paths, dict) or len(paths) > 32:
            raise ValueError("invalid_paths")
        checked = {}
        for key, path in paths.items():
            if not isinstance(key, str) or not 1 <= len(key) <= 64 or not isinstance(path, list) or not 1 <= len(path) <= 64:
                raise ValueError("invalid_path")
            checked[key] = [point(p) for p in path]
        world = {"mode": data["mode"], "session": session, "revision": revision, "root": point(data.get("root")),
                 "max_speed": number(data.get("max_speed", 1), 0, 3), "paths": checked}
        if data["mode"] == "neko":
            if self.generator.fps != 20 or not 2 <= self.generator.frames <= 40 or self.generator.frames % 2:
                raise ValueError("unsupported_pose_rate")
            op = data.get("op_id")
            if (not isinstance(op, str) or len(op) != 32 or any(c not in "0123456789abcdef" for c in op)
                    or not isinstance(data.get("world_id"), str) or not 1 <= len(data["world_id"]) <= 128
                    or type(data.get("pose_epoch")) is not int or data["pose_epoch"] <= 0
                    or data.get("instance") != self.instance):
                raise ValueError("invalid_world_preparation")
            world.update(world_id=data["world_id"], pose_epoch=data["pose_epoch"], op_id=op)
        with self.condition:
            self._expire()
            if self.world and session == self.world["session"] and revision < self.world["revision"]:
                raise ValueError("stale_world_revision")
            if self.world != world:
                self._cancel("world_changed")
            self.world, self.seen = world, self.clock()
            self.condition.notify_all()
            return self.health()

    def submit(self, data):
        allowed = {"session", "prompt", "target_key", "replace_active", "request_id", "duration_s", "epoch", "source", "instance"}
        if set(data) - allowed:
            raise ValueError("unknown_task_fields")
        if data.get("instance") != self.instance:
            raise ValueError("stale_service_instance")
        prompt, key = data.get("prompt"), data.get("request_id")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 320:
            raise ValueError("invalid_prompt")
        if not isinstance(key, str) or not 1 <= len(key) <= 96:
            raise ValueError("invalid_request_id")
        duration = number(data.get("duration_s", 4), .4, 30)
        if type(data.get("replace_active", False)) is not bool:
            raise ValueError("invalid_replace")
        if data.get("source", "explicit") not in {"explicit", "base"}:
            raise ValueError("invalid_source")
        fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        with self.condition:
            self._expire()
            identity = (data.get("session"), key)
            if identity in self.keys:
                old_hash, task_id = self.keys[identity]
                if old_hash != fingerprint:
                    raise ValueError("idempotency_conflict")
                return deepcopy(self.records[task_id])
            if not self.world or data.get("session") != self.world["session"]:
                raise ValueError("world_unavailable")
            if type(data.get("epoch")) is not int or data["epoch"] != self.epoch:
                raise ValueError("stale_epoch")
            if not self.health()["ready"]:
                raise ValueError("generator_unavailable")
            if self.active and not data.get("replace_active", False):
                raise ValueError("task_busy")
            task = dict(data, duration_s=duration)
            compile_constraints(self.world, task, 0, 1, self.generator.fps)
            task_id = self.world["op_id"] if self.world["mode"] == "neko" else uuid.uuid4().hex
            if task_id in self.records:
                raise ValueError("world_preparation_already_used")
            self._cancel("replaced")
            task.update(op_id=task_id, epoch=self.epoch, prompt_version=self.epoch,
                        status="accepted", generated=0, delivered=-1, acked=-1,
                        created=self.clock(), world=deepcopy(self.world))
            self.active = task
            self.records[task_id] = self._receipt(task)
            self.keys[identity] = (fingerprint, task_id)
            while len(self.keys) > 128:
                _, (_, old_id) = self.keys.popitem(last=False)
                self.records.pop(old_id, None)
            self.condition.notify_all()
            return deepcopy(self.records[task_id])

    def submit_intent(self, data):
        """消费插件已编译的低频基础动作，不再调用语义模型。"""
        intent = data.get("intent")
        if not isinstance(intent, dict) or intent.get("version") != 1:
            raise ValueError("invalid_intent")
        if intent.get("movement") not in {"blocked", "planned_only"}:
            raise ValueError("invalid_movement")
        if intent.get("player_slot") is not None or intent.get("action_key") is not None:
            raise ValueError("interaction_bridge_not_implemented")
        with self.lock:
            if self.active and self.active.get("source", "explicit") != "base":
                raise ValueError("explicit_task_owns_motion")
            return self.submit({"session": data.get("session"), "epoch": data.get("epoch"), "instance": data.get("instance"),
                "request_id": data.get("request_id"), "source": "base", "replace_active": True,
                "prompt": intent.get("prompt"), "duration_s": intent.get("duration_s", 4),
                "target_key": intent.get("target_key") if intent["movement"] == "planned_only" else None})

    def _receipt(self, task):
        return {k: task[k] for k in ("op_id", "status", "epoch", "prompt_version", "session")}

    def _cancel(self, reason):
        self.epoch += 1
        if self.active:
            receipt = self.records[self.active["op_id"]]
            receipt.update(status="cancelled", error=reason)
        self.active = None
        self.buffer.clear()

    def cancel(self, data):
        with self.condition:
            if self.world and data.get("session") != self.world["session"]:
                raise ValueError("stale_session")
            if "op_id" in data and (not self.active or any(data.get(k)!=self.active[k] for k in ("op_id","epoch"))):
                raise ValueError("stale_task_cancel")
            self._cancel("explicit_stop")
            self.condition.notify_all()
            return {"status": "cancelled", "epoch": self.epoch, "world_stopped": False}

    def status(self, task_id):
        with self.lock:
            self._expire()
            return deepcopy(self.records.get(task_id, {"status": "unknown", "error": "task_not_found"}))

    def request_status(self, data):
        with self.lock:
            entry = self.keys.get((data.get("session"), data.get("request_id")))
            return self.status(entry[1]) if entry else {"status": "unknown", "error": "request_not_found"}

    def pull(self, data):
        with self.condition:
            self._expire()
            if not self.active or any(data.get(k)!=self.active[k] for k in ("op_id","session","epoch")):
                raise ValueError("stale_task")
            # 一次只允许一个未确认分段；拉取重试返回同一段。
            chunk = deepcopy(self.buffer[0]) if self.buffer else None
            if chunk:
                self.active["delivered"] = chunk["sequence"]
            return {"chunk": chunk, "status": self.active["status"]}

    def ack(self, data):
        with self.condition:
            self._expire()
            if type(data.get("terminal", False)) is not bool:
                raise ValueError("invalid_terminal")
            task = self.active
            if not task or any(data.get(k) != task[k] for k in ("op_id", "epoch", "session")):
                raise ValueError("stale_ack")
            seq = data.get("sequence")
            if type(seq) is not int or seq != task["delivered"]:
                raise ValueError("unissued_ack")
            if data.get("terminal") and (task["status"] != "awaiting_world"
                    or seq != task["generated"] - 1 or len(self.buffer) > 1):
                raise ValueError("premature_completion")
            if data.get("terminal") and task["world"]["mode"] == "neko":
                self._validate_completion(task, data.get("completion"))
            if seq > task["acked"]:
                if not self.buffer or self.buffer[0]["sequence"] != seq:
                    raise ValueError("out_of_order_ack")
                self.buffer.popleft()
                task["acked"] = seq
            if data.get("terminal"):
                if task["status"] != "awaiting_world" or self.buffer or task["acked"] != task["generated"] - 1:
                    raise ValueError("premature_completion")
                task["status"] = "succeeded" if task["world"]["mode"] == "neko" else "isolated_completed"
                self.records[task["op_id"]] = self._receipt(task)
                self.active = None
            self.condition.notify_all()
            return deepcopy(self.records[task["op_id"]])

    def _validate_completion(self, task, completion):
        """只接受同一世界动作的双终态；校验失败不能消费末段缓冲。"""
        world = task["world"]
        if not isinstance(completion, dict):
            raise ValueError("world_completion_required")
        operation = completion.get("operation_receipt")
        identity = {"world_id": world["world_id"], "session": task["session"],
                    "npc_id": "yui", "op_id": task["op_id"]}
        # 当前传输固定降采样到 10fps，finish 紧随最后一个姿态帧。
        final_sequence = task["generated"] * (self.generator.frames // 2) + 1
        if (completion.get("production_receipt") is not True
                or completion.get("type") != "npc.pose_completed" or completion.get("state") != "succeeded"
                or any(completion.get(k) != v for k, v in identity.items())
                or any(type(completion.get(k)) is not int for k in ("pose_session", "pose_epoch", "pose_sequence", "log_seq"))
                or (completion["pose_session"], completion["pose_epoch"], completion["pose_sequence"])
                    != (task["session"], world["pose_epoch"], final_sequence)
                or not isinstance(operation, dict) or any(operation.get(k) != v for k, v in identity.items())
                or operation.get("type") != "npc.operation_completed" or operation.get("kind") != "motion"
                or operation.get("result") != "motion_completed" or type(operation.get("log_seq")) is not int
                or not 0 < completion["log_seq"] < operation["log_seq"]):
            raise ValueError("world_completion_mismatch")

    def step(self):
        with self.lock:
            self._expire()
            task = self.active
            if not task or task["status"] == "awaiting_world" or len(self.buffer) >= self.buffer_size:
                return False
            epoch, sequence = self.epoch, task["generated"]
            constraints = compile_constraints(task["world"], task,
                sequence * self.generator.frames / self.generator.fps, self.generator.frames, self.generator.fps)
        # 推理不持有状态锁；停止和健康请求不等待 GPU。
        try:
            output = self.generator.generate(task["op_id"], task["prompt"], constraints)
            json.dumps(output, allow_nan=False)
            roots = output.get("root_positions")
            if not isinstance(roots, list) or len(roots) != self.generator.frames:
                raise ValueError("invalid_pose_frames")
            tolerance = .03 if constraints["movement"] == "blocked" else .2
            for actual, expected in zip(roots, constraints["root_xz"]):
                if (not isinstance(actual,list) or len(actual)!=3
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in actual)
                    or math.dist([actual[0], actual[2]], expected) > tolerance):
                    raise ValueError("root_constraint_violation")
        except Exception as exc:
            with self.lock:
                if self.epoch == epoch:
                    self.records[task["op_id"]].update(status="failed", error=str(exc) if isinstance(exc,ValueError) else "generation_failed")
                    self.active = None
                    self.buffer.clear()
                    # 单次随机生成不满足约束只结束该任务；模型运行故障才熔断服务。
                    if not isinstance(exc,ValueError):
                        self.error = type(exc).__name__
                else:
                    self.dropped += 1
            return False
        with self.condition:
            self._expire()
            if self.epoch != epoch or self.active is not task:
                self.dropped += 1
                return False
            chunk = {"op_id": task["op_id"], "session": task["session"], "epoch": epoch,
                     "prompt_version": task["prompt_version"], "sequence": sequence,
                     "fps": self.generator.fps, "constraints": constraints, "pose": output}
            chunk["final"] = (task["generated"]+1)*self.generator.frames/self.generator.fps >= task["duration_s"]
            self.buffer.append(chunk)
            task["generated"] += 1
            if task["generated"] * self.generator.frames / self.generator.fps >= task["duration_s"]:
                task["status"] = "awaiting_world"
            self.records[task["op_id"]] = self._receipt(task)
            return True

    def start(self):
        def worker():
            while not self.closed:
                if not (self.streams.step() or self.step()):
                    with self.condition:
                        self.condition.wait(.05)
        self.thread = threading.Thread(target=worker, name="ardy-motion-generator", daemon=True)
        self.thread.start()

    def close(self):
        self.streams.close()
        with self.condition:
            self.closed = True
            self._cancel("service_shutdown")
            self.condition.notify_all()
        if self.thread:
            self.thread.join(2)
