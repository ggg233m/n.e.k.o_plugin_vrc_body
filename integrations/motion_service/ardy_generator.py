"""驻留模型与有界自回归历史；不包含 UI、聊天或 LLM 规划调用。"""
import time
import threading
from collections import OrderedDict
from contextlib import nullcontext, contextmanager


class ArdyGenerator:
    def __init__(self, *, model_name="core40", device="xpu", checkpoints=None, text_device="xpu", graph_dir=None, continuous_graph=True):
        self.ready = False
        self.continuous_graph = continuous_graph
        from pathlib import Path
        if not checkpoints or not Path(checkpoints).is_dir():
            raise ValueError("必须指定现有本地权重目录；独立服务不自动下载模型")
        import torch
        from ardy.model import load_model
        started = time.perf_counter()
        self.torch = torch
        self.model = load_model(model_name, device=device, checkpoints_dir=checkpoints,
                                text_encoder_device=text_device)
        self.model.eval()
        self.load_s = time.perf_counter() - started
        self.graph = None
        self.warmup_s = 0
        if graph_dir:
            from .optimized import install
            self.graph, self.warmup_s = install(self.model, graph_dir)
        self.frames = self.model.gen_horizon_len
        self.fps = float(self.model.motion_rep.fps)
        token = self.model.num_frames_per_token
        self.history_limit = 20 // token * token
        self.generation_lock = threading.RLock()
        self.text_cache = OrderedDict()
        self.task_id = None
        self.history = None
        self.text = None
        self.last_generation_s = None
        self.execution_warmup_s = 0.
        if self.graph:
            # 原图预热只有零文本和伪约束，不能证明真实首段编码/约束/回传已热。
            # 发布 ready 前走完整首段与续段，保留随机状态，清空预热历史。
            warm_started = time.perf_counter()
            index = torch.device(self.model.device).index
            if index is None:
                index = torch.xpu.current_device()
            try:
                with torch.random.fork_rng(devices=[index], device_type="xpu"):
                    constraints = {"root_xz": [[0., 0.]] * self.frames, "origin_xz": [0., 0.], "heading": [0.] * self.frames}
                    self.graph.max_entries = max(self.graph.max_entries, 16)
                    self.graph.enabled = False
                    _, sample = self.generate_candidate("A person stands calmly with relaxed arms.", constraints)
                    # 先编译新历史形状，再捕获图，避免图内落入 XPU 不支持的即时算子。
                    for length in range(token, self.history_limit + 1, token):
                        print(f"[连续历史预热] history={length}, graph={self.graph.enabled}",flush=True)
                        self.generate_candidate("A person stands calmly with relaxed arms.", constraints,
                                                sample[:, -length:].detach().clone())
                    self.graph.enabled = True
                    self.graph.allow_capture = True
                    for length in range(token, self.history_limit + 1, token):
                        print(f"[连续历史预热] history={length}, graph={self.graph.enabled}",flush=True)
                        self.generate_candidate("A person stands calmly with relaxed arms.", constraints,
                                                sample[:, -length:].detach().clone())
                    torch.xpu.synchronize()
            finally:
                self.graph.allow_capture = False
                self.task_id = self.history = self.text = self.last_generation_s = None
            self.execution_warmup_s = time.perf_counter() - warm_started
        self.ready = True

    def reseed(self, seed):
        if type(seed) is not int or not 0<=seed<=0x7fffffff:
            raise ValueError("invalid_motion_seed")
        with self.generation_lock:
            self.torch.manual_seed(seed)
            if str(self.model.device).startswith("xpu"):
                self.torch.xpu.manual_seed_all(seed)

    def history_at(self, sample, generated_frames):
        """返回指定提交边界的独立检查点；被撤销未来帧不进入下一次历史。"""
        history_len = sample.shape[1] - self.frames
        if not 0 <= generated_frames <= self.frames or generated_frames % self.model.num_frames_per_token:
            raise ValueError("invalid_history_boundary")
        end = history_len + generated_frames
        start = max(0, end - self.history_limit)
        return sample[:, start:end].detach().clone() if end else None

    def generate(self, task_id, prompt, constraints):
        """兼容有限任务接口；持续会话使用无提交副作用的 generate_candidate。"""
        with self.generation_lock:
            history = self.history if self.task_id == task_id else None
            result, sample = self.generate_candidate(prompt, constraints, history,
                                                     graph_policy=bool(self.graph and self.graph.enabled))
            self.history = self.history_at(sample, self.frames)
            self.task_id = task_id
            return result

    @contextmanager
    def _select_graph(self, policy=None):
        previous=None
        if self.ready and self.graph:
            previous=self.graph.enabled
            self.graph.enabled=self.continuous_graph if policy is None else policy
        try:yield
        finally:
            if previous is not None:self.graph.enabled=previous

    def generate_candidate(self, prompt, constraints, history=None, *, graph_policy=None):
        """生成候选而不提交会话历史；调用方验证意图版本后决定是否采用。"""
        from ardy.constraints import Root2DConstraintSet, FullBodyConstraintSet, EndEffectorConstraintSet
        torch, model = self.torch, self.model
        started = time.perf_counter()
        with self.generation_lock, self._select_graph(graph_policy), (torch._dynamo.config.patch(recompile_limit=32) if not self.ready else nullcontext()), torch.no_grad(), torch.compiler.set_stance("eager_on_recompile" if self.graph and self.ready else "default"):
            text = self.text_cache.get(prompt)
            if text is None:
                with torch.inference_mode():
                    encoded = model._encode_text([prompt])
                text = (encoded[0].clone().float().contiguous(), encoded[1].clone())
                self.text_cache[prompt] = text
                if len(self.text_cache) > 8:
                    self.text_cache.popitem(last=False)
            else:
                self.text_cache.move_to_end(prompt)
            history_len = 0 if history is None else history.shape[1]
            count = history_len + self.frames
            if self.graph and history_len:
                count = max(64, count)
            sets = []
            points = constraints.get("root_xz")
            if points is not None:
                if len(points) != self.frames:
                    raise ValueError("invalid_constraint_horizon")
                padded = points + [points[-1]] * (count-history_len-len(points))
                heading = constraints.get("heading")
                if heading is not None:
                    if len(heading) != self.frames:
                        raise ValueError("invalid_heading_horizon")
                    heading = torch.tensor(heading + [heading[-1]]*(count-history_len-len(heading)),
                                           dtype=torch.float32, device=model.device)
                selected=constraints.get('root_indices',list(range(count-history_len)))
                if (not isinstance(selected,list) or not selected or any(type(i) is not int or not 0<=i<count-history_len for i in selected)
                    or selected!=sorted(set(selected))):
                    raise ValueError('invalid_root_constraint_indices')
                indices=torch.tensor(selected,dtype=torch.long,device=model.device)
                sets.append(Root2DConstraintSet(model.skeleton,indices+history_len,
                    torch.tensor(padded,dtype=torch.float32,device=model.device)[indices],
                    heading[indices] if heading is not None else None))
            for item in constraints.get("keyframes", []):
                frame = item["frame"]
                if type(frame) is not int or not 0 <= frame < self.frames:
                    raise ValueError("invalid_constraint_frame")
                indices = torch.tensor([history_len+frame], device=model.device)
                positions = torch.tensor([item["positions"]],dtype=torch.float32,device=model.device)
                rotations = torch.tensor([item["rotations"]],dtype=torch.float32,device=model.device)
                if item["kind"] == "fullbody":
                    sets.append(FullBodyConstraintSet(model.skeleton,indices,positions,rotations))
                elif item["kind"] == "end_effector":
                    sets.append(EndEffectorConstraintSet(model.skeleton,indices,positions,rotations,None,
                                                        joint_names=item["joints"]))
                else:
                    raise ValueError("unsupported_constraint_kind")
            observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
                sets,torch.tensor([count],device=model.device),to_normalize=True,device=model.device)
            # 历史已提交，不允许未来约束重新写入历史帧。
            if history_len:
                mask[:, :history_len] = 0
                observed[:, :history_len] = 0
            sample = model.autoregressive_step(
                num_frames=count,num_denoising_steps=min(10,int(model.diffusion.num_base_steps)),
                motion_mask=mask,observed_motion=observed,cfg_weight=(2.,2.),
                text_feat=text[0],text_pad_mask=text[1],init_history_sequence=history,
                init_global_translation=(torch.tensor([[constraints["origin_xz"][0],0.,constraints["origin_xz"][1]]],
                    dtype=torch.float32,device=model.device) if history is None else None))
            if sample.shape[1] != history_len + self.frames or not torch.isfinite(sample).all():
                raise ValueError("invalid_generated_window")
            # 图输出可能复用存储，提交检查点必须拥有自己的张量。
            sample = sample.detach().clone()
            output = model.motion_rep.inverse(sample,is_normalized=True)
            result = {key:output[key][0,history_len:history_len+self.frames].detach().float().cpu().tolist()
                      for key in ("local_rot_mats","root_positions","foot_contacts") if key in output}
            angles=model.motion_rep.get_root_heading_angle(model.motion_rep.unnormalize(sample))
            result["root_heading"]=angles[0,history_len:history_len+self.frames].detach().float().cpu().tolist()
        self.last_generation_s = time.perf_counter()-started
        return result,sample
