# Lightweight Alternatives to End-to-End VLA for a Navigation Agent

**Compiled:** 2026-10-04 · Sources: arXiv full texts, official blogs, Jetson AI Lab benchmarks, GitHub, Papers-with-Code-style leaderboards.
**Note on method:** `web_fetch` was blocked in this environment (DNS resolves all hosts to non-public IPs); full texts were retrieved through PowerShell `Invoke-WebRequest` and parsed locally. Every number below is traced to a fetched source file or a search snippet with a live URL.

---

## 0. Executive summary (read this first)

1. **No VLA on the ObjectNav leaderboards.** As of mid-2026, every top method on Habitat ObjectNav HM3D-v1/v2 is **training-free, zero-shot, and modular** (frontier exploration + semantic map + A*/search + a small scoring model). The end-to-end learned baseline DD-PPO is *dead last* on HM3D-v2 (27.9% SR / 14.2% SPL). ([HM3D v2 leaderboard](https://www.sota2.com/research/sota/object-navigation-on-hm3d-v2))
2. **A 2025 ablation paper found a pure-geometry frontier explorer beats the LLM-guided pipeline** on HM3D at $0 API cost and ~2.5× lower runtime. ([arXiv 2507.20021](https://arxiv.org/html/2507.20021v3))
3. **NaVILA is the strongest existing "decoupled" navigation VLA** and is the closest published thing to what you want — but it is an **8B model at ~1 FPS on an RTX 4090, 18.5 GB FP16 / 8.6 GB W4A16**. That is a datacenter GPU, not edge. ([arXiv 2412.04453](https://arxiv.org/html/2412.04453v2))
4. **The cheapest real VLA that runs anywhere is SmolVLA (450M)** and it beats π0 (3.3B, ~7×) on real-robot pick-place. ([arXiv 2506.01844](https://arxiv.org/html/2506.01844v1))
5. **Small VLAs (ViNT/NoMaD, 30–31M) genuinely run at 4 Hz on Jetson Orin** and give you point-goal/image-goal local control — but **no persistent metric map**. You must bolt on a topological graph or a costmap yourself.
6. **V-JEPA 2 (1B encoder, 1M hours video)** is the strongest published example of a world model doing zero-shot robot planning, and it is 10× cheaper per action than Cosmos diffusion (16 s vs 4 min per action, both on one RTX 4090). ([arXiv 2506.09985](https://arxiv.org/html/2506.09985v1))
7. **Your 5–20 TOPS budget cannot run any generative VLM at interactive rates.** It can run NanoOWL+EfficientViT-SAM at 47.5 FPS / 84.6 mIoU on an Orin 64GB. ([Frontiers 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1693988/full))

**Bottom line:** build a **modular stack** (SLAM + 3D semantic map + geometric/classical planner + a small learned perception head + a ~1–3B VLM called at 0.2–1 Hz for frontier scoring and task decomposition). The single most valuable 2026 reference is **AstraNav-Memory** (Qwen2.5-VL-3B + 16× visual token compression, 62.7% SR / 56.9% SPL on GOAT-Bench lifelong nav) — it is the only paper that simultaneously demonstrates (a) a small VLM as a nav policy, (b) a genuine long-term memory mechanism, and (c) real compute measurements.

---

## 1. Recent VLA scaling and the critical literature

### 1.1 Scale table (params / Hz / data / compute)

| Model | Params | Control / infer rate | Data | Compute | Memory mechanism | Link |
|---|---|---|---|---|---|---|
| **RT-2** (2023) | 55B (PaLI-X) / 55B (PaLM-E) | ~1–3 Hz, action tokens, **not chunked** | Google internal EWD + QT-100K | Closed | **None** (single-frame, no history) | [paper](https://deepmind.google/discover/blog/rt-2-new-model-transfers-web-knowledge-to-robot-action/) |
| **OpenVLA** (2024) | **7B** (Llama-2 + DINOv2+SigLIP) | **~6 Hz on RTX 4090** (bf16, no tricks); 15 GB VRAM; 0.33 s/timestep on A100 | **970k OXE episodes** | **64 A100 × 14 d = 21,500 A100-h**; LoRA FT = 10–15 h on 1 A100 (1.4% params) | **None** — no persistent state, no map | [arxiv](https://arxiv.org/html/2406.09246v1) · [code](https://github.com/openvla/openvla) |
| **π0** (2024) | **3.3B** (PaliGemma 3B + 300M action expert) | **50 Hz chunk** H=50, 10 flow steps; **73 ms on-board total on RTX 4090** (14 enc + 32 obs + 27 flow) / 86 ms off-board; infer every 0.5 s | **~10,000 h** own robot data (903M timesteps, 7 embodiments, 68 tasks) + OXE/Bridge/DROID | "largest robot pre-training mixture ever used for manipulation" | 8-frame context only | [arxiv](https://arxiv.org/html/2410.24164v3) · [openpi](https://github.com/Physical-Intelligence/openpi) |
| **π0-FAST** | same | autoregressive, 2–13× faster but **750 ms inter-chunk latency** | same | ~5× more training than π0 | — | [cited in OFT](https://arxiv.org/html/2502.19645v1) |
| **π0.5** | same family | 76 ms (RTC table) | 2-stage pretraining, "knowledge insulation" | — | — | [arxiv 2504.16054](https://arxiv.org/html/2504.16054v1) |
| **Knowledge Insulation** (PI, 2025) | — | applies to π0/π0.5 | "VLAs that Train Fast, Run Fast, and Generalize Better" | — | — | [pi.website](https://www.pi.website/research/knowledge_insulation) · [arxiv 2505.23705](https://arxiv.org/html/2505.23705v1) |
| **OpenVLA-OFT** (2025) | 7.5B | **109.7 Hz** (K=8 chunk, A100), latency **0.0729 s**; with wrist cam + state: 71.4 Hz / 0.112 s; ALOHA K=25: **77.9 Hz** | LIBERO fine-tuning | 8×A100/H100, 50–150K steps | 853M added params (LoRA 111M + action head 269M + proprio 17M + FiLM 456M) | [openvla-oft.github.io](https://openvla-oft.github.io/) · [arxiv](https://arxiv.org/html/2502.19645v1) |
| **GR00T N1** (2025) | **2.2B** (1.34B Eagle-2 VLM = SmolLM2 LLM + SigLIP-2) | **System 2 @ 10 Hz** on L40, **System 1 @ 120 Hz**; **63.9 ms for a 16-action chunk** on L40 bf16 | 88 h teleop → 827 h synthetic video + 7 real datasets | **~50,000 H100 GPU-hours** pre-training; **~105k L40 GPU-hours** for video synthesis (2 min per 1 s video) | None explicit | [arxiv](https://arxiv.org/html/2503.14734v2) · [HF](https://huggingface.co/nvidia/GR00T-N1-2B) |
| **GR00T N1.5 / N1.6** | N1.6 = "2.5× faster rollout" | — | — | — | — | [N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5) · [N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) |
| **GR00T N1.7** (Apr 2026) | **3.14B** (nvidia/GR00T-N1.7-3B) | 16 GB VRAM minimum for inference; 16 GB+ for fine-tuning | "human data is the most scalable source of robot intelligence" | — | — | [HF blog](https://huggingface.co/blog/nvidia/gr00t-n1-7) · [arena](https://www.ay-robots.com/arena/groot-n1-7) |
| **Figure Helix** (2025) | **S2 = 7B VLM, S1 = 80M visuomotor** | **S2 @ 7–9 Hz**, S1 @ 200 Hz | Closed | Closed | — | [figure.ai/news/helix](https://www.figure.ai/news/helix) |
| **Helix 02** (Jan 2026) | — | full-body control | Closed | Closed | — | [figure.ai/helix](https://www.figure.ai/helix) |
| **Gemini Robotics** (2025) | Gemini 2.0 Flash-based, closed | ASIMOV benchmark | — | Cloud API | None | [blog](https://deepmind.google/models/gemini-robotics/) |
| **Gemini Robotics On-Device** (Jun 2025) | "on-device Gemma models", closed | **low-latency local inference**; adapts from **50–100 demos**; ALOHA → bi-arm Franka FR3 + Apollo humanoid | Trusted tester program | Runs on the robot itself | — | [blog](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/) · [model card](https://genie.caelinya.im/_ext/storage/deepmind-media/Model-Cards/Gemini-Robotics-On-Device-Model-Card.pdf) |
| **Gemini Robotics 1.5 / ER 1.5** (Sep 2025) | closed | Gemini Robotics 1.5 adds embodied reasoning + thinking + motion transfer | — | — | — | [tech report](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1-5-Tech-Report.pdf) · [arxiv 2510.03342](https://arxiv.org/html/2510.03342v3) |
| **SmolVLA** (2025) | **450M** (~100M action expert), SmolVLM2-500M backbone; layer-skipping + interleaved cross-attn | Runs on **consumer GPUs or CPUs**; async inference: **9.7 s vs 13.75 s per pick-place (~30% faster), 19 vs 9 cycles in 60 s** | LeRobot community datasets | **~30k GPU-hours total project**; trainable on 1 GPU | 8-frame context | [HF blog](https://huggingface.co/blog/smolvla) · [arxiv](https://arxiv.org/html/2506.01844v1) |
| **Octo** | 93M (27M small) | diffusion chunk | OXE | — | — | cited in [π0](https://arxiv.org/html/2410.24164v3) |
| **RDT-1B** | 1.2B | **84.1 Hz** on A100 (K=25) | 6K bimanual episodes | — | — | [OFT Table III](https://arxiv.org/html/2502.19645v1) |

### 1.2 The critical literature — what actually goes wrong

**Latency vs. control frequency is the single hardest constraint.**
- Physical Intelligence's own Real-Time Chunking paper states it plainly: *"with an RTX 4090 GPU, the 3 billion parameter π0 VLA spends 46 ms on the KV cache prefill alone, before any denoising steps, and targets a 50 Hz control frequency (Δt = 20 ms). Run in remote inference for mobile manipulation, π0 lists 13 ms of network latency, in perfect conditions with a wired connection."* And: *"Kim et al., who optimize the 7B OpenVLA model specifically for inference speed, achieve no better than 321 ms of latency on a server-grade A100 GPU."* ([arXiv 2506.07339](https://arxiv.org/html/2506.07339v2))
- Even with RTC, π0.5 needs **97 ms** per chunk (2.5× the 76 ms synchronous baseline) and they only run the *6th* robot. ([RTC Table 3](https://arxiv.org/html/2506.07339v2))
- **Action-chunk execution horizon is an unacknowledged free parameter.** "VLA Knows Its Limits" (2026) shows π0.5 on LIBERO has a *peaked* success curve in execution horizon — success first rises then collapses, and *"when e > p, performance drops sharply."* Real-robot observation: at `e ≤ 5` the robot "hesitates or stalls"; at `e > 40` it "struggles to maintain accurate object localization, leading to frequent object drops." ([arXiv 2602.21445](https://arxiv.org/html/2602.21445v1))
- **Execution horizon is usually a human heuristic, not measured** — and the optimal value differs by task and model with no generalizable pattern.

**Benchmark success ≠ embodied reasoning.** The 2026 *BeTTER* benchmark is the sharpest critique:
> "We reveal that state-of-the-art VLAs catastrophically fail in dynamic scenarios, exhibiting severe lexical-kinematic shortcuts, behavioral inertia, and semantic feature collapse. Crucially, our mechanistic analysis traces these symptoms to fundamental architectural bottlenecks — such as **capacity compression and myopic downsampling** — which systematically degrade the model's foundational semantic representation. We demonstrate that **highly static evaluation protocols effectively mask this degradation** by allowing optimization to overfit to sensorimotor priors." ([arXiv 2604.18000](https://arxiv.org/abs/2604.18000), Apr 2026)

**VLA long-horizon failures are attributed to a missing harness, not a missing model.** AdaHVLA (Sep 2026): *"Vision-language-action (VLA) models offer strong local control and instruction following but often struggle with long-horizon tasks requiring persistent memory and planning. Task harnesses provide persistent context for agent reasoning by retaining task history and tracking progress across execution."* ([arXiv 2609.29204](https://arxiv.org/abs/2609.29204)) — this is an admission that the memory/planning layer *is* a separate system you must build.

**OpenVLA's own zero-shot claim is weak.** OpenVLA's fine-tuned numbers are strong (beats RT-2-X by 16.5% absolute with 7× fewer params), but the paper's own framing — it outperforms RT-2-X on *fine-tuned* WidowX/Google Robot tasks — plus OFT's observation that *"VLAs struggle with novel robot setups and require fine-tuning to achieve good performance"* is the real story. ([OpenVLA](https://arxiv.org/html/2406.09246v1), [OFT](https://arxiv.org/html/2502.19645v1))

**VLA-Bench / LIBERO-style evaluation is contested.** Search-surfaced titles: *"The Evaluation Bottleneck of Vision-Language-Action Models"* (2026), *"Where Did It Go Wrong? Capability-Oriented Failure Attribution for Vision-and-Language Navigation Agents"* (ACL Findings 2026) — the latter notes VLN failures are hard to localize because *"perception, memory, planning, decision"* are coupled ([aclanthology](https://aclanthology.org/2026.findings-acl.1402.pdf)).

**Data-hunger / training cost is brutal.** OpenVLA 21,500 A100-h. GR00T N1 50,000 H100-h **plus** 105,000 L40-h just to synthesize its own video data. π0 uses ~10,000 h of exclusively-collected robot data. None of this is available to you.

---

## 2. "Decouple reasoning from control" — the pattern, in depth

### 2.1 NaVILA — the flagship (and its hard limits)

[arXiv 2412.04453](https://arxiv.org/html/2412.04453v2) · [project](https://navila-bot.github.io/) · [code](https://github.com/AnjieCheng/NaVILA) (UCSD / USC / NVIDIA, RSS'25, Cheng et al.)

**Architecture — exactly the pattern you described:**
- **Level 2 (slow, VLM):** a **VILA-8B** image-based VLM (deliberately *not* a video encoder — image VLMs generalize better; frames are uniformly sampled from history with the latest frame singled out as "current observation"). Fine-tuned to emit **mid-level actions as natural language with metric arguments**: `"moving forward 75cm"`, `"turn right 30 degrees"`. Parsed with a regex at inference — "we empirically found that all actions throughout all experiments are successfully matched and mapped."
- **Level 1 (fast, RL):** a **vision-based locomotion policy** (PPO, single-stage, not teacher-student) on a Unitree Go2. Action = 12 joint position targets. Perception = **2.5D height map built from LiDAR** (Unitree L1, 15 Hz, 360°×90° FOV, min-filter per voxel + max-filter over 5 sweeps). Casts language to fixed command velocities {0.5 m/s, ±π/6 rad/s, 0} and runs for the stated duration. Critic gets privileged terrain scan; actor only gets real-available sensors. **Locomotion RL trained at >60,000 FPS on one RTX 4090** (Isaac Lab ray-casting).
- **Does the VLM run off a costmap? No.** This is the important negative. NaVILA is **mapless** — it has no occupancy grid, no SLAM, no global planner, no odometry. The VLM's only spatial memory is **8 sampled history frames** (ablation: "8 frames are sufficient to cover most instruction horizons"; they use 8 in the real world "due to latency constraints"). The two levels communicate only through a scalar velocity command.

**Compute (the killer number):**
> "The first two stages of NaVILA are inherited from VILA, which is trained on **16 A100 GPU nodes, with each node having 8 GPUs**… connector initialization takes 4 hours, visual language pre-training takes 30 hours. The final visual instruction-tuning stage is experimented on **4 A100 GPU nodes, taking 18 hours**. **During inference time, the VLA model in NaVILA can be served using a single RTX 4090 GPU with roughly 1 FPS.**" — [Appendix C.4](https://arxiv.org/html/2412.04453v2)

**Quantization (their own edge-oriented experiment):** AWQ **W4A16** on the 8B FP16 model → latency **594.58 ms → 367.80 ms**, VRAM **18.5 GB → 8.6 GB**, SR 49.7 → 48.2, SPL 45.5 → 43.6 (RTX 4090, 1737 ctx tokens, 10 generated tokens). They state this "make[s] NaVILA deployable directly on the robot" but list it as future work — it is **not** 8B-on-4090-class hardware (a laptop-class GPU), it is a $1,600 desktop GPU.

**Performance (R2R-CE / RxR-CE Val-Unseen, single RGB view, no depth/odom/panorama, no simulator waypoint predictor):**
| | R2R NE↓ | OS↑ | **SR↑** | SPL↑ | RxR NE↓ | SR↑ | SPL↑ |
|---|---|---|---|---|---|---|---|
| NaVid (prior single-view SOTA) | 5.47 | 49.0 | 37.0 | 35.0 | — | — | — |
| Best *with* pano+depth+odo+waypoint predictor (HAMT+ScaleVLN) | 4.80 | — | 55.0 | 51.0 | — | — | — |
| **NaVILA** | **5.22** | **62.5** | **54.0** | **49.0** | **6.77** | **44.0** | **58.8** |

The abstract's "+17% success rate" is **54.0 vs NaVid's 37.0** — against single-view methods only. It still loses SPL to waypoint-prior methods.
**ScanQA spatial 3D-QA:** beats NaviLLM by ~20 CIDEr; with 64 frames beats 3D-LMMs that need depth+poses.
**New benchmark VLN-CE-Isaac** (Isaac Sim, joint-level control): vision policy beats blind policy by 14% SR.
**Real world:** **88% success over 25 instructions** (75% on complex multi-scene instructions); deployed on Go2, H1, Booster T1 by swapping only the locomotion policy.
**Ablation:** YouTube human-touring-video data (2K videos → 20K trajectories, MASt3R pose estimation + VLM caption + LLM rephrase) adds ~5% OS/SR/SPL — "the first work to show that direct training on human videos improves navigation in continuous environments."

**Stated failure mode (relevant to you):** the real-world failure case is "the robot's inability to perform effective error correction when deviations occur." They propose "incorporating explicit reasoning data during training" and "larger-scale training on more realistic simulations… with diverse error recovery cases."

### 2.2 The "VLM as mid-level planner over a map" family

| Work | Structure | Key result | Link |
|---|---|---|---|
| **L3MVN** (IROS'23) | LLM (offline RoBERTa fine-tune or zero-shot GPT-3) scores frontiers on a semantic map; FMM computes cost; A*/Dijkstra executes | **Gibson 76.1/77.0% SR**; **HM3D 50.4/54.2% SR** — vs SemExp 65.2/37.9 and PONI 73.6 (Gibson). *No training.* Ablation: removing the LLM drops SR and SPL. | [arxiv](https://arxiv.org/html/2304.05501v2) · [code](https://github.com/ybgdgh/L3MVN) |
| **VLMaps** (ICRA'23) | LSeg open-vocab 2D segmentation → 3D point cloud; LLM writes code to query "navigable points to <landmark>"; UniTS / PointNet localization | Long-horizon language navigation with open-vocab landmark indexing; sim-to-real via semantic domain invariance | [vlmaps.github.io](https://vlmaps.github.io/) · [code](https://github.com/vlmaps/vlmaps) |
| **MapGPT** (ACL'24) | **Training-free.** Top-down map rendered to an image; GPT-4V/4o takes ("instruction, top-down semantic map, robot pose") and outputs the next waypoint; adaptive path planning | Zero-shot VLN; explicitly removes RL training | [arxiv](https://arxiv.org/html/2401.07314v3) |
| **SayNav** (ICAPS'24) | LLM dynamically breaks a long instruction into a *rollout plan* of embodied sub-goals, re-planned as the world changes | Generalizes to new large-scale environments without training | [arxiv](https://arxiv.org/html/2309.04077v4) · [code](https://github.com/arajv/SayNav) |
| **InstructNav** (2024) | Dynamic Chain-of-Navigation prompting + open-vocab GLEE detector + Intuition saliency value map + A* | **HM3D 58.0% SR / 20.9% SPL** — the "LLM earns its keep" reference point | [arxiv 2402.11758](https://arxiv.org/html/2402.11758) |
| **VLFM** (ICRA'24) | Vision-Language Frontier Maps: geometry-only frontiers + a VLM (BLIP-2) computing a value map | HM3D **52.5% SR / 30.4% SPL** | cited in [2507.20021](https://arxiv.org/html/2507.20021v3) |
| **MooG** (2024) | Training-free modular VLM pipeline: LLM decomposes instruction, VLM grounds objects, frozen zero-shot detectors, affordance costmap | Zero-shot ObjNav/PointNav | project/code by Neurath et al. |
| **NaVid** (RSS'24) | Video-based VLM; **no maps, no odometry, no depth** — a sliding window of history frames, diffusion action head | R2R-CE Val-Unseen is the NaVILA baseline: **SR 37.0 / SPL 35.0** | [arxiv](https://arxiv.org/html/2402.15852v7) |
| **VLMnav** (CoRL'24) | The *anti-decoupling* position: "we do not rely on a separation between perception, planning, and control; instead, we use a VLM to directly select actions in one step." Surprising navigation results. | [PDF](https://jirl-upenn.github.io/VLMnav/static/VLMnav.pdf) |
| **OpenVLA-OFT** | See §1 — the decoupled-fine-tuning version: parallel decode + chunk + L1. | 76.5% → **97.1%** LIBERO avg; 26× throughput | [openvla-oft](https://openvla-oft.github.io/) |
| **NaVILA** | See §2.1. | | |
| **Hydra-Nav** (ICML'26) | Adaptive dual-process reasoning: a fast System-1 + a deliberative System-2 for ObjectNav; attacks "weak temporal-spatial reasoning" | HM3D-v2 **72.9% SR (SFT variant)** | [arxiv](https://arxiv.org/html/2602.09972v1) · [project](https://zixuan-wang99.github.io/Hydra-Nav/) |

### 2.3 "Affordance-based navigation with a frozen VLM" — the closest to what you want

- **FOM-Nav** (Frontier-Object Maps): segment objects → store geometric + visual + textual features in a 3D object map → a VLM scores (frontier, object, history-path) triples → A*/FMM plans. [HAL PDF](https://hal.science/hal-05392088v1/file/FOMNav.pdf)
- **GAMap** (Geometric-part + Affordance Maps), zero-shot OGN. [arxiv 2410.23978](https://arxiv.org/pdf/2410.23978)
- **WMNav** — "Integrating Vision-Language Models into World Models for Object Goal Navigation": a world model of VLMs + novel modules. HM3D-v2 **72.2% SR / 33.3% SPL**, HM3D-v1 58.1/31.2. [arxiv 2503.02247](https://arxiv.org/html/2503.02247v5) · [alphaXiv](https://www.alphaxiv.org/abs/2503.02247)
- **R2F** (2026) — the purest "frozen encoder, no LLM" result. Language-aligned features from **NA-RADIO** are accumulated **along out-of-range depth rays** and attached to frontier regions as **direction-conditioned embeddings**; frontiers are scored by cosine similarity with a **SigLIP** query embedding. Purely geometric occupancy map; frontier extraction every N=5 steps, semantic accumulation every step. **Runs at 25 Hz on a laptop with an RTX 4070 8GB (Intel Core Ultra 185H, 32 GB).** [arxiv 2603.08475](https://arxiv.org/html/2603.08475v1) · [project](https://lab-rococo-sapienza.github.io/r2f/)

  ObjectNav (HM3D, 60 eps) / VLN (50 eps):

  | Method | SR% | SPL% | **t (s)** |
  |---|---|---|---|
  | VLN-Game | 76.7 | 28.0 | 122.0 |
  | 3D-MEM | 26.7 | 15.9 | 61.2 |
  | VLFM | 40.0 | 7.71 | 84.2 |
  | OpenFrontier | 23.3 | 7.12 | 245.0 |
  | **R2F (LLM-free, training-free)** | **78.3** | **29.6** | **32.7** |
  | R2F-VLN | 28.0 | 13.94 | 40.3 |

  **This is the single most important row in this report.** Best accuracy, 6× faster than the best VLM-based competitor, zero API cost, runs on a laptop. Its weakness (self-reported): compositional/relational reasoning — "the main errors arise from false positives, where objects with landmarks similar to the query are detected but appear in a different configuration."

### 2.4 The "geometry beats the LLM" result

[arXiv 2507.20021](https://arxiv.org/html/2507.20021v3) "When Engineering Outruns Intelligence" (2025) — [code](https://github.com/matinaghaei/instructnav-scrutinized). HM3D-v1 val (2,000 eps) and MP3D val (2,195 eps), RGB-D 640×480, 500-step cap.

| Method | GT sem.? | HM3D SR | HM3D SPL | HM3D ASPL | MP3D SR | MP3D SPL | MP3D ASPL |
|---|---|---|---|---|---|---|---|
| ZSON | No | 25.5 | 12.6 | – | 15.3 | 4.8 | – |
| ESC | No | 39.2 | 22.3 | – | 28.7 | 14.2 | – |
| L3MVN | No | 50.4 | 23.1 | – | – | – | – |
| VoroNav | No | 42.0 | 26.0 | – | – | – | – |
| VLMNav | No | 50.4 | 21.0 | – | – | – | – |
| VLFM | No | 52.5 | **30.4** | – | 36.4 | 17.5 | – |
| **InstructNav** | No | **58.0** | 20.9 | – | – | – | – |
| InstructNav-GT | **Yes** | 60.6 | 33.8 | 21.3 | 50.6 | 24.0 | 17.0 |
| **FPE (ours, geometry-only, no API)** | Yes | **61.4** | **36.0** | **23.5** | 48.0 | **24.5** | **17.5** |
| **SHF (ours, LLM frontier votes, k=5)** | Yes | 61.2 | **36.2** | 23.0 | 47.1 | 23.1 | 16.4 |

| | HM3D $ | HM3D runtime (h) | MP3D $ | MP3D runtime (h) |
|---|---|---|---|---|
| InstructNav-GT | 242.5 | 237.0 | 207.8 | 204.4 |
| **FPE** | **0** | **95.8** | **0** | **87.4** |
| SHF | 366.3 | 190.5 | 424.2 | 180.0 |

Their prescription, worth adopting verbatim as a design rule: *"language should inject local, metric-aware preferences that a geometry-aware planner can arbitrate—rather than attempting free-form plan generation."*

---

## 3. Small visual navigation models

| Model | Params | Rate | What it is | Memory | Link |
|---|---|---|---|---|---|
| **ViNT** (CoRL'23) | **31M** | **4 Hz** on the robot, PD-tracked waypoints | Transformer over EfficientNet-B0 context + goal-fusion encoder; predicts (temporal distance, H-step action chunk). Trained on **100+ h of real trajectories across 8 robot platforms**. | **Semi-parametric topological graph** `M` — nodes are subgoal images, edges added when traversed. *Best performance comes from pairing it with a 300M image-diffusion subgoal proposer.* | [paper](https://proceedings.mlr.press/v229/shah23a.html) · [general-navigation-models.github.io/vint](https://general-navigation-models.github.io/vint) · [code](https://github.com/robodhruv/visualnav-transformer) |
| **NoMaD / ViNTv2** (ICRA'24 Best Paper) | **30M** | *"can run directly on the less powerful onboard computers (e.g., NVIDIA Jetson Orin)"* | Adds a **goal-mask token** (Bernoulli p=0.5) so one diffusion policy does both goal-reaching *and* undirected frontier exploration. Diffusion over **actions**, not images. | Topological graph (ViKiNG-style episodic memory) | [arxiv 2310.07896](https://arxiv.org/html/2310.07896) · [nomad project](https://general-navigation-models.github.io/nomad/) |

  **NoMaD's key claim: >25% better than the ViNT system (Subgoal Diffusion) in undirected exploration, with 15× fewer parameters, "running entirely on-the-edge."** It also matches ViNT on goal-conditioned nav with the same capacity. Explicit limitation: *"our exploration method uses a standard frontier-based exploration strategy for high-level planning… Intelligently selecting which regions to explore, such as strategies based on semantics and prior knowledge, could further improve performance."* ← **This is exactly the slot you should fill with a small VLM.**

- **ViNTv2** — the released successor in `robodhruv/visualnav-transformer` (GNM, ViNT, NoMaD checkpoints). ViNT is trained on 100+ h; ViNTv2 scales the data.
- **GNM** — 8.7M trainable params, the minimal ablation baseline.
- **ViPlanner** (CVPR'24, ETH) — local planner, not a foundation model. ResNet-18 perception nets from scratch + a light CNN/MLP planning net with two heads (sparse keypoint path + **collision probability**); only trajectories with collision prob < δμ=0.5 are executed. The **semantic costmap is training-only**; at inference the network emits paths straight from the image stream. **Deployed on ANYmal on a Jetson AGX Orin; by asynchronously reusing Mask2Former semantic frames at ~3 Hz it achieves 10 Hz planner rate.** Its differentiator is that semantics let it "recognize geometrically invisible obstacles" where the geometric baseline iPlanner halts. [arxiv](https://arxiv.org/html/2310.00982v3) · [code](https://github.com/leggedrobotics/viplanner) · [project](https://leggedrobotics.github.io/viplanner.github.io/)
- **M3TDP** (Multi-modal, Multi-target, Task-driven Policy, NeurIPS'20) — third-person+egocentric RGB fusion, single model for 24 navigation tasks, point+image goals. [arxiv 2009.02796](https://arxiv.org/html/2009.02796v1)
- **Active Neural SLAM** (2018) and **Neural SLAM** (2019) — the classic "learned mapping module + learned global policy + learned local policy" decomposition. ANSL: *"Memory: Neural SLAM; Policy: Active Neural SLAM."* [arxiv 2005.08229](https://arxiv.org/html/2005.08229v1)
- **PONI** (CVPR'22 Oral, UT Austin) — see §5.3.
- **MOPA** (WACV'24) — modular ObjectNav with *point-goal* agents: shows the modular decomposition (detector + mapping + local policy) is a deliberate design, not a fallback. [thecvf PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Raychaudhuri_MOPA_Modular_Object_Navigation_With_PointGoal_Agents_WACV_2024_paper.pdf)
- **PANO / panoramic goal networks**, **ViNG** (topological graph + image-goal subgoals), **Diffusion Policy** (157M) as the universal from-scratch local baseline.
- **VANP** (self-supervised vision-action pretraining) — learns *where to look* from action supervision. [CMU PDF](https://people.cs.gmu.edu/~xxiao2/papers/vanp.pdf)

---

## 4. World models as the alternative to VLA

### 4.1 V-JEPA 2 — the strongest published case, and the numbers you need

[arXiv 2506.09985](https://arxiv.org/html/2506.09985v1) (Meta, Jun 2025) · [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)

- **Encoder: ViT-g, 1B params** (scaled 300M ViT-L → 1B; +1.5 avg points, +1.7 with cooldown). Pretrained on **>1M hours of internet video** + 1M images (VM22M mix: SSv2, K400/600/700, HowTo100M, **Curated YT1B**, ImageNet). Progressive resolution: 252K iters @16f/256² then 12K cooldown @64f/384² — **8.4× GPU-time reduction** vs. training at full res throughout (which would have needed **~60 GPU-years** on A100s).
- Understanding: **77.3% top-1 SSv2**, **39.7 R@5 Epic-Kitchens-100**; with a Qwen2-7B LLM, 84.0 PerceptionTest / 76.9 TempCompass at the 8B scale. **300M V-JEPA 2 beats the 8B PlausiVL by +12.1 R@5 (44% relative).**
- **V-JEPA 2-AC**: frozen encoder + frame-causal action-conditioned latent predictor trained on **<62 h of unlabeled DROID** video (2 context frames, horizon 4, 256×256). Planning = minimize a goal-conditioned energy function by **CEM (800 samples, 10 refinement steps, horizon 1)**, executing the first action (receding horizon).
- **Zero-shot on two Franka arms in two different labs, no data from those labs, no task-specific training, no reward.** Best success across grasp / reach-with-object / pick-and-place; Octo gets 0% on most object-interaction tasks.
- **The compute comparison that matters:**
  > "With 80 samples, 10 refinement steps, and a planning horizon of 1, it takes **4 minutes** to compute a single action in each planning step with **Cosmos**… a full pick & place trajectory requires over one hour of robot execution. By contrast, with **10× more samples in each refinement step, the V-JEPA 2-AC world model requires only 16 seconds per action** and leads to higher performance across all considered robot skills." — all on a **single RTX 4090**.
  Cosmos still gets 80% on *reach*; it collapses on object interaction. **JEPA latent prediction beats pixel diffusion for planning by ~15× per action.**
- **Their own critique of VLAs** (a gift for your write-up): *"Although these approaches show promising generalization results, it is unclear whether they will be able to learn to predict behaviors that were not demonstrated in the training data since they **lack an explicit predictive model of the world** and do leverage inference-time computation for planning. They require high-quality large scale teleoperation data, and can only utilize successful trajectories. In contrast, we focus on leveraging **any** interaction data whether it comes from a successful or failed interaction."*
- **Stated limitation for you:** predictions only reach ~16 s; *"extending this to longer-horizon tasks… without requiring sub-goals will require further innovations in modeling."* Planning cost grows exponentially with horizon.
- **Weight availability:** V-JEPA 2 is **open-weight (non-commercial research license) via `facebookresearch/vjepa2`** — this is the main practical caveat.

### 4.2 Genie / Cosmos / GR-2 / DreamerV3

| Model | What it is | Numbers | Verdict for you |
|---|---|---|---|
| **Genie 3** (DeepMind, Aug 2025) | Text→interactive photoreal world | **720p @ 24 fps**, consistency for **a few minutes**. **Not open weights.** Self-declared limits: *"Limited action space… the range of actions agents can perform directly is currently constrained"* and *"Limited interaction duration… a few minutes of continuous interaction, rather than extended hours."* | Simulator/generator, **not** a nav memory. [blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) |
| **Cosmos-1.0** (NVIDIA) | World Foundation Model family | Diffusion **7B/14B** Text2World + Video2World; **Autoregressive 4B/12B** (+13B Video2World); 6B/7B/14B Predict variants. AdaLN-LoRA cut 11B→7B (−36%) at parity. 14B needs 280 GB states + 310 GB activations on H100s. **Action-conditioned variants exist** (`-Sample-ActionCond`). **Camera-conditioned variant = 3D world simulator with Plücker camera-pose control.** | Useful for **synthetic data**, not runtime. [arxiv 2501.03575](https://arxiv.org/html/2501.03575v1) |
| **Cosmos Policy** (Jan 2026) | Fine-tunes video models for visuomotor control; criticizes the 3-stage post-training complexity | — | [arxiv 2601.16163](https://arxiv.org/html/2601.16163v1) |
| **GR00T-Dreams** | Cosmos generates synthetic robot trajectories from one image + prompt; expands 88 h teleop → **827 h** | NVIDIA reports this scaling as essential to GR00T N1 | Data engine. [docs](https://docs.nvidia.com/cosmos/latest/predict2/post-training_video2world/groot-dreams.html) |
| **DreamerV3** | Latent imagination RL | Mastered 150+ games from pixels; needs a GPU to *train* | Not a nav memory. [arxiv](https://arxiv.org/html/2301.04104v3) |

### 4.3 Is any world model the *long-term memory* of a nav agent? (the direct question)

**Mostly no — and this is a real gap you would be filling.** The memory research in 2026 has converged on *explicit* or *compressed-image* memory rather than generative world models:

- **AstraNav-Memory** (CVPR 2026, Amap/Alibaba + CUHK) — **the closest existing answer.** [arxiv 2512.21627](https://arxiv.org/html/2512.21627v1) · [code](https://github.com/amap-cvlab/AstraNav-Memory) · [project](https://astra-amap.github.io/AstraNav-Memory.github.io/)
  - **Image-centric implicit memory, no explicit map, no object queries.** A ViT-native visual tokenizer: **frozen DINOv3-ViT-Base** → two **PixelUnshuffle** stages (each + a 3×3 stride-1 conv, BatchNorm, SiLU) → injected into **block 1 of Qwen2.5-VL-3B's own ViT** → 2×2 patch-merger. A 720×640 frame goes **598 native Qwen visual tokens → ~30 tokens (≈16×, ≈20× compression)**. Camera poses are **serialized as text tokens** immediately before the visual tokens.
  - Camera pose `P_t` + image `I_t` pairs + instruction; model emits natural language naming a frontier or target, with 2D coords in `<coordinate>x, y</coordinate>` tags.
  - **Up to 300 historical frames in one context.** Qwen2.5-VL-3B language model and projector unchanged. Trained on 1.5M samples (OVON-500K + GOAT-1M-50/100/200/500L) on 32 H20s, lr 1e-5, ≤2 epochs.
  - **Results:** GOAT-Bench (lifelong, persistent environment+state across tasks) Val-Unseen **62.7% SR / 56.9% SPL**, beating prior SOTA MTU3D (47.2/27.7) by **+15.5 SR / +29.2 SPL**, and 2.4× SR / 3.2× SPL over Modular GOAT. HM3D-OVON (open-vocab) **62.5% SR / 34.8% SPL** vs MTU3D +21.7/+22.8.
  - **Efficiency measured:** 50 compressed images → training time/iter **26.4 s → 6.5 s**, GPU memory **90.6 GB → 46.8 GB**. Best compression is 16×; 64× collapses (42.5–48.1% SR). DINOv3 ViT-B > ViT-S by 4.2%, ViT-L only +1.2% for 3× params.
  - **Failure mode to know:** DINOv3 features blur texture boundaries — worse than raw Qwen ViT on *carpet*, *island*, *microwave*. They explicitly propose adding SAM/boundary cues.
  - **Important caution:** attention cost is **O((tokens_per_frame × history)²)** — 300 frames × 30 tokens = 9,000 tokens, quadratic. On a 5–20 TOPS part this is *not* real-time. You would need to shrink hard (e.g. 50 frames).
- **GLAM** (Sep 2026) — *"Training a latent world model over global spatiotemporal memory for active exploration and navigation."* The most on-the-nose title. [arxiv 2609.14561](https://arxiv.org/abs/2609.14561)
- **WorldMAP** (2026) — VLMs as direct planners / trajectory predictors vs. generative world models for look-ahead; finds trajectory prediction from a world model is unreliable. [arxiv 2604.07957](https://arxiv.org/html/2604.07957v1)
- **WMNav** (see §2.3) — VLM modules inside a world model for ObjectNav. 72.2 SR on HM3D-v2.
- **VTM-Nav** (2026) — hierarchical visual-topological memory for **cross-episode** ObjectNav: *"Training-free ObjectNav agents… typically discard acquired scene knowledge after each request."* [arxiv 2607.14514](https://arxiv.org/html/2607.14514v2)
- **3D-MEM** (Memory Snapshots / Frontier Snapshots + VLM) — [cited in R2F](https://arxiv.org/html/2603.08475v1)
- **A Survey of Spatial Memory Representations for Efficient Robot Navigation** (CVPRW 2026, WiCV) — explicitly targets your constraint: *"As vision-based robots navigate larger environments, their spatial memory grows without bound… particularly on embedded platforms (**8–16 GB shared memory, <30 W**) where adding hardware is not an option."* [PDF](https://openaccess.thecvf.com/content/CVPR2026W/WiCV/papers/Pangaliman_A_Survey_of_Spatial_Memory_Representations_for_Efficient_Robot_Navigation_CVPRW_2026_paper.pdf)

### 4.4 What is the cheapest thing that provides "strong environment understanding" today?

Ranked by edge cost, for a **navigation** agent:

1. **Metric depth + geometry** — Depth Anything V2 / UniDepth / ZoeDepth at ~384px. A 2.5D height map from this plus a SLAM pose *is* terrain understanding, and it is what NaVILA's own low-level policy uses (via LiDAR).
2. **Open-vocab detection with a frozen CLIP text head** — NanoOWL (OWL-ViT) / YOLO-World. On Orin Nano Super: **NanoOWL(OWLViT-base-patch32) FP16 = 9.81 ms**, 2.65× faster than YOLO-World-S (26.07 ms). ([Frontiers 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1693988/full))
3. **DINOv2 / DINOv3 features as a general map backbone** — DINOv2-base-patch14 = **126 FPS on Orin Nano Super**. ([Jetson AI Lab](https://www.jetson-ai-lab.com/archive/benchmarks.html))
4. **CLIP / SigLIP text-image scoring over a 3D map** — clip-vit-base-patch32 = **314 FPS on Orin Nano Super**; R2F does exactly this over ray-frontier embeddings.
5. **A 1–2B VLM called at 0.2–1 Hz** — Qwen3-VL-2B / InternVL3.5-2B / SmolVLM2-500M, only for frontier scoring, decomposition, and scene graph extraction. Not in the hot loop.

**This ladder is the architecture.** Do not put a VLM in the inner loop.

---

## 5. Hierarchical planning stacks that work today without (or with minimal) learning

### 5.1 The current SOTA is modular and training-free

**Habitat ObjectNav, HM3D v1** ([leaderboard](https://www.sota2.com/research/sota/object-navigation-on-hm3d-v1), mid-2026):

| Rank | Method | Training | SR | SPL |
|---|---|---|---|---|
| 1 | VLingNav | ✗ | **79.1** | **42.9** |
| 2 | Uni-NaVid | ✓ | 73.7 | 37.1 |
| 3 | IntentNav | ✓ | 70.5 | 34.6 |
| 4–5 | SysNav | ✓ / ✗ | 63.7 | 30.5 |
| 6 | ConsistNav | unsupervised, zero-shot | 63.2 | 34.8 |
| 7 | STRIVE | ✗ | 62.9 | 34.2 |
| 8 | **Hierarchical 3D Scene Graph** | **✓ training-free** | 61.6 | 33.9 |
| 9 | BeliefMapNav | ✓ / training-free | 61.4 | 30.4–30.6 |
| 10 | ApexNav | training-free, zero-shot | 59.6 | 33.0 |
| … | FBN-Nav | training-free | 58.8 | 31.2 |
| … | **WMNav** | ✗ | 58.1 | 31.2 |
| … | ProcTHOR | ✗ | 54.4 | 31.8 |
| … | SG-Nav | training-free | 54.0 | 24.9 |
| … | VLFM | training-free | 52.5 | 30.4 |
| … | **L3MVN** | training-free | 50.4 | 23.1 |
| … | VoroNav | training-free | 42.0 | 26.0 |
| … | **Navid** | ✓ | 32.5 | 21.6 |
| … | **ZSON** | ✓ | 25.5 | 12.6 |
| 32 | ProcTHOR-ZS | ✗ | 13.2 | 7.7 |

**Habitat ObjectNav, HM3D v2** ([leaderboard](https://www.sota2.com/research/sota/object-navigation-on-hm3d-v2)):

| Method | Training | SR | SPL |
|---|---|---|---|
| **ConsistNav** | unsupervised, zero-shot | **84.2** | 41.2 |
| VLingNav | ✗ | 83.0 | 40.5 |
| IntentNav | ✓ | 82.2 | 38.5 |
| **Hierarchical 3D Scene Graph** | **training-free** | 80.1 | 39.5 |
| **Hydra-Nav-SFT** | ✓ | 72.9 | 27.7 |
| **WMNav** | ✗ | 72.2 | 33.3 |
| VLFM | training-free | 63.6 | 32.5 |
| **InstructNav** | training-free | 58.0 | 20.9 |
| L3MVN | training-free | 36.3 | 15.7 |
| **DD-PPO** (canonical end-to-end RL) | ✓ | **27.9** | **14.2** |

**Read this carefully:** the *only* end-to-end learned policy in the comparison (DD-PPO) is last. The 2026 winner on v2 is **unsupervised + zero-shot**. The 2026 winner on v1 uses a **hierarchical 3D scene graph, training-free**. This is direct evidence for your hypothesis.

### 5.2 The canonical components

- **Habitat / HM3D / Gibson / MP3D / AI2-THOR / ProcTHOR / GOAT-Bench / OVON** as simulators and benchmarks. [aihabitat.org](https://aihabitat.org/challenge/2023) · [habitat-challenge code](https://github.com/facebookresearch/habitat-challenge)
- **SemExp** (CVPR'20, semantic exploration; the origin of the frontier idea as a learned thing)
- **Active Neural SLAM** (global policy + local policy + differentiable mapping). [arxiv 2005.08229](https://arxiv.org/html/2005.08229v1)
- **Hydras** (multi-resolution, attention-based exploration) — [arxiv 2306.12148](https://arxiv.org/html/2306.12148v1)
- **OVMM** (Explore-Then-Execute): modular ObjectNav with an explicit exploration phase then an execution phase. [arxiv 2011.08111](https://arxiv.org/abs/2011.08111)
- **SMACL** (Semantic-aware Multi-goal Curiosity Learning) — the RL counterpart. [arxiv 2012.08815](https://arxiv.org/html/2012.08815v1)
- **Nav2** (ROS 2 navigation server: costmaps, planners, controllers, behavior trees) — the production substrate. [docs.nav2.org](https://docs.nav2.org/)
- **OpenVSLAM / RTAB-Map / ORB-SLAM3** for SLAM.
- **PointNav** numbers: classical global-planar frontier + A* pipelines are essentially solved; the *hard part is ObjectNav / open-vocabulary / lifelong*, and that is a perception+memory problem, not a control problem.

### 5.3 PONI (CVPR'22 Oral) — the compute-efficiency proof

[arxiv 2201.10029](https://arxiv.org/html/2201.10029) · [project](https://vision.cs.utexas.edu/projects/poni/) · [code](https://github.com/srama2512/PONI)

> "PONI outperforms the prior SoTA on Gibson (SemExp) with **7× lower training cost**, and the prior SoTA on MP3D (THDA) with **1@600× lower training cost**."

Mechanism: a small **UNet** trained purely on **2D semantic-map** pairs (no RL, no interaction) predicts two explicit potential functions at map frontiers — an **area potential** (exploration) and an **object potential** (goal-directed). Take argmax frontier, A* on `1 − V_aff`. Training data: 400,000 map tuples per dataset, 3 epochs.
**Its honest limitation, which is your open problem:** *"Our performance is sensitive to the image segmentation quality. The success rate goes down by 14.9% in Gibson and 45.4% on MP3D relative to the performance with ground-truth segmentation… Unlike end-to-end RL methods which may learn to be robust to the sensory noise, we do not have an inbuilt mechanism to handle failures in segmentation."*

### 5.4 L3MVN, revisited — the "LLM as local prior over a map" numbers

From [Table I](https://arxiv.org/html/2304.05501v2):

| Method | Gibson SR | Gibson SPL | Gibson DTG↓ | HM3D SR | HM3D SPL |
|---|---|---|---|---|---|
| Random walk | 0.030 | 0.030 | 2.580 | 0.000 | 0.000 |
| Frontier-based | 0.417 | 0.214 | 2.634 | 0.237 | 0.123 |
| Random sample on map | 0.544 | 0.288 | 1.918 | 0.300 | 0.143 |
| SemExp | 0.652 | 0.336 | 1.520 | 0.379 | 0.188 |
| **PONI** | 0.736 | **0.410** | 1.250 | – | – |
| **L3MVN (zero-shot)** | 0.761 | 0.377 | 1.101 | 0.504 | 0.231 |
| **L3MVN (feed-forward)** | **0.769** | 0.388 | **1.008** | **0.542** | **0.255** |

Note the ablation: removing the cost-utility exploration module (→ nearest-frontier) drops performance *more* than removing the LLM. **The geometry carries more of the load than the language.**

---

## 6. Compute budget reality check

### 6.1 Hardware

| Module | Sparse INT8 TOPS | Dense INT8 TOPS | RAM | Bandwidth | Power | Link |
|---|---|---|---|---|---|---|
| **Jetson AGX Thor** | — | — | 128 GB | 273 GB/s | 15–130 W | [benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks) |
| **Jetson AGX Orin 64GB** | **275** | 85 | 64 GB LPDDR5 | 204.8 GB/s | 15–60 W | [specs](https://developer.nvidia.com/embedded/jetson-orin) |
| Jetson AGX Orin 32GB | 248 | — | 32 GB | 204.8 GB/s | 15–60 W | ″ |
| **Jetson Orin NX 16GB (Super)** | 200 | 100 | 16 GB | 102.4 GB/s | 10–40 W | ″ |
| Jetson Orin NX 8GB | 157 | 78 | 8 GB | 102.4 GB/s | 10–40 W | ″ |
| **Jetson Orin Nano 8GB (Super)** | 67 | 33 | 8 GB | 102 GB/s | 7–25 W | ″ |
| Jetson Orin Nano 4GB (Super) | 34 | 17 | 4 GB | 51 GB/s | 7–15 W | ″ |
| **RK3588** (Rock 5B/5C, Orange Pi 5) | **~6** | — | 8 GB | — | 5–12 W | [Qengineering NPU ports](https://github.com/Qengineering/Qwen3-VL-2B-NPU) |

**Your "5–20 TOPS" is roughly Orin-Nano-4GB-class or RK3588-class.** That is *not* enough for any generative VLM in a control loop. The Frontier survey is blunt: *"Securing a response rate of over 10 FPS, the minimum requirement for seamless human interaction, presents a significant optimization challenge when deploying VLMs in edge environments."* ([Frontiers 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1693988/full))

The 2026 CVPRW survey on spatial memory literally names your class of target: *"8–16 GB shared memory, <30 W."*

### 6.2 Measured throughput

**Jetson AI Lab, Orin Nano Super** (MLC, INT4 for LLM; live-streaming end-to-end for VLM; TensorRT for ViT) — [source](https://www.jetson-ai-lab.com/archive/benchmarks.html):

| Model | tok/s (Nano Super) | tok/s (Nano orig) |
|---|---|---|
| LLaVA-1.6 **7B** | **0.57** | 0.412 |
| **VILA-1.5 8B** | **0.83** | 0.574 |
| **VILA-1.5 3B** | **1.06** | 0.7 |
| Qwen2-VL 2B | 4.4 | 2.8 |
| InternVL2.5 4B | 5.1 | 2.5 |
| SmolVLM 2B | 12.9 | 8.1 |
| PaliGemma2 3B | 21.6 | 13.7 |
| *CLIP ViT-B/32* | *314* | *196* |
| *CLIP ViT-B/16* | *161* | *95* |
| *DINOv2-base-patch14* | *126* | *75* |
| *ViT-B/16* | *158* | *98* |
| *Grounding DINO* | *6.23* | *4.11* |
| *SAM2-base* | *6.34* | *4.42* |

**VILA-1.5-8B at 0.83 tok/s.** NaVILA-8B is a VILA-8B fine-tune. **NaVILA is un-runnable on a 67-TOPS Nano Super, let alone a 6-TOPS RK3588.** This is the single most decision-relevant number in the report.

**Jetson AGX Thor, JetPack 7 / CUDA 13 / TensorRT 10.13, vLLM, ISL/OSL 2048/128** — [source](https://developer.nvidia.com/embedded/jetson-benchmarks):

| Model | tok/s (concurrency 1) | tok/s (concurrency 8) |
|---|---|---|
| **Qwen2.5-VL 3B** | **71.7** | 356.9 |
| Qwen2.5-VL 7B | 45.0 | 252.0 |
| Llama 3.2 11B Vision | 26.3 | 69.6 |
| Qwen3-30B-A3B | 61.0 | 226.4 |
| DeepSeek R1 7B | 41.3 | 304.8 |

Note this is **prefill-heavy short-output** throughput. Real navigation prompts (multi-image + a long instruction + ~20 generated tokens) behave worse. Thor is a ~2000-TOPS-class part; it's roughly 10× beyond your budget.

**Jetson AGX Orin 64GB, JetPack 6 / TensorRT 8.6.2** — open-vocab detect-then-segment, RefCOCO+ mIoU, [Frontiers 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1693988/full):

| Component | Latency | Note |
|---|---|---|
| YOLO-World-S (PyTorch) | 26.07 ms | 38.4 FPS |
| YOLO-World-X (PyTorch) | 45.59 ms | |
| **NanoOWL OWLViT-base-patch32 FP16** | **9.81 ms** | ~102 FPS |
| NanoOWL OWLv2-large-patch14 | 195.69 ms | 5 FPS |
| EfficientViT-SAM-L0 | 7.88–17.58 ms | |
| EfficientViT-SAM-XL1 | 27.78–77.8 ms | |
| **NanoOWL + EfficientViT-SAM-L0 (best)** | — | **47.51 FPS @ 84.64 mIoU** |
| YOLO-World-S + EfficientViT-SAM-L0 | — | 26.68 FPS (−43%) |

**Critical deployment warning from that paper:** EfficientViT-SAM L0/XL0/XL1 **collapse to mIoU ≈ 0 under FP16 encoder** (catastrophic failure, not graceful degradation), while **NanoSAM (distilled ResNet-18) never failed at any precision**. Also: TensorRT-ing YOLO-World destroys its zero-shot ability, whereas NanoOWL is TensorRT-native. **Distillation buys you quantization robustness** — that transfers directly to your design.

**RK3588 (6 TOPS NPU), w8a8 LLM + fp16 vision, Qengineering ports** — [source](https://github.com/Qengineering/Qwen3-VL-2B-NPU):

| Model | RAM (GB) | LLM cold (s) | LLM warm (s) | VLM cold (s) | VLM warm (s) | Res | **tok/s** |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 1.3 | 10.6 | 1.9 | 2.7 | 0.2 | 448² | **21.6** |
| Qwen3.5-2B | 2.9 | 23.9 | 3.2 | 8.5 | 0.8 | 448² | 11.0 |
| Qwen3-2B | 3.1 | 21.9 | 2.6 | 10.0 | 0.9 | 448² | 11.5 |
| **InternVL3.5-1B** | 1.9 | 8.3 | 8.0 | 1.5 | 0.8 | 448² | **24.0** |
| InternVL3.5-2B | 3.0 | 22 | 8.0 | 2.7 | 0.8 | 448² | 11.2 |
| Qwen3-4B | 8.7 | 49.6 | 5.6 | 10.6 | 1.1 | 448² | 5.7 |
| InternVL3.5-4B | 5.4 | 50 | 8.0 | 5.9 | 0.8 | 448² | 5.0 |
| Qwen2.5-3B | 4.8 | 48.3 | 4.0 | 17.9 | 1.8 | 392² | 7.0 |
| Qwen2-7B | 8.7 | 86.6 | 34.5 | 37.1 | 20.7 | 392² | 3.7 |
| InternVL3.5-8B | 8.8 | 92 | 8.0 | 50.5 | 5.8 | 448² | 3.5 |
| Qwen3.5-9B | 9.2 | 97.1 | 97.1 | 11.5 | 11.5 | 448² | 3.2 |

**At ~5–20 TOPS, a 1–2B VLM gives you ~11–24 tok/s.** That is roughly one decision every 3–6 seconds, with a ~1.9–3.0 GB model + 2.7–10 s of image-encoder warm-up. **Conclusion: a 1B VLM is viable as a 0.2–1 Hz deliberator. An 8B VLM is not viable at all.**

### 6.3 Small VLM options and their robotics relevance

| Model | Params | Notes | Link |
|---|---|---|---|
| **Qwen3-VL** | **2B / 4B / 8B / 32B** dense + 30B-A3B / 235B-A22B MoE | Apache 2.0. Native 256K interleaved context. **Explicitly built for embodied AI:** spatial-relationship reasoning, object affordances, action planning, **3D grounding from a single monocular image**, 9-DoF 3D boxes, relative (not absolute) spatial references. Benchmarked on ERQA, VSIBench, EmbSpatial, RefSpatial, **RoboSpatialHome**. 2B → 8B MMBench-EN thinking 79.9 → 85.3; MMStar 68.1 → 75.3. Qwen3-ViT backbone (replaces SigLIP-2, 1.5T tokens, stronger on OmniBench). | [tech report](https://arxiv.org/html/2511.21631v2) · [repo](https://github.com/qwenlm/qwen3-vl) |
| **InternVL3.5** | **1B / 2B / 4B / 8B / 14B / 38B / 20B-A4B / 30B-A3B / 241B-A28B** | Cascade RL, **Visual Resolution Router (ViR)** (dynamically picks visual-token resolution), **DvD** (split ViT / LLM across devices). **4.05× inference speedup** and +16% reasoning over InternVL3. Lightweight scores: 2B overall 50.7 (vs InternVL3-2B 32.4), 4B 57.4 (vs MiniCPM-V-4 33.5), 8B 60.3. **RK3588 NPU ports exist for 2B/4B.** | [arxiv](https://arxiv.org/html/2508.18265v2) · [RK3588-2B](https://github.com/Qengineering/InternVL3.5-2B-NPU) |
| **SmolVLM2** | **256M / 500M / 2.2B** (+ IBM 256M base) | Video-capable; 500M ≈ 2.2B video understanding at <¼ params; **"smallest video language models ever released"**; iPhone app runs 500M fully on-device. MLX-ready (Python + Swift). | [HF blog](https://huggingface.co/blog/smolvlm2) |
| **Phi-4-multimodal** | 5.6B | Too big for your budget; listed for completeness. | — |
| **Moondream** | ~1.9B (Moondream 3 family) | Edge-oriented VLM; commonly used for robot pointing/grounding. | [arxiv 2502.07858](https://arxiv.org/html/2502.07858v1) |
| **Florence-2** | **0.23B / 0.77B** | Open-vocab detection + captioning + grounding in one tiny model. **The best value-per-TOPS open-vocab detector that exists.** Pairs with SAM/SAM2 for masks. | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| **PaliGemma2 3B** | 3B | **21.6 tok/s on Orin Nano Super** — the fastest 3B-class VLM on Jetson by a wide margin (vs VILA-8B's 0.83). Designed for embodied control; π0's backbone. | [Jetson AI Lab](https://www.jetson-ai-lab.com/archive/benchmarks.html) |
| **Eagle-2** (GR00T's VLM) | 1.34B total | SmolLM2 LLM + SigLIP-2 image encoder, 64 tokens/frame at 224². | [GR00T N1](https://arxiv.org/html/2503.14734v2) |

---

## 7. The strongest current argument FOR and AGAINST a VLA for a navigation agent

### FOR (the honest case)

1. **A VLM-planner gives open-vocabulary goal grounding with no labels.** "Find the thing we use to open the fridge" requires commonsense that a 21-class detector cannot supply. Every modular pipeline that reaches >70% SR on HM3D-v2 does so with a VLM somewhere in the loop.
2. **Compositional language → sub-goal decomposition is a genuinely LLM-shaped problem.** SayNav, MapGPT, L3MVN, and every top-leaderboard method assume it works.
3. **A pretrained VLM brings visual priors you cannot train.** Scanning VA-22W and the OXE mixture is out of reach forever; VLM weights are not. GR00T N1 demonstrates *data* efficiency from pretraining: with 10% of the data it beat Diffusion Policy on full data by only 3.8% on real robot.
4. **Generalization to unseen object categories/scene layouts is exactly where modular pipelines still leak.** NoMaD says so in its limitations section.
5. **NaVILA proves the hierarchical *VLA* variant is real and effective:** +17% SR over the previous single-view SOTA, works across three robot embodiments by swapping only the low-level policy, 88% real-world success over 25 instructions, transfers to zero-shot mobile manipulation (a task NaVILA wasn't trained for).
6. **2026 SOTA on lifelong navigation is a 3B VLM.** AstraNav-Memory (Qwen2.5-VL-3B) at 62.7/56.9 GOAT-Bench Val-Unseen. The long-term-memory problem is being solved *by* small VLMs, not in spite of them.
7. **Small VLAs are now real.** SmolVLA-450M runs on a laptop, trains on one GPU, and beats π0-3.3B on real-robot pick-place with ~30k GPU-hours of total project compute.
8. **V-JEPA 2 shows a world model can do zero-shot planning with 62 h of unlabeled data and no rewards** — a data-efficiency regime no learned policy can match.

### AGAINST (the honest case, and it is stronger for *navigation* specifically)

1. **A VLA has no persistent state.** This is not a bug report; it is structural. OpenVLA, π0, GR00T N1, and Helix all condition on a fixed short context window. NaVILA uses **8 frames** and explicitly ablates that "8 frames are sufficient." A 3,000-step mission cannot fit. This alone disqualifies a monolithic VLA as your memory architecture.
2. **The compute doesn't exist at your budget.** VILA-8B (= NaVILA's architecture) runs at **0.83 tok/s on a 67-TOPS Orin Nano Super**. You have 5–20 TOPS. Even a 1B VLM is ~11–24 tok/s and 2–3 GB. There is no monolithic-VLA configuration that fits and is fast.
3. **The frequency gap is unclosable without more machinery.** π0 at 50 Hz still needs 73 ms (RTX 4090). OpenVLA-OFT reaches 109.7 Hz but only by throwing away autoregression and adding an MLP head — and it still needs 8×A100s to fine-tune. For navigation on a wheeled/legged base, the *inner* loop must be 30–50 Hz obstacle avoidance; a VLA at even 5 Hz is a path planner, not a controller.
4. **NaVILA's own compute section disqualifies it for edge.** 16 A100 nodes for pretraining stages; ~1 FPS on a 4090 for the 8B SFT. Even the AWQ-compressed version is 8.6 GB and 368 ms.
5. **The "reasoning" is an illusion, per 2026 diagnostics.** BeTTER: VLAs "catastrophically fail in dynamic scenarios… severe lexical-kinematic shortcuts, behavioral inertia, and semantic feature collapse," traced to "capacity compression and myopic downsampling," and "highly static evaluation protocols effectively mask this degradation."
6. **Chunking has a hidden, brittle hyperparameter.** π0.5's success is a peaked function of execution horizon; the optimum differs per task/model with no pattern; `e > prediction horizon` collapses; real robots stall at short horizons and drop objects at long ones.
7. **Geometry often beats the LLM outright.** 2025's controlled re-evaluation: a **training-free geometry-only frontier explorer** beats InstructNav-GT on HM3D SR *and* SPL at **$0 API cost, 2.5× lower runtime** (61.4/36.0 vs 60.6/33.8). SHF's language prior is best used as a *local* vote, not a plan generator.
8. **The 2026 ObjectNav leaderboards are stacked with training-free, zero-shot, modular methods, and the end-to-end learned baseline is last.** DD-PPO 27.9/14.2 on HM3D-v2 vs ConsistNav 84.2/41.2 and a training-free Hierarchical 3D Scene Graph at 80.1/39.5. "Hierarchical 3D Scene Graph" being #1–4 is the most direct empirical vindication of your hypothesis.
9. **Long-horizon VLA failures are attributed to a *missing harness*, not a missing model.** AdaHVLA: VLAs "struggle with long-horizon tasks requiring persistent memory and planning. **Task harnesses provide persistent context for agent reasoning.**" You would build the harness anyway; making it carry the VLM just adds a 3B dependency with no benefit.
10. **R2F is the existence proof for the alternative.** 78.3% SR / 29.6% SPL on HM3D ObjectNav at 25 Hz on a laptop, LLM-free, training-free, 6× faster than the best VLM competitor. The gap to VLM methods is 1.6 points of SR — noise.
11. **Persistent map is not an optional feature for a navigating agent — it is the task.** "Given a subgoal image, run at 4 Hz" (ViNT) and NoMaD's frontier-exploration loop *both* need a graph/occupancy structure around them. NaVILA's maplessness is listed as its main weakness; its own failure case is "inability to perform effective error correction when deviations occur" — a state-estimation problem, not a reasoning problem.

**Synthesis:** the winning position is **not "no VLA" but "VLA in the wrong place."** The VLM belongs in the *deliberative* layer — 0.2–1 Hz, on a 1–3B model, reading an explicit map and returning a ranking or a sub-goal list. The reactive layer should be classical (DWA/MPPI/TebyLocalPlanner over a costmap) with a small learned residual.

---

## 8. Recommended architecture (not a monolithic VLA), sized for edge

**Target: 1× Jetson AGX Orin 64GB (275 sparse INT8 TOPS, 85 dense, 64 GB, 204.8 GB/s, 15–60 W) — or degrade to Orin NX 16GB by dropping the deliberative VLM to 1B and shortening memory.** If you are truly at 5–20 TOPS, see §8.6.

### 8.0 Dataflow

```
Sensors (RGB-D / stereo / LiDAR / IMU / wheel-odom / joint encoders)
   │
   ├─[50–100 Hz]──► L0  State Estimation       ORB-SLAM3 / RTAB-Map / OpenVSLAM → T_wc(t)
   │                    + depth integration    Depth-Anything-V2-S @ 640×384, ~30 Hz
   │
   ├─[10–20 Hz]──► L1  Geometric Map           Nav2 costmap layers (voxel/rolling) at 0.10 m
   │                    (rolling, 12 m)        + static traversability layer from a TINY
   │                                            terrain net (see §8.4)
   │
   ├─[5–10 Hz]────► L2  Semantic Map            3D open-vocab object map:
   │                    (persistent)            - Florence-2-large (0.77B) or NanoOWL-base-patch32
   │                                            - SAM2 (Turbo, distilled) masks
   │                                            - LERF / ConceptGraphs-style feature lifting
   │                                            - 3D scene graph over object instances
   │                                            + **topological graph** (ViNT/NoMaD style) for
   │                                              fast long-range planning and dead-end detection
   │
   ├─[10 Hz]─────► L3  Local Planner           MPPI / DWA / hybrid-A* on the layered costmap
   │                                            + learned local residual (ViPlanner or ViNT/NoMaD 30M)
   │                                            → (v, ω) velocity command @ 20–50 Hz
   │
   ├─[0.2–1 Hz]──► L4  Deliberative Planner     InternVL3.5-2B or Qwen3-VL-2B (INT4/INT8)
   │                    (the ONLY "VLA")        Inputs: instruction, downsampled top-down semantic
   │                                            render, frontier list w/ crops, scene-graph summary,
   │                                            + compressed visual memory (see §8.3)
   │                                            Outputs: frontier ranking  +  sub-goal list
   │                                            (structured, ≤40 tokens, hard schema)
   │
   └─[0.1–0.2 Hz]► L5  Lifelong Memory          explicit: 3D object map + topo graph + visit log
                                             implicit: AstraNav-Memory-style compressed frame memory
                                             + episode/episode summary embeddings
```

### 8.1 L0/L1 — State estimation & geometry (never learned, never a bottleneck)

- ORB-SLAM3 (or RTAB-Map with a loop-closure-protected keyframe graph) at 30–100 Hz, or a simpler visual-inertial odometry + loop closure if the base is wheeled and ground-plane-constrained.
- **Depth Anything V2 Small** or **UniDepth** at 640×384. Depth is what makes open-vocab mapping and terrain costmaps possible; it is the highest value-per-TOPS sensor-derived signal.
- Nav2 layered costmaps (robot footprint layer, static obstacle layer, semantic traversability layer, inflation layer). `docs.nav2.org`

### 8.2 L2 — Persistent semantic map (**this is your memory**)

Use **both** memory representations, because they fail differently:

- **Explicit object-centric**: 3D object instances with (a) category from an open-vocab detector, (b) CLIP/SigLIP embedding, (c) oriented 3D box or point cluster, (d) first-seen frame + last-seen frame, (e) traversability/affordance. Queryable in O(1). This is LERF / ConceptGraphs / 3D-MEM / PONI's potential-function substrate.
- **Topological graph**: nodes = keyframes with 6-DoF poses + image embeddings + learned edge costs (from NoMaD's temporal-distance predictor, or from geodesic distance on the costmap). Edges invalidated when the costmap says the path is blocked. This gives you *global* route planning, loop detection, and "I've been here" — the things a metric grid does badly.
- **Do not** store raw point clouds at full rate. Store keyframe RGB at ~0.5 Hz + semantic/embedding summaries. The CVPRW 2026 spatial-memory survey's whole point is that unbounded memory is what kills embedded platforms.

**Open-vocab detector choice:** NanoOWL (OWLv2-base-patch32, **9.81 ms on Orin**) if your queries are noun phrases, or **YOLO-World-S (26 ms)** if you need relational expressions — and note the TensorRT caveat: *TensorRT-ing YOLO-World destroys its zero-shot ability*, while NanoOWL is TensorRT-native.

**Segmenter choice:** **NanoSAM** (distilled ResNet-18 MobileSAM teacher), *not* EfficientViT-SAM — NanoSAM had zero catastrophic failures across FP32/FP16/"best" precision while EfficientViT-SAM L0/XL0/XL1 collapsed to mIoU≈0 under FP16 encoders. Measured end-to-end: **47.51 FPS @ 84.64 mIoU** (Orin AGX 64GB).

### 8.3 L4/L5 — The deliberative layer, sized correctly

- **Model: InternVL3.5-2B or Qwen3-VL-2B, INT8/INT4.** At 11–24 tok/s on RK3588 / ~5 tok/s InternVL3.5-4B. Qwen3-VL-2B is the better pick if you need 3D grounding and affordance reasoning — its training data explicitly includes *relative spatial references, object affordances, feasible actions in 2D scenes, and 3D localization from a single monocular image.* InternVL3.5-2B is the better pick if you need its Visual Resolution Router to cut tokens adaptively.
- **Frequency: 0.2–1 Hz, never higher.** Budget ≤40 output tokens (a ranked list of frontier IDs + a sub-goal list). This is the design of R2F, FPE, and the Hydra-Nav "dual-process" argument.
- **Interface: a hard schema, not free text.** Frontier ID, score, expected info gain, one-line justification. This is the FPE/SHF lesson — *language as a local metric-aware prior, not a plan generator.*
- **Memory: adopt AstraNav-Memory's compression idea, but budget for quadratic attention.** DINOv3-ViT-Base + PixelUnshuffle×2 + 3×3 conv, injected at block 1 of the VLM's own ViT, giving ~30 tokens/frame (**16× compression from 598**) with 8–16× speedup on training and 2× memory saving (90.6→46.8 GB). **But cap history at 30–50 frames on edge, not 300.** Their own ablation says 200 frames at 16× is already worse than 100 because the model can't attend to recent frames. At 50 frames × 30 tokens = 1,500 visual tokens — tractable.
- **Add boundary cues, which AstraNav-Memory explicitly flags as missing.** Blend a few SAM/edge channels into the compressed tokens so carpet/floor and island/table don't blur. DINOv3's texture failure is documented and reproducible.

### 8.4 Local planner + learned residual

- **Classical core:** MPPI or DWA over the layered costmap at 20–50 Hz. This handles obstacle avoidance, which is exactly the thing NaVILA's low-level RL policy exists to do, and which you can get for free from a planner.
- **Learned residual (optional, 30M class):**
  - **NoMaD (30M)** if you need learned exploration *and* goal-reaching in one policy, runs on Jetson Orin, and plugs into your topological graph. This is the highest-value 30M model on the market for you.
  - **ViNT (31M) @ 4 Hz** if your goals are point/image-goal and you want smoother motion.
  - **ViPlanner** if terrain semantics matter (it runs at 10 Hz on a Jetson AGX Orin, reusing Mask2Former frames at 3 Hz asynchronously) and you can tolerate the ~3 Hz semantic dependency.
- **Terrain costmap from a tiny net**: train a small CNN (a few M params, distilled — distillation is what makes you quantization-robust) on height maps → traversability/slope/step-height cost. This is where NaVILA's real-world robustness against glass, holes, and rocks came from, and it is far cheaper than a VLM.

### 8.5 What you get, versus the alternatives

| Capability | Your modular stack | NaVILA-8B | SmolVLA-450M | Monolithic VLA |
|---|---|---|---|---|
| Persistent metric map | ✅ explicit | ❌ none (8 frames) | ❌ none | ❌ none |
| Long-horizon / lifelong memory | ✅ topo graph + object map + compressed frames | ❌ | ❌ | ❌ |
| Global route planning | ✅ hybrid-A*/Dijkstra | ❌ (implicit in LM) | ❌ | ❌ |
| Open-vocab goal understanding | ✅ 1–2B VLM @ 0.5 Hz | ✅ 8B @ 1 Hz | ✅ 450M | ✅ |
| Reactive obstacle avoidance @ 30–50 Hz | ✅ MPPI/DWA | ✅ (RL policy) | ❌ (4 Hz chunks) | ❌ |
| Terrain/semantic traversability | ✅ tiny CNN costmap | ✅ (LiDAR height map + RL) | ⚠️ | ⚠️ |
| Fits 5–20 TOPS | ✅ | ❌ (needs 4090) | ⚠️ (450M fits) | ❌ |
| Fits 64 GB Orin | ✅ comfortable | ⚠️ 8.6–18.5 GB VRAM | ✅ | ❌ 7B needs 15 GB min |
| Delivers its own training data | N/A (you train ~5M-param nets) | ❌ 128 A100s | ⚠️ 30k GPU-h project | ❌ 21,500 A100-h+ |
| Expected HM3D ObjectNav SR | **~60–78%** (modular SOTA band) | 54% on R2R-CE | N/A | N/A |
| Expected HM3D-v2 SR | **~63–84%** (training-free band) | N/A | N/A | 27.9% (DD-PPO) |

### 8.6 If you are truly at 5–20 TOPS

Drop L4 to **InternVL3.5-1B (24 tok/s, 1.9 GB) or Qwen3.5-0.8B (21.6 tok/s, 1.3 GB)**, or **remove the generative VLM entirely and use the R2F recipe** (NA-RADIO ray-frontier embeddings + SigLIP query scoring, cosine similarity, classical planner) — which is *LLM-free, training-free, beats VLFM and 3D-MEM, and runs at 25 Hz on a laptop.* In that regime, budget:
- 30% to depth + SLAM
- 30% to open-vocab detection + segmentation (NanoOWL + NanoSAM ≈ 20 ms/frame)
- 20% to the costmap + MPPI
- ≤20% to a 0.8–1B deliberative model at 0.1–0.2 Hz, or zero if you take the R2F path.

---

## 9. Open problems you would have to solve yourself

Ranked by how much they'd hurt you.

1. **Vision-language action-space binding for a navigation action space.** NaVILA showed language mid-level actions (`"forward 75cm"`) let you train on YouTube videos + ScanQA + general VQA. But on a *wheeled* base over terrain, the action space is richer than `{forward, turn left, turn right, stop}` at fixed velocities — you need speed selection, gait/mode switches, slope-aware speed, and recovery maneuvers. **No public work covers this well for wheels over rough terrain.** NaVILA is legged; its locomotion policy is a separate RL artifact you don't have. This is the #1 unsolved piece.
2. **Closed-loop error recovery.** NaVILA's own named failure: "the robot's inability to perform effective error correction when deviations occur." Almost every cited work is open-loop between sub-goals. You need to decide *when the deliberative layer gets re-invoked* — and every fix is a heuristic. RTC's guidance-based inpainting is a VLA-internal fix; for a modular stack you need the equivalent at the *plan* level (e.g. partial re-planning of the remaining sub-goal list when deviation exceeds a threshold).
3. **Time-budgeted replanning / anytime behavior.** The deliberative VLM at 0.2 Hz means a 5-second staleness gap. NaVILA runs at ~1 Hz. No published method defines how to choose the VLM call rate from a formal freshness/utility tradeoff for a mobile robot. (AutoHorizon is the closest analogue, but for action chunks inside a VLA, not for plan-level replanning.)
4. **Memory budget policy.** What to keep, at what rate, under a hard RAM cap. AstraNav-Memory shows *within-context* compression works; it does not address the sliding-window eviction problem, and its ablations show naive "more history is better" is false. The CVPRW 2026 spatial-memory survey is the right starting bibliography, and the field is explicitly open here.
5. **Quantization robustness of your deliberative path.** The Frontiers study shows hardware-aware architectures (EfficientViT-SAM) can collapse to mIoU=0 under FP16 while distilled ones (NanoSAM) don't. **Test every component at your target precision before you build anything on top of it.** Expect surprises.
6. **Open-vocab perception on the platforms that matter.** All ObjectNav numbers are indoor HM3D/MP3D/Gibson. Your environment is presumably terrain. There is no open-vocab terrain affordance dataset. ViPlanner's outdoor Zurich set is a handful of images. NaVILA's YouTube-touring pipeline is the only real method, and it is legged-specific.
7. **Evaluating "strong environment understanding" without a benchmark.** There is no ObjectNav-for-outdoor-terrain leaderboard. You will have to build one, and — per §2.4 — you must include a **strong geometry-only baseline** and report **runtime and cost**, or you will not know whether your VLM is doing anything.
8. **Sim-to-real for the semantic map.** PONI measured a 14.9%/45.4% SR drop from segmentation failure alone. ObjectNav detectors are trained on indoor RGB. On outdoor/terrain imagery the open-vocab detector is your dominant error source and you have no simulation for it.
9. **Cross-episode / lifelong consolidation.** GOAT-Bench shows the field only started measuring this in 2025–2026 (AstraNav-Memory, VTM-Nav, FSR-VLN). Nobody has a standard representation, and it directly conflicts with your RAM cap.
10. **Whether a learned local residual actually beats tuned MPPI on *your* terrain.** ViPlanner, NoMaD, and ViNT all beat their baselines on their authors' terrains. On yours, MPPI + a good costmap may be all you need, and the 30M network is dead weight.
11. **Licensing.** V-JEPA 2 is **non-commercial**. SmolVLA is Apache-2.0-ish (HF), Qwen3-VL is Apache 2.0, InternVL3.5 is MIT-ish, but Gemini Robotics On-Device, Helix, and GR00T are gated/commercial. If you need a world-model planner, V-JEPA 2's license is a real constraint — budget for distilling your own.
12. **Safety coupling.** DeepMind's own Gemini Robotics guidance: "interface our models with low-level safety critical controllers to execute the actions" and recommends red-teaming end-to-end. A 3B VLM must never be in the braking path. Design the arbitration now, not later.

---

## 10. One-paragraph answer to "should I use a VLA?"

**Use a VLM, not a VLA, and put it at 0.2–1 Hz above a real map.** Every piece of evidence points the same way: (i) VLAs have no persistent state, which is disqualifying for a navigating agent whose whole job is building and using one; (ii) the arithmetic doesn't close — VILA-8B runs at 0.83 tok/s on a 67-TOPS Orin Nano Super, and you have 5–20; (iii) the 2026 ObjectNav leaderboards are topped by training-free, zero-shot, *hierarchical* methods while the end-to-end learned policy sits last; (iv) a 2025 controlled study found a $0-API geometry-only frontier explorer beating the LLM pipeline on both SR and SPL; (v) R2F runs an LLM-free, training-free system at 78.3% SR on HM3D ObjectNav at 25 Hz on a laptop, only 1.6 points behind the best VLM method and 6× faster. Meanwhile, the two things VLAs genuinely give you — open-vocab grounding and language decomposition — are fully available from a 1–2B model called rarely, and the 2026 long-memory SOTA (AstraNav-Memory, Qwen2.5-VL-3B) is itself a *small* VLM above an explicit memory. The failure mode to design against is not "VLA is dumb," it is "VLA has no map": you must build the map, the planner, and the recovery logic yourself either way, so put the 3B model where it adds something a costmap can't.

---

## Appendix: source manifest

**Papers read in full** (local copies under `_research/out/`): `2412.04453` NaVILA · `2507.20021` When Engineering Outruns Intelligence · `2506.09985` V-JEPA 2 · `2406.09246` OpenVLA · `2502.19645` OpenVLA-OFT · `2410.24164` π0 · `2503.14734` GR00T N1 · `2506.01844` SmolVLA · `2306.14846` ViNT · `2310.07896` NoMaD · `2310.00982` ViPlanner · `2304.05501` L3MVN · `2402.15852` NaVid · `2510.03342` Gemini Robotics 1.5 · `2602.09972` Hydra-Nav · `2201.10029` PONI · `2005.08229` ANSL · `2309.04077` SayNav · `2401.07314` MapGPT · `2505.23705` Knowledge Insulation · `2506.07339` RTC · `2511.21631` Qwen3-VL · `2508.18265` InternVL3.5 · `2210.05782` VLMaps · `2009.02796` M3TDP · `2306.12148` Hydras · `2012.08815` SMACL · `2011.08111` OVMM · `2512.21627` AstraNav-Memory · `2603.08475` R2F · `2607.05122` · `2602.21445` VLA Knows Its Limits · `2501.03575` Cosmos · `2601.16163` Cosmos Policy.

**Web sources:** [figure.ai Helix](https://www.figure.ai/news/helix) · [DeepMind Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) · [DeepMind Gemini Robotics On-Device](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/) · [DeepMind GR 1.5](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1-5-Tech-Report.pdf) · [Meta V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) · [Jetson AI Lab benchmarks](https://www.jetson-ai-lab.com/archive/benchmarks.html) · [NVIDIA Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks) · [NVIDIA Jetson Orin specs](https://developer.nvidia.com/embedded/jetson-orin) · [HF SmolVLA](https://huggingface.co/blog/smolvla) · [HF SmolVLM2](https://huggingface.co/blog/smolvlm2) · [Frontiers edge VLM perception](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1693988/full) · [SOTA2 HM3D v1](https://www.sota2.com/research/sota/object-navigation-on-hm3d-v1) / [v2](https://www.sota2.com/research/sota/object-navigation-on-hm3d-v2) · [NVIDIA GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5) / [N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) · [PI knowledge insulation](https://www.pi.website/research/knowledge_insulation) · [Cosmos GR00T-Dreams](https://docs.nvidia.com/cosmos/latest/predict2/post-training_video2world/groot-dreams.html).

**Code:** [openpi](https://github.com/Physical-Intelligence/openpi) · [openvla](https://github.com/openvla/openvla) · [openvla-oft](https://github.com/moojink/openvla-oft) · [NaVILA](https://github.com/AnjieCheng/NaVILA) · [visualnav-transformer (ViNT/NoMaD/GNM)](https://github.com/robodhruv/visualnav-transformer) · [viplanner](https://github.com/leggedrobotics/viplanner) · [L3MVN](https://github.com/ybgdgh/L3MVN) · [vlmaps](https://github.com/vlmaps/vlmaps) · [PONI](https://github.com/srama2512/PONI) · [PONI alternative](https://github.com/srama2512/poni) · [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) · [AstraNav-Memory](https://github.com/amap-cvlab/AstraNav-Memory) · [instructnav-scrutinized (FPE/SHF)](https://github.com/matinaghaei/instructnav-scrutinized) · [Qwen3-VL-2B on RK3588](https://github.com/Qengineering/Qwen3-VL-2B-NPU) · [InternVL3.5-2B on RK3588](https://github.com/Qengineering/InternVL3.5-2B-NPU) · [Qwen3-VL](https://github.com/qwenlm/qwen3-vl) · [habitat-challenge](https://github.com/facebookresearch/habitat-challenge) · [Nav2](https://docs.nav2.org/) · [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) · [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) · [LeRobot async inference](https://huggingface.co/docs/lerobot/en/async).
