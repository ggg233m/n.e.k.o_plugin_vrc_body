# Semantic / spatial-map papers — full read (semmap-full, 2026-09)

> **状态**：外部调研（**全文精读**，不是搜索片段）；快照日期 **2026-09-30**。
> 这不是"当前实现"的描述；任何与代码不符处**以代码为准**（见 `Docs/README.md` §一）。
> 已登记于 `Docs/README.md` §三（📑）。文中所有数字均来自实读原文，未读到的标 **NOT FOUND**。

Scope: papers 1–5 of the brief, read in FULL from arXiv HTML, not snippets.
All page/section references are to the fetched full text. Anything I could not find is marked **NOT FOUND**.

## 0. Two corrections to the brief (act on these)

1. **Hydra's arXiv ID in the brief is wrong.** `arXiv:2210.13420` is *"Twitter Users' Behavioral Response to Toxic Replies"* (Aleksandric, Roy, Nilizadeh; WWW '23) — a social-media toxicity paper, not robotics. The real Hydra is **arXiv:2201.13360** (Hughes, Chang, Carlone; **RSS 2022**, `roboticsproceedings.org/rss18/p050.html`). RayFronts' own reference list confirms 2201.13360. I read 2201.13360.
2. **The "no depth sensor" premise in `Docs/spatial-memory-research-2026.md:3` is obsolete.** Per current context the rig now has SGBM stereo (720×405/eye, fx=202.5@720, b=0.063 m) and a working 2.5D three-state occupancy grid. That changes the verdict on §6.2 of that doc, which "explicitly rejected ... any 3D/occupancy/voxel map (**needs depth**)". A depth-gated map is now available; only *learned-metric* depth is forbidden. This re-opens a whole class of papers that doc eliminated.

---

## (a) Comparison table

| System | Memory type | Inputs required | Depth required? | Runtime (exact) | Training / annotation | Code released |
|---|---|---|---|---|---|---|
| **RayFronts** 2504.06994 (CMU) | Semantic sparse **voxel map** `V_t` + OpenVDB occupancy `O_t` + frontiers `F_t` + **semantic ray fronts** `R_sem={(o_r,θ_r,φ_r,f_r)}` (§III) | posed **RGB-D**; depth from "**stereo, LiDAR, or monocular depth estimation**" (§III-E Observe) | **Yes.** RGB alone insufficient — §III-B takes `D_t∈R^{H×W}` | **8.84 Hz** end-to-end on Jetson AGX Orin @224×224, 30 cm voxel, base encoder, top-100 PCA (~80% var), ray tracing **off** (§V-C). Encoder-only **17.5 Hz**, **16.5×** faster than Trident, **46%** of its params (§V-C, C3) | **None** — zero-shot, frozen RADIO v2.5-L + SIGLIP + NACLIP attention trick (§III-A). No 3D training | **Yes** — `github.com/RayFronts/RayFronts`, standalone encoder offered |
| **R2F** 2603.08475 (Sapienza) | **Purely geometric occupancy** (WaveMap hashed chunked wavelet octree, 0.1 m voxel, 0.05 m min cell) **+ semantics stored ONLY at frontier regions**, each with discrete direction bins holding a weighted-average feature `f_{m,b}` (§III-C/D) | posed **RGB-D** (`o_t=(I_t,D_t)`), depth clipped `r_max=3.5 m`, camera height 1.25 m (§III-A, III-F) | **Yes** for the frontier/ray step; occupancy itself needs depth | **25 Hz** avg inference rate on laptop: Intel Core 9 Ultra 185H, 32 GB RAM, **RTX 4070 8 GB VRAM** (§IV-E). Episode time 32.7 s vs VLN-Game 122.0 s (Table I) | **None** — "real-time LLM-free **and training-free**" (§I); NA-RADIO + SigLIP, frozen | **Yes** — `github.com/Lab-RoCoCo-Sapienza/r2f` |
| **Hydra** 2201.13360 (MIT, RSS 2022) | Hierarchical **3D scene graph**: agent / mesh / places / objects / rooms, built on Kimera's metric-semantic mesh (§IV, §V) | visual-**inertial** + **stereo-depth reconstruction at keyframe rate**; Kinect depth on real datasets (§V-A, VI-A) | **Yes** — depth per keyframe | "real-time on a multi-core CPU; **the only module that relies on GPU computing is the 2D semantic segmentation**" (§V-C). Target rate = **keyframe rate (5 Hz)** (§VI-D). On Xavier NX: objects **75±35 ms**, places **33±6 ms**, rooms **55±41 ms** (§VI-D). Batch baseline >40 s per scene (§VI-C) | Requires 2D semantic segmentation + stereo; not training-free-competitive with modern OV encoders | **Yes** — `github.com/MIT-SPARK/Hydra` |
| **ConceptGraphs** 2309.16650 (ICRA 2024) | **Object-centric 3D scene graph** `M_t=<O_t,E_t>`; node = point cloud + CLIP feature + LLM caption; edge = inter-object relation (§III) | posed **RGB-D** `I_t=<I^rgb, I^depth, θ_t>` (§III-A) | **Yes** — "builds an open-vocabulary 3D scene graph from a sequence of **posed RGB-D** images" (Fig. 2) | **NOT FOUND** — no runtime, latency or FPS number anywhere in the paper; no "real-time" claim for itself (only cites Hydra-lineage for real-time 3DSGs, §II) | "off-the-shelf models (**no training/finetuning**)" (§II), **but** calls **GPT-4 (gpt-4-0613)** per scene for captions/relations + LLaVA + SAM + CLIP (§IV) | **Yes** — `github.com/concept-graphs/concept-graphs` |
| **LM-Nav** 2207.04429 (CoRL 2022) | **Topological graph** `G(V,E)`, V = observed images, E = traversals with learned temporal distance `D(v_i,v_j)` (§3) | RGB images + GPS + odometry; **no depth** used for the map | **No** | **NOT FOUND** — "The LLM and VLM queries are pre-computed on a remote workstation"; VNM (ViNG) runs onboard on CPU/GPU unspecified (§5.2) | **None** for LLM/VLM (GPT-3 + CLIP off-the-shelf); the VNM (ViNG) is **trained** | **Yes** — code + Colab at `sites.google.com/view/lmnav` |

### Item 4 — the best 2026 open-vocab 3D mapping / object-level SLAM option

I evaluate four 2026 candidates head-to-head. **No single paper satisfies "monocular/stereo + real-time + open-vocabulary + instance-level" simultaneously**, and one of them says so explicitly (VOIM).

| System | Modality | Real-time? | Runtime / hardware | Code |
|---|---|---|---|---|
| **OVI-MAP** 2603.26541 — **CVPR 2026 Highlight** | **RGB-D + poses** (depth from *any* source, incl. stereo) | **Yes — 30 FPS** | RTX 3090 + i7-12700K. Per-component (Table 8): RGB seg **964.8 ms** (skip 30), depth seg **188.4 ms**, 2D-3D assoc **176.3 ms**, view selection **140.8 ms** (skip 10), feature extract **131.3 ms** (skip 3); multi-threaded so the pipeline stays 30 FPS | `github.com/OVI-MAP/OVI-MAP` |
| **Ov3R** 2507.22052 — **CVPR 2026 Highlight** | **RGB-only video**, no depth sensor, no explicit pose estimation (poses are a pointmap by-product) | Yes, **15 FPS** | Replica: ATE RMSE **6.00**, **15 FPS** (Table 1; Spann3R 32.79@>50 FPS, SLAM3R 6.61@24 FPS). Trained on **4×A100 64 GB**; inference needs a single **3090 24 GB** (§4.2 Impl. Details). Peak GPU observed up to **22572 MB** in the efficiency table | `github.com/ZoranGong/Ov3R` |
| **VOIM** 2609.00775 | **RGB-D *or* monocular RGB alone** — "runs unchanged from RGB-D with known poses down to a bare RGB stream" (Abstract) | **No — "the pipeline is offline-rate (Sec. IV-G)"** (Limitations, §VI) | — Per-point feature buffer ~**4 M points at 1024-D exhausts a 23 GiB GPU** (§VI). Monocular map is defined only up to a **global similarity** → "metric deployment needs an external scale reference **such as a calibrated stereo baseline or an IMU**" (§VI) | **NOT FOUND** |
| **OVO-SLAM** 2411.15043 | **RGB-D** | **No** — "its current processing time of **~1 frame per second** limits its application to platforms with multiple GPUs available" (§4 Limitations) | RTX-3090 (§4, §5) | `github.com/tberriel/OVO` |

**Verdict for item 4, for THIS rig:** **OVI-MAP is the best fit**, not Ov3R, because the project *has* a stereo camera, and stereo SGBM supplies the depth OVI-MAP needs **with the metric coming from the calibrated baseline** — which satisfies the project's hard constraint that the metric source must be stereo baseline / HMD displacement / OSC speed, never a learned metric depth. Ov3R is the better answer only if you refuse to use stereo depth at all; it also costs 22.5 GB peak vs the A770's 16 GB. OVO-SLAM and VOIM are both ruled out on rate.
Note the VOIM quote for the record: "To our knowledge no concurrent system is simultaneously **training-free, online, instance-level, and monocular**" (§II).

---

## 1. RayFronts — what it actually stores, and the architectural detail that matters

**What is stored** (§III, one paragraph each):
- `V_t` — sparse semantic voxels: coordinate + feature `f_i ∈ R^{3+D+1}` (3 RGB, `D` feature, 1 hit-count) + hit count. Fusion is a **plain weighted average with hit count as the weight**, done by a parallel **scatter-reduce** over discretized coordinates (§III-B). Local updates buffer `m` frames, then voxelize at resolution `α`.
- `O_t` — occupancy in **OpenVDB**, log-odds in a **signed byte**, clamped. Lives on **CPU**; `V_t`, `F_t`, `R_t` live on **GPU as PyTorch tensors** (§III).
- `F_t` — frontiers: iterate free voxels, a voxel is a frontier if its neighbours meet `min_unobsrv`, `min_occ`, `min_free`; then **subsampled** onto a coarser grid `β` (§III-D). Online values: neighbourhood_r=1, min_unobsrv=9, min_occ=0, min_free=4, subsampling=4, subsampling_min_fronti=5 (Table A.2).
- `R_t` — **semantic ray fronts** `R_sem={(o_r, θ_r, φ_r, f_r)}`, θ∈[−π,π), φ∈[0,π) (§III-E). This is the paper's key contribution: rays, not just frontiers, because a single feature per frontier causes **feature collisions** when two different objects are seen through the same frontier.

**Ray recipe (verbatim parameters, Table A.2):** out-of-range mask from `D_t` → **eroded** (`ray_erosion=32` online, `0` in the throughput run) to stop semantic leakage at boundaries → two-stage frontier filtering: discard frontiers not in front of the ray, discard `d_ortho > β`, discard `d_orig > 4×depth_range`; cost `d_cost = (d_ortho/max{d_ortho} + d_orig/max{d_orig})/2 ∈ [0,1]`, pick min (§III-E Eqs. 1–2). Discretize into angle bins `ψ=30°` via `θ=atan2(d1,d0)`, `φ=acos(d2)`; merge local+global with weight `1−d_cost`. `max_dirs_per_frame=10000`. Optional ray tracing marches each ray through `O_t`; when disabled, removed rays would "propagate indefinitely, so we disable the behavior under that setting" (§III-E).

**Pruning:** semantic voxels with occupancy < 0.5 are removed (§III-C) — this is the paper's *entire* dynamic-object mechanism.

**Encoder (§III-A):** RADIO v2.5 (distils CLIP+DINOv2+SAM) + **NACLIP's** locality trick — augment the final ViT block's attention with an unnormalized multivariate Gaussian kernel centred on each patch, `gauss_std σ=7.0`. Then project spatial features through RADIO's **SIGLIP summary-feature adapter** into the SIGLIP CLS-token space. Straightforward "project spatial features onto CLIP/SIGLIP with their adapters" "yields subpar performance"; the SIGLIP-summary-adapter route is what works.

**VRAM:** **NOT FOUND.** No GPU-memory figure is given. The paper says RayFronts "has the highest memory consumption" of the online baselines (Limitations) and that ground truth was generated at an 80 m cutoff, "the **highest value that fits in our memory**" (Fig. A.2). Do not quote a number.

**Depth needed?** Yes for the metric map. Explicitly: depth is "obtained via **stereo, LiDAR, or monocular depth estimation**" (§III-E). So stereo is sanctioned by the paper. But note that RayFronts itself **does not derive depth** — it consumes `D_t`. On this rig that means SGBM.

**Training:** none. Frozen encoder, no 3D supervision.

---

## 2. R2F — how it reuses RayFronts, and why it is the better template here

**What it keeps from RayFronts:** the NA-RADIO dense encoder (§III-B, identical recipe incl. the SIGLIP summary adapter), the out-of-range semantic-ray idea, the ray→frontier geometric-compatibility test, and the 30° angular binning.

**What it changes (the load-bearing differences):**
1. **Semantics are NOT fused volumetrically.** "In contrast to RayFronts, semantic evidence is **not** fused volumetrically into the occupancy grid and **no ray re-casting is performed**. Instead, semantic information is stored exclusively at frontier regions." (§III-D, *Storing semantics efficiently*). Occupancy stays **purely geometric**.
2. **Directions become explicit navigation goals**, not just exploration priors. Region score `S_m = max_b f_{m,b}ᵀ t_q`, both `ℓ2`-normalized; argmax bin gives the preferred direction (§III-E).
3. **Two-rate execution schedule** (§III-D *Execution schedule*, §III-F): semantic ray accumulation + dense feature extraction run **every timestep**; frontier extraction, occupancy query and region synchronization run **every `N_map = 5` steps**.

**Map representation:** WaveMap hashed chunked wavelet octree, **voxel 0.1 m**, min cell width 0.05 m (§III-F). Frontier = voxel with `y_min ≤ y ≤ y_max` (navigable band), ≥ `k_u=3` unknown 6-connected neighbours, ≥ `k_f=1` free neighbours (§III-C). Regions = clustered frontier voxels with centroid `c_m`; merge radius 0.8 m.

**Exact hyperparameters (§III-F) — copy these, they are the useful deliverable:**
`r_max = 3.5 m`, camera height 1.25 m, `τ_r = 14.0 m`, `τ_⊥ = 1.0 m`, angular bin **30°**, invalidation radius 1.0 m, visited filter 2.0 m, `k_u=3`, `k_f=1`, `N_map=5`.
Detection: `N_cons = 3` consecutive frames above **`τ_g = 0.14`**. R2F-VLN: `N_confirm = 3`, landmark threshold **`τ_ℓ = 0.11`**, `τ_syn = 0.60`, `K_syn = 5`.

**The single most transferable number in this whole report:** R2F Fig. 3 states "**informative similarity values typically lie in the range [−0.10, 0.15]**" in the RADIO→SIGLIP space. So `τ_g = 0.14` sits essentially **at the top of the informative band** — the entire discriminative budget is **0.25 wide**. Any domain shift that compresses this band destroys the channel. This is a measured, quotable fragility, not speculation.

**Real-world / CPU numbers:** real-world validation is on a **TIAGo** robot as a ROS package, finding "a sink" (§IV-E). "Our software runs on a laptop equipped with an Intel Core 9 Ultra 185H, 32 GB of RAM, and an NVIDIA GeForce 4070 with 8 GB of VRAM. It achieves an average inference rate of **25 Hz**" (§IV-E). CPU-only numbers: **NOT FOUND**. Training: none.

**Results (Table I):** ObjectNav R2F SR **78.3 / SPL 29.6 / 32.7 s** vs VLN-Game 76.7 / 28.0 / 122.0 s, VLFM 40.0 / 7.71 / 84.2 s, 3D-MEM 26.7 / 15.9 / 61.2 s, OpenFrontier 23.3 / 7.12 / 245.0 s. VLN (R2F-VLN) SR 28.0 / SPL 13.94 / 40.3 s vs VLN-Game 43.7 / 22.7 / 504.0 s.

---

## 3. Hydra and ConceptGraphs — why scene graphs need RGB-D, and whether a 2D-only or stereo-only variant exists

**Why they need RGB-D.**
- **Hydra**: the mesh/places/objects/rooms layers are built on **Kimera's metric-semantic mesh** (`§V-A "The real-time construction of the metric-semantic 3D mesh (Layer 1) is an extension of Kimera"`), and early perception does "**stereo-depth reconstruction (at keyframe rate)**"; the real datasets use "the depth reconstruction from the **Kinect**" (§VI-A). Places and rooms are **derived from the ESDF**, i.e. from geometry. Remove depth and the ESDF — and therefore the top three layers — does not exist.
- **ConceptGraphs**: object nodes are **3D point clouds** `p_{o_j}` obtained by back-projecting masks with depth (§III-A), and association uses geometric similarity on those clouds (voxel 2.5 cm, `δ_nn` 2.5 cm). Fig. 2 caption: "builds an open-vocabulary 3D scene graph from a sequence of **posed RGB-D** images." No depth → no object point clouds → no geometric association → no graph.

**Real-time claims.**
- Hydra: **yes, and it is unusually honest about the target** — "real-time" means **keyframe rate 5 Hz** for the mid/high-level layers (§VI-D). The impressive part for this project is that **only the 2D semantic segmentation touches the GPU**; everything else is multi-core CPU (§V-C). The authors' stated motive is exactly our situation: "Running on CPU has the advantage of (i) leaving the GPU to learning-oriented components" (§V-C).
- ConceptGraphs: **NOT FOUND.** The paper never claims real-time for itself and reports no runtime/latency. In practice it is offline-rate by construction (SAM per frame + LLaVA + **GPT-4 API** per scene).

**2D-only / stereo-only variant:** **NOT FOUND for ConceptGraphs.** For Hydra, stereo is a *supported input* (stereo-depth reconstruction at keyframe rate), so a stereo-only variant is not needed — Hydra already is one, provided the stereo is dense enough. The nearest genuine "2D-only but semantically open" relatives named in RayFronts §II-C are **VLFM** (2D frontier semantics, single-object, indoor only) and **EPG** (Embedding Pose Graph: one feature vector per pose node, no fine-grained map). RayFronts explicitly criticises EPG: whole-image compression means "non-prominent objects" are lost — which is exactly the failure signature of this project's removed grid-token channel.

---

## 4. LM-Nav — the exact recipe for putting language landmarks into a topological graph

This is the part of the brief that most directly specifies the persistent-memory interface, so here is the **verbatim recipe**, step by step.

**Step 1 — build the topological graph (before any language is involved).** Nodes = images observed during a prior traversal. Edges are deduced from three cues (§4.3, Appendix B `Algorithm 2`):
- **temporal proximity during data collection**: if node timestamps are within `ε = 1 s` (`<2 s` stated in §4.3), connect — these were physically traversed;
- **learned distance from the VNM (ViNG)**: if the VNM estimates the two node images are close (within `τ = 80 time steps ≈ 20 m`), connect — this links distant nodes along the same route and across different times of day;
- **spatial proximity from GPS**: `η = 100 m`.
Then: "To avoid cases of underestimated distances by the model due to aliased observations, e.g., green open fields or a white wall, we **filter out prospective edges that are significantly further away as per their GPS estimates**" — the paper's own name for the failure is a **"wormhole"**. Finally, "we perform a **transitive reduction** operation on the graph to remove redundant edges" (Appendix B).

**Step 2 — LLM extracts ordered textual landmarks.** GPT-3, **temperature 0**, with **3 in-context examples** in the prompt; `ℓ̄ = ℓ₁,…,ℓₙ` (§4.2, Appendix A). Table 2 compares GPT-3 against fairseq, GPT-J-6B, GPT-NeoX-20B, and a **spaCy base-noun-phrase chunker** — the noun chunker "performs surprisingly reliably, correctly solving many simple prompts". (Relevant if you want to avoid an LLM call entirely.)

**Step 3 — VLM grounds each landmark on each node.** CLIP applied to the node's stored image with the caption prompt **`"This is a photo of a [ℓ_j]"`** (Appendix A: "This simple prompt was sufficient to detect over 95% of the landmarks encountered"). Convert logits to a distribution by softmax **over nodes**:
`P(v_i | ℓ_j) = exp CLIP(v_i, ℓ_j) / Σ_{v∈V} exp CLIP(v, ℓ_j)` (§4.3).

**Step 4 — the objective and the search.** With `t̄ = (t₁,…,tₙ)` a monotonically increasing index sequence (§4.1):
`R(v̄, t̄) = Σ_{i=1..n} CLIP(v_{t_i}, ℓ_i) − α Σ_{j=1..T−1} D(v_j, v_{j+1})`, where `α = −log γ`.
DP/Dijkstra-style recursion (§4.4 Eq. 5, Algorithm 1):
`Q(i,v) = max( Q(i−1,v) + CLIP(v, ℓ_i),  max_{w∈neighbors(v)} Q(i,w) − α·D(v,w) )`
base case `Q(0,v) = −` (shortest path length from start S). Result: a walk `v̄ = (v₁,…,v_k)` executed by the VNM using its action estimates.

**Step 5 — reported performance and honest failure modes.** 85% instruction success over 20 instructions, "an average of 1 intervention per 6.4 km" (§5.3). Ablations:
- **Max-likelihood planning** (drop the `P_t` traversability term, pick only the highest-CLIP node per landmark, connect by shortest path / Floyd–Warshall) — "suffers greatly in the form of efficiency" (5× worse in the illustrated case) **and** its planning success also drops (Appendix C.3, Table 4). The traversability term is not optional.
- **CLIP cannot retrieve "hard" landmarks.** "CLIP is unable to retrieve a small number of 'hard' landmarks, including **fire hydrants and cement mixers**" (§5.3, Missing landmarks). Also variable-binding errors: for `"A photo of a blue dumpster"` the max-likelihood planner picked an image containing "a blue semi-truck and an orange trailer, but no blue dumpsters" (Fig. 8).
- Without the VNM, a straight-line controller "results in collisions with a curb, a tree, and a wall in 3 individual attempts" (§5.4).

---

## (b) Concrete blueprint for the "什么在哪" channel on stereo RGB + pose + a frozen encoder

### B0. The depth budget that constrains everything (derived, not quoted)

With fx = 202.5 px at 720 width and b = 0.063 m: `fx·b = 12.7575 m·px`, so `Z = 12.7575 / d` and `σ_Z ≈ Z²·σ_d / 12.7575`.

| Z (m) | disparity d (px) | σ_Z @ σ_d = 1 px | σ_Z @ σ_d = 0.5 px | rel. σ_Z @ 0.5 px |
|---|---|---|---|---|
| 1 | 12.76 | 0.078 | 0.039 | 3.9% |
| 2 | 6.38 | 0.313 | 0.157 | 7.8% |
| 3 | 4.25 | 0.705 | 0.353 | 11.8% |
| 4 | 3.19 | 1.254 | 0.627 | 15.7% |
| **5** | 2.55 | **1.962** | 0.981 | **19.6%** |
| 6 | 2.13 | 2.82 | 1.41 | 23.5% |
| 8 | 1.59 | 5.02 | 2.51 | 31.4% |
| 10 | 1.28 | 7.84 | 3.92 | 39.2% |

**Conclusion: set `r_max ≈ 3.0–4.0 m`.** Beyond ~5 m a single-pixel disparity error exceeds 1 m. R2F independently chose `r_max = 3.5 m` (§III-F) — the numbers agree, which is a strong signal that the R2F template is the right one for a 63 mm baseline. This also means the semantic channel is *mostly* a **beyond-range ray channel**, exactly RayFronts/R2F's design point.
Two extra in-repo facts to fold in: ORB-SLAM3 evaluation measured **`s = d_osc / d_stereo = 0.755`** (`Docs/archive/ORB-SLAM3双目参照评估（2026-09-26）.md:46`), i.e. stereo metric runs ~1.32× long and must be scaled by ~0.755 to agree with the OSC/HMD metric; and run5 recorded `baseline_tracking_m = 0.126` = 2×0.063 (`Docs/archive/双目序列run5-7结论汇总（2026-09-27）.md:22`). **Doubling the effective baseline halves σ_Z** — free accuracy if the rig can use run5's configuration.

**Feature patch resolution (also constraining):** with a pinhole model the frame width at depth Z is `3.5556·Z` m. At 720→224 input (16 patches across) each patch spans **0.222·Z m** → 0.67 m at 3 m, 1.33 m at 6 m. A 1.7 m avatar at 3 m is then only ~2.5 patches wide. At 448 input each patch spans **0.111·Z m** (0.33 m at 3 m). **Recommend ≥448 input for landmark-scale semantics**, and preserve the 16:9 aspect (720×405 → e.g. 448×252) rather than square-resizing, since the papers' square inputs assume their own cameras.

### B1. Layer 1 — geometry (keep, add a gate). No new paper needed.
`backend/nav_mapping.py` + `nav_grid.py` (FREE/OCC/UNK, `GridMeta`) stays the **only** metric source. Add the `r_max` gate so everything past ~3.5 m becomes explicitly **UNKNOWN/"beyond-range"** rather than a noisy metric reading. Metric provenance: SGBM baseline (and/or HMD displacement / OSC speed), never a learned metric depth. This honours R2F's separation principle (§III-D) — semantics never write into occupancy.

### B2. Layer 2 — the frozen dense language-aligned encoder (replaces the failed grid tokens). Source: **RayFronts §III-A / R2F §III-B**.
Replace `landmark_probe.py`'s `--mode grid` token `(brightness band, edge-density band)` — whose measured cause of failure was documented in-repo as *"网格下标根本不对应"* + relative-to-frame-mean normalisation degenerating into a scene-independent self-similar quantity (`Docs/archive/世界模型导航方案-可行性评估.md:1815-1817`) — with a **dense patch-level language-aligned feature map**. Recipe, from R2F §III-B verbatim: RADIO backbone → **NACLIP neighbourhood-aware attention** (replace global query-key similarity with `sim_ij = k_iᵀk_j/√D · G(‖u_i − u_j‖)`) → **SIGLIP summary-feature adapter** to project patches into SIGLIP CLS space. Query with `t_q` (SigLIP text embedding of the phrase) via cosine.

*Encoder choice on Arc A770 + OpenVINO:* RADIO v2.5-L + the NACLIP modification is a PyTorch ViT-L — **NOT FOUND** any OpenVINO port, and the repo's `.venv` has no torch (`Docs/archive/世界模型导航方案-可行性评估.md:1834`). Two honest routes: (i) export the RayFronts standalone encoder (`github.com/RayFronts/RayFronts` offers it "independent of the rest of the codebase") to OpenVINO IR — unverified effort, and note the in-repo OpenVINO finding that **`grid_sample` "runs very slowly" on Arc A770** (`Docs/业界世界模型方案落地评估（2026-09-29）.md:215`), which affects warping-based stereo/flow nets, not plain ViT inference; or (ii) substitute a smaller SigLIP/CLIP-ViT patch-feature encoder with the same NACLIP locality trick. Whichever you pick, the **feature space must be one that a text encoder can address**, and the similarity band must be re-measured (see B5). Coordinate the export with the `vlm-ov` workstream.

### B3. Layer 3 — in-range semantics. Source: **RayFronts §III-B**.
Semantic voxels/cells: unproject occupied points with the gated stereo depth, nearest-neighbour-interpolate the patch feature onto each point, accumulate in a buffer of `m` frames, voxelize at `α`, and fuse by **hit-count-weighted average via scatter-reduce** (§III-B). Store RGB + feature + hit count. This is the *"what is here"* half.

### B4. Layer 4 — beyond-range semantics = the actual "什么在哪" fix. Source: **R2F §III-D** (simplified RayFronts).
1. `OOR mask`: pixels with `D(u,v) ≥ r_max`; **erode in the image plane** (R2F §III-D; RayFronts uses `ray_erosion`).
2. Sample a bounded subset of rays (RayFronts caps at `max_dirs_per_frame = 10000`); compute world direction `d` from the current pose.
3. Extend the existing 2.5D grid to expose **frontier regions** (boundary between known-free and unknown), clustered with a merge radius (R2F: 0.8 m), restricted to a navigable height band.
4. Ray→region association (R2F §III-D): keep regions with `dᵀv_m > 0`, perpendicular distance < `τ_⊥` (R2F: 1.0 m), radial distance < `τ_r` (R2F: 14.0 m); assign to the min normalized cost. Accumulate the feature into that region's **direction bin** (30°) by weighted running average.
5. Store semantics **only at frontier regions**; keep occupancy purely geometric; **no re-casting** (R2F §III-D).
6. Query: `S_m = max_b f_{m,b}ᵀ t_q` (§III-E). Goal detected when the max spatial response exceeds `τ_g = 0.14` for `N_cons = 3` consecutive frames (§III-F) — **re-calibrate these, do not copy them** (see B5).
7. Update cadence: semantics every frame, frontier/region sync every `N_map = 5` frames (§III-D/§III-F).

### B5. Layer 5 — place identity + language routing (the persistent part). Sources: **R2F (region scoring) + LM-Nav §4 (graph grounding)**.
- Keep the two-layer place graph from `Docs/spatial-memory-research-2026.md` §6.2 for persistence, but **upgrade the node descriptor** from a bare appearance vector to the same frozen encoder's output, so place identity and object semantics share one embedding space and one query path.
- Language routing: use **LM-Nav §4.3–4.4 verbatim** — softmax `P(v_i|ℓ_j)` over nodes, objective `R(v̄,t̄) = Σ CLIP(v_{t_i},ℓ_i) − αΣ D(v_j,v_{j+1})`, DP/Dijkstra `Q(i,v)`. Replace ViNG's `D(·,·)` with the project's existing **per-sector traversability/divergence-risk cost + dead-reckoned displacement**; replace GPS spatial proximity with the self-built frame's uncertainty scalar; keep the **transitive-reduction** and **wormhole filter** steps.
- Replace GPT-3 with the local LLM at temperature 0 and 3 in-context examples; **first try the spaCy noun-chunker baseline** (LM-Nav Table 2 shows it solves many simple prompts with no LLM at all).
- **Mandatory calibration before any threshold is trusted.** R2F's informative band is only [−0.10, 0.15] wide (§Fig. 3). Run the same-place vs different-place separation experiment already prescribed at `Docs/spatial-memory-research-2026.md` §6.1 Step 0, in the *new* embedding space, and derive `τ_g`, `τ_ℓ` and the margin from it. Do **not** inherit 0.14 / 0.11.

### B6. What this fixes about the recorded zero-discrimination failure
The in-repo diagnosis is *partially* right and *incomplete*. It correctly identifies (i) no frame-to-frame alignment under translation and (ii) relative-to-frame-mean normalisation. But the "phrase" mode failure was quantified as **vocabulary discriminability**: 73% (292/400) of the v400 words appear in >30% of frames, df median 50.6%, and the tension *"多帧稳定 ⇒ 词必须常见 ⇒ 不判别；判别 ⇒ 词必须稀有 ⇒ 不稳定"* (`Docs/archive/世界模型导航方案-可行性评估.md:1825-1828`). Both failures are properties of **hand-built / BoW tokens**. A dense language-aligned patch feature from a frozen foundation encoder removes all three: there is no grid index to misalign (features are interpolated at unprojected 3D points), no per-frame mean normalisation, and no fixed vocabulary — the query is free-form text, not a 400-word list. This is *precisely* the R2F/RayFronts mechanism, which is why it is the recommended template.

---

## (c) Failure modes in a stylized VRChat scene

### C1. Mirrors — **NOT FOUND in any paper read; no literature solution**
Mechanism: SGBM returns the **mirror plane's** disparity (a real surface at real depth), so occupancy correctly gets a wall — but the **semantic features painted on that wall are the reflection's content**. A landmark that exists only as a reflection is placed at the mirror's metric depth, in the wrong direction. Worse, the OOR/beyond-range ray path (B4) casts rays *through* the mirror, so the frontier behind it accumulates reflected semantics and the agent searches for a door that exists only in a reflection. Simultaneously, a mirror creates two views of the same object (direct + reflected), which is the exact **feature-collision** condition RayFronts §III-E was built to avoid — and its fix (multiple rays per frontier) does not help, because both rays carry the same feature.
Mitigations available to us: (i) mirrors in VRChat are usually a discrete world component; (ii) detect the agent's **own avatar** in the reflection (the project already runs a person detector + OSNet ReID) and suppress features in that screen region; (iii) flag mirror-like cells by their abnormally high semantic self-similarity to the directly-observed scene. Treat all three as hypotheses to test — there is no published method to copy.

### C2. UI overlays — high risk, and a plausible *second* cause of the zero-discrimination result
Mechanism: RayFronts/R2F assume the whole image is world content. VRChat menus, nameplates, HUD and the desktop window are **composited after stereo rendering**, so UI pixels carry **no valid disparity** — they inherit whatever world surface happens to be behind them. They are then unprojected, and their dense features ("button", "text", user names) get bound to arbitrary world cells and frontier regions. OCR (the `det-ocr` workstream) would faithfully read these as landmark labels.
**The load-bearing property: screen-fixed content is invariant across frames.** Any channel dominated by screen-fixed content has near-zero across-time variance and therefore cannot discriminate places. This is a mechanism *independent* of the two already documented in-repo, and it would produce exactly the observed zero discriminative power. I flag it as a **hypothesis to test, not a claim** — I could not confirm from the repo whether UI was present in the probe frames.
Mitigations: (i) mask UI regions (the OCR/person-detector boxes *are* the UI boxes) before unprojection and encoding; (ii) exploit the invariance directly — compute per-pixel feature variance over a sliding window and suppress minimum-variance (screen-fixed) content. This is cheap, training-free, and directly testable.
**NOT FOUND:** neither RayFronts nor R2F handles overlays.

### C3. Avatars — partially solvable, and we are better equipped than the papers
Mechanism: (i) a moving avatar in front of a landmark writes its features into the *same* frontier region, and since `S_m = max_b f_{m,b}ᵀt_q` takes a max over direction bins, a single contaminated bin can win → false positive; (ii) avatar geometry writes occupancy → phantom obstacles; (iii) VRChat avatar shaders (toon flat shading, outlines, transparency, animated normals) break SGBM outright, producing dropouts or a wrong surface.
What the papers offer is weak: RayFronts' only dynamic mechanism is clamping log-odds to tolerate dynamics and **pruning semantic voxels with occupancy < 0.5** (§III-C); R2F has **no** explicit dynamic-object handling. **This is where the project has an advantage the papers lack: it already runs a person detector + OSNet ReID.** Mask person boxes before encoding/unprojection and route persons exclusively into the existing person-memory channel, never the landmark channel. That single change removes the most common contamination source and is strictly better than anything RayFronts or R2F does.

### C4. Toon/flat shading → SGBM collapse (already observed in-repo)
Flat-shaded, texture-poor surfaces yield no disparity, so occupancy is sparse exactly where walls should be. The repo has already logged *"A large number (397/554) of stereo correspondences are rejected"* (`Docs/archive/双目序列run5-7结论汇总（2026-09-27）.md:130`, flagged unresolved at `Docs/自动到达能力差距清单.md:53`). Constraint-compatible mitigation: `models/depth_anything_v2_small` (already present) may be used **only as a non-metric shape/gradient prior** for obstacle inflation and gap-filling, never as a metre source — matching the "soft depth … traversability only" position in the existing doc. Note RayFronts *does* sanction monocular depth as its `D_t` (§III-E), so the paper cannot be cited as licence to break the project's own metric rule.

### C5. Stylized-domain feature collapse (the biggest unverified risk)
RADIO/SIGLIP are trained on natural images; anime/toon rendering with UI is out of distribution. In-domain, R2F's informative similarity band is only **[−0.10, 0.15]** (≈0.25 wide, Fig. 3), and its operating thresholds sit at its edge (`τ_g=0.14`, `τ_ℓ=0.11`). A domain shift that compresses or shifts the band makes those thresholds arbitrary. **NOT FOUND:** no evaluation of RADIO/SigLIP-style dense open-vocabulary features on stylized/anime game rendering exists in any paper I read. This is why B5's in-domain calibration is non-optional, and why the existing doc's `where_am_i()` design — returning ranked candidates plus a `confidence`/`margin` rather than a bare id — remains the right interface.

### C6. Two more, brief
- **Per-eye rendering mismatch (TAA/post-processing):** open and suspected in-repo (`Docs/业界世界模型方案落地评估（2026-09-29）.md:215`). Photometric inconsistency between eyes degrades SGBM directly; no semantic method fixes it.
- **Transparency / alpha** (windows, holograms, water, glass) and **skybox/infinite background:** SGBM invalid or disparity→0/random. The `r_max` OOR mask (B4 step 1) correctly quarantines skybox, but transparent surfaces remain a semantic-vs-geometric contradiction (semantics says "window", geometry has no surface).

---

## NOT FOUND register (do not let these be invented downstream)

| Item | Status |
|---|---|
| RayFronts VRAM / GPU-memory figure | **NOT FOUND** (only "highest memory consumption" of the baselines) |
| RayFronts desktop-GPU numbers | **NOT FOUND** (all timings are Jetson AGX Orin) |
| Hydra GPU/VRAM figures | **NOT FOUND** (CPU-real-time; only Xavier NX ms figures) |
| ConceptGraphs runtime / FPS / any real-time claim for itself | **NOT FOUND** |
| ConceptGraphs 2D-only or stereo-only variant | **NOT FOUND** |
| LM-Nav compute, hardware, or runtime | **NOT FOUND** (LLM/VLM queries pre-computed on a remote workstation) |
| R2F CPU-only numbers; R2F simulated FPS | **NOT FOUND** (only seconds/episode) |
| VOIM code release | **NOT FOUND** |
| Any paper handling mirrors, UI overlays, or screen-fixed content in semantic mapping | **NOT FOUND** |
| Any evaluation of dense open-vocab features (RADIO/SigLIP/CLIP) on stylized/anime game rendering | **NOT FOUND** |
| Ov3R efficiency-table column mapping (Peak GPU 6222/11132/22572/10968/22572 MB, FPS 2.5/15/15/1.9) | **PARTIAL** — values extracted from PDF text but the header-to-column alignment is unverified; the per-dataset FPS (Ov3R 15, SLAM3R 24, Spann3R >50 on Replica) **is** verified from the HTML Table 1 |

## Repos verified this session

RayFronts `github.com/RayFronts/RayFronts` · R2F `github.com/Lab-RoCoCo-Sapienza/r2f` · Hydra `github.com/MIT-SPARK/Hydra` · ConceptGraphs `github.com/concept-graphs/concept-graphs` · LM-Nav `sites.google.com/view/lmnav` · OVI-MAP `github.com/OVI-MAP/OVI-MAP` · Ov3R `github.com/ZoranGong/Ov3R` · OVO-SLAM `github.com/tberriel/OVO`
