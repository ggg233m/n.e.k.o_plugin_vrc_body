# Embodied Navigation Literature Review for a VRChat LLM Agent (late 2026)

**Scope.** Research for a Windows VRChat plugin that gives an LLM agent embodied navigation
ability inside VRChat. Sensors: egocentric frames from a window capture of VRChat
(anime/stylized rendering, menus, UI overlays), VRChat OSC telemetry (avatar velocity in m/s
plus a few built-in params), SteamVR HMD/controller poses. **No depth sensor, no VRChat world
coordinates** — only a self-built map frame. Existing modules: person detector, OSNet
person-ReID embedder, traversability prediction (divergence risk per sector), and LLM tools.
Missing: map tools for the LLM. Failed: a semantic landmark channel (grid tokens had zero
discriminative power).

## Method and confidence statement

**`web_fetch` was completely unavailable in this session.** Every hostname (`arxiv.org`,
`github.com`, `example.com`, `en.wikipedia.org`) returned *"resolves to a non-public IP
address"*. **No page was ever opened.** Every fact below comes from search-result titles and
snippets only. Therefore:

- Titles, arXiv IDs, venue tags and repo names that appear **verbatim in a snippet** are
  reported as confirmed.
- Anything a snippet did not state is explicitly marked **unverified**.
- **No numeric benchmark number is reported unless a snippet literally contained it.**

Search backends that worked: `keenable`, `exa`, `tavily` (until hourly cap), `bing`, `ddg`,
`firecrawl`, `parallel`. Backends that failed: `anysearch` (HTTP 402 throughout), `searxng`
(all instances 429 / invalid JSON), `perplexity`/`serpbase`/`deepseek-official` (no key),
`reddit` platform search (HTTP 403/404). Tavily hit its keyless hourly cap mid-run.

---

## 1. Language/VLM-driven navigation without a metric map

### 1.1 The Berkeley "Visual Navigation Transformer" family — GNM, ViNT, NoMaD

| | GNM | ViNT | NoMaD |
|---|---|---|---|
| Paper | arXiv:2210.03370 | arXiv:2306.14846 | arXiv:2310.07896 |
| Date | Oct 2022 | Jun 2023 | Oct 2023 |
| Venue | **unverified** (widely cited) | **CoRL 2023** (confirmed on project page) | **ICRA 2024** (confirmed via UCLA-VAIL model zoo listing) |
| Language input? | No | No | No |

**What they give you:** a goal-conditioned *visual* navigation policy. Input is the current RGB
observation plus a **goal image** (or a distance-to-goal), output is a local action / waypoint
sequence. They do **not** take language and do **not** build or query a map. Come with official
code and checkpoints in `robodhruv/visualnav-transformer` (also mirrored as
`RobotiXX/visualnav-transformer` for ROS 2). **License unverified.**

**Training vs zero-shot:** *trained once on cross-embodiment robot data, then used zero-shot on
new robots and environments*; also designed for efficient fine-tuning/adaptation to new robots.
So: no training needed by you, but you cannot retrain it without a data-collection effort.

**Hardware — this is the good news.** The HuggingFace `UCLA-VAIL/Navigation-Model-Zoo-Public`
listing describes `NoMaD_GL` / "NoMaD · ICRA 2024 goal-free (diffusion)" as **4 × 96×96 input,
8×8 samples, 3× .onnx (+.data), 111 MB**. That is a genuinely small, ONNX-exportable model —
deployable on a consumer GPU, plausibly even CPU. **Caveat: 96×96 is a very low resolution**,
and these models were trained on real robot camera streams.

**Relevance to this project: medium.** They solve *local* goal-reaching, not "where is X" or
"how do I get back to the bar". Their real value here is as a **drop-in local controller** for a
subgoal (e.g. the traversability module's per-sector risk → a chosen bearing → a waypoint).
Note the parent already has its own traversability + motion stack, so this may be redundant.

### 1.2 VLFM — Vision-Language Frontier Maps

- **Paper:** arXiv:2312.03275, Yokoyama et al. (RAI Institute / Boston Dynamics lineage).
- **Venue:** **ICRA 2024** (confirmed in the official repo description). A snippet from Semantic
  Scholar mentioning "IEEE Transactions on Robotics 2026" appears in a garbled context alongside
  an unrelated "VLN-Game" TLDR — treat the T-RO extension as **unverified**.
- **Open source: YES** — `rai-opensource/vlfm` (also mirrored at `bdaiinstitute/vlfm`).
- **Training: none — explicitly zero-shot.** This is its headline claim.
- **What it gives you:** the single most relevant published architecture to this project's
  *problem shape*. It builds a **frontier map** (unexplored boundaries of the explored region)
  and scores each frontier by semantic value against a **text prompt** (e.g. "chair"), using
  **BLIP-2** to produce a value map. It then navigates to the highest-value frontier. So it is
  "semantic exploration guided by a VLM, without any task-specific training."
- **Sensor requirement — important caveat:** a frontier map requires **occupancy/depth**. A
  SemNav snippet states "VLFM [20] leverages BLIP-2…". VLFM was deployed on a Boston Dynamics
  Spot, which carries depth sensing. **I could not confirm from snippets whether VLFM can run on
  monocular RGB alone — assume it needs depth or an occupancy source** (unverified for the
  monocular case).
- **Why it matters anyway:** the *concept* — maintain an explored/occupied region, score
  unexplored boundaries against a language goal, go to the best one — is directly transferable
  to a monocular setup if you substitute monocular depth or a learned occupancy proxy.
- **Successor worth reading:** **SemNav** (arXiv:2506.03516) — model-based planner for zero-shot
  object-goal navigation using vision-foundation models.

### 1.3 VLM/VLA navigation models — NaVid, NaVILA, Uni-NaVid, StreamVLN

| Name | Paper | Venue | Open source | Needs training? |
|---|---|---|---|---|
| **NaVid** | arXiv:2402.15852 | **RSS 2024** (confirmed: `roboticsproceedings.org/rss20/p079.html`) | Eval code: `jzhzhang/NaVid-VLN-CE` | Yes — fine-tuned video-VLM |
| **NaVILA** | arXiv:2412.04453 | **RSS 2025** (confirmed: `github.com/AnjieCheng/NaVILA`, "[RSS'25]") | Yes — training code, weights, YouTube human-touring data all released per repo checklist | Yes |
| **Uni-NaVid** | arXiv:2412.06224 | **RSS 2025** (confirmed: `roboticsconference.org/2025/program/papers/13/`) | Project page at `pku-epic.github.io/Uni-NaVid` | Yes |
| **StreamVLN** | arXiv:2507.05240 | **ICRA 2026** (confirmed: `InternRobotics/StreamVLN`) | Yes + HF checkpoint `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln` | Yes |

**NaVid is the most important of these for this project**, because the official repo states
verbatim: *"Video-Language in, Actions out! **No odometry, Depth, or Map required!**"* That is
**exactly** this project's sensor constraint (plus language). The cost is that it is a
fine-tuned video-VLM, so it needs a GPU and it was fine-tuned on real/simulated indoor video.

**NaVILA** is a two-level VLA: a VLM (Qwen2.5-VL-based, per repo/search context) emits mid-level
*language* actions ("move forward 1 m"), and a separate RL locomotion policy executes them for a
legged robot. The locomotion half is irrelevant to VRChat; the **VLM-emits-language-actions**
half is a clean fit for a game avatar, whose locomotion you already control via OSC input.

### 1.4 Pure-LLM navigators — NavGPT and the 2026 "tool-calling harness" turn

- **NavGPT** — Zhou et al., **AAAI 2024** (confirmed: `ojs.aaai.org/index.php/AAAI/article/view/28597`).
  Official code `GengzeZhou/NavGPT`. **Purely LLM-based, zero-shot, no training.** A GPT-4 ReAct
  loop that explicitly reasons over a textual view of the scene. **Its constraint:** it operates
  on *discrete panorama nodes* in R2R — i.e. it needs a pre-built discretization of the world
  and turn-based actions.
- **AgenticNav** — arXiv:2606.10577 (Jul 2026): *"Zero-Shot Vision-and-Language Navigation as a
  **Tool-Calling Harness**"*. A 2026 abstract states the bottleneck "is no longer just the
  vision" — the framing is that an LLM orchestrating tools is the right architecture.
- **LG-VLN** — arXiv:2609.15098 (Sep 2026): zero-shot VLN framework with **LangGraph state
  orchestration**.
- **AnyGoal** — arXiv:2606.13878 (2026): vision-language guided multi-agent exploration for
  **training-free** lifelong navigation. Abstract explicitly criticises modular systems that
  "delegate to fragile detectors".
- **OpenFrontier** — arXiv:2603.05377, **RSS 2026**, code `cvg/OpenFrontier` (Marc Pollefeys
  group): detects and semantically evaluates **visual frontiers directly in the image** — i.e.
  the monocular-friendly reformulation of VLFM's frontier idea.
- **NavFoM** — arXiv:2509.12129, **ICLR 2026** (confirmed: `pku-epic.github.io/NavFoM-Web/`,
  papernotes ICLR2026 index): cross-embodiment, cross-task Navigation Foundation Model trained on
  **eight million navigation samples**. Code status: a paper-note snippet says "Open-sourced upon
  paper acceptance" — **whether it is actually released is unverified**.
- **"What Limits Vision-and-Language Navigation?"** — arXiv:2605.13328 (2026).

**Architectural verdict for this project:** the 2026 literature has *converged on exactly this
project's shape* — an LLM orchestrating tools over visual observations, zero-shot, no map
required (AgenticNav, LG-VLN, NaVGPT, OpenFrontier). The parent is not off-architecture. The
uncommon part is the *sensor set* (window capture + OSC + SteamVR poses, no depth, no world
coords).

---

## 2. Object-goal navigation via open-vocabulary detection and segmentation

### 2.1 Components

- **GroundingDINO** — IDEA Research, arXiv:2303.05499, **ECCV 2024** (venue widely stated;
  treat as high-confidence but not snippet-confirmed). A text-prompted **open-set detector**:
  you pass "a chair" and it returns boxes. Snippet confirms it is "an open-set (open-vocabulary)
  object detector: instead of pre-[defined classes]…". **License: believed Apache-2.0 —
  unverified.** Roboflow publishes a latency/benchmark page (`roboflow.com/model/grounding-dino`).
- **Successors:** Grounding DINO 1.5 / 1.6 and **DINO-X** are referenced in the ecosystem but I
  **could not confirm their license or availability from snippets — unverified**. MM-Grounding-DINO
  likewise unverified.
- **YOLO-World** — Tencent AI Lab, **CVPR 2024**. Real-time open-vocabulary detection.
  **YOLO-World v2 Small (YOLO-World-S-v2)** is confirmed as "the smallest variant of Tencent AI
  Lab's YOLO-World v2 family" via Roboflow. **License is the critical unknown for a shipping
  product** — Roboflow maintains a dedicated `roboflow.com/model-licenses/yolo-world` page,
  which strongly implies a non-permissive license in play (YOLO lineage is typically GPL/AGPL).
  **Verify the exact license before shipping — unverified here.**
- **SAM / SAM 2** — Meta. SAM 2 (arXiv:2408.00714) is the video-capable promptable segmenter.
  Confirmed as an active 2026 framework in a "Computer Vision Frameworks 2026" roundup.
  **License believed Apache-2.0 — unverified.** Lightweight variants (MobileSAM, FastSAM,
  EfficientSAM) surfaced only as ecosystem names, **details unverified**.
- **CLIP / OpenCLIP / SigLIP** — used as **region classifiers**: crop a detected box, embed it,
  compare to the text embedding of the goal. SigLIP/SigLIP 2 specifics: **unverified** from this
  session's searches.

### 2.2 The canonical target-finding pipeline

Assembled from the 2024-2026 object-goal-navigation literature (all of these are
**zero-shot / training-free** unless noted):

1. **Open-vocab detector** proposes boxes for a language goal (GroundingDINO / YOLO-World).
2. **CLIP-family region scoring** re-ranks the boxes against the goal text.
3. **Instance association across frames** — this is where **ReID** lives (the project's OSNet).
4. **Bearing/range to target** passed to a policy or the LLM.
5. **Frontier exploration** supplies the "where to look next" when the target is not visible.

Representative papers (all confirmed to exist via snippets):
- **Dense CLIP semantic mapping for open-vocab object-goal nav** — arXiv:2407.09016.
- **AECNav: Active Evidence Consolidation for Efficient Zero-Shot Open-Vocabulary Object
  Navigation** — arXiv:2608.10817 (Aug 2026).
- **Room-Mediated Co-occurrence for Zero-Shot Object-Centric Semantic Navigation via Frontier
  Scoring** — arXiv:2607.25448 (Jul 2026). Note the *room-co-occurrence prior* — i.e. "a kettle
  is likely in a kitchen" — which is a cheap, high-value LLM prior you can inject without any
  detector.
- **Context-Nav** — **CVPR 2026**, project page `autocompsyslab.github.io/ContextNav`; consumes
  RGB-D + odometry + free-form text.
- **SemNav** — arXiv:2506.03516 (2025), zero-shot object-goal nav from vision foundation models.
- **Uncertainty-Informed Active Perception for Open Vocabulary Object Goal Navigation** (2026) —
  described as "a training-free, open-vocabulary, uncertainty-informed active perception
  pipeline".
- **Open-Vocabulary Object-Goal Navigation by Generalizing Semantic Mapping with Dense CLIP** —
  arXiv:2407.09016.

**Pattern to note:** essentially all of these consume **RGB-D + odometry**. The ones that don't
(NaVid, AgenticNav) are VLM-policy approaches. **There is no widely-adopted training-free
object-goal pipeline that assumes monocular RGB with no metric map and no odometry** — this
project's constraint set remains unusual.

### 2.3 Anime / stylized / game-rendering robustness — the honesty section

**No study measures CLIP, GroundingDINO, SAM, YOLO, or OSNet on VRChat avatar renders
specifically. That is a genuine measurement gap.**

Indirect evidence that the risk is real:
- A snippet from **YOLOv14** (arXiv:2608.04720) states detectors "degrade sharply on non-ideal
  inputs — fisheye distortion, **game-rendered content**…". This is the closest thing to a direct
  statement that game rendering is a known detector stressor.
- **SADGE** (arXiv:2605.22467) is cited as quantifying a synthetic-vs-real gap.
- CLIP-based ReID domain gap is studied in **MDPI Sensors 25(2):363 (2025)**.
- **"Intent Recognition under Rendered Avatar Distortions"** — arXiv:2609.27560.
- "Limitations of Zero-Shot CLIP on Fine-Grained Cultural Art" — exact venue **unverified**.

**The single most useful positive finding for the OSNet problem:** Wang, Liao, Shao,
*"Surpassing Real-World Source Training Data: **Random 3D Characters** for Generalizable Person
Re-Identification"*, **ACM MM 2020**, DOI 10.1145/3394171.3413815. Random **3D characters as
source data generalised BETTER than real-world data** for person ReID. This is direct evidence
that ReID trained on synthetic 3D humans can beat real-data training on a stylized target — a
strong argument for fine-tuning OSNet on rendered-avatar data if the current one underperforms.

Adjacent anime/manga ReID literature exists and confirms the problem is taken seriously:
**DAF:RE** anime character recognition dataset (arXiv:2101.08674); *Occlusion-Aware **Manga**
Character Re-identification with Self-Paced Contrastive Learning* (ACM, DOI
10.1145/3595916.3626401); **Re:Cognize** open-set comic character ReID (arXiv:2609.34032).

Anime-specific tooling found: `deepghs/imgutils` (YOLOv8-based **AniDet3** anime whole-body person
detection, plus a classifier into `3d`/`bangumi`/`comic`/`illustration`/`not_painting` — the
"is this a 3D render" gate is a clever pre-filter); `nagadomi/lbpcascade_animeface` (faces only,
**no published 3D-render accuracy**); `OysterQAQ/DanbooruCLIP`; `v2ray/clipbooru`;
`pixai-labs/pixai-tagger-v0.9`; `deepghs/csip`.

---

## 3. Semantic 3D scene graphs for LLM querying

| System | Paper / venue | Open source | Sensor need | Compute |
|---|---|---|---|---|
| **ConceptGraphs** | arXiv:2309.16650, **ICRA 2024** (venue widely stated) | Yes, `concept-graphs/concept-graphs` | **RGB-D + poses** | Heavy: detector + SAM + CLIP per frame, then object merging |
| **Hydra / Hydra-Multi** | arXiv:2202.12498, **RSS 2022** | Yes (MIT-SPARK) | **RGB-D + IMU** | Real-time claim; unverified numbers |
| **Kimera / Kimera-Semantics** | **ICRA 2020** lineage | Yes | **Stereo or RGB-D** | VIO + volumetric mesh |
| **OpenScene** | **CVPR 2023** | Yes | **Posed RGB-D** | CLIP features lifted into 3D |
| **The Bare Necessities** | arXiv:2412.01539 | unverified | RGB-D | Explicitly argues simpler is better |
| **Seeing Fast and Slow: Bimodal 3D Scene Graphs** | arXiv:2605.31067 (2026) | unverified | unverified | 2026 successor |
| **FreeOcc** | arXiv:2604.28115, **RSS 2026** | Yes, `the-masses/FreeOcc` | **training-free, open-vocab occupancy** | 2026 |
| **SceneVerse / SpatialVLM / SpatialRGPT / VSI-Bench** | details **unverified** in this session | — | — | — |

**Context:** a ConceptGraphs snippet confirms the field's own complaint — *"3D open-vocabulary
scene graph methods are a promising map representation for embodied agents, however… building
scalable scene graphs in real-time remains a significant challenge"* (from *The Bare
Necessities*, arXiv:2412.01539). ConceptGraphs' repo has a refactored **`ali-dev` branch offering
"a real-time, streamlined re[lease]"** per an Ecosystems snippet — indicating real-time was a
later retrofit, not the original design.

**Monocular metric depth (the enabling substitute for a missing depth sensor):**
- **Depth Anything V2** — a snippet from `damionrashford/media-os` states Depth-Anything v2 is
  **Apache 2.0**. Variants ViT-S/B/L; weights as `.pth` (snippet from `rizwanai.com`). The ViT-S
  variant is the real-time candidate; **exact FPS not confirmed from a snippet**.
- **Video Depth Anything** — **CVPR 2025 Highlight** (confirmed: `reka-ai/Video-Depth-Anything`),
  built on Depth Anything V2, temporally consistent for arbitrarily long video. A codehub
  release-note snippet dated **2025-08-28** records: *"Metric depth models released"* with a
  comparison table listing **MoGe-2-L, UniDepthV2-L, DepthPro, VDA-S-Metric**. So metric
  monocular depth is available as of Aug 2025.
- Others named: **MoGe-2**, **UniDepth v2**, **Apple Depth Pro**. Apple Depth Pro license:
  **unverified** (historically a research-only license).

**Feasibility verdict (my inference, not a sourced claim):** a full ConceptGraphs-class pipeline
(per-frame detector + SAM + CLIP + object merging + depth fusion) is **not real-time on a
consumer GPU** and is designed around RGB-D. A **monocular** substitute (Depth Anything V2
ViT-S → back-project keyframes → fuse CLIP or DINOv2 features into a sparse 3D landmark set →
cluster into objects) is plausible but:
1. monocular metric scale is ambiguous and drifts, and this project already lacks world coords;
2. `FreeOcc` (RSS 2026) is the best published starting point precisely because it is
   **training-free open-vocabulary occupancy without a depth sensor**;
3. *RADIO-ViPE* (2026) is described as "online tightly coupled multi-modal fusion for
   open-vocab[ulary]…" — worth investigating further (**details unverified**).

---

## 4. Episodic / spatial memory for embodied LLM agents

### 4.1 The canonical precedent — LM-Nav

**LM-Nav** — Shah, Osiński, et al., arXiv:2207.04429, **CoRL 2022** (venue widely stated).
Confirmed components: **GPT-3** parses a natural-language instruction into a sequence of
**landmark** phrases; **CLIP** grounds each landmark onto nodes of a previously built
**topological graph**; graph search yields the route; a learned low-level policy (ViNG) drives
between waypoints. A third-party implementation exists (`blazejosinski/lm_nav`); the original
repo name is **unverified**.

This is the **closest published analogue to what this project needs**: it is exactly
"LLM + semantic graph + language-queried route", with **no metric map required** — the graph can
come from appearance-based place recognition.

### 4.2 Place recognition: the fix for the failed landmark channel

The project's semantic landmark channel failed because **grid tokens had zero discriminative
power**. The literature's answer to "what makes a good place descriptor" is well established:

- **AnyLoc** — Keetha, Mishra, Karhade, Jatavallabhula et al., **RA-L 2023 / ICRA 2024**
  (confirmed: `anyloc.github.io`). **Training-free "universal" visual place recognition**: take
  **DINOv2** foundation-model features and aggregate them **unsupervised** (VLAD / GeM). The
  paper's own comparison table lists `AnyLoc-GeM-DINOv2`, `MixVPR`, `NetVLAD`,
  `AnyLoc-VLAD-DINOv2`. **This is the single strongest candidate to replace grid tokens** — it is
  zero-shot, needs no domain training, and relies on *appearance* rather than *semantic labels*.
- **SALAD** — Izquierdo & Civera, **CVPR 2024**. Strong learned place descriptor, but
  **requires training** on domain data.
- **MixVPR**, **NetVLAD** — learned, require training.
- **MegaLoc** — named in a 2026 foundation-model-VPR roundup; **details unverified**.

**Important epistemic point (my inference):** stylization destroys *semantic* discriminability
(a rendered wall is still "a wall" everywhere) far more than it destroys *geometric and
appearance* structure (which particular wall, which particular texture). This is consistent with
the project's own observed failure — grid-token *semantics* had zero discriminative power. So
the recommendation is to move place identity onto **framework-feature + local-geometry**
descriptors (AnyLoc/DINOv2-VLAD, DINOv2 patch matching, BoW on ORB, SeqSLAM) rather than onto
CLIP text semantics.

This is corroborated by the parent workspace's own existing probes:
`research/tools/seqslam_probe.py`, `bow_loop_eval.py`, `orb_place_probe.py`,
`location_distinguishability_probe.py`, `loop_place_vote_eval.py`,
`regression_place_identity.py`, `world_region_similarity.py` — i.e. appearance/geometric place
identity is already the direction being probed.

### 4.3 2026 memory systems (all confirmed to exist)

- **HAM-VLN** — arXiv:2607.29600 (Jul 2026): *Harnessing **Hierarchical Agentic Memory** for
  Zero-Shot VLN*. "Brings agentic memory into the robot's navigation loop… keeps observations
  from the most recent waypoints…" — a two-tier (short-term waypoint buffer + long-term)
  memory, zero-shot.
- **eMEM** — arXiv:2606.03374 (Jun 2026): *"a hybrid **graph-based memory system** for embodied
  agents operating in physical environments"* — spatio-temporal graph memory.
- **VTM-Nav** — arXiv:2607.14514 (Jul 2026): *Hierarchical **Visual-Topological Memory** for
  Cross-Episode Object-Goal Navigation*.
- **Embodied-RAG** — "a retriev[al]…" hierarchical memory for embodied agents (2026 roundup);
  **details unverified**.
- **NaviAgent** — arXiv:2506.19500: bilevel planning on a **tool navigation graph** for
  large-scale orchestration — relevant if the LLM tool set grows large.
- **Embodied Task Planning via Graph-Informed Action Generation with LLM** — arXiv:2601.21841.

### 4.4 Recommended memory representation and LLM tool surface

**Recommendation (my design inference, grounded in the above):**

Representation: a **topological graph**, `node = place`, `edge = traversable adjacency with
estimated relative transform`. Each node stores: (a) an **AnyLoc / DINOv2-VLAD global
descriptor** as its identity key; (b) 1-3 representative keyframes; (c) an optional short
**LLM-generated text label** produced *asynchronously* (never in the control loop); (d) an
optional list of observed object tags with the CLIP confidence that produced them.

Deliberately **do not** make semantics the *identity* of a node — make it an *annotation*. That
is the direct lesson of the grid-token failure and the reason AnyLoc is the right primitive.

Proposed LLM tools (5):
1. `where_am_i()` → current node id + confidence + descriptor distance to best match.
2. `list_places()` → node ids with their text labels and visit counts (text-only, cheap).
3. `describe_place(node_id)` → labels + linked keyframe image(s) + object tags.
4. `route_to(node_id)` → adjacency path as a list of bearings/"go to node N", returned as a
   *plan*, not motor commands.
5. `mark_place(label)` → attaches/overwrites the human/LLM-facing label on the current node.
6. *(optional)* `recall_object("couch")` → nodes whose object tags match, ranked by confidence —
   this is the object-goal bridge and it needs no 3D scene graph, only per-node tag lists.

Why this shape: it keeps the LLM in a **symbolic, text-cheap** loop (GraphRAG-style over a tiny
graph), never asks the LLM to reason in pixels about place identity, and matches the
LM-Nav/HAM-VLN/eMEM pattern of *graph + LLM query*. It also degrades gracefully: if labels are
wrong or absent, `route_to` and `where_am_i` still work off descriptors alone.

---

## 5. Sim-to-real and domain-shift warnings — is zero-shot transfer to VRChat plausible?

### 5.1 The documented real-world gap

An unusually strong 2025-2026 cluster of papers exists precisely because the field got burned:

- **"A Comprehensive Survey and Systematic Real-World Evaluation of Embodied Vision-and-Language
  Navigation"** — arXiv:2607.09792 (Jul 2026). A snippet states: *"Experiments across ten diverse
  real-world scenes show a **substantial performance gap between simulation and real-world
  de[ployment]**."*
- **"Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical
  and Visual Disparities"** — arXiv:2507.13019 (2025): idealised assumptions "about robot movement
  and control fail to reflect physically emb[odied]…" reality.
- **"What Limits Vision-and-Language Navigation?"** — arXiv:2605.13328 (2026).
- **NavTrust** — arXiv:2603.19229 (2026): *"Benchmarking Trustworthiness for Embodied
  Navigation"*, covering "spatial corruptions that arise in real-world settings".
- **"Can Vision Foundation Models Navigate? Zero-Shot Real-World Evaluation and Lessons
  Learned"** — Guerrier, Soma, et al., arXiv:2603.25937 (Mar 2026). Evaluates **five**
  state-of-the-art visual navigation models — **GNM, ViNT, NoMaD, NaviBridger, CrossFormer** —
  **zero-shot in the real world**. Companion repo `MaevaGuerrier/vnm-zeroshot-eval` confirms the
  model list and notes each model's backbone/key feature/output. **This is the paper to read
  first if you care about deploying ViNT/NoMaD/GNM.** Note it also surfaces two models I had not
  otherwise seen — **NaviBridger** and **CrossFormer** — worth chasing.
- **"Deploying Foundation Models for Embodied Navigation"** — Dorbala & Manocha, University of
  Maryland, arXiv:2609.25666 (Sep 2026). Directly about the deployment problems.
- **"Scaffolding Foundation Models into Physical-World Agents Pushes the Frontier of Long-Horizon
  Navigation"** — arXiv:2608.30396 (2026).
- **vnm-zeroshot-eval** also referenced *"Synthetic vs. Real Training Data for Visual
  Navigation"* (FAINT architecture) — directly on point for synthetic-domain training.

### 5.2 Is VRChat stylized rendering covered?

**No.** I found **no paper** measuring any of these navigation models, CLIP, GroundingDINO, SAM,
YOLO, or OSNet on VRChat or on a comparable real-time stylized first-person render.

The honest position:

1. **Direction of risk is documented.** Game-rendered content is explicitly named as a condition
   where detectors "degrade sharply" (YOLOv14, arXiv:2608.04720). Synthetic-vs-real gaps are
   quantified elsewhere (SADGE, arXiv:2605.22467).
2. **Zero-shot transfer is *plausible but unvalidated*.** VRChat differs from the training
   domain on several axes at once: non-photorealistic shading and textures, arbitrary user-made
   worlds with no physical plausibility constraints, extreme avatar proportions, anime faces, and
   a persistent 2D UI overlay (menus, nameplates, HUD) that occupies a meaningful fraction of
   frames and is pure out-of-distribution input.
3. **Asymmetry argument (my inference).** Cues that depend on *optical geometry* — optical flow,
   feature matching, loop closure, structure from motion, relative bearing — should transfer
   far better than cues that depend on *photometric realism or semantic priors*. Renderers still
   obey projective geometry and consistent texture mapping. This predicts: the traversability /
   geometry stack should transfer; the open-vocabulary semantic stack is the fragile half.
   This is consistent with the project's own empirical result that grid-token semantics had zero
   discriminative power.
4. **The UI-overlay axis is under-researched everywhere.** A 2D overlay is a *visual* corruption
   with no analogue in any navigation benchmark. There is adjacent work — **GUI grounding**
   (GUI-Actor arXiv:2506.03143; Qwen-GUI-3B arXiv:2506.23491; GUI-Lens arXiv:2608.03270) and
   **"Naive Visual Memory is Not Enough: A Failure-Mode Study of GUI Agents"**
   (arXiv:2606.14106) — but that literature treats the UI as *the task*, not as *noise on a
   navigation observation*. **Practical implication: detect and mask the UI overlay (or crop to
   the 3D viewport) rather than hoping the model ignores it.** That is cheap and likely
   high-leverage.
5. **Do not trust any zero-shot claim for VRChat until measured.** The correct move is to build a
   small in-domain evaluation set from recorded VRChat sessions (which the workspace already has
   recording tooling for: `research/recorder/`, `research/tools/recording_channels…`) and measure
   each foundation model on it before betting architecture on it.

---

## 6. Embodied AI inside VRChat specifically

**Verdict: no peer-reviewed paper does embodied navigation inside VRChat from egocentric pixels
with no world-coordinate access. That exact combination is unpublished.** But the space is not
empty. Four buckets:

### 6.1 Published LLM agents inside VRChat — conversational, not navigating
- **ELLMA-T** — *ACM DIS 2025*, arXiv:2410.02406, code `HeyMengxu/ELLMA-T`. An embodied English-
  tutor agent in VRChat.
- **Building LLM-based AI Agents in Social Virtual Reality** — *ACM CHI EA 2024*, DOI
  10.1145/3613905.3651026 — GPT-4 NPC deployed in VRChat.
- A **CHI 2026** ECA paper, DOI 10.1145/3772318.3791068.
- A **CHI 2026** LLM "sighted guide" for blind/low-vision social VR users, arXiv:2603.09964.

### 6.2 The one navigation agent in a commercial metaverse — but not VRChat
**Navigation Pixie** — *IEEE ISMAR 2025*, arXiv:2508.03216 (Cluster Metaverse Lab with U. Tsukuba
/ U. Tokyo). Implemented on **`cluster`, NOT VRChat**, though its introduction names
VRChat/Resonite/Cluster as the target platform family. It **deliberately avoids the hard
problem** by consuming **"structured spatial metadata"** plus an LLM, explicitly "minimizing
platform dependencies". No repo found → **closed source**. This is the paper to cite and to
position against.

### 6.3 The architecturally closest published analogue
**NavAI** — arXiv:2601.03251 (Jan 2026); journal version in *Automated Software Engineering*,
DOI 10.1007/s10515-026-00676-z (Aug 2026). An "application-agnostic LLM framework for navigation
in VR" that "interprets the virtual world using **screenshots captured from the VR
application**". **Which VR applications it was evaluated on is unverified** from snippets.

### 6.4 Unpublished open-source projects that already do much of this — the real novelty threat
- `gamio-22/vrchat-ai-agent` — ASR + local LLM + YOLO/Pose + **VLM** + OSC + autonomous avatar
  movement/gaze/expression.
- `HoppouAI/ProjectGabriel-Remastered` (2026) — Gemini Live + **YOLO person/face tracking** + OSC
  control; described as "Gabriel walks around".
- `Pomelo32141/VRchat_api_agent` — screen/audio observation + low-frequency LLM planning + OSC/
  input execution + an instinct loop.
- `MLShukai/vrcpilot` — capture / OCR / template detection / input synthesis.
- `MLShukai/pamiq-vrchat` — RL.

**Defensible novelty is therefore the specific constraint set** (window capture + OSC velocity +
SteamVR poses, no depth, no world coordinates, plus person detector / OSNet ReID /
traversability / LLM tools) — **not** "an LLM agent navigates a metaverse".

### 6.5 SIMA / SIMA 2 — the closest general precedent, but closed
- **SIMA 1** — *Scaling Instructable Agents Across Many Simulated Worlds*, arXiv:2404.10179
  (Mar 2024). **Nine commercial games** (No Man's Sky, Teardown, Valheim, Space Engineers, …),
  pixels → keyboard/mouse. **No peer-reviewed venue found → treat as a technical report.**
- **SIMA 2** — arXiv:2512.04797 (Dec 2025), announced **13 Nov 2025**, Gemini-based (Flash-Lite
  backbone + Gemini Pro steering, per secondary sources). Also a technical report.
- **Both are closed: no weights, no code, no API.** The only public code is `kyegomez/SIMA`, an
  **unofficial reimplementation**. Use as framing/citation, **not as a baseline**.

### 6.6 Simulators others actually use — and why none reproduces this constraint
Habitat 3.0 (arXiv:2310.13724, **ICLR 2024**, `facebookresearch/habitat-lab`; robots + humanoid
avatars + human-in-the-loop — closest *task* analogue, opposite *observability*, since it grants
ground-truth state); AI2-THOR (arXiv:1712.05474); ProcTHOR (**NeurIPS 2022**); iGibson 2.0
(**CoRL 2021**); ThreeDWorld (arXiv:2007.04954); BEHAVIOR-1K (arXiv:2403.09227); MineDojo
(**NeurIPS 2022**, arXiv:2206.08853); Voyager (**TMLR**, arXiv:2305.16291); SAPIEN specifics
**unverified**. **Every one grants privileged ground-truth state.**

**Game-engine navigation precedents:** GTA V for robotics/navigation synthetic data
(arXiv:2502.12303); Project Malmo (**IJCAI 2016**); ViZDoom (arXiv:1809.03470); Dreamer 4
(arXiv:2509.24527, trains inside a world model in Minecraft); REGEN (arXiv:2508.17061); a
sim2real game-engine appearance-gap paper (arXiv:2605.02291). **None is "third-party plugin
inside someone else's live commercial social world."**

### 6.7 The OSC telemetry answer — most actionable operational finding

**Avatar world position is NOT obtainable via OSC.** This is not inference; it is the still-open
state of the official channels:

- VRChat official feedback request (11 Nov 2022, still a *request*, not a feature):
  *"send the absolute position and rotation in the world… **It is technically possible to
  calculate it using the velocity, but this doesn't account for teleporting/respawning.**"*
  → https://feedback.vrchat.com/feature-requests/p/additional-parameters-for-reading-current-players-position
  **This single sentence validates the velocity-integration workaround AND names its failure mode
  (teleport/respawn).**
- Open GitHub issues since 2022: `vrchat-community/osc#43` ("Expose the player's own position via
  Avatar OSC parameters") and `#147` ("Native Avatar Absolute X Y Z and rotation Parameters").
- Proof that world-object poses do not export: OSC discussion #154 — sending a cube's
  position/rotation over OSC works **in the Unity Editor** but emits **zero output in a built,
  published world**.
- **`/tracking/trackers/{1..8}/position|rotation` is input *TO* VRChat, not an avatar-pose
  output.** Official docs state *"You must create your own program to transmit this data to
  VRChat"* (`docs.vrchat.com/docs/osc-trackers`). **Do not design around reading it back.**
- Built-in readable params: `VelocityX/Y/Z/VelocityMagnitude` (Float, m/s), `Grounded`,
  `Upright`, `Seated`, `AFK`, `AngularY` (caps ±1024), `ScaleFactor`.
- **CRITICAL CAVEAT:** VRChat Wiki / VRC School state that these velocity params are
  avatar-**local** axes and that *"Locally, playspace movement does not count; remotely, it
  does."* → **naive integration under-counts real-world (HMD) walking when running locally.**
  Avatar world scale also changes the velocity→metres mapping.
- **NEW (2025.3.3+, Nov 2025): VRChat added OSC User-Camera / Camera-Dolly endpoints, read/write,
  and they expose WORLD-SPACE position.** A complaint thread confirms they only offer world-space.
  **This is the only recent feature handing external software a world coordinate.** Whether the
  camera can be anchored to the avatar is **unverified — prototype before relying on it.**
- Only true world-coords path = be the world author and use **Udon** (`GetPosition`/`GetVelocity`,
  Vector3 World Space) — unavailable in others' worlds.
- Movement control: `/input/Vertical|Horizontal|LookHorizontal` (float −1..1), buttons int 1/0;
  **must reset axes to 0** or the avatar moves forever.
- Reusable tooling: `kushiemoon-dev/vrchat-osc-bridge` (`POST /move {vertical,horizontal,look,
  duration}`), `ZenithVal/OSCLeash` (OSC-in → movement-out closed loop via PhysBone),
  `sandraschi/vrchat-mcp`, `Duinrahaic/VRCDollyManager`, `theepicsnail/vrchat_oscquery`,
  `Lioncat6/OSC-Chat-Tools` (unofficial docs). **Udon AI Navigation** gives NavMesh pathfinding
  **only if you author the world**.

---

## 7. Consolidated recommendations

1. **Adopt the 2026 consensus architecture.** An LLM orchestrating tools over visual
   observations, zero-shot, no metric map is what AgenticNav (arXiv:2606.10577), LG-VLN
   (arXiv:2609.15098), NavGPT (AAAI 2024) and OpenFrontier (RSS 2026) all converge on. Do not
   invest in training a VLA.
2. **Fix place identity with appearance, not semantics.** Replace the dead grid-token landmark
   channel with **AnyLoc (DINOv2 + unsupervised VLAD, RA-L 2023/ICRA 2024)** global descriptors,
   and keep LLM text labels as *annotations on top*, never as node identity. This is the single
   highest-value change and it matches the parent's existing `seqslam_probe` / `bow_loop_eval` /
   `orb_place_probe` direction.
3. **Expose exactly the 5-6 graph tools in §4.4** (`where_am_i`, `list_places`,
   `describe_place`, `route_to`, `mark_place`, optional `recall_object`). Keep the LLM out of the
   pixel loop.
4. **Do not attempt a ConceptGraphs-class 3D scene graph.** It requires RGB-D, is not real-time on
   consumer hardware by the field's own admission (arXiv:2412.01539), and this project has no
   depth. If a 3D map is wanted, start from **FreeOcc** (RSS 2026, training-free open-vocab
   occupancy, no depth sensor) and a **Depth Anything V2 ViT-S** metric path.
5. **Assume the semantic stack will not transfer; measure before betting.** Record an in-domain
   VRChat evaluation set and test CLIP/GroundingDINO/OSNet on it. Budget for **fine-tuning OSNet
   on rendered avatars** — Wang et al., ACM MM 2020 (DOI 10.1145/3394171.3413815) shows random
   3D characters can *beat* real-data source training for ReID.
6. **Mask the UI overlay.** Menus/HUD/nameplates are a purely out-of-distribution visual axis
   with no benchmark coverage; cropping to the 3D viewport is cheap and likely high-leverage.
7. **Use the Nov 2025 OSC camera/dolly world-space endpoints if you need a world anchor**, and
   treat the velocity-integration path as broken under teleport/respawn — exactly as VRChat's own
   feedback thread warns.
8. **Read first, in order:** (a) can-VFMs-navigate zero-shot real-world eval (arXiv:2603.25937)
   for ViNT/NoMaD/GNM deployment reality; (b) LM-Nav (arXiv:2207.04429) for graph+LLM;
   (c) AnyLoc (arXiv:2306/2308.00688) for place descriptors; (d) Navigation Pixie
   (arXiv:2508.03216) and NavAI (arXiv:2601.03251) as the prior art to differentiate from.

---

## Appendix: URL list

Navigation foundation models / VLFM
- https://arxiv.org/abs/2312.03275 (VLFM, ICRA 2024)
- https://github.com/rai-opensource/vlfm
- https://general-navigation-models.github.io/ (GNM/ViNT/NoMaD hub)
- https://github.com/robodhruv/visualnav-transformer
- https://arxiv.org/abs/2210.03370 (GNM) · https://arxiv.org/abs/2306.14846 (ViNT) ·
  https://arxiv.org/abs/2310.07896 (NoMaD)
- https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public
- https://arxiv.org/pdf/2603.25937.pdf (Can VFMs Navigate? zero-shot real-world eval)
- https://github.com/MaevaGuerrier/vnm-zeroshot-eval

VLM/VLA navigation
- https://arxiv.org/abs/2402.15852 (NaVid, RSS 2024) · https://github.com/jzhzhang/NaVid-VLN-CE
- https://arxiv.org/abs/2412.04453 (NaVILA, RSS 2025) · https://github.com/AnjieCheng/NaVILA
- https://arxiv.org/html/2412.06224v2 (Uni-NaVid, RSS 2025)
- https://arxiv.org/pdf/2507.05240v2 (StreamVLN, ICRA 2026) · https://github.com/InternRobotics/StreamVLN
- https://ojs.aaai.org/index.php/AAAI/article/view/28597 (NavGPT, AAAI 2024) ·
  https://github.com/GengzeZhou/NavGPT
- https://arxiv.org/pdf/2606.10577v1 (AgenticNav, 2026)
- https://arxiv.org/pdf/2609.15098 (LG-VLN, 2026)
- https://arxiv.org/html/2606.13878 (AnyGoal, 2026)
- https://arxiv.org/pdf/2603.05377v3 (OpenFrontier, RSS 2026) · https://github.com/cvg/OpenFrontier
- https://arxiv.org/pdf/2509.12129v1 (NavFoM, ICLR 2026) · https://pku-epic.github.io/NavFoM-Web/
- https://arxiv.org/html/2605.13328v1 (What Limits VLN?)

Open-vocabulary detection / segmentation / object-goal nav
- https://roboflow.com/model/grounding-dino · https://roboflow.com/model-licenses/yolo-world
- https://arxiv.org/pdf/2407.09016v2 (Dense CLIP semantic mapping)
- https://arxiv.org/pdf/2608.10817 (AECNav, 2026)
- https://arxiv.org/pdf/2607.25448v1 (Room-Mediated Co-occurrence frontier scoring, 2026)
- https://autocompsyslab.github.io/ContextNav (Context-Nav, CVPR 2026)
- https://arxiv.org/html/2506.03516 (SemNav, 2025)
- https://arxiv.org/pdf/2608.04720 (YOLOv14 — "game-rendered content" degradation)
- https://doi.org/10.1145/3394171.3413815 (Random 3D Characters for generalizable ReID, ACM MM 2020)
- https://arxiv.org/html/2101.08674v1 (DAF:RE anime character dataset)
- https://dl.acm.org/doi/fullHtml/10.1145/3595916.3626401 (Manga character ReID)
- https://arxiv.org/pdf/2609.34032 (Re:Cognize open-set comic character ReID)
- https://github.com/DepthAnything/Video-Depth-Anything (Video Depth Anything, CVPR 2025 Highlight)

Scene graphs / spatial reasoning
- https://ar5iv.labs.arxiv.org/html/2309.16650 (ConceptGraphs) ·
  https://github.com/concept-graphs/concept-graphs
- https://arxiv.org/html/2412.01539v1 (The Bare Necessities, open-vocab scene graphs)
- https://arxiv.org/html/2605.31067 (Seeing Fast and Slow: Bimodal 3D Scene Graphs, 2026)
- https://arxiv.org/pdf/2604.28115.pdf (FreeOcc, RSS 2026) · https://github.com/the-masses/FreeOcc
- https://arxiv.org/pdf/2603.06166v1 (FreeOcc panoptic, 2026)

Memory / place recognition
- https://arxiv.org/html/2207.04429v2 (LM-Nav, CoRL 2022)
- https://github.com/blazejosinski/lm_nav
- https://arxiv.org/pdf/2308.00688 (AnyLoc, RA-L 2023/ICRA 2024) · https://anyloc.github.io
- https://arxiv.org/html/2607.29600 (HAM-VLN, 2026)
- https://arxiv.org/pdf/2606.03374v2 (eMEM, 2026)
- https://arxiv.org/pdf/2607.14514v1 (VTM-Nav, 2026)
- https://arxiv.org/pdf/2506.19500 (NaviAgent tool navigation graph)
- https://arxiv.org/pdf/2601.21841.pdf (Graph-informed action generation with LLM)
- https://engineermaxxing.com/veanors/papers/foundation-vpr.html (AnyLoc/SALAD/MegaLoc roundup)

Sim-to-real / domain shift
- https://alphaxiv.org/abs/2607.09792 (Survey + real-world eval of embodied VLN)
- https://arxiv.org/pdf/2507.13019.pdf (Rethinking the Embodied Gap in VLN)
- https://arxiv.org/pdf/2603.19229v1.pdf (NavTrust)
- https://arxiv.org/pdf/2609.25666 (Deploying Foundation Models for Embodied Navigation)
- https://arxiv.org/html/2608.30396v1 (Scaffolding Foundation Models, 2026)
- https://alphaxiv.org/abs/2505.01458 (Sim-to-real discrepancy survey)
- https://arxiv.org/html/2512.19021v1 (VLNVerse benchmark)

VRChat / metaverse embodied AI (from the companion prior-art review)
- https://arxiv.org/abs/2410.02406 (ELLMA-T, ACM DIS 2025) · https://github.com/HeyMengxu/ELLMA-T
- https://doi.org/10.1145/3613905.3651026 (LLM agents in social VR, CHI EA 2024)
- https://arxiv.org/abs/2508.03216 (Navigation Pixie, IEEE ISMAR 2025)
- https://arxiv.org/abs/2601.03251 (NavAI) · https://doi.org/10.1007/s10515-026-00676-z
- https://arxiv.org/html/2404.10179v3 (SIMA 1) · arXiv:2512.04797 (SIMA 2)
- https://arxiv.org/html/2310.13724 (Habitat 3.0, ICLR 2024)
- https://feedback.vrchat.com/feature-requests/p/additional-parameters-for-reading-current-players-position
- https://docs.vrchat.com/docs/osc-avatar-parameters · https://docs.vrchat.com/docs/osc-trackers
- https://github.com/vrchat-community/vrc-oscquery-lib/blob/main/osc-trackers.md
- https://github.com/Lioncat6/OSC-Chat-Tools/wiki/Unofficial-OSC-Documentation
- https://github.com/kushiemoon-dev/vrchat-osc-bridge · https://github.com/ZenithVal/OSCLeash

Companion file: `research/vrchat-embodied-ai-prior-art-2026.md` (full prior-art review, ~60
queries of negative-result evidence).
