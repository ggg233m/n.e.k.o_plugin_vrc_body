# Episodic / Topological Spatial Memory for Embodied LLM Agents — Engineering Research Report

**Target system:** Windows VRChat plugin. Egocentric frames from a VRChat window capture (stylized/anime rendering, 2D menus + UI overlays), VRChat OSC avatar velocity telemetry (m/s), SteamVR HMD/controller poses. **No depth sensor, no VRChat world coordinates** — only a self-built map frame. Existing modules: person detector, OSNet person-ReID embedder, per-sector "traversability" divergence-risk predictor, LLM tool set. **Currently no map tools; a grid-token semantic landmark channel failed (zero discriminative power).**

---

## 0. Method, evidence grading, and a hard limitation

**Limitation (important):** `web_fetch` is **completely unavailable** in this session — every hostname, including `arxiv.org`, `github.com` and `en.wikipedia.org`, returns *"resolves to a non-public IP address"*. **I did not open a single page.** Every factual claim below comes from search-engine result titles, URLs and snippets, or from third-party aggregator/summary pages.

Evidence grades used throughout:

| Grade | Meaning |
|---|---|
| **[S1]** | Primary source visible in search results — an arXiv abstract/HTML/PDF URL, a proceedings URL (PMLR, ACL Anthology, CVF Open Access, NeurIPS proceedings, RSS, IJCAI, ECVA, IEEE, AAAI OJS), a DOI, or an official GitHub/project page. The URL and title are directly observed; the *detail* is from the snippet. |
| **[S2]** | Third-party page: AI-summary aggregators (emergentmind, alphaxiv, arcxiv, paperlayer, chatpaper), blogs, vendor pages. Weaker — use as lead, not proof. |
| **[U]** | **unverified** — could not confirm from any snippet. Stated as such, never guessed. |
| **[I]** | **my inference** — engineering judgement, not sourced. |

Time frame: search results carry dates through **September 2026**, so "2025–2026 successors" below are the current frontier as observed.

---

## 1. Topological memory / place graphs

### 1.1 LM-Nav — the canonical LLM + VLM + topological graph system

| Field | Value |
|---|---|
| **Name** | LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action |
| **Authors** | Dhruv Shah, Błażej Osiński, Brian Ichter, Sergey Levine (UC Berkeley / Univ. of Warsaw / Robotics at Google) — author list and affiliations observed in the arXiv snippet. **[S1]** |
| **Venue / year** | **CoRL 2022** (Conference on Robot Learning). Published as PMLR **v205** → `proceedings.mlr.press/v205/shah23b` and PDF `proceedings.mlr.press/v205/shah23b/shah23b.pdf`. Also OpenReview `forum?id=UW5A3SweAH` (created 2022-08-15). arXiv **2207.04429**. **[S1]** |
| **Open source** | **Yes** — `github.com/blazejosinski/lm_nav` (observed: 269 stars, 26 forks). Project page `sites.google.com/view/lmnav`. Third party re-use confirmed by `github.com/programminglove08/RobustnessVisualNav`, which states its LM-Nav code came from that repo. **[S1]** |

**How it combines the three components** (confirmed from snippets):

1. **LLM (GPT-3) = landmark extractor.** The project page: *"The Large Language Model is used to parse the natural language instruction into a sequence of landmarks that can serve as intermediary subgoals for navigation."* An OpenReview comment confirms: *"The proposed method uses large language models (LLM) such as GPT-3 for landmark extraction, vision-language models (VLM)..."*, and an alphaXiv summary notes *"The landmark extraction uses GPT-3 with carefully designed prompts that demonstrate the desired parsing behavior."* **[S1]**
2. **VLM (CLIP) = grounding landmarks onto the graph.** CLIP scores each extracted landmark string against the *images* already attached to graph nodes, producing a per-node cost. The PMLR PDF snippet shows the actual algorithm: a matrix `Q[i, v]` over landmarks `i` and graph vertices `v`, then **`Dijkstra algorithm(G, Q[0, *])`** — i.e. landmark grounding is written into a cost matrix and the route is a shortest path over the topological graph. **[S1]**
3. **The topological graph itself is pre-built, not learned by the LLM.** The project page describes step 1: *"Given a bunch of observations in the target environment, the goal-conditioned distance function (part of the Visual Navigation model)..."* — the graph comes from the visual-navigation model (ViNG-lineage) plus the robot's observations; the LLM never edits the graph. **[S1]**

**What it gives you:** the exact, battle-tested decomposition for *your* problem — **LLM parses, VLM grounds language to appearance, a graph search produces the route.** The LLM produces no geometry.

**Compute / hardware:** not verified in this session **[U]**. Qualitatively it ran on a real outdoor mobile robot with GPT-3 API + CLIP. **[S1 for the platform claim]**

**Training needed:** none for the LLM/CLIP (off-the-shelf); the *navigation* component is learned. **[S1/S2]**

**URLs:** <https://arxiv.org/abs/2207.04429> · <https://proceedings.mlr.press/v205/shah23b> · <https://proceedings.mlr.press/v205/shah23b/shah23b.pdf> · <https://openreview.net/forum?id=UW5A3SweAH> · <https://sites.google.com/view/lmnav> · <https://github.com/blazejosinski/lm_nav>

### 1.2 ViNT — foundation model + topological graph planning

- **Venue/year:** arXiv **2306.14846**; **CoRL 2023** → PMLR **v229** (`proceedings.mlr.press/v229/shah23a/shah23a.pdf`). Project page `general-navigation-models.github.io/vint/`. **[S1]**
- **Relevance:** the ViNT page states ViNT *"can be deployed for long-horizon navigation in combination with a **topological graph planning method**"*, and the arXiv HTML adds it plans *"on a topological graph and executed using the ViNT policy"*. This is the pattern your agent needs: a cheap graph planner on top of a reactive/render-level policy. **[S1]**
- **Open source:** GitHub repo referenced by the project page — repo URL not directly observed **[U]**; the project page and BibTeX are observed **[S1]**.
- **Training:** ViNT is a pretrained foundation model; **exact training data/compute unverified [U]**.

### 1.3 ViKiNG — geographic hints as soft planning heuristics

- **Venue/year:** arXiv **2202.11271**; **RSS 2022** (Robotics: Science and Systems XVIII, `roboticsproceedings.org/rss18/p019.html`, PDF `rss18/p019.pdf`). Authors Dhruv Shah, Sergey Levine. **[S1]**
- **Relevance to you:** the abstract states the method *"can utilize side information such as schematic roadmaps, satellite maps and GPS coordinates as a **planning heuristic, without relying on them being accurate**."* This is the strongest published precedent for **treating a noisy, non-authoritative spatial prior as a soft cost — exactly the situation of a self-built map frame derived from OSC velocity + SteamVR poses.** **[S1]**

### 1.4 TSGM — Topological Semantic Graph Memory

- **Venue/year:** **CoRL 2022** (PMLR **v205**, `papers/v205/kim23a/kim23a.pdf`); arXiv **2209.08274**. **[S1]**
- **What it gives you:** the abstract: *"A novel framework is proposed to incrementally collect landmark based graph memory..."* — node = landmark, memory is built **online**. Code: `github.com/rllab-snu/TopologicalSemanticGraphMemory`, described as *"CoRL 2022 oral"*. **[S1]**
- **Failure mode worth knowing:** PRISM-TopoMap's paper text states *"state-of-the-art topological method **TSGM links locations at opposite ends of the environment**, as shown..."* — i.e. **this class of method produces false place merges.** Directly relevant to §4. **[S1]**

### 1.5 PRISM-TopoMap — online topological mapping without global metrics

- **Venue/year:** arXiv **2404.01674** (v3 dated Feb 2025). **[S1]**
- **What it gives you (this is architecturally the closest prior art to your constraint):** *"a topological mapping method that maintains a graph of **locally aligned locations not relying on global metric coordinates**"*, combining **place recognition with 2D feature scan matching**. **[S1]**
- **Open source:** **Yes** — `github.com/KirillMouraviev/PRISM-TopoMap`. **[S1]**
- **Caveat for you:** the "2D feature scan matching" implies a range/scan modality (LiDAR or depth-derived scan). **Requires a range channel — your agent has none. [S1 for scan matching + I for the incompatibility]**

### 1.6 ETPNav — evolving topological planning

- **Venue/year:** **IEEE TPAMI 2024**, DOI `10.1109/TPAMI.2024.3386695`; arXiv **2304.03047**; PubMed 38593013. Repo: `github.com/MarSaKi/ETPNav` (*"[TPAMI 2024] Official repo"*). **[S1]**
- **Relevance:** canonical "online topological map + LLM/VLN planning" in continuous environments (Habitat). **Requires RGB-D** (Habitat-class simulators) — **incompatible with your no-depth constraint. [I, based on the standard task setup; the depth requirement itself is [U] from my snippets]**

### 1.7 2026 successor: Unordered Landmark Visual Navigation (ULVN) — **RGB-only**

- **Venue/year:** **ECCV 2026 (5-min Oral)** per the project page `hren20.github.io/ulvn-website`; arXiv **2608.06833**. Authors Hao Ren et al. **[S1/S2]**
- **Why it matters most for you:** the abstract snippet says image-goal navigation *"is strained by strong prior..."* and an aggregator characterises UL VN as *"a unified **RGB-only** framework that eliminates t[he need for depth]"* **[S2]**. The cognitive-science framing — *"human navigation relies primarily on visual observations to form a topological..."* — is precisely the topometric-plus-appearance design you want. **[S1]**
- **Training / compute / open source:** **[U]** — not verified in this session.

**Takeaway for §1 [I]:** every system that works **without depth** falls back to **appearance/landmark node identity + relative or soft geometry** (ViKiNG's soft hints, TSGM's landmark graph, PRISM-TopoMap's *local* alignment, UL VN's RGB-only landmarks). Every system that builds **metric** graphs (ETPNav, HOV-SG, ConceptGraphs, Clio) assumes depth/RGB-D or LiDAR.

---

## 2. Experience graphs / episodic memory for LLM agents

| System | Venue/Year | Memory representation | How the LLM reads it | Open source | Grade |
|---|---|---|---|---|---|
| **AriGraph** | **IJCAI 2025** (`ijcai.org/proceedings/2025/2`, DOI `10.24963/ijcai.2025/2`); arXiv **2407.04363** | *"Knowledge Graph World Models with **Episodic Memory** for LLM Agents"* — KG of entities + episodes | Graph traversal feeding the LLM | **[U]**, not observed | **[S1]** |
| **MapNav** | **ACL 2025** (`aclanthology.org/2025.acl-long.638/`); arXiv **2502.13451** | *"A Novel Memory Representation via **Annotated Semantic Maps** for VLM-based VLN"* — memory **is** a map, annotated | Map is fed to the VLM as the memory substrate | **[U]** | **[S1]** |
| **MapGPT** | **ACL 2024** (`aclanthology.org/2024.acl-long.529/`); arXiv **2401.07314** | Map + topological graph, converted into **text prompts** for **global path planning**, with adaptive path planning | Prompt-stuffed map; LLM revises the global plan | **[U]** | **[S1]** |
| **NavGPT** | **AAAI 2024** (`ojs.aaai.org/index.php/AAAI/article/view/28597`); arXiv **2305.16986**; repo `github.com/gengzezhou/navgpt` (*"[AAAI 2024] Official implementation"*) | **Zero-shot** LLM navigator; textual history of observations/actions | Serialized history in prompt | **Yes** | **[S1]** |
| **DBM (Discuss Before Moving)** | **EMNLP 2023** (researchr `Long0C024`; exact track **[U]**); arXiv **2309.11382** | Multi-expert discussion before committing an action | Debate → action | **[U]** | **[S1/S2]** |
| **ETPNav** | TPAMI 2024; arXiv **2304.03047** | Evolving online topological map | Graph → planner → low-level | **Yes** (`MarSaKi/ETPNav`) | **[S1]** |
| **SG-Nav** | **NeurIPS 2024** (proceedings hash `098491b37deebbe6c007e69815729e09`); arXiv **2410.08189**; repo `github.com/bagh2178/SG-Nav` | **Online 3D scene graph**, prompted into an LLM with hierarchical chain-of-thought for zero-shot object navigation | Scene graph serialized into the LLM prompt | **Yes** | **[S1]** |
| **3D-Mem** | **CVPR 2025** (CVF Open Access); arXiv **2411.17735**; repo `github.com/UMass-Embodied-AGI/3D-Mem` | *"3D Scene Memory for Embodied Exploration and Reasoning"* — stores rich visual info as a scene memory | Snapshot/retrieval over 3D memory | **Yes** | **[S1]** |
| **3DLLM-Mem** | **NeurIPS 2025** (proceedings PDF) | *"Long-Term **Spatial-Temporal Memory** for Embodied 3D Large Language Model"* | Memory tokens into the 3D-LLM | **[U]** | **[S1]** |
| **GraphPad** | arXiv **2506.01174** (2025) | *"**Inference-Time 3D Scene Graph Updates** for Embodied Question Answering"* — the **scene graph is updated at inference time** rather than precomputed | VLM + updatable graph | **[U]** | **[S1]** |
| **Open Scene Graphs (OSG)** | arXiv **2508.04678** (2025) | Open-world semantic scene graph for object-goal navigation | Graph for search | **[U]** | **[S1]** |
| **Generative Agents** | **UIST 2023** (DOI `10.1145/3586183.3606763`); arXiv **2304.03442** | **Memory stream**: complete record of experience; retrieval scored by **recency, importance, relevance**, plus **reflection** into higher-level summaries | Retrieved memories injected into prompt | **[U]** | **[S1]** |
| **A-Mem** | arXiv **2502.12110** (2025) | *"Agentic Memory"*: LLM-driven **note construction + linking** into a memory graph | Memory operations (link/retrieve) as agent actions | **[U]** | **[S1]** |

### 2.1 The 2026 papers that most directly constrain your design

- **"What Spatial Memory Must Store: Occlusion as the Test for Language-Agent Memory"** — arXiv **2606.10299** (submitted 2026-06-09). This is the single most relevant 2026 paper for your situation. Snippet: *"Language-agent **'memory palace'** systems anchor each memory to a **world coordinate**, on the intuition that geometry adds som[ething]..."* and *"Our contribution is upstream of rendering: we **isolate what must be stored from how it is read**, showing that once geometr[y]..."* **[S1 for those quotes; the completion of the argument is [U] because I could not open the PDF]**
  **Why it matters [I]:** the prevailing 2026 design (anchor every memory to a world coordinate) is exactly what your system **cannot** do — you have no VRChat world coordinates. This paper is your citation for the claim that the *choice of stored representation* is the load-bearing decision, and that it can be evaluated independently of the reader (LLM). It also gives you a testable criterion (occlusion) rather than "does it feel right".
- **"From Reactive to Map-Based AI: Tuned Local LLMs for Semantic Zone Inference in Object-Goal Navigation"** — arXiv **2603.08086** (2026-03-10), Yudai Noda, Kanji Tanaka. **[S1]** Directly relevant to §4: a **local, tuned** LLM is used to infer **semantic zones** for ObjectNav — i.e. "region" is treated as an *inferred semantic label over map structure*, not a geometric measurement.
- **"Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration"** — arXiv **2512.02458**. **[S1]**
- **GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning** — arXiv **2603.19137** (2026-03-19). **[S1]** Note: 3DGS memory reconstruction **needs depth/pose**; likely unsuitable for you **[I]**.
- **CapMem: A Benchmark for Caption-Based Episodic Memory in Egocentric Video** — arXiv **2609.17688** (2026-09). Snippet: *"recovering **what happened, where it happened, and when it happened**"* — i.e. the triad your three questions map onto. **[S1]**
- **EGOSTREAM: A Diagnostic Benchmark for Streaming Episodic Memory in Egocentric Vision** — arXiv **2605.31557** (2026). **[S1]**
- **ESCA: Contextualizing Embodied Agents via Scene-Graph Generation** — arXiv **2510.15963**. **[S1]**
- **OP3DSG: Open-Vocabulary Part-Aware 3D Scene Graph Generation** — arXiv **2606.29786**. **[S1]**
- **SoftNav: Injecting 3D Scene Tokens into VLMs for Embodied Navigation** — arXiv **2607.14586** (2026). **[S1]**
- **Thea / "Towards the Harness of Embodied Agents"** — arXiv **2608.11246** (2026), and **HarnessVLN** — arXiv **2609.15195** (2026). Framework-level agent harnesses for embodied navigation. **[S1]**

**Representation summary [I, from the table]:** the field has converged on **graph nodes carrying a *combination* of (a) an image/observation, (b) a text caption or object list, and (c) an embedding (CLIP/OpenCLIP/DINO-family)** — with the LLM reading a **filtered** subset, never the whole graph. Text-only nodes without an image/embedding are the exception, not the norm.

---

## 3. Semantic landmark channels that actually work (and why your grid tokens failed)

### 3.1 The diagnosis of "grid tokens had zero discriminative power" **[I, with [S1] support]**

Every working VPR descriptor is a **learned or statistically-aggregated function of local image content**:
- **NetVLAD**: *"CNN architecture for weakly supervised place recognition"* — a learnable VLAD layer aggregating CNN features. **[S1]**
- **AnyLoc**: *"aggregating per-pixel [DINOv2] features"* into a global descriptor; the paper's own comparison table lists `AnyLoc-GeM-DINOv2`, `MixVPR`, `NetVLAD`, `AnyLoc-VLAD-DINOv2`. **[S1]**
- **SALAD**: *"Optimal Transport Aggregation for Visual Place Recognition"*, explicitly building on DINOv2; the snippet notes *"AnyLoc [28] proposed to leverage foundation models, using DINOv2 [41] as a feature extractor for VPR. However, AnyLoc us[es]..."* **[S1]**

A **uniform grid of cells with no content-derived value** is a *spatial index*, not a *descriptor*. Two different rooms produce the same grid token unless the token encodes something discriminative observed there. So the failure is expected, not surprising **[I]**. I **could not verify** a paper that experimentally demonstrates "grid encodings are weak place descriptors" **[U]** — so present the grid-token failure as a *design analysis*, not as a literature-backed law.

### 3.2 The descriptor menu

| Method | Venue/Year | Open source | Needs training on domain data? | Compute | Grade |
|---|---|---|---|---|---|
| **NetVLAD** | **CVPR 2016** (DOI `10.1109/CVPR.2016.572`, `cv-foundation.org` CVPR16 PDF); arXiv **1511.07247** | **[U]** (widely reimplemented; official release not observed) | **Yes** — weakly supervised on place-labelled imagery | **[U]** | **[S1]** |
| **MixVPR** | **WACV 2023** (CVF Open Access WACV2023 repository) | **[U]** from my snippets; author page `amaralibey.github.io` observed | **Yes** (trained aggregator) | **[U]** | **[S1]** |
| **SALAD** (DINOv2 SALAD) | **ICCV 2023**; arXiv **2311.15937**; HF paper page `huggingface.co/papers/2311.15937` | **Yes** — `github.com/serizba/salad` (*"Code and models for Optimal Transport Aggregation for Visual Place Recognition (DINOv2 SALAD)"*) | Backbone DINOv2 is pretrained (no domain labels); the aggregation head is trained | **[U]** | **[S1]** |
| **AnyLoc** | **RA-L 2023 / ICRA 2024** (IEEE Xplore doc `10361537`); arXiv **2308.00688**; project `anyloc.github.io` | **Yes** — `github.com/AnyLoc/AnyLoc` (*"RA-L 2023"*) | **Closest to zero-shot** — off-the-shelf DINOv2 features + VLAD aggregation; no gradient training of the backbone **[S1 for the mechanism]** | **[U]**; there is a HF Space `TheProjectsGuy/AnyLoc` with a CUDA-or-CPU device switch **[S1]** | **[S1]** |
| **BoQ** | **CVPR 2024** — *not 2025* (CVF Open Access `CVPR2024/.../Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.html`); arXiv **2405.07364** | **Yes** — `github.com/amaralibey/Bag-of-Queries` | Trained (learnable query bag) | **[U]** | **[S1]** |
| **MegaLoc** | **CVPR 2025 Workshop (IMW)** (`openaccess.thecvf.com/content/CVPR2025W/IMW/.../Berton_MegaLoc_...`); arXiv **2502.17237** | **Yes** — `github.com/gmberton/MegaLoc` (*"An image retrieval model for any localization task"*, 203 stars) + weights `huggingface.co/gberton/MegaLoc` | Trained once on a **fused multi-dataset corpus**; snippet: *"we combine a variety of existing methods, training techniques, and datasets to train a retrieval model"* — aimed at **zero-shot transfer** to new tasks | **[U]** | **[S1]** |
| **CliqueMining** ("Close, But Not There") | **ECCV 2024** (`ecva.net/papers/eccv_2024/.../09275.pdf`); arXiv **2407.02422** | **Yes** — `github.com/serizba/cliquemining` | Trained | **[U]** | **[S1]** |
| **"A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For VPR"** | **ICCV 2025** (per PDF header *"Accepted to ICCV 2025"*); arXiv **2412.06153** | **[U]** | **[U]** | **[U]** | **[S1]** |
| **Revisit Anything** (segment-level VPR) | **ECCV 2024** (`ecva.net/papers/eccv_2024/.../08592.pdf`); arXiv **2409.18049**; project `revisit-anything.github.io` | **Yes** — `github.com/AnyLoc/revisit-anything` | Uses foundation features; segmentation-based retrieval | **[U]** | **[S1]** |
| **CricaVPR** | Venue **[U]** — only third-party review pages (themoonlight.io) found; arXiv ID **[U]**. Title confirmed as *"CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition"* | **[U]** | **[U]** | **[U]** | **[S2]** |

**Corrections to the brief:**
- **BoQ is CVPR 2024, not 2025.** **[S1]**
- **I could not verify any VPR method named "CliqueNet."** The closest verified real name is **CliqueMining** (ECCV 2024, `serizba/cliquemining`). Treat "CliqueNet" as **unverified / likely a name confusion**. **[U]**
- **Compute for all of the above is unverified in this session [U].** I refuse to quote parameter counts, VRAM figures or latency numbers I did not observe. Qualitatively they are DINOv2-ViT-class image encoders (descriptor inference is a single forward pass per keyframe), and MegaLoc is explicitly *"an image retrieval model"* — the retrieval side is a vector index over L2-normalised descriptors **[I]**.

**Useful index to mine rather than re-derive:** `github.com/gmberton/awesome-Visual-Place-Recognition` (a curated VPR paper/repo table; a citation file `Izquierdo_2024_cliquemining.txt` was observed in it). **[S1]**

### 3.3 The domain-gap problem you will hit **[I, with [S1] leads]**

Your imagery is **stylized/anime game rendering with 2D UI overlays** — a domain no VPR benchmark covers. Searches turned up **no evaluation of any VPR descriptor on anime/stylized game renders [U]**. What I *did* find is the adjacent literature on synthetic/game data for robotics:
- *"From Gaming to Research: GTA V for Synthetic Data Generation for Robotics and Navigations"* — arXiv **2502.12303** (2025). **[S1]**
- *"LychSim: A Controllable and Interactive Simulation Framework for Vision Research"* — arXiv **2605.12449** (2026). **[S1]**
- *"SegGen: An Unreal Engine 5 Pipeline for Generating Multimodal Semantic Segmentation Datasets"* — PMC12432427, repo `Secure-and-Intelligent-Systems-Lab/SegGen`. **[S1]**

**Consequence [I]:** do **not** select a descriptor from published urban-driving leaderboards. Run a **one-day in-domain calibration experiment on your own captured VRChat frames** — see the recommendation, step 0.

### 3.4 2026 work on *when to trust* a match (directly relevant to §4)

- **"Through the Lens of Doubt: Robust and Efficient Uncertainty Estimation for VPR"** — arXiv **2510.13464** (2025-10-16). Snippet explicitly frames the application as *"loop closure detection in SLAM pipelines"*. **[S1]**
- **"Quantile Transfer for Reliable Operating Point Selection in Visual Place Recognition"** — arXiv **2602.04401** (2026-02-05). Snippet: VPR *"enables mobile robots to localize in GNSS-denied environments by recognizing previously v[isited places]"*; the paper is about **choosing the operating point** (i.e. the decision threshold). **[S1]**
- **"Breaking Déjà Vu: Independent Auditing of Visual Place Recognition through Vision-Language Reasoning"** — arXiv **2607.12818** (2026-07-16). **[S1]**

**These three are the strongest 2025–2026 support for the claim that place-identity decisions must be *threshold-calibrated and uncertainty-aware*, not argmax.** **[S1]**

---

## 4. Room / region segmentation from a topological map, and "is this a new place?"

### 4.1 Classical region segmentation

| Method | Venue/Year | Mechanism | Grade |
|---|---|---|---|
| **Voronoi Random Fields** | **IJCAI 2007** (`ijcai.org/Proceedings/07/Papers/340.pdf`, abstract `ijcai.org/Abstract/07/340`) | *"Extracting the **Topological Structure** of Indoor Environments via **Place Labeling**"* — probabilistic place labeling on a generalized Voronoi graph | **[S1]** |
| **Room Segmentation: Survey, Implementation, and Analysis** | Venue **[U]** (hosted at Fraunhofer publica) | Survey; *"The most popular approaches to segment floor plans base upon **generalized Voronoi graphs**"* | **[S1]** |
| **Bayesian SegNet** | arXiv **1511.02680** (Kendall, Badrinarayanan, Cipolla) — *"Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for **Scene Understanding**"* | Monte-Carlo-dropout uncertainty in semantic segmentation | **[S1]** |
| SeqSLAM line | Original SeqSLAM (Milford & Wyeth) venue **[U]**; **OpenSeqSLAM2.0** toolbox and *"CNN Feature boosted SeqSLAM"* (arXiv 1704.05016) observed; *"Visual place recognition using HMM sequence matching"* (IEEE 6943207, 2014) | **Sequence** matching instead of single-frame matching | **[S1/S2]** |

**Honest gap:** the brief asked for **"Bayesian SegNet"-based room segmentation**. I verified Bayesian SegNet as an *uncertainty-aware segmentation* method, but **could not verify a room-segmentation-specific application of it [U]**. Likewise, "Bayesian/probabilistic place merging" is best supported in my evidence by Voronoi Random Fields (2007) + the HMM sequence matching line, **not** by a modern (2025–2026) Bayesian place-merging paper — I did not find one. **[U]**

### 4.2 Modern systems that actually segment rooms — and their sensor demands

| System | Venue/Year | Segments rooms how | Sensors | Open source | Grade |
|---|---|---|---|---|---|
| **HOV-SG** | **RSS 2024** (`roboticsproceedings.org/rss20/p077.html`); arXiv **2403.17846**; site `hovsg.github.io` | Hierarchical **floor → room → object** open-vocabulary 3D scene graph for large multi-story indoor spaces | **RGB-D.** Its README states: *"HOV-SG uses the Open CLIP model to extract features from **RGB-D frames**."* | **Yes** — `github.com/hovsg/HOV-SG` (*[RSS2024] Official implementation*) | **[S1]** |
| **ConceptGraphs** | **ICRA 2024** (OpenReview `4oKEEQL1bJ`); arXiv **2309.16650**; site `concept-graphs.github.io` | Open-vocabulary 3D scene graph (objects + relations) for perception and planning | RGB-D + poses **[I]** | **[U]**; project site observed | **[S1]** |
| **Clio** | arXiv **2404.13696** (2024); a related listing appears at `semrob.github.io` (RSS workshop) — **venue [U]** | *"Real-time construction of compact **open-set** 3D scene graphs"*, task-driven | **LiDAR/depth** (MIT-SPARK Hydra-class) **[I]** | **Yes** — `github.com/MIT-SPARK/Clio` | **[S1]** |
| **Occupancy-Grounded Room Segmentation for Hierarchical 3D Scene Graphs** | arXiv **2606.13727** (2026-06-11), Cueto Zumaya, Catalano, Peña-Queralta et al. | **2026 state of the art in exactly your §4 question** — room segmentation *grounded in occupancy* for 3DSGs | **Requires occupancy (depth/LiDAR).** Snippet: *"Hierarchical 3D scene graphs (3DSGs) for indoor robots organize geometric and semantic information across spati[al scales]"* | **[U]** | **[S1]** |
| **From Reactive to Map-Based AI** | arXiv **2603.08086** (2026) | **Semantic zone inference** from a map, using a **tuned local LLM** | Map-based (ObjectNav); depth implied by ObjectNav sim **[I]** | **[U]** | **[S1]** |
| **Layout Anything** | arXiv **2512.02952** (2025) | *"One Transformer for Universal **Room Layout Estimation**"* | Monocular layout estimation — task is single-image, but **needs the layout prior/indoor imagery [U]** | **[U]** | **[S1]** |
| **FloorSAM** | arXiv **2509.15750** (2025) | SAM-guided floorplan reconstruction, semantic-geometric fusion | Point clouds/geometry **[S1 for "reconstructing building floorplans"]** | **[U]** | **[S1]** |

### 4.3 So how *should* "this is a new place" be decided? — synthesised mechanism **[I]**

From the sources above, a defensible four-part decision rule:

1. **Descriptor similarity + margin, not argmax.** Retrieve top-k node descriptors; if `sim(best) < τ` **and** `sim(best) − sim(second) < margin`, propose a new node. Support: quantile-transfer operating-point selection (arXiv 2602.04401) and uncertainty estimation for loop closure (arXiv 2510.13464). **[S1 for the components; the specific rule is [I]]**
2. **Sequence/temporal consistency.** A single frame is a weak decision; a *sequence* of frames matching a *sequence* of nodes is strong. Support: SeqSLAM line, OpenSeqSLAM2.0, HMM sequence matching. **[S1]**
3. **Topological/odometric reachability check.** A candidate merge must also be *reachable* from the current node within the dead-reckoned displacement budget. This is the direct antidote to the documented TSGM failure (*"links locations at opposite ends of the environment"*). **[S1 for the failure mode; the check is [I]]**
4. **Room = a cluster of nodes, not a node.** A "place/room" should be discovered as a **densely-connected cluster in the place graph** (community detection over the graph), exactly as GraphRAG forms communities over a lexical graph via **hierarchical Leiden clustering** and summarises each. Support: GraphRAG docs on hierarchical Leiden community detection. **[S1]** This gets you **regions for free from topology, with no depth and no world coordinates. [I — this is the key recommendation]**

---

## 5. Graph query interfaces for LLMs — what serialization actually gets used

### 5.1 GraphRAG — the reference implementation of "graph out of context, LLM in front"

| Field | Value |
|---|---|
| **Name** | *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — Darren Edge et al. |
| **Venue/Year** | arXiv **2404.16130** (2024-04-24); Microsoft Research publication page. **[S1]** |
| **Open source** | **Yes** — `github.com/microsoft/graphrag`, self-described as *"A modular graph-based Retrieval-Augmented Generation (RAG) system"*, and flagged as *"a research project"*. **[S1]** |
| **Serialization** | A **lexical graph**: **TextUnits** → extracted **entities and relationships** → **hierarchical communities** → **community summaries**. The indexing dataflow has **six phases**, Phase 1 = *"Compose TextUnits"*. Community detection docs: *"How hierarchical **Leiden** clustering organizes knowledge graphs into multi-level community structure."* **[S1]** |
| **How the LLM queries it** | **Not** by dumping the graph. Query modes are **local** (entity-centric neighbourhood) and **global** (community summaries). The project frames the whole approach as *"a structured, hierarchical approach to RAG, as opposed to naive semantic-se[arch]"*. **[S1]** |
| **Compute** | **High, and unverified in this session [U]** — the indexing pipeline makes many LLM calls (entity/relationship extraction per TextUnit + community report generation). This is the well-known cost objection; I did not verify a number. |

### 5.2 Structured Scene Interfaces (SSI) — the 2025/2026 academic name for your design problem

- **"Structured Interfaces for Automated Reasoning with 3D Scene Graphs"** — Aaron Ray, Jacob Arkin, Harel Biggie, Chuchu Fan et al. (MIT). arXiv **2510.16643** (submitted 2025-10-18; also a talk at `youtube.com/watch?v=zY_YI9giZSA`, 2025-10-30). Snippets: *"Our goal is to enable users to give natural language commands to robots for tasks such as navigating to a goal or retrie[ving]..."* and *"the natural language ... [to] 3D Scene Graphs (3DSGs) and large language models (LLMs) have become popular choices for r[obotics]"*. **[S1]**
- **"Structured Scene Interface (SSI)"** as a named class of representation — aggregator definition: *"SSI is a class of representations that **externalizes scene organization for enhanced reasoning**"* (emergentmind topic page, updated 2026-07). **[S2]**
- **"SSI-Policy: Learning Structured Scene Interfaces for Vision-Language Robotic Manipulation"** — arXiv **2606.26800** (2026). Aggregator: *"**RGB-only, robot-agnostic** framework"*, *"decouples perception from control"*. **[S2 for the RGB-only claim]**
- **Concrete query language / exact serialization of SSI: [U]** — I could not observe the interface definition (JSON schema, DSL, or query language) without opening the paper. Claiming one would be fabrication.

### 5.3 What the serializations actually look like **[I, grounded in [S1]]**

Across SG-Nav (NeurIPS 2024), ConceptGraphs (ICRA 2024), HOV-SG (RSS 2024), GraphPad (arXiv 2506.01174), MapNav (ACL 2025), AriGraph (IJCAI 2025) and GraphRAG, the observed pattern is **three tiers**:

1. **Per-node record**: an image/observation + a text description/object list + an embedding. (Confirmed in structure for scene-graph systems and VPR systems; exact field names **[U]**.)
2. **Compact text/JSON adjacency** for the *relevant* subgraph — **not** the whole graph. MapGPT does prompt-level map serialization for global planning **[S1]**; NavGPT serializes observation/action history **[S1]**; SG-Nav prompts the scene graph into the LLM **[S1]**.
3. **Retrieval or tool calls that return a filtered subgraph.** GraphRAG local vs global modes **[S1]**; NaviAgent (*"Bilevel Planning on **Tool Navigation Graph**"*, arXiv 2506.19500, 2025) is the tool-graph analogue **[S1]** — non-spatial, but the clearest example of an LLM planning *over a graph of tools*.

**Criticism / limits of naive serialization:** I looked for a paper that *quantifies* token bloat or structure loss from dumping a graph into a prompt. The only [S1] support is GraphRAG's own framing of itself as *"opposed to naive semantic-search"* — i.e. the critique is implicit in the system design, not a measured result I can cite. **Treat "prompt-stuffing the graph does not scale" as [I], not as a sourced finding.**

### 5.4 MCP / tool-surface practice

- **VRChat MCP OSC** (`github.com/krekun/vrchat-mcp-osc`) — *"provides a bridge between AI assistants and VRChat using the Model Context Protocol (MCP)"*, exposing avatar control/interaction. **[S1]** This is directly reusable plumbing for your tool layer.
- **Prior art in exactly your application space:** `HoppouAI/ProjectGabriel-Remastered` (2026) — *"A real-time VRChat AI powered by Gemini Live with voice, YOLO person and face tracking, **OSC control, memory**, and a Discord bot."* **[S1]** Confirms the architecture family exists in the wild; its memory design details are **[U]**.
- **Graph-memory MCP read surfaces** — a non-peer-reviewed write-up claims graph-memory MCP servers *"expose sharply different read surfaces: ranked facts, completed answers, raw Cypher resul[ts]"* (mnemoverse.com, 2026). **[S2]** Useful as an informal design taxonomy, not evidence.

---

## 6. Concrete recommendation for the VRChat agent

### 6.0 Design constraints, restated as the argument

1. **No depth → no metric 3D scene graph.** HOV-SG needs RGB-D frames (**[S1]**, its README). ConceptGraphs, Clio, ETPNav, GSMem, FloorSAM and the 2026 occupancy-grounded room segmentation paper all sit on depth/LiDAR/occupancy (**[S1] for HOV-SG and for the papers' stated reliance on geometric maps; the rest is [I]**). Eliminate them.
2. **No VRChat world coordinates → all geometry must be *relative and soft*.** This is *published practice*, not a hack: ViKiNG uses *"schematic roadmaps, satellite maps and GPS coordinates as a planning heuristic, **without relying on them being accurate**"* (**[S1]**), and PRISM-TopoMap maintains *"locally aligned locations **not relying on global metric coordinates**"* (**[S1]**).
3. **The landmark channel must be appearance-derived.** Grid tokens failed because they carry no content-derived information; every functioning place descriptor is a learned/aggregated function of image content (NetVLAD, AnyLoc, SALAD, MixVPR, BoQ, MegaLoc — all **[S1]**).
4. **A do-not-repeat:** keep the LLM *out* of geometry, exactly as LM-Nav does (GPT-3 extracts landmarks, CLIP grounds them, Dijkstra routes — **[S1]**).

### 6.1 Step 0 — a one-day in-domain calibration before any architecture commitment **[I]**

Nothing in the literature tells you which descriptor survives **anime/stylized VRChat rendering with UI overlays** — I found **no such evaluation [U]**. So:

Capture ~500–2000 frames with rough place labels (you can label by walking back and forth). Compute descriptors with **AnyLoc** (training-free DINOv2+VLAD, `github.com/AnyLoc/AnyLoc`) and **MegaLoc** (`github.com/gmberton/MegaLoc`, weights on HF). Measure same-place vs different-place similarity separation, and pick the operating point with the **quantile-transfer / uncertainty** method from arXiv 2602.04401 and arXiv 2510.13464. **This experiment decides your τ and your descriptor; everything below is contingent on its outcome.**

### 6.2 Recommended memory representation: a two-layer place graph **[I]**

**Layer A — appearance-indexed topological place graph** (persistent, is your "map"):

- **Node = a discrete visited place.**
  - `place_id` (stable)
  - **`descriptor`** — one L2-normalised appearance vector (MegaLoc primary; AnyLoc fallback). **This is the channel that replaces the failed grid tokens.**
  - **`keyframe`** — the raw frame (or a handle to it). *Store the image, not only text.* Every working system keeps the observation; 3D-Mem (*"store rich visual info"*) and MapNav make this the point. **[S1]**
  - **`caption`** — VLM-generated, **created lazily** on first read, cached. Not the primary key.
  - **`breadcrumb`** — accumulated 2D pose in the **self-built frame**, from OSC velocity + SteamVR pose deltas, **plus an explicit uncertainty scalar**. ViKiNG's "hints you don't rely on" (**[S1]**) is the licence for this.
  - `visit_count`, `first_seen`/`last_seen` (supports Generative Agents' recency/importance retrieval weighting — **[S1]**)
  - `person_stats` — aggregated OSNet ReID IDs seen here (you already have this embedder; it is a free, non-visual node attribute)
- **Edge = an observed traversal**: `from`, `to`, dead-reckoned relative displacement, uncertainty, **traversability/divergence-risk cost from your existing module**, direction of travel.
- **Room/region = a cluster**, discovered by community detection over the place graph (GraphRAG's hierarchical Leiden approach, **[S1]**) — not by a geometric segmentation you cannot compute. This is how you get "what places exist" without depth.

**Explicitly rejected:** storing the map as grid tokens (**already falsified empirically in your system**), text-only nodes (no discriminative key), any 3D/occupancy/voxel map (needs depth), 3DGS memory (needs depth+pose).

**Optional soft depth [I]:** a monocular metric-depth model (Depth Anything V2, NeurIPS 2024, arXiv 2406.09414, metric variants on HF — **[S1]**) can feed **traversability only**. Do **not** promote it to metric mapping: its behaviour on stylized anime rendering is **[U]**, and GeoLoco (arXiv 2603.07624, 2026) shows that using visual-foundation geometric priors for RGB-only *control* is an active research problem, not a solved drop-in — **[S1]**.

### 6.3 Recommended LLM tool signatures — 5 tools **[I]**

Design rules: the LLM never receives the graph; each tool returns a **bounded, ranked, uncertainty-annotated** result; every tool maps to your three questions.

```
1. where_am_i()
   -> {
        current_place_id: str | null,
        confidence: float,                  # 0..1, from descriptor margin
        candidates: [ {place_id, caption, similarity, margin, last_seen} ],  # top-k
        odometric_pose_estimate: {x, y, theta, sigma},   # SOFT, self-built frame
        is_likely_new_place: bool
      }

2. list_places(query: str | null, limit: int = 10)
   -> [ {place_id, caption, visit_count, last_seen,
         distance_edges: int,                        # graph hops, NOT metres
         representative_keyframe_ref} ]
      # `query` is grounded by text->descriptor/caption retrieval over nodes
      # (the LM-Nav CLIP-grounding step, but as a tool).

3. describe_place(place_id: str, include_frames: bool = false)
   -> { place_id, caption, first_seen, last_seen, visit_count,
        persons_seen: [...],
        outgoing_edges: [ {to_place_id, to_caption, traversability_cost,
                           displacement_estimate, displacement_sigma} ],
        keyframe_refs: [...] }        # returned as handles, not inline bytes

4. find_route(from_place_id: str | null, to_place_id: str)
   -> { reachable: bool,
        waypoints: [place_id, ...],                # ordered, Dijkstra-style (as LM-Nav)
        total_cost: float, total_edge_hops: int,
        low_confidence_edges: [ ... ] }            # flag uncertain hops

5. mark_place(label: str, place_id: str | null = null)
   -> { place_id, label, applied: bool }
      # LLM WRITES BACK NAMES ONLY. It never creates geometry or edges.
      # Optionally: current_frontier_candidates() returning sectors the
      # traversability module believes are unexplored.
```

**Why this set is the most defensible choice:**

- **It mirrors LM-Nav's proven division of labour** — LLM does language, the descriptor/index does identity, a graph search does routing (**[S1]**, and the PMLR PDF shows the actual Dijkstra over graph vertices).
- **`where_am_i()` returns ambiguity instead of a guess.** Given that no descriptor is validated on your rendering domain **[U]**, exposing `confidence` + ranked `candidates` is the only honest interface. This is the direct fix for the grid-token failure: a *quantified* identity channel rather than a bare token.
- **`list_places` + `describe_place` answer "what places exist"** with a bounded payload, following GraphRAG's retrieve-a-subgraph discipline rather than prompt-stuffing the graph (**[S1]** for GraphRAG's design stance; the token-bloat reasoning is **[I]**).
- **`find_route` returns `low_confidence_edges`** so the LLM can say "the map is unsure about the last stretch" — matching MapGPT's adaptive *global-plan-then-revise* pattern and the soft-prior philosophy of ViKiNG (**[S1]**).
- **`mark_place` keeps the LLM out of map construction.** Every surveyed system that works keeps the LLM as a *reader/labeller*, and the map as a *maintained data structure* (NavGPT, SG-Nav, GraphPad's inference-time graph updates — **[S1]**).
- **Distance is in graph hops plus flagged uncertain displacement, never claimed metres.** Given the self-built frame's drift, quoting metres would be an unforced error.

### 6.4 Three failure modes to guard against from day one **[I]**

1. **False place merges.** Documented in TSGM (*"links locations at opposite ends of the environment"* — **[S1]**). Guard with the four-part rule in §4.3 (threshold + margin + sequence + reachability).
2. **Descriptor domain collapse.** Stylized rendering + UI overlays may put all frames in a tight similarity ball, making every place look the same. **Step 0 exists to detect exactly this**; if it happens, fall back to AnyLoc's training-free DINOv2 features, or fine-tune an aggregator on your own frames in the style of MegaLoc's multi-dataset fusion (**[S1]** for MegaLoc's approach).
3. **Odometry silence during thumbstick locomotion.** If OSC exposes velocity **magnitude only** (no direction) and SteamVR poses track only the *physical* playspace, then smooth locomotion produces **no physical HMD translation** and dead reckoning has no observable direction. **This is my single largest unverified dependency [U]** — verify (a) whether OSC gives a velocity *vector* or a scalar, and (b) how avatar yaw relates to HMD yaw during smooth turn. If direction is unavailable, heading must come from the visual channel or controller orientation, and `breadcrumb.sigma` must be large and grow.

---

## 7. What I explicitly could NOT verify

| Item | Status |
|---|---|
| A VPR method named **"CliqueNet"** | **[U]** — no such method found; **CliqueMining (ECCV 2024)** is the closest verified name |
| **BoQ venue** | Corrected: **CVPR 2024**, not 2025 **[S1]** |
| **Compute/VRAM/latency** for NetVLAD, MixVPR, SALAD, AnyLoc, BoQ, MegaLoc | **[U]** — no numbers observed; deliberately not invented |
| Any **VPR evaluation on anime/stylized game rendering** (VRChat-like) | **[U]** — not found |
| **"CliqueNet" / CricaVPR** arXiv IDs and venues | **[U]** / **[S2]** only |
| **Clio** venue | **[U]** (arXiv 2404.13696 confirmed; a workshop listing exists) |
| **Room Segmentation: Survey** venue, **original SeqSLAM** venue | **[U]** |
| A **2025–2026 Bayesian place-merging** paper | **[U]** — not found; nearest evidence is Voronoi Random Fields (2007) + HMM sequence matching (2014) |
| A **room-segmentation-specific Bayesian SegNet** application | **[U]** — Bayesian SegNet confirmed as uncertainty-aware segmentation only |
| Exact **SSI serialization / query language** (arXiv 2510.16643) | **[U]** — could not open the PDF |
| Exact **GraphRAG cost figures** | **[U]** |
| Whether any paper **quantifies grid-token / naive-serialization failure** | **[U]** — the grid-token diagnosis in §3.1 is my analysis **[I]** |
| Whether **VRChat OSC provides velocity direction** and how **SteamVR poses behave under thumbstick locomotion** | **[U]** — flagged as the top engineering risk |
| **DMZ/DBM** exact EMNLP 2023 track; **ViNT** repo URL; **NetVLAD/MixVPR** official repos | **[U]** |

---

## 8. Full URL list

**§1 Topological memory / place graphs**
1. <https://arxiv.org/abs/2207.04429> — LM-Nav, arXiv
2. <https://proceedings.mlr.press/v205/shah23b> — LM-Nav, CoRL 2022 (PMLR v205)
3. <https://proceedings.mlr.press/v205/shah23b/shah23b.pdf> — LM-Nav PDF (Dijkstra-over-graph algorithm)
4. <https://openreview.net/forum?id=UW5A3SweAH> — LM-Nav OpenReview
5. <https://sites.google.com/view/lmnav> — LM-Nav project page
6. <https://github.com/blazejosinski/lm_nav> — LM-Nav code
7. <https://arxiv.org/abs/2306.14846> — ViNT arXiv
8. <https://proceedings.mlr.press/v229/shah23a/shah23a.pdf> — ViNT, CoRL 2023 (PMLR v229)
9. <https://general-navigation-models.github.io/vint/> — ViNT project page
10. <https://arxiv.org/abs/2202.11271> — ViKiNG arXiv
11. <https://www.roboticsproceedings.org/rss18/p019.html> — ViKiNG, RSS 2022
12. <https://www.roboticsproceedings.org/rss18/p019.pdf> — ViKiNG PDF
13. <https://proceedings.mlr.press/v205/kim23a/kim23a.pdf> — TSGM, CoRL 2022
14. <https://arxiv.org/abs/2209.08274> — TSGM arXiv
15. <https://github.com/rllab-snu/TopologicalSemanticGraphMemory> — TSGM code
16. <https://arxiv.org/abs/2404.01674> — PRISM-TopoMap arXiv
17. <https://github.com/KirillMouraviev/PRISM-TopoMap> — PRISM-TopoMap code
18. <https://arxiv.org/abs/2304.03047> — ETPNav arXiv
19. <https://doi.org/10.1109/TPAMI.2024.3386695> — ETPNav, TPAMI 2024
20. <https://github.com/MarSaKi/ETPNav> — ETPNav code
21. <https://arxiv.org/abs/2608.06833> — Unordered Landmark Visual Navigation (ULVN), ECCV 2026
22. <https://hren20.github.io/ulvn-website> — ULVN project page
23. <https://www.cs.cmu.edu/~iwan/papers/localization.pdf> — Appearance-Based Place Recognition for Topological Localization (background)

**§2 Episodic / experience memory for LLM agents**
24. <https://www.ijcai.org/proceedings/2025/2> — AriGraph, IJCAI 2025
25. <https://arxiv.org/abs/2407.04363> — AriGraph arXiv
26. <https://dl.acm.org/doi/10.24963/ijcai.2025/2> — AriGraph, ACM DL
27. <https://aclanthology.org/2025.acl-long.638/> — MapNav, ACL 2025
28. <https://arxiv.org/html/2502.13451v3> — MapNav arXiv
29. <https://aclanthology.org/2024.acl-long.529/> — MapGPT, ACL 2024
30. <https://arxiv.org/abs/2401.07314> — MapGPT arXiv
31. <https://ojs.aaai.org/index.php/AAAI/article/view/28597> — NavGPT, AAAI 2024
32. <https://arxiv.org/abs/2305.16986> — NavGPT arXiv
33. <https://github.com/gengzezhou/navgpt> — NavGPT code
34. <https://arxiv.org/abs/2309.11382> — Discuss Before Moving arXiv
35. <https://researchr.org/publication/Long0C024> — DBM publication record
36. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html> — SG-Nav, NeurIPS 2024
37. <https://arxiv.org/abs/2410.08189v1> — SG-Nav arXiv
38. <https://github.com/bagh2178/SG-Nav> — SG-Nav code
39. <https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_3D-Mem_3D_Scene_Memory_for_Embodied_Exploration_and_Reasoning_CVPR_2025_paper.pdf> — 3D-Mem, CVPR 2025
40. <https://github.com/UMass-Embodied-AGI/3D-Mem> — 3D-Mem code
41. <https://proceedings.neurips.cc/paper_files/paper/2025/file/61f527a737e4ba61f3e10d6c3f0c4b55-Paper-Conference.pdf> — 3DLLM-Mem, NeurIPS 2025
42. <https://arxiv.org/abs/2506.01174> — GraphPad
43. <https://arxiv.org/abs/2508.04678> — Open Scene Graphs
44. <https://arxiv.org/pdf/2304.03442> — Generative Agents arXiv
45. <https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763> — Generative Agents, UIST 2023
46. <https://arxiv.org/pdf/2502.12110> — A-Mem
47. <https://arxiv.org/abs/2606.10299> — What Spatial Memory Must Store (2026)
48. <https://arxiv.org/pdf/2603.08086v1.pdf> — From Reactive to Map-Based AI / Semantic Zone Inference (2026)
49. <https://alphaxiv.org/abs/2512.02458> — Vision to Geometry: 3D Spatial Memory (2026)
50. <https://alphaxiv.org/abs/2603.19137> — GSMem (2026)
51. <https://arxiv.org/pdf/2609.17688> — CapMem (2026)
52. <https://arxiv.org/pdf/2605.31557v2> — EGOSTREAM (2026)
53. <https://arxiv.org/pdf/2510.15963v1> — ESCA
54. <https://arxiv.org/pdf/2606.29786v2> — OP3DSG
55. <https://arxiv.org/pdf/2607.14586v1> — SoftNav
56. <https://arxiv.org/html/2608.11246v1> — Towards the Harness of Embodied Agents (Thea)
57. <https://arxiv.org/pdf/2609.15195.pdf> — HarnessVLN

**§3 Place-recognition descriptors**
58. <http://cv-foundation.org/openaccess/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf> — NetVLAD, CVPR 2016
59. <https://arxiv.org/abs/1511.07247> — NetVLAD arXiv
60. <https://doi.org/10.1109/cvpr.2016.572> — NetVLAD DOI
61. <https://openaccess.thecvf.com/WACV2023> — MixVPR, WACV 2023
62. <https://amaralibey.github.io> — MixVPR / BoQ author page
63. <https://arxiv.org/abs/2311.15937> — SALAD arXiv
64. <https://github.com/serizba/salad> — SALAD code (DINOv2 SALAD)
65. <https://huggingface.co/papers/2311.15937> — SALAD on HF Papers
66. <https://arxiv.org/html/2308.00688v2> — AnyLoc arXiv
67. <https://anyloc.github.io/> — AnyLoc project page
68. <https://github.com/AnyLoc/AnyLoc> — AnyLoc code (RA-L 2023)
69. <https://ieeexplore.ieee.org/ielaam/7083369/10360389/10361537-aam.pdf> — AnyLoc, IEEE
70. <https://openaccess.thecvf.com/content/CVPR2024/html/Ali-bey_BoQ_A_Place_is_Worth_a_Bag_of_Learnable_Queries_CVPR_2024_paper.html> — BoQ, CVPR 2024
71. <https://arxiv.org/abs/2405.07364> — BoQ arXiv
72. <https://github.com/amaralibey/Bag-of-Queries> — BoQ code
73. <https://openaccess.thecvf.com/content/CVPR2025W/IMW/html/Berton_MegaLoc_One_Retrieval_to_Place_Them_All_CVPRW_2025_paper.html> — MegaLoc, CVPRW 2025
74. <https://arxiv.org/abs/2502.17237> — MegaLoc arXiv
75. <https://github.com/gmberton/MegaLoc> — MegaLoc code
76. <https://huggingface.co/gberton/MegaLoc> — MegaLoc weights
77. <https://arxiv.org/abs/2407.02422> — CliqueMining arXiv
78. <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09275.pdf> — CliqueMining, ECCV 2024
79. <https://github.com/serizba/cliquemining> — CliqueMining code
80. <https://arxiv.org/pdf/2412.06153v2> — Hyperdimensional One Place Signature, ICCV 2025
81. <https://arxiv.org/html/2409.18049v1> — Revisit Anything arXiv
82. <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08592.pdf> — Revisit Anything, ECCV 2024
83. <https://github.com/AnyLoc/revisit-anything> — Revisit Anything code
84. <https://github.com/gmberton/awesome-Visual-Place-Recognition> — VPR index
85. <https://github.com/CV4RA/SOTA-Place-Recognitioner> — VPR SOTA tracker
86. <https://arxiv.org/pdf/2406.09414> — Depth Anything V2 arXiv
87. <https://proceedings.neurips.cc/paper_files/paper/2024/file/26cfdcd8fe6fd75cc53e92963a656c58-Paper-Conference.pdf> — Depth Anything V2, NeurIPS 2024
88. <https://depth-anything-v2.github.io/> — Depth Anything V2 project page
89. <https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base> — DA-V2 metric weights
90. <https://arxiv.org/html/2502.12303v1> — GTA V synthetic data for robotics
91. <https://arxiv.org/pdf/2605.12449.pdf> — LychSim (2026)
92. <https://ncbi.nlm.nih.gov/pmc/articles/PMC12431427> — SegGen (UE5 semantic data)
93. <https://arxiv.org/abs/2510.13464> — Through the Lens of Doubt (VPR uncertainty, 2025)
94. <https://arxiv.org/abs/2602.04401> — Quantile Transfer / operating point (2026)
95. <https://arxiv.org/abs/2607.12818> — Breaking Déjà Vu (VPR auditing, 2026)
96. <https://arxiv.org/html/2608.27226> — DINOcular (2026)
97. <https://arxiv.org/pdf/2603.07624> — GeoLoco (RGB-only, 2026)

**§4 Room / region segmentation**
98. <https://www.ijcai.org/Proceedings/07/Papers/340.pdf> — Voronoi Random Fields, IJCAI 2007
99. <https://www.ijcai.org/Abstract/07/340> — Voronoi Random Fields abstract
100. <https://publica.fraunhofer.de/bitstreams/64c0cb3e-af64-469e-b52c-484cd0e8b38f/download> — Room Segmentation: Survey, Implementation, and Analysis
101. <https://arxiv.org/html/1511.02680v2> — Bayesian SegNet
102. <https://arxiv.org/abs/2403.17846> — HOV-SG arXiv
103. <https://www.roboticsproceedings.org/rss20/p077.html> — HOV-SG, RSS 2024
104. <https://github.com/hovsg/HOV-SG> — HOV-SG code (RGB-D requirement in README)
105. <https://hovsg.github.io/> — HOV-SG project page
106. <https://arxiv.org/abs/2309.16650> — ConceptGraphs arXiv
107. <https://concept-graphs.github.io/> — ConceptGraphs project page
108. <https://openreview.net/forum?id=4oKEEQL1bJ> — ConceptGraphs OpenReview
109. <https://arxiv.org/abs/2404.13696> — Clio arXiv
110. <https://github.com/MIT-SPARK/Clio> — Clio code
111. <https://arxiv.org/pdf/2606.13727.pdf> — Occupancy-Grounded Room Segmentation for H3DSGs (2026)
112. <https://alphaxiv.org/abs/2606.13727> — same, aggregator
113. <https://arxiv.org/pdf/2509.15750v1> — FloorSAM
114. <https://doi.org/10.48550/arxiv.2512.02952> — Layout Anything
115. <https://emergentmind.com/topics/seqslam> — SeqSLAM topic summary [S2]
116. <https://emergentmind.com/topics/openseqslam2-0> — OpenSeqSLAM2.0 [S2]
117. <https://ar5iv.labs.arxiv.org/html/1704.05016> — CNN Feature boosted SeqSLAM
118. <https://ieeexplore.ieee.org/document/6943207> — VPR using HMM sequence matching

**§5 Graph query interfaces for LLMs**
119. <https://arxiv.org/abs/2404.16130> — GraphRAG arXiv
120. <https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/> — GraphRAG, Microsoft Research
121. <https://github.com/microsoft/graphrag> — GraphRAG code
122. <https://microsoft.github.io/graphrag/index/default_dataflow> — GraphRAG indexing dataflow
123. <https://microsoft-graphrag.mintlify.app/concepts/community-detection> — GraphRAG hierarchical Leiden
124. <https://microsoft-graphrag.mintlify.app/indexing/dataflow> — GraphRAG six-phase pipeline
125. <https://graphrag.com/reference/knowledge-graph/lexical-graph-extracted-entities-community-summaries/> — GraphRAG lexical graph
126. <https://arxiv.org/abs/2510.16643> — Structured Interfaces for Automated Reasoning with 3D Scene Graphs
127. <https://youtube.com/watch?v=zY_YI9giZSA> — SSI talk (2025-10-30)
128. <https://emergentmind.com/topics/structured-scene-interface-ssi> — SSI definition [S2]
129. <https://arxiv.org/pdf/2606.26800v2> — SSI-Policy (2026)
130. <https://arxiv.org/pdf/2506.19500> — NaviAgent (tool navigation graph)
131. <https://arxiv.org/abs/2502.18470v5> — Spatial-RAG
132. <https://github.com/krekun/vrchat-mcp-osc> — VRChat MCP OSC
133. <https://glama.ai/mcp/servers/Krekun/vrchat-mcp-osc> — VRChat MCP OSC listing
134. <https://github.com/HoppouAI/ProjectGabriel-Remastered> — VRChat AI + OSC + memory prior art
135. <https://wiki.vrchat.com/?oldid=69278&title=Open_Sound_Control> — VRChat OSC wiki
136. <https://github.com/Lioncat6/OSC-Chat-Tools/wiki/Unofficial-OSC-Documentation> — unofficial OSC docs
137. <https://mnemoverse.com/docs/library/graph-memory-mcp-tools> — graph-memory MCP read surfaces [S2]
