# Visual Traversability / Affordance Perception for Small Edge Robots (2023–2026)

**Compiled 2026-10-04.** Covers cs.RO work through Sept 2026 (arXiv 26xx IDs).

---

## 0. How this was compiled, and what to distrust

**Tooling limitation (important).** In this environment `web_fetch` is non-functional — DNS resolution fails for *every* domain (`arxiv.org`, `github.com`, `openreview.net`, PMLR, IEEE, ar5iv all return "resolves to a non-public IP address"). All facts below were extracted from **search-engine result snippets**, not full texts. Consequences:

- Numbers marked **[unverified]** could not be confirmed against a paper body and must be checked before use.
- Where a claim rests on an abstract only, I say so.
- Two items in the original request could **not be confirmed to exist** and are flagged prominently (§1.1, §1.2).

---

## 1. Supervised / label-based traversability nets

### 1.1 ⚠️ "URMA" — the name is ambiguous; the traversability paper could not be verified

There are **three different things** called "URMA"/"U-Mamba", and search engines collapse them:

| What | Actually is | Source |
|---|---|---|
| **U-Mamba** | *Biomedical* 2D/3D image segmentation hybrid CNN-SSM. arXiv:2401.04722, Ma/Li/Wang (Toronto Vector). 1000+ GitHub stars. **Not robotics.** | [arxiv.org/abs/2401.04722](https://arxiv.org/abs/2401.04722), [github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba) |
| **URMA (traversability)** | "Enhancing Long-range Dependency for Vision-based Traversability Assessment", attributed to Frey/Mattamala/Khattak/Hutter, IEEE RA-L 2024. **Could not be verified** — every search for the title resolved to the biomedical paper. Latency/param claims (e.g. "~40 ms") are **[unverified]**. | *No verified URL.* Closest legitimate source: [Jonas Frey publications](https://jonasfrey96.github.io/publications) (list snippet was truncated before the entry) |
| **URMA (locomotion)** | **Unified Robot Morphology Architecture** — a single policy controlling many legged embodiments. Bohlinger et al., CoRL 2024, PMLR v270:3356–3378, arXiv:2409.06366. [URMAv2](https://arxiv.org/pdf/2509.02815v1) scales to 50 robots. | [nico-bohlinger.github.io](https://nico-bohlinger/one_policy_to_run_them_all_website), [github.com/nico-bohlinger/one_policy_to_run_them_all](https://github.com/nico-bohlinger/one_policy_to_run_them_all) |

**Also note the attribution error in the request:** METAVerse is *not* MIT. It is Seo, Kim, **Ahn (U. Michigan)**, Kwak (UMD/ADD).

**Practical substitute for the intended URMA:** the same ETH RSL group publishes the strongest verified real-time vision traversability nets — **RoadRunner** and **RoadRunner M&M** (§1.2), plus the DHT/LHT cost-map heads from RSL. Cite those instead unless you can pull the RA-L PDF directly.

### 1.2 RoadRunner / RoadRunner M&M (Frey, Khattak, Hutter et al., ETH RSL)

**RoadRunner** — [arXiv:2402.19341](https://arxiv.org/abs/2402.19341). Camera + LiDAR fusion predicting **traversability AND an elevation map** jointly, aimed at high-speed off-road driving where degraded image quality (low light, motion blur, dust) breaks single-sensor pipelines. Venue per the author's page: **IEEE Transactions on Field Robotics, 2024** (not RA-L). Authors: Jonas Frey, Manthan Patel, Deegan Atha, Julian Nubert, David Fan, Ali Agha, Curtis Padgett, Patrick Spieler, Marco Hutter, Shehryar Khattak. Project page: [alphaxiv.org/abs/2402.19341](https://alphaxiv.org/abs/2402.19341).

**RoadRunner M&M** — [arXiv:2409.10940](https://arxiv.org/abs/2409.10940), RA-L, [leggedrobotics.github.io/roadrunner_mm](https://leggedrobotics.github.io/roadrunner_mm). The "M&M" is **multi-range, multi-resolution**: one network serves a far-range low-res map for global planning and a near-range high-res map for local driving, because degraded perception plus sparse LiDAR makes a single scale unworkable. This is the most directly copyable design idea in this family for an edge robot.

### 1.3 Noah (Meta) — could not be verified

I found **no verifiable paper named "Noah"** from Meta. The only "Noah" hits were Huawei Noah's Ark Lab. The verified Meta-lineage outdoor-navigation work is:

- **ViNT: A Foundation Model for Visual Navigation** — Shah et al., CoRL 2023, [arXiv:2306.14846](https://arxiv.org/abs/2306.14846), [general-navigation-models.github.io/vint](https://general-navigation-models.github.io/vint/). A single ViT backbone with a navigation-specific head, pretrained on many navigation datasets, decoding a future context/action.
- **CityWalker** — [arXiv:2411.17820](https://arxiv.org/abs/2411.17820), CVPR 2025, **NYU** (Xinhao Liu, Jintong Li, …, Chen Feng), [ai4ce.github.io/CityWalker](https://ai4ce.github.io/CityWalker). Trained on **2,000+ h of web-scale city walking/driving video + 6 h of expert robot data**; benchmarks against GNM, ViNT, and NoMaD.

If you need a large self-supervised affordance model, ViNT/CityWalker are the citable anchors; treat the "Noah / 33M-param fisheye ViT + IMU" description in the request as **unconfirmed**.

### 1.4 Verified supervised set

| System | Year | Input | Output | Notes / source |
|---|---|---|---|---|
| **METAVerse** | 2023 (ICCV) | RGB(-D) + vehicle states | Dense continuous **cost map** | Meta-learns over *task distributions of terrains* so a single global model generalizes. [arXiv:2307.13991](https://arxiv.org/abs/2307.13991) |
| **ViPlanner** | 2024 (RA-L) | Monocular depth + semantic | 2D **collision cost map** + local plan | Trains against a **differentiable semantic costmap** assigning per-class costs; code [leggedrobotics/viplanner](https://github.com/leggedrobotics/viplanner), [arXiv:2310.00982](https://arxiv.org/abs/2310.00982), [project](https://leggedrobotics.github.io/viplanner.github.io) |
| **Shaban et al.** | 2022 (CoRL) | Camera + proprioception | **4 cost classes: free / low-cost / medium-cost / obstacle** | The canonical "not binary" terrain classifier. [PMLR v164](https://proceedings.mlr.press/v164/shaban22a.html) |
| **3DTTNet / RELLIS-OCC** | 2024/25 | Multi-modal (camera/LiDAR) fused into **voxels** | 4 voxel cost labels: **lethal / medium-cost / low-cost / free** | Rule-based physics pipeline generates labels from step height, slope, unevenness. [arXiv:2412.08195](https://arxiv.org/abs/2412.08195) |
| **LP (Locomotion-Policy-Guided)** | 2022 (IROS) | **3D volumetric** + sparse conv | Traversability cost | A locomotion policy *is* the label oracle — teaches traversability by asking "could the robot walk here?" [arXiv:2203.15854](https://arxiv.org/abs/2203.15854) |
| **Cat** (Frey et al.) | 2026 | Point cloud | Traversability | **Open-vocabulary** category-conditioned costmap; companion to Frey's PhD thesis. [jonasfrey96.github.io](https://jonasfrey96.github.io) |
| **W-RIZZ** | 2024 | Monocular images | Relative traversability | Weakly supervised (few labels), learns *relative* rather than absolute cost. [arXiv:2406.02822](https://arxiv.org/abs/2406.02822) |

**Ranger** (Husarenka et al., RAL 2019, KITTI) and the **TTP** line (Miki et al., IROS 2022; Guan et al., *Autonomous Robots* 2024, [10.1007/s10514-024-10158-4](https://link.springer.com/article/10.1007/s10514-024-10158-4)) are real and canonical but I could not verify parameter/latency numbers for them — treat as **[unverified]**.

---

## 2. Self-supervised / experience-driven learning

### 2.1 The mechanisms that actually work (ranked by how directly they transfer to your problem)

1. **Motion divergence → cost map (ViNL).** [arXiv:2210.14791](https://arxiv.org/abs/2210.14791), ICRA 2023, Fu, Drobny, Hollister, Dragan (CMU). A quadruped explores an apartment; wherever the *commanded* body motion diverges from the *realized* motion, that cell in the local image frame is labelled "high cost". This yields an affordance cost map with **zero manual labelling**, then a locomotion policy consumes it. Code: [github.com/SimarKareer/ViNL](https://github.com/SimarKareer/ViNL). *This is the single most important paper for your use case.*
2. **Motion-image difference → traversability (SCATE).** Dohan, Trulls, Frazzoli, RA-L 2022. Diffs a sparse depth frame against its successor; where a candidate region changed the image, something moved and cost depends on motion magnitude. Handful of seeds → self-improving costmap. [taekyung.me/research/scate](https://taekyung.me/research/scate).
3. **Fused model + failure-event prediction (PrePARE).** [arXiv:2208.00322](https://arxiv.org/abs/2208.00322), IROS 2022 (Dey, Frey, Schilling, Tranzatto, Agha-mohammadi; Caltech/ETH/JPL). Learns a **navigation stack** (mapping → local path planning → terrain scoring) from proprioception and imagery, plus a **contact-state classifier** and an **LSTM that predicts failure events** (ledge/pit/slip) *before* they occur, so the planner can react early. Explicitly aimed at "extreme terrains". This is the closest published thing to "learned safety critic".
4. **How does it *feel*? (HDIF).** [arXiv:2209.10788](https://arxiv.org/abs/2209.10788), ICRA 2023 (Guaman Castro, Triest, …, Scherer, CMU), [mateoguaman.github.io/hdif](https://mateoguaman.github.io/hdif). Associates a **proprioception-derived "feeling" cost** with visual+geometric context to learn a costmap — i.e. treats the robot's bodily experience as the label.
5. **Risk-aware self-training from a short manual drive (LeSTA).** RA-L 2024, [github.com/Ikhyeon-Cho/traversability_learning](https://github.com/Ikhyeon-Cho/traversability_learning), [dataset](https://github.com/Ikhyeon-Cho/urban-traversability-dataset). A few minutes of manual driving generates pseudo-labels; a **risk-aware** loss down-weights uncertain pseudo-labels. Same author maintains the field's best survey list: [awesome-traversability-analysis](https://github.com/Ikhyeon-Cho/awesome-traversability-analysis) — start here.
6. **Representation learning, then linear probe (STERLING).** [arXiv:2309.15302](https://arxiv.org/abs/2309.15302), CoRL 2023, Karnan, Yang, Farkash, Warnell, Biswas, Stone (UT Austin), [hareshkarnan.github.io/sterling](https://hareshkarnan.github.io/sterling). SimCLR-style contrastive pretraining on *unconstrained* robot experience, then **visual traversability estimation from very few labels**. Strong for "I have lots of unlabeled driving video and 20 labelled frames".
7. **Label-free via visual foundation models (Velociraptor).** CoRL 2024, Triest, Sivaprakasam, Scherer et al., [PMLR v270](https://proceedings.mlr.press/v270/triest25a.html). Uses VFM features (DINOv2/CLIP/SAM family) instead of human labels to build a **risk-aware** costmap.
8. **Online adaptation with a handful of labels (SALON).** [arXiv:2412.07826](https://arxiv.org/abs/2412.07826), [theairlab.org/SALON](https://theairlab.org/SALON). Self-supervised **costmap + speedmap** adaptation; claim per secondary review: matching navigation performance with ~**one hand-labelled image plus seconds of own driving**. *(Third-party review claim, not from the paper — verify.)*
9. **Online self-supervision at speed (MTP / WVN).** [arXiv:2305.08510](https://arxiv.org/abs/2305.08510), RSS 2023 (Frey, Mattamala, Chebrolu, Cadena, Fallon, Hutter). Superpixels cut compute so online self-supervised learning runs in real time; reported **2.5 Hz on NVIDIA Orin with unoptimized code**. Journal version: [10.1007/s10514-025-10202-x](https://doi.org/10.1007/s10514-025-10202-x) (2025).
10. **Satellite supervision (2026).** [arXiv:2607.17984](https://arxiv.org/abs/2607.17984), IROS 2026, Sivaprakasam, Triest, Nye, Atha, Khattak, Fan, Wang, Scherer, [theairlab.org/ss_frontiers_iros](https://theairlab.org/ss_frontiers_iros). Generates **satellite traversability maps** to supervise an image-based affordance net, fixing the myopia of purely local self-supervision.

### 2.2 Privileged teacher → onboard student (the dominant 2024–2026 pattern)

- **Pre-conditioned locomotion policies** (Levine et al., 2024). Teacher consumes a **privileged heightmap**; student consumes onboard exteroception; student is trained by **DAgger imitation** to match teacher actions. The paper's thesis is that hand-designed reward shaping fails at *task selection* (deciding what the robot should try), while pre-conditioning the policy on terrain makes the choice fall out. *(I could not retrieve the arXiv ID via search — search for the exact title; commonly cited as 2404.10338 but **[unverified]**.)*
- **Learn to Teach** (ETH RSL, 2024) — [alphaxiv.org/abs/2402.06783](https://www.alphaxiv.org/abs/2402.06783). Privileged teacher + student + a **task-inference network**, with an explicit **cost-to-go map**.
- **SleepWalking** (2026) — [arXiv:2608.30883](https://arxiv.org/html/2608.30883). "Privileged Representation Shaping", explicitly framing exteroceptive terrain info (stairs, gaps, obstacles) as what a blind policy lacks.
- **VIRAL** (2025) — [arXiv:2511.15200](https://arxiv.org/pdf/2511.15200.pdf). Privileged-RL teacher on full state → vision student, zero-shot sim-to-real.

### 2.3 Risk modelling — the formal home of "partially valid but risky"

- **Learned Speed Distribution Map** (Cai, Everett, Fink et al., MIT, 2022) — [arXiv:2203.13429](https://arxiv.org/abs/2203.13429). Predicts a *distribution* of achievable speeds and converts it to a costmap with a **CVaR (conditional value at risk)** term. Motivating example is exactly your problem: *"a robot may be able to drive through soft bushes but not a fallen log."* **This is the single most principled way to encode water as "passable but expensive."**
- **Risk-aware costmaps via IRL** (Triest, Guaman Castro, …, Scherer) — [arXiv:2302.00134](https://arxiv.org/abs/2302.00134).
- **Fan et al., RA-L 2021** — [PMC8740562](https://pmc.ncbi.nlm.nih.gov/articles/PMC8740562/): CVaR-based traversability costmap, "more robust to outliers, better captures tail risks."
- **STEP** (DARPA Subterranean) — [arXiv:2303.01614](https://arxiv.gg/abs/2303.01614). Stochastic traversability evaluation + planning.

### 2.4 Remaining self-supervised landmarks

**TravSUITE** (Abbeel group, RSS 2022) — [roboticsproceedings.org/rss22/p070.pdf](https://www.roboticsproceedings.org/rss22/p070.pdf): offline IRL cost estimation + online model-based RL with a **terrain uncertainty map**. **V-STRONG** — [arXiv:2312.16016](https://arxiv.org/abs/2312.16016). **WayFASTER** — [arXiv:2402.00683](https://arxiv.org/abs/2402.00683) (sensor-fusion, self-supervised). **Resilient legged local navigation** — [arXiv:2310.03581](https://arxiv.org/abs/2310.03581), ICRA 2024: end-to-end traverse-under-degraded-perception.

---

## 3. Water as a first-class traversability class

**Bottom line: there is no strong published system that learns water *from experience alone*.** The literature splits into (a) water as a *semantic/marine perception* problem, and (b) water as a *risk/likelihood* problem. Nobody has closed the loop.

### 3.1 Water perception (you need a detector)

| Work | Year | What it gives you |
|---|---|---|
| **MARVIS: Motion & Geometry Aware Real and Virtual Image Segmentation** (Wu, Lin, Negahdaripour, Fermüller, Aloimonos — UMD) | [arXiv:2403.09850](https://arxiv.org/abs/2403.09850) | Real-vs-virtual **water-surface segmentation** robust to reflection/refraction; ships **AquaSim**, a simulator that generates realistic air-water-interface training data. Motivation is explicitly that reflection + refraction + irregular flow break perception near water. |
| **Self-supervised monocular depth on water via specular-reflection prior** | [arXiv:2404.07176](https://arxiv.org/abs/2404.07176) | Depth on water scenes without stereo; introduces the **WRS** water-reflection dataset rendered in Unreal Engine 4. |
| **UDepth** (fast monocular depth for underwater robots) | [arXiv:2209.12358](https://arxiv.org/abs/2209.12358) | Edge-friendly depth for low-cost underwater vehicles. |
| **Through-Water Stereo SLAM with Refraction Correction** (Suresh, Westman, Kaess, RA-L 2019) | [cs.cmu.edu/~kaess/pub/Suresh19ral.pdf](https://cs.cmu.edu/~kaess/pub/Suresh19ral.pdf) | Geometric model of the air-water interface — the principled fix for "the depth net sees a fake bottom." |
| **WDNet** (USV water depth perception) | 2023 | Water-environment-aware stereo depth. |
| **Path Planning in Physically Viable World Models** | [arXiv:2607.00673](https://arxiv.org/abs/2607.00673) | 3D-Gaussian world model with volumetric **hazard encoding**; evaluated on a real field site under **simulated flooding at multiple severity levels**. Directly models flooding as changing traversability over time. |
| **Rankin et al., JPL** (classic) | 2006 | [ro-botics.jpl.nasa.gov](https://www-robotics.jpl.nasa.gov/media/documents/asc2006-rankin-water-detection-final.pdf) — the original statement of the problem: water bodies are terrain hazards; deep water destroys UGV electronics. |

### 3.2 Water depth labels — the datasets

- **FRED: Flooded Road Environments Dataset** (2026) — [arXiv:2605.22018](https://arxiv.org/abs/2605.22018). Claimed to be the **first multimodal driving dataset dedicated to flooded roads**. This is the closest thing to a water-depth supervision set.
- **WaterScenes** — [arXiv:2307.06505](https://arxiv.org/abs/2307.06505). 4D radar + camera for driving *on* water surfaces.
- **VLM fine-tuning for cm-level flood depth** (2026) — [arXiv:2608.07562](https://arxiv.org/abs/2608.07562). Continuous street-level flood depth from images.
- **Geometric flood depth: segmentation + DEM** (2026) — [arXiv:2605.08521](https://arxiv.org/abs/2605.08521). Fuses semantic segmentation with digital elevation models to get volumetric depth from imagery.
- Driving datasets with a `water` semantic class usable for pretraining: **Mapillary Vistas**, **BDD100K** ([arxiv 2004.06320 is A2D2; BDD100K CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.pdf)), **WildDash 2**, **IDD**. Water is a *semantic* class there, not a *depth* class.
- **A2D2** (Audi) — [arXiv:2004.06320](https://arxiv.org/abs/2004.06320), 6 cameras + 5 LiDAR, 3D semantic + instance segmentation, CC BY-ND 4.0.

### 3.3 Amphibious hardware (context, not perception)

WorMa, an undulatory robot that shifts internal fluid to cross land/steps/water — [NYU Tandon, Sept 2026](https://engineering.nyu.edu/news/robot-shifts-its-own-weight-cross-land-steps-and-water). Relevant because it shows water is treated as a *locomotion mode switch*, not a perception problem.

---

## 4. Semantic / affordance taxonomy

**There is no accepted 5-class standard.** The published taxonomies, in increasing specificity:

1. **Binary free / not-free.** Ranger-lineage work; the baseline everything else argues against.
2. **Binary + separate geometric channels.** Free space (planar, no geometry) vs. negative obstacle (missing geometry) vs. positive obstacle. This is what DARPA Subterranean / CERBERUS teams standardize on, and it is the reason pits are *separable* at all.
3. **4 discrete cost classes — the dominant published taxonomy:**
   - Shaban et al., CoRL 2022 — **free / low-cost / medium-cost / obstacle** ([PMLR v164](https://proceedings.mlr.press/v164/shaban22a.html))
   - RELLIS-OCC (3DTTNet, 2024) — **lethal / medium-cost / low-cost / free**, per voxel ([arXiv:2412.08195](https://arxiv.org/abs/2412.08195))
4. **Continuous learned cost.** METAVerse, RoadRunner, ViPlanner, ViNL, METAVerse. Smooth but destroys the "is it a pit?" question.
5. **Class-agnostic + semantic, jointly (2026).** **Trinity** ([arXiv:2605.27644](https://arxiv.org/abs/2605.27644), UZH/Auterion) — one transformer learns semantic classes *and* a class-agnostic traversability head, trained on synthetic data, explicitly to stop requiring robot-specific class maps. **This is the direction to watch for a {floor, stairs, pit, water, obstacle} head.**

Supporting taxonomy work: **Zech & Boeker, "A Survey of Affordance in Robotics"** ([arXiv:2105.06706](https://www.alphaxiv.org/abs/2105.06706) / [iis.uibk.ac.at](https://iis.uibk.ac.at/public/papers/Zech-2017-AB.pdf)); **Chen, Li, Pathak "Learning to Move with Affordance Maps" (A2L, ICLR 2020)** — [arXiv:2001.02364](https://arxiv.org/abs/2001.02364), [Prof-pengyin/A2L](https://github.com/Prof-pengyin/A2L) — the canonical "affordance map for navigation rather than geometry" paper; **CAPABILITY-aware traversability (Cat)** — [capability-aware-traversability.github.io](https://capability-aware-traversability.github.io/) — conditions cost on *embodiment*, since the same terrain is free for one platform and lethal for another.

---

## 5. Efficient / point- and voxel-based architectures

- **3D point clouds:** Cylinder3D ([arXiv:2008.01550](https://arxiv.gg/abs/2008.01550)) cylindrical/asymmetric 3D conv for LiDAR segmentation; LP ([arXiv:2203.15854](https://arxiv.org/abs/2203.15854)) uses **sparse convolutions over 3D volumes** for traversability — the right primitive if you already have a LiDAR. CurveCloudNet ([arXiv:2303.12050](https://arxiv.org/abs/2303.12050)) exploits LiDAR's 1D range-image structure, which is far cheaper than full 3D.
- **SSM/Mamba for 3D:** OccMamba ([github.com/jmwang0117/Occ-Mamba](https://github.com/jmwang0117/Occ-Mamba)) for semantic occupancy; RS3Mamba for remote sensing ([arXiv:2404.02457](https://arxiv.org/abs/2404.02457)). Linear-complexity SSMs are attractive at 10–20 Hz because they avoid attention's quadratic cost.
- **⚡ The concrete deployment number you want:** **LiteViLNet** ([arXiv:2605.21007](https://arxiv.org/abs/2605.21007), 2026) — a MobileNetV3 RGB encoder **plus a 0.12M-parameter depth-wise-separable geometry encoder**, multi-scale fusion. On **Jetson Orin NX**: PyTorch FP16 **22.18 FPS**, TensorRT FP16 **68.73 FPS**; **96.74% F-score / 93.68% IoU** on the ORFD benchmark. That is a directly copyable template for a small RGB+depth terrain net.
- Also confirmed: **LRASPP MobileNetV3-Large** used for drivable-surface + path-intersection detection on a **Jetson Nano** (USF 2026 project). LR-ASPP is the standard trick for a fast, accurate MobileNet segmentation head.
- Context for why not a VLA: **RT-2 = 55B params, 1–3 Hz on cloud TPUs**; **OpenVLA = 7.5B, ~6 Hz on an RTX 4090** (a desktop GPU, not onboard); **π₀ ≈ 3.3B** ([efficient-embodied-ai.github.io](https://efficient-embodied-ai.github.io/), [particula.tech comparison](https://particula.tech/blog/openvla-vs-pi0-vs-smolvla-vs-groot-n1)). Surveys: [A Survey on Efficient VLA Models](https://arxiv.org/html/2510.24795v2), [Embodied Foundation Models at the Edge](https://arxiv.org/abs/2603.16952) (deployment constraints: memory traffic, latency variability, thermal envelope).

---

## 6. Code and datasets

**Curated list (start here):** [Ikhyeon-Cho/awesome-traversability-analysis](https://github.com/Ikhyeon-Cho/awesome-traversability-analysis)

| Resource | URL |
|---|---|
| ViNL (affordance cost map, self-sup) | [github.com/SimarKareer/ViNL](https://github.com/SimarKareer/ViNL) |
| ViPlanner (semantic costmap planner) | [github.com/leggedrobotics/viplanner](https://github.com/leggedrobotics/viplanner) |
| Physical terrain parameter decoder (friction/softness) | [github.com/leggedrobotics/physical_terrain_parameter_learning](https://github.com/leggedrobotics/physical_terrain_parameter_learning) · [paper 2408.16567](https://arxiv.org/abs/2408.16567) |
| LeSTA (self-sup risk-aware traversability) | [code](https://github.com/Ikhyeon-Cho/traversability_learning) · [dataset](https://github.com/Ikhyeon-Cho/urban-traversability-dataset) |
| Elevational mapping (CPU, ROS1/2) | [github.com/Ikhyeon-Cho/FastDEM](https://github.com/Ikhyeon-Cho/FastDEM) |
| ORFD off-road freespace benchmark (12,198 LiDAR frames) | [github.com/chaytonmin/Off-Road-Freespace-Detection](https://github.com/chaytonmin/Off-Road-Freespace-Detection) · [arXiv:2206.09907](https://arxiv.org/abs/2206.09907) · **ORAD-3D released 2025-12** |
| **GrandTour** (ANYmal-D multimodal, 49 missions, HF-hosted) | [github.com/leggedrobotics/grand_tour_dataset](https://github.com/leggedrobotics/grand_tour_dataset) · [arXiv:2602.18164](https://arxiv.org/abs/2602.18164) |
| GOOSE unstructured-outdoor dataset + ICRA 2025 challenge | [arXiv:2310.16788](https://arxiv.org/abs/2310.16788) · [challenge report arXiv:2505.11769](https://arxiv.org/abs/2505.11769) |
| GO: The Great Outdoors (adds thermal + radar) | [arXiv:2501.19274](https://arxiv.org/abs/2501.19274) |
| GA3T ground-aerial traversability (Husky + UAV) | [arXiv:2605.06478](https://arxiv.org/abs/2605.06478) |
| **Survey of datasets for unstructured outdoor perception** | [arXiv:2404.18750](https://arxiv.org/abs/2404.18750) |
| Survey of off-road datasets + traversability analysis | [ouci.dntb.gov.ua/works/4aBVeGR6](https://ouci.dntb.gov.ua/works/4aBVeGR6) |
| A2D2 (Audi) | [a2d2-dataset.github.io](https://a2d2-dataset.github.io) |
| SALON / Velociraptor | [theairlab.org/SALON](https://theairlab.org/SALON) |
| Distilling Global Traversability Priors | [theairlab.org/ss_frontiers_iros](https://theairlab.org/ss_frontiers_iros) |
| URMA multi-embodiment locomotion | [github.com/nico-bohlinger/one_policy_to_run_them_all](https://github.com/nico-bohlinger/one_policy_to_run_them_all) |

**Synthetic generation (Isaac/Gazebo/Unity):**
- **Isaac Sim Replicator** is the standard pipeline — randomizes scene, lighting, materials, poses, intrinsics; emits RGB + depth + semantic/instance seg + normals. [strands-labs.github.io](https://strands-labs.github.io/robots-sim/simulation/domain-randomization), [NVIDIA blog](https://perspectives.nvidia.com/physical-ai/omniverse/task/blog/synthetic-data-pipeline-robot-perception).
- **Isaac Lab terrain generators** are the substrate for heightmap-conditioned locomotion; terrain-gen docs in [leggedrobotics/sru-navigation-sim](https://github.com/leggedrobotics/sru-navigation-sim/blob/main/docs/TERRAIN_AND_GOALS.md).
- **AquaSim** (from MARVIS) specifically simulates air-water interfaces — the only sim I found that generates water-surface training data.
- **TRIDENT**/synthetic multimodal off-road generation framework (ECU et al., J. Intell. Robot. Syst. 2025) — [10.1007/s10846-025-02340-2](https://link.springer.com/content/pdf/10.1007/s10846-025-02340-2.pdf).

---

## 7. The 8 most important papers

1. **ViNL — [arXiv:2210.14791](https://arxiv.org/abs/2210.14791)** (ICRA 2023). Trains an affordance cost map with **zero labels** by finding image cells where commanded motion diverged from realized motion, then feeds it to a locomotion policy. Proof that the signal you need for pits and water is already in the robot's joints.
2. **PrePARE — [arXiv:2208.00322](https://arxiv.org/abs/2208.00322)** (IROS 2022). A learned map-plan-score stack plus an **LSTM that predicts failure events (ledge/pit/slip) before they happen**, turning traversability into explicit forward risk.
3. **Risk-Aware Off-Road Navigation via a Learned Speed Distribution Map — [arXiv:2203.13429](https://arxiv.org/abs/2203.13429)** (2022). Predicts a speed *distribution* and penalises with **CVaR** — the correct formalism for "wadeable but expensive" water.
4. **RoadRunner M&M — [arXiv:2409.10940](https://arxiv.org/abs/2409.10940)** (RA-L 2025). One net, **multi-range and multi-resolution** traversability + elevation maps, built for degraded perception. Best architectural template for an edge robot.
5. **Velociraptor — [PMLR v270](https://proceedings.mlr.press/v270/triest25a.html)** (CoRL 2024) + **SALON — [arXiv:2412.07826](https://arxiv.org/abs/2412.07826)** + **Distilling Global Traversability Priors — [arXiv:2607.17984](https://arxiv.org/abs/2607.17984)** (IROS 2026). The 2024–2026 arc: use frozen visual foundation models to get **label-free** risk-aware traversability, then adapt online with almost no human labelling, then distil a *global* (satellite) prior into an onboard net.
6. **3DTTNet / RELLIS-OCC — [arXiv:2412.08195](https://arxiv.org/abs/2412.08195)** (2024). Voxel-level **4-cost** labels (lethal/medium/low/free) generated by a physics rule pipeline from step height, slope, and unevenness — the most reusable *label generator* found.
7. **Shaban et al. — [PMLR v164](https://proceedings.mlr.press/v164/shaban22a.html)** (CoRL 2022). Establishes that terrain should be classified into **4 cost classes**, not a binary mask, because robot-terrain interaction is continuous.
8. **Identifying Terrain Physical Parameters from Vision — [arXiv:2408.16567](https://arxiv.org/abs/2408.16567)** (RA-L 2024, Chen/Frey/Zhou/Miki/Martius/Hutter). Predicts **friction and deformability** from vision — i.e. the non-geometric hazards (slippery, soft) that a pure geometric classifier cannot represent. Directly the right frame for "mud vs. sand vs. water."

**Honourable mentions (2026, the current frontier):** **PIVOT** — [arXiv:2609.20983](https://arxiv.org/abs/2609.20983) (Jiao, Zhao, Sahak, Barfoot, U. Toronto): VLM + physical priors for off-road terrain assessment; reported **human rescues on a 4-mile route dropping 11 → 3** *(secondary source; verify in paper)*. **RECAST** — [arXiv:2609.32595](https://arxiv.org/abs/2609.32595) (Korea Univ. + KAIST AI): "VLM Semantic Recasting + VFM Spatial Grounding → actionable cost map." **Trinity** — [arXiv:2605.27644](https://arxiv.org/abs/2605.27644) (UZH): one transformer, class-agnostic **and** semantic terrain heads from synthetic data. Both PIVOT and RECAST are exactly the pattern you want: **a heavy teacher offline, a cheap student onboard.**

---

## 8. What the literature actually says about PITS and WATER from self-supervision

### 8.1 PITS — there *is* a workable recipe

Pits are the one hazard where self-supervision genuinely works, and the mechanisms are consistent across papers:

- **Motion divergence (ViNL).** Commanding forward motion into a hole produces large forward slip and zero height gain. The ViNL divergence signal is *maximally* diagnostic for negative obstacles — a pit produces the largest possible commanded-vs-realized mismatch of any terrain type. This is why pit detection is the easiest class to bootstrap self-supervised.
- **Absence of contact / no-return (PrePARE, SCATE).** A pre-contact state classifier infers "expected ground contact that never happened." PrePARE additionally runs this through an LSTM that *predicts* the failure event ahead of time, so a pit is anticipated from a few frames of approach rather than confirmed after the fall.
- **Post-hoc label mining from falls (V-STRONG, LeSTA, RL²).** After a fall or a stuck event, the terrain cells in the traversed region are relabelled high-cost. RL² ([Margolis & Agrawal](https://arxiv.org/abs/2002.11177)) is the archetype: deliberately induce failures, mine the affordance map from the failure region, and add a latent "self-stuck"/"stairs" hazard code.
- **Geometry fallback.** Pits are the one class that raw geometry nails: a depth *discontinuity* plus **LiDAR no-return** is close to unambiguous. Combine it with the self-supervised signal and the class becomes near-trivial.

**Honest gaps:** I found **no** published system that names "pit" as an explicit output class learned end-to-end from experience. The field detects "negative obstacle," "failure event," or "high cost" instead, and lumps pits together with cliffs and drops. If you need a named `pit` class, you will be building it — but every ingredient exists.

### 8.2 WATER — the literature is thin, and that is the real finding

**No work was found in which a robot learns that a surface is water purely from experience and failure/success signals.** Specifically absent:

- No work where water is learned as a *terrain material* from wading experience.
- No work using traction/slippage as a water cue (this is the most obvious idea and it is unclaimed — see recommendations).
- Water is handled as a **semantic** class by marine/aerial work (MARVIS, A2D2, Mapillary) or as a **risk** to be avoided by planning work (JPL 2006, PIVOT, physically-viable world models). The two are never unified.

**Why self-supervision struggles with water specifically**, and what to exploit:
- Water produces a **weak, graded** failure: the robot doesn't fall, it just gets slower, slips, and may work fine. Binary failure signals are near-silent. This is why the CVaR/speed-distribution formulation ([arXiv:2203.13429](https://arxiv.org/abs/2203.13429)) matters — you need a *graded* cost, not a success bit.
- Water is **geometrically ambiguous**: refraction makes the true bottom appear shallower/deeper, and the surface is at a *different* place than any texture suggests. MARVIS and the refraction-SLAM literature ([Suresh et al.](https://cs.cmu.edu/~kaess/pub/Suresh19ral.pdf)) show that static monocular geometry is unreliable here. **Therefore water is the one class you should NOT expect a depth camera to give you.**
- Water is, however, the class best suited to **temporal + motion cues** (surface motion, reflection shimmer, ripple parallax) — which is exactly MARVIS's "Motion & Geometry Aware" design, and exactly what a small recurrent or frame-differencing head can capture cheaply.

---

## 9. Concrete design recommendation

**Target: 5-class {floor, stairs, pit, water, obstacle} at 10–20 Hz on an edge NPU.**

### 9.1 Why not a VLA — with numbers
RT-2 is 55B at 1–3 Hz on a cloud TPU; OpenVLA is 7.5B at ~6 Hz on an RTX 4090; π₀ is ~3.3B ([efficient-embodied-ai.github.io](https://efficient-embodied-ai.github.io/)). All are desktop-or-cloud. A terrain classifier is a *dense, per-pixel, high-rate* task — the exact opposite of a sparse, low-rate, language-conditioned task. Use a heavy model **offline as a teacher**, never onboard.

### 9.2 Architecture (target: 2–6M params, INT8, ≤50 ms/frame)

**Two-branch, late-fusion, LR-ASPP head.** Copy LiteViLNet's proven split ([arXiv:2605.21007](https://arxiv.org/abs/2605.21007)): a **MobileNetV3-Large RGB encoder** + a **~0.12M-parameter depth-wise-separable geometry encoder** (depth/normal/height), fused at multiple scales. That design already hits 96.74% F on ORFD and **68.73 FPS FP16 TensorRT on Jetson Orin NX**. Add an LR-ASPP decoder — the standard fix for MobileNet segmentation quality (used in the Jetson-Nano off-road work).

**Output: two heads, not one.**
- **Head A — 5-way softmax** {floor, stairs, pit, water, obstacle} for the interpretable class the planner reasons about.
- **Head B — continuous cost + log-variance** (aleatoric uncertainty). This is what lets you abstain instead of guessing, and it's the standard output of METAVerse/RoadRunner/CVaR work. Feed `mean + λ·σ` to the planner.

Follow the **4-cost precedent** (Shaban; RELLIS-OCC) for the *planner* even if the perception head is 5-way: map your 5 classes onto `{free, low-cost, medium-cost, lethal}` at the cost-map layer. Don't make the network learn absolute cost — let it learn identity, and let a small hand-written table own embodiment-specific cost. [Cat](https://capability-aware-traversability.github.io/) argues exactly this: cost is embodiment-dependent, so it should be a late, cheap, adjustable stage.

### 9.3 The one design decision that matters most: **don't ask depth for `pit` and `water`**

- **`pit`: geometry-first.** Train the pit head on **depth discontinuity + LiDAR no-return**, with the RGB branch as a weak auxiliary. A pit has literally no surface for a depth net to hallucinate onto, which is why depth nets over-smooth holes. Add an explicit "drop-off" detector: run a min-pool over the predicted elevation map and flag cells whose local minimum falls below the robot's support plane. This is a ~20-line geometric post-process and it will beat any learned head.
- **`water`: motion-first, not geometry-first.** Refraction makes static geometry unreliable ([MARVIS](https://arxiv.org/abs/2403.09850), [refraction SLAM](https://cs.cmu.edu/~kaess/pub/Suresh19ral.pdf)). Give the water branch a **short temporal receptive field**: frame differencing or a 2–4 frame GRU on the RGB stream, plus the geometric context that water is *low, flat, and specular*. Do not rely on a single-frame depth net to find it.

### 9.4 Supervision strategy (three tiers, in order)

1. **Bootstrapping with a frozen foundation model (free labels).** Run DINOv2/VFM features through a linear/light head exactly as in Velociraptor ([PMLR v270](https://proceedings.mlr.press/v270/triest25a.html)), or distil from a VLM offline in the RECAST/PIVOT style ([arXiv:2609.32595](https://arxiv.org/abs/2609.32595), [arXiv:2609.20983](https://arxiv.org/abs/2609.20983)). This is the highest-leverage move and it costs no field data.
2. **Privileged teacher → onboard student.** Train a teacher on a **simulated heightmap** with ground-truth class labels generated by a rule pipeline (RELLIS-OCC's approach: step height → stairs, no-return → pit, low+specular → water, slope/unevenness → cost). Distil to the depth-student via **DAgger on motion divergence**, ViNL-style. Synthesise the teacher in Isaac Lab/Replicator; generate water surfaces in **AquaSim** or a Unity/Unreal water shader, since generic sims do not model an air-water interface.
3. **Online self-supervised correction.** Run ViNL-style motion divergence and LeSTA-style risk-weighted self-training *in the field*, with a risk term that down-weights uncertain pseudo-labels. This is what keeps the water class honest — the robot is the only thing that can tell you a puddle was 8 cm deep and merely slow rather than 60 cm deep and dangerous.

### 9.5 The water-specific self-supervision signal nobody has claimed

**Use drag/traction loss, not failure.** For a wheeled or legged platform, measure commanded-vs-realized velocity **and the mechanical power/energy actually expended** to achieve it. Mud, sand, and shallow water all produce a characteristic **high energy, moderate speed, no fall** signature. Bucket by (Δv, power) into a graded cost rather than a binary, and water separates from mud because the *ratio* differs: water reduces effective mass/friction without the shear-resistance signature of mud. Then let the CVaR formulation ([arXiv:2203.13429](https://arxiv.org/abs/2203.13429)) turn that graded cost into a conservative plan. This is the concrete mechanism the literature is missing, and it costs you no labels.

### 9.6 Deployment checklist

- Export INT8 with TensorRT; LiteViLNet's 68.73 FPS FP16 → comfortable 20–30 Hz INT8 headroom on Orin NX.
- Run perception at a **fixed low rate (10–20 Hz) and fuse temporally** — a per-frame independent classifier cannot see motion, and motion is your only reliable water cue.
- Do the elevation-drop pit check on CPU (FastDEM-style, [github.com/Ikhyeon-Cho/FastDEM](https://github.com/Ikhyeon-Cho/FastDEM)) so the network is never the only line of defence against a hole.
- Multi-range like RoadRunner M&M if you also need far-field planning: one weight-shared net, two resolutions, instead of two models.

---

## 10. Open items I could not resolve

1. **URMA (traversability)** — no verified source. Name collides with biomedical U-Mamba and with Bohlinger's locomotion URMA. Pull the RA-L PDF directly before citing.
2. **Noah (Meta)** — no verifiable paper found. Use ViNT ([arXiv:2306.14846](https://arxiv.org/abs/2306.14846)) and CityWalker ([arXiv:2411.17820](https://arxiv.org/abs/2411.17820)) instead.
3. **Pre-conditioned locomotion policies** — arXiv ID commonly given as 2404.10338 but **unverified**; the mechanism is corroborated by many papers.
4. All latency/parameter numbers in §1 and §5 that aren't attributed above are **[unverified]** — this environment could not fetch full texts.
5. `web_fetch` being fully DNS-blocked means I worked from snippets only. A pass with working fetch on PIVOT, RECAST, Trinity, and the RELLIS-OCC paper would firm up §7 considerably.
