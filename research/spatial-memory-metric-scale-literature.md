# Reconstruct-Current-Geometry literature + the monocular metric scale question

**Scope:** the "reconstruct the geometry that is actually there" family (multi-view / monocular feed-forward 3D
reconstruction and metric depth), **not** generative future-prediction world models.

**Constraint set used for every verdict:** Windows, Intel Arc A770 16 GB, no CUDA, OpenVINO + ONNX Runtime,
old Xeon CPU, **VRChat running on the same GPU during VR play**, hard rule = never fabricate metric distances.

**Method / honesty note.** The `web_fetch` tool is unavailable in this sandbox (every hostname resolves to a
non-public IP), so **no primary PDF, README or model card was read directly**. Every number below comes from a
search snippet; where a snippet was truncated or ambiguous I say so and mark it **UNVERIFIED**. Nothing here was
inferred from memory. There is also a systematic speed/VRAM caveat: essentially all published numbers are for
**A100 / H100 / A40 / M1 Max / RTX 4090**. *None of them transfer to an Arc A770*, and no Arc benchmark for any
model in this report was found. Treat every "inference time" column as a relative ordering only.

***

## 1. DUSt3R / MASt3R / MonST3R (Naver Labs Europe and collaborators)

| Name            | Org                                                                | Year                                          | What it does                                                                                                                                                                                              | Params/VRAM                                                                                                                                              | Inference time                                                                                                         | Open weights (license)                                                                                    | ONNX/OpenVINO                                                                                                                                                                                                | Verdict                                                                                                                                           |
| --------------- | ------------------------------------------------------------------ | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DUSt3R**      | Naver Labs Europe (+ ETH/KIT)                                      | 2024, CVPR 2024                               | Dense unconstrained stereo 3D reconstruction: pairwise point-map regression + global alignment; cameras + point clouds, **up to one unknown global scale**                                                | ViT-L encoder + ViT-B decoder; checkpoint **2.28 GB** (`nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt`); VRAM **UNVERIFIED** (no official number retrieved) | No official per-pair latency retrieved → **UNVERIFIED**                                                                | Yes — **CC BY-NC-SA 4.0** (non-commercial)                                                                | ONNX: community fork `ibaiGorordo/dust3r-pytorch-inference-minimal` exists and documents the *code surgery* required (fixed image size, pre-computed RoPE, `expm1`→`exp-1`). OpenVINO: **no evidence found** | Not metric. Scale-free by construction, so it cannot satisfy the hard rule without an external anchor. Too heavy for a shared 16 GB Arc during VR |
| **MASt3R**      | Naver Labs Europe (+ ETH)                                          | 2024 (arXiv 2406.09756; **venue UNVERIFIED**) | Adds a matching head + fast reciprocal matching on top of DUSt3R → dense pixel correspondences with 3D grounding; still scale-ambiguous                                                                   | Same ViT-L/B recipe (`MASt3R_ViTLarge_B`); checkpoint sizes not retrieved                                                                                | **\~1.x s per image pair on an A40** (per Speedy MASt3R abstract snippet; exact value truncated → treat as ≈1 s scale) | Yes — repo license **CC BY-NC-SA 4.0**; checkpoints carry *separate* dataset terms (`CHECKPOINTS_NOTICE`) | No ONNX/OpenVINO evidence                                                                                                                                                                                    | Accurate matching, but scale-free + non-commercial + \~1 s/pair on an A40 → far slower on A770                                                    |
| **MASt3R-SLAM** | Murai, Dexheimer, Davison (Imperial/Oxford)                        | 2025, CVPR 2025                               | Real-time **monocular dense SLAM** built bottom-up from MASt3R; pointmap-based tracking + local fusion, uncalibrated, "no assumption on a fixed or parametric camera model beyond a unique camera center" | Not retrieved → **UNVERIFIED**                                                                                                                           | **\~15 FPS on RTX 4090** (secondary source)                                                                            | Code/weights follow MASt3R lineage → **CC BY-NC-SA 4.0** (UNVERIFIED for this repo specifically)          | No evidence                                                                                                                                                                                                  | **Monocular ⇒ up-to-scale.** Even at 15 FPS on a 4090 it does not solve metric scale, and an A770 is many times slower                            |
| **MonST3R**     | Zhang et al. (Berkeley/Adobe/Meta mix; **affiliation UNVERIFIED**) | 2024 (arXiv 2410.03825; **venue UNVERIFIED**) | DUSt3R fine-tuned with optical flow to handle **dynamic** scenes: per-frame point maps + camera poses in the presence of motion                                                                           | ViT-L based; checkpoints via the repo's `download_ckpt.sh`; size not retrieved                                                                           | Not retrieved → **UNVERIFIED**                                                                                         | Yes — listed as **CC BY-NC(-SA)** non-commercial (plumbline bundling note; Naver headers in the tree)     | No evidence                                                                                                                                                                                                  | Not metric; useful conceptually only (dynamic scenes), not for honest meters on Arc                                                               |

**Additional member worth knowing:** `plumbline` (kmatzen) is an evaluation harness whose own code is Apache-2.0
but which **bundles DAGE / CUT3R / DUSt3R / MASt3R / MonST3R under NonCommercial licenses**, i.e. the whole
distribution is non-commercial. Useful as a ready-made evaluator, but it inherits the NC constraint.

***

## 2. VGGT (Meta AI + Oxford VGG) and its 2025–2026 successors

| Name                       | Org                                                                                | Year                                                             | What it does                                                                                                                                                                                                                   | Params/VRAM                                                                                                                                                                                                                                       | Inference time                                                                                                                                                                                                                                                                     | Open weights (license)                                                                                                                                                                                                                                                                                 | ONNX/OpenVINO                                                                                                                                                                                                                                                                                      | Verdict                                                                                                                                                                                                                                                                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **VGGT**                   | Meta AI + Oxford VGG                                                               | 2025, **CVPR 2025 Best Paper**                                   | One feed-forward transformer → camera extrinsics/intrinsics, point maps, depth maps, point tracks from 1…hundreds of views                                                                                                     | **≈1.2 B params** (paper §3.4); input capped at **518 px**; DPT head adds ≈0.03 s + 0.2 GB per frame. **Reported 1.88 GB for 1 frame vs 7–8 GB actually measured by a user on A100** (issue #81); \~4 GB for 10 frames per maintainer (issue #11) | Paper scale: "less than a second" for 1–2 views; **200 frames = 8.75 s / 40.6 GB on H100 with FA3** at 336×518 (secondary summary of paper Table 9). TensorRT deployment blog claims **5 images → 3D in 62 ms** (GPU per blog = RTX 5090 class) → **not transferable, UNVERIFIED** | Yes — **"VGGT License v1", Last Updated 2025-07-29** = Meta non-commercial research license + Acceptable Use Policy                                                                                                                                                                                    | ONNX: two community exporters exist (`akretz/vggt-onnx` with "tiny modifications" to upstream; `samolego/VGGT-1B-ONNX`, author states "this repo is not finished"). OpenVINO: **no evidence found**                                                                                                | **Not metric**: users explicitly struggle with scene-scale on ScanNet++ (issue #64). Non-commercial + multi-GB + view-count-quadratic attention → infeasible on a shared 16 GB Arc during VR. A VRAM-optimised fork (`harry7557558/vggt-low-vram`) reports 150 images in 8 GB / 1100 images in 32 GB, **but that is a CUDA fork** |
| **VGGT-Ω (Omega)**         | Meta AI + Oxford VGG                                                               | **2026, CVPR 2026 Oral** (Best-Paper finalist), arXiv 2605.15195 | Scaling sequel: **Register Attention** replaces global self-attention with bottleneck communication through register tokens → "70 % less memory" (training); scales to **10 B params**; better static + dynamic reconstruction | `VGGT-Omega-1B-512` = 1 B params, 512 px; README benchmarks **A100, 624×416 peak GPU memory, end-to-end incl. weight load** — **exact GB values NOT retrieved → UNVERIFIED**                                                                      | Not retrieved → **UNVERIFIED**                                                                                                                                                                                                                                                     | Yes — **FAIR Noncommercial Research License v1**                                                                                                                                                                                                                                                       | No ONNX/OpenVINO evidence                                                                                                                                                                                                                                                                          | Better memory profile than VGGT, but still non-commercial, still not a metric guarantee, still no Arc evidence. Third-party run reports say "needs an A100-class GPU"                                                                                                                                                             |
| **VGGT-Long**              | DengKaiCQ et al.                                                                   | 2025 (arXiv 2507.16443)                                          | "Chunk it, loop it, align it": pushes VGGT to kilometre-scale RGB sequences                                                                                                                                                    | Builds on VGGT → same 1.2 B backbone                                                                                                                                                                                                              | Not retrieved                                                                                                                                                                                                                                                                      | Repo exists; inherits VGGT licensing (non-commercial)                                                                                                                                                                                                                                                  | No evidence                                                                                                                                                                                                                                                                                        | Not relevant at room scale                                                                                                                                                                                                                                                                                                        |
| **FastVGGT**               | Deng, Ti, Xu, Yang, Xie                                                            | 2025 (arXiv 2509.02560)                                          | **Training-free acceleration** of the VGGT-style visual geometry transformer                                                                                                                                                   | VGGT backbone                                                                                                                                                                                                                                     | Speed-ups reported, exact ms not retrieved                                                                                                                                                                                                                                         | Ships VGGT License v1 (per `FastVGGT/LICENSE.txt`)                                                                                                                                                                                                                                                     | No evidence                                                                                                                                                                                                                                                                                        | Speed helps; licensing/metric unchanged                                                                                                                                                                                                                                                                                           |
| **AVGGT**                  | Sun et al.                                                                         | **2026, CVPR 2026** (highlight)                                  | Layer-wise dissection of global attention; K/V subsampling to accelerate VGGT/π³                                                                                                                                               | VGGT-scale                                                                                                                                                                                                                                        | Not retrieved                                                                                                                                                                                                                                                                      | **Code "None (Repository not disclosed)"** per paper-notes                                                                                                                                                                                                                                             | No code → moot                                                                                                                                                                                                                                                                                     | Interesting theory; unusable today                                                                                                                                                                                                                                                                                                |
| **π³ (Pi3)**               | Wang, Zhou, Zhu et al. (Shanghai AI Lab/ByteDance-ish; **affiliation UNVERIFIED**) | **ICLR 2026**, arXiv 2507.02114? (**ID UNVERIFIED**)             | Permutation-equivariant visual geometry: no fixed reference view, so it is robust to view ordering                                                                                                                             | Not retrieved                                                                                                                                                                                                                                     | Not retrieved                                                                                                                                                                                                                                                                      | Repo `yyfz/Pi3` (≈2 k stars); license not verified                                                                                                                                                                                                                                                     | No evidence                                                                                                                                                                                                                                                                                        | Scale is still ambiguous; no Arc evidence                                                                                                                                                                                                                                                                                         |
| **Depth Anything 3 (DA3)** | ByteDance Seed                                                                     | 2025-11 (arXiv 2511.10647); ICLR 2026 per third-party            | A **plain transformer** (DINOv2 ViT backbone + DPT head) using a unified **depth-ray** representation; single model for any number of views; poses + depth + 3DGS support in the Nested series                                 | **Any-View zoo: DA3-Small 0.08 B, DA3-Base 0.12 B, DA3-Large 0.35 B, DA3-Giant 1.15 B.** Metric series: **DA3Metric-Large 0.35 B (334 M)**                                                                                                        | Not retrieved for DA3Metric-Large → **UNVERIFIED**. A Replicate deployment and a ROS2 TensorRT wrapper exist, so it does run in practice                                                                                                                                           | Yes. Licenses per DA3 model-zoo table: **DA3-Giant / DA3-Large = CC BY-NC 4.0; DA3-Base (and Small) = Apache 2.0.** **Conflict flagged:** the HF card metadata for `depth-anything/DA3METRIC-LARGE` shows `license: apache-2.0`, which contradicts the family table → **verify before commercial use** | **Best export story of anything here**: `Heliosoph/da3metric-large-onnx` (ONNX export), `HeliosophLLC/DatumV` `scripts/export-da3metric.ps1`, and `ika-rwth-aachen/ros2-depth-anything-v3-trt` which loads `models/DA3METRIC-LARGE.onnx`. **ONNX ≠ OpenVINO**: no OpenVINO evidence for DA3 either | **The most plausible learned metric component for this project** — small enough (0.35 B ViT-L), ONNX-exportable, and *requires focal length*, which the plugin can obtain exactly from OpenVR (see §6). Still not a measurement                                                                                                   |
| **Dens3R**                 | Dens3R team                                                                        | 2025 (arXiv 2507.16290)                                          | Feed-forward foundation model taking **unposed** images → high-quality point maps, multi-view + multi-resolution                                                                                                               | Not retrieved                                                                                                                                                                                                                                     | Not retrieved                                                                                                                                                                                                                                                                      | Repo exists; license unverified                                                                                                                                                                                                                                                                        | No evidence                                                                                                                                                                                                                                                                                        | Same class; not metric                                                                                                                                                                                                                                                                                                            |
| **MV-DUSt3R+**             | Tang et al.                                                                        | 2025, CVPR 2025                                                  | Single-stage multi-view scene reconstruction from sparse views **in \~2 s** (paper title claims 2 s; table snippet shows 0.89/1.54 s RGB-only)                                                                                 | DUSt3R-scale                                                                                                                                                                                                                                      | **0.89–1.54 s** (paper table snippet)                                                                                                                                                                                                                                              | Repo exists; DUSt3R lineage → likely NC                                                                                                                                                                                                                                                                | No evidence                                                                                                                                                                                                                                                                                        | Speed is not the blocker; scale is                                                                                                                                                                                                                                                                                                |

***

## 3. MapAnything (Meta + CMU, 2025)

| Name            | Org           | Year                               | What it does                                                                                                                                                                                                                                                                                                                                                                                                       | Params/VRAM                                                                                                                                                                                                                                                       | Inference time                                                                                                     | Open weights (license)                                                                                                                                                             | ONNX/OpenVINO             | Verdict                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------- | ------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MapAnything** | Meta AI + CMU | 2025 (arXiv 2509.13414, Sept 2025) | **Unified feed-forward** ***metric*** **3D reconstruction**: ingests 1..N images **plus optional** intrinsics / poses / depth / partial reconstructions, and regresses factored metric geometry + cameras. "12+ reconstruction tasks". It is a *framework*: the model table lists `mapanything` (518 px, **DINOv2 base**), `mapanything_ablations`, `modular_dust3r`, **`vggt`** **1 B**, **`vggt_omega`** **1 B** | Default MapAnything = DINOv2-**base** backbone (so smaller than the 1 B VGGT variants); **exact param count and VRAM NOT retrieved → UNVERIFIED**. Official note: `memory_efficient_inference` "trades off speed for more views (**up to 2000 views on 140 GB**)" | Paper profiles all models "on an **H200-140 GB** GPU" (Fig. S.1) — **no per-image seconds retrieved → UNVERIFIED** | Yes, **dual-licensed by design**: code = **Apache-2.0**; weights = `facebook/map-anything` (**CC-BY-NC 4.0**, research) **and** `facebook/map-anything-apache-v1` (**Apache-2.0**) | No ONNX/OpenVINO evidence | Conceptually the closest match to the project's needs (metric + accepts calibration/depth as *inputs*), and it has an Apache-2.0 weights variant — but its published envelope is a **140 GB H200**, there is no Arc evidence, and "metric" behaviour on **uncalibrated stylized game renders** is **UNVERIFIED**. There is also a public GitHub issue where the Apache model performed *worse* than the NC model for one user's case → the two variants are not equivalent |

***

## 4. CUT3R — persistent state (CVPR 2025) — the most architecturally relevant one

| Name             | Org                                                                                              | Year                                  | What it does                                                                                                                                                              | Params/VRAM                                                                                                                                                                                                                                                                                                                              | Inference time                                                                                                        | Open weights (license)                                                                             | ONNX/OpenVINO             | Verdict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CUT3R**        | Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei Efros, Angjoo Kanazawa (Berkeley/Google) | 2025, **CVPR 2025**, arXiv 2501.12387 | Online **stateful recurrent** 3D reconstruction: a persistent state is updated with each new image and read out to produce point maps / camera poses for the whole stream | ViT-Large image encoder (initialised from DUSt3R weights) + **ViT-Base decoders**, 16×16 patches, 224²→512 px training. **STATE = 768 tokens** (paper "Implementation Detail"). Per-token dimension and VRAM **NOT retrieved → UNVERIFIED** (`emergentmind` notes the state is "a set of learnable tokens", consistent but not decisive) | Not retrieved on any GPU → **UNVERIFIED**. Repo notes they parallelise the encoder, so memory is **linear** in frames | Yes — repo `CUT3R/CUT3R`; listed as **NonCommercial (CC BY-NC-SA)** by the plumbline bundling note | No ONNX/OpenVINO evidence | **Two independent worries.** (a) **Drift / length generalisation**: the follow-up **TTT3R** (arXiv 2509.26645) states outright that modern recurrent 3D models "**degrade significantly when applied beyond the training context length, revealing limited length generalization**" — i.e. the persistent state does drift, and TTT3R exists to patch it via test-time training. (b) **Scale**: the CUT3R abstract claims the state yields "**metric-scale pointmaps**", but the mechanism is not visible in any snippet I could retrieve and the lineage is scale-ambiguous DUSt3R → **treat the metric claim as UNVERIFIED** |
| *CUT3R variants* | <br />                                                                                           | <br />                                | <br />                                                                                                                                                                    | <br />                                                                                                                                                                                                                                                                                                                                   | <br />                                                                                                                | <br />                                                                                             | <br />                    | **G-CUT3R** (arXiv 2508.11379) adds camera/intrinsic guidance; **TTT3R** (arXiv 2509.26645) fixes length generalisation with test-time training. Both non-commercial lineage                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

**Why this matters for the project:** the persistent-state design is the right *shape* for a rolling VR session
(no re-running a batch over all history), but CUT3R itself gives no license-clean, scale-honest, Arc-portable
artifact today.

***

## 5. Metric monocular depth — is any of it *actually* metric and how accurate?

### 5a. The table

| Name                                    | Org                                     | Year                                                                | Metric by construction?                                                                                                                                                                      | Params / size                                                                                                                                                | Inference time                                                                                                                                                                                              | Open weights (license)                                                                                                                                                               | ONNX/OpenVINO                                                                                                                                           | Verdict                                                                                                                                                                                                                                                                                                         |
| --------------------------------------- | --------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Metric3D v2**                         | Yin, Hu et al. (Tsinghua/USTC/…)        | 2024 (arXiv 2404.15506; TPAMI-family)                               | **Yes, but needs camera intrinsics** — the "canonical camera transformation" is precisely how arbitrary intrinsics are handled; without intrinsics you get canonical-space depth, not metres | DINOv2 encoders (ViT-S / ViT-L / **ViT-giant2**), DPT decoders. Exact params **UNVERIFIED**. ViT-g checkpoints are multi-GB                                  | Not retrieved → **UNVERIFIED**                                                                                                                                                                              | Yes — code **BSD-2-Clause "for non-commercial usage"**; **weights not clearly licensed** → flag                                                                                      | No ONNX/OpenVINO evidence                                                                                                                               | Its own README reports **ViT-giant2 δ1≈0.989 (NYU) / ≈0.987 (KITTI)** — the snippet's column layout is ambiguous, so treat those as *approximate, self-reported*. Needs intrinsics; ViT-g is far too big for a shared 16 GB Arc                                                                                 |
| **UniDepth / UniDepthV2**               | Piccinelli et al. (ETH Zürich + Huawei) | UniDepth 2024 (CVPR 2024); **UniDepthV2 2025** (arXiv 2502.20110)   | **Yes — and self-calibrating**: it predicts camera intrinsics itself, so "no extra information about the scene or camera" is needed                                                          | ViT-L/14 (≈4.05 M? — the `q4km.ai` figure of 4,052,968 looks like a mis-parsed metadata field → **do not trust it**); ViT-L class ≈0.3 B. VRAM not retrieved | Not retrieved for UniDepthV2 alone → **UNVERIFIED**                                                                                                                                                         | Yes — **CC BY-NC 4.0** (HF model cards for v1/v2)                                                                                                                                    | No ONNX/OpenVINO evidence                                                                                                                               | Accuracy on standard benchmarks is genuinely strong: **δ1 = 0.989 (NYUv2), 0.982 (KITTI)** in the Video-Depth-Anything comparison table. But **δ1 only asks "within 1.25×"** — see 5c                                                                                                                           |
| **Apple Depth Pro**                     | Apple ML Research                       | 2024 (arXiv 2410.02073)                                             | **Yes — "metric, with absolute scale, without relying on metadata"** (it estimates focal length internally)                                                                                  | ViT-L, **2.2 GB** checkpoint                                                                                                                                 | Paper's selling point: "**less than a second**" (paper framing; the measured GPU is a **V100-class** in Apple's report → **not transferable to A770**)                                                      | Yes — **`apple-amlr`** **(Apple ML Research Model License)**, i.e. research-only terms, *not* MIT                                                                                    | No ONNX/OpenVINO evidence                                                                                                                               | Very sharp boundaries; **δ1 = 0.953 (NYUv2) but only 0.822 (KITTI)** in the VDA table. License is the practical blocker                                                                                                                                                                                         |
| **Depth Anything V2 — metric variants** | DepthAnything / TikTok                  | 2024                                                                | **Yes, needs focal length.** Fine-tuned on **Hypersim (indoor) and VKITTI2 (outdoor)** with an added focal-length conditioning module                                                        | Encoder Small/Base/Large (as today)                                                                                                                          | VDA's own metric models on A100: **S = 7.5 ms / 6.8 GB fp16; L = 14 ms / 23.6 GB fp16** (that is Video-Depth-Anything, a *different* model, but the same family and the only concrete A100 latency I found) | Small encoder = **Apache-2.0**; Base/Large = **CC BY-NC 4.0** (family convention; not re-verified for the metric checkpoints)                                                        | OpenVINO has official DA/DA-V2 notebooks — **but they document** ***relative*** **depth**, not the metric variants                                      | **The plugin already uses the relative Small model correctly.** The metric variant would give approximate metres, but issue #152 in the DA-V2 repo makes the dependency explicit: to output metres the model "must somehow estimate" the focal length. We can *supply* focal length instead of letting it guess |
| **MoGe / MoGe-2**                       | Microsoft Research                      | MoGe CVPR 2025 **Oral**; **MoGe-2 NeurIPS 2025** (arXiv 2507.02546) | **Yes — claims "metric-scale 3D point map" zero-shot from one image**; MoGe (v1) is explicitly **affine-invariant with unknown scale**, and MoGe-2 is the metric successor                   | ViT-S ≈**35 M** (DINOv2 ViT-S backbone), ViT-L variant; MoGe-2 also has a "-normal" variant adding normals                                                   | **MoGe-2 ViT-S on Apple M1 Max: 212 ms (MPS) / 1259 ms (CPU)** (dev.to) — **no discrete-GPU number, no A770 number**                                                                                        | Repo is MIT + Apache-2.0; **weights' license is contested** — HF says MIT, but a maintainer-adjacent issue (#98) says the licence of the provided weights is "not clearly mentioned" | **Yes, practical**: ONNX-adjacent ports exist — `litert-community/MoGe-2-LiteRT` (TFLite/GPU), a Tenstorrent `tt-nn` port, and community ONNX pipelines | **Important negative datum: in the Video-Depth-Anything table, MoGe-2-L scores δ1 = 0.967 on NYUv2 but only 0.415 on KITTI.** A 0.415 δ1 means it is essentially wrong on that domain. That is the single most useful warning in this whole report: *"metric" models are domain-fragile*                        |
| **MoGe-3**                              | Kong, Li, Wang et al. (Microsoft)       | **2026** (arXiv 2607.17967)                                         | Metric geometry + **Self-Guided Sparse Volumetric Refinement** for fine detail; the stated motivation is that 2D-convolution decoders put image-plane-nearby pixels together in 3D           | Not retrieved                                                                                                                                                | Not retrieved                                                                                                                                                                                               | Repo `microsoft/MoGe` documents it; license unverified                                                                                                                               | Unknown                                                                                                                                                 | Too new to verify; indicative that the frontier moved to *detail*, not to *trust*                                                                                                                                                                                                                               |
| **UniK3D**                              | Piccinelli et al. (ETH/Huawei)          | 2025, CVPR 2025 (arXiv 2503.16591)                                  | Metric monocular 3D for **arbitrary camera models** (fisheye, 360°)                                                                                                                          | Not retrieved                                                                                                                                                | Not retrieved                                                                                                                                                                                               | **CC BY-NC 4.0**; third-party notes: "Linux and CUDA are effectively required; macOS and Windows are untested"                                                                       | No                                                                                                                                                      | **Directly relevant to VR lens handling** — but the Windows/CUDA caveat is a red flag for this project                                                                                                                                                                                                          |
| **FoundationGeo**                       | (2026) arXiv 2607.11588                 | 2026                                                                | Two-stage: affine-invariant geometry (DINOv3, 10.2 M samples) then **explicit spatial calibration** to metric; explicitly "bridges relative and metric prediction"                           | Not retrieved                                                                                                                                                | Not retrieved                                                                                                                                                                                               | Not retrieved                                                                                                                                                                        | Unknown                                                                                                                                                 | The *architecture of the fix* (relative → calibrated metric) matches what this project needs, but nothing is verified or portable                                                                                                                                                                               |

### 5b. So: is anything **really** metric zero-shot?

Yes, by claim, in two distinct senses — and the distinction is exactly what the plugin's hard rule turns on:

1. **Self-calibrating metric models** (no camera info needed): **UniDepth/UniDepthV2** (predicts intrinsics),
   **Depth Pro** (predicts focal length), **MoGe-2**. These output metres from a single image with no calibration.
2. **Intrinsics-required metric models**: **Metric3D v2** (canonical-camera transform), **Depth Anything V2
   metric**, **DA3Metric-Large** ("canonical metric depth; **multiplying by focal length gives metric depth**").

Nobody in this list is metric *by measurement*. They are all metric *by learned prior*.

### 5c. Why benchmark numbers like δ1 = 0.98 are **not** good enough for this project's hard rule

- **δ1 = fraction of pixels within a factor of 1.25 of the ground truth.** δ1 = 0.98 still permits a **25 %**
  error on 2 % of pixels, and there is no constraint on *global scale error* at all. The evaluation literature
  says this explicitly: see *Toward A Better Understanding of Monocular Depth Evaluation* /
  *How Should One Evaluate Monocular Depth Estimation?* (arXiv 2510.19814), which documents that metric choice
  changes conclusions.
- **The MMDE survey (arXiv 2501.11841 / MDPI Computers 14(11):502, 2025)** frames metric depth as requiring
  strong learned priors, and an overview of the topic states the core problem plainly: metric depth from a single
  image is **fundamentally ill-posed**.
- **Empirical domain fragility** (the KITTI 0.415 MoGe-2-L result above) plus the *PDE* robustness benchmark
  (arXiv 2507.00981: SOTA monocular depth estimators are "surprisingly fragile" under procedural scene
  perturbations) mean a "metric" model's metres can be systematically off on imagery it has not seen.
- **2026 confirmation that this is still open** — *VFM-Recon* (arXiv 2603.12657) describes VFM depth priors as
  "**scale-ambiguous**" and needing explicit scale alignment under severe domain shifts. *ParaScale*
  (arXiv 2606.19805) restates the gauge argument: monocular reconstruction recovers (T, Z) only up to scale,
  and translation enters depth only through ‖T‖/Z.
- **2026 successors exist but change nothing structural**: MoGe-3, FoundationGeo, DepthART
  (arXiv 2607.17099, scaling foundation depth to *tiny* models — relevant to Arc, unverified).

***

## 6. Stereo metric scale — does a known baseline actually solve it?

**Short answer: yes, and the project's stated blocker ("IPD known only approximately") is weaker than it looks —
but only if the capture pipeline really hands you the per-eye frames.**

### What OpenVR actually gives you (metric, not approximate)

- **`Prop_UserIpdMeters_Float`** is a real OpenVR tracked-device property in **metres**. The OpenVR wiki states
  that whenever it changes, **SteamVR recomputes the** **`HeadFromEyePose`** **matrix** and hands the new matrix to the
  application — the matrix "contains only the simple translation of the eyes based on IPD in the X-direction".
  ⇒ Both the **baseline** and the per-eye **extrinsics** are available from the runtime, exactly.
- **`IVRSystem::GetProjectionRaw`** returns **the tangents of the half-angles** of each eye's frustum
  (left/right/top/bottom). From those plus the render-target size you get metric-consistent pixel focal lengths:
  `f_x = W / (r − l)`, `f_y = H / (t − b)`. ⇒ **intrinsics in pixels, exactly**, no calibration target needed.
- So `Z = f·B/d` becomes a **measurement with real units**, not a learned guess.

### Realistic accuracy (this is arithmetic, not a cited measurement — the formula is standard)

Standard stereo error propagation from `Z = fB/d`:

```
ΔZ = Z² · Δd / (f · B)          ⇒   ΔZ/Z = (Z / (f·B)) · Δd
```

Concrete numbers for a plausible VR HMD (B = 0.063 m, W ≈ 1832 px/eye, HFOV ≈ 95° ⇒ **f ≈ 840 px**,
`f·B ≈ 52.9 px·m`), using **sub-pixel** disparity noise `Δd = 0.2 px` (achievable on noise-free game renders):

| Depth | disparity d | ΔZ (0.2 px)     | ΔZ (1.0 px)      |
| ----- | ----------- | --------------- | ---------------- |
| 1 m   | 52.9 px     | ≈ 4 mm (0.4 %)  | ≈ 1.9 cm (1.9 %) |
| 5 m   | 10.6 px     | ≈ 9 cm (1.9 %)  | ≈ 47 cm (9.4 %)  |
| 10 m  | 5.3 px      | ≈ 47 cm (4.7 %) | ≈ 1.9 m (19 %)   |

⇒ **Metric scale is trustworthy in the near field and degrades quadratically.** This is the standard,
long-known limitation: long-range stereo VO papers state that "for large scene depths, triangulation from a
single stereo pair is inadequate and noisy" (IROS 2013, *Robust Scale Initialization for Long-Range Stereo Visual
Odometry*), and the baseline-to-depth ratio is treated as the governing quantity (RSS 2013, *High Altitude Stereo
Visual Odometry*: "without additional sensing, metric scale is considered lost"). A 2026 treatment
(*Fisheye Stereo Vision: Depth and Range Error*, arXiv 2602.02973) notes the classical ΔZ model
(Hartley & Zisserman) assumes a rectilinear pinhole and a fixed horizontal baseline.

⇒ **Design implication:** the spatial map must carry per-point metric uncertainty and *refuse* to emit a metre
value past a configured threshold (e.g. where ΔZ/Z exceeds \~10 %), rather than extrapolating.

### The honest failure modes of the stereo route here

1. **Does the SteamVR mirror give a genuine per-eye pair?** The stock desktop mirror commonly shows **one eye**;
   Unity/SteamVR discussions confirm people specifically hunt for the "both eyes side by side" mode. Whether the
   captured image is the app's **pre-distortion eye textures** or a compositor-resampled/encoded composite is
   **UNVERIFIED**, and it is the single most important thing to test empirically.
2. **VRChat's own camera ≠ the eye camera.** VRChat's stream/desktop camera has its own FOV; a capture from it is
   not the HMD eye view.
3. **Game FOV vs HMD recommended projection.** If VRChat renders with its own FOV, use the *game's* projection,
   not the HMD's recommended one.
4. **IPD offsets beyond** **`Prop_UserIpdMeters_Float`.** SteamVR 1.0.9 added
   `TrackedDeviceDisplayTransformUpdated` so drivers can specify explicit **4×3 eye-to-head transforms**, with
   the old IPD-property mechanism still supported → there may be per-device eye offsets that the IPD scalar alone
   misses. **Read the transform matrices, not just the scalar.**
5. **Texture-poor / specular / transparent game surfaces** break block matching — but the plugin can *refuse*
   there instead of inventing depth.

### The second honest metric anchor (often overlooked)

OpenVR tracked **HMD and controller poses are in metres**. The magnitude of measured head translation therefore
fixes the scale of any monocular structure — i.e. the capture rig is effectively a *calibrated* camera with
*known metric motion*. Related literature: *Metrically-Scaled Monocular SLAM using Learned Scale Factors*
(MIT, ICRA 2020) — which explicitly benchmarks against **Stereo ORB-SLAM2** as the metric reference; and 2026's
*Metric-scale monocular SLAM via lightweight depth-…* (ScienceDirect, 2026). Combined with DA3Metric-Large
*(needs focal length → we have it exactly)* this is a defensible "known-scale, not predicted-scale" pipeline.

### Also relevant: fusing relative depth with metric anchors

- **Prior Depth Anything / "Depth Anything with Any Prior"** (arXiv 2505.10565): a framework that combines
  **incomplete but metric** depth measurements with a relative depth model to produce complete metric depth.
- **Prompt Depth Anything** (CVPR 2025, `promptda.github.io`): uses a **low-cost LiDAR** prompt for 4K metric
  depth — no LiDAR here, but the *pattern* (metric prompt + relative model) is exactly the safe design.
- **OpenVINO already ships the visual-inertial version of this idea**: *Monocular Visual-Inertial Depth Estimation
  using OpenVINO™* (VIDepth) produces **dense metric-scale depth** by fusing monocular depth with VIO. That is a
  directly portable, Intel-supported precedent for "metric scale from known motion, not from a depth model".

***

## 7. Synthetic / game / VR renders and sim-to-real scale mismatch

| Work                                                                                                                                                                 | Year / venue | Relevance                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hypersim** (via the DA-V2 metric recipe)                                                                                                                           | —            | The DA-V2 metric-indoor model is fine-tuned **on Hypersim, a** ***synthetic*** **photorealistic indoor dataset**, plus VKITTI2 for outdoor. ⇒ *synthetic renders are already partly in-distribution* for the metric variants — but Hypersim is offline ray-traced photoreal, not a real-time stylised game                                                                                                                                              |
| **Playing for Depth** (arXiv 1810.06268)                                                                                                                             | 2018         | Builds a depth dataset **from video games**, arguing open-world games give high-quality depth maps "in the wild". Establishes that game frames are usable depth training data                                                                                                                                                                                                                                                                           |
| **Origin Lab** **`game-depth`** **dataset** + METHODOLOGY                                                                                                            | 2026         | Dense **in-engine z-buffer depth**, normals, camera pose, inputs, **frame-locked across 10 modalities**; 48,615 RGB + engine depth (17,799 used in training after session-balanced split). **Caveat: their depth is encoded as "log-nearness"** — engine z-buffer depth is *non-linear*, so converting to metres needs the near/far planes. This is the most concrete path to **supervised fine-tuning with exact metric ground truth on game renders** |
| **A Framework for Real-Time Stereoscopic VR Conversion of Mainstream Game Output Using Synthetic Supervision and GPU-Resident Inference** (Research Square preprint) | 2026         | Direct precedent for exactly this problem shape: mainstream game output → stereoscopic VR, synthetic supervision, GPU-resident inference. **Preprint, not peer-reviewed → treat as indicative only**                                                                                                                                                                                                                                                    |
| **AerialMetric** (arXiv 2606.29716 / ECCV 2026)                                                                                                                      | 2026         | **52 K real + 16 K synthetic** aerial image-depth pairs. Its premise is the finding this project needs to take seriously: metric models trained on street-view/indoor data show a **"significant domain gap"** when the viewpoint/appearance changes. Exact δ1 deltas **NOT retrieved → UNVERIFIED**                                                                                                                                                    |
| **Domain Decluttering** (arXiv 2002.12114)                                                                                                                           | 2020         | Explicitly studies and mitigates **synthetic→real domain shift for depth**                                                                                                                                                                                                                                                                                                                                                                              |
| **Domain-Transferred Synthetic Data Generation for Improving MDE** (arXiv 2405.01113)                                                                                | 2024         | Uses **game engines (Unreal/Unity)** to generate synthetic depth data and transfer domains                                                                                                                                                                                                                                                                                                                                                              |
| **PDE robustness benchmark** (arXiv 2507.00981)                                                                                                                      | 2025         | SOTA monocular depth is fragile to procedural scene-content perturbations                                                                                                                                                                                                                                                                                                                                                                               |
| **VFM-Recon** (arXiv 2603.12657)                                                                                                                                     | 2026         | VFM depth priors are **scale-ambiguous** and require scale alignment **under severe domain shifts**                                                                                                                                                                                                                                                                                                                                                     |
| **"Can These Views Be One Scene?"** (arXiv 2605.18754)                                                                                                               | 2026         | Shows **VGGT, MASt3R, DUSt3R and Fast3R can hallucinate dense geometry and cross-view support for unrelated images** — the standard no-GT consistency metrics (MEt3R) do *not* reliably catch it. This is the closest thing in the literature to a formal warning about the exact failure the project's hard rule forbids                                                                                                                               |

**Bottom line for §7:** there is **no published evaluation of metric monocular depth on VRChat-like social-VR game
renders** that I could find → **UNVERIFIED / genuine evidence gap**. The transferable evidence says: synthetic
domains are usable for *training* (§5a, game-depth, Hypersim, Playing-for-Depth), but zero-shot absolute scale on
a *new* synthetic domain is not something the literature supports, and the one aerial case study that looks at a
clean viewpoint/appearance shift reports a significant gap.

***

## 8. Can any of this run under OpenVINO on Intel Arc? (honest evidence audit)

**Confirmed to exist:**

- **OpenVINO official notebooks for Depth Anything / DepthAnythingV2** — the DA-V2 notebook describes it as
  "a solution for robust **relative** depth estimation". So *relative* monocular depth on Intel GPUs is a
  first-class, Intel-maintained path (and it is what the plugin already does).
- **ONNX Runtime OpenVINO Execution Provider** — Intel publishes pre-built OV-EP packages for ONNX Runtime,
  targeting Intel CPUs, **GPUs** and NPUs. This is the sanctioned bridge from an ONNX model to an Arc GPU.
- **OpenVINO also ships a** ***metric*****-scale notebook**: *Monocular Visual-Inertial Depth Estimation using OpenVINO*
  (VIDepth) → dense **metric-scale** depth by fusing monocular depth with visual-inertial odometry.
- **OpenVINO has a Stable Fast 3D (image→textured mesh) notebook** — proof Intel invests in 3D, but this is
  object-level generative reconstruction, *not* metric scene reconstruction.

**Explicitly not found (searched repeatedly):**

- **No evidence whatsoever** of DUSt3R, MASt3R, MASt3R-SLAM, MonST3R, CUT3R, VGGT, VGGT-Ω or MapAnything running
  under **OpenVINO**. Their published runtimes are CUDA/TensorRT. This is an evidence gap, not a proven
  impossibility.
- **ONNX export for the multi-view transformer family requires code changes**, which is itself evidence the
  graphs are not export-clean:
  - `ibaiGorordo/dust3r-pytorch-inference-minimal`: assumes constant image size, pre-computes RoPE sin/cos, and
    **replaces** **`expm1`** **with** **`exp − 1`** "to make it possible" to export.
  - `akretz/vggt-onnx`: "**Tiny modifications** have been done to the upstream code to enable ONNX export".
  - `samolego/VGGT-1B-ONNX`: author states the repo "**is not finished**".
    ⇒ Whether OpenVINO's op set covers the result (RoPE-fused attention, dynamic view counts, register tokens) is
    **UNVERIFIED**, and the honest expectation is *significant porting work*.
- **No Arc A770 benchmark for any model in this report.** Every FPS/GB figure above is A100/H100/A40/V100/M1 —
  **explicitly not transferable**.
- **DA3 Metric is the exception with a real ONNX artifact** (`Heliosoph/da3metric-large-onnx`; a ROS2/TRT wrapper
  loads `DA3METRIC-LARGE.onnx`) — and its conversion formula is documented. But note the formula I could retrieve
  (`metric_depth = focal * net_output / 300`) comes from a **third-party playbook site, not the official repo**
  → **do not implement from it; read the official model card / issue #244 first.**

**VRAM reality check for the A770 (16 GB, shared with VRChat):** a single-image, single-view ViT-L + DPT model
such as DA3Metric-Large (\~0.35 B, ≈0.7 GB fp16 weights) is *plausibly* a 1–2 GB inference. The multi-view
transformer family is not: VGGT's own community measurements are 7–8 GB for **one** frame on an A100, its
attention grows with view count, and MapAnything's documented "memory-efficient" envelope is literally
**140 GB**. Given that VRChat must keep running on the same GPU, the multi-view family should be considered
**out of scope during VR play**.

***

# Can monocular metric scale be honestly solved here?

**Yes — but almost certainly not by monocular metric depth, and the project's stated blocker is currently
mis-diagnosed.**

The project says: *"known hard blocker is monocular metric scale."* That is true **only if the only input is one
monocular image**. The moment SteamVR is in the loop, the rig is no longer uncalibrated:

1. **OpenVR hands over metrically exact baseline and intrinsics** (`Prop_UserIpdMeters_Float` +
   per-eye `HeadFromEyePose` transforms + `GetProjectionRaw` tangents). Monocular metric scale is *not* the
   blocker; **the actual blocker is whether the capture path preserves a genuine per-eye stereo pair**.
2. **OpenVR tracked poses are in metres**, giving a second, independent metric anchor for the *trajectory* scale.

### Recommended design (in order of trust)

1. **Metric scale from stereo triangulation only** — `Z = f·B/d`, with `f`, `B` read from OpenVR in metres, and a
   **per-pixel uncertainty** attached (`ΔZ = Z²Δd/(fB)`). Publish a metre value **only** while `ΔZ/Z` is under a
   configured budget (≈10 %); beyond that, degrade the point to *relative/topological* or drop it. Never
   extrapolate metric depth into the far field.
2. **Metric scale from known camera motion** — the measured head/controller translation fixes the scale of any
   monocular structure. This works even where stereo disparity is unusable (far objects, low texture), and it is
   exactly the VIDepth-style fusion that OpenVINO already ships as a notebook.
3. **Learned metric depth as a prior/cross-check, never as the source of a published metre.** If used:
   **DA3Metric-Large** is the best candidate (0.35 B, ONNX exists, needs focal length which we can supply
   exactly) — but its metres are a *learned prediction* on out-of-distribution stylized content and must be
   validated against (1)/(2) on real VRChat captures before any metre output is enabled. Metric3D v2 / UniDepthV2
   / Depth Pro / MoGe-2 are interesting comparisons but are heavier and/or research-licensed.
4. **Explicit refusal policy** (enforce in code, not in prose): a metre value may be emitted **only** if it came
   from stereo triangulation with a known baseline, or from a known-metric pose/size anchor. Every other
   quantity stays dimensionless. This is what makes the project's rule verifiable rather than aspirational.
5. **Optional later step:** fine-tune a small metric model on game-render depth. The Origin Lab `game-depth`
   dataset (in-engine z-buffer depth + pose, 10 modalities) plus the AerialMetric/Playing-for-Depth precedent make
   this a legitimate project — but remember engine z-buffer depth is stored as **log-nearness** and needs the
   near/far planes to become metres.

### Residual uncertainty (explicit)

| #  | Uncertainty                                                                                            | Why it matters                                             | How to close it                                                                                                                    |
| -- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Does the SteamVR mirror yield a genuine per-eye pair with the app's projection?**                    | Everything in step 1 depends on it                         | Empirical test: capture mirror, verify two viewpoints, compare a known geometry's disparity against `GetProjectionRaw` predictions |
| 2  | Does VRChat render the HMD view with its own FOV, or the HMD-recommended projection?                   | A wrong `f` silently corrupts every metre                  | Read the game's submitted projection, or fit `f` from the stereo pair and cross-check                                              |
| 3  | Are there per-eye offsets beyond the IPD scalar?                                                       | SteamVR 1.0.9 added explicit 4×3 eye-to-head transforms    | Use the transform matrices, not only the scalar                                                                                    |
| 4  | **Zero published evidence of any metric model on VRChat-like renders**                                 | Absolute scale error is unbounded and unaudited            | Measure it: build a small ground-truth set from (1) and report δ1 / scale error honestly                                           |
| 5  | **`DA3METRIC-LARGE`** **licence conflict** (HF card `apache-2.0` vs family table `CC BY-NC 4.0`)       | Legal, not technical                                       | Ask upstream / read the repo LICENSE directly                                                                                      |
| 6  | **No Arc A770 evidence for any model here**; ONNX exports of the multi-view family needed code surgery | Feasibility is unproven                                    | Prototype DA3Metric-Large ONNX on the A770 via ONNX Runtime + OpenVINO EP; everything else assume infeasible                       |
| 7  | 16 GB shared with VRChat                                                                               | Even a 2 GB model competes for the same VRAM and bandwidth | Budget-test under live VR; single-view ViT-L only                                                                                  |
| 8  | MoGe weights licence ambiguity (issue #98)                                                             | Legal                                                      | Read the model card, not the issue thread                                                                                          |
| 9  | All reported latencies are A100/H100/A40/V100/M1                                                       | **None transferable to A770**                              | Measure locally; do not plan from these numbers                                                                                    |
| 10 | `metric_depth = focal * net_output / 300` is from a third-party playbook                               | Implementing from it risks a silently wrong scale          | Read the official DA3 model card + issue #244                                                                                      |

### One-line answer

> **Do not try to solve monocular metric scale. Read the metric baseline, intrinsics and poses out of OpenVR,
> triangulate stereo where disparity is reliable, anchor the rest to measured metric camera motion, and treat
> every learned "metric" depth model as an unvalidated prior — with a hard-coded refusal to emit metres otherwise.**

***

## URL list (all sources referenced above)

**DUSt3R / MASt3R / MonST3R**

- <https://github.com/naver/dust3r/blob/main/LICENSE>
- <https://github.com/naver/mast3r>
- <https://github.com/naver/mast3r/blob/mast3r_sfm/CHECKPOINTS_NOTICE>
- <https://huggingface.co/nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt>
- <https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt>
- <https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html>
- <https://openaccess.thecvf.com/content/CVPR2025/papers/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf>
- <https://edexheim.github.io/mast3r-slam>
- <https://learnopencv.com/mast3r-slam-realtime-dense-slam-explained/>
- <https://arxiv.org/html/2412.12392v1>
- <https://arxiv.org/html/2503.10017> (Speedy MASt3R)
- <https://github.com/Junyi42/monst3r>
- <https://monst3r-project.github.io>
- <https://awesome.ecosyste.ms/projects/github.com%2Fkmatzen%2Fplumbline> (bundled NC licence note)
- <https://pypi.org/project/stereodemo/> (DUSt3R weights ≈2.3 GB)
- <https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal>
- <https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_MV-DUSt3R_Single-Stage_Scene_Reconstruction_from_Sparse_Views_In_2_Seconds_CVPR_2025_paper.pdf>
- <https://arxiv.org/pdf/2412.06974>

**VGGT and successors**

- <https://github.com/facebookresearch/vggt>
- <https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt>
- <https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf>
- <https://vgg-t.github.io/>
- <https://github.com/facebookresearch/vggt/issues/81>
- <https://github.com/facebookresearch/vggt/issues/11>
- <https://github.com/facebookresearch/vggt/issues/64>
- <https://vitavision.dev/atlas/vggt>
- <https://github.com/harry7557558/vggt-low-vram>
- <https://github.com/akretz/vggt-onnx>
- <https://huggingface.co/samolego/VGGT-1B-ONNX>
- <https://medium.com/@kynkynkyn/vggt-5-images-3d-in-62ms-tensorrt-optimization-for-cvpr-2025-best-paper-cb6e96e14018>
- <https://github.com/facebookresearch/vggt-omega>
- <https://github.com/facebookresearch/vggt-omega/blob/main/LICENSE>
- <https://arxiv.org/abs/2605.15195>
- <https://vggt-omega.github.io/>
- <https://cvpr.thecvf.com/virtual/2026/oral/40381>
- <https://ehakblog.com/blog/2026/06/02/vggt-omega-paper-review-en>
- <https://github.com/DengKaiCQ/VGGT-Long>
- <https://arxiv.org/html/2509.02560v1> (FastVGGT)
- <https://github.com/mystorm16/FastVGGT/blob/main/LICENSE.txt>
- <https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_AVGGT_Rethinking_Global_Attention_for_Accelerating_VGGT_CVPR_2026_paper.pdf>
- <https://en.papernotes.org/CVPR2026/3d_vision/avggt_rethinking_global_attention_for_accelerating_vggt>
- <https://github.com/yyfz/Pi3>
- <https://yyfz.github.io/pi3/>
- <https://arxiv.org/html/2507.16290v1> (Dens3R)

**Depth Anything 3**

- <https://arxiv.org/html/2511.10647v1>
- <https://github.com/ByteDance-Seed/Depth-Anything-3>
- <https://deepwiki.com/ByteDance-Seed/Depth-Anything-3/1.2-model-zoo-and-pretrained-models>
- <https://huggingface.co/depth-anything/DA3METRIC-LARGE>
- <https://github.com/ByteDance-Seed/Depth-Anything-3/issues/244>
- <https://huggingface.co/Heliosoph/da3metric-large-onnx>
- <https://github.com/HeliosophLLC/DatumV/blob/main/scripts/export-da3metric.ps1>
- <https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt>
- <https://replicate.com/david20321/depth-anything-v3-metric-large>
- <https://fullstackcv.github.io/playbook/depth> (third-party; formula **unverified**)

**MapAnything**

- <https://arxiv.org/html/2509.13414v1>
- <https://arxiv.org/pdf/2509.13414>
- <https://github.com/facebookresearch/map-anything>
- <https://github.com/facebookresearch/map-anything/blob/main/LICENSE>
- <https://github.com/facebookresearch/map-anything/blob/main/CHANGELOG.md>
- <https://deepwiki.com/facebookresearch/map-anything/3.1-model-variants-and-licensing>
- <https://huggingface.co/facebook/map-anything-apache-v1>
- <https://github.com/facebookresearch/map-anything/issues/19>
- <https://map-anything.github.io/>

**CUT3R and recurrent-state follow-ups**

- <https://arxiv.org/abs/2501.12387>
- <https://arxiv.org/html/2501.12387v1>
- <https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html>
- <https://github.com/CUT3R/CUT3R>
- <https://cut3r.github.io/>
- <https://arxiv.org/html/2509.26645v4> (TTT3R)
- <https://rover-xingyu.github.io/TTT3R/>
- <https://arxiv.org/html/2508.11379v1> (G-CUT3R)

**Metric monocular depth**

- <https://arxiv.org/html/2404.15506v4> (Metric3D v2)
- <https://github.com/YvanYin/Metric3D>
- <https://huggingface.co/zachL1/Metric3D> (BSD-2 non-commercial quote)
- <https://arxiv.org/html/2502.20110v2> (UniDepthV2)
- <https://github.com/lpiccinelli-eth/UniDepth>
- <https://huggingface.co/lpiccinelli/unidepth-v1-vitl14> (cc-by-nc-4.0)
- <https://github.com/PozzettiAndrea/ComfyUI-UniDepth> (CC-BY-NC 4.0 note)
- <https://arxiv.org/pdf/2410.02073> (Depth Pro)
- <https://machinelearning.apple.com/research/depth-pro>
- <https://huggingface.co/apple/DepthPro/blob/main/LICENSE> (apple-amlr)
- <https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md>
- <https://github.com/DepthAnything/Depth-Anything-V2/issues/152>
- <https://github.com/DepthAnything/Depth-Anything-V2/issues/88>
- <https://github.com/DepthAnything/Video-Depth-Anything> (δ1 table + A100 latency/VRAM table)
- <https://github.com/microsoft/moge>
- <https://arxiv.org/html/2507.02546v1> (MoGe-2)
- <https://proceedings.neurips.cc/paper_files/paper/2025/hash/336572db3e99930814d6b328d4220cb6-Abstract-Conference.html>
- <https://github.com/microsoft/MoGe/issues/98>
- <https://arxiv.org/pdf/2607.17967v2> (MoGe-3)
- <https://huggingface.co/litert-community/MoGe-2-LiteRT>
- <https://huggingface.co/changh95/moge-2-p150>
- <https://arxiv.org/abs/2503.16591> (UniK3D)
- <https://openaccess.thecvf.com/content/CVPR2025/papers/Piccinelli_UniK3D_Universal_Camera_Monocular_3D_Estimation_CVPR_2025_paper.pdf>
- <https://heatdrop.ai/repo/lpiccinelli-eth/UniK3D> (CC BY-NC 4.0; Windows untested)
- <https://arxiv.org/html/2501.11841v2> + <https://mdpi.com/2073-431X/14/11/502> (MMDE survey)
- <https://arxiv.org/pdf/2510.19814v3> (how to evaluate monocular depth)
- <https://arxiv.org/pdf/2507.00981v1> (PDE robustness)
- <https://arxiv.org/pdf/2607.11588v2> (FoundationGeo)
- <https://arxiv.org/html/2607.17099v1> (DepthART)
- <https://arxiv.org/html/2505.10565v1> (Prior Depth Anything)
- <https://groups.csail.mit.edu/rrg/papers/greene_icra20.pdf> (metrically-scaled monocular SLAM)

**Stereo metric scale**

- <https://github.com/ValveSoftware/openvr/wiki/vr::ITrackedDeviceServerDriver-Overview> (`Prop_UserIpdMeters_Float`, HeadFromEyePose)
- <https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetProjectionRaw>
- <https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h>
- <https://github.com/ValveSoftware/openvr/commit/5d0574bf6473130d25dd296ad30206ccd148590b> (TrackedDeviceDisplayTransformUpdated)
- <https://www.roboticsproceedings.org/rss09/p03.pdf> (High Altitude Stereo VO — baseline-to-depth)
- <https://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_IROS_2013/media/files/1203.pdf> (Long-range stereo scale init)
- <https://arxiv.org/html/2602.02973> (Fisheye Stereo Vision: Depth and Range Error)
- <https://www.ni.com/docs/en-US/bundle/ni-vision/page/what-to-expect-from-a-stereo-vision-system.html>
- <https://arxiv.org/abs/2606.19805> (ParaScale — monocular gauge ambiguity)
- <https://www.sciencedirect.com/science/article/pii/S2090447926004272> (metric-scale monocular SLAM, 2026)

**Synthetic / game / VR renders and sim-to-real**

- <https://arxiv.org/html/1810.06268v1> (Playing for Depth)
- <https://huggingface.co/datasets/originlab/game-depth>
- <https://huggingface.co/datasets/originlab/game-depth/raw/main/METHODOLOGY.md>
- <https://huggingface.co/datasets/originlab/game-recordings-v3>
- <https://arxiv.org/html/2405.01113v1> (domain-transferred synthetic data via game engines)
- <https://ar5iv.labs.arxiv.org/html/2002.12114> (Domain Decluttering — synthetic→real depth shift)
- <https://arxiv.org/pdf/2603.12657.pdf> (VFM-Recon — scale-ambiguous priors under domain shift)
- <https://arxiv.org/html/2606.29716v2> + <https://arxiv.org/abs/2606.29716> (AerialMetric, ECCV 2026)
- <https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026>
- <https://kuieless.github.io/AerialMetric-ECCV2026-page/>
- <https://researchsquare.com/article/rs-9399054/v1.pdf> (game output → stereo VR, 2026 preprint)
- <https://arxiv.org/abs/2605.18754> + <https://mvp18.github.io/3d-consistency-metrics/> (3D foundation models hallucinate)
- <https://huggingface.co/datasets/VR-VLA/VR-egodex-depth-cache-full> (DA-V2 Metric Indoor used on VR egocentric data)
- <https://hubertshum.com/publications/vr2022depth/files/vr2022depth.pdf> (depth for VR/360 content)

**OpenVINO / ONNX on Intel**

- <https://docs.openvino.ai/2024/notebooks/depth-anything-v2-with-output.html>
- <https://docs.openvino.ai/2024/notebooks/depth-anything-with-output.html>
- <https://docs.openvino.ai/2023.3/notebooks/246-depth-estimation-videpth-with-output.html>
- <https://docs.openvino.ai/2024/notebooks/stable-fast-3d-with-output.html>
- <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html>
- <https://docs.openedgeplatform.intel.com/2026.2/edge-ai-suites/robotics-ai-suite/components/ai_resources/openvino/models/model_depthanythingv2.html>
- <https://github.com/openvinotoolkit/openvino/releases/tag/2026.1.0>
- <https://github.com/lhl/intel-inference> (Arc/Lunar Lake toolchain notes)

