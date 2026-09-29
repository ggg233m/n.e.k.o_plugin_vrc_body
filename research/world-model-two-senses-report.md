# "World Model" — Industry Senses vs. This Project (late-2026 research)

**Method caveat / uncertainty.** `web_fetch` is blocked in this sandbox (every host resolves to a
non-public proxy IP); I bypassed it with a Python/OpenSSL fetcher, so the primary pages cited below
*were* read directly. However only 1 of 3 delegated research threads completed (geometry + metric
scale); the **VR/VRChat-agent** and **Intel-Arc-limits** threads failed, so those two sections rest on
my own searches and are flagged. No Arc A770 benchmark exists for any model here.

**The three buckets (keep these separate — the project needs 2 and 3):**

| Bucket | Question it answers | Useful here? |
|---|---|---|
| **A. Generative / predictive WM** | "What will I see next?" | **No** (past bucket, wrong direction) |
| **B. Feed-forward geometry** | "What is the 3D shape right now?" | **Partly** — but heavy + scale-ambiguous |
| **C. Persistent spatial memory** | "Where have I been / what is where?" | **Yes — this is the project's job** |

The project's existing naming is already ambiguous and will collide: `backend/world_model.py` is
**W1 world identity** ("which world am I in"), while `recorder/world_model_build.py` produces the
**map** artifact `world_model.json`. `Docs/SLAM代码地图.md` documents this collision explicitly.
Adopting the ML meaning of "world model" on top of both would make the term unusable in-repo.

---

## 1. Generative / learned world models (Bucket A)

| Name | Org | Year | What it does | Hardware | Open weights | Verdict |
|---|---|---|---|---|---|---|
| **Cosmos 3** (Predict/Transfer/Reason lineage; Cosmos-Predict2.5) | NVIDIA | 2024→2026 | Omni world foundation model (Mixture-of-Transformers): reasoning VLM + world + action generation; Predict = future video, Transfer = sim2real, Reason = VLM over physical scenes | Tuned for NVIDIA; RTX PRO 6000 Blackwell-class cited; CUDA/TensorRT | **Yes** — models downloadable (e.g. `nvidia/Cosmos-Predict2.5-2B`, 2B DiT) | **No.** CUDA-only; wrong bucket |
| **Genie 3** | Google DeepMind | 2025-08 | Text → interactive world navigable **real-time at 24 fps**, consistency "a few minutes", limited action space | Undisclosed cloud (TPU-class) | **No** | **No** |
| **Project Genie** | Google | 2026-01-29 | Product wrapper exposing Genie 3 to Google AI Ultra subscribers (US first) | Cloud only | No | **No** |
| **Marble** | World Labs (Fei-Fei Li) | 2025-11-12 GA | Multimodal world model → **persistent, editable 3D worlds** (Gaussian splats) from text/image/video | Cloud service, tiered pricing | **No** | **No for runtime.** Only conceivable as an *offline asset/scene generator* |
| **GAIA-1 / GAIA-2** | Wayve | 2023 / 2025-03 | Driving video prediction, multi-view controllable scenarios | Undisclosed training; not consumer | **No** | No |
| **DreamerV3** | Hafner et al. (DeepMind) | 2023 / Nature 2025 | Latent WM + RL; learns behaviours "in imagination". 200M default; **trained on a single A100** | 1× A100 to train; small models → CPU inference feasible | Code open (`danijar/dreamerv3`); UNVERIFIED whether checkpoints ship | Conceptually relevant, practically out of reach |
| **Dreamer 4** | Hafner, Yan, Lillicrap | 2025-09 | Minecraft WM, "**real-time interactive inference on a single GPU**" via shortcut forcing; first agent to get diamonds from offline data only | "Single GPU" — **model/GPU not specified = UNVERIFIED** | **No** | No locally |
| **DIAMOND** | Alonso et al. | 2024 | Diffusion WM as Atari engine, 64×64 | Consumer GPU feasible | Yes (`eloialonso/diamond`, MIT) | No — wrong bucket |
| **GameNGen** | Google | 2024 | DOOM neural engine, ~20 fps | Single TPU-v5 | Code/weights not released (UNVERIFIED) | No |
| **Oasis / Oasis 500M** | Decart + Etched | 2024 | Keyboard-conditioned autoregressive DiT Minecraft | Full model H100-class; **500M downscaled variant** | **Yes** — `Etched/oasis-500m`, MIT | No — downscaled ≠ interactive quality; CUDA PyTorch |
| **Decart Mirage / Odyssey-2 Pro, Odyssey-3** | Decart / Odyssey | 2025-2026 | Cloud "world transformation" and interactive world models, WebRTC low-latency | Cloud only | **No** | No |

**Bucket-A verdict:** none of this runs on an Arc A770, and **none of it is the right tool.** These
models answer "what comes next" — the project needs "what is true now". Using a generative model for
spatial facts would *manufacture* the hallucination the hard rule forbids.

## 2. Reconstruct-current-geometry (Bucket B) + monocular metric scale

(Delegated thread; report at `research/spatial-memory-metric-scale-literature.md`; all figures from
search snippets, none from primary PDFs, and **no Arc benchmark exists for any of these**.)

- **DUSt3R / MASt3R / MonST3R** (Naver, 2024) — open weights but **CC BY-NC-SA**; scale-ambiguous
  (up to one unknown global scale) ⇒ cannot give honest metres.
- **MASt3R-SLAM** (CVPR 2025) — real-time dense SLAM, **~15 FPS on RTX 4090**; monocular ⇒ up-to-scale.
- **VGGT** (Meta+Oxford, CVPR 2025 Best Paper) — ~1.2B; community-reported **7–8 GB for one frame**
  on A100; **non-commercial** license; not metric.
- **VGGT-Ω** (CVPR 2026 Oral) — exists; ~70% less memory via Register Attention; still
  non-commercial. **MapAnything** (Meta, 2025) — genuinely metric, **Apache-2.0 weights variant
  exists**, but profiling envelope is H200/140 GB-class.
- **CUT3R** (CVPR 2025) — persistent **768-token state**; follow-up **TTT3R** documents that such
  recurrent 3D models "degrade significantly beyond the training context length" ⇒ **it drifts**.
- **Depth Anything 3 Metric-Large** — ~0.35B, ONNX export exists, needs focal length (OpenVR can give
  it exactly). Best small candidate; **licence metadata conflicts (Apache-2.0 vs CC BY-NC)**.
- **Sharpest warning:** *"Can These Views Be One Scene?"* (2026) shows VGGT/MASt3R/DUSt3R/Fast3R
  **can hallucinate geometry** and standard no-GT metrics don't catch it. Also MoGe-2-L scores 0.967 δ1
  on NYUv2 but **0.415 on KITTI** — metric depth models are domain-fragile, and δ1=0.98 still permits
  25% error with no global-scale guarantee.

**The reframe worth acting on:** if SteamVR/OpenVR is in the loop, "monocular metric scale" is the
wrong diagnosis. `Prop_UserIpdMeters_Float` is in **metres**, and `IVRSystem::GetProjectionRaw` yields
exact pixel focal length — so scale can come from **stereo triangulation with a metrically known
baseline**, not from a monocular prior. Residual uncertainty: whether the mirror capture yields a
genuine per-eye pair (the repo's default is WGC window capture → likely a composited single eye), and
error grows as ΔZ ∝ Z², so metres are only honest while ΔZ/Z ≲ 10%.

## 3. Persistent spatial memory (Bucket C) — the correct bucket

Topological place graphs, 3D scene graphs (ConceptGraphs, Hydra, Kimera), and hybrids such as
*Learning 3D Persistent Embodied World Models* (arXiv 2505.05495), which pairs video diffusion with
**explicit 3D spatial memory**. The transferable lesson is the *explicit, non-hallucinating* memory
structure — not the diffusion. **Bucket C needs no learned model here;** it is symbolic bookkeeping the
repo already has (place groups, loop verify, pose graph, navmesh). Reported "Navigation Pixie"
(ISMAR 2025) — **UNVERIFIED**, found by a failed thread.

## 4. Is there *any* honest use for generative world models here?

1. **Offline synthetic style-matched data** (Marble/Cosmos cloud, not local) — genuinely possible for
   stress-testing the refusal policy or measuring a depth model's domain gap on VRChat-like renders.
   But the project's blockers are metric scale and place identity, not data volume. Also: generating
   VRChat-derived content raises TOS/consent questions. Cost: cloud, not Arc.
2. **Offline testing/replay — do NOT use a generative model.** Deterministic replay of recorded
   frames is cheaper, reproducible, and cannot hallucinate. A neural engine would *add* the exact
   failure mode under test.
3. **Data augmentation for the detector/embedder** — marginal, and again needs CUDA.
4. **Monocular metric scale — no.** Generative WMs do not resolve scale; they hallucinate geometry.
5. **DreamerV3-style latent controller** — conceptually the closest honest fit (a small latent WM used
   as a *controller*, not a mapper), but it requires training infrastructure and a task-specific
   reward this project does not have.

## 5-6. Modest-hardware navigation & VR/VRChat

DreamerV3 is the only credible "runs on modest hardware" data point (single-A100 *training*; small
models plausibly CPU-capable at inference — **no measured Arc numbers found**). DINO-WM (ICML 2025)
and 2026 latent-WM planning work exist but again target NVIDIA. **No evidence found** of any
world-model navigation running under OpenVINO on Intel Arc, and **no evidence found** of generative
world models applied to VRChat or social-VR agents. LLM-based VR agents (2024-2026) use
symbolic/graph/topological memory, not predictive world models. Treat "no evidence found" as the
finding, not as a gap to fill with speculation.

## Bottom line

- **Bucket A: unusable and wrong.** Cloud-only, CUDA-only, predicts futures.
- **Bucket B: right idea, too heavy.** Multi-GB CUDA transformers are out while VRChat renders VR on
  the same 16 GB. The only borderline-plausible local learned component is a ~0.35B metric-depth model
  as an *unvalidated prior*, never as the source of a published metre.
- **Bucket C: already the project's design, and correct.** Keep it symbolic.
- **Actionable:** get metric scale from OpenVR baselines/focal lengths + stereo triangulation, anchor
  the rest to measured metric camera motion, and enforce the "no metre without a metric anchor" rule
  **in code** so the hard rule is verifiable rather than aspirational.

**Flagged uncertainty:** Dreamer 4's "single GPU" is unspecified; DreamerV3 CPU-inference cost is
unmeasured; Oasis-500M real-time viability unverified; GameNGen weights unverified; DA3Metric-Large
licence conflict; "Navigation Pixie" unverified; mirror-capture stereo validity untested; zero Arc
benchmarks for every model cited.
