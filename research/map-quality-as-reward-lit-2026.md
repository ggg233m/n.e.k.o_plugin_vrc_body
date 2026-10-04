# Map Quality as an RL Reward / Training Signal — Research Report

Scope: an agent that must build a **navigable semantic-traversability map** of an unknown
environment (floors, stairs, pits, water) and then **plan routes through it**.

Tooling note for whoever re-runs this: `web_fetch` is DNS-blocked for `arxiv.org`,
`proceedings.neurips.cc`, `*.github.io`, `aihabitat.org` ("resolves to a non-public IP
address"). The `pwsh` tool **does** have network access. Working recipes used throughout:
- full LaTeX-rendered paper text: `https://ar5iv.labs.arxiv.org/html/<arxivid>`
- paper metadata/abstract: `http://export.arxiv.org/api/query?search_query=...`
- reference implementation: `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<file>`

---

## 0. Three corrections to the premise (verified by full-text search)

These matter because they change what you should copy.

| Premise | Reality | Evidence |
|---|---|---|
| "**Neural Map: Structured Memory for Deep RL** (Chaplot et al., ICML 2019)" | That paper is by **Parisotto & Salakhutdinov**, not Chaplot ([arXiv:1702.08360](https://arxiv.org/abs/1702.08360)). Chaplot's related line is *Exploration via Neural Map* and *Semantic Neural Maps*. | author block of [arXiv:1702.08360](https://arxiv.org/abs/1702.08360) |
| "**SemExp's predictive information gain reward**" | **SemExp has no information-gain reward.** Full-text grep of [arXiv:2007.00643](https://arxiv.org/abs/2007.00643): the token `predictive` occurs **0 times**. SemExp's map is trained by *supervised cross-entropy + binary cross-entropy*; its RL reward is *decrease in distance to nearest goal object*. The predictive-information-gain idea is real, but it lives in **SEER** ([2209.11034](https://arxiv.org/abs/2209.11034)) and **MapEx** ([2409.15590](https://arxiv.org/abs/2409.15590)). | §1.3 |
| "**Percent Complete Exploration (PCE)**" and "**MPCE**" | I could not verify either as a standard, citable named metric (exhaustive arXiv-API + Bing + DDG + GitHub search). The canonical exploration metric in this literature is **%Cov / Cov** from Active Neural SLAM. Do not cite PCE/MPCE without finding a primary source first. | §2.3 |

Also: **Exploration metrics do not require the map to be *good* — only large.** ANS's reward
is literally "proportional to the increase" in coverage. That is the single most important
thing to un-learn when designing a *map-quality* reward.

---

## 1. Using map quality as a reward / objective

### 1.1 Active Neural SLAM — coverage reward (Chaplot et al., ICLR 2019 / [arXiv:2004.05155](https://arxiv.org/abs/2004.05155))

Modular: Mapper (egocentric 2×V×V map of obstacles + explored area) → Pose Estimator → Global
Policy → local planner via **Fast Marching Method** (Sethian).

- **Reward (verbatim):** "The Global Policy is trained using Reinforcement Learning with
  reward proportional to the increase [in coverage]."
  $$ r_t = \lambda \cdot \big(\mathrm{Cov}_{t} - \mathrm{Cov}_{t-1}\big) $$
- **Supervision:** Mapper trained on the *ground-truth* egocentric projection computed by
  geometric projection from GT depth. Loss is a weighted sum of obstacle-map BCE, explored-area
  BCE, and pose MSE (loss coefficients 1, 1, 10000).
- Global policy samples a new long-term goal every **25** timesteps; PPO.
- Reported: Gibson 0.948 %Cov / 32.701 m²; MP3D 0.521 %Cov / 73.281 m².

> ⚠️ The Mapper **predicts the "explored area" channel as an output**. That channel is directly
> hackable — an agent can paint its own explored mask wide. This is the canonical reward-hacking
> trap in mapping RL. See §5.

### 1.2 SemExp — supervised map loss, distance-reduced goal reward ([arXiv:2007.00643](https://arxiv.org/abs/2007.00643))

- **Map training (differentiable, supervised):**
  $$ \mathcal{L}_{\text{map}} = \text{CE}(\text{semantic map pred.},\ M^*) + \text{BCE}(\hat{m}_{\text{explored}}, m^*_{\text{explored}}) $$
  Verbatim: "The geometric projection is implemented using differentiable operations such that
  the loss on the semantic map prediction can be backpropagated through the entire module if
  desired." → **map-space loss, not reward**. This is the pattern to copy.
- **Goal-oriented semantic policy reward:** $$ r_t = D_{t-1} - D_t $$ where `D` is distance to
  the **nearest goal object**. PPO, γ=0.99, entropy coef 1e-3, value-loss coef 0.5, long-term
  goal resampled every **u = 25** steps (which "reduces the time-horizon for exploration in RL
  exponentially").
- Metrics reported: **Success, SPL, DTS** (distance-to-success).

### 1.3 Predictive information gain — the actual sources

- **SEER: Safe Efficient Exploration for Aerial Robots using Learning to Predict Information
  Gain** ([arXiv:2209.11034](https://arxiv.org/abs/2209.11034), ICRA 2023). "An incremental
  detection and prediction module that detects semantic objects, classifies frontiers,
  **predicts information gain**", sampling viewpoints to predict it.
- **MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map
  Predictions** ([arXiv:2409.15590](https://arxiv.org/abs/2409.15590)). Key insight quoted:
  prior methods "do not consider sensor coverage or visibility, which do not reflect actual
  sensor acquisition. Therefore, this method does not consider..." → MapEx forms a
  **probabilistic sensor model** to compute information gain jointly over what the robot *can*
  observe and its uncertainty.
- Generic Bayesian form to implement:
  $$ IG(i) = H\big(M^*_i\big) - \mathbb{E}_{s \sim p(\cdot \mid o_{1:t}, a)} \big[\, H(M^*_i \mid s) \,\big] $$
  estimated by a deep-ensemble as mutual information (see M2IDQN below), or with a
  particle/Gaussian-process belief.

### 1.4 L2M / M2IDQN — ensemble-disagreement as information gain ([arXiv:2106.15648](https://arxiv.org/abs/2106.15648), ICRA 2020)

- Verbatim reward: $$ R(s,a) = D(s,c) - D(s',c) $$ — "distance reduced between the agent and
  the target, where D(.,.) is distance on the shortest path." Motivation: this yields a
  *target-dependent* policy, so they instead **reformulate the policy as goal selection over
  unobserved map cells** using uncertainty.
- Uncertainty = **epistemic** uncertainty via **ensemble disagreement** (2-stage segmentation
  ensemble, [Pathak et al. 2019](https://arxiv.org/abs/1906.01520)), used both (i) to
  actively select labeling samples during training and (ii) at test time as an **UCB over map
  cells** for long-term goal selection. "Maximizing epistemic uncertainty is used as a proxy
  for maximizing information gain."
- **This is the closest published thing to a "map-quality-as-objective" for navigation**, and it
  is the right structural template: *uncertainty over the map, used to choose where to act.*

### 1.5 Semantic Curiosity — an intrinsic reward that survives without labels ([arXiv:2006.09367](https://arxiv.org/abs/2006.09367))

Excellent for your case because it needs **no ground-truth labels at all**:

> "A good semantic exploration policy is one which generates observations of objects and not
> free-space or the wall/ceiling. But not only should the observations be objects, but a good
> exploration policy should also observe many unique objects. Finally, a good exploration policy
> will move to parts of the observation space where the current object detection model fails."

> "In the standard curiosity reward, a policy is rewarded if the predicted future observation
> does not match the true future observation... Instead, we formulate semantic curiosity based
> on the meta-supervisory signal of **consistency in semantic prediction** — that is, if our
> model truly understands the object, it should predict the same label for the object even as we
> move around and change viewpoints. Therefore, we exploit consistency in label prediction to
> reward our policies."

This is the *viewpoint-consistency* term you want if you have no simulator ground truth.

### 1.6 Reward-vs-perception failure already documented in the literature

- **Ye et al., "Auxiliary Tasks and Exploration Enable ObjectNav"** ([arXiv:2104.04112](https://arxiv.org/abs/2104.04112),
  ICCV 2021). Same research group as SemExp. Verbatim observation about the learned agents:
  *"Qualitative examination of these agents indicate they prefer to **wander around goals for
  some time before stopping** at them correctly, likely due to the **exploration reward**."*
  → A pure coverage/IG reward demonstrably degrades the task metric it was supposed to support.
- **Remember to be Curious** ([arXiv:2605.22814](https://arxiv.org/abs/2605.22814), 2026) —
  the most on-point reward-hacking paper for your setting. Verbatim:
  > "agents frequently collapse into repetitive behavior driven by **cyclic curiosity rewards**…"
  > "without historical context, agents repeatedly revisit the same locations; simultaneously,
  > without a **persistent and continuously updating world model**, predictive errors spuriously
  > arise in these revisited areas, yielding **false novelty rewards for forgotten states**."

  Their fix: a **persistent** world model (online 3D Gaussian Splatting, continuously updated)
  + an agent with **episodic context** (sequence model over observations). Trained purely on
  curiosity on HM3D; zero-shot to Gibson. **Read this before designing any IG reward.**

### 1.7 Modern reward-hacking mitigations (general RL, applicable)

- **Reward as an Agent for Embodied World Models** ([arXiv:2606.19990](https://arxiv.org/abs/2606.19990),
  2026). Verbatim thesis: "the core limitation is not exploration itself, but the **lack of
  reliable verification strategies**… expanded exploration becomes highly susceptible to
  **reward hacking**, where policies exploit imperfect rewards without achieving genuine
  improvement." Their fix is an **agentic verifier** that actively evaluates rollouts.
- **GARDO: Reinforcing Diffusion Models without Reward Hacking** ([arXiv:2512.24138](https://arxiv.org/abs/2512.24138)).
- Curiosity/ICM baselines: [ICM, arXiv:1705.05363](https://arxiv.org/abs/1705.05363);
  [RND, arXiv:1810.12894](https://arxiv.org/abs/1810.12894).
- Active-learning-for-annotation equivalents: [SEAL, arXiv:2112.01001](https://arxiv.org/abs/2112.01001).

---

## 2. Metrics for map quality (with formulas and provenance)

### 2.1 Task-level navigation metrics

**SPL** — Success weighted by (normalized inverse) Path Length,
[Anderson et al. 2018, arXiv:1807.06757](https://arxiv.org/abs/1807.06757):

$$ \mathrm{SPL} = \frac{1}{N}\sum_{i=1}^{N} S_i \cdot \frac{l_i}{\max(p_i,\, l_i)} $$

where $S_i \in \{0,1\}$ is success, $l_i$ the **shortest-path (oracle) length**, $p_i$ the
agent's travelled path length. Confirmed verbatim in
[Ye et al. arXiv:2104.04112](https://arxiv.org/abs/2104.04112) and in Habitat's own code
(`habitat-lab/habitat/tasks/nav/nav.py`, see below).

**SPL (Habitat implementation, exact):**
```python
self._metric = ep_success * (self._start_end_episode_distance
                / max(self._start_end_episode_distance, self._agent_episode_distance))
```
([`habitat/tasks/nav/nav.py`](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/nav/nav.py), class `SPL`)

**SoftSPL (Habitat, exact)** — relaxes the boolean success:
```python
ep_soft_success = max(0, (1 - distance_to_target / self._start_end_episode_distance))
self._metric = ep_soft_success * (self._start_end_episode_distance
                / max(self._start_end_episode_distance, self._agent_episode_distance))
```
$$ \mathrm{SoftSPL} = \max\!\Big(0,\, 1 - \tfrac{d_t}{d_0}\Big) \cdot \frac{d_0}{\max(d_0, d_{\text{traveled}})} $$
Docs: [SoftSPLMeasurementConfig](https://aihabitat.org/docs/habitat-lab/habitat.config.default_structured_configs.SoftSPLMeasurementConfig.html).

**ObjectNav Success (Habitat 2023, verbatim from the challenge README):**
> "an episode is deemed successful if on calling the STOP action, the agent is within **1.0 m
> Euclidean distance** from any instance of the target object category AND the object *can be
> viewed by an oracle* from that stopping position by turning the agent or looking up/down…
> **Oracle-visibility is our proxy for 'the agent is close enough to interact with the object'.**"

Also verbatim: "for ObjectNav, optimal path = shortest path from the agent's starting position
to the **closest** instance of the target object category… if an agent spawns very close to
'chair1' but stops at a distant 'chair2', it will achieve 100% success … but a fairly low SPL."
([habitat-challenge README](https://github.com/facebookresearch/habitat-challenge/blob/main/README.md))

> 💡 That "oracle-visibility as interaction proxy" is a clean, citable example of a **utility**
> metric (can I act on it?) rather than a **fidelity** metric (is my label right?).

### 2.2 Coverage / exploration metrics

From Active Neural SLAM ([arXiv:2004.05155](https://arxiv.org/abs/2004.05155)), verbatim:
> "We use two evaluation metrics, the absolute coverage area in m² (**Cov**) and the percentage
> of area explored in the scene (**%Cov**), i.e. ratio of coverage area in the map known to be
> traversable… **A traversable point to be known if it is in the field-of-view of the agent and
> is less than 3.2 m away.**"

$$ \mathrm{Cov} = \big|\{\text{known-traversable cells}\}\big| \cdot a_{\text{cell}},
\qquad \%\mathrm{Cov} = \frac{|\{\text{known-traversable}\}|}{|\{\text{traversable in scene}\}|} $$

**The 3.2 m visibility rule is the important part**: coverage is defined by the *sensor frustum
on ground truth*, not by the agent's self-reported explored channel. Use this convention.

### 2.3 Trajectory / pose accuracy

- **ATE** (Absolute Trajectory Error):
  $$ \mathrm{ATE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\|\hat{p}_i - p_i^{\mathrm{gt}}\|^2} $$
  (or the mean form). TUM RGB-D uses the absolute translational error RMSE after Umeyama
  alignment; see the [TUM RGB-D online evaluation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/online_evaluation)
  and Sturm et al. [IROS 2012](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf).
- **KITTI odometry** — [benchmark page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php),
  verbatim: *"we compute translational and rotational errors for **all possible subsequences of
  length (100,…,800) meters**… errors are measured in **percent** (for translation) and in
  **degrees per meter** (for rotation)."* Standard devkit definitions:
  $$ t_{rel} = \frac{1}{n-1}\sum_{i=1}^{n}\|\Delta p_i\| \times 100\ \ [\%],
  \qquad r_{rel} = \frac{1}{n-1}\sum_{i=1}^{n}\arccos\!\frac{\Delta R_i\!\cdot\!\Delta R_0}{\|\Delta R_i\|\|\Delta R_0\|}\cdot\frac{180}{\pi} \ [\text{deg/m}] $$
  Sequences 00–10 train, 11–21 test; "fully automatic, no manual loop-closure tagging".
  Devkit: [pykitti/eval](https://github.com/utiasSTARS/pykitti).
- Note the **drift-vs-local-accuracy** distinction: KITTI's t_rel is a *sub-sequence* drift
  measure, so it penalizes long-horizon accumulation but is blind to local structure.

### 2.4 Point-cloud / SLAM map quality — MapEval (RA-L 2025)

[MapEval: Towards Unified, Robust and Efficient SLAM Map Evaluation Framework](https://github.com/JokerJohn/Cloud_Map_Evaluation)
([RA-L 2025](https://ieeexplore.ieee.org/document/10910156), [arXiv:2411.17928](https://arxiv.org/abs/2411.17928)).
**Its own framing is exactly the fidelity-vs-utility split you asked about** (verbatim):

> "addressing **two fundamentally distinct aspects of map quality assessment**:
> 1. **Global Geometric Accuracy** — measures the absolute geometric fidelity of the reconstructed
>    map compared to ground truth. This aspect is crucial as SLAM systems often accumulate drift
>    over long trajectories, leading to global deformation.
> 2. **Local Structural Consistency** — Evaluates the preservation of local geometric features
>    and structural relationships, **which is essential for tasks like obstacle avoidance and
>    local planning, even when global accuracy may be compromised.**
> … global drift may exist despite excellent local reconstruction, or conversely, good global
> alignment might mask local inconsistencies."

Metrics implemented: **AC** (accuracy, point-level geometric error), **COM** (completeness /
coverage), **CD** (Chamfer), **MME** (Mean Map Entropy, information-theoretic local
consistency), plus novel **AWD** (Average Wasserstein Distance, robust global accuracy) and
**SCS** (Spatial Consistency Score, local consistency). Datasets: MS-Dataset, FusionPortableV2,
New College, GEODE.

### 2.5 Semantic map / occupancy metrics (perception-side)

- **mIoU / IoU against GT semantic labels** is the standard for 3D semantic occupancy
  prediction: [Occ3D benchmark, NeurIPS 2023 D&B](https://proceedings.neurips.cc/paper_files/paper/2023/file/cabfaeecaae7d6540ee797a66f0130b0-Paper-Datasets_and_Benchmarks.pdf),
  [OccFormer, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_OccFormer_Dual-path_Transformer_for_Vision-based_3D_Semantic_Occupancy_Prediction_ICCV_2023_paper.pdf),
  [PanoOcc, CVPR 2024](https://arxiv.org/html/2306.10013). Also semantic BEV mIoU (BEVFormer
  line). Soft/differentiable IoU (soft-Jaccard) is standard and is what you'd want as a loss.
- **Depth-map quality, incl. scale-consistent loss** (MiDaS line,
  [arXiv:1907.01341](https://arxiv.org/abs/1907.01341)): AbsRel, RMSE, δ < 1.25, and the
  scale-invariant term $s = \arg\min_s \sum_i (s d_i - \hat d_i)^2$ giving **scale-consistent
  AbsRel**, which is what makes zero-shot cross-dataset transfer work. Relevant if your
  traversability head runs off monocular depth — a mis-scaled depth map wrecks traversability
  inference even when the shape is right.
- **Traversability classification**: precision / recall / F1 from a confusion matrix on
  GT-labelled terrain cells. Self-supervised proxies: [ScaTE, arXiv:2209.06522](https://arxiv.org/abs/2209.06522)
  (RA-L 2023) — "Actual driving experience can be utilized in a **self-supervised** fashion to
  learn **vehicle-specific** traversability"; [Ghigliazza et al.,
  "Learning to Detect Traversable Surfaces"](https://arxiv.org/abs/2005.10672) (RA-L 2020);
  [Traversability-Aware Legged Navigation by Learning from Real-World Visual Data, arXiv:2410.10621](https://arxiv.org/abs/2410.10621).

### 2.6 Newer benchmarks — and a warning about name collisions

| Benchmark | arXiv / link | What it measures | Usable as map-quality metric? |
|---|---|---|---|
| **GOAT / GOAT-Bench** | [2404.06609](https://arxiv.org/abs/2404.06609) | Universal **lifelong** navigation: a *sequence* of open-vocabulary goals (category / language / image). Explicitly studies "the **impact of memory in lifelong scenarios**" | **Yes — the best task-level proxy for a persistent-map agent** |
| **MapEval (SLAM)** | [2411.17928](https://arxiv.org/abs/2411.17928) / [GitHub](https://github.com/JokerJohn/Cloud_Map_Evaluation) | Point-cloud map: AC, COM, CD, MME, AWD, SCS | **Yes — geometric fidelity** |
| **MapEval (geo-spatial reasoning)** | [2501.00316](https://arxiv.org/abs/2501.00316) / [site](https://mapeval.github.io) | 700 MCQs, 180 cities, 54 countries, for LLMs using map APIs | **No** — LLM QA, not map quality |
| **MapBench (LLM map reading)** | [2503.14607](https://arxiv.org/html/2503.14607) | >1600 pixel-space path-finding problems from 100 maps for LVLMs | Partial — measures map *legibility* |
| **NavBench** | [2506.01031](https://arxiv.org/abs/2506.01031) | Probing MLLMs for embodied navigation | No |
| **Habitat ObjectNav 2023+** | [habitat-challenge](https://github.com/facebookresearch/habitat-challenge) | ObjectNav/ImageNav on HM3D-Semantics v0.2, 216 scenes, 145/36/35, 6 categories (chair, couch, potted plant, bed, toilet, tv), Stretch agent, **continuous** action space | Yes (SR/SPL) |
| **Habitat 3.0** | [2310.13724](https://arxiv.org/abs/2310.13724) | Human-robot co-habitation + navigation | Infrastructure |
| **occWorld** | [2311.16038](https://arxiv.org/abs/2311.16038) (ECCV 2024, driving); [Occupancy World Model for Robots, 2505.05512](https://arxiv.org/abs/2505.05512) (2025, embodied) | 3D occupancy world model | Training objective source |
| **MPCE** | — | **Could not verify.** Do not cite. | — |

---

## 3. Differentiable / learnable memory maps

### 3.1 Neural Map — the read / write / update algebra ([arXiv:1702.08360](https://arxiv.org/abs/1702.08360), ICML 2019)

The map $M_t \in \mathbb{R}^{H\times W\times C}$ is written by four differentiable ops.
**Context read** (soft attention addressing):
$$ q_t = W[s_t, r_t] \quad(8) $$
$$ a_t^{(x,y)} = q_t \cdot M_t^{(x,y)} \quad(9) $$
$$ \alpha_t^{(x,y)} = \frac{e^{a_t^{(x,y)}}}{\sum_{(w,z)} e^{a_t^{(w,z)}}} \quad(10) $$
$$ c_t = \sum_{(x,y)} \alpha_t^{(x,y)}\, M_t^{(x,y)} \quad(11) $$

**Key–value variant** (stronger associative bias): $M_t^{(x,y)} = [k_t^{(x,y)}, v_t^{(x,y)}]$, with
$a = q\cdot k$, softmax over the grid, and $c_t = \sum \alpha v$ (Eqs. 15–19).

**Local write** (a candidate vector at the *current* cell only):
$$ w_{t+1}^{(x_t,y_t)} = f\big([\,s_t,\ r_t,\ c_t,\ M_t^{(x_t,y_t)}\,]\big) \quad(12) $$

**Map update** (in-place, local):
$$ M_{t+1}^{(a,b)} = \begin{cases} w_{t+1}^{(x_t,y_t)} & \text{if } (a,b)=(x_t,y_t) \\ M_t^{(a,b)} & \text{otherwise} \end{cases} $$

The **spatial-differencing convolution** is the broadcasting trick that turns this local write
into a full-map differentiable op:
$$ w = \sigma\!\big(\max(W_c \circledast (\max(W_n \circledast M,\ W_e \circledast E) + b_n + b_e,\ 0)\big) $$
(the "spatial differencing" wording is not in the ar5iv render of this paper; it is the
standard implementation of the `update` operator).

### 3.2 Persistent, editable, topologically-consistent vs one-shot transformers

| System | Year | Representation | Temporal? | Edited in place? | Topologically consistent? | End-to-end differentiable? |
|---|---|---|---|---|---|---|
| **Neural Map** ([1702.08360](https://arxiv.org/abs/1702.08360)) | 2019 | dense grid features | ✅ | ✅ (local write) | ❌ (raster only) | ✅ (read/write/update) |
| **ANS Mapper** ([2004.05155](https://arxiv.org/abs/2004.05155)) | 2019 | egocentric 2×V×V obstacle+explored raster | ✅ | ✅ (aggregated by spatial transform + channel-wise max-pool) | ❌ | ✅ (differentiable projection) |
| **SemExp Semantic Mapping** ([2007.00643](https://arxiv.org/abs/2007.00643)) | 2020 | H×W×(C+2) semantic raster, 3-layer CNN denoiser, **channel-wise max-pool fusion over time** | ✅ | ✅ | ❌ | ✅ (explicitly stated) |
| **Semantic Neural Maps** (Chaplot et al., CoRL-era 2020) | 2020 | STM (spatial working memory tensor) + LTM (semantic/goal memory), CNN-LSTM, feeding an ORB global policy | ✅ | ✅ | ❌ | ✅ |
| **Neural Topological SLAM** ([2005.12256](https://arxiv.org/abs/2005.12256)) | 2020 | **graph**: nodes with semantic features, interconnected by coarse geometric edges; supervised build/maintain/use under noisy actuation | ✅ | ✅ (add/merge nodes) | ✅ | ❌ (discrete) |
| **Differentiable Spatial Planning using Transformers** ([2112.01010](https://arxiv.org/abs/2112.01010)) | 2021 | transformer planner, spatially grounded | ✅ | n/a | partial | ✅ |
| **SEAL** ([2112.01001](https://arxiv.org/abs/2112.01001)) | 2021 | online 3D reconstruction, self-supervised by 3D consistency | ✅ | ✅ | ❌ | ✅ |
| **SNAP: Self-Supervised Neural Maps** (NeurIPS 2023, [OpenReview](https://openreview.net/forum?id=LCHmP68Gtj)) | 2023 | neural maps trained by *self-supervision*, for visual positioning + semantics | ✅ | ✅ | ❌ | ✅ |
| **AutoNeRF** ([2304.11241](https://arxiv.org/abs/2304.11241)) | 2023 | implicit NeRF scene representation optimized by an RL agent | ✅ | ✅ (photometric) | ❌ | partial |
| **occWorld** ([2311.16038](https://arxiv.org/abs/2311.16038)) | 2023/24 | 3D occupancy tokenizer → **discrete scene tokens** → GPT-like spatio-temporal generative transformer → future occupancy + ego traj | ✅ (rollout) | ❌ (autoregressive generation) | ❌ | ❌ (tokenized) |
| **MapTR / MapTRv2** ([2208.14437](https://arxiv.org/abs/2208.14437), [2308.05736](https://arxiv.org/abs/2308.05736)) | 2022/23 | vectorized HD map, per-element queries, **Hungarian matching** + Chamfer position cost + directional + focal, evaluated with **Chamfer-based AP / mAP** | ⚠️ streaming (BEV) | ❌ (one-shot per frame) | ❌ | ✅ (training only) |
| **StreamMapNet** ([2308.12570](https://arxiv.org/abs/2308.12570)) | 2023/24 | **streaming** BEV vectorized HD map, propagates a query-based BEV memory across time | ✅ | ⚠️ (memory fusion, not map editing) | ❌ | ✅ |
| **DynaVol / DynaVol-S** ([2305.00393](https://arxiv.org/abs/2305.00393), [2407.20908](https://arxiv.org/abs/2407.20908)) | 2023/24 | object-centric voxelization of dynamic scenes, differentiable volume rendering, unsupervised | ✅ (video) | ❌ (one-shot scene) | ❌ | ✅ (rendering) |
| **MapAnything** ([2509.13414](https://arxiv.org/abs/2509.13414), Meta) | 2025 | **unified feed-forward** transformer; "a collection of depth maps, local ray maps, camera poses, and a **metric scale factor** that effectively upgrades local reconstructions into a globally consistent metric frame"; one pass, 12+ 3D tasks | ❌ | ❌ | ✅ *metric frame only* | ✅ |
| **D-Map / "Diffusion as Reasoning"** ([2410.21842](https://arxiv.org/abs/2410.21842)) | 2024 | **diffusion model of the semantic map**, conditioned on the explored region, generates the *unknown* region; "global target bias" + "local LLM bias" | ✅ (conditioned on map) | ❌ (generative) | ❌ | ❌ (sampling) |
| **Occupancy World Model for Robots** ([2505.05512](https://arxiv.org/abs/2505.05512)) | 2025 | occupancy world model for embodied agents | ✅ | ❌ | ❌ | ❌ |

**Verdict for your brief.** Only two families give you a **long-term, incrementally-edited,
topologically-consistent** map:
1. **Raster + recurrent fusion** (SemExp-style: differentiable projection → denoiser → spatial
   warp + channel-wise max-pool). Best for *end-to-end differentiability* and cheap edits, but
   gives you **no topology** — you must run a planner (FMM/Dijkstra) on top to get consistency.
2. **Graph / topological** (Neural Topological SLAM, and the 3D scene-graph line:
   [D-Lite, 2209.06111](https://arxiv.org/abs/2209.06111), [SGN-CIRL, 2506.04505](https://arxiv.org/abs/2506.04505),
   [Predicting Topological Maps, 2211.12649](https://arxiv.org/abs/2211.12649)). Best for
   *persistent structure and long-horizon* but **not differentiable**.

**Recommended hybrid (and what the literature supports):** learn the raster with a
SemExp-style differentiable module (supervised, §6), and separately maintain a cheap graph
layer (rooms / portals / stairs / water edges) rebuilt from the raster. Use the raster for
**metric** cost-to-traverse and the graph for **topological** reachability and loop closure.
DynaVol / occWorld / MapAnything are **not** appropriate for this: they are one-shot or
autoregressive *generators* of a scene, with no in-place editing and no metric map you can run
A* on. MapTR/MapTRv2/StreamMapNet are valuable only for their **losses** (Hungarian +
Chamfer + directional) and their **metric** (Chamfer-AP / mAP), which transfer directly to
evaluating a vectorized traversability map.

---

## 4. Memory architectures for long-horizon navigation

### 4.1 Topological / graph memory
- **Neural Topological SLAM** ([arXiv:2005.12256](https://arxiv.org/abs/2005.12256)) — the
  canonical formulation: *"we design topological representations for space that effectively
  leverage semantics and afford approximate geometric reasoning. At the heart of our
  representations are **nodes with associated semantic features, that are interconnected using
  coarse geometric information**."* Verbatim on the motivation (an oven lives in the kitchen,
  so path 2): supervised algorithms that *build, maintain and use* these representations
  "under noisy actuation"; **>50 % relative improvement** on image-goal navigation.
- 3D scene graphs: [D-Lite: Navigation-Oriented Compression of 3D Scene Graphs for Multi-Robot
  Collaboration, 2209.06111](https://arxiv.org/abs/2209.06111);
  [SGN-CIRL: Scene Graph-based Navigation with Curriculum, Imitation, and RL, 2506.04505](https://arxiv.org/abs/2506.04505);
  [Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs, 2309.15940](https://arxiv.org/abs/2309.15940).
- Topological map *prediction* for unexplored regions: [2211.12649](https://arxiv.org/abs/2211.12649).

### 4.2 Episodic memory transformers
- **GridMM: Grid Memory Map for Vision-and-Language Navigation** ([arXiv:2307.12907](https://arxiv.org/abs/2307.12907),
  ICCV 2023) — a *persistent grid memory map* updated each step and consumed by the navigator.
  Architecturally the closest modern analogue to what you want.
- **ESceme: Vision-and-Language Navigation with Episodic Scene Memory** ([arXiv:2303.01032](https://arxiv.org/abs/2303.01032)).
- **MapGPT: Map-Guided Prompting with Adaptive Path Planning for VLN** ([arXiv:2401.07314](https://arxiv.org/abs/2401.07314)).

### 4.3 The map as a POMDP state, and **cross-episode / lifelong** maps ← the most relevant subsection

This is where the recent, directly-on-point work lives.

- **GOAT / GOAT-Bench** ([arXiv:2404.06609](https://arxiv.org/abs/2404.06609), NeurIPS 2024
  spotlight, CMU/Motta/Chaplot). "the agent is directed to navigate to a **sequence** of
  targets specified by the category name, language description, or image in an
  open-vocabulary fashion… they analyze their performance across modalities, the role of
  **explicit and implicit scene memories**, their robustness to noise in goal specifications, and
  the **impact of memory in lifelong scenarios**." → **This is your benchmark.** It isolates
  the persistent-memory value proposition, which single-goal ObjectNav (with memory reset)
  structurally cannot.
- **VTM-Nav: Harnessing Cross-Episode Experience for Object-Goal Navigation with Hierarchical
  Visual-Topological Memory** ([arXiv:2607.14514](https://arxiv.org/abs/2607.14514)). Verbatim
  problem statement:
  > "Training-free ObjectNav agents increasingly use vision-language models (VLMs), yet
  > **typically discard acquired scene knowledge after each request.** We study **cross-episode
  > ObjectNav**, where each request is an independently initialized, single-goal episode and
  > **only self-acquired, scene-scoped memory persists across episodes**. We ask whether an
  > agent with **fixed model parameters and navigation components can reuse such experience
  > without retraining or oracle information.**"

  Method: a persistent **Hierarchical Visual-Topological Memory** — a coarse **room topology**
  indexes room-owned visual memories; it **distinguishes in-room from remote-visible evidence**
  and retains successful approach cues; each new request **re-localizes the agent in the
  accumulated scene structure**; a **conservative execution guard** handles local failures.
  Result: **+4.6 / +2.0 / +0.8 SR** over the memory-reset control on HM3D v0.1 / v0.2 / MP3D
  at matched 40-step budgets, with comparable-or-higher SPL, and **+3.1 / +5.5 SR** over a
  text-memory-harnessed WMNav on HM3D. **Note the design constraints: fixed parameters, no
  oracle, and a re-localization step** — all three are directly transferable to a small
  trained agent.
- **OVAL: Open-Vocabulary Augmented Memory Model for Lifelong Object Goal Navigation**
  ([arXiv:2604.12872](https://arxiv.org/abs/2604.12872), 2026). Verbatim:
  > "navigation rarely involves pursuing an isolated objective (e.g., to locate a *cup*), but
  > rather requires sequentially achieving lifelong open-vocabulary goals… in different unseen
  > room scenes." Introduces **memory descriptors** for structured management of the memory
  model, plus a **probability-based exploration strategy using multi-value frontier scoring**:
  $$ P(F) \propto o_d(F) + o_s(F) + o_f(F) $$
  and a **confidence-based stopping rule**:
  $$ C_i = \exp\!\Big(-\sigma\,\big\|\bar p_c - p_c\big\|\,\tfrac{D_t}{\bar p_c A_b}\Big) $$
- **LiDAR lifelong mapping** (analogous infrastructure, if you ever go outdoors):
  [LT-mapper, 2107.07712](https://arxiv.org/abs/2107.07712);
  [Lifelong update of semantic maps in dynamic environments, 2010.08846](https://arxiv.org/abs/2010.08846);
  [Ephemerality meets LiDAR-based Lifelong Mapping, 2502.13452](https://arxiv.org/abs/2502.13452);
  [2501.18110](https://arxiv.org/abs/2501.18110).

---

## 5. Reward design: pitfalls, failure cases, and fidelity-vs-utility

### 5.1 Published failure cases (use these as your design constraints)

1. **Cyclic / stale curiosity rewards** — *Remember to be Curious*
   ([arXiv:2605.22814](https://arxiv.org/abs/2605.22814)): "agents frequently collapse into
   repetitive behavior driven by cyclic curiosity rewards"; "false novelty rewards for forgotten
   states" arise when the world model is not persistent. **Constraint: your map must be
   persistent and continuously updated, and your agent must carry episodic context.**
2. **Exploration reward degrades the task metric** — Ye et al.
   ([arXiv:2104.04112](https://arxiv.org/abs/2104.04112)): agents "wander around goals for some
   time before stopping… likely due to the exploration reward."
3. **Self-declared coverage is forgeable** — ANS's Mapper *predicts* the explored channel
   ([arXiv:2004.05155](https://arxiv.org/abs/2004.05155)). **Constraint: define coverage from the
   sensor frustum on ground truth, never from the agent's own map.**
4. **Action-conditioned video world models lack spatial persistence** — *Remember to be Curious*
   shows LingBot-World "fails to maintain scene consistency, generating an entirely different
   image after a simple in-place 360° turn," and that ICM's "learned world model acts as a
   **statistical prior over lifelong experience rather than an episodic record** of the
   environment." **Constraint: don't use a generative video model as the memory for a
   navigational map.**
5. **General RL reward hacking under distribution shift** — *Reward as an Agent for Embodied
   World Models* ([arXiv:2606.19990](https://arxiv.org/abs/2606.19990)); GARDO
   ([arXiv:2512.24138](https://arxiv.org/abs/2512.24138)). The stated fix is a **verifier that
   actively evaluates rollouts** rather than trusting a static reward.

### 5.2 The fidelity-vs-utility distinction — the citable version

**MapEval (RA-L 2025)** states the split explicitly ([GitHub](https://github.com/JokerJohn/Cloud_Map_Evaluation)):
"global drift may exist despite excellent local reconstruction, or conversely, good global
alignment might mask local inconsistencies" — and it states that **local structural
consistency** is the property "essential for tasks like obstacle avoidance and local planning."

So:
- **Fidelity metrics** (perception): ATE/RPE, map Chamfer/AC/COM, mIoU, AbsRel, traversability F1.
  These answer *"is my map right?"*
- **Utility metrics** (behaviour): SPL, SoftSPL, success-with-oracle-visibility, %Cov, planning
  regret, **counterfactual path cost**, and cross-episode SR gains ([2404.06609](https://arxiv.org/abs/2404.06609),
  [2607.14514](https://arxiv.org/abs/2607.14514)). These answer *"does my map get me there?"*

Habitat's own **oracle-visibility** criterion is a good precedent for a utility proxy:
*"Oracle-visibility is our proxy for 'the agent is close enough to interact with the object'."*
([habitat-challenge](https://github.com/facebookresearch/habitat-challenge/blob/main/README.md))

**Empirical caveat:** I found no paper that cleanly ablates "prettier map → better navigation."
The closest is MapEval's *local structural consistency* claim, plus the widespread practice in
ObjectNav of using explicit maps + a classical planner rather than end-to-end policies —
SemExp itself argues that "learnt representations are implicit and the models need to learn
obstacle avoidance, episodic memory, planning as well as semantic priors implicitly from the
goal-driven reward," and that **explicit map representation improves both performance and
sample efficiency**. ([arXiv:2007.00643](https://arxiv.org/abs/2007.00643)). **Treat "better map
⇒ better navigation" as an assumption to be ablated, not a given.**

### 5.3 Safer shaping alternatives (with the caveat each carries)

| Shaping term | Formula | Risk |
|---|---|---|
| Coverage bonus | $\Delta\,\mathrm{Cov}_t$ computed on **GT frustum** (§2.2) | if GT-free, agent paints its own explored channel |
| Entropy of unknown region | $H(\text{Unknown}_t) - H(\text{Unknown}_{t-1})$ | **exactly the cyclic-curiosity failure** ([2605.22814](https://arxiv.org/abs/2605.22814)); needs a persistent map to be safe |
| Predictive information gain | IG formula §1.3, MC over ensemble | same; also degenerates to "go where the model is uncertain forever" |
| Uncertainty/UCB goal selection | $\text{argmax}_i \, \mu_i + \beta\,\sigma_i$ ([2106.15648](https://arxiv.org/abs/2106.15648)) | needs the epistemic/deterministic split or it just finds aleatoric noise |
| Viewpoint-consistency (Semantic Curiosity) | reward trajectories causing **label inconsistency for the same object under viewpoint change** ([2006.09367](https://arxiv.org/abs/2006.09367)) | "wrong but stable" maps score well — it's a *proxy*, not ground truth |
| **Counterfactual planning regret** (see §6) | $\sum_k [\hat C^*(g_k) - \hat C_t(g_k)]_+$ | costs $K$ shortest paths per checkpoint |

---

## 6. Fusing map quality into planning

### 6.1 Cost-to-traverse as a soft field, not a binary mask

Build a per-cell cost from the semantic-traversability map, with **bounded but non-zero** costs
for hazards, so a detour is always representable:

$$ C(i) = \underbrace{1}_{\text{free}} + \sum_{k} w_k \cdot \mathbb{1}\!\left[\text{class}(i)=k\right] \cdot (1 - \text{confidence}(i)) + w_{\text{slope}}\cdot|\nabla h(i)| $$

Practical weights for your four hazard classes: `floor ≈ 1`, `stairs ≈ 1 + ε`
(direction-dependent!), `pit/shaft → C_max` (truly impassable), `shallow water ≈ 1 + c_w` with
$c_w$ tuned to the *real* cost of fording, `deep water → C_max`. Cap at $C_{\max}$ (e.g. 1000)
so Dijkstra terminates and so the reward cannot be driven to infinity by map blow-ups.

> **This is the single most important design choice for pits/water.** With hard blocking, a
> correct map is often *uninformative*: the shortest path is "infinite" both for a good and a bad
> map, so the utility reward is flat and the agent gets no gradient. With **soft costs**, "the
> agent's map claims a cheap route through shallow water that GT says is deep" produces a
> dense, informative penalty. Your reward becomes a *function* of map quality instead of a
> step function. This is the same reason MapEval insists on measuring local structure "even when
> global accuracy may be compromised."

Run the planner with **Fast Marching Method** (SemExp and ANS both use it,
[arXiv:2007.00643](https://arxiv.org/abs/2007.00643), [arXiv:2004.05155](https://arxiv.org/abs/2004.05155))
or Dijkstra/A* on an 8-connected grid. FMM handles the *continuous* cost field better and gives
a natural `Eikonal` gradient, which you need if the planner is inside the training loop.

### 6.2 Learned cost maps (exactly the losses to copy)

**Predicting Dense and Context-aware Cost Maps for Semantic Robot Navigation**
([arXiv:2210.08952](https://arxiv.org/abs/2210.08952), IROS 2022, Chebrolu/Stachniss et al.).
Pipeline: semantic mapping (Chaplot-style) → **cost map prediction** (U-Net, fuses egocentric
mid-level visual features, robot-orientation-aware) → **sampling-based MPC**.
Three losses, verbatim:

$$\mathcal{L}_{occ} = \frac{1}{HW}\sum_{i,j}\Big[-c^{occ}_{i,j}\log \hat c^{occ}_{i,j} - (1-c^{occ}_{i,j})\log\big(1-\hat c^{occ}_{i,j}\big)\Big] \tag{1}$$
$$\mathcal{L}_{cost} = \frac{1}{HW}\sum_{i,j}\Big\|\big(c^{nav}_{i,j}-\hat c^{nav}_{i,j}\big)\big(1-c^{occ}_{i,j}\big)\Big\|_1 \tag{2}$$
$$\mathcal{L}_{dir} = \frac{1}{HW}\sum_{i,j}\Big(1-\frac{\mathbf{g}_{i,j}\cdot\hat{\mathbf{g}}_{i,j}}{|\mathbf{g}_{i,j}|\cdot|\hat{\mathbf{g}}_{i,j}|}\Big)\big(1-c^{occ}_{i,j}\big) \tag{3}$$
with $\mathbf{g}_{i,j} = (\delta \mathcal{C}^{nav}/\delta x,\ \delta \mathcal{C}^{nav}/\delta y)$.

Note the pattern: **all three are masked by $(1-c^{occ})$** — every loss is computed only on
*navigable* cells. Do the same in your reward: never charge the agent for cells it could not
have observed.

Reported ablation: mid-level visual features improve **success rate by 7 percentage points**.

### 6.3 Sampling-based planners with learned cost

- The same paper drives its predicted cost map with **sampling-based MPC** (continuous
  velocity control, differential-drive model) — "the predicted cost maps can be used for
  semantic navigation in a loosely coupled scheme with sampling-based model predictive control
  (MPC)."
- **GP-guided MPPI** ([arXiv:2307.04019](https://arxiv.org/abs/2307.04019)) — learns the
  *unknown-region* cost from data and injects it into MPPI for cluttered unknown environments.
  This is the "DIAL/MPPI with learned cost" pattern: the prior only fills **unobserved** cells.
- **Differentiable Composite Neural Signed Distance Fields for Robot Navigation in Dynamic
  Indoor Environments** ([arXiv:2502.02664](https://arxiv.org/abs/2502.02664)) — differentiable
  geometry layer under a conventional planner.
- **Differentiable Spatial Planning using Transformers** ([arXiv:2112.01010](https://arxiv.org/abs/2112.01010))
  — end-to-end gradients through planning.
- Weakly-supervised cost-to-traverse without precise localization:
  [arXiv:1906.02468](https://arxiv.org/abs/1906.02468);
  terrain: [TerraPN, 2202.12873](https://arxiv.org/abs/2202.12873),
  [Online Hierarchical Policy Learning using Physics Priors, 2510.01519](https://arxiv.org/abs/2510.01519).

**Recommended coupling:** keep the planner **fixed and non-learned** during reward computation.
Dijkstra/FMM with hand-set weights on $\hat C_t$ (agent map) and on $\mathcal{C}^*$ (GT cost
field). If you let the planner be learned and co-trained, you get **collusion** — the map and
planner will agree with each other and be jointly wrong. Freeze it.

---

## 7. Recommendation: a trainable map-quality reward

### 7.1 First, the target map

Priority order:

1. **Simulator ground truth (recommended if you are in Habitat / iGibson / a custom sim).**
   Build $\mathcal{M}^*$ by rasterizing the scene mesh + navmesh + instance/vertex semantic
   labels into the *same* world frame and grid resolution the agent predicts. HM3D-Semantics
   provides per-vertex semantics and a navmesh; MP3D similarly
   ([Habitat-Matterport 3D Semantics, 2210.05633](https://arxiv.org/abs/2210.05633);
   [Habitat 3.0, 2310.13724](https://arxiv.org/abs/2310.13724)). Derive the GT cost field
   $\mathcal{C}^*$ from the same map with §6.1 weights, so GT and prediction are scored through
   an *identical* lens.
   - Also precompute the **observability set** $\mathcal{O}^*$ per cell: cells the sensor could
     ever see from anywhere in the scene (this is what makes the "you should have known this"
     term well-defined). Precompute it once; it is expensive.
2. **Teacher / pseudo-labels (if GT semantics are unavailable).** Run a deliberately
   privileged mapper — GT depth + GT poses + pretrained segmentation — to produce $\mathcal{M}^*$
   once per scene, and **distill**: the small online agent regresses toward the teacher map.
   Accept that the ceiling is the teacher's accuracy; report the teacher as an upper bound.
3. **Self-supervised only (last resort).** Viewpoint-consistency ("Semantic Curiosity",
   [arXiv:2006.09367](https://arxiv.org/abs/2006.09367)) + multi-view photometric/geometric
   consistency ([SEAL, arXiv:2112.01001](https://arxiv.org/abs/2112.01001)). This is easy to hack
   with the wrong rationale (retexturing, specular water) and gives no absolute scale, so use it
   only for pretraining, then switch to (1) or (2).

### 7.2 The reward — split into a *supervised* fidelity loss and an RL *utility* reward

Do **not** put map accuracy in the RL reward. Put it in a dense supervised loss (which cannot
be gamed by a policy) and let the RL reward measure only **utility**.

**(A) Fidelity loss (differentiable, supervised, backprop through SemExp-style projection).**
Computed on **observed** cells only.

$$\mathcal{L}_{fid} = \underbrace{\lambda_{cls}\cdot \mathrm{CE}(p_t(i), \mathcal{M}^*(i))}_{\text{per-cell class}}
\;+\; \underbrace{\big(1 - \mathrm{IoU}^{soft}_t\big)}_{\text{segmentation-aligned}}
\;+\; \underbrace{\lambda_{cal}\cdot \big(\mathrm{Brier}(p_t, \mathcal{M}^*) - \mathrm{Brier}(\text{uniform}, \mathcal{M}^*)\big)}_{\text{calibration}}$$
$$\mathrm{IoU}^{soft}_t = \frac{\sum_{i\in\mathcal{O}_t} p_t(i)\,g(i)}{\sum_{i\in\mathcal{O}_t}\big(p_t(i) + g(i) - p_t(i)\,g(i)\big)},
\qquad g(i) = \mathcal{M}^*(i)\ \text{one-hot}$$

*Why soft-IoU and not only CE:* it is the exact metric you will be judged on (mIoU/IoU, §2.5),
and unlike CE it is not satisfiable by predicting the majority class.
*Why the Brier term:* the dominant map failure is not wrong labels, it is **confident wrong
labels**. Calibration makes overconfidence expensive, which is what stops the
"self-consistent but wrong" map from winning.

**(B) Anti-fabrication loss — the term that kills "paint it and declare victory."**

$$\mathcal{L}_{fab} = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \mathbb{1}\!\left[\exists \mathcal{O}^*_t \ni i\ \text{reachable from the trajectory}\right]\cdot \mathbb{1}\!\left[\arg\max_c p_t(i) \neq \texttt{unknown}\right]$$

i.e. **penalize confident non-`unknown` predictions on cells that were actually observable but
that the agent never looked at.** Without this, the cheapest way to reduce $\mathcal{L}_{fid}$
is to mark every cell `unknown`.

**(C) The RL reward — counterfactual planning regret (the "utility" term).**

Every $N$ steps (not every step — it costs $K$ shortest-path computations):

1. Sample $K$ probe goals $G = \{g_1..g_K\}$ uniformly from the GT-reachable set
   $\mathcal{R}^*$. **Resample every episode** so the map cannot memorize them.
2. $\hat C_t(g)$ = cost of the path from the agent's current pose to $g$ under a **frozen**
   Dijkstra/FMM on the agent's own cost field $\hat C_t$, capped at $C_{\max}$.
3. $\hat C^*(g)$ = the same planner on the GT cost field $\mathcal{C}^*$.
4. **Reward:**
$$ R^{util}_t = \frac{1}{K}\sum_{k=1}^{K} \frac{\big(\hat C^*(g_k) - \min(\hat C_t(g_k),\, C_{\max})\big)_+}{\max(\hat C^*(g_k),\ \epsilon)} \;\in\;[-1,\,1]$$

5. **Coverage shaping** (GT-defined, §2.2):
$$ R^{cov}_t = \frac{\big|\mathcal{O}^{traj}_t \cap \mathcal{R}^*\big|}{\big|\mathcal{R}^*\big|} - \frac{\big|\mathcal{O}^{traj}_{t-1} \cap \mathcal{R}^*\big|}{\big|\mathcal{R}^*\big|}$$
   where $\mathcal{O}^{traj}_t$ is the union of GT sensor frustums over the trajectory. **Never**
   read this from the agent's own explored channel.

6. **Information-gain shaping** (optional, only with a persistent map — see [2605.22814](https://arxiv.org/abs/2605.22814)):
$$ R^{ig}_t = \frac{1}{K}\sum_{k} \mathrm{IG}(g_k) \quad\text{with IG from §1.3, MC over a deep ensemble}$$

7. **Total RL objective:**
$$\mathcal{J}_{RL} = \mathbb{E}\Big[\sum_t \gamma^t\big(R^{util}_t + \beta R^{cov}_t + \eta R^{ig}_t\big)\Big]$$
and the **total training loss**:
$$\boxed{\ \mathcal{L} = \mathcal{L}_{fid} \;+\; \lambda_{fab}\,\mathcal{L}_{fab} \;-\; \alpha\,\mathcal{J}_{RL}\ }$$

### 7.3 Why this design resists "good enough to fool the reward"

| Failure mode | Why it fails here |
|---|---|
| Mark everything `unknown` | $\mathcal{L}_{fab}$ is maximal; $\hat C_t(g_k)=C_{\max}$ for all $k$, so $R^{util}$ is **negative-maximal**, not neutral. Coverage gives zero. |
| Paint a large, confident, wrong map | $\mathcal{L}_{fid}$ (CE + soft-IoU + Brier) charges per-cell on GT; and $R^{util}$ is positive **only** when the agent's own planner finds a path *matching the GT cost*, so a wrong-but-confident map routes through hazards and loses reward. |
| Explore forever to farm coverage | $R^{cov}$ is a **difference**, not a level; and $R^{util}$ is bounded in $[-1,1]$, so it cannot be dominated. Reported failure of exactly this: [arXiv:2104.04112](https://arxiv.org/abs/2104.04112). |
| Cycle in a loop to farm novelty | $R^{ig}$ is computed against the **persistent accumulated map**; once the loop is in the map, IG there is ~0. Reported failure: [arXiv:2605.22814](https://arxiv.org/abs/2605.22814). |
| Co-adapt map and planner to look good together | The planner is **frozen and non-learned** during reward computation (§6.3). |
| Memorize the probe goals | Probe goals are **resampled every episode** from the GT-reachable set. |
| Get rewarded on cells it could never have seen | Every term is masked by the observability set $\mathcal{O}_t$ (and $\mathcal{L}_{cost}$-style masking in [arXiv:2210.08952](https://arxiv.org/abs/2210.08952) is the same trick). |
| Drift that keeps local structure but breaks global frame | This is exactly the **global vs local** split in [MapEval](https://github.com/JokerJohn/Cloud_Map_Evaluation) and the ATE-vs-local-consistency split of KITTI $t_{rel}$ (§2.3). Add a **loop-closure consistency** term: when the trajectory revisits a cell, penalize the disagreement between the pre- and post-loop-closure label. $R^{util}$ with long probe paths already partially captures this. |
| Overfit the reward (train metric ≠ eval metric) | Hold out a disjoint cell subset for reporting. Report on GOAT-Bench-style **cross-episode SR/SPL** ([2404.06609](https://arxiv.org/abs/2404.06609)) so the *fidelity* terms cannot be the whole story. |

### 7.4 Concrete implementation sketch for a small agent

```
# Map head output, per world cell i
p[i]        : Categorical(C+1)      # C semantic/traversability classes + `unknown`
# GT side
M*[i]       : Categorical(C)        # simulator ground truth
Obs[i]      : bool                  # cell in GT sensor frustum over trajectory so far
ObsAny[i]   : bool                  # cell observable from anywhere in the scene (precomputed)
Reach[i]    : bool                  # cell in GT reachable set
C*[i]       : float                 # GT cost-to-traverse  (section 6.1 weights)
```

1. **Mapper (supervised, differentiable).** SemExp-style: RGB-D → (differentiable geometric
   projection) → voxel sum → denoiser CNN → spatial warp + channel-wise max-pool into the
   global map → per-cell `Categorical`. Backprop $\mathcal{L}_{fid} + \mathcal{L}_{fab}$ through
   the whole thing — SemExp explicitly designed the projection to be differentiable for this
   ([arXiv:2007.00643](https://arxiv.org/abs/2007.00643)).
2. **Planner (frozen).** Build $\hat C_t$ from $\arg\max p[i]$ + the §6.1 class weights
   (capped at $C_{\max}$). Run 8-connected Dijkstra (or FMM) to all $K$ probe goals, and the
   same on $C^*$.
3. **Reward (every $N$ steps).** $R^{util} + \beta R^{cov} + \eta R^{ig}$.
4. **Policy.** PPO with a 25-step long-term-goal abstraction (the ANS/SemExp trick, which
   "reduces the time-horizon for exploration in RL exponentially") for the high-level goal
   choice over map cells, and a deterministic local controller for the low level.
5. **Graph layer.** Rebuild a room/portal/stairs/water graph from the raster every $N$ steps;
   run topological reachability on it and use it to (a) short-circuit $R^{util}$ for
   provably-unreachable probes, (b) do loop closure. (Follows
   [arXiv:2005.12256](https://arxiv.org/abs/2005.12256); VTM-Nav's re-localization idea,
   [arXiv:2607.14514](https://arxiv.org/abs/2607.14514).)
6. **Evaluation harness.** (a) Fidelity: soft-IoU / mIoU on held-out cells, ATE/RPE, %Cov.
   (b) Utility: SPL / SoftSPL on a route task, and **cross-episode** SR/SPL with the map
   *carried over* ([arXiv:2404.06609](https://arxiv.org/abs/2404.06609),
   [arXiv:2607.14514](https://arxiv.org/abs/2607.14514)). Report both — if fidelity improves
   without utility improving, you have exactly the MapEval "local/global" problem and the
   reward needs reweighting.

### 7.5 What to read first, in order

1. [arXiv:2605.22814](https://arxiv.org/abs/2605.22814) — *Remember to be Curious* (why naive IG rewards fail, and what fixes them)
2. [arXiv:2007.00643](https://arxiv.org/abs/2007.00643) — SemExp (the modular decomposition + differentiable map loss)
3. [arXiv:2004.05155](https://arxiv.org/abs/2004.05155) — Active Neural SLAM (%Cov definition, the 3.2 m visibility rule)
4. [arXiv:2210.08952](https://arxiv.org/abs/2210.08952) — learned cost maps + the three masked losses
5. [arXiv:2106.15648](https://arxiv.org/abs/2106.15648) — ensemble disagreement as information gain, UCB over map cells
6. [arXiv:2006.09367](https://arxiv.org/abs/2006.09367) — Semantic Curiosity (label-free self-supervision)
7. [GitHub: JokerJohn/Cloud_Map_Evaluation](https://github.com/JokerJohn/Cloud_Map_Evaluation) — AC/COM/CD/MME/AWD/SCS and the fidelity-vs-consistency framing
8. [arXiv:2005.12256](https://arxiv.org/abs/2005.12256) — Neural Topological SLAM (the graph layer)
9. [arXiv:2404.06609](https://arxiv.org/abs/2404.06609) + [arXiv:2607.14514](https://arxiv.org/abs/2607.14514) — lifelong / cross-episode evaluation and design constraints
