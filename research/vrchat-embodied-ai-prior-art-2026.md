# Prior Work: Embodied AI / Navigation Agents Inside VRChat
### Literature map as of late 2026 · research memo for an LLM-navigation VRChat plugin

**Scope of this search.** `web_fetch` was unavailable in this session (every hostname returned
*"resolves to a non-public IP address"*), so **no source page was opened directly**. Every factual
claim below comes from search-result titles/snippets returned by `web_search`,
`advanced_search`, and `platform_search` (github / hn / reddit / wikipedia). Where a snippet did
not establish a fact, it is marked **unverified**. URLs are cited even though they could not be
opened.

---

## (a) Direct verdict on VRChat-specific prior work

**No peer-reviewed paper was found that does embodied navigation inside VRChat from egocentric
pixels with no world-coordinate access.** That exact combination appears to be unpublished.

But the honest answer is not "nobody has touched this". The space splits into four buckets:

1. **Published LLM agents *inside* VRChat — yes, several; none of them navigate.**
   They are conversational/tutoring/NPC agents: ELLMA-T (ACM DIS 2025), an LLM NPC agent
   (ACM CHI EA 2024), an embodied conversational agent for speaking anxiety (ACM CHI 2026),
   and an LLM "sighted guide" for blind/low-vision users (CHI 2026).
2. **Published navigation agents in *commercial metaverse* — one, and it is explicitly
   VRChat-adjacent.** *Navigation Pixie* (IEEE ISMAR 2025) names VRChat, Resonite and Cluster in
   its introduction, but was **implemented on Cluster**, and it sidesteps the hard problem by
   consuming **structured spatial metadata** rather than solving pixels→coordinates.
3. **A generic VR navigation LLM framework — NavAI (2026)** — which reads *screenshots captured
   from the VR application*, i.e. architecturally the closest published analogue to this project.
   Its specific target VR application(s) could not be verified from snippets.
4. **Open-source, unpublished VRChat LLM-agent projects that already do much of this** —
   `HoppouAI/ProjectGabriel-Remastered` (Gemini Live + YOLO person/face tracking + OSC control,
   "walks around"), `Pomelo32141/VRchat_api_agent` (screen/audio observation + LLM planning +
   OSC/input execution), `gamio-22/vrchat-ai-agent` (ASR + local LLM + YOLO/Pose + VLM + OSC +
   autonomous avatar movement/gaze/expression), `MLShukai/pamiq-vrchat` (RL), `MLShukai/vrcpilot`
   (capture/OCR/input synthesis). **These are not papers, but they constrain novelty claims.**

**Practical reading:** the defensible novelty is *not* "an LLM agent that navigates a metaverse",
and *not* even "an LLM agent that navigates VRChat via OSC" (that exists in open source). It is
the **specific constraint set**: egocentric window capture + OSC velocity telemetry + SteamVR
poses, with **no depth and no world coordinates**, closed by a person detector + OSNet ReID +
traversability module + LLM tool use. That precise formulation was not found in any publication.

---

## (b) The five leads

### 1. VRChat as a robotics / embodied-AI platform

**Papers that use VRChat as a research platform (verified):**

| Work | Venue / year | What it does | VRChat role |
|---|---|---|---|
| *Crowdsourcing Virtual Reality Experiments using VRChat* (Saffo et al.) | ACM CHI EA 2020, DOI 10.1145/3334480.3382829 | Remote VR user-study methodology | Platform as experiment venue |
| *Techniques for using VRChat to Replace On-site Experiments* | IEEE ISMAR-Adjunct 2022 | Lab→VRChat study porting | Platform as experiment venue |
| *AvatarJudger: Content-Independent Avatar Ownership Detection…* | ACM UIST 2025, DOI 10.1145/3746059.3747665 | Cross-account avatar ownership detection | VRChat-style social metaverse |
| *Building LLM-based AI Agents in Social Virtual Reality* | ACM CHI EA 2024, DOI 10.1145/3613905.3651026 | GPT-4 LLM agent deployed **in VRChat as an NPC** for human-agent interaction | Direct VRChat agent |
| *ELLMA-T: an Embodied LLM-agent for Supporting English Language Learning in Social VR* | **ACM DIS 2025**, pp. 576–594, DOI 10.1145/3715336.3735786 (arXiv:2410.02406) | LLM embodied conversational agent, situated language tutoring | **Direct VRChat embodied LLM agent** |
| *LLM-based Embodied Conversational Agent for Reducing Foreign Language Speaking Anxiety in Social VR* | **ACM CHI 2026**, DOI 10.1145/3772318.3791068 | ECA in social VR | Social VR / VRChat-adjacent |
| *Understanding the Use of an LLM-Powered Guide to Make VR Accessible for BLV People* | arXiv:2603.09964, DOI 10.1145/3772318.3791143 (CHI 2026) | LLM "sighted guide" for navigation + Q&A in social VR | **Closest to navigation assistance** |

**None of these is a robotics/embodied-navigation paper.** They are HCI studies. Note in
particular that the BLV "sighted guide" is *navigation assistance for a human*, not an
autonomous embodied agent with its own pose estimate.

**The one navigation agent in a commercial metaverse — Navigation Pixie (this is the key paper):**

- **Title:** *Navigation Pixie: Implementation and Empirical Study Toward On-demand Navigation
  Agents in Commercial Metaverse*
- **Authors/affiliation:** Hikari Yanagawa (Cluster Metaverse Lab / Univ. of Tsukuba),
  Yuichi Hiroi (Cluster Metaverse Lab), Satomi Tokida (Univ. of Tokyo) — additional co-authors
  possible (**unverified**).
- **Year/venue:** submitted **5 Aug 2025**, arXiv:2508.03216; **IEEE ISMAR 2025** (author copy
  hosted as `2025_ISMAR_Navigation_Pixie.pdf`; a Cluster, Inc. press release announces an ISMAR
  2025 accepted paper with an oral presentation 8–12 Oct 2025).
- **Open source:** no repository found → **not found / presumed closed**.
- **Hardware:** **unverified**.
- **What it does:** an on-demand navigation agent with a "loosely coupled architecture that
  integrates **structured spatial metadata** with LLM-based natural language processing while
  **minimizing platform dependencies**".
- **Caveat that matters most for you:** it deliberately avoids the hard problem. It consumes
  **structured spatial metadata** from the metaverse rather than doing egocentric pixel→pose
  inference under a no-coordinates constraint. Its introduction explicitly frames
  **VRChat, Resonite and Cluster** as the motivating commercial platforms — so it is
  *VRChat-adjacent in framing, Cluster-implemented in fact*.
- **Also:** the press release states the technology was already productized into a commercial
  AI-agent reception service (**unverified** whether that service is the same system).

**Simulator platforms people actually use (the contrast set):**

| Platform | Paper | Venue / year | Open source | Notes |
|---|---|---|---|---|
| **Habitat 3.0** | arXiv:2310.13724 | **ICLR 2024** | Yes — `facebookresearch/habitat-lab` | **Most conceptually relevant**: co-habitat of robots + *humanoid avatars* + humans-in-the-loop |
| **AI2-THOR** | arXiv:1712.05474 | arXiv 2017 (**peer-reviewed venue unverified**) | Yes | Interactive near-photorealistic indoor scenes |
| **ProcTHOR** | arXiv:2206.06994 | **NeurIPS 2022** | Yes (AllenAI) | Procedural generation of embodied environments |
| **iGibson 2.0** | arXiv:2108.03272 | **CoRL 2021** (PMLR v164) | Yes | Object-centric household tasks |
| **ThreeDWorld** | arXiv:2007.04954 | arXiv 2020/2021 (**venue unverified**) | Yes | Multi-modal physical simulation |
| **BEHAVIOR-1K** | arXiv:2403.09227 | 2024; CoRL 2024 acceptance **unverified** | Yes (OmniGibson) | 1,000 household activities grounded in human surveys |
| **SAPIEN** | — | — | — | Named as a comparison platform inside the iGibson 2.0 paper; **not independently verified in this search** |
| **MineDojo** | arXiv:2206.08853 | **NeurIPS 2022** | Yes | Minecraft; 3,000+ tasks + internet-scale knowledge base |
| **Voyager** | arXiv:2305.16291 | **TMLR** (per secondary source; venue confidence moderate) | Yes | LLM lifelong-learning agent in Minecraft, skill library |

**Why this matters:** every one of these hands the agent **privileged ground-truth state**
(pose, depth, semantic maps, object states). None of them reproduces your constraint of
*pixels + velocity + HMD poses only*. Habitat 3.0 is the closest **task** analogue (humans as
avatars co-habiting with robots) but the polar opposite in **observability**.

---

### 2. SIMA / SIMA 2 (Google DeepMind) — the closest "commercial game from pixels" precedent

**SIMA 1**
- **Title:** *Scaling Instructable Agents Across Many Simulated Worlds*; SIMA Team, Google DeepMind.
- **Year:** arXiv:2404.10179, March 2024; blog 13 Mar 2024.
- **Venue:** arXiv technical report. **No peer-reviewed venue found — unverified whether published.**
- **What it does:** follows free-form natural-language instructions via **keyboard-and-mouse
  actions across a broad range of commercial video games** — reported as **nine** commercial 3D
  games including *No Man's Sky, Teardown, Valheim, Space Engineers*. Trained from human
  playthroughs (video + actions), **not** reward functions.
- **Input modality:** **pixels only** (video feed) — this is precisely the "agent navigates a
  commercial game from pixels" precedent.
- **Open source:** **No.** Closed weights. No official code/weights release found. The only
  public code is an **unofficial** third-party reimplementation (`kyegomez/SIMA`) that is not
  the real model.
- **Hardware:** **N/A / unverified** — no published weights, so it cannot be self-hosted.
  (Model-aggregator pages state there are "no hardware requirements" precisely because the
  weights are not published.)

**SIMA 2**
- **Title:** *SIMA 2: A Generalist Embodied Agent for Virtual Worlds*; SIMA Team, Google DeepMind.
- **Year:** arXiv:2512.04797, December 2025; blog **13 Nov 2025**.
- **Open source:** **No.** Closed weights, no download, no API (per aggregator listings).
- **Architecture:** built on a **Gemini** foundation model; backbone reported as a
  **Gemini Flash-Lite** checkpoint (tens of billions of parameters) with optional hierarchical
  "steering" by larger **Gemini Pro** (>100B) models — **secondary-source, moderate confidence**.
- **Capability delta vs SIMA 1:** moves from instruction-follower to interactive agent; layered
  training (human demonstrations + synthetic data + RL); reported to **self-generate training
  tasks and self-score**, "nearly doubling task success" and approaching human performance in
  unseen games — **secondary-source paraphrase, treat numbers as unverified**.
- **Hardware:** not applicable (closed).

**Verdict on SIMA as precedent:** it is the strongest *scientific* precedent for
"LLM/VLA agent that navigates a commercial 3D world purely from pixels". It is also **completely
unavailable** — closed weights, no API, trained at a scale no individual can reproduce. It is a
**citation and framing device, not a baseline**.

---

### 3. VRChat OSC ecosystem and the world-position question

#### What OSC actually exposes

| Namespace | Direction | Content | Source |
|---|---|---|---|
| `/avatar/parameters/*` | **read + write** | Custom avatar parameters (up to 8192 total custom params; 256-bit sync budget for synced ones) plus **built-in** params. `/avatar/change` fires on avatar switch. | [OSC Avatar Parameters](https://docs.vrchat.com/docs/osc-avatar-parameters), [Expressions wiki](https://wiki.vrchat.com/wiki/Expressions) |
| `/avatar/parameters/` built-ins | read | **`VelocityX`, `VelocityY`, `VelocityZ`, `VelocityMagnitude` (Float, m/s along each axis + total)**, `AngularY`, `Grounded` (bool, touching floor), `Upright`, `Seated`, `AFK`, `Voice`, `Viseme`, `GestureLeft/Right`, `ScaleFactor`, … | [VRChat Creation — Animator Parameters](https://creators.vrchat.com/avatars/animator-parameters/), [VRC School — Built-In VRC Parameters](https://vrc.school/docs/Avatars/VRC-Parameters), [VRChat Wiki — Expressions](https://wiki.vrchat.com/wiki/Expressions) |
| `/tracking/trackers/{1..8}/position` and `/rotation` | **input to VRChat** | Full-body tracker data you **send in** for calibrated FB IK. The docs are explicit: *"You must create your own program to **transmit** this data to VRChat."* It is **not** an avatar-pose output. | [OSC Trackers](https://docs.vrchat.com/docs/osc-trackers), [vrc-oscquery-lib/osc-trackers.md](https://github.com/vrchat-community/vrc-oscquery-lib/blob/main/osc-trackers.md) |
| `/input/*` | **write** | Movement/action injection: **axes expect float −1..1** (`/input/Vertical`, `/input/Horizontal`, `/input/LookHorizontal`), **buttons expect int 1/0** (`/input/Jump`, …). Warning in the docs: a `MoveForward` left at `1` "will continue to move you forward forever". | [OSC as Input Controller](https://docs.vrchat.com/docs/osc-as-input-controller), [osc docs Input.md](https://github.com/vrchat-community/osc/blob/db0a0d36/docs/Input.md), [Unofficial OSC Documentation](https://github.com/Lioncat6/OSC-Chat-Tools/wiki/Unofficial-OSC-Documentation) |
| **User Camera / Camera Dolly endpoints** | **read + write** | Added in **VRChat 2025.3.3**: *"Added OSC endpoints for interacting with the camera… All of the new endpoints have read/write access."* A follow-up feedback item states these give **world-space position only**: *"The new OSC camera endpoints only give world-space position info, and can only be set to world-space coordinates."* | [VRChat 2025.3.3 release notes](https://docs.vrchat.com/docs/vrchat-202533), [2025.3.3 Open Beta notes](https://docs.vrchat.com/docs/vrchat-202533-openbeta), [OSC wiki](https://wiki.vrchat.com/wiki/OSC), [camera-endpoints feedback](https://feedback.vrchat.com/feature-requests/p/new-osc-camera-endpoints-local-global-coordinates-anchor-mode) |
| `/chatbox/*` | write | Chatbox text (community-documented limits/wrapping) | [vrc-live-caption chatbox reference](https://github.com/Harry-Jing/vrc-live-caption/blob/main/docs/research/vrchat-chatbox-reference.md) |

Coordinate convention (verified): **left-handed Unity system**, Cartesian distances **in metres of
real-world space**, +X right, +Y up, +Z forward ([VRChat Wiki — OSC](https://wiki.vrchat.com/wiki/Open_Sound_Control)).

#### **Is VRChat world position / coordinate data obtainable via OSC? — Short answer: the avatar's, no. Not today.**

This is not an inference; it is the repeated, still-open state of the official channels:

- **Official feature request, opened 11 Nov 2022, still a request:**
  *"A useful addition to the OSC parameters would be to either send the **absolute position and
  rotation in the world** to the application or allow it to request it. **It is technically
  possible to calculate it using the velocity, but this doesn't account for teleporting/respawning.**"*
  → [feedback.vrchat.com — additional parameters for reading current player's position](https://feedback.vrchat.com/feature-requests/p/additional-parameters-for-reading-current-players-position)

  **This single sentence is the authoritative confirmation of your design's core workaround and
  its core failure mode.** Velocity integration is the platform-sanctioned stopgap, and
  teleport/respawn breaks it.

- **GitHub issue, open since 17 Feb 2022:** *"Expose the player's own position via Avatar OSC
  parameters"* — [vrchat-community/osc#43](https://github.com/vrchat-community/osc/issues/43)
- **GitHub issue, 11 Nov 2022:** *"Native Avatar Absolute X Y Z and rotation Parameters"* —
  [vrchat-community/osc#147](https://github.com/vrchat-community/osc/issues/147)
- **Discussion proving world-object poses do not export:** a user tried to send a cube's
  position/rotation out of VRChat over OSC — it works **in Editor play mode** but yields
  **0 output in a built/published world** —
  [vrchat-community/osc discussion #154](https://github.com/vrchat-community/osc/discussions/154)
- Consequence: **third-party OSC clients only ever speak for the local player** and cannot read
  world transforms. Anything requiring geometry must come from **Udon inside the world** — and
  `Player Positions` (`GetPosition`/`GetVelocity`, `Vector3` **in World Space**) is documented as
  world-author-only API — [VRChat Creation — Player Positions](https://creators.vrchat.com/worlds/udon/players/player-positions/),
  [UdonSharp VRChat API](https://udonsharp.docs.vrchat.com/vrchat-api/).

**Caveat on the built-in velocity params** — verified and important:
> *"VelocityX, VelocityY, VelocityZ, VelocityMagnitude (Float) — Movement speed in m/s along each
> axis and total magnitude. **Locally, playspace movement does not count; remotely, it does.**"*
> — [VRChat Wiki — Expressions](https://wiki.vrchat.com/wiki/Expressions), corroborated by
> [VRC School](https://vrc.school/docs/Avatars/VRC-Parameters) ("Note that locally, playspace
> moving doesn't count, but remotely it does") and
> [VRLabs/VRCSchool VRC-Parameters.md](https://github.com/VRLabs/VRCSchool/blob/main/docs/Avatars/VRC-Parameters.md).

So VelocityX/Y/Z are **avatar-local axes**, are **zero for real-world playspace translation when
measured locally**, and are clipped (AngularY documented as capping at ±1024). Naive integration
of them will silently under-count physical walking.

**The one genuinely new channel (2025.3.3+):** the **User Camera / Camera Dolly OSC endpoints**
are read/write and expose **world-space position** to a third-party OSC app. This is the only
recent VRChat feature found that hands external software a *world-space* coordinate. Two caveats:
(a) it is the **camera object**, not the avatar — whether it can be attached/anchored to the
avatar reliably is **unverified**; (b) the same feedback thread complains the endpoints *only*
offer world-space (no local/anchor mode), which was a limitation for the requester's use case.
Worth prototyping, but do not present it as an avatar-pose solution.

#### Open-source OSC ecosystem (practical prior art)

- **[vrchat-community/osc](https://github.com/vrchat-community/osc)** — official community spec repo (⭐248).
- **[vrchat-community/vrc-oscquery-lib](https://github.com/vrchat-community/vrc-oscquery-lib)** — OSCQuery service library; hosts the machine-readable `osc-trackers.md`.
- **[theepicsnail/vrchat_oscquery](https://github.com/theepicsnail/vrchat_oscquery)** — Python OSCQuery client/proxy (`pip install`).
- **[Lioncat6/OSC-Chat-Tools — Unofficial OSC Documentation](https://github.com/Lioncat6/OSC-Chat-Tools/wiki/Unofficial-OSC-Documentation)** — the de-facto complete parameter table.
- **[ZenithVal/OSCLeash](https://github.com/ZenithVal/OSCLeash)** (⭐168) — reads a stretched PhysBone parameter over OSC **and drives movement**; the canonical "OSC in → movement out" closed loop. Uses OSCQuery for parameter receiving.
- **[kushiemoon-dev/vrchat-osc-bridge](https://github.com/kushiemoon-dev/vrchat-osc-bridge)** — HTTP→OSC bridge; `POST /move {vertical, horizontal, look, duration}` — i.e. exactly the `/input/` axis-injection pattern you need, already packaged.
- **[MLShukai/vrcpilot](https://github.com/MLShukai/vrcpilot)** (PyPI `vrcpilot`) — Python automation toolkit for the VRChat desktop client (Windows/Linux): **launch, focus, capture, OCR, image-template detection, synthetic input**. This is the closest existing tooling to your window-capture + input-synthesis loop.
- **[MLShukai/pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat)** — "VRChat interfaces for PAMIQ", trains an agent that interacts with VRChat (RL lab tooling).
- **[sandraschi/vrchat-mcp](https://github.com/sandraschi/vrchat-mcp)** — MCP server exposing `manage_avatar / manage_osc / manage_world / manage_economy / manage_input / manage_system` as **runtime/OSC/REST telemetry**; explicitly **no world-authoring**.
- **[Duinrahaic/VRCDollyManager](https://github.com/Duinrahaic/VRCDollyManager)** — OSCQuery-based camera-dolly manager (relevant to the new camera endpoints).
- **[benaclejames/VRCFaceTracking](https://github.com/benaclejames/VRCFaceTracking)** (⭐912), **[200Tigersbloxed/HRtoVRChat_OSC](https://github.com/200Tigersbloxed/HRtoVRChat_OSC)**, **[vard88508/vrc-osc-miband-hrm](https://github.com/vard88508/vrc-osc-miband-hrm)**, **[cyberkitsune/vrc-osc-scripts](https://github.com/cyberkitsune/vrc-osc-scripts)** — ecosystem examples of "external sensor/ML → OSC parameter".
- **[OscToys/OscGoesBrrr](https://github.com/OscToys/OscGoesBrrr)** — external ML-driven OSC integration (⭐123).
- **[AndrewAltimit/avatar — osc-runtime](https://github.com/AndrewAltimit/avatar/blob/main/docs/reference/osc-runtime.md)** — Rust OSC runtime: set params, **push input-controller axes/buttons**, switch avatars, **read parameter updates back**.

**Open-source VRChat LLM agents (not papers — but direct engineering prior art):**

- **[gamio-22/vrchat-ai-agent](https://github.com/gamio-22/vrchat-ai-agent)** — described as: speech recognition, **local LLM**, VOICEVOX-family TTS, **VRChat OSC**, **YOLO/Pose**, **VLM**, long-term memory, emotion/desire model, autonomous behaviour — *"not only conversation but avatar **movement**, gaze, expressions"*. **This is the closest thing to your exact project that exists.**
- **[HoppouAI/ProjectGabriel-Remastered](https://github.com/HoppouAI/ProjectGabriel-Remastered)** — "2026 remaster… real-time VRChat AI powered by **Gemini Live** with voice, **YOLO person and face tracking**, **OSC control**, memory… **Gabriel walks around**, talks to people, remembers who they are." Also see org page [HoppouAI](https://github.com/HoppouAI) ("Screen capture, person and face tracking, and spatial awareness").
- **[Pomelo32141/VRchat_api_agent](https://github.com/Pomelo32141/VRchat_api_agent)** — "**Human-like VRChat agent with low-frequency LLM planning and instinct loops**… observes **screen/audio**, plans with LLM at controlled frequency, executes actions via **OSC/local input**." (Note: mirrored at `p.rst.im` in results.)
- **[savan1304/VRChat-Guide](https://github.com/savan1304/VRChat-Guide)** — task-oriented onboarding/event-recommendation AI agent in VRChat.
- **[MLShukai/pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat)**, **[HeyMengxu/ELLMA-T](https://github.com/HeyMengxu/ELLMA-T)** (the published ELLMA-T system's code).

**Also relevant:** **Udon AI Navigation** — official VRChat world-authoring feature providing NavMesh pathfinding so NPCs "can intelligently move around the game world"
([VRChat Creation — AI Navigation](https://github.com/vrchat-community/creator-docs/blob/fce3082e/Docs/docs/worlds/udon/ai-navigation.md?plain=1),
[AI Navigation Example](https://vrc-beta-docs.netlify.app/worlds/examples/ai-navigation)).
If you author/own the world, the entire coordinate problem disappears. If you visit someone else's
world, it is unavailable — which is exactly your situation.

---

### 4. Game-engine / game-observation navigation datasets

These are the real sim-to-real-adjacent precedents.

| Work | Venue / year | Notes |
|---|---|---|
| **From Gaming to Research: GTA V for Synthetic Data Generation for Robotics and Navigations** — arXiv:2502.12303 | arXiv 2025 | Direct precedent: **commercial game → robotics/navigation synthetic data** |
| **Project Malmo** (Johnson, Hofmann, Hutton, Bignell, Microsoft) | **IJCAI 2016** | Minecraft-based AI experimentation platform; open source |
| **MineDojo** — arXiv:2206.08853 | **NeurIPS 2022** | Open-ended agents in Minecraft; 3,000+ tasks; internet-scale knowledge base (YouTube/wiki/Reddit), open source |
| **Voyager** — arXiv:2305.16291 | **TMLR** (moderate confidence) | LLM lifelong-learning agent in Minecraft, skill library, open source |
| **ViZDoom Competitions: Playing Doom from Pixels** — arXiv:1809.03470 | IEEE ToG (per secondary listing) | Deep RL navigation from raw pixels in Doom |
| **Dreamer 4** — arXiv:2509.24527 | arXiv 2025 | Trains agents **inside a learned world model**; Minecraft; surpassing prior world models |
| **REGEN: Real-Time Photorealism Enhancement in Games** — arXiv:2508.17061 | arXiv 2025 | Game→photorealism translation |
| **A Hybrid Approach for Closing the Sim2real Appearance Gap in Game Engine Synthetic Datasets** — arXiv:2605.02291 | arXiv 2026 | Explicit **sim2real appearance gap** from game-engine data |
| **Walking to Learn / "Grand Theft Auto-Based Cycling Simulator…"** — PMC10098922 | PMC 2023 | GTA as a study simulator |
| **"Improving Deep Object Detection Algorithms for Game Scenes"** | via Academia.edu | **Low-quality source; treat as a pointer only** |

**Cross-cutting caveat:** virtually all of the game-navigation literature either (a) has access to
the game's internal state/API (Malmo, MineDojo, ViZDoom), or (b) learns a world model, or (c) is
DeepMind-scale and closed. **None of them is "third-party plugin, pixels + velocity + HMD only,
inside someone else's commercial online social world."** That combination is your gap.

---

### 5. Anime / stylized-rendering robustness for vision models

**Be honest: I found no study that measures CLIP / person detectors / OSNet ReID / VLMs
specifically on VRChat avatar renders.** There is a large adjacent literature, and it is mostly
built for *2D illustration* (Danbooru-style art), not *real-time 3D anime-style avatar rendering* —
which is a different domain (3D shading, PhysBones, per-frame motion blur, occluding nameplates,
mirrors, non-photoreal lighting).

**Evidence that the domain gap is real:**
- **YOLOv14** (arXiv:2608.04720) states real-time detectors "**degrade sharply on non-ideal
  inputs — fisheye distortion, game-rendered content**, aerial views, and 360° panoramas." This is
  the most direct published acknowledgement that **game-rendered content degrades detectors**.
- **SADGE: Structure and Appearance Domain Gap Estimation of Synthetic and Real Data**
  (arXiv:2605.22467) — a quantitative synthetic-vs-real similarity metric.
- **An Investigation of the Domain Gap in CLIP-Based Person Re-Identification**,
  *MDPI Sensors* 25(2):363, 2025 — **CLIP-based ReID degrades on unseen domains**; the canonical
  statement of the gap.
- **Limitations of Zero-Shot CLIP on Fine-Grained Cultural Art, and the Effectiveness of Linear
  Adaptation** (Baldrati, Bertini, Uricchio, Del Bimbo) — **zero-shot CLIP is weak on
  non-photographic art; linear adaptation helps.** Directly relevant to "will CLIP/VLM understand
  an anime avatar frame".
- **When Visual Quality Misleads: Intent Recognition under Rendered Avatar Distortions**
  (arXiv:2609.27560, 2026) — rendered-avatar distortion degrades downstream recognition.
- **A multidimensional measurement of photorealistic avatar quality of experience**
  (arXiv:2411.09066) — avatar QoE metrics.
- Methodology point: **AI Avatar / VRChat expression-syncing comparison** (commercialtrade blog) —
  not peer-reviewed; ignore for claims.

**Evidence that synthetic/anime data can *help* rather than hurt:**
- **Surpassing Real-World Source Training Data: Random 3D Characters for Generalizable Person
  Re-Identification** — Wang, Liao, Shao; **ACM Multimedia 2020**, DOI 10.1145/3394171.3413815
  (arXiv:2006.12774). **The single most useful result for your OSNet design**: random **3D
  characters** as source training data generalised **better** than real-world source data. This
  is a strong argument for fine-tuning ReID on synthetic/3D avatar imagery.
- **OSNet** baseline: *Omni-Scale Feature Learning for Person Re-Identification*,
  **ICCV 2019**; extended as *Learning Generalisable Omni-Scale Representations for Person
  Re-Identification*, **arXiv:1910.06827** (TPAMI 2021). Note the paper's own framing is
  **cross-dataset generalisation** — precisely your risk.

**Anime-specific detector / embedder tooling (verified, open source):**
- **[nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)** — the
  classic OpenCV LBP cascade anime/manga face detector (original release **2011**, repo since 2014);
  mirror: [Mukosame/lbpcascade_animeface](https://github.com/Mukosame/lbpcascade_animeface). Face
  detection only — **not** a full-body person detector, and no published accuracy on 3D renders.
- **[deepghs/imgutils](https://github.com/deepghs/imgutils)** — `detect_person()` uses **YOLOv8
  models trained on the AniDet3 dataset** (from Roboflow) for anime **whole-body person**
  detection ([docs](https://dghs-imgutils.deepghs.org/HEAD/api_doc/detect/person.html),
  [source](https://github.com/deepghs/imgutils/blob/main/imgutils/detect/person.py)).
  Also ships an anime image **classifier into `3d` / `bangumi` / `comic` / `illustration` /
  `not_painting`** ([docs](https://dghs-imgutils.deepghs.org/HEAD/api_doc/validate/classify.html))
  — i.e. **an off-the-shelf model that explicitly distinguishes 3D renders from illustrations**,
  directly useful as a sanity gate for "am I looking at a 3D avatar or a drawn image".
- Datasets: **[deepghs/anime_person_detection](https://huggingface.co/datasets/deepghs/anime_person_detection)**,
  **[deepghs/csip](https://huggingface.co/datasets/deepghs/csip)** and
  **[deepghs/csip_v1](https://huggingface.co/datasets/deepghs/csip_v1)** (Contrastive anime Style
  Image Pre-Training; zero-shot style classification),
  **[Library-Mutsumi/csip_eval](https://huggingface.co/datasets/Library-Mutsumi/csip_eval)**,
  **[huggan/anime-faces](https://huggingface.co/datasets/huggan/anime-faces)**.
- **Anime CLIP variants:**
  **[OysterQAQ/DanbooruCLIP](https://huggingface.co/OysterQAQ/DanbooruCLIP)** (CLIP ViT-L/14
  fine-tuned on Danbooru2021),
  **[v2ray/clipbooru](https://huggingface.co/v2ray/clipbooru)** (CLIP + Danbooru-tag head),
  **[pixai-labs/pixai-tagger-v0.9](https://huggingface.co/pixai-labs/pixai-tagger-v0.9)**
  (multi-label anime tagger, fresh Danbooru 2025-01 snapshot, ~13.5k tags).
- Related: **AniMatrix** (arXiv:2605.03652) — anime video generation "thinks in art, not physics";
  useful as evidence that anime motion *deliberately* violates physical priors, which is a real
  hazard for a traversability module trained on physical assumptions.

**Verdict for lead 5:** the *direction* of the risk is well documented (stylized/game-rendered
input degrades CLIP, ReID and detectors), and the *mitigation* is documented (fine-tune on
synthetic 3D characters; ACM MM 2020). But **the specific numbers for VRChat avatar renders do
not exist in the literature** — that is a genuine, publishable measurement gap, and you should
present it as such rather than citing a number.

---

## (c) Practical mechanisms for telemetry — what actually exists

1. **Egocentric frames:** VRChat exposes no official frame API to third parties. The established
   routes are desktop **window capture** (`MLShukai/vrcpilot` does capture + OCR + template
   detection), SteamVR **overlay/camera** APIs, or VRChat's **stream mode / Spout** output. The
   VRChat **camera** system also has new OSC endpoints (§3) — the camera *object* is addressable,
   its *image* is not delivered over OSC.
2. **Avatar velocity:** `/avatar/parameters/VelocityX|Y|Z|VelocityMagnitude` over OSC, **m/s**,
   **avatar-local**, and **playspace translation is excluded locally**. This is the official
   "velocity integration" workaround named in VRChat's own feedback thread, and it **breaks on
   teleport/respawn**. Integration drift is therefore not a bug in your design — it is the
   documented limitation of the only sanctioned signal.
3. **Built-in avatar params:** `Grounded`, `Upright`, `Seated`, `AFK`, `Voice`, `Viseme`,
   `GestureLeft/Right`, `AngularY`, `ScaleFactor`, and custom params you author. All
   read-over-OSC **only if the local avatar defines them in its Playable Layers** — so your plugin
   must ship a companion avatar (or a parameter contract) to guarantee the telemetry exists.
   **`ScaleFactor` matters**: avatar world scale changes the mapping from velocity to world metres.
4. **SteamVR HMD/controller poses:** available from the runtime as **tracking-space** poses.
   They are **not aligned to VRChat world coordinates** and contain no playspace-origin↔world
   mapping, so they give **relative** motion and body/head orientation only. (Verified
   indirectly: the community's known limitation is that SteamVR recentering/standing recenter
   changes the tracking origin — [steamvr_unity_plugin#829](https://github.com/ValveSoftware/steamvr_unity_plugin/issues/829).)
5. **Tracker positions are inputs, not outputs.** `/tracking/trackers/*` is data *you send in*.
   Do not design around reading it back.
6. **World-object/world-coordinate data: not obtainable from outside.** Editor works, published
   world emits nothing (osc discussion #154); absolute avatar X/Y/Z has been an **open request
   since 2022** (#43, #147, official feedback). **The only way to get true world coordinates is
   to be the world author and use Udon** (`GetPosition`, `GetVelocity` in World Space), or to own
   a world you can publish a telemetry bridge into.
7. **The one new world-space channel: OSC User Camera / Camera Dolly endpoints (2025.3.3+),
   read/write, world-space.** This is the most promising recent development for absolute
   coordinates. Whether it can be bound to the avatar rather than a free camera is **unverified** —
   prototype before relying on it.
8. **Movement control:** `/input/Vertical|Horizontal|LookHorizontal` as float −1..1, buttons as
   int 1/0, **and axes must be reset to 0** or the avatar moves forever. Packaged precedents:
   `vrchat-osc-bridge` `POST /move`, `OSCLeash`, `pamiq-vrchat`.
9. **Your remaining state estimate must be reconstructed** from velocity integration + HMD
   relative poses + visual odometry/landmark matching from the captured frames — and you should
   detect and re-anchor on **teleport/respawn events** (`Grounded`/`Seated` transitions, sudden
   velocity discontinuities) since the platform gives you no teleport signal.

---

## (d) Full URL list

**VRChat official docs & API**
- https://docs.vrchat.com/docs/osc-avatar-parameters
- https://docs.vrchat.com/docs/osc-trackers
- https://docs.vrchat.com/docs/osc-as-input-controller
- https://docs.vrchat.com/docs/player-positions
- https://docs.vrchat.com/docs/vrchat-202533
- https://docs.vrchat.com/docs/vrchat-202533-openbeta
- https://docs.vrchat.com/
- https://creators.vrchat.com/avatars/animator-parameters/
- https://creators.vrchat.com/worlds/udon/players/player-positions/
- https://wiki.vrchat.com/wiki/OSC
- https://wiki.vrchat.com/wiki/Open_Sound_Control
- https://wiki.vrchat.com/wiki/Expressions
- https://wiki.vrchat.com/wiki/Expressions/zh-hans
- https://wiki.vrchat.com/wiki/Inverse_Kinematics
- https://wiki.vrchat.com/wiki/Community:OSC_Resources
- https://udonsharp.docs.vrchat.com/vrchat-api/
- http://vrchat.wikidot.com/worlds:guides:player-tracking
- https://vrc.school/docs/Avatars/VRC-Parameters
- https://vrc-beta-docs.netlify.app/worlds/examples/ai-navigation

**VRChat OSC position: the negative evidence**
- https://feedback.vrchat.com/feature-requests/p/additional-parameters-for-reading-current-players-position
- https://feedback.vrchat.com/feature-requests/p/new-osc-camera-endpoints-local-global-coordinates-anchor-mode
- https://github.com/vrchat-community/osc/issues/43
- https://github.com/vrchat-community/osc/issues/147
- https://github.com/vrchat-community/osc/discussions/154
- https://feedback.vrchat.com/bug-reports/p/avatar-position-and-bone-synchronization-stopped-after-osc-update

**VRChat OSC tooling / libraries / agents**
- https://github.com/vrchat-community/osc
- https://github.com/vrchat-community/osc/blob/db0a0d36/docs/Input.md
- https://github.com/vrchat-community/osc/wiki/Avatar-Parameters
- https://github.com/vrchat-community/vrc-oscquery-lib/blob/main/osc-trackers.md
- https://github.com/vrchat-community/creator-docs/blob/fce3082e/Docs/docs/avatars/animator-parameters/index.md
- https://github.com/vrchat-community/creator-docs/blob/fce3082e/Docs/docs/worlds/udon/ai-navigation.md
- https://github.com/theepicsnail/vrchat_oscquery
- https://github.com/Lioncat6/OSC-Chat-Tools/wiki/Unofficial-OSC-Documentation
- https://github.com/Lioncat6/OSC-Chat-Tools/wiki/Unofficial-OSC-Documentation/b875fb73ebe994f1b902334dbdd860a40ea8e198
- https://github.com/Lioncat6/OSC-Chat-Tools/wiki/Unofficial-OSC-Documentation/5ce419f79c08d8269845129a2088061dd95dd061
- https://github.com/ZenithVal/OSCLeash
- https://github.com/ZenithVal/OSCLeash/releases
- https://github.com/kushiemoon-dev/vrchat-osc-bridge
- https://github.com/MLShukai/vrcpilot
- https://pypi.org/project/vrcpilot
- https://github.com/MLShukai/pamiq-vrchat
- https://github.com/sandraschi/vrchat-mcp
- https://github.com/sandraschi/vrchat-mcp/blob/master/docs/Building_VRChat_Worlds_With_Unity3D_MCP.md
- https://github.com/Duinrahaic/VRCDollyManager
- https://github.com/benaclejames/VRCFaceTracking
- https://github.com/200Tigersbloxed/HRtoVRChat_OSC
- https://github.com/vard88508/vrc-osc-miband-hrm
- https://github.com/cyberkitsune/vrc-osc-scripts
- https://github.com/OscToys/OscGoesBrrr
- https://github.com/AndrewAltimit/avatar/blob/main/docs/reference/osc-runtime.md
- https://github.com/Harry-Jing/vrc-live-caption/blob/main/docs/research/vrchat-chatbox-reference.md
- https://github.com/VRLabs/VRCSchool/blob/main/docs/Avatars/VRC-Parameters.md
- https://github.com/gamio-22/vrchat-ai-agent
- https://github.com/HoppouAI/ProjectGabriel-Remastered
- https://github.com/HoppouAI/ProjectGabriel-Remastered/blob/main/README.md
- https://github.com/HoppouAI
- https://github.com/Pomelo32141/VRchat_api_agent
- https://github.com/Pomelo32141/VRchat_api_agent/issues
- https://github.com/savan1304/VRChat-Guide
- https://github.com/gummidot/vrchat-agentic-tools
- https://github.com/ggg123124/vrchat-assistant
- https://github.com/HeyMengxu/ELLMA-T
- https://github.com/niaka3dayo/agent-skills-vrc-udon
- https://github.com/ValveSoftware/steamvr_unity_plugin/issues/829

**VRChat / social VR published research**
- https://arxiv.org/abs/2508.03216 — Navigation Pixie (ISMAR 2025)
- https://arxiv.org/html/2508.03216v1
- https://arxiv.org/pdf/2508.03216v1.pdf
- https://arxiv.org/pdf/2508.03216
- https://doi.org/10.48550/arxiv.2508.03216
- https://yhiroi.github.io/assets/pdf/2025_ISMAR_Navigation_Pixie.pdf
- https://alphaxiv.org/abs/2508.03216
- https://arxiv.gg/abs/2508.03216
- https://prtimes.jp/main/html/rd/p/000000376.000017626.html
- https://arxiv.org/abs/2410.02406 — ELLMA-T
- https://arxiv.org/html/2410.02406
- https://arxiv.org/pdf/2410.02406
- https://dl.acm.org/doi/10.1145/3715336.3735786 — ELLMA-T, ACM DIS 2025
- https://dblp.org/pid/389/4584.html
- https://dl.acm.org/doi/full/10.1145/3613905.3651026 — LLM agents in social VR (CHI EA 2024)
- https://dl.acm.org/doi/pdf/10.1145/3613905.3651026
- https://dl.acm.org/doi/full/10.1145/3772318.3791068 — CHI 2026 ECA
- https://dl.acm.org/doi/10.1145/3772318.3791143 — LLM BLV guide
- https://arxiv.org/pdf/2603.09964v1
- https://arxivlens.com/paperview/details/understanding-the-use-of-a-large-language-model-powered-guide-to-make-virtual-reality-accessible-for-blind-and-low-vision-people-8174-83b06d77
- https://dl.acm.org/doi/fullHtml/10.1145/3334480.3382829 — Crowdsourcing VR experiments using VRChat
- https://dl.acm.org/doi/10.1145/3544548.3581103 — Social VR as a mental health tool
- https://vis.khoury.northeastern.edu/pubs/Saffo2020CrowdsourcingVirtualReality/
- https://www.researchgate.net/profile/David-Saffo/publication/339372695_Crowdsourcing_Virtual_Reality_Experiments_using_VRChat/links/5e5697d24585152ce8f251f3/Crowdsourcing-Virtual-Reality-Experiments-using-VRChat.pdf
- https://exa.ai/library/publication/qw5fp1ygn9h — Techniques for using VRChat to replace on-site experiments (ISMAR-Adjunct 2022)
- https://dl.acm.org/doi/full/10.1145/3746059.3747665 — AvatarJudger (UIST 2025)
- https://arxiv.org/abs/2603.25223 — Understanding newcomer persistence in social VR
- https://arxiv.org/html/2410.11869 — SLR of VR as a social platform
- https://arxiv.org/html/2412.20266v1 — 20-year SLR of social VR collaboration
- https://arxiv.org/html/2104.05030v1 — Social VR ethics
- https://arxiv.org/pdf/2604.00592 — social VR over the past decade
- https://www.sciencedirect.com/science/article/pii/S1755458624000392 — social relations/spatiality in VR
- https://www.tandfonline.com/doi/full/10.1080/10447318.2026.2620647 — Dialogs with GenAI NPCs
- https://arxiv.org/abs/2601.03251 — NavAI (arXiv)
- https://arxiv.org/pdf/2601.03251v1
- https://doi.org/10.48550/arxiv.2601.03251
- https://link.springer.com/article/10.1007/s10515-026-00676-z — NavAI (Automated Software Engineering, 2026)
- https://2025.aiwareconf.org/profile/xueqin
- http://papers.iafor.org/wp-content/uploads/papers/acp2021/ACP2021_59485.pdf — VRChat study 2021

**SIMA / SIMA 2**
- https://arxiv.org/abs/2404.10179 — SIMA 1
- https://arxiv.org/html/2404.10179v3
- https://arxiv.org/pdf/2404.10179
- https://doi.org/10.48550/arxiv.2404.10179
- https://deepmind.google/blog/sima-generalist-ai-agent-for-3d-virtual-environments/
- https://newscientist.com/article/2422101-google-ai-learns-to-playopen-world-video-games-by-watching-them
- https://the-decoder.com/google-deepminds-new-ai-agent-plays-games-using-only-natural-language
- https://academy.dair.ai/papers/sima
- https://encord.com/blog/google-deepmind-sima-ai-agent/
- https://gputps.com/ai-models/1391-sima
- https://github.com/kyegomez/SIMA — UNOFFICIAL reimplementation
- https://arxiv.org/html/2512.04797v1 — SIMA 2
- https://arxiv.org/html/2512.04797
- https://arxiv.org/pdf/2512.04797.pdf
- https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/
- https://genie.caelinya.im/_ext/storage/deepmind-media/DeepMind.com/Blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/SIMA_Tech_Report_2025.pdf
- https://www.youtube.com/watch?v=Zphax4f6Rls
- https://alphaxiv.org/overview/2512.04797
- https://emergentmind.com/papers/2512.04797
- https://emergentmind.com/topics/sima-2
- https://paper-graph-ui.vercel.app/p/agents/2512.04797-SIMA-2-A-Generalist-Embodied-Agent-for-Virtual-Worlds
- https://gputps.com/ai-models/110-sima-2
- https://simagame.com/sima-2-download
- https://vinesmsuic.github.io/paper-world-model-agent
- https://aionda.blog/en/posts/sima-2-gemini-3-agi-real-time-agent
- https://artificialintelligencemonaco.substack.com/p/sima-2-and-general-purpose-robotics
- https://lifetips.alibaba.com/tech-efficiency/what-is-google-deepmind-sima
- https://kaustubhsridhar.github.io/cv.pdf
- https://wew.twstalker.com/janexwang

**Embodied-AI simulators / navigation**
- https://arxiv.org/abs/2310.13724 — Habitat 3.0
- https://arxiv.org/html/2310.13724v1
- https://proceedings.iclr.cc/paper_files/paper/2024/hash/430894999584d0bd358611e2ecf00b15-Abstract-Conference.html — Habitat 3.0 @ ICLR 2024
- https://aihabitat.org/habitat3/
- https://ai.meta.com/research/publications/habitat-3-0-a-co-habitat-for-humans-avatars-and-robots/
- https://github.com/facebookresearch/habitat-lab
- https://arxiv.org/abs/1712.05474 — AI2-THOR
- http://ai2thor.allenai.org/publications
- https://arxiv.org/html/2206.06994v1 — ProcTHOR
- https://proceedings.neurips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html
- https://proceedings.neurips.cc/paper_files/paper/2022/file/27c546ab1e4f1d7d638e6a8dfbad9a07-Paper-Conference.pdf
- https://arxiv.org/pdf/2108.03272 — iGibson 2.0
- https://proceedings.mlr.press/v164/li22b/li22b.pdf
- https://alphaxiv.org/abs/2108.03272
- https://arxiv.org/html/2007.04954v2 — ThreeDWorld
- https://arxiv.org/abs/2403.09227 — BEHAVIOR-1K
- https://doi.org/10.48550/arxiv.2403.09227
- https://behavior.stanford.edu/index.html
- https://ui.adsabs.harvard.edu/abs/arXiv:2403.09227
- https://huggingface.co/datasets/StarVLA/BEHAVIOR-1K
- https://2024.corl.org/
- https://arxiv.org/pdf/2206.08853 — MineDojo
- https://voyager.minedojo.org/
- https://mrpeppersdev.github.io/agent-infrastructure-landscape/systems/minedojo--minedojo-org
- https://arxiv.org/abs/2305.16291 — Voyager
- https://azimuth.plus/en/paper/voyager
- https://github.com/ndqkhanh/agent-research-book/blob/main/docs/89-voyager-deep.md
- https://arxiv.org/html/2509.12129v1 — Embodied Navigation Foundation Model
- https://arxiv.org/pdf/2607.26148v3 — Minimal-interface zero-shot VLN agents
- https://arxiv.org/pdf/2609.15195.pdf — HarnessVLN
- https://arxiv.org/pdf/2609.34276 — NavHarness
- https://arxiv.org/pdf/2606.19948.pdf — DialNav
- https://arxiv.org/pdf/2609.20388v2 — Navi-Agent
- https://arxiv.org/pdf/2608.17512v1 — Embodied-Navigator
- https://ui.adsabs.harvard.edu/abs/2026arXiv260206427Z/abstract — Indoor-outdoor embodied navigation

**Game engines / game-observation navigation**
- https://arxiv.org/html/2502.12303v1 — GTA V for synthetic data
- https://www.ijcai.org/Proceedings/16/Papers/643.pdf — Project Malmo (IJCAI 2016)
- https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/johnson-malmo-platform-camera-ready.pdf
- https://ar5iv.labs.arxiv.org/html/1809.03470 — ViZDoom
- https://www.semanticscholar.org/paper/ViZDoom-Competitions%3A-Playing-Doom-From-Pixels-Wydmuch-Kempka/de0f96ebabb75700a9febcfe4ee0c5eb0adea7ba
- https://arxiv.org/pdf/2509.24527.pdf — Dreamer 4
- https://arxiv.org/html/2508.17061v2 — REGEN
- https://arxiv.org/html/2605.02291 — sim2real appearance gap, game engines
- https://arxiv.org/html/2107.08398 — Unsupervised skill discovery in Minecraft
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10098922/ — GTA-based cycling simulator
- https://www.researchgate.net/publication/380007638_Weapon_Violence_Dataset_20_A_synthetic_dataset_for_violence_detection
- https://dl.acm.org/doi/10.1145/3795886 — SFDA on GTA5/SYNTHIA
- https://www.academia.edu/78182962/Improving_Deep_Object_Detection_Algorithms_for_Game_Scenes
- https://riunet.upv.es/bitstreams/b5696b26-e8fd-499f-966a-6360e6cc2a37/download — DeepMind Lab
- https://sscalabrino.github.io/files/2025/TOSEM2025AutomaticIdentificationOf.pdf
- https://arxiv.org/html/2406.08231v1 — DCNN glitch detection in games

**HRI / XR reviews**
- https://www.sciopen.com/article/10.3934/era.2023121 — VR in HRI
- https://www.aimspress.com/article/doi/10.3934/era.2023121?viewType=HTML
- https://arxiv.org/pdf/2609.31138 — Adaptive XR on HRI across reality-virtuality continuum
- https://arxiv.org/pdf/2602.15840.pdf — Decade of HRI through immersive lenses
- https://www.reddit.com/r/virtualreality/comments/1lknrwj/ — H2L Capsule Interface robot teleoperation
- https://researchsquare.com/article/rs-10484850/v1.pdf?c=1785324820000
- https://arxiv.org/html/2506.0555v1 — Embodied AI Agents: Modeling the World
- https://news.ncsu.edu/2025/10/embodied-ai-vr-learning/

**Anime / stylized-rendering robustness**
- https://arxiv.org/html/2608.04720 — YOLOv14 (game-rendered degradation)
- https://arxiv.org/pdf/2605.22467 — SADGE domain gap estimation
- https://www.mdpi.com/1424-8220/25/2/363 — Domain gap in CLIP-based person ReID (Sensors 2025)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11769178/
- https://exa.ai/library/publication/l848rb7w5fv — Limits of zero-shot CLIP on fine-grained art
- https://arxiv.org/abs/2609.27560 — Intent recognition under rendered avatar distortions
- https://arxiv.org/html/2411.09066v3 — Photorealistic avatar QoE
- https://arxiv.org/pdf/2006.12774v2 — Random 3D characters for generalizable ReID
- https://arxiv.org/html/2006.12774v1
- https://dl.acm.org/doi/10.1145/3394171.3413815 — ACM MM 2020
- https://research.uaeu.ac.ae/en/publications/surpassing-real-world-source-training-data-random-3d-characters-f
- https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html — OSNet
- https://arxiv.org/html/1910.06827v5 — OSNet TPAMI extension
- https://github.com/nagadomi/lbpcascade_animeface
- https://github.com/Mukosame/lbpcascade_animeface
- https://github.com/deepghs/imgutils
- https://github.com/deepghs/imgutils/blob/main/imgutils/detect/person.py
- https://dghs-imgutils.deepghs.org/HEAD/api_doc/detect/person.html
- https://dghs-imgutils.deepghs.org/HEAD/api_doc/validate/classify.html
- https://huggingface.co/datasets/deepghs/anime_person_detection
- https://huggingface.co/datasets/deepghs/csip
- https://huggingface.co/datasets/deepghs/csip_v1
- https://huggingface.co/datasets/Library-Mutsumi/csip_eval
- https://huggingface.co/datasets/huggan/anime-faces
- https://huggingface.co/OysterQAQ/DanbooruCLIP
- https://huggingface.co/v2ray/clipbooru
- https://huggingface.co/pixai-labs/pixai-tagger-v0.9
- https://model.aibase.com/models/details/1915694074650320898
- https://arxiv.org/html/2605.03652 — AniMatrix
- https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_PAniC-3D_Stylized_Single-View_3D_Reconstruction_From_Portraits_of_Anime_Characters_CVPR_2023_paper.pdf
- https://link.springer.com/article/10.1007/s00371-025-04047-9
- https://github.com/haohua13/anime-your-photos
- https://github.com/freedomofkeima/transfer-learning-anime
- https://awesome.ecosyste.ms/projects/github.com%2Ffreedomofkeima%2Ftransfer-learning-anime
- https://github.com/ryogrid/anime-illust-image-searcher

**Other adjacent**
- https://arxiv.org/html/2506.04606v1 — SmartAvatar VLM avatar generation
- https://dl.acm.org/doi/10.1145/3746059.3747665
- https://arxiv.org/abs/2511.18677 — Sketch person ReID
- https://fugumt.com/fugumt/paper_check/2511.18677v1_enmode
- https://fugumt.com/fugumt/paper_check/2508.03216v1_enmode
- https://github.com/Future-AiLab/ABot-Navigation
- https://pure.kaist.ac.kr/en/publications/target-driven-visual-navigation-in-indoor-scenes-using-deep-reinf
- https://www.freelancer.com/projects/transformer-model/vrchat-transformer-model-development
- https://github.com/ruslanmv/3D-Avatar-Chatbot/commit/eddd65a3186a333aa6fcd24bdb39cc81b8bb48f3
- https://p.rst.im/q/Github.com/Pomelo32141/VRchat_api_agent
- https://github-wiki-see.page/m/vrchat-community/osc/wiki/Avatar-Parameters
- https://commercialtoolry.com/industrial/ai-anime-avatar-creators-vs-vrchat-avatars-which-offers-smoother-real-time-expression-syncing

---

## Search methodology (evidence of the negative result)

Queries run via `web_search` / `advanced_search` (engine auto-fell back from `anysearch`, which
was returning HTTP 402, to exa/tavily/keenable; `searxng` and the key-gated engines were down):

- `VRChat embodied AI research paper`; `VRChat embodied AI agent navigation arxiv`;
  `VRChat robotics platform paper IEEE ACM`; `VRChat reinforcement learning agent trained in VRChat`;
  `VRChat AI agent LLM NPC research paper avatar`; `VRChat navigation agent paper 2026 embodied`;
  `VRChat embodied navigation robot paper arxiv 2026 OSC egocentric`;
  `VRChat NPC LLM agent paper 2025 2026`; `embodied agent social VR VRChat navigation paper 2025 2026`;
  `VRChat navigation assistance agent published research prior work`;
  `VRChat egocentric vision robot agent research project`;
  `social VR robot teleoperation VRChat HRI study`; `VRChat robot teleoperation avatar control robot physical`;
  `VRChat as simulation environment research dataset`; `VRChat dataset collected research avatars interactions`;
  `arxiv VRChat research papers social VR list`; `"VRChat" arxiv 2026 agent`;
  `"Navigation Pixie" VRChat metaverse agent platform`;
  `"on-demand navigation" metaverse agent ISMAR 2025 cluster paper abstract`;
  `Has anyone published an embodied AI navigation agent inside VRChat?` (anysearch);
  `Habitat 3.0 paper arxiv year`; `AI2-THOR ProcTHOR paper venue`; `BEHAVIOR-1K benchmark paper`;
  `MineDojo Voyager open-ended embodied agent Minecraft`; `iGibson SAPIEN ThreeDWorld simulation platform paper comparison`;
  `SIMA DeepMind Scalable Instructable Multiworld Agent paper venue 2024`; `SIMA 2 Gemini DeepMind agent`;
  `SIMA DeepMind open source code release weights not released`; `SIMA 2 open source code released github DeepMind`;
  `SIMA DeepMind nine games No Man's Sky Valheim 600 language instructions`;
  `SIMA 2 games list Genie 3 world model training`; `SIMA 2 Gemini robotics transfer real world`;
  `Grand Theft Auto V as a research platform paper`; `Project Malmo Microsoft research platform paper`;
  `Minecraft navigation agent from pixels dataset`; `learning to navigate video games from pixels dataset CSGO Doom`;
  `VRChat OSC world position avatar not exposed limitation`;
  `VRChat OSC avatar position tracking exposed`; `VRChat world position coordinate get out of VRChat OSC impossible`;
  `VRChat does not expose avatar world position OSC workaround`;
  `vrchat-community osc issue 43 expose player position avatar parameters`;
  `Victory OSC trackers position head hands values`; `VRChat OSC input endpoint /input/ docs`;
  `VRChat OSC camera endpoints 2025.3.3 position rotation read write`;
  `VRChat OSC camera dolly world position attach avatar telemetry`;
  `VRChat built-in animator parameters VelocityX Grounded Upright Seated OSC output`;
  `VRChat VelocityX local space caveat playspace movement m/s`;
  `VRChat OSC library open source python oscquery`; `VRChat OSC third party tool window capture computer vision`;
  `VRChat world position tracker SteamVR OpenVR overlay get player position`;
  `SteamVR HMD pose world position offset playspace Unity`;
  `CLIP anime images domain gap performance stylized rendering`;
  `anime face detection lbpcascade_animeface dataset`;
  `AniDet3 dataset anime person detection roboflow YOLOv8`;
  `CSIP contrastive anime style image pre-training CLIP paper`;
  `person re-identification virtual world game characters dataset`;
  `person re-identification synthetic 3D characters generalizable ACM MM 2020`;
  `CLIP game rendering vs real photos evaluation`; `stylized rendering object detection benchmark sim-to-real gap game`;
  `vision model performance virtual avatar screenshots evaluation`;
  `VRChat avatar rendering anime style detector benchmark study`;
  `NavAI application-agnostic LLM framework navigation virtual reality ...`

`platform_search` was run on **github** (`VRChat embodied AI agent navigation LLM` → 0 results;
`vrchat osc`, `vrchat embodied agent`, `vrchat ai agent`, `vrchat navigation agent movement OSC`,
`vrchat ai companion vision OSC autonomous`), on **hn** (`VRChat AI agent LLM embodied bot`,
`VRChat OSC automation agent` → 0 results), on **reddit** (API HTTP 404 — unavailable), and on
**wikipedia** (`VRChat platform OSC Open Sound Control` → 0 results).

**Conclusion of the negative search:** across all of these, **not one result describes a
peer-reviewed embodied navigation system operating inside VRChat from egocentric pixels without
world-coordinate access.**
