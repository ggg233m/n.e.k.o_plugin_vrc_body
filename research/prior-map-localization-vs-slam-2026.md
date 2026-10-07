# Prior-map localization vs. incremental SLAM drift correction
### A research brief for a small mobile robot / embodied agent with a head-mounted stereo camera in an indoor Unity-rendered environment

**Question under evaluation.** If a complete 3D map of the environment already exists (a watertight triangle mesh with colliders, or a point cloud derived from it), is *localizing against that map* a sound replacement for *incremental SLAM + loop closure* as a way to eliminate drift?

---

## Short answer (3 lines)

1. **Yes for observable directions, no for degenerate ones.** Registering each frame against a known map replaces an unbounded random-walk error with a *bounded, map-referenced* error — but only in the degrees of freedom the current geometry actually constrains. In a symmetric corridor the along-corridor translation stays effectively unconstrained and can still wander.
2. **The map does not remove error, it re-bases it.** Residual error becomes a function of map accuracy, registration accuracy, initial guess and observability — typically **1–5 cm / 0.2–1°** indoors with rich geometry, degrading to **decimetres and unbounded along-axis error** in featureless or symmetric spaces.
3. **The real cost is the *initial* global localization and the robustness engineering** (multi-hypothesis MCL, place recognition + geometric verification, degeneracy detection, dynamic-object rejection, covariance/health monitoring), not the per-frame ICP. In a Unity-rendered world the sensor-noise term nearly vanishes (you can render the exact same mesh), so **initialization + observability become the whole problem**.

---

## 0. Scope, assumptions, method

**Assumed system.** Indoor room / room-scale to building-scale. Head-mounted stereo camera (ZED-class, RealSense D4xx-class, OAK-D, or Unity synthetic depth), 1–2 m typical range, 40k+ points/frame, ~2 cm depth noise. Prior map available as a triangle mesh (+ colliders) or a point cloud. Robot is small; locomotion may be wheeled, legged, or VR-style teleport.

**Method note / limitation.** In this session `web_fetch` was blocked (every hostname resolved to a non-public IP) and direct HTTPS from the shell failed TLS authentication, so **no paper was read end-to-end**. All findings below come from search-engine retrieval (Exa / Keenable / Tavily / Bing / DDG) — abstracts, indexed full-text fragments, and documentation pages. Where a number comes from an abstract or an indexed snippet rather than a verified table, it is marked `[snippet]`. Claims I could not corroborate at all are marked **UNVERIFIED**. Section 8 collects the confidence notes.

---

## 1. Terminology and the standard architecture

### 1.1 The two families, stated precisely

| | Incremental SLAM + loop closure | Prior-map localization ("map-based localization") |
|---|---|---|
| What is estimated | Map **and** trajectory jointly | Pose only; map is frozen input |
| Error growth | Front-end drifts; **bounded only where loop closures / absolute constraints exist**; VINS-style systems "inherently accumulate unbounded drift over time" while SLAM achieves bounded error through loop closure ([Multi-cam Multi-map VIL, arXiv 2412.04287](https://arxiv.org/html/2412.04287)) | Every frame is measured against an absolute reference; error does not integrate over time in constrained DOF |
| Failure mode | Drift, wrong loop closure, scale drift | Wrong initial pose, unobservable DOF, map staleness |
| Cost | Loop detection machinery | Map acquisition + maintenance + global localization |

The clean statement of the trade is in the OpenLiDARMap paper: *"Map-based localization offers an alternative by using pre-existing maps to constrain localization and eliminate drift… However, these algorithms rely on the availability of maps, which limits their applicability"* ([OpenLiDARMap, arXiv 2501.11111](https://arxiv.org/html/2501.11111v1), [DOI](https://doi.org/10.5220/0013405400003941)) `[snippet]`.

### 1.2 Vocabulary you should use (and the standard architecture each implies)

- **Prior-map / map-based localization** — pose estimation with the map given. Umbrella term.
- **Global (re)localization** — pose from **no** initial guess, over the whole map. Hard problem; usually *place recognition → candidate retrieval → geometric verification (ICP/NDT/PnP) → hypothesis scoring*. See [InstaLoc (RSS 2023, arXiv 2305.09552)](https://arxiv.org/abs/2305.09552), [G-PROBE / PROBE-X (2026, arXiv 2607.06782)](https://arxiv.org/html/2607.06782v2).
- **Kidnapped-robot problem** — the robot is moved without being told; the localizer must detect the inconsistency and re-globalize. Solved canonically by particle filters with random particle injection ([Thrun et al., *Robust Monte Carlo Localization for Mobile Robots*](http://robots.stanford.edu/papers/thrun.robust-mcl.pdf)).
- **MCL / AMCL** — Monte Carlo Localization; AMCL = adaptive (KLD-sampling) MCL. In ROS 2 it is *"a probabilistic localization module which estimates the Pose of a robot in a **given known map** using a 2D laser scanner"* ([nav2_amcl docs](https://docs.ros.org/en/jazzy/p/nav2_amcl/), [index.ros.org](https://index.ros.org/p/nav2_amcl/)). Note: this reference implementation is **2D + 2D occupancy grid**; 3D/6-DoF variants exist but are not the ROS default.
- **Scan-to-map vs scan-to-scan** — matching the current scan directly to the (frozen) map, vs. to the previous scan. Prior-map localization is scan-to-map.
- **Point-to-point / point-to-plane / plane-to-plane ICP** — ICP ([Besl & McKay 1992; Chen & Medioni 1991]) in its three correspondence regimes; point-to-plane converges faster and has a wider basin than point-to-point; **Generalized-ICP** unifies both as plane-to-plane ([Segal et al., RSS 2009](https://www.roboticsproceedings.org/rss05/p21.pdf)). Modern robust variants: VGICP, Nano-GICP, small_gicp.
- **NDT (Normal Distributions Transform)** — represents the map as piecewise Gaussians rather than points, giving a smooth, voxel-resolution cost ([Biber & Straßer, IROS 2003](https://ieeexplore.ieee.org/document/1249285); [Wikipedia summary](https://en.wikipedia.org/wiki/Normal_distributions_transform)). This is the workhorse of HD-map localization in Autoware.
- **Degeneracy / observability** — directions in state space that the current geometry does not constrain (classic: translation along a corridor axis). Canonical treatment: [Zhang, Kaess & Singh, *On Degeneracy of Optimization-based State Estimation Problems*, ICRA 2016](https://ieeexplore.ieee.org/document/7487211) — it separates the degenerate directions and "only partially solves the problem in well-conditioned directions" (solution remapping / D-factor).
- **Covariance estimation / health monitoring** — propagating registration uncertainty into the filter. Foundational: [Censi, *An Accurate Closed-Form Estimate of ICP's Covariance*, ICRA 2007](https://doi.org/10.1109/robot.2007.363961); 3D extension: [CELLO-3D, ICRA 2019](https://doi.org/10.1109/icra.2019.8793516) and [*A New Approach to 3D ICP Covariance Estimation*, arXiv 1909.05722](https://arxiv.org/pdf/1909.05722).
- **Place recognition** — appearance/geometry descriptor retrieval used for loop closure *and* global localization: [Scan Context (Kim & Kim, IROS 2018)](https://dl.acm.org/doi/abs/10.1109/iros.2018.8593953), [Scan Context++ (arXiv 2109.13494)](https://arxiv.org/pdf/2109.13494), [OverlapNet (RA-L 2021)](https://link.springer.com/content/pdf/10.1007/s10514-021-09999-0.pdf), NetVLAD for the visual case.
- **ESDF / TSDF / SDF registration** — registering against a *continuous* distance field rather than discrete points. See §4.
- **Analysis-by-synthesis / render-and-compare** — render synthetic depth (or range image) from the mesh and align it to the observed depth. This is how you register against a **triangle mesh** without converting it to points. See §4.

**Correction on "PoseCNN".** PoseCNN is **not** a map-localization method. It is *"a Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes"* ([Xiang et al., arXiv 1711.00199](https://arxiv.org/pdf/1711.00199)) — estimating the 6-DoF pose of a **known object instance** from an RGB image. It belongs to a different problem family (object pose), and its "pose" is object-relative, not map-relative. If you meant a *learned absolute pose regressor* (APR) for scenes, the relevant families are APR / Map-Relative Pose Regression ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Map-Relative_Pose_Regression_for_Visual_Re-Localization_CVPR_2024_paper.pdf)) and [VIO-APR (arXiv 2308.05394)](https://arxiv.org/html/2308.05394v2).

### 1.3 Canonical references and packages

**Books / foundational**
- Thrun, Burgard & Fox, *Probabilistic Robotics* (MIT Press 2005) — Ch. 7 (MCL), Ch. 8 (grid localization), Ch. 10–11 (SLAM, loop closure). The standard textbook.
- Barfoot, *State Estimation for Robotics* (Cambridge; free PDF from the author) — the rigorous treatment of observability, FEJ, and information matrices.
- [Thrun et al., Robust MCL (AI Journal 2001, PDF)](http://robots.stanford.edu/papers/thrun.robust-mcl.pdf) — global localization + kidnapped robot.

**Registration primitives**
| Method | Reference |
|---|---|
| ICP | Besl & McKay, TPAMI 1992; Chen & Medioni 1991 (point-to-plane) |
| Generalized-ICP | [Segal, Haehnel & Thrun, RSS 2009](https://www.roboticsproceedings.org/rss05/p21.pdf) |
| NDT / 3D-NDT | [Biber & Straßer, IROS 2003](https://ieeexplore.ieee.org/document/1249285); Magnusson, *Scan registration for autonomous mining vehicles using 3D-NDT* |
| ICP covariance | [Censi, ICRA 2007](https://doi.org/10.1109/robot.2007.363961); [CELLO-3D, ICRA 2019](https://doi.org/10.1109/icra.2019.8793516) |
| Degeneracy | [Zhang, Kaess & Singh, ICRA 2016](https://ieeexplore.ieee.org/document/7487211) |
| Open3D (point-to-point / point-to-plane / colored ICP, FPFH+RANSAC global registration, `RaycastingScene` for mesh queries) | [Open3D ICP tutorial](https://www.open3d.org/docs/latest/tutorial/t_pipelines/t_icp_registration.html), [global registration tutorial](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html), Zhou, Park & Koltun, arXiv 1801.09847 |
| libpointmatcher (2D/3D ICP framework: filters, outlier rejectors, error minimizers) | [norlab-ulaval/libpointmatcher](https://github.com/norlab-ulaval/libpointmatcher), [docs](https://libpointmatcher.readthedocs.io/en/latest/) |
| PCL | pointclouds.org (`pcl::IterativeClosestPoint`, `pcl::NormalDistributionsTransform`) |

**ROS / production localization stacks**
| Stack | Role | Link |
|---|---|---|
| `nav2_amcl` | 2D MCL against a known 2D map | [docs.ros.org](https://docs.ros.org/en/jazzy/p/nav2_amcl/) |
| Cartographer **pure localization** | freezes the map trajectory and constrains the live trajectory against it; docs explicitly discuss the "large number of inter constraints between the frozen trajectory and the current one" | [Cartographer tuning docs](https://google-cartographer-ros.readthedocs.io/en/latest/tuning.html) |
| `hdl_graph_slam` | 3D graph SLAM, NDT scan matching + loop detection (mapping side) | [koide3/hdl_graph_slam](https://github.com/koide3/hdl_graph_slam) |
| `hdl_localization` | **real-time 3D localization against a prebuilt map**, UKF + NDT | [koide3/hdl_localization](https://github.com/koide3/hdl_localization) |
| Autoware `ndt_scan_matcher` | NDT scan matching against a prebuilt pointcloud map, plus *Monte Carlo initial-pose estimation* as a ROS service | [Autoware Core docs](https://autowarefoundation.github.io/autoware_core/main/localization/autoware_ndt_scan_matcher/) |
| FAST-LIO + prior map variants | LIO front end forced to a fixed map | [FAST_LIO_LOCALIZATION](https://github.com/iral-ntua/fast_lio_localization), [FAST-LOCALIZATION](https://github.com/YWL0720/FAST-LOCALIZATION), [Fast-LIO2-Localization](https://github.com/PolarisXQ/Fast-LIO2-Localization) |
| KISS-ICP | minimal, parameter-free point-to-point ICP odometry ([Vizzo et al., RA-L 2023, arXiv 2209.15397](https://arxiv.org/html/2209.15397v2)) | [PRBonn/kiss-icp](https://github.com/PRBonn/kiss-icp) |
| PIN-SLAM | implicit neural map SLAM (T-RO 2024) | [PRBonn/PIN_SLAM](https://github.com/PRBonn/PIN_SLAM) |
| VoxelMap | adaptive voxel mapping (RA-L 2022) | [hku-mars/VoxelMap](https://github.com/hku-mars/VoxelMap) |
| `fast_gicp` / `small_gicp` | fast GICP implementations | repo URL **UNVERIFIED** in this session |
| `range-mcl` | MCL that localizes LiDAR against a **mesh** map | [PRBonn/range-mcl](https://github.com/PRBonn/range-mcl) |

### 1.4 How prior-map localization compares with loop closure for *eliminating drift*

- **Loop closure** makes error *bounded* by adding relative constraints when a place is revisited; it is opportunistic (needs revisits) and can fail silently via false positives ([Gesar write-up on loop-closure failure modes](https://gesarrobots.com/blog/post/2349329/why-loop-closure-fails-in-slam-and-how-gesar-inc-fixes-it-every-time) — vendor blog, treat as context not evidence).
- **Prior-map localization** adds an *absolute, per-frame* constraint. This is strictly stronger **when the map is correct and the geometry is observable**, and strictly weaker when the robot is in an unobservable region or the map disagrees with reality.
- **They are not exclusive.** The common production pattern is: LIO/VIO front end + scan-to-map correction + optional loop closure + optional global relocalization. `hdl_localization` (UKF over NDT scan matching), Autoware (NDT + EKF + stop filter), and Cartographer pure localization (frozen trajectory as a constraint target) are all this shape.

---

## 2. Expected accuracy: registering a dense stereo point cloud against a mesh, indoors

### 2.1 The sensor term

Stereo depth error is dominated by triangulation geometry:

    σ_z ≈ (z² / (f · B)) · σ_d

i.e. **depth error grows quadratically with range** and linearly with disparity error; `f·B` is the focal-length × baseline product. This is textbook stereo material (Scharstein & Szeliski, *A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms*, IJCV 2002) `[textbook, not re-verified from a fetched PDF]`.

Measured values that agree with the models:

| Sensor / condition | Reported error | Source |
|---|---|---|
| Intel RealSense D435 (B = 50 mm) | **< 2 % of depth at 2 m** (≈ 40 mm @ 2 m); 2–5 mm at 1 m; RealSense staff report error growing **linearly** with distance with drift onset near **3 m**; optimal range 0.3–5 m | [D400 datasheet](https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf), [librealsense #7806](https://github.com/realsenseai/librealsense/issues/7806), [#8879](https://github.com/IntelRealSense/librealsense/issues/8879) `[snippet]` |
| Intel RealSense D455 | **< 2 % at 4 m** | [RealSense compare page](https://www.realsenseai.com/compare-depth-cameras/) |
| D435 (table-top, close range) | **error < 1 cm in every tested table-top setting up to about 1 m** | [arXiv 2501.07421 summary via pith.science](https://pith.science/paper/2501.07421) `[snippet, second-hand — AI summary of the paper]` |
| StereoLabs ZED 2 | **< 3 cm Chamfer error at 4 m** — best overall in that comparison, but needs a CUDA GPU | same source `[snippet, second-hand]` |
| StereoLabs ZED 2i (B = 12 cm, 120° FOV), indoor office, 1→20 m | **< 1 % error up to 3 m**; usable depth to **18 m** (HD1080/HD2K), **14 m** (HD720), **7 m** (VGA); errors explicitly **heavy-tailed / non-Gaussian** | [Abdelsalam et al., *Robotics and Autonomous Systems* 2024](https://www.sciencedirect.com/science/article/pii/S0921889024001374), [Aalto full text](https://aaltodoc.aalto.fi/server/api/core/bitstreams/13a72a5e-3836-4cf3-9b61-5791e06b1002/content) |
| Four-camera empirical comparison (D435, D455, ZED 2, OAK-D Pro) | full study | [Rustler et al., arXiv 2501.07421](https://arxiv.org/html/2501.07421v2), [project page](https://lukasrustler.cz/rgbd-comparison/) |

So the user's assumed "~2 cm depth noise at 1–2 m" is realistic for a mid-range stereo rig (expect roughly **5–30 mm** in that band); at 2 m the *spec* allows closer to 4 cm for a 2 %-class sensor, and beyond 3–5 m the stereo cloud stops constraining pose at all. Two consequences worth internalising: **(a)** your perception noise is the *same order of magnitude as the registration residual you are trying to measure*, so it sets a floor on how well you can validate a localizer; **(b)** stereo depth errors are **heavy-tailed, not Gaussian** ([RAS 2024](https://www.sciencedirect.com/science/article/pii/S0921889024001374)), which is exactly why least-squares ICP needs robust kernels (§3.4). Range-image representations exist precisely to make this non-uniform sampling explicit — LiDAR/stereo scans are *not* uniformly sampled point sets ([*Revisiting LiDAR Registration and Reconstruction: A Range Image Perspective*, Dong, Ryu, Kaess & Park, arXiv 2112.02779](https://arxiv.org/abs/2112.02779)).

### 2.2 The registration term — what to expect

Reported figures, with their conditions:

- **Open3D's own documented ICP numbers** — the single most useful calibration point. On the Redwood indoor RGB-D pair (`cloud_bin_0` → `cloud_bin_1`, `threshold = 0.02 m`, *with* a supplied rough `trans_init`):

| Stage | fitness (inlier fraction) | inlier_rmse | correspondences |
|---|---|---|---|
| Initial alignment | **0.1747** | **11.77 mm** | 34 741 |
| Point-to-point ICP | **0.3724** | **7.76 mm** | 74 056 |
| Point-to-plane ICP | **0.6210** | **6.58 mm** | 123 471 |

  ([Open3D ICP tutorial](https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html); archived 0.7.0 values agree — `fitness = 0.620972`, `inlier_rmse = 0.006581`.) **Reading:** even after point-to-plane ICP on a clean, static, tripod-captured indoor scene, only **~62 % of points are inliers** and residual RMSE is **6.6 mm**. Point-to-plane nearly *doubles* the inlier fraction (0.37 → 0.62) versus point-to-point and cuts RMSE ~15 %. **Caveat:** Open3D changed the `fitness` denominator between versions (0.7.0 divides by target points, latest by source points) — do not compare `fitness` across versions ([0.7.0 docs](https://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html) vs [latest](https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html)).
- **Indoor/urban LiDAR scan-to-map localization** is generally quoted in the **few-centimetre** range: e.g. a real-time scan-to-map system on a lightweight pre-built occupancy HD map ([Remote Sensing 15(3):595, 2023](https://www.mdpi.com/2072-4292/15/3/595)) `[snippet]`; building/BIM-referenced indoor localization ([BIM-Loc, arXiv 2606.14237](https://arxiv.org/html/2606.14237v1)) works specifically in "feature-sparse indoor environments". A 2026 ISPRS contribution reports precision *increases* when an LIO output is constrained by a predefined global map ([ISPRS Archives XLIX-B1-2026, 161](https://isprs-archives.copernicus.org/articles/XLIX-B1-2026/161/2026/isprs-archives-XLIX-B1-2026-161-2026.pdf)) `[snippet]`.
- **Continuous-distance-field localization** reaches **0.06 m localization error** (and 0.22 % odometry error) over 750 km of automotive + handheld data ([2Fast-2Lamaa, arXiv 2410.05433](https://arxiv.org/pdf/2410.05433v3)) `[snippet]`.
- **Autonomous-driving map localization targets < 0.1 m at ≥ 95 % confidence** ([*Evaluating Localization Accuracy of Automated Driving Systems*, Sensors 21(17):5855, 2021](https://www.mdpi.com/1424-8220/21/17/5855)); semantic-HD-map localization is reported "on the order of a few centimeters" ([arXiv 1908.03274](https://ar5iv.labs.arxiv.org/html/1908.03274)).
- **Industrial docking** (a *local* high-precision regime, with fiducials/targets) reaches **sub-centimetre and sub-degree** accuracy and precision ([multi-stage AMR docking framework, Robotica 2024](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/7C4B7AB733A729E538F50B9CCDBAC3F3/S0263574724000602a.pdf/multistage_localization_framework_for_accurate_and_precise_docking_of_autonomous_mobile_robots_amrs.pdf)).
- **Comparison anchor:** KITTI-style LiDAR odometry ATE values are metres-to-decimetres, i.e. 1–2 orders of magnitude worse than scan-to-map localization against a good map — e.g. KISS-ICP's own KITTI table lists systems in the 1.14–2.21 m range on some sequences ([KISS-ICP, arXiv 2209.15397](https://arxiv.org/html/2209.15397v2)) `[snippet, table fragment without its header — read in context before quoting]`.

**Practical honest range for this system:** with rich indoor geometry, dense 40k-point frames, and a decent initial guess, expect **≈ 1–5 cm translation and ≈ 0.2–1° rotation per frame**; error is dominated by *how much of the frame is actually informative* and by the initial guess. In poorly constrained directions expect **decimetres and growing**, not a fixed bound. (Note that even the clean Open3D static-scene case left a **6.6 mm** residual — so treat 1 cm as an optimistic-best-case floor, not a typical value, for a moving head-mounted rig.)

### 2.3 What drives the numbers

| Driver | Effect | Evidence |
|---|---|---|
| **Geometric richness** (corners, wall junctions, furniture, ceiling/floor variation) | The single biggest factor; constrains all 6 DOF | [Zhang et al. 2016](https://ieeexplore.ieee.org/document/7487211); [BIM-Loc](https://arxiv.org/html/2606.14237v1) notes feature-sparse environments as the core difficulty |
| **Symmetry / repetition** (corridors, identical cubicles, aisles) | Creates *multi-modal* belief; ambiguity is a belief problem, not a solver problem | [Multi-component information-equalized filter for global localization "robust to kidnapping and symmetrical environments"](https://sciencedirect.com/science/article/pii/S0921889010000382) |
| **Initial guess / basin of attraction** | "ICP is subject to error due to wrong initialization that makes the algorithm converge to a local minimum out of the attraction basin of the true solution… In practice it often proves to be the dominant error." | [*A New Approach to 3D ICP Covariance Estimation*, arXiv 1909.05722](https://arxiv.org/pdf/1909.05722) |
| **Point-to-plane vs point-to-point** | Wider convergence basin, faster convergence | [Robust symmetric ICP, ISPRS 2022](https://skyearth.org/publication/papers/2022_rsicp.pdf) — point-to-plane metric "has a wider convergence basin and higher convergence speed" |
| **Occlusion / partial overlap** | Fewer inliers; fitness drops; covariance inflates | Open3D `fitness` semantics |
| **Dynamic objects** | Corrupt correspondences; robust kernels needed | §3.4 |
| **Sensor noise** | Enters as σ_z ∝ z², heavy-tailed | §2.1 |
| **Point density** | More points ≠ more constraint if they lie on the same plane; plane-to-plane methods reduce the point count while keeping the geometry | [Plane-based approach for indoor point clouds registration (ICPR 2020)](http://rainbow-doc.irisa.fr/pdf/2020_icpr_favre.pdf) |
| **Map–reality discrepancy** | Systematically biases the pose; the localizer will happily lock onto the *old* geometry | [BIM-Loc](https://arxiv.org/html/2606.14237v1) ("simultaneously estimating a BIM-aligned trajectory and detecting which BIM structures actually exist"); [Cartographer issue #1534 on removed objects](https://github.com/googlecartographer/cartographer/issues/1534) |

**The convergence basin, quantified (weakly).** Every source agrees ICP "requires sufficiently good initial parameters" and fails "when the rotation angle between the two point sets is large" ([PLOS ONE 2017](https://dx.plos.org/10.1371/journal.pone.0188039), [ISPRS J. 2019](https://sciencedirect.com/science/article/pii/S0924271619302965)). The most explicit quantification I could retrieve — *good initial guess = within a few degrees and a few centimetres; 30° rotation with no prior ⇒ trapped in a local minimum with a plausible but wrong result* — comes from an **informal vendor blog** ([smartbotparts](https://smartbotparts.com/articles/point-cloud-registration-algorithms-compared.html), **LOW CONFIDENCE**). **No peer-reviewed, quantified basin figure in degrees for indoor point clouds was retrievable in this session** — treat "≈30°" as folklore, not a spec. What *is* well established is the escape hatch and the standard recipe: **coarse global registration → multi-scale local refinement**, i.e. FPFH + RANSAC (arbitrary initial poses, coarser and **non-deterministic even with a fixed seed** — [Open3D #7475](https://github.com/isl-org/Open3D/issues/7475)), TEASER/TEASER++ (certifiable, heavy outliers — [arXiv 2001.07715](https://ar5iv.labs.arxiv.org/html/2001.07715)), or coarse-to-fine multi-scale ICP ([EM-ICP](http://www-sop.inria.fr/asclepios/Publications/Granger/eccv-2002.pdf), [coarse-to-fine GICP](https://ieeexplore.ieee.org/document/9007699)). Variant benchmarks on real indoor/laser data: [iralabdisco/point_clouds_registration_benchmark](https://github.com/iralabdisco/point_clouds_registration_benchmark) / [RAS 2021](https://doi.org/10.1016/j.robot.2021.103734); an eight-system indoor+outdoor SLAM comparison is in [arXiv 2208.02063](https://ar5iv.labs.arxiv.org/html/2208.02063).

### 2.4 Special case: the world is *rendered from the same mesh* (Unity)

This materially changes the error budget, and it should be stated explicitly because it is the actual situation:

- If depth is rendered from the prior mesh itself, **the sensor-noise term is ~0** by construction, and the "map" and the "measurement" are two views of the *same* geometry. The classical stereo noise model in §2.1 only applies if stereo noise is deliberately simulated.
- What remains is **initialization, observability, estimator bias, and any deliberate model mismatch** (colliders/visual mesh differences, mesh simplification/LOD, floating-point precision, LOD-swapped geometry, hidden faces, back-face culling, materials/transparency).
- Also note the trap: in a Unity-rendered world you can read the agent's true pose from the engine. A prior-map localizer evaluated in that world must be scored on *how little privileged information it uses*, otherwise the comparison against SLAM is meaningless. `[Reasoning, no citation.]`
- One more consequence: mesh-derived depth is **exact and full-FOV**, so a render-and-compare localizer gets a much cleaner likelihood than a real stereo rig would. Accuracy results measured in this world will be **optimistic relative to hardware deployment**.

---

## 3. Failure modes and standard mitigations

### 3.1 The mechanism of degeneracy — precisely

Gauss–Newton point-to-plane ICP minimises `Σ (nᵢᵀ(Rpᵢ + t − qᵢ))²`; the normal-equation matrix `H = JᵀJ` **is** the (inverse) information matrix. The key structural fact:

> **Translational observability comes only from the span of the correspondence surface normals.**

Consequences by geometry:

| Environment | Available normal directions | Unobservable DOF | Symptom |
|---|---|---|---|
| **Long straight corridor / tunnel** | ±y walls, ±z floor & ceiling → spans y, z, roll, pitch | **translation along the corridor axis (+x)** and often yaw | `λ_min(H) → 0` for +x; the update is driven by noise and asymmetry; the estimate **slides along the corridor** |
| **Flat, empty room** | mostly ±z floor/ceiling | yaw (and weakly x, y) | heading wanders; position weakly constrained |
| **Large open hall** | distant walls only → weak, low-weight constraints | all horizontal directions poorly conditioned | large covariance in-plane; confident but loose |

Zhang, Kaess & Singh give the formal statement: the shift along a unit direction `c` is

    δx_c = cᵀ (AᵀA + ccᵀ)⁻¹ c · δd

which is a function of the **geometry and the residual, not of the measurement `b`** — i.e. **the degenerate direction is not constrained by data at all**. Their online method detects this, separates the degenerate directions, and "only partially solves the problem in well-conditioned directions" — the **solution-remapping / D-factor** technique. [Zhang, Kaess & Singh, ICRA 2016, pp. 809–816](https://ieeexplore.ieee.org/document/7487211), [author PDF](https://frc.ri.cmu.edu/~zhangji/publications/ICRA_2016.pdf)

**A trap that specifically bites a stereo/GICP stack.** With point-to-distribution registration plus covariance regularisation, every correspondence contributes an *isotropic information floor* to the translational block of the Hessian. The Hessian therefore looks well-conditioned **while the pose is still wrong** — degeneracy is structurally masked. [LF-GICP, arXiv 2608.19522, Aug 2026](https://arxiv.org/html/2608.19522) — *brand-new, single-author: treat its claims as preliminary.* This is the single most important warning in this section: **"my Hessian is well-conditioned" is not evidence that the pose is right.**

### 3.2 Detection — how standard systems find out

| Method | What it measures | Reference |
|---|---|---|
| Min eigenvalue of `H` ("degeneracy factor", D-factor) | Weakest constrained direction | [Zhang et al. 2016](https://ieeexplore.ieee.org/document/7487211) |
| Condition number of `H` | Overall conditioning | classical |
| **Per-direction eigen-decomposition + localizability categories** (well-conditioned / degenerate / none), replacing binary thresholds | Directional observability *and* correspondence alignment strength | [X-ICP, IEEE T-RO 2023, arXiv 2211.16335](https://arxiv.org/pdf/2211.16335), [DOI](https://dl.acm.org/doi/abs/10.1109/TRO.2023.3335691) |
| Probabilistic point-to-plane degeneracy detection | Statistical test instead of a threshold | [arXiv 2410.10784](https://arxiv.org/html/2410.10784) |
| D-optimality | Design-based observability measure | CompSLAM (reported via [LF-GICP](https://arxiv.org/html/2608.19522)) |
| Critique: D-factor detects *overall*, not per-correspondence, degeneracy | — | [LP-ICP, arXiv 2501.02580](https://arxiv.org/html/2501.02580) |
| ICP covariance (closed form, from the Hessian) | Per-DOF uncertainty used to **gate** updates | [Censi, ICRA 2007](https://doi.org/10.1109/robot.2007.363961), [arXiv 1410.7632](https://arxiv.org/pdf/1410.7632), [CELLO-3D, ICRA 2019](https://doi.org/10.1109/icra.2019.8793516) |

Related degeneracy-robust systems: [GenZ-ICP, RA-L 2025](https://arxiv.org/html/2411.06766v1) (adaptive point-to-plane ↔ point-to-distribution weighting), [AdaLIO](https://arxiv.org/html/2304.12577v1) (LOAM parameters tuned for open space diverge in corridors and spiral stairs), [D²-LIO](https://arxiv.org/html/2508.14355v1), [LODESTAR](https://arxiv.org/html/2511.09142v1) (adaptive Schmidt–Kalman), DAMM-LOAM, VINA-SLAM.

### 3.3 Mitigations — and what each still does **not** fix

| Mitigation | What it fixes | What it does **not** fix |
|---|---|---|
| **Solution remapping / freezing the degenerate directions** | Stops the estimator from inventing motion along an unobservable axis | Converts sliding into **prior-driven drift** — the DOF is still unobserved; you have just delegated it to odometry/IMU |
| **Multi-hypothesis MCL / particle filter** | Genuinely multimodal belief ("either end of a symmetric hallway") — the filter can **defer commitment**; a Gaussian cannot | Particle depletion; large-area coverage needs very large sample counts; convergence time scales with injection rate ([Thrun, Robust MCL](http://robots.stanford.edu/papers/thrun.robust-mcl.pdf), [multi-hypothesis PF](https://ieeexplore.ieee.org/document/5355068), [symmetrical-environment global localization](https://sciencedirect.com/science/article/pii/S0921889010000382), [Portable Multi-Hypothesis MCL, arXiv 2209.07586](https://ar5iv.labs.arxiv.org/html/2209.07586)) |
| **IMU / wheel odometry coupling** to carry the unobservable DOF | The dominant practical fix in the LIO line (FAST-LIO, Point-LIO, AdaLIO, DAMS-LIO) | Does not remove drift — doubly-integrated biased IMU translation is itself a random walk |
| **Place recognition + geometric verification** (Scan Context/++, DBoW2, NetVLAD, SeqSLAM, MinkLoc3D, [SegMatch](https://arxiv.org/pdf/1609.07720.pdf)) | The only way out of the **local** convergence basin; global initialisation | False positives: [Scan Context++ (T-RO 2021)](https://gisbi-kim.github.io/uploads/gkim-2021-tro.pdf) argues standard precision–recall curves **under-report** loop-closure risk |
| **Absolute references** (AprilTag/ArUco, QR floor codes, reflectors) | Makes the pose genuinely observable and effectively drift-free locally | Requires infrastructure; one secondary aggregator claims ±1–4 cm for AprilTag under favourable conditions (**LOW CONFIDENCE**) |
| **Covariance / health monitoring + update gating** | Prevents a bad registration from being trusted; enables fallback | Needs a correct noise model; Censi's own formula does **not** model wrong convergence |

**The honest field study you should cite when someone claims a single winner.** [*Informed, Constrained, Aligned* (arXiv 2408.11809)](https://arxiv.org/html/2408.11809v2), code [leggedrobotics/perfectlyconstrained](https://github.com/leggedrobotics/perfectlyconstrained), compares **active vs passive** degeneracy mitigation and adds sub-space Tikhonov regularisation, truncated SVD, and inequality-constrained optimisation. Its takeaway: **no single method wins; mitigation must be tuned per environment.** That is the correct expectation to set.

### 3.4 Dynamic humans and moving objects

They are **not modelled by design** — they enter the correspondence set and corrupt the solve. Standard mitigations, in increasing order of sophistication:

- **Dynamic point removal before matching**: [Removert, IROS 2020](https://gisbi-kim.github.io/uploads/gkim-2020-iros.pdf), [ERASOR, RA-L 2021](https://arxiv.org/pdf/2103.04316v1.pdf), [ERASOR2, RSS 2023](https://www.ipb.uni-bonn.de/pdfs/lim2023rss.pdf), [Dynablox](https://github.com/ethz-asl/dynablox) (now inside NVIDIA nvblox), [DynamicMap Benchmark, ITSC 2023](https://arxiv.org/html/2307.07260) / [DUFOMap](https://kth-rpl.github.io/DynamicMap_Benchmark/).
- **Honest caveat:** removal is a precision/recall trade-off, and ERASOR2's evaluation explicitly reports **both** "remaining dynamic points" and "rejected static points" ([Fig. 1](https://www.ipb.uni-bonn.de/pdfs/lim2023rss.pdf)). **Over-aggressive removal punches holes in the map and creates *new* degeneracy** — a real risk for small robots with few points.
- **Robust kernels instead of hard removal**: Cauchy / Geman-McClure M-estimators, correntropy-based ICP (CoBigICP, IROS 2020), RANSAC, adaptive truncated least squares ([DynaVINS++, arXiv 2410.15373](https://arxiv.org/html/2410.15373v1)).
- **Measured damage** (visual odometry proxy, since these are the numbers I could retrieve): ORB-SLAM3 reaches **ATE = 0.259 m** on TUM RGB-D `fr3_w_xyz` — a *walking-people* sequence; dynamic-aware systems report **53 % average and up to 93.6 % ATE improvement** over it ([Springer 2026](https://link.springer.com/article/10.1007/s44163-026-01578-5)). For scale, a good indoor neural SLAM system reports **ATE ≈ 5.5 cm** on clean sequences ([GlORIE-SLAM, arXiv 2403.19549](https://arxiv.org/html/2403.19549v1)) — i.e. **moving people can be roughly 5× the clean-scene error.**

### 3.5 Map change, lighting, and material

- A fixed prior map imports **scan-to-BIM / as-planned vs as-built deviation**: furniture and clutter absent from the model, quasi-static change, renovation ([TUM ECPPM 2022](https://publications.cms.bgu.tum.de/2022_ECPPM_Vega.pdf)). Recent work is explicitly *discrepancy-aware*: [BIM-Loc (arXiv 2606.14237 / IJRR 2026)](https://ar5iv.labs.arxiv.org/html/2606.14237) simultaneously estimates the trajectory and **detects which BIM structures actually exist**.
- **Long-term / appearance-change literature**: [lifelong SLAM survey, JFR](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.22170), [OpenLORIS-Scene](https://arxiv.org/html/1911.05603v2), [4Seasons](https://arxiv.org/html/2009.06364v3), [UTIAS multi-season dataset (159 549 stereo pairs)](https://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/), [summary maps, JFR](https://onlinelibrary.wiley.com/doi/10.1002/rob.21595), [map management](https://ar5iv.labs.arxiv.org/html/1808.02658).
- **Stereo/IR-specific failures — the ones that actually bite a head-mounted stereo rig:**
  - **Mirrors** return the *virtual image behind the mirror* → geometrically plausible but **wrong depth** ([Mirror3D, ICCV 2021](https://ar5iv.labs.arxiv.org/html/2106.06629)).
  - **Glass / transparent surfaces**: IR passes through, so depth lands on whatever is behind the pane ([librealsense #1680](https://github.com/IntelRealSense/librealsense/issues/1680)); LiDAR has an analogous multi-return ambiguity ([arXiv 2406.10494](https://arxiv.org/pdf/2406.10494)). See also [GlassFormer, arXiv 2609.36844](https://arxiv.org/pdf/2609.36844) ("transparent surfaces… remain a persistent failure case") and [Reliability-Guided Depth Fusion for Glare-Resilient Costmaps, arXiv 2604.12753](https://arxiv.org/abs/2604.12753) (specular glare "produces holes and spikes that accumulate as persistent phantom obstacles").
  - **Black / tinted / absorptive surfaces** absorb IR → depth holes ([librealsense #13694](https://github.com/IntelRealSense/librealsense/issues/13694)).
  - **Textureless flat-shaded walls break *passive* stereo** (no photometric match); active stereo survives. **In a Unity render, a flat-shaded wall is exactly this pathological case.**
  - Specular surfaces generally need polarisation or multi-modal sensing ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611810/)).

### 3.6 Kidnapped robot / global recovery

- MCL handles the kidnapped-robot problem via **random particle injection / augmented MCL**; recovery time and success rate scale with particle count and injection rate. **No quantitative recovery-rate or convergence-time figures were retrievable in this session — UNVERIFIED** ([Thrun, Robust MCL](http://robots.stanford.edu/papers/thrun.robust-mcl.pdf), [augmented MCL explainer](http://cs.gettysburg.edu/~tneller/cs371/17sp/mcl/2017/dmy/index.html)).
- Global 3D localization against a **mesh** is directly studied: [Transformer-based MCL in Construction Meshes, arXiv 2609.31357](https://arxiv.org/abs/2609.31357) and earlier single-shot global localization in meshes ([ISARC 2021](https://www.iaarc.org/publications/2021_proceedings_of_the_38th_isarc/global_localization_in_meshes.html)) — the closest published analogue to the proposed setup.
- Place-recognition false-positive rates: **UNVERIFIED** in this session.

### 3.7 Summary of failure modes for *this* system

Ranked by expected impact in a Unity-rendered indoor world:

1. **Geometric degeneracy (corridors, blank walls, empty rooms)** — fully reproduced by a renderer, and the dominant risk. Mitigation is *architectural*, not parametric.
2. **Wrong global initialisation / perceptual aliasing** — the failure is a confident, stable, wrong pose.
3. **Map–reality mismatch** — only if the renderer's world and the prior mesh diverge (LOD swaps, procedural variation, dynamic props).
4. **Dynamic actors** — only if the scene contains them; if so, they are outliers to the solve and need separate treatment.
5. **Passive-stereo texturing** — flat-shaded geometry; irrelevant if depth is rendered, relevant if stereo is simulated.
6. **Sensor noise** — near-zero if depth is rendered from the mesh; otherwise §2.1 applies.

---

## 4. Registering against a mesh (not a point cloud), and absolute metric scale

### 4.1 Is point-to-mesh / SDF registration standard? Yes — with a distinction

There are **two standard ways** to use a triangle mesh as the map, and both are established:

**(a) Convert/sample the mesh into the registration primitive.** Sample the mesh surface into a point cloud (or into surfels / voxel Gaussians) and use ordinary ICP/NDT/GICP. libpointmatcher, PCL, VoxelMap, hdl_graph_slam and Open3D all work this way. Open3D additionally gives you exact mesh queries via `RaycastingScene` (`raycast`, `compute_signed_distance`, `compute_distance`) so you can compute point-to-mesh distances and normals directly instead of discretizing — that is the practical "point-to-mesh ICP" in a modern library.

**(b) Register against a continuous field / a direct mesh query** — an SDF/ESDF/TSDF, a hardware-accelerated ray cast against the triangles, or **render synthetic depth/range images from the mesh and align them** (analysis-by-synthesis / render-and-compare). This is the important branch here because it is *exactly* the "head-mounted camera + mesh map" architecture:

- **MICP-L — the direct answer for point-to-mesh.** *Monte Carlo ICP for LiDAR* (IROS 2024) registers arbitrary range sensors **directly to a triangle mesh map** using **hardware-accelerated ray casting** (RTX), motivated by mesh maps being "a versatile 3D environment representation… in challenging indoor and outdoor environments." Code: [uos/rmcl](https://github.com/uos/rmcl). ([arXiv 2210.13904](https://arxiv.org/html/2210.13904v4), [IROS 2024 DOI](https://doi.org/10.1109/IROS58592.2024.10802360)) — **I could not retrieve MICP-L's accuracy figures; UNVERIFIED.**
- **range-mcl** localizes against a **triangular mesh map** by rendering synthetic range images from the mesh and matching the live scan: [Chen, Vizzo, Läbe, Behley & Stachniss, *Range Image-based LiDAR Localization for Autonomous Vehicles*, ICRA 2021, arXiv 2105.12121](https://arxiv.org/abs/2105.12121v1), [author PDF](https://www.ipb.uni-bonn.de/pdfs/chen2021icra.pdf), code [PRBonn/range-mcl](https://github.com/PRBonn/range-mcl). **Accuracy figures not retrieved — UNVERIFIED.**
- **KMCL — Kinect Monte Carlo Localization** renders simulated depth from an a priori 3D model and uses it as the particle likelihood: [Fallon, Johannsson & Leonard, ICRA 2012](https://ieeexplore.ieee.org/document/6224951). Foundational; an RGB-D particle filter against a mesh model.
- **PUMA / Poisson Surface Reconstruction for LiDAR Odometry and Mapping** (ICRA 2021) treats the mesh itself as the map representation ([PRBonn/puma](https://github.com/PRBonn/puma)).
- **SDF-based localization**: [*Freetures: Localization in Signed Distance Function Maps*, arXiv 2010.09378](https://ar5iv.labs.arxiv.org/html/2010.09378) — explicitly framed as "reducing estimation drift"; [SDF-Loc](https://ieeexplore.ieee.org/document/8814347) — hybrid **ESDF + TSDF** relocalization; [*Metric Monocular Localization Using Signed Distance Fields*, IROS 2019, arXiv 2003.14157](https://arxiv.org/html/2003.14157v1).
- **Older but exactly on-point**: *Scale Invariant Robust Registration of 3D Point Data and a Triangle Mesh by Global Optimization* (2006) — point-set ↔ triangle-mesh registration robust to noise, outliers **and wrong scale** ([PDF](https://www.tnt.uni-hannover.de/papers/data/457/UrfMikSte06RobustRegistration.pdf)).
- **Transformer-based MCL in Construction Meshes** (2026) learns the observation model — PointNet++ encoder + place-recognition decoder — and uses it as the likelihood inside MCL against a **building mesh**, explicitly because "similar room layouts and low-texture surfaces pose a challenge" ([arXiv 2609.31357](https://arxiv.org/pdf/2609.31357)). This is the closest published analogue to the proposed architecture.
- **Distance-field registration** is a live research line: [2Fast-2Lamaa](https://arxiv.org/pdf/2410.05433v3) reports **0.06 m localization error using continuous distance fields** over 750 km of data `[snippet]`.
- **Neural / rendered maps** extend the same idea: [Loc-NeRF](https://arxiv.org/abs/2209.09050) runs MCL with a pre-trained NeRF as the map; [iNeRF](https://arxiv.org/pdf/2012.05877); [NeRF-VINS](https://arxiv.org/html/2309.09295) renders synthetic views from a NeRF map to create loop-closure-like constraints "not susceptible to [place-recognition] failure modes"; [GSplatLoc](https://arxiv.org/html/2412.20056v2); [Fast Global Localization on NeRF](https://arxiv.org/html/2406.12202v2); [NeRF-VIO](https://arxiv.org/html/2503.07952). 3DGS-based pose refinement is active but still **init-sensitive** ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kong_Rethinking_Pose_Refinement_in_3D_Gaussian_Splatting_under_Pose_Prior_CVPR_2026_paper.pdf)) `[snippet]`.

**Tooling for direct point-to-mesh residuals** (all confirmed to exist):
- **Open3D `RaycastingScene`** — `compute_signed_distance`, `compute_unsigned_distance`, `compute_closest_points`, `compute_occupancy`, with an internal acceleration structure. **Note it is the `o3d.t.geometry` (tensor) API.** ([Distance Queries tutorial](https://open3d.org/docs/latest/tutorial/geometry/distance_queries.html), [API](https://open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html))
- **libigl** `igl::copyleft::cgal::point_mesh_squared_distance` ([docs](https://libigl.github.io/dox/copyleft_2cgal_2point__mesh__squared__distance_8h.html)); **trimesh** `proximity.closest_point` / `signed_distance` ([docs](https://trimesh.org/trimesh.proximity.html)).
- **Caveat:** I could not verify whether Open3D's `evaluate_registration` accepts a `RaycastingScene`/mesh target — I believe it does **not**, so you build point-to-triangle residuals yourself. **UNVERIFIED.**

**Benefits vs a discrete point cloud**: no discretization error; a continuous field with **smooth gradients almost everywhere** (versus a piecewise-constant nearest-neighbour field that is locally flat between samples) — the standard argument for implicit representations ([Pan et al., T-RO 2024](https://www.ipb.uni-bonn.de/pdfs/pan2024tro.pdf)); **dense correspondences** (a rendered range image gives one correspondence per pixel rather than per detected feature); distances available at arbitrary resolution; you can render *exactly what the sensor should have seen*, which makes occlusion and visibility modelling natural; and the same mesh serves collision, rendering and localization. **Costs / failure modes**: the mesh must be watertight and consistent for *signed* distances (non-manifold or open meshes break the sign); implicit-neural SDF maps are sensitive to **incorrect SDF value estimation**, which "can have a significant impact on localization and mapping" ([UcPIN-SLAM](https://ieeexplore.ieee.org/document/11134702)); faces are flat so the SDF has gradient kinks at edges; ray casting wants a BVH or RTX-class device; render-and-compare is sensitive to appearance/material/lighting.

**The one claim you must not make:** **a mesh or SDF does not remove degeneracy.** *"A mesh corridor is still a corridor."* Degeneracy is a property of the *geometry*, not of the representation. **No source retrieved in this session claims mesh/SDF registration fixes degeneracy** — treat any such claim as unsupported.

### 4.2 Absolute metric scale from a prior map — solved and standard practice

**Yes, this is standard.** A prior map is an absolute metric reference; aligning to it recovers scale for any sensor suite that lacks it. Concretely:

- VIO/monocular systems drift *and* (monocular) have scale ambiguity; SLAM-style methods "achieve bounded error through loop closure" while VINS "inherently accumulate[s] unbounded drift over time" and needs external constraints ([arXiv 2412.04287](https://arxiv.org/html/2412.04287)).
- **Prior-map-constrained VIO** is a published, standard formulation: [Zuo et al., *Visual-Inertial Localization With Prior LiDAR Map Constraints*, RA-L 2019](https://april.zju.edu.cn/wp-content/papercite-data/pdf/zuo2019visualinertiallw.pdf) (`MSCKF` VIO plus an a-priori LiDAR map delivering *bounded-error* 3D navigation); [*Visual-Inertial Localization With Prior LiDAR Map Constraints* (IROS/RA-L 2019)](https://yangyulin.net/papers/2019_iros_ral_map_resue.pdf) registers to the prior map to bound an MSCKF VIO; [*Map-based Visual-Inertial Localization: A Numerical Study* (ICRA 2022)](https://pgeneva.com/downloads/papers/Geneva2022ICRA.pdf) shows prior-map methods beat odometry VIO in both position and orientation "even at large noise levels of 12 cm"; [*Map-based VIL: Consistency and Complexity*, arXiv 2204.12173](https://ar5iv.labs.arxiv.org/html/2204.12173); [*Global Visual–Inertial Localization for Autonomous Vehicles with Pre-Built Map*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10181573).
- **Explicit scale correction** from a map or from loop closure: [*Visual-LiDAR Odometry with Monocular Scale Correction*, arXiv 2304.08978](https://arxiv.org/html/2304.08978v2) (ORB-SLAM + A-LOAM with an explicit scale corrector); [*Bayesian Scale Estimation for Monocular SLAM… Correcting Scale Drift*, arXiv 1711.02768](https://ar5iv.labs.arxiv.org/html/1711.02768); map-reuse relocalization for monocular-inertial [arXiv 1803.01549](https://arxiv.org/html/1803.01549v1).

**A precision that matters for framing.** **Stereo is already metric** — a calibrated baseline puts scale *inside* every measurement, so scale is observable without any map ([scale observability note](https://cv-learn.com/visual-slam-roadmap/level-07-stereo-slam/scale-observability)). The prior map's job is therefore **not to supply scale but to bound drift**. Scale recovery is the *monocular* problem.

**In the Unity case**, scale is trivially exact by construction: the engine's units are the ground truth and the prior mesh is expressed in them. So the scale problem does not exist there at all. The only residual scale/geometry risks are `[Reasoning]`: (i) float precision in the Unity→map transform, (ii) any deliberate scale factor applied to the XR rig/camera, and (iii) depth-buffer **near/far clipping and non-linearity** in the synthetic depth (a real, non-obvious failure mode: near-plane clipping silently deletes the closest geometry, which is exactly the geometry that constrains pose best).

---

## 5. Is this standard practice in the "map already available" regime? Yes — it is the default in several industries

### 5.1 Warehouse / intralogistics AMRs

- The oldest and still most common architecture is **infrastructure-based absolute localization, not SLAM**: Kiva/Amazon uses **QR-code matrices on the warehouse floor read by a downward camera** ([AGV/AMR industry report](https://faxiangongchang.com/en/reports/china-mobile-robot-agv-amr-2026); [review of AMRs for warehouses, arXiv 2406.08333](https://arxiv.org/html/2406.08333v1)); magnetic tape and 2D barcode tags are the classical AGV localization stack ([ICAROB paper](https://alife-robotics.co.jp/members2019/icarob/data/html/data/OS_pdf/OS4/OS4-1.pdf)).
- Modern AMRs use **2D LiDAR + a pre-built 2D map in localization mode**, and frequently **reflector targets** for high-precision docking; the research frontier is *fusing* AMCL with fiducials to reach higher precision (e.g. *improved AMCL + QR-code assistance* for transport robots, 2025 `[snippet via Exa]`).
- Precision anchor: **sub-centimetre / sub-degree docking** is achieved by multi-stage localization ([Robotica 2024](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/7C4B7AB733A729E538F50B9CCDBAC3F3/S0263574724000602a.pdf/multistage_localization_framework_for_accurate_and_precise_docking_of_autonomous_mobile_robots_amrs.pdf)) — but that regime uses *docking-specific* references, not a room mesh.
- Vendor/blog framing worth knowing (not peer-reviewed): "a prior map is typically constructed in advance, allowing multiple robots to perform **drift-free localization using the same map**" ([*A Robust Visual SLAM System for Warehouse Robots Using Ground…*, arXiv 1710.05502](https://arxiv.org/html/1710.05502v4)) — useful because it states the industry assumption plainly.

### 5.2 Autonomous driving / HD maps

- **Autoware's default localization is literally NDT scan matching against a prebuilt pointcloud map**, plus Monte Carlo initial pose estimation exposed as a ROS service, plus regularization ([autoware_ndt_scan_matcher](https://autowarefoundation.github.io/autoware_core/main/localization/autoware_ndt_scan_matcher/)).
- Autoware publishes a **localization evaluation methodology and results** for urban environments using exactly this stack ([Autoware urban environment evaluation](https://autowarefoundation.github.io/autoware-documentation/main/simulation-evaluation/components_evaluation/localization_evaluation/urban-environment-evaluation/), [community discussion with measured results](https://github.com/orgs/autowarefoundation/discussions/5135)).
- Required accuracy is quantified in the literature: **< 0.1 m at ≥ 95 % confidence** for automated driving ([Sensors 2021](https://www.mdpi.com/1424-8220/21/17/5855)); semantic HD maps give "a few centimetres" ([arXiv 1908.03274](https://ar5iv.labs.arxiv.org/html/1908.03274)).
- **Counter-evidence that matters for failure modes:** in a 15-city Japanese field trial, NDT-based localization met accuracy and real-time requirements in most sites, *but at some sites the accuracy degraded or localization broke down due to insufficient 3D shape features in the environment* ([IATSS Review 42(2)](https://www.iatss.or.jp/entry_img/42-2-06.pdf)) `[Japanese-language PDF, snippet-translated]`. This is the single best "real deployment" evidence that prior-map localization is not unconditional.

### 5.3 AR / VR

- **HoloLens spatial anchors** exist precisely to persist a coordinate frame across time/sessions; the design guidance is about keeping holograms stable relative to the anchor ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/mixed-reality/design/spatial-anchors)); Unity's **World Locking Tools** is the standard way to stabilize a large tracked space ([WLT + ASA sample](https://learn.microsoft.com/en-us/mixed-reality/world-locking-tools/documentation/howtos/samples/wlt_asa_sample)).
- **Measured drift**: on HoloLens 2, mean drift was **< 2 cm** across all six tested user actions, while **smartphone AR drift is "frequently greater than 5 cm"** ([*Here to Stay: A Quantitative Comparison of Virtual Object Stability in Markerless Mobile AR*, arXiv 2109.14757](https://ar5iv.labs.arxiv.org/html/2109.14757), [author PDF](https://maria.gorlatova.com/wp-content/uploads/2022/03/HereToStay_CR.pdf)).
- **Measured limits**: on a 1 600 m² industrial floor with 60 m walks and highly dynamic features, ARCore/ARKit/HoloLens localization degraded substantially against a sub-millimetre optical ground truth ([Feigl et al., *Localization Limitations of ARCore, ARKit, and HoloLens in Dynamic Large-scale Industry Environments*, 2020](https://www.scitepress.org/Papers/2020/89899/89899.pdf)).
- **Prior-map AR is a published architecture**: [VIO-APR](https://arxiv.org/html/2308.05394v2) couples VIO with an absolute pose regressor so "the APR… compensate[s] for VIO drift", implemented in Unity for mobile AR; [Map-Relative Pose Regression, CVPR 2024 (Niantic)](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Map-Relative_Pose_Regression_for_Visual_Re-Localization_CVPR_2024_paper.pdf) reports APR position errors "in the range of a few centimeters" and uses the map as the reference frame.
- **Cloud anchor services** (Azure Spatial Anchors, ARCore Cloud Anchors / Geospatial VPS) are the productized form: relocalize against a stored map/anchor. Third-party marketing claims **sub-5 cm** persistence for self-hosted anchor replacement ([multiset.ai](https://multiset.ai/migrate-azure-spatial-anchors)) — **vendor claim, UNVERIFIED**; note the same source states **Azure Spatial Anchors was retired in November 2024** — **UNVERIFIED**.
- ARCore's Geospatial API "uses a combination of Google's Visual Positioning System (VPS) and GPS to determine the geospatial pose", i.e. an explicit prior-map relocalization product ([ARCore docs](https://developers.google.com/ar/develop/java/geospatial/enable)).

### 5.4 Research systems where "localize on a prebuilt map" is first-class

Cartographer pure localization ([docs](https://google-cartographer-ros.readthedocs.io/en/latest/tuning.html)), `hdl_localization` ([repo](https://github.com/koide3/hdl_localization)), FAST-LIO localization variants ([example](https://github.com/iral-ntua/fast_lio_localization)), [InstaLoc](https://arxiv.org/abs/2305.09552) (RSS 2023, one-shot indoor global localization in a prior map), [range-mcl](https://github.com/PRBonn/range-mcl) (mesh maps), [Loc-NeRF](https://arxiv.org/abs/2209.09050), and [OpenLiDARMap](https://arxiv.org/html/2501.11111v1) — whose reported result is the crispest version of the claim: **mean ATE below a metre where pure LiDAR odometry accumulates tens of metres** on multi-kilometre datasets `[snippet]`.

### 5.5 Does it actually *eliminate* drift?

**In the observable directions, yes** — the error becomes bounded by (map error + registration error + observability), and does not integrate over time. The literature is explicit that this is the whole point: map-based localization "constrain[s] localization and **eliminate[s] drift**" ([OpenLiDARMap](https://arxiv.org/html/2501.11111v1)); prior maps let "multiple robots … perform **drift-free localization**" ([arXiv 1710.05502](https://arxiv.org/html/1710.05502v4)).

**Caveats that are not optional:**
1. Drift is replaced by **pose error against the map**. If the map is wrong, that error is systematic and permanent.
2. **Unobservable directions still drift.** §3.
3. The localizer itself can **lock onto a wrong hypothesis** (perceptual aliasing) and then track it consistently — the failure is a confident, stable, wrong pose, which is worse than visible drift.
4. Global initialization is a separate, harder problem than tracking.

---

## 6. What prior-map localization does **not** give you

Prior-map localization answers **"where am I"**. It does not answer **"where may I go"** or **"what is there"**. Each of the following is a genuinely separate pipeline.

### 6.1 Traversability for a specific locomotion model — separate, and unsolved for teleport/VR

**A mesh with colliders gives you physics collision, not locomotion feasibility.** Colliders tell you nothing about riser height, gap width, slope, friction, whether a step is climbable, or whether a *particular* locomotion mode can execute a transition. Terrain traversability is its own research stack:

- [**STEP**: Stochastic Traversability Estimation and Planning, RSS 2021](https://arxiv.org/abs/2103.02828) — GP-based traversability with risk awareness.
- [**Wild Visual Navigation**, RSS 2023 / AURO 2025](https://arxiv.org/html/2404.07110) — self-supervised traversability from in-the-wild data.
- [**STEPP**, arXiv 2501.17594](https://arxiv.org/html/2501.17594); [**leggedrobotics/traversability_estimation**](https://github.com/leggedrobotics/traversability_estimation); [**GBPlanner**](https://github.com/ntnu-arl/gbplanner_ros) (a planner that *consumes* a traversability map).
- [**Towards Capability-Aware Traversability Navigation for Unstructured Environments**, arXiv 2607.20679](https://arxiv.org/html/2607.20679) — the crucial framing: "the same terrain can be traversable for one platform and unsafe for another", i.e. **traversability is conditioned on embodiment**. A wheeled base, a legged robot, and a teleporting VR avatar have *different* traversability functions over the *same* mesh.

**Gap to flag honestly:** I found **no literature on traversability for VR teleport locomotion**. Teleporting is not a locomotion constraint the terrestrial-robotics literature models — you can teleport through a wall, onto a table, or across a gap. **Treat "what can this avatar do" as a design problem, not a solved one**, and expect to encode it yourself (e.g. an explicit reachability/teleport-target volume derived from the mesh, plus a navmesh or capsule-sweep query).

### 6.2 Semantics — separate pipeline, not recoverable from geometry

A geometric mesh carries no labels. Open-vocabulary semantics come from VLM features fused into a map:

- [**ConceptGraphs** (arXiv 2309.16650)](https://arxiv.org/pdf/2309.16650.pdf) — open-vocabulary 3D scene graphs.
- [**ConceptFusion** (arXiv 2302.07241)](https://ar5iv.labs.arxiv.org/html/2302.07241) — multimodal open-set fusion.
- **OpenScene** (arXiv 2211.15654) and **VLMaps / 3D-VLMaps** (reported as `3DVLMapsOrg/3dvlmpas`-style repo by the research pass — **URL UNVERIFIED**).

You do not need any of this for metric localization; you cannot get it out of a mesh.

### 6.3 Dynamic obstacles — invisible by construction

Localizing against a **static** prior map treats people as *outliers to be rejected* (§3.4). It does **not** track, predict, or avoid them. Obstacle avoidance lives in a separate perception → prediction → planning stack:

- [Mixed crowd navigation survey (Annual Reviews)](https://www.annualreviews.org/content/journals/10.1146/annurev-control-032024-023929)
- [**SICNav** (arXiv 2310.10982)](https://arxiv.org/html/2310.10982)
- [Occlusion-aware crowd navigation (arXiv 2210.00552)](https://ar5iv.labs.arxiv.org/html/2210.00552)

Note the coupling direction: better localization *improves* the avoidance stack, but the avoidance stack is not derivable from localization.

### 6.4 What the mesh *can* legitimately be reused for

- **Offline ray-casting into an elevation / traversability prior** — the same geometry that gives you collision gives you the substrate for a costmap.
- **Rendering depth to pre-train or sanity-check place-recognition front-ends**, or to generate synthetic training data for the observation model (this is what MICP-L and range-mcl exploit at run time).
- **Localization itself** (§1–§4).

Nothing else transfers: **labels and locomotion costs are not contained in the mesh.**

---

## 7. Bottom line

**Is localization-against-a-known-map a sound way to eliminate drift entirely when the map is available?**

**Yes — it is the right architecture, and it is genuinely better than loop closure *for this regime*, but "entirely" is the wrong word.** Precisely:

- **It removes drift as an unbounded random walk.** Error stops integrating over time and becomes a bounded, map-referenced quantity. That is a categorical improvement over loop closure, which only bounds error where revisits happen and can be actively harmful when it produces false positives.
- **It does not make the error zero, and it does not bound error in degenerate directions.** In a symmetric corridor the along-axis translation is *structurally* unconstrained — Zhang/Kaess/Singh show the along-axis shift is a function of geometry and residual, **not of the measurement** — so the localizer will slide. Expect **1–5 cm / 0.2–1° indoors with rich geometry**, and **decimetres to unbounded along-axis error** in featureless or symmetric spaces. Your stated sensor (2 cm stereo at 1–2 m, 40k points) is adequate; **your geometry is the limiting factor, not the sensor.**
- **The real failure mode is degeneracy, not noise — and a mesh does not fix it.** "A mesh corridor is still a corridor." No retrieved source claims mesh/SDF registration removes degeneracy. Worse, **point-to-distribution/GICP registration with covariance regularisation puts an isotropic information floor on every correspondence, so the Hessian can look perfectly well-conditioned while the pose is wrong** ([LF-GICP, 2026](https://arxiv.org/html/2608.19522), preliminary). Do not use "my solver converged and my Hessian is fine" as evidence of correctness.
- **ICP-type registration has a narrow convergence basin**, so a *global* stage is not optional: coarse (FPFH+RANSAC / TEASER++ / place recognition) → multi-scale local refinement. And no single mitigation wins — the [Informed, Constrained, Aligned](https://arxiv.org/html/2408.11809v2) study finds mitigation must be tuned per environment.
- **In a Unity-rendered world with the mesh as both world and map, the sensor-noise term nearly disappears**, so the accuracy ceiling is set entirely by initialization, observability, and estimator design — and by whether you allow yourself privileged engine pose. Results measured this way will be optimistic versus hardware. Watch synthetic-depth near/far clipping and mesh-LOD swaps as the two ways this world still lies to you.
- **Point-to-mesh is standard**, not exotic: **MICP-L** (IROS 2024) registers range sensors *directly* to a triangle mesh via hardware ray casting; KMCL (ICRA 2012) rendered synthetic depth from an a priori model for MCL; range-mcl (ICRA 2021) localizes against triangle meshes; Freetures/SDF-Loc localize in SDF maps; and 2026 work does Transformer-based MCL directly against a building mesh. You do not have to convert the mesh to points — though doing so is also completely standard and is what Open3D/PCL/libpointmatcher/VoxelMap expect.
- **Absolute metric scale from a prior map is solved and standard** (prior-map-constrained VIO is a published formulation), and in Unity it is exact by construction. Scale is not your problem.
- **The industry verdict is unambiguous**: warehouse AMRs run localization against pre-built maps (often with fiducials/reflectors), Autoware's default is NDT against a prebuilt pointcloud map, HoloLens/ARCore anchor and relocalize against stored maps, and every major LiDAR stack ships a "localization mode" against a frozen map.

**What it costs — four real line items:**

1. **Initial global localization (the expensive, under-appreciated part).** Tracking needs a prior; a prior needs a search. You need place recognition (Scan Context / OverlapNet / NetVLAD-class) + geometric verification + hypothesis scoring, or a learned global relocalizer (InstaLoc, Transformer-MCL-in-meshes), or a fiducial/QR/engine hint. Budget for **multi-hypothesis belief** — a single-hypothesis filter cannot represent "I am in aisle 3 *or* aisle 7", which is exactly the indoor failure mode.
2. **Robustness engineering.** Multi-hypothesis MCL with random-particle injection (kidnapped-robot recovery), degeneracy detection and solution remapping (Zhang/Kaess/Singh, X-ICP, GenZ-ICP), ICP covariance estimation and health monitoring (Censi 2007, CELLO-3D), dynamic-object rejection, robust kernels, and a fallback when the map and the world disagree.
3. **Map as a liability.** A wrong or stale map produces confident wrong poses. Cartographer users hit this with parked forklifts; BIM-Loc exists specifically because real buildings differ from their models; the 15-city Autoware trial shows geometry-poor sites breaking NDT. You need change detection / partial-map updating, or you accept bounded-but-wrong.
4. **Everything else is still a separate pipeline.** Prior-map localization gives you *pose*, not traversability for your locomotion model, not semantics, and it does not detect dynamic obstacles — it treats them as outliers. For a **VR teleport avatar there is no traversability literature at all** — that is a design gap you must close yourself (§6.1). The mesh itself is reusable as an elevation/traversability substrate and as a depth renderer for training, but labels and locomotion costs are not in it.

**Recommended shape for this project:** LIO/VIO front end → scan-to-map (or render-and-compare/D-SDF) registration against the frozen mesh → UKF/EKF or pose-graph fusion with covariance → global relocalization via place recognition + ICP verification when confidence collapses → degeneracy detection that *freezes* the unobservable DOF and hands them to odometry/IMU. Add loop closure only if you also need to work when the map is absent. In a Unity world, build the ground-truth-pose read-out as an evaluation-only channel, and score the localizer with it disabled.

---

## 8. Verification notes and confidence

| Claim | Confidence | Note |
|---|---|---|
| Architecture, terminology, package roles | High | Documentation pages retrieved directly |
| Open3D ICP numbers (fitness 0.17/0.37/0.62, inlier_rmse 11.77/7.76/6.58 mm) | High | Open3D's own docs; consistent across doc versions. It *is* an indoor RGB-D scene (Redwood), but a **clean, static, tripod capture with a supplied initial guess** — treat as a best case, not a room result |
| Stereo depth magnitudes (D435 <2 % @2 m; ZED 2i <1 % to 3 m; ZED 2 <3 cm @4 m) | Medium-high | Datasheet + a 2024 RAS study + a 2025 four-camera comparison. The "<3 cm @4 m" and "<1 cm @1 m" phrasings come from an **AI-generated summary**, not the paper text |
| σ_z ∝ z²/(f·B) relation | High (textbook) | Multiply confirmed in sources; no PDF fetched |
| Stereo errors are heavy-tailed, non-Gaussian | Medium-high | Stated explicitly by the RAS 2024 ZED 2i study `[snippet]` |
| Indoor scan-to-map accuracy "1–5 cm / 0.2–1°" | Medium | **Synthesised range** across heterogeneous sources (LiDAR, automotive, BIM); a *range with conditions*, not a measured result for this system |
| ICP convergence basin "≈30° rotation ⇒ local minimum" | **LOW — folklore** | Only an informal vendor blog. **No peer-reviewed quantified basin figure in degrees was retrievable.** Do not cite the number |
| Degeneracy mechanism, δx_c formula, solution remapping | High | Zhang/Kaess/Singh ICRA 2016, multiple mirrored PDFs |
| "GICP covariance regularisation masks degeneracy" | Low-medium | [LF-GICP, Aug 2026](https://arxiv.org/html/2608.19522), single-author, brand new — treat as preliminary but worth guarding against |
| X-ICP localizability categories, numerical thresholds | Medium / **UNVERIFIED numbers** | Paper confirmed to exist (T-RO 2023); threshold values not retrieved |
| Dynamic-object damage (ORB-SLAM3 ATE 0.259 m on `fr3_w_xyz`) | Medium | From a 2026 Springer article `[snippet]`; it is a *visual* odometry proxy, not scan-to-map |
| MICP-L / range-mcl / Freetures / SDF-Loc **accuracy figures** | **NOT RETRIEVED** | The methods and their codes are confirmed to exist; **none of their RMSE/ATE numbers were retrievable**. This is the biggest hole in the report |
| Autoware NDT as default localization | High | Official docs |
| 15-city NDT breakdowns | Medium | Japanese PDF, snippet-translated |
| HoloLens 2 < 2 cm drift; smartphones > 5 cm | Medium-high | Peer-reviewed 2021/2022 measurement study, snippet |
| Feigl 2020 AR degradation (specific cm values) | Low | Abstract/snippet only — numbers **not** extracted |
| Azure Spatial Anchors retirement (Nov 2024) and sub-5 cm claim | **UNVERIFIED** | Vendor page only |
| `fast_gicp` / `small_gicp` repo URLs | **UNVERIFIED** | Quoted from memory; not confirmed by search in this session |
| KISS-ICP KITTI table values | Low | Snippet of a table without its header; do not quote without checking |
| arXiv 2112.02779 attribution | Corrected | By **Dong, Ryu, Kaess & Park** (range-image perspective) — *not* Vizzo, and it is not primarily an SDF-registration paper |
| AprilTag ±1–4 cm | Low | Secondary aggregator only |
| Place-recognition false-positive rates; KRP recovery success rates/times | **UNVERIFIED** | Not retrievable in this session |
| "AdaCorr" | **NOT LOCATED** | Possibly renamed/merged; treat as unconfirmed |
| "Unity world ⇒ negligible sensor noise / exact scale" | Reasoning | Follows from rendering the same mesh; not an empirical citation |
| VR/teleport traversability | **NO LITERATURE FOUND** | Treated as a design gap, not a solved problem |

**Methodological caveat for the whole report:** `web_fetch` was blocked in this session (all hostnames resolved to non-public IPs) and shell HTTPS failed TLS authentication, so **no source was read end-to-end**. Every number above is from an abstract, an indexed full-text fragment, or documentation, and most are marked `[snippet]`. Before those numbers go into a paper or a design doc, open the PDFs and confirm them. The **highest-value follow-up** is full-text retrieval of MICP-L (arXiv 2210.13904) and range-mcl (arXiv 2105.12121), because those two papers are the closest published match to the proposed architecture and their accuracy figures are exactly what is missing here.
