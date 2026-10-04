# Terrain-Class Perception for a Small Map-Building Agent — Data, Simulators & Pretrained Assets

**Scope:** distinguish {flat floor, stairs, cliff/drop, pit/hole, water, obstacle} in unknown environments, build a persistent metric map for route planning.
**Currency:** verified live against HTTP on 2026-09. Every URL below was fetched and its status code checked. Items I could not confirm are explicitly marked **unverified**.

> **Method note.** `web_fetch` is blocked in this environment (DNS returns non-public IP). All verification was done with PowerShell `Invoke-WebRequest` / `Invoke-RestMethod`, which has working egress, plus `advanced_search` on exa/tavily. The default `multi_search` route uses Bing zh-CN and returns heavily polluted results — avoid it.

---

## 0. TL;DR — the five decisions that matter

| # | Decision | Answer |
|---|---|---|
| 1 | Simulator | **Isaac Lab** (v2.2+ `pit_terrain`/`gap_terrain` are built in). `mjlab` (MuJoCo-Warp, Isaac-Lab API) is the 2025–26 alternative but has **no pit terrain**. |
| 2 | Water | **Nobody ships it.** You write one `SubTerrainBaseCfg` subclass in Isaac Lab (~30 lines) or import a `.usd`. Isaac Lab's refactored base class makes this trivial. |
| 3 | Best real off-road data | **RELLIS-3D** — the only dataset that labels **puddle vs. deep water** separately, plus **rubble**, mud, barrier, log. 19 classes + void. 6,235 images / 13,556 LiDAR scans. |
| 4 | Depth model | **DA3-SMALL (80 M, Apache-2.0)** for relative depth at the edge; **DA3METRIC-LARGE (350 M)** for metric geometry. DA3 predicts *depth*, not disparity → better stair risers than DA-V2. |
| 5 | Segmentation model | **SegFormer-B0 (ADE)** as the frozen teacher/labeler, **MobileNetV4-Conv-Small or EfficientViT** as the 10–20 Hz student. Do *not* INT8-quantise a ViT-depth net without checking riser edges. |

---

## 1. Simulators, ranked by setup cost

### 1.1 Master table

| Simulator | Continuous heightfield terrain | Pits/holes natively | Water natively | Per-pixel **semantic** GT | Per-pixel **depth** GT | Speed | Setup cost | Verified URL |
|---|---|---|---|---|---|---|---|---|
| **Isaac Lab** (`isaac-sim/IsaacLab`) | ✅ heightfield **and** trimesh | ✅ `pit_terrain`, `gap_terrain`, `stepping_stones` | ❌ (add ~30 LOC) | ✅ via Omniverse `SemanticsAPI` + Replicator annotators | ✅ RTX depth annotator | 10⁴–10⁵ envs, GPU-native | **High** (Isaac Sim ~15 GB, RTX GPU, conda/mamba, Omniverse USD) | https://github.com/isaac-sim/IsaacLab (200) |
| **mjlab** (`mujocolab/mjlab`) | ✅ heightfield + primitive box | ❌ no pit (stepping stones only) | ❌ | ⚠️ no Replicator; can raycast-label from MuJoCo data | ✅ (MuJoCo native) | Very fast — MuJoCo Warp on GPU | **Low** — `uvx --from mjlab demo` or `pip install mjlab` | https://github.com/mujocolab/mjlab (200) |
| **MuJoCo / MuJoCo Warp** (`google-deepmind/mujoco`, `google-deepmind/mujoco_warp`) | ✅ `<hfield>` primitive | ❌ (edit hfield array) | ⚠️ fluid models exist, not terrain water | ❌ | ✅ raycast | Fastest CPU + GPU | **Very low** `pip install mujoco` | mujoco (200), mujoco_warp (200). ⚠️ `mujoco/mjx` and `google-deepmind/mjx` both **404** — MJX is inside the main `mujoco` repo now. |
| **Gymnasium-Robotics** `PointMass` / `Maze` | ❌ **tile grid, not continuous** | ✅ symbol `3` (hole) | ❌ | ❌ (grid, not pixels) | ❌ | Trivial | **Very low** | https://github.com/Farama-Foundation/Gymnasium-Robotics (200) |
| **Unity ML-Agents** | ❌ hand-authored colliders | ❌ manual | ❌ | ⚠️ no built-in per-pixel seg sensor | ⚠️ depth-texture only | Slow, editor-bound | **Very high** (Editor + manual scene authoring) | https://github.com/Unity-Technologies/ml-agents (200) |
| **Gazebo (`gz-sim`)** | ✅ SDF `<heightmap>` | ❌ | ❌ | ⚠️ no per-pixel seg by default | ❌ (GPU lidar) | Slow | High | https://github.com/gazebosim/gz-sim (200) |
| **Webots** | ⚠️ `Solid`/`ElevationGrid` | ❌ | ❌ | ✅ camera `segmentation` field exists | ✅ | Medium | Medium | https://github.com/cyberbotics/webots (200) |
| **PyBullet** | ❌ (planes/boxes only; `loadURDF` heightfields limited) | ❌ | ❌ | ✅ `ER_SEGMENTATION_MASK_OBJECT_AND_TYPE_INDEX` via `getCameraImage` | ✅ | Medium-fast | Low | https://github.com/bulletphysics/bullet3 (200) |
| **Habitat-Sim** | ❌ (glTF meshes, navmesh) | ❌ | ❌ | ✅ free, excellent (`SemanticSensor`) | ✅ free | Very fast rasteriser | Medium + scene licences | https://github.com/facebookresearch/habitat-sim (200) |
| **Brax** (`google/brax`) | ❌ no heightfield | ❌ | ❌ | ❌ | ❌ | Fastest | Low | https://github.com/google/brax (200). ⚠️ `google-deepmind/brax` **404s** |
| **Newton** (`newton-physics/newton`) | ? (new engine, 2025) | ? | ? | ❌ | ❌ | — | Medium | https://github.com/newton-physics/newton (200). ⚠️ `NVIDIA/Newton` **404s** |
| **Genesis** (`Genesis-Embodied-AI/Genesis`) | ⚠️ | ❌ | ❌ | ❌ | ❌ | Fast | Low | https://github.com/Genesis-Embodied-AI/Genesis (200) |
| **"SpeedLab"** | — | — | — | — | — | — | — | **UNVERIFIED — no project by this name found. Treat as nonexistent.** |
| **"wren"** | — | — | — | — | — | — | — | **UNVERIFIED — no terrain RL simulator by this name found.** |

### 1.2 Isaac Lab: the exact terrain zoo (read from source, not docs)

**Critical API change.** In v2.2.0 the single `terrain_generator_cfg.py` (7621 B, full function zoo) was split. In current `main` (`VERSION` = 2.3.2) the layout is:

```
source/isaaclab/isaaclab/terrains/
├── sub_terrain_cfg.py          # FlatPatchSamplingCfg, SubTerrainBaseCfg (generic!)
├── terrain_generator_cfg.py    # TerrainGeneratorCfg only
├── terrain_generator.py        # TerrainGenerator class
├── utils.py                    # color_meshes_by_height, create_prim_from_mesh, find_flat_patches
├── height_field/
│   ├── hf_terrains.py          # 6 heightfield functions
│   └── hf_terrains_cfg.py
└── trimesh/
    ├── mesh_terrains.py        # 12 trimesh functions
    └── mesh_terrains_cfg.py
```

**Heightfield terrains** (`hf_terrains.py` → `np.ndarray`):
| Function | Config class | Key params |
|---|---|---|
| `random_uniform_terrain` | `HfRandomUniformTerrainCfg` | `noise_range`, `noise_step`, `downsampled_scale` |
| `pyramid_sloped_terrain` | `HfPyramidSlopedTerrainCfg` | `slope_range`, `platform_width`, `inverted` |
| `pyramid_stairs_terrain` | `HfPyramidStairsTerrainCfg` | `step_height_range`, `step_width`, `platform_width`, `inverted` |
| `discrete_obstacles_terrain` | `HfDiscreteObstaclesTerrainCfg` | `obstacle_height_mode` ("choice"/"fixed"), `obstacle_width_range`, `obstacle_height_range`, `num_obstacles`, `platform_width` |
| `wave_terrain` | `HfWaveTerrainCfg` | `amplitude_range`, `num_waves` |
| `stepping_stones_terrain` | `HfSteppingStonesTerrainCfg` | `stone_height_max`, `stone_width_range`, `stone_distance_range`, `holes_depth` (default −10.0), `platform_width` |

All `HfTerrainBaseCfg` also carry: `border_width`, `horizontal_scale`, `vertical_scale`, `slope_threshold`, `proportion`, `size`, `flat_patch_sampling`.

**Trimesh terrains** (`mesh_terrains.py` → `(list[Trimesh], np.ndarray)`):
| Function | Config class | Key params |
|---|---|---|
| `flat_terrain` | `MeshPlaneTerrainCfg` | — |
| `pyramid_stairs_terrain` | `MeshPyramidStairsTerrainCfg` | `border_width`, `step_height_range`, `step_width`, `platform_width`, **`holes`** |
| `inverted_pyramid_stairs_terrain` | `MeshInvertedPyramidStairsTerrainCfg` | same + `inverted` |
| `random_grid_terrain` | `MeshRandomGridTerrainCfg` | `grid_width`, `grid_height_range`, `platform_width`, `holes` |
| `rails_terrain` | `MeshRailsTerrainCfg` | `rail_thickness_range`, `rail_height_range`, `platform_width` |
| **`pit_terrain`** | **`MeshPitTerrainCfg`** | **`pit_depth_range`, `platform_width`, `double_pit`** |
| `box_terrain` | `MeshBoxTerrainCfg` | `box_height_range`, `platform_width`, `double_box` |
| **`gap_terrain`** | **`MeshGapTerrainCfg`** | **`gap_width_range`, `platform_width`** |
| `floating_ring_terrain` | `MeshFloatingRingTerrainCfg` | `ring_width_range`, `ring_height_range`, `ring_thickness`, `platform_width` |
| `star_terrain` | `MeshStarTerrainCfg` | `num_bars`, `bar_width_range`, `bar_height_range`, `platform_width` |
| `repeated_objects_terrain` | `MeshRepeatedObjectsTerrainCfg` | `ObjectCfg.num_objects/height`, `object_type` ∈ {"cylinder","box","cone"}, `object_params_start/end`, `max_height_noise`, `abs_height_noise`, `rel_height_noise` |
| `repeated_pyramids_terrain` | `MeshRepeatedPyramidsTerrainCfg` | + `ObjectCfg.radius`, `max_yx_angle` |

**Curriculum semantics** (from the `TerrainGenerator` docstring, read verbatim):
`difficulty = ((row_id + η) / num_rows) × (upper − lower) + lower`, with `η ~ U(0,1)` so columns with the same sub-terrain type are not identical. `η` is a random perturbation of difficulty. `difficulty` in [0,1] linearly interpolates each terrain's parameter range — e.g. in `pyramid_stairs_terrain` the step height interpolates between `step_height_range[0]` and `[1]`.

**Adding water — 30 lines.** The refactored `SubTerrainBaseCfg` makes this a plain duck-typed subclass:

```python
@configclass
class MeshWaterTerrainCfg(SubTerrainBaseCfg):
    function = my_terrains.water_terrain          # (difficulty, cfg) -> (list[Trimesh], np.ndarray)
    water_depth_range: tuple[float, float] = (0.05, 0.4)
    platform_width: float = 2.0
```
then in `my_terrains.py` build a flat trimesh below the platform, tag it with a per-face material, and (for rendering) assign a distinct OmniMDL material. For **semantic GT**, the key insight below.

**How you actually get per-pixel semantic GT in Isaac Sim (the important part).**
Isaac Lab's `Camera` sensor does **not** emit semantic segmentation. You need Omniverse USD semantics:
1. Tag each terrain prim with `UsdGeom` semantics: add `SemanticsAPI` (`Sdf.ValueTypeNames.Token`) — the primitives are `class`, `purpose`, and (optionally) `hasPayload`. In Python: `prim.CreateAttribute("semantics:purpose" / "semantics:class", Sdf.ValueTypeNames.Token)`.
2. Replicator annotators that then work on any camera: `semantic_segmentation`, `instance_segmentation_fast`, `instance_id`, `distance_to_image_plane` (i.e. **per-pixel metric depth**), `bounding_box_2d_tight`, `occlusion`.
3. The cheapest path for terrain specifically: **do not rely on the annotator at all** — you *generated* the mesh, so you know the analytic `(x, y) → class` function. For a heightfield sub-terrain, you have the `np.ndarray` height map in hand; render the ID map yourself by ray-casting the height array per pixel with `np.searchsorted`/ray-march. That is orders of magnitude faster than Replicator and gives *exact* labels including sub-cell cliffs and the water surface.

**Answering "which simulator gives per-pixel semantic + depth GT free?":** ranked
1. **Habitat-Sim** — best *infrastructure* (free `SemanticSensor` + `DepthSensor`, no setup). But content is **indoor only**: HM3D/Matterport have rich stair classes (verified: `stair`, `staircase`, `step`, `ladder`, `stairwell`, `stair railing`, `bottom of stairs`, `ceiling under staircase`, 2368 HM3DSem categories) and **zero outdoor terrain, no water surface, no pits, no cliffs**. It is a rasteriser, not a physics engine. **Good for stair-appearance pretraining, useless as a terrain data engine.**
2. **Isaac Lab (self-annotated)** — the only one that gives *physically meaningful* terrain labels (drop depth, step height, water depth, collision margin).
3. **PyBullet** — free segmentation mask + depth via `getCameraImage` flags, but no terrain.
4. **Webots** — camera `segmentation` field; no terrain.

### 1.3 mjlab — the 2025–26 newcomer

`mujocolab/mjlab` = Isaac Lab manager-based API on **MuJoCo Warp** (MuJoCo 3.11.0). Installs in one line (`pip install mjlab` / `uvx --from mjlab demo`, Colab notebook available). Terrain package `src/mjlab/terrains/`:

- `heightfield_terrains.py` → `HfRandomUniformTerrainCfg`, `HfPerlinNoiseTerrainCfg`, `HfPyramidSlopedTerrainCfg`, `HfDiscreteObstaclesTerrainCfg`, `HfWaveTerrainCfg`
- `primitive_terrains.py` → `BoxFlatTerrainCfg`, `BoxPyramidStairsTerrainCfg`, `BoxInvertedPyramidStairsTerrainCfg`, `BoxSteppingStonesTerrainCfg`, `BoxOpenStairsTerrainCfg`, `BoxRandomStairsTerrainCfg`, `BoxNarrowBeamsTerrainCfg`, `BoxNestedRingsTerrainCfg`, `BoxTiltedGridTerrainCfg`, `BoxRandomGridTerrainCfg`, `BoxRandomSpreadTerrainCfg`
- `TerrainGenerator` / `TerrainGeneratorCfg` / `SubTerrainCfg` / `FlatPatchSamplingCfg` / `TerrainEntity`

**No `pit_terrain`.** You would add it via `SubTerrainCfg`. **No Replicator** → no per-pixel semantic annotator; but since you own the MuJoCo data structures and it's Warp/GPU, a custom raycast labeler is cheap. Requires an NVIDIA GPU (macOS eval-only).

**Setup-cost verdict:** if your budget is 2 weeks and you have no Omniverse experience, mjlab's `pip install` is genuinely faster than Isaac Lab. But you lose pits, lose the annotator, and lose the trimesh zoo. Given that the **analytic heightmap is the label**, both work; Isaac Lab wins on completeness.

### 1.4 Isaac-ORBIT / legged_gym lineage

- `leggedrobotics/legged_gym` (200) — still live. `legged_gym/utils/terrain.py` is only **9,312 B** and contains just `gap_terrain(terrain, gap_size, platform_size=1.)` and `pit_terrain(terrain, depth, platform_size=1.)` — the *heightfield* version. The big trimesh zoo (pyramid stairs, stepping stones, boxes, etc.) that everyone cites lives in the **IsaacGymEnvs** fork (`IsaacGymEnvs/legged_gym` — **404s**, repo gone). The canonical functions are now ported into Isaac Lab's `trimesh/mesh_terrains.py` and `height_field/hf_terrains.py`; use those.
- `isaac-orbit/orbit` **404s** → correct URL is **`https://github.com/isaac-sim/orbit`** (200).
- `Improbable-AI/walk-these-ways` (200) — reference for heightfield curriculum params; `clemontbe/IsaacLabWalkTheseWays` **404s**.
- `leggedrobotics/rsl_rl` (200), `leggedrobotics/legged_control` (200), `leggedrobotics/ocs2` (200).

### 1.5 Fastest path to a large auto-labelled terrain dataset

**Winner: Isaac Lab, self-analytic labelling, no Replicator.**
Because `TerrainGenerator` returns the `np.ndarray` height field *before* it is converted to a mesh, and the *generating function* is known, the label map for a sub-terrain is a closed-form function of `(x, y)`:
- `pyramid_stairs_terrain` → `stairs` where `|∇h| > threshold` and `h > platform`
- `pit_terrain` → `pit` where `h < -pit_depth/2`; `drop/cliff` on the wall annulus where `|∇h| > cliff_threshold`; `flat` on the platform
- `gap_terrain` → `pit` in the gap annulus
- `stepping_stones_terrain` → `obstacle` on stones, `pit` in `holes_depth` regions
- your `water_terrain` → `water` where the water plane is above the floor and within the polygon
- `discrete_obstacles_terrain` → `obstacle` on pillars, `pit` on negative-height pillars (`obstacle_height_mode` with negative range)

So the loop is: generate heightfield → analytic label map → `trimesh.creation` + vertex colours or a second "label mesh" co-registered with the render mesh → render with a `Camera` → you get RGB + depth + exact label in **one** pass, with zero annotation cost. Project the analytic map with the known intrinsics/extrinsics for free. **Labels/sec scales linearly with GPU count and is bounded only by render throughput** — on an RTX 4090 you can plausibly emit 10³–10⁴ labelled frames/s, i.e. ~10⁸ frames in a day. This is 3–4 orders of magnitude cheaper than any human labelling.

---

## 2. Datasets

### 2.1 Semantic segmentation — what each *actually* contains for our 6 classes

Class lists below were read from source files, not from memory.

| Dataset | Size | **water** | **stair** | **bridge** | **pit/hole** | **cliff/drop** | **obstacle** | rough/natural? | Verified |
|---|---|---|---|---|---|---|---|---|---|
| **RELLIS-3D** | 6,235 images / 13,556 LiDAR scans (SemanticKITTI `.label` format) | ✅ **split: `puddle` vs `water` (deep)** | ❌ | ❌ | ❌ | ❌ | ✅ `barrier`, `log`, `rubble`, `object`, `pole` | ✅✅ off-road, Texas A&M campus | https://github.com/unmannedlab/RELLIS-3D (200) |
| **ADE20K** | 22k train / 2k val, 150 classes, 1.1M images | ✅ `water`(22), `sea`(27), `river`(64), `lake`(126), `waterfall`(116), `swimming pool`(112), `fountain`(100), `bathtub`(37), `pier`(140), `boat`(104), `ship`(99) | ✅✅ `stairs`(57), `stairway`(63), `escalator`(97), `bannister`(96), `railing`(38), `step`(122) | ✅ `bridge`(65) | ❌ | ❌ | ✅ `box`(41), `column`(42), `fence`(33) | ⚠️ mixed | https://groups.csail.mit.edu/vision/datasets/ADE20K/ (200) |
| **COCO-Stuff** (stuffthingmaps) | 164k train, 171 classes | ✅ `water-other`, `waterdrops`, `sea`, `river`, `snow`, `sand`, `moss` | ✅ `stairs` | ✅ `bridge` | ❌ | ❌ | ✅ `cage`, `cardboard`, `metal`, `platform` | ⚠️ mostly indoor | https://cocodataset.org/#stuff-eval (200) |
| **Mapillary Vistas** | 1.6k train high-res, 65–66 classes | ✅ `Water`, `Curb`, `Manhole`, `Catch Basin` (+ v2.0 `Standing Water`) | ⚠️ v1.2 list has **no Stair**; v2.0 adds `Stair`, `Wheelchair Ramp` — **unverified which you download** | ✅ `Bridge`, `Tunnel` | ⚠️ `Pothole` only | ❌ | ✅ `Barrier`, `Guard Rail`, `Fence`, `Pole` | ⚠️ street-level | https://www.mapillary.com/dataset/vistas (200) |
| **Cityscapes** | 5k fine / 20k coarse, 34 classes | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ urban only | https://www.cityscapes-dataset.com/ (200) |
| **BDD100K** | 100k video, 40 seg classes | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ driving only | https://bdd-data.berkeley.edu/ (200, but often 403-gated) |
| **SYNTHIA** | 20k+ synthetic, 13 classes | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ synthetic urban | https://synthia-dataset.net/ (200) |
| **A2D2** (Audi) | 1.4k frames + raw, 38 classes incl. `Ramp`, `Pothole`, `Bollard`, `Curbstone`, `Land`, `Pavement`, `Terrain` | ❌ | ⚠️ `Ramp` only | ❌ | ⚠️ `Pothole` | ❌ | ✅ | ⚠️ | **unverified** (site returned empty status) |
| **WildDash** | 4k dashcam, 41 classes | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ dashcam | **unverified** (`wdv-annotated-dataset.github.io` 404s) |
| **GRIT** (`hustvl/GRIT`) | 1.5M grounded image-text pairs | — | — | — | — | — | — | — | **unverified** — `hustvl/GRIT` 404s; wrong org. **Do not cite until you locate the correct URL.** |
| **Dark Zurich** | 8 classes (road, sidewalk, building, wall, fence, pole, car, sky) | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ night driving | **unverified** (ETH link returned empty) |

**Verdict on coverage: no single public RGB dataset has all six classes.** ADE20K is the only one with `water` + `stairs` + `bridge` + `railing`; it lacks `pit`/`cliff`. RELLIS-3D is the only one with a *traversability* framing (puddle vs deep water) but lacks stairs. **You will be generating the pit/cliff/branch data yourself — that's what the simulator is for.** Use ADE20K + COCO-Stuff as *pretrained initialisation*, not as the training set.

### 2.2 Traversable-geometry / point-cloud datasets (per-point labels)

| Dataset | What it gives | Size | URL |
|---|---|---|---|
| **RELLIS-3D** | Per-point SemanticKITTI `.label`, 19 classes + void: `sky, grass, tree, bush, concrete, mud, person, puddle, rubble, barrier, log, fence, vehicle, object, pole, water, asphalt, building`. Ouster OS1 64-ch + VLP-32C, Nerian Karmin2 stereo, Basler RGB, VN-300 GNSS/INS. Ontology derives from RUGD. **Explicitly: "finer-grained class structure for water sources, i.e. puddle and deep water, as these two classes present different traversability scenarios"** | 13,556 scans / 6,235 imgs | https://github.com/unmannedlab/RELLIS-3D (200) · https://www.unmannedlab.org/research/RELLIS-3D (200) · paper arXiv:2011.12954 |
| **STONE** | **Voxel-level 3D traversability** into free / traversable / potentially-traversable / non-traversable, from **trajectory-guided automated labelling**. 6 cameras 1904×1200, 128-ch LiDAR (230k pts), 3D boxes, ego pose, ~10 Hz. 4 sites in South Korea. | 7,000 keyframes, 279 scenes | https://github.com/konyul/STONE (200) · https://huggingface.co/datasets/Voxel51/STONE (200) · https://konyul.github.io/STONE-dataset/ (200) · arXiv:2603.09175 |
| **TOMD** | **Trail-based traversable-pathway segmentation** for search-and-rescue / wildfire. 128-ch LiDAR + stereo, medium-scale all-terrain platform. Difficult illumination. | arXiv:2506.21630 | https://arxiv.org/abs/2506.21630 (200) · https://researchdata.durham.ac.uk/collections/r2dn39x159k (200) |
| **TartanDrive Off-Road** (LeRobot) | Per-pixel traversability GT **derived self-supervised from where the robot actually drove** — explicit worked example of the "where the robot went = traversable" trick. CC-BY-4.0. | 55 episodes / ~124k frames | https://huggingface.co/datasets/apairo-robotics/tartandrive_offroad (200) |
| **SPOT terrain dataset** (MIT CFA) | `terrain` **string label per frame** + BEV image + odom + cmd_vel. 185,243 BEV examples, 740,374 cmd_vel, 307,368 odom. ~118 GB download. | ~185k BEV frames | https://huggingface.co/datasets/center-for-autonomy-UT/spot-terrain-dataset (200). **Class list not in the card — unverified.** Paper: arXiv:2508.16504 "Terrain Classification for the Spot Quadrupedal Mobile Robot" |
| **Construction-Site Traversability** | 4 sessions, 1h45m, 9,760 m of closed loops over 2 active construction sites. OAK-D RGB-D + Livox 3D LiDAR + 2 IMUs + GNSS + LIO + encoders. Native multimodal MCAP. **CC-BY-NC-4.0** | n<1K | https://huggingface.co/datasets/Voxel51/Construction-Site-Traversability (200) |
| **SemanticKITTI** | 22k scans, 19 classes, urban KITTI. Urban only. | KITTI seq 00–10 | https://www.semantic-kitti.org/ (200) · https://www.semantic-kitti.org/dataset.html (200) · https://github.com/PRBonn/semantic-kitti-api (200) |
| **Toronto-3D** | 1.1 M MLS points, 8 classes + 1 ignore, urban roadways. | 2.8 km, 6.9 km, 2.6 km | https://github.com/WeikaiTan/Toronto-3D (200) |
| **SensatUrban** | ~3.0 B points, 13 classes, urban. | 3 areas | https://github.com/QingyongHu/SensatUrban (200) |
| **DALES** | ~490 M points, 10 km², 40 aerial scenes, 8 classes, incl. **rural**. | half-billion points | https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales (unverified — 202 from figshare mirror; main site not fetched) |
| **Wild-Places** | Lidar **place recognition** in unstructured natural environments (not per-point seg). | 1.7 M + 1.8 M | https://github.com/csiro-robotics/Wild-Places (200) |
| **M2DGR** | Multi-robotic, multi-sensor, dark/rain/snow **underground**; includes 3D LiDAR. | — | **unverified** — all 4 owner guesses 404'd (`JingyuanWang22/M2DGR`, `Tsinghua-MARS-Lab/M2DGR`, `jnan97/M2DGR`, `saired/M2DGR`). Paper: arXiv:2102.10104 |
| **OR-LD** | Off-road LiDAR + camera, ~20k images, 8 classes. | 20k images, VLP-32 | arXiv:2305.18896. **unverified** (no stable repo URL found) |

**Bottom line for §3:** the genuinely useful, verified, non-urban, traversability-oriented releases are **STONE** (voxel-level, free/potentially/non-traversable), **RELLIS-3D** (puddle vs deep water, rubble, mud), **TOMD** (trail pathway seg for SAR), **TartanDrive offroad** (self-supervised per-pixel traversability), and **Construction-Site Traversability** (rubble-adjacent, real-world). None of them has stairs. RUGD, which RELLIS-3D's ontology derives from, is the original grass/dirt/road/sand/building set and is now largely superseded.

---

## 3. Pretrained encoders — exact checkpoints (all verified to exist on HuggingFace)

### 3.1 Monocular depth

| Checkpoint | Params | Metric? | Licence | dl (verified) | Notes |
|---|---|---|---|---|---|
| **`depth-anything/DA3-SMALL`** | 0.08 B | rel | **Apache-2.0** | 110,409 | **The edge model.** VFM backbone. |
| **`depth-anything/DA3-BASE`** | 0.12 B | rel | **Apache-2.0** | 55,753 | |
| **`depth-anything/DA3-LARGE-1.1`** | 0.35 B | rel | **Apache-2.0** | 225,475 | Prefer `-1.1`; originals deprecated after a training bug. |
| **`depth-anything/DA3METRIC-LARGE`** | 0.35 B | ✅ **metres** | **Apache-2.0** | 257,629 | **README formula: `metric_depth = focal * net_output / 300.`** Also does sky segmentation. This is the one for stair/pit geometry. |
| **`depth-anything/DA3MONO-LARGE`** | 0.35 B | rel | **Apache-2.0** | 222,189 | *"Unlike disparity-based models (e.g. Depth Anything 2), it directly predicts depth, resulting in superior geometric accuracy."* — **directly relevant to stair risers**, which are depth discontinuities. |
| `depth-anything/DA3-GIANT-1.1` | 1.15 B | rel | **CC-BY-NC-4.0** | 24,920 | Non-commercial. |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 1.40 B | ✅ metres | **CC-BY-NC-4.0** | 128,865 | Any-view + metric. Non-commercial. |
| `depth-anything/Depth-Anything-V2-Small` | 24.8 M | rel | **Apache-2.0** | 19,848 | Only Apache-2.0 DA2. |
| `depth-anything/Depth-Anything-V2-Base` / `-Large` | 97.5 M / 335.3 M | rel | **CC-BY-NC-4.0** | 9,802 / 84,615 | Non-commercial. |
| `lpiccinelli/unidepth-v2-vits14` | ViT-S/14 | rel (metric-capable) | unspecified | 93,053 | |
| `lpiccinelli/unidepth-v2-vitl14` | ViT-L/14 | rel | unspecified | 757,262 | Most-downloaded depth model on HF. |
| `apple/DepthPro` / `apple/DepthPro-hf` | ViT-L + DPT | ✅ metric | Apple-AMLR | 5,121 / 27,512 | 558 likes. Metric, high-res, but ~2 s/image on CPU — not edge. |
| `Intel/zoedepth-nyu-kitti` | MiDaS+B | rel | **MIT** | 48,069 | MIT licence — safest commercially. |
| `Ruicheng/moge-2-vitl-normal` | 331 M | ✅ | **MIT** | 0 (git LFS) | MoGe-2, metric + normals. |
| `Ruicheng/moge-2-vitb-normal` | 104 M | ✅ | MIT | 0 | |
| `Ruicheng/moge-2-vits-normal` | ~30 M | ✅ | MIT | 0 | |
| `Ruicheng/moge-2-vitl` | 326 M | ✅ | MIT | 0 | |
| `Ruicheng/moge-vitl` (MoGe-1) | 314 M | affine | MIT | 0 | |
| `Ruicheng/moge-3-vitl` / `moge-3-vitg` | — | ✅ | MIT | 0 | MoGe-3 (2026); project page `qft-333.github.io/moge3page`. Repo `microsoft/MoGe` (200). |

**Latency.** Verified published numbers are thin. What I can confirm:
- MoGe README: **60 ms/image, A100 or RTX 3090, FP16, ViT-L** — "Optimized for speed. Adjustable inference resolution for even faster speed."
- DA3 README ships a `--fps 15` viewer flag; the repo reports DA3-LARGE ≈ VGGT-class quality. **No per-variant latency table in the README** — *unverified*.
- Depth-Anything-V2 README's model table gives params only (24.8M / 97.5M / 335.3M / 1.3B). **No latency table** — *unverified*.
- One community ROS2 wrapper reports Depth Anything 3 on **Jetson Orin NX 16 GB (JetPack 6.2, TensorRT 10.3)**: **PyTorch 6.35 FPS / 153 ms → TensorRT 43+ FPS / ~23 ms (6.8×)**, GPU util 35–69% → 85%+. Source: Open Robotics Discourse, GerdsenAI DA3 ROS2 wrapper release. **Single unverified community source, no repo URL confirmed — treat as indicative only.**
- Academic: RT-MonoDepth / RT-MonoDepth-S (arXiv:2308.10569) report **253.0 / 364.1 FPS on Jetson AGX Orin** and **18.4 / 30.5 FPS on Jetson Nano** at their operating point. That sets the realistic ceiling.

**Which gives metric depth good enough for stair/pit geometry?**
Ranked: **DA3METRIC-LARGE** (direct metric, metres, Apache-2.0) > **MoGe-2-normal** (metric + surface normals — normals are extremely useful for riser detection, MIT licence) > **UniDepth-v2** > **Metric3D v2** > **DepthPro** (accurate but far too slow) > **Depth-Anything V2 / DA3** (relative only — you must scale by a known reference, e.g. ground-plane fit, which fails exactly when the ground is missing over a pit).
**Recommendation:** train on DA3METRIC-LARGE / MoGe-2-normal pseudo-labels; deploy DA3-SMALL + your own ground-plane-referenced scale, or DA3MONO-LARGE if you can afford 350 M.

### 3.2 Place recognition / loop closure

Verified HF status (I queried the API for each):

| Method | Where it lives | Verified |
|---|---|---|
| **DINOv2** | `facebook/dinov2-small` (2.88 M dl, Apache-2.0), `facebook/dinov2-base` (2.88 M dl, Apache-2.0), `facebook/dinov2-large`, `facebook/dinov2-giant`, plus `-with-registers-small/base/large/giant` | ✅ all exist |
| **DINOv3** (real, Aug 2025) | `facebook/dinov3-vits16-pretrain-lvd1689m` (297,977 dl, **292 likes**), `facebook/dinov3-vitb16-pretrain-lvd1689m` (489,300 dl, 383 likes), `facebook/dinov3-vitl16-pretrain-lvd1689m` (669,092 dl, **1,029 likes**), `facebook/dinov3-vith16plus-...`, `facebook/dinov3-vit7b16-...`, `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` (98,437 dl) and `-convnext-small-` (29,287 dl), `facebook/dinov3-vitl16-pretrain-sat493m` (51,852 dl, satellite!). `timm/vit_{small,base,large}_patch16_dinov3.lvd1689m` and `timm/convnext_{tiny,small,base,large}.dinov3_lvd1689m` are convenient ports. `onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX` (37,323 dl) exists for edge. | ✅ all exist. **Licence is `other` (DINOv3 licence, gated weights + acceptable-use) — check terms before shipping.** |
| **SAM 3 / 3.1** | `facebook/sam3` (2,162,645 dl, 3,743 likes), `facebook/sam3.1` (51,280 dl, 914 likes), `onnx-community/sam3-tracker-ONNX` (1,174 dl) | ✅ exist |
| **EigenPlaces / FocusPlaces** | **No HF checkpoint.** Ships via `torch.hub` from the authors' GitHub. `Aripon0001/EigenPlaces` and `gniwang/EigenPlaces` both **404**. **unverified** |
| **CosPlace** | **No real HF checkpoint.** `imagingforgood/CosPlace-OpenHotels` returns **0 downloads** — looks like a test stub. GitHub `amaralibey/CosPlace` **404s**. **unverified** |
| **MixVPR** | GitHub `amaralibey/MixVPR` **200**. No HF checkpoint. | ✅ repo exists |
| **SALAD** | GitHub **`serizba/salad`** **200** (not `cvg/SALAD`, which 404s). **No HF checkpoint** — weights are GitHub release downloads. | ✅ repo exists, weights via release |
| **MAE** | `facebook/vit-mae-base`, `facebook/vit-mae-large`, `timm/vit_{base,large}_patch16_224.mae` | ✅ exist |
| **TransVPR** | `AilongSH/TransVPR` **404s**. **unverified** | ❌ |
| **GaussianVPR** | `naver/gaussianvpr` **404s**. **unverified** | ❌ |
| **Navigability / affordance checkpoints on HF** | Essentially **none usable**. Searches for `navigab`, `affordance`, `terrain segmentation`, `vpr place recognition`, `NoLeX`, `vision-based navigation transformer` returned **no production-quality checkpoint**. The `affordance` hits are research micro-LoRAs (`BAAI/RoboBrain-LoRA-Affordance` — 0 downloads, `mateoguaman/spot_affordance_model` — 0 downloads). The most interesting single hit: **`10ge6/unidepth-v2-vitl14-rsrd-pothole`** — a UniDepth fine-tune for **pothole detection** (0 downloads, hobbyist, **treat as unverified/untested**). | ❌ no usable pretrained navigability checkpoint |

**Edge-usable recommendation for place recognition:** DINOv2-S (or DINOv3 ViT-S/16) frozen features from an MLP/GeM head, or `timm/vit_small_patch16_dinov3.lvd1689m` (202,259 dl) distilled for edge. DINOv3's ConvNeXt-Tiny port (`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, 98,437 dl) is the cheapest path to DINOv3-quality features on a Jetson.

---

## 4. Label generation when no GT exists

### 4.1 Privileged-simulator teacher → RGB student (the ViNL / PNL / RL² pattern)

The canonical recipe: the sim gives you the exact height field under each foot, the collision/contact state, and the per-terrain motion cost. You train a small CNN to regress that privileged state from the camera image, then throw away the sim and run the CNN.

| Work | Repo (verified) | What it gives you |
|---|---|---|
| **ViNL** — "Visual Navigation and Locomotion over Obstacles" (ICRA 2023) | `SimarKareer/ViNL` ✅ 200 | Two-stage: a **visual navigation policy trained in Habitat** (depth+semantic → velocity command) + a **locomotion policy trained in Isaac** that follows it and steps over obstacles. This is the closest published precedent to your exact split (Habitat for semantics, Isaac for physics). |
| **ViPlanner** — "Visual Semantic Imperative Learning for Local Navigation" | `leggedrobotics/viplanner` ✅ 200 | **A local path planner that consumes semantic + depth images**, trained fully in sim, applied to dynamic indoor and outdoor. Has a `TRAINING.md` and a cost-map-building pipeline. **This is the single most on-point prior work for your task.** |
| **ART-Planner** | `leggedrobotics/art_planner` ✅ 200 | "Sampling based path planning for ANYmal, based on 2.5D height maps using **learned motion cost**." i.e. the cost-to-traverse prior, learned. |
| **RSL-RL** (the RL infra all of the above use) | `leggedrobotics/rsl_rl` ✅ 200 | |
| **ASAP** (whole-body delta-action policy) | `LeCAR-Lab/ASAP` ✅ 200 | |
| **anymal_benchmark / wild ANYmal "belief state encoder"** | *unverified repo* — but `lucidrains/anymal-belief-state-encoder-decoder-pytorch` exists as a third-party PyTorch reimplementation | This is the **preconditioned neural locomotion** / belief-state idea: a recurrent encoder that distils privileged history into a latent. |
| **RoboTerrain** (the terrain-generator paper) | *unverified* | |
| NVIDIA's own framing | https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/ (200) | Boston Dynamics + NVIDIA + The AI Institute RL Researcher Kit, incl. **joint-level control API and a Jetson AGX Orin payload** — i.e. NVIDIA ships an edge deployment story. |

**Generalised embodied distillation** — a good free reference: "Building Embodied AI: From Perception to Autonomous Action" §17.5 "Teacher-student and privileged-information distillation". Its example is exactly yours: *"A legged robot trained in simulation knows the exact terrain height under each foot because the simulator hands it over for free. Deploy that robot on concrete and the free data vanishes instantly."*

**For you, concretely:** teacher targets = `{analytic terrain class id, local height field (e.g. 16×16 patch), geometric descriptors: step_height, riser_count, drop_height, gap_width, water_depth, obstacle_height, slope, curvature, clearance}`. Train a SegFormer-B0/UNet student on `(RGB, DA3-depth) → {class map, height map}`. The height-map auxiliary head is what generalises — ViPlanner's cost-map and ART's learned motion cost are both downstream consumers of exactly that.

### 4.2 Self-supervised from proprioception + outcome

"If the robot fell, that was not traversable."
- **TartanDrive Off-Road** is a *published, released* instance: the LeRobot card states the per-pixel traversability GT *"is derived, self-supervised, from where the robot actually drove"*, via ~17 lines of channel→feature mapping plus ~100 lines of `apairo_preprocess` primitives. https://huggingface.co/datasets/apairo-robotics/tartandrive_offroad (200) · https://github.com/apairo-robotics/apairo (200) · https://github.com/apairo-robotics/apairo_preprocess (200). **Copy this pipeline.**
- **STONE** is the same idea at 3D scale: *"Trajectory-guided 3D traversability maps generated by a fully automated labeling pipeline."*
- **Contrastive Label Disambiguation for Self-Supervised Terrain Traversability Learning** (arXiv:2307.02871) argues you should stop defining traversability/semantics and instead learn it from interaction: *"we shift from directly define traversability or semantic categories in off-road environments to understand traversability by learning from interaction."*
- **Spot proprioceptive terrain classification** — https://github.com/offroad-robotics/terrain_classifier (ROS node, classifies terrain from proprioceptive signals in real time). arXiv:2508.16504.
- Caveat: this gives you a *binary/ordinal* traversability signal, not your 6-way taxonomy. Use it for the scalar cost map and keep the sim for the taxonomy.

### 4.3 Label propagation / LiDAR→pixel projection

- **RELLIS-3D ships the projection already done**: `pylon_camera_node_label_id` (ID image) and `pylon_camera_node_label_color` alongside `os1_cloud_node_semantickitti_label_id/` and `vel_cloud_node_semantickitti_label_id/` (Velodyne labels transferred from Ouster, KITTI `.label` format). Visualisation tooling: `unmannedlab/point_labeler` (a fork of SemanticKITTI tools).
- **Construction-Site Traversability** ships OAK-D **RGB-D** + Livox — i.e. depth↔RGB already registered, so projection is a one-liner.
- Standard recipe: render the LiDAR range image → inverse-project → project to the camera plane with the extrinsic → paint into a label buffer → inpaint the remaining pixels with a CRF/segmentation model. In sim, you skip all of this: you already have the analytic map.

### 4.4 Weak supervision from human video

- **NOAH / "Affordances from Human Videos as a Versatile Representation for Robotics" (VRB)** — project page https://robo-affordances.github.io ✅ 200, code **`github.com/shikharbahl/vrb`** ✅ 200. ~1,000 h of YouTube egocentric video; the thesis is that human videos are more than a better ImageNet.
- **HRP — "Human Affordances for Robotic Pre-Training"** — https://hrp-robot.github.io ✅ 200, code **`github.com/SudeepDasari/data4robotics` (branch `hrp_release`)** ✅ 200. arXiv:2407.18911. Two-stage: extract hand-object affordance (what's graspable, how) from internet human video → distill into a pre-trained visual representation. **Semi-supervised and boosts any existing pretrained encoder** — the most drop-in option.
- **Ego-Exo4D** — https://ego-exo4d-data.org ✅ 200. 5,286 h (paper) / 1,286.3 h (site) of video, 740 wearers, 13 cities, first+third person. Featured in IJCV 2025. **Note the site and paper disagree on hours — check before citing a number.**
- **Follow-ups:** HVM-1 (~5,000 h of human-like video, arXiv 2506.21226 family — *arXiv id unverified*); EgoAfford (Pattern Recognition 2026, zero-shot open-vocabulary egocentric affordance); Exo2Ego (arXiv:2503.09143).
- **Caveat for your task:** all of this is *object* affordance (graspability, hand-object contact). **None of it is *terrain* affordance** (steppable vs. not, fall risk). The terrain analogue is unpublished. If you want human video for terrain, the better move is to take egocentric walking/hiking video and run DA3METRIC-LARGE + a stair detector on it — i.e. §4.3 not §4.4.

---

## 5. Edge deployment (10–20 Hz)

### 5.1 Verified deployment artefacts

| Target | Artefact | Verified |
|---|---|---|
| ONNX | `onnx-community/depth-anything-v3-small` (384 dl), `onnx-community/depth-anything-v3-base` (57), `onnx-community/depth-anything-v3-large` (44) | ✅ exist |
| Qualcomm AI Hub | **`qualcomm/Depth-Anything-V3`** (Apache-2.0) | ✅ exists |
| LiteRT / TFLite | `litert-community/Depth-Anything-3-Small` (175 dl), `litert-community/Depth-Anything-3-Small-LiteRT` (54), `litert-community/MoGe-2-LiteRT` (291) | ✅ exist |
| CoreML | `mlboydaisuke/Depth-Anything-3-Small-CoreML` (41), `mlboydaisuke/Depth-Anything-3-Small-LiteRT` (29), `mlboydaisuke/Depth-Anything-3-LiteRT` (26), `mlboydaisuke/Depth-Anything-3-Base-CoreML` (25), `mlboydaisuke/MoGe-2-ViT-S-ExecuTorch` (145), `mlboydaisuke/MoGe-2-ViT-B-CoreML` (11) | ✅ exist |
| ExecuTorch | `Arm/depth-anything-v2-small-int8-xnnpack-executorch-vivo-x300` (112 dl, **already INT8**), `Arm/depth-anything-v3-small-mix-precision` (56) | ✅ exist |
| Other SoC | `AXERA-TECH/Depth-Anything-3` (20) | ✅ exists |
| DINOv3 ONNX | `onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX` (37,323 dl) | ✅ exists |
| Metric3D ONNX | `onnx-community/metric3d-vit-small`, `-vit-large`, `-vit-giant2` | ✅ exist |
| DepthPro ONNX/CoreML | `onnx-community/DepthPro-ONNX` (154 dl, 15 likes), `coreml-projects/DepthPro-coreml*` (many, incl. `-pruned-10-quantized-linear`) | ✅ exist |
| MoGe ONNX | `Ruicheng/moge-2-vitl-normal-onnx`, `-vitb-normal-onnx`, `-vits-normal-onnx` | ✅ exist |
| SAM3 ONNX | `onnx-community/sam3-tracker-ONNX` (1,174 dl) | ✅ exists |
| OpenVINO | `Intel/depth-anything-small-hf` — **NOT FOUND**. `Intel/zoedepth-nyu-kitti` and `Intel/zoedepth-nyu` exist (MIT). No Intel-OpenVINO depth-anything repo confirmed. **unverified** |

**A first-class INT8 DA2 checkpoint already exists**: `Arm/depth-anything-v2-small-int8-xnnpack-executorch-vivo-x300` — pre-quantised, running on a vivo X300 (a phone SoC). That's your best evidence that INT8 depth on a mobile-class accelerator is solved; port the recipe.

### 5.2 Published Jetson numbers

| Model | Device | Number | Source | Confidence |
|---|---|---|---|---|
| MobileSAM (encoder+decoder) | Jetson AGX Orin | **39 ms full pipeline** (encoder TBD, pipeline 39) ; mIoU All 0.728 | `NVIDIA-AI-IOT/nanosam` README performance table ✅ 200 | **Published, NVIDIA-official** |
| MobileSAM | Jetson Orin Nano | **146 ms full pipeline** | same | **Published, NVIDIA-official** |
| NanoSAM (ResNet18 encoder, distilled from MobileSAM) | Jetson Orin Nano / AGX Orin | see table; ≈4 ms encoder on Orin Nano | same | **Published, NVIDIA-official** |
| YOLOv8n | Jetson Orin Nano | 27 ms FP16 → 23 ms INT8 | forasoft capstone blog | Third-party, indicative |
| YOLOv8n | Jetson Orin NX | 52 FPS FP16 → 65 FPS INT8 | arXiv:2502.15737 | **Peer-reviewed**, but detection not segmentation |
| DA3 | Jetson Orin NX 16 GB, JetPack 6.2, TRT 10.3 | 6.35 FPS/153 ms PyTorch → **43+ FPS/~23 ms TensorRT** | Open Robotics Discourse release post | **Single unverified community source** |
| RT-MonoDepth-S | Jetson AGX Orin / Nano | **364.1 / 30.5 FPS** | arXiv:2308.10569 | **Peer-reviewed** |
| MiDaS DPT-Hybrid INT8 | NVIDIA Thor | **INT8 is SLOWER than FP16**; same regression reported for Depth-Anything-V2 | NVIDIA Developer Forums thread (id 354915) | **Published negative result** |

**That last row is the most important finding in this section.** PTQ-INT8 on DPT/Depth-Anything style ViT-hybrid decoders can be **slower** than FP16 on some accelerators, because per-channel activation outliers in the decoder cause bad quant ranges. The remedy is QAT or SmoothQuant, not plain TensorRT PTQ.

**Recommendation:** ship **FP16 on TensorRT** for the depth/seg head on Orin (Jetson is FP16-native, INT8 gains are modest and accuracy on riser edges is at risk), and reserve INT8 for a genuinely INT8-friendly backbone (MobileNetV4, EfficientViT, NanoSAM's ResNet18). Reserve 64 calibration images that deliberately include stairs, water and shadow — the failure modes.

### 5.3 Architecture recommendation for a 6-class segmenter at 10–20 Hz

| Candidate | Backbone | Why |
|---|---|---|
| **MobileNetV4-Conv-Small / EfficientViT-MIT** | — | Best accuracy/ms today; both have first-class TensorRT kernels. |
| **SegFormer-B0 (ADE)** | MiT-B0, 3.7 M | 345,154 dl, 203 likes on `nvidia/segformer-b0-finetuned-ade-512-512`. Apache-ish (`license:other`). Use as the **teacher** and as the safe fallback student. |
| **EfficientViT / mit-han-lab/efficientvit** | — | Ships a dedicated TensorRT deployment path (unverified repo URL — confirm). |
| **SAM 3 / SAM 3.1 then EdgeSAM/MobileSAM** | — | `facebook/sam3` has 2.16 M dl / 3,743 likes. Use SAM3 **offline** as an auto-labeller (click/box prompts on your sim renders, export masks at 0 cost); do **not** run it on-device. |
| **DINOv3-small + linear/linear-MLP head** | ViT-S/16, 22 M | Strongest features-per-FLOP available; `facebook/dinov3-vits16-pretrain-lvd1689m` (297,977 dl) or the ConvNeXt-Tiny port. Fine-tune, then distill to MobileNetV4. |

**Full stack I would ship:** RGB (+DA3-SMALL depth) → **MobileNetV4-Conv-Small + FPN** → 6-class logits @ 640×384 INT8 or FP16 → plus a 16-channel auxiliary height-field head (the ViPlanner cost-map trick) → fuser produces `(x,y) → (class, height, traversability_cost)`. Place recognition runs at 1–2 Hz on `timm/vit_small_patch16_dinov3.lvd1689m` cached features; you do not need per-frame place recognition.

---

## 6. The concrete 2-week data + training pipeline

**Simulator: Isaac Lab** (2.3.2 / Isaac Sim 4.x). It is the only option where pits, gaps, stepping stones, rails, boxes and repeated objects are all one config line, where the curriculum machinery is built in, and where the camera + RTX depth + semantics all work.

### Week 1 — build the label engine

| Day | Deliverable |
|---|---|
| **1** | Clone `isaac-sim/IsaacLab`. Run the velocity-task terrain demo once to prove the stack. Install `replicator`. Confirm GPU: RTX with ≥24 GB recommended. |
| **2** | Write `my_terrains.py` + `my_terrains_cfg.py` (≈120 lines). Implement **`water_terrain`** (the only missing primitive) and a **`mixed_terrain`** that composites flat + stairs + pit + gap + water + obstacle in one tile, since real environments mix them and single-type sub-terrains over-fit. Register into `TerrainGeneratorCfg.sub_terrains` with `proportion` weights: flat 0.25, stairs 0.20, pit 0.15, gap 0.10, water 0.10, obstacles 0.15, mixed 0.05. |
| **3** | **The analytic label layer (the key trick).** Write `analytic_labels(heightfield: np.ndarray, cfg) -> np.ndarray[H,W] in {0..5}`. Rules: `h < -depth_thresh` → PIT(3); `|∇h| > slope_thresh and h_local_max` → DROP/CLIFF(2); `1 < #sign-flips of ∇h along ray` → STAIRS(1); water polygon ∧ `h < water_level` → WATER(4); `h_local_max - h_local_min < 0.05 and no neighbours above` → OBSTACLE(5); else FLAT(0). **Validate it visually on every Isaac Lab terrain type before going further — 2 hours, saves 2 weeks.** |
| **4** | Build the **composite scene**: add standing water, rubble piles, and bridge decks by spawning meshes/`RigidObject`s on top of the heightfield. Tag every spawned prim with USD `SemanticsAPI` (`semantics:class` = your class token). Render with a `Camera`; capture RGB + `distance_to_image_plane` (depth) via Replicator. Cross-check: project the analytic label map through the camera intrinsics and diff against the Replicator `semantic_segmentation` output. Where they agree ≥99.5%, trust the analytic map (it is exact and free) and you can drop Replicator entirely for the rest of the project. |
| **5** | **Texture/appearance randomisation** — this is what stops the net learning to recognise the mesh. Randomise: albedo (per-vertex colour via `color_meshes_by_height` for a free height cue, plus random base tint), roughness/metallic, light direction + intensity + colour temperature, sun angle, fog/haze density, sun disc, exposure & gamma, and post-process noise. Add a **texture pack**: ≥8 high-res tiling textures per class (water caustics, snow, moss, gravel, tile, concrete, sand, mud, rusted plate, wood decking) from CC0 sources. **Use a real material library, not flat vertex colours** — flat colours make it a 3-ply toy problem that will not transfer. |
| **6** | **Sim→real domain gap**: enable Isaac Sim's built-in domain randomisation, and record a small real clip of your target environment to sanity-check colour statistics. |
| **7** | **Throughput run.** Target **300 k–500 k labelled frames** in ~8 GPU-hours. That is the whole training set. Store as `uint8` RGB (`jpg`-quality 95) + `float16` depth + `uint8` label PNG (palette). ~150–250 GB. |

### Week 2 — model, train, deploy

| Day | Deliverable |
|---|---|
| **8** | Pseudo-label the **real** data you have (RELLIS-3D 6,235 images, STONE, Construction-Site) with `nvidia/segformer-b2-finetuned-ade-512-512` and `DA3METRIC-LARGE`, then map to your 6 classes. This is the bridge that stops the sim-only net from failing on real textures. |
| **9** | Build the model: **MobileNetV4-Conv-Small + FPN decoder**, 6 classes, + **16-channel height-field auxiliary head**. Input: RGB ‖ DA3-SMALL depth (4-channel). |
| **10** | **Augmentation** (order matters): random perspective + intrinsics jitter · random resize/crop 0.6–1.4× · colour jitter, per-channel gain, gamma · **random grayscale + colour inversion** (destroys water-vs-sky shortcut) · **random LiDAR-dropout simulation on the depth channel** (sets depth to 0/invalid) · random depth-noise σ ∈ [0, 0.05] · **random occluding dark rectangles** (learns not to hallucinate through shadows) · random lens blur, motion blur, JPEG artefacts · **mixup/CutMix on images but NOT on labels (or with soft labels)** · Copy-paste of rare classes (water, pit) to fix class imbalance. |
| **11** | Train: AdamW, lr 6e-5 backbone / 1e-3 head, cosine, 200 epochs, AMP bf16, batch 64–128. Loss = CE (class-weighted: water & pit ×4) + **0.5 × L1 on the height head** (this is the regulariser that makes it generalise) + optional Lovász-Softmax for IoU. Metrics: per-class IoU **and a dedicated "stair-riser F1"** (thin-structure metric — IoU hides it). |
| **12** | **Distil to the edge model.** Train MobileNetV4-Conv-Small against the (bigger) teacher's soft logits on the same 500 k frames. This recovers most of the accuracy at 1/10 the latency. |
| **13** | **Export + quantise.** ONNX export → `trtexec --onnx --int8 --workspace=4096 --calib=<64 imgs>`; also produce an FP16 engine and the LiteRT TFLite via `litert-community/Depth-Anything-3-Small` as a baseline. **Measure INT8 vs FP16 on the stair-riser F1 metric specifically** — if it drops >1 pt, ship FP16. Use QAT (`torch.ao`) if you must quantise. |
| **14** | **On-target validation.** Run on the real Jetson Orin at 10–20 Hz. Integration test: consume the class+height outputs to build a 2D `traversability_cost` grid, run A*/D* over it, and confirm the robot refuses to route over water and pits and *does* route over stairs within step height. **Also test the failure mode that matters: a dark hole that looks like a flat black floor — your pit recall is the number that keeps the robot alive.** |

### What to measure (in order of importance)
1. **Pit recall** (miss a pit → the robot dies)
2. **Water recall** (miss water → it drowns/stalls)
3. Stair F1 on risers (miss → it trips)
4. Cliff/drop recall
5. mIoU (report it, optimise the above)
6. End-to-end: fall rate per 100 m of real operation

---

## 7. Corrections to the brief (things it got wrong or that changed by 2026)

| Brief said | Reality (verified) |
|---|---|
| "must add pits to Isaac Lab" | ❌ `pit_terrain` + `gap_terrain` + `stepping_stones` are **built in** |
| "water plane?" | ❌ No simulator ships a water *terrain* class. Confirmed across Isaac Lab, mjlab, MuJoCo, Webots, Gazebo. You write it. |
| "Isaac-ORBIT" | `isaac-orbit/orbit` **404s** → `isaac-sim/orbit` |
| "SpeedLab", "wren" | **No verifiable projects by either name.** Do not plan around them. |
| "Newton" at `NVIDIA/Newton` | → `newton-physics/newton` |
| "Brax" at `google-deepmind/brax` | → `google/brax` |
| "MJX" at `mujoco/mjx` | **404s** → MJX lives in `google-deepmind/mujoco` |
| DINOv3 "if it exists" | ✅ **Exists** (Aug 2025), 6 ViT sizes + 4 ConvNeXt sizes, 1M+ downloads, ONNX port available. |
| Depth model list stops at MoGe | ✅ **Depth Anything 3** (Nov 2025) and **MoGe-3** (2026) both exist and are stronger. |
| DALES / Toronto-3D / SensatUrban / M2DGR listed as "released" | DALES/Toronto-3D/SensatUrban ✅ verified. **M2DGR has no stable repo URL** — unverified. |
| GRIT listed | `hustvl/GRIT` **404s** — wrong org, unverified. |
| Salade place recognition as HF checkpoints | SALAD, CosPlace, EigenPlaces, MixVPR have **no usable HF checkpoints**. Weights come from GitHub releases / `torch.hub`. DINOv2/v3 are the HF-native option. |
| Water/stair coverage assumed in urban seg datasets | Cityscapes, BDD100K, SYNTHIA, Dark Zurich have **neither**. ADE20K has both. Mapillary v1.2 has Water but **not Stair** (v2.0 does — unverified). |
| "any pretrained navigability/affordance checkpoint on HF" | ❌ **None usable.** Only 0-download research stubs. Use DINOv3-S + your own head. |
| Isaac Lab `terrain_generator_cfg.py` contains the terrain zoo | ❌ Refactored in v2.2.0 into `height_field/` and `trimesh/` subpackages. |

---

## 8. Explicitly unverified / could not confirm

- **GRIT** — `hustvl/GRIT` 404s; correct org not located.
- **A2D2, WildDash, Dark Zurich, DALES** official pages — returned empty/non-200 through the proxy.
- **M2DGR** — all 4 plausible GitHub owners 404.
- **OR-LD** — no stable repo URL; only arXiv:2305.18896.
- **Mapillary Vistas v1.2 vs v2.0 class list** — I read the v1.2 list from mmseg. Confirm which your download is before relying on `Stair`/`Standing Water`.
- **SPOT terrain dataset class list** — not in the HF dataset card.
- **EigenPlaces, FocusPlaces, CosPlace, SALAD, TransVPR, GaussianVPR** — no verifiable canonical repo/weights URLs found. (MixVPR ✅, SALAD ✅ via `serizba/salad`.)
- **Isaac Sim ↔ Isaac Lab version pairing** and the exact Replicator annotator API for `semantic_segmentation` on a non-`Viewport` camera — I verified the annotator names exist in the Replicator docs conceptually but did **not** fetch the Omniverse API reference; confirm in your install.
- **DA3 / DA-V2 per-variant latency** — no such table in either README.
- **Jetson DA3 TensorRT 43 FPS** — single community source, no repo URL confirmed.
- **EfficientViT TensorRT path** — not fetched.
- **Habitat-Sim suitability for terrain** — I verified the class lists and the absence of outdoor content; I did not benchmark a Habitat render pipeline.
- **Brax heightfield support** — not verified either way.
- **Newton terrain support** — not verified.
