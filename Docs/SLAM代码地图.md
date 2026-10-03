# SLAM 代码地图（2026-09-30 重写）

> **本文已重写。** 原版（2026-09-23）描述的"两条 SLAM 路线"里，**在线单目 SLAM 那一条的实现在
> 之后的重构中被整体删除**（`slam_core.py` / `live_mapping.py` / `relocalization.py` /
> `online_pose.py` / `nav_plan.py` / `nav_target.py` / `route_executor.py` / `realtime_depth.py` /
> `relocalization_observer.py` —— 源码已删，仅剩孤儿 `.pyc`，因此原版多处已失效）。
> 原版全文见 git `688e420` 之前的历史。
>
> **权威口径与 21 组冲突登记见 [`Docs/README.md`](README.md)。** 本文只描述**现状**。
> ⚠️ 2026-10-04 复核：§三「活跃链」表**漏了** 2026-10 起上线的 `nav_xsession.py`（跨会话 P0 主体，默认开）、
> `nav_bow.py`、`nav_memory.py`，已补入 `Docs/README.md` §二 权威矩阵。

---

## 一、一句话结论

1. **现在不是"两套 SLAM"**，是**一条离线 2.5D 流水线 + 一条在线 navmesh**，
   而且两者的产物**至今没有绑定**（"去某地点"因此无入口）。
2. **`.slam_probe` 不能删 —— 但理由变了**：**不是** backend 运行期 `import` 它
   （实扫证明那个依赖是 **0**），而是**未迁移的半条流水线 + 全部录制素材都在里面，且它被 gitignore**。
3. **项目自有目录内的文件重名只剩 2 组**，其中 **1 组是真隐患**（见 §四）。
4. **`world model` 一词在本仓库有三个含义**，新增代码**不要再用它**（见 §四）。

---

## 二、两条路线

| | **路线 A：离线 2.5D 经验图** | **路线 B：在线 navmesh** |
|---|---|---|
| 代码 | `research/recorder/` + `.slam_probe/offline_probe/recorder/` | `backend/nav_online.py`、`nav_mapping.py`、`nav_loop.py`、`nav_grid.py`、`nav_follow.py` |
| 输入 | OBS 录像 + OSC + HMD | **镜像双目 SGBM** + OSC 速度 + HMD 朝向 |
| 位姿 | 航位推算（`backend/pose_math.py`）+ 位姿图 | 20 Hz 航位推算 + `nav_loop.LoopCloser`（ORB+PnP，**只修平移**，朝向来自 HMD） |
| 产物 | `topo_map` / `nav_map.json` / `world_model.json` / 2.5D 障碍层 | 三态栅格（free/obstacle/unknown）+ navmesh |
| 接线 | **离线跑，人工触发**（入口 `map_from_capture.py`） | **已接进后端**：`service.py` 构造 `OnlineNavigator`，11 处调用 |
| HTTP | 无 | `GET /worldmodel/navmesh`、`POST /worldmodel/navmesh/{start,stop,goto,explore,cancel}` |
| 状态 | 主样本 `20260920-233456` 验收 **pass**（21 节点 / 20 边），manifest 为 `degraded` | 局部建图跑通；**跨会话检索 ✅ / 采纳下游消费 ❌**（2026-10-04 更正，见 `Docs/README.md` §三 **C18**） |

**两条路线各有坐标系与尺度、没有绑定** —— 这是当前最大的结构洞。
详见 [`Docs/自动到达能力差距清单.md`](自动到达能力差距清单.md)（2026-09-30 已重写）。

---

## 三、活跃链（按流水线层次）

| 层 | 现行文件 | 职责 |
|---|---|---|
| **采集（桌面/窗口）** | `backend/wgc_capture.py`（**遗留，决定移除**）、`backend/vision.py`（`WgcWindowFrameSource` / `desktop_mirror` / mss / dxcam 四种源） | 2D 帧 |
| **采集（双目镜像）** | `backend/openvr_mirror.py`（`MirrorEye`、`read_stereo`、`projection_intrinsics`） | SteamVR 合成器镜像纹理，两眼同帧；GPU mip 降采样 |
| **时间基准** | `backend/time_alignment.py`；`research/tools/estimate_video_offset.py`；`.slam_probe/.../recorder/time_align_check.py` | 视频↔OSC↔HMD 统一单调钟 |
| **深度** | **在线**：`backend/nav_mapping.py` 的 `make_sgbm` / `stereo_disparity` / `stereo_points`（**backend 内没有任何深度模型**）；**离线**：`recorder/openvino_depth_probe.py` + Depth Anything V2 | 视差 → 点 → 栅格 |
| **在线建图** | `backend/nav_mapping.py`（关键帧增量三态栅格）、`nav_grid.py`（`FREE`/`OCC`/`UNK`、`GridMeta`）、`nav_online.py`（`OnlineNavigator`：位姿 / 关键帧 / 控制 / 快照） | 首访世界边走边建 |
| **回环** | `backend/nav_loop.py`（`LoopCloser`） | ORB 互最近邻 → 候选关键帧双目 3D 点 + 当前帧 2D → `solvePnPRansac` + LM；**HMD 相对 yaw 一致性是最强外点过滤** |
| **航位推算数学** | `backend/pose_math.py` | **唯一实现**（纯 `math`，无 numpy）。⚠️ **在线 navmesh 不经过它**，见其 docstring |
| **规划 / 跟随** | `backend/nav_follow.py`（跟随器）；`backend/navigator.py`（LLM 侧 10 Hz 闭环 + 卡墙判据） | 局部移动 |
| **世界状态** | `backend/world_state.py`、根目录 `world_salience.py` | 实体/事件账本 + 唤醒分级 |
| **世界身份** | `backend/world_model.py` | **W1：当前在哪个世界** + 启停骨架；按世界分区记忆路径 |
| **感知** | `backend/local_perception.py`（检测器）、`reid_embedder.py`（OSNet）、`traversability.py` | 人 / 可通行性 |
| **可视化** | `GET /worldmodel/navmesh` → `grid_view()`（栅格 PNG + 像素↔世界米换算）、前端 `ui/navmesh.js`；离线 `recorder/map_preview.py` + HTML | 地图预览 |
| **回归门 / 工具** | `research/tools/regression_place_identity.py`、**`import_audit.py`**（AST 跨根依赖审计）、**`doc_health.py`**（文档体检） | 防复发 |

`map_from_capture.py` 用 **subprocess** 串起离线那一串（不是 import）：
`estimate_video_offset → scale_calib → build_topo_map → place_group → loop_verify → pose_graph → nav_map → obstacle_map_2d → check_pointcloud_orientation → map_preview → world_model_build`。
⚠️ 迁移曾把它打断（两个步骤的目标脚本已迁走），已于 2026-09-29 修复并按脚本所在目录解析。

---

## 四、⚠️ 陷阱

### 4.1 文件重名：现在只剩 2 组（原版列的 5 组已大多消失）

实扫（项目自有目录：根、`backend/`、`research/`、`tests/`、`.slam_probe/offline_probe/recorder/`）：

| 重名 | 状态 |
|---|---|
| `__init__.py` | 根 与 `backend/` —— **包初始化，不是隐患** |
| **`test_world_model.py`** | `tests/`（现行单测）与 `.slam_probe/offline_probe/recorder/`（旧的） —— **唯一真隐患，改测试时别改错那份** |

原版的 `pose_graph.py` / `obstacle_map_2d.py` / `offline_route_replay.py` / `probe.py` 重名
**都已随重构消失**（对应文件已删或只剩一份）。

### 4.2 同一算法存两份：**已收敛，但这段历史必须留着**

> 在线 `online_pose.py` 与离线 `pose_graph.py` 曾各写了一份"本地矢量位移按朝向旋进世界"。
> 在线那份一直对；**离线那份退化成标量路程沿 yaw 前进**（`x += ds*sin(yaw)`），
> 等价于把横移当前进 —— 实测该 run 终点差 **1.419 m**（当次横移报文仅占 2.9%）。

2026-09-24 收敛：公式只留一份在 **`backend/pose_math.py`**，离线三处都调它
（`research/recorder/run_motion.py`、`research/recorder/pose_graph.py`、
`research/tools/stereo_seq_ground_truth.py`）。

**规矩**：航位推算家族里**不允许出现任何 `cos/sin/tan` 调用** —— 需要旋转就调 `pose_math`。

**已知有意保留的其它旋转副本**（形状不同，不是航位推算，**勿照抄**）：
`research/recorder/scale_calib.py::_R_wc`（3×3 矩阵，相机→世界三维落点）、
`pointcloud_build.py` / `pointcloud_splat.py` 里的 GL/JS 着色器旋转（渲染坐标变换）。

### 4.3 🔴 `world model` 一词三义 —— **新增代码不要用这个词**

| 含义 | 位置 |
|---|---|
| **W1 世界身份**（"当前在哪个世界"+启停骨架） | `backend/world_model.py` |
| **建图产物** | `recorder/world_model_build.py` → `world_model.json`（schema `neko.world_model/v1`） |
| **持久空间记忆**（新北极星里的那个） | `ROADMAP.md` 北极星；`Docs/业界世界模型方案落地评估（2026-09-29）.md` |

⇒ **说"世界身份"或"地点记忆"，不要写"世界模型"**。这条**已经在文档里造成过实际混淆**
（见 `Docs/README.md` §三 C7 / C15）。

---

## 五、🔴 为什么 `.slam_probe` 不能删（理由已更正）

**原版的理由是错的。** 它说"`backend/` 有 4～5 个模块运行时去 `.slam_probe` import"，
举的证据（`backend/live_mapping.py:22-31`、`backend/nav_plan.py`、`backend/realtime_depth.py`、
`backend/offline_route_replay.py`、`backend/relocalization_observer.py`）**全部不存在**。

2026-09-29 三重实扫证明**发行面对 `.slam_probe` 的运行时依赖是 0**：

1. `backend/` 与插件根里对 `slam_probe` 的**字面引用：0 处**（只有 `pose_math.py` 的 docstring 提及）；
2. `backend/` 唯一的动态导入是 `process.py`，插的是 **`PROJECT_DIR.parent`**（插件包自身命名空间）
   与 `VENDOR_DIR`（`vendor/`），**与 `.slam_probe` 无关**；
3. **AST 全量导入审计**（`research/tools/import_audit.py`）：解析不到的只有
   `plugin`（宿主 SDK）、`mss` / `openvino`（可选依赖，代码用 `find_spec` 守卫）。

**⇒ 但结论仍然是"不能删"，理由换成两条：**

1. **未迁移的那半条离线流水线只在里面**（`build_topo_map.py`、`place_group.py`、
   `loop_verify.py`、`nav_map.py`、`obstacle_map_2d.py`、`world_model_build.py`、
   `pointcloud_build.py`、`map_preview.py`、`map_from_capture.py` 等 39 个 `.py`），
   **而 `.slam_probe/` 在 `.gitignore:19`** ⇒ 删掉等于丢掉**没有备份的代码**；
2. **全部录制素材与负结果证据在里面**：`runs/*`（29 个，含唯一完整的主样本 `20260920-233456`）、
   `stereo_seq/run1–7`、`rtabmap*`、`orbslam3_run*`、`depth_scale/`、`learned_probe/`、
   `loop_probe/`、`roomscale/`、`ov_env/`（OpenVINO 2026.4）、`deps/` + `ORB_SLAM3/` 构建树。

**清理时只能清坟场子目录，不能动 `recorder/` 与素材。**

> 附带：`.slam_probe/ov_env`（OpenVINO）是**开发期 PYTHONPATH 便利**
> （`research/recorder/scale_calib.py` 用 `PYTHONPATH=.slam_probe/ov_env`），
> 不是 `import .slam_probe`。但**部署面若依赖它就说明装错了** —— 用户机器上没有这个目录。

---

## 六、坟场清单（已证伪，别再投时间）

| 目录 / 文件 | 结论 |
|---|---|
| `ORB_SLAM3/`、`deps/`（Pangolin/glew/libepoxy）、`orbslam3_run*`、`run_mono_tum.sh`、`analyze_orbslam3.py` | **单目**：走—停全败，直接喂录像不可行。**双目**（2026-09-26）已重新评估：几何与尺度成立，但**因 GPLv3 + 30 Hz 做不到而不采用**，只做参照 → `Docs/ORB-SLAM3双目参照评估（2026-09-26）.md` |
| `depth_scale/*`（`vio_new`、`osc_odometry`、`pose_graph_opt`、`loop_closure_check`、`metric_scale`、`umeyama_check`、`batch_scale`、`overlap_new`…） | VO 闭合误差 12–14% ❌，不用它替代 OSC 里程 |
| `sfm_ba/`、`triangulation/triangulate_check.py` | 增量几何已止损：单帧墙 0.0098 m → 融合 0.071 m，瓶颈在**跨帧位姿** |
| `loop_probe/`、`learned_probe/` | 学习描述子替 BoW、全局指纹：**判别力不足** |
| `jump_detect.py`、`show_jumps.py`、`compare_runs.py`、`measure_reloc.py`、`plot_traj.py`、`show_traj.py` | 09-18 路线 B 的配套分析脚本 |
| `research/tools/seqslam_probe.py`、`orb_place_probe.py`、`world_fingerprint_probe.py`、`landmark_probe.py`、`location_distinguishability_probe.py` | 离线探针，**不接实时链路**（docstring 自己写明） |
| `runs/` 下 29 个早期 run（除 `20260920-233456` 外） | 只有 2–4 个文件的标定/自测残留 |

---

## 七、建议（按价值排序）

1. **P0：跨会话重定位** —— 给 `nav_loop.LoopCloser` 补**全局地点检索入口**
   （`_candidates()` 现在只按会话内路径距离取候选）。**保留** ORB+PnP 与 HMD 一致性门。
2. **绑定两条路线** —— `nav_map` 需要绑定 `visual_map_revision` + 同一尺度 + 同一坐标系，
   否则"去某地点"永远没有入口。
3. **把未迁移的那半条流水线纳入版本控制** —— 它现在**没有任何备份**（gitignore）。
   这是结构性风险，不是整洁问题。
4. **消歧改名**（低风险、收益高）：`.slam_probe/.../test_world_model.py` 与 `tests/test_world_model.py` 之一改名。
   **新增代码不要用 "world model" 这个词**（§4.3）。
