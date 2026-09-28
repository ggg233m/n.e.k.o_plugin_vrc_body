# SLAM 代码地图（2026-09-23 全仓扫描）

> 目的：这个项目里有**两套并行的 SLAM**，加上若干已证伪的探针，名字又高度重复。
> 本文只做梳理与定性，不改动任何代码。所有结论基于实扫（mtime / docstring / import 关系），不是推断。

---

## 一、一句话结论

1. **不是一堆乱代码，是两条路线 + 一片实验坟场**：
   - **路线 A（离线 2.5D）**：`.slam_probe/offline_probe/recorder/*`，靠 OBS 录像 + OSC + HMD 建经验图。已验收（P1 `degraded`）。
   - **路线 B（在线单目 SLAM）**：`.slam_probe/slam_core.py` + `relocalization.py`，被 `backend/live_mapping.py` **动态 import**。P2 在建。
   - **坟场**：`ORB_SLAM3/`、`deps/`、`depth_scale/`、`sfm_ba/`、`triangulation/`、`loop_probe/`、`learned_probe/` —— 全部是负结果，已证伪。（`ORB_SLAM3/` 的双目模式 09-26 另有参照评估，仍不采用，见 §六。）
   - **双目采集（09-26 新增，探针）**：`research/tools/openvr_mirror_probe.py`（合成器镜像纹理，GPU mip 缩放）、`research/tools/record_stereo_euroc.py`（录 EuRoC 格式序列 + OSC）、`research/tools/downscale_stereo_seq.py`（受控降分辨率）、`research/tools/analyze_orbslam3_stereo.py`（覆盖/丢追/尺度）、`research/tools/stereo_seq_ground_truth.py`（OSC+HMD 位置真值打分）。**不接实时链路。**
   - **RTAB-Map 双目（09-26，离线评估）**：`research/tools/prepare_rtabmap_euroc.py`（转 RTAB-Map 的 EuRoC 布局，z-up T_BS）、`research/tools/rtabmap_poses_to_euroc.py`（位姿转回左目光学系）。当前最优候选，结论见 `Docs/RTAB-Map双目评估（2026-09-26）.md`。
2. **`.slam_probe` 不能删**：`backend/` 有 4 个模块运行时去这个目录里 `import`（见 §五），删了在线建图直接崩。
3. **5 组重名文件**是当前最危险的混淆源，见 §四。

---

## 二、两条路线的分工

| | 路线 A：离线 2.5D 经验图 | 路线 B：在线单目 SLAM |
|---|---|---|
| 入口 | `recorder/map_from_capture.py`（P1 一键） | `backend/live_mapping.py` |
| 核心 | 录像抽帧 + OSC 里程 + HMD yaw | `slam_core.py`（MonocularSlam）+ `relocalization.py` |
| 输出 | `topo_map / place_groups / loop_verify / pose_graph / nav_map / world_model` | 视觉地图段 + `relocalization_map.json/.npz` |
| 尺度 | OSC 路程当量反解 `cam_h`（`scale_calib.py`） | 视觉里程 + `metric_scale_calibrator` 待 OSC 配对 |
| 状态 | ✅ 2.5D 主验收 pass，因缺 `action_timeline` 标 degraded | 🟡 `preview_ready`，`scale_validated=false` ⇒ 只能预览不能执行 |
| 生产接线 | `backend/nav_plan.py`（逐字复制 `recorder/nav_map.py` 的规划器） | `backend/nav_map_builder / nav_target / route_executor` |

**两条路线目前没有合并**：A 出 2.5D 导航图，B 出视觉坐标系下的定位；`nav_target` 现在因为两者 revision/尺度/坐标系没绑定而**安全拒绝**（`navigation_map_frame_mismatch`）——这是当前 P2 最大的一个洞。

---

## 三、活跃链（按流水线层次）

| 层 | 文件 | 职责 |
|---|---|---|
| 采集 | `recorder/record_stage1.py`、`obs_ctl.py`、`record_snap/axis/calib.py` | 受控录制（v2.1 手动 HMD 采集） |
| 采集自检 | `research/tools/check_recording_channels.py`、`recorder/verify_run.py` | 录完先验通道，全 PASS 才进后处理 |
| 时间基准 | `research/tools/estimate_video_offset.py` → `recorder/time_align_check.py`；`research/tools/offline_timebase_harness.py`；`backend/time_alignment.py` | 视频↔OSC↔HMD 统一单调钟 |
| 深度/尺度 | `recorder/stage3_horizon.py`（地平线/地板归一）、`scale_calib.py`（cam_h 自动标定）、`openvino_depth_probe.py`；`backend/realtime_depth.py`、`backend/metric_scale_calibrator.py` | 单目深度 + 米制尺度 |
| 拓扑/回环 | `recorder/build_topo_map.py` → `place_group.py` → `loop_verify.py` | 观测簇 → 地点组 → 回环几何验证（四级确认协议） |
| 位姿图 | **`recorder/pose_graph.py`**（2D 米制）← 活跃 | 航位推算 + 闭环修正 |
| 航位推算数学 | **`backend/pose_math.py`**（唯一实现，纯 `math`，无 numpy） | 本地矢量位移按朝向旋进世界；在线 `backend/online_pose.py` 与离线 `recorder/pose_graph.py` **共用同一份** |
| 导航图/规划 | `recorder/nav_map.py`（A*/Dijkstra 原实现）→ `backend/nav_plan.py`（无 numpy 副本） | 2.5D 可导航图 |
| 避障 | `recorder/local_avoid.py`（三态）+ `tools/local_avoid_replay.py`（冻结回放验证） | passable/unknown/stop |
| 障碍层 | `recorder/obstacle_map_2d.py`（离线产物） vs `backend/obstacle_map_2d.py`（自建坐标系） | 见 §四 |
| 世界模型/预览 | `recorder/world_model_build.py`、`map_preview.py`、`pointcloud_build/splat.py`、`check_pointcloud_orientation.py` | 收敛成单一世界模型 + HTML 预览 |
| 重定位 | `.slam_probe/relocalization.py`（被 backend 加载）；实验区 `recorder/relocalize.py`、`relocalize_geom.py`；回放 `tools/relocalize_video.py`、`replay_navtarget_video.py` | 只读视觉重定位 |
| 执行/回放 | `backend/route_executor.py`、`backend/offline_route_replay.py`（实现）+ `tools/offline_route_replay.py`（CLI） | 安全状态机，缺证据就暂停 |
| 门禁 | `research/tools/regression_place_identity.py`（15 断言）、`research/tools/check_2d_map_acceptance.py` | 回归与验收 |

`map_from_capture.py` 用 **subprocess** 串起这一串（不是 import）：`estimate_video_offset → scale_calib → build_topo_map → place_group → loop_verify → pose_graph → nav_map → obstacle_map_2d → check_pointcloud_orientation → map_preview → world_model_build`。

---

## 四、⚠️ 5 组重名陷阱

| 同名 | 谁活跃 | 另一个是什么 |
|---|---|---|
| `pose_graph.py` | `recorder/pose_graph.py`（2D 米制，P1 链上） | `.slam_probe/pose_graph.py`＝SE(3) 早期图优化（09-18，路线 B 早期） |
| `obstacle_map_2d.py` | 两者都在用 | `recorder/`＝离线深度出证据图；`backend/`＝自建坐标系在线三态障碍层 |
| `offline_route_replay.py` | `research/tools/` 是 CLI 入口 | `backend/` 是实现（纯函数 `_contract_checks` / `_safety_checks`） |
| `probe.py` | 都不是活跃链 | `loop_probe/` 与 `learned_probe/` 各一份，均为负结果探针 |
| `test_world_model.py` | `tests/`（13 条，全过） | `recorder/` 里还有一份旧的单测 |

**还有一个语义重名**：`backend/world_model.py` 是 **W1 世界身份**（"当前在哪个世界"+启停骨架），跟建图的世界模型（`recorder/world_model_build.py` 产物 `world_model.json`）**完全不是一回事**。

### 四之二、⚠️ 更危险的一类：**同一算法存两份拷贝**

重名至少会当场报错；**同一公式存两份**不会报错，只会**悄悄漂移**。这一类已经真实
发生过一次：

> `backend/online_pose.py`（在线）与 `recorder/pose_graph.py`（离线）各写了一份
> "本地矢量位移按朝向旋进世界"。在线那份一直是对的；离线那份退化成
> **标量路程沿 yaw 前进**（`x += ds*sin(yaw)`），等价于把横移当成前进 ——
> 该 run 终点差 **1.5496 m**，且 596/895 步都带一点横移，不是"横移很少见"。

2026-09-24 已收敛：公式只留一份在 `backend/pose_math.py`，两侧都调它。
防复发靠**门禁**而不是自觉：

```
python tools/pose_math_equivalence.py --selftest
```

六道闸门：共享模块纯净 / yaw 符号常量被钉住 / 同符号下两侧数值一致 /
家族文件里不再出现第二份三角函数 / 在线与离线输出与重构前**逐位一致** /
以及"闸门本身不是空转"的自检（喂历史 bug 会红、注入第二份公式会红）。

**规矩**：航位推算家族（`backend/online_pose.py`、`recorder/pose_graph.py`、
`recorder/run_motion.py`、`tools/slam_metric_retest.py`、
`depth_scale/osc_odometry.py`）里**不允许出现任何 `cos/sin/tan` 调用** ——
需要旋转就调 `backend/pose_math.py`。

**已知有意保留的其它旋转副本**（形状不同，不是航位推算，勿照抄）：
`recorder/scale_calib.py::_R_wc`（同一旋转的 3×3 矩阵，相机→世界三维落点）、
`pointcloud_build.py` / `pointcloud_splat.py` 里的 GL/JS 着色器旋转（渲染坐标变换）。

**仍未决**：在线侧的 yaw 符号（`ONLINE_YAW_SIGN = +1.0`）在本仓库内
**既无标定证据、也无测试钉住** —— `POST /worldmodel/navroute/feedback` 的调用方
不在本仓库内，符号约定由外部决定。离线侧的 `-1` 有画面证据
（`yaw_sign_check.json`）。**动在线符号前先补在线侧的证据。**

---

## 五、🔴 为什么 `.slam_probe` 不能删

`backend/` 运行时依赖它（实扫 import 关系）：

- `backend/live_mapping.py:22-31`：`sys.path.insert(0, ".slam_probe")` → `importlib.import_module("slam_core")` 取 `MonocularSlam/SlamConfig`；再 `import_module("relocalization")` 取 `export_relocalization_map`
- `backend/nav_plan.py`：`recorder/nav_map.py` 的 `plan`/`_format_plan` **逐字复制**（为在无 numpy、无 recorder 依赖的环境下也能规划）
- `backend/realtime_depth.py`、`backend/offline_route_replay.py`、`backend/relocalization_observer.py`：同样按路径加载 `.slam_probe` 里的模块

⇒ 清理时只能清理**坟场子目录**，不能动 `slam_core.py` / `relocalization.py` / `recorder/`。

---

## 六、坟场清单（已证伪，别再投时间）

| 目录/文件 | 结论 |
|---|---|
| `ORB_SLAM3/`、`deps/`（Pangolin/glew/libepoxy）、`orbslam3_run*`、`run_mono_tum.sh`、`analyze_orbslam3.py`、`orb_slam3_result.json` | **单目**：走—停全败，直接喂录像不可行。**双目**（2026-09-26）已重新评估：几何与尺度成立，但因 GPLv3 和 30 Hz 做不到而不采用，只做参照 → `Docs/ORB-SLAM3双目参照评估（2026-09-26）.md` |
| `depth_scale/*`（`vio_new`、`vio_traj`、`osc_odometry`、`pose_graph_opt`、`loop_closure_check`、`metric_scale`、`umeyama_check`、`batch_scale`、`check_consistency`、`scale_drift_check`、`overlap_new`、`diag_motion`） | VO 闭合误差 12–14% ❌，不用它替代 OSC 里程 |
| `sfm_ba/`、`triangulation/triangulate_check.py` | 增量几何已止损：单帧墙 0.0098 m → 融合 0.071 m，瓶颈在跨帧位姿 |
| `loop_probe/`、`learned_probe/` | 学习描述子替 BoW、全局指纹：判别力不足 |
| `jump_detect.py`、`show_jumps.py`、`compare_runs.py`、`measure_reloc.py`、`plot_traj.py`、`show_traj.py` | 09-18 路线 B 的配套分析脚本 |
| `research/tools/seqslam_probe.py`、`orb_place_probe.py`、`world_fingerprint_probe.py`、`landmark_probe.py`、`location_distinguishability_probe.py` | 离线探针，**不接实时链路**（docstring 自己写明） |
| `runs/` 下 29 个早期 run（除 `20260920-233456` 外） | 只有 2–4 个文件的标定/自测残留；`20260920-233456` 有 67 个产物，是唯一完整素材 |

---

## 七、建议（未执行，等确认）

1. **先立规矩再动文件**：给坟场目录加 `Docs/` 级别的归档标记或统一挪到 `.slam_probe/_archive/`，**不要直接删**（有运行时 import 风险）。
2. **消歧改名**（低风险、收益高）：`recorder/pose_graph.py` → `pose_graph_2d.py`；`backend/obstacle_map_2d.py` → `obstacle_layer_2d.py`。注意改后要同步改 `map_from_capture.py` 里的 subprocess 调用与所有产物文件名。
3. **两条路线的合并点**才是该花时间的地方：`nav_map` 需要绑定 `visual_map_revision` + `meters_per_unit` + 同一坐标系，否则 `nav_target` 会一直拒绝（`navigation_map_frame_mismatch`）。
