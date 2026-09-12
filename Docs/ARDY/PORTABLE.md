# ARDY / NEKO 轻量接入包

ARDY源码和模型由使用者自行下载。本包只提供接入代码、49个补丁覆盖文件、CUDA/XPU配置、安装/检查工具、已构建的viser可视化依赖、NEKO插件与Unity集成；不包含ARDY完整工程、ARDY wheel或模型权重。

补丁源文件由插件仓库的 `integrations/ardy-patches/` 维护，打包时导出到本包 `ardy-patches/`；应用工具、清单和许可证一起发布。

## 准备官方ARDY

适用于Windows x64、支持AVX的CPU、64位Python 3.11。安装对应显卡驱动、Git和Visual Studio C++ Build Tools（原生扩展需要C++编译器）。NVIDIA CUDA与Intel XPU二选一；当前实际硬件验证使用Intel Arc A770。

自行下载[官方ARDY](https://github.com/nv-tlabs/ardy)，已验证基线为 `693f74d13b3d04a0a22ce127ee79c929dd89756b`：

```powershell
git clone https://github.com/nv-tlabs/ardy.git ardy
git -C ardy checkout 693f74d13b3d04a0a22ce127ee79c929dd89756b
```

可以放在接入包根目录的 `ardy`，也可以放在任意位置，安装时用 `-ArdySource` 指定。安装器逐文件检查基线，兼容Windows CRLF换行；全部通过后才备份并应用补丁。已有其他修改或版本不匹配会在写入前失败，不强制覆盖。备份位于下载源码的 `.ardy-integration-backups`，其中 `added-files.json` 记录本次新增文件。重复应用同版本补丁不会重复改写。

## 自行下载模型

按照官方ARDY说明取得访问权限并下载模型。本包启动时使用本地离线资源，不代替用户登录或下载模型。需要：

- [Core40](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40)，Core8为可选配置：[Core8](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon8)。
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)，含配置和完整权重分片。
- [LLM2Vec mntp](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp) 与 [mntp-supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised)，含适配器、配置和tokenizer。

默认目录为接入包下的 `checkpoints/ARDY-Core-RP-20FPS-Horizon40`、`text_encoders/meta-llama/Meta-Llama-3-8B-Instruct` 和 `text_encoders_local/McGill-NLP/上述两个LLM2Vec目录`。已有下载无需复制：将 `portable/config.example.json` 复制成根目录 `ardy.local.json`，修改 `checkpoints`、`base_model`、`text_encoders`、`ardy_source` 为实际位置。相对路径以接入包根目录解析，支持空格和中文。

## 安装和启动

```powershell
# 二选一；省略-ArdySource时使用配置中的源码位置，默认./ardy
powershell -NoProfile -ExecutionPolicy Bypass -File portable/install.ps1 -Device xpu -ArdySource D:/Projects/ardy
powershell -NoProfile -ExecutionPolicy Bypass -File portable/install.ps1 -Device cuda -ArdySource D:/Projects/ardy
```

也可双击 `安装环境.cmd`。安装器建立本机 `.venv`，应用补丁并从下载源码编译ARDY；CMake首次可能联网取得Eigen/pybind11。已存在环境默认拒绝覆盖，更新时使用 `-Reuse`。可用 `-Python` 指定Python 3.11，`-TorchWheel` 指定用户已有且匹配设备的官方2.14.0 wheel。安装和编译临时文件、pip缓存都放在项目 `.cache`，不复用旧电脑环境、GPU缓存或令牌。

PyTorch固定2.14.0，XPU默认官方xpu索引，CUDA默认cu130；可用 `-TorchIndex` 指定相应官方索引。依据：[PyTorch安装](https://pytorch.org/get-started/locally/)、[Windows XPU编译说明](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)。CUDA不会加载XPU专用图重放，使用标准推理，性能需在目标显卡实测；没有可用GPU会明确失败。

```powershell
# 接入包完整性检查（不扫描用户下载的全部模型）
.venv/Scripts/python.exe portable/verify.py
# 路径、权重分片、设备与原生模块检查
powershell -NoProfile -ExecutionPolicy Bypass -File start-ardy.ps1 -Check
# 真正加载并生成固定种子动作
powershell -NoProfile -ExecutionPolicy Bypass -File start-npc-motion.ps1 -Smoke
# NPC动作后台，端口2346
powershell -NoProfile -ExecutionPolicy Bypass -File start-npc-motion.ps1
# 可视化Demo，端口2333
powershell -NoProfile -ExecutionPolicy Bypass -File start-ardy.ps1
```

可双击 `启动ARDY.cmd` 或 `启动NPC动作.cmd`。Core40/Core8在XPU使用图重放；Core40Eager/Core8Eager关闭编译用于排查。不要同时加载Demo与NPC后台争用显存；启动器不关闭已有端口进程。

首次量化/编译可能耗时较长，初始化等待上限由 `startup_timeout_s` 配置（不超过3600秒），加载期间 `health.ready=false`。仅初始化等待可放宽，运行中的推理、世界握手和急停时限不变。LoRA重定位配置写入本机缓存，用户原模型不改写。检查通过、模型生成通过和世界实际执行通过分别记录。

## NEKO与Unity

无需令牌、环境变量或宿主命令行。双击 `启动NPC动作.cmd`，等待模型就绪；在NEKO插件面板启用ARDY并保存、重载插件。动作服务仅监听本机 `127.0.0.1`，宿主必须与ARDY在同一台电脑。旧令牌文件和环境变量不再参与正常启动。

按宿主插件流程导入 `neko-plugin`，配置动作后台 `http://127.0.0.1:2346`；mido、python-rtmidi需要安装在宿主环境，ARDY环境不代替宿主环境。新机建立 `NEKO_MIDI` 虚拟端口。主模型仍只有 `npc.perform` / `npc.stop`，环境去重注入不变；ARDY离线保留原动作系统，急停不会因重连解除。

`unity-integration/Assets` 是当前集成文件，不是完整家园工程。场景、人物、第三方资源和项目设置从原Unity工程/UVCS取得，使用Unity2022.3.22f1及项目锁定SDK。覆盖前备份目标改动，重新编译Udon并核对引用；不复制Library/Temp。先在ClientSim验证回退、连续动作和断线恢复。当前不启动VRChat，多客户端仍延后。

## 验证边界

另一台电脑配置未知，CUDA和XPU都保留；本机XPU验证不能代表NVIDIA实机性能。后空翻的模型支持、世界完成、严格单次翻转、腾空物理与稳定落地不等价。具体结果见交付报告。

[本机验证记录](evidence/backflip-portable-20260912/REPORT.md)列出了已通过检查、初始化耗时和仍未通过的动作质量条件。

本包只供本地迁移，没有自动上传工程或模型。`MANIFEST.json`记录接入文件SHA256，不包含使用者自行下载的资源。原ARDY、viser和其他依赖保留各自许可证。
