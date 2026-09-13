# N.E.K.O 一键安装包

在目标电脑的 N.E.K.O 插件管理面板选择「导入」，选择发行的 `.neko-plugin` 文件，确认安装，然后启动插件。
不需要手动解压或运行 pip。包内带人物检测模型和第三方 Python 依赖，不包含 API 密钥、本机设置或世界记忆。

本次二进制依赖针对 **Windows 10/11 x64、CPython 3.11** 构建；目标 N.E.K.O 宿主须使用 Python 3.11 和兼容 SDK（见 plugin.toml）。其他 Python 版本需要重新构建 vendor，不能直接复用本包中的 .pyd。

目标电脑仍需安装并配置 VRChat、SteamVR 和 AnyaDance 驱动，并启用 VRChat OSC。身体输出默认禁用，自主移动仍需面板手动授权。
允许与 `yui_npc_controller` 共存。不应同时启动另一份 AnyaDance 独立后端占用相同端口。

视觉使用自带模型，包内提供 ONNX Runtime CPU 推理，无需 CUDA；宿主已有兼容 OpenVINO/CUDA 运行库时仍使用项目原来的自动选择逻辑。不同设备上的捕获权限、多屏布局及推理速度需实际检查。
宿主主多模态 LLM 可继续处理语义识别；独立 VLM 接口及密钥需在目标电脑自行配置。
本仓库没有 `motions/` 预制素材，不包含额外 VMD/NYA 动作；程序化动作仍可使用。

包内 `DEPENDENCIES.json` 列出实际依赖版本；各依赖的许可保留在 `vendor/*.dist-info/`。包外 `.sha256` 用于校验下载完整性。
发行验证包含宿主包检查器、隔离目录安装、干净 Python 依赖导入、真实模型推理和后端 HTTP/UI 启动；不替代目标设备的 VRChat 实机联调。
