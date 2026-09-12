# ARDY 源码迁移（历史记录）

> 历史阶段记录：本文保留当时的结果与限制，不代表当前部署或验收状态。当前文档入口见[ARDY文档索引](../README.md)。

当前启动方式以 [PORTABLE.md](../PORTABLE.md) 为准：双击接入包的 启动NPC动作.cmd，无需令牌。下文绝对路径仅用于定位原始备份，不是当前运行依赖。

候选源码来自 H:/AI/NVidia/integrations/yui_ardy。主仓库是后续插件代码的维护位置；此次不替换活动 N.E.K.O 宿主插件，不改正式 SAVE，不上传世界。

- runtime、主模型精简工具、环境去重注入、后台语义动作接口和共享 MIDI 已迁入。
- 独立服务在 integrations/motion_service，Unity 可选执行桥在 unity/Assets/NEKO/ArdyLab。不携带权重、角色、烘焙资源和测试场景；这些源码不代表正式场景已启用。
- 迁移初期使用本机独立环境与图缓存；当前已统一为可迁移接入包启动器，不再使用当时的路径或令牌设置。
- ARDY 默认关闭。后台动作模型独立负责填补与过渡，字幕不决定动作起止；连续填补与对话状态联动尚未验收，语音播放回执尚未接通。
- 单客户端完整桥已完成待机与一米步行，实测 4.994/5.007 秒；证据在 evidence。约 5 秒播放的是标称 4 秒素材，不是生成延迟或 P95。证据来自迁移前独立场景，迁移后未重新做真实客户端测试。
- [PLAN.md](../PLAN.md)、[PROTOCOL.md](../PROTOCOL.md) 包含历史记录，旧路径指向原实验目录。实验辅助工具、完整原始记录和模型环境继续保留在 H:/AI/NVidia/integrations；本仓库 evidence 保存最终通过的轻量记录。

验证命令：`.venv/Scripts/python.exe -B -m pytest tests integrations/motion_service -q -p no:cacheprovider --import-mode=importlib`。
独立服务目录已从宿主插件打包中排除。后续宿主部署和正式场景接线需要单独验收。
