# YUI Unity 源码包

本目录是 YUI NPC v1.1 的 Unity/UdonSharp 可版本化源码，不是完整 Unity 工程。

- `Assets/NEKO/Npc/`：MIDI 路由、移动、同步、感知、遥测、触摸、名牌和可选 EyeCam。
- `Assets/NEKO/Editor/`：NPC、Rig、Animator、模型和 EyeCam 的 Editor 生成工具。
- 所有 `.meta` 均来自已通过火柴盒验收的 Unity 工程，迁移时必须一并保留。

此处刻意不收录场景、模型、动画、材质、RenderTexture、NavMesh 以及测试场专用
对象。目标世界应自行维护这些资产和 Inspector 引用，避免测试场内容污染正式地图。

协议常量仍以 `../Docs/Protocols/` 为唯一事实源；不得根据这些 C# 文件另行扩展
命令、能力或事件。
