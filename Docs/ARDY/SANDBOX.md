# Unity 动作回放沙盒

打开 Assets/NEKO/ArdyLab/ArdyWorldBridgeLab.unity，使用菜单 NEKO → ARDY → 动作回放沙盒。选择主仓库 integrations/acceptance/sandbox-fixtures/walk.json 或 idle.json；无需启动模型服务或 VRChat。

默认使用固定绑定基准；勾选“重现旧的姿势叠加错误”可对照之前的扭曲问题。停止按钮或关闭窗口会清理临时角色与地板。仅允许在 ArdyLab 路径的隔离场景中编辑器回放，不能在正式 SAVE 内运行。

这是保存素材的编辑器 C# 与 Physics 回放，不代表新一轮实时生成、ClientSim/Udon VM 或真实客户端通过。仍需观察鞋底接触、摆动脚和连续步态，不能仅凭回执和脚踝 IK 数值宣称无滑步。
