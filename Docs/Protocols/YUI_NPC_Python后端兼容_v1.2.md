# YUI NPC Python 后端兼容 v1.2

Python 同时接受 `spec=1.1` 与 `spec=1.2`。v1.1 世界的工具面完全不变；只有 v1.2 世界同时发布对应 capability 和完整目录后，才暴露世界查询、语义导航和行为计划工具。

后台计划执行器由 MCP 与 N.E.K.O 插件共同复用。提交返回 `plan_id` 后在独立线程运行，stdio/MCP 请求线程不得被长路线占用。计划只保存当前 session 的最后 16 项诊断记录，不持久化。

世界目录、玩家槽位和 operation 日志是唯一事实源。任何 ACK 或 operation 终态缺失最终都为 `unknown`；禁止把 cancelled、failed 或 unknown 当成到达。
