# YUI NPC 宿主自主行为 v0.5

## 范围

0.5.4 实现无视觉自主闭环、独立文本意图模型、只读聊天记忆上下文、主 LLM 完整回答到 NPC 头顶 `TEXT_UTF8` 的默认开启投影，以及玩家按 `T` 唤起、跟随视野、闲时完全隐藏的世界输入 UI。视觉、Spout、画面队列和视觉模型入口均不在本版本中；
`AutonomyStimulusProvider` 仅保留异步接口，运行时固定使用不采图的空实现。

## 控制规则

1. 自主计划来源为 `autonomy`，LLM 工具计划来源为 `explicit`。
2. 显式计划无需 `replace_active=true` 即可抢占自主移动计划；自主计划不能抢占任何
   显式移动计划。
3. 普通显式工具必须等 plan/operation 终态，再延迟 `resume_delay_s` 恢复自主。
4. `npc.stop`、`npc.estop`、watchdog、人工暂停和人工断开均保持暂停。
5. `accepted` 只表示受理。自主循环必须等待
   `npc.operation_completed/failed/cancelled` 投影出的终态。

## 决策来源

- 驻足：在 `dwell_range_s` 内选择时长。
- 区域探索：只选择 `explorable=true` 的 Region，持续时间来自 `explore_range_s`。
- 跨区域游览：从最近 Anchor 出发，对发布的 `route_edge` 做广度优先路径规划，逐个
  提交语义 Anchor；NavMesh 和 operation lifecycle 最终裁决是否成功。
- 社交：触摸、挥手、有效凝视和靠近事件生成接近、注视与可用短动作；同一
  `player_slot` 按 `social_cooldown_s` 限频。
- 失败：目标进入 15、30、60、120、240、300 秒封顶的指数退避。
- 自然活动：动作模型可选择观察语义目标、轻微转身、0.5～1.5 米附近闲逛或小环路；
  只有明确的活动目标才进行语义导航，不为区域多样性强制跨区。
- 路线：实际语义边序列使用方向无关的规范签名，成功后冷却 10 分钟；强度不低于
  0.7 的有效兴趣可消费一次覆盖。每个生活片段最多跨区一次，普通跨区至少间隔 180 秒。

自主候选从不生成绝对坐标。移动和驻足不再追逐固定比例，规则循环只防止永久站立。
独立 OpenAI 兼容 API 在启动、完整聊天轮次更新、片段边界、重要世界事件和 3～6 分钟保底
周期生成 2～4 项生活活动。插件每秒只读当前角色 `recent.json` 修订，最多注入最近
6 个完整用户到主 LLM 轮次，并把它明确标记为不可信上下文；不会订阅或扩展
`conversations` 总线，也不会回写聊天或触发 TTS。API 不响应或校验失败
不影响规则循环。YUI 插件仅对验证通过的 `player.chat_submit` 发送
`ai_behavior="respond"`；自主、社交和失败事件不会触发宿主回复或 TTS。

## 验收入口

- `yui_autonomy_start`：启动或人工恢复。
- `yui_autonomy_pause`：持久暂停并停止当前自主计划。
- `yui_autonomy_status`：查看当前状态、访问/路线历史、失败黑名单和脱敏记忆状态。
- `yui_autonomy_intent_probe`：脱敏验证记忆路径、解析结果、独立 API 与结构化响应，不显示正文、不应用活动。
- `yui_player_chat_status`：查看按键唤起式输入 UI 就绪、提交计数和脱敏错误，不显示正文。

以上入口均为宿主专用，不注册给 LLM。
