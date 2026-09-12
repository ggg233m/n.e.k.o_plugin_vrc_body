import { Page, Card, Stack, Text, ActionButton, RefreshButton, StatusBadge, useEffect, useState } from "@neko/plugin-ui"
import type { PluginSurfaceProps } from "@neko/plugin-ui"
import { ErrorDiagnosis } from "./diagnostics"

type Value = boolean | number | string
type TokenUsage = { input_tokens: number | null; output_tokens: number | null; total_tokens: number | null }
type ModelCall = { number: number; started_at: string; status: string; output: string | null; truncated: boolean; error: string | null; latency_ms: number | null; format: string | null; source: string; disposition?: string; usage?: TokenUsage; usage_reported_requests?: number }
type IntentModelState = {
  enabled: boolean; configured: boolean; model: string; in_flight: boolean
  requests: number; http_requests: number; successes: number; failures: number; schema_fallbacks: number
  last_call?: ModelCall | null
  history?: ModelCall[]
  usage_totals?: TokenUsage
  usage_reported_requests?: number
  configuration_error?: string | null
  validation_retries?: number
}
type ActionProgress = {
  state: string; pending_intent: boolean
  preference?: { motivation: string; mood: string; expires_in_s: number; interests: { target_key: string; strength: number }[] } | null
  activity?: { kind: string; targets: string[]; regions: string[]; elapsed_s: number; index: number | null; count: number | null } | null
  proactive?: { enabled: boolean; reason: string; retry_in_s: number }
  chat?: { input_supported?: boolean; active: boolean; phase: string; player_slot: number; remaining_s: number | null }
  plan?: { status: string; origin: string; elapsed_s: number; completed_nodes: number; total_nodes: number; error?: string | null; active_nodes: { id: string; kind: string; target: string | null; player_slot: number | null; remaining_s: number | null }[] } | null
}
type State = {
  key_configured?: boolean
  key_reload_required?: boolean
  profile?: string
  settings?: Record<string, Value>
  applied?: Record<string, Value>
  status?: Record<string, unknown>
  intent_model?: IntentModelState
  action_progress?: ActionProgress
  fields?: { key: string; label: string; type: string; min?: number; max?: number; step?: number }[]
}

function visibleSaveError(error: unknown) {
  const detail = error instanceof Error ? error.message : String(error ?? "")
  for (const marker of ["设置校验失败：", "宿主配置服务读取失败", "宿主配置服务写入失败", "配置档已切换"]) {
    const start = detail.indexOf(marker)
    if (start >= 0) return detail.slice(start).split("\n", 1)[0].slice(0, 240)
  }
  return "保存失败，修改已保留。请刷新配置档后重试。"
}

const groups = [
  { title: "世界连接", description: "日志文件与日志目录二选一；均留空时自动寻找VRChat日志。", keys: ["midi_port", "claim_code", "log_path", "log_directory", "ardy.endpoint"] },
  { title: "模型连接", description: "配置独立意图模型；密钥写入宿主本机配置，不回显，也不保存在面板草稿缓存中。", keys: ["autonomy.intent_model.endpoint", "autonomy.intent_model.model", "autonomy.intent_model.api_key", "autonomy.intent_model.clear_api_key", "autonomy.intent_model.api_key_env", "autonomy.intent_model.persona_prompt", "autonomy.intent_model.timeout_s", "autonomy.intent_model.min_interval_s", "autonomy.intent_model.max_output_tokens"] },
  { title: "对白与字幕", description: "设置玩家对话入口和世界中的回复显示。", keys: ["player_chat.enabled", "chat_bridge.enabled", "chat_bridge.display_seconds", "chat_bridge.max_pages"] },
  { title: "自主陪伴", description: "管理自动连接、日常活动和对话陪伴。", keys: ["autonomy.auto_connect", "autonomy.enabled", "autonomy.chat_engagement.enabled", "autonomy.proactive_chat_enabled"] },
  { title: "动作决策", description: "为自主活动提供独立的意图判断。", keys: ["autonomy.intent_model.enabled", "autonomy.intent_model.chat_context.enabled", "ardy.enabled"] },
]
const hints: Record<string, string> = {
  "midi_port": "填写已创建的虚拟MIDI输出端口，例如NEKO_MIDI。",
  "claim_code": "与Unity世界中的控制码一致，范围0–16383。",
  "log_path": "填写完整日志文件路径。Unity测试使用当前用户 AppData/Local/Unity/Editor/Editor.log；留空可自动查找VRChat日志。",
  "log_directory": "只填写目录时自动选择其中最新VRChat日志；配置日志文件时请清空此项。",
  "ardy.endpoint": "默认 http://127.0.0.1:2346；只支持本机HTTP服务，无需令牌。",
  "autonomy.intent_model.endpoint": "完整的HTTPS Chat Completions地址，例如 https://example.com/v1/chat/completions。",
  "autonomy.intent_model.model": "填写API服务支持的模型标识。",
  "autonomy.intent_model.api_key": "留空保留原密钥，填写新值则替换；直接配置的密钥优先于环境变量。",
  "autonomy.intent_model.clear_api_key": "勾选并保存后清除直接配置的密钥；如需同时禁用备用密钥，请清空环境变量名称。",
  "autonomy.intent_model.api_key_env": "可选兼容旧配置；直接填写API Key后无需设置环境变量。",
  "autonomy.intent_model.persona_prompt": "描述角色性格与动作表达偏好，不需要编写骨骼或坐标。",
  "autonomy.intent_model.min_interval_s": "控制日常活动请求频率；聊天动作最短间隔仍为2秒。",
  "autonomy.proactive_chat_enabled": "空闲时靠近附近玩家并说一句开场白；全局至少间隔 2 分钟，同一玩家至少 5 分钟。",
  "player_chat.enabled": "玩家在世界聊天框发言时，请求角色回复。",
  "ardy.enabled": "启用可选 ARDY 服务；世界动作执行端尚未就绪时继续使用原动作系统。保存后需重载。",
  "chat_bridge.enabled": "将角色回复显示为世界中的 NPC 字幕。",
  "chat_bridge.display_seconds": "每页至少显示 10–25 秒；较长内容会自动延长。",
  "chat_bridge.max_pages": "单次回复最多展示 1–4 页字幕。",
  "autonomy.auto_connect": "插件启动或重载后，自动尝试连接世界。",
  "autonomy.enabled": "允许 NPC 自主安排活动；即时控制请使用上方按钮。",
  "autonomy.chat_engagement.enabled": "玩家发言后暂缓日常活动，优先留在发言者附近。",
  "autonomy.intent_model.enabled": "使用单独配置的模型选择动作，需要配置模型连接信息。",
  "autonomy.intent_model.chat_context.enabled": "读取近期聊天上下文，辅助独立模型选择动作。",
}
const labels: Record<string, string> = {
  disabled: "未启用", paused: "已暂停", explicit_control: "执行指定任务", chat_engaged: "对话陪伴中",
  executing: "自主活动中", waiting_control: "等待控制连接", ready: "准备就绪", not_initialized: "尚未启动",
  idle: "等待新回复", baseline: "等待新回复", empty: "暂无可显示内容", displayed: "本轮字幕已发送",
  displaying: "字幕发送中", waiting: "等待发送条件", queued_proactive_bus: "回复已排队", queued_recent_file: "回复已排队",
}
function statusLabel(value: unknown) {
  if (typeof value !== "string" || !value) return "暂无状态"
  return labels[value] || "其他状态（" + value + "）"
}

const dispositions: Record<string, string> = {
  generating: "生成中", validated: "通过校验，等待交付", invalid: "调用或校验失败", probe_only: "仅测试，不执行",
  queued: "已接收，等待活动边界", active: "已采纳", superseded: "已被新指令替代", expired: "已过期",
  paused: "已暂停，停止使用", chat_engaged: "转入对话陪伴，停止使用", stopped: "已停止使用",
  completed: "活动片段已完成", execution_failed: "活动执行失败",
}

const actionNames: Record<string, string> = {
  visit: "前往目标", navigate: "导航前往", explore: "探索区域", linger: "停留休息", socialize: "社交陪伴",
  perform: "表演动作", observe: "观察目标", local_roam: "附近活动", approach: "接近玩家", follow: "跟随玩家",
  orbit: "绕行目标", move_relative: "相对移动", turn_relative: "转身", look_at_target: "注视目标", look_at: "注视",
  act: "执行动作", set_expression: "切换表情", say: "显示字幕", wait: "停留等待", stop: "停止动作", chat_engagement: "对话陪伴",
}
function actionName(kind: string) { return actionNames[kind.replace(/^(intent_|preference_|rule_)/, "")] || kind }
function ActionProgressCard({ progress }: { progress?: ActionProgress }) {
  if (!progress) return <Card title="当前动作进度"><Text>动作进度尚未加载。</Text></Card>
  const { activity, plan, chat } = progress
  const live = plan?.status === "running" || plan?.status === "accepted"
  const phases: Record<string, string> = { waiting_input: "等待玩家输入", input_closed: "输入框关闭后的等待", proactive_approach: "前往搭话", preparing_opening: "准备开场", waiting_reply: "等待角色回复", reply_displaying: "等待字幕显示结束", final_reply_displaying: "等待最后一页字幕消失", post_reply_hold: "等待玩家续聊" }
  const states: Record<string, string> = { accepted: "等待调度", running: "正在执行", succeeded: "已完成", cancelled: "已取消", failed: "执行失败", unknown: "缺少完成证据" }
  const proactiveReasons: Record<string, string> = { disabled: "未启用", idle_wait: "等待空闲", cooldown: "冷却中", player_cooldown: "附近玩家仍在搭话冷却中", chat_busy: "正在聊天", no_eligible_player: "没有同区域、8 米内且已确认可达的玩家", approaching: "正在接近玩家" }
  return <Card title="当前动作进度">
    <div className="yui-row"><strong>{activity ? actionName(activity.kind) : progress.state === "explicit_control" ? "指定任务／恢复等待" : statusLabel(progress.state)}</strong>{activity?.index != null && activity.count != null ? <StatusBadge>第 {activity.index} / {activity.count} 个活动</StatusBadge> : null}</div>
    {activity ? <p className="yui-muted">目标：{activity.targets.length ? activity.targets.join("、") : "当前位置"}{activity.regions.length ? " · 区域：" + activity.regions.join("、") : ""} · 已进行 {activity.elapsed_s} 秒</p> : null}
    {activity ? <p className="yui-muted">决策来源：{activity.kind.startsWith("intent_") ? "模型活动片段" : activity.kind.startsWith("preference_") ? "模型偏好延续" : activity.kind === "chat_engagement" ? "对话陪伴" : "规则行为"}</p> : null}
    {progress.preference ? <div className="yui-metric"><strong>当前模型目标</strong><p>{progress.preference.motivation}</p><p className="yui-muted">关注：{progress.preference.interests.map(item => item.target_key).join("、") || "按当前活动偏好延续"} · 有效期剩余 {progress.preference.expires_in_s} 秒</p></div> : null}
    {chat?.active ? <p>{phases[chat.phase] || "正在陪伴玩家"} · 玩家槽位 {chat.player_slot}{chat.remaining_s != null ? " · 当前阶段等待上限剩余 " + chat.remaining_s + " 秒" : ""}</p> : null}
    {chat?.input_supported === false ? <p className="yui-muted">当前世界不支持输入中检测。更新世界聊天脚本后，打开输入框即可让 NPC 等候。</p> : null}
    {progress.proactive ? <p className="yui-muted">主动搭话：{proactiveReasons[progress.proactive.reason] || progress.proactive.reason}{progress.proactive.retry_in_s > 0 ? " · 等待 " + progress.proactive.retry_in_s + " 秒" : ""}</p> : null}
    {plan ? <div>
      <p>{live ? "当前计划" : "最近计划结果"}：{states[plan.status] || plan.status} · {plan.origin === "explicit" ? "指定任务" : "自主活动"} · {plan.elapsed_s} 秒</p>
      <p className="yui-muted">已成功节点 {plan.completed_nodes} / {plan.total_nodes}；计划可能包含分支、并行、重复和重试，此数值不是完成百分比。</p>
      {plan.active_nodes.map(node => <div className="yui-metric" key={node.id} style={{ marginTop: 8 }}>
        <strong>{actionName(node.kind)}</strong><p className="yui-muted">{node.target ? "目标：" + node.target : node.player_slot != null ? "玩家槽位：" + node.player_slot : "当前位置／当前对象"}</p>
        <span>{node.kind === "wait" ? "等待停留时间结束" + (node.remaining_s != null ? " · 剩余 " + node.remaining_s + " 秒" : "") : "等待指令返回或世界执行结果"}</span>
      </div>)}
      {plan.error || plan.status === "unknown" || plan.status === "failed" ? <ErrorDiagnosis code={plan.error || (plan.status === "unknown" ? "execution_unknown" : "execution_failed")} scope="execution" /> : null}
    </div> : <p className="yui-muted">{progress.state === "waiting_control" || progress.state === "not_initialized" ? "等待世界控制就绪，暂无执行计划。" : "当前没有行为图节点记录。"}</p>}
    {progress.pending_intent ? <p className="yui-muted">新动作建议已接收，等待当前活动边界后应用。</p> : null}
  </Card>
}
function tokenValue(value: number | null | undefined) { return value == null ? "未提供" : value.toLocaleString("zh-CN") }
function TokenSummary({ usage }: { usage?: TokenUsage }) {
  return <span>输入 {tokenValue(usage?.input_tokens)} · 输出 {tokenValue(usage?.output_tokens)} · 总量 {tokenValue(usage?.total_tokens)}</span>
}

function ModelOutput({ call }: { call: ModelCall }) {
  let output = call.output ?? ""
  // 模型输出只按文本呈现，不执行其中的 HTML。
  try { output = JSON.stringify(JSON.parse(output), null, 2) } catch (_) { /* 保留无效 JSON 原文。 */ }
  return <div>
    <p className="yui-muted">{new Date(call.started_at).toLocaleString("zh-CN", { hour12: false })} · {call.source === "probe" ? "手动测试" : "自主决策"}{call.latency_ms != null ? " · 耗时 " + call.latency_ms + " ms" : ""}{call.format ? " · " + call.format : ""}</p>
    <p>采纳情况：{dispositions[call.disposition || ""] || "暂无采纳记录"}</p>
    <p className="yui-muted">本次 Token：<TokenSummary usage={call.usage} />（仅累计本次已报告值）</p>
    {call.error ? <ErrorDiagnosis code={call.error} scope="model" /> : null}
    {output ? <pre aria-label="模型输出内容" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: 360, overflow: "auto", padding: 14, borderRadius: 10, background: "var(--bg)", border: "1px solid var(--border)", fontSize: 13, lineHeight: 1.65 }}>{output}</pre> : <p className="yui-muted">{call.status === "running" ? "正在等待模型返回内容。" : call.output === "" ? "模型返回了空内容。" : "本次调用未取得模型输出。"}</p>}
    {call.truncated ? <p className="yui-muted">输出较长，仅展示前 8,000 个字符。</p> : null}
  </div>
}

function ModelActivity({ model }: { model?: IntentModelState }) {
  if (!model) return <Card title="动作辅助模型"><Text>调用记录尚未加载。更新后请重载插件，再刷新面板。</Text></Card>
  const call = model.last_call
  const activity = model.in_flight ? "正在调用" : !model.enabled ? "未启用" : !model.configured ? "配置未完成" : "等待下一次调用"
  return <Card title="动作辅助模型">
    <div className="yui-row"><div><strong>{model.model || "尚未指定模型"}</strong><p className="yui-muted">{activity}</p></div><RefreshButton label="刷新调用记录" /></div>
    {model.enabled && !model.configured ? <ErrorDiagnosis code={model.configuration_error || "not_configured"} scope="model" /> : null}
    <div className="yui-metrics">
      {[ ["调用次数", model.requests], ["成功（通过校验）", model.successes], ["失败", model.failures], ["HTTP 请求次数", model.http_requests] ].map(([label, value]) => <div key={String(label)} className="yui-metric"><span className="yui-muted">{label}</span><strong>{value}</strong></div>)}
    </div>
    <p className="yui-muted">自插件启动或重载后累计，包含手动模型测试。兼容格式重试 {model.schema_fallbacks} 次，计入 HTTP 请求次数；缺少配置和忙碌时跳过的调用不计数。点击刷新读取最新快照。</p>
    <p className="yui-muted">输出纠正重试 {model.validation_retries || 0} 次；每次调用最多纠正一次，仍需通过目标与动作校验。</p>
    <div className="yui-metric"><strong>本次运行 Token 累计</strong><p><TokenSummary usage={model.usage_totals} /></p><p className="yui-muted">{model.usage_reported_requests || 0} / {model.http_requests} 次 HTTP 请求返回用量。仅累计接口报告的字段，包含已报告用量的重试和失败响应；未提供不等于零。重载后清零。</p></div>
    {call ? <div>
      <div className="yui-row"><strong>最近一次输出 · 第 {call.number} 次</strong><StatusBadge tone={call.status === "succeeded" ? "success" : "warning"}>{call.status === "succeeded" ? "通过校验" : call.status === "running" ? "生成中" : "调用失败"}</StatusBadge></div>
      <ModelOutput call={call} />
      <p className="yui-muted">这里展示模型返回的动作建议；通过校验不代表世界中的动作已执行完成。</p>
    </div> : <p className="yui-muted">尚无调用记录。自主决策或手动模型测试完成后，输出会显示在这里。</p>}
    <h3 style={{ fontSize: 15 }}>调用历史 · 最近 {(model.history || []).length} / 20 次</h3>
    <p className="yui-muted">按时间倒序保留在本次运行内，展开查看各次输出。重载后清空。</p>
    {(model.history || []).map(record => <details className="yui-help" key={record.number} style={{ borderTop: "1px solid var(--border)", padding: "10px 0" }}>
      <summary>第 {record.number} 次 · {record.status === "succeeded" ? "通过校验" : record.status === "running" ? "生成中" : "失败"} · {dispositions[record.disposition || ""] || "暂无采纳记录"}</summary>
      <ModelOutput call={record} />
    </details>)}
  </Card>
}

export default function Panel(props: PluginSurfaceProps<State>) {
  const { state, actions } = props
  const profileKey = JSON.stringify(state.profile ?? null)
  // 密钥只存在当前组件内存；切换配置档或关闭面板即清除。
  const [draft, setDraft] = useState<Record<string, Value>>({})
  useEffect(() => { setDraft({}) }, [profileKey])
  const [busy, setBusy] = props.useLocalState("saving", false)
  const [message, setMessage] = props.useLocalState("message:" + profileKey, "")
  const [autoRefresh, setAutoRefresh] = props.useLocalState("auto-refresh", true)
  const [refreshError, setRefreshError] = props.useLocalState("refresh-error", "")
  useEffect(() => {
    if (!autoRefresh || busy) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    // 串行刷新，慢请求不会堆积；隐藏面板不发请求，卸载时清理定时器。
    async function refresh() {
      if (stopped) return
      if (!document.hidden) {
        try {
          await props.api.refresh()
          if (!stopped) setRefreshError("")
        } catch (_) {
          if (!stopped) setRefreshError("自动刷新失败，保留上次快照，稍后重试。")
        }
      }
      if (!stopped) timer = setTimeout(refresh, 5000)
    }
    timer = setTimeout(refresh, 5000)
    return () => { stopped = true; clearTimeout(timer) }
  }, [autoRefresh, busy])
  const fields = state.fields || []
  // 比较实际值，改回原值不计为修改；只提交后端提供的字段。
  const changed = fields.filter(field => draft[field.key] !== undefined && draft[field.key] !== state.settings?.[field.key])
  const pending = fields.filter(field => field.type === "password" ? state.key_reload_required : state.settings?.[field.key] !== state.applied?.[field.key])
  const invalid = changed.some(field => field.type === "number" && (typeof draft[field.key] !== "number" || !Number.isFinite(draft[field.key]) || ((field.step ?? 1) === 1 && !Number.isInteger(draft[field.key])) || Number(draft[field.key]) < (field.min ?? -Infinity) || Number(draft[field.key]) > (field.max ?? Infinity)))
  const status = state.status || {}
  const ready = status["控制已就绪"] === true
  const midi = status["MIDI 已打开"] === true
  const manual = status["人工断开"] === true
  const estop = status["急停已锁存"] === true
  const connection = estop ? "已紧急停止" : ready ? "世界控制已就绪" : manual ? "已手动断开" : midi ? "等待世界握手" : "尚未连接世界"
  const connectionHint = estop ? "确认可以继续后，点击解除急停恢复控制；需要自主陪伴时再点击启动自主。" : ready ? "可以发送控制指令，并查看自主陪伴状态。" : manual ? "点击连接世界，即可重新建立控制连接。" : midi ? "MIDI 通道已打开，世界控制尚未就绪。请确认已进入支持 YUI 的世界。" : "进入支持 YUI 的世界后，点击连接世界。"
  function action(id: string) {
    const item = actions.find(a => a.id === id)
    return item ? <ActionButton key={id} action={item} refresh={true} /> : null
  }
  async function save() {
    if (busy || invalid || !changed.length) return
    setBusy(true)
    setMessage("")
    try {
      await props.api.call("yui_panel_save", { changes: Object.fromEntries(changed.map(field => [field.key, draft[field.key]])), expected_profile: state.profile ?? null })
      setDraft({})
      setMessage("已保存。重载配置后生效。")
      try { await props.api.refresh() } catch (_) { setMessage("已保存，但状态刷新失败。请刷新状态确认，再重载配置。") }
    } catch (error) {
      setMessage(visibleSaveError(error))
    } finally { setBusy(false) }
  }
  // 后端缺失时不能把空数据当作正常配置展示。
  if (!fields.length || !actions.some(a => a.id === "yui_panel_save")) {
    return <Page title="YUI · 世界陪伴" subtitle="连接世界，让陪伴自然发生">
      <Card title="面板尚未就绪">
        <p role="alert">未能读取插件设置。请点击插件详情右上角的“重载”，等待插件运行后重新读取面板。</p>
        <Text>如果仍无法加载，请检查插件日志。</Text>
        <RefreshButton label="重新读取面板" />
      </Card>
    </Page>
  }
  return <Page title="YUI · 世界陪伴" subtitle="连接世界，让陪伴自然发生">
    <style>{`
      .yui-row { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }
      .yui-muted { color:var(--muted); font-size:13px; line-height:1.65; margin:6px 0; }
      .yui-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:18px 0; }
      .yui-metric { background:var(--bg); border:1px solid var(--border); border-radius:12px; padding:14px; }
      .yui-metric strong { display:block; font-size:16px; margin-top:6px; overflow-wrap:anywhere; }
      .yui-actions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
      .yui-field { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:16px 0; border-bottom:1px solid var(--border); }
      .yui-field:last-child { border-bottom:0; }
      .yui-field label { font-weight:600; cursor:pointer; }
      .yui-field input[type=checkbox] { width:20px; height:20px; accent-color:var(--primary); cursor:pointer; flex-shrink:0; }
      .yui-text { flex:1; min-width:180px; max-width:440px; }
      .yui-text input { box-sizing:border-box; width:100%; padding:9px; border:1px solid var(--border); border-radius:8px; background:var(--bg); color:var(--text); }
      .yui-number { display:flex; align-items:center; gap:8px; flex-shrink:0; }
      .yui-number input { width:76px; min-height:38px; padding:6px 8px; border:1px solid var(--border); border-radius:8px; background:var(--bg); color:var(--text); }
      .yui-field input:focus-visible, .yui-help summary:focus-visible { outline:2px solid var(--primary); outline-offset:4px; }
      .yui-notice { border-left:3px solid var(--warning); padding:10px 14px; background:var(--bg); border-radius:6px; margin-top:14px; }
      .yui-save { padding:16px 0 4px; }
      .yui-help summary { cursor:pointer; font-weight:600; padding:4px 0; }
      @media(max-width:420px) { .yui-field { gap:12px; } .yui-metrics { grid-template-columns:1fr; } }
    `}</style>
    <Stack>
      <div className="yui-row"><label><input type="checkbox" checked={autoRefresh} onChange={event => setAutoRefresh(event.target.checked)} /> 每 5 秒自动刷新</label><span className="yui-muted">{autoRefresh ? "面板隐藏或保存设置时暂停刷新" : "自动刷新已关闭"}</span></div>
      {refreshError ? <p role="status">{refreshError}</p> : null}
      <Card title="连接与运行">
        <div className="yui-row">
          <div><StatusBadge tone={ready ? "success" : "warning"}>{connection}</StatusBadge><p className="yui-muted">{connectionHint}</p></div>
          <RefreshButton label="刷新状态" />
        </div>
        <div className="yui-metrics">
          <div className="yui-metric"><span className="yui-muted">控制通道</span><strong>{midi ? "MIDI 已打开" : "MIDI 未打开"}</strong></div>
          <div className="yui-metric"><span className="yui-muted">自主陪伴</span><strong>{statusLabel(status["自主状态"])}</strong></div>
          <div className="yui-metric"><span className="yui-muted">字幕发送</span><strong>{statusLabel(status["字幕状态"])}</strong><p className="yui-muted">{typeof status["最近字幕秒数"] === "number" && Number(status["最近字幕秒数"]) > 0 ? "最近一页 · " + status["最近字幕秒数"] + " 秒" : "暂无发送时长"}</p></div>
        </div>
        <div className="yui-actions">{estop && action("yui_clear_estop")}{action("yui_connect")}{action("yui_disconnect")}{action("yui_autonomy_start")}{action("yui_autonomy_pause")}</div>
        <p className="yui-muted">状态以最近一次刷新为准。暂停自主只停止自主活动；断开连接会停止 NPC 控制。</p>
      </Card>
      <ActionProgressCard progress={state.action_progress} />
      <ModelActivity model={state.intent_model} />
      <Card title="陪伴设置">
        <div className="yui-row"><Text>当前配置档：{state.profile || "基础配置"}</Text><StatusBadge tone={pending.length ? "warning" : "success"}>{pending.length ? pending.length + " 项等待重载" : "已保存配置与运行一致"}</StatusBadge></div>
        {pending.length ? <div className="yui-notice" role="status">已保存但尚未应用：{pending.map(field => field.label).join("、")}。</div> : null}
        {/* 宿主沙盒禁止原生表单提交，保存按钮直接调用接口。 */}
        <div role="group" aria-label="陪伴设置编辑">
          {groups.map(group => <section key={group.title} aria-label={group.title} style={{ marginTop: 24 }}>
            <h3 style={{ margin: 0, fontSize: 16 }}>{group.title}</h3>
            <p className="yui-muted">{group.description}</p>
            {group.keys.map(key => {
              const field = fields.find(item => item.key === key)
              if (!field) return null
              const value = draft[key] ?? state.settings?.[key]
              const modified = changed.some(item => item.key === key)
              const inputId = "yui-" + key
              return <div className="yui-field" key={key}>
                <div><label for={inputId}>{field.label}</label>{modified ? <span className="yui-muted"> · 未保存</span> : null}<p id={inputId + "-hint"} className="yui-muted">{hints[key]}</p></div>
                {field.type === "boolean" ? <input id={inputId} aria-describedby={inputId + "-hint"} type="checkbox" disabled={busy} checked={Boolean(value)} onChange={event => setDraft({ ...draft, [key]: event.target.checked })} />
                  : field.type === "text" || field.type === "password" ? <div className="yui-text"><input id={inputId} aria-describedby={inputId + "-hint"} type={field.type} autoComplete={field.type === "password" ? "new-password" : "off"} disabled={busy} value={String(value ?? "")} placeholder={field.type === "password" ? state.key_configured ? "已配置，留空保留" : "尚未直接配置" : ""} onInput={event => setDraft({ ...draft, [key]: event.target.value })} /></div>
                  : <div className="yui-number"><input id={inputId} aria-describedby={inputId + "-hint"} type="number" required disabled={busy} min={field.min} max={field.max} step={field.step ?? 1} value={value === undefined ? "" : String(value)} onInput={event => setDraft({ ...draft, [key]: event.target.value === "" ? "" : Number(event.target.value) })} /><span className="yui-muted">{key === "chat_bridge.max_pages" ? "页" : key.endsWith("_s") || key.endsWith("seconds") ? "秒" : ""}</span></div>}
              </div>
            })}
          </section>)}
          <div className="yui-save">
            <p className="yui-muted" aria-live="polite">{invalid ? "请输入范围内的整数后再保存。" : changed.length ? changed.length + " 项未保存。保存后需重载配置。" : "没有未保存的修改。"}</p>
            <div className="yui-actions"><button className="neko-button" data-tone="primary" type="button" onClick={() => { void save() }} disabled={busy || invalid || !changed.length}>{busy ? "保存中…" : "保存设置"}</button><button className="neko-button" data-tone="default" type="button" disabled={busy || !changed.length} onClick={() => { setDraft({}); setMessage("") }}>撤销修改</button></div>
            {message ? <p role="status" aria-live="polite">{message}</p> : null}
          </div>
        </div>
        <div className="yui-notice"><div className="yui-row"><div><strong>应用已保存的配置</strong><p className="yui-muted">重载会中断当前控制并重新初始化；启用自动连接后会尝试重连。未保存的修改不会应用。</p></div>{busy ? <Text>请等待保存完成</Text> : action("yui_reload_config")}</div></div>
      </Card>
      <Card title="使用提示">
        <details className="yui-help"><summary>世界聊天与字幕显示</summary><p className="yui-muted">在支持聊天输入的世界中，按 T 打开输入框，Enter 发送，Esc 关闭。字幕逐字效果和全文至少留存 6 秒由世界端实现，需要上传包含该功能的世界版本。</p></details>
        <details className="yui-help"><summary>模型配置与控制说明</summary><p className="yui-muted">独立意图模型用于动作决策。可在模型连接中输入密钥并保存；直接配置优先，环境变量作为备用。密钥保存在宿主本机配置，面板不回显。自主暂停和断开连接不会自动解除紧急停止（ESTOP）。</p></details>
      </Card>
    </Stack>
  </Page>
}
