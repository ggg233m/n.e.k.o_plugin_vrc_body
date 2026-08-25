import {
  Alert,
  Button,
  ButtonGroup,
  Card,
  Divider,
  Field,
  Grid,
  Input,
  JsonView,
  KeyValue,
  LogViewer,
  NumberInput,
  Page,
  Progress,
  RefreshButton,
  Select,
  Slider,
  Stack,
  StatCard,
  StatusBadge,
  Switch,
  Text,
  Toolbar,
  ToolbarGroup,
  useEffect,
  useToast,
} from "@neko/plugin-ui"
import type { HostedAction, PluginSurfaceProps } from "@neko/plugin-ui"

type ClipSummary = {
  name?: string
  duration_s?: number
  frame_count?: number
  is_pose?: boolean
  indexed?: boolean
  file_size_bytes?: number
  metadata?: {
    label?: string
    description?: string
    intents?: string[]
    source_kind?: string
    source_name?: string
  }
}

type DebugState = {
  version?: string
  updated_at_unix?: number
  body?: Record<string, any>
  awareness?: Record<string, any>
  vrchat_osc?: Record<string, any>
  driver_log?: Record<string, any>
  host_vmc?: Record<string, any>
  world?: Record<string, any>
  autonomy?: Record<string, any>
  clips?: {
    clips?: ClipSummary[]
    invalid_clips?: Array<{ name?: string; error?: string }>
    directory?: string
    indexed_count?: number
    unindexed_count?: number
    cache?: Record<string, any>
    motion_catalog?: {
      entries?: Array<Record<string, any>>
      errors?: string[]
      missing_clips?: string[]
    }
  }
  config?: Record<string, any>
  ui_events?: Array<Record<string, any>>
}

const sideOptions = [
  { value: "left", label: "左手" },
  { value: "right", label: "右手" },
  { value: "both", label: "双手" },
]

const palmOptions = [
  { value: "neutral", label: "自然" },
  { value: "forward", label: "掌心向前" },
  { value: "down", label: "掌心向下" },
  { value: "inward", label: "掌心向内" },
]

const handOptions = [
  { value: "open", label: "张开" },
  { value: "fist", label: "握拳" },
  { value: "grip", label: "抓握" },
  { value: "point", label: "指向" },
]

const gestureOptions = [
  { value: "wave", label: "挥手" },
  { value: "nod", label: "点头" },
  { value: "bow", label: "鞠躬" },
  { value: "shake_head", label: "摇头" },
  { value: "shrug", label: "耸肩" },
  { value: "think", label: "思考" },
  { value: "point", label: "指向" },
  { value: "beckon", label: "招手靠近" },
  { value: "clap", label: "鼓掌" },
  { value: "surprise", label: "惊讶" },
  { value: "comfort", label: "安慰" },
  { value: "sigh", label: "叹气" },
]

const expressionOptions = [
  { value: "greet", label: "问候" },
  { value: "agree", label: "同意" },
  { value: "disagree", label: "否定" },
  { value: "explain", label: "解释" },
  { value: "present", label: "展示" },
  { value: "think", label: "思考" },
  { value: "celebrate", label: "庆祝" },
  { value: "question", label: "疑问" },
  { value: "emphasize", label: "强调" },
  { value: "beckon", label: "招手靠近" },
  { value: "comfort", label: "安慰" },
  { value: "apologize", label: "道歉" },
  { value: "surprise", label: "惊讶" },
  { value: "shrug", label: "无奈" },
  { value: "clap", label: "鼓掌" },
  { value: "laugh", label: "发笑" },
  { value: "sigh", label: "叹气" },
]

function fixed(value: any, digits = 1) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—"
}

function timestamp(value: any) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? new Date(numeric * 1000).toLocaleTimeString() : "—"
}

export default function AnyaDanceDebugPanel(props: PluginSurfaceProps<DebugState>) {
  const state = props.state || {}
  const body = state.body || {}
  const awareness = state.awareness || {}
  const osc = state.vrchat_osc || {}
  const driverLog = state.driver_log || {}
  const idleRelay = body.idle_relay || awareness.idle_relay || {}
  const hostVmc = state.host_vmc || {}
  const autonomy = state.autonomy || {}
  const navigation = autonomy.navigation || {}
  // 卡墙判据的两个数据源：OSC 快照里的实时读数，以及导航器 tick 里的最后一次
  // 采样。后者在解除授权后就冻住了，所以优先用前者。
  const stall = navigation.stall || {}
  const oscMotion = osc.motion || stall.last_motion || { available: false, reason: "osc_unavailable" }
  const metrics = body.metrics || {}
  const udp = body.udp || {}
  const currentAction = body.current_action || null
  const behavior = body.behavior || awareness.behavior || {}
  const clips = Array.isArray(state.clips?.clips) ? state.clips?.clips || [] : []
  const invalidClips = Array.isArray(state.clips?.invalid_clips) ? state.clips?.invalid_clips || [] : []
  const debugAction = (props.actions || []).find((item) => item.id === "debug_command") as HostedAction | undefined
  const toast = useToast()

  const [autoRefresh, setAutoRefresh] = props.useLocalState("autoRefresh", true)
  const [busy, setBusy] = props.useLocalState("busy", false)
  const [clientLog, setClientLog] = props.useLocalState<string[]>("clientLog", [])

  const [armSide, setArmSide] = props.useLocalState("armSide", "right")
  const [elevation, setElevation] = props.useLocalState("elevation", 90)
  const [azimuth, setAzimuth] = props.useLocalState("azimuth", 0)
  const [reach, setReach] = props.useLocalState("reach", 0.9)
  const [palm, setPalm] = props.useLocalState("palm", "neutral")
  const [wristPitch, setWristPitch] = props.useLocalState("wristPitch", 0)
  const [wristYaw, setWristYaw] = props.useLocalState("wristYaw", 0)
  const [wristRoll, setWristRoll] = props.useLocalState("wristRoll", 0)
  const [armDuration, setArmDuration] = props.useLocalState("armDuration", 600)

  const [handSide, setHandSide] = props.useLocalState("handSide", "right")
  const [handPose, setHandPose] = props.useLocalState("handPose", "open")
  const [handStrength, setHandStrength] = props.useLocalState("handStrength", 1)

  const [gestureName, setGestureName] = props.useLocalState("gestureName", "wave")
  const [gestureSide, setGestureSide] = props.useLocalState("gestureSide", "right")
  const [gestureIntensity, setGestureIntensity] = props.useLocalState("gestureIntensity", 0.65)
  const [expressionIntent, setExpressionIntent] = props.useLocalState("expressionIntent", "explain")
  const [expressionSide, setExpressionSide] = props.useLocalState("expressionSide", "auto")
  const [expressionIntensity, setExpressionIntensity] = props.useLocalState("expressionIntensity", 0.45)

  const [reachSide, setReachSide] = props.useLocalState("reachSide", "right")
  const [reachHeight, setReachHeight] = props.useLocalState("reachHeight", "chest")
  const [reachDirection, setReachDirection] = props.useLocalState("reachDirection", "forward")
  const [reachDistance, setReachDistance] = props.useLocalState("reachDistance", 0.35)
  const [reachDuration, setReachDuration] = props.useLocalState("reachDuration", 700)

  const [clipName, setClipName] = props.useLocalState("clipName", "")
  const [clipSpeed, setClipSpeed] = props.useLocalState("clipSpeed", 1)
  const [clipLoops, setClipLoops] = props.useLocalState("clipLoops", 1)
  const [clipTransition, setClipTransition] = props.useLocalState("clipTransition", 400)
  const [clipRestore, setClipRestore] = props.useLocalState("clipRestore", false)

  const [parameterName, setParameterName] = props.useLocalState("parameterName", "NEKO_Action")
  const [parameterType, setParameterType] = props.useLocalState("parameterType", "int")
  const [parameterValue, setParameterValue] = props.useLocalState("parameterValue", "1")
  const [inputAction, setInputAction] = props.useLocalState("inputAction", "grab")
  const [inputSide, setInputSide] = props.useLocalState("inputSide", "right")
  const [inputHold, setInputHold] = props.useLocalState("inputHold", 100)

  useEffect(() => {
    if (!autoRefresh) return
    const timer = window.setInterval(() => {
      props.api.refresh().catch(() => undefined)
    }, 1000)
    return () => window.clearInterval(timer)
  }, [autoRefresh])

  const appendLog = (line: string) => {
    setClientLog((previous) => [...previous.slice(-39), line])
  }

  const run = async (command: string, args: Record<string, any> = {}) => {
    if (!debugAction) {
      toast.error("调试 action 尚未注册，请重启插件。")
      return null
    }
    setBusy(true)
    try {
      const response = await props.api.call(debugAction.id, { command, arguments: args })
      const serialized = JSON.stringify(response)
      appendLog(`${new Date().toLocaleTimeString()}  ${command}  ${serialized.slice(0, 1400)}`)
      await props.api.refresh()
      toast.success(`${command} 已返回`)
      return response
    } catch (error) {
      const message = error && (error as any).message ? (error as any).message : String(error)
      appendLog(`${new Date().toLocaleTimeString()}  ${command}  ERROR ${message}`)
      toast.error(message)
      return null
    } finally {
      setBusy(false)
    }
  }

  const parameterPayload = () => {
    if (parameterType === "bool") return String(parameterValue).trim().toLowerCase() === "true"
    if (parameterType === "int") return Math.trunc(Number(parameterValue))
    return Number(parameterValue)
  }

  const effectiveClip = clipName || clips[0]?.name || ""
  const clipOptions = clips.map((clip) => ({
    value: clip.name || "",
    label: clip.indexed === false
      ? `${clip.metadata?.label || clip.name || "未命名"} · 未索引 · ${fixed((clip.file_size_bytes || 0) / 1048576, 1)} MiB`
      : `${clip.metadata?.label || clip.name || "未命名"} · ${fixed(clip.duration_s, 2)}s · ${clip.frame_count || 0}帧`,
  }))
  const backendLines = (state.ui_events || []).map((event) => {
    const status = event.accepted ? "OK" : "REJECT"
    const reason = event.reason ? ` · ${event.reason}` : ""
    return `${timestamp(event.at_unix)}  ${status}  ${event.command || "unknown"}${reason}`
  })
  const logText = [...backendLines, ...clientLog].join("\n") || "暂无调试命令。"

  return (
    <Page title="AnyaDance 身体调试台" subtitle={`插件 ${state.version || "—"} · ${state.config?.anyadance_target || "127.0.0.1:39570"}`}>
      <Toolbar>
        <ToolbarGroup>
          <StatusBadge tone={body.output_enabled ? "success" : "warning"}>
            {body.output_enabled ? "身体输出中" : "身体输出关闭"}
          </StatusBadge>
          <StatusBadge tone={body.safety_state === "normal" ? "success" : "danger"}>
            {body.safety_state || "unknown"}
          </StatusBadge>
          <StatusBadge tone={osc.connection === "detected" ? "success" : "warning"}>
            OSC {osc.connection || "unknown"}
          </StatusBadge>
          <StatusBadge tone={idleRelay.connection === "detected" ? "success" : (idleRelay.connection === "listening" ? "info" : "warning")}>
            VMC {idleRelay.connection || "unknown"}
          </StatusBadge>
          <StatusBadge tone={hostVmc.active ? "success" : "warning"}>
            宿主 VMC {hostVmc.active ? "enabled" : "disabled"}
          </StatusBadge>
          <StatusBadge tone="info">行为 {behavior.mode || "disabled"}</StatusBadge>
          <StatusBadge tone={autonomy.armed ? "success" : "warning"}>
            自主 {autonomy.state || "disarmed"}
          </StatusBadge>
          {busy ? <StatusBadge tone="info">命令执行中</StatusBadge> : null}
        </ToolbarGroup>
        <ToolbarGroup>
          <Switch checked={autoRefresh} label="每秒刷新" onChange={setAutoRefresh} />
          <RefreshButton>刷新状态</RefreshButton>
        </ToolbarGroup>
      </Toolbar>

      {!debugAction ? <Alert tone="danger">调试 action 不可用。请确认插件已经重启并加载 0.13.21。</Alert> : null}
      {osc.last_error ? <Alert tone="warning">OSC：{String(osc.last_error)}</Alert> : null}
      {idleRelay.last_error ? <Alert tone="warning">VMC 待机中转：{String(idleRelay.last_error)}</Alert> : null}
      {idleRelay.frame_error ? <Alert tone="warning">VMC 待机帧：{String(idleRelay.frame_error)}</Alert> : null}
      {hostVmc.last_error ? <Alert tone="warning">宿主 VMC 控制：{String(hostVmc.last_error)}</Alert> : null}
      {body.last_error ? <Alert tone="danger">身体调度器：{String(body.last_error)}</Alert> : null}

      <Grid cols={4}>
        <StatCard label="状态" value={body.state || "shutdown"} />
        <StatCard label="实际发送频率" value={`${fixed(metrics.actual_hz)} Hz`} />
        <StatCard label="AnyaDance 数据包" value={udp.sent_packets || 0} />
        <StatCard label="驱动已接受" value={driverLog.accepted_commands ?? "—"} />
        <StatCard label="OSC 收/发" value={`${osc.received_packets || 0} / ${osc.sent_packets || 0}`} />
      </Grid>

      <Grid cols={2}>
        <Card title="运行控制">
          <Stack>
            <Text>{awareness.summary || "暂无身体自知摘要。"}</Text>
            <ButtonGroup>
              <Button tone="success" disabled={busy} onClick={() => run("body_enable")}>启用输出</Button>
              <Button tone="warning" disabled={busy} onClick={() => run("body_disable")}>平滑禁用</Button>
              <Button tone="info" disabled={busy} onClick={() => run("body_reset", { duration_ms: 600 })}>复位 T Pose</Button>
              <Button tone="danger" disabled={busy} onClick={() => run("body_stop")}>立即急停</Button>
            </ButtonGroup>
            <KeyValue
              items={[
                { key: "action", label: "当前动作", value: currentAction?.motion_label || currentAction?.clip_name || currentAction?.name || "无" },
                { key: "behavior", label: "行为状态", value: behavior.mode || "disabled" },
                { key: "layers", label: "活动层", value: (behavior.active_layers || []).join(", ") || "无" },
                { key: "progress", label: "动作进度", value: currentAction ? `${fixed((currentAction.progress || 0) * 100)}%` : "—" },
                { key: "queue", label: "队列", value: body.queue_length || 0 },
                { key: "skipped", label: "跳过帧", value: metrics.skipped_frames || 0 },
              ]}
            />
            <Progress label="当前动作" value={currentAction ? Number(currentAction.progress || 0) * 100 : 0} />
          </Stack>
        </Card>

        <Card title="连接诊断">
          <KeyValue
            items={[
              { key: "udp", label: "AnyaDance UDP", value: udp.target || state.config?.anyadance_target || "—" },
              { key: "driverLog", label: "驱动遥测", value: driverLog.enabled === false ? "已禁用" : `${driverLog.connection || "unknown"} @ ${driverLog.listen_address || state.config?.driver_log_address || "—"}` },
              { key: "driverAck", label: "驱动确认", value: udp.connected === "detected" ? `已确认（接受 ${driverLog.accepted_commands || 0} / 拒绝 ${driverLog.rejected_commands || 0}）` : (udp.connected === "stale" ? "曾确认，已过期" : "无法确认") },
              { key: "concurrentSender", label: "并发发送者", value: body.concurrent_sender_detection === "concurrent" ? `存在其他发送者：${(udp.other_senders || []).join("、") || "未知"}` : (body.concurrent_sender_detection || "unsupported") },
              { key: "oscSend", label: "OSC 发送", value: osc.send_target || state.config?.osc_send_target || "—" },
              { key: "oscListen", label: "OSC 监听", value: osc.listen_address || state.config?.osc_listen_address || "—" },
              { key: "oscListening", label: "9001 监听", value: osc.receiver_listening ? "正常" : "未监听" },
              { key: "vmcListen", label: "N.E.K.O VMC 监听", value: idleRelay.listen_address || "—" },
              { key: "hostVmc", label: "宿主 VMC 输出", value: hostVmc.active ? `已启用 → ${hostVmc.target || "—"}` : "未启用" },
              { key: "vmcSource", label: "VMC 待机来源", value: idleRelay.applied ? "正在中转" : (idleRelay.source_available ? "已检测，当前未应用" : "尚未检测") },
              { key: "avatar", label: "Avatar ID", value: osc.avatar_id || "尚未收到" },
              { key: "sendFailures", label: "发送失败", value: `${udp.send_failures || 0} / ${osc.send_failures || 0}` },
            ]}
          />
        </Card>

        <Card title="自主控制授权">
          <Stack>
            <Text>{autonomy.reason || "必须手动授权；世界观测失败会自动降级。"}</Text>
            <KeyValue items={[
              { key: "state", label: "状态", value: autonomy.state || "disarmed" },
              { key: "ttl", label: "剩余授权", value: autonomy.remaining_seconds == null ? "—" : `${fixed(autonomy.remaining_seconds, 0)} 秒` },
              { key: "revision", label: "世界 revision", value: autonomy.world_revision ?? 0 },
              { key: "goal", label: "当前目标", value: autonomy.goal?.text || "无" },
              { key: "navReason", label: "导航决策", value: navigation.last_decision?.reason || "—" },
              { key: "behaviorPhase", label: "本地行为阶段", value: navigation.behavior?.phase || "idle" },
              { key: "behaviorOutcome", label: "最近行为结果", value: navigation.behavior?.last_outcome?.reason || "—" },
              {
                key: "speed",
                label: "实测水平速度",
                // available=false 是「读不到」，不是「速度为零」——这两者混淆
                // 正是「卡墙不自知」的根源，所以这里必须显式写出来。
                value: oscMotion.available
                  ? `${fixed(oscMotion.horizontal_speed_mps, 2)} m/s`
                  : `不可测（${oscMotion.reason || "unknown"}）`,
              },
              {
                key: "stall",
                label: "卡住判定",
                // detectable 只在导航器 tick 时刷新，未授权时恒为 false；这里用
                // OSC 快照的实时读数判断「能不能观测」，避免把「还没跑」显示成
                // 「观测不到」。
                value: !oscMotion.available
                  ? "无法观测（VRChat 未回传内置移动参数）"
                  : stall.stalled
                    ? `已卡住 ×${stall.stall_count ?? 0}（换目标才解除）`
                    : `正常 ${stall.consecutive_ticks ?? 0}/${stall.threshold_ticks ?? 0}`,
              },
            ]} />
            <ButtonGroup>
              <Button tone="success" disabled={busy || autonomy.armed} onClick={() => run("vrc_autonomy_arm")}>手动授权 30 分钟</Button>
              <Button tone="danger" disabled={busy || !autonomy.armed} onClick={() => run("vrc_autonomy_disarm")}>解除授权并释放</Button>
              <Button tone="warning" disabled={busy || !autonomy.armed} onClick={() => run("vrc_autonomy_stop")}>停止自主目标</Button>
            </ButtonGroup>
          </Stack>
        </Card>
      </Grid>

      <Card title="手臂角度测试">
        <Grid cols={3}>
          <Field label="手臂">
            <Select value={armSide} options={sideOptions} onChange={setArmSide} />
          </Field>
          <Field label={`抬升角 ${elevation}°`}>
            <Slider value={elevation} min={0} max={180} step={1} showValue onChange={setElevation} />
          </Field>
          <Field label={`方位角 ${azimuth}°`}>
            <Slider value={azimuth} min={-180} max={180} step={1} showValue onChange={setAzimuth} />
          </Field>
          <Field label={`伸展 ${fixed(reach, 2)}`}>
            <Slider value={reach} min={0.3} max={1} step={0.05} showValue onChange={setReach} />
          </Field>
          <Field label="掌心">
            <Select value={palm} options={palmOptions} onChange={setPalm} />
          </Field>
          <Field label={`手腕俯仰 ${wristPitch}°`}>
            <Slider value={wristPitch} min={-90} max={90} step={1} showValue onChange={setWristPitch} />
          </Field>
          <Field label={`手腕偏航 ${wristYaw}°`}>
            <Slider value={wristYaw} min={-180} max={180} step={1} showValue onChange={setWristYaw} />
          </Field>
          <Field label={`手腕翻滚 ${wristRoll}°`}>
            <Slider value={wristRoll} min={-180} max={180} step={1} showValue onChange={setWristRoll} />
          </Field>
          <Field label="过渡时间 ms">
            <NumberInput value={armDuration} min={100} max={5000} step={50} onChange={(value) => setArmDuration(Number(value))} />
          </Field>
        </Grid>
        <Button tone="primary" disabled={busy} onClick={() => run("body_arm_pose", {
          side: armSide,
          elevation_deg: elevation,
          azimuth_deg: azimuth,
          reach,
          palm,
          wrist_pitch_deg: wristPitch,
          wrist_yaw_deg: wristYaw,
          wrist_roll_deg: wristRoll,
          duration_ms: armDuration,
        })}>应用手臂姿态</Button>
      </Card>

      <Grid cols={2}>
        <Card title="手部与手势">
          <Stack>
            <Grid cols={3}>
              <Field label="手部">
                <Select value={handSide} options={sideOptions} onChange={setHandSide} />
              </Field>
              <Field label="手型">
                <Select value={handPose} options={handOptions} onChange={setHandPose} />
              </Field>
              <Field label={`力度 ${fixed(handStrength, 2)}`}>
                <Slider value={handStrength} min={0} max={1} step={0.05} showValue onChange={setHandStrength} />
              </Field>
            </Grid>
            <Button tone="primary" disabled={busy} onClick={() => run("body_hand", {
              side: handSide,
              pose: handPose,
              strength: handStrength,
              duration_ms: 300,
            })}>应用手型</Button>
            <Divider />
            <Grid cols={3}>
              <Field label="手势">
                <Select value={gestureName} options={gestureOptions} onChange={setGestureName} />
              </Field>
              <Field label="侧别">
                <Select value={gestureSide} options={sideOptions} onChange={setGestureSide} />
              </Field>
              <Field label={`强度 ${fixed(gestureIntensity, 2)}`}>
                <Slider value={gestureIntensity} min={0} max={1} step={0.05} showValue onChange={setGestureIntensity} />
              </Field>
            </Grid>
            <Button tone="info" disabled={busy} onClick={() => run("body_gesture", {
              name: gestureName,
              side: gestureSide,
              intensity: gestureIntensity,
            })}>播放程序化手势</Button>
            <Divider />
            <Text>语义表达优先选择真实 VMD；没有匹配时回退到程序化覆盖层。舞蹈、序列和抓取中的全身动作会受到保护。</Text>
            <Grid cols={3}>
              <Field label="身体动作意图">
                <Select value={expressionIntent} options={expressionOptions} onChange={setExpressionIntent} />
              </Field>
              <Field label="侧别">
                <Select value={expressionSide} options={[
                  { value: "auto", label: "自动选择" },
                  ...sideOptions,
                ]} onChange={setExpressionSide} />
              </Field>
              <Field label={`强度 ${fixed(expressionIntensity, 2)}`}>
                <Slider value={expressionIntensity} min={0} max={1} step={0.05} showValue onChange={setExpressionIntensity} />
              </Field>
            </Grid>
            <Button tone="success" disabled={busy} onClick={() => run("body_express", {
              intent: expressionIntent,
              side: expressionSide,
              intensity: expressionIntensity,
            })}>请求身体语义动作</Button>
          </Stack>
        </Card>

        <Card title="伸手抓取测试">
          <Stack>
            <Grid cols={2}>
              <Field label="手部">
                <Select value={reachSide} options={sideOptions.slice(0, 2)} onChange={setReachSide} />
              </Field>
              <Field label="高度">
                <Select value={reachHeight} options={[
                  { value: "waist", label: "腰部" },
                  { value: "chest", label: "胸口" },
                  { value: "head", label: "头部" },
                ]} onChange={setReachHeight} />
              </Field>
              <Field label="方向">
                <Select value={reachDirection} options={[
                  { value: "forward", label: "向前" },
                  { value: "inward", label: "向内" },
                  { value: "outward", label: "向外" },
                ]} onChange={setReachDirection} />
              </Field>
              <Field label={`距离 ${fixed(reachDistance, 2)} m`}>
                <Slider value={reachDistance} min={0.15} max={0.7} step={0.05} showValue onChange={setReachDistance} />
              </Field>
              <Field label="动作时间 ms">
                <NumberInput value={reachDuration} min={100} max={5000} step={50} onChange={(value) => setReachDuration(Number(value))} />
              </Field>
            </Grid>
            <Button tone="warning" disabled={busy} onClick={() => run("body_reach_and_grab", {
              side: reachSide,
              height: reachHeight,
              direction: reachDirection,
              distance_m: reachDistance,
              duration_ms: reachDuration,
            })}>伸手并触发 Grip + OSC Grab</Button>
            <Alert tone="info">该测试只能确认输入已发送，不能确认 VRChat Pickup 已经附着。</Alert>
          </Stack>
        </Card>
      </Grid>

      <Card title=".nya 预制动作">
        {clips.length > 0 ? (
          <Stack>
            <Grid cols={3}>
              <Field label="动作片段">
                <Select value={effectiveClip} options={clipOptions} onChange={setClipName} />
              </Field>
              <Field label={`速度 ${fixed(clipSpeed, 2)}x`}>
                <Slider value={clipSpeed} min={0.25} max={3} step={0.05} showValue onChange={setClipSpeed} />
              </Field>
              <Field label="循环次数">
                <NumberInput value={clipLoops} min={1} max={10} step={1} onChange={(value) => setClipLoops(Number(value))} />
              </Field>
              <Field label="切换时间 ms">
                <NumberInput value={clipTransition} min={0} max={5000} step={50} onChange={(value) => setClipTransition(Number(value))} />
              </Field>
              <Switch checked={clipRestore} label="结束后恢复原姿态" onChange={setClipRestore} />
            </Grid>
            <Button tone="success" disabled={busy || !effectiveClip} onClick={() => run("body_play_clip", {
              clip_name: effectiveClip,
              speed: clipSpeed,
              loop_count: clipLoops,
              transition_ms: clipTransition,
              anchor: true,
              restore_after: clipRestore,
            })}>播放动作</Button>
            <KeyValue
              items={[
                { key: "indexed", label: "已索引", value: state.clips?.indexed_count || 0 },
                { key: "unindexed", label: "待首次解析", value: state.clips?.unindexed_count || 0 },
                { key: "resident", label: "内存缓存", value: (state.clips?.cache?.resident_clips || []).join(", ") || "无" },
                { key: "lastParse", label: "上次解析", value: `${fixed(state.clips?.cache?.last_parse_ms, 0)} ms` },
                { key: "metadata", label: "语义元数据", value: state.clips?.motion_catalog?.entries?.length || 0 },
              ]}
            />
            {clips.find((clip) => clip.name === effectiveClip)?.metadata ? (
              <Text>{clips.find((clip) => clip.name === effectiveClip)?.metadata?.description} · 意图：{(clips.find((clip) => clip.name === effectiveClip)?.metadata?.intents || []).join(", ")}</Text>
            ) : null}
            <Text>未索引的大动作会在首次播放时后台解析；期间界面仍可刷新，后续播放直接使用缓存。</Text>
          </Stack>
        ) : (
          <Alert tone="warning">motions 目录中没有有效动作。</Alert>
        )}
        {invalidClips.length > 0 ? <Alert tone="danger">无效动作：{invalidClips.map((clip) => `${clip.name}: ${clip.error}`).join("；")}</Alert> : null}
        {(state.clips?.motion_catalog?.errors || []).length > 0 ? <Alert tone="danger">动作元数据错误：{state.clips?.motion_catalog?.errors?.join("；")}</Alert> : null}
        {(state.clips?.motion_catalog?.missing_clips || []).length > 0 ? <Alert tone="warning">尚未烘焙：{state.clips?.motion_catalog?.missing_clips?.join("、")}</Alert> : null}
      </Card>

      <Grid cols={2}>
        <Card title="Avatar Parameter">
          <Stack>
            <Field label="参数名">
              <Input value={parameterName} placeholder="NEKO_Action" onChange={setParameterName} />
            </Field>
            <Grid cols={2}>
              <Field label="类型">
                <Select value={parameterType} options={[
                  { value: "bool", label: "Bool" },
                  { value: "int", label: "Int" },
                  { value: "float", label: "Float" },
                ]} onChange={setParameterType} />
              </Field>
              <Field label="值">
                {parameterType === "bool" ? (
                  <Select value={parameterValue} options={[
                    { value: "true", label: "true" },
                    { value: "false", label: "false" },
                  ]} onChange={setParameterValue} />
                ) : (
                  <Input value={parameterValue} placeholder="1" onChange={setParameterValue} />
                )}
              </Field>
            </Grid>
            <Button tone="primary" disabled={busy || !parameterName.trim()} onClick={() => run("body_avatar_parameter", {
              name: parameterName,
              value: parameterPayload(),
            })}>发送 Avatar 参数</Button>
          </Stack>
        </Card>

        <Card title="VRChat 输入脉冲">
          <Stack>
            <Grid cols={3}>
              <Field label="输入">
                <Select value={inputAction} options={[
                  { value: "grab", label: "Grab" },
                  { value: "use", label: "Use" },
                  { value: "drop", label: "Drop" },
                ]} onChange={setInputAction} />
              </Field>
              <Field label="手部">
                <Select value={inputSide} options={sideOptions.slice(0, 2)} onChange={setInputSide} />
              </Field>
              <Field label="按住 ms">
                <NumberInput value={inputHold} min={20} max={1000} step={10} onChange={(value) => setInputHold(Number(value))} />
              </Field>
            </Grid>
            <Button tone="warning" disabled={busy} onClick={() => run("body_vrchat_input", {
              action: inputAction,
              side: inputSide,
              hold_ms: inputHold,
            })}>发送并自动释放</Button>
            <Text>VRChat 必须启用 OSC；Grab、Use、Drop 的部分行为只在 VR 模式有效。</Text>
          </Stack>
        </Card>
      </Grid>

      <Grid cols={2}>
        <Card title="身体自知快照">
          <JsonView data={{ behavior, idle_relay: idleRelay, motion: awareness.motion, pose: awareness.pose, transition: awareness.transition }} />
        </Card>
        <Card title="OSC 参数回传">
          <JsonView data={{ avatar_id: osc.avatar_id, connection: osc.connection, motion: oscMotion, parameters: osc.parameters || {} }} />
        </Card>
      </Grid>

      <Card title="调试命令日志">
        <Stack>
          <LogViewer text={logText} autoScroll deps={[logText]} />
          <Button onClick={() => { setClientLog([]) }}>清除本地结果</Button>
        </Stack>
      </Card>
    </Page>
  )
}
