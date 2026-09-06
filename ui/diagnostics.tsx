type Diagnosis = { title: string; reason: string; suggestion: string }
type Scope = "model" | "execution"

const modelErrors: Record<string, [string, string, string]> = {
  timeout: ["模型请求超时", "未在配置的时限内取得模型结果。", "检查模型服务是否可用；持续超时时检查网络和服务耗时，再调整请求超时。"],
  network_error: ["模型服务连接失败", "请求未能通过网络正常完成，可能涉及 DNS、代理或连接中断。", "检查宿主网络、代理和模型服务地址，再等待下一次调用。"],
  missing_api_key: ["缺少模型密钥", "宿主进程未读取到配置指定的环境变量密钥。", "检查该环境变量是否设置；若刚修改系统环境变量，请重启宿主以读取新值。"],
  not_configured: ["模型配置未完成", "模型服务地址、模型名称或认证配置尚未满足调用条件。", "检查意图模型配置中的服务地址、模型名称和密钥环境变量，保存后重载配置。"],
  invalid_json: ["输出不是有效 JSON", "模型返回了内容，但内容无法作为 JSON 解析。", "展开本次输出，检查是否包含 Markdown 或说明文字；确认模型支持结构化输出。"],
  invalid_response: ["模型响应结构不兼容", "响应中缺少可读取的 choices/message/content 内容。", "确认服务提供 Chat Completions 兼容响应；检查服务端日志和模型接口配置。"],
  response_too_large: ["模型响应过大", "接口返回内容超过插件允许的响应大小。", "减少输出 Token 上限，检查服务是否附加了大量额外内容。"],
  schema_not_supported: ["结构化输出格式不受支持", "未取得符合当前结构化输出协议的响应。", "确认模型支持 json_schema 或 json_object，并检查服务兼容性。"],
  unknown_target: ["语义目标不可用", "模型引用的目标不在本次目录中，或不满足该动作的目标要求。", "检查本次输出中的目标键及世界发布的目录；目录变化后等待新的动作建议。"],
  unknown_tag: ["模型引用了未知标签", "输出标签不在本次世界目录允许的标签中。", "检查输出标签与世界目录，确认模型只选择已发布的标签。"],
  unknown_action: ["模型引用了未知动作", "输出动作键未出现在本次世界发布的动作目录中。", "检查世界是否发布该动作；更新目录后再生成动作建议。"],
  unknown_player_slot: ["玩家目标不可用", "模型引用的玩家槽位不在本次上下文中，或社交动作没有可用玩家。", "确认目标玩家仍在世界中，等待使用最新玩家上下文生成的建议。"],
  request_busy: ["已有模型请求进行中", "此次请求未开始，因为上一请求尚未结束。", "等待当前请求结束再试，避免重复触发。"],
  request_expired: ["调用已失效", "请求已超时或被更新的调用取代，后台响应不再用于本次动作。", "查看最近一条调用记录，无需重复执行旧输出。"],
  disabled: ["动作辅助模型未启用", "当前运行配置关闭了独立意图模型。", "如需使用，在动作决策设置中启用并保存、重载配置。"],
  request_error: ["模型请求处理异常", "插件处理请求时出现未分类异常，详情未传入面板。", "按调用时间检查宿主插件日志，确认模型服务和配置后再试。"],
  request_worker_error: ["模型调用线程异常", "后台调用未能正常完成。", "按调用时间检查插件日志；持续发生时重载插件并保留错误记录。"],
}

const executionErrors: Record<string, [string, string, string]> = {
  timeout: ["动作执行超时", "计划或节点超过执行时限，不能据此确认世界动作完成。", "检查 NPC 当前状态和世界日志，确认是否卡住，再根据实际状态决定是否重新下达动作。"],
  operation_timeout: ["世界执行结果未确认", "命令发出后未取得 operation 终态证据；执行器已请求状态快照。", "先检查世界连接和日志回传，刷新状态确认结果，避免立即重复下达同一动作。"],
  operation_missing: ["缺少动作跟踪标识", "命令已被接受，但没有可关联的 operation_id，无法跟踪完成状态。", "检查世界版本是否支持操作生命周期，以及命令响应中的标识是否完整。"],
  approach_timeout: ["接近玩家的结果未确认", "等待期间未取得满足距离要求的终态证据。", "确认玩家仍在场、路径可达并检查状态回传，再决定是否重新接近。"],
  target_missing: ["没有可用导航目标", "当前目标区域没有可用的已发布导航锚点。", "检查世界目录中的区域和 Anchor，确认目标区域发布了可到达的锚点。"],
  slot_unknown: ["目标玩家已不可用", "动作使用的玩家槽位无法继续取得，玩家可能已离开。", "刷新玩家状态，选择仍在场的玩家再执行。"],
  invalid_state: ["世界控制尚未就绪", "NPC 当前不处于允许宿主控制的状态。", "确认已进入支持 YUI 的世界，连接成功后再执行动作。"],
  not_handshaken: ["世界握手尚未完成", "当前控制会话还没有完成世界握手。", "等待自动重连并刷新状态；持续失败时检查世界和 MIDI 通道。"],
  ack_timeout: ["世界未及时确认指令", "在时限内没有收到指令确认，不能判断动作是否已执行。", "检查 MIDI 输入和世界日志回传，等待连接恢复后先确认 NPC 状态。"],
  unsupported_capability: ["世界未提供所需能力", "当前世界没有发布该动作需要的能力。", "选择世界已支持的动作，或更新包含该能力的世界版本。"],
  explicit_control_active: ["指定任务正在占用控制", "自主动作需要等待优先级更高的指定任务。", "等待指定任务结束；仅在确实要中止时手动停止任务。"],
  plan_conflict: ["动作计划冲突", "已有计划占用了新计划需要的控制范围。", "等待当前计划结束，或确认后通过已有停止操作取消当前任务。"],
  plan_capacity: ["计划数量达到上限", "当前会话已达到可保留的计划容量。", "等待现有任务结束再提交，避免连续重复发送计划。"],
  plan_not_found: ["计划记录不存在", "所查询计划已不在当前执行器中，可能已切换会话或重载。", "刷新面板查看当前会话中的计划。"],
  behavior_graph_invalid: ["行为计划格式不合法", "行为图未通过执行前的结构和参数校验。", "检查计划节点、目标和参数是否符合当前世界能力，再提交修正后的计划。"],
  invalid_param: ["动作参数不合法", "动作参数未通过执行器校验。", "检查目标、距离、时长等参数是否在允许范围内。"],
  invalid_origin: ["计划来源不合法", "计划来源不是受支持的指定任务或自主活动。", "检查调用入口，使用插件支持的计划提交方式。"],
  unsupported_node: ["行为节点尚未支持", "执行器没有该节点类型的实现。", "检查插件与行为图版本，改用已支持的节点。"],
  condition_false: ["动作前置条件未满足", "行为图中的条件判断结果为不成立。", "检查所需玩家、距离或控制状态，条件满足后再执行。"],
  selector_exhausted: ["所有备选动作均未成功", "行为图已尝试其备选分支，但没有分支成功。", "查看计划及世界日志，检查各分支依赖的目标和条件。"],
  retry_exhausted: ["动作重试已用尽", "执行器已达到该动作允许的重试次数。", "先检查目标可达性及连接状态，处理原因后再重新提交。"],
  parallel_failed: ["并行动作未成功", "并行节点未取得成功结果。", "检查同时执行的节点及世界日志，定位失败分支。"],
  internal_error: ["计划执行器异常", "后台执行器捕获了内部异常。", "按计划时间检查插件日志，保留错误代码用于定位。"],
  execution_unknown: ["缺少完成证据", "执行结果为未知，当前没有足够证据确认成功或失败。", "刷新世界状态并检查操作终态日志，确认实际结果后再决定下一步。"],
  execution_failed: ["动作未成功", "执行器报告失败，但没有附带具体错误代码。", "按计划时间检查插件及世界日志，定位对应节点。"],
}

const invalidModelFields = new Set([
  "invalid_root", "invalid_root_fields", "invalid_motivation", "invalid_mood", "invalid_activity_count",
  "invalid_activity_fields", "invalid_activity_kind", "invalid_duration", "invalid_tags", "invalid_local_roam_style",
  "invalid_avoid_targets", "invalid_interests", "invalid_interest_fields", "invalid_interest_strength", "invalid_interest_ttl", "invalid_ttl",
  "missing_visit_target", "missing_explore_target", "missing_perform_action", "missing_observe_target",
])

export function diagnose(code: string, scope: Scope): Diagnosis {
  const dictionary = scope === "model" ? modelErrors : executionErrors
  let match = Object.prototype.hasOwnProperty.call(dictionary, code) ? dictionary[code] : undefined
  if (!match && scope === "model" && invalidModelFields.has(code)) match = ["模型输出未通过动作校验", "输出字段、活动数量、时长或必要目标不符合动作协议。", "展开本次输出并对照原始代码检查对应字段；插件不会将这份无效输出交给自主控制器。"]
  if (!match && scope === "model" && /^http_[1-5][0-9]{2}$/.test(code)) {
    const status = Number(code.slice(5))
    if (status === 401 || status === 403) match = ["模型服务拒绝访问", "服务返回认证或访问权限错误；可能涉及密钥、模型权限或访问策略。", "检查密钥是否有效、账号能否访问该模型，以及服务的访问限制。"]
    else if (status === 429) match = ["模型服务限制了请求", "服务返回请求限制响应，可能是频率、并发或配额限制。", "查看服务端限制说明或配额，适当增大调用间隔；避免连续重试。"]
    else if (status >= 500) match = ["模型服务端异常", "接口或中转服务返回服务器错误。", "查看服务状态，等待恢复后再试；持续出现时检查中转服务日志。"]
    else match = ["模型接口返回 HTTP 错误", "服务未接受本次请求或返回了非成功状态。", "检查服务地址、模型名称和请求格式，并按原始 HTTP 代码查看服务端日志。"]
  }
  const [title, reason, suggestion] = match || ["暂未识别的错误", "该代码尚无对应的中文诊断，不能仅凭代码判断具体原因。", "保留原始代码和发生时间，检查对应的插件或世界日志。"]
  return { title, reason, suggestion }
}

export function ErrorDiagnosis({ code, scope }: { code: string; scope: Scope }) {
  const diagnosis = diagnose(code, scope)
  return <div className="yui-notice" role="status" aria-label="中文故障诊断">
    <strong>{diagnosis.title}</strong>
    <p className="yui-muted">原因：{diagnosis.reason}</p>
    <p>建议：{diagnosis.suggestion}</p>
    <details className="yui-help"><summary>原始错误代码</summary><code style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{code}</code></details>
  </div>
}
