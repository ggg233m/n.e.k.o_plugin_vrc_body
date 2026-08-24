"use strict";

const state = {
  token: "",
  snapshot: null,
  perception: null,
  config: null,
  configInitialized: false,
  timer: null,
};

const byId = (id) => document.getElementById(id);
const get = (value, path, fallback = null) => {
  let current = value;
  for (const key of path.split(".")) {
    if (current == null || typeof current !== "object" || !(key in current)) return fallback;
    current = current[key];
  }
  return current == null ? fallback : current;
};
const text = (id, value) => { byId(id).textContent = value == null || value === "" ? "—" : String(value); };
const number = (value, digits = 1) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const boolLabel = (value, yes = "可用", no = "不可用") => value ? yes : no;

function toast(message, error = false) {
  const element = byId("toast");
  element.textContent = String(message);
  element.classList.toggle("error", error);
  element.classList.remove("hidden");
  window.clearTimeout(element._hideTimer);
  element._hideTimer = window.setTimeout(() => element.classList.add("hidden"), 3600);
}

function readToken() {
  const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
  const fromHash = hash.get("token") || "";
  if (fromHash) {
    sessionStorage.setItem("nekoBackendToken", fromHash);
    history.replaceState(null, "", location.pathname + location.search);
  }
  state.token = fromHash || sessionStorage.getItem("nekoBackendToken") || "";
  byId("authGate").classList.toggle("hidden", Boolean(state.token));
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Neko-Backend-Token", state.token);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) {
    if (response.status === 401) {
      sessionStorage.removeItem("nekoBackendToken");
      state.token = "";
      byId("authGate").classList.remove("hidden");
    }
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function post(path, payload = {}) {
  return api(path, { method: "POST", body: JSON.stringify(payload) });
}

function setConnection(ok, detail = "") {
  const badge = byId("connectionBadge");
  badge.textContent = ok ? `已连接${detail ? ` · ${detail}` : ""}` : "连接失败";
  badge.className = `badge ${ok ? "ok" : "bad"}`;
}

function renderOverview() {
  const snapshot = state.snapshot || {};
  const perception = state.perception || {};
  const body = snapshot.body || {};
  const world = perception.world || snapshot.world || {};
  const vision = get(world, "backends.vision_runtime", {});
  const detector = vision.detector || {};
  const semantic = vision.semantic || {};
  const mainLlmSemantic = vision.main_llm_semantic || {};
  const navigation = perception.navigation || snapshot.navigation || {};
  const autonomy = snapshot.autonomy || {};
  const worker = perception.worker || snapshot.vision_worker || {};
  const memory = world.memory || {};

  text("bodyStatus", body.output_enabled ? "运行中" : "已关闭");
  text("bodyDetail", `${body.state || "unknown"} · ${body.safety_state || "unknown"}`);
  text("detectorStatus", detector.available ? "可用" : "不可用");
  text("detectorDetail", `${detector.runtime || detector.name || "none"} · ${detector.resolved_device || detector.device || "—"}`);
  text("semanticStatus", semantic.available ? "可用" : "未连接");
  text("semanticDetail", semantic.model || semantic.reason || semantic.last_error || "not configured");
  text("navigationStatus", autonomy.armed ? "已授权" : "待授权");
  text("navigationDetail", get(navigation, "last_decision.reason", autonomy.reason || "idle"));

  text("anyadanceFact", `${get(body, "udp.target", "—")} · ${get(body, "udp.connected", "unknown")}`);
  text("oscFact", `${get(snapshot, "vrchat_osc.connection", "unknown")} · ${get(snapshot, "vrchat_osc.send_target", "—")}`);
  text("latencyFact", `${number(get(snapshot, "control_latency.last_latency_ms"), 1)} ms`);
  text("diskFact", `${get(memory, "persistence_write_count", 0)} 次 · 帧不持久化`);
  text("revisionFact", get(world, "status.revision", 0));
  text("entityCountFact", get(world, "status.entity_count", 0));
  text("observationAgeFact", `${number(get(world, "status.last_observation_age_ms"), 0)} ms`);
  text("journalFact", `${get(memory, "revision_journal.entry_count", 0)} / ${get(memory, "revision_journal.capacity", 512)}`);
  const uncertainties = Array.isArray(world.uncertainties) ? world.uncertainties : [];
  text("uncertainties", uncertainties.length ? `不确定性：${uncertainties.join("、")}` : "当前没有报告不确定性。");

  text("captureFact", `${boolLabel(worker.running, "运行中", "已停止")} · ${worker.frames_processed ?? 0} 帧`);
  text("deviceFact", `${detector.runtime || "none"} / ${detector.resolved_device || detector.device || "—"}`);
  text(
    "semanticQueueFact",
    mainLlmSemantic.enabled
      ? `${mainLlmSemantic.request_state || "none"} · revision ${mainLlmSemantic.request_revision ?? "—"} · 已提交 ${mainLlmSemantic.requests_committed ?? 0}`
      : `${get(vision, "semantic_worker.queue_depth", 0)} / ${get(vision, "semantic_worker.queue_size", 1)} · 已处理 ${get(vision, "semantic_worker.processed", 0)}`,
  );
  text("candidateFact", `${get(vision, "semantic_candidates.candidate_count", 0)} / ${get(vision, "semantic_candidates.max_candidates", 32)} · 仅内存`);

  const lines = [
    `后端：${get(snapshot, "backend.started", false) ? "running" : "unknown"}`,
    `视觉：${worker.last_error || get(vision, "last_error", "ok") || "ok"}`,
    `导航：${get(navigation, "last_decision.state", "idle")} / ${get(navigation, "last_decision.reason", "—")}`,
    `授权：${autonomy.state || "disarmed"} / ${autonomy.reason || "—"}`,
  ];
  byId("runtimeLog").textContent = lines.join("\n");
  renderEntities(world.entities || []);
  renderAutonomy(autonomy, navigation);
}

function renderEntities(entities) {
  const body = byId("entityRows");
  body.replaceChildren();
  if (!Array.isArray(entities) || entities.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty";
    cell.textContent = "暂无实体";
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const entity of entities.slice(0, 64)) {
    const attrs = entity.attributes || {};
    const row = document.createElement("tr");
    const values = [
      entity.id,
      entity.label,
      attrs.semantic_type || "未分类",
      number(entity.confidence, 2),
      attrs.bearing_deg == null ? "—" : `${number(attrs.bearing_deg, 1)}°`,
      attrs.identity_method || "—",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value == null ? "—" : String(value);
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
}

function renderAutonomy(autonomy, navigation) {
  const explorer = navigation.explorer || {};
  text("autonomySummary", `${autonomy.state || "disarmed"} · ${autonomy.reason || "—"} · 剩余 ${number(autonomy.remaining_seconds, 0)} 秒`);
  text("goalFact", get(autonomy, "goal.text", "无"));
  text("decisionFact", `${get(navigation, "last_decision.state", "idle")} / ${get(navigation, "last_decision.reason", "—")}`);
  text("scanFact", explorer.scan_turns ?? 0);
  text("llmLoopFact", explorer.llm_calls_in_loop ?? 0);
}

function value(id, fallback = "") { return byId(id).value.trim() || fallback; }
function numericValue(id) { return Number(byId(id).value); }

function populateConfig() {
  if (!state.config || state.configInitialized) return;
  const cfg = state.config.config || {};
  const vision = cfg.vision || {};
  const osc = cfg.vrchat_osc || {};
  const anya = cfg.anyadance || {};
  const autonomy = cfg.autonomy || {};
  const memory = cfg.world_memory || {};
  byId("cfgVisionEnabled").checked = Boolean(vision.enabled);
  byId("cfgCapture").value = vision.capture || "desktop_mirror";
  byId("cfgWindowTitle").value = vision.window_title || "";
  byId("cfgDevice").value = vision.device || "AUTO";
  byId("cfgModelPath").value = vision.model_path || "";
  byId("cfgLabelsPath").value = vision.labels_path || "";
  byId("cfgConfidence").value = vision.confidence_threshold ?? 0.25;
  byId("cfgInterval").value = vision.interval_ms ?? 100;
  byId("cfgDetectorInterval").value = vision.detector_interval_ms ?? 500;
  byId("cfgAcceleratorInterval").value = vision.detector_accelerator_interval_ms ?? 100;
  byId("cfgSemanticBackend").value = vision.semantic_backend || "main_llm";
  byId("cfgSemanticEndpoint").value = vision.semantic_endpoint || "";
  byId("cfgSemanticModel").value = vision.semantic_model || "gpt-4o-mini";
  byId("cfgSemanticRate").value = vision.semantic_max_per_minute ?? 30;
  byId("cfgMainLlmInterval").value = vision.semantic_main_llm_min_interval_s ?? 12;
  byId("cfgAnyaHost").value = anya.host || "127.0.0.1";
  byId("cfgAnyaPort").value = anya.port ?? 39570;
  byId("cfgOscEnabled").checked = Boolean(osc.enabled);
  byId("cfgOscSendPort").value = osc.send_port ?? 9000;
  byId("cfgOscListenPort").value = osc.listen_port ?? 9001;
  byId("cfgAutonomyTtl").value = autonomy.session_ttl_minutes ?? 30;
  byId("cfgPersistWorld").checked = Boolean(memory.persist_world);
  byId("cfgPersistPlayers").checked = Boolean(memory.persist_players);
  text("settingsPath", state.config.settings_path || "配置由宿主管理");
  byId("restartBanner").classList.toggle("hidden", !state.config.restart_required);
  byId("managedBanner").classList.toggle("hidden", state.config.editable);
  byId("saveSettings").disabled = !state.config.editable;
  const hasKey = Boolean(get(state.config, "secrets.vlm_api_key", false));
  const keyBadge = byId("apiKeyBadge");
  if ((vision.semantic_backend || "main_llm") === "main_llm") {
    keyBadge.textContent = "需要 N.E.K.O 宿主桥接";
    keyBadge.className = "badge muted";
  } else {
    keyBadge.textContent = hasKey ? "API Key 已配置" : "API Key 未配置";
    keyBadge.className = `badge ${hasKey ? "ok" : "bad"}`;
  }
  state.configInitialized = true;
}

function collectConfig() {
  return {
    anyadance: {
      host: value("cfgAnyaHost", "127.0.0.1"),
      port: numericValue("cfgAnyaPort"),
    },
    vrchat_osc: {
      enabled: byId("cfgOscEnabled").checked,
      send_port: numericValue("cfgOscSendPort"),
      listen_port: numericValue("cfgOscListenPort"),
    },
    autonomy: { session_ttl_minutes: numericValue("cfgAutonomyTtl") },
    world_memory: {
      persist_world: byId("cfgPersistWorld").checked,
      persist_players: byId("cfgPersistPlayers").checked,
    },
    vision: {
      enabled: byId("cfgVisionEnabled").checked,
      source: "none",
      capture: value("cfgCapture", "desktop_mirror"),
      local_backend: "openvino",
      window_title: value("cfgWindowTitle"),
      device: value("cfgDevice", "AUTO"),
      model_path: value("cfgModelPath") || null,
      labels_path: value("cfgLabelsPath") || null,
      confidence_threshold: numericValue("cfgConfidence"),
      interval_ms: numericValue("cfgInterval"),
      detector_interval_ms: numericValue("cfgDetectorInterval"),
      detector_accelerator_interval_ms: numericValue("cfgAcceleratorInterval"),
      semantic_backend: value("cfgSemanticBackend", "main_llm"),
      semantic_endpoint: value("cfgSemanticEndpoint") || null,
      semantic_model: value("cfgSemanticModel", "gpt-4o-mini"),
      semantic_max_per_minute: numericValue("cfgSemanticRate"),
      semantic_main_llm_min_interval_s: numericValue("cfgMainLlmInterval"),
    },
  };
}

async function refresh({ includeConfig = false } = {}) {
  if (!state.token) return;
  try {
    const requests = [api("/snapshot"), api("/perception")];
    if (includeConfig || !state.config) requests.push(api("/config"));
    const results = await Promise.all(requests);
    state.snapshot = results[0];
    state.perception = results[1];
    if (results[2]) state.config = results[2];
    setConnection(true, get(state.snapshot, "backend.dry_run", false) ? "offline" : "live");
    renderOverview();
    populateConfig();
  } catch (error) {
    setConnection(false);
    toast(error.message || String(error), true);
  }
}

async function command(label, path, payload = {}) {
  try {
    const result = await post(path, payload);
    toast(`${label}：${result.accepted === false ? result.reason || "被拒绝" : "已接受"}`, result.accepted === false);
    await refresh();
    return result;
  } catch (error) {
    toast(`${label}：${error.message || error}`, true);
    return null;
  }
}

function bind() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === tab));
    document.querySelectorAll(".page").forEach((page) => page.classList.toggle("active", page.id === tab.dataset.tab));
  }));
  byId("refreshButton").addEventListener("click", () => refresh({ includeConfig: true }));
  byId("enableBody").addEventListener("click", () => command("启用身体", "/action", { kind: "enable", params: {} }));
  byId("disableBody").addEventListener("click", () => command("禁用身体", "/action", { kind: "disable", params: {} }));
  byId("emergencyStop").addEventListener("click", () => command("急停", "/action", { kind: "stop", params: {} }));
  byId("startVision").addEventListener("click", () => command("启动视觉", "/vision/start"));
  byId("stopVision").addEventListener("click", () => command("停止视觉", "/vision/stop", { reason: "standalone_ui" }));
  byId("armAutonomy").addEventListener("click", () => command("自主授权", "/autonomy/arm"));
  byId("stopAutonomy").addEventListener("click", () => command("停止自主目标", "/autonomy/stop", { reason: "standalone_ui" }));
  byId("disarmAutonomy").addEventListener("click", () => command("解除自主授权", "/autonomy/disarm", { reason: "standalone_ui" }));
  byId("submitGoal").addEventListener("click", async () => {
    const targetId = value("targetId");
    const payload = {
      text: value("goalText", "寻找目标"),
      kind: value("goalKind", "explore"),
      selector: {
        semantic_type: value("selectorType", "npc"),
        min_confidence: numericValue("selectorConfidence"),
      },
      constraints: {
        max_duration_s: numericValue("goalDuration"),
        max_scan_turns: numericValue("goalScans"),
        max_forward_axis: numericValue("goalForward"),
      },
      based_on_revision: Number(get(state.perception, "world.status.revision", 0)),
    };
    if (targetId) payload.target_id = targetId;
    await command("提交目标", "/autonomy/goal", payload);
  });
  byId("loadFrame").addEventListener("click", async () => {
    try {
      const frame = await api("/vision/frame?max_age_ms=3000&overlay=1");
      if (!frame.data_base64) throw new Error(frame.reason || "没有可用帧");
      const image = byId("frameImage");
      image.src = `data:${frame.mime_type || "image/jpeg"};base64,${frame.data_base64}`;
      image.classList.add("loaded");
      text("frameMeta", `${frame.frame_id || "frame"} · revision ${frame.revision ?? "—"} · ${number(frame.age_ms, 0)} ms`);
    } catch (error) {
      toast(`读取画面：${error.message || error}`, true);
    }
  });
  byId("settingsForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      state.config = await post("/config", { config: collectConfig() });
      state.configInitialized = false;
      populateConfig();
      toast("配置已校验并保存；重启独立后端后生效。");
    } catch (error) {
      toast(`保存配置：${error.message || error}`, true);
    }
  });
  byId("authForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const token = value("tokenInput");
    if (!token) return;
    state.token = token;
    sessionStorage.setItem("nekoBackendToken", token);
    byId("authGate").classList.add("hidden");
    await refresh({ includeConfig: true });
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  bind();
  readToken();
  if (state.token) await refresh({ includeConfig: true });
  state.timer = window.setInterval(() => {
    if (!document.hidden) refresh();
  }, 1000);
});
