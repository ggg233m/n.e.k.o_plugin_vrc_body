---
uid: 9e3c0105
id: neko-vrc-body.backend.ipc.http-handler.post-routes
parent: neko-vrc-body.backend.ipc.http-handler
name: {zh: "命令 POST 路由表", en: "Command POST Routes"}
description:
  zh: >
      命令 POST 路由表：动作提交、OSC 与虚拟输入、自主授权与目标、视觉与 navmesh 控制、记忆管理、认知与停机。
      
  en: >
      The command POST route table: action submission, OSC and virtual input, autonomy arming and goals, vision and navmesh control, memory management, cognition and shutdown.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.604Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 321
    end_line: 505
apis:
  - protocol: http
    method: POST
    path: "/config"
    description:
      zh: >
          写入并持久化后端配置。
          
      en: >
          Writes and persists the backend configuration.
          
  - protocol: http
    method: POST
    path: "/action"
    description:
      zh: >
          提交一次身体动作命令。
          
      en: >
          Submits a body action command.
          
  - protocol: http
    method: POST
    path: "/osc/parameter"
    description:
      zh: >
          写入一个 OSC 化身参数。
          
      en: >
          Writes one OSC avatar parameter.
          
  - protocol: http
    method: POST
    path: "/osc/input"
    description:
      zh: >
          发送一次 OSC 输入事件。
          
      en: >
          Sends one OSC input event.
          
  - protocol: http
    method: POST
    path: "/osc/locomotion"
    description:
      zh: >
          设置移动速度矢量。
          
      en: >
          Sets the locomotion velocity vector.
          
  - protocol: http
    method: POST
    path: "/osc/turn"
    description:
      zh: >
          设置转向速度。
          
      en: >
          Sets the turn rate.
          
  - protocol: http
    method: POST
    path: "/osc/stop_movement"
    description:
      zh: >
          立即停止一切移动。
          
      en: >
          Stops all movement immediately.
          
  - protocol: http
    method: POST
    path: "/osc/chatbox"
    description:
      zh: >
          发送聊天框消息。
          
      en: >
          Sends a chatbox message.
          
  - protocol: http
    method: POST
    path: "/osc/batch"
    description:
      zh: >
          批量提交一组 OSC 命令。
          
      en: >
          Submits a batch of OSC commands.
          
  - protocol: http
    method: POST
    path: "/osc/cancel"
    description:
      zh: >
          取消已排程的 OSC 输入。
          
      en: >
          Cancels scheduled OSC inputs.
          
  - protocol: http
    method: POST
    path: "/input/axes"
    description:
      zh: >
          写入虚拟控制器摇杆轴。
          
      en: >
          Writes virtual controller stick axes.
          
  - protocol: http
    method: POST
    path: "/input/button"
    description:
      zh: >
          按下或释放虚拟控制器按键。
          
      en: >
          Presses or releases a virtual controller button.
          
  - protocol: http
    method: POST
    path: "/input/release"
    description:
      zh: >
          释放全部虚拟输入。
          
      en: >
          Releases all virtual inputs.
          
  - protocol: http
    method: POST
    path: "/autonomy/arm"
    description:
      zh: >
          启用自主代理。
          
      en: >
          Arms the autonomy agent.
          
  - protocol: http
    method: POST
    path: "/autonomy/disarm"
    description:
      zh: >
          解除自主代理授权。
          
      en: >
          Disarms the autonomy agent.
          
  - protocol: http
    method: POST
    path: "/autonomy/goal"
    description:
      zh: >
          设置自主代理目标。
          
      en: >
          Sets the autonomy agent goal.
          
  - protocol: http
    method: POST
    path: "/autonomy/wander-step"
    description:
      zh: >
          驱动自主代理单步游走。
          
      en: >
          Drives one wander step of the autonomy agent.
          
  - protocol: http
    method: POST
    path: "/autonomy/intent"
    description:
      zh: >
          提交自主代理的行动意图。
          
      en: >
          Submits an autonomy agent intent.
          
  - protocol: http
    method: POST
    path: "/autonomy/stop"
    description:
      zh: >
          立即停止自主代理。
          
      en: >
          Stops the autonomy agent immediately.
          
  - protocol: http
    method: POST
    path: "/vision/start"
    description:
      zh: >
          启动视觉采集。
          
      en: >
          Starts vision capture.
          
  - protocol: http
    method: POST
    path: "/vision/stop"
    description:
      zh: >
          停止视觉采集。
          
      en: >
          Stops vision capture.
          
  - protocol: http
    method: POST
    path: "/worldmodel/start"
    description:
      zh: >
          启动世界模型。
          
      en: >
          Starts the world model.
          
  - protocol: http
    method: POST
    path: "/worldmodel/stop"
    description:
      zh: >
          停止世界模型。
          
      en: >
          Stops the world model.
          
  - protocol: http
    method: POST
    path: "/worldmodel/world"
    description:
      zh: >
          注入世界快照数据。
          
      en: >
          Injects world snapshot data.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/start"
    description:
      zh: >
          启动 navmesh 构建。
          
      en: >
          Starts navmesh building.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/stop"
    description:
      zh: >
          停止 navmesh 构建。
          
      en: >
          Stops navmesh building.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/goto"
    description:
      zh: >
          下发导航到目标点。
          
      en: >
          Commands navigation to a target point.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/explore"
    description:
      zh: >
          下发 navmesh 探索任务。
          
      en: >
          Commands a navmesh exploration task.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/cancel"
    description:
      zh: >
          取消当前的 navmesh 任务。
          
      en: >
          Cancels the active navmesh task.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/memory/update"
    description:
      zh: >
          更新 navmesh 记忆条目。
          
      en: >
          Updates a navmesh memory entry.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/memory/delete"
    description:
      zh: >
          删除 navmesh 记忆条目。
          
      en: >
          Deletes a navmesh memory entry.
          
  - protocol: http
    method: POST
    path: "/worldmodel/navmesh/memory/prune"
    description:
      zh: >
          裁剪过期的 navmesh 记忆。
          
      en: >
          Prunes stale navmesh memories.
          
  - protocol: http
    method: POST
    path: "/vmc/recalibrate"
    description:
      zh: >
          重新校准 VMC 中继。
          
      en: >
          Recalibrates the VMC relay.
          
  - protocol: http
    method: POST
    path: "/clips/list"
    description:
      zh: >
          列出可用的动作片段。
          
      en: >
          Lists the available action clips.
          
  - protocol: http
    method: POST
    path: "/semantic_express"
    description:
      zh: >
          提交一次语义表达请求。
          
      en: >
          Submits a semantic expression request.
          
  - protocol: http
    method: POST
    path: "/semantic/commit"
    description:
      zh: >
          提交语义结果并落盘。
          
      en: >
          Commits a semantic result for persistence.
          
  - protocol: http
    method: POST
    path: "/world/ingest"
    description:
      zh: >
          注入外部世界观测。
          
      en: >
          Ingests external world observations.
          
  - protocol: http
    method: POST
    path: "/cognition/plan"
    description:
      zh: >
          提交认知计划。
          
      en: >
          Submits a cognition plan.
          
  - protocol: http
    method: POST
    path: "/cognition/feedback"
    description:
      zh: >
          提交认知反馈。
          
      en: >
          Submits cognition feedback.
          
  - protocol: http
    method: POST
    path: "/shutdown"
    description:
      zh: >
          请求后端进程退出。
          
      en: >
          Requests shutdown of the backend process.
          
---
