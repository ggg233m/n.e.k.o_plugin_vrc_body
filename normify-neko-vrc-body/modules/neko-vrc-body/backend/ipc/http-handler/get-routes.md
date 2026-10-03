---
uid: 9e3c0104
id: neko-vrc-body.backend.ipc.http-handler.get-routes
parent: neko-vrc-body.backend.ipc.http-handler
name: {zh: "只读 GET 路由表", en: "Read-only GET Routes"}
description:
  zh: >
      只读 GET 路由表：健康检查、世界模型、navmesh 与记忆、配置、感知、自主与世界增量。
      
  en: >
      The read-only GET route table: health checks, world model, navmesh and memory, configuration, perception, autonomy and world deltas.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.602Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 218
    end_line: 319
apis:
  - protocol: http
    method: GET
    path: "/vision/mjpeg"
    description:
      zh: >
          视觉画面的 MJPEG 视频流。
          
      en: >
          MJPEG video stream of the vision frame.
          
  - protocol: http
    method: GET
    path: "/health"
    description:
      zh: >
          后端进程的健康与存活状态。
          
      en: >
          Health and liveness status of the backend process.
          
  - protocol: http
    method: GET
    path: "/worldmodel/status"
    description:
      zh: >
          查询世界模型的运行状态。
          
      en: >
          Queries the world model runtime status.
          
  - protocol: http
    method: GET
    path: "/worldmodel/navmesh"
    description:
      zh: >
          读取当前 navmesh 概览。
          
      en: >
          Reads the current navmesh overview.
          
  - protocol: http
    method: GET
    path: "/worldmodel/navmesh/coverage"
    description:
      zh: >
          查询 navmesh 覆盖度统计。
          
      en: >
          Queries navmesh coverage statistics.
          
  - protocol: http
    method: GET
    path: "/worldmodel/navmesh/memory"
    description:
      zh: >
          读取 navmesh 记忆库概览。
          
      en: >
          Reads the navmesh memory bank overview.
          
  - protocol: http
    method: GET
    path: "/worldmodel/navmesh/memory/sessions"
    description:
      zh: >
          列出 navmesh 记忆的会话。
          
      en: >
          Lists the sessions of navmesh memory.
          
  - protocol: http
    method: GET
    path: "/worldmodel/navmesh/memory/session"
    description:
      zh: >
          查询指定会话的 navmesh 记忆。
          
      en: >
          Queries the navmesh memory of a given session.
          
  - protocol: http
    method: GET
    path: "/worldmodel/navmesh/memory/thumb"
    description:
      zh: >
          获取 navmesh 记忆条目的缩略图。
          
      en: >
          Gets the thumbnail of a navmesh memory entry.
          
  - protocol: http
    method: GET
    path: "/config"
    description:
      zh: >
          读取后端当前配置。
          
      en: >
          Reads the backend's current configuration.
          
  - protocol: http
    method: GET
    path: "/awareness"
    description:
      zh: >
          读取 OSC 代理的 awareness 状态。
          
      en: >
          Reads the awareness state of the OSC agent.
          
  - protocol: http
    method: GET
    path: "/cognition"
    description:
      zh: >
          读取认知模块状态。
          
      en: >
          Reads the cognition module state.
          
  - protocol: http
    method: GET
    path: "/perception"
    description:
      zh: >
          读取感知模块状态。
          
      en: >
          Reads the perception module state.
          
  - protocol: http
    method: GET
    path: "/autonomy"
    description:
      zh: >
          读取自主模块的授权与目标状态。
          
      en: >
          Reads the arming and goal state of the autonomy module.
          
  - protocol: http
    method: GET
    path: "/world/delta"
    description:
      zh: >
          获取自上次拉取以来的世界增量。
          
      en: >
          Gets the world deltas accumulated since the last poll.
          
  - protocol: http
    method: GET
    path: "/vision/frame"
    description:
      zh: >
          获取最新视觉帧。
          
      en: >
          Gets the latest vision frame.
          
  - protocol: http
    method: GET
    path: "/semantic/request"
    description:
      zh: >
          轮询待处理的语义请求。
          
      en: >
          Polls for pending semantic requests.
          
---
