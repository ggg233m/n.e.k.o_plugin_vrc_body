---
uid: 9e3a022b
id: neko-vrc-body.plugin.tools.body.stop
parent: neko-vrc-body.plugin.tools.body
name: {zh: "停止", en: "Stop"}
description:
  zh: >
      停掉一切：当前动作、控制器输入，并可连同驱动它的自主目标一起停。
      
  en: >
      Stopping everything: the active motion, controller inputs, and optionally the autonomy goal driving it.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.893Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2477
    end_line: 2563
apis:
  - protocol: rpc
    path: "body_stop"
    description:
      zh: >
          工具：停止所有身体动作，并可同时停止自主目标。
          
      en: >
          Tool: stop all body motion and optionally the autonomy goal too.
          
deps:
  - kind: call
    to: neko-vrc-body.backend.client.remote-scheduler
    label: {zh: "经 IPC 停止", en: "Stop over IPC"}
---
