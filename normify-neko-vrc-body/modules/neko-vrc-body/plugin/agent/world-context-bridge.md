---
uid: 9e3a0212
id: neko-vrc-body.plugin.agent.world-context-bridge
parent: neko-vrc-body.plugin.agent
name: {zh: "世界上下文桥", en: "World Context Bridge"}
description:
  zh: >
      启动、运行与停止把后端世界增量转成 agent 上下文的后台线程。
      
  en: >
      Start, run and stop the background thread that turns backend world deltas into agent context.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.855Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 700
    end_line: 741
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._start_world_context_bridge"
    description:
      zh: >
          启动轮询世界增量的后台线程。
          
      en: >
          Start the background thread that polls world deltas.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._stop_world_context_bridge"
    description:
      zh: >
          停止后台世界增量轮询器。
          
      en: >
          Stop the background world-delta poller.
          
---
