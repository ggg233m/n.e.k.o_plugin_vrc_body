---
uid: 9e3a0233
id: neko-vrc-body.plugin.tools.vision.start-stop
parent: neko-vrc-body.plugin.tools.vision
name: {zh: "视觉启停", en: "Vision Start and Stop"}
description:
  zh: >
      由宿主模型启停后端感知 worker。
      
  en: >
      Starting and stopping the backend perception worker from the host model.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.899Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2802
    end_line: 2818
apis:
  - protocol: rpc
    path: "vrc_vision_start"
    description:
      zh: >
          工具：启动感知 worker。
          
      en: >
          Tool: start the perception worker.
          
  - protocol: rpc
    path: "vrc_vision_stop"
    description:
      zh: >
          工具：停止感知 worker。
          
      en: >
          Tool: stop the perception worker.
          
---
