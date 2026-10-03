---
uid: 9e3f0017
id: neko-vrc-body.backend.world-state.event
parent: neko-vrc-body.backend.world-state
name: {zh: "世界事件", en: "World Event"}
description:
  zh: >
      检测器或 VLM 发出的短时事件假设。
      
  en: >
      A short-lived event hypothesis emitted by a detector or a VLM.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.775Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 272
    end_line: 307
apis:
  - protocol: file
    path: "backend/world_state.py#WorldEvent"
    description:
      zh: >
          世界事件假设的数据结构定义。
          
      en: >
          The data structure definition of a world event hypothesis.
          
  - protocol: file
    path: "backend/world_state.py#WorldEvent.from_mapping"
    description:
      zh: >
          从外部映射构造事件，忽略未知或非法字段。
          
      en: >
          Builds an event from an external mapping, ignoring unknown or invalid fields.
          
---
