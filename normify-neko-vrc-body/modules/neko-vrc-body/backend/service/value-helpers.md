---
uid: 9e3e0001
id: neko-vrc-body.backend.service.value-helpers
parent: neko-vrc-body.backend.service
name: {zh: "取值归一化", en: "Value Normalisation"}
description:
  zh: >
      对所有经 IPC 传入的取值做范围与类型归一化，以及只有人形类别才算本地可检测这条规则。
      
  en: >
      Range and type normalisation for every value arriving over IPC, plus the rule that only humanoid selectors are locally detectable.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.757Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 93
    end_line: 210
apis:
  - protocol: file
    path: "backend/service.py#_selector_is_locally_detectable"
    description:
      zh: >
          本地检测器能否真正为某个语义选择器产出实体框。
          
      en: >
          Whether a local detector can actually produce boxes for a semantic selector.
          
  - protocol: file
    path: "backend/service.py#_effective_detector_interval_ms"
    description:
      zh: >
          按实际推理设备选择检测间隔，未知一律取 CPU 安全值。
          
      en: >
          Pick the detector interval by the actual inference device, defaulting to the CPU-safe value.
          
  - protocol: file
    path: "backend/service.py#_osc_axis_value"
    description:
      zh: >
          归一化一个 OSC 轴值。
          
      en: >
          Normalise an OSC axis value.
          
  - protocol: file
    path: "backend/service.py#_controller_value"
    description:
      zh: >
          把扳机或握把数值归一化到 0..1 的协议范围。
          
      en: >
          Normalise a trigger or grip value into the 0..1 protocol range.
          
  - protocol: file
    path: "backend/service.py#_osc_duration_ms"
    description:
      zh: >
          把 OSC 时长收敛到协议的毫秒范围。
          
      en: >
          Clamp an OSC duration to the protocol's millisecond range.
          
  - protocol: file
    path: "backend/service.py#_controller_hold_ms"
    description:
      zh: >
          收敛控制器保持时长。
          
      en: >
          Clamp a controller hold duration.
          
---
