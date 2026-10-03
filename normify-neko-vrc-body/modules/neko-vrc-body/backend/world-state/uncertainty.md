---
uid: 9e3f0015
id: neko-vrc-body.backend.world-state.uncertainty
parent: neko-vrc-body.backend.world-state
name: {zh: "不确定性分类", en: "Uncertainty Classification"}
description:
  zh: >
      区分仅供参考的能力边界标记与真正应当停止移动的不确定性。
      
  en: >
      Separates informational capability-boundary markers from the uncertainties that should actually stop movement.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.781Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 64
    end_line: 80
apis:
  - protocol: file
    path: "backend/world_state.py#blocking_uncertainties"
    description:
      zh: >
          筛出真正应当停止移动的阻断型不确定性。
          
      en: >
          Filters out the blocking uncertainties that should actually stop movement.
          
---
