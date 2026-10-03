---
uid: "9e420011"
id: neko-vrc-body.backend.navigator.decision.target
parent: neko-vrc-body.backend.navigator.decision
name: {zh: "目标选择", en: "Target selection"}
description:
  zh: >
      按置信度、新鲜度与可达性挑选目标实体。
      
  en: >
      Picks the target entity by confidence, freshness, and reachability.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.673Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 2337
    end_line: 2366
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._select_target"
    description:
      zh: >
          挑选要接近的目标实体。
          
      en: >
          Selects the target entity to approach.
          
---
