---
uid: "9e500016"
id: neko-vrc-body.backend.navigator.wander.direction-memory.advice-api
parent: neko-vrc-body.backend.navigator.wander.direction-memory
name: {zh: "方向建议接口", en: "Direction Advice API"}
description:
  zh: >
      对外的方向建议、偏好更新与拒绝判定。
      
  en: >
      Outward direction advice, preference updates and refusal decisions.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.684Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1156
    end_line: 1206
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator.direction_advice"
    description:
      zh: >
          给出某个方位上的方向建议。
          
      en: >
          Returns the direction advice for a bearing.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator.update_direction_scores"
    description:
      zh: >
          更新各方向的偏好分数。
          
      en: >
          Updates the per-direction preference scores.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator.should_refuse_bearing"
    description:
      zh: >
          判断某个方位是否应当拒绝。
          
      en: >
          Tells whether a bearing should be refused.
          
---
