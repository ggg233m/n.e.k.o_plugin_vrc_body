---
uid: "9e500010"
id: neko-vrc-body.backend.navigator.stall-guard.unreachable
parent: neko-vrc-body.backend.navigator.stall-guard
name: {zh: "不可达标记", en: "Unreachable Marking"}
description:
  zh: >
      把无进展的目标标记为不可达，避免反复撞同一障碍。
      
  en: >
      Marks a target with no progress as unreachable so the same obstacle is not hit again.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.681Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1620
    end_line: 1641
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._mark_unreachable_locked"
    description:
      zh: >
          在锁内把目标标记为不可达。
          
      en: >
          Marks a target as unreachable under the lock.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._unreachable_ids"
    description:
      zh: >
          列出当前被标记为不可达的目标 id。
          
      en: >
          Lists the target ids currently marked unreachable.
          
---
