---
uid: "9e600002"
id: neko-vrc-body.backend.navigator.decision.free-roam
parent: neko-vrc-body.backend.navigator.decision
name: {zh: "自由漫游与守卫", en: "Free Roam and Guard"}
description:
  zh: >
      自由漫游决策，以及可以否决任何方向的可通行性守卫 —— 判断不了时如实回答 unknown。
      
  en: >
      Free-roam decisions and the traversability guard that may veto any direction, answering unknown when it cannot tell.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.671Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 2177
    end_line: 2335
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._free_roam_decision"
    description:
      zh: >
          没有目标时的自由漫游决策。
          
      en: >
          Free-roam decision when there is no target.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._traversability_guard_reason"
    description:
      zh: >
          可通行性守卫拒绝某个方向的理由。
          
      en: >
          Reason the traversability guard refuses a direction.
          
deps:
  - kind: call
    to: neko-vrc-body.backend.traversability.optical-flow
    label: {zh: "光流可通行性预测", en: "Optical-flow prediction"}
  - kind: call
    to: neko-vrc-body.backend.traversability.ground-extent
    label: {zh: "单帧地面可见范围", en: "Single-frame ground extent"}
---
