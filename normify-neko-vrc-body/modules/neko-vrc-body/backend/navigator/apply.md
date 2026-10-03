---
uid: "9e600001"
id: neko-vrc-body.backend.navigator.apply
parent: neko-vrc-body.backend.navigator
name: {zh: "决策施加", en: "Decision Application"}
description:
  zh: >
      把决策落到化身上，并在出错时仍安全地释放输入。
      
  en: >
      Bringing the decision out to the avatar, and releasing inputs safely when anything goes wrong.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.661Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 2442
    end_line: 2527
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._apply"
    description:
      zh: >
          把决策施加到化身输入上。
          
      en: >
          Apply the decision to the avatar inputs.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._safe_release"
    description:
      zh: >
          即使失败也释放全部按住的输入。
          
      en: >
          Release every held input even on failure.
          
deps:
  - kind: call
    to: neko-vrc-body.backend.service.navigator-io.dispatch
    label: {zh: "经服务驱动", en: "Move via service"}
  - kind: call
    to: neko-vrc-body.backend.navigator.wander.direction-memory
    label: {zh: "拒绝受阻方位", en: "Refuse blocked bearing"}
---
