---
uid: 9e3e0019
id: neko-vrc-body.backend.service.navigator-io.feedback
parent: neko-vrc-body.backend.service.navigator-io
name: {zh: "导航器反馈", en: "Navigator Feedback"}
description:
  zh: >
      允许导航器看到的内容：动作反馈与转向状态，不含它不得假设的世界信息。
      
  en: >
      What the navigator is allowed to see: motion feedback and turn state, nothing about the world it must not assume.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.741Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2666
    end_line: 2700
apis:
  - protocol: file
    path: "backend/service.py#BackendService._navigator_motion_feedback"
    description:
      zh: >
          交给导航器闭环的动作反馈。
          
      en: >
          Motion feedback handed to the navigator's closed loop.
          
  - protocol: file
    path: "backend/service.py#BackendService._navigator_turn_state"
    description:
      zh: >
          交给导航器的当前转向状态。
          
      en: >
          Current turn state handed to the navigator.
          
---
