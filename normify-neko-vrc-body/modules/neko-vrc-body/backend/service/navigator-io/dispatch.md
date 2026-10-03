---
uid: 9e3e0018
id: neko-vrc-body.backend.service.navigator-io.dispatch
parent: neko-vrc-body.backend.service.navigator-io
name: {zh: "导航器下发", en: "Navigator Dispatch"}
description:
  zh: >
      导航器驱动化身的唯一通道：移动轴、受门控的转向与输入释放。
      
  en: >
      The only channel the navigator may drive the avatar through: locomotion axes, gated turns and input release.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.740Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2597
    end_line: 2664
apis:
  - protocol: file
    path: "backend/service.py#BackendService._navigator_send_axes"
    description:
      zh: >
          代表导航器发送一条移动轴命令。
          
      en: >
          Send a locomotion axis command on behalf of the navigator.
          
  - protocol: file
    path: "backend/service.py#BackendService._record_turn_submission"
    description:
      zh: >
          记录一次已提交的转向及其朝向。
          
      en: >
          Record that a turn was submitted, with its heading.
          
  - protocol: file
    path: "backend/service.py#BackendService._navigator_send_turn"
    description:
      zh: >
          代表导航器发送一条转向命令。
          
      en: >
          Send a turn command on behalf of the navigator.
          
  - protocol: file
    path: "backend/service.py#BackendService._navigator_release_inputs"
    description:
      zh: >
          释放导航器仍按住的输入。
          
      en: >
          Release held navigator inputs.
          
  - protocol: file
    path: "backend/service.py#BackendService.turn_trace_snapshot"
    description:
      zh: >
          读取近期转向轨迹。
          
      en: >
          Read the recent turn trace.
          
---
