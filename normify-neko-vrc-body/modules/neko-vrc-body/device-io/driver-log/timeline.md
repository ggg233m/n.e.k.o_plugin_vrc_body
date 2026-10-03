---
uid: 9e3a0364
id: neko-vrc-body.device-io.driver-log.timeline
parent: neko-vrc-body.device-io.driver-log
name: {zh: "动作时间轴", en: "Action Timeline"}
description:
  zh: >
      运行时动作时间轴：路线历史的唯一写入口，决定哪些内容真正落盘。
      
  en: >
      The runtime action timeline: the only writer into the route history, deciding what reaches disk.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.821Z"
fingerprint: c8a4fd5ddeaad285f69e1b67a07314be7c0738f57a94deaa19af7b24210179e9
source:
  - path: "driver_log.py"
    line: 717
    end_line: 896
apis:
  - protocol: file
    path: "driver_log.py#ActionTimeline.record_command"
    description:
      zh: >
          记录一条真正提交出去的命令。
          
      en: >
          Record a command that was actually submitted.
          
  - protocol: file
    path: "driver_log.py#ActionTimeline.record_turn"
    description:
      zh: >
          记录一次转向及其偏航增量与是否真的发出。
          
      en: >
          Record a turn with its yaw delta and whether it was sent.
          
  - protocol: file
    path: "driver_log.py#ActionTimeline.begin"
    description:
      zh: >
          打开时间轴并锚定时间基准。
          
      en: >
          Open the timeline and anchor the timebase.
          
---
