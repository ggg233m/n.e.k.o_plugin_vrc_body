---
uid: 9e3a0332
id: neko-vrc-body.body-kernel.scheduler.awareness
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "Awareness 负载", en: "Awareness Payload"}
description:
  zh: >
      aware 负载：动作反馈、语义姿态、空闲中继状态，以及宿主展示的一句话摘要。
      
  en: >
      The awareness payload: motion feedback, semantic pose, idle relay state and the one-line summary the host shows.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.791Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1658
    end_line: 1780
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._build_awareness"
    description:
      zh: >
          构建宿主每一拍都会读取的 aware 负载。
          
      en: >
          Build the awareness payload the host reads every tick.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._awareness_summary"
    description:
      zh: >
          把动作与姿态汇总成一句可读的话。
          
      en: >
          Summarise motion and pose into one readable sentence.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._idle_relay_snapshot"
    description:
      zh: >
          汇总空闲中继状态。
          
      en: >
          Summarise the idle relay state.
          
---
