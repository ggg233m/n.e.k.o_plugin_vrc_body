---
uid: 9e3a0324
id: neko-vrc-body.body-kernel.scheduler.run-loop
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "实时循环", en: "Real-time Loop"}
description:
  zh: >
      固定周期循环本身，以及尊重优先级与速率限制的逐拍命令泵。
      
  en: >
      The fixed-period loop itself and the per-tick command pump that respects priority and rate limits.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.799Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 499
    end_line: 749
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._run"
    description:
      zh: >
          固定周期的实时循环。
          
      en: >
          The fixed-period real-time loop.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._process_one_command"
    description:
      zh: >
          按优先级与速率限制取出并启动下一条命令。
          
      en: >
          Pop and start the next command, honouring priority and rate limits.
          
---
