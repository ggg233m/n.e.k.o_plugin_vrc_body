---
uid: 9e3a0321
id: neko-vrc-body.body-kernel.scheduler.types
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "命令类型", en: "Command Types"}
description:
  zh: >
      命令数据类、优先级排序、当前动作记账与截止时间推进规则。
      
  en: >
      The command dataclasses, the priority ordering, active-motion bookkeeping and the deadline advance rule.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.803Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 83
    end_line: 124
apis:
  - protocol: file
    path: "scheduler.py#BodyCommand"
    description:
      zh: >
          队列中等待的一条已归一化身体命令。
          
      en: >
          One normalised body command waiting in the queue.
          
  - protocol: file
    path: "scheduler.py#PrioritizedCommand"
    description:
      zh: >
          已解析调度优先级的命令。
          
      en: >
          A command with its scheduling priority resolved.
          
  - protocol: file
    path: "scheduler.py#ActiveMotion"
    description:
      zh: >
          当前正在采样并输出的动作。
          
      en: >
          The motion currently being sampled and rendered.
          
  - protocol: file
    path: "scheduler.py#advance_deadline"
    description:
      zh: >
          把截止时间推过已错过的时隙，且不产生补偿性突发。
          
      en: >
          Advance a deadline past missed slots without a catch-up burst.
          
---
