---
uid: 9e3a0323
id: neko-vrc-body.body-kernel.scheduler.submit
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "命令提交", en: "Command Submission"}
description:
  zh: >
      所有工具汇入的唯一入口：参数归一化、安全检查与入队。
      
  en: >
      The single entry point every tool funnels into: parameter normalisation, safety checks and enqueue.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.801Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 326
    end_line: 497
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler.submit"
    description:
      zh: >
          按种类与参数提交一条命令。
          
      en: >
          Submit a command by kind and parameters.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._target_pose_summary"
    description:
      zh: >
          为动作描述符汇总命令的目标姿态。
          
      en: >
          Summarise a command's target pose for the action descriptor.
          
---
