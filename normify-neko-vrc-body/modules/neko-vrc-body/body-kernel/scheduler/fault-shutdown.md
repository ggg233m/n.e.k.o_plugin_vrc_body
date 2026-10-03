---
uid: 9e3a0330
id: neko-vrc-body.body-kernel.scheduler.fault-shutdown
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "故障停机", en: "Fault Shutdown"}
description:
  zh: >
      故障路径：进入故障态、干净停止输出、释放输入并清空队列，而不是带伤继续。
      
  en: >
      The fault path: enter fault, stop output cleanly, release inputs and clear the queue rather than limping on.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.792Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1505
    end_line: 1561
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._enter_fault"
    description:
      zh: >
          进入故障态并记录异常。
          
      en: >
          Enter the fault state and record the exception.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._shutdown_output"
    description:
      zh: >
          关闭输出通道并回报最终状态。
          
      en: >
          Shut the output path down and report the final state.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._clear_commands"
    description:
      zh: >
          清空队列中的全部命令。
          
      en: >
          Clear every queued command.
          
---
