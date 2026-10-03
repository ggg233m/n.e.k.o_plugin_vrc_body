---
uid: 9e3a0363
id: neko-vrc-body.device-io.driver-log.recorder
parent: neko-vrc-body.device-io.driver-log
name: {zh: "动作日志", en: "Action Log"}
description:
  zh: >
      把动作事件按共享时间基准落成 JSONL，并读回为按 episode 汇总的路线历史。
      
  en: >
      Persisting action events as JSONL on the shared timebase, and reading them back into an episode-level route history.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.820Z"
fingerprint: c8a4fd5ddeaad285f69e1b67a07314be7c0738f57a94deaa19af7b24210179e9
source:
  - path: "driver_log.py"
    line: 619
    end_line: 714
  - path: "driver_log.py"
    line: 899
    end_line: 1004
apis:
  - protocol: file
    path: "driver_log.py#ActionLogRecorder.record"
    description:
      zh: >
          写出一行字段固定的 JSONL 动作记录。
          
      en: >
          Write one correctly-shaped JSONL action record.
          
  - protocol: file
    path: "driver_log.py#load_action_timeline"
    description:
      zh: >
          读回 JSONL 动作日志，坏行跳过而不抛错。
          
      en: >
          Read a JSONL action log back, skipping bad lines instead of raising.
          
  - protocol: file
    path: "driver_log.py#episode_action_summary"
    description:
      zh: >
          按 episode 把动作日志汇总成路线历史。
          
      en: >
          Summarise an action log per episode as a route history.
          
---
