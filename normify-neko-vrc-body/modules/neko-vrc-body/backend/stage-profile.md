---
uid: 9e3a0043
id: neko-vrc-body.backend.stage-profile
parent: neko-vrc-body.backend
name: {zh: "阶段计时器", en: "Stage Profiler"}
description:
  zh: >
      在线链路的阶段耗时观测：只做观测，不改变任何行为。
      
  en: >
      Per-stage latency observation for the online loop: it only measures, it never changes behaviour.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.763Z"
fingerprint: e1d3833575be6abbf0acd7f51f4685bfa57d135a3cf2c1632600c838a6956835
source:
  - path: "backend/stage_profile.py"
    line: 18
    end_line: 75
apis:
  - protocol: file
    path: "backend/stage_profile.py#StageProfiler.snapshot"
    description:
      zh: >
          分阶段耗时计数，输出中位数、p90 与峰值。
          
      en: >
          Per-stage latency counters with median, p90 and peak.
          
  - protocol: file
    path: "backend/stage_profile.py#StageProfiler.measure"
    description:
      zh: >
          在具名阶段下给一段代码计时。
          
      en: >
          Time a block of code under a named stage.
          
---
