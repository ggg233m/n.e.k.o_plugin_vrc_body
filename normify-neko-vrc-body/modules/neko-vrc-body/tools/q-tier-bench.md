---
uid: 9e3a002f
id: neko-vrc-body.tools.q-tier-bench
parent: neko-vrc-body.tools
name: {zh: "建图耗时基准", en: "Mapping Benchmark"}
description:
  zh: >
      按真实节奏灌入录制，对 rasterize() 的每个阶段计时，量出建图成本。
      
  en: >
      Benchmarks mapping cost by feeding recordings at their real cadence and timing every stage of rasterize().
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.942Z"
fingerprint: 85667bff20bece180781de0ff59d0784cd354ba75f6cec37eeaa9370d79087cd
source:
  - path: "tools/q_tier_bench.py"
    line: 24
    end_line: 131
apis:
  - protocol: file
    path: "tools/q_tier_bench.py#run"
    description:
      zh: >
          按真实节奏灌入录制，对每个建图阶段计时。
          
      en: >
          Feed a recording at its real cadence and time every mapping stage.
          
  - protocol: file
    path: "tools/q_tier_bench.py#instrument"
    description:
      zh: >
          给 mapper 实例挂上分阶段计时器。
          
      en: >
          Attach per-stage timers to a mapper instance.
          
---
