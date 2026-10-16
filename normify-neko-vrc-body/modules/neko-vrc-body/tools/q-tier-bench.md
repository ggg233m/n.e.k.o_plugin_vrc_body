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
      
revision: 0eb991034d815aafdf6ad8342f413fecb385d5b5
updated_at: "2026-10-16T21:51:59.430Z"
fingerprint: edc041c4db8aa8e752d82d26c671a2a3a28ba027a4c3946f4d7a7bee35bb42b3
source:
  - path: "research/tools/q_tier_bench.py"
    line: 24
    end_line: 131
apis:
  - protocol: file
    path: "research/tools/q_tier_bench.py#run"
    description:
      zh: >
          按真实节奏灌入录制，对每个建图阶段计时。
          
      en: >
          Feed a recording at its real cadence and time every mapping stage.
          
  - protocol: file
    path: "research/tools/q_tier_bench.py#instrument"
    description:
      zh: >
          给 mapper 实例挂上分阶段计时器。
          
      en: >
          Attach per-stage timers to a mapper instance.
          
---
