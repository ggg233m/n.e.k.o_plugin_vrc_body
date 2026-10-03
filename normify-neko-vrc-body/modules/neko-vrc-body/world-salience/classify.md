---
uid: 9e3a002b
id: neko-vrc-body.world-salience.classify
parent: neko-vrc-body.world-salience
name: {zh: "叫醒判定", en: "Wake-up Classification"}
description:
  zh: >
      纯判定函数：给定每个实体上一次的档位，哪些世界增量值得打断 agent。
      
  en: >
      The pure wake-up decision: which world deltas deserve to interrupt the agent, given the previous band per entity.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.945Z"
fingerprint: 9514e5be7514a04afbe83acf4af23398f6939c9ad2d179fbcb3a12d4af685844
source:
  - path: "world_salience.py"
    line: 149
    end_line: 229
apis:
  - protocol: file
    path: "world_salience.py#classify"
    description:
      zh: >
          给定每个实体上一次的档位，判定这次世界增量是否值得打断 agent。
          
      en: >
          Decide whether a world delta deserves to interrupt the agent, given the previous band per entity.
          
---
