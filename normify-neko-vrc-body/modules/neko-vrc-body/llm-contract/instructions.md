---
uid: 9e3a0027
id: neko-vrc-body.llm-contract.instructions
parent: neko-vrc-body.llm-contract
name: {zh: "行为指令文本", en: "Behaviour Instructions"}
description:
  zh: >
      注入每个 N.E.K.O 会话的行为规则文本，与工具 Schema 争抢同一份注意力预算。
      
  en: >
      The behaviour rule text injected into every N.E.K.O session, competing for the same attention budget as the tool schemas.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.836Z"
fingerprint: dbffb90acd2682cf352933ee1ce1366c858e5bda11162679f183ab3967d70472
source:
  - path: "instructions.py"
    line: 22
    end_line: 77
apis:
  - protocol: file
    path: "instructions.py#BODY_AI_INSTRUCTIONS"
    description:
      zh: >
          注入每个 N.E.K.O 会话的行为规则文本。
          
      en: >
          The behaviour rule text injected into every N.E.K.O session.
          
---
