---
uid: 9e3a03a2
id: neko-vrc-body.clips.expression-motion.apply
parent: neko-vrc-body.clips.expression-motion
name: {zh: "增量施加", en: "Delta Application"}
description:
  zh: >
      把采样出的动作以增量形式施加，使当前基础姿态始终保有权威性。
      
  en: >
      Applying the sampled motion as a delta, so the current base pose always stays authoritative.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.804Z"
fingerprint: c259af45ca5ae068f69fda4bf52ec86545024c5ec90b3354281011688df04f0f
source:
  - path: "expression_motion.py"
    line: 254
    end_line: 275
apis:
  - protocol: file
    path: "expression_motion.py#apply_expression_overlay"
    description:
      zh: >
          把采样出的动作以增量形式施加在权威基础姿态上。
          
      en: >
          Apply the sampled motion as a delta on the authoritative base pose.
          
---
