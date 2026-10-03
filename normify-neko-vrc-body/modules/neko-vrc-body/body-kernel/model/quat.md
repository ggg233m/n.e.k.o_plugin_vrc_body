---
uid: 9e3a0303
id: neko-vrc-body.body-kernel.model.quat
parent: neko-vrc-body.body-kernel.model
name: {zh: "四元数净化", en: "Quaternion Hygiene"}
description:
  zh: >
      四元数归一化，以及每一帧出站前必须通过的有限值检查。
      
  en: >
      Quaternion normalisation and the finite-value check that every outbound frame must pass.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.785Z"
fingerprint: 172f824e6246028ebf80474b7ac6a938e3b15068a82f1e384a1b87744984bc0c
source:
  - path: "model.py"
    line: 113
    end_line: 126
apis:
  - protocol: file
    path: "model.py#normalized_quat"
    description:
      zh: >
          归一化四元数，退化时回落到单位四元数。
          
      en: >
          Normalise a quaternion, falling back to identity when degenerate.
          
  - protocol: file
    path: "model.py#all_finite"
    description:
      zh: >
          检查序列中的每个值是否有限。
          
      en: >
          Check that every value in a sequence is finite.
          
---
