---
uid: "9e410011"
id: neko-vrc-body.backend.avatar-identity.registry.absorb
parent: neko-vrc-body.backend.avatar-identity.registry
name: {zh: "身份合并", en: "Identity Absorption"}
description:
  zh: >
      身份合并与歧义候选消解，优先用几何与上下文而非单一分数。
      
  en: >
      Identity merging and ambiguous-candidate resolution, preferring geometry and context over any single score.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.577Z"
fingerprint: 89b5d2d1b91411d8de7f700e453b25567d25e099f70a2b5e51581f661c2a2793
source:
  - path: "backend/avatar_identity.py"
    line: 442
    end_line: 617
apis:
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry._absorb"
    description:
      zh: >
          把一条轨迹的证据吸收进目标身份的记录。
          
      en: >
          Absorbs the evidence of one track into a target identity's record.
          
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry._resolve_ambiguous_candidate"
    description:
      zh: >
          在多个候选身份之间用几何与上下文消解歧义。
          
      en: >
          Resolves ambiguity among several candidate identities using geometry and context.
          
---
