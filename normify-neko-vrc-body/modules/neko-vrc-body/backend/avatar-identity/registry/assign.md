---
uid: "9e410012"
id: neko-vrc-body.backend.avatar-identity.registry.assign
parent: neko-vrc-body.backend.avatar-identity.registry
name: {zh: "轨迹分配", en: "Track Assignment"}
description:
  zh: >
      把易变的检测轨迹映射到有界的化身身份，并对外报告统计。
      
  en: >
      Maps volatile detection tracks onto a bounded set of avatar identities and reports statistics outward.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.578Z"
fingerprint: 89b5d2d1b91411d8de7f700e453b25567d25e099f70a2b5e51581f661c2a2793
source:
  - path: "backend/avatar_identity.py"
    line: 619
    end_line: 806
apis:
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry.assign"
    description:
      zh: >
          把当前检测轨迹分配到已有身份或新建身份。
          
      en: >
          Assigns the current detection tracks to existing identities or creates new ones.
          
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry.status"
    description:
      zh: >
          对外报告身份数量、合并与消歧等统计。
          
      en: >
          Reports identity counts, merges and disambiguation statistics outward.
          
---
