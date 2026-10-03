---
uid: 9e41000f
id: neko-vrc-body.backend.avatar-identity.registry.init
parent: neko-vrc-body.backend.avatar-identity.registry
name: {zh: "注册表构造", en: "Registry Construction"}
description:
  zh: >
      身份注册表的构造、id 生成与有界淘汰。
      
  en: >
      Construction of the identity registry, id generation and bounded eviction.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.579Z"
fingerprint: 89b5d2d1b91411d8de7f700e453b25567d25e099f70a2b5e51581f661c2a2793
source:
  - path: "backend/avatar_identity.py"
    line: 251
    end_line: 367
apis:
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry.__init__"
    description:
      zh: >
          建立身份注册表并载入容量与阈值等有界参数。
          
      en: >
          Builds the identity registry with bounded capacity and threshold parameters.
          
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry._prune"
    description:
      zh: >
          淘汰过期或超额的记录以保持注册表有界。
          
      en: >
          Evicts stale or excess records to keep the registry bounded.
          
---
