---
uid: 9e3e0011
id: neko-vrc-body.backend.service.autonomy.target-refs
parent: neko-vrc-body.backend.service.autonomy
name: {zh: "目标引用缓存", en: "Target Reference Cache"}
description:
  zh: >
      实体引用的短 TTL 缓存，使上一次工具调用提到的目标在下一次仍能解析。
      
  en: >
      A short TTL cache of entity references, so a target named one tool call ago still resolves on the next one.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.730Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1422
    end_line: 1527
apis:
  - protocol: file
    path: "backend/service.py#BackendService._visible_entity_by_id"
    description:
      zh: >
          按引用查出当前可见的实体。
          
      en: >
          Look up one visible entity by its reference.
          
  - protocol: file
    path: "backend/service.py#BackendService._remember_target_refs"
    description:
      zh: >
          连同帧修订号记住实体引用。
          
      en: >
          Remember entity references with their frame revision.
          
  - protocol: file
    path: "backend/service.py#BackendService._resolve_target_ref"
    description:
      zh: >
          针对特定帧修订号解析目标引用。
          
      en: >
          Resolve a target reference against a specific frame revision.
          
  - protocol: file
    path: "backend/service.py#BackendService._prune_target_refs_locked"
    description:
      zh: >
          使过旧而不可信的目标引用过期。
          
      en: >
          Expire target references that are too old to trust.
          
---
