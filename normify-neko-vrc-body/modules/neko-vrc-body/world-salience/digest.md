---
uid: 9e3a002a
id: neko-vrc-body.world-salience.digest
parent: neko-vrc-body.world-salience
name: {zh: "实体摘要与签名", en: "Entity Digest and Signature"}
description:
  zh: >
      稳定的实体摘要，以及建立在量化距离与方位之上的增量去重签名。
      
  en: >
      Stable per-entity digests and the delta de-duplication signature built on quantised distance and bearing.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.946Z"
fingerprint: 9514e5be7514a04afbe83acf4af23398f6939c9ad2d179fbcb3a12d4af685844
source:
  - path: "world_salience.py"
    line: 101
    end_line: 146
apis:
  - protocol: file
    path: "world_salience.py#entity_digest"
    description:
      zh: >
          稳定的实体摘要，含量化后的距离档与方位档。
          
      en: >
          Stable per-entity digest including the quantised distance and bearing bands.
          
  - protocol: file
    path: "world_salience.py#delta_signature"
    description:
      zh: >
          世界增量的去重签名。
          
      en: >
          De-duplication signature for a world delta.
          
---
