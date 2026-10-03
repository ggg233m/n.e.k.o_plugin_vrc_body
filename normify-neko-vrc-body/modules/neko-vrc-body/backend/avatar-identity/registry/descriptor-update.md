---
uid: "9e410010"
id: neko-vrc-body.backend.avatar-identity.registry.descriptor-update
parent: neko-vrc-body.backend.avatar-identity.registry
name: {zh: "描述子更新", en: "Descriptor Update"}
description:
  zh: >
      累积外观原型并按需刷新背景指纹。
      
  en: >
      Accumulates the appearance prototype and refreshes the background fingerprint on demand.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.578Z"
fingerprint: 89b5d2d1b91411d8de7f700e453b25567d25e099f70a2b5e51581f661c2a2793
source:
  - path: "backend/avatar_identity.py"
    line: 370
    end_line: 440
apis:
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry._update_descriptor"
    description:
      zh: >
          把新观测累积进身份的外观原型。
          
      en: >
          Folds a new observation into an identity's appearance prototype.
          
  - protocol: file
    path: "backend/avatar_identity.py#AvatarIdentityRegistry._context_similarity"
    description:
      zh: >
          衡量两个身份所处背景视角的一致程度。
          
      en: >
          Measures how consistent the surrounding background view is between two identities.
          
---
