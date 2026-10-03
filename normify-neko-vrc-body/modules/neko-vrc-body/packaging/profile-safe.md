---
uid: 9e3a0034
id: neko-vrc-body.packaging.profile-safe
parent: neko-vrc-body.packaging
name: {zh: "Profile 清理", en: "Profile Sanitiser"}
description:
  zh: >
      去掉非必需的默认 profile，同时保留目标宿主已有配置及其归属记录。
      
  en: >
      Strips the non-essential default profile while preserving the target host's existing config and ownership records.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.842Z"
fingerprint: 7b5cb9817828763cd2d4348ea62620555d71db122da237e4e251b4a4fc038e8e
source:
  - path: "packaging/profile_safe.py"
    line: 11
    end_line: 34
apis:
  - protocol: file
    path: "packaging/profile_safe.py#omit_default_profile"
    description:
      zh: >
          复制包目录树，同时去掉非必需的默认 profile。
          
      en: >
          Copy a package tree while dropping the non-essential default profile.
          
---
