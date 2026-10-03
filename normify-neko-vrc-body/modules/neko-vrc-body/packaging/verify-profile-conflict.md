---
uid: 9e3a0035
id: neko-vrc-body.packaging.verify-profile-conflict
parent: neko-vrc-body.packaging
name: {zh: "Profile 冲突验证", en: "Profile Conflict Verifier"}
description:
  zh: >
      复现旧包 profile 归属冲突，验证修订包安装且不改旧配置。
      
  en: >
      Reproduces the old profile-ownership conflict to prove the revised package installs without touching existing config.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.843Z"
fingerprint: f0e01c5683ff9186392ce1dec521e44e870f2940c7576ab7cbf0d28d04a3ed0e
source:
  - path: "packaging/verify_profile_conflict.py"
    line: 16
    end_line: 83
apis:
  - protocol: file
    path: "packaging/verify_profile_conflict.py#main"
    description:
      zh: >
          端到端复现冲突场景并断言旧配置未被改动。
          
      en: >
          Reproduce the conflict scenario end to end and assert the old config is untouched.
          
---
