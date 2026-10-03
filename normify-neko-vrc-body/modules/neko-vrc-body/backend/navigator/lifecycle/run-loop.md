---
uid: "9e700011"
id: neko-vrc-body.backend.navigator.lifecycle.run-loop
parent: neko-vrc-body.backend.navigator.lifecycle
name: {zh: "运行循环", en: "Run Loop"}
description:
  zh: >
      十赫兹控制循环本身，tick() 由它调用。
      
  en: >
      The 10 Hz control loop itself, which owns the call into tick().
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.678Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1767
    end_line: 1777
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._run"
    description:
      zh: >
          十赫兹控制循环线程。
          
      en: >
          The 10 Hz control loop thread.
          
---
