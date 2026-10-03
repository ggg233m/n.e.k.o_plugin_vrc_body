---
uid: "9e500011"
id: neko-vrc-body.backend.navigator.stall-guard.motion-sample
parent: neko-vrc-body.backend.navigator.stall-guard
name: {zh: "运动采样", en: "Motion Sample"}
description:
  zh: >
      采样一次运动观测供守卫判定使用。
      
  en: >
      Samples one motion observation for the guard decision.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.680Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1643
    end_line: 1652
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._sample_motion"
    description:
      zh: >
          为守卫采样一次运动观测。
          
      en: >
          Samples one motion observation for the guard.
          
---
