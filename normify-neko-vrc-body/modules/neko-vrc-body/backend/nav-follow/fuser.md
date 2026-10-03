---
uid: 9e3d0206
id: neko-vrc-body.backend.nav-follow.fuser
parent: neko-vrc-body.backend.nav-follow
name: {zh: "定位融合器", en: "Localization Fusion"}
description:
  zh: >
      把 OSC 航位推算与视觉重定位约束融合成一份带协方差的位姿估计。
      
  en: >
      Fuses OSC dead reckoning with visual relocalization constraints into one pose estimate with covariance.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.618Z"
fingerprint: e04443cf978aa8fa70bfa2a90b855b2665df6c03489b8102b037e97a24315555
source:
  - path: "backend/nav_follow.py"
    line: 39
    end_line: 133
apis:
  - protocol: file
    path: "backend/nav_follow.py#RelocFuser.predict"
    description:
      zh: >
          由航位推算状态预测下一位姿。
          
      en: >
          Predicts the next pose from the dead-reckoning state.
          
  - protocol: file
    path: "backend/nav_follow.py#RelocFuser.add_reloc"
    description:
      zh: >
          加入一条视觉重定位约束。
          
      en: >
          Adds a visual relocalization constraint.
          
  - protocol: file
    path: "backend/nav_follow.py#RelocFuser.estimate"
    description:
      zh: >
          输出融合后的位姿估计与其协方差。
          
      en: >
          Estimates the fused pose together with its covariance.
          
---
