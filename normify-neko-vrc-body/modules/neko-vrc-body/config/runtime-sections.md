---
uid: 9e3a0021
id: neko-vrc-body.config.runtime-sections
parent: neko-vrc-body.config
name: {zh: "身体与安全配置段", en: "Body and Safety Sections"}
description:
  zh: >
      体型、安全护栏、行为仲裁与 VMC 中继调参。
      
  en: >
      Body shape, safety guardrails, behaviour arbitration and the VMC relay tuning block.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.814Z"
fingerprint: 55757bfaa420ee4e8c4479614d38026395bc79552cfe74aec0d249ef070ee23f
source:
  - path: "config.py"
    line: 86
    end_line: 121
apis:
  - protocol: file
    path: "config.py#BodyProfile"
    description:
      zh: >
          用于缩放全部程序化姿态的化身体型参数。
          
      en: >
          Avatar body proportions used to scale every procedural pose.
          
  - protocol: file
    path: "config.py#SafetyConfig"
    description:
      zh: >
          运动护栏：关节限位、最短过渡时长与姿态合理性边界。
          
      en: >
          Motion guardrails: joint limits, minimum transition duration and pose sanity bounds.
          
  - protocol: file
    path: "config.py#BehaviorConfig"
    description:
      zh: >
          行为状态机的基础层/叠加层仲裁调参。
          
      en: >
          Base/overlay arbitration tuning for the behaviour state machine.
          
  - protocol: file
    path: "config.py#VmcIdleConfig"
    description:
      zh: >
          VMC 中继端点、骨骼求解缩放与静止姿态校准窗口。
          
      en: >
          VMC relay endpoint, bone-solve scaling and rest-pose calibration window.
          
---
