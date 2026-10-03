---
uid: 9e3a0238
id: neko-vrc-body.plugin.tools.autonomy.arm-disarm
parent: neko-vrc-body.plugin.tools.autonomy
name: {zh: "授权与撤销", en: "Arm and Disarm"}
description:
  zh: >
      授予与撤销会话授权 —— 全部自主工具都以此为门控。
      
  en: >
      Arming and disarming the session authorization that every autonomy tool is gated on.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.880Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 3274
    end_line: 3284
apis:
  - protocol: rpc
    path: "vrc_autonomy_arm"
    description:
      zh: >
          工具：授予自主操作授权，可指定存活时间。
          
      en: >
          Tool: arm autonomy authorization, optionally with a time-to-live.
          
  - protocol: rpc
    path: "vrc_autonomy_disarm"
    description:
      zh: >
          工具：立即撤销自主操作授权。
          
      en: >
          Tool: withdraw autonomy authorization immediately.
          
---
