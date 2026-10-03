---
uid: 9e3a020a
id: neko-vrc-body.plugin.driver-bridge.delivery-awareness
parent: neko-vrc-body.plugin.driver-bridge
name: {zh: "投递感知", en: "Delivery Awareness"}
description:
  zh: >
      告诉宿主驱动是否接收了转发出去的遥测，使它能显示真实的投递状态。
      
  en: >
      Telling the host whether the driver accepted the forwarded telemetry, so it can show a real delivery state.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.870Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 1776
    end_line: 1807
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._driver_delivery_awareness"
    description:
      zh: >
          回报驱动是否真的接收了上一次投递。
          
      en: >
          Report whether the driver actually accepted the last delivery.
          
---
