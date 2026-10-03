---
uid: 9e3a0209
id: neko-vrc-body.plugin.driver-bridge.udp-apply
parent: neko-vrc-body.plugin.driver-bridge
name: {zh: "快照写入 UDP", en: "Snapshot to UDP"}
description:
  zh: >
      把驱动遥测快照写回本机 UDP 驱动，拒绝非本机发送者。
      
  en: >
      Writing a driver telemetry snapshot back into the local UDP driver, refusing senders that are not this machine.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.871Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 1717
    end_line: 1773
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._driver_log_snapshot"
    description:
      zh: >
          读取驱动当前快照。
          
      en: >
          Read the driver's current snapshot.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._apply_driver_log_to_udp"
    description:
      zh: >
          把驱动日志快照写进本机 UDP 驱动，非本机发送者会被丢弃。
          
      en: >
          Push a driver log snapshot into the local UDP driver, dropping non-local senders.
          
---
