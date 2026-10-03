---
uid: 9e3a0333
id: neko-vrc-body.body-kernel.scheduler.publish
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "快照发布", en: "Snapshot Publication"}
description:
  zh: >
      对宿主的限速快照发布，让 60Hz 循环不会淹没前端。
      
  en: >
      Rate-limited snapshot publication to the host, so the 60 Hz loop does not flood the UI.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.799Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1782
    end_line: 1846
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._publish_snapshot"
    description:
      zh: >
          向宿主发布一份新鲜快照。
          
      en: >
          Publish a fresh snapshot to the host.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._publish_snapshot_if_due"
    description:
      zh: >
          仅在发布间隔到期时才发布。
          
      en: >
          Publish only when the publish interval has elapsed.
          
---
