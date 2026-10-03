---
uid: 9e40001e
id: neko-vrc-body.backend.perception.runtime.stop-mask
parent: neko-vrc-body.backend.perception.runtime
name: {zh: "停止态遮挡", en: "Stopped-State Mask"}
description:
  zh: >
      感知停止后仍然对外可见的遮挡，使宿主不会误以为世界变空了。
      
  en: >
      The masking that stays visible after perception stops, so the host does not mistake it for an emptied world.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.713Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 3700
    end_line: 3758
apis:
  - protocol: file
    path: "backend/vision.py#VisionRuntime._mask_stopped_snapshot"
    description:
      zh: >
          感知停止后为快照补上遮挡标记。
          
      en: >
          Adds an occlusion marker to the snapshot after perception has stopped.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime._mask_stopped_delta"
    description:
      zh: >
          感知停止后为增量补上遮挡标记。
          
      en: >
          Adds an occlusion marker to the delta after perception has stopped.
          
---
