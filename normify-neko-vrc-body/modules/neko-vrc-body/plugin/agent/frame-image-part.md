---
uid: 9e3a0216
id: neko-vrc-body.plugin.agent.frame-image-part
parent: neko-vrc-body.plugin.agent
name: {zh: "帧图片部分", en: "Frame Image Part"}
description:
  zh: >
      把采集帧转成有界的 base64 图片部分，超过调用方最大帧龄的一律拒绝。
      
  en: >
      Converting a captured frame into a bounded base64 image part, refusing anything older than the caller's max age.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.845Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 894
    end_line: 931
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._frame_image_part"
    description:
      zh: >
          把采集到的帧转成有界的 base64 图片部分。
          
      en: >
          Turn a captured frame into a bounded base64 image part.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._fetch_frame_image_part"
    description:
      zh: >
          取一帧图片部分，并强制检查最大帧龄。
          
      en: >
          Fetch a frame image part, enforcing a maximum frame age.
          
---
