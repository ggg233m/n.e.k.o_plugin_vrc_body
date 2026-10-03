---
uid: "9e400001"
id: neko-vrc-body.backend.perception.frame-encoding
parent: neko-vrc-body.backend.perception
name: {zh: "帧编码与降采样", en: "Frame Encoding and Downsampling"}
description:
  zh: >
      把一帧编码成 JPEG 字节并按比例降采样，给 LLM 的图不需要原始分辨率。
      
  en: >
      Encodes one frame into JPEG bytes with proportional downsampling, so the image handed to the LLM does not need the original resolution.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.696Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 132
    end_line: 183
apis:
  - protocol: file
    path: "backend/vision.py#encode_frame_jpeg"
    description:
      zh: >
          把 BGR 帧按最大边长降采样后编码为 JPEG 字节。
          
      en: >
          Downsamples a BGR frame by its longest side and encodes it as JPEG bytes.
          
---
