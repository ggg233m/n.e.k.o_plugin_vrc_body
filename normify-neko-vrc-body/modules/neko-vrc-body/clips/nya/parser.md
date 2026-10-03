---
uid: 9e3a0381
id: neko-vrc-body.clips.nya.parser
parent: neko-vrc-body.clips.nya
name: {zh: "解析与采样", en: "Parser and Sampler"}
description:
  zh: >
      严格的文本解析器，以及把片段加基础帧转成插值身体帧的采样器。
      
  en: >
      The strict text parser and the sampler that turns a clip plus a base frame into an interpolated body frame.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.808Z"
fingerprint: af36750f4baf9f8ec574db149233d6e34f52a4cca5324558b7d87e202a6a0414
source:
  - path: "nya.py"
    line: 62
    end_line: 187
apis:
  - protocol: file
    path: "nya.py#parse_nya"
    description:
      zh: >
          把 .nya 文本解析成经过校验的片段，任何越界都被拒绝。
          
      en: >
          Parse .nya text into a validated clip, rejecting anything out of range.
          
  - protocol: file
    path: "nya.py#sample_clip"
    description:
      zh: >
          在给定时间上相对基础帧采样一个片段。
          
      en: >
          Sample a clip at a given time against a base frame.
          
---
