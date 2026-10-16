---
uid: 9e3a002d
id: neko-vrc-body.tools.precision-probe
parent: neko-vrc-body.tools
name: {zh: "精度探针", en: "Precision Probe"}
description:
  zh: >
      按不同关键帧预算回放同一段录制，把逐帧深度噪声与多视角错位分开量。
      
  en: >
      Replays a recording at different keyframe budgets to separate per-frame depth noise from multi-view misregistration.
      
revision: 0eb991034d815aafdf6ad8342f413fecb385d5b5
updated_at: "2026-10-16T21:51:59.428Z"
fingerprint: e0391b74ac69cfe256f16a80ebd7696b371c6aadd1b21b4cfc7fb716c8fe3852
source:
  - path: "research/tools/precision_probe.py"
    line: 24
    end_line: 217
apis:
  - protocol: file
    path: "research/tools/precision_probe.py#main"
    description:
      zh: >
          按 1/2/4/8 个关键帧回放同一段录制，分离深度噪声与多视角错位。
          
      en: >
          Replays one recording at 1/2/4/8 keyframes to separate depth noise from multi-view misregistration.
          
  - protocol: file
    path: "research/tools/precision_probe.py#pose_ab"
    description:
      zh: >
          只换位姿做 A/B，关键帧、点云与判定规则全部固定。
          
      en: >
          A/B only the poses, holding keyframes, point clouds and the decision rule fixed.
          
---
