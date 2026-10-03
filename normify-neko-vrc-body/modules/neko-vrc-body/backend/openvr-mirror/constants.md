---
uid: 9e41001a
id: neko-vrc-body.backend.openvr-mirror.constants
parent: neko-vrc-body.backend.openvr-mirror
name: {zh: "D3D11 常量", en: "D3D11 Constants"}
description:
  zh: >
      D3D11 COM 常量与 mip 链选择，返回仍不小于目标尺寸的最深 mip。
      
  en: >
      D3D11 COM constants and mip-chain selection returning the deepest mip still no smaller than the target size.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.690Z"
fingerprint: f3d92ecec9db56dafe3d7aa0aaa351a8790ddcceb20a157ad8745cebf1318124
source:
  - path: "backend/openvr_mirror.py"
    line: 38
    end_line: 73
apis:
  - protocol: file
    path: "backend/openvr_mirror.py#_mip_for"
    description:
      zh: >
          为源尺寸与目标尺寸选出满足约束的最深 mip 级别。
          
      en: >
          Picks the deepest mip level that still satisfies the source/target size constraint.
          
  - protocol: file
    path: "backend/openvr_mirror.py#_RGBA_FORMATS"
    description:
      zh: >
          按像素格式排列的 DXGI 格式候选表。
          
      en: >
          The ordered table of DXGI format candidates per pixel format.
          
---
