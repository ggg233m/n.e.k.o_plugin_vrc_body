---
uid: 9e41001b
id: neko-vrc-body.backend.openvr-mirror.mirror-eye
parent: neko-vrc-body.backend.openvr-mirror
name: {zh: "单眼镜像纹理", en: "Single-Eye Mirror Texture"}
description:
  zh: >
      单眼镜像纹理路径：SRV → 纹理 → staging，staging 只建一次。
      
  en: >
      Single-eye mirror texture path SRV → texture → staging, with the staging texture created only once.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.691Z"
fingerprint: f3d92ecec9db56dafe3d7aa0aaa351a8790ddcceb20a157ad8745cebf1318124
source:
  - path: "backend/openvr_mirror.py"
    line: 76
    end_line: 252
apis:
  - protocol: file
    path: "backend/openvr_mirror.py#MirrorEye.copy"
    description:
      zh: >
          把眼镜像纹理拷进复用的 staging 缓冲供回读。
          
      en: >
          Copies the eye mirror texture into a reused staging buffer for readback.
          
  - protocol: file
    path: "backend/openvr_mirror.py#MirrorEye.map_rgb"
    description:
      zh: >
          映射 staging 内存并输出 RGB 像素。
          
      en: >
          Maps the staging memory and yields RGB pixels.
          
  - protocol: file
    path: "backend/openvr_mirror.py#MirrorEye._setup_scratch"
    description:
      zh: >
          首次使用时按需创建 staging 与中间纹理。
          
      en: >
          Creates the staging and intermediate textures on first use when needed.
          
---
