---
uid: "9e420022"
id: neko-vrc-body.backend.nav-mapping.stereo
parent: neko-vrc-body.backend.nav-mapping
name: {zh: "双目立体匹配", en: "Stereo matching"}
description:
  zh: >
      SGBM 视差与校正后的双目点到 base 系的转换。
      
  en: >
      SGBM disparity and the conversion of rectified stereo points into the base frame.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.634Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 253
    end_line: 281
apis:
  - protocol: file
    path: "backend/nav_mapping.py#stereo_disparity"
    description:
      zh: >
          计算双目图像对的 SGBM 视差。
          
      en: >
          Computes the SGBM disparity of a stereo pair.
          
  - protocol: file
    path: "backend/nav_mapping.py#stereo_points"
    description:
      zh: >
          把视差转换为 base 系中的点。
          
      en: >
          Turns disparity into points in the base frame.
          
  - protocol: file
    path: "backend/nav_mapping.py#make_sgbm"
    description:
      zh: >
          构造按配置好的 SGBM 匹配器。
          
      en: >
          Builds the configured SGBM matcher.
          
---
