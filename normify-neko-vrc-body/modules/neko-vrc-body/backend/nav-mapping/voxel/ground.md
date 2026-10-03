---
uid: "9e420020"
id: neko-vrc-body.backend.nav-mapping.voxel.ground
parent: neko-vrc-body.backend.nav-mapping.voxel
name: {zh: "体素化与地面拟合", en: "Voxelization and ground fitting"}
description:
  zh: >
      点云体素化与逐点的地面高度拟合。
      
  en: >
      Voxelizes the point cloud and fits a ground height for each point.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.635Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 120
    end_line: 178
apis:
  - protocol: file
    path: "backend/nav_mapping.py#_voxelize"
    description:
      zh: >
          把点云体素化到栅格中。
          
      en: >
          Voxelizes a point cloud into the grid.
          
  - protocol: file
    path: "backend/nav_mapping.py#_ground_model"
    description:
      zh: >
          为这些点建立地面高度模型。
          
      en: >
          Builds the ground height model for the points.
          
---
