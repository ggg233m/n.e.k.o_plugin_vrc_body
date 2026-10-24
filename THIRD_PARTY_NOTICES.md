# 第三方声明

## AnyaDance

本插件与 `AnyaDance/docs/protocol.md` 中描述的公开 AnyaDance UDP v1 JSON 协议进行互操作。AnyaDance 项目及其协议文档采用 Apache License 2.0 许可协议发布。本插件不包含或修改 AnyaDance 源代码；它仅实现已发布的线上传输格式，并在发行版中保留 AnyaDance 的归属声明。

## OSNet（osnet_x0_25_msmt17）

`models/osnet_x0_25_msmt17/` 携带 OSNet x0_25 行人重识别网络的 ONNX 导出，用于会话级 Avatar 外观重识别。模型结构出自 Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification*（ICCV 2019），参考实现为 [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)（MIT 许可）；本副本取自 Hugging Face 仓库 [anriha/osnet_x0_25_msmt17](https://huggingface.co/anriha/osnet_x0_25_msmt17)（MIT 许可），仅将输入维度改写为动态 batch，未改动权重。

## 人物检测模型（person_detect_v1.3_s）

`models/person_detect_v1.3_s/` 携带 deepghs 发布的二次元人物检测模型，用于画面中的 Avatar 检测。副本取自 Hugging Face 仓库 [deepghs/anime_person_detection](https://huggingface.co/deepghs/anime_person_detection) 的 `person_detect_v1.3_s/`（模型卡声明 MIT 许可，训练数据为 [deepghs/anime_person_detection](https://huggingface.co/datasets/deepghs/anime_person_detection) 数据集，同为 MIT 声明）。`person_detect_v1.3_s_model.onnx` 与上游 `model.onnx` 字节一致（SHA-256 `6da88929438cd442e31e45ff4f934dd2d7eb9cf7a423c22885d650ee52550f90`），仅改了文件名，未转换、量化或重新训练。

该模型由 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 8.0.20 训练导出（见 `person_detect_v1.3_s_model_artifacts.json`）。Ultralytics 以 AGPL-3.0 发布，并主张用其训练的模型同样受 AGPL-3.0 约束；这一主张与上游模型卡的 MIT 声明之间的关系未经确认。本插件只通过 ONNX Runtime / OpenVINO 推理该 ONNX 文件，不包含也不调用 Ultralytics 代码。

## LiteAnyStereo

`backend/las2_model.py` 携带 LiteAnyStereo v2 Medium（LAS2-M）的网络结构，用于可选的学习型双目视差后端（`backend/stereo_backend.py`，配置 `navmesh.stereo_backend = "las2"`）。代码取自 GitHub 仓库 [TomTomTommi/LiteAnyStereo](https://github.com/TomTomTommi/LiteAnyStereo) 提交 8c97bd4 的 `core/liteanystereov2.py` / `aggregation_fasternet.py` / `fnet.py` / `submodule.py`，按 MIT License（Copyright (c) 2025 Junpeng Jing）发布。只保留 M 尺寸的前馈推理路径，模块名与层级逐字保持不变以便 `strict=True` 载入发布权重；训练分支（`kd_mode`、线性上采样）已删除，两眼特征提取改为合批一次前向（eval 下数值等价）。

预训练权重 `las_ckpt/LAS2_M.pth`（同一上游，MIT，SHA-256 `c41b61efce9b36e6b8b38b580421cd20cc9e84a47f0f45361767281938f69696`）**不随发行包分发**（`pyproject.toml` 的 `exclude_dirs`），需另行放到该路径。LAS2 后端运行时依赖 PyTorch（XPU）与 timm（FasterNet-T0 主干，`pretrained=False`，主干权重已在 checkpoint 里，不联网下载），二者都不在发行包依赖内；默认的 SGBM 后端不依赖它们。

