# 第三方声明

## AnyaDance

本插件与 `AnyaDance/docs/protocol.md` 中描述的公开 AnyaDance UDP v1 JSON 协议进行互操作。AnyaDance 项目及其协议文档采用 Apache License 2.0 许可协议发布。本插件不包含或修改 AnyaDance 源代码；它仅实现已发布的线上传输格式，并在发行版中保留 AnyaDance 的归属声明。

## OSNet（osnet_x0_25_msmt17）

`models/osnet_x0_25_msmt17/` 携带 OSNet x0_25 行人重识别网络的 ONNX 导出，用于会话级 Avatar 外观重识别。模型结构出自 Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification*（ICCV 2019），参考实现为 [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)（MIT 许可）；本副本取自 Hugging Face 仓库 [anriha/osnet_x0_25_msmt17](https://huggingface.co/anriha/osnet_x0_25_msmt17)（MIT 许可），仅将输入维度改写为动态 batch，未改动权重。

## 人物检测模型（person_detect_v1.3_s）

`models/person_detect_v1.3_s/` 携带 deepghs 发布的二次元人物检测模型，用于画面中的 Avatar 检测。副本取自 Hugging Face 仓库 [deepghs/anime_person_detection](https://huggingface.co/deepghs/anime_person_detection) 的 `person_detect_v1.3_s/`（模型卡声明 MIT 许可，训练数据为 [deepghs/anime_person_detection](https://huggingface.co/datasets/deepghs/anime_person_detection) 数据集，同为 MIT 声明）。`person_detect_v1.3_s_model.onnx` 与上游 `model.onnx` 字节一致（SHA-256 `6da88929438cd442e31e45ff4f934dd2d7eb9cf7a423c22885d650ee52550f90`），仅改了文件名，未转换、量化或重新训练。

该模型由 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 8.0.20 训练导出（见 `person_detect_v1.3_s_model_artifacts.json`）。Ultralytics 以 AGPL-3.0 发布，并主张用其训练的模型同样受 AGPL-3.0 约束；这一主张与上游模型卡的 MIT 声明之间的关系未经确认。本插件只通过 ONNX Runtime / OpenVINO 推理该 ONNX 文件，不包含也不调用 Ultralytics 代码。

