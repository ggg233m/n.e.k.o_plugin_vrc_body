# 第三方声明

## AnyaDance

本插件与 `AnyaDance/docs/protocol.md` 中描述的公开 AnyaDance UDP v1 JSON 协议进行互操作。AnyaDance 项目及其协议文档采用 Apache License 2.0 许可协议发布。本插件不包含或修改 AnyaDance 源代码；它仅实现已发布的线上传输格式，并在发行版中保留 AnyaDance 的归属声明。

## OSNet（osnet_x0_25_msmt17）

`models/osnet_x0_25_msmt17/` 携带 OSNet x0_25 行人重识别网络的 ONNX 导出，用于会话级 Avatar 外观重识别。模型结构出自 Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification*（ICCV 2019），参考实现为 [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)（MIT 许可）；本副本取自 Hugging Face 仓库 [anriha/osnet_x0_25_msmt17](https://huggingface.co/anriha/osnet_x0_25_msmt17)（MIT 许可），仅将输入维度改写为动态 batch，未改动权重。

