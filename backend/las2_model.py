# -*- coding: utf-8 -*-
"""LAS2-M（LiteAnyStereo v2）推理用网络结构，只给 ``stereo_backend`` 惰性导入。

取自 https://github.com/TomTomTommi/LiteAnyStereo （MIT，Copyright (c) 2025 Junpeng Jing，
提交 8c97bd4）的 ``core/liteanystereov2.py`` / ``aggregation_fasternet.py`` / ``fnet.py`` /
``submodule.py``。只留 M 前馈模型的推理路径：模块名与层级**逐字保持**，
发布权重才能 ``strict=True`` 载入（改名 = 载不进）。训练分支（``kd_mode``、线性上采样）删了。

依赖 torch + timm（FasterNet-T0 主干，``pretrained=False``：主干权重已在 LAS2 checkpoint 里，
不联网）。发行包没有这两个依赖，所以**本模块不得在 import 期被 backend 其它模块引用**
（``stereo_backend`` 只在选了 ``las2`` 时才在函数里 import 它）。
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

FASTERNET_T0_MODEL = "fasternet_t0"
LAS2_FEATURE_CHANNELS = [40, 80, 160, 320]
# 上游 LAS2_MODEL_SIZE_CONFIGS。⚠️ L 的 expanse_ratio 是 8（论文写 4），发布权重 27.3 M ≠ 论文 11.4 M。
SIZE_CONFIGS = {
    "s": {"blocks": [1, 2, 4], "expanse_ratio": 4},
    "m": {"blocks": [4, 8, 16], "expanse_ratio": 4},
    "l": {"blocks": [8, 16, 32], "expanse_ratio": 8},
}
AGG_IN_CHANNELS = 48     # = 代价体视差层数 = 全分辨率 192 px ÷ 4


# ---- submodule.py ----
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False,
                 norm_layer=None, act_layer=None, **kwargs):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BasicDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 norm_layer=None, act_layer=None, **kwargs):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def _leaky02():
    return nn.LeakyReLU(negative_slope=0.2, inplace=True)


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d, act_layer=_leaky02)
        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d, act_layer=_leaky02)

    def forward(self, low, high):
        return self.conv(torch.cat([high, self.deconv(low)], 1))


def disparity_regression(x, maxdisp):
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device).view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def build_correlation_volume(left_feature, right_feature, max_disp):
    B, C, H, W = left_feature.shape
    left_volume = left_feature.unsqueeze(2).expand(B, C, max_disp, H, W)
    padded_right = F.pad(right_feature, (max_disp - 1, 0, 0, 0))
    unfolded_right = padded_right.unfold(3, W, 1)              # (B, C, H, max_disp, W)
    right_volume = torch.flip(unfolded_right, [3]).permute(0, 1, 3, 2, 4)
    return (left_volume * right_volume).mean(dim=1).contiguous()


def context_upsample(depth_low, up_weights):
    b, c, h, w = depth_low.shape
    depth_unfold = F.unfold(depth_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    depth_unfold = F.interpolate(depth_unfold, (h * 4, w * 4), mode="nearest").reshape(b, 9, h * 4, w * 4)
    return torch.sum(depth_unfold * up_weights, dim=1, keepdim=True)


# ---- fnet.py ----
class FeatureNetFasterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(FASTERNET_T0_MODEL, pretrained=False, features_only=True,
                                          out_indices=(0, 1, 2, 3))
        self.feature_channels = list(self.backbone.feature_info.channels())
        if self.feature_channels != LAS2_FEATURE_CHANNELS:
            raise ValueError(f"Expected {FASTERNET_T0_MODEL} channels {LAS2_FEATURE_CHANNELS}, "
                             f"got {self.feature_channels}")
        ch = self.feature_channels
        self.fpn_layer4 = FPNLayer(ch[3], ch[2])
        self.fpn_layer3 = FPNLayer(ch[2], ch[1])
        self.fpn_layer2 = FPNLayer(ch[1], ch[0])
        self.out_conv = BasicConv2d(ch[0], ch[0], kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)

    def forward(self, images):
        c2, c3, c4, c5 = self.backbone(images)
        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.out_conv(self.fpn_layer2(p3, c2))
        return [p2, p3, p4, c5]


# ---- aggregation_fasternet.py ----
class PartialConv3(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv3(x1), x2), dim=1)


class FasterNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, n_div=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.spatial_mixing = PartialConv3(dim, n_div)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False))

    def forward(self, x):
        return x + self.mlp(self.spatial_mixing(x))


class FasterNetResidual(nn.Module):
    def __init__(self, inp, oup, stride, mlp_ratio, n_div=4):
        super().__init__()
        if stride == 1 and inp == oup:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))
        self.block = FasterNetBlock(oup, mlp_ratio=mlp_ratio, n_div=n_div)

    def forward(self, x):
        return self.block(self.proj(x))


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)
        attn = attn + self.conv0_2(self.conv0_1(attn)) + self.conv1_2(self.conv1_1(attn)) + \
               self.conv2_2(self.conv2_1(attn))
        return self.conv3(attn) * cost


class Aggregation(nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()
        cfg = SIZE_CONFIGS["m"]
        blocks, r = cfg["blocks"], cfg["expanse_ratio"]
        c = AGG_IN_CHANNELS
        self.conv0 = nn.Sequential(*[FasterNetResidual(c, c, 1, r) for _ in range(blocks[0])])
        self.conv1 = FasterNetResidual(c, c * 2, 2, r)
        self.conv2 = nn.Sequential(*[FasterNetResidual(c * 2, c * 2, 1, r) for _ in range(blocks[1] - 1)])
        self.conv3 = FasterNetResidual(c * 2, c * 4, 2, r)
        self.conv4 = nn.Sequential(*[FasterNetResidual(c * 4, c * 4, 1, r) for _ in range(blocks[2] - 1)])
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(c * 4, c * 2, 3, padding=1, output_padding=1,
                                                      stride=2, bias=False), nn.BatchNorm2d(c * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(c * 2, c, 3, padding=1, output_padding=1,
                                                      stride=2, bias=False), nn.BatchNorm2d(c))
        self.redir1 = FasterNetResidual(c, c, 1, r)
        self.redir2 = FasterNetResidual(c * 2, c * 2, 1, r)
        self.att0 = AttentionModule(c, backbone_channels[0])
        self.att2 = AttentionModule(c * 2, backbone_channels[1])
        self.att4 = AttentionModule(c * 4, backbone_channels[2])

    def forward(self, x, features_left):
        x = self.att0(self.conv0(x), features_left[0])
        conv2 = self.att2(self.conv2(self.conv1(x)), features_left[1])
        conv4 = self.att4(self.conv4(self.conv3(conv2)), features_left[2])
        return F.relu(self.conv6(F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)) +
                      self.redir1(x), inplace=True)


# ---- liteanystereov2.py ----
class LiteAnyStereoV2M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = FeatureNetFasterNet()
        self.fnet_channels = self.fnet.feature_channels
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.cost_agg = Aggregation(backbone_channels=self.fnet_channels)
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.fnet_channels[0], self.fnet_channels[0], kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(self.fnet_channels[0], self.fnet_channels[0], kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))
        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(self.fnet_channels[0], 16)
        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def normalize_image(self, img):
        return ((img / 255.0 - self.image_mean) / self.image_std).contiguous()

    def forward(self, left, right):
        """推理前向：(1,3,H,W) ×2，0–255 → (1,1,H,W) 左目视差（像素）。H、W 须是 32 的倍数。

        与上游 ``forward(test_mode=True)`` 等价，只是两眼**合批**过一次 fnet（eval 下 BN 用
        running stats、InstanceNorm 按样本归一，合批不改数值）；实测部署块 720 档 51 ms /
        1080 档 58.7 ms 就是这个形态（`.tmp/las2_block.py`）。代价体 48 层 = 全分辨率 192 px。
        """
        x = self.normalize_image(torch.cat([left, right], 0))
        f = self.fnet(x)
        fl = [t[:1] for t in f]
        cv = self.cost_agg(build_correlation_volume(fl[0], f[0][1:], AGG_IN_CHANNELS), fl)
        disp = disparity_regression(F.softmax(cv, dim=1), AGG_IN_CHANNELS)
        xspx = self.refine_3(self.refine_2(self.refine_1(fl[0]), self.stem_2(x[:1])))
        return context_upsample(disp * 4.0, F.softmax(xspx, 1).float())

