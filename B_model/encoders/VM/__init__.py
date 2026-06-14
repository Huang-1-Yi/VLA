# -*- coding: utf-8 -*-
"""B_model.encoders.VM —— Vision Model 编码器族(7 个单图变体)。

v5-1 + 2026-06-14 扩展:**所有 VM 都是单图接口**,输入 (B, C, H, W),输出 (B, D)。
多相机由 Adapter 沿 batch 维 cat 处理,本目录不感知。

阶段 1 主用:
  - RobomimicObsEncoder:robomimic bc_rnn 包装(阶段 1 baseline)

阶段 1 备选/阶段 2/3:
  - TorchvisionResNetObsEncoder:纯 torchvision ResNet(去 timm 依赖)
  - R3MObsEncoder:R3M 预训练 ResNet(机器人泛化强)
  - TimmImageObsEncoder:timm backbone 单图(resnet18/convnext/vit)
  - CLIPImageObsEncoder:CLIP ViT-B/32(可选下载)
  - DP3ObsEncoder:3D 点云(robomimic 仿真用不到,真机/3D 仿真用)
  - TransformerObsEncoder:DINO/CLIP-based ViT(语义特征强)
"""
from B_model.encoders.VM.robomimic_obs_encoder import RobomimicObsEncoder
from B_model.encoders.VM.timm_image_obs_encoder import TimmImageObsEncoder
from B_model.encoders.VM.clip_image_obs_encoder import CLIPImageObsEncoder
from B_model.encoders.VM.torchvision_resnet_obs_encoder import TorchvisionResNetObsEncoder
from B_model.encoders.VM.r3m_obs_encoder import R3MObsEncoder
from B_model.encoders.VM.dp3_obs_encoder import DP3ObsEncoder
from B_model.encoders.VM.transformer_obs_encoder import TransformerObsEncoder

__all__ = [
    "RobomimicObsEncoder",
    "TimmImageObsEncoder",
    "CLIPImageObsEncoder",
    "TorchvisionResNetObsEncoder",
    "R3MObsEncoder",
    "DP3ObsEncoder",
    "TransformerObsEncoder",
]
