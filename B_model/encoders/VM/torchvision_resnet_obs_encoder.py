"""B_model.encoders.VM.torchvision_resnet_obs_encoder —— 纯 torchvision ResNet 单图像编码器(去 timm 依赖版)。

════════════════════════════════════════════════════════════════════════
安装命令:
  # conda install pytorch::pytorch torchvision
  # (timm 不是本类依赖;本类只依赖 torchvision,部署/精简环境时使用)
════════════════════════════════════════════════════════════════════════

源:参考 PADP `model/vision/model_getter.py:get_resnet(name, weights, **kwargs)`。
与 `TimmImageObsEncoder` 的区别:
  - `TorchvisionResNetObsEncoder`(本类):吃单图 `(B, C, H, W)`,用 `torchvision.models.resnet18/34/50`
  - `TimmImageObsEncoder`:用 timm 库加载同样架构(可换 resnet50/convnext/vit 等)
两者 output_shape() 相同 `(proj_dim,)`,policy 用 output_shape() 自动适配下游 dim。

════════════════════════════════════════════════════════════════════════
默认 config 注释(参考 PADP config/robomimic_padp_position_wise_v3.yaml):
  # model_name:        "resnet18"         # 也可 "resnet34" / "resnet50"
  # weights:           "IMAGENET1K_V1"    # 或 None(随机初始化)
  # crop_shape:        [76, 76]           # 训练时 RandomCrop,推理时 CenterCrop
  # obs_encoder_group_norm: True          # BN → GN 替换(原 conv 训练稳定)
  # eval_fixed_crop:   True               # 推理时固定 crop 位置
  # proj_dim:          64                 # Linear 投影后的输出维度
════════════════════════════════════════════════════════════════════════
"""
import logging

import torch
import torch.nn as nn
import torchvision

from B_model.encoders.interface_vm import VisionEncoderInterface

logger = logging.getLogger(__name__)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _replace_bn_with_gn(root_module):
    """递归把 BN2d 替换为 GN,模仿 PADP `replace_submodules` 行为。"""
    for name, module in root_module.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            num_groups = (num_features // 16) if (num_features % 16 == 0) else (num_features // 8)
            setattr(root_module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_features))
        else:
            _replace_bn_with_gn(module)


class TorchvisionResNetObsEncoder(VisionEncoderInterface):
    """单图像 torchvision ResNet 编码器,吃 (B, C, H, W) → (B, proj_dim)。"""

    def __init__(self,
                 model_name: str = "resnet18",
                 weights: str = "IMAGENET1K_V1",
                 crop_shape=(76, 76),
                 obs_encoder_group_norm: bool = True,
                 eval_fixed_crop: bool = True,
                 proj_dim: int = 64):
        super().__init__()
        self.model_name = model_name
        self.proj_dim = int(proj_dim)

        # backbone 加载(去 fc,只留 conv feature)
        weights_enum = getattr(torchvision.models, "Weights").DEFAULT if weights == "IMAGENET1K_V1" else None
        if weights == "IMAGENET1K_V1":
            backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif weights is None:
            backbone = torchvision.models.resnet18(weights=None)
        else:
            backbone = getattr(torchvision.models, model_name)(weights=weights)
        backbone.fc = nn.Identity()
        # 去掉 avgpool(本类要 spatial feature flatten 之前的所有 conv)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        feature_dim = 512  # resnet18 last conv channel

        if obs_encoder_group_norm and weights is None:
            _replace_bn_with_gn(self.backbone)

        # 训练/推理 preprocess
        ch, cw = crop_shape
        train_t = [
            torchvision.transforms.RandomCrop(size=(ch, cw), padding=4),
            torchvision.transforms.Resize(size=(ch, cw), antialias=True),
            torchvision.transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        eval_t = [
            torchvision.transforms.CenterCrop(size=(ch, cw)),
            torchvision.transforms.Resize(size=(ch, cw), antialias=True),
            torchvision.transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        if not eval_fixed_crop:
            eval_t = [t for t in eval_t if not isinstance(t, torchvision.transforms.CenterCrop)]
        self.train_preprocess = nn.Sequential(*train_t)
        self.eval_preprocess = nn.Sequential(*eval_t)

        self.proj = nn.Linear(feature_dim, proj_dim)
        logger.info("TorchvisionResNetObsEncoder built: model=%s, weights=%s, proj_dim=%d",
                    model_name, weights, proj_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, proj_dim)"""
        if self.training:
            img = self.train_preprocess(img)
        else:
            img = self.eval_preprocess(img)
        feat = self.backbone(img)
        # feat: (B, 512, h, w) → spatial mean pool
        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        else:
            feat = feat.mean(dim=1)
        return self.proj(feat)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)
