"""B_model.encoders.VM.timm_image_obs_encoder —— Timm 单图像编码器(单相机版)。

与 TimmObsEncoder(多图 dict 版本)的区别:
- `TimmImageObsEncoder`(本文件):吃单张图 `(B, C, H, W)`,输出 `(B, proj_dim)`
- `TimmObsEncoder`:吃 dict,内部为每 key deepcopy 一份 backbone,concat 多图

两者 output_shape() 相同 `(proj_dim,)`,policy 用 output_shape() 自动适配。
"""
import logging

import timm
import torch
import torch.nn as nn
import torchvision

from B_model.encoders.interface_vm import VisionEncoderInterface

logger = logging.getLogger(__name__)


class TimmImageObsEncoder(VisionEncoderInterface):
    """单图像 timm 编码器(每相机一个实例)。"""

    def __init__(self,
                 model_name: str = "resnet18",
                 pretrained: bool = False,
                 frozen: bool = False,
                 use_group_norm: bool = False,
                 imagenet_norm: bool = False,
                 proj_dim: int = 64):
        super().__init__()
        self.model_name = model_name
        self.frozen = frozen
        self.proj_dim = int(proj_dim)

        global_pool = ""
        if model_name.startswith("resnet") or model_name.startswith("convnext"):
            model = timm.create_model(
                model_name=model_name, pretrained=pretrained,
                global_pool=global_pool, num_classes=0,
            )
            model = nn.Sequential(*list(model.children())[:-2])  # 去掉 avgpool + fc
            feature_dim = 512
        elif model_name.startswith("vit"):
            model = timm.create_model(
                model_name=model_name, pretrained=pretrained,
                global_pool=global_pool, num_classes=0,
                img_size=224,
            )
            feature_dim = model.num_features
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        if frozen:
            for p in model.parameters():
                p.requires_grad = False

        if use_group_norm and not pretrained:
            self._replace_bn_with_gn(model)

        # 训练时增强(简化:RandomCrop + Resize),推理时只用 Resize
        train_t = [
            torchvision.transforms.RandomCrop(size=84, padding=4),
            torchvision.transforms.Resize(size=84, antialias=True),
        ]
        eval_t = [torchvision.transforms.Resize(size=84, antialias=True)]
        if imagenet_norm:
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            train_t.append(torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
            eval_t.append(torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.train_preprocess = nn.Sequential(*train_t)
        self.eval_preprocess = nn.Sequential(*eval_t)

        self.backbone = model
        self.proj = nn.Linear(feature_dim, proj_dim)

        logger.info("TimmImageObsEncoder built: model=%s, proj_dim=%d", model_name, proj_dim)

    @staticmethod
    def _replace_bn_with_gn(root_module):
        for name, module in root_module.named_children():
            if isinstance(module, nn.BatchNorm2d):
                gn = nn.GroupNorm(
                    num_groups=(module.num_features // 16) if module.num_features % 16 == 0 else (module.num_features // 8),
                    num_channels=module.num_features,
                )
                setattr(root_module, name, gn)
            else:
                TimmImageObsEncoder._replace_bn_with_gn(module)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, proj_dim)"""
        # === 形状:input (B, C, H, W) → output (B, D_vm) ===
        # === D_vm = self.proj_dim(默认 64,见 __init__)===
        # === backbone 支持 resnet18/34/50 / convnext / vit,__init__ 按 model_name 字符串分支===
        if self.training:
            img = self.train_preprocess(img)
        else:
            img = self.eval_preprocess(img)
        raw = self.backbone(img)
        if raw.dim() == 4:
            feat = raw.mean(dim=(2, 3))
        else:
            feat = raw.mean(dim=1)
        return self.proj(feat)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)