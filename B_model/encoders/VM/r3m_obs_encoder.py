"""B_model.encoders.VM.r3m_obs_encoder —— R3M 预训练 ResNet 单图像编码器(阶段 2/3 启用)。

════════════════════════════════════════════════════════════════════════
安装命令(使用本类时按需安装,**不进** F_envs/base_train/environment.yml):
  # pip install r3m
  # 或: pip install git+https://github.com/facebookresearch/r3m.git
  #
  # 注:未装 r3m 时本类**自动 fallback 到 torchvision resnet18**(不 raise),
  #     用户可继续跑代码;只是少了 R3M 预训练权重。
════════════════════════════════════════════════════════════════════════

源:参考 PADP `model/vision/model_getter.py:get_r3m(name, **kwargs)`(11 行,调 `r3m.load_r3m(name)`)。
特点:R3M 在 ego-centric 视频上预训练,机器人领域泛化强(v4-1 §七 阶段 2 LoRA 路径里提过)。

════════════════════════════════════════════════════════════════════════
默认 config 注释(参考 PADP model_getter.py:get_r3m):
  # model_name:    "resnet18"     # r3m 支持 resnet18/34/50
  # proj_dim:      64             # Linear 投影后的输出维度
  # crop_shape:    [76, 76]       # 与 TorchvisionResNetObsEncoder 一致
  # imagenet_norm: True           # 输入归一化(与 ImageNet 一致)
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
    """递归把 BN2d 替换为 GN(参考 PADP `replace_submodules`)。"""
    for name, module in root_module.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            num_groups = (num_features // 16) if (num_features % 16 == 0) else (num_features // 8)
            setattr(root_module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_features))
        else:
            _replace_bn_with_gn(module)


class R3MObsEncoder(VisionEncoderInterface):
    """R3M 预训练 ResNet 单图像编码器,吃 (B, C, H, W) → (B, proj_dim)。

    若 r3m 库未装,自动 fallback 到 torchvision resnet18(无 R3M 预训练权重)。
    """

    def __init__(self,
                 model_name: str = "resnet18",
                 crop_shape=(76, 76),
                 imagenet_norm: bool = True,
                 proj_dim: int = 64):
        super().__init__()
        self.model_name = model_name
        self.proj_dim = int(proj_dim)

        backbone, source = self._load_backbone(model_name)

        # 去掉最后一层(avgpool + fc),只留 conv feature
        if isinstance(backbone, torchvision.models.ResNet):
            backbone.fc = nn.Identity()
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            self.backbone = backbone
        feature_dim = 512  # resnet18 last conv channel

        _replace_bn_with_gn(self.backbone)

        # preprocess
        ch, cw = crop_shape
        train_t = [
            torchvision.transforms.RandomCrop(size=(ch, cw), padding=4),
            torchvision.transforms.Resize(size=(ch, cw), antialias=True),
        ]
        eval_t = [
            torchvision.transforms.CenterCrop(size=(ch, cw)),
            torchvision.transforms.Resize(size=(ch, cw), antialias=True),
        ]
        if imagenet_norm:
            train_t.append(torchvision.transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD))
            eval_t.append(torchvision.transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD))
        self.train_preprocess = nn.Sequential(*train_t)
        self.eval_preprocess = nn.Sequential(*eval_t)

        self.proj = nn.Linear(feature_dim, proj_dim)
        logger.info("R3MObsEncoder built: model=%s, source=%s, proj_dim=%d",
                    model_name, source, proj_dim)

    def _load_backbone(self, model_name: str):
        """尝试加载 R3M 预训练 backbone,失败 fallback 到 torchvision。"""
        try:
            import r3m
            r3m.device = "cpu"
            model = r3m.load_r3m(model_name)
            r3m_model = model.module
            r3m_model = r3m_model.convnet
            r3m_model = r3m_model.to("cpu")
            return r3m_model, "r3m"
        except Exception as e:
            logger.warning("R3M 加载失败: %s. Fallback 到 torchvision %s(无 R3M 预训练权重).",
                           e, model_name)
            backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            return backbone, "torchvision_fallback"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, proj_dim)"""
        if self.training:
            img = self.train_preprocess(img)
        else:
            img = self.eval_preprocess(img)
        feat = self.backbone(img)
        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        else:
            feat = feat.mean(dim=1)
        return self.proj(feat)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)
