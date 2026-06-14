"""B_model.encoders.VM.transformer_obs_encoder —— DINO/CLIP-based ViT 单图像编码器(语义特征强,阶段 2/3 启用)。

════════════════════════════════════════════════════════════════════════
安装命令(使用本类时按需安装,**不进** F_envs/base_train/environment.yml):
  # pip install peft         # LoRA 启用时需要;未启用可不装
  # (transformers / timm 已在 base_train/environment.yml 中)
  #
  # 注:本类需要从 HuggingFace / OpenAI 下载 ViT 预训练权重,
  #     第一次跑会下载数 GB,需要联网。
════════════════════════════════════════════════════════════════════════

源:参考 PADP `model/vision/ds_transformer_obs_encoder.py:TransformerObsEncoder`
      (model_name 默认 'vit_base_patch16_clip_224.openai')。
特点:用 DINOv2 / CLIP 预训练的 ViT,特征含丰富语义信息,适合任务级泛化。

════════════════════════════════════════════════════════════════════════
默认 config 注释(参考 PADP config/robomimic_padp_position_wise_dinov3.yaml):
  # model_name:    "vit_base_patch16_clip_224.openai"
  #                # 也可:
  #                #   "vit_small_patch16_dinov3"
  #                #   "vit_base_patch16_dino"
  #                #   "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"
  # pretrained:     True
  # frozen:         False                       # 全量微调;True 时锁 backbone
  # global_pool:    ''                          # timm: '' = no pool
  # use_lora:       False                       # True 时启用 peft LoRA(per PADP)
  # lora_rank:      8
  # proj_dim:       64
  # crop_shape:     [76, 76]                    # ViT 必须 Resize 到 224(见 forward 内)
════════════════════════════════════════════════════════════════════════
"""
import logging

import timm
import torch
import torch.nn as nn
import torchvision

from B_model.encoders.interface_vm import VisionEncoderInterface

logger = logging.getLogger(__name__)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformerObsEncoder(VisionEncoderInterface):
    """DINO/CLIP-based ViT 单图像编码器,吃 (B, C, H, W) → (B, proj_dim)。"""

    def __init__(self,
                 model_name: str = "vit_base_patch16_clip_224.openai",
                 pretrained: bool = True,
                 frozen: bool = False,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 crop_shape=(224, 224),
                 proj_dim: int = 64):
        super().__init__()
        self.model_name = model_name
        self.proj_dim = int(proj_dim)

        # backbone(timm 加载,global_pool='' 保留所有 token 或 CLS)
        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool="",  # 不做 pool
            num_classes=0,  # 去掉 classification head
        )

        if frozen:
            assert pretrained, "frozen=True 必须 pretrained=True"
            for p in self.backbone.parameters():
                p.requires_grad = False

        # LoRA(可选,use peft)
        if use_lora:
            self._inject_lora(lora_rank)

        # 探测 feature dim
        with torch.no_grad():
            mock = torch.zeros(1, 3, *crop_shape)
            mock_out = self.backbone(mock)
            # ViT 输出: (B, num_tokens, embed_dim) 或 (B, embed_dim)
            if mock_out.dim() == 3:
                # 取 CLS token(第 0 个)或 mean pool
                feat = mock_out.mean(dim=1)  # (1, embed_dim)
            else:
                feat = mock_out
            feature_dim = feat.shape[-1]

        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, proj_dim)

        # ViT 通常要求 224x224 输入;preprocess 做 Resize+Normalize
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
        self.train_preprocess = nn.Sequential(*train_t)
        self.eval_preprocess = nn.Sequential(*eval_t)

        logger.info("TransformerObsEncoder built: model=%s, pretrained=%s, frozen=%s, "
                    "use_lora=%s, feature_dim=%d, proj_dim=%d",
                    model_name, pretrained, frozen, use_lora, feature_dim, proj_dim)

    def _inject_lora(self, lora_rank: int):
        """用 peft 给 ViT attention 注入 LoRA(per PADP `ds_timm_obs_encoder.py:use_lora`)。"""
        try:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["qkv"],  # timm ViT 的 attention qkv 线性层
                lora_dropout=0.05,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, config)
            logger.info("LoRA injected: rank=%d, target=qkv", lora_rank)
        except ImportError as e:
            logger.warning("peft 未装,LoRA 注入跳过: %s. (按需 pip install peft)", e)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, proj_dim)"""
        if self.training:
            img = self.train_preprocess(img)
        else:
            img = self.eval_preprocess(img)
        raw = self.backbone(img)
        # ViT 输出 (B, num_tokens, embed_dim) → mean pool
        if raw.dim() == 3:
            feat = raw.mean(dim=1)
        else:
            feat = raw
        return self.proj(feat)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)
