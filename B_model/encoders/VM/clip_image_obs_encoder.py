"""B_model.encoders.VM.clip_image_obs_encoder —— CLIP 单图像编码器(单相机版)。

源:参考 PADP `model/task_padp/clip_multi_image_obs_encoder_clip.py` 但简化为只吃单图。

与 CLIPMultiImageObsEncoder 的区别:
- `CLIPImageObsEncoder`(本文件):吃单张图 `(B, C, H, W)`,输出 `(B, D)`
- `CLIPMultiImageObsEncoder`:吃 dict 或 batched `(B, N, C, H, W)`,内部聚合为 `(B, D)`

两者 output_shape() 相同 `(D,)`,policy 通过 output_shape() 自动适配下游 dim,
不依赖具体选哪种 encoder。
"""
import logging

import torch
import torch.nn as nn
import torchvision

from B_model.encoders.interface_vm import VisionEncoderInterface

logger = logging.getLogger(__name__)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPImageObsEncoder(VisionEncoderInterface):
    """单图像 CLIP 编码器:吃一张图 `(B, C, H, W)`,输出 `(B, proj_dim)`。

    用法:每个相机一个实例,Adapter 把 N 个实例的输出 concat 为 `(B, N*proj_dim)`。
    """

    def __init__(self,
                 clip_model_name: str = "ViT-B/32",
                 clip_freeze: bool = True,
                 clip_normalize_features: bool = True,
                 proj_dim: int = 512):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip_freeze = clip_freeze
        self.clip_normalize_features = clip_normalize_features
        self.proj_dim = int(proj_dim)

        try:
            import clip
            self._clip_available = True
            # 强制 CPU 加载,避免设备不一致
            model, _ = clip.load(clip_model_name, device="cpu", jit=False)
            self.clip_model = model.visual.float()
            if clip_freeze:
                for p in self.clip_model.parameters():
                    p.requires_grad = False
            self.clip_output_dim = 512  # ViT-B/32 默认
        except Exception as e:
            logger.warning("CLIP load failed: %s. Fallback to identity + global avg pool.", e)
            self._clip_available = False
            self.clip_model = nn.Identity()
            # === 修复 B: fallback 时 in_dim=3(图像 RGB 三通道,经 spatial mean pool 后)===
            # 而不是 in_dim=proj_dim(那是输出维,不能当下游 proj 的输入维)
            self.clip_output_dim = 3

        # CLIP 标准预处理:Resize → CenterCrop → Normalize
        self.preprocess = nn.Sequential(
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        )

        # === 修复 B:in_dim 必须等于 backbone 输出维(ViT-B/32 输出 512,fallback 输出 3)===
        in_dim = self.clip_output_dim
        self.proj = nn.Linear(in_dim, proj_dim)

        logger.info("CLIPImageObsEncoder built: clip=%s, available=%s, "
                    "in_dim=%d, proj_dim=%d, params=%.3e",
                    clip_model_name, self._clip_available, in_dim, proj_dim,
                    sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, proj_dim)"""
        # === 形状:input (B, C, H, W) → output (B, D_vm) ===
        # === D_vm = self.proj_dim(默认 512,见 __init__)===
        # === 注意:CLIP preprocess 强制 Resize 到 224x224 ===
        img = self.preprocess(img)
        if not self._clip_available:
            # === 修复 B:fallback 路径 ===
            # Identity 不会降维 → 必须先 spatial mean pool 成 (B, C=3) 再投影
            raw = img.mean(dim=(2, 3))  # (B, 3, 224, 224) → (B, 3)
        else:
            with torch.no_grad() if self.clip_freeze else torch.enable_grad():
                raw = self.clip_model(img)
            if self.clip_normalize_features:
                raw = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
            # === 鲁棒性:CLIP 输出可能是 (B, D) 或 (B, N, D) 或 (B, D, h, w) ===
            if raw.dim() == 4:        # spatial feature map
                raw = raw.mean(dim=(2, 3))  # (B, D, h, w) → (B, D)
            elif raw.dim() == 3:      # (B, N, D) patch tokens → mean pool
                raw = raw.mean(dim=1)      # (B, N, D) → (B, D)
            # raw.dim() == 2: (B, D) already, no-op
        return self.proj(raw)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)
