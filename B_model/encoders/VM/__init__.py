# -*- coding: utf-8 -*-
"""B_model.encoders.VM —— Vision Model 编码器。

阶段 1:用 RobomimicObsEncoder 包装(沿用 PADP 工作版本,无 LoRA / 无 DINOv2)。
"""
from .robomimic_obs_encoder import RobomimicObsEncoder

__all__ = ["RobomimicObsEncoder"]
