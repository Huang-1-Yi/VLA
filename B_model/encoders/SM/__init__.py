# -*- coding: utf-8 -*-
"""B_model.encoders.SM —— Stage Model 多相机编码器族(阶段 2/3 启用)。

PADP 对应源码(参考):
  - diffusion_policy/model/vision/stage_timm_obs_encoder.py
  - diffusion_policy/model/vision/stage_multi_image_obs_encoder_onehot.py

特点(阶段 2/3 补全):
  - 多相机 backbone(per-key deepcopy,concat 沿 feature 维)
  - one-hot stage 标签作为额外输入
  - 适配任务分段训练

VLA 阶段 1:**占位目录,不写实现**。
"""
# 阶段 1 不写任何 class / function
__all__ = []
