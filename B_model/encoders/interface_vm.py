# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""契约 2:VisionEncoderInterface —— VM 抽象契约(单图版)。

v5-1 简化后:所有 VM 子类**必须**实现 `forward(img: Tensor[B, C, H, W]) -> Tensor[B, D]`。
多相机由 Adapter 沿 batch 维 cat 处理(本接口不感知多相机)。

output_shape() 报告**单相机**输出维度 `(D,)`。
Policy 在 __init__ 时调 output_shape() 决定 adapter 维度。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class VisionEncoderInterface(nn.Module):
    """图像编码器(单图接口):输入 (B, C, H, W),输出 (B, D)。"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, img):
        """编码单张 / 单 batch 的图像。

        Args:
            img: (B, C, H, W) Tensor(B=batch,C=channels,H=height,W=width)
        Returns:
            (B, D) Tensor —— 单相机输出特征
        """
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """返回单相机输出 shape tuple,如 (D,)。Policy 用此决定 Adapter fusion 维度。"""
        raise NotImplementedError