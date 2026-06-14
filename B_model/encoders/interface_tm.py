"""契约 4:TimestepEncoderInterface —— TM 抽象契约。

v5-1 §13 现行规范:每个 interface 单文件,最细粒度。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class TimestepEncoderInterface(nn.Module):
    """扩散时间步编码器。"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """输入 [B] 时间步,输出 [B, D_h] 嵌入。"""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple:
        """返回编码后特征 shape。"""
        raise NotImplementedError