"""契约 4:TimestepEncoderInterface —— TM(扩散时间步编码器)统一接口。

铁律 3:本抽象类必须放在 A_common/types/。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class TimestepEncoderInterface(nn.Module):
    """扩散时间步编码器。"""

    def __init__(self):
        super().__init__()()

    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """输入 [B] 时间步,输出 [B, D_h] 嵌入。"""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple:
        """返回编码后特征 shape。"""
        raise NotImplementedError