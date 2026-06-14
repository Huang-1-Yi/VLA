"""契约 3:ActionEncoderInterface —— AM 抽象契约。

v5-1 §13 现行规范:每个 interface 单文件,最细粒度。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class ActionEncoderInterface(nn.Module):
    """历史动作 / 低维状态编码器。"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入历史动作 [B, T, D_a] 或低维状态 [B, T, D_s],输出嵌入。"""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple:
        """返回编码后特征 shape。"""
        raise NotImplementedError