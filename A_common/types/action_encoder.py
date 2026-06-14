"""契约 3:ActionEncoderInterface —— AM(历史动作编码器)统一接口。

铁律 3:本抽象类必须放在 A_common/types/。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class ActionEncoderInterface(nn.Module):
    """历史动作序列编码器。"""

    def __init__(self):
        super().__init__()()

    @abstractmethod
    def forward(self, action_seq: torch.Tensor) -> torch.Tensor:
        """输入 [B, T, D_a] 历史动作,输出嵌入。"""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple:
        """返回编码后特征 shape。"""
        raise NotImplementedError