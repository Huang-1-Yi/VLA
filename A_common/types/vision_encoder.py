"""契约 2:VisionEncoderInterface —— VM(图像编码器)统一接口。

铁律 3:本抽象类必须放在 A_common/types/。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class VisionEncoderInterface(nn.Module):
    """图像编码器。输入 obs_dict(标准 lerobot 格式),输出特征 Tensor。

    Policy 在 __init__ 时调 output_shape() 决定 adapter 维度。
    """

    def __init__(self):
        super().__init__()  # 子类继续 super().__init__() 完成参数注册

    @abstractmethod
    def forward(self, obs_dict: dict) -> torch.Tensor:
        """输入图像 obs(lerobot 标准),输出特征 Tensor。"""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple:
        """返回编码后特征 shape。"""
        raise NotImplementedError