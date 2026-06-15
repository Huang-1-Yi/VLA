# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""契约 5:DiffusionNetworkInterface —— DM 抽象契约。

v5-1 §13 现行规范:每个 interface 单文件,最细粒度。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class DiffusionNetworkInterface(nn.Module):
    """动作去噪网络。吃加噪动作 + 条件,吐去噪预测。

    DM 的 sample 输入/输出维度相同(D_a),由 __init__ 的 input_dim 决定。
    """

    def __init__(self):
        super().__init__()
        # 子类继续:接收 input_dim / global_cond_dim / local_cond_dim 等参数

    @abstractmethod
    def forward(self, sample: torch.Tensor, **cond) -> torch.Tensor:
        """输入加噪动作 [B, H, D_a] + 由 Adapter 投影好的全局条件,输出预测 [B, H, D_a]。"""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple:
        """返回 sample 输入/输出的最后一维 (D_a,)。"""
        raise NotImplementedError

    def shape_info(self) -> str:
        """返回 DM 的形状摘要字符串,供调试用。"""
        return f"sample=({self.output_shape()[-1]},)"