"""B_model.adapters.base_adapter —— Adapter 抽象基类。

Adapter 在 B_model/adapters/<algo>_adapter.py 继承本类。
Adapter 内部实例化 VM/AM,并做特征融合投影,产出 DM 期望的 global_cond。

由 Gpolicy/<algo>/<algo>_policy.py 在 __init__ 实例化。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class BaseAdapter(nn.Module):
    """所有算法 Adapter 的基类。"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs) -> torch.Tensor:
        """输入 obs(dict),输出 global_cond [B, global_cond_dim]。"""
        raise NotImplementedError