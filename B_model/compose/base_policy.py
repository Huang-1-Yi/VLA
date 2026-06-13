"""契约 2:BaseVLAPolicy —— 所有 Policy 的统一基类。"""
from abc import abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn

from A_common.types.action_output import ActionOutput


class BaseVLAPolicy(nn.Module):
    """所有 policy 的统一接口。采/训/推外围代码只跟这个接口打交道。

    关键修正(v4):不包含 `compute_loss`!loss 在 G_algo 计算,
                `forward` 只暴露"加噪轨迹 + t + context → 预测输出"的纯前向。
    """

    def __init__(self):
        super().__init__()
        self._normalizer = None

    @abstractmethod
    def encode_inputs(self, obs_dict: Dict[str, torch.Tensor],
                       prompt: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """把 obs 编码成 features dict。"""
        raise NotImplementedError

    def forward(self, a_t: torch.Tensor, t: torch.Tensor,
                encoded: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """G_algo.compute_loss 调用:已知 features + noisy action + t,出 denoiser 预测。

        训练时返回 pred(噪声预测 / 样本预测,看 pred_type)
        推理时由 G_algo 走 denoise loop 调多次,本函数每次都返回一步预测
        """
        raise NotImplementedError

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                        prompt: Optional[str] = None) -> ActionOutput:
        """高层 API:由 G_algo 编排(denoise loop)调,本基类给个默认实现。"""
        raise NotImplementedError

    def reset(self):
        """推理 episode 切换时清空状态。"""
        pass

    def set_normalizer(self, normalizer):
        self._normalizer = normalizer

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
