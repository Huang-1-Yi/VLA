"""B_model.encoders.AM.linear_state_encoder —— 简单 nn.Linear 投影 state。

PADP 现状:state 直接 `.reshape(B, -1)` 然后 cat(没有任何专门的 AM)。
VLA 改进:把 state 线性投影到 hidden_dim,让 Adapter 拿到独立 state 投影向量,
可以做更精细的 fusion(比如 cross-attention)。

本类继承 B_model.encoders.interface_am.ActionEncoderInterface,实现 forward + output_shape。
"""
import logging

import torch
import torch.nn as nn

from B_model.encoders.interface_am import ActionEncoderInterface

logger = logging.getLogger(__name__)


class LinearStateEncoder(ActionEncoderInterface):
    """把任意维度 state 投影到 output_dim。

    输入 shape:[B, T, D_in](B batch,T 时间步,D_in 输入维度)
    输出 shape:[B, T, D_out]
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        # === L1b:__init__ 末尾 logger.info(参数 + 维度摘要)===
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("LinearStateEncoder built: input_dim=%d, output_dim=%d, params=%.3e",
                    self.input_dim, self.output_dim, total_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D_in] → [B, T, D_out]"""
        # === 形状:input (B, T_obs, D_in) → output (B, T_obs, D_am) ===
        # === D_am = self.output_dim(由 cfg 决定,典型 64)===
        # === 不缩减 T_obs 维(供 Adapter 保持时间窗一致)===
        return self.proj(x)

    def output_shape(self) -> tuple:
        return (self.output_dim,)
