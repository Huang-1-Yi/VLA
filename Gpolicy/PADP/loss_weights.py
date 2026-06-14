"""Gpolicy.PADP.loss_weights —— per-horizon 权重曲线工具函数。

(本阶段 Fat Policy 直接在 __init__ 里构造 window_weights,本文件作为工具预留)
"""
import torch


def padp_loss_weights(horizon: int, decay: str = "exp", alpha: float = 0.5) -> torch.Tensor:
    """per-horizon weight curve。

    Args:
        horizon: 窗口长度 H
        decay: 'exp' / 'linear' / 'const'
        alpha: 衰减系数
    """
    if decay == "exp":
        return torch.exp(-torch.arange(horizon, dtype=torch.float32) * alpha)
    if decay == "linear":
        return 1.0 - torch.arange(horizon, dtype=torch.float32) * alpha / max(horizon - 1, 1)
    if decay == "const":
        return torch.ones(horizon)
    raise ValueError(f"Unknown decay: {decay}")