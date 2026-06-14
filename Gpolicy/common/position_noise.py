"""G_algo.common.position_noise —— PADP 位置感知加噪(DP / PADP 共用)。

源:抄自 PADP `diffusion_policy/policy/schedulers_padp.py`。
"""
from typing import Optional
import torch


def add_position_noise(original_samples: torch.Tensor,
                       noise: torch.Tensor,
                       sqrt_alpha_bar_h: torch.Tensor,
                       sqrt_one_minus_alpha_bar_h: torch.Tensor,
                       mode: str = "positionwise",
                       **kwargs) -> torch.Tensor:
    """按 mode 加噪。

    Args:
        original_samples / noise: 形状 [B, H, D_a]
        sqrt_alpha_bar_h:  预对齐 [1, H, 1]
        sqrt_one_minus_alpha_bar_h: 预对齐 [1, H, 1]
        mode: 'positionwise' (本阶段唯一支持) / 'linear' / 'constant' / 'random' / 'chunkwise'

    Returns:
        noisy: [B, H, D_a]
    """
    if mode == "positionwise":
        return sqrt_alpha_bar_h * original_samples + sqrt_one_minus_alpha_bar_h * noise
    elif mode == "linear":
        H = original_samples.shape[1]
        device = original_samples.device
        dtype = original_samples.dtype
        alpha_bar = torch.linspace(1.0 - 1.0 / H, 0.0, H, device=device, dtype=dtype)
        sa = torch.sqrt(alpha_bar).view(1, H, 1)
        so = torch.sqrt(1.0 - alpha_bar).view(1, H, 1)
        return sa * original_samples + so * noise
    else:
        # 其它模式留接口(本阶段不实现)
        raise NotImplementedError(
            f"Noise mode '{mode}' not implemented. Use 'positionwise' or 'linear'."
        )
