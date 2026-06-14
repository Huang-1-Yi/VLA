"""B_model.encoders.TM.sinusoidal —— SinusoidalPosEmb + MLP 包装。

源:抄自 PADP `diffusion_policy/model/diffusion/positional_embedding.py:12-27`
    SinusoidalPosEmb + `conditional_unet1d.py:154-159` 的 MLP 包装
    (SinusoidalPosEmb → Linear → Mish → Linear)。

v5-1 §铁律 3:继承 B_model.encoders.interface_tm.TimestepEncoderInterface。
"""
import logging
import math
import torch
import torch.nn as nn

from B_model.encoders.interface_tm import TimestepEncoderInterface

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """无 learnable param 的 sinusoidal 投影。输入 [B] timestep,输出 [B, dim]。"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"SinusoidalPosEmb dim must be even, got {dim}"
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B] int/float timestep → [B, dim] sinusoidal embedding"""
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SinusoidalTimestepEncoder(TimestepEncoderInterface):
    """标准 diffusion timestep 编码:SinusoidalPosEmb → Linear → Mish → Linear。

    继承 TimestepEncoderInterface,实现 forward + output_shape。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        # === L1b:__init__ 末尾 logger.info(MLP 维度摘要 + 参数量)===
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("SinusoidalTimestepEncoder built: dim=%d (SinusoidalPosEmb → Linear(%d→%d) → Mish → Linear(%d→%d)), params=%.3e",
                    self.dim, self.dim, self.dim * 4, self.dim * 4, self.dim, total_params)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] timestep → [B, dim]"""
        # === 形状:input (B,) timestep → output (B, D_tm) ===
        # === D_tm = self.dim(默认 512,见 __init__)===
        return self.encoder(t)

    def output_shape(self) -> tuple:
        return (self.dim,)
