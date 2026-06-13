"""PadpUnetPolicy —— PADP 的 SlidingWindowDiffusionPolicy,套 BaseVLAPolicy 接口。

源:抄自 PADP `diffusion_policy/policy/robomimic/diffusion_unet_hybrid_padp.py`,
   删去 Hydra 反射,改为显式构造参数。
   核心创新(position-aware noise + sliding window + 预对齐 buffer)完整保留。
"""
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from A_common.registry.policy_registry import register_policy
from A_common.types.normalizer import LinearNormalizer
from A_common.types.action_output import ActionOutput
from A_common.logger import get_logger

from B_model.compose.base_policy import BaseVLAPolicy
from B_model.networks.DM.unet1d_padp import Unet1DPadp
from B_model.encoders.VM import RobomimicObsEncoder

logger = get_logger(__name__)


def _build_horizon_alpha_bar(horizon: int, beta_schedule: str = "squaredcos_cap_v2"):
    """预计算 PADP 位置感知余弦表,长度 horizon+1,索引 0=纯动作,索引 horizon=纯高斯。"""
    if beta_schedule != "squaredcos_cap_v2":
        raise NotImplementedError(beta_schedule)
    s = 0.008
    t = torch.arange(0, horizon + 1, dtype=torch.float32)
    alpha_bar = torch.cos((t / horizon + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha_bar[0] = 1.0
    alpha_bar[-1] = 0.0
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    return alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar


def _apply_position_noise(original_samples, noise, sqrt_alpha_bar_h,
                          sqrt_one_minus_alpha_bar_h):
    """positionwise 模式:O(1) 广播加噪。"""
    return sqrt_alpha_bar_h * original_samples + sqrt_one_minus_alpha_bar_h * noise


@register_policy("padp_unet")
class PadpUnetPolicy(BaseVLAPolicy):
    """PADP 位置感知扩散策略。

    Args:
        shape_meta:   {obs: {key: {shape, type}}, action: {shape}}
        obs_encoder:  RobomimicObsEncoder 实例
        horizon:      预测窗口
        n_obs_steps:  历史观测帧数
        n_action_steps: 每次推理输出的 action 步数
        down_dims:    UNet 通道
        kernel_size:  卷积核
        n_groups:     GroupNorm 组数
        cond_predict_scale: FiLM 风格(scale+shift)
        noise_schedule_mode: 'positionwise' 唯一支持
        window_loss_weights: 'exponential' / 'linear' / 'constant'
        window_exp_gamma: 指数衰减系数
        window_min_weight: 权重最小值(防止远端动作 loss=0)
        pred_type:    'epsilon'(预测噪声) / 'sample'(预测 x0)
    """

    def __init__(self, shape_meta: dict,
                 obs_encoder: Optional[nn.Module] = None,
                 horizon: int = 40, n_action_steps: int = 8,
                 n_obs_steps: int = 1, num_inference_steps: int = 40,
                 down_dims=(256, 512, 1024), kernel_size: int = 5,
                 n_groups: int = 8, cond_predict_scale: bool = True,
                 window_loss_weights: str = "exponential",
                 window_exp_gamma: float = 0.2,
                 window_min_weight: float = 0.02,
                 pred_type: str = "sample", **kwargs):
        super().__init__()

        self.pred_type = pred_type

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # === bug fix 1: 漏传 3 个 obs_encoder kwarg(PADP yaml 里显式给) ===
        crop_shape = kwargs.pop("crop_shape", (76, 76))
        obs_encoder_group_norm = kwargs.pop("obs_encoder_group_norm", True)
        eval_fixed_crop = kwargs.pop("eval_fixed_crop", True)

        if obs_encoder is None:
            obs_encoder = RobomimicObsEncoder(
                shape_meta=shape_meta,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )
        self.obs_encoder = obs_encoder

        obs_shape = obs_encoder.output_shape()
        # 兼容 encoder output_shape 格式:1D tuple (D,) 或 2D tensor (1, D)
        if hasattr(obs_shape, '__len__') and len(obs_shape) >= 1:
            try:
                obs_feature_dim = int(obs_shape[-1])
            except (TypeError, IndexError):
                obs_feature_dim = int(obs_shape[0])
        else:
            obs_feature_dim = int(obs_shape)

        global_cond_dim = obs_feature_dim * n_obs_steps
        logger.info("global_cond_dim=%d (obs_feature_dim=%d × n_obs_steps=%d)",
                     global_cond_dim, obs_feature_dim, n_obs_steps)

        self.denoiser = Unet1DPadp(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # 预计算 PADP 余弦表
        _, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = _build_horizon_alpha_bar(horizon)
        self.register_buffer('sqrt_alpha_bar_h', sqrt_alpha_bar[1:horizon + 1].view(1, horizon, 1))
        self.register_buffer('sqrt_one_minus_alpha_bar_h', sqrt_one_minus_alpha_bar[1:horizon + 1].view(1, horizon, 1))
        self.register_buffer('sqrt_one_minus_alpha_bar_tail', sqrt_one_minus_alpha_bar[horizon:horizon + 1].view(1, 1, 1))

        # 窗口权重
        if window_loss_weights == "linear":
            window_weights = 1.0 - torch.arange(horizon, dtype=torch.float32) / horizon
        elif window_loss_weights == "exponential":
            window_weights = torch.exp(-torch.arange(horizon, dtype=torch.float32) * window_exp_gamma)
        elif window_loss_weights == "constant":
            window_weights = torch.ones(horizon)
        else:
            raise ValueError(f"Unknown window loss weights: {window_loss_weights}")
        if window_min_weight > 0.0:
            window_weights = torch.clamp(window_weights, min=window_min_weight)
        self.register_buffer('window_weights_h', window_weights.view(1, horizon, 1))

        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.obs_feature_dim = obs_feature_dim

        # 推理 buffer
        self._inference_buffer: Optional[torch.Tensor] = None
        self._inference_global_cond: Optional[torch.Tensor] = None

        logger.info("PadpUnetPolicy built. UNet params=%.3e, Vision params=%.3e",
                     sum(p.numel() for p in self.denoiser.parameters()),
                     sum(p.numel() for p in self.obs_encoder.parameters()))

    # ============== G_algo 训练入口 ==============
    def encode_inputs(self, obs_dict, prompt=None):
        """取最近 n_obs_steps 帧,VM 编码,reshape 成 [B, n_obs*D]。"""
        n = self.normalizer.normalize(obs_dict)
        # 取前 n_obs_steps 帧
        this_nobs = {k: v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:]) for k, v in n.items()}
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features  # [B*n_obs, D]

    def forward(self, a_t, t, encoded, **kwargs):
        """G_algo.compute_loss 调用。

        a_t:     [B, H, D_a]
        t:       [B]
        encoded: [B*n_obs, D]  (从 encode_inputs)
        return:  [B, H, D_a]
        """
        B = a_t.shape[0]
        global_cond = encoded.reshape(B, -1)
        return self.denoiser(a_t, local_cond=None, global_cond=global_cond)

    # ============== 推理入口 ==============
    def _init_inference_buffer(self, obs_dict):
        B = next(iter(obs_dict.values())).shape[0]
        H, D = self.horizon, self.action_dim
        base_noise = torch.randn(B, H, D, device=self.device, dtype=self.dtype)
        # 用零动作 + 同样 noise schedule 初始化 buffer
        with torch.no_grad():
            self._inference_buffer = _apply_position_noise(
                torch.zeros_like(base_noise), base_noise,
                self.sqrt_alpha_bar_h, self.sqrt_one_minus_alpha_bar_h,
            )
        encoded = self.encode_inputs(obs_dict)
        self._inference_global_cond = encoded.reshape(B, -1)

    def predict_action(self, obs_dict, prompt=None) -> ActionOutput:
        """sliding-window 一步推理。

        实现 PADP 的 inference buffer 左移 + 末位注入新噪声。
        返回 [B, n_action_steps, D_a] 动作 + is_chunk=True,latency_ms。
        """
        import time
        t0 = time.time()
        B = next(iter(obs_dict.values())).shape[0]

        if (self._inference_buffer is None
                or self._inference_buffer.shape[0] != B):
            self._init_inference_buffer(obs_dict)
        else:
            encoded = self.encode_inputs(obs_dict)
            self._inference_global_cond = encoded.reshape(B, -1)

        H, D = self.horizon, self.action_dim

        # 单步去噪:用 buffer 作为 x0 估计
        x0 = self._inference_buffer
        noise = torch.randn_like(x0)
        noisy = _apply_position_noise(
            x0, noise, self.sqrt_alpha_bar_h, self.sqrt_one_minus_alpha_bar_h,
        )
        pred = self.denoiser(noisy, local_cond=None, global_cond=self._inference_global_cond)

        if self.pred_type == 'epsilon':
            pred_actions = noisy - pred
        else:  # 'sample'
            pred_actions = pred

        # 取第一步动作,左移 buffer
        action_to_execute = pred_actions[:, 0:self.n_action_steps, :]  # [B, n_act, D]
        self._inference_buffer[:, :H - 1, :] = pred_actions[:, 1:, :]
        new_noise = torch.randn(B, 1, D, device=self.device, dtype=self._inference_buffer.dtype)
        self._inference_buffer[:, H - 1:, :] = self.sqrt_one_minus_alpha_bar_tail * new_noise

        # 反归一化
        if self._normalizer is not None and "action" in self._normalizer:
            action_to_execute = self._normalizer["action"].unnormalize(action_to_execute)

        latency_ms = (time.time() - t0) * 1000.0
        return ActionOutput(
            actions=action_to_execute,
            is_chunk=(self.n_action_steps > 1),
            latency_ms=latency_ms,
        )

    def reset(self):
        self._inference_buffer = None
        self._inference_global_cond = None
