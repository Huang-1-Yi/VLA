"""G_algo.PADP.pipeline —— PADP 算法的 compute_loss + predict_action。

源:逻辑抄自 PADP `diffusion_policy/workspace/robomimic/train_padp_workspace_v3.py` 的
   `run_step` 内部(去 workspace class / 去 EMA / 去 checkpoint 逻辑)。
"""
from typing import Dict
import torch
import torch.nn.functional as F

from A_common.logger import get_logger
from G_algo.common.position_noise import add_position_noise

logger = get_logger(__name__)


def compute_loss(policy, batch: dict, num_train_timesteps: int = 1000,
                 noise_schedule_mode: str = "positionwise",
                 window_exp_gamma: float = 0.2) -> torch.Tensor:
    """PADP 算法 compute_loss。

    1. 归一化 obs / action
    2. VM 编码 obs → global_cond
    3. positionwise 加噪
    4. UNet 前向
    5. per-position weighted MSE

    Args:
        policy:  PadpUnetPolicy 实例(已 set_normalizer)
        batch:   {'obs': dict, 'action': Tensor [B, H, D_a]}
    Returns:
        scalar loss (per-sample mean)
    """
    obs = batch["obs"]
    a0 = batch["action"]      # [B, H, D_a]

    # 1. 归一化
    nobs = policy.normalizer.normalize(obs)
    nactions = policy.normalizer["action"].normalize(a0)

    # 2. 编码 obs
    encoded = policy.encode_inputs(nobs)        # [B*n_obs, D]
    B = a0.shape[0]
    global_cond = encoded.reshape(B, -1)        # [B, n_obs*D]

    # 3. 注入噪声
    noise = torch.randn_like(nactions)
    noisy = add_position_noise(
        nactions, noise,
        policy.sqrt_alpha_bar_h,
        policy.sqrt_one_minus_alpha_bar_h,
        mode=noise_schedule_mode,
    )

    # 4. 前向预测
    pred = policy(noisy, t=None, encoded=encoded)

    # 5. 选 target
    if policy.pred_type == "epsilon":
        target = noise
    elif policy.pred_type == "sample":
        target = nactions
    else:
        raise ValueError(policy.pred_type)

    # 6. per-position weighted MSE
    loss_mse = F.mse_loss(pred, target, reduction="none")   # [B, H, D]
    loss_weighted = (loss_mse * policy.window_weights_h).sum(dim=1)  # [B, D]
    loss = loss_weighted.mean(dim=-1)                              # [B]
    return loss.mean()


@torch.no_grad()
def predict_action(policy, obs_dict: dict, **kwargs):
    """PADP 推理:走 policy.predict_action(已经在 padp_policy 实现 sliding window)。"""
    return policy.predict_action(obs_dict)
