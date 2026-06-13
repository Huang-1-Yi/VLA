"""G_algo.DP.pipeline —— baseline DP 的 compute_loss + predict_action。

与 PADP pipeline 的唯一区别:loss 不乘 window_weights_h(无位置加权)。
其它全部相同。
"""
import torch
import torch.nn.functional as F

from A_common.logger import get_logger
from G_algo.common.position_noise import add_position_noise

logger = get_logger(__name__)


def compute_loss(policy, batch: dict, num_train_timesteps: int = 1000,
                 noise_schedule_mode: str = "positionwise") -> torch.Tensor:
    """DP baseline compute_loss:位置加噪但**无 per-position weight**。"""
    obs = batch["obs"]
    a0 = batch["action"]

    nobs = policy.normalizer.normalize(obs)
    nactions = policy.normalizer["action"].normalize(a0)

    encoded = policy.encode_inputs(nobs)
    B = a0.shape[0]
    global_cond = encoded.reshape(B, -1)

    noise = torch.randn_like(nactions)
    noisy = add_position_noise(
        nactions, noise,
        policy.sqrt_alpha_bar_h,
        policy.sqrt_one_minus_alpha_bar_h,
        mode=noise_schedule_mode,
    )

    pred = policy(noisy, t=None, encoded=encoded)

    if policy.pred_type == "epsilon":
        target = noise
    elif policy.pred_type == "sample":
        target = nactions
    else:
        raise ValueError(policy.pred_type)

    return F.mse_loss(pred, target)


@torch.no_grad()
def predict_action(policy, obs_dict: dict, **kwargs):
    return policy.predict_action(obs_dict)
