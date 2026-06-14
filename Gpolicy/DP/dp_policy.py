"""Gpolicy.DP.dp_policy —— DP (baseline) Fat Policy。

DP 与 PADP 共享同一个 UNet + adapter,只是 compute_loss 无 per-position 加权。
"""
from A_common.registry.policy_registry import register_policy

from Gpolicy.PADP.padp_policy import SlidingWindowDiffusionPolicy


@register_policy("dp_unet")
class DpUnetPolicy(SlidingWindowDiffusionPolicy):
    """DP baseline:完全继承 PADP 策略大脑,只是 loss 不做位置加权。

    在 Gpolicy.DP.compute_loss 的 cfg 里传入 window_loss_weights='constant',window_exp_gamma=0,
    即可关掉 per-horizon 权重(因为 window_weights 已是 1)。
    但因为 PADP 默认 window_loss_weights='exponential',本类通过覆写 compute_loss 直接用
    平凡损失。
    """

    def __init__(self, cfg: dict):
        # 强制把 window_loss_weights 设为 constant,alpha=0(等价于无位置加权)
        cfg = dict(cfg)
        cfg["window_loss_weights"] = "constant"
        cfg["window_exp_gamma"] = 0.0
        cfg["window_min_weight"] = 0.0
        super().__init__(cfg)