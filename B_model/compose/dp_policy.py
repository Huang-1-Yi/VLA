"""DpUnetPolicy —— baseline DP,继承 PadpUnetPolicy 但强制 pred_type='epsilon' + no window weights。

实际差异只在 G_algo 那边(DP pipeline 不加位置权重),B_model 这层结构完全相同。
"""
from A_common.registry.policy_registry import register_policy
from B_model.compose.padp_policy import PadpUnetPolicy


@register_policy("dp_unet")
class DpUnetPolicy(PadpUnetPolicy):
    """完全继承 PadpUnetPolicy。区别仅在 G_algo pipeline 是否加位置加权。

    保留独立 class 名,方便:
        1. policy registry 区分
        2. 未来 DP 专属改动有空间
    """
    pass
