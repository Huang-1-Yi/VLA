# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""Gpolicy —— 策略大脑(Fat Policy)层。"""
# re-export BasePolicy
from Gpolicy.base_policy_abstract import BasePolicy

# 触发子包(触发 @register_policy 注册 padp_unet)
from Gpolicy import PADP  # noqa: F401

__all__ = [
    "BasePolicy",
    "PADP",
]
