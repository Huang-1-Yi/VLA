# -*- coding: utf-8 -*-
"""Gpolicy —— 策略大脑(Fat Policy)层。

v5 关键事实:
  - Policy 类在这里,继承 A_common.types.base_policy.BasePolicy
  - __init__ 实例化 B_model 组件(adapter + DM)+ 自己的 scheduler
  - 暴露 3 个必须方法:forward / compute_loss / predict_action
  - E_cti 通过 build_policy(cfg.policy) 拿到实例,只调 compute_loss / predict_action
"""
from Gpolicy import PADP  # noqa: F401  (触发 @register_policy("padp_unet"))
from Gpolicy import DP    # noqa: F401  (触发 @register_policy("dp_unet"))