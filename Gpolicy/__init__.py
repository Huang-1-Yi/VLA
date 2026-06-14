# -*- coding: utf-8 -*-
"""Gpolicy —— 策略大脑(Fat Policy)层。

v5-1 偏离版:
  - BasePolicy abstract class 跟消费者同包(本目录的 PADP/padp_policy.py、
    DP/dp_policy.py 直接 `from Gpolicy import BasePolicy` 即可)。
  - 原 v5-1 把 BasePolicy 放 A_common/types/base_policy.py,本目录实施时
    改放 Gpolicy/base_policy_abstract.py 并 re-export。

导入顺序(避免循环 import):
  1) 先把 BasePolicy 从 base_policy_abstract.py re-export 到本模块
  2) 再触发子包 PADP / DP(它们的 *_policy.py 继承 BasePolicy)
"""
# 1) re-export BasePolicy
from Gpolicy.base_policy_abstract import BasePolicy

# 2) 触发子包(触发 @register_policy 注册)
from Gpolicy import PADP  # noqa: F401  (注册 padp_unet)
from Gpolicy import DP    # noqa: F401  (注册 dp_unet)

__all__ = [
    "BasePolicy",
    "PADP", "DP",
]