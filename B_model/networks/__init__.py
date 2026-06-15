# -*- coding: utf-8 -*-
# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""B_model.networks —— 核心主干(本阶段只有 DM/Unet1D)。

v5-1 偏离版:DM 的 abstract interface 跟消费者同包(本目录的
DM/unet1d_padp.py 直接 import 这里)。原 v5-1 把 DiffusionNetworkInterface
放 A_common/types/,本目录实施时改放这里。

导入顺序(避免循环 import):
  1) 先把 abstract interface 从 interface_dm.py re-export
  2) 再触发子包 DM(子包内的 Unet1DPadp 继承 DiffusionNetworkInterface)
"""
# 1) re-export abstract interface
from B_model.networks.interface_dm import DiffusionNetworkInterface

# 2) 触发子包 DM
from B_model.networks import DM  # noqa: F401

__all__ = [
    "DiffusionNetworkInterface",
    "DM",
]