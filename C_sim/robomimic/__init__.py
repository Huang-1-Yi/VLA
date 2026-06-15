# -*- coding: utf-8 -*-
# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""C_sim.robomimic —— 仿真边界接口层。

调用方 (E_cti / Gpolicy) 通过:
    from C_sim.robomimic import (
        BaseRobomimicEnv, make_robomimic_env, register_robomimic_factory,
    )
不直接 import 任何 F_envs/robomimic 的实现。
"""
from C_sim.robomimic.interface_robomimic_env import (  # noqa: F401
    BaseRobomimicEnv, make_robomimic_env, register_robomimic_factory,
)
