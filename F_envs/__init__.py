# -*- coding: utf-8 -*-
# ============================================================
# F_envs —— 环境管理(仿真 / 真机的统一入口)
# ============================================================

"""F_envs —— 环境管理层。

子模块:
  - F_envs.robomimic  robomimic 0.3.0 仿真 + VLA server TCP 客户端
  - F_envs.libero     (TODO)LIBERO 仿真
  - F_envs.real_robot (TODO)真机
"""
from F_envs import robomimic  # noqa: F401

__all__ = ["robomimic"]
