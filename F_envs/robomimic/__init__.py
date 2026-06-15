# -*- coding: utf-8 -*-
# ============================================================
# F_envs.robomimic —— 自包含 robomimic 仿真环境 + TCP 客户端
# ============================================================

"""F_envs.robomimic —— 真实 robomimic 0.3.0 仿真 + VLA server TCP 客户端。

**模块自包含**(仅依赖 VLA 之外的纯协议):
  - `robomimic_image_env.py`:robomimic 0.3.0 env 包装(继承 BaseRobomimicEnv)
  - `tcp_rollout_client.py`:连 VLA server,跑 n_train+n_test ep
  - `make_env.py`:工厂函数 + 注册到 C_sim
  - `env_meta.py`:任务元信息(TASK_MAX_STEPS, get_env_meta)
  - `wrappers.py`:可选的 gym 包装

**用法**:
  ```python
  from F_envs.robomimic import make_robomimic_env, PadpRolloutClient
  env = make_robomimic_env(task_name="square", shape_meta=..., max_steps=400)
  ```
"""
# 顺序很重要:先 import env_meta(纯数据),再 import make_env(注册),
# 最后 import tcp_rollout_client(可选,启动时才用)
from F_envs.robomimic.env_meta import (
    TASK_MAX_STEPS,
    TASK_DESCRIPTIONS,
    get_env_meta,
    infer_task_name_from_dataset,
)
from F_envs.robomimic.robomimic_env import RobomimicEnv, remap_env_name
from F_envs.robomimic.make_env import make_robomimic_env
from F_envs.robomimic.tcp_rollout_client import (
    PadpRolloutClient,
    env_obs_to_msgpack_obs,
)

# 触发 register_robomimic_factory(import 时副作用)
import F_envs.robomimic.make_env  # noqa: F401

__all__ = [
    "RobomimicEnv",
    "make_robomimic_env",
    "PadpRolloutClient",
    "TASK_MAX_STEPS",
    "TASK_DESCRIPTIONS",
    "get_env_meta",
    "infer_task_name_from_dataset",
    "env_obs_to_msgpack_obs",
    "remap_env_name",
]
