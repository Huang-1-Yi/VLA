"""C_sim.robomimic.padp_for_test_make_env —— 新 robomimic 0.3.0 API 适配的 make_env。

> 为什么不复用 C_sim.robomimic.env_impl.make_env ?
>   VLA 现有 `C_sim/robomimic/env_impl/env_meta.py::get_env_meta` 调
>   `from robomimic import get_config`,但本环境装的是 robomimic 0.3.0,
>   `get_config` 已从 `robomimic` 顶层移到 `robomimic.config.config_factory`。
>   用户禁止改现有代码,故新增本文件,用新 API 重新实现 make_env。
"""
from typing import Optional
import numpy as np
import gym
from gym import spaces

import robomimic.utils.obs_utils as ObsUtils


# ---- 任务最大步数(同 C_sim/robomimic/env_impl/env_meta.py 的 TASK_MAX_STEPS)----
TASK_MAX_STEPS = {
    "square_d0": 400, "square_d1": 400, "square_d2": 400, "square_d3": 400,
    "can_d0": 400, "can_d1": 400, "can_d2": 400, "can_d3": 400,
    "lift_d0": 400, "lift_d1": 400, "lift_d2": 400, "lift_d3": 400,
    "transport_d0": 700, "transport_d1": 700,
    "tool_hang_d0": 700, "tool_hang_d1": 700,
    "stack_d0": 500, "stack_d1": 500, "stack_d2": 500, "stack_d3": 500,
    "nut_assembly_d0": 700, "nut_assembly_d1": 700, "nut_assembly_d2": 700, "nut_assembly_d3": 700,
    "pick_place_d0": 400, "pick_place_d1": 400, "pick_place_d2": 400, "pick_place_d3": 400,
    "square": 400, "can": 400, "lift": 400, "transport": 700, "tool_hang": 700,
    "stack": 500, "nut_assembly": 700, "pick_place": 400, "kitchen": 1500,
}


def get_max_steps(task_name: str) -> int:
    return TASK_MAX_STEPS.get(task_name, 400)


class RobomimicImageWrapper(gym.Env):
    """robomimic env → gym 接口的最小包装(同 VLA 现有 C_sim 实现)。"""

    def __init__(self, env, shape_meta: dict, max_steps: int = 400):
        self.env = env
        self.shape_meta = shape_meta
        self.max_steps = max_steps
        self._step = 0

        obs_space = {}
        for key, attr in shape_meta["obs"].items():
            shp = attr["shape"]
            if attr.get("type", "low_dim") == "rgb":
                obs_space[key] = spaces.Box(low=0, high=255, shape=shp, dtype=np.uint8)
            else:
                obs_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shp, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_space)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=shape_meta["action"]["shape"], dtype=np.float32,
        )

    def _get_obs(self) -> dict:
        raw = self.env.get_observation()
        out = {}
        for k in self.shape_meta["obs"]:
            out[k] = raw[k]
        return out

    def reset(self):
        self.env.reset()
        self._step = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self.max_steps:
            done = True
        # robomimic 0.3.0 的 env 有 is_success() 方法
        info["is_success"] = self.env.is_success() if hasattr(self.env, "is_success") else False
        return obs, reward, done, info


def make_env(task_name: str, shape_meta: dict, init_state: Optional[np.ndarray] = None,
             max_steps: int = 400):
    """robomimic env 的工厂(用 robomimic 0.3.0 新 API)。"""
    max_steps = max_steps or get_max_steps(task_name)

    # 新 API:robomimic.config.config_factory(替代 robomimic.get_config)
    from robomimic.config import config_factory
    from robomimic.scripts.generate_paper_configs import (
        modify_config_for_default_image_exp, modify_config_for_dataset,
    )
    config = config_factory(algo_name="bc")
    config = modify_config_for_default_image_exp(config)
    config = modify_config_for_dataset(
        config=config, task_name=task_name, dataset_type="ph",
        hdf5_type="image", base_dataset_dir="/tmp", filter_key=None,
    )

    # 初始化 obs utils
    import collections
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    # 构造 env(robomimic 0.3.0)
    from robomimic.envs.env_robosuite import EnvRobosuite

    # 🚧 robomimic 0.3.0 config 默认 lock,直接读会 RuntimeError
    # 用 config.unlocked() 上下文解锁(同 VLA RobomimicObsEncoder 的做法)
    with config.unlocked():
        env_name = config.env_name
        env_type = config.env_type
        env_kwargs = dict(config.env_kwargs)
        camera_names = list(config.observation.camera_names)
        camera_widths = list(config.observation.camera_widths)
        camera_heights = list(config.observation.camera_heights)

    env_meta = dict(
        env_name=env_name,
        env_type=env_type,
        env_kwargs=env_kwargs,
    )
    if init_state is not None:
        env_meta["init_state"] = init_state

    # robomimic 0.3.0: EnvRobosuite.create_for_data_processing(env_name, camera_names, ...)
    # 注意:env_name 必须是字符串,不能传 Config 对象(env_meta 是 dict 单独传)
    env = EnvRobosuite.create_for_data_processing(
        env_name=str(env_meta["env_name"]),
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=camera_heights[0] if camera_heights else 84,
        camera_width=camera_widths[0] if camera_widths else 84,
        reward_shaping=False,
    )

    return RobomimicImageWrapper(env, shape_meta=shape_meta, max_steps=max_steps)
