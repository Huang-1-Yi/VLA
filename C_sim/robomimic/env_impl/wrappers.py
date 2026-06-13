"""env_impl.wrappers —— gym wrapper + make_env 工厂。

源:抄自 PADP `diffusion_policy/env/robomimic/robomimic_image_wrapper.py`,
   简化为不依赖 BaseImageDataset 的纯 gym wrapper。
"""
from typing import Optional

import numpy as np
import gym
from gym import spaces

import robomimic.utils.obs_utils as ObsUtils


class RobomimicImageWrapper(gym.Env):
    """robomimic env → gym 接口的最小包装。

    observation:
        - rgb: dict 形态 {key: HWC uint8 ndarray}
        - low_dim: dict 形态 {key: float32 ndarray}
        - state: 由 low_dim 拼成的 state(本阶段 9 维:eef_pos[3]+quat[4]+gripper[2])
    action: 绝对动作 [D_a]
    """

    def __init__(self, env, shape_meta: dict, init_state: Optional[np.ndarray] = None,
                 max_steps: int = 400, use_legacy_normalizer: bool = False):
        self.env = env
        self.shape_meta = shape_meta
        self.max_steps = max_steps
        self._init_state = init_state
        self._step = 0

        obs_space = {}
        for key, attr in shape_meta["obs"].items():
            shp = attr["shape"]
            if attr.get("type", "low_dim") == "rgb":
                # HWC uint8
                obs_space[key] = spaces.Box(low=0, high=255, shape=shp, dtype=np.uint8)
            else:
                obs_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shp, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_space)
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=shape_meta["action"]["shape"],
            dtype=np.float32,
        )

    def _get_obs(self) -> dict:
        raw = self.env.get_observation()
        out = {}
        for k in self.shape_meta["obs"]:
            out[k] = raw[k]
        return out

    def reset(self):
        if self._init_state is not None:
            self.env.reset_to({"states": self._init_state})
        else:
            self.env.reset()
        self._step = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self.max_steps:
            done = True
        info["is_success"] = self.env.is_success() if hasattr(self.env, "is_success") else False
        return obs, reward, done, info


def make_env(task_name: str, shape_meta: dict, init_state: Optional[np.ndarray] = None,
             max_steps: int = 400):
    """robomimic env 的工厂。"""
    from C_sim.robomimic.env_impl.env_meta import get_max_steps
    max_steps = max_steps or get_max_steps(task_name)

    # 初始化 robomimic obs utils
    from C_sim.robomimic.env_impl.env_meta import get_env_meta
    config = get_env_meta(task_name)
    ObsUtils.initialize_obs_utils_with_config(config)

    from robomimic.envs.config import EnvUtils
    env_meta = dict(
        env_name=config.env_name,
        env_type=config.env_type,
        env_kwargs=config.env_kwargs,
    )
    if init_state is not None:
        env_meta["init_state"] = init_state

    import robomimic.utils.file_utils as FileUtils
    from robomimic.envs.env_base import EnvBase
    env = EnvBase.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=config.observation.camera_names,
        camera_widths=config.observation.camera_widths,
        camera_heights=config.observation.camera_heights,
        reward_shaping=False,
    )

    return RobomimicImageWrapper(env, shape_meta=shape_meta, max_steps=max_steps)
