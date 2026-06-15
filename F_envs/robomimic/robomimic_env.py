# -*- coding: utf-8 -*-
# ============================================================
# F_envs.robomimic.robomimic_env
# 从 PADP_v3/diffusion_policy/env/robomimic/robomimic_image_wrapper.py
# 简化移植 + 适配 C_sim.robomimic.interface_robomimic_env.BaseRobomimicEnv
# ============================================================

"""F_envs.robomimic.robomimic_env —— robomimic 0.3.0 仿真环境的自包含实现。

**关键职责**:
  1. 从 hdf5 读 env_args → 用 `robomimic.utils.env_utils.create_env_from_metadata`
     加载 `EnvRobosuite` 实例
  2. 包装为 gym.Env(暴露 obs dict,符合 shape_meta)
  3. 实现 `BaseRobomimicEnv` 5 个抽象方法
  4. action 维度 = 7 (axis_angle: pos3+rot3+gripper1)

**env_name 映射**(robomimic 0.3.0 旧数据集存的 env_name 与新 robosuite 不一致):
  - Lift_D0/D1         → Lift
  - Can_D0/D1/D2       → PickPlaceCan
  - Square_D0/D1/D2    → NutAssemblySquare
  - Transport_D0/D1/D2 → TwoArmTransport
  - ToolHang_D0/D1/D2  → ToolHang
"""
from __future__ import annotations

import collections
import os
from typing import Optional, Union, Dict, Any
import json

import gym
import numpy as np
from gym import spaces

# 屏蔽 robosuite/robomimic 的 verbose WARNING
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from C_sim.robomimic.interface_robomimic_env import BaseRobomimicEnv

# robomimic 0.3.0 旧数据集的 env_name (e.g. "Square_D0") → 新 robosuite env_name
# 数据源:robomimic 论文(https://arxiv.org/abs/2108.03298)
ROBOMIMIC_ENV_NAME_MAP = {
    # Lift
    "Lift": "Lift",
    "Lift_D0": "Lift",
    "Lift_D1": "Lift",
    # Can (PickPlaceCan)
    "Can": "PickPlaceCan",
    "Can_D0": "PickPlaceCan",
    "Can_D1": "PickPlaceCan",
    "Can_D2": "PickPlaceCan",
    # Square (NutAssemblySquare)
    "Square": "NutAssemblySquare",
    "Square_D0": "NutAssemblySquare",
    "Square_D1": "NutAssemblySquare",
    "Square_D2": "NutAssemblySquare",
    # Transport (TwoArmTransport)
    "Transport": "TwoArmTransport",
    "Transport_D0": "TwoArmTransport",
    "Transport_D1": "TwoArmTransport",
    "Transport_D2": "TwoArmTransport",
    # ToolHang
    "ToolHang": "ToolHang",
    "ToolHang_D0": "ToolHang",
    "ToolHang_D1": "ToolHang",
    "ToolHang_D2": "ToolHang",
    # NutAssembly
    "NutAssembly": "NutAssembly",
    "NutAssemblySingle": "NutAssemblySingle",
    "NutAssemblySingle_D0": "NutAssemblySingle",
    "NutAssemblySingle_D1": "NutAssemblySingle",
    "NutAssemblyRound": "NutAssemblyRound",
    "NutAssemblyRound_D0": "NutAssemblyRound",
    "NutAssemblyRound_D1": "NutAssemblyRound",
    # PickPlace
    "PickPlaceCan": "PickPlaceCan",
    "PickPlaceBread": "PickPlaceBread",
    "PickPlaceCereal": "PickPlaceCereal",
    "PickPlaceMilk": "PickPlaceMilk",
    "PickPlaceSingle": "PickPlaceSingle",
}


def remap_env_name(orig_name: str) -> str:
    """robomimic 0.3.0 旧 env_name → 新 robosuite 注册名。
    若不存在,直接返回原名(让 robosuite 自己报错,信息更明确)。
    """
    return ROBOMIMIC_ENV_NAME_MAP.get(orig_name, orig_name)


def _init_obs_utils(shape_meta: dict) -> None:
    """根据 shape_meta 初始化 robomimic 内部 obs modality 映射。"""
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)


class RobomimicEnv(BaseRobomimicEnv):
    """robomimic 0.3.0 gym wrapper(自包含,无外部依赖)。

    Example:
        >>> env = RobomimicEnv(
        ...     dataset_path="/path/to/square_d0_abs.hdf5",
        ...     shape_meta={"obs": {...}, "action": {"shape": [10]}},
        ...     max_steps=400,
        ... )
        >>> obs = env.reset(seed=10000)
        >>> obs, r, d, info = env.step(np.zeros(7, dtype=np.float32))
    """

    # gym interface
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        max_steps: int = 400,
        abs_action: bool = True,
        render: bool = False,
        render_offscreen: bool = True,
    ):
        """
        Args:
            dataset_path: hdf5 路径,用于读 env_args
            shape_meta: dict,形如 ``{"obs": {"agentview_image": {"shape": [3,84,84], "type": "rgb"}, ...},
                                       "action": {"shape": [10]}}``
                       obs 决定 env 返回的 obs dict keys / shape
            init_state: 可选,初始 sim state (np.ndarray)
                        None 时用 seed 随机 reset
            max_steps: 单 episode 最大步数(超过则 done=True)
            abs_action: True=绝对动作(robosuite OSC controller default),
                        False=delta 动作(本环境默认 abs=True)
            render/render_offscreen: robomimic 渲染开关
        """
        self.dataset_path = os.path.expanduser(dataset_path)
        self.shape_meta = shape_meta
        self.init_state = init_state
        self._max_steps = int(max_steps)
        self.abs_action = bool(abs_action)
        self.render_offscreen = bool(render_offscreen)
        self.render = bool(render)

        # 从 hdf5 读 env_meta
        env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        # 🚨 关键:env_name 映射
        orig_env_name = env_meta.get("env_name", "")
        mapped_env_name = remap_env_name(orig_env_name)
        if mapped_env_name != orig_env_name:
            env_meta["env_name"] = mapped_env_name
        # 关掉 object state(只要 image + low_dim)
        env_meta["env_kwargs"]["use_object_obs"] = False
        # abs_action → controller 非 delta
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        # 强制 offscreen(不弹窗)
        env_meta["env_kwargs"]["has_renderer"] = False
        env_meta["env_kwargs"]["has_offscreen_renderer"] = self.render_offscreen
        # use_camera_obs 需要为 True 才能拿到 *_image keys
        if self.render_offscreen:
            env_meta["env_kwargs"]["use_camera_obs"] = True
        self.env_meta = env_meta

        # 初始化 obs_utils modality
        _init_obs_utils(shape_meta)

        # 创建底层 env
        self.env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,  # 用 offscreen
            render_offscreen=self.render_offscreen,
            use_image_obs=self.render_offscreen,
        )
        # Robosuite's hard reset causes excessive memory consumption
        # (PADP_v3: https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py)
        self.env.env.hard_reset = False

        # robomimic 0.3.0 在 reset 后才会把 obs_space 正确填进 self.env
        # 先做一次 dry reset 让 obs space 就绪
        self.env.reset()
        self._step_count = 0
        self._seed = None
        self._has_reset_before = False

        # 构建 action / observation space
        action_shape = shape_meta["action"]["shape"]  # [10]
        self._policy_action_dim = action_shape[0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.ACTION_DIM,),
            dtype=np.float32,
        )

        # observation_space 是 shape_meta['obs'] 的镜像
        obs_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1.0, 1.0
            if key.endswith("image"):
                min_value, max_value = 0.0, 1.0
            elif key.endswith("depth"):
                min_value, max_value = 0.0, 1.0
            elif key.endswith("quat"):
                min_value, max_value = -1.0, 1.0
            elif key.endswith("qpos") or key.endswith("pos"):
                min_value, max_value = -1.0, 1.0
            else:
                raise RuntimeError(f"Unsupported obs key: {key}")
            obs_space[key] = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
        self.observation_space = obs_space

    # ============== BaseRobomimicEnv 接口实现 ==============

    def reset(self, seed: Optional[int] = None) -> dict:
        """重置到随机状态(seed 可控)或 init_state(若提供)。"""
        if seed is not None:
            self.seed(seed)
        return self._do_reset()

    def reset_to(self, init_state: Union[np.ndarray, dict]) -> dict:
        """重置到指定的初始 state。

        Args:
            init_state: 可以是 np.ndarray(robosuite sim state)
                        或 dict(必须含 "states" 键)
        """
        if isinstance(init_state, np.ndarray):
            state_dict = {"states": init_state}
        else:
            state_dict = dict(init_state)
            if "states" not in state_dict:
                # 兼容性:可能直接是 {k: v} 但第一个 value 是 sim state
                # robomimic 必须有 "states" key
                raise ValueError(
                    f"reset_to: init_state dict must contain 'states' key, got {list(state_dict.keys())}"
                )
        # 第一次 reset 需要先做一次 empty reset 让 robosuite 内部状态机就绪
        if not self._has_reset_before:
            self.env.reset()
            self._has_reset_before = True
        raw_obs = self.env.reset_to(state_dict)
        self._step_count = 0
        return self._format_obs(raw_obs)

    def seed(self, seed: int) -> None:
        """设置全局 numpy seed(robosuite reset 时用)。"""
        np.random.seed(int(seed))
        self._seed = int(seed)

    def step(self, action: np.ndarray):
        """执行一步。

        Args:
            action: shape=(7,) axis_angle 动作
        Returns:
            (obs, reward: float, done: bool, info: dict)
        """
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"action shape must be ({self.ACTION_DIM},), got {action.shape}"
            )
        action = action.astype(np.float32)
        raw_obs, reward, done, info = self.env.step(action)
        self._step_count += 1
        # time limit
        if self._step_count >= self._max_steps and not done:
            done = True
            info = dict(info) if info else {}
            info["TimeLimit.truncated"] = True
        obs = self._format_obs(raw_obs)
        return obs, float(reward), bool(done), dict(info) if info else {}

    def get_robot_state(self) -> np.ndarray:
        """返回 [x, y, z, qw, qx, qy, qz, gripper_open] 8D state。"""
        try:
            obs = self.env.get_observation()
            state = []
            state.extend(obs.get("robot0_eef_pos", np.zeros(3))[:3])
            state.extend(obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))[:4])
            state.append(obs.get("robot0_gripper_qpos", np.zeros(2))[0])
            return np.asarray(state, dtype=np.float32)
        except Exception:
            return np.array([0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def close(self) -> None:
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ============== 内部辅助 ==============

    def _do_reset(self) -> dict:
        """根据 _seed 决定 reset 路径。"""
        if self.init_state is not None:
            return self.reset_to(self.init_state)
        if self._seed is not None:
            # robosuite 内部使用 numpy global random
            np.random.seed(self._seed)
        raw_obs = self.env.reset()
        self._step_count = 0
        self._has_reset_before = True
        return self._format_obs(raw_obs)

    def _format_obs(self, raw_obs: dict) -> dict:
        """把 robomimic 原始 obs 转成符合 shape_meta 的 dict。

        约定:
          - rgb keys: (H, W, C) uint8  →  (C, H, W) uint8(供 server 端 to-torch 后 float/255)
          - lowdim keys: (D,) float64  →  (D,) float32
        """
        out = {}
        for key in self.observation_space.keys():
            v = raw_obs.get(key, None)
            if v is None:
                # 用 zeros 占位
                spec = self.observation_space[key]
                out[key] = np.zeros(spec.shape, dtype=spec.dtype)
                continue
            if key.endswith("image"):
                # raw_obs image: (H, W, C) uint8
                if v.ndim == 3 and v.shape[-1] in (1, 3, 4):
                    v = np.moveaxis(v, -1, 0)  # (C, H, W)
                out[key] = v.astype(np.uint8, copy=False)
            else:
                out[key] = v.astype(np.float32, copy=False)
        return out

    # gym 兼容
    def render(self, mode: str = "rgb_array"):
        if mode == "rgb_array":
            # 用 agentview_image 作为 render
            obs = self.env.get_observation()
            img = obs.get("agentview_image", None)
            if img is not None:
                if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                    return img
                return np.moveaxis(img, 0, -1) if img.ndim == 3 else img
        return None

    def __str__(self) -> str:
        return (
            f"RobomimicEnv(task={self.env_meta.get('env_name')}, "
            f"action_dim={self.ACTION_DIM}, max_steps={self._max_steps})"
        )
