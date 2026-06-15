# -*- coding: utf-8 -*-
# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""C_sim.robomimic.interface_robomimic_env —— robomimic env 的抽象基类。

> **设计原则**:C_sim 只声明接口契约,具体实现由 F_envs.robomimic 提供。
>   - E_cti / Gpolicy 只能 `from C_sim.robomimic.interface_robomimic_env import BaseRobomimicEnv`
>   - 实现细节(robomimic 0.3.0 / mujoco / hdf5 加载等)隔离在 F_envs.robomimic
>   - 这样后续切到 LIBERO / 真机时,只换 F_envs 即可

**接口契约**(实现必须遵守):
  - action 维度:固定 7 = pos(3) + axis_angle(3) + gripper(1)
  - obs: dict[str, np.ndarray],key 与 shape_meta["obs"] 一致
  - step 返回: (obs, reward: float, done: bool, info: dict)
  - reward: robomimic sparse 0/1
  - get_robot_state: 返回 [x, y, z, qw, qx, qy, qz, gripper] 8D
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseRobomimicEnv(ABC):
    """robomimic 仿真环境的抽象基类。

    实现方: `F_envs.robomimic.robomimic_env.RobomimicEnv`
    调用方: `E_cti.train.tcp_rollout_client` (子 AI 实现)、
            `E_cti.train.eval_loop` 等
    """

    # action 维度约定(7 维 axis_angle 格式)
    ACTION_DIM: int = 7

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> dict:
        """重置到随机(或 seed 控制的)初始状态。

        Args:
            seed: 若给,会先 `self.seed(seed)` 再 reset
        Returns:
            obs dict(各 key 与 shape_meta["obs"] 一致)
        """
        raise NotImplementedError

    @abstractmethod
    def reset_to(self, init_state) -> dict:
        """重置到指定的初始 state(来自 hdf5 demo)。

        Args:
            init_state: dict,至少含 "states" 字段(np.ndarray,robosuite sim state)
                       也可直接传 np.ndarray(向下兼容)
        Returns:
            obs dict
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        """设置全局 numpy seed(影响 robosuite 内部随机)。"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray):
        """执行一步仿真。

        Args:
            action: np.ndarray, shape=(7,)=pos(3)+axis_angle(3)+gripper(1)
        Returns:
            (obs, reward: float, done: bool, info: dict)
        """
        raise NotImplementedError

    @abstractmethod
    def get_robot_state(self) -> np.ndarray:
        """获取当前 robot state。

        Returns:
            np.ndarray, [x, y, z, qw, qx, qy, qz, gripper_open] (8D)
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """关闭底层 mujoco / 渲染资源。"""
        raise NotImplementedError

    # ----------------- 辅助属性(子类可 override) -----------------

    # 默认实例属性,子类在 __init__ 里 override
    # (用 @property 容易导致子类 "can't set attribute" 问题)
    action_space = None        # type: gym.spaces.Space
    observation_space = None   # type: gym.spaces.Space
    max_steps: int = 400

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# 工厂函数:F_envs.robomimic 应当 patch 此函数
# ============================================================
_DEFAULT_FACTORY = None


def make_robomimic_env(task_name: str,
                        shape_meta: dict,
                        init_state: Optional[np.ndarray] = None,
                        max_steps: int = 400,
                        abs_action: bool = True) -> BaseRobomimicEnv:
    """工厂函数,根据 task_name 构造 BaseRobomimicEnv 实例。

    Args:
        task_name: robomimic 任务名 (e.g. "square", "can", "lift")
        shape_meta: obs/action shape 元信息
        init_state: 可选,初始 state(从 hdf5 demo 来的 1D np.ndarray)
        max_steps: 单 episode 最长步数(默认 400 for square_d0)
        abs_action: 是否绝对动作(default True)
                   - True:  动作是绝对世界坐标,需要 axis_angle 旋转
                   - False: 动作是 delta,通常是相对偏移

    F_envs/robomimic 应当在 import 时调用 `register_robomimic_factory` 注册真实工厂。
    若未注册,本函数抛 NotImplementedError。
    """
    if _DEFAULT_FACTORY is None:
        raise NotImplementedError(
            "F_envs/robomimic 应当通过 register_robomimic_factory() 注册真实工厂。"
            "请确认 F_envs.robomimic 已 import 并注册。"
        )
    return _DEFAULT_FACTORY(task_name=task_name, shape_meta=shape_meta,
                             init_state=init_state, max_steps=max_steps,
                             abs_action=abs_action)


def register_robomimic_factory(factory):
    """F_envs/robomimic 调用本函数注册真实工厂(必须在 import 时调用)。"""
    global _DEFAULT_FACTORY
    _DEFAULT_FACTORY = factory

