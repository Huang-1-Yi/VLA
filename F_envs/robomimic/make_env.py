# -*- coding: utf-8 -*-
# ============================================================
# F_envs.robomimic.make_env
# 工厂函数 + 注册到 C_sim.robomimic.interface_robomimic_env
# ============================================================

"""F_envs.robomimic.make_env —— 工厂函数 + 注册到 C_sim。

调用方(在 E_cti / Gpolicy 中):
    from C_sim.robomimic.interface_robomimic_env import make_robomimic_env
    env = make_robomimic_env(task_name="square", shape_meta=..., max_steps=400)

底层:
    F_envs.robomimic 在 import 时调用 `register_robomimic_factory(robomimic_env_factory)`
    把自己的工厂注册到 C_sim。
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np

from C_sim.robomimic.interface_robomimic_env import (
    register_robomimic_factory,
    BaseRobomimicEnv,
)
from F_envs.robomimic.env_meta import (
    TASK_MAX_STEPS,
    get_env_meta,
    infer_task_name_from_dataset,
)


# 数据集路径表(task_name -> hdf5 路径相对 VLA/ 根)
# 路径前缀用 {VLA_ROOT},运行时替换成 VLA 绝对路径
_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DATASETS = {
    "lift": f"{_VLA_ROOT}/data/robomimic/datasets/lift_d0/lift_d0_abs.hdf5",
    "can": f"{_VLA_ROOT}/data/robomimic/datasets/can_d0/can_d0_abs.hdf5",
    "square": f"{_VLA_ROOT}/data/robomimic/datasets/square_d0/square_d0_abs.hdf5",
    "transport": f"{_VLA_ROOT}/data/robomimic/datasets/transport_d0/transport_d0_abs.hdf5",
    "toolhang": f"{_VLA_ROOT}/data/robomimic/datasets/toolhang_d0/toolhang_d0_abs.hdf5",
}

# 兜底:用 VLA 根的 "data/..." 相对路径(常见用法 cwd=VLA/)
_DEFAULT_DATASETS_REL = {
    "lift": "data/robomimic/datasets/lift_d0/lift_d0_abs.hdf5",
    "can": "data/robomimic/datasets/can_d0/can_d0_abs.hdf5",
    "square": "data/robomimic/datasets/square_d0/square_d0_abs.hdf5",
    "transport": "data/robomimic/datasets/transport_d0/transport_d0_abs.hdf5",
    "toolhang": "data/robomimic/datasets/toolhang_d0/toolhang_d0_abs.hdf5",
}


def _resolve_dataset_path(task_name: str, dataset_path: Optional[str] = None) -> str:
    """解析 task_name -> 实际 hdf5 路径。

    优先级:
      1. dataset_path 参数(显式传入,绝对或相对 cwd)
      2. _DEFAULT_DATASETS 绝对路径(基于 VLA root)
      3. _DEFAULT_DATASETS_REL 相对路径(要求 cwd 在 VLA/ 或父目录)
      4. 报错
    """
    if dataset_path is not None:
        # 显式传入:相对路径按 cwd 解析
        abs_path = os.path.abspath(dataset_path)
        if os.path.exists(abs_path):
            return abs_path
    if task_name in _DEFAULT_DATASETS:
        candidate = _DEFAULT_DATASETS[task_name]
        if os.path.exists(candidate):
            return candidate
    if task_name in _DEFAULT_DATASETS_REL:
        candidate = _DEFAULT_DATASETS_REL[task_name]
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Cannot find dataset for task {task_name!r}. "
        f"Pass `dataset_path` explicitly or place hdf5 at "
        f"{_DEFAULT_DATASETS.get(task_name, _DEFAULT_DATASETS_REL.get(task_name, '<unknown>'))}"
    )


def make_robomimic_env(task_name: str,
                       shape_meta: dict,
                       init_state: Optional[np.ndarray] = None,
                       max_steps: int = 400,
                       dataset_path: Optional[str] = None,
                       abs_action: bool = True) -> BaseRobomimicEnv:
    """工厂函数:F_envs.robomimic 实现版。

    实现 `C_sim.robomimic.interface_robomimic_env.make_robomimic_env` 的接口契约。

    Args:
        task_name: 任务名(见 `F_envs.robomimic.env_meta.TASK_MAX_STEPS`)
        shape_meta: dict,决定 obs keys / shapes
        init_state: 可选,初始 sim state(np.ndarray)
        max_steps: 单 episode 最大步数
        dataset_path: 可选,显式 hdf5 路径
        abs_action: 绝对动作 vs delta 动作
    Returns:
        RobomimicEnv 实例(BaseRobomimicEnv 子类)
    """
    # 懒加载,避免 import 时就 load robomimic(mujoco 0.5s 启动)
    from F_envs.robomimic.robomimic_env import RobomimicEnv

    if task_name not in TASK_MAX_STEPS:
        raise ValueError(
            f"Unknown robomimic task: {task_name!r}. "
            f"Supported: {list(TASK_MAX_STEPS.keys())}"
        )

    hdf5_path = _resolve_dataset_path(task_name, dataset_path)
    return RobomimicEnv(
        dataset_path=hdf5_path,
        shape_meta=shape_meta,
        init_state=init_state,
        max_steps=max_steps,
        abs_action=abs_action,
        render=False,
        render_offscreen=True,
    )


# ============== 注册到 C_sim(必须在 import 时调用) ==============
# 显式注册,把本工厂挂到 `C_sim.robomimic.interface_robomimic_env.make_robomimic_env` 上
# 注:register_robomimic_factory 只把 _DEFAULT_FACTORY 替换;调用方始终从 C_sim 拿,
# 所以本文件被 import 一次就够了。
register_robomimic_factory(make_robomimic_env)


__all__ = ["make_robomimic_env", "TASK_MAX_STEPS", "get_env_meta"]
