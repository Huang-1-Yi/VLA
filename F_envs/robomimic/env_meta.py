# -*- coding: utf-8 -*-
# ============================================================
# F_envs.robomimic.env_meta
# 从 PADP_v3/diffusion_policy/env_runner/robomimic_image_runner.py
# 与 robomimic 0.3.0 内置任务表整合
# ============================================================

"""F_envs.robomimic.env_meta —— robomimic 任务的元信息。

**TASK_MAX_STEPS**:任务单 episode 的最大步数
  (与 robomimic `env_meta["env_kwargs"]["horizon"]` 一致,400 是默认值)
**get_env_meta(task_name)**:返回 dict 形式的元信息
  (供 `make_robomimic_env` 决定要加载哪个 robosuite env class)

**支持的任务**(robomimic 0.3.0 内置 5 个 image benchmark):
  - lift     Pick up the cube
  - can      Pick up the soup can
  - square   Pick up the nut and place on the peg (square)
  - transport Pick up two objects and place in bins
  - toolhang Hang the tool on the rack

每个任务对应一个 robomimic config 文件,如 `lift_image.hdf5` 配 `lift_image.json`。
但本环境直接读 hdf5 + `robomimic.scripts.generate_paper_configs.modify_config_for_dataset`
自动推断 config,无需手动配置。
"""
from typing import Dict, Optional


# 单 episode 最大步数(robomimic 默认 horizon=400,square 等长任务就是 400)
TASK_MAX_STEPS: Dict[str, int] = {
    "lift": 200,
    "can": 200,
    "square": 400,
    "transport": 700,
    "toolhang": 700,
    # Phases & other
    "nut_assembly": 500,
    "pick_place_can": 500,
    "block_pushing": 500,
}


# 任务名 -> 简短描述(供日志 / wandb)
TASK_DESCRIPTIONS: Dict[str, str] = {
    "lift": "Pick up the red cube.",
    "can": "Pick up the soup can.",
    "square": "Pick up the nut and place on the square peg.",
    "transport": "Pick up two objects and place in bins.",
    "toolhang": "Hang the tool on the rack.",
}


def get_env_meta(task_name: str) -> dict:
    """返回任务元信息。

    Args:
        task_name: 任务名(见 TASK_MAX_STEPS keys)
    Returns:
        dict 形如:
            {
                "task_name": str,
                "max_steps": int,
                "description": str,
                "env_type": "robomimic_image",
            }
    """
    if task_name not in TASK_MAX_STEPS:
        raise ValueError(
            f"Unknown robomimic task: {task_name!r}. "
            f"Supported: {list(TASK_MAX_STEPS.keys())}"
        )
    return {
        "task_name": task_name,
        "max_steps": TASK_MAX_STEPS[task_name],
        "description": TASK_DESCRIPTIONS.get(task_name, ""),
        "env_type": "robomimic_image",
    }


def infer_task_name_from_dataset(dataset_path: str) -> Optional[str]:
    """从 hdf5 文件名推断 task_name。

    约定:`square_d0_abs.hdf5` -> "square",`lift_image.hdf5` -> "lift",`can_lowdim.hdf5` -> "can"。
    """
    import os
    if dataset_path is None:
        return None
    fname = os.path.basename(dataset_path).lower()
    for task in TASK_MAX_STEPS:
        if fname.startswith(task):
            return task
    return None
