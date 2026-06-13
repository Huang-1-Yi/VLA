# -*- coding: utf-8 -*-
"""C_sim —— 仿真边界(铁律 4:对外只暴露 factory)。

公开 API(本模块顶层):
    - make_env_meta(cfg) -> dict
    - make_env(cfg) -> callable zero-arg gym env
    - make_dataset(cfg) -> A_common.data.base_dataset.BaseVLADataset
    - make_eval_runner(cfg) -> object with .run(predict_fn) -> dict
"""
from typing import Callable


def make_env_meta(cfg: dict) -> dict:
    """构造 shape_meta 字典(给 encoder / dataset 共同用)。"""
    return cfg["data"]["shape_meta"]


def make_env(cfg: dict) -> Callable:
    """构造 zero-arg env factory(供 eval runner 调)。"""
    from C_sim.robomimic.env_impl import make_env as _make_env

    sim = cfg.get("sim", {})
    return _make_env(
        task_name=sim["task_name"],
        shape_meta=sim["shape_meta"],
        max_steps=sim.get("max_steps", 400),
    )


def make_dataset(cfg: dict):
    """构造 BaseVLADataset 实例。"""
    from C_sim.robomimic.zarr_dataset import RobomimicZarrDataset

    sim = cfg.get("sim", {})
    data = cfg["data"]
    return RobomimicZarrDataset(
        shape_meta=data["shape_meta"],
        dataset_path=data["dataset_path"],
        n_demo=data.get("n_demo", 200),
        horizon=data.get("horizon", 40),
        n_obs_steps=data.get("n_obs_steps", 1),
        n_action_steps=data.get("n_action_steps", 8),
        abs_action=data.get("abs_action", True),
        use_legacy_normalizer=data.get("use_legacy_normalizer", False),
    )


def make_eval_runner(cfg: dict):
    """构造评估 runner(带 .run(predict_fn) -> dict 接口)。"""
    from C_sim.robomimic.env_runner import RobomimicImageRunner

    sim = cfg.get("sim", {})
    eval_cfg = cfg.get("eval", {})
    return RobomimicImageRunner(
        shape_meta=sim["shape_meta"],
        task_name=sim["task_name"],
        dataset_path=cfg["data"]["dataset_path"],
        n_test=eval_cfg.get("n_test", 50),
        n_train=eval_cfg.get("n_train", 6),
        max_steps=sim.get("max_steps", 400),
        render=eval_cfg.get("render", False),
    )
