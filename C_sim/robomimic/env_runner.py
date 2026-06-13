"""C_sim.robomimic.env_runner —— 接收 predict_fn,跑 N 个 episode,返回指标。

源:简化自 PADP `diffusion_policy/env_runner/robomimic_image_runner_padp.py`。
本阶段不接 remote env server(单 4090 时,本进程内 step env)。
"""
import os
from typing import Dict, Callable
import numpy as np
import torch
from tqdm import tqdm

from A_common.logger import get_logger
from A_common.types.action_output import ActionOutput
from A_common.data.base_collator import base_collate
from C_sim.robomimic.env_impl import make_env, get_max_steps
from C_sim.robomimic.zarr_dataset import RobomimicZarrDataset

logger = get_logger(__name__)


class RobomimicImageRunner:
    """单进程 sim 评估 runner。"""

    def __init__(self, shape_meta: dict, task_name: str,
                 dataset_path: str, n_test: int = 50, n_train: int = 6,
                 max_steps: int = 400, render: bool = False):
        self.shape_meta = shape_meta
        self.task_name = task_name
        self.dataset_path = dataset_path
        self.n_test = n_test
        self.n_train = n_train
        self.max_steps = max_steps or get_max_steps(task_name)
        self.render = render
        self.env = make_env(task_name, shape_meta=shape_meta, max_steps=self.max_steps)

    def run(self, predict_fn: Callable, n_test: int = None) -> dict:
        """跑 n_test 个 episode,返回 success rate。"""
        n = n_test or self.n_test
        successes = []
        for ep in tqdm(range(n), desc=f"Eval {self.task_name}"):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            while not done:
                # 把 obs dict 转 batch (1, ...) 形态
                obs_b = {}
                for k, v in obs.items():
                    if k in self.shape_meta["obs"]:
                        arr = np.asarray(v)
                        if arr.ndim == 3:  # HWC → 添加 S=1 维
                            arr = arr[None]   # (1, h, w, c)
                        elif arr.ndim == 1:  # D → 添加 S=1
                            arr = arr[None]   # (1, D)
                        obs_b[k] = torch.from_numpy(arr.astype(np.float32) if arr.dtype != np.uint8 else arr)
                # 图像转 CHW
                for k in self.shape_meta["obs"]:
                    if self.shape_meta["obs"][k].get("type", "low_dim") == "rgb":
                        v = obs_b[k]
                        if v.ndim == 4 and v.shape[-1] in (1, 3):
                            obs_b[k] = v.permute(0, 3, 1, 2).float() / 255.0
                # state
                state_parts = []
                for k in self.shape_meta["obs"]:
                    if self.shape_meta["obs"][k].get("type", "low_dim") == "low_dim":
                        v = obs_b[k]
                        state_parts.append(v.reshape(v.shape[0], -1))
                if state_parts:
                    obs_b["state"] = torch.cat(state_parts, dim=-1)

                # predict
                with torch.no_grad():
                    out = predict_fn(obs_b)   # ActionOutput
                action = out.actions.cpu().numpy()  # [H, D_a] 或 [1, D_a]
                if action.ndim == 3:
                    action = action[0]   # 取 batch 0

                # step H 次(只取第 0 步给 env 算 rolling)
                if action.ndim == 1:
                    a = action
                else:
                    a = action[0]  # 第一步
                obs, reward, done, info = self.env.step(a)
                ep_reward += reward
                ep_steps += 1
                if info.get("is_success", False):
                    break
            successes.append(float(info.get("is_success", False)))
        success_rate = float(np.mean(successes))
        logger.info("Eval %s: success_rate=%.4f over %d episodes", self.task_name, success_rate, n)
        return {
            "success_rate": success_rate,
            "n_episodes": n,
            "per_episode_success": successes,
        }
