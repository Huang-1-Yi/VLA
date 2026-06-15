# -*- coding: utf-8 -*-
# ============================================================
# F_envs.robomimic.wrappers
# 额外的 gym 包装(可选)
# ============================================================

"""F_envs.robomimic.wrappers —— 额外的 gym 包装(可选,推荐)。

提供:
  - TimeLimit       :max_steps 截断(已在 RobomimicEnv 内置,这里只做参考)
  - DictObsWrapper  :保证 obs 是 dict[str, np.ndarray]
  - RewardSparse    :reward 强制转 0/1
  - ActionClip      :action clip 到 [-1, 1]
"""
from __future__ import annotations
import gym
import numpy as np
from typing import Any


class TimeLimit(gym.Wrapper):
    """按 step 计数截断(done 但 info["TimeLimit.truncated"]=True)。"""

    def __init__(self, env, max_steps: int = 400):
        super().__init__(env)
        self._max_steps = int(max_steps)
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self._max_steps and not done:
            done = True
            info = dict(info) if info else {}
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info


class DictObsWrapper(gym.Wrapper):
    """确保 obs 是 dict 格式(robomimic 默认就是,但万一走 vec env 可能 flatten)。"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._ensure_dict(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._ensure_dict(obs), reward, done, info

    @staticmethod
    def _ensure_dict(obs):
        if isinstance(obs, dict):
            return obs
        return {"obs": obs}


class RewardSparse(gym.Wrapper):
    """把 reward 强制转 0/1(robomimic 默认就是 0/1,这里只是 sanity)。"""

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward > 0.5), done, info


class ActionClip(gym.Wrapper):
    """把 action clip 到 [-1, 1](防御上游发非法值)。"""

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self.env.step(a)
