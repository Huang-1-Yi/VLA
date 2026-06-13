"""铁律 6:BaseVLADataset 显式声明 n_obs_steps 与 horizon。"""
from typing import Dict, Tuple
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceSampler:
    """简化版 SequenceSampler:从一个 zarr replay buffer 按 episode 切窗口,处理首尾 padding。

    与 PADP RTV8 的区别:不做 per-epoch mapping,简单的 `__getitem__(idx) → (episode, window_start)`。
    """

    def __init__(self, replay_buffer, episode_ends: np.ndarray,
                 horizon: int, n_obs_steps: int = 1,
                 pad_strategy: str = "edge_repeat"):
        """
        replay_buffer: zarr group,内含 'action' / rgb 键 / lowdim 键
        episode_ends: cumsum 数组,每集真实结束索引
        horizon:        预测窗口长度
        n_obs_steps:    历史观测窗口长度(1 即可,留扩展)
        pad_strategy:   'edge_repeat' / 'zero' / 'none'
        """
        self.rb = replay_buffer
        self.episode_ends = episode_ends.astype(np.int64)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.pad_strategy = pad_strategy

        # 切 episode,统计每集真实长度
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]]).astype(np.int64)
        self.real_lens = (self.episode_ends - self.episode_starts).astype(np.int64)
        # 每集窗口数 = real_len + horizon - 1(末位被 pad 到 horizon - 1 个 pad)
        self.window_nums = self.real_lens + self.horizon - 1

        # 总样本数 = sum(window_nums)
        self.total = int(self.window_nums.sum())

    def __len__(self) -> int:
        return self.total

    def locate(self, idx: int) -> Tuple[int, int]:
        """idx → (episode_id, window_start_in_padded_buffer)。"""
        ep_id = int(np.searchsorted(np.cumsum(self.window_nums), idx, side="right"))
        prev = 0 if ep_id == 0 else int(np.cumsum(self.window_nums)[ep_id - 1])
        win = idx - prev
        return ep_id, win

    def get_action(self, ep_id: int, win: int) -> np.ndarray:
        """取 [win : win + horizon] 的 action window,首尾 pad 用 edge_repeat。"""
        ep_start = int(self.episode_starts[ep_id])
        ep_end = int(self.episode_ends[ep_id])
        real_len = ep_end - ep_start
        H = self.horizon

        # win ∈ [0, real_len + H - 1)
        # 我们要采 [win : win + H] 这 H 个 index
        # 在 ep 局部坐标系下,index ∈ [win, win + H)
        # 首尾 pad 用 edge_repeat 扩展虚拟边界
        out = np.zeros((H,) + self.rb["action"].shape[1:], dtype=np.float32)
        for i in range(H):
            pos = win + i
            if pos < 0:
                src = 0                       # 头 pad
            elif pos >= real_len:
                src = real_len - 1            # 尾 pad
            else:
                src = pos
            out[i] = self.rb["action"][ep_start + src]
        return out

    def get_obs_window(self, ep_id: int, win: int, rgb_keys, lowdim_keys) -> dict:
        """取最近 n_obs_steps 帧 obs(在 win 时刻),同样 edge_repeat 边界。"""
        ep_start = int(self.episode_starts[ep_id])
        ep_end = int(self.episode_ends[ep_id])
        real_len = ep_end - ep_start
        S = self.n_obs_steps

        # win 表示 "现在"的位置
        out = {}
        for k in rgb_keys:
            arr = self.rb[k]                              # (T, H, W, C) uint8
            buf = np.zeros((S,) + arr.shape[1:], dtype=arr.dtype)
            for i in range(S):
                pos = win - (S - 1 - i)
                if pos < 0:
                    src = 0
                elif pos >= real_len:
                    src = real_len - 1
                else:
                    src = pos
                buf[i] = arr[ep_start + src]
            out[k] = buf
        for k in lowdim_keys:
            arr = self.rb[k]                              # (T, ...)
            buf = np.zeros((S,) + arr.shape[1:], dtype=np.float32)
            for i in range(S):
                pos = win - (S - 1 - i)
                if pos < 0:
                    src = 0
                elif pos >= real_len:
                    src = real_len - 1
                else:
                    src = pos
                buf[i] = arr[ep_start + src]
            out[k] = buf
        return out


class BaseVLADataset(Dataset):
    """铁律 6:抽象 Dataset 基类,显式声明 n_obs_steps / horizon。"""
    n_obs_steps: int
    horizon: int

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Dict, torch.Tensor]:
        """返回 (obs_dict, action) 二元组。
        action shape 必须严格对齐 [self.horizon, D_a]。
        """

    @abstractmethod
    def get_normalizer(self, **kwargs):
        """返回 LinearNormalizer。"""
