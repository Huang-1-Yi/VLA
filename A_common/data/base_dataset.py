"""铁律 6:BaseVLADataset 显式声明 n_obs_steps 与 horizon。"""
from typing import Dict, List, Optional, Tuple
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceSampler:
    """简化版 SequenceSampler:从一个 zarr replay buffer 按 episode 切窗口,处理首尾 padding。

    与 PADP RTV8 的区别:不做 per-epoch mapping,简单的 `__getitem__(idx) → (episode, window_start)`。

    参数:
      - episode_ends: cumsum 数组,每集结束索引(若是 padded 后的总长,可同时传 real_lens)
      - real_lens:    (可选)每集**原始**真实长度(不含头尾 padding);若传入,window_nums
                       按 `real_lens + horizon - 1` 算,而不是按 padded 长度,保证与
                       RTV8 (real_len + horizon - 1) 公式一致。向后兼容:不传则退回
                       `episode_ends - episode_starts`(旧行为)。
    """

    def __init__(self, replay_buffer, episode_ends: np.ndarray,
                 horizon: int, n_obs_steps: int = 1,
                 pad_strategy: str = "edge_repeat",
                 real_lens: Optional[np.ndarray] = None):
        """
        replay_buffer: zarr group,内含 'action' / rgb 键 / lowdim 键
        episode_ends: cumsum 数组,每集真实结束索引
        horizon:        预测窗口长度
        n_obs_steps:    历史观测窗口长度(1 即可,留扩展)
        pad_strategy:   'edge_repeat' / 'zero' / 'none'
        real_lens:      (可选)每集原始 real_len,用于正确算 window_nums
        """
        self.rb = replay_buffer
        self.episode_ends = episode_ends.astype(np.int64)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.pad_strategy = pad_strategy

        # 切 episode,统计每集真实长度
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]]).astype(np.int64)
        if real_lens is None:
            # 旧行为:从 episode_ends - episode_starts 算(若 caller 传的是 padded
            # cumsum,这里会得到 padded real_len,有 2*(horizon-1) 偏差)
            self.real_lens = (self.episode_ends - self.episode_starts).astype(np.int64)
        else:
            assert real_lens.shape == self.episode_ends.shape, (
                f"real_lens {real_lens.shape} != episode_ends {self.episode_ends.shape}"
            )
            # 用 caller 提供的原始 real_len(精确,正确)
            self.real_lens = real_lens.astype(np.int64)
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


class BalancedColumnsSampler:
    """RTV8-aligned per-epoch balanced-columns sampler。

    ========================================================================
    【创新点】per-batch 跨 episode 平衡(RTV8 范式,VLA 侧 1:1 复刻)
    ========================================================================

    痛点:
        Diffusion policy 的 DataLoader 走 `shuffle=True` 时,batch 内 N 个 sample
        撞同一 episode 的概率 = 1 - C(M, N) / M_(N)(M = 总 episode 数)。
        robomimic 演示轨迹长 ~100~200 步,horizon=40,window_num 约 60~160;
        200 demos × 平均 100 win ≈ 2 万样本,batch=64 → M 远大于 N,撞 ep
        概率小但**不为零**。RTV8 通过"per-epoch 全局映射表"**强制** batch 内 N 个
        sample 来自 N 个不同 episode(每行一个 batch,行内严格不重 ep)。

    数学契约(与 PADP_v3/position_aware_diffusion_policy/dataset/robomimic/
    replay_image_dataset_padp_v8.py 的 `_build_and_save_epoch_mapping` 1:1 对齐):

        ┌────────────────────────────────────────────────────────────────┐
        │ 输入                                                            │
        │   episode_ends: (N,)  每集在 *padded buffer* 中的累计结束索引   │
        │   real_lens:    (N,)  每集**原始**真实长度(不含 padding)        │
        │   horizon:      预测窗口长度                                    │
        │   batch_size:   DataLoader 的 batch_size (B)                    │
        │   seed:         随机种子                                        │
        │   epoch_index:  epoch 序号(决定性随机源)                        │
        ├────────────────────────────────────────────────────────────────┤
        │ 派生(关键公式,与 RTV8 完全一致)                                │
        │   window_nums[i] = real_lens[i] + horizon - 1                   │
        │   (而非 padded 后的 real_lens_new - horizon + 1,二者代数恒等)  │
        │   total_windows  = sum(window_nums)                             │
        ├────────────────────────────────────────────────────────────────┤
        │ 平衡分配(RTV8 lines 385-419 复刻,数学逻辑逐行 1:1)             │
        │   rng   = np.random.RandomState(seed + epoch_index)             │
        │   ep_ids = shuffle(arange(N), rng)         # 确定性洗牌         │
        │   rows    = [[] for _ in range(B)]        # B 个空 row          │
        │   row_lens = np.zeros(B, dtype=np.int64)   # 各 row 累积长度    │
        │   for e in ep_ids:                                              │
        │       win = window_nums[e]                                      │
        │       r   = int(np.argmin(row_lens))   # 选"当前最短"row       │
        │       rows[r].extend([(e, m) for m in range(win)])              │
        │       row_lens[r] += win                                        │
        │   max_cols = int(row_lens.max())    # 决定 __len__ = B*max_cols │
        │   # 短 row 补齐:循环 ep_ids,逐个 ep 头取 win 段                │
        │   for r in range(B):                                            │
        │       deficit = max_cols - len(rows[r])                         │
        │       fill_ptr = 0                                              │
        │       while len(rows[r]) < max_cols:                            │
        │           e    = int(ep_ids[fill_ptr % len(ep_ids)])            │
        │           win  = int(window_nums[e])                            │
        │           need = max_cols - len(rows[r])                         │
        │           take = min(win, need)                                 │
        │           rows[r].extend([(e, m) for m in range(take)])         │
        │           fill_ptr += 1                                         │
        ├────────────────────────────────────────────────────────────────┤
        │ 物化                                                            │
        │   _global_pairs: (B, max_cols, 2) int32 array                   │
        │       pairs[r, c, 0] = episode_id                               │
        │       pairs[r, c, 1] = window_start_in_episode                  │
        │       (window_start ∈ [0, real_len + horizon - 1))               │
        ├────────────────────────────────────────────────────────────────┤
        │ 寻址(__getitem__ 内部使用)                                     │
        │   p = idx % B                                                   │
        │   q = idx // B                                                  │
        │   e, m = _global_pairs[p, q]                                    │
        │   → DataLoader 拿到 [B, ...] 时,各 sample 必来自不同 ep        │
        └────────────────────────────────────────────────────────────────┘

    与 SequenceSampler 的对比:
        ┌───────────────────┬──────────────────────┬────────────────────────────┐
        │ 维度              │ SequenceSampler      │ BalancedColumnsSampler     │
        ├───────────────────┼──────────────────────┼────────────────────────────┤
        │ __len__           │ sum(window_nums)     │ B * max_cols               │
        │ idx 解码          │ searchsorted(cumsum) │ p=idx%B; q=idx//B          │
        │                   │ → (ep_id, win)       │ → _global_pairs[p, q]      │
        │ batch 内跨 ep     │ 不保证(DataLoader    │ **强制**不同 ep            │
        │                   │ shuffle 决定)        │ (贪心 argmin 最短行)       │
        │ DataLoader shuffle│ 必须 True            │ False(per-epoch 映射       │
        │                   │                      │ 已含洗牌)                  │
        │ DataLoader drop_last│ 任意                │ **True**(否则末 batch     │
        │                   │                      │ 可能不完整)                │
        │ per-epoch 重建    │ 不需要               │ **必须**(set_epoch)        │
        └───────────────────┴──────────────────────┴────────────────────────────┘

    Args:
        episode_ends:  (N,) padded buffer 中每集累计结束索引
        real_lens:     (N,) 每集**原始**真实长度(不含 padding)
        horizon:       预测窗口长度
        batch_size:    DataLoader 的 batch_size(>0)
        seed:          基础随机种子(默认 42)
        epoch_index:   起始 epoch 索引(默认 0)
    """

    def __init__(self, episode_ends: np.ndarray, real_lens: np.ndarray,
                 horizon: int, batch_size: int,
                 seed: int = 42, epoch_index: int = 0):
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        assert self.batch_size > 0, f"batch_size 必须 > 0, 实际={self.batch_size}"

        # 1. 输入检查
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.real_lens = np.asarray(real_lens, dtype=np.int64)
        assert self.real_lens.shape == self.episode_ends.shape, (
            f"real_lens {self.real_lens.shape} != episode_ends {self.episode_ends.shape}"
        )
        assert (self.real_lens >= 0).all(), "real_lens 必须 >= 0"

        # 2. episode_starts(padded buffer 内的累计起始)
        self.episode_starts = np.concatenate(
            [[0], self.episode_ends[:-1]]
        ).astype(np.int64)

        # 3. 关键公式(与 RTV8 1:1):window_nums[i] = real_lens[i] + horizon - 1
        self.window_nums = self.real_lens + self.horizon - 1
        assert (self.window_nums >= 1).all() or (self.real_lens == 0).all(), (
            f"window_nums 全部 < 1: max={self.window_nums.max()}, "
            f"horizon={self.horizon}, real_lens 范围=[{self.real_lens.min()}, {self.real_lens.max()}]"
        )

        # 4. 物化 _global_pairs
        self._seed = int(seed)
        self._epoch_index = int(epoch_index)
        self._global_pairs: np.ndarray
        self._max_cols: int
        self._global_pairs, self._max_cols = self._build_epoch_mapping(
            self._seed, self._epoch_index
        )

    # ------------------------------------------------------------------
    # 核心:per-epoch 平衡映射表(数学逻辑 1:1 复刻 RTV8 _build_and_save_epoch_mapping)
    # ------------------------------------------------------------------
    def _build_epoch_mapping(self, seed: int, epoch_index: int) -> Tuple[np.ndarray, int]:
        """贪心 argmin 最短行 + 循环补齐,产出 [B, max_cols, 2] (ep, win) 矩阵。"""
        B = self.batch_size
        N = self.real_lens.shape[0]
        ep_ids = np.arange(N, dtype=np.int64)

        # 确定性洗牌(per-epoch 不同)
        rng = np.random.RandomState(int(seed) + int(epoch_index))
        rng.shuffle(ep_ids)

        # 贪心分配:每个 ep 的所有窗口按"当前最短行"塞入
        rows: List[List[Tuple[int, int]]] = [[] for _ in range(B)]
        row_lens = np.zeros(B, dtype=np.int64)
        for e in ep_ids:
            win = int(self.window_nums[e])
            if win == 0:
                continue
            r = int(np.argmin(row_lens))          # 选当前最短行
            rows[r].extend([(int(e), m) for m in range(win)])
            row_lens[r] += win

        max_cols = int(row_lens.max()) if B > 0 else 0
        if max_cols == 0:
            raise RuntimeError(
                f"BalancedColumnsSampler: 全 {N} 个 episode 的 window_nums 全为 0,"
                f"无法构建映射。horizon={self.horizon} 是否过大?"
            )

        # 短行补齐:循环 ep_ids,每轮从 ep 头取 win 段(可能跨多 ep)
        for r in range(B):
            deficit = max_cols - len(rows[r])
            if deficit <= 0:
                continue
            fill_ptr = 0
            while len(rows[r]) < max_cols:
                e_fill = int(ep_ids[fill_ptr % len(ep_ids)])
                win_fill = int(self.window_nums[e_fill])
                if win_fill == 0:
                    fill_ptr += 1
                    continue
                need = max_cols - len(rows[r])
                take = min(win_fill, need)
                rows[r].extend([(e_fill, m) for m in range(take)])
                fill_ptr += 1

        # 物化为 [B, max_cols, 2] int32
        pairs = np.zeros((B, max_cols, 2), dtype=np.int32)
        for r in range(B):
            for c, (e, m) in enumerate(rows[r]):
                pairs[r, c, 0] = e
                pairs[r, c, 1] = m
        return pairs, max_cols

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def set_epoch(self, epoch_index: int):
        """切换到新 epoch:重新构建 _global_pairs(per-epoch 不同洗牌)。"""
        self._epoch_index = int(epoch_index)
        self._global_pairs, self._max_cols = self._build_epoch_mapping(
            self._seed, self._epoch_index
        )

    def __len__(self) -> int:
        """__len__ = batch_size * max_cols(已是 batch 维度,DataLoader 关 shuffle 即可)。"""
        return self.batch_size * self._max_cols

    def locate(self, idx: int) -> Tuple[int, int]:
        """idx → (episode_id, window_start_in_episode)。

        寻址规则(与 RTV8 __getitem__ p=idx%B / q=idx//B 一致):
            p = idx % B
            q = idx // B
            若 q 越界(q >= max_cols),clamp 到 max_cols - 1,与 RTV8 lines 290-291 同
            e, m = _global_pairs[p, q]
        """
        B = self.batch_size
        p = idx % B
        q = idx // B
        if q >= self._max_cols:
            q = self._max_cols - 1
        e = int(self._global_pairs[p, q, 0])
        m = int(self._global_pairs[p, q, 1])
        return e, m

    @property
    def max_cols(self) -> int:
        return self._max_cols

    @property
    def epoch_index(self) -> int:
        return self._epoch_index


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
