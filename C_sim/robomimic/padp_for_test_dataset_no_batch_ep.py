"""C_sim.robomimic.padp_for_test_dataset_no_batch_ep —— 【消融基线】无 batch 跨 ep 平衡的版本。

==========================================================================
🚨 消融实验基线(ABLATION BASELINE)🚨 —— 不可删,必须先经用户确认 🚨
==========================================================================

本文件是 **永久 ablation baseline 快照**,与同目录
`padp_for_test_dataset.py`(实验组)构成消融对。

本文件作为 baseline 永久保留,任何情况下都不得删除/重构/合并;
任何修改前必须先 @用户确认(2026-06-14 用户明令)。

消融对照表
----------
  ┌─────────┬─────────────────────────────────┬──────────────────────────┐
  │ 组别     │ 配置                             │ 走的 sampler             │
  ├─────────┼─────────────────────────────────┼──────────────────────────┤
  │ 实验组   │ padp_for_test_dataset.py        │ BalancedColumnsSampler   │
  │         │ + balanced_sampler=True         │ (行内强制不同 ep)        │
  │ 对照组 1 │ padp_for_test_dataset.py        │ SequenceSampler          │
  │         │ + balanced_sampler=False        │ (与本文件字节级等价)     │
  │ 对照组 2 │ 本文件 (NoBatchEp class)        │ SequenceSampler          │
  │ (基线)  │                                  │ (独立 class 快照)        │
  └─────────┴─────────────────────────────────┴──────────────────────────┘

TODO 清单(实验待跑)
-------------------
  [ ] 跑实验组:   padp_for_test_dataset.py + balanced_sampler=True
  [ ] 跑对照组 1: padp_for_test_dataset.py + balanced_sampler=False
  [ ] 跑对照组 2: import 本文件 NoBatchEp 类
  [ ] 三组差异 < 0.5% →  消融无显著影响
  [ ] 三组差异 > 2%   →  balanced sampler 是必要设计
  [ ] 三个不同 seed(42 / 123 / 2024)取均值
  [ ] 对照组 1 vs 对照组 2 应**完全字节级等价**(sanity check)

不可删/不可改红线(违反前必须先 @用户确认)
-----------------------------------------
  ❌ 不可删本文件(任何时候都不删,不允许任何理由)
  ❌ 不可改 class 名 `RobomimicZarrDatasetPadpForTestNoBatchEp`
  ❌ 不可改 `__init__` 签名(不得加 balanced_sampler / batch_size / sampler_seed)
  ❌ 不可加 `set_epoch()` 方法
  ❌ 不可加 `balanced_sampler` / `batch_size` @property
  ❌ 不可合并到 `padp_for_test_dataset.py`
  ❌ 不可改 window_nums 公式(real_lens + horizon - 1)
  ❌ 不可改 __getitem__ / get_normalizer 的数值(必须与对照组 1 字节级一致)

如果 reviewer 说"这个文件没人用,删了吧",请回:
  "这是消融实验的永久 baseline 快照,删除需要先经用户(@user)确认,
  见文件顶部 '🚨 消融实验基线' 块。"

相关文件
--------
  - C_sim/robomimic/padp_for_test_dataset.py                  (实验组,含本文件镜像)
  - A_common/data/base_dataset.py::BalancedColumnsSampler    (实验组 sampler)
  - A_common/data/base_dataset.py::SequenceSampler           (本文件用的 sampler)
  - E_cti/train/verify_balanced_sampler.py                   (4 项验证脚本,第 3 项
                                                                专门验证本文件 vs
                                                                对照组 1 字节级一致)
==========================================================================
================================================================================
🧪 消融实验专用(ABLATION BASELINE)
================================================================================

背景:
    `padp_for_test_dataset.RobomimicZarrDatasetPadpForTest` 在 RTV8 数据对齐基础上
    又新增了 `balanced_sampler=True` 选项,启用 RTV8 风格的"per-epoch 全局映射表 +
    balanced columns"(行=batch,列=sample,行内强制不同 ep,每 epoch 重建)。

    本文件是新增 balanced_sampler 之前**一字不差的旧版本**,作为消融基线:
    - 走 SequenceSampler(flat index),DataLoader 配 `shuffle=True`
    - **没有** per-epoch 全局映射表
    - **没有** balanced columns 贪心分配
    - **没有** `set_epoch` 钩子
    - **没有** `balanced_sampler` / `batch_size` property

与新版本(同目录 `padp_for_test_dataset.py`)的数学差异(本文件 100% 缺少):
    ┌──────────────────────┬──────────────────────────────────────────────────┐
    │ 维度                  │ 本文件(消融基线)                                  │
    ├──────────────────────┼──────────────────────────────────────────────────┤
    │ Sampler 类型          │ SequenceSampler(flat index)                     │
    │ window_nums 公式     │ real_lens + horizon - 1(与新版本同)              │
    │ __len__              │ sum(window_nums)                                 │
    │ idx 解码              │ searchsorted(cumsum) → (ep_id, win)              │
    │ batch 内跨 ep 分布   │ **不保证**:DataLoader shuffle 随机打散           │
    │ DataLoader 配置      │ shuffle=True, drop_last=False                   │
    │ per-epoch 重排        │ 无(每 epoch 同 idx 取同 (ep, win))              │
    │ set_epoch 方法        │ 无(本类**没有** set_epoch)                      │
    │ balanced_sampler 属性 │ 无(本类**没有**此属性)                          │
    └──────────────────────┴──────────────────────────────────────────────────┘

================================================================================
🧪 消融实验使用方法
================================================================================

方法 A:在 train 脚本里改 import(推荐)

    # E_cti/train/padp_for_test_train.py
    - from C_sim.robomimic.padp_for_test_dataset import RobomimicZarrDatasetPadpForTest
    + from C_sim.robomimic.padp_for_test_dataset_no_batch_ep import \
    +     RobomimicZarrDatasetPadpForTestNoBatchEp as RobomimicZarrDatasetPadpForTest

    然后 yaml 里:
      data:
        balanced_sampler: false   # 显式关掉(本基线类无此参数,但 yaml 解析会忽略)
    训练脚本里:
        dl = DataLoader(dataset, batch_size=..., shuffle=True, drop_last=False, ...)

方法 B:在 yaml 切到 balanced_sampler=false(等同本基线)

    不改 import,直接在 yaml 里:
      data:
        balanced_sampler: false
    效果与本基线类**几乎一致**(`SequenceSampler` 路径),
    唯一区别是 yaml 路径会走新版本类,本文件是独立快照。

建议两种都用:
  1. yaml balanced_sampler=true  + 新版本类       → "RTV8-aligned"  实验组
  2. yaml balanced_sampler=false + 新版本类       → "SequenceSampler"对照组 1
  3. import 本基线类(无视 yaml 开关)             → "消融基线"      对照组 2

对照组 1 vs 2 应**数值一致**(同 SequenceSampler 代码),对照组 2 vs 3 仅作冗余 sanity check。

================================================================================
📌 复刻本文件时的依据(供 review)
================================================================================
本文件源码复刻自 `padp_for_test_dataset.py` 2026-06-14 引入 balanced_sampler 之前的版本,
差异点逐项列出(本文件保持修改前状态,新版本在另一文件):

    [D1] line 21:  `from typing import Dict, Optional`     → `from typing import Dict`
    [D2] line 27:  imports 中**删除** `BalancedColumnsSampler`
    [D3] line 57-60: __init__ 末尾**删除** 3 个新参数
                          (balanced_sampler, batch_size, sampler_seed)
    [D4] line 214-253: Sampler 选择块**还原**为单一 SequenceSampler 调用
    [D5] line 258-264: **删除** set_epoch 方法
    [D6] line 266-278: **删除** balanced_sampler / batch_size 两个 property

其余代码(数据加载、__getitem__、get_normalizer、window_nums 公式等)**与新版本完全一致**。

================================================================================
"""
from typing import Dict
import numpy as np
import h5py
import torch
import zarr

from A_common.data.base_dataset import BaseVLADataset, SequenceSampler
from A_common.types.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer,
)
from A_common.types.normalizer_utils import (
    array_to_stats,
    robomimic_abs_action_only_normalizer_from_stat,
    get_range_normalizer_from_stat, get_image_range_normalizer,
    get_identity_normalizer_from_stat,
)
from A_common.logger import get_logger
from C_sim.robomimic.rotation_numpy import axis_angle_to_rotation_6d_batch

logger = get_logger(__name__)


class RobomimicZarrDatasetPadpForTestNoBatchEp(BaseVLADataset):
    """RTV8-aligned RobomimicZarrDataset(zarr v2 API 兼容版)**消融基线**。

    与 `C_sim.robomimic.zarr_dataset_padp.RobomimicZarrDatasetPadp` 行为等价,
    但用 zarr v2 API 实现,避免 'create_array' AttributeError。

    【消融基线特征】**没有**以下特性(用于和 `RobomimicZarrDatasetPadpForTest`
    balanced_sampler=True 路径做对照实验):
      - ❌ 不支持 `balanced_sampler` / `batch_size` / `sampler_seed` 参数
      - ❌ 不支持 `set_epoch(epoch_index)`(无 per-epoch 重建)
      - ❌ 不暴露 `balanced_sampler` / `batch_size` property
      - ❌ 不走 RTV8 `_build_and_save_epoch_mapping` 平衡列贪心分配
      - ✅ 保留所有 RTV8 数据对齐:
          * action 7 维 hdf5 → 10 维(同 `_convert_actions_v8`)
          * window_nums = real_len + horizon - 1(原始 real_len,非 padded)
          * __getitem__ 返回 dict{obs, action, window_info} 5 维 pos_info
          * 物理化填充:头尾各 pad horizon-1 帧(edge_repeat)
    """

    n_obs_steps: int = 1
    horizon: int = 40

    def __init__(self, shape_meta: dict, dataset_path: str,
                 n_demo: int = 200, horizon: int = 40, n_obs_steps: int = 1,
                 n_action_steps: int = 8, abs_action: bool = True,
                 use_legacy_normalizer: bool = False, seed: int = 42):
        super().__init__()
        self.shape_meta = shape_meta
        self.dataset_path = dataset_path
        self.n_demo = n_demo
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.abs_action = abs_action
        self.use_legacy_normalizer = use_legacy_normalizer
        self.seed = seed

        # 解析 obs keys
        self.rgb_keys = [k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "rgb"]
        self.lowdim_keys = [k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "low_dim"]

        # === 加载 hdf5,直接存到 zarr.MemoryStore ===
        store = zarr.MemoryStore()
        root = zarr.group(store=store)

        action_dim = shape_meta["action"]["shape"][0]
        rgb_shapes = {k: shape_meta["obs"][k]["shape"] for k in self.rgb_keys}
        lowdim_shapes = {k: shape_meta["obs"][k]["shape"] for k in self.lowdim_keys}

        with h5py.File(dataset_path, "r") as f:
            demos = f["data"]
            real_lens = []
            for i in range(n_demo):
                d = demos[f"demo_{i}"]
                real_lens.append(int(d["actions"].shape[0]))
            real_lens = np.asarray(real_lens, dtype=np.int64)

            # === 防漂移:以 hdf5 实际维度为准 ===
            actual_action_dim = int(demos[f"demo_0"]["actions"].shape[1])
            if action_dim != actual_action_dim:
                if abs_action and actual_action_dim == 7 and action_dim == 10:
                    # 预期:config 期望 10D(6D),hdf5 实际 7D,会做 7→10 转换
                    pass
                else:
                    logger.warning(
                        "[padp_for_test_no_batch_ep] config action_dim=%d 与 hdf5 实际 %d 不一致;"
                        "以 hdf5 为准(防御 config 漂移)。",
                        action_dim, actual_action_dim,
                    )
                    action_dim = actual_action_dim

            # 物理化填充
            pad = horizon - 1
            real_lens_new = real_lens + 2 * pad
            total = int(real_lens_new.sum())

            # 内部 action_dim(转换后维度)
            internal_action_dim = action_dim
            if abs_action and actual_action_dim == 7:
                internal_action_dim = 10

            # 创建数组 —— 全部用 zarr v2 API (create_dataset)
            data_g = root.create_group("data")
            meta_g = root.create_group("meta")

            action_arr = data_g.create_dataset(
                "action",
                shape=(total, internal_action_dim),
                chunks=(min(total, 1024), internal_action_dim),
                dtype="float32",
            )
            for k in self.lowdim_keys:
                shp = (total,) + tuple(lowdim_shapes[k])
                data_g.create_dataset(
                    k, shape=shp,
                    chunks=(min(total, 1024),) + tuple(lowdim_shapes[k]),
                    dtype="float32",
                )
            for k in self.rgb_keys:
                c, h, w = rgb_shapes[k]
                data_g.create_dataset(
                    k, shape=(total, h, w, c), chunks=(16, h, w, c), dtype="uint8",
                )

            pos_info_arr = data_g.create_dataset(
                "pos_info",
                shape=(total, 5),
                chunks=(min(total, 1024), 5),
                dtype="int64",
            )

            episode_ends = []
            ep_meta = []
            write_ptr = 0
            for i in range(n_demo):
                d = demos[f"demo_{i}"]
                rl = real_lens[i]
                rln = rl + 2 * pad

                # actions: 7→10 转换
                act = d["actions"][:].astype(np.float32)
                if abs_action and actual_action_dim == 7:
                    act = axis_angle_to_rotation_6d_batch(act)
                # 边缘复制填充
                padded_act = np.zeros((rln,) + act.shape[1:], dtype=np.float32)
                if pad > 0:
                    padded_act[:pad] = act[0:1].repeat(pad, axis=0)
                padded_act[pad:pad + rl] = act
                if pad > 0:
                    padded_act[pad + rl:] = act[-1:].repeat(pad, axis=0)
                action_arr[write_ptr:write_ptr + rln] = padded_act

                # lowdim
                for k in self.lowdim_keys:
                    arr = d["obs"][k][:].astype(np.float32)
                    out = np.zeros((rln,) + arr.shape[1:], dtype=np.float32)
                    if pad > 0:
                        out[:pad] = arr[0:1].repeat(pad, axis=0)
                    out[pad:pad + rl] = arr
                    if pad > 0:
                        out[pad + rl:] = arr[-1:].repeat(pad, axis=0)
                    data_g[k][write_ptr:write_ptr + rln] = out

                # rgb
                for k in self.rgb_keys:
                    arr = d["obs"][k][:]
                    out = np.zeros((rln,) + arr.shape[1:], dtype=np.uint8)
                    if pad > 0:
                        out[:pad] = arr[0:1].repeat(pad, axis=0)
                    out[pad:pad + rl] = arr
                    if pad > 0:
                        out[pad + rl:] = arr[-1:].repeat(pad, axis=0)
                    data_g[k][write_ptr:write_ptr + rln] = out

                # pos_info 5 维
                pos = np.zeros((rln, 5), dtype=np.int64)
                if pad > 0:
                    pos[:pad, 0] = 1
                if rl > 0:
                    pos[pad:pad + rl, 1] = np.arange(rl, dtype=np.int64)
                    if pad > 0:
                        pos[:pad, 1] = 0
                        pos[pad + rl:, 1] = rl - 1
                pos[:, 2] = np.arange(rln, dtype=np.int64)
                pos[:, 3] = i
                buf_start_1b = write_ptr + 1
                pos[:, 4] = buf_start_1b + pos[:, 2]
                pos_info_arr[write_ptr:write_ptr + rln] = pos

                ep_meta.append([i, rl, rln, write_ptr + 1, write_ptr + rln, rl + pad])
                episode_ends.append(write_ptr + rln)
                write_ptr += rln

            meta_g.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
            meta_g.create_dataset("episode_map", data=np.asarray(ep_meta, dtype=np.int64))

        self.replay_buffer = data_g
        self._internal_action_dim = internal_action_dim

        # ===============================================================
        # 【消融基线】单一 SequenceSampler(flat index)
        # ===============================================================
        # 与 RTV8 唯一共享的部分:
        #   - 公式:window_nums[i] = real_lens[i] + horizon - 1(传 real_lens 给 sampler)
        # 与 RTV8 不同的部分(本基线刻意保留):
        #   - 不建 per-epoch 全局映射表
        #   - DataLoader 走 shuffle=True,batch 内不强制不同 ep
        #   - 每 epoch 同 idx 取到同 (ep, win)
        # ===============================================================
        self.sampler = SequenceSampler(
            replay_buffer=data_g,
            episode_ends=np.asarray(episode_ends, dtype=np.int64),
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            pad_strategy="edge_repeat",
            real_lens=real_lens,                # 原始 hdf5 real_len(不传则用 padded,有 2*(h-1) 偏差)
        )
        logger.info(
            "[padp_for_test_no_batch_ep] %d demos, total %d windows, action_dim=%d (6D if abs_action),"
            " sampler=SequenceSampler(flat index, NO per-epoch balanced mapping)",
            n_demo, len(self.sampler), internal_action_dim,
        )

    def __len__(self) -> int:
        return len(self.sampler)
        # ⚠️ 【消融基线】**没有** set_epoch 方法
        # ⚠️ 【消融基线】**没有** balanced_sampler / batch_size property
        # 调用方须用 DataLoader(shuffle=True) 替代 RTV8 的 per-epoch 洗牌

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回 {obs, action, window_info} dict。"""
        ep_id, win = self.sampler.locate(idx)
        ep_start = int(self.sampler.episode_starts[ep_id])
        ep_real_len = int(self.sampler.real_lens[ep_id])
        abs_start = ep_start + win

        H = self.horizon
        S = self.n_obs_steps
        D = self._internal_action_dim

        # action: edge_repeat 取 [win, win+H)
        action_buf = self.replay_buffer["action"]
        out = np.zeros((H,) + action_buf.shape[1:], dtype=np.float32)
        for i in range(H):
            pos = win + i
            if pos < 0:
                src = 0
            elif pos >= ep_real_len:
                src = ep_real_len - 1
            else:
                src = pos
            out[i] = action_buf[ep_start + src]
        action = torch.from_numpy(out.astype(np.float32))   # [H, D]

        # obs: 取 [abs_start, abs_start+S)
        obs_dict: Dict[str, torch.Tensor] = {}
        for k in self.rgb_keys:
            seq = self.replay_buffer[k][abs_start: abs_start + S]
            seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0
            obs_dict[k] = torch.from_numpy(seq)
        for k in self.lowdim_keys:
            seq = self.replay_buffer[k][abs_start: abs_start + S].astype(np.float32)
            obs_dict[k] = torch.from_numpy(seq)
        if self.lowdim_keys:
            obs_dict["state"] = torch.cat(
                [obs_dict[k].reshape(S, -1) for k in self.lowdim_keys], dim=-1
            )

        # window_info 5 维
        pos = np.zeros((H, 5), dtype=np.int64)
        if win < 0 or win >= ep_real_len:
            pos[:, 0] = 1
        if ep_real_len > 0:
            ei = min(max(win, 0), ep_real_len - 1)
            pos[:, 1] = ei
        pos[:, 2] = np.arange(H, dtype=np.int64)
        pos[:, 3] = ep_id
        pos[:, 4] = abs_start + np.arange(H, dtype=np.int64) + 1

        return {
            "obs": obs_dict,
            "action": action,                 # [H, D]
            "window_info": torch.from_numpy(pos),  # [H, 5]
        }

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """action 10D 逐维 max_abs scale,lowdim 用 range/identity,rgb 用 [0,1]。"""
        normalizer = LinearNormalizer()

        act = self.replay_buffer["action"][:]
        stat = array_to_stats(act)
        dim = stat["mean"].shape[-1]
        if self.abs_action and dim == 7:
            if self.use_legacy_normalizer:
                max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
                scale = 1.0 / max_abs
                offset = np.zeros_like(stat["max"])
                normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
                    scale=scale, offset=offset, input_stats_dict=stat,
                )
            else:
                normalizer["action"] = robomimic_abs_action_only_normalizer_from_stat(stat)
        elif self.abs_action and dim == 10:
            # 10D 黄金标准:逐维 max_abs scale 到 [-1, 1]
            max_abs = np.maximum(np.abs(stat["min"]), np.abs(stat["max"]))
            scale = 1.0 / np.maximum(max_abs, 1e-7)
            offset = np.zeros(dim, dtype=np.float32)
            normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
                scale=scale.astype(np.float32),
                offset=offset,
                input_stats_dict=stat,
            )
        else:
            normalizer["action"] = SingleFieldLinearNormalizer.create_identity(dim)

        for k in self.lowdim_keys:
            arr = self.replay_buffer[k][:]
            stat = array_to_stats(arr)
            if k.endswith("pos") or k.endswith("qpos"):
                normalizer[k] = get_range_normalizer_from_stat(stat)
            else:
                normalizer[k] = get_identity_normalizer_from_stat(stat)

        for k in self.rgb_keys:
            normalizer[k] = get_image_range_normalizer()
        return normalizer
