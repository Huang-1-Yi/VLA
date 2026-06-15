# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""C_sim.robomimic.padp_for_test_dataset —— padp_for_test 专用 RTV8-aligned Dataset。

==========================================================================
🚨 消融实验文件(ABLATION EXPERIMENT FILE)🚨
==========================================================================

本文件是 **消融实验"实验组"代码**,与同目录 `_no_batch_ep.py`(消融基线)
构成一对。**严禁任何形式的删除/重构/合并;任何修改必须先经用户确认**。

消融目的
--------
  验证 RTV8-aligned "per-batch 跨 ep 平衡"(BalancedColumnsSampler)对
  Diffusion Policy 训练效果的贡献度。

消融对照表
----------
  ┌─────────┬─────────────────────────────────┬──────────────────────────┐
  │ 组别     │ 配置                             │ 走的 sampler             │
  ├─────────┼─────────────────────────────────┼──────────────────────────┤
  │ 实验组   │ 本文件 + balanced_sampler=True  │ BalancedColumnsSampler   │
  │         │                                  │ (行内强制不同 ep)        │
  │ 对照组 1 │ 本文件 + balanced_sampler=False │ SequenceSampler          │
  │         │                                  │ (flat index + DataLoader │
  │         │                                  │  shuffle)                │
  │ 对照组 2 │ 切到 _no_batch_ep.py 导入       │ SequenceSampler          │
  │ (基线)  │ (字节级等价于对照组 1,           │ (独立 class 快照)        │
  │         │  作 sanity check)                │                          │
  └─────────┴─────────────────────────────────┴──────────────────────────┘

TODO 清单(实验待跑)
-------------------
  [ ] 跑实验组:   balanced_sampler=True, 记录 test_mean_score
  [ ] 跑对照组 1: balanced_sampler=False, 记录 test_mean_score
  [ ] 跑对照组 2: import 切到 _no_batch_ep.py,记录 test_mean_score
  [ ] 三组差异 < 0.5% →  消融无显著影响(论文可写"balanced sampler 不影响")
  [ ] 三组差异 > 2%   →  balanced sampler 是必要设计(论文可写"创新点")
  [ ] 同时记录 train_loss 曲线、sample entropy、gradient norm
  [ ] 跑 3 个不同 seed(42 / 123 / 2024)取均值

不可删/不可改红线(违反前必须先 @用户确认)
-----------------------------------------
  ❌ 不可删 `balanced_sampler` / `batch_size` / `sampler_seed` 三个参数
  ❌ 不可删 `set_epoch()` 方法
  ❌ 不可删 `balanced_sampler` / `batch_size` 两个 @property
  ❌ 不可删 `if self._balanced_sampler:` 分支(否则 balanced 模式失效)
  ❌ 不可删 `_no_batch_ep.py`(对照组)
  ❌ 不可改 window_nums 公式(real_lens + horizon - 1,已与 RTV8 对齐)
  ❌ 不可改 __getitem__ 中 `self.sampler.locate(idx)` 调用接口
  ❌ 不可改 get_normalizer 的 10D 逐维 max_abs scale 公式

相关文件
--------
  - C_sim/robomimic/padp_for_test_dataset_no_batch_ep.py   (消融基线)
  - A_common/data/base_dataset.py::BalancedColumnsSampler  (sampler 实现)
  - A_common/data/base_dataset.py::SequenceSampler         (对照组 sampler)
  - E_cti/train/verify_balanced_sampler.py                  (4 项验证脚本)
  - E_cti/train/padp_for_test_train.py                      (训练入口)
  - E_cti/configs/padp_for_test_golden.yaml                 (yaml 配置)
==========================================================================


> **为什么新增?** VLA 现有 `RobomimicZarrDatasetPadp` 用了 zarr v3 API (`create_array`),
> 本环境装的是 zarr 2.12,只有 v2 API (`create_dataset`)。直接调现有 dataset 会
> `AttributeError: 'Group' object has no attribute 'create_array'`。
> 用户禁止修改现有代码,故新增本文件,行为 1:1 等价于 RTV8-aligned 路径。

行为契约 (与 `RobomimicZarrDatasetPadp` 保持一致):
  - action 7 维 hdf5 → 10 维 (pos3 + rot6d6 + gripper1),与 RTV8 _convert_actions_v8 同公式
  - window_nums = real_len + horizon - 1(原始 real_len,非 padded)
  - __getitem__ 返回 dict{obs, action, window_info},window_info 是 5 维 pos_info
    (is_warmup, episode_index, episode_new_pos_id, episode_id, buffer_pos_id)
  - 不走 PADP RTV8 的 per-epoch mapping(DataLoader 用 shuffle=True 替代)
  - 物理化填充:头尾各 pad horizon-1 帧(用 edge_repeat 复制首尾)

实现差异 (本文件 vs 现有 RobomimicZarrDatasetPadp):
  - zarr 全部用 v2 API:`Group.create_dataset(...)`,而不是 `create_array(...)`
  - axis_angle → 6D 转换用本地 `C_sim.robomimic.rotation_numpy` (与现有实现一致)
  - normalizer 逻辑同 `RobomimicZarrDatasetPadp.get_normalizer` (10D 用逐维 max_abs scale)
"""
from typing import Dict, Optional
import numpy as np
import h5py
import torch
import zarr

from A_common.data.base_dataset import BaseVLADataset, SequenceSampler, BalancedColumnsSampler
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


class RobomimicZarrDatasetPadpForTest(BaseVLADataset):
    """RTV8-aligned RobomimicZarrDataset (zarr v2 API 兼容版)。

    与 `C_sim.robomimic.zarr_dataset_padp.RobomimicZarrDatasetPadp` 行为等价,
    但用 zarr v2 API 实现,避免 'create_array' AttributeError。
    """

    n_obs_steps: int = 1
    horizon: int = 40

    def __init__(self, shape_meta: dict, dataset_path: str,
                 n_demo: int = 200, horizon: int = 40, n_obs_steps: int = 1,
                 n_action_steps: int = 8, abs_action: bool = True,
                 use_legacy_normalizer: bool = False, seed: int = 42,
                 # ========== 新增:RTV8-aligned balanced-columns sampler 开关 ==========
                 balanced_sampler: bool = False,
                 batch_size: Optional[int] = None,
                 sampler_seed: int = 42):
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
                        "[padp_for_test] config action_dim=%d 与 hdf5 实际 %d 不一致;"
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

        # ===================================================================
        # Sampler 选择:flat index (SequenceSampler) vs balanced columns
        # ===================================================================
        # 两种 sampler 的 window_nums 公式都是 real_len + horizon - 1(已对齐 RTV8)
        # 区别只在 batch 组成:
        #   - SequenceSampler:  __len__ = sum(window_nums),DataLoader shuffle=True
        #   - BalancedColumnsSampler: __len__ = B * max_cols,DataLoader shuffle=False
        #     (per-epoch 重排,行内必不同 ep,数学逻辑与 RTV8 1:1)
        # ===================================================================
        self._balanced_sampler = bool(balanced_sampler)
        self._batch_size = int(batch_size) if batch_size is not None else None
        if self._balanced_sampler:
            assert self._batch_size is not None and self._batch_size > 0, (
                "balanced_sampler=True 时必须显式传 batch_size > 0"
            )
            self.sampler = BalancedColumnsSampler(
                episode_ends=np.asarray(episode_ends, dtype=np.int64),
                real_lens=real_lens,
                horizon=horizon,
                batch_size=self._batch_size,
                seed=sampler_seed,
                epoch_index=0,
            )
        else:
            # SequenceSampler 用原始 real_lens 保证 window_nums = real_len + horizon - 1
            self.sampler = SequenceSampler(
                replay_buffer=data_g,
                episode_ends=np.asarray(episode_ends, dtype=np.int64),
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                pad_strategy="edge_repeat",
                real_lens=real_lens,
            )
        logger.info(
            "[padp_for_test] %d demos, total %d windows, action_dim=%d (6D if abs_action),"
            " sampler=%s%s",
            n_demo, len(self.sampler), internal_action_dim,
            type(self.sampler).__name__,
            f" (batch_size={self._batch_size})" if self._balanced_sampler else "",
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def set_epoch(self, epoch_index: int) -> None:
        """balanced_sampler 模式下,每 epoch 重建 _global_mapping(per-epoch 洗牌)。

        SequenceSampler 模式下为空操作(no-op),保持向后兼容。
        """
        if self._balanced_sampler and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch_index)

    @property
    def balanced_sampler(self) -> bool:
        """True 表示当前 dataset 启用了 RTV8-aligned balanced-columns sampler。

        调用方(DataLoader 配置)据此决定:
          - shuffle=True  vs  shuffle=False
          - drop_last=cfg.get("drop_last", False)  vs  drop_last=True
        """
        return self._balanced_sampler

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

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
