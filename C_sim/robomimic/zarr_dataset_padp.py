"""C_sim.robomimic.zarr_dataset_padp —— RTV8-对齐版,产出 {obs, action, window_info} dict。

RTV8 对齐(与 PADP_v3/diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py 一致):
    - action: 7 维 hdf5 → 10 维 (pos3 + rot6d6 + gripper1),与 RTV8 _convert_actions_v8 同公式
    - window_nums = real_len + horizon - 1(原始 real_len,非 padded)
    - __getitem__ 返回 dict{obs, action, window_info},window_info 是 5 维 pos_info
      (is_warmup, episode_index, episode_new_pos_id, episode_id, buffer_pos_id)
    - 不走 PADP RTV8 的 per-epoch mapping(DataLoader 用 shuffle=True 替代)

无 pytorch3d 依赖:用本地 `C_sim.robomimic.rotation_numpy` 的 numpy Rodrigues
实现(数值与 pytorch3d 在 float32 ULP 内一致)。

设计简化:不走 PADP RTV8 的 per-epoch mapping,直接 __len__ = sum(window_nums)。
"""
import os
from typing import Dict, Tuple, List, Optional
import numpy as np
import h5py
import torch

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


def _robomimic_action_only_normalizer_from_stat(stat, use_legacy: bool = False):
    """PADP 的 10 维 robomimic abs action normalizer(RTV8 对齐)。"""
    if use_legacy:
        max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
        scale = 1.0 / max_abs
        offset = np.zeros_like(stat["max"])
        return SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stat,
        )
    return robomimic_abs_action_only_normalizer_from_stat(stat)


class RobomimicZarrDatasetPadp(BaseVLADataset):
    """RTV8-aligned RobomimicZarrDataset。

    Args:
        shape_meta:        dict,obs 含 rgb / low_dim,action.shape 应是 [10](已 6D 化)
        dataset_path:      hdf5 路径(7 维 abs action 原始数据)
        n_demo:            加载多少集
        horizon:           预测窗口长度
        n_obs_steps:       历史观测窗口
        n_action_steps:    一次推理出的动作数
        abs_action:        True(RTV8 黄金)
        use_legacy_normalizer: 是否用 legacy 7D normalizer
        seed:              随机种子(预留)
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

        # === 加载 hdf5,直接存到 zarr MemoryStore ===
        # 2026-06-14 适配 zarr v3 API:MemoryStore 已迁到 zarr.storage;create_dataset 仍可但
        # 写 data=... 的形式 v3 不支持,改用 create_array。
        import zarr
        store = zarr.storage.MemoryStore()
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

            # === 防漂移:以 hdf5 实际维度为准,config 漂移时 warning 但不报错 ===
            actual_action_dim = int(demos[f"demo_0"]["actions"].shape[1])
            if action_dim != actual_action_dim:
                # 如果 config 期望 10D(已 6D)而 hdf5 实际 7D,我们会做转换,所以 warning 即可
                # 反之如果 config 是 7D 而 hdf5 是 10D,则以 hdf5 为准
                if abs_action and actual_action_dim == 7 and action_dim == 10:
                    logger.info(
                        "[RobomimicZarrDatasetPadp] config action_dim=%d(6D) 与 hdf5 实际 %d(axis_angle) 一致;"
                        "将在转换阶段 axis_angle → 6D。", action_dim, actual_action_dim,
                    )
                else:
                    logger.warning(
                        "[RobomimicZarrDatasetPadp] config action_dim=%d 与 hdf5 实际 %d 不一致;"
                        "以 hdf5 为准(防御 config 漂移)。请把 E_cti/configs/*.yaml 的 "
                        "data.shape_meta.action.shape 改成 [%d]",
                        action_dim, actual_action_dim, actual_action_dim,
                    )
                    action_dim = actual_action_dim
            # abs_action + 7D hdf5 → 内部 action_dim = 10
            if abs_action and actual_action_dim == 7 and action_dim == 10:
                # 这是预期路径:写到 zarr 时维度用 10
                pass

            # 物理化填充(头尾各 pad horizon-1 帧)
            pad = horizon - 1
            real_lens_new = real_lens + 2 * pad
            total = int(real_lens_new.sum())

            # 内部 action_dim(转换后维度)
            internal_action_dim = action_dim  # 默认同 hdf5
            if abs_action and actual_action_dim == 7 and self.use_axis_angle_to_6d():
                internal_action_dim = 10

            # 创建数组
            data_g = root.create_group("data")
            meta_g = root.create_group("meta")

            action_arr = data_g.create_array(
                "action", shape=(total, internal_action_dim), chunks=(min(total, 1024), internal_action_dim),
                dtype="float32",
            )
            for k in self.lowdim_keys:
                shp = (total,) + tuple(lowdim_shapes[k])
                data_g.create_array(k, shape=shp, chunks=(min(total, 1024),) + tuple(lowdim_shapes[k]),
                                     dtype="float32")
            for k in self.rgb_keys:
                c, h, w = rgb_shapes[k]
                # HWC 存
                data_g.create_array(k, shape=(total, h, w, c), chunks=(16, h, w, c), dtype="uint8")

            episode_ends = []
            ep_meta = []
            write_ptr = 0
            for i in range(n_demo):
                d = demos[f"demo_{i}"]
                rl = real_lens[i]
                rln = rl + 2 * pad
                # actions
                act = d["actions"][:].astype(np.float32)
                if abs_action and actual_action_dim == 7:
                    # 7 维 robomimic abs action: pos(3) + axis_angle(3) + gripper(1)
                    # 与 RTV8 _convert_actions_v8 一致,转 rotation_6d,action_dim=10
                    # 纯 numpy 实现,无 pytorch3d 依赖
                    act = axis_angle_to_rotation_6d_batch(act)
                # 填充
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
                    arr = d["obs"][k][:]   # (T, h, w, c) uint8
                    out = np.zeros((rln,) + arr.shape[1:], dtype=np.uint8)
                    if pad > 0:
                        out[:pad] = arr[0:1].repeat(pad, axis=0)
                    out[pad:pad + rl] = arr
                    if pad > 0:
                        out[pad + rl:] = arr[-1:].repeat(pad, axis=0)
                    data_g[k][write_ptr:write_ptr + rln] = out

                # pos_info 5-elem 格式(同 RTV8:704-723)
                #   col 0: is_warmup (前 pad 段 = 1)
                #   col 1: episode_index(钳到 [0, rl-1])
                #   col 2: episode_new_pos_id(0..rln-1)
                #   col 3: episode_id(i)
                #   col 4: buffer_pos_id(1-based,全局)
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
                # 写入 pos_info(同 RTV8 的 data[pos_info_arr])
                if "pos_info" not in data_g.array_keys():
                    pos_info_arr = data_g.create_array(
                        "pos_info",
                        shape=(total, 5),
                        chunks=(min(total, 1024), 5),
                        dtype="int64",
                    )
                else:
                    pos_info_arr = data_g["pos_info"]
                pos_info_arr[write_ptr:write_ptr + rln] = pos

                # episode_map(同 RTV8):[ep_id, real_len, real_len_new, buf_start(1b), buf_end(1b), window_num]
                ep_meta.append([i, rl, rln, write_ptr + 1, write_ptr + rln, rl + pad])
                episode_ends.append(write_ptr + rln)
                write_ptr += rln

            meta_g.create_array("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
            meta_g.create_array("episode_map", data=np.asarray(ep_meta, dtype=np.int64))

        self.replay_buffer = data_g  # SequenceSampler 期望 rb 直接含 'action' / rgb / lowdim 键
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
            # === 关键修复:传 original real_lens(不是 padded),让 window_nums = real_len + horizon - 1 ===
            self.sampler = SequenceSampler(
                replay_buffer=data_g,
                episode_ends=np.asarray(episode_ends, dtype=np.int64),
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                pad_strategy="edge_repeat",
                real_lens=real_lens,                # 原始 hdf5 real_len(不传则用 padded,有 2*(h-1) 偏差)
            )
        logger.info(
            "RobomimicZarrDatasetPadp: %d demos, total %d windows, action_dim=%d (6D if abs_action),"
            " sampler=%s%s",
            n_demo, len(self.sampler), internal_action_dim,
            type(self.sampler).__name__,
            f" (batch_size={self._batch_size})" if self._balanced_sampler else "",
        )

    def use_axis_angle_to_6d(self) -> bool:
        """是否做 axis_angle → 6D 转换(由 abs_action + 实际 hdf5 维度共同决定)。"""
        return self.abs_action

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
        """RTV8-aligned:返回 {obs, action, window_info} dict。

        window_info: [H, 5] = [is_warmup, episode_index, episode_new_pos_id,
                                 episode_id, buffer_pos_id(1-based)]
        """
        ep_id, win = self.sampler.locate(idx)
        ep_start = int(self.sampler.episode_starts[ep_id])
        ep_real_len = int(self.sampler.real_lens[ep_id])
        abs_start = ep_start + win

        H = self.horizon
        S = self.n_obs_steps
        D = self._internal_action_dim

        # ============ action: 沿 ep 局部坐标,edge_repeat 取 [win, win+H) ============
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

        # ============ obs: 取 [abs_start, abs_start+S) ============
        obs_dict: Dict[str, torch.Tensor] = {}
        for k in self.rgb_keys:
            seq = self.replay_buffer[k][abs_start: abs_start + S]
            seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0
            obs_dict[k] = torch.from_numpy(seq)
        for k in self.lowdim_keys:
            seq = self.replay_buffer[k][abs_start: abs_start + S].astype(np.float32)
            obs_dict[k] = torch.from_numpy(seq)
        # 拼 state(所有 lowdim concat,沿用 C_sim 传统)
        if self.lowdim_keys:
            obs_dict["state"] = torch.cat(
                [obs_dict[k].reshape(S, -1) for k in self.lowdim_keys], dim=-1
            )

        # ============ window_info(5-elem pos_info, 同 RTV8:704-723) ============
        pos = np.zeros((H, 5), dtype=np.int64)
        # col 0: is_warmup — 起点 win 落在 pad 段,则全段都算 warmup
        if win < 0 or win >= ep_real_len:
            pos[:, 0] = 1
        # col 1: episode_index — 钳到 [0, real_len-1]
        if ep_real_len > 0:
            ei = min(max(win, 0), ep_real_len - 1)
            pos[:, 1] = ei
        # col 2: episode_new_pos_id — 0..H-1(窗口内偏移)
        pos[:, 2] = np.arange(H, dtype=np.int64)
        # col 3: episode_id
        pos[:, 3] = ep_id
        # col 4: buffer_pos_id — 1-based 全局下标
        pos[:, 4] = abs_start + np.arange(H, dtype=np.int64) + 1

        return {
            "obs": obs_dict,
            "action": action,                 # [H, D]
            "window_info": torch.from_numpy(pos),  # [H, 5]
        }

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """参考 PADP get_normalizer:action 用 abs-only normalizer(7D/10D),lowdim 用 range/image。

        - 7D:  pos range [-1, 1] + axis_angle identity + gripper range [-1, 1]
        - 10D: 逐维 max_abs scale(整条动作归一到 [-1, 1] 内,等价 RTV8 `use_legacy_normalizer` 路径)
        """
        normalizer = LinearNormalizer()

        act = self.replay_buffer["action"][:]
        stat = array_to_stats(act)
        dim = stat["mean"].shape[-1]
        if self.abs_action and dim == 7:
            this_n = _robomimic_action_only_normalizer_from_stat(
                stat, use_legacy=self.use_legacy_normalizer,
            )
        elif self.abs_action and dim == 10:
            # 10D:逐维 max_abs 缩放到 [-1, 1]
            max_abs = np.maximum(np.abs(stat["min"]), np.abs(stat["max"]))  # (10,)
            scale = 1.0 / np.maximum(max_abs, 1e-7)                          # (10,)
            offset = np.zeros(dim, dtype=np.float32)
            this_n = SingleFieldLinearNormalizer.create_manual(
                scale=scale.astype(np.float32),
                offset=offset,
                input_stats_dict=stat,
            )
        else:
            this_n = SingleFieldLinearNormalizer.create_identity(dim)
        normalizer["action"] = this_n

        # lowdim obs
        for k in self.lowdim_keys:
            arr = self.replay_buffer[k][:]
            stat = array_to_stats(arr)
            if k.endswith("pos") or k.endswith("qpos"):
                this_n = get_range_normalizer_from_stat(stat)
            elif k.endswith("quat"):
                this_n = get_identity_normalizer_from_stat(stat)
            else:
                this_n = get_identity_normalizer_from_stat(stat)
            normalizer[k] = this_n

        # rgb 用固定 [0,1] 归一化
        for k in self.rgb_keys:
            normalizer[k] = get_image_range_normalizer()
        return normalizer
