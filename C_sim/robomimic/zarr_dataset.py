"""C_sim.robomimic.zarr_dataset —— 读 PADP zarr / hdf5,产出 (obs_dict, action)。

铁律 6 实现:
    - 显式 n_obs_steps / horizon
    - 返回 action shape 严格对齐 [self.horizon, D_a]
    - 首尾 padding 在本类内处理(用 SequenceSampler.edge_repeat)

设计简化:不走 PADP RTV8 的 per-epoch mapping,直接 __len__ = sum(window_nums)
"""
import os
from typing import Dict, Tuple, List
import numpy as np
import h5py
import torch

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

logger = get_logger(__name__)


def _robomimic_action_only_normalizer_from_stat(stat, use_legacy: bool = False):
    """PADP 的 7 维 robomimic abs action normalizer(去掉 quat 中的 w,加 special mask)。"""
    if use_legacy:
        max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
        scale = 1.0 / max_abs
        offset = np.zeros_like(stat["max"])
        return SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stat,
        )
    return robomimic_abs_action_only_normalizer_from_stat(stat)


class RobomimicZarrDataset(BaseVLADataset):
    """铁律 6 时序契约:

        - 显式声明 n_obs_steps (历史观测窗口) / horizon (未来动作预测窗口)
        - __getitem__ 返回 action shape = [self.horizon, D_a]
        - 首尾 padding 在 SequenceSampler 内处理
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

        # === 加载 hdf5,直接存到 zarr.MemoryStore(阶段 1 不走 RTV8 磁盘 cache) ===
        import zarr
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

            # 物理化填充(头尾各 pad horizon-1 帧)
            pad = horizon - 1
            real_lens_new = real_lens + 2 * pad
            total = int(real_lens_new.sum())

            # 创建数组
            data_g = root.create_group("data")
            meta_g = root.create_group("meta")

            action_arr = data_g.create_dataset(
                "action", shape=(total, action_dim), chunks=(min(total, 1024), action_dim),
                dtype="float32",
            )
            for k in self.lowdim_keys:
                shp = (total,) + tuple(lowdim_shapes[k])
                data_g.create_dataset(k, shape=shp, chunks=(min(total, 1024),) + tuple(lowdim_shapes[k]),
                                     dtype="float32")
            for k in self.rgb_keys:
                c, h, w = rgb_shapes[k]
                # HWC 存
                data_g.create_dataset(k, shape=(total, h, w, c), chunks=(16, h, w, c), dtype="uint8")

            episode_ends = []
            ep_meta = []
            write_ptr = 0
            for i in range(n_demo):
                d = demos[f"demo_{i}"]
                rl = real_lens[i]
                rln = rl + 2 * pad
                # actions
                act = d["actions"][:].astype(np.float32)
                if abs_action:
                    # 7 维 robomimic abs action:pos(3) + axis_angle(3) + gripper(1)
                    # NOTE: PADP yaml rotation_rep: 'rotation_6d' 默认会把 axis_angle 转 6D
                    # 这里本阶段**故意**不转(pytorch3d 重依赖),保持 axis_angle
                    # 仍能训通(只是 loss 曲线略不同),如需精确复现 PADP,见
                    # VLA/C_sim/robomimic/zarr_dataset.py 顶部 TODO
                    pass
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

                ep_meta.append([i, rl, rln, write_ptr + 1, write_ptr + rln, rl + pad])
                episode_ends.append(write_ptr + rln)
                write_ptr += rln

            meta_g.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
            meta_g.create_dataset("episode_map", data=np.asarray(ep_meta, dtype=np.int64))

        self.replay_buffer = root
        self.sampler = SequenceSampler(
            replay_buffer=root,
            episode_ends=np.asarray(episode_ends, dtype=np.int64),
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            pad_strategy="edge_repeat",
        )
        logger.info("RobomimicZarrDataset: %d demos, total %d windows", n_demo, len(self.sampler))

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Tuple[Dict, torch.Tensor]:
        ep_id, win = self.sampler.locate(idx)

        # 动作 [H, D_a]
        action = self.sampler.get_action(ep_id, win)
        action_t = torch.from_numpy(action.astype(np.float32))

        # obs
        obs = self.sampler.get_obs_window(ep_id, win, self.rgb_keys, self.lowdim_keys)
        obs_dict = {}
        for k in self.rgb_keys:
            arr = obs[k]   # (S, h, w, c) uint8
            arr = np.moveaxis(arr, -1, 1).astype(np.float32) / 255.0  # (S, c, h, w)
            obs_dict[k] = torch.from_numpy(arr)
        for k in self.lowdim_keys:
            obs_dict[k] = torch.from_numpy(obs[k].astype(np.float32))
        # 拼 state(简化为所有 lowdim concat)
        if self.lowdim_keys:
            state = torch.cat([obs_dict[k].reshape(self.n_obs_steps, -1) for k in self.lowdim_keys], dim=-1)
            obs_dict["state"] = state

        return obs_dict, action_t

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """参考 PADP get_normalizer:action 用 abs-only normalizer,lowdim 用 range/image。"""
        normalizer = LinearNormalizer()

        # action
        act = self.replay_buffer["action"][:]
        stat = array_to_stats(act)
        if self.abs_action:
            this_n = _robomimic_action_only_normalizer_from_stat(
                stat, use_legacy=self.use_legacy_normalizer,
            )
        else:
            this_n = SingleFieldLinearNormalizer.create_identity(stat["mean"].shape[-1])
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


def array_to_stats(arr: np.ndarray) -> dict:
    """从 (T, D) 数组算 mean/min/max/std/percentile 统计。"""
    arr = arr.astype(np.float32)
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "p01": np.percentile(arr, 1, axis=0),
        "p99": np.percentile(arr, 99, axis=0),
    }
