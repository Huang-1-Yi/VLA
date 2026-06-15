# ============================================================
# PADP-VLA v1.2
# 1.2 版本,新加 lerobot parquet 数据源支持(为 LIBERO 训练做准备)
# ============================================================

"""C_sim.robomimic.padp_for_libero_dataset —— padp_for_libero 专用 RTV8-aligned Dataset (lerobot)。

==========================================================================
v1.2 lerobot 数据集版(独立线,跟 padp_for_test 并行)
==========================================================================

本文件 fork 自 `padp_for_test_dataset.py` (1.1 干净版),但**数据源**:
  - padp_for_test (1.1):  robomimic 原生 hdf5, 5 个 lowdim keys (pos/quat/qpos)
  - padp_for_libero (1.2): lerobot parquet, 单一 8D 'observation.state' LIBERO 标准

两条线**共享**:`__getitem__` / `get_normalizer` / `set_epoch` / sampler 选择 / normalizer 公式
两条线**独立**:`__init__` 加载块(本文件改写为 lerobot parquet loader)

调用方:
  - `E_cti/train/padp_for_libero_train.py`:monkey-patch 替换 dataset 类
  - 配置文件:`E_cti/configs/lerobot_libero.yaml` / `lerobot_libero10d.yaml`

行为契约 (跟 1.1 保持一致):
  - window_nums = real_len + horizon - 1
  - __getitem__ 返回 dict{obs, action, window_info}
  - normalizer 10D 逐维 max_abs scale
  - 物理化填充:头尾各 pad horizon-1 帧(用 edge_repeat 复制首尾)
  - balanced_sampler / batch_size / set_epoch 行为一致

lerobot 路径差异 (vs 1.1 hdf5):
  - 数据源:HF `datasets.load_from_disk()` 读 parquet
  - 边界重建:从 `meta/episode_ends.json` 读 (由 convert_to_lerobot.py 写)
  - 7→10 转换:lerobot 端已经做了 (`--action-dim 10`),dataset 内部不再做
  - state 字段:8D LIBERO 标准 (pos3 + quat_xyzw + gripper1),不是 hdf5 的 5 个 lowdim
==========================================================================
"""
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np
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

logger = get_logger(__name__)


class RobomimicZarrDatasetPadpForLibero(BaseVLADataset):
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

        # === 加载 lerobot parquet,直接存到 zarr.MemoryStore (跟 1.1 hdf5 路径同样的 schema) ===
        from datasets import load_from_disk
        dataset_path_p = Path(dataset_path)
        if not (dataset_path_p / "meta" / "episode_ends.json").is_file():
            raise FileNotFoundError(
                f"lerobot 路径 {dataset_path_p} 缺 meta/episode_ends.json, "
                f"请先用 convert_to_lerobot.py 生成。"
            )
        with open(dataset_path_p / "meta" / "episode_ends.json") as f:
            ep_meta_json = json.load(f)
        full_real_lens = ep_meta_json["real_lens"]
        full_episode_ends = ep_meta_json["episode_ends"]
        n_ep_available = len(full_real_lens)
        n_ep = min(n_demo, n_ep_available)
        real_lens = np.asarray(full_real_lens[:n_ep], dtype=np.int64)
        episode_ends_parquet = full_episode_ends[:n_ep]

        # 读 parquet (HF datasets),一次性转 numpy 格式,避免逐行 .with_format 的开销
        ds = load_from_disk(str(dataset_path_p))
        # ds[0]["action"] 默认是 list (Sequence feature), 转 numpy 后才能用 .shape
        try:
            ds = ds.with_format("numpy")
        except Exception:
            logger.warning("[padp_for_libero] datasets.with_format('numpy') 不支持,"
                           " 退回到 np.asarray 包裹 (略慢)")
        n_total_parquet = len(ds)
        # 取前 n_ep 个 episode 的总帧数(== episode_ends_parquet[-1])
        assert n_total_parquet >= episode_ends_parquet[-1], (
            f"parquet 总帧数 {n_total_parquet} < episode_ends[-1] {episode_ends_parquet[-1]}"
        )
        logger.info("[padp_for_libero] lerobot 加载: %d 帧, %d 个 ep (限 %d)",
                    episode_ends_parquet[-1], n_ep, n_demo)

        # === v1.2.4 路消融:config 跟 parquet 必须严格一致 ===
        action_dim_cfg = shape_meta["action"]["shape"][0]
        actual_action_dim = int(ds[0]["action"].shape[-1]) if episode_ends_parquet[-1] > 0 else action_dim_cfg
        if action_dim_cfg != actual_action_dim:
            raise ValueError(
                f"[padp_for_libero] config action.shape={action_dim_cfg} 跟 parquet 实际 "
                f"action.shape[-1]={actual_action_dim} 不一致。"
                f" 7D 用 `lerobot_libero7d.yaml` (action.shape=[7]),"
                f" 10D 用 `lerobot_libero10d.yaml` (action.shape=[10])。"
            )
        action_dim = action_dim_cfg

        # 物理化填充(跟 1.1 一致:头尾各 pad horizon-1 帧, edge_repeat)
        pad = horizon - 1
        real_lens_new = real_lens + 2 * pad
        total = int(real_lens_new.sum())

        # lerobot 端已经做了 7→10 转换(--action-dim 10),dataset 内部不再做
        internal_action_dim = action_dim

        # 创建 zarr v2 arrays(跟 1.1 同样的 schema:data_g + meta_g)
        store = zarr.MemoryStore()
        root = zarr.group(store=store)
        data_g = root.create_group("data")
        meta_g = root.create_group("meta")

        action_arr = data_g.create_dataset(
            "action",
            shape=(total, internal_action_dim),
            chunks=(min(total, 1024), internal_action_dim),
            dtype="float32",
        )
        rgb_shapes = {k: shape_meta["obs"][k]["shape"] for k in self.rgb_keys}
        lowdim_shapes = {k: shape_meta["obs"][k]["shape"] for k in self.lowdim_keys}
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

        # 写入数据:用 HF datasets 的索引访问(0-based),按 ep 边界切片
        episode_ends = []
        ep_meta_list = []
        write_ptr = 0
        prev_end = 0
        for i in range(n_ep):
            ep_end = episode_ends_parquet[i]  # 该 ep 在 parquet 中的 1-based 累计结束位置
            rl = int(real_lens[i])
            rln = rl + 2 * pad

            # 用 ds[i_start:i_end] 切片读一个 ep
            ep_data = ds.select(range(prev_end, ep_end))
            # select 后的子集可能丢失 numpy 格式, 重新 with_format
            try:
                ep_data = ep_data.with_format("numpy")
            except Exception:
                pass  # 老版本不支持时, 后续用 np.asarray 兜底

            # actions:不转换(lerobot 已经是 7/10D)
            act = np.stack([row["action"] for row in ep_data]).astype(np.float32)
            # 边缘复制填充
            padded_act = np.zeros((rln,) + act.shape[1:], dtype=np.float32)
            if pad > 0:
                padded_act[:pad] = act[0:1].repeat(pad, axis=0)
            padded_act[pad:pad + rl] = act
            if pad > 0:
                padded_act[pad + rl:] = act[-1:].repeat(pad, axis=0)
            action_arr[write_ptr:write_ptr + rln] = padded_act

            # lowdim (lerobot 路径是单一 'observation.state' 8D)
            for k in self.lowdim_keys:
                arr = np.stack([row[k] for row in ep_data]).astype(np.float32)
                out = np.zeros((rln,) + arr.shape[1:], dtype=np.float32)
                if pad > 0:
                    out[:pad] = arr[0:1].repeat(pad, axis=0)
                out[pad:pad + rl] = arr
                if pad > 0:
                    out[pad + rl:] = arr[-1:].repeat(pad, axis=0)
                data_g[k][write_ptr:write_ptr + rln] = out

            # rgb
            # zarr schema 是 HWC: shape=(total, h, w, c) (跟 1.1 一致)
            # lerobot parquet 也是 HWC,直接 stack 即可,不做 moveaxis
            for k in self.rgb_keys:
                arr = np.stack([np.asarray(row[k]) for row in ep_data]).astype(np.uint8)
                # arr shape 应当是 (T, H, W, C) — 直接验证
                assert arr.ndim == 4 and arr.shape[-1] in (1, 3, 4), (
                    f"lerobot rgb 字段 {k} 期望 (T, H, W, C), 实际 shape={arr.shape}"
                )
                out = np.zeros((rln,) + arr.shape[1:], dtype=np.uint8)
                if pad > 0:
                    out[:pad] = arr[0:1].repeat(pad, axis=0)
                out[pad:pad + rl] = arr
                if pad > 0:
                    out[pad + rl:] = arr[-1:].repeat(pad, axis=0)
                data_g[k][write_ptr:write_ptr + rln] = out

            # pos_info 5 维(跟 1.1 同公式)
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

            ep_meta_list.append([i, rl, rln, write_ptr + 1, write_ptr + rln, rl + pad])
            episode_ends.append(write_ptr + rln)
            write_ptr += rln
            prev_end = ep_end

        meta_g.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
        meta_g.create_dataset("episode_map", data=np.asarray(ep_meta_list, dtype=np.int64))

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
            # v1.2 lerobot 兼容:nn.Module 不允许 key 含 '.', 用 sanitized name 存
            #   'observation.state' -> 'state'  (state 已经是拼接后的 8D)
            sanitized_k = k.replace(".", "_") if "." in k else k
            # 选用 normalizer:pos/qpos 用 range,其他(quat/state 拼接后)用 identity
            #   state 8D 实际上 (pos3 + quat_xyzw + gripper1) 范围不一, identity 简化
            if k.endswith("pos") or k.endswith("qpos"):
                normalizer[sanitized_k] = get_range_normalizer_from_stat(stat)
            else:
                normalizer[sanitized_k] = get_identity_normalizer_from_stat(stat)
            if sanitized_k != k:
                logger.info("[padp_for_libero] normalizer key '%s' (含 '.') 改用 '%s'",
                            k, sanitized_k)

        for k in self.rgb_keys:
            # rgb keys 同样可能含 '.', sanitized
            sanitized_k = k.replace(".", "_") if "." in k else k
            normalizer[sanitized_k] = get_image_range_normalizer()
            if sanitized_k != k:
                logger.info("[padp_for_libero] normalizer key '%s' (含 '.') 改用 '%s'",
                            k, sanitized_k)
        return normalizer
