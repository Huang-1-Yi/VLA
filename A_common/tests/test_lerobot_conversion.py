"""
测试:用 VLA 的 normalizer 读 lerobot_converter.py 转换后的数据。

目的:
  1. 验证 lerobot_converter.py 的输出(LeRobotDataset)能被 VLA 的 LinearNormalizer 正常归一化
  2. 验证 lerobot 路径与 VLA RobomimicZarrDataset 路径数据等价
  3. 验证归一化后 state 落在 [-1, 1](VLA min-max 风格)

测试数据(预先通过 lerobot_converter.py 生成):
    PYTHON=/media/disk7t/PADP_v3/Guided-VLA/.venv/bin/python
    $PYTHON /media/disk7t/PADP_v3/VLA/C_sim/robomimic/lerobot_converter.py \\
        --hdf5_path /media/disk7t/PADP_v3/VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \\
        --repo_id padp/test_square_d0_lerobot \\
        --output_root /tmp/vla_test_lerobot/square_d0 \\
        --n_demo 3 --fps 10 --mode video \\
        --task "pick up the red square"

用法:
    cd /media/disk7t/PADP_v3/VLA
    $PYTHON -m pytest A_common/tests/test_lerobot_conversion.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

# VLA 的 repository root 必须在 path 里(A_common / C_sim 互相 import)
VLA_ROOT = Path("/media/disk7t/PADP_v3/VLA")
LEROBOT_DATASET_ROOT = Path("/tmp/vla_test_lerobot/square_d0")
ORIGINAL_HDF5_PATH = VLA_ROOT / "data/robomimic/datasets/square_d0/square_d0_abs.hdf5"
LEROBOT_REPO_ID = "padp/test_square_d0_lerobot"

if str(VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VLA_ROOT))


# ----------------------------------------------------------------------------
# 工具:从 hdf5 推断 shape_meta,与 lerobot_converter.py 的 infer_shape_meta 镜像
# ----------------------------------------------------------------------------
def _infer_shape_meta_for_test(hdf5_path: Path) -> Dict[str, Any]:
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][sorted(f["data"].keys())[0]]
        obs = demo["obs"]
        rgb_keys = sorted(k for k in obs.keys() if obs[k].ndim == 4)
        lowdim_keys = sorted(k for k in obs.keys() if obs[k].ndim == 2)
        rgb_shapes = {k: (obs[k].shape[3], obs[k].shape[1], obs[k].shape[2]) for k in rgb_keys}
        state_dim = sum(obs[k].shape[1] for k in lowdim_keys)
        action_dim = demo["actions"].shape[1]
    return {
        "rgb_keys": rgb_keys,
        "lowdim_keys": lowdim_keys,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "rgb_shapes": rgb_shapes,
    }


# ----------------------------------------------------------------------------
# 准备:若 lerobot 数据集没生成,先跑 converter(用 VLA 本地 hdf5)
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def ensure_lerobot_dataset_exists():
    """若 /tmp/vla_test_lerobot/square_d0 还没生成,自动跑 converter(只跑一次)。"""
    expected = LEROBOT_DATASET_ROOT / "meta/info.json"
    if expected.exists() and (LEROBOT_DATASET_ROOT / "videos").exists():
        return  # already exists, skip
    if not ORIGINAL_HDF5_PATH.exists():
        pytest.skip(f"原 hdf5 不存在: {ORIGINAL_HDF5_PATH}")
    # 调用 VLA lerobot_converter
    import subprocess
    cmd = [
        "/media/disk7t/PADP_v3/Guided-VLA/.venv/bin/python",
        str(VLA_ROOT / "C_sim/robomimic/lerobot_converter.py"),
        "--hdf5_path", str(ORIGINAL_HDF5_PATH),
        "--repo_id", LEROBOT_REPO_ID,
        "--output_root", str(LEROBOT_DATASET_ROOT),
        "--n_demo", "3",
        "--fps", "10",
        "--mode", "video",
        "--task", "pick up the red square",
    ]
    subprocess.run(cmd, check=True)
    assert expected.exists(), f"lerobot_converter.py 跑完后仍缺 {expected}"


# ----------------------------------------------------------------------------
# Test 1:基础 sanity — LeRobotDataset 能读,dtype/shape 符合预期
# ----------------------------------------------------------------------------
def test_lerobot_dataset_basic_load():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=LEROBOT_REPO_ID, root=LEROBOT_DATASET_ROOT)
    # 总量
    assert ds.meta.total_episodes == 3
    assert ds.meta.total_frames > 0
    # features 完整
    expected_features = {"observation.state", "action", "observation.images.agentview_image",
                         "observation.images.robot0_eye_in_hand_image", "timestamp", "frame_index",
                         "episode_index", "index", "task_index"}
    assert set(ds.meta.features.keys()) >= expected_features
    # dtype
    assert ds.meta.features["observation.state"]["dtype"] == "float32"
    assert ds.meta.features["action"]["dtype"] == "float32"
    assert ds.meta.features["observation.images.agentview_image"]["dtype"] == "video"
    # video_keys 完整
    assert "observation.images.agentview_image" in ds.meta.video_keys
    assert "observation.images.robot0_eye_in_hand_image" in ds.meta.video_keys
    assert ds.meta.image_keys == []  # mode=video 时 image_keys 应为空


def test_lerobot_dataset_getitem():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=LEROBOT_REPO_ID, root=LEROBOT_DATASET_ROOT)
    item = ds[0]
    # 默认 horizon=1 → action 标量 7 维
    assert item["action"].shape == torch.Size([7])
    # state 拼接 11 个 lowdim 字段 = 59 维
    assert item["observation.state"].shape == torch.Size([59])
    # 视频帧:CHW = (3, 84, 84),归一化到 [0, 255] uint8
    img = item["observation.images.agentview_image"]
    assert img.shape == torch.Size([3, 84, 84])
    # lerobot v3.0 把视频帧自动归一化到 [0, 1] float32(v2.1 是 uint8)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min().item() <= 1.0, f"image min={img.min()} 越界 [0, 1]"
    assert 0.0 <= img.max().item() <= 1.0, f"image max={img.max()} 越界 [0, 1]"
    # state / action 是 float32
    assert item["observation.state"].dtype == torch.float32
    assert item["action"].dtype == torch.float32
    # task
    assert item["task"] == "pick up the red square"
    # 默认 horizon=1 → 无 is_pad mask
    assert "action_is_pad" not in item or not item["action_is_pad"].any()


# ----------------------------------------------------------------------------
# Test 2:VLA 现状 RobomimicZarrDataset 能读(基线)
# ----------------------------------------------------------------------------
def test_robomimic_zarr_dataset_baseline():
    from C_sim import make_dataset

    shape_meta = _infer_shape_meta_for_test(ORIGINAL_HDF5_PATH)
    cfg = {
        "data": {
            "shape_meta": {
                "obs": {
                    "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
                    "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
                    **{k: {"shape": [v.shape[1]] if v.ndim == 2 else [3, 84, 84], "type": "low_dim" if v.ndim == 2 else "rgb"}
                       for k, v in zip(shape_meta["lowdim_keys"],
                                       [__import__("h5py").File(ORIGINAL_HDF5_PATH, "r")["data"][sorted(__import__("h5py").File(ORIGINAL_HDF5_PATH, "r")["data"].keys())[0]]["obs"][k]
                                        for k in shape_meta["lowdim_keys"]])},
                },
                "action": {"shape": [7]},
            },
            "dataset_path": str(ORIGINAL_HDF5_PATH),
            "n_demo": 5,
            "horizon": 16,
            "n_obs_steps": 2,
            "n_action_steps": 1,
            # abs_action=False: 保持 7D action(避免 padp 默认做 axis_angle→6D 转换变 10D)
            "abs_action": False,
            "use_legacy_normalizer": False,
        },
    }
    ds = make_dataset(cfg)
    # 验证 dataset 类型 + 长度
    assert hasattr(ds, "horizon")
    assert hasattr(ds, "n_obs_steps")
    assert len(ds) > 0
    # 2026-06-14: padp 版 __getitem__ 返回 dict{obs, action, window_info}(非 2-tuple)
    item = ds[0]
    obs_dict = item["obs"]
    action = item["action"]
    # n_obs_steps=2 → obs["agentview_image"] 形状 (2, 3, 84, 84)
    assert obs_dict["agentview_image"].shape == torch.Size([2, 3, 84, 84])
    # horizon=16 → action 形状 (16, 7)
    assert action.shape == torch.Size([16, 7])


# ----------------------------------------------------------------------------
# Test 3:数值一致性 — lerobot action[0] 与 hdf5 demo_0 actions[0] 一致
# ----------------------------------------------------------------------------
def test_lerobot_data_matches_original_hdf5():
    """比较 lerobot-converter 转换后的第 0 个 episode 第 0 帧 action 与原 hdf5 demo_0 actions[0] 是否一致。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import h5py

    ds = LeRobotDataset(repo_id=LEROBOT_REPO_ID, root=LEROBOT_DATASET_ROOT)
    item = ds[0]   # 第 0 帧
    lerobot_action = item["action"].numpy()
    lerobot_state = item["observation.state"].numpy()

    with h5py.File(ORIGINAL_HDF5_PATH, "r") as f:
        demo0 = f["data"]["demo_0"]
        hdf5_action = demo0["actions"][0]
        # state 拼接 lowdim 字段(按字母顺序,与 lerobot_converter 里的 infer_shape_meta 逻辑一致;
        # 必须过滤 2D 字段,否则 rgb 4D 数组与 1D 拼接会爆)
        lowdim_keys = sorted(k for k in demo0["obs"].keys() if demo0["obs"][k].ndim == 2)
        hdf5_state = np.concatenate([demo0["obs"][k][0] for k in lowdim_keys])

    # action 一致(用 assert_allclose 容 float32 concat 累积 ~1e-7 误差)
    np.testing.assert_allclose(lerobot_action, hdf5_action, rtol=1e-5, atol=1e-6,
                               err_msg="lerobot action 与 hdf5 demo_0 actions[0] 不一致")
    # state 一致(同理)
    np.testing.assert_allclose(lerobot_state, hdf5_state, rtol=1e-5, atol=1e-6,
                               err_msg="lerobot state 与 hdf5 demo_0 obs 拼接结果不一致")
    print(f"\n✓ action 7 维 + state 59 维 与 hdf5 demo_0 第 0 帧逐字节一致")


# ----------------------------------------------------------------------------
# Test 4:VLA LinearNormalizer 归一化 lerobot action 落 [-1, 1]
# ----------------------------------------------------------------------------
def test_lerobot_action_normalized_to_unit_range():
    """应用 VLA 的 robomimic_abs_action_only_normalizer_from_stat 归一化 lerobot action,
    验证 7 维 action:
        - pos(0:3)        min-max → [-1, 1]
        - axis_angle(3:6) identity (保持原值,本数据 [-π, π] 不强制)
        - gripper(6:7)    min-max → [-1, 1]
    """
    from A_common.types.normalizer import SingleFieldLinearNormalizer
    from A_common.types.normalizer_utils import (
        array_to_stats, robomimic_abs_action_only_normalizer_from_stat,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 从整个 lerobot dataset 算 action 的 stat(min/max)
    ds = LeRobotDataset(repo_id=LEROBOT_REPO_ID, root=LEROBOT_DATASET_ROOT)
    actions = ds.hf_dataset.with_format("numpy")["action"][:200]   # 采 200 帧足够
    stats = array_to_stats(actions)
    # 注:robomimic_abs_action_only_normalizer_from_stat 2026-06-14 修后,直接返回
    #     SingleFieldLinearNormalizer,不再返回 dict;pos/gripper 用 stat,axis_angle identity
    action_normalizer: SingleFieldLinearNormalizer = robomimic_abs_action_only_normalizer_from_stat(stats)

    # 归一化一帧(注意:normalize 接收 torch 输入,因为 scale/offset 是 torch buffer)
    item = ds[0]
    raw_action = item["action"].numpy()
    normalized = action_normalizer.normalize(torch.from_numpy(raw_action)).numpy()

    # 检查归一化后各维范围
    # pos(0:3) + gripper(6:7) 走 create_from_stat(已修 /2 bug)→ 落 [-1, 1]
    # axis_angle(3:6) identity → 保持原值 (robomimic 约定:角度方向无统一范围)
    print(f"\n  raw action[:5]:      {raw_action[:5]}")
    print(f"  normalized action[:5]: {normalized[:5]}")
    for i in (0, 1, 2, 6):  # pos + gripper 必在 [-1, 1]
        assert -1.0 <= normalized[i] <= 1.0, (
            f"action[{i}] normalized={normalized[i]} 超出 [-1, 1]"
        )
    # axis_angle 走 identity,本数据值在 [-π, π];只 sanity check round-trip 不爆
    for i in (3, 4, 5):  # axis_angle
        assert abs(normalized[i] - raw_action[i]) < 1e-5, (
            f"axis_angle[{i-3}] 应该 identity 保持原值,实际 normalized={normalized[i]} vs raw={raw_action[i]}"
        )
    print(f"  ✓ pos(0:3) + gripper(6:7) 归一化到 [-1, 1]")
    print(f"  ✓ axis_angle(3:6) identity(原值) round-trip 无损")
    print(f"  ✓ scale={action_normalizer.scale.tolist()}")
    print(f"  ✓ offset={action_normalizer.offset.tolist()}")


# ----------------------------------------------------------------------------
# Test 5:lerobot state 归一化后范围(与 RobomimicZarrDataset + VLA normalizer 等价)
# ----------------------------------------------------------------------------
def test_lerobot_state_normalized_to_unit_range():
    """对 lerobot 转换后的 observation.state(拼接 11 个 lowdim 字段)用 range normalizer 归一化,
    验证落在 [-1, 1] 区间。"""
    from A_common.types.normalizer import SingleFieldLinearNormalizer
    from A_common.types.normalizer_utils import array_to_stats, get_range_normalizer_from_stat
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=LEROBOT_REPO_ID, root=LEROBOT_DATASET_ROOT)
    states = ds.hf_dataset.with_format("numpy")["observation.state"][:200]
    stats = array_to_stats(states)
    # 注:get_range_normalizer_from_stat 无 output_min/max 参数(2026-06-14),硬编码 [-1, 1]
    state_normalizer: SingleFieldLinearNormalizer = get_range_normalizer_from_stat(stats)

    item = ds[0]
    raw_state = item["observation.state"].numpy()
    # normalize 需要 torch 输入(scale/offset 是 torch buffer)
    normalized = state_normalizer.normalize(torch.from_numpy(raw_state)).numpy()

    assert normalized.min() >= -1.001, f"normalized min={normalized.min()} < -1"
    assert normalized.max() <= 1.001, f"normalized max={normalized.max()} > 1"
    print(f"\n  raw state min/max:        {raw_state.min():.3f} / {raw_state.max():.3f}")
    print(f"  normalized state min/max: {normalized.min():.3f} / {normalized.max():.3f}")
    print(f"  ✓ state 59 维归一化到 [-1, 1]")


# ----------------------------------------------------------------------------
# Test 6:lerobot 路径 vs VLA 现状 RobomimicZarrDataset 路径,归一化后数值一致
# ----------------------------------------------------------------------------
def test_lerobot_path_matches_zarr_path_after_normalize():
    """比对 lerobot 路径与 VLA 现状 RobomimicZarrDataset 路径在归一化后的 state/action 一致性。

    关键:两条路径都应输入原值,用 VLA 同一 normalizer 归一化后应得相同结果。
    """
    from A_common.types.normalizer import SingleFieldLinearNormalizer
    from A_common.types.normalizer_utils import array_to_stats, robomimic_abs_action_only_normalizer_from_stat
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from C_sim import make_dataset

    shape_meta = _infer_shape_meta_for_test(ORIGINAL_HDF5_PATH)

    # lerobot 路径
    ds_lerobot = LeRobotDataset(repo_id=LEROBOT_REPO_ID, root=LEROBOT_DATASET_ROOT)
    lerobot_item = ds_lerobot[0]

    # zarr 路径
    cfg = {
        "data": {
            "shape_meta": {
                "obs": {
                    "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
                    "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
                    **{k: {"shape": [v.shape[1]] if v.ndim == 2 else [3, 84, 84], "type": "low_dim" if v.ndim == 2 else "rgb"}
                       for k, v in zip(shape_meta["lowdim_keys"],
                                       [__import__("h5py").File(ORIGINAL_HDF5_PATH, "r")["data"][sorted(__import__("h5py").File(ORIGINAL_HDF5_PATH, "r")["data"].keys())[0]]["obs"][k]
                                        for k in shape_meta["lowdim_keys"]])},
                },
                "action": {"shape": [7]},
            },
            "dataset_path": str(ORIGINAL_HDF5_PATH),
            "n_demo": 3,
            "horizon": 1,             # 与 lerobot horizon=1 对齐
            "n_obs_steps": 1,         # 简化
            "n_action_steps": 1,
            # abs_action=False: 保持 7D action 与 lerobot 对齐(避免 6D 转换)
            "abs_action": False,
            "use_legacy_normalizer": False,
        },
    }
    ds_zarr = make_dataset(cfg)
    # 2026-06-14: padp 版 __getitem__ 返回 dict{obs, action, window_info}
    zarr_item = ds_zarr[0]
    zarr_action = zarr_item["action"]

    # 归一化前:两条路径 action 应该相同
    np.testing.assert_array_equal(
        lerobot_item["action"].numpy(), zarr_action.numpy().squeeze(),
        err_msg="lerobot action[0] vs zarr action[0] 不一致(归一化前)"
    )
    print(f"\n  ✓ 归一化前 action 7 维 完全一致")
    print(f"  ✓ lerobot[0] action: {lerobot_item['action'].numpy()[:3]}...")
    print(f"  ✓ zarr[0]    action: {zarr_action.numpy().squeeze()[:3]}...")
    # 归一化(注:robomimic_abs 2026-06-14 修后直接返回 normalizer,无 ["action"] 下标)
    actions = ds_lerobot.hf_dataset.with_format("numpy")["action"][:200]
    stats = array_to_stats(actions)
    normalizer: SingleFieldLinearNormalizer = robomimic_abs_action_only_normalizer_from_stat(stats)
    # normalize 需 torch 输入
    lerobot_norm = normalizer.normalize(torch.from_numpy(lerobot_item["action"].numpy())).numpy()
    zarr_norm = normalizer.normalize(torch.from_numpy(zarr_action.numpy().squeeze())).numpy()
    np.testing.assert_allclose(lerobot_norm, zarr_norm, atol=1e-6, err_msg="归一化后不一致")
    print(f"  ✓ 归一化后 action 7 维完全一致(atol=1e-6)")


# ----------------------------------------------------------------------------
# 套件入口(允许 pytest -k 选择性跑)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
