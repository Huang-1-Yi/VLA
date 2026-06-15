# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""normalizer_utils —— normalizer 工具函数(从 PADP normalize_util.py 抽出来)。"""
import numpy as np

from A_common.types.normalizer import SingleFieldLinearNormalizer


def get_range_normalizer_from_stat(stat: dict):
    """range normalizer:output [-1, 1] / input [min, max]。"""
    return SingleFieldLinearNormalizer.create_from_stat(stat, output_min=-1.0, output_max=1.0)


def get_image_range_normalizer():
    """图像专用:input [0, 255] / output [0, 1]"""
    return SingleFieldLinearNormalizer(
        scale=np.array([1.0 / 255]),
        offset=np.array([0.0]),
        input_stats_dict={"min": 0, "max": 255},
    )


def get_identity_normalizer_from_stat(stat: dict):
    """identity normalizer:不做任何归一化(用于 quat 等)。"""
    dim = stat["mean"].shape[-1]
    return SingleFieldLinearNormalizer.create_identity(dim)


def robomimic_abs_action_only_normalizer_from_stat(stat: dict):
    """PADP 7 维 abs action 专用 normalizer。

    7 维动作分段归一化(2026-06-14 修,旧版硬编码 scale=[2,2,2,1,1,1,1] 完全没用 stat):
    - pos(0:3):       min-max → [-1, 1]  (create_from_stat)
    - axis_angle(3:6): identity          (方向角度无统一范围,保持原值)
    - gripper(6:7):   min-max → [-1, 1]  (create_from_stat)
    """
    dim = stat["mean"].shape[-1]
    assert dim == 7, f"Only support 7-dim robomimic abs action, got {dim}"

    # pos(0:3) 和 gripper(6:7) 用 create_from_stat(已修 /2 bug,真正落 [-1, 1])
    pos_n = SingleFieldLinearNormalizer.create_from_stat(
        {"min": stat["min"][:3], "max": stat["max"][:3]}, output_min=-1, output_max=1)
    gripper_n = SingleFieldLinearNormalizer.create_from_stat(
        {"min": stat["min"][6:7], "max": stat["max"][6:7]}, output_min=-1, output_max=1)
    # axis_angle(3:6) identity
    axis_angle_n = SingleFieldLinearNormalizer.create_identity(3)

    # 拼接 7 维(转 numpy 避开 torch buffer)
    scale = np.concatenate([
        pos_n.scale.numpy(), axis_angle_n.scale.numpy(), gripper_n.scale.numpy()
    ]).astype(np.float32)
    offset = np.concatenate([
        pos_n.offset.numpy(), axis_angle_n.offset.numpy(), gripper_n.offset.numpy()
    ]).astype(np.float32)
    return SingleFieldLinearNormalizer(scale=scale, offset=offset, input_stats_dict=stat)


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
