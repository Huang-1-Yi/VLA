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

    - pos(3): range [-1, 1]
    - axis_angle(3): identity(角度方向)
    - gripper(1): range [-1, 1]
    """
    dim = stat["mean"].shape[-1]
    assert dim == 7, f"Only support 7-dim robomimic abs action, got {dim}"
    scale = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
