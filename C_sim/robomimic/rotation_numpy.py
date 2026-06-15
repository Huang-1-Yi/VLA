# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""C_sim.robomimic.rotation_numpy —— 纯 numpy 的 axis_angle ↔ rotation_6d 工具。

为去掉 pytorch3d 重依赖(原 PADP `RotationTransformer` 顶层 import pytorch3d),
本模块用 Rodrigues 公式自行实现,数值上与 pytorch3d 在 float32 内等价
(0.1° 角度容差;见 A_common/tests/test_dataset_compare.py)。

约定:
  - axis_angle: 3D 向量, 模 = 旋转角, 方向 = 旋转轴
  - rotation_6d: 3x3 旋转矩阵 R 的前 2 列 flatten(Zhou et al. 2019)
  - 7 维 robomimic abs action = pos(3) + axis_angle(3) + gripper(1)
  - 10 维 PADP action       = pos(3) + rotation_6d(6) + gripper(1)
"""
import numpy as np


def axis_angle_to_matrix_aa(aa: np.ndarray) -> np.ndarray:
    """axis_angle (..., 3) → rotation matrix (..., 3, 3) via Rodrigues。

    与 pytorch3d.transforms.axis_angle_to_matrix 数值一致 (float32 ULP 级)。
    θ=0 时返回单位矩阵(而非 pytorch3d 的"任意 axis × 0")。
    """
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)             # (..., 1)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = aa / safe                                                   # 单位 axis

    # 反对称矩阵 K
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=np.float64)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] =  k[..., 0]

    I = np.broadcast_to(np.eye(3, dtype=np.float64), K.shape)
    sin_t = np.sin(theta)[..., None]                                # (..., 1, 1)
    cos_t = np.cos(theta)[..., None]
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R.astype(aa.dtype, copy=False)


def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    """rotation matrix (..., 3, 3) → first 2 rows flattened (..., 6)。

    ⚠ 重要:与 pytorch3d.transforms.matrix_to_rotation_6d 保持一致(取行,不取列)。
    pytorch3d 的源码是 `R[..., :, :2].reshape(..., 6)`,但实测发现它取的是前 2 行
    (即 R[..., 0, :] 和 R[..., 1, :]),与 Zhou et al. 2019 论文描述(前 2 列)相反。
    这里采用 pytorch3d 实际行为,以保证与 PADP RTV8 数值一致。
    """
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def axis_angle_to_rotation_6d(aa: np.ndarray) -> np.ndarray:
    """axis_angle (T, 3) → rotation_6d (T, 6)。"""
    return matrix_to_rotation_6d(axis_angle_to_matrix_aa(aa))


def axis_angle_to_rotation_6d_batch(aa7: np.ndarray) -> np.ndarray:
    """PADP 7 维 abs action → 10 维。

    input  (T, 7) = pos(3) + axis_angle(3) + gripper(1)
    output (T, 10) = pos(3) + rot_6d(6)   + gripper(1)
    """
    pos, rot, grip = aa7[..., :3], aa7[..., 3:6], aa7[..., 6:7]
    rot6 = axis_angle_to_rotation_6d(rot)
    return np.concatenate([pos, rot6, grip], axis=-1).astype(np.float32)


def rotation_6d_to_axis_angle_batch(rot10: np.ndarray) -> np.ndarray:
    """PADP 10 维 abs action → 7 维(用于反查校验)。

    input  (T, 10) = pos(3) + rot_6d(6)   + gripper(1)
    output (T, 7)  = pos(3) + axis_angle(3) + gripper(1)
    """
    pos, rot6, grip = rot10[..., :3], rot10[..., 3:9], rot10[..., 9:10]

    # 6D 重建 R:与 pyt3d 对称,把 6D 视为 R 的前 2 行,然后做 Gram-Schmidt
    a1 = rot6[..., 0:3]  # 第一行
    a2 = rot6[..., 3:6]  # 第二行
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    # pyt3d 存储的是行,我们要的是 R;直接 stack (b1, b2, b3) 当作行
    R = np.stack([b1, b2, b3], axis=-2)                            # (..., 3, 3)

    # rotation matrix → axis_angle (Rodrigues inverse)
    cos_t = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    ax = R[..., 2, 1] - R[..., 1, 2]
    ay = R[..., 0, 2] - R[..., 2, 0]
    az = R[..., 1, 0] - R[..., 0, 1]
    sin_t = np.sin(theta)
    # θ=0 时 sin_t=0,令 factor=0.5 防止除零(pytorch3d 用 0.5)
    safe_sin = np.where(sin_t < 1e-8, 1.0, sin_t)
    factor = np.where(sin_t < 1e-8, 0.5, theta / (2.0 * safe_sin))
    aa = factor[..., None] * np.stack([ax, ay, az], axis=-1)
    return np.concatenate([pos, aa, grip], axis=-1).astype(np.float32)
