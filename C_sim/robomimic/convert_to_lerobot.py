"""VLA v1.2 Lerobot 数据集转换器

读 robomimic hdf5 (原 PADP 录制格式)
写 LeRobot v0.4.2 parquet (标准 lerobot 格式,能被 lerobot.LeRobotDataset 直接读)

保留 PADP 处理逻辑:
  - 7D action (axis_angle) 默认,与 lerobot 生态兼容
  - --action-dim 10 切到 rot6d 模式 (PADP 金标, 训练时 7→10 转换)
  - 8D state (pos3 + quat_xyzw + gripper1) 标准 LIBERO 格式
  - 正常化留给 E_cti 训练时做 (LerobotDataset + LinearNormalizer),此处不拟合 normalizer

技术决策 (v1.2):
  - 用 HF datasets.Dataset.save_to_disk() 直接写 (lerobot v0.4.2 内部用同样格式)
  - 原因:lerobot 要求 Python 3.10+,但 VLA 用 equidiff 是 Python 3.9
  - 实际 parquet 格式跟 lerobot 完全兼容,可在 Python 3.10+ env 装 lerobot 直接读

用法:
  # 7D (默认,LEROBOT 标准)
  python VLA/C_sim/robomimic/convert_to_lerobot.py \\
    --hdf5 VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \\
    --output_dir VLA/data/lerobot/square_d0_lerobot7d

  # 10D (PADP 模式)
  python VLA/C_sim/robomimic/convert_to_lerobot.py \\
    --hdf5 VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \\
    --output_dir VLA/data/lerobot/square_d0_lerobot10d \\
    --action-dim 10
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
from tqdm import tqdm

# VLA 内部 import(用绝对路径,避免 sys.path 漂移)
VLA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VLA_ROOT))
from C_sim.robomimic.rotation_numpy import (
    axis_angle_to_rotation_6d_batch,  # 7D axis_angle → 10D rot6d
)


# ============================================================
# 核心转换
# ============================================================
def convert_hdf5_episode(h5_root, demo_key: str, action_dim: int = 7) -> Dict[str, np.ndarray]:
    """读一个 hdf5 episode,转成 lerobot 格式 dict。

    Args:
        h5_root:  h5py.File 根节点
        demo_key: e.g. "demo_0"
        action_dim: 7 (axis_angle) 或 10 (rot6d)

    Returns:
        dict 包含:
            observation.image:       (T, H, W, 3) uint8    主相机
            observation.wrist_image: (T, H, W, 3) uint8    腕相机
            observation.state:       (T, 8) float32         8D LIBERO 标准
            action:                  (T, action_dim) float32 7 or 10
    """
    # 1. 读 obs (主相机 + 腕相机)
    images = h5_root[f"{demo_key}/obs/agentview_image"][:]  # (T, H, W, 3) uint8
    wrist_images = h5_root[f"{demo_key}/obs/robot0_eye_in_hand_image"][:]
    states_pos = h5_root[f"{demo_key}/obs/robot0_eef_pos"][:]  # (T, 3)
    quats_wxyz = h5_root[f"{demo_key}/obs/robot0_eef_quat"][:]  # (T, 4) wxyz
    grippers = h5_root[f"{demo_key}/obs/robot0_gripper_qpos"][:]  # (T, 1)

    # 2. 读 action (原始 7D)
    actions_7d = h5_root[f"{demo_key}/actions"][:]  # (T, 7)

    # 3. 7↔10 转换 (action 转换, state 保持原样 8D)
    if action_dim == 7:
        actions = actions_7d
    elif action_dim == 10:
        # axis_angle (T, 3) → rotation_6d (T, 6) → 拼 (T, 10)
        actions = axis_angle_to_rotation_6d_batch(actions_7d)
    else:
        raise ValueError(f"action_dim must be 7 or 10, got {action_dim}")

    # 4. 拼 state (8D LIBERO 标准: pos3 + quat_xyzw + gripper1)
    # grippers shape 是 (T, 2) 不是 (T, 1),只取第一列
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    states_8d = np.concatenate(
        [states_pos, quats_xyzw, grippers[:, :1]], axis=-1
    ).astype(np.float32)

    return {
        "observation.image":       images.astype(np.uint8),
        "observation.wrist_image": wrist_images.astype(np.uint8),
        "observation.state":       states_8d,
        "action":                  actions.astype(np.float32),
    }


def save_as_lerobot_parquet(episodes: List[Dict], output_dir: Path,
                             action_dim: int, fps: int = 10):
    """用 HF datasets.Dataset.save_to_disk() 写 lerobot 兼容 parquet。

    HF datasets 写出的 Arrow table 跟 lerobot LeRobotDataset 读的一致。
    """
    from datasets import Dataset, Features, Value, Image, Sequence

    # 把所有 episode 的 data 拼成 1D/2D 数组
    # 关键:HF datasets 期望 uniform shape,变长会让它迷茫
    # 我们把所有时间步拼成一个长数组(N_total, ...)
    flat_data = {
        "observation.image":       np.concatenate(
            [ep["observation.image"] for ep in episodes], axis=0),
        "observation.wrist_image": np.concatenate(
            [ep["observation.wrist_image"] for ep in episodes], axis=0),
        "observation.state":       np.concatenate(
            [ep["observation.state"] for ep in episodes], axis=0).astype(np.float32),
        "action":                  np.concatenate(
            [ep["action"] for ep in episodes], axis=0).astype(np.float32),
    }

    # 拿 image shape(给 meta/info.json 用)
    H, W = flat_data["observation.image"].shape[1:3]

    # Features schema
    features = Features({
        "observation.image":       Image(),
        "observation.wrist_image": Image(),
        "observation.state":       Sequence(Value("float32"), length=8),
        "action":                  Sequence(Value("float32"), length=action_dim),
    })

    # 构造 HF Dataset
    ds = Dataset.from_dict(flat_data, features=features)
    ds.save_to_disk(str(output_dir))

    # 写 meta/info.json (lerobot 标准)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    import json
    info = {
        "fps": fps,
        "robot_type": "panda",
        "features": {
            "observation.image": {
                "dtype": "video",
                "shape": [H, W, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.wrist_image": {
                "dtype": "video",
                "shape": [H, W, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": (["dx", "dy", "dz", "dax", "day", "daz", "gripper"]
                          if action_dim == 7
                          else ["dx", "dy", "dz", "r00", "r01", "r02", "r10", "r11", "r12", "gripper"]),
            },
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[convert] meta/info.json: {meta_dir / 'info.json'}")

    # v1.2 新增:写 meta/episode_ends.json (供 padp_for_test_dataset.py 重建 episode 边界)
    # 格式:[end_0, end_1, ..., end_N] (累计帧数),跟原 hdf5 严格对齐
    real_lens = [int(ep["observation.image"].shape[0]) for ep in episodes]
    episode_ends = np.cumsum(real_lens).tolist()
    (meta_dir / "episode_ends.json").write_text(json.dumps({
        "real_lens": real_lens,
        "episode_ends": episode_ends,
        "n_episodes": len(real_lens),
        "total_frames": int(sum(real_lens)),
    }, indent=2))
    print(f"[convert] meta/episode_ends.json: {len(real_lens)} episodes,"
          f" total {sum(real_lens)} frames")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="把 robomimic hdf5 转成 LeRobot v0.4.2 parquet(保留 PADP 处理)")
    ap.add_argument("--hdf5", type=str, required=True,
                    help="robomimic hdf5 路径")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="lerobot 输出目录(将被创建)")
    ap.add_argument("--action-dim", type=int, default=7, choices=[7, 10],
                    help="7=LEROBOT 标准 (axis_angle), 10=PADP rot6d")
    ap.add_argument("--n_demo", type=int, default=None,
                    help="限制 episode 数(None = 全部)")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    hdf5_path = Path(args.hdf5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] hdf5 = {hdf5_path}")
    print(f"[convert] output = {output_dir}")
    print(f"[convert] action_dim = {args.action_dim} (7=LEROBOT axis_angle, 10=PADP rot6d)")

    with h5py.File(hdf5_path, "r") as f:
        demos = f["data"]
        all_keys = [k for k in demos.keys() if k.startswith("demo_")]
        n_demos = args.n_demo or len(all_keys)
        n_demos = min(n_demos, len(all_keys))
        print(f"[convert] {len(all_keys)} total episodes, processing {n_demos}")

        episodes = []
        for i in tqdm(range(n_demos), desc="[convert]"):
            ep = convert_hdf5_episode(demos, f"demo_{i}", action_dim=args.action_dim)
            episodes.append(ep)

    save_as_lerobot_parquet(episodes, output_dir, action_dim=args.action_dim, fps=args.fps)

    print(f"[convert] ✅ Done. Saved to {output_dir}")
    print(f"[convert] Verify: python -c \"from datasets import load_from_disk; ds = load_from_disk('{output_dir}'); print(len(ds), ds[0].keys())\"")


if __name__ == "__main__":
    main()
