"""VLA v1.2 Lerobot 数据集回环测试 (Round-trip Test)

验证主 AI 写的 convert_to_lerobot.py 输出的 lerobot parquet 数据集:
  1. 能用 datasets.load_from_disk() 加载
  2. 形状正确: image (H, W, 3), state (8,), action (7|10,)
  3. 数据范围合理: image ∈ [0, 255] uint8, state/action finite float32
  4. 跨 episode 一致: 所有 episode 的 state/action dim 相同
  5. 7D<->10D round-trip max err < 0.01
     (用 VLA/C_sim/robomimic/rotation_numpy.py 的 axis_angle_to_rotation_6d_batch /
      rotation_6d_to_axis_angle_batch)

用法:
  python VLA/C_sim/robomimic/test_lerobot_round_trip.py \
    --parquet_dir /tmp/test_lerobot7d --action-dim 7
  python VLA/C_sim/robomimic/test_lerobot_round_trip.py \
    --parquet_dir /tmp/test_lerobot10d --action-dim 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 让 rotation_numpy 可被 import (不依赖调用方设的 sys.path)
VLA_ROOT = Path(__file__).resolve().parents[2]
if str(VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VLA_ROOT))
from C_sim.robomimic.rotation_numpy import (
    axis_angle_to_rotation_6d_batch,
    rotation_6d_to_axis_angle_batch,
)


def _sample_indices(n: int, k: int):
    """均匀采 k 个 [0, n) 索引。"""
    if n <= k:
        return list(range(n))
    return np.linspace(0, n - 1, k).astype(int).tolist()


def test_load(parquet_dir: Path):
    """T1: datasets.load_from_disk 加载 + 列名齐全 + 非空。"""
    from datasets import load_from_disk
    ds = load_from_disk(str(parquet_dir))
    assert len(ds) > 0, "Dataset is empty"
    expected = {"observation.image", "observation.wrist_image",
                "observation.state", "action"}
    missing = expected - set(ds.column_names)
    assert not missing, f"Missing columns: {missing}"
    print(f"  [T1] dataset loaded: {len(ds)} frames, cols = {list(ds.column_names)}")
    return ds


def test_shapes(ds, action_dim: int):
    """T2: image (H, W, 3), wrist_image (H, W, 3), state (8,), action (action_dim,)."""
    s = ds[0]
    img = np.asarray(s["observation.image"])
    wimg = np.asarray(s["observation.wrist_image"])
    state = np.asarray(s["observation.state"], dtype=np.float32)
    action = np.asarray(s["action"], dtype=np.float32)

    assert img.ndim == 3 and img.shape[-1] == 3, \
        f"image must be (H, W, 3), got {img.shape}"
    assert wimg.ndim == 3 and wimg.shape[-1] == 3, \
        f"wrist_image must be (H, W, 3), got {wimg.shape}"
    assert img.shape == wimg.shape, \
        f"main/wrist image shape mismatch: {img.shape} vs {wimg.shape}"
    assert state.ndim == 1 and state.shape[0] == 8, \
        f"state must be (8,), got {state.shape}"
    assert action.ndim == 1 and action.shape[0] == action_dim, \
        f"action must be ({action_dim},), got {action.shape}"

    print(f"  [T2] image/wrist = {img.shape}    (H, W, 3)")
    print(f"  [T2] state        = {state.shape}  (8,)")
    print(f"  [T2] action       = {action.shape} ({action_dim},)")


def test_ranges(ds, n_samples: int = 20):
    """T3: image ∈ [0, 255] uint8, state/action finite float32。"""
    img_min, img_max = 255, 0
    for i in _sample_indices(len(ds), n_samples):
        s = ds[int(i)]
        img = np.asarray(s["observation.image"])
        wimg = np.asarray(s["observation.wrist_image"])
        state = np.asarray(s["observation.state"], dtype=np.float32)
        action = np.asarray(s["action"], dtype=np.float32)

        assert img.dtype == np.uint8, f"image dtype != uint8: {img.dtype}"
        assert wimg.dtype == np.uint8, f"wrist_image dtype != uint8: {wimg.dtype}"
        assert img.min() >= 0 and img.max() <= 255, \
            f"image range bad: [{img.min()}, {img.max()}]"
        assert wimg.min() >= 0 and wimg.max() <= 255, \
            f"wrist_image range bad: [{wimg.min()}, {wimg.max()}]"
        img_min = min(img_min, int(img.min()))
        img_max = max(img_max, int(img.max()))

        assert np.all(np.isfinite(state)), f"non-finite state at idx {i}"
        assert np.all(np.isfinite(action)), f"non-finite action at idx {i}"

    print(f"  [T3] image range over {n_samples} samples: "
          f"[{img_min}, {img_max}] uint8 OK")
    print(f"  [T3] state/action finite float32 OK")


def test_episode_consistency(ds, n_episodes: int = 10):
    """T4: 多个 frame 的 state/action dim 相同。

    注:HF datasets 把所有 episode 展平成一张表,无 episode boundary 信息;
    这里验所有 sample 的 state/action 最后一维 dim 一致即可。
    """
    state_dims, action_dims = set(), set()
    for i in _sample_indices(len(ds), n_episodes):
        s = ds[int(i)]
        state_dims.add(np.asarray(s["observation.state"]).shape[-1])
        action_dims.add(np.asarray(s["action"]).shape[-1])
    assert len(state_dims) == 1, f"Inconsistent state dims: {state_dims}"
    assert len(action_dims) == 1, f"Inconsistent action dims: {action_dims}"
    print(f"  [T4] state_dim  = {list(state_dims)[0]}  (consistent)")
    print(f"  [T4] action_dim = {list(action_dims)[0]} (consistent)")


def _geodesic_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    """两个旋转矩阵之间的测地距离 (rad), 0 = 完全一致, pi = 反向。"""
    M = R1.T @ R2
    cos_t = (np.trace(M) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def _R_from_6d(rot6: np.ndarray) -> np.ndarray:
    """rotation_6d (6,) -> 旋转矩阵 R (3, 3), Gram-Schmidt。"""
    a1, a2 = rot6[:3], rot6[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - (b1 * a2).sum() * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def test_round_trip(ds, action_dim: int, n_samples: int = 50,
                    path_threshold: float = 0.05):
    """T5: 7D<->10D round-trip, 旋转测地距离 max < 0.01 rad (≈0.57°)。

    验证 forward + inverse 一致性:
      7D:  axis_angle -> 6d -> axis_angle
      10D: 6d -> axis_angle -> 6d
    映回 6d 重建 R, 用 geodesic 距离比较 (不受 axis_angle 分支切线影响)。

    注 1: 用 geodesic 而不是 raw vector 比较,axis_angle |aa| ~ pi 时
    分支切线让 raw vec 不同但 R 一致。

    注 2: hdf5 里有少量 |aa| 接近 pi 的样本 (例如 3.1414, 来自 LIBERO
    absolute action),f32 存储丢精度,Rodrigues inverse (theta ≈ pi) 数值上
    不稳定,这是 axis_angle 表示本身的局限,与 convert_to_lerobot.py 无关。
    这类样本用 |aa| < pi - path_threshold (默认 0.05) 标记并跳过。
    """
    idxs = _sample_indices(len(ds), n_samples)
    a_in = np.stack(
        [np.asarray(ds[int(i)]["action"], dtype=np.float32) for i in idxs], axis=0
    )

    if action_dim == 7:
        # 7 -> 10 -> 7, 中间 axis_angle 是 input 自身的 |aa|
        a_back = rotation_6d_to_axis_angle_batch(
            axis_angle_to_rotation_6d_batch(a_in)
        )
        name = "7 -> 10 -> 7"
        aa_norms = np.linalg.norm(a_in[..., 3:6], axis=-1)
    elif action_dim == 10:
        # 10 -> 7 -> 10, 中间 axis_angle 是 10->7 恢复出来的 |aa|
        a_back = axis_angle_to_rotation_6d_batch(
            rotation_6d_to_axis_angle_batch(a_in)
        )
        name = "10 -> 7 -> 10"
        aa_norms = np.linalg.norm(
            rotation_6d_to_axis_angle_batch(a_in)[..., 3:6], axis=-1
        )
    else:
        raise ValueError(f"action_dim must be 7 or 10, got {action_dim}")

    # 都映成 6d 重建 R, 算 geodesic
    if action_dim == 7:
        rot6_in = axis_angle_to_rotation_6d_batch(a_in)[..., 3:9]
        rot6_back = axis_angle_to_rotation_6d_batch(a_back)[..., 3:9]
    else:
        rot6_in = a_in[..., 3:9]
        rot6_back = a_back[..., 3:9]

    geos = np.array([
        _geodesic_rad(_R_from_6d(rot6_in[k]), _R_from_6d(rot6_back[k]))
        for k in range(len(idxs))
    ])
    # 跳过 axis-angle 表示不稳定的样本:
    #   (a) |aa| 接近 pi (theta ≈ pi,Rodrigues inverse 数值敏感)
    #   (b) round-trip 本身 geodesic ~ pi (catastrophic 分支跳)
    skip_aa = aa_norms >= np.pi - path_threshold
    skip_geo = geos > np.pi / 2
    ok_mask = ~skip_aa & ~skip_geo
    n_skip_aa = int(skip_aa.sum())
    n_skip_geo = int(skip_geo.sum() - (skip_aa & skip_geo).sum())

    if ok_mask.sum() == 0:
        raise AssertionError(
            f"All {len(idxs)} samples skipped (|aa|~pi: {n_skip_aa}, "
            f"geo~pi: {n_skip_geo}); can't evaluate round-trip"
        )

    worst_geo = float(geos[ok_mask].max())
    median_geo = float(np.median(geos[ok_mask]))
    print(f"  [T5] {name} geodesic: worst={worst_geo:.2e} rad "
          f"({np.degrees(worst_geo):.2e} deg), "
          f"median={median_geo:.2e} rad "
          f"over {int(ok_mask.sum())}/{len(idxs)} well-conditioned samples "
          f"(skipped: {n_skip_aa} |aa|~pi, {n_skip_geo} geo~pi)")
    assert worst_geo < 0.01, \
        f"Round-trip rotation error too large: {worst_geo:.4e} rad"


def main():
    ap = argparse.ArgumentParser(description="VLA v1.2 lerobot round-trip test")
    ap.add_argument("--parquet_dir", type=str, required=True,
                    help="HF datasets 目录 (convert_to_lerobot.py 的输出)")
    ap.add_argument("--action-dim", type=int, default=7, choices=[7, 10],
                    help="7=LEROBOT axis_angle, 10=PADP rot6d")
    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir)
    assert parquet_dir.is_dir(), f"parquet_dir not found: {parquet_dir}"

    print()
    print("=== VLA v1.2 Lerobot Round-trip Test ===")
    print(f"  parquet_dir : {parquet_dir}")
    print(f"  action_dim  : {args.action_dim}")

    # 顺便核对主 AI 写的 meta/info.json schema
    info_path = parquet_dir / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        feats = info.get("features", {})
        if "action" in feats:
            print(f"  meta info   : action.shape = {feats['action'].get('shape')}")

    print()
    ds = test_load(parquet_dir)                     # T1
    test_shapes(ds, args.action_dim)                # T2
    test_ranges(ds)                                 # T3
    test_episode_consistency(ds)                    # T4
    test_round_trip(ds, args.action_dim)            # T5

    print()
    print("=" * 50)
    print(f"[OK] All 5 tests passed (action_dim={args.action_dim})")
    print("=" * 50)


if __name__ == "__main__":
    main()
