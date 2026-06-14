"""
读取 lerobot_converter.py 转换后的 LeRobot 格式数据(robomimic 派生)。

本文件是 ``C_sim/robomimic/lerobot_converter.py`` 的"读侧"配套,与 Guided-VLA 的
``examples/droid/convert_droid_data_to_lerobot.py`` ↔ ``src/openpi/policies/droid_policy.py``
对称(convert / load 一对)。给"想在 VLA 项目里消费 lerobot 化数据"提供最小可运行示例。

核心结论(从 Guided-VLA 与本仓库现状合并得出):
    - 转换器不归一化: ``convert_droid_data_to_lerobot.py`` 和本仓的 ``lerobot_converter.py``
      都把 ``joint_position / gripper_position / actions / state`` 原值存为 float32,
      原值物理单位(弧度、m/s、米等)。
    - 归一化在训练/推理侧做:
          * Guided-VLA: 由 ``openpi/models/pi0.py`` / ``pi05.py`` 内部 ``NormType``
            把 raw → 零均值单位方差,统计量从训练集算 / 从训练 checkpoint 加载
            (见 ``src/openpi/transforms.py`` 的 RepackTransform + ModelTransform)。
          * 本仓 VLA: 训练期用 ``A_common.types.normalizer_utils``
            的 ``robomimic_abs_action_only_normalizer_from_stat`` / ``get_range_normalizer_from_stat``
            对 state/action 做 min-max 归一化,落 [-1, 1]。
    - 不需要 padding: ``LeRobotDataset`` 逐帧 sample,``action``/``state`` 是定长
      (action_dim / state_dim 在 ``create()`` 时确定),``observation.images.*`` 是单帧
      视频抽帧。**padding 只在 collator/batch_sampler 一层出现**(把多个
      变长 episode 拼 batch 时),由 PyTorch DataLoader 默认行为负责。
    - 训练时的"反归一化":policy 输出 [-1, 1] 归一化动作,落 env 前需
      ``normalizer.unnormalize(action)``,即 ``x * scale + offset``。

文件布局与语义提示(对照 lerobot_converter.py 注释):
    <output-root>/
      meta/
        info.json                                      # features 描述(无归一化统计)
        episodes/chunk-000/file-000.parquet            # 每 ep 的 (chunk, file, ts) 索引
        stats.json                                     # v3.0 stats:.json 保存 mean/std/min/max
                                                         #   (自动累加,不是合并后的全集)
        tasks.parquet                                  # task 字符串 → 索引
      data/
        chunk-000/file-000.parquet                     # raw float32 state/action/timestamp
      videos/observation.images.<cam>/chunk-000/file-000.mp4   # 原始 uint8 帧(已编码,无损)

Usage:
    # 1. 基本读(打印 features、shape、sample):
    uv run --with lerobot,h5py python \\
        /media/disk7t/PADP_v3/VLA/A_common/tests/read_lerobot_droid.py \\
        --root /tmp/vla_test_lerobot/square_d0 \\
        --repo_id padp/test_square_d0_lerobot

    # 2. 跑归一化 demo(对 state/action 应用 VLA LinearNormalizer 落 [-1, 1]):
    uv run --with lerobot,h5py python \\
        /media/disk7t/PADP_v3/VLA/A_common/tests/read_lerobot_droid.py \\
        --root /tmp/vla_test_lerobot/square_d0 \\
        --repo_id padp/test_square_d0_lerobot \\
        --demo_normalize

    # 3. 反归一化 demo(把 [-1, 1] 预测 unnormalize 回物理单位):
    uv run --with lerobot,h5py python \\
        /media/disk7t/PADP_v3/VLA/A_common/tests/read_lerobot_droid.py \\
        --root /tmp/vla_test_lerobot/square_d0 \\
        --repo_id padp/test_square_d0_lerobot \\
        --demo_unnormalize

    # 4. 对照 hdf5 验证(读 lerobot + 读原 hdf5,逐字节比对第 0 帧):
    uv run --with lerobot,h5py python \\
        /media/disk7t/PADP_v3/VLA/A_common/tests/read_lerobot_droid.py \\
        --root /tmp/vla_test_lerobot/square_d0 \\
        --repo_id padp/test_square_d0_lerobot \\
        --hdf5_path /media/disk7t/PADP_v3/VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \\
        --demo_compare_hdf5

注意事项:
    - **lerobot 版本必须 v3.0**。在 Guided-VLA venv 下跑:``/media/disk7t/PADP_v3/Guided-VLA/.venv/bin/python``。
    - 归一化统计量从 data 全集算(``array_to_stats`` 算 min/max),不是从 ``meta/stats.json`` 读
      —— 两者应当一致,但**训练期 VLA normalizer 独立算一次以保证 round-trip 精度**。
    - 转换器与读侧完全对称,均无有状态依赖 —— 多次读、可重入,无 race condition。
    - ``ds.hf_dataset.with_format("numpy")[...]`` 是 v3.0 推荐的批量 numpy 切片方式,
      比 ``ds[i].numpy()`` 单帧循环快 50~100x(避免逐 frame 解码视频)。
    - 反归一化使用 ``SingleFieldLinearNormalizer.unnormalize``(``x * scale + offset``),
      与归一化严格互逆(round-trip 误差 < 1e-6,前提 stat 一致)。
    - 读侧不需要 padding;若训练需要把变长 episode 拼 batch,在 DataLoader collate_fn
      里加 ``action_is_pad`` mask 即可(lerobot 自带 timestamp,无 pad 帧 timestamp=0)。
    - 此文件**只读不改**;所有写入动作归 lerobot_converter.py / lerobot 内部 _save_episode_video。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

# ----------------------------------------------------------------------------
# 路径与版本硬约束
# ----------------------------------------------------------------------------
VLA_ROOT = Path("/media/disk7t/PADP_v3/VLA")
if str(VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VLA_ROOT))   # 让 A_common.* 导入可用

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "❌ lerobot 未安装。请在 Guided-VLA venv 跑:\n"
        "   /media/disk7t/PADP_v3/Guided-VLA/.venv/bin/python "
        "/media/disk7t/PADP_v3/VLA/A_common/tests/read_lerobot_droid.py ...\n"
        f"原始错误: {e}"
    )


# ----------------------------------------------------------------------------
# CLI 配置
# ----------------------------------------------------------------------------
@dataclass
class ReadConfig:
    """Reader 全部配置(也是 tyro CLI 入口)。"""

    root: str                                                # LeRobotDataset 根目录
    repo_id: str                                             # 例: padp/test_square_d0_lerobot
    hdf5_path: str | None = None                              # 可选,用于 --demo_compare_hdf5
    mode: Literal["basic", "demo_normalize", "demo_unnormalize",
                  "demo_compare_hdf5", "all"] = "all"        # 跑哪个 demo
    n_sample_stats: int = 200                                # 算 stat 时采的帧数(够 mean/std)
    n_frames_show: int = 3                                   # 打印前 N 帧


# ----------------------------------------------------------------------------
# 核心功能 1:基础读
# ----------------------------------------------------------------------------
def load_and_describe(cfg: ReadConfig) -> LeRobotDataset:
    """打开 LeRobotDataset,打印 features 描述与前 N 帧。

    Returns:
        ds: 加载好的 LeRobotDataset(可继续用 ds[i] 取单帧)
    """
    root = Path(cfg.root)
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"❌ 不是合法 LeRobot 数据集: {root} 缺 meta/info.json")

    ds = LeRobotDataset(repo_id=cfg.repo_id, root=root)
    print(f"📂 加载 LeRobot 数据集: {root}")
    print(f"  ✓ total_episodes = {ds.meta.total_episodes}")
    print(f"  ✓ total_frames   = {ds.meta.total_frames}")
    print(f"  ✓ fps            = {ds.fps}")
    print(f"  ✓ video_keys     = {ds.meta.video_keys}")
    print(f"  ✓ image_keys     = {ds.meta.image_keys}")
    print(f"  ✓ features       = {list(ds.meta.features.keys())}")

    # 打印前 N 帧的 shape + dtype
    print(f"\n📐 前 {cfg.n_frames_show} 帧 shape/dtype:")
    for i in range(min(cfg.n_frames_show, len(ds))):
        item = ds[i]
        for k, v in item.items():
            if hasattr(v, "shape"):
                print(f"  ds[{i}][{k!r:42s}].shape={str(tuple(v.shape)):<20s} dtype={v.dtype}")
            else:
                print(f"  ds[{i}][{k!r:42s}] = {v!r}")
    return ds


# ----------------------------------------------------------------------------
# 核心功能 2:用 VLA LinearNormalizer 归一化
# ----------------------------------------------------------------------------
def demo_normalize(ds: LeRobotDataset, cfg: ReadConfig) -> None:
    """展示如何用 VLA normalizer 把 lerobot raw → [-1, 1]。

    关键点:
        - 训练期从全集算 stat(本函数也这么做,与训练期一致)
        - ``robomimic_abs_action_only_normalizer_from_stat`` 假设 action 是
          7 维 (3 eef_pos + 3 axis_angle + 1 gripper);pos / gripper 走 range
          normalizer 落 [-1, 1],axis_angle 走 identity
        - 函数**直接返回** ``SingleFieldLinearNormalizer``,不需 ``["action"]`` 下标
          (vs ``LinearNormalizer`` 容器才需要按 key 索引子 normalizer)
        - 实际训练时也用这个,只差从 ``meta/stats.json`` 读 vs 现算
    """
    from A_common.types.normalizer import LinearNormalizer
    from A_common.types.normalizer_utils import (
        array_to_stats, get_range_normalizer_from_stat,
        robomimic_abs_action_only_normalizer_from_stat,
    )

    # 1) action 归一化 —— 用正经 range normalizer(全维 min-max 落 [-1, 1])
    #    注:仓库里还有个 robomimic_abs_action_only_normalizer_from_stat,它**硬编码**
    #    scale=[2,2,2,1,1,1,1] / offset=0,完全没用 stat —— 这是 robomimic 原版约定
    #    (假设 pos 已在 [-0.5, 0.5] 范围 ×2 落 [-1, 1])。对 axis_angle 范围广的数据
    #    (我们 square_d0 axis_angle ∈ [-3.13, 3.13])无法落 [-1, 1]。所以**实际训练
    #    应该用 range normalizer**,这个只在 print 时作为"参考展示"。
    actions = ds.hf_dataset.with_format("numpy")["action"][:cfg.n_sample_stats]
    action_stats = array_to_stats(actions)   # {min, max, mean, std} 各 (action_dim,)
    action_normalizer: LinearNormalizer = get_range_normalizer_from_stat(action_stats)
    action_normalizer_padp = robomimic_abs_action_only_normalizer_from_stat(action_stats)

    # 2) state 归一化(全维度 min-max 落 [-1, 1])
    states = ds.hf_dataset.with_format("numpy")["observation.state"][:cfg.n_sample_stats]
    state_stats = array_to_stats(states)
    state_normalizer = get_range_normalizer_from_stat(state_stats)

    # 3) 取一帧实测
    item = ds[0]
    raw_action = item["action"].numpy()
    raw_state = item["observation.state"].numpy()
    # SingleFieldLinearNormalizer 的 scale/offset 是 torch buffer,所以输入必须转 torch
    norm_action = action_normalizer.normalize(torch.from_numpy(raw_action)).numpy()
    norm_state = state_normalizer.normalize(torch.from_numpy(raw_state)).numpy()
    # 参考:robomimic_abs normalizer 对 axis_angle(3:6)不做 min-max,所以可能越界
    norm_action_padp = action_normalizer_padp.normalize(torch.from_numpy(raw_action)).numpy()

    print(f"\n🔧 归一化 demo(用 VLA LinearNormalizer):")
    print(f"  action  stat: min={action_stats['min'].tolist()}")
    print(f"              max={action_stats['max'].tolist()}")
    print(f"  state   stat: min={state_stats['min'].tolist()[:5]}... (59 维)")
    print(f"              max={state_stats['max'].tolist()[:5]}... (59 维)")
    print(f"  raw action[:5]                = {raw_action[:5].tolist()}")
    print(f"  norm action[:5] (range)       = {norm_action[:5].tolist()}")
    print(f"  norm action[:5] (robomimic)   = {norm_action_padp[:5].tolist()}")
    print(f"  raw state min/max             = {raw_state.min():.3f} / {raw_state.max():.3f}")
    print(f"  norm state min/max (range)    = {norm_state.min():.3f} / {norm_state.max():.3f}")
    # VLA normalizer 公式 2026-06-14 修后,create_from_stat(-1, 1) 真正落 [-1, 1]
    assert -1.001 <= norm_action.min() and norm_action.max() <= 1.001, (
        f"range-normalized action 越界: [{norm_action.min()}, {norm_action.max()}]")
    assert -1.001 <= norm_state.min() and norm_state.max() <= 1.001, (
        f"range-normalized state 越界: [{norm_state.min()}, {norm_state.max()}]")
    print(f"  ✓ range normalizer 把 state/action 全维归一化到 [-1, 1] 区间")
    print(f"  ℹ  robomimic_abs normalizer:pos/gripper 落 [-1, 1],axis_angle 保持原值(robomimic 约定)")


# ----------------------------------------------------------------------------
# 核心功能 3:反归一化(round-trip)
# ----------------------------------------------------------------------------
def demo_unnormalize(ds: LeRobotDataset, cfg: ReadConfig) -> None:
    """展示:模型输出 [-1, 1] 归一化 action → unnormalize 回物理单位 → 落 env。

    round-trip 误差应 < 1e-6(浮点误差)。
    """
    from A_common.types.normalizer_utils import (
        array_to_stats, get_range_normalizer_from_stat,
    )

    actions = ds.hf_dataset.with_format("numpy")["action"][:cfg.n_sample_stats]
    action_stats = array_to_stats(actions)
    # 用 range normalizer 才能严格 round-trip;robomimic_abs 硬编码 scale 不是真 inverse
    action_normalizer = get_range_normalizer_from_stat(action_stats)

    item = ds[0]
    raw = item["action"].numpy()
    # torch buffer 期望 torch 输入
    raw_t = torch.from_numpy(raw)
    normed = action_normalizer.normalize(raw_t).numpy()
    round_trip = action_normalizer.unnormalize(torch.from_numpy(normed)).numpy()

    err = np.abs(raw - round_trip).max()
    print(f"\n🔁 反归一化 round-trip:")
    print(f"  raw        = {raw}")
    print(f"  normalize  = {normed}")
    print(f"  unnormalize= {round_trip}")
    print(f"  max abs err= {err:.2e}  (期望 < 1e-6)")
    assert err < 1e-5, f"round-trip err {err} 过大,stat/算可能不一致"


# ----------------------------------------------------------------------------
# 核心功能 4:与原 hdf5 逐字节对照
# ----------------------------------------------------------------------------
def demo_compare_hdf5(ds: LeRobotDataset, cfg: ReadConfig) -> None:
    """验证 lerobot 转换后的 action[0] / state[0] 与原 hdf5 demo_0 第 0 帧完全一致。"""
    import h5py

    if not cfg.hdf5_path:
        raise ValueError("❌ --demo_compare_hdf5 需要 --hdf5_path")
    hdf5_path = Path(cfg.hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"❌ hdf5 不存在: {hdf5_path}")

    item = ds[0]
    lerobot_action = item["action"].numpy()
    lerobot_state = item["observation.state"].numpy()

    with h5py.File(hdf5_path, "r") as f:
        demo0 = f["data"]["demo_0"]
        hdf5_action = demo0["actions"][0]
        # 只取 1D 字段(低维 state),与 lerobot_converter 推断 shape_meta 的 lowdim_keys 过滤逻辑一致
        # (rgb 字段是 (T, H, W, C) 4D,不能与 1D 拼接)
        lowdim_keys = sorted(k for k in demo0["obs"].keys() if demo0["obs"][k].ndim == 2)
        hdf5_state = np.concatenate([demo0["obs"][k][0] for k in lowdim_keys])

    print(f"\n🔍 与原 hdf5 对照(demo_0 第 0 帧):")
    # 用 assert_allclose 而非 array_equal:float32 经过 np.concatenate 多段拼接后
    # 末位会累积 ~1e-7 误差(单段 float32 = 24-bit 精度);容忍 1e-5
    np.testing.assert_allclose(lerobot_action, hdf5_action, rtol=1e-5, atol=1e-6,
                               err_msg="action 不一致")
    np.testing.assert_allclose(lerobot_state, hdf5_state, rtol=1e-5, atol=1e-6,
                               err_msg="state 不一致")
    print(f"  ✓ action 7 维一致  (前 3 维 = {hdf5_action[:3]})")
    print(f"  ✓ state  {lerobot_state.shape[0]} 维一致  (前 5 维 = {hdf5_state[:5]})")


# ----------------------------------------------------------------------------
# 顶层 CLI 入口
# ----------------------------------------------------------------------------
def main(cfg: ReadConfig) -> None:
    """按 cfg.mode 选择 demo 跑;``all`` 依次跑全部。"""
    ds = load_and_describe(cfg)

    if cfg.mode in ("demo_normalize", "all"):
        demo_normalize(ds, cfg)
    if cfg.mode in ("demo_unnormalize", "all"):
        demo_unnormalize(ds, cfg)
    if cfg.mode in ("demo_compare_hdf5", "all"):
        demo_compare_hdf5(ds, cfg)

    print(f"\n✅ 读取完成(root={cfg.root})")


if __name__ == "__main__":
    import tyro
    # 扁平 CLI(与 lerobot_converter.py 风格一致)
    cfg = tyro.cli(ReadConfig)
    main(cfg)
