"""
将 VLA robomimic hdf5 数据集转换为 LeRobot 格式的离线转换脚本。

数据源(预期 schema,见 VLA RobomimicZarrDataset 实践):
  /path/to/<dataset>.hdf5
    /data/
      demo_<i>/
        actions                  # (T, action_dim)         np.float32,  e.g. (T,7) 或 (T,14)
        obs/
          agentview_image            # (T, H, W, 3)  np.uint8
          robot0_eye_in_hand_image   # (T, H, W, 3)  np.uint8
          robot0_eef_pos             # (T, 3)         np.float32
          robot0_eef_quat            # (T, 4)         np.float32
          robot0_gripper_qpos        # (T, 2)         np.float32

输出:标准 LeRobot 数据集目录(可被 LeRobotDataset 加载)
  <output-root>/
    meta/
      info.json                                      # features 描述
      episodes/chunk-000/file-000.parquet           # episode 索引(每 parquet ~1000 episodes)
      episodes/chunk-000/file-001.parquet
      stats.safetensors                              # 归一化统计(consolidate 触发)
    data/
      chunk-000/file-000.parquet                     # 非视觉字段(state/action/timestamp/episode_index/task)
    videos/
      observation.images.<cam1>/chunk-000/file-000.mp4   # 视觉字段 1
      observation.images.<cam1>/chunk-000/file-001.mp4   # 多个连续 episode 串接到同一 mp4
      observation.images.<cam1>/chunk-001/file-000.mp4   # 达 chunks_size=1000 后开新 chunk
      observation.images.<cam2>/chunk-000/file-000.mp4   # 视觉字段 2(每个 camera 独立子目录)
      observation.images.<cam2>/chunk-000/file-001.mp4
      ...

视频文件组织详解(videos/ 三层结构,默认 --mode image 时是 PNG 路径类似):
    1. 路径模板: videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
       - video_key: camera 全名,例 observation.images.agentview
       - chunk_index: 0..N,当 file_index 累积到 chunks_size(默认 1000)后递增
       - file_index: 0..chunks_size-1,每存满一个 mp4(默认 500MB)后递增
    2. 1 个 episode × 1 个 camera = 1 段视频,**不强制独占 1 个 mp4**
       - LeRobot 在 _save_episode_video 时把 episode 串接到当前未满 500MB 的 mp4
       - 多个连续 episode 可共享同一个 mp4(从 episode_index 0 起算)
       - meta/episodes/...parquet 用 (chunk_index, file_index, from_timestamp, to_timestamp) 字段
         索引每个 episode 在共享 mp4 里的起止时间,读侧按 timestamp 抽帧
    3. 同一 episode 多个 camera **绝不合并**到同一 mp4,各自独立子目录
       - 2 个 camera = 2 套独立 mp4/PNG 序列
       - 例 5 episodes × 2 cámaras × --mode image ≈ 20 张 PNG(2 套,各 5 张)
    4. 默认 --mode image 时:实际写 PNG 到 images/{image_key}/episode-{ep:06d}/frame-{fr:06d}.png
       - 注:save_episode 完成后 images/ 默认保留在本地但 push_to_hub 跳过(ignore_patterns=["images/"])

Usage:
    # 1. 基本用法(默认从 hdf5 path 自动推断 shape_meta,mode=image):
    uv run --with lerobot,h5py,tyro,tqdm python \\
        /media/disk7t/PADP_v3/VLA/C_sim/robomimic/lerobot_converter.py \\
        --hdf5_path /media/disk7t/PADP_v3/PADP_v3/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \\
        --repo_id padp/square_d0_lerobot

    # 2. 指定 task 描述 + 输出目录(单条 language instruction):
    uv run --with lerobot,h5py,tyro,tqdm python \\
        /media/disk7t/PADP_v3/VLA/C_sim/robomimic/lerobot_converter.py \\
        --hdf5_path /path/to/<dataset>.hdf5 \\
        --output_root /data/robomimic/lerobot/square_d0 \\
        --repo_id padp/square_d0_lerobot \\
        --task "pick up the red square"

    # 3. 视频模式(更快,占用更少 inode):
    uv run --with lerobot,h5py,tyro,tqdm python \\
        /media/disk7t/PADP_v3/VLA/C_sim/robomimic/lerobot_converter.py \\
        --hdf5_path /path/to/<dataset>.hdf5 \\
        --repo_id padp/square_d0_lerobot \\
        --mode video

    # 4. 推送到 Hugging Face Hub(默认 private,需手动设 HF token):
    uv run --with lerobot,h5py,tyro,tqdm python \\
        /media/disk7t/PADP_v3/VLA/C_sim/robomimic/lerobot_converter.py \\
        --hdf5_path /path/to/<dataset>.hdf5 \\
        --repo_id your_org/<dataset> \\
        --push_to_hub

输出文件布局示例(n_demo=5, 2 cámaras, --mode image):
    <output-root>/
      meta/
        info.json
        episodes/chunk-000/file-000.parquet              # 5 行
        stats.safetensors                                 # state/action mean/std
      data/
        chunk-000/file-000.parquet                        # state/action/timestamp
      images/                                              # --mode image 时生成;--mode video 时不生成
        observation.images.agentview_image/
          episode-000000/frame-000000.png
          ...
          episode-000004/frame-<last>.png
        observation.images.robot0_eye_in_hand_image/
          episode-000000/frame-000000.png
          ...
          episode-000004/frame-<last>.png
    # 切换 --mode video 后:
      videos/                                              # 替代 images/,mp4 容器
        observation.images.agentview_image/chunk-000/file-000.mp4
        observation.images.robot0_eye_in_hand_image/chunk-000/file-000.mp4
      # (5 episode 串接到 1 个 mp4;6 个以上 episode 时自动开 file-001.mp4)

注意事项:
    - **lerobot 版本必须用 `/media/disk7t/PADP_v3/Guided-VLA/.venv/lib/python3.11/site-packages/lerobot/`(v3.0)**,不要在 VLA conda env 里另装;若用其它 venv 跑,需先把 PYTHONPATH 指向 Guided-VLA 的 venv site-packages
    - 需要 h5py;若已激活 padp-train conda env 则全部依赖已就绪
    - 图像默认以 dtype="image" 逐帧存 PNG(更简单可调试);如需更快用 --mode video
    - 输出目录若已存在会被清空,请提前备份
    - 转换时间大约:image 模式 1~2 小时/1000 episode;video 模式快 3-5x
    - 推荐先用 --n_demo 5 冒烟测试,再跑全量
    - 切勿手动编辑 videos/ 下的 mp4(读侧依赖精确 timestamp)
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------
# lerobot 导入(版本硬约束:必须用 Guided-VLA venv 里的 v3.0)
# 实际跑时需保证 sys.path 包含:
#   /media/disk7t/PADP_v3/Guided-VLA/.venv/lib/python3.11/site-packages
# ----------------------------------------------------------------------------
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "❌ lerobot 未安装。请先:\n"
        "   export PYTHONPATH=/media/disk7t/PADP_v3/Guided-VLA/.venv/lib/python3.11/site-packages:$PYTHONPATH\n"
        "或激活 Guided-VLA 的 venv 跑此脚本。\n"
        f"原始错误: {e}"
    )


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------
@dataclass
class PortConfig:
    """Converter 的全部配置(也是 CLI 入口)。"""

    hdf5_path: str                                # 必需,例: data/robomimic/.../square_d0_abs.hdf5
    repo_id: str                                  # 必需,例: padp/square_d0_lerobot
    output_root: str | None = None                # 默认 = $LEROBOT_HOME/<repo_id>
    task: str = "perform the demonstrated task"   # 单条 language instruction
    n_demo: int = 200                             # 转换的 demo 数量
    fps: int = 10                                 # 帧率,robomimic 单臂典型 10
    mode: Literal["image", "video"] = "image"     # 视觉存储模式
    push_to_hub: bool = False                     # 推送到 HF Hub
    use_videos: bool = True                       # 与 mode 配套
    tolerance_s: float = 1e-4                    # lerobot 时间戳容差
    image_writer_threads: int = 4                 # PNG 写盘的线程数
    image_writer_processes: int = 2               # PNG 写盘的进程数
    video_backend: str | None = None               # None = lerobot 自动选


@dataclass
class ShapeMeta:
    """从 hdf5 第一个 demo 自动推断的 shape 元信息。"""

    rgb_keys: list[str] = field(default_factory=list)          # 例: ["agentview_image", "robot0_eye_in_hand_image"]
    lowdim_keys: list[str] = field(default_factory=list)      # 例: ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    state_dim: int = 0                                          # sum(lowdim key shape[1])
    action_dim: int = 0                                        # demo["actions"].shape[1]
    rgb_shapes: dict[str, tuple] = field(default_factory=dict) # rgb_key -> (C, H, W) CHW


# ----------------------------------------------------------------------------
# 核心功能函数
# ----------------------------------------------------------------------------
def infer_shape_meta(hdf5_path: str, n_demo_for_inspect: int = 1) -> tuple[ShapeMeta, int]:
    """从 hdf5 第一个(或前 n 个)demo 自动推断 shape_meta。

    Returns:
        shape_meta: 推断的形状元信息
        total_demos: 整个 hdf5 里 demo 数量(用于 n_demo 上限检查)
    """
    with h5py.File(hdf5_path, "r") as f:
        demos = f["data"]
        demo_keys = sorted(demos.keys())
        total_demos = len(demo_keys)

        # 用第一个 demo 推断(robomimic 同 task 多个 demo 的 obs key 列表一致)
        demo = demos[demo_keys[0]]
        obs = demo["obs"]
        rgb_keys = sorted(k for k in obs.keys() if obs[k].ndim == 4)
        lowdim_keys = sorted(k for k in obs.keys() if obs[k].ndim == 2)

        # rgb shape: obs[k] shape = (T, H, W, C),转 CHW 给 B_model
        rgb_shapes = {k: (obs[k].shape[3], obs[k].shape[1], obs[k].shape[2]) for k in rgb_keys}
        # state_dim = sum(lowdim shape[1])
        state_dim = sum(obs[k].shape[1] for k in lowdim_keys)
        # action_dim: 取第一个 demo 的 actions
        action_dim = demo["actions"].shape[1]

    shape_meta = ShapeMeta(
        rgb_keys=rgb_keys,
        lowdim_keys=lowdim_keys,
        state_dim=state_dim,
        action_dim=action_dim,
        rgb_shapes=rgb_shapes,
    )
    return shape_meta, total_demos


def build_features(shape_meta: ShapeMeta, mode: str) -> dict:
    """构造 LeRobot features 字典(镜像 VLA B_model 输入契约 + aloha 命名风格)。

    关键约定(v3.0 lerobot):
        - mode="image"  → dtype="image"  → use_videos=False
        - mode="video"  → dtype="video"  → use_videos=True(默认)
        v3.0 内部依据 dtype 把 key 分到 video_keys 或 image_keys,只对 video_keys
        触发 _save_episode_video → MP4 编码。所以 dtype 与 use_videos 必须配套。
    """
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (shape_meta.state_dim,),
            "names": [shape_meta.lowdim_keys],   # 具名 joints,便于 normalizer 按 dim 解释
        },
        "action": {
            "dtype": "float32",
            "shape": (shape_meta.action_dim,),
            "names": ["action"],
        },
    }
    for rgb_key in shape_meta.rgb_keys:
        C, H, W = shape_meta.rgb_shapes[rgb_key]
        features[f"observation.images.{rgb_key}"] = {
            "dtype": mode,                            # "image" 或 "video"
            "shape": (C, H, W),                      # CHW(aloha 风格,匹配 B_model)
            "names": ["channels", "height", "width"],
        }
    return features


def derive_use_videos_from_mode(mode: str) -> bool:
    """mode → use_videos 映射:image=False, video=True。"""
    if mode == "image":
        return False
    elif mode == "video":
        return True
    else:
        raise ValueError(f"Unknown mode: {mode!r}; must be 'image' or 'video'")


def create_dataset(cfg: PortConfig, shape_meta: ShapeMeta) -> LeRobotDataset:
    """创建(覆盖)LeRobot 数据集目录 + LeRobotDataset 实例。

    清理旧 output(避免残留干扰)+ 调 LeRobotDataset.create()。
    """
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

    output_root = Path(cfg.output_root) if cfg.output_root else Path(HF_LEROBOT_HOME) / cfg.repo_id
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    features = build_features(shape_meta, cfg.mode)
    # mode 自动决定 use_videos(见 derive_use_videos_from_mode 注释)
    use_videos = derive_use_videos_from_mode(cfg.mode)
    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        root=output_root,
        fps=cfg.fps,
        features=features,
        robot_type="panda",   # robomimic 默认 panda
        use_videos=use_videos,
        tolerance_s=cfg.tolerance_s,
        image_writer_threads=cfg.image_writer_threads,
        image_writer_processes=cfg.image_writer_processes,
        video_backend=cfg.video_backend,
    )
    # 小批量(< 10) 立即 flush episodes metadata(LeRobotDataset.create 不接受此 kwarg,只能 set attr)
    dataset.meta.metadata_buffer_size = 1
    return dataset


def populate_dataset(dataset: LeRobotDataset, hdf5_path: str, shape_meta: ShapeMeta,
                     task: str, n_demo: int) -> int:
    """遍历 hdf5 里的 demo,逐帧 add_frame + 每集 save_episode。

    Returns:
        n_success: 成功写入的 episode 数
    """
    n_success = 0
    with h5py.File(hdf5_path, "r") as f:
        demos = f["data"]
        demo_keys = sorted(demos.keys())[:n_demo]

        for demo_key in tqdm(demo_keys, desc="Converting demos"):
            demo = demos[demo_key]
            T = demo["actions"].shape[0]
            try:
                # 一次性读整集 state(action 也流式读,但加 frame 时 numpy 切即可,内存 OK)
                if shape_meta.lowdim_keys:
                    state = np.concatenate(
                        [demo["obs"][k][:] for k in shape_meta.lowdim_keys], axis=1
                    ).astype(np.float32)
                else:
                    state = np.zeros((T, 0), dtype=np.float32)
                action = demo["actions"][:].astype(np.float32)

                for t in range(T):
                    frame: dict = {
                        "observation.state": state[t],
                        "action": action[t],
                    }
                    for rgb_key in shape_meta.rgb_keys:
                        # 保持 hdf5 里的 (H, W, C) uint8;LeRobot 会按 features shape 自动转置
                        frame[f"observation.images.{rgb_key}"] = demo["obs"][rgb_key][t]
                    # v3.0 lerobot 风格:task 字段写在 frame 里(vs v2.1 的 save_episode(task=...))
                    frame["task"] = task
                    dataset.add_frame(frame)

                dataset.save_episode()
                n_success += 1
            except Exception as e:
                # 单集失败不中断整批
                print(f"  ⚠️  {demo_key} 转换失败: {e}")
                continue
    return n_success


# ----------------------------------------------------------------------------
# 顶层 CLI 入口
# ----------------------------------------------------------------------------
def port_robomimic(cfg: PortConfig) -> LeRobotDataset:
    """主入口:推断 shape → 建 dataset → populate → consolidate → 可选 push。"""
    hdf5_path = Path(cfg.hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"❌ hdf5 文件不存在: {hdf5_path}")

    print(f"🔍 推断 shape_meta 从 {hdf5_path.name} ...")
    shape_meta, total_demos = infer_shape_meta(cfg.hdf5_path)
    print(f"  ✓ rgb_keys: {shape_meta.rgb_keys}")
    print(f"  ✓ lowdim_keys: {shape_meta.lowdim_keys}")
    print(f"  ✓ state_dim={shape_meta.state_dim}, action_dim={shape_meta.action_dim}")
    print(f"  ✓ hdf5 总 demo 数: {total_demos}")

    n_demo = min(cfg.n_demo, total_demos)
    if n_demo < cfg.n_demo:
        print(f"  ⚠️  --n_demo={cfg.n_demo} 大于 hdf5 总数 {total_demos},实际转换 {n_demo}")

    print(f"📦 创建 LeRobot 数据集 ...")
    dataset = create_dataset(cfg, shape_meta)
    print(f"  ✓ 输出目录: {dataset.root}")

    print(f"🔄 转换 {n_demo} 个 demo ...")
    n_success = populate_dataset(dataset, cfg.hdf5_path, shape_meta, cfg.task, n_demo)
    print(f"  ✓ 成功 {n_success}/{n_demo}")

    # 触发 video 编码 + final parquet 落盘
    # (v3.0 lerobot 的 create() 已自动 start_image_writer,save_episode 内部 _wait_image_writer,
    #  但 videos 编码 + meta/episodes/*.parquet 在 stop_image_writer 时才 finalize)
    if dataset.image_writer is not None:
        print(f"⏹  停止 image writer(触发 video 编码 + final parquet 写入)...")
        dataset.stop_image_writer()

    # 注:v3.0 lerobot 的 stats 在 save_episode 内部自动 compute(每集算一次,LeRobotDatasetMetadata
    #     累加并落盘到 meta/stats.safetensors),无需显式调 consolidate()

    if cfg.push_to_hub:
        print(f"☁️  推送到 Hugging Face Hub(默认 private)...")
        dataset.push_to_hub(private=True, push_videos=True, license="apache-2.0")

    print(f"✅ 转换完成: {dataset.root}")
    return dataset


# ----------------------------------------------------------------------------
# tyro CLI 入口
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import tyro
    # 直接从 PortConfig dataclass 生成扁平 CLI(--hdf5_path / --repo_id 等,而不是 --cfg.hdf5-path)
    cfg = tyro.cli(PortConfig)
    dataset = port_robomimic(cfg)
    print(f"   ├── n_episodes = {dataset.meta.total_episodes}")
    print(f"   ├── n_frames   = {dataset.meta.total_frames}")
    print(f"   └── fps        = {dataset.fps}")
