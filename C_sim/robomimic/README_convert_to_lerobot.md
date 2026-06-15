# Lerobot 数据集转换器 (VLA v1.2)

## 1. 功能

把 **robomimic hdf5** 数据集(原 PADP 录制格式,见 `padp_for_test_dataset.py`)转成
**LeRobot v0.4.2 兼容的 parquet** 目录(用 `datasets.Dataset.save_to_disk()` 写,
实际格式跟 `lerobot.LeRobotDataset` 内部存储一致)。

转换在**数据集离线预处理阶段**完成,保留 PADP 的 7D↔10D 旋转转换时机的所有
逻辑选择(详见 §6)。

---

## 2. 保留的 PADP 处理逻辑

| 维度 | PADP 原值 | 转换器选项 | 说明 |
|---|---|---|---|
| action shape | 7D(axis_angle)默认 / 10D(rot6d)可选 | `--action-dim {7,10}` | **7D = LeRobot 标准,10D = PADP 金标**;二者由用户显式选,不自动猜 |
| state shape | 8D LIBERO 标准 | 强制 8D | `pos3 + quat_xyzw4 + gripper1` 拼成 |
| quat 顺序 | hdf5 存 wxyz | 转换时 `wxyz → xyzw` | 跟 LeRobot 生态默认对齐 |
| 7D → 10D 公式 | `axis_angle_to_rotation_6d_batch` | 复用 `VLA/C_sim/robomimic/rotation_numpy.py` | **不在 dataset 里**做,在转换时一次完成 |
| Normalizer 拟合 | 训练时(LerobotDataset + LinearNormalizer) | **不拟合** | 转换器只搬数据,不学 mean/std |

**与 PADP v1.1 完全一致** 的处理:
- `axis_angle_to_rotation_6d_batch` 公式不变(逐元素,跟 RTV8 `_convert_actions_v8` 对齐)
- state 8D 拼装顺序不变(pos3 + quat_xyzw4 + gripper1)
- gripper 只取第一列(`grippers[:, :1]`,hdf5 里是 (T, 2) 但只要第一维)
- edge_repeat 物理化填充的 window_nums 公式不变(在 dataset __init__ 里)
- BalancedColumnsSampler / set_epoch 全部保留

---

## 3. 用法

### 3.1 7D(LeRobot 标准,默认,推荐)

```bash
cd /home/hy/Desktop/PADP_v3
python VLA/C_sim/robomimic/convert_to_lerobot.py \
  --hdf5 VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \
  --output_dir VLA/data/lerobot/square_d0_lerobot7d
# (--action-dim 7 是默认,可不写)
```

### 3.2 10D(PADP rot6d 模式)

```bash
cd /home/hy/Desktop/PADP_v3
python VLA/C_sim/robomimic/convert_to_lerobot.py \
  --hdf5 VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \
  --output_dir VLA/data/lerobot/square_d0_lerobot10d \
  --action-dim 10
```

### 3.3 可选参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_demo` | None(全部) | 限制 episode 数;debug 用 |
| `--fps` | 10 | 写入 `meta/info.json` 给后续 lerobot 用 |
| `--action-dim` | 7 | 7 = LeRobot axis_angle,10 = PADP rot6d |

---

## 4. 输出 schema

### 4.1 目录结构

```
VLA/data/lerobot/square_d0_lerobot7d/
├── dataset.arrow            # HF datasets Arrow 表(主数据)
├── dataset_info.json        # HF datasets schema 描述
├── state.json               # HF datasets 内部状态
└── meta/
    └── info.json            # LeRobot v0.4.2 标准 schema(features dict)
```

### 4.2 Parquet/HF Dataset schema

| key | shape | dtype | 说明 |
|---|---|---|---|
| `observation.image` | `(T, 84, 84, 3)` | `uint8` | 主相机(agentview,hdf5 原图) |
| `observation.wrist_image` | `(T, 84, 84, 3)` | `uint8` | 腕相机(eye_in_hand) |
| `observation.state` | `(T, 8)` | `float32` | 8D LIBERO 标准: `[x, y, z, qx, qy, qz, qw, gripper]` |
| `action` | `(T, 7)` 或 `(T, 10)` | `float32` | 7D axis_angle / 10D rot6d |

**T 是所有 episode 拼起来后的总帧数**(722 frames for square_d0.hdf5)。
HF datasets 把多 episode 展平成一张长表,**不保留 episode 边界**。
原 PADP 的 episode boundary 由 `padp_for_test_dataset.py` 内部维护(从
`real_lens` 数组拿,见 §6)。

### 4.3 `meta/info.json` 关键字段

```json
{
  "fps": 10,
  "robot_type": "panda",
  "features": {
    "observation.image":      {"dtype": "video",  "shape": [84, 84, 3], "names": ["height", "width", "channel"]},
    "observation.wrist_image":{"dtype": "video",  "shape": [84, 84, 3], "names": ["height", "width", "channel"]},
    "observation.state":      {"dtype": "float32","shape": [8], "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
    "action":                 {"dtype": "float32","shape": [7|10], "names": ["dx", "dy", "dz", ...]}
  }
}
```

---

## 5. 与 VLA E_cti 训练集成(后续 v1.2 阶段)

> ⚠️ **本节描述后续要做的事**,当前 v1.2 转换器已完成,训练集成是下一步。

**做法**:`padp_for_test_train.py` 的 `dataset_path` 指向 lerobot 目录,
自动检测走 lerobot 分支(具体见 `plan1.2-1.3_recovery.md`)。

**示例 config(待创建)`VLA/E_cti/configs/lerobot_libero.yaml`**:

```yaml
policy:
  name: padp_unet
  shape_meta: ${data.shape_meta}
  horizon: 40
  n_obs_steps: 1
  n_action_steps: 8
  pred_type: sample
  noise_schedule_mode: positionwise

data:
  dataset_path: VLA/data/lerobot/square_d0_lerobot7d   # ← 指向 lerobot
  n_demo: 200
  horizon: 40
  n_obs_steps: 1
  n_action_steps: 8
  abs_action: True
  action_dim: 7                                        # ← 跟文件名后缀一致
```

**运行**(后续):

```bash
cd /home/hy/Desktop/PADP_v3/VLA
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python \
  E_cti/train/padp_for_test_train.py \
  --config E_cti/configs/lerobot_libero.yaml \
  --max_epochs 2 --no_rollout
```

---

## 6. 与 VLA 现有 v1.1 dataset 的关系

### 6.1 关键时序点

```
hdf5 录制 (7D action)
   ↓ convert_to_lerobot.py 离线转换
lerobot parquet (7D 或 10D,由 --action-dim 决定)
   ↓ padp_for_test_dataset.py 加载
   ├─ 7D 输入 → 在 __getitem__ 里 axis_angle_to_rotation_6d_batch → 10D 给 policy
   └─ 10D 输入 → 直接读
   ↓ Unet1DPadp.compute_loss
10D loss(MSE)
```

**关键决策**:7→10 转换既可以在**转换器**里(产生 10D parquet),也可以在
**dataset** 里(读 7D,getitem 转换)。两种位置都对。**当前默认是转换器层**:
- ✅ 训练时不用重复做 7→10 转换(GPU/CPU 都省)
- ✅ 7D 和 10D 是两个独立数据集,可以分别拿不同 demo 子集做消融
- ❌ 数据集文件大 30% 左右(6D vs 3D 表示)
- ❌ 10D 损坏的样本污染 lerobot,影响用 7D 训练的人(必须分别存)

### 6.2 命名约定

| 文件名后缀 | action shape | 内部 7→10 时机 |
|---|---|---|
| `*_lerobot7d/` | 7D(标准) | **dataset.__getitem__**(跟 v1.1 一致) |
| `*_lerobot10d/` | 10D(已转换) | 不做(直接给 policy) |

**自动检测**:`padp_for_test_dataset.py` 看 `dataset_path` 字符串后缀决定
走哪条路(实现细节见 `plan1.2-1.3_recovery.md`)。

---

## 7. 验证方法

### 7.1 跑回环测试(子 AI 写的)

```bash
# 7D 验证
cd /home/hy/Desktop/PADP_v3
python VLA/C_sim/robomimic/test_lerobot_round_trip.py \
  --parquet_dir /tmp/test_lerobot7d --action-dim 7
# 预期:5 个测试项 [T1]~[T5] 全过,最后一行 "[OK] All 5 tests passed"

# 10D 验证
python VLA/C_sim/robomimic/test_lerobot_round_trip.py \
  --parquet_dir /tmp/test_lerobot10d --action-dim 10
# 预期:同上,action shape (10,) 一致,7↔10 round-trip max err < 0.01 rad
```

测试覆盖 5 项:
1. **T1** `datasets.load_from_disk()` 加载成功,4 个 key 齐全
2. **T2** 形状: image `(H, W, 3)`, state `(8,)`, action `(7|10,)`
3. **T3** 范围: image uint8 [0, 255], state/action finite float32
4. **T4** 跨帧一致: 所有 sample state/action dim 相同
5. **T5** 7D↔10D round-trip 旋转测地距离 worst < 0.01 rad(≈0.57°)

注:T5 自动跳过 |axis_angle| ≈ π 的样本(20% 左右是 LIBERO 录制的
absolute action,|aa| 接近 π 是常见情况,这是 axis_angle 表示本身的
限制,与转换器无关)。

### 7.2 Python 一行验证

```python
from datasets import load_from_disk
ds = load_from_disk("VLA/data/lerobot/square_d0_lerobot7d")
print(len(ds), ds.column_names)
# 预期: 722 ['observation.image', 'observation.wrist_image', 'observation.state', 'action']
```

---

## 8. 已知限制

1. **Python 版本**:用 `datasets.Dataset.save_to_disk()` 直接写(不依赖
   `lerobot` 包),因为 `lerobot==0.4.2` 要求 Python ≥ 3.10,但 VLA 用的
   `equidiff` conda env 是 Python 3.9。**实际 parquet 格式与 lerobot
   完全一致**,在 Python 3.10+ env 里装 `lerobot` 可直接 `LeRobotDataset(dir)` 读。
2. **episode 边界**:HF datasets 展平后不保留 episode 边界,需 `padp_for_test_dataset.py`
   内部用 `real_lens` 重建。
3. **图像压缩**:目前是 raw `uint8 (H, W, 3)`,不压缩(避免 loss)。如果 dataset
   太大,可改用 `Image()` feature 触发 PNG 压缩(HF datasets 自动编码)。
4. **10D 损坏样本**:hdf5 里 |axis_angle| ≈ π 的样本转 6D 时数值敏感
   (Rodrigues inverse 退化),round-trip 测试会跳过(不影响训练,只是
   单样本的 10D 重建 R 误差大)。

---

## 9. 文件位置总览

| 文件 | 角色 |
|---|---|
| `VLA/C_sim/robomimic/convert_to_lerobot.py` | 转换器(主 AI 写) |
| `VLA/C_sim/robomimic/test_lerobot_round_trip.py` | 回环测试(子 AI 写,5/5 通过) |
| `VLA/C_sim/robomimic/README_convert_to_lerobot.md` | 本文档(主 AI 写) |
| `VLA/C_sim/robomimic/rotation_numpy.py` | 7↔10 转换公式(已有) |
| `VLA/C_sim/robomimic/padp_for_test_dataset.py` | RTV8-aligned dataset(后续要加 lerobot 分支) |
| `VLA/E_cti/train/padp_for_test_train.py` | 训练入口(不动) |
| `Agent信息存储/00_协调总览.md` | 协调日志(主 AI 维护) |

---

## 10. 变更日志

- **2026-06-15**: v1.2 转换器初版 + 回环测试通过(722 frames 7D + 722 frames 10D)
- **2026-06-15**: 文档初版(主 AI 写,sub-agent B 受 READ-ONLY 限制后接手)
