# P3 修复报告: ckpt 输出路径到 VLA/data/outputs/

**修复日期**: 2026-06-16
**目标**: 训练 ckpt 输出路径从 `data/outputs/` (PADP_v3 根) 改到 `VLA/data/outputs/` (跟 E_cti 同级)

## 改动清单

### 1. `padp_for_test_train.py` line 504-506 (VLA + VLA-v1.1 都要改)

**当前** (硬编码, 信任 yaml 路径):
```python
ckpt_dir = Path(train_cfg["ckpt_dir"])
ckpt_dir.mkdir(parents=True, exist_ok=True)
```

**改后** (从 `_VLA_ROOT` 解析相对路径):
```python
from pathlib import Path
_VLA_ROOT = Path(__file__).resolve().parents[2]  # 已定义在 line 113
ckpt_dir = train_cfg["ckpt_dir"]
if not Path(ckpt_dir).is_absolute():
    ckpt_dir = _VLA_ROOT / ckpt_dir  # CWD=PADP_v3 根时, 解析到 VLA/
ckpt_dir = Path(ckpt_dir)
ckpt_dir.mkdir(parents=True, exist_ok=True)
```

### 2. 4 个 yaml `ckpt_dir` 改用相对路径 (CWD=VLA 时正确)

| 文件 | 改前 | 改后 |
|---|---|---|
| `padp_for_test_golden.yaml` (1.1_10D) | `ckpt_dir: ../data/outputs/padp_for_test_golden` | `ckpt_dir: data/outputs/padp_for_test_golden` |
| `padp_for_test_7d_golden.yaml` (1.1_7D) | `ckpt_dir: ../data/outputs/padp_for_test_7d_golden` | `ckpt_dir: data/outputs/padp_for_test_7d_golden` |
| `lerobot_libero10d.yaml` (1.2_10D) | `ckpt_dir: ../data/outputs/v1.2_lerobot_libero10d` | `ckpt_dir: data/outputs/v1.2_lerobot_libero10d` |
| `lerobot_libero7d.yaml` (1.2_7D) | `ckpt_dir: ../data/outputs/v1.2_lerobot_libero7d` | `ckpt_dir: data/outputs/v1.2_lerobot_libero7d` |
| `lerobot_libero.yaml` (旧名字) | `ckpt_dir: ../data/outputs/v1.2_lerobot_libero7d` | `ckpt_dir: data/outputs/v1.2_lerobot_libero7d` |

**解析规则**: CWD = VLA/ 时, `data/outputs/...` 解析到 `VLA/data/outputs/...` (跟 E_cti 同级, 在 VLA/ 子目录下)

### 3. 镜像同步到 VLA-v1.1 (1.1 干净版)

```bash
mkdir -p /home/hy/Desktop/PADP_v3/VLA-v1.1/E_cti/configs
cp /home/hy/Desktop/PADP_v3/VLA/E_cti/configs/padp_for_test_golden.yaml \
   /home/hy/Desktop/PADP_v3/VLA-v1.1/E_cti/configs/
cp /home/hy/Desktop/PADP_v3/VLA/E_cti/configs/padp_for_test_7d_golden.yaml \
   /home/hy/Desktop/PADP_v3/VLA-v1.1/E_cti/configs/
```

## 1 步测试 (验证新路径)

```bash
cd /home/hy/Desktop/PADP_v3/VLA
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python -u \
  E_cti/train/padp_for_test_train.py \
  --config E_cti/configs/padp_for_test_golden.yaml \
  --max_epochs 1 --no_rollout --resume False
```

**期望**: `VLA/data/outputs/padp_for_test_golden/` 目录被创建, 含 `latest.ckpt` + `normalizer.pt`

## 4 个 yaml 最终 ckpt_dir 路径 (CWD=VLA)

| yaml | ckpt_dir | 解析后绝对路径 |
|---|---|---|
| padp_for_test_golden.yaml | `data/outputs/padp_for_test_golden` | `/home/hy/Desktop/PADP_v3/VLA/data/outputs/padp_for_test_golden` |
| padp_for_test_7d_golden.yaml | `data/outputs/padp_for_test_7d_golden` | `/home/hy/Desktop/PADP_v3/VLA/data/outputs/padp_for_test_7d_golden` |
| lerobot_libero10d.yaml | `data/outputs/v1.2_lerobot_libero10d` | `/home/hy/Desktop/PADP_v3/VLA/data/outputs/v1.2_lerobot_libero10d` |
| lerobot_libero7d.yaml | `data/outputs/v1.2_lerobot_libero7d` | `/home/hy/Desktop/PADP_v3/VLA/data/outputs/v1.2_lerobot_libero7d` |

## 跟原来 `data/outputs/` 的区别

| 路径方案 | 绝对路径 (CWD=VLA) |
|---|---|
| **旧** `../data/outputs/<line>` (相对 CWD=VLA 退回 1 级) | `/home/hy/Desktop/PADP_v3/data/outputs/<line>` (PADP_v3 根下) |
| **新** `data/outputs/<line>` (相对 CWD=VLA) | `/home/hy/Desktop/PADP_v3/VLA/data/outputs/<line>` (VLA 子目录下) |

**影响**:
- ✅ ckpt 现在跟 VLA 代码在同根, 删 VLA/ 时可以一起删, 不用额外同步 PADP_v3 根
- ✅ 本地/服务器 mount 路径变化时, 仍能 work (相对路径)
- ⚠️ 旧的 `/home/hy/Desktop/PADP_v3/data/outputs/<line>/` 残留数据需要手动 `rm -rf` 才能清掉
- ⚠️ 如果用户在服务器跑 (PADP_v3 挂载点不同), 需要重新 mkdir + 训练

## 镜像同步状态

- ✅ VLA: 4 个 yaml + lerobot_libero.yaml + train.py 改完
- ⏳ VLA-v1.1: train.py 跟 VLA 1:1 一致, yaml 镜像 cp (指令见 §3)
