# ============================================================
# PADP-VLA v1.2
# 1.2 版本,新加 lerobot 数据源训练入口(为 LIBERO 训练做准备)
# ============================================================

"""E_cti.train.padp_for_libero_train —— v1.2 lerobot 数据集训练入口(独立线)。

==========================================================================
架构:thin wrapper + monkey-patch
==========================================================================

本文件**不重复实现**训练逻辑,只做三件事:
  1. 在 import 1.1 train 模块之前,monkey-patch 两个关键符号:
     a) `C_sim.robomimic.padp_for_test_dataset.RobomimicZarrDatasetPadpForTest`
        → 替换为 `C_sim.robomimic.padp_for_libero_dataset.RobomimicZarrDatasetPadpForLibero`
     b) `B_model.encoders.VM.robomimic_obs_encoder.RobomimicObsEncoder`
        → 替换为 `B_model.encoders.VM.lerobot_obs_encoder.LerobotRobomimicObsEncoder`
        (因为 policy 内部 `padp_policy.py:96` 写死了 import 名)
  2. import 1.1 `padp_for_test_train` 模块(此时 1.1 内部 `from X import Y` 已经
     拿到 libero 版本)
  3. 调 `main()`,跟 1.1 一样支持 --config / --max_epochs / --no_rollout

为什么要 monkey-patch 而不是 fork?
  - 1.1 train 681 行,fork 维护成本高
  - 两条线 99% 逻辑相同(sampler / normalizer / loss / EMA / ckpt 策略)
  - lerobot 只影响 dataset 加载 + rgb_key 兼容
  - monkey-patch 让 1.1 train "变成" lerobot train,而不用改 1.1 一行代码

跟 1.1 的区别:
  - 数据源:hdf5 → lerobot parquet
  - rgb_key:agentview_image → observation.image
  - state 字段:5 个 lowdim (pos+quat+qpos) → 1 个 8D (pos+quat_xyzw+gripper)
  - 7→10 转换:dataset 内部做(1.1) → lerobot 端做(1.2)

用法:
  cd /home/hy/Desktop/PADP_v3
  CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python -u \\
    VLA/E_cti/train/padp_for_libero_train.py \\
    --config VLA/E_cti/configs/lerobot_libero10d.yaml \\
    --max_epochs 5 --no_rollout
==========================================================================
"""
import os
import sys
from pathlib import Path

# VLA root 加 sys.path(跟 1.1 train 一致)
_VLA_ROOT = Path(__file__).resolve().parents[2]
if str(_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLA_ROOT))


# ============================================================
# 关键 Step 1:monkey-patch 在 import 1.1 train 之前完成
# ============================================================
# 1a) Patch dataset 类
import C_sim.robomimic.padp_for_test_dataset as _test_ds_mod
from C_sim.robomimic.padp_for_libero_dataset import RobomimicZarrDatasetPadpForLibero
_test_ds_mod.RobomimicZarrDatasetPadpForTest = RobomimicZarrDatasetPadpForLibero
# 注意:这里 patch _test_ds_mod 模块的属性,1.1 train 内部
# `from C_sim.robomimic.padp_for_test_dataset import RobomimicZarrDatasetPadpForTest`
# 会从 _test_ds_mod 模块查这个属性,所以拿到的是 libero 版本

# 1b) Patch encoder 类(policy 内部 `padp_policy.py:96` 写死 import RobomimicObsEncoder)
import B_model.encoders.VM.robomimic_obs_encoder as _test_enc_mod
from B_model.encoders.VM.lerobot_obs_encoder import LerobotRobomimicObsEncoder
_test_enc_mod.RobomimicObsEncoder = LerobotRobomimicObsEncoder
# 同时也 patch VM/__init__.py 的导出
import B_model.encoders.VM as _vm_mod
_vm_mod.RobomimicObsEncoder = LerobotRobomimicObsEncoder

# 还要 patch padp_policy 模块内部的引用
import Gpolicy.PADP.padp_policy as _padp_policy_mod
_padp_policy_mod.RobomimicObsEncoder = LerobotRobomimicObsEncoder


# ============================================================
# 关键 Step 2:import 1.1 train(此时 1.1 内部已经拿到 libero 版本)
# ============================================================
from E_cti.train.padp_for_test_train import main, DEFAULT_CONFIG

# 把 1.1 DEFAULT_CONFIG 跟 lerobot yaml merge(用 lerobot yaml 覆盖)
# 注意:这里不主动 merge,1.1 train main() 内部已经支持 --config 覆盖


# ============================================================
# 关键 Step 3:CLI 默认值改为 lerobot config
# ============================================================
_DEFAULT_LEROBOT_CONFIG = "VLA/E_cti/configs/lerobot_libero7d.yaml"


def main_lerobot():
    """跟 1.1 main() 一致,只是默认 --config 改为 lerobot config。"""
    # 如果用户没显式传 --config,自动补 lerobot 默认
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", _DEFAULT_LEROBOT_CONFIG])
    main()


if __name__ == "__main__":
    main_lerobot()
