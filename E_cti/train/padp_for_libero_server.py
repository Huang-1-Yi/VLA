# ============================================================
# PADP-VLA v1.2
# 1.2 版本,新加 lerobot ckpt rollout server 入口(为 LIBERO 训练做准备)
# ============================================================

"""E_cti.train.padp_for_libero_server —— v1.2 lerobot ckpt rollout server(独立线)。

==========================================================================
架构:thin wrapper + monkey-patch (跟 train 一致)
==========================================================================

本文件**不重复实现**server 逻辑,只做三件事:
  1. 在 import 1.1 server 模块之前,monkey-patch encoder 类:
     `B_model.encoders.VM.robomimic_obs_encoder.RobomimicObsEncoder`
     → `B_model.encoders.VM.lerobot_obs_encoder.LerobotRobomimicObsEncoder`
     (因为 policy 内部 `padp_policy.py:96` 写死了 import 名)
  2. import 1.1 `padp_for_test_server` 模块
  3. 调 `main()`,跟 1.1 一样支持 --ckpt / --config / --host / --port

为什么 server 也需要 monkey-patch?
  - server 端加载 lerobot-trained ckpt
  - ckpt 对应的 cfg 来自 lerobot yaml (rgb_key='observation.image')
  - policy 构造 encoder 时,如果用 1.1 RobomimicObsEncoder,会因为含 '.' 的 key 失败
  - monkey-patch 让 policy 内部拿到 LerobotRobomimicObsEncoder

跟 1.1 server 的区别:
  - 加载的 ckpt 来自 lerobot 训练(`padp_for_libero_train.py` 产出)
  - 默认 --config 改为 lerobot yaml
  - encoder 类用 lerobot 兼容版

用法:
  CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python -u \\
    VLA/E_cti/train/padp_for_libero_server.py \\
    --ckpt VLA/data/outputs/v1.2_lerobot_libero10d/last.ckpt \\
    --port 8766
==========================================================================
"""
import os
import sys
from pathlib import Path

# VLA root 加 sys.path(跟 1.1 server 一致)
_VLA_ROOT = Path(__file__).resolve().parents[2]
if str(_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLA_ROOT))


# ============================================================
# 关键 Step 1:monkey-patch encoder 类(在 import 1.1 server 之前)
# ============================================================
import B_model.encoders.VM.robomimic_obs_encoder as _test_enc_mod
from B_model.encoders.VM.lerobot_obs_encoder import LerobotRobomimicObsEncoder
_test_enc_mod.RobomimicObsEncoder = LerobotRobomimicObsEncoder
import B_model.encoders.VM as _vm_mod
_vm_mod.RobomimicObsEncoder = LerobotRobomimicObsEncoder
import Gpolicy.PADP.padp_policy as _padp_policy_mod
_padp_policy_mod.RobomimicObsEncoder = LerobotRobomimicObsEncoder


# ============================================================
# 关键 Step 2:import 1.1 server
# ============================================================
from E_cti.train.padp_for_test_server import main as _test_main
from E_cti.train.port_utils import find_free_port


# ============================================================
# 关键 Step 3:CLI 默认值改为 lerobot config + 端口自动避让
# ============================================================
_DEFAULT_LEROBOT_CONFIG = "VLA/E_cti/configs/lerobot_libero7d.yaml"


def main_lerobot():
    """跟 1.1 main() 一致,只是默认 --config 改为 lerobot config。

    v1.2.4 路消融改动:
      - 默认 --config 改为 lerobot_libero7d.yaml (7D, 默认线)
      - 启动前自动避让端口 (8765 → 8764 → 8763 ...)
    """
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", _DEFAULT_LEROBOT_CONFIG])
    # 解析 --port 然后自动避让
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            try:
                preferred = int(sys.argv[idx + 1])
                actual = find_free_port(preferred)
                if actual != preferred:
                    print(f"[lerobot_server] port {preferred} 被占用, 自动 -1 到 {actual}")
                sys.argv[idx + 1] = str(actual)
            except (ValueError, OSError) as e:
                print(f"[lerobot_server] 端口避让失败: {e}, 沿用原值")
    else:
        # 没指定 --port, 用默认 8765 + 自动避让
        actual = find_free_port(8765)
        sys.argv.extend(["--port", str(actual)])
    _test_main()


if __name__ == "__main__":
    main_lerobot()
