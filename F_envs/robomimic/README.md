# F_envs/robomimic

自包含的 robomimic 0.3.0 仿真环境 + VLA policy server TCP 客户端。

## 模块结构

```
F_envs/robomimic/
├── __init__.py            # 模块入口,触发 register_robomimic_factory
├── env_meta.py            # 任务元信息(TASK_MAX_STEPS, get_env_meta)
├── robomimic_env.py       # RobomimicEnv 主体(env wrapper, 继承 BaseRobomimicEnv)
├── make_env.py            # 工厂函数 + 注册到 C_sim
├── wrappers.py            # 可选 gym 包装
├── tcp_rollout_client.py  # PadpRolloutClient + CLI
└── README.md              # 本文件
```

## 用法

### 1. 程序化创建 env

```python
from F_envs.robomimic import make_robomimic_env
env = make_robomimic_env(
    task_name="square",
    shape_meta={
        "obs": {
            "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3]},
            "robot0_eef_quat": {"shape": [4]},
            "robot0_gripper_qpos": {"shape": [2]},
        },
        "action": {"shape": [10]},
    },
    max_steps=400,
)
obs = env.reset(seed=10000)
for _ in range(5):
    a = env.action_space.sample()
    o, r, d, i = env.step(a)
```

### 2. 通过 C_sim 工厂(推荐)

```python
from C_sim.robomimic.interface_robomimic_env import make_robomimic_env
env = make_robomimic_env(
    task_name="square",
    shape_meta=...,
    max_steps=400,
)
```

这会走 `C_sim.robomimic.interface_robomimic_env.make_robomimic_env`,
由 `F_envs.robomimic` 在 import 时通过 `register_robomimic_factory` 注册的工厂。

### 3. TCP rollout 客户端(连 VLA server)

**Server 端** (主 AI 写的):
```bash
cd /home/hy/Desktop/PADP_v3
python VLA/E_cti/train/padp_for_test_server.py \
  --ckpt VLA/data/outputs/padp_for_test_golden/latest.ckpt --port 8765
```

**Client 端** (本子 AI 写的):
```bash
cd /home/hy/Desktop/PADP_v3
python -m F_envs.robomimic.tcp_rollout_client \
  --host 127.0.0.1 --port 8765 \
  --task_name square \
  --dataset_path VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5 \
  --n_train 2 --n_test 4
```

输出(JSON):
```json
{
  "test/mean_score": 0.0,
  "train/mean_score": 0.0,
  "test/per_seed": {"10000": 0.0, "10001": 0.0, ...},
  "train/per_seed": {"0": 0.0, "1": 0.0},
  "episodes": 6,
  "total_steps": 2400,
  "elapsed_sec": 12.3,
  "avg_inference_time_ms": 4.2
}
```

## 与 PADP_v3 同步

参考文件(`/home/hy/Desktop/PADP_v3/PADP_v3/diffusion_policy/`):
- `env/robomimic/robomimic_image_wrapper.py`  → `robomimic_env.py` 的基础
- `env_runner/robomimic_image_runner.py`     → obs/action 处理逻辑参考
- `env_runner/robomimic_image_runner_padp.py`→ 7D → 10D action 转换

**主要差异**:
- 本环境是单 env(非 vec),`tcp_rollout_client` 串行跑 ep
- action 处理:server 端 10D → `rotation_6d_to_axis_angle_batch` → 7D → env.step
- env_name 映射:robomimic 0.3.0 旧数据集存的 "Square_D0" → robosuite "NutAssemblySquare"

## 注意事项

1. **不要 import VLA 内部任何东西**(除了 `E_cti.train.padp_for_test_protocol` 的帧工具)
2. **保持本目录完全自包含**,这样后续可独立 pip install / git 化
3. **C_sim 只提供接口**,实现细节由本目录提供
4. 若 robomimic 0.3.0 报新 API 错误,优先去 `PADP_v3/diffusion_policy/env/robomimic/robomimic_image_wrapper.py` 找答案
5. **cond 环境**:`/opt/miniconda3/envs/equidiff/bin/python`
   (含 `robomimic==0.3.0 + gym==0.21.0 + mujoco==2.3.2 + h5py + msgpack + numpy`)
