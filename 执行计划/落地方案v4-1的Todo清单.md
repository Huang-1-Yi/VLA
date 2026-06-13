# 落地方案v4-1 的 Todo 清单(60 文件,最终版)

> **配套文档**
> - 上位架构:[落地方案v4.md](落地方案v4.md)
> - 阶段 1 实施(最终版):[落地方案v4-1.md](落地方案v4-1.md)
> - 文件级映射(概要):[与现有代码的映射关系.md](与现有代码的映射关系.md)
>
> **本文件定位**:**精确到每个文件的 Todo 清单**(v4-1 最终版的"实施预算")。每个文件 = 一个 checkbox + 一段实现说明。
>
> **本版与上一版差异**
> 1. 锚定 [v4-1.md 最终版](落地方案v4-1.md) 的 60 文件结构(原 65 − 9 项优化)
> 2. emoji 已清理(上一版已做),本版维持
> 3. v4-1.1.md 已归档(变动合到 v4-1),本文以 v4-1 为准
>
> **代码组织原则(7-Layer)**:详见 [落地方案v4.md §一](落地方案v4.md)。`E_cti` 无 class,`B_model` 不 import G_algo/E_cti,数据结构在 A_common。

---

## 0. 总体进度

| 层级 | 文件数(v4-1 目标) | 进度 |
|---|---|---|
| 根目录 | 4 | 0/4 |
| F_envs | 3 | 0/3 |
| A_common | 15(核心) | 0/15 |
| B_model | 12(核心) | 0/12 |
| C_sim | 5(核心,含 env_impl 拆) | 0/5 |
| D_real | 0(阶段 1 占位) | 0/0 |
| E_cti | 4(2 yaml + 2 py) | 0/4 |
| F_envs | 3 | 0/3 |
| G_algo | 5 | 0/5 |
| **核心小计** | **48** | **0/48** |
| + 各层 `__init__.py` 占位(命名空间) | ~12 | 0/12 |
| **总文件数(含 __init__.py)** | **~60** | **0/60** |

> v4-1.md 的"60 文件"包含命名空间 `__init__.py`。本清单按"核心 48 + 命名空间 12"分两段列。

## 实施顺序(按依赖深度)

1. **F_envs**(3 文件,独立)— 配环境
2. **A_common**(15 + 7 init = 22)— 装底座契约
3. **B_model**(12 + 10 init = 22)— 装网络
4. **G_algo**(5 + 4 init = 9)— 装算法
5. **C_sim**(5 + 7 init = 12)— 装仿真
6. **E_cti**(4 + 0 init = 4)— 装流水线
7. **D_real**(0 + 1 init = 1)— 占位
8. **根目录**(4 文件)— 收尾

---

## 1. 根目录文件(4)

- [ ] **pyproject.toml**
  - 目的:uv 管理的 Python 依赖声明
  - 关键:`[project]`、`[tool.uv]`、`[tool.ruff]`
  - 源:参考 [GuidedVLA/pyproject.toml](GuidedVLA/pyproject.toml)
  - 步骤:列依赖(对应 F_envs/base_train/requirements.txt)
- [ ] **uv.lock**
  - 目的:uv 锁定文件(自动生成)
  - 步骤:第一次 `uv sync` 后自动产生
- [ ] **.python-version**
  - 目的:声明 Python 版本
  - 内容:`3.11`
- [ ] **.gitignore**
  - 目的:忽略 `__pycache__/`、`.venv/`、`outputs/`、`*.zarr` 等
  - 源:参考 PADP `.gitignore`

---

## 2. F_envs/(3 文件)

> F_envs 是**唯一无代码**的字母,只放 conda/uv 描述文件。

- [ ] **F_envs/base_train/environment.yml**
  - 目的:conda 主环境描述
  - 关键 channels:`pytorch`, `nvidia`, `conda-forge`
  - 内容:`name: padp-train`, `python=3.11`, `pytorch::pytorch=2.4`, `pytorch-cuda=12.4`
  - Python 包段(`pip:` 字段):`einops`, `huggingface-hub`, `hydra-core`, `robomimic==0.3.0`, `timm`, `transformers`, `wandb`, `zarr`, `numcodecs`, `imagecodecs`, `diffusers`
  - 源:参考 [PADP/conda_environment_sim.yaml](PADP/conda_environment_sim.yaml) + 合并原 requirements.txt
  - **铁律 7**:单一文件,弃用 uv
- [ ] **~~F_envs/base_train/requirements.txt~~ —— 删除**
  - 铁律 7:不再维护 requirements.txt;Python 包全部进 environment.yml
- [ ] **F_envs/base_train/README.md**
  - 目的:本 env 的安装 / 激活 / 升级命令
  - 内容:仅一行 `conda env create -f environment.yml`

---

## 3. A_common/(15 核心 + 7 init = 22)

> A_common 是**所有其他字母的底座**。先做这一节。

### 3.1 `A_common/types/`(4 核心 + 1 init = 5)

- [ ] **A_common/types/__init__.py**
  - 目的:导出所有公开 dataclass / 函数
  - 内容:`from .action_output import ActionOutput` 等
- [ ] **A_common/types/action_output.py**
  - 目的:**契约 1** —— 统一动作输出,抹平 step vs chunk
  - 关键:`@dataclass class ActionOutput(actions: Tensor, is_chunk: bool, latency_ms: float)`
  - 源:**自创**(v4 阶段引入)
  - 实现:纯 dataclass,加 `__post_init__` 校验 `actions.dim() == 2`
- [ ] **A_common/types/observation.py**
  - 目的:统一 obs 数据类,支持"什么有什么发什么"原则
  - 关键:`@dataclass class Observation(rgb: dict, state: Tensor, prompt: str, timestamp: float)`,加 `from_dict` 工厂方法
  - 源:参考 [GuidedVLA/src/openpi/models/model.py](GuidedVLA/src/openpi/models/model.py) 的 `Observation` 思路
- [ ] **A_common/types/state.py**
  - 目的:统一本体状态(关节 + 夹爪)
  - 关键:`@dataclass class State(joint_pos: Tensor, joint_vel: Tensor, gripper: Tensor)`,属性 `vector -> Tensor`
- [ ] **A_common/types/normalizer.py**(阶段 1 简化版)
  - 目的:线性归一化基类(阶段 1 只单字段 LinearNormalizer)
  - 关键:`class LinearNormalizer` 含 `normalize(obs/action)` / `unnormalize`,字段按 `key` 分桶
  - 源:合并 [PADP/diffusion_policy/common/normalize_util.py](PADP/diffusion_policy/common/normalize_util.py) + [PADP/diffusion_policy/model/common/normalizer.py](PADP/diffusion_policy/model/common/normalizer.py)

### 3.2 `A_common/registry/`(1 核心 + 1 init = 2)

- [ ] **A_common/registry/__init__.py**
  - 目的:导出 registry 函数
- [ ] **A_common/registry/policy_registry.py**
  - 目的:**契约 3** —— Policy 注册表
  - 关键:`register_policy(name)` 装饰器、`build_policy(config)` 工厂、`list_policies()`
  - 源:**自创**(v4 阶段引入)
  - 实现:全局 dict + `threading.Lock` + 校验 `BaseVLAPolicy` 派生

### 3.3 `A_common/data/`(2 核心 + 1 init = 3)

- [ ] **A_common/data/__init__.py**
- [ ] **A_common/data/base_dataset.py**
  - 目的:抽象 Dataset 基类
  - 关键:`class BaseVLADataset(Dataset)`
  - **铁律 6 时序契约**:
    - 显式类属性 / 构造参数:`n_obs_steps: int`(历史观测窗口长度)+ `horizon: int`(未来动作块长度)
    - `__init__` 必须接收这两个参数并存为 self.n_obs_steps / self.horizon
    - 派生类约定:返回的 `action` shape 严格对齐 `[self.horizon, D_a]`
    - episode 首尾 padding 在派生类(各 sim 的 dataset)内处理,基类不兜底
  - 抽象方法 `__getitem__` 返回 `(obs_dict, action, prompt)`
  - 实现:**不** import B_model / G_algo(铁律 3)
- [ ] **A_common/data/base_collator.py**
  - 目的:把 list of `(obs, action, prompt)` 拼成 batch dict
  - 关键:`def base_collate(batch) -> dict` 处理嵌套 `obs["rgb"]["cam_high"]`
  - 实现:基于 `torch.utils.data.default_collate`

### 3.4 `A_common/ipc/`(0 核心 + 1 init = 1)

- [ ] **A_common/ipc/__init__.py**
  - 目的:阶段 1 占位(阶段 2 写 shm + ws)

### 3.5 `A_common/logger/`(2 核心 + 1 init = 3)

- [ ] **A_common/logger/__init__.py**
- [ ] **A_common/logger/wandb_logger.py**
  - 目的:Wandb 统一封装
  - 实现:`class WandbLogger`,方法 `init / log / log_video / finish`
  - 源:参考 PADP `diffusion_policy/common/json_logger.py` 风格
- [ ] **A_common/logger/json_logger.py**
  - 目的:本地 JSON 行日志
  - 关键:`class JsonLogger` 一行一条 JSON
  - 源:抄 [PADP/diffusion_policy/common/json_logger.py](PADP/diffusion_policy/common/json_logger.py) 重构

### 3.6 `A_common/ckpt/`(1 核心 + 1 init = 2)

- [ ] **A_common/ckpt/__init__.py**
- [ ] **A_common/ckpt/checkpoint.py**
  - 目的:统一 ckpt save / load
  - 关键:`save_checkpoint(path, policy, optimizer, epoch, cfg)` / `load_checkpoint(path, policy)`
  - 实现:用 safetensors(后续阶段 2 完整化)

### 3.7 `A_common/tests/`(4 核心 + 1 init = 5)

- [ ] **A_common/tests/__init__.py**
- [ ] **A_common/tests/test_action_output.py**
  - 测试:`ActionOutput` 的 chunk/step 两种 shape 正确
- [ ] **A_common/tests/test_observation.py**
  - 测试:`Observation.from_dict` 工厂方法
- [ ] **A_common/tests/test_policy_registry.py**
  - 测试:register / build / list / duplicate 报错 / unknown 报错
- [ ] **A_common/tests/test_base_dataset.py**
  - 测试:派生 Dataset 跑通 `__getitem__`

---

## 4. B_model/(12 核心 + 10 init = 22)

> B_model **只**依赖 A_common + 第三方,**严禁** import G_algo / E_cti / C_sim / D_real。

### 4.1 `B_model/encoders/VM/`(1 核心 + 2 init+interface = 3)

- [ ] **B_model/encoders/VM/__init__.py**
- [ ] **B_model/encoders/VM/interface.py**
  - 抽象:`class VisionEncoderInterface(nn.Module)`, `def encode(rgb) -> Tensor` 输入 `[B,N,3,H,W]` 输出 `[B,N·T_v,D_h]`
- [ ] **B_model/encoders/VM/resnet18.py**
  - 目的:ResNet18 baseline(阶段 1 唯一 VM 实现)
  - 关键:`class ResNet18Encoder(VisionEncoderInterface)`, torchvision resnet18 去 fc,加 Linear 投影
  - 源:参考 [PADP/diffusion_policy/model/vision/dp_multi_image_obs_encoder.py](PADP/diffusion_policy/model/vision/dp_multi_image_obs_encoder.py)

### 4.2 `B_model/encoders/AM/`(1 核心 + 2 init+interface = 3)

- [ ] **B_model/encoders/AM/__init__.py**
- [ ] **B_model/encoders/AM/interface.py**
  - 抽象:`class ActionEncoderInterface(nn.Module)`, `def embed(action_seq) -> Tensor` 输入 `[B,H,D_a]` 输出 `[B,H,D_h]`
- [ ] **B_model/encoders/AM/conv1d_embed.py**
  - 目的:Conv1d 动作嵌入
  - 关键:`Conv1d(d_a, d_h, kernel_size=3, padding=1)`
  - 源:抽 [PADP/diffusion_policy/model/diffusion/conv1d_components.py](PADP/diffusion_policy/model/diffusion/conv1d_components.py) 通用组件

### 4.3 `B_model/encoders/TM/`(1 核心 + 2 init+interface = 3)

- [ ] **B_model/encoders/TM/__init__.py**
- [ ] **B_model/encoders/TM/interface.py**
  - 抽象:`class TimestepEncoderInterface(nn.Module)`, `def embed(t) -> Tensor` 输入 `[B]` 输出 `[B,D_h]`
- [ ] **B_model/encoders/TM/sinusoidal.py**
  - 目的:正弦时间步嵌入 + MLP
  - 关键:`SinusoidalPosEmb` + 2-layer MLP
  - 源:抄 [PADP/diffusion_policy/model/diffusion/positional_embedding.py](PADP/diffusion_policy/model/diffusion/positional_embedding.py)

### 4.4 `B_model/networks/DM/`(1 核心 + 2 init+interface = 3)

- [ ] **B_model/networks/__init__.py**
- [ ] **B_model/networks/DM/__init__.py**
- [ ] **B_model/networks/DM/interface.py**
  - 抽象:`class DiffusionNetworkInterface(nn.Module)`, `forward(sample, t, context) -> Tensor`
- [ ] **B_model/networks/DM/unet1d_padp.py**
  - 目的:**核心** — PADP 的 1D Conditional UNet
  - 关键:Down/Mid/Up ResBlock + FiLM + 可选 cross-attn
  - 源:**主迁移目标** — 抄 [PADP/diffusion_policy/model/diffusion/conditional_unet1d_padp.py](PADP/diffusion_policy/model/diffusion/conditional_unet1d_padp.py)
  - 实现:内联 `conv1d_components.py` 和 `modules.py` 的 `SinusoidalPosEmb/Downsample1d/Upsample1d/Conv1dBlock` 全部 inline 到本文件
  - 改名:类 `ConditionalUnet1D` → `Unet1DPadp`
  - 改签名:`forward(sample, t, global_cond)` → `forward(sample, t, context)`,`context` 包含 time_emb + state + text_pool
  - 删:任何 `from diffusion_policy.model.common.normalizer` / `compute_loss` 引用

### 4.5 `B_model/adapters/`(0 核心 + 1 init = 1,阶段 1 占位)

- [ ] **B_model/adapters/__init__.py**
  - 阶段 1 空,阶段 3 加 `openvla.py` / `octo.py`
  - 源(待加):参考 [GuidedVLA/src/openpi/models_pytorch/pi0_pytorch.py](GuidedVLA/src/openpi/models_pytorch/pi0_pytorch.py)

### 4.6 `B_model/compose/`(3 核心 + 1 init = 4)

- [ ] **B_model/compose/__init__.py**
- [ ] **B_model/compose/base_policy.py**
  - 目的:**契约 2** —— 所有 Policy 的统一基类
  - 关键:`class BaseVLAPolicy(nn.Module)`, 抽象 `predict_action` + `encode_inputs`
  - 源:参考 [PADP/diffusion_policy/policy/base_image_policy.py](PADP/diffusion_policy/policy/base_image_policy.py) + [GuidedVLA/packages/openpi-client/src/openpi_client/base_policy.py](GuidedVLA/packages/openpi-client/src/openpi_client/base_policy.py)
  - 关键修正:**不**含 `compute_loss`(v4 关键修正:loss 在 G_algo)
- [ ] **B_model/compose/fusion.py**
  - 目的:FiLM / cross-attn / concat 三种 fusion 工具
  - 关键:`def film(global_cond, h) -> h`、`def cross_attn(q, kv) -> h`、`def concat(*xs) -> h`
  - 源:参考 [GuidedVLA/src/openpi/models_pytorch/attention/attn_paths.py](GuidedVLA/src/openpi/models_pytorch/attention/attn_paths.py)
- [ ] **B_model/compose/padp_policy.py**
  - 目的:**PADP 顶层 policy**,通过 `@register_policy("padp_unet")` 注册
  - 关键:`class PadpUnetPolicy(BaseVLAPolicy)`,组合 `ResNet18Encoder + Conv1dActionEncoder + SinusoidalTimestepEncoder + state_proj + Unet1DPadp`
  - 源:结构抄 [PADP/diffusion_policy/policy/PADP_diffusion_unet_image_policy.py](PADP/diffusion_policy/policy/PADP_diffusion_unet_image_policy.py)
  - 关键修正:`predict_action` 不实现(raise NotImplementedError,改由 G_algo 控制);`forward(a_t, t, obs)` 暴露给 G_algo.compute_loss 用
- [ ] **B_model/compose/dp_policy.py**
  - 目的:baseline DP 顶层 policy,`@register_policy("dp_unet")`
  - 关键:`class DpUnetPolicy(PadpUnetPolicy): pass` 继承复用
  - 源:同 padp_policy,仅类名 / 注册名不同

---

## 5. C_sim/(5 核心 + 7 init = 12)

> C_sim 把硬件 / 引擎的异构 obs / action 翻译成 A_common.types 的统一 schema。

### 5.1 `C_sim/robomimic/`(5 核心 + 2 init = 7)

- [ ] **C_sim/__init__.py**
  - 目的:**铁律 4 factory 入口** —— E_cti 唯一通过本文件拿 env / runner / dataset
  - 关键函数:
    - `def make_env(cfg) -> Callable`:按 `cfg["sim"]["type"]` 选 robomimic / pusht / kitchen,内部 import 对应 env_impl
    - `def make_eval_runner(cfg) -> object`:返回带 `.run(predict_fn) -> dict` 的 runner
    - `def make_dataset(cfg) -> BaseVLADataset`:读 `cfg["data"]["horizon"]` / `n_obs_steps`
  - **铁律 4 强制**:
    - E_cti 只能 `import C_sim` 然后 `C_sim.make_*`
    - **禁止** `from C_sim.robomimic.env_runner import RobomimicImageRunner`
    - 具体类名是 C_sim 的**私有实现细节**
- [ ] **C_sim/robomimic/__init__.py**
  - 占位 / 内部 init
- [ ] **C_sim/robomimic/env_impl/__init__.py**
  - 目的:`make_env(task_name, ...)` 工厂导出
- [ ] **C_sim/robomimic/env_impl/env_meta.py**
  - 目的:23 个 robomimic task 元信息
  - 源:抄 [PADP/diffusion_policy/common/robomimic_config_util.py](PADP/diffusion_policy/common/robomimic_config_util.py)
- [ ] **C_sim/robomimic/env_impl/wrappers.py**
  - 目的:gym wrapper
  - 源:抄 [PADP/diffusion_policy/env/robomimic/robomimic_image_wrapper.py](PADP/diffusion_policy/env/robomimic/robomimic_image_wrapper.py) + [robomimic_lowdim_wrapper.py](PADP/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py)
- [ ] **C_sim/robomimic/env_runner.py**
  - 目的:接收 action,step env,返回 obs
  - 关键:`class RobomimicImageRunner`,方法 `reset() -> obs`, `step(action) -> (obs, done, info)`, `run(predict_fn) -> metrics`
  - 源:迁移 + 改造 [PADP/diffusion_policy/env_runner/robomimic_image_runner_padp.py](PADP/diffusion_policy/env_runner/robomimic_image_runner_padp.py)
  - 关键改造:`run(policy)` 改成 `run(predict_fn)`,接受 callable 而非 policy 类(让 runner 不感知 B_model)
- [ ] **C_sim/robomimic/zarr_dataset.py**
  - 目的:把 PADP zarr replay buffer 转 torch batch
  - 关键:`class RobomimicZarrDataset(BaseVLADataset)`
  - **铁律 6 实现**:
    - `__init__(zarr_path, horizon, n_obs_steps, pad_strategy='edge_repeat')` 显式接收时序参数并存为 self.horizon / self.n_obs_steps
    - 内部用 `SequenceSampler` 的 `pad_before=n_obs_steps-1` + `pad_after=horizon-1` 处理首尾 padding(策略 `edge_repeat` / `zero`)
    - `__getitem__` 返回的 `action` shape 严格对齐 `[self.horizon, D_a]`
    - `_extract_obs` 取最近 n_obs_steps 帧作为 obs window
  - 源:抄 [PADP/diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py](PADP/diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py) 的 sampler 逻辑
  - 不转 LeRobot(阶段 1 直接读 zarr)
  - **注意**:`C_sim/robomimic/zarr_dataset.py` 是 C_sim **内部**模块,E_cti 不应直接 import(铁律 4);通过 `C_sim.make_dataset(cfg)` 拿
- [ ] **C_sim/robomimic/adapter.py**
  - 目的:把 sim 的 obs / action 翻译成 A_common.types 统一 schema
  - 关键:`def obs_sim_to_common(sim_obs) -> dict`, `def action_common_to_sim(action_out) -> sim_action`
  - 源:抄 [PADP/diffusion_policy/env/robomimic/robomimic_image_wrapper.py](PADP/diffusion_policy/env/robomimic/robomimic_image_wrapper.py) 的 obs / action 转换
- [ ] **C_sim/robomimic/conda.yaml**
  - 目的:阶段 1 暂不真用,留接口

### 5.2 `C_sim/pusht/`(0 核心 + 1 init = 1,占位)

- [ ] **C_sim/pusht/__init__.py**
  - 阶段 1 占位
  - 源(待加):参考 [PADP/diffusion_policy/env/pusht/](PADP/diffusion_policy/env/pusht/)

### 5.3 `C_sim/kitchen/`(0 核心 + 1 init = 1,占位)

- [ ] **C_sim/kitchen/__init__.py**
  - 阶段 1 占位
  - 源(待加):参考 [PADP/diffusion_policy/env/kitchen/](PADP/diffusion_policy/env/kitchen/)

### 5.4 `C_sim/shared/`(0 核心 + 1 init = 1,占位)

- [ ] **C_sim/shared/__init__.py**
  - 阶段 1 占位(后续放 vector / video wrapper)

---

## 6. D_real/(0 核心 + 1 init = 1,阶段 1 不启用)

- [ ] **D_real/__init__.py**
  - 阶段 1 仅占位
  - 阶段 2 填:`D_real/franka/{controller, state_reader, interpolation, data_converter, adapter, spacemouse, realsense_config/, conda.yaml}.py`
  - 源(阶段 2 准备):[PADP/diffusion_policy/real_world/](PADP/diffusion_policy/real_world/) 整目录

---

## 7. E_cti/(4 核心 + 0 init = 4)

> **铁律 1**:E_cti 内**所有文件无 `class`**,只放 `if __name__ == "__main__"` 脚本。

### 7.1 `E_cti/configs/`(2 yaml)

- [ ] **E_cti/configs/train_padp.yaml**
  - 目的:PADP 训练超参配置
  - 源:改自 [PADP/diffusion_policy/config/robomimic_padp_position_wise_v3.yaml](PADP/diffusion_policy/config/robomimic_padp_position_wise_v3.yaml)
  - 改:`policy.name: padp_unet`,加 `padp.decay: exp, alpha: 0.5`
- [ ] **E_cti/configs/train_dp.yaml**
  - 目的:DP baseline 配置
  - 改:`policy.name: dp_unet`,无 `padp` 段

### 7.2 `E_cti/train/`(2 py)

- [ ] **E_cti/train/run_train.py**
  - 目的:**训练主入口**(纯脚本)
  - 关键:`if __name__ == "__main__": main()`,内部调 `build_policy` + `padp_compute_loss` / `dp_compute_loss`
  - **不写 class**
  - 源:参考 [PADP/train_sim.py](PADP/train_sim.py) 的 main loop 模式(但把所有 class 抽到 G_algo)
- [ ] **E_cti/train/run_eval.py**
  - 目的:**仿真评估入口**(纯脚本)
  - 关键:`main()` 加载 ckpt → 调 `padp_predict` / `dp_predict` → 调 `runner.run(predict_fn)`
  - **铁律 4 核心体现**:
    - **禁止** `from C_sim.robomimic.env_runner import RobomimicImageRunner`
    - **只** `import C_sim` + `runner = C_sim.make_eval_runner(cfg)`
    - 内部细节(env_impl / runner / dataset)由 C_sim 屏蔽
  - 关键修正:用 `--config` 复用 `train_*.yaml`(不读独立 eval.yaml)
  - 源:参考 [PADP/eval_sim.py](PADP/eval_sim.py) + [PADP/eval_sim_client.py](PADP/eval_sim_client.py)

### 7.3 `E_cti/collect/`(0 核心 + 1 init = 1,占位)

- [ ] **E_cti/collect/__init__.py**
  - 阶段 1 占位,阶段 2 填:`run_sim_collect.py`, `run_real_collect.py`
  - 源(阶段 2 准备):[GuidedVLA/examples/aloha_real/convert_aloha_data_to_lerobot.py](GuidedVLA/examples/aloha_real/convert_aloha_data_to_lerobot.py) 模板

### 7.4 `E_cti/deploy/`(0 核心 + 1 init = 1,占位)

- [ ] **E_cti/deploy/__init__.py**
  - 阶段 1 占位,阶段 2 填:`start_server.py`, `start_client.py`
  - 源(阶段 2 准备):[GuidedVLA/scripts/serve_policy.py](GuidedVLA/scripts/serve_policy.py)

---

## 8. G_algo/(5 核心 + 4 init = 9)

> G_algo 包含 loss / scheduler / pipeline,**B_model 不 import G_algo**,但 G_algo 可 import B_model。

### 8.1 `G_algo/common/`(1 核心 + 1 init = 2)

- [ ] **G_algo/__init__.py**
- [ ] **G_algo/common/__init__.py**
- [ ] **G_algo/common/denoise_loop.py**
  - 目的:DP / PADP 共享的 denoise 推理模板
  - 关键:`@torch.no_grad() def denoise_loop(policy, encoded, scheduler, action_dim, horizon, normalizer=None) -> Tensor`
  - **铁律 5 关键体现**:
    - 签名必须含 `scheduler` 参数
    - 本文件**禁止** `from diffusers import DDIMScheduler` 或 `DDIMScheduler(...)` 调用
    - scheduler 由调用方(DP / PADP 的 `predict_action`)实例化后注入
  - 源:参考 [PADP/diffusion_policy/policy/schedulers.py](PADP/diffusion_policy/policy/schedulers.py) 的 DDIMScheduler 协议

### 8.2 `G_algo/DP/`(1 核心 + 1 init = 2)

- [ ] **G_algo/DP/__init__.py**
- [ ] **G_algo/DP/dp_pipeline.py**
  - 目的:DP 算法的 compute_loss + predict_action
  - 关键:`def compute_loss(policy, batch) -> Tensor`(无位置加权)、`def predict_action(policy, obs, ...) -> ActionOutput`
  - **铁律 5 体现**:`predict_action` **内部**实例化自己的 `DDIMScheduler`(`num_inference_steps=10` 等),作为参数传给 `common.denoise_loop(policy, encoded, scheduler, ...)`
  - `compute_loss` 内部实例化训练用 `DDIMScheduler` 用于 `add_noise`
  - loss = `F.mse_loss(pred_noise, noise)`(无位置加权)
  - 源:参考 [PADP/diffusion_policy/policy/robomimic/diffusion_unet_hybrid_image_policy.py](PADP/diffusion_policy/policy/robomimic/diffusion_unet_hybrid_image_policy.py) 的训练 loop 逻辑(抽到 G_algo)

### 8.3 `G_algo/PADP/`(3 核心 + 1 init = 4)

- [ ] **G_algo/PADP/__init__.py**
- [ ] **G_algo/PADP/padp_pipeline.py**
  - 目的:**PADP 算法主入口**
  - 关键:`def compute_loss(policy, batch, decay, alpha) -> Tensor`、`def predict_action(policy, obs, ...) -> ActionOutput`
  - **铁律 5 体现**:`predict_action` **内部**实例化自己的 `DDIMScheduler`(可与 DP 不同超参:`eta=0.0` / `beta_schedule='squaredcos_cap_v2'`),作为参数传给 `common.denoise_loop(policy, encoded, scheduler, ...)`
  - loss = `(pred - target)^2 * w[None,:,None]`.mean()
  - 源:参考 [PADP/diffusion_policy/workspace/robomimic/train_padp_workspace_v3.py](PADP/diffusion_policy/workspace/robomimic/train_padp_workspace_v3.py) 抽出的算法核心(去 workspace class)
- [ ] **G_algo/PADP/loss_weights.py**
  - 目的:**per-position 加权曲线**(v4 关键修正:从 v3 的 B_model 移过来)
  - 关键:`def padp_loss_weights(horizon, decay, alpha) -> Tensor[H]`
  - 源:抄 [PADP/diffusion_policy/policy/schedulers_padp.py](PADP/diffusion_policy/policy/schedulers_padp.py) 的 weight 部分
- [ ] **G_algo/PADP/metrics.py**
  - 目的:per-position MSE / NMSE(供阶段 1 对照 PADP 原版)
  - 关键:`def per_position_mse(pred, target) -> Tensor[H]`, `per_position_nmse`
  - 源:参考 [PADP/训练padp_abition.md](PADP/训练padp_abition.md) 的指标定义

---

## 9. 7-Layer 验收检查(完成所有文件后跑)

```bash
cd padp-vla

# 铁律 1: E_cti 无 class
[ -z "$(grep -rE '^class ' E_cti/)" ] && echo "OK 1" || echo "FAIL 1"

# 铁律 2: B_model 不 import G_algo / E_cti
[ -z "$(grep -rE 'import G_algo|import E_cti' B_model/)" ] && echo "OK 2" || echo "FAIL 2"

# 铁律 3: 数据结构在 A_common/types/
[ -z "$(grep -rE 'class ActionOutput|class Observation|class State' B_model/ C_sim/ D_real/ E_cti/ G_algo/)" ] && echo "OK 3" || echo "FAIL 3"

# B_model 清洁(无 loss / scheduler)
[ -z "$(grep -rE 'def compute_loss|class.*Loss|DDPMScheduler' B_model/)" ] && echo "OK BM clean" || echo "FAIL BM"

# A_common 不 import B/G/E
[ -z "$(grep -rE 'import B_model|import G_algo|import E_cti' A_common/ | grep -v TYPE_CHECKING)" ] && echo "OK A clean" || echo "FAIL A"

# F_envs 独立
[ -z "$(grep -rE 'import A_common|import B_model' F_envs/)" ] && echo "OK F clean" || echo "FAIL F"

# === 铁律 4: C_sim / D_real factory 检查 ===
[ -z "$(grep -rE 'from C_sim\.[a-z_]+(\.[a-z_]+)+ import' E_cti/)" ] && echo "OK 4 (E_cti 不 import 内部)" || echo "FAIL 4"
grep -E "^def make_env|^def make_eval_runner|^def make_dataset" C_sim/__init__.py
# 期望:3 行匹配

# === 铁律 5: denoise_loop 解耦检查 ===
[ -z "$(grep -E 'DDIMScheduler\(|import.*DDIMScheduler|from diffusers' G_algo/common/denoise_loop.py)" ] && echo "OK 5 (denoise_loop 无硬编码)" || echo "FAIL 5"
grep -E 'def denoise_loop\(.*scheduler' G_algo/common/denoise_loop.py
# 期望:1 行匹配

# === 铁律 6: 时序契约检查 ===
grep -E "self\.n_obs_steps\s*[:=]|self\.horizon\s*[:=]|^\s+n_obs_steps\s*:|^\s+horizon\s*:" A_common/data/base_dataset.py
# 期望:≥2 行匹配
grep -E "n_obs_steps\s*=\s*[0-9]+|horizon\s*=\s*[0-9]+" C_sim/robomimic/zarr_dataset.py
# 期望:有匹配

# === 铁律 7: F_envs 纯 conda 检查 ===
[ ! -f F_envs/base_train/requirements.txt ] && echo "OK 7 (无 requirements.txt)" || echo "FAIL 7"
grep -A 20 "^dependencies:" F_envs/base_train/environment.yml | grep -E "^\s*-\s*pip:" && echo "OK 7 (pip 段)" || echo "FAIL 7"

# C/D 数据流对称(阶段 1 D_real 占位 OK)
for f in env_impl env_runner data_converter adapter; do
    [ -d C_sim/robomimic/$f ] || [ -f C_sim/robomimic/$f.py ] && echo "  OK C_sim/robomimic/$f" || echo "  FAIL C_sim/robomimic/$f"
done

# pytest
pytest A_common/tests/ -v
```

预期:**全部 `OK`**,4 个 pytest 全过。

---

## 10. 优化点(已合到 v4-1,本节只列摘要)

> 详细推导见 v4-1.md 修订记录。本节为反向追溯,方便日后查。

| # | 优化 | 行动 | v4-1 文件 |
|---|---|---|---|
| 1 | 阶段 1 删 `LM/` 整目录 | 阶段 3 再加 | 删除 |
| 2 | 阶段 1 删 `dinov2_small.py` | 阶段 2 加速 ablation 再加 | 删除 |
| 3 | 阶段 1 删 `transformer.py` | 阶段 3 变体再加 | 删除 |
| 4 | `env_impl/` 拆 3 文件 | `__init__/env_meta/wrappers` | 拆 3 |
| 5 | `DP/schedulers.py` 与 `PADP/schedulers.py` 共享 | 抽 `G_algo/common/denoise_loop.py` | 抽 1 |
| 6 | 删 `E_cti/configs/eval.yaml` | `run_eval.py` 复用 `train_*.yaml` | 删除 |
| 7 | 删 `A_common/types/chunk.py` | `ActionOutput` 已含 `is_chunk` | 删除 |
| 8 | 简化 `normalizer.py` | 阶段 1 单字段 | 简化 |
| 9 | `data_converter.py` 拆 2 | `zarr_to_lerobot.py`(阶段 2)+ `zarr_dataset.py`(阶段 1) | 改名 |
| | **净变化** | | **65 → 60** |

---

## 修订记录

| 日期 | 修订人 | 内容 |
|---|---|---|
| 2026-06-13 | Claude(60 文件最终版) | 锚定 [v4-1.md 最终版](落地方案v4-1.md) 的 60 文件结构;emoji 维持清理状态;v4-1.1.md 已归档;按 7-Layer + 8 字母文件夹给出逐文件 Todo + PADP/GuidedVLA 引用 |
| 2026-06-13 | Claude(4 铁律加固) | 在每个 Todo 文件的关键条目上显式标注 4 个铁律的体现位置:铁律 4(C_sim factory)/ 铁律 5(denoise_loop 解耦)/ 铁律 6(时序契约 n_obs_steps/horizon)/ 铁律 7(纯 conda)。完整自检脚本见 [v4-1 执行清单 §10](落地方案v4-1的执行清单.md) |
| 2026-06-13 | Claude(代码修复 4 bug) | Todo 清单体现的 4 bug 修复(obs_encoder kwarg / n_action_steps / window_exp_gamma / EMA) |
| 2026-06-13 | Claude(初版 65 文件) | 按 v4-1 初版 65 文件出 Todo 清单,反推 9 项优化(→ 60 文件) |
| 2026-06-13 | Claude(v4 架构) | 7-Layer DDD 规则 + 4 契约 + 4 提示词,详见 [落地方案v4.md](落地方案v4.md) |
