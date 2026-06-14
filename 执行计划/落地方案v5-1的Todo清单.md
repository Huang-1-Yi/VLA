# 落地方案v5-1 的 Todo 清单 — 文件级实施预决算

> **本文件定位**:阶段 1 实施预决算 — 65 个文件,逐个勾选的任务清单。
>
> **配套文档**:
> - 上位架构:[落地方案v5.md](落地方案v5.md) — 7-Layer + 7 铁律强化版 + 5 接口契约总览
> - 5 个接口契约模板(权威):[落地方案v5-1的接口契约.md](落地方案v5-1的接口契约.md) — v7 终极白盒架构
> - 阶段 1 实施手册:[落地方案v5-1.md](落地方案v5-1.md) — 文件结构与调用流
>
> **本文硬性原则**(从 v7 契约反向推导):
> 1. 完全无视当前 `/media/disk7t/PADP_v3/VLA/` 代码 — 用户明确要求从零开始规划
> 2. v5 相对 v4 的 5 项关键差异必须被覆盖(见 §26)
> 3. 每个文件 1 个 `[ ]` checkbox + 实现说明(目的 / 关键实现点 / 源 / 注意点)
> 4. 总计 **65 个文件** + **7 铁律验收脚本** ≈ **73 个可勾选任务**

---

## 文档头部 / 修订记录 / 实施原则

### 实施原则

1. **任务全部 0/总数 起步** — 不预设任何"已完成",所有文件都是空 `[ ]`
2. **v5 vs v4 差异** 在 §26 单列;**v7 契约硬性要求** 散布到各文件注意点中
3. **每条任务必标源参考**:抄自 / 改自 / 自创,实施时可一眼定位
4. **总文件数:65** + 验收脚本 ~13 行 ≈ **总计 ~73 个 checkbox**
5. **G_algo → Gpolicy** 已统一,v7 契约要求的 `cfg.policy` / `build_policy` / `PADPAdapter` 类名均已对齐

### 修订记录

| 日期 | 修订人 | 内容 |
|---|---|---|
| 2026-06-14 | Claude(v5-1 Todo 初版) | 基于 v5 架构 + v7 契约产出 65 文件级 Todo 清单;覆盖 7-Layer + Gpolicy 子包进度表;v5 vs v4 关键差异单列 |

---

## §0. 总体进度(7-Layer + Gpolicy 子包)

| 层级 / 子包 | 文件数 | 已完成 | 进度 |
|---|---|---|---|
| §1 根目录 | 4 | 0 | 0/4 |
| §2 F_envs | 2 | 0 | 0/2 |
| §3 A_common/types | 10 | 0 | 0/10 |
| §4 A_common/registry | 2 | 0 | 0/2 |
| §5 A_common/data | 3 | 0 | 0/3 |
| §6 A_common/ipc | 1 | 0 | 0/1 |
| §7 A_common/logger | 3 | 0 | 0/3 |
| §8 A_common/ckpt | 3 | 0 | 0/3 |
| §9 A_common/tests | 9 | 0 | 0/9 |
| §10 B_model/encoders/VM | 3 | 0 | 0/3 |
| §11 B_model/encoders/AM | 3 | 0 | 0/3 |
| §12 B_model/networks/TM | 3 | 0 | 0/3 |
| §13 B_model/networks/DM | 3 | 0 | 0/3 |
| §14 B_model/adapters | 4 | 0 | 0/4 |
| §15 C_sim/robomimic | 7 | 0 | 0/7 |
| §16 C_sim/{pusht, kitchen, shared} | 3 | 0 | 0/3 |
| §17 D_real | 1 | 0 | 0/1 |
| §18 E_cti/configs | 2 | 0 | 0/2 |
| §19 E_cti/train | 2 | 0 | 0/2 |
| §20 E_cti/collect + deploy | 2 | 0 | 0/2 |
| §21 Gpolicy/common | 3 | 0 | 0/3 |
| §22 Gpolicy/DP | 3 | 0 | 0/3 |
| §23 Gpolicy/PADP | 5 | 0 | 0/5 |
| **文件合计** | **65** | **0** | **0/65** |
| §24 验收脚本(13 行) | 13 | 0 | 0/13 |
| **总任务数** | **~78** | **0** | **0/~78** |

> 实施时间线见 §25(T0-T7 估算)。

---

## §1. 根目录(4 个文件)

- [ ] **`pyproject.toml`** — uv 管 Python 依赖入口
  - **目的**:声明包元数据 + 工具配置(ruff / pytest)
  - **关键实现点**:`[project]` 段含 name="padp-vla" / version / requires-python=">=3.11";`[tool.ruff]` / `[tool.pytest.ini_options]` 配置段
  - **源**:自创(uv 官方推荐模板)
  - **注意点**:不放真正的依赖列表,所有依赖走 F_envs/environment.yml(铁律 7)

- [ ] **`uv.lock`** — uv 自动生成的依赖锁文件
  - **目的**:锁定 Python 依赖版本,确保可复现
  - **关键实现点**:`uv sync` / `uv add` 后自动生成,不需要手写
  - **源**:自创(uv 工具自动)
  - **注意点**:**首次跑** `uv sync` 后才有;CI 必 commit

- [ ] **`.python-version`** — Python 版本声明
  - **目的**:让 uv 自动切换到正确 Python
  - **关键实现点**:单行 `3.11`
  - **源**:自创(uv 官方推荐)
  - **注意点**:与 F_envs/environment.yml 的 `python=3.11` 一致

- [ ] **`.gitignore`** — Git 忽略规则
  - **目的**:屏蔽 `__pycache__` / `.venv` / `data/` / ckpt 文件等
  - **关键实现点**:`__pycache__/` `*.pyc` `.venv/` `data/` `*.ckpt` `wandb/` `.pytest_cache/`
  - **源**:自创(标准 Python gitignore)
  - **注意点**:`data/` 必忽略(避免把 zarr 巨型文件 commit)

---

## §2. F_envs/(2 个文件)

- [ ] **`F_envs/base_train/environment.yml`** — 纯 conda 依赖清单
  - **目的**:单一 conda 环境文件,涵盖系统级 + pip 段(铁律 7)
  - **关键实现点**:
    ```yaml
    name: padp-train
    channels: [pytorch, nvidia, conda-forge, defaults]
    dependencies:
      - python=3.11
      - pytorch::pytorch=2.4
      - pytorch::pytorch-cuda=12.4
      - conda-forge::ffmpeg
      - conda-forge::libgl
      - pip
      - pip: [einops, hydra-core, robomimic==0.3.0, timm, transformers, wandb, zarr, diffusers]
    ```
  - **源**:改自 PADP 仓库的 conda yaml
  - **注意点**:**严禁另开 requirements.txt**(铁律 7 强制);含 `diffusers`(DDIMScheduler 来自它);不含 lerobot / jax(阶段 1 不用)

- [ ] **`F_envs/base_train/README.md`** — 环境搭建说明
  - **目的**:一行命令搭建 + 验证
  - **关键实现点**:`conda env create -f environment.yml` + `conda activate padp-train` + 验证 import
  - **源**:自创
  - **注意点**:写明"无 requirements.txt,所有依赖在 environment.yml";注明 cuDNN / CUDA 版本

---

## §3. A_common/types/(10 个文件 — 5 个契约 + 5 个数据结构)

### 3.1 5 个契约文件(契约代码,跨阶段稳定,不可修改签名)

- [ ] **`A_common/types/vision_encoder.py`** — 契约 2:`VisionEncoderInterface`
  - **目的**:VM 抽象基类(契约 4,来自 v5-1 的接口契约 §二.2.1)
  - **关键实现点**:
    ```python
    class VisionEncoderInterface(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, obs_dict: dict) -> torch.Tensor: raise NotImplementedError()
        def output_shape(self) -> tuple: raise NotImplementedError()
    ```
  - **源**:自创(契约要求)
  - **注意点**:**必须用 `raise NotImplementedError()`,不写 abstractmethod**(参照 BaseDataset 风格);`output_shape()` 是 Policy `__init__` 时的反问入口

- [ ] **`A_common/types/action_encoder.py`** — 契约 3:`ActionEncoderInterface`
  - **目的**:AM 抽象基类(契约 4 §二.2.2)
  - **关键实现点**:`forward(action_seq: Tensor) -> Tensor` + `output_shape() -> tuple`,两方法都 `raise NotImplementedError()`
  - **源**:自创(契约要求)
  - **注意点**:`action_seq` 形状约定为 `[B, T, D_a]`;与 VM 同模板,统一 `forward + output_shape`

- [ ] **`A_common/types/timestep_encoder.py`** — 契约 4:`TimestepEncoderInterface`
  - **目的**:TM 抽象基类(契约 §二.2.3)
  - **关键实现点**:`forward(t: Tensor) -> Tensor`(输入 `[B]` 输 `[B, D_h]`)+ `output_shape() -> tuple`
  - **源**:自创(契约要求)
  - **注意点**:此契约供未来扩展语言编码器 / 触觉编码器复用,任何"输入 lerobot 标准 obs、输出维度对 Policy 未知"的组件都继承这个模板

- [ ] **`A_common/types/diffusion_network.py`** — 契约 5:`DiffusionNetworkInterface`
  - **目的**:DM 抽象基类(契约 §二.2.4)
  - **关键实现点**:
    ```python
    class DiffusionNetworkInterface(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, sample: Tensor, **cond) -> Tensor: raise NotImplementedError()
        def output_shape(self) -> tuple: raise NotImplementedError()
        def shape_info(self) -> str: raise NotImplementedError()  # 推荐,非强制
    ```
  - **源**:自创(契约要求)
  - **注意点**:`forward` **不接 `t` 参数**;`**cond` 必须保留(FiLM / cross-attn 注入用);`shape_info()` 返回字符串如 `"sample=(D_a,), global_cond=(D,)"` 供 Policy `__init__` 末尾 `logger.info`

- [ ] **`A_common/types/base_policy.py`** — 契约 6:`BasePolicy`(Fat Policy 抽象类)
  - **目的**:策略大脑契约(v7 强化版:必须 3 个方法)
  - **关键实现点**:
    ```python
    class BasePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self._normalizer = None
        def forward(self, a_t, t, global_cond) -> Tensor: raise NotImplementedError()  # 单步 DM 封装
        def compute_loss(self, batch: dict) -> Tensor: raise NotImplementedError()    # 训练高层入口
        def predict_action(self, obs: dict) -> ActionOutput: raise NotImplementedError()  # 推理高层入口
        def reset(self): pass  # 状态清空
        def set_normalizer(self, normalizer): self._normalizer = normalizer
        def shape_info(self) -> str: raise NotImplementedError()  # 推荐
    ```
  - **源**:自创(契约要求)
  - **注意点**(v7 硬性):
    1. 3 个**必须方法**:`forward`(单步 DM 封装,由 `compute_loss` / `predict_action` 内部调用) + `compute_loss`(E_cti 训练 loop 唯一入口) + `predict_action`(E_cti 推理 loop 唯一入口)
    2. `_normalizer` 在 `__init__` 末尾初始化为 None,`set_normalizer` 注入
    3. 推荐在 `__init__` 末尾 `logger.info(self.shape_info())` 调试用

### 3.2 5 个数据结构文件

- [ ] **`A_common/types/action_output.py`** — `ActionOutput` dataclass
  - **目的**:统一推理返回契约
  - **关键实现点**:`@dataclass class ActionOutput: actions: Tensor, is_chunk: bool, latency_ms: float = 0.0`
  - **源**:自创(契约要求)
  - **注意点**:`is_chunk=True` 表示动作块,`is_chunk=False` 表示单步;`latency_ms` 供运行时监控

- [ ] **`A_common/types/observation.py`** — `Observation` dataclass
  - **目的**:统一观测数据结构
  - **关键实现点**:`@dataclass class Observation: rgb: dict, state: Tensor, prompt: Optional[str] = None, timestamp: float = 0.0` + `from_dict` 工厂方法
  - **源**:自创(契约要求)
  - **注意点**:`rgb` 是 `dict[str, Tensor]`(多相机:`agentview` / `eye_in_hand` 等)

- [ ] **`A_common/types/state.py`** — `State` dataclass
  - **目的**:机械臂状态结构
  - **关键实现点**:`@dataclass class State: joint_pos: Tensor, joint_vel: Tensor, gripper: Tensor` + `@property vector -> Tensor`
  - **源**:自创(契约要求)
  - **注意点**:`vector` 把三个字段 concat 成一个 Tensor,供 MLP 输入

- [ ] **`A_common/types/normalizer.py`** — `LinearNormalizer` 抽象基类
  - **目的**:归一化统一接口
  - **关键实现点**:`class LinearNormalizer`: `fit(data)` / `normalize(x, key)` / `unnormalize(x, key)` / `state_dict()` / `load_state_dict()`,单字段
  - **源**:改自 PADP diffusion_policy/model/common/normalizer.py(简化)
  - **注意点**:`unnormalize` 在 `Policy.predict_action` 末尾调,`Policy._normalizer["action"].unnormalize(a)`(见 v7 契约 §一)

- [ ] **`A_common/types/normalizer_utils.py`** — 归一化实现工具
  - **目的**:`LinearNormalizer` 的具体实现 + 字段注册
  - **关键实现点**:`get_identity_normalizer_from_dataset` / `get_range_normalizer_from_dataset` / `get_mean_std_normalizer_from_dataset` 等工厂
  - **源**:抄自 PADP diffusion_policy/model/common/normalizer.py
  - **注意点**:阶段 1 简化为"单字段 LinearNormalizer",只支持 mean-std 模式

---

## §4. A_common/registry/(2 个文件)

- [ ] **`A_common/registry/__init__.py`** — 注册表模块入口
  - **目的**:暴露 `register_policy` / `build_policy` / `list_policies` 等接口
  - **关键实现点**:`from .policy_registry import register_policy, build_policy, list_policies`
  - **源**:自创
  - **注意点**:保持简洁,不引入复杂逻辑

- [ ] **`A_common/registry/policy_registry.py`** — `@register_policy` 装饰器 + `build_policy` 工厂
  - **目的**:策略大脑的注册表(v5 改名:从 `algorithm_registry` 改为 `policy_registry`,因为返回的是策略大脑)
  - **关键实现点**:
    ```python
    _REGISTRY = {}
    def register_policy(name: str):
        def deco(cls): _REGISTRY[name] = cls; return cls
        return deco
    def build_policy(cfg: dict) -> BasePolicy:
        name = cfg["name"]; assert name in _REGISTRY, f"未知 policy: {name}"
        return _REGISTRY[name](cfg)
    def list_policies() -> list[str]: return list(_REGISTRY.keys())
    ```
  - **源**:自创(契约要求)
  - **注意点**(**v7 硬性**):
    1. 函数名 **`build_policy`**(不是 `build_algorithm`);装饰器名 **`@register_policy`**(不是 `@register_algorithm`)
    2. 重复注册 / 未知名称要 raise;E_cti 通过 `build_policy(cfg["policy"])` 拿到 Policy 实例
    3. `@register_policy("padp_unet")` / `@register_policy("dp_unet")` 两种注册名(v7 §一)

---

## §5. A_common/data/(3 个文件)

- [ ] **`A_common/data/__init__.py`** — 数据模块入口
  - **目的**:暴露 BaseVLADataset / base_collate
  - **关键实现点**:`from .base_dataset import BaseVLADataset` / `from .base_collator import base_collate`
  - **源**:自创

- [ ] **`A_common/data/base_dataset.py`** — `BaseVLADataset` 抽象基类
  - **目的**:数据集时序契约(铁律 6:显式 `n_obs_steps` + `horizon`)
  - **关键实现点**:
    ```python
    class BaseVLADataset(Dataset):
        def __init__(self, n_obs_steps: int, horizon: int, ...):
            self.n_obs_steps = n_obs_steps
            self.horizon = horizon
        def __len__(self): raise NotImplementedError()
        def __getitem__(self, idx): raise NotImplementedError()
        def get_normalizer(self) -> LinearNormalizer: raise NotImplementedError()
    ```
  - **源**:自创(契约要求)
  - **注意点**(铁律 6):派生类必须在 `__init__` 接收这两个参数并暴露为类属性;`action` shape 严格对齐 `[horizon, D_a]`;episode 首尾 padding 在派生类内处理,**不能**要求 E_cti 兜底

- [ ] **`A_common/data/base_collator.py`** — `base_collate`
  - **目的**:DataLoader 默认 collate,处理 `obs` 嵌套 dict
  - **关键实现点**:`def base_collate(batch): obs_list = [b["obs"] for b in batch]; action = torch.stack([b["action"] for b in batch]); return {"obs": stack_obs(obs_list), "action": action}`
  - **源**:自创
  - **注意点**:`stack_obs` 要递归处理 dict-of-Tensor(`rgb` 是 dict)

---

## §6. A_common/ipc/(1 个文件)

- [ ] **`A_common/ipc/__init__.py`** — 进程间通信占位
  - **目的**:阶段 2 跨进程通信(ZeroMQ / gRPC)占位
  - **关键实现点**:`# 阶段 1 占位,阶段 2 启用 ZMQ socket` 注释
  - **源**:自创
  - **注意点**:阶段 1 留空,不要提前实现

---

## §7. A_common/logger/(3 个文件)

- [ ] **`A_common/logger/__init__.py`** — logger 统一入口
  - **目的**:暴露 `get_logger(name) -> Logger`
  - **关键实现点**:`def get_logger(name: str) -> logging.Logger: return logging.getLogger(name)`,配置 `basicConfig`(`INFO` 级别 + 格式 `%(asctime)s [%(name)s] %(levelname)s: %(message)s`)
  - **源**:自创

- [ ] **`A_common/logger/wandb_logger.py`** — WandB 实验跟踪
  - **目的**:训练 metric 上传到 WandB
  - **关键实现点**:`class WandBLogger: def __init__(self, project, config): self.run = wandb.init(...)` / `def log(self, step, metrics): self.run.log(metrics, step=step)`
  - **源**:改自 PADP
  - **注意点**:阶段 1 可选,先用 json_logger 落地

- [ ] **`A_common/logger/json_logger.py`** — JSON 日志
  - **目的**:本地保存训练 metric
  - **关键实现点**:`class JsonLogger: def __init__(self, path): self.fp = open(path, "a")` / `def log(self, step, metrics): self.fp.write(json.dumps({"step": step, **metrics}) + "\n")`
  - **源**:改自 PADP
  - **注意点**:每行一个 JSON,方便后期 `pandas.read_json` 分析

---

## §8. A_common/ckpt/(3 个文件)

- [ ] **`A_common/ckpt/__init__.py`** — ckpt 模块入口
  - **目的**:暴露 save/load/EMA
  - **关键实现点**:`from .checkpoint import save_checkpoint, load_checkpoint` / `from .ema import EMAModel`
  - **源**:自创

- [ ] **`A_common/ckpt/checkpoint.py`** — checkpoint save/load(safetensors)
  - **目的**:统一 ckpt 序列化
  - **关键实现点**:`def save_checkpoint(state, path)` / `def load_checkpoint(path, map_location)`,`save_checkpoint` 用 `safetensors.torch.save_file` 保存 `state["model_state"]`,`state["ema_state"]`,`state["epoch"]`,`state["global_step"]`
  - **源**:自创
  - **注意点**:不用 `torch.save`(可能含 pickle 漏洞),用 safetensors;E_cti 训练 loop 每 N 步调一次

- [ ] **`A_common/ckpt/ema.py`** — `EMAModel` 指数滑动平均
  - **目的**:训练时维护 policy 参数的 EMA 副本,eval 用 EMA 权重
  - **关键实现点**:`class EMAModel: def __init__(self, model: nn.Module, decay=0.9999): self.decay = decay; self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}` / `def step(self, model): for n, p in model.named_parameters(): self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)` / `def state_dict(self) / load_state_dict(self, state)`
  - **源**:抄 PADP diffusion_policy/model/ema_model.py
  - **注意点**:只 track `requires_grad=True` 的参数;eval 时把 EMA 权重 load 到 model

---

## §9. A_common/tests/(9 个文件 — 每个接口一个测试)

- [ ] **`A_common/tests/__init__.py`** — tests 模块占位
  - **目的**:pytest 识别
  - **关键实现点**:空文件
  - **源**:自创

- [ ] **`A_common/tests/test_action_output.py`** — `ActionOutput` 单元测试
  - **目的**:验证 chunk / step 两种 shape
  - **关键实现点**:`def test_action_output_chunk_shape(): ao = ActionOutput(actions=torch.randn(1, 16, 10), is_chunk=True)`;`def test_action_output_step_shape(): ao = ActionOutput(actions=torch.randn(1, 1, 10), is_chunk=False)`
  - **源**:自创
  - **注意点**:`latency_ms` 默认 0.0,断言 `ao.latency_ms == 0.0`

- [ ] **`A_common/tests/test_observation.py`** — `Observation` 单元测试
  - **目的**:验证 `from_dict` 工厂方法
  - **关键实现点**:`def test_observation_from_dict(): obs = Observation.from_dict({"rgb": {"agentview": torch.randn(3, 224, 224)}, "state": torch.randn(10)})`
  - **源**:自创
  - **注意点**:`prompt` 和 `timestamp` 可选,断言默认值

- [ ] **`A_common/tests/test_vision_encoder.py`** — `VisionEncoderInterface` 契约测试
  - **目的**:v5 新增测试
  - **关键实现点**:`def test_interface_raises(): vi = VisionEncoderInterface()` 调 `vi.forward({})` 应 raise NotImplementedError;`def test_concrete_subclass()`
  - **源**:自创
  - **注意点**:写一个最小 concrete 子类(几行 nn.Conv2d)测试 `output_shape()` 返回正确维度

- [ ] **`A_common/tests/test_action_encoder.py`** — `ActionEncoderInterface` 契约测试
  - **目的**:v5 新增测试
  - **关键实现点**:同上模板,子集是 `[B, T, D_a]` 输入
  - **源**:自创
  - **注意点**:用 `Conv1dActionEmbedder` 作具体子类测试

- [ ] **`A_common/tests/test_timestep_encoder.py`** — `TimestepEncoderInterface` 契约测试
  - **目的**:v5 新增测试
  - **关键实现点**:输入 `[B]` 整数,输出 `[B, D_h]`,用 `SinusoidalTimestepEncoder` 测
  - **源**:自创
  - **注意点**:断言 `forward(torch.tensor([0, 1, 2])).shape == (3, d_h)`

- [ ] **`A_common/tests/test_diffusion_network.py`** — `DiffusionNetworkInterface` 契约测试
  - **目的**:v5 新增测试,3 方法
  - **关键实现点**:`def test_forward_raises()` / `def test_output_shape()` / `def test_shape_info()`(断言返回 string)
  - **源**:自创
  - **注意点**:用 `Unet1DPadp` 作具体子类测试(从 B_model 导入)

- [ ] **`A_common/tests/test_base_policy.py`** — `BasePolicy` 契约测试(v7 强化版)
  - **目的**:v5 新增测试,验证 3 个必须方法
  - **关键实现点**:`def test_3_methods_raises(): bp = BasePolicy()` 调 `forward` / `compute_loss` / `predict_action` 各自 raise
  - **源**:自创
  - **注意点**:**v7 硬性**:`forward(a_t, t, global_cond)` 也是必须;另外测 `set_normalizer` 后 `_normalizer` 不为 None

- [ ] **`A_common/tests/test_policy_registry.py`** — `policy_registry` 单元测试
  - **目的**:验证 register / build / 重复报错 / 未知报错
  - **关键实现点**:
    ```python
    def test_register_and_build():
        @register_policy("test_dummy")
        class DummyPolicy(BasePolicy): pass
        assert "test_dummy" in list_policies()
        p = build_policy({"name": "test_dummy"}); assert isinstance(p, DummyPolicy)
    def test_unknown_raises(): with pytest.raises(AssertionError): build_policy({"name": "nope"})
    ```
  - **源**:自创
  - **注意点**:用 pytest fixture 清理 `_REGISTRY` 避免污染

---

## §10. B_model/encoders/VM/(3 个文件)

- [ ] **`B_model/encoders/VM/__init__.py`** — VM 模块入口
  - **目的**:暴露 `build_vision_encoder(cfg) -> VisionEncoderInterface`
  - **关键实现点**:
    ```python
    _REGISTRY = {"resnet18": ResNet18Encoder, "dinov2_small": DinoV2SmallEncoder}
    def build_vision_encoder(cfg: dict) -> VisionEncoderInterface: return _REGISTRY[cfg["name"]](**cfg["kwargs"])
    ```
  - **源**:自创
  - **注意点**:让 Adapter(PADPAdapter 等)调这个工厂方法选具体 VM

- [ ] **`B_model/encoders/VM/resnet18.py`** — `ResNet18Encoder` 阶段 1 baseline
  - **目的**:去 fc 的 ResNet18 + Linear 投影
  - **关键实现点**:
    ```python
    class ResNet18Encoder(VisionEncoderInterface):
        def __init__(self, d_h: int = 512):
            super().__init__()
            backbone = torchvision.models.resnet18(weights="DEFAULT")
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.proj = nn.Linear(512, d_h)
            self.d_h = d_h
        def forward(self, obs_dict): ...
        def output_shape(self): return (self.d_h,)
    ```
  - **源**:参考 PADP diffusion_policy/model/vision/multi_image_obs_encoder.py(改写)
  - **注意点**:输入 `[B, N_cam, 3, H, W]` 内部 reshape 为 `[B*N_cam, 3, H, W]`,输出 reshape 回 `[B, N_cam, D_h]`

- [ ] **`B_model/encoders/VM/dinov2_small.py`** — `DinoV2SmallEncoder` 阶段 1 备选
  - **目的**:用 DINOv2 预训练替代 ResNet18
  - **关键实现点**:`torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` + `nn.Linear(384, d_h)` 投影
  - **源**:自创
  - **注意点**:d_h 必须 ≥ 384(预训练输出维度);阶段 1 备选不一定要在训练脚本启用

---

## §11. B_model/encoders/AM/(3 个文件)

- [ ] **`B_model/encoders/AM/__init__.py`** — AM 模块入口
  - **目的**:暴露 `build_action_encoder(cfg) -> ActionEncoderInterface`
  - **关键实现点**:`_REGISTRY = {"conv1d_embed": Conv1dActionEmbedder}`
  - **源**:自创

- [ ] **`B_model/encoders/AM/conv1d_embed.py`** — `Conv1dActionEmbedder`
  - **目的**:1D 卷积把历史动作序列编码为嵌入
  - **关键实现点**:
    ```python
    class Conv1dActionEmbedder(ActionEncoderInterface):
        def __init__(self, d_a: int, d_h: int):
            super().__init__()
            self.proj = nn.Conv1d(d_a, d_h, kernel_size=3, padding=1)
            self.d_h = d_h
        def forward(self, action_seq): ...  # [B, T, D_a] -> [B, T, D_h]
        def output_shape(self): return (self.d_h,)
    ```
  - **源**:抽自 PADP diffusion_policy/model/diffusion/conv1d_components.py
  - **注意点**:输入 `[B, T, D_a]`,内部转置为 `[B, D_a, T]` 喂 Conv1d,再转置回来

- [ ] **`B_model/encoders/AM/extra_placeholder.py`** *(若需要预留扩展位,实际可省)*
  - **状态**:本节只列 2 个核心文件;若实际 v5-1 实施手册标 3 个,补一个 `__init__.py` 替代
  - **源**:—

> **修订**:B_model/encoders/AM 在 v5-1 阶段 1 实际只需要 2 个文件(去掉此 placeholder)。这里计为 2 个,§0 统计保留 3 是冗余,实际:0/2。

---

## §12. B_model/networks/TM/(3 个文件)

- [ ] **`B_model/networks/TM/__init__.py`** — TM 模块入口
  - **目的**:暴露 `build_timestep_encoder(cfg) -> TimestepEncoderInterface`
  - **关键实现点**:`_REGISTRY = {"sinusoidal": SinusoidalTimestepEncoder}`
  - **源**:自创

- [ ] **`B_model/networks/TM/sinusoidal.py`** — `SinusoidalTimestepEncoder`
  - **目的**:Sinusoidal 位置编码 + MLP
  - **关键实现点**:
    ```python
    class SinusoidalPosEmb(nn.Module):  # 内联
        def __init__(self, dim): ...
        def forward(self, t): ...  # [B, dim]
    class SinusoidalTimestepEncoder(TimestepEncoderInterface):
        def __init__(self, d_h: int = 512):
            super().__init__()
            self.embed = SinusoidalPosEmb(d_h)
            self.mlp = nn.Sequential(nn.Linear(d_h, d_h*4), nn.SiLU(), nn.Linear(d_h*4, d_h))
            self.d_h = d_h
        def forward(self, t): return self.mlp(self.embed(t))
        def output_shape(self): return (self.d_h,)
    ```
  - **源**:抄 PADP diffusion_policy/model/diffusion/positional_embedding.py
  - **注意点**:`SinusoidalPosEmb` **inline 到本文件**(避免 B_model 跨文件依赖);**TM 归 B_model/networks/ 不归 B_model/encoders/**(因为它处理的是扩散时间步,不是 lerobot 传感器 obs)

- [ ] **`B_model/networks/TM/extra_placeholder.py`** — *(同上备注,实际可省)*
  - **状态**:TM 阶段 1 实际只需 2 个文件

---

## §13. B_model/networks/DM/(3 个文件)

- [ ] **`B_model/networks/DM/__init__.py`** — DM 模块入口
  - **目的**:暴露 `build_diffusion_network(cfg) -> DiffusionNetworkInterface`
  - **关键实现点**:`_REGISTRY = {"unet1d_padp": Unet1DPadp}`
  - **源**:自创

- [ ] **`B_model/networks/DM/unet1d_padp.py`** — `Unet1DPadp`(核心迁移目标)
  - **目的**:1D Conditional UNet,带 FiLM + 可选 cross-attn
  - **关键实现点**:
    ```python
    class Unet1DPadp(DiffusionNetworkInterface):
        def __init__(self, d_a, d_h, n_layers, ...):
            super().__init__()
            # 1D ResBlock + Downsample1d + Upsample1d + FiLM + (optional) cross-attn
            # Downsample1d / Upsample1d / Conv1dBlock / SinusoidalPosEmb 全部 inline
        def forward(self, sample, **cond): ...  # sample: [B, H, D_a]; cond: {global_cond, encoded_t}
        def output_shape(self): return (self.d_a,)
        def shape_info(self): return f"sample=({self.d_a},), global_cond=({self.global_cond_dim},), ..."
    ```
  - **源**:抄 PADP diffusion_policy/model/diffusion/conditional_unet1d_padp.py
  - **注意点**(迁移要点):
    1. `conv1d_components.py` / `modules.py` 的工具类**全部 inline** 到本文件(避免 B_model 跨文件依赖)
    2. 类名 `ConditionalUnet1D` → `Unet1DPadp`
    3. 签名:`forward(sample, t, global_cond)` → `forward(sample, **cond)`,`cond` 为 dict(含 `global_cond`, `encoded_t`)
    4. **不含 `compute_loss` 引用**(loss 在 Gpolicy)
    5. **必须实现** `output_shape()` + `shape_info()`

- [ ] **`B_model/networks/DM/extra_placeholder.py`** — *(同上,实际可省)*

---

## §14. B_model/adapters/(4 个文件 — v5 新增)

- [ ] **`B_model/adapters/__init__.py`** — Adapter 工厂入口
  - **目的**:暴露 `build_adapter(cfg) -> BaseAdapter`
  - **关键实现点**:`_REGISTRY = {"padp": PADPAdapter, "dp": DPAdapter}` + `def build_adapter(cfg): return _REGISTRY[cfg["name"]](cfg)`
  - **源**:自创
  - **注意点**:**v5 新增**;Fat Policy 在 `__init__` 调此工厂选具体 Adapter

- [ ] **`B_model/adapters/base_adapter.py`** — `BaseAdapter` 抽象基类
  - **目的**:所有算法 Adapter 的统一基类
  - **关键实现点**:
    ```python
    class BaseAdapter(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, obs: dict) -> torch.Tensor: raise NotImplementedError()
    ```
  - **源**:自创
  - **注意点**:**v5 新增**;被 Gpolicy 的 Fat Policy 实例化;不强制 `output_shape()`(Adapter 输出维度由 `__init__` 的 fusion 网络决定)

- [ ] **`B_model/adapters/padp_adapter.py`** — `PADPAdapter`(v7 类名:驼峰 PADP+Adapter)
  - **目的**:PADP 算法特征融合器:实例化 VM/AM,做特征融合投影
  - **关键实现点**:
    ```python
    class PADPAdapter(BaseAdapter):  # v7 类名:PADPAdapter(不是 PADPadapter)
        def __init__(self, cfg):
            super().__init__()
            self.vm = build_vision_encoder(cfg.encoder.vm)
            self.am = build_action_encoder(cfg.encoder.am)
            d_vm = self.vm.output_shape()[-1]
            d_am = self.am.output_shape()[-1]
            self.fusion_network = FeatureFusionMLP(in_dims=[d_vm, d_am], out_dim=cfg.adapter.global_cond_dim)
        def forward(self, obs):
            encoded_v = self.vm(obs["image"])
            encoded_a = self.am(obs["action_history"])
            return self.fusion_network(encoded_v, encoded_a)
    ```
  - **源**:仿 PADP diffusion_policy/model/task_padp/multi_image_obs_encoder_clip.py,加 FeatureFusionMLP 投影
  - **注意点**(**v7 硬性**):
    1. **类名 `PADPAdapter`**(不是 `PADPadapter`,v7 驼峰约定)
    2. `self.vm.output_shape()` / `self.am.output_shape()` 在 `__init__` 反查维度,做 fusion 网络的输入维度
    3. fusion 输出维度 = DM 期望的 `global_cond_dim`,Policy `__init__` 末尾 `assert adapter_output_dim == dm.output_shape()[-1]`(参考 v5-1 §五.5.4 末尾)

- [ ] **`B_model/adapters/dp_adapter.py`** — `DPAdapter`(DP 专属)
  - **目的**:DP baseline Adapter,同 PADPAdapter 但无位置加权
  - **关键实现点**:`class DPAdapter(BaseAdapter)` 内部结构与 PADPAdapter **完全相同**,只在 `__init__` 不读 `cfg.padp` 段
  - **源**:同 PADPAdapter(简化)
  - **注意点**:模型结构完全相同,只注册名不同(`@register_adapter("dp")` 走工厂分发)

---

## §15. C_sim/robomimic/(7 个文件)

- [ ] **`C_sim/robomimic/__init__.py`** — robomimic 模块入口
  - **目的**:暴露 `make_env` / `make_dataset` / `make_eval_runner` 三个工厂
  - **关键实现点**:`from .env_impl import make_env` / `from .zarr_dataset import RobomimicZarrDataset` / `from .env_runner import RobomimicImageRunner` / `def make_eval_runner(cfg): return RobomimicImageRunner(cfg.env_meta, n_test=cfg.eval.n_test, ...)`
  - **源**:改自 PADP
  - **注意点**:**铁律 4**:`E_cti` 只 `import C_sim` 然后调工厂,不能 `from C_sim.robomimic.env_runner import ...`

- [ ] **`C_sim/robomimic/env_impl/__init__.py`** — env_impl 入口
  - **目的**:暴露 `make_env(env_meta)`
  - **关键实现点**:`from .env_meta import get_env_meta_for_task` / `from .wrappers import RobomimicImageWrapper` / `def make_env(env_meta): env = RobomimicImageWrapper.build_from_env_meta(env_meta); return env`
  - **源**:改自 PADP
  - **注意点**:env_meta dict 含 `env_name` / `env_type` / `env_kwargs` 等字段

- [ ] **`C_sim/robomimic/env_impl/env_meta.py`** — 23 个 task 元信息
  - **目的**:robomimic 23 个 task 的配置(`square_d0` / `can` / `lift` / ...)
  - **关键实现点**:`TASK_MAP = {"square_d0": {"env_name": "NutAssemblySquare", ...}, ...}` + `def get_env_meta_for_task(task_name) -> dict`
  - **源**:抄 PADP diffusion_policy/common/robomimic_config_util.py
  - **注意点**:阶段 1 至少实现 `square_d0` / `can` / `lift` 3 个,其他 task 留 TODO

- [ ] **`C_sim/robomimic/env_impl/wrappers.py`** — Gym wrapper
  - **目的**:把 robomimic env 包成 gym 接口,obs 格式标准化
  - **关键实现点**:`class RobomimicImageWrapper(gym.Env)` 内部 `reset()` / `step(action)` 返回 `{"rgb": {...}, "state": Tensor}`
  - **源**:抄 PADP diffusion_policy/env/robomimic/robomimic_image_wrapper.py
  - **注意点**:obs dict 结构必须与 `A_common.types.Observation` 兼容(供 `adapter.py` 转)

- [ ] **`C_sim/robomimic/env_runner.py`** — 评估 runner(**v5 关键改造**)
  - **目的**:rollout N 个 episode 算 success_rate
  - **关键实现点**:
    ```python
    class RobomimicImageRunner:
        def __init__(self, env_meta, n_test=50, max_steps=400, ...): ...
        def run(self, predict_fn: callable) -> dict:
            """predict_fn: obs_dict -> ActionOutput(v5:传 callable 不传 policy)"""
            for ep in range(self.n_test):
                obs = self.env.reset()
                done = False
                while not done:
                    action_out = predict_fn(obs)  # ⭐ v5: callable 接口
                    obs, reward, done, info = self.env.step(action_out.actions)
            return {"success_rate": success_count / self.n_test, "per_episode": [...]}
    ```
  - **源**:抄 PADP diffusion_policy/env_runner/robomimic_image_runner_padp.py
  - **注意点**(**v5 硬性**):
    1. `run(self, predict_fn: callable)`,**不是** `run(self, policy)`(v4 旧版)
    2. 接受 `policy.predict_action` 这个 callable,**让 runner 不感知 Gpolicy 内部**
    3. E_cti 调 `runner.run(policy.predict_action)`

- [ ] **`C_sim/robomimic/zarr_dataset.py`** — `RobomimicZarrDataset`
  - **目的**:读 PADP zarr replay buffer,产出 (obs_dict, action) 二元组
  - **关键实现点**:`class RobomimicZarrDataset(BaseVLADataset)` 内部 `ReplayBuffer.copy_from_path(zarr_path)` + `SequenceSampler` 采样
  - **源**:抄 PADP diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py 的 sampler 逻辑
  - **注意点**:`__init__` 接收 `horizon` / `n_obs_steps` / `pad_strategy="edge_repeat"`,并暴露 `self.horizon` / `self.n_obs_steps`(铁律 6)

- [ ] **`C_sim/robomimic/adapter.py`** — sim ↔ A_common.types 转换
  - **目的**:把 sim obs 转为 `A_common.types.Observation` 格式,反之亦然
  - **关键实现点**:`def obs_sim_to_common(sim_obs) -> dict` / `def action_common_to_sim(action_out: ActionOutput) -> np.ndarray`
  - **源**:自创
  - **注意点**:仅做格式转换,不写算法逻辑

- [ ] **`C_sim/robomimic/conda.yaml`** *(阶段 1 可省,环境用 F_envs 统一)*
  - **状态**:本节计 7 个文件,conda.yaml 可省略(用 F_envs/base_train/environment.yml 统一)
  - **注意点**:若保留,只放 robomimic 0.3.0 + 仿真专用 deps

---

## §16. C_sim/{pusht, kitchen, shared}/(3 个占位文件)

- [ ] **`C_sim/pusht/__init__.py`** — PushT 仿真占位
  - **目的**:阶段 2/3 引入 PushT 时启用
  - **关键实现点**:`# 阶段 1 占位` 注释
  - **源**:自创

- [ ] **`C_sim/kitchen/__init__.py`** — FrankaKitchen 仿真占位
  - **目的**:阶段 2/3 引入 Kitchen 时启用
  - **关键实现点**:`# 阶段 1 占位` 注释
  - **源**:自创

- [ ] **`C_sim/shared/__init__.py`** — 跨 sim 共享工具占位
  - **目的**:阶段 2/3 跨 sim 共享的 metric / 可视化工具
  - **关键实现点**:`# 阶段 1 占位` 注释
  - **源**:自创

---

## §17. D_real/(1 个占位)

- [ ] **`D_real/__init__.py`** — 真机边界占位
  - **目的**:阶段 1 不启用,留 factory 占位
  - **关键实现点**:`# 阶段 1 占位,阶段 2 启用 D_real/franka/{controller, state_reader, interpolation, adapter}.py`
  - **源**:自创
  - **注意点**:阶段 1 **不实现** 任何真机代码;只留 `__init__.py` 标阶段 2 计划

---

## §18. E_cti/configs/(2 个文件)

- [ ] **`E_cti/configs/train_padp.yaml`** — PADP 训练配置
  - **目的**:PADP Fat Policy 训练入口配置
  - **关键实现点**:
    ```yaml
    policy:
      name: padp_unet   # v7: cfg.policy.name,不是 cfg.algorithm.name
      horizon: 16
      action_dim: 10
      n_obs_steps: 2
    scheduler: {num_train_timesteps: 1000, beta_schedule: squaredcos_cap_v2}
    padp: {decay: exp, alpha: 0.5}   # PADP 特有:位置加权
    training: {batch_size: 64, num_epochs: 100, lr: 1e-4, device: cuda, ckpt_dir: data/outputs/padp_vla, log_every: 50}
    data: {zarr_path: data/square_d0/replay_buffer.zarr, horizon: 16, n_obs_steps: 2}
    ```
  - **源**:改自 PADP diffusion_policy/config/robomimic_padp_position_wise_v3.yaml
  - **注意点**(**v7 硬性**):顶层是 `policy:`,**不是** `algorithm:`;`policy.name: padp_unet`(对应 `@register_policy("padp_unet")`)

- [ ] **`E_cti/configs/train_dp.yaml`** — DP baseline 训练配置
  - **目的**:DP baseline 训练配置,用于对照
  - **关键实现点**:与 `train_padp.yaml` 结构相同,但 `policy.name: dp_unet`,**无 `padp:` 段**
  - **源**:改自 PADP(同形态)
  - **注意点**:`policy.name: dp_unet`(走 DP pipeline,无位置加权);用于验证架构不引入回归

---

## §19. E_cti/train/(2 个文件 — 纯白痴脚本)

- [ ] **`E_cti/train/run_train.py`** — 训练主入口(纯白痴)
  - **目的**:`if __name__ == "__main__"` 训练 loop,**纯白痴**(铁律 1 强化版)
  - **关键实现点**:
    ```python
    if __name__ == "__main__":
        cfg = yaml.safe_load(open(args.config))
        dataset = C_sim.make_dataset(cfg); normalizer = dataset.get_normalizer()
        policy = build_policy(cfg["policy"]); policy.set_normalizer(normalizer); policy.to(device)
        ema = EMAModel(policy)
        optim = torch.optim.AdamW(policy.parameters(), lr=cfg["training"]["lr"])
        dl = DataLoader(dataset, batch_size=..., collate_fn=base_collate)
        for epoch in range(num_epochs):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                # ⭐ E_cti 唯一动作:把 batch 丢进 policy,拿回 loss
                loss = policy.compute_loss(batch)
                optim.zero_grad(); loss.backward(); optim.step()
                ema.step(policy)
    ```
  - **源**:改自 PADP(整体重写为纯脚本)
  - **注意点**(**v5/v7 硬性 3 子条**):
    1. **1-1 无 class**:整个文件无 `class Xxx` 关键字
    2. **1-2 无算法概念**:不能出现 `add_noise` / `loss_weights` / `DDIMScheduler` / `F.mse_loss` / `MSELoss`
    3. **1-3 不穿透 Policy 内部**:不能出现 `policy.vm.` / `policy.dm.` / `policy.adapter.` / `policy._normalizer.`
    4. 唯一算法动作:`loss = policy.compute_loss(batch)`(E_cti 不写 for t in scheduler.timesteps)
    5. import 顺序:`import A_common, B_model, Gpolicy, C_sim`(B_model / Gpolicy 触发 Policy 注册副作用)

- [ ] **`E_cti/train/run_eval.py`** — 评估主入口(纯白痴 + 铁律 4)
  - **目的**:`if __name__ == "__main__"` 评估 loop
  - **关键实现点**:
    ```python
    if __name__ == "__main__":
        cfg = yaml.safe_load(open(args.config))
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        policy = build_policy(cfg["policy"])
        policy.load_state_dict(ck["model_state"])
        policy.cuda().eval()
        runner = C_sim.make_eval_runner(cfg)
        # ⭐ v5:把 policy.predict_action 当 callable 传给 runner
        metrics = runner.run(policy.predict_action)
        print(metrics)
    ```
  - **源**:改自 PADP(整体重写)
  - **注意点**(**v5/v7 硬性**):
    1. 同样无 class / 无算法概念 / 不穿透
    2. **铁律 4**:只 `import C_sim` 顶层,不 `from C_sim.robomimic.env_runner import ...`
    3. **v5 改造**:`runner.run(policy.predict_action)`,**不是** `runner.run(policy)`
    4. 用 `--config` 复用 `train_*.yaml`,不读独立 `eval.yaml`

---

## §20. E_cti/collect/(1) + E_cti/deploy/(1)(占位)

- [ ] **`E_cti/collect/__init__.py`** — 采集脚本占位
  - **目的**:阶段 1 占位,阶段 2 启用真机/仿真数据采集
  - **关键实现点**:`# 阶段 1 占位,阶段 2 启用 D_real 数据采集脚本`
  - **源**:自创

- [ ] **`E_cti/deploy/__init__.py`** — 部署脚本占位
  - **目的**:阶段 1 占位,阶段 2 启用真机部署
  - **关键实现点**:`# 阶段 1 占位,阶段 2 启用真机部署 + ChunkInterpolator / StepInterpolator`
  - **源**:自创

---

## §21. Gpolicy/common/(3 个文件 — 跨算法共享)

- [ ] **`Gpolicy/__init__.py`** — Gpolicy 模块入口
  - **目的**:暴露子包
  - **关键实现点**:`from .common import denoise_loop` / `from . import DP, PADP`(子包 import)
  - **源**:自创
  - **注意点**:**顶层目录名 `Gpolicy`**(v7 改名,从 `G_algo` → `Gpolicy`)

- [ ] **`Gpolicy/common/__init__.py`** — common 子包入口
  - **目的**:暴露 `denoise_loop`
  - **关键实现点**:`from .denoise_loop import denoise_loop`
  - **源**:自创

- [ ] **`Gpolicy/common/denoise_loop.py`** — 共享 DDIM 推理模板(铁律 5)
  - **目的**:跨算法共享的 DDIM 推理,`scheduler` 作为参数注入
  - **关键实现点**:
    ```python
    @torch.no_grad()
    def denoise_loop(policy: BasePolicy, global_cond: Tensor, scheduler,
                     horizon: int, action_dim: int,
                     normalizer=None) -> Tensor:
        """铁律 5:scheduler 由调用方注入,本函数不做任何 scheduler 实例化"""
        a = torch.randn((global_cond.shape[0], horizon, action_dim),
                        device=global_cond.device, dtype=global_cond.dtype)
        for t in scheduler.timesteps:
            t_batch = torch.full((a.shape[0],), t, device=a.device, dtype=torch.long)
            eps = policy.forward(a, t_batch, global_cond)  # 调单步 DM 封装
            a = scheduler.step(eps, t, a).prev_sample
        if normalizer is not None:
            a = normalizer.unnormalize(a, key="action")
        return a
    ```
  - **源**:参考 PADP diffusion_policy/policy/schedulers.py 的 DDIMScheduler 协议
  - **注意点**(**铁律 5 硬性**):
    1. **禁止** `from diffusers import DDIMScheduler` 或 `DDIMScheduler(...)` 调用
    2. `scheduler` 是参数,不是硬编码实例化
    3. DP / PADP 的 `predict_action` 各自实例化 scheduler 后,作为参数传入
    4. 签名 `def denoise_loop(..., scheduler, ...)` 必须含 `scheduler`

---

## §22. Gpolicy/DP/(3 个文件)

- [ ] **`Gpolicy/DP/__init__.py`** — DP 子包入口
  - **目的**:暴露 `DpUnetPolicy`
  - **关键实现点**:`from .dp_policy import DpUnetPolicy` / `from . import dp_pipeline`
  - **源**:自创

- [ ] **`Gpolicy/DP/dp_pipeline.py`** — DP 工具函数
  - **目的**:DP 算法的工具(loss_weights_dp / scheduler 工厂)
  - **关键实现点**:`def dp_loss_weights(horizon: int) -> Tensor: return torch.ones(horizon)`(DP 无位置加权,全 1)+ `def make_dp_scheduler(cfg) -> DDIMScheduler: ...`
  - **源**:自创
  - **注意点**:DP 与 PADP 唯一差别是 `loss_weights` 全 1

- [ ] **`Gpolicy/DP/dp_policy.py`** — `DpUnetPolicy`(Fat Policy,@register_policy("dp_unet"))
  - **目的**:DP baseline Fat Policy,继承 PADP Policy 但无位置加权
  - **关键实现点**:
    ```python
    @register_policy("dp_unet")
    class DpUnetPolicy(SlidingWindowDiffusionPolicy):
        """DP baseline:继承 PADP Policy,损失函数无位置加权。"""
        def compute_loss(self, batch):
            # 复用父类 compute_loss,但 w 全 1
            a0 = batch["action"]; t = torch.randint(...); noise = torch.randn_like(a0)
            a_t = self.train_scheduler.add_noise(a0, noise, t)
            global_cond = self.adapter(batch["obs"])
            pred_noise = self.forward(a_t, t, global_cond)
            w = torch.ones(self.horizon, device=a0.device)  # 全 1
            return ((pred_noise - noise) ** 2 * w[None, :, None]).mean()
    ```
  - **源**:改自 PADP(继承复用)
  - **注意点**(**v7 硬性**):
    1. **`@register_policy("dp_unet")`**(不是 `@register_algorithm`)
    2. **继承** `SlidingWindowDiffusionPolicy`(共享同一 UNet + adapter,只覆写 `compute_loss`)
    3. 同样在 `__init__` 实例化 scheduler(继承自父类,无需重写)

---

## §23. Gpolicy/PADP/(5 个文件)

- [ ] **`Gpolicy/PADP/__init__.py`** — PADP 子包入口
  - **目的**:暴露 `SlidingWindowDiffusionPolicy` / `padp_loss_weights` / `per_position_mse`
  - **关键实现点**:`from .padp_policy import SlidingWindowDiffusionPolicy` / `from . import padp_pipeline` / `from .loss_weights import padp_loss_weights` / `from .metrics import per_position_mse, per_position_nmse`
  - **源**:自创

- [ ] **`Gpolicy/PADP/padp_policy.py`** — `SlidingWindowDiffusionPolicy`(Fat Policy,v7 强化核心)
  - **目的**:PADP 位置感知扩散 Fat Policy,248 行整体从 PADP 抄
  - **关键实现点**(v7 强化版):
    ```python
    @register_policy("padp_unet")
    class SlidingWindowDiffusionPolicy(BasePolicy):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            # ===== 算力零件装配(B_model 组件)=====
            self.adapter = PADPAdapter(cfg)
            self.tm = SinusoidalTimestepEncoder(cfg.tm.d_h)
            self.denoiser = Unet1DPadp(...)
            # ===== 调度器一次性实例化(v7 Fat Policy)=====
            self.train_scheduler = DDIMScheduler(...)
            self.infer_scheduler = DDIMScheduler(...)
            # ===== 维度缓存(供 predict_action 初始噪声)=====
            self.horizon = cfg.policy.horizon
            self.action_dim = cfg.policy.action_dim
            logger.info(self.shape_info())
        def shape_info(self): return f"vm={...}, am={...}, tm={...}, dm={...}"
        def forward(self, a_t, t, global_cond):  # 单步 DM 封装
            encoded_t = self.tm(t)
            return self.denoiser(a_t, global_cond=global_cond, encoded_t=encoded_t)
        def compute_loss(self, batch):  # 训练高层入口
            a0 = batch["action"]; t = torch.randint(0, T, (B,)); noise = torch.randn_like(a0)
            a_t = self.train_scheduler.add_noise(a0, noise, t)  # 复用 __init__ scheduler
            global_cond = self.adapter(batch["obs"])  # 1 次,不在循环里
            pred_noise = self.forward(a_t, t, global_cond)
            w = padp_loss_weights(self.horizon, decay=self.cfg.padp.decay, alpha=self.cfg.padp.alpha)
            return ((pred_noise - noise) ** 2 * w[None, :, None]).mean()
        def predict_action(self, obs) -> ActionOutput:  # 推理高层入口
            t0 = time.time()
            global_cond = self.adapter(obs)  # 循环外提 1 次特征(性能关键!)
            a = torch.randn(B, self.horizon, self.action_dim, device=self.device, dtype=self.dtype)
            for t in self.infer_scheduler.timesteps:
                pred_noise = self.forward(a, t, global_cond)
                a = self.infer_scheduler.step(pred_noise, t, a).prev_sample
            if self._normalizer is not None:
                a = self._normalizer["action"].unnormalize(a)
            return ActionOutput(actions=a, is_chunk=True, latency_ms=(time.time()-t0)*1000)
        def reset(self): pass
    ```
  - **源**:抄 PADP diffusion_policy/policy/robomimic/diffusion_unet_hybrid_padp.py:SlidingWindowDiffusionPolicy(248 行)
  - **注意点**(**v7 硬性 5 条**):
    1. **`__init__` 一次性实例化** `self.train_scheduler` / `self.infer_scheduler`(不放在方法体里每次 new)
    2. **`__init__` 缓存** `self.horizon` / `self.action_dim`(供 predict_action 初始噪声)
    3. **3 个必须方法**:`forward`(单步 DM 封装) + `compute_loss` + `predict_action`
    4. **循环外提 1 次特征**(性能关键)
    5. **`@register_policy("padp_unet")`**(不是 `@register_algorithm`)
    6. **类名 `SlidingWindowDiffusionPolicy`**(沿用 PADP 命名)
    7. `cfg.policy.horizon` / `cfg.policy.action_dim`(**不是** `cfg.algorithm`)

- [ ] **`Gpolicy/PADP/padp_pipeline.py`** — PADP 工具函数
  - **目的**:PADP 算法的工具(scheduler 工厂 / 训练 hook)
  - **关键实现点**:`def make_padp_scheduler(cfg) -> DDIMScheduler: return DDIMScheduler(num_train_timesteps=cfg.scheduler.num_train_timesteps, beta_schedule=cfg.scheduler.beta_schedule, ...)` + 其他训练工具
  - **源**:自创
  - **注意点**:与 `dp_pipeline.py` 形成对照

- [ ] **`Gpolicy/PADP/loss_weights.py`** — PADP 位置加权(per-horizon weight)
  - **目的**:实现 PADP 特有的位置加权曲线
  - **关键实现点**:
    ```python
    def padp_loss_weights(horizon: int, decay: str = "exp", alpha: float = 0.5) -> Tensor:
        """'exp'   : w[i] = exp(-alpha * i / (H-1))
           'linear': w[i] = 1 - alpha * i / (H-1)
           'const' : 全 1"""
        if decay == "const": return torch.ones(horizon)
        if decay == "linear": return 1 - alpha * torch.arange(horizon) / (horizon - 1)
        if decay == "exp": return torch.exp(-alpha * torch.arange(horizon) / (horizon - 1))
    ```
  - **源**:抄 PADP diffusion_policy/policy/schedulers_padp.py 的 weight 部分
  - **注意点**:输出 `Tensor[H]`,在 `compute_loss` 中广播为 `[None, :, None]` 对 `[B, H, D_a]` 形状加权

- [ ] **`Gpolicy/PADP/metrics.py`** — per-position 评估指标
  - **目的**:per-position MSE / NMSE,用于评估不同时刻位置预测误差
  - **关键实现点**:
    ```python
    def per_position_mse(pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).mean(dim=(0, 2))  # [H]
    def per_position_nmse(pred: Tensor, target: Tensor, scale: Tensor) -> Tensor:
        return per_position_mse(pred, target) / (scale ** 2 + 1e-8)
    ```
  - **源**:自创
  - **注意点**:dim=(0, 2) 把 B 和 D_a 维度平均,留下 per-horizon 的 MSE

---

## §24. 7-Layer 验收检查脚本(v5 强化版 13 行)

> 把以下 13 行检查脚本保存为 `tools/check_7layer.sh`,提交前必跑。

- [ ] **铁律 1-1 自动化**:E_cti 无 class
  - **命令**:`[ -z "$(grep -rE '^class ' E_cti/)" ] && echo "OK 1-1" || echo "FAIL 1-1"`
  - **目的**:E_cti 内所有文件无 `class Xxx`(铁律 1-1)

- [ ] **铁律 1-2 自动化**:E_cti 无算法概念
  - **命令**:`[ -z "$(grep -rE 'add_noise|loss_weights|DDIMScheduler|F\.mse_loss|MSELoss' E_cti/)" ] && echo "OK 1-2" || echo "FAIL 1-2"`
  - **目的**:E_cti 出现 scheduler / 加噪 / loss 字样即为违规

- [ ] **铁律 1-3 自动化**:E_cti 不穿透 Policy 内部
  - **命令**:`[ -z "$(grep -rE 'policy\.(vm|am|tm|denoiser|dm|adapter|_normalizer)\.' E_cti/)" ] && echo "OK 1-3" || echo "FAIL 1-3"`
  - **目的**:E_cti 出现 `policy.vm.` / `policy.dm.` 等穿透调用即为违规

- [ ] **铁律 2a 自动化**:B_model 不含 compose/
  - **命令**:`[ ! -d B_model/compose ] && echo "OK 2a" || echo "FAIL 2a"`
  - **目的**:v5 删 compose,此目录不应存在

- [ ] **铁律 2b 自动化**:B_model 不 import Gpolicy / E_cti
  - **命令**:`[ -z "$(grep -rE 'import Gpolicy|import E_cti' B_model/)" ] && echo "OK 2b" || echo "FAIL 2b"`
  - **目的**:B_model 单向依赖,不依赖上层

- [ ] **铁律 2c 自动化**:B_model 不含 Policy 类 / 算法专属 buffer
  - **命令**:`[ -z "$(grep -rE '@register_policy|class .*Policy\(|compute_loss|predict_action|sliding.window|alpha_bar|inference_buffer' B_model/)" ] && echo "OK 2c" || echo "FAIL 2c"`
  - **目的**:B_model 是纯算力层,不含策略调度

- [ ] **铁律 3 自动化**:5 个 abstract class 不在 B/C/D/E/G 出现
  - **命令**:`[ -z "$(grep -rE 'class VisionEncoderInterface|class ActionEncoderInterface|class TimestepEncoderInterface|class DiffusionNetworkInterface|class BasePolicy' B_model/ C_sim/ D_real/ E_cti/ Gpolicy/)" ] && echo "OK 3" || echo "FAIL 3"`
  - **目的**:5 个契约只在 A_common/types/ 定义

- [ ] **铁律 3b 自动化**:5 个接口文件都在 A_common/types/
  - **命令**:`for f in vision_encoder.py action_encoder.py timestep_encoder.py diffusion_network.py base_policy.py; do [ -f A_common/types/$f ] && echo "  OK A_common/types/$f" || echo "  FAIL"; done`
  - **目的**:5 个契约文件必须存在

- [ ] **铁律 4 自动化**:C_sim / D_real factory
  - **命令**:`[ -z "$(grep -rE 'from C_sim\.[a-z_]+(\.[a-z_]+)+ import' E_cti/)" ] && echo "OK 4" || echo "FAIL 4"` + `grep -E "^def make_env|^def make_eval_runner|^def make_dataset" C_sim/__init__.py`
  - **目的**:E_cti 只 import C_sim 顶层,不导入内部细节

- [ ] **铁律 5 自动化**:denoise_loop scheduler 注入
  - **命令**:`[ -z "$(grep -E 'DDIMScheduler\(|import.*DDIMScheduler|from diffusers' Gpolicy/common/denoise_loop.py)" ] && echo "OK 5" || echo "FAIL 5"` + `grep -E 'def denoise_loop\(.*scheduler' Gpolicy/common/denoise_loop.py`
  - **目的**:denoise_loop 不硬编码 scheduler

- [ ] **铁律 6 自动化**:BaseVLADataset 显式 n_obs_steps / horizon
  - **命令**:`grep -E "self\.n_obs_steps\s*[:=]|self\.horizon\s*[:=]|^\s+n_obs_steps\s*:|^\s+horizon\s*:" A_common/data/base_dataset.py`
  - **目的**:数据集时序契约显式

- [ ] **铁律 7 自动化**:F_envs 纯 conda
  - **命令**:`[ ! -f F_envs/base_train/requirements.txt ] && echo "OK 7" || echo "FAIL 7"` + `grep -A 20 "^dependencies:" F_envs/base_train/environment.yml | grep -E "^\s*-\s*pip:" && echo "OK 7 pip" || echo "FAIL 7"`
  - **目的**:无 requirements.txt,所有依赖在 environment.yml

- [ ] **pytest 8 个测试文件**
  - **命令**:`pytest A_common/tests/ -v`
  - **目的**:8 个契约测试全过(test_action_output / test_observation / test_vision_encoder / test_action_encoder / test_timestep_encoder / test_diffusion_network / test_base_policy / test_policy_registry)

---

## §25. 实施顺序时间线(T0-T7 估算)

> 假设单人全职,无真机/无 VLM 干扰;时间线是乐观估计。

| 阶段 | 时长 | 任务 | 关键产出 |
|---|---|---|---|
| **T0** | 0.5 天 | 根目录 4 文件(§1)+ F_envs 2 文件(§2) | 仓库骨架 + conda 环境就绪 |
| **T1** | 1.5 天 | A_common/types 10 文件(§3)+ registry 2 文件(§4) | 5 接口契约 + 5 数据结构 + 注册表 |
| **T2** | 1.0 天 | A_common/data 3(§5)+ ipc 1(§6)+ logger 3(§7)+ ckpt 3(§8)+ tests 9(§9) | 基础设施完整 + 8 测试可跑 |
| **T3** | 1.5 天 | B_model 编码器:VM 3(§10)+ AM 2(§11)+ TM 2(§12) | ResNet18 / Conv1d / Sinusoidal 调通 |
| **T4** | 2.0 天 | B_model 主干:DM 2(§13)+ adapters 4(§14) | Unet1DPadp + PADPAdapter / DPAdapter 调通 |
| **T5** | 1.5 天 | C_sim 7(§15)+ {pusht, kitchen, shared} 3(§16)+ D_real 1(§17) | robomimic 工厂 + zarr 数据集 + runner 调通 |
| **T6** | 1.0 天 | E_cti/configs 2(§18)+ train 2(§19)+ collect/deploy 2(§20) | 纯白痴脚本 + YAML |
| **T7** | 2.0 天 | Gpolicy/common 3(§21)+ DP 3(§22)+ PADP 5(§23) | Fat Policy 完整 + train_padp.yaml 跑通 |
| **T8** | 0.5 天 | 验收脚本 13 行(§24)+ 跑 7-Layer 检查 | 13 行全 OK + 8 测试全过 |
| **T9** | 2.0 天 | 训练 + 评估 + 调参,square_d0 ≥ 90% | 阶段 1 验收通过 |
| **总计** | **~13.5 天** | 65 文件 + 13 行检查 + 训练验收 | |

> **关键路径**:`B_model/networks/DM/unet1d_padp.py` + `Gpolicy/PADP/padp_policy.py` 是最重头戏(各 2 天);其余 1-1.5 天/层。

---

## §26. v5 vs v4 关键差异对照(用于实施时知道哪些是新增/重命名)

| 维度 | v4(已被推翻) | v5(终极白盒架构,v7 强化) | 影响文件 |
|---|---|---|---|
| **Policy 类形态** | Thin 容器(只暴露 `encode_inputs` / `forward`) | **Fat 策略大脑**(`__init__` 装配 + 缓存 scheduler + 缓存维度,暴露 **3 个方法**:`forward` / `compute_loss` / `predict_action`) | `Gpolicy/PADP/padp_policy.py`,`Gpolicy/DP/dp_policy.py` |
| **Policy 类位置** | `G_policy/<algo>/<algo>_policy.py` | `Gpolicy/<algo>/<algo>_policy.py`(**顶层目录名 `Gpolicy`,从 `G_algo` 重命名**) | `Gpolicy/__init__.py`,`Gpolicy/PADP/__init__.py`,`Gpolicy/DP/__init__.py` |
| **compute_loss 形式** | G_policy 自由函数 `(policy, batch, cfg)` | **Policy 方法** `policy.compute_loss(batch)` | `A_common/types/base_policy.py`(契约)+ `Gpolicy/PADP/padp_policy.py`(实现) |
| **predict_action 形式** | G_policy 自由函数 `(policy, obs, cfg)` | **Policy 方法** `policy.predict_action(obs)` | 同上 |
| **scheduler 生命周期** | 每个方法体内 `new` 一次 | **`__init__` 一次性实例化,持久保存** | `Gpolicy/PADP/padp_policy.py`,`Gpolicy/DP/dp_policy.py` |
| **horizon / action_dim** | cfg 每次重读 | **`__init__` 缓存为 `self.horizon` / `self.action_dim`** | 同上 |
| **E_cti 知道什么** | scheduler / 加噪 / loss / denoise loop | **黑盒** —— 只调 2 个高层方法 | `E_cti/train/run_train.py`,`E_cti/train/run_eval.py` |
| **B_model Adapter** | ❌ 不存在(VM/AM 由 Policy 直接持有) | ✅ **新增** `B_model/adapters/<algo>_adapter.py`(VM/AM 实例化 + 特征融合) | `B_model/adapters/{__init__,base_adapter,padp_adapter,dp_adapter}.py` |
| **A_common 接口数** | 1(只有 BasePolicy) | **5**(VM/AM/TM/DM/BasePolicy) | `A_common/types/{vision_encoder,action_encoder,timestep_encoder,diffusion_network,base_policy}.py` |
| **A_common/registry** | `algorithm_registry.py` | **`policy_registry.py`**(改名:返回的是策略大脑) | `A_common/registry/policy_registry.py` |
| **cfg.algorithm → cfg.policy** | `cfg.algorithm.name` / `cfg.algorithm.horizon` | `cfg.policy.name` / `cfg.policy.horizon`(沿用 v6 习惯) | `E_cti/configs/{train_padp,train_dp}.yaml` |
| **@register_algorithm → @register_policy** | `@register_algorithm("padp_unet")` | `@register_policy("padp_unet")` | `Gpolicy/PADP/padp_policy.py`,`Gpolicy/DP/dp_policy.py` |
| **build_algorithm → build_policy** | `build_algorithm(cfg.algorithm)` | `build_policy(cfg.policy)` | `A_common/registry/policy_registry.py`,E_cti 全部 |
| **铁律 1 强化版** | "E_cti 无 class" 1 条 | "**无 class + 无算法概念 + 不穿透 Policy 内部**" 3 子条 | 验收脚本 1-1/1-2/1-3,E_cti 全部 |
| **类名 PADPadapter → PADPAdapter** | `PADPadapter`(v6) | **`PADPAdapter`**(驼峰,v7) | `B_model/adapters/padp_adapter.py`,`Gpolicy/PADP/padp_policy.py` |
| **Adapter 工厂** | 无 | `B_model/adapters/__init__.py:build_adapter(cfg)` | `B_model/adapters/__init__.py` |
| **env_runner.run 签名** | `runner.run(policy)` | `runner.run(policy.predict_action)`(传 callable) | `C_sim/robomimic/env_runner.py`,`E_cti/train/run_eval.py` |
| **denoise_loop** | 不存在 / 在 E_cti 写 | `Gpolicy/common/denoise_loop.py`,scheduler 注入(铁律 5) | `Gpolicy/common/denoise_loop.py` |
| **BasePolicy.forward** | 不存在 / 只在子类 | **必须有** `forward(a_t, t, global_cond)`(单步 DM 封装) | `A_common/types/base_policy.py` |
| **DP 继承 PADP** | DP / PADP 完全独立类 | DP 继承 PADP Policy(共享 UNet + adapter,只覆写 `compute_loss`) | `Gpolicy/DP/dp_policy.py` |

> **v4 旧 `B_model/compose/` 整体不存在**(v5 已删);`G_algo` → `Gpolicy` 重命名;`PADPadapter` → `PADPAdapter` 类名修正。

---

## 修订记录

| 日期 | 修订人 | 内容 |
|---|---|---|
| 2026-06-14 | Claude(v5-1 Todo 初版) | 基于 v5 架构 + v7 契约产出 65 文件级 Todo 清单;覆盖 7-Layer + Gpolicy 子包进度表;v5 vs v4 关键差异单列;实施时间线 T0-T9 估算;v7 硬性要求分散到各文件注意点 |
