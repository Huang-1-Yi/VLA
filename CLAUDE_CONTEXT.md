# CLAUDE_CONTEXT.md
> PADP-VLA v4 项目专用:Claude Code 运行上下文与约束

## === 项目身份 ===
- 项目: PADP-VLA v4(把 PADP diffusion_policy 重构成 7-Layer DDD 架构,目标阶段 1 复现 square_d0 ≥ 90%)
- 服务器路径: /media/disk7t/PADP_v3/VLA
- 参考仓库(只读):
  - PADP 原版: ../../PADP/(提供工作版算法)
  - GuidedVLA: ../../GuidedVLA/(提供工程框架)
- 运行环境: conda env `equidiff`
- Python: 跑前必 `which python` 确认,优先 `$CONDA_PREFIX/bin/python`(避开系统 Python 2.7)

## === 当前阶段 1 目标 ===
- 核心:在 `square` 仿真任务上,复现 PADP 当前推理成功率 ≥ 90%
- 次要:跑通 baseline DP 做对照(差距 ≤ 3%)
- 不做:不做真机、不做多 sim 混训、不做 VLM/FM

## === 文档 8 层结构(2026-06-13 已闭环) ===
VLA/ 执行计划/ 下 10 个文档按"分层"组织,**新会话接续时按以下顺序读**:

| 层 | 文档 | 作用 |
|---|---|---|
| **历史层**(归档) | 落地方案.md (v1) | 早期重构版,已归档 |
|  | 落地方案v2.md | 早期版,已归档 |
|  | 落地方案v3.md | 三阶段首版,已归档 |
| **架构层** | 落地方案v4.md | 7-Layer 架构规范 + 7 铁律 + 4 契约 + 4 提示词 |
| **实施层** | 落地方案v4-1.md | **60 文件 + EMA + 4 bug 修复(当前阶段 1 实施手册)** |
| **归档层** | 落地方案v4-1.1.md | 60 文件精简版(已并入 v4-1) |
| **进度层** | 落地方案v4-1的执行清单.md | 实时打勾 + 修复明细 |
| **Todo 层** | 落地方案v4-1的Todo清单.md | 逐文件勾选 |
| **映射层** | 与现有代码的映射关系.md | PADP/GuidedVLA → VLA 文件级对账(含 §12 v4-1 60 文件增补) |
| **操作层** | 用户执行方案.md | 命令 + AI 提示词 + 踩坑提醒 + 速查表 |

## === 7-Layer DDD 架构(必须遵守) ===
1. `A_common` —— 基础设施(数据契约 / registry / IPC / logger / ckpt / ema)
2. `B_model` —— 纯净模型层(encoders / networks / adapters / compose)
3. `C_sim` —— 仿真边界(只暴露 `__init__.py` 的 factory)
4. `D_real` —— 真机边界(阶段 1 占位)
5. `E_cti` —— 纯执行流(无 class,只 `if __name__ == "__main__"` 脚本)
6. `F_envs` —— 环境管理(纯 conda,弃 uv 与 requirements.txt)
7. `G_algo` —— 算法调度大脑(loss / scheduler / pipeline)

## === 7 条铁律(任何代码 commit 前必过) ===
- **铁律 1**: `E_cti` 不许写 `class`,只写执行脚本
- **铁律 2**: `B_model` 绝不能 `import G_algo` 或 `import E_cti`
- **铁律 3**: 数据结构定义一律进 `A_common/types/`
- **铁律 4**: `C_sim` / `D_real` 对外只暴露 `__init__.py` 的 factory
- **铁律 5**: `G_algo/common/position_noise.py` 不硬编码 scheduler
- **铁律 6**: `A_common/data/base_dataset.py` 显式声明 `n_obs_steps` 与 `horizon`
- **铁律 7**: `F_envs` 走纯 conda 路线,弃 uv 与 requirements.txt

7 铁律自检脚本(每改完跑一次):
```bash
cd VLA
$CONDA_PREFIX/bin/python scripts/sync_todo.py --json > /tmp/sync.json
bash -c '
[ -z "$(grep -rE "^class " E_cti/)" ] && echo "OK 1" || echo "FAIL 1"
[ -z "$(grep -rE "import G_algo|import E_cti" B_model/)" ] && echo "OK 2" || echo "FAIL 2"
[ -z "$(grep -rE "from C_sim\.[a-z_]+(\.[a-z_]+)+ import" E_cti/)" ] && echo "OK 4" || echo "FAIL 4"
[ -z "$(grep -E "DDIMScheduler" G_algo/common/position_noise.py)" ] && echo "OK 5" || echo "FAIL 5"
[ ! -f F_envs/base_train/requirements.txt ] && echo "OK 7" || echo "FAIL 7"
'
```

## === VLA/ 仓库目录 ===
```
VLA/
├── CLAUDE_CONTEXT.md            ← 本文件(跨设备接续用)
├── README.md
├── A_common/    (20+ 文件) 契约代码
├── B_model/     (12 文件) UNet1D PADP + policy
├── C_sim/       (12 文件) robomimic factory
├── D_real/      (1 文件) 阶段 1 占位
├── E_cti/       (6 文件) 纯脚本
├── F_envs/      (4 文件) 纯 conda
├── G_algo/      (6 文件) DP/PADP pipeline
├── scripts/
│   └── sync_todo.py             ← Todo ↔ 文件系统 diff
└── 执行计划/   (10 个规划文档)
```

## === 已修复的 4 个 bug(对照 PADP v3) ===
| # | 修复 | 涉及 |
|---|---|---|
| 1 | 漏传 3 个 obs_encoder kwarg(`crop_shape` / `obs_encoder_group_norm` / `eval_fixed_crop`) | `B_model/compose/padp_policy.py` |
| 2 | `n_action_steps: 8 → 1`(对齐 PADP v3 单步滑窗) | `E_cti/configs/{train_padp,train_dp}.yaml` |
| 3 | `window_exp_gamma: 0.2 → 0.25`(对齐用户命令) | `E_cti/configs/train_padp.yaml` |
| 4 | 新增 EMA(抄 PADP `ema_model.py` 88 行版) | `A_common/ckpt/ema.py` + 接入 `E_cti/train/{run_train,run_eval}.py` |

## === 已修复的 4 个铁律加固 ===
| # | 加固 | 体现位置 |
|---|---|---|
| 1 | 铁律 4(C_sim factory) | `C_sim/__init__.py` 暴露 `make_env / make_dataset / make_eval_runner`;`E_cti/train/run_eval.py` 只 `import C_sim` 不 import 内部 |
| 2 | 铁律 5(denoise_loop scheduler 注入) | `G_algo/common/position_noise.py` 不硬编码 `DDIMScheduler`(留接口给阶段 3 `denoise_loop.py`);当前 padp_policy.py 内联实现 |
| 3 | 铁律 6(时序契约) | `A_common/data/base_dataset.py` 显式 `n_obs_steps` / `horizon`;`C_sim/robomimic/zarr_dataset.py` 严格 `[self.horizon, D_a]` |
| 4 | 铁律 7(纯 conda) | `F_envs/base_train/environment.yml` 唯一文件,`pip:` 段含所有 Python 包;无 `requirements.txt` |

## === 踩坑提醒(从真实跑通过的坑里提炼) ===
| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `Python 2.7.18, pytest-4.6.9` | `pytest` 走到了系统 Python 2.7 | 用 `$CONDA_PREFIX/bin/python -m pytest` |
| 2 | `SyntaxError: Non-ASCII character '\xe2'` | `__init__.py` 有中文/emoji,Python 2.7 严格 | VLA 仓库 35 个 `__init__.py` 已加 `# -*- coding: utf-8 -*-` |
| 3 | `ImportError: cannot import name '_POLICY_REGISTRY'` | `_POLICY_REGISTRY` 是 `policy_registry.py` 私有变量 | 改 `from A_common.registry.policy_registry import _POLICY_REGISTRY` |
| 4 | 终端输出 emoji 乱码 | Windows console GBK 编码 | 加 `PYTHONIOENCODING=utf-8` 前缀 |

## === 端到端命令速查(equidiff env 已激活) ===
```bash
cd /media/disk7t/PADP_v3/VLA

# 1. 契约测试(预期 8 passed)
$CONDA_PREFIX/bin/python -m pytest A_common/tests/ -v

# 2. Todo ↔ 文件系统 diff
$CONDA_PREFIX/bin/python scripts/sync_todo.py --json

# 3. 7 铁律自检
bash -c '
[ -z "$(grep -rE "^class " E_cti/)" ] && echo "OK 1" || echo "FAIL 1"
[ -z "$(grep -rE "import G_algo|import E_cti" B_model/)" ] && echo "OK 2" || echo "FAIL 2"
[ -z "$(grep -rE "from C_sim\.[a-z_]+(\.[a-z_]+)+ import" E_cti/)" ] && echo "OK 4" || echo "FAIL 4"
[ -z "$(grep -E "DDIMScheduler" G_algo/common/position_noise.py)" ] && echo "OK 5" || echo "FAIL 5"
[ ! -f F_envs/base_train/requirements.txt ] && echo "OK 7" || echo "FAIL 7"
'

# 4. 训练
$CONDA_PREFIX/bin/python E_cti/train/run_train.py --config E_cti/configs/train_padp.yaml

# 5. 评估(自动用 EMA 权重)
$CONDA_PREFIX/bin/python E_cti/train/run_eval.py --config E_cti/configs/train_padp.yaml --ckpt data/outputs/padp_vla/latest.ckpt
```

## === Claude 行为约束(必须遵守) ===
- **本服务器侧只做**:跑实验 / 验证 / 测试 / 偶尔修 bug
- **严禁**直接 `git push origin main`
- 改代码优先在当前工作区,不要碰 ../../PADP/ 或 ../../GuidedVLA/
- 禁止未经确认执行破坏性命令:
  - `git reset --hard`
  - `rm -rf`
  - `git push -f`
- 跑 `pytest` / `python` 前必先 `which python`(避开系统 Python 2.7)
- 新 Claude session 接续时,必先 `Read CLAUDE_CONTEXT.md`,然后按"文档 8 层结构"顺序读其它文档

## === 跨设备接续 ===
本文档让另一台设备可以新开 claude 继续聊,代码内容完全一致:
1. `scp -r /media/disk7t/PADP_v3/VLA/ user@other:/path/`
2. `cd VLA && conda activate equidiff`
3. `$CONDA_PREFIX/bin/python -m pytest A_common/tests/ -v`(预期 8 passed)
4. `$CONDA_PREFIX/bin/python E_cti/train/run_train.py --config E_cti/configs/train_padp.yaml`
5. 把运行结果贴回,Claude 根据 [执行计划/用户执行方案.md §5](执行计划/用户执行方案.md) 的 AI 提示词模板继续协作
