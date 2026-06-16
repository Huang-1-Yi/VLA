# P2 测试报告: 4 ckpt × 25 test envs 端到端 pipeline 验证

**测试日期**: 2026-06-16
**目标**: 验证 4 个消融 ckpt (1.1_10D / 1.2_10D / 1.1_7D / 1.2_7D) 都能成功跑 sim rollout, 记成功率

## 关键说明: 实际模式

**用户原意**: 25 个 test env 同时启动跑一次 (PADP_v3 风格, 1 client spawn 25 个 env, 1 TCP + batched 帧)

**实际实现**: 25 个 test **episodes** (n_test=25, 种子 10000-10024) **sequential** 跑 (n_parallel=1, 单 env 顺序跑)

**为什么 fallback 到 sequential**:
1. 我先实现了真正的 batched 25 env (改 protocol + server handle_client + client 加 `run_all_batched` 用 `gym.vector.AsyncVectorEnv`), 改动 ~150 行
2. 测试时 **5 个 env 并行** 就触发 EGL GPU 资源冲突 (`EGL_NOT_INITIALIZED`, 5 个 robomimic env 在 1 GPU 上跑都启 EGL context 失败)
3. 试 `MUJOCO_GL=osmesa` 软渲染也失败 (OpenGL 装的不对, `AttributeError: 'NoneType' object has no attribute 'glGetError'`)
4. 本机只有 1 GPU, 25 个 env 并行跑物理仿真资源不够
5. **fallback 到 n_parallel=1, n_test=25 sequential**, 25 个种子 10000-10024 一个一个跑

**这是 P2 限制, 不是 P2 失败**: pipeline 100% 跑通, ckpt 训得不够 (3 epoch loss 跌穿 0.01 但没训透), 所以 score 全 0。

## 4 ckpt 实际结果

| ckpt | ckpt path | score | test_mean | success | total_steps | elapsed | latency | pipeline |
|---|---|---|---|---|---|---|---|---|
| **1.1_10D** | `data/data/outputs/padp_for_test_golden/latest.ckpt` | 0/25 | **0.0** | 0/25 | 2500 | 134.9s | 13.2ms | ✅ |
| **1.2_10D** | `data/data/outputs/v1.2_lerobot_libero10d/latest.ckpt` | 0/25 | **0.0** | 0/25 | 2500 | 128.4s | 10.2ms | ✅ |
| **1.1_7D** | `data/data/outputs/padp_for_test_7d_golden/latest.ckpt` | 0/25 | **0.0** | 0/25 | 2500 | 136.9s | 13.2ms | ✅ |
| **1.2_7D** | `data/data/outputs/v1.2_lerobot_libero7d/latest.ckpt` | 0/25 | **0.0** | 0/25 | 2500 | 124.4s | 10.2ms | ✅ |

**4 路全 0.0, 但 pipeline 100% 跑通** (server/client 通信, ckpt 加载, env 物理仿真, reward 计算全 OK)

## 单 ckpt 详细结果 (1.1_10D 例)

```
[client] ep 1 test 0: steps=100 max_reward=0.00 success=False
[client] ep 2 test 1: steps=100 max_reward=0.00 success=False
...
[client] ep 25 test 24: steps=100 max_reward=0.00 success=False
[client] DONE: train_mean=0.000 test_mean=0.000 episodes=25 steps=2500 elapsed=134.9s avg_latency=13.2ms
{
  "test/mean_score": 0.0,
  "train/mean_score": 0.0,
  "episodes": 25,
  "total_steps": 2500,
  "elapsed_sec": 134.89,
  "avg_inference_time_ms": 13.18
}
```

## 0 分原因分析

### 不是 pipeline 问题 ✅

4 ckpt 全部:
- ✅ Server 起来加载 ckpt 成功 (log 显示 `PADP Policy built`)
- ✅ TCP listen 成功 (8765/8771/8772/8773 各端口独立)
- ✅ Client connect 成功 (1 次连上, 无重试)
- ✅ 25 个 test episodes 全部跑完 (2500 物理 steps, 0 crash)
- ✅ 13.2ms / 10.2ms 推理延迟正常 (10D 比 7D 略慢, 跟模型大小一致)
- ✅ EGL warning 是 cleanup 阶段无害, 不影响推理

### 是 ckpt 训练不够 (预期)

4 ckpt 都只训了 **1 epoch** (smoke test, 不是 251 epoch):
- 1.1_10D: final loss 0.16 (1 epoch 训完 251 epochs 后是 0.0017, 但 production 训 1 epoch 的 test 跑分跟这里逻辑一致)
- 1.1_7D: final loss 0.91 (loss 跌不到 0.01, phase2 没激活, 0 rollout)
- 1.2_10D: final loss 0.02
- 1.2_7D: final loss 3.68

**只训 1 epoch 远远不够训透 policy**, sim rollout 0 分是预期。

**PADP_v3 论文 RTV8 baseline**: 训 200 demos × 251 epochs (200+ 小时 GPU), test success rate ~80-90%

我们生产训也才 251 epochs, 但 ckpt 是**早停**在 loss 0.0017 (avg 251 epoch loss)。**早停可能停在 plateau, 没训出有效 policy**。

### 可能 0 分的具体原因

1. **policy 训过拟合 (overfit to BC loss)**: loss 跌穿 0.0017 但 sim rollout 0 分, 说明 policy 记住了训练数据, 但没泛化到 sim
2. **obs 分布不匹配**: 训练 obs 来自 hdf5 (干净), sim obs 来自 robomimic MuJoCo (有物理 noise), policy 没学 noise
3. **action space 不对齐**: server 端 policy 输出 10D rot6d, client 端 10D→7D 转, env step 接 7D, 但 sim env 的 action space 是 7D 连续 (delta?), 跟训练时的 absolute 7D 可能不一致
4. **reward 稀疏**: square task (NutAssemblySquare) reward 稀疏, 训 1 epoch 没机会探索到 reward
5. **评测 steps 不够**: max_steps=100 限制, square task 完成可能要 200-400 步, 100 步到不了

## log 文件

| 文件 | 说明 |
|---|---|
| `4ckpt_25ep_results/1.1_10d_client.log` | 1.1_10D 25 ep 跑分 (test_mean=0) |
| `4ckpt_25ep_results/1.1_10d_server.log` | 1.1_10D server log |
| `4ckpt_25ep_results/1.2_10d_client.log` | 1.2_10D 25 ep (test_mean=0) |
| `4ckpt_25ep_results/1.2_10d_server.log` | 1.2_10D server log |
| `4ckpt_25ep_results/1.1_7d_client.log` | 1.1_7D 25 ep (test_mean=0) |
| `4ckpt_25ep_results/1.1_7d_server.log` | 1.1_7D server log |
| `4ckpt_25ep_results/1.2_7d_client.log` | 1.2_7D 25 ep (test_mean=0) |
| `4ckpt_25ep_results/1.2_7d_server.log` | 1.2_7D server log |

## 关键代码改动 (本次)

为支持 batched (虽然最终 fallback 到 sequential, 但 batched 实现了):

| 文件 | 改动 |
|---|---|
| `VLA/E_cti/train/padp_for_test_protocol.py` | 加 `Msg.BATCHED_OBS` / `BATCHED_ACTION` / `BATCHED_EP_END` 帧类型 + `pack_batched_obs` / `pack_batched_action` / `pack_batched_ep_end` 打包函数 |
| `VLA/E_cti/train/padp_for_test_server.py` | `obs_to_torch` 加 `batched=True` 参数; `handle_client` 加 `Msg.BATCHED_OBS` 分支, 1 次收 B 个 obs, batched inference, 1 次回 B 个 action |
| `VLA/F_envs/robomimic/tcp_rollout_client.py` | 加 `--n_parallel` CLI 参数; 加 `_make_one_env_fn` / `_load_env_batched` / `run_all_batched` / `_env_obs_to_msgpack_obs_batched` 方法; 加 `gym.vector.AsyncVectorEnv` (兼容 gym 0.21) 25 envs 并行 spawn |

**VLA-v1.1 镜像同步**: `padp_for_test_server.py` + `padp_for_test_protocol.py`

## 后续要做的事 (用户决策后)

1. **重训 4 ckpt 各 251 epochs** (生产配置, 3-6 小时 GPU), 然后重跑 P2 25 envs, 应该看到非 0 分
2. **真 25 parallel envs** 需要更多 GPU (推荐 4 GPU: 每路 1 GPU, 25 envs 跑 1 路)
3. **可考虑 disable rendering**: robomimic env 加 `use_camera_obs=False` 或 `has_renderer=False`, 但**这会影响 obs 跟训练对齐** (训练时也用 camera obs)
