# 提示词: VLA 拆分 model_server + env_client (2 程序)

> 把下面整段内容复制给服务器 AI 即可。提示词以 VLA 路径为主, 必要时往 PADP_v3 找参考。

---

## 任务 (1 句话)

把 VLA 当前的 "1 process 又 spawn server 又跑 env" 拆成 **2 个独立程序**:
- **model_server** (1 process): 加载 ckpt, 接 TCP, 收 obs → 跑 inference → 回 action
- **env_client** (1 process): 跑 robomimic env, 用 **1 个 conn** 跟 model_server 通信, 跑 N episodes

> 参考原 PADP_v3 架构: 1 process + `env_runner.run(policy)` 调 env, 现在拆 client / server 两端。

---

## 必读路径 (服务器 AI 必须先读)

**主路径**: `/home/hy/Desktop/PADP_v3/VLA/`

### 1. 现状代码 (VLA 必须读, 按这个文件夹往下面找)

| 顺序 | 路径 | 说明 |
|---|---|---|
| 1 | `VLA/E_cti/train/padp_for_test_server.py` | 当前 server (加载 ckpt + 接受 client TCP + 调 policy inline, 跟 policy 同 process) |
| 2 | `VLA/E_cti/train/padp_for_libero_server.py` | 1.2 server (Fix-B 后 dumb listener, 不再 auto-avoid port) |
| 3 | `VLA/E_cti/train/padp_for_test_rollout.py` | 当前 rollout (spawn server + 调 client), Fix-A 后用 `port_utils.find_free_port` |
| 4 | `VLA/F_envs/robomimic/tcp_rollout_client.py` | 当前 rollout client (single-env + 跑 25 sequential episodes) |
| 5 | `VLA/E_cti/train/padp_for_test_train.py` | 1.1 train main (251 epochs, 训 + 周期性 rollout) |
| 6 | `VLA/E_cti/train/padp_for_test_protocol.py` | 帧类型 + pack/unpack (BATCHED_OBS 已加, single OBS/ACTION 也有) |
| 7 | `VLA/E_cti/train/port_utils.py` | `find_free_port` helper (Fix-A 改) |
| 8 | `VLA/E_cti/configs/` | 4 个 yaml (1.1_10D / 1.1_7D / 1.2_10D / 1.2_7D) |
| 9 | `VLA/E_cti/configs/lerobot_libero.yaml` | 旧名字, 跟 lerobot_libero7d.yaml 内容一样 |

### 2. 必读报告 (VLA 上级目录)

| 路径 | 说明 |
|---|---|
| `/home/hy/Desktop/PADP_v3/与用户交流/代码现状.md` | VLA 当前架构, 4 路消融, 已改的 17 个文件, 4 步全流程, 端口分配, ckpt 状态 |
| `/home/hy/Desktop/PADP_v3/与用户交流/P0_1.2_10d_server_fail_investigation.md` | 之前 server 60s 没起的根因 (H1: server 偷偷改 sys.argv port, rollout 端 port 局部变量没变), 已 Fix-A/B/E |
| `/home/hy/Desktop/PADP_v3/与用户交流/P2_4ckpt_25_test_envs_result.md` | 4 ckpt 25 sequential envs 跑分 (全 0, pipeline 100% 通, ckpt 训 1 epoch) |
| `/home/hy/Desktop/PADP_v3/与用户交流/P3_output_path_fix.md` | ckpt 输出路径改 VLA/data/outputs/ |

### 3. PADP_v3 参考 (按这个文件夹往下面找, 服务器 AI 可选)

| 路径 | 用途 |
|---|---|
| `PADP_v3/diffusion_policy/config/robomimic_padp_position_wise_v3.yaml` | **原 1 个 yaml** (用户提到, 参考) |
| `PADP_v3/diffusion_policy/env_runner/base_image_runner.py` | env_runner.run(policy) 模式 (用户原话 "env 里面是同时像 env_runner.run(policy) 调用的程序里面一样") |
| `PADP_v3/serve_env.py` | server 端 25 envs (用户不要, 但参考) |
| `PADP_v3/diffusion_policy/serving/env_server.py` | server batched step/reset |
| `PADP_v3/diffusion_policy/serving/env_client.py` | client 1 conn |
| `PADP_v3/diffusion_policy/serving/websocket_async_vector_env_v6.py` | v6 合并版 (推荐参考) |

---

## 目标架构 (2 程序, 1 conn)

```
+-----------------------------+    1 conn     +-----------------------+
|  env_client (1 process)     | <----------> |  model_server (1 proc) |
|                             |  TCP         |                        |
| - 跑 N episodes             |  (1 次建立)  | - 加载 ckpt            |
| - 每 episode: reset + loop  |               | - accept 1 conn = 1 thr |
| - 每 step:                  |               | - 收 obs → predict      |
|   obs → 发到 server          |               | - 回 action            |
|   action → env.step          |               |                        |
| - 报告 metrics              |               |                        |
+-----------------------------+               +-----------------------+
       ^
       | (用户跑 4 路消融时启 4 个 model_server + 4 个 env_client, 各占不同 port)
       v
4 路消融 (A/B/C/D):
  A. 1.1_10D: model_server port 8765, env_client connect
  B. 1.1_7D:  model_server port 8764
  C. 1.2_10D: model_server port 8763
  D. 1.2_7D:  model_server port 8762
```

**关键约束** (用户原话):
- **2 个程序, 1 个连接 (只需要通信一次就行)**
- env client **不** 启 25 个并行 env (P2 报告 1 GPU 撞 EGL, 不可行)
- env client **不** 跑 batched 25 obs (那是 PADP_v3 的搞法, 用户不要)
- model server **不** 跑 25 envs (那是 PADP_v3 的搞法)
- env client 1 process 跑 N episodes (1 GPU 跑 1 env sequential)

**通信协议 (1 conn 复用, 跟现在 VLA 一样)**:
```
# client → server (env_client.send_obs)
{"type": "OBS", "ep": int, "step": int, "obs": dict}

# server → client (model_server.predict_action)
{"type": "ACTION", "ep": int, "step": int, "action": ndarray (D_a,), "latency_ms": float}

# 复用 VLA/E_cti/train/padp_for_test_protocol.py 的 Msg.OBS / Msg.ACTION
```

---

## 实现要求 (服务器 AI 实施)

### Step 1: 新建 `VLA/E_cti/train/model_server.py`

**复用 VLA 现有代码**:
- `VLA/E_cti/train/padp_for_test_server.py:load_policy_from_ckpt` (line 142-180) — ckpt 加载
- `VLA/E_cti/train/padp_for_test_server.py:serve_forever` (line 346-380) — TCP accept + thread spawn
- `VLA/E_cti/train/padp_for_test_server.py:handle_client` (line 218-280) — 处理 OBS 帧 + 调 policy.predict_action + 回 ACTION 帧
- `VLA/E_cti/train/padp_for_test_server.py:obs_to_torch` (line 183-216) — obs 转 torch

**新建 `VLA/E_cti/train/model_server.py` 逻辑**:
- `argparse` 接受 `--ckpt --config --port --host --device`
- 调 `load_policy_from_ckpt(args.ckpt, args.config, args.device)` 加载 policy
- 启 TCP server, accept 1 conn (env_client), 1 thread 处理
- 线程内循环: 收 OBS → `obs_to_torch` → `policy.predict_action` → 回 ACTION
- 不要 spawn env (env 在 client 端)

**参考实现**:
- 几乎 1:1 复制 `VLA/E_cti/train/padp_for_test_server.py` 的 server 部分 (line 142-380)
- 删除 `padp_for_libero_server.py` 的 auto-avoid (Fix-B 已删)

### Step 2: 改 `VLA/F_envs/robomimic/tcp_rollout_client.py`

**加 CLI 参数**:
- `--server_host` (default 127.0.0.1)
- `--server_port` (default 8765)
- 删除 `--port` (client 不再 listen, 改成 connect)

**改 `connect()` 方法** (line 117-150):
- 之前: client 自己 listen + accept
- 现在: client connect 到 `--server_host:server_port`
- 1 conn 复用, 跑 N episodes 不重连

**改 `run_all()` 和 `run_all_batched()`**:
- 1 conn 复用, 跑完所有 episodes 才 close
- `_run_episode` 内部 `send_obs` / `recv_action` 用同一个 `self.sock`

### Step 3: 改 `VLA/E_cti/train/padp_for_test_rollout.py`

**删 spawn server 逻辑** (line 75-118):
- 不再 `subprocess.Popen(server)`
- 改成: 打印 "请先在另一个 terminal 启 model_server" 然后 `sys.exit(0)` 退出
- 或者: 改成 wrapper 脚本 `Agent信息存储/run_4_ckpts_25ep.sh` 同时启 model_server + env_client (用户可以选 auto-spawn flag)

**保留**:
- `_resolve_server_port` (Fix-A) - rollout 端用 `find_free_port`
- `_run_episode` 逻辑

### Step 4: 8 个 yaml 拆分 (可选, 用户没强求, 建议做)

参考 `PADP_v3/diffusion_policy/config/robomimic_padp_position_wise_v3_model_v6.yaml` + `*_v3_env_v6.yaml`, 把现有 4 个 yaml 拆成 8 个:

| model yaml (policy 配置) | env yaml (env/rollout 配置) |
|---|---|
| `padp_for_test_model_v6.yaml` (1.1_10D) | `padp_for_test_env_v6.yaml` |
| `padp_for_test_7d_model_v6.yaml` (1.1_7D) | `padp_for_test_7d_env_v6.yaml` |
| `lerobot_libero10d_model_v6.yaml` (1.2_10D) | `lerobot_libero10d_env_v6.yaml` |
| `lerobot_libero7d_model_v6.yaml` (1.2_7D) | `lerobot_libero7d_env_v6.yaml` |

**model yaml 包含**: policy (name, shape_meta, down_dims, scheduler) + ckpt_path (新增字段)
**env yaml 包含**: data (dataset_path, n_demo) + train (loss_threshold, rollout.interval) + rollout (server, client) + logging

**保留现有 yaml 字段**, 加 `_model` / `_env` 后缀即可。

### Step 5: 4 路消融启动流程

```bash
# Terminal 1: 4 路各起 1 个 model_server (4 个 port)
python VLA/E_cti/train/model_server.py \
  --ckpt /path/to/1.1_10d/latest.ckpt \
  --config VLA/E_cti/configs/padp_for_test_model_v6.yaml \
  --port 8765 --device cuda:0

python VLA/E_cti/train/model_server.py \
  --ckpt /path/to/1.1_7d/latest.ckpt \
  --config VLA/E_cti/configs/padp_for_test_7d_model_v6.yaml \
  --port 8764 --device cuda:0

python VLA/E_cti/train/model_server.py \
  --ckpt /path/to/1.2_10d/latest.ckpt \
  --config VLA/E_cti/configs/lerobot_libero10d_model_v6.yaml \
  --port 8763 --device cuda:1

python VLA/E_cti/train/model_server.py \
  --ckpt /path/to/1.2_7d/latest.ckpt \
  --config VLA/E_cti/configs/lerobot_libero7d_model_v6.yaml \
  --port 8762 --device cuda:1

# Terminal 2: 4 路各跑 1 个 env_client (每个连 1 个 model_server, 25 episodes)
python VLA/F_envs/robomimic/tcp_rollout_client.py \
  --server_host 127.0.0.1 --server_port 8765 \
  --config VLA/E_cti/configs/padp_for_test_env_v6.yaml \
  --action_dim 10 --n_test 25 --max_steps 100

python VLA/F_envs/robomimic/tcp_rollout_client.py \
  --server_host 127.0.0.1 --server_port 8764 \
  --config VLA/E_cti/configs/padp_for_test_7d_env_v6.yaml \
  --action_dim 7 --n_test 25 --max_steps 100

# ... 等等 (1.2 两路同上)
```

---

## 关键约束 (服务器 AI 不要做错)

1. **2 个程序**, 不是 3 个 (用户原话)
2. **1 个 conn 复用** (用户原话 "只需要通信一次就行")
3. **不要** 把 25 envs 跑在 model_server 端 (用户原话 "env 里面是同时像 env_runner.run(policy) 调用的程序里面一样, 直接单个进程使用了")
4. **不要** 用 WebSocket (VLA 现在用 raw TCP, 保持一致)
5. **不要** 跑 batched 25 obs (1 conn 1 step obs, 不 batched)
6. **必须** 保留 VLA 现有的 4 路 yaml 配置 (`loss_threshold=0.01` / `rollout.interval=5` / `enabled=true` 等)
7. **必须** 复用 `padp_for_test_protocol.py` 的 `Msg.OBS` / `Msg.ACTION` 帧 (single-env 协议)
8. **必须** 复用 `port_utils.find_free_port` (Fix-A 后的行为)
9. **必须** 保留 4 路端口分配 (8765/8764/8763/8762, Fix-E 改的)

---

## VLA 现有 4 路 yaml 当前状态 (供参考)

| yaml | port | loss_threshold | enabled | interval | dataset_path |
|---|---|---|---|---|---|
| `padp_for_test_golden.yaml` | 8765 | 0.01 | true | 5 | `data/robomimic/...` (1.1_10D) |
| `padp_for_test_7d_golden.yaml` | 8764 | 0.01 | true | 5 | `data/robomimic/...` (1.1_7D) |
| `lerobot_libero10d.yaml` | 8763 | 0.01 | true | 5 | `data/lerobot/square_d0_lerobot10d` |
| `lerobot_libero7d.yaml` | 8762 | 0.01 | true | 5 | `data/lerobot/square_d0_lerobot7d` |

`ckpt_dir` 全部 `data/outputs/...` (Fix-P3 改的, CWD=VLA 时)

---

## 验证步骤 (服务器 AI 完成后)

```bash
# 1. 启 1 个 model_server (1.1_10D ckpt)
python VLA/E_cti/train/model_server.py \
  --ckpt /path/to/padp_for_test_golden/latest.ckpt \
  --config VLA/E_cti/configs/padp_for_test_golden.yaml \
  --port 8765 --device cuda:0 &

# 2. 跑 1 个 env_client (1 conn, 25 episodes)
python VLA/F_envs/robomimic/tcp_rollout_client.py \
  --server_host 127.0.0.1 --server_port 8765 \
  --task_name square --dataset_path /path/to/square_d0_abs.hdf5 \
  --shape_meta /path/to/1.1_10d_shape_meta_robomimic.json \
  --action_dim 10 --n_test 25 --max_steps 100

# 3. 期望: test_mean_score 输出 (生产 ckpt 训 251 epochs 后应非 0, 当前 ckpt 训 1 epoch 应 0)
```

---

## 完成后产出

服务器 AI 完成后, 报告写到 `/home/hy/Desktop/PADP_v3/与用户交流/PADPv3_拆分实施报告.md`:

1. **新建文件清单** (`model_server.py` + 8 个 yaml)
2. **改的文件清单** (`tcp_rollout_client.py` + `padp_for_test_rollout.py` + 4 个 yaml 改名)
3. **4 路消融启动命令** (上面 Step 5 抄过去)
4. **跟 VLA 当前架构对比**:
   - 代码行数 (新增 vs 删除)
   - 启动时间 (server 启动 + client connect)
   - 内存 (server 加载 ckpt 后, client 启动后)
   - 灵活性 (1 model_server 可服务多 env_client 吗?)
5. **测试结果**: 4 ckpt 25 test_mean_score, latency 跟 VLA 当前对比
6. **已知限制**: 1 GPU 跑 1 env (sequential), 没法 parallel 25 (跟用户决策一致, 不需要改)

---

## 重点: 不要做的事

- **不要** 跑 AsyncVectorEnv 25 envs (1 GPU 撞 EGL, P2 报告已证明)
- **不要** 改 transport 协议 (保持 raw TCP, 跟 VLA 当前一致)
- **不要** 拆 3 个 process (用户只要 2 个)
- **不要** 改 `padp_for_test_protocol.py` 的 `Msg.OBS` / `Msg.ACTION` 帧格式 (保持向后兼容)
- **不要** 删除 VLA 现有的 4 路消融相关代码 (1.2 dataset / 1.2 encoder / 1.2 train wrapper / 1.2 server 都在, 别动)

---

## 关键文件 (服务器 AI 必读清单)

服务器 AI 必须在 VLA 端 读这些文件 (跟 `与用户交流/代码现状.md` 重叠):

1. `VLA/E_cti/train/padp_for_test_server.py` (line 142-380 server + handle_client)
2. `VLA/E_cti/train/padp_for_libero_server.py` (Fix-B 后 dumb listener)
3. `VLA/E_cti/train/padp_for_test_rollout.py` (Fix-A rollout find_free_port)
4. `VLA/F_envs/robomimic/tcp_rollout_client.py` (1 conn / 25 sequential)
5. `VLA/E_cti/train/padp_for_test_train.py` (1.1 train main)
6. `VLA/E_cti/train/padp_for_test_protocol.py` (BATCHED_OBS 协议)
7. `VLA/E_cti/train/port_utils.py` (find_free_port)
8. 4 个 yaml (`VLA/E_cti/configs/`)

PADP_v3 参考 (本地 `/home/hy/Desktop/PADP_v3/PADP_v3/`):
- `diffusion_policy/config/robomimic_padp_position_wise_v3.yaml` (原 1 个 yaml, 用户的 ref)
- `diffusion_policy/env_runner/base_image_runner.py` (env_runner.run 模式)
- `serve_env.py`, `diffusion_policy/serving/env_server.py`, `diffusion_policy/serving/env_client.py`
- `diffusion_policy/serving/websocket_async_vector_env_v6.py` (v6 合并版)
