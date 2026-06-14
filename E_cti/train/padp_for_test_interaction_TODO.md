# padp_for_test rollout 交互逻辑 TODO

> **本文件是 server+client rollout 的"待用户确认"清单**。
> 当前已实现的真版 server+client(robomimic + msgpack 协议)能跑通,但仍有以下 TODO 待补。

---

## ✅ 已完成(本次提交)

- ✅ **A-1 / A-2**: 用 **msgpack + 4B 长度前缀**(同 payload 比 base64+JSON 省 25%,pack 2.8μs / unpack 5.4μs)
- ✅ **D-1**: env 走 VLA `C_sim.robomimic.env_impl.make_env` (robomimic square_d0)
- ✅ **C-1**: 协议已加 `EP_CHANGE` 帧,server 收到时调 `policy.reset()` 并回 `RESET_ACK`
- ✅ protocol 模块:`padp_for_test_protocol.py` 提供 `send_framed` / `recv_framed` / numpy 编码
- ✅ server 调真 `policy.predict_action`,client 调真 `env.step` + `env.is_success()`

---

## 🚧 仍待补 TODO

### Group A: 协议(Protocol) — ✅ 已完成基础,P2 异步待办

| ID | 状态 | 描述 | 涉及文件 |
|---|---|---|---|
| **A-1** | ✅ 完成 | obs 序列化用 msgpack + 4B 长度前缀 | `padp_for_test_protocol.py` |
| **A-2** | ✅ 完成 | action 反序列化同协议 | 同上 |
| **A-3** | ✅ 完成 | 请求/响应边界 = 4B 长度前缀 | 同上 |
| **A-4** | ✅ 完成 | episode/step 边界用 `EP_CHANGE` 帧 | 同上 |
| **A-5** | 🟢 P2 | error/retry 策略(指数 backoff) | `padp_for_test_rollout.py` |

### Group B: 数据格式 — ✅ 已完成基础

| ID | 状态 | 描述 |
|---|---|---|
| **B-1** | ✅ 完成 | rgb 形状: client 做 HWC uint8 → CHW uint8(再加 batch 由 server 端做) |
| **B-2** | ✅ 完成 | lowdim 形状: client 直接发 (D,) float32 |
| **B-3** | ✅ 完成 | action 形状: server 端取第 0 步发 [D_a] float32 |
| **B-4** | 🟡 P1 | n_action_steps 实际步数: 当前写死 1,要支持 chunk > 1 |

### Group C: Policy Reset — ✅ EP_CHANGE 已实现,消融待办

| ID | 状态 | 描述 |
|---|---|---|
| **C-1** | ✅ 完成 | `policy.reset()` 由 server 收到 `EP_CHANGE` 帧时触发,client 发完等 `RESET_ACK` |
| **C-2** | 🟡 P1 | warmup 行为: predict_action 第一次调用会预热 horizon+n_obs_steps-1 步,server 已自动处理(client 不用额外触发) |

#### 🚧 消融实验(后续 TODO,先标记,先不实现)

| 消融开关 | 位置 | 当前默认 | 描述 |
|---|---|---|---|
| **`server_reset_on_ep_change`** | `rollout.server_reset_on_ep_change` (yaml) + `--server_reset_on_ep_change` (server CLI) | `true` | 消融组 A: 收到 EP_CHANGE 帧时是否真调 `policy.reset()`?true = 每 ep 都 reset(源端默认);false = 永远不 reset(测试 sliding window buffer 的累积效应) |
| **`reset_per_epoch`** | `rollout.reset_per_epoch` (yaml,待补) | `false` | 消融组 B: 训练时,每个 epoch 开头是否要 reset inference buffer?PADP 默认不 reset(让 buffer 跨 batch 累积) |
| **`reset_per_rollout`** | `rollout.reset_per_rollout` (yaml,待补) | `true` | 消融组 C: rollout 开头(第一次 rollout 调用)是否要 reset?目前 reset 跟着 EP_CHANGE 走,每 ep reset。如果改成 `false`,整个 rollout 期间 buffer 不 reset |

### Group D: Env 适配(GuidedVLA)

| ID | 状态 | 描述 |
|---|---|---|
| **D-1** | ✅ 完成(robomimic) | env 走 VLA `C_sim.robomimic.env_impl.make_env` |
| **D-2** | 🟡 P1 | n_test 来源: 当前固定 6 个 train init states(可从 hdf5 读);后续可切 libero task id |
| **D-3** | 🟡 P1 | max_steps 按 task 查表(robomimic 已查,libero 待补) |
| **D-4** | 🟡 P1 | success metric: robomimic 用 `env.is_success()`,libero 待适配 |

### Group E: Ckpt 序列化

| ID | 状态 | 描述 |
|---|---|---|
| **E-1** | 🟡 P1 | ckpt 体积: 现在 dump 1.28GB × 每次 rollout。要瘦身: (a) 跳过 optim_state (b) 跳过 EMA (c) 用 safetensors |
| **E-2** | ✅ 完成 | normalizer 单独存 `payload["normalizer_state"]`,server 端 `LinearNormalizer.load_state_dict` 还原 |
| **E-3** | ✅ 完成 | config 走独立 yaml 文件(不嵌 ckpt),server 端 `--config` 读 |

### Group F: 性能 / 可观测性

| ID | 状态 | 描述 |
|---|---|---|
| **F-1** | 🟡 P1 | latency 统计: client 已在 log 中累加 `resp["latency_ms"]` 并打 `avg_latency=%.1fms` |
| **F-2** | 🟡 P1 | WandB / 日志接入: train_logs.json.txt 待加 test_mean_score 字段 |
| **F-3** | 🟡 P1 | GPU 隔离: server 走 `cuda:0`,env(MuJoCo)用 CPU 即可 |

---

## 🚧 异步流水线 TODO(用户提出的核心改进)

> **目标**: 解决"环境观测频率固定、模型推理时间长"的瓶颈——client 发完 obs 后不等 server 回包,直接继续发下一个 obs;server 算完一个就发一个。
> 双方各持一个发送 / 接收缓冲区,允许 K 个 in-flight 请求。

### 设计草图(待实现)

```
client 端:                                          server 端:
┌──────────────┐                                  ┌──────────────┐
│ obs gen      │                                  │ predict_queue│
│   ↓          │                                  │   ↓          │
│ inflight_buf │ ── OBS(seq=0) ─────────────────→ │   ↓          │
│   ↓          │   OBS(seq=1) ──────────────────→ │ worker thread│
│ step env     │   OBS(seq=2) ──────────────────→ │   ↓          │
│   ↓          │                                  │ result_queue │
│ inflight_buf │ ←── ACTION(seq=0) ────────────── │   ↓          │
│   ↓          │     ACTION(seq=1) ────────────   │ inflight_buf │
│ step env     │     ACTION(seq=2) ────────────   │   ↓          │
└──────────────┘                                  └──────────────┘
```

**关键设计点**:
- 每帧 OBS 带 `seq` 编号(0, 1, 2, ...),server 回 ACTION 时带回 `seq`,client 按 seq 排序对齐
- 缓冲区大小 K = 4 (P1) / 8 (P2) / 16 (极限) — 待消融
- 必须保留 EP_CHANGE 帧的特殊地位(它要等 RESET_ACK 才能继续,不能并行)
- 错误处理: server 端 worker thread 失败时,所有 in-flight 请求都标失败

### 实现路径(分步)

1. **P1 - 双缓冲区骨架**: client 端起一个发送线程 + 接收线程,server 端起一个 worker pool
2. **P1 - seq 编号 + 排序**: 每个 OBS/ACTION 帧带 seq,client 端维护"已收到的最大连续 seq"
3. **P2 - 动态 K**: K 可由 rollout_cfg 配置,消融不同 K 对吞吐的影响
4. **P2 - 性能统计**: 打 client 端"平均在飞请求数",server 端"worker 队列长度"

---

## 📞 推荐与用户确认的对话顺序

```
✅ Q1 (P0 / 协议):   "用哪种 IPC 协议?"  →  msgpack+4B 长度前缀
✅ Q2 (P0 / env):    "env 走 robomimic 还是 libero?"  →  robomimic 起步
✅ Q3 (P0 / reset):  "policy.reset() 怎么触发?"  →  EP_CHANGE 帧 + 3 个消融开关
🚧 Q4 (P1 / 异步):   "是否上异步流水线?"  ← 用户已要求,留 TODO 待补
🚧 Q5 (P1 / 消融):   "消融实验跑哪些组合?"  ← 用户已要求留开关,待消融矩阵设计
```

---

## 🛠 当前可跑的最小验证

### A. 验证 server 加载真 policy

```bash
cd /media/disk7t/PADP_v3/VLA
# 1) 训练 1 epoch 产 ckpt(若已有则跳过)
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python \
  E_cti/train/padp_for_test_train.py --max_epochs 1

# 2) 起 server
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python \
  E_cti/train/padp_for_test_server.py \
  --ckpt data/outputs/padp_for_test_golden/latest.ckpt \
  --config E_cti/configs/padp_for_test_golden.yaml --port 8765
# 等看到 "Listening on 127.0.0.1:8765"

# 3) 跑 client(robomimic 真 env)
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python \
  E_cti/train/padp_for_test_client.py \
  --host 127.0.0.1 --port 8765 --n_test 2 --max_steps 50 \
  --config E_cti/configs/padp_for_test_golden.yaml
# 预期: TEST_MEAN_SCORE=0.xxxx (roundtrip 真的成功)
```

### B. 验证 train 脚本内嵌 rollout

```yaml
# E_cti/configs/padp_for_test_golden.yaml
train:
  skip_rollout: false
rollout:
  enabled: true
  rollout_every: 1
  n_test: 2
  max_steps: 50
```

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python \
  E_cti/train/padp_for_test_train.py --max_epochs 2
# 看到 "Epoch X ROLLOUT: test_mean_score=0.xxxx" 说明 server+client 真的被 spawn
```
