"""E_cti.train.padp_for_test_client —— 双协议 rollout client。

## 双协议架构(2026-06-14)

本 client 同时连两个 server,各走各的协议:

```
client 进程
  │
  ├─── WebSocket ──→  ws://<env_host>:<env_port>     (PADP 源端 serve_env_v6.py)
  │                   协议: msgpack_numpy
  │                   接口: diffusion_policy.serving.env_client.EnvClientPolicy
  │                     - reset()       → obs dict
  │                     - step(action)  → (obs, reward, done, info)
  │                     - close()
  │
  └─── TCP ───────→  <policy_host>:<policy_port>     (VLA 端 padp_for_test_server.py)
                     协议: msgpack + 4B 长度前缀
                     接口: padp_for_test_protocol
                       - EP_CHANGE → RESET_ACK     (新 ep 触发 policy.reset)
                       - OBS → ACTION               (action = [D_a] numpy)
                       - EP_END                     (上报 success, 可选)
```

## 流程(每个 step)
1. 通知 policy server: 发 EP_CHANGE → 等 RESET_ACK (若需要, e.g. 第 1 ep)
2. env.reset() → obs
3. while not done:
   - obs → 编 msgpack → 发给 policy server
   - 等 policy server 回 ACTION(action = [D_a])
   - action → env.step() → 新 obs
4. 发 EP_END 报告 success (可选)

## TODO(进 v5-1 残余问题 doc,本阶段先跑通)
- 异步流水线:env 端发 obs 不等 server 回包,server 端并行算下一个
- 消融实验:reset 触发策略对比(per-ep / per-rollout / 全不 reset)
- ckpt 瘦身:rollout ckpt 跳过 optim_state
- GPU 隔离:server 用 cuda:0, env 走 CPU 即可
"""
import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# 路径:同时 import VLA 端 和 PADP 源端
_VLA_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = Path("/media/disk7t/PADP_v3/PADP_v3")
sys.path.insert(0, str(_VLA_ROOT))
sys.path.insert(0, str(_SRC_ROOT))

from A_common.logger import get_logger
from E_cti.train.padp_for_test_protocol import (
    send_framed, recv_framed,
    Msg, pack_obs, pack_ep_change, pack_ep_end,
)

logger = get_logger("padp_for_test_client")


# ====================================================================
# Config 解析(简单支持 ${a.b} 嵌套引用)
# ====================================================================
def _resolve(obj, root):
    if isinstance(obj, dict):
        return {k: _resolve(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(x, root) for x in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        v = root
        for p in obj[2:-1].split("."):
            v = v[p]
        return _resolve(v, root)
    return obj


def env_obs_to_msgpack_obs(obs: dict, shape_meta: dict) -> dict:
    """env 出的 obs dict → msgpack-可序列化的 obs dict。

    Policy server 期望 (msgpack 编码,server 端解):
      - rgb keys:  (C, H, W) uint8       (单帧,S=1 由 server 加 batch)
      - lowdim:    (D,)  float32
    """
    out = {}
    for k, v in obs.items():
        if k not in shape_meta["obs"]:
            continue
        attr = shape_meta["obs"][k]
        if attr.get("type", "low_dim") == "rgb":
            arr = np.asarray(v)
            # HWC uint8 → CHW uint8
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                arr = np.moveaxis(arr, -1, 0)
            out[k] = arr.astype(np.uint8, copy=False)
        else:
            out[k] = np.asarray(v, dtype=np.float32)
    return out


# ====================================================================
# 主 rollout 入口(双协议)
# ====================================================================
def run_rollout(env_host: str, env_port: int,
                policy_host: str, policy_port: int,
                n_test: int, max_steps: int,
                config_path: str) -> float:
    """双协议 rollout 入口:
      - env server:    ws://<env_host>:<env_port>      (PADP 源端)
      - policy server: tcp://<policy_host>:<policy_port> (VLA 端)
    """
    # === 1. 读 config ===
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = _resolve(raw, raw)
    shape_meta = cfg["data"]["shape_meta"]

    # === 2. 连 env server (WebSocket, PADP 源端 EnvClientPolicy) ===
    logger.info("[client] Connecting to env server ws://%s:%d ...", env_host, env_port)
    from diffusion_policy.serving.env_client import EnvClientPolicy
    env_client = EnvClientPolicy(host=env_host, port=env_port)
    logger.info("[client] Env server connected. num_envs=%d, server_metadata=%s",
                env_client.num_envs, env_client.server_metadata)

    # === 3. 连 policy server (TCP, VLA 端 msgpack) ===
    logger.info("[client] Connecting to policy server tcp://%s:%d ...", policy_host, policy_port)
    successes = []
    total_latency_ms = 0.0
    total_actions = 0
    with socket.create_connection((policy_host, policy_port), timeout=30.0) as policy_sock:
        logger.info("[client] Policy server connected. Running %d episodes, max_steps=%d",
                    n_test, max_steps)

        for ep in range(n_test):
            # 3.1 通知 policy server: 新 ep,触发 reset
            send_framed(policy_sock, pack_ep_change())
            ack = recv_framed(policy_sock)
            if ack is None or ack.get("type") != Msg.RESET_ACK:
                logger.error("[client] Policy server did not ack reset: %s", ack)
                successes.append(0.0)
                continue

            # 3.2 重置 env (走 WebSocket → PADP 源端)
            obs = env_client.reset()
            logger.info("[client] ep=%d env.reset() done, obs keys=%s", ep, list(obs.keys()))

            # 3.3 rollout loop
            done = False
            ep_steps = 0
            ep_success = False
            while not done and ep_steps < max_steps:
                # obs → msgpack-可序列化 → 发给 policy server
                obs_msg = env_obs_to_msgpack_obs(obs, shape_meta)
                send_framed(policy_sock, pack_obs(ep=ep, step=ep_steps, obs=obs_msg))
                resp = recv_framed(policy_sock)
                if resp is None:
                    logger.error("[client] Policy server closed at ep=%d step=%d", ep, ep_steps)
                    break
                if resp.get("type") == Msg.ERROR:
                    logger.error("[client] Policy server returned error: %s", resp.get("msg"))
                    break
                if resp.get("type") != Msg.ACTION:
                    logger.error("[client] Unexpected msg type: %s", resp.get("type"))
                    break
                action = np.asarray(resp["action"], dtype=np.float32)
                total_latency_ms += float(resp.get("latency_ms", 0.0))
                total_actions += 1

                # 3.4 action → env.step() (走 WebSocket → PADP 源端)
                try:
                    obs, reward, done, info = env_client.step(action)
                except Exception as e:
                    logger.exception("[client] env.step failed at ep=%d step=%d", ep, ep_steps)
                    break
                ep_steps += 1
                # 25 envs 并行, 但本 client 跑单 ep 走默认 n_envs=1 模式
                # is_success 可能在 info 里; 兼容两种 key
                if info.get("is_success", False):
                    ep_success = True
                    break
                if ep_steps >= max_steps:
                    done = True

            # 3.5 上报 success (可选)
            send_framed(policy_sock, pack_ep_end(ep=ep, success=ep_success))
            successes.append(float(ep_success))
            logger.info("[client] ep=%d done, success=%s, steps=%d",
                        ep, ep_success, ep_steps)

    # === 4. 收尾: 关 env server 连接 ===
    try:
        env_client.close()
    except Exception:
        pass

    success_rate = sum(successes) / max(1, len(successes))
    avg_latency = total_latency_ms / max(1, total_actions)
    logger.info("[client] success_rate=%.4f (n=%d), avg_policy_latency=%.1fms",
                success_rate, len(successes), avg_latency)
    return success_rate


def main():
    ap = argparse.ArgumentParser()
    # env server (PADP 源端, WebSocket)
    ap.add_argument("--env_host", type=str, default="127.0.0.1",
                    help="PADP 源端 serve_env_v6.py 的 host")
    ap.add_argument("--env_port", type=int, default=8766,
                    help="PADP 源端 serve_env_v6.py 的 port (默认 8766)")
    # policy server (VLA 端, TCP)
    ap.add_argument("--policy_host", type=str, default="127.0.0.1",
                    help="VLA 端 padp_for_test_server.py 的 host")
    ap.add_argument("--policy_port", type=int, default=8765,
                    help="VLA 端 padp_for_test_server.py 的 port (默认 8765)")
    ap.add_argument("--n_test", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--config", type=str, required=True,
                    help="yaml config(读 shape_meta)")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("padp_for_test_client starting (双协议: WS+msgpack_numpy ↔ TCP+msgpack)")
    logger.info("=" * 60)
    logger.info("  env server    (WS)  = ws://%s:%d", args.env_host, args.env_port)
    logger.info("  policy server (TCP) = tcp://%s:%d", args.policy_host, args.policy_port)

    t0 = time.time()
    success_rate = run_rollout(
        env_host=args.env_host, env_port=args.env_port,
        policy_host=args.policy_host, policy_port=args.policy_port,
        n_test=args.n_test, max_steps=args.max_steps,
        config_path=args.config,
    )
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done %d episodes in %.1fs, success_rate=%.4f",
                args.n_test, elapsed, success_rate)
    logger.info("=" * 60)
    print(f"TEST_MEAN_SCORE={success_rate:.4f}  (N={args.n_test}, elapsed={elapsed:.1f}s)")


if __name__ == "__main__":
    main()
