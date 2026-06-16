# -*- coding: utf-8 -*-
# ============================================================
# F_envs.robomimic.tcp_rollout_client
# VLA policy server 的 TCP 客户端 + 真实 robomimic env rollout
# ============================================================

"""F_envs.robomimic.tcp_rollout_client —— 连 VLA server,跑 n_train+n_test ep。

**协议**:`VLA/E_cti/train/padp_for_test_protocol.py`
  - msgpack + 4B big-endian 长度前缀
  - 帧类型:`OBS / ACTION / EP_CHANGE / EP_END / RESET_ACK / PING / PONG / ERROR`

**调用方**:
  1. CLI: `python -m F_envs.robomimic.tcp_rollout_client --host ... --port ... --task_name square ...`
  2. 代码: `client = PadpRolloutClient(...); client.connect(); result = client.run_all()`

**返回**: dict with
  - test/mean_score, train/mean_score
  - test/per_seed: {seed: max_reward}
  - train/per_seed: {demo_idx: max_reward}
  - episodes, total_steps, elapsed_sec, avg_inference_time_ms
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any, List

import h5py
import numpy as np

# 协议:从 VLA 根目录加 path(本子 AI 不应 import 任何 VLA 内部模块,只 import 协议)
# VLA root = parents[3](F_envs/robomimic/tcp_rollout_client.py -> VLA/F_envs/robomimic/ -> VLA/F_envs/ -> VLA/)
_VLA_ROOT = Path(__file__).resolve().parents[2]
if str(_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLA_ROOT))

from E_cti.train.padp_for_test_protocol import (
    send_framed, recv_framed,
    Msg, pack_obs, pack_batched_obs, pack_ep_change, pack_ep_end,
    pack_batched_ep_end, pack_ping, pack_pong,
)

# 6D→axis_angle 反转换
from C_sim.robomimic.rotation_numpy import rotation_6d_to_axis_angle_batch


# ============================================================
# Obs 编码:env dict -> msgpack-able dict
# ============================================================
def env_obs_to_msgpack_obs(obs: dict, shape_meta: dict) -> dict:
    """把 env 返回的 obs dict 转成 msgpack 可序列化的 dict。

    输入约定(本 env wrapper):
      - rgb keys: (C, H, W) uint8
      - lowdim keys: (D,) float32
    输出:同样格式,值是 numpy,msgpack hook 会自动编码
    """
    out = {}
    for k, v in obs.items():
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        out[k] = v
    return out


# ============================================================
# 主客户端类
# ============================================================
class PadpRolloutClient:
    """连 VLA server,在真实 robomimic env 中跑 n_train+n_test 个 episode。"""

    def __init__(
        self,
        host: str,
        port: int,
        task_name: str,
        dataset_path: str,
        shape_meta: dict,
        n_train: int = 2,
        n_test: int = 4,
        train_start_idx: int = 0,
        test_start_seed: int = 10000,
        max_steps: int = 400,
        abs_action: bool = True,
        timeout_sec: float = 30.0,
        verbose: bool = True,
        wait_server_ready: bool = True,
        # === v1.2 lerobot 并行:env 包装 + init state 源 ===
        use_lerobot_env: bool = False,
        hdf5_for_init_states: Optional[str] = None,
        # === 4 路消融:action 维度 ===
        # 10 = rot6d (PADP 标准, 1.1_10D / 1.2_10D)
        # 7  = axis_angle (1.1_7D / 1.2_7D, server 直接发 7D 不转)
        action_dim: int = 10,
        # === v1.3:25 parallel envs (PADP_v3 AsyncVectorEnv + 1 TCP + batched 帧) ===
        n_parallel: int = 1,
    ):
        self.host = host
        self.port = int(port)
        self.task_name = task_name
        self.dataset_path = os.path.abspath(dataset_path)
        self.shape_meta = shape_meta
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.train_start_idx = int(train_start_idx)
        self.test_start_seed = int(test_start_seed)
        self.max_steps = int(max_steps)
        self.abs_action = bool(abs_action)
        self.timeout_sec = float(timeout_sec)
        self.verbose = bool(verbose)
        self.wait_server_ready = bool(wait_server_ready)
        self.use_lerobot_env = bool(use_lerobot_env)
        # v1.2 lerobot:init states 不能从 lerobot parquet 读(没存 sim state)
        # 默认用 dataset_path(若是 lerobot, caller 必须显式传 hdf5_for_init_states)
        self.hdf5_for_init_states = (
            os.path.abspath(hdf5_for_init_states) if hdf5_for_init_states
            else self.dataset_path
        )
        # 4 路消融:action 维度(10=rot6d, 7=axis_angle)
        assert int(action_dim) in (7, 10), f"action_dim 必须是 7 或 10, 收到 {action_dim}"
        self.action_dim = int(action_dim)
        # v1.3:25 parallel envs 模式 (>1 时走 batched 帧)
        assert int(n_parallel) >= 1, f"n_parallel 必须是 >=1"
        self.n_parallel = int(n_parallel)

        self.sock: Optional[socket.socket] = None
        self.env = None
        self.train_init_states: List[np.ndarray] = []
        self._latencies: List[float] = []
        self._ep_count = 0
        self._step_count = 0

    # ============== 生命周期 ==============

    def connect(self, retry_sec: float = 0.5, max_retry: int = 30) -> None:
        """TCP 连接 + 简单 handshake(发 PING 等 PONG)。

        server 启动通常需要 2-5s 加载 ckpt,这里支持重试。
        """
        if self.verbose:
            print(f"[client] Connecting to {self.host}:{self.port} ...")
        last_err: Optional[Exception] = None
        for attempt in range(max_retry):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout_sec)
                sock.connect((self.host, self.port))
                sock.settimeout(self.timeout_sec)
                self.sock = sock
                if self.wait_server_ready:
                    self._wait_pong()
                if self.verbose:
                    print(f"[client] Connected to {self.host}:{self.port} (attempt {attempt+1})")
                return
            except (ConnectionRefusedError, OSError, socket.timeout) as e:
                last_err = e
                if self.verbose:
                    print(f"[client] attempt {attempt+1}/{max_retry} failed: {e}; "
                          f"retry in {retry_sec:.1f}s ...")
                time.sleep(retry_sec)
        raise ConnectionError(
            f"Failed to connect to {self.host}:{self.port} after {max_retry} attempts"
        ) from last_err

    def _wait_pong(self) -> None:
        """PING/PONG 握手,确认 server 端协议层就绪。"""
        send_framed(self.sock, pack_ping())
        reply = recv_framed(self.sock)
        if reply is None or reply.get("type") != Msg.PONG:
            raise ConnectionError(
                f"Handshake failed: expected PONG, got {reply}"
            )

    def _load_env(self) -> None:
        """懒加载 env + 预读 hdf5 init states。

        v1.2 lerobot 并行:
          - use_lerobot_env=False (1.1 默认):跟原 1.1 一致
          - use_lerobot_env=True  (1.2 lerobot):用 LerobotRobomimicEnv 包装 RobomimicEnv,
            init states 从 hdf5_for_init_states 读(默认 = dataset_path, 若是 lerobot 则 caller 必须显式传)
        """
        if self.verbose:
            print(f"[client] Loading env for task {self.task_name!r} "
                  f"(lerobot_wrapper={self.use_lerobot_env}) ...")
        # 通过 C_sim 注册的工厂造 env
        from C_sim.robomimic.interface_robomimic_env import make_robomimic_env
        # 触发 F_envs.robomimic 注册
        import F_envs.robomimic  # noqa: F401
        from F_envs.robomimic.make_env import _resolve_dataset_path

        # 把 env_factory 用的 dataset_path 强制注入
        # 注意:v1.2 lerobot 时,env_factory 必须用 hdf5 路径(读 env_args),
        #      不是 lerobot 目录
        from F_envs.robomimic import make_env as _make_env_mod
        # 决定 env_factory 用的 dataset path
        if self.use_lerobot_env:
            # v1.2 lerobot: env_factory 用 hdf5_for_init_states (hdf5), dataset_path 留 lerobot
            factory_dataset_path = self.hdf5_for_init_states
        else:
            factory_dataset_path = self.dataset_path
        if factory_dataset_path and os.path.exists(factory_dataset_path):
            # 临时 patch _DEFAULT_DATASETS
            _make_env_mod._DEFAULT_DATASETS[self.task_name] = factory_dataset_path

        raw_env = make_robomimic_env(
            task_name=self.task_name,
            shape_meta=self.shape_meta,
            max_steps=self.max_steps,
            abs_action=self.abs_action,
        )

        # v1.2 lerobot:用包装类
        if self.use_lerobot_env:
            from F_envs.robomimic.lerobot_robomimic_env import LerobotRobomimicEnv
            self.env = LerobotRobomimicEnv(raw_env)
            if self.verbose:
                print(f"[client] Wrapped with LerobotRobomimicEnv (obs_to_lerobot active)")
        else:
            self.env = raw_env

        # 预读 n_train 个 demo 的 init states
        # v1.2 lerobot 路径:用 hdf5_for_init_states 读原 hdf5 的 sim state
        hdf5_path = self.hdf5_for_init_states
        if hdf5_path.endswith(".hdf5") and os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, "r") as f:
                for i in range(self.n_train):
                    demo_idx = self.train_start_idx + i
                    key = f"data/demo_{demo_idx}/states"
                    if key not in f:
                        raise KeyError(
                            f"Dataset missing {key}; available demos: "
                            f"{list(f['data'].keys())[:5]}..."
                        )
                    self.train_init_states.append(f[key][0])
        else:
            raise FileNotFoundError(
                f"Init state hdf5 not found: {hdf5_path}\n"
                f"如果是 lerobot 训练 (1.2), 请用 --hdf5_for_init_states "
                f"指向原 robomimic hdf5 路径"
            )
        if self.verbose:
            print(f"[client] Loaded {len(self.train_init_states)} train init states from {hdf5_path}")

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    # ============== Episode 循环 ==============

    def _run_episode(self, ep_id: int, is_train: bool,
                     init_state: Optional[np.ndarray] = None,
                     seed: Optional[int] = None) -> float:
        """跑一个 episode,返回 max_reward。"""
        # 1. 通知 server 切 ep
        send_framed(self.sock, pack_ep_change())
        ack = recv_framed(self.sock)
        if ack is None or ack.get("type") != Msg.RESET_ACK:
            raise ConnectionError(f"EP_CHANGE no RESET_ACK: got {ack}")

        # 2. 重置 env
        if is_train:
            obs = self.env.reset_to({"states": init_state})
        else:
            self.env.seed(seed)
            obs = self.env.reset()

        # 3. 主循环
        done = False
        total_reward = 0.0
        max_reward = 0.0
        step = 0
        while not done and step < self.max_steps:
            # 3a. 编码 obs 发到 server
            mp_obs = env_obs_to_msgpack_obs(obs, self.shape_meta)
            send_framed(self.sock, pack_obs(ep=ep_id, step=step, obs=mp_obs))

            # 3b. 等 server 回 action
            reply = recv_framed(self.sock)
            if reply is None:
                raise ConnectionError("Server closed connection mid-episode")
            if reply.get("type") == Msg.ERROR:
                raise RuntimeError(f"Server ERROR: {reply.get('msg')}")
            if reply.get("type") != Msg.ACTION:
                raise RuntimeError(f"Expected ACTION, got {reply.get('type')}")
            action = np.asarray(reply["action"], dtype=np.float32)
            if action.ndim != 1 or action.shape[0] != self.action_dim:
                raise ValueError(
                    f"Expected action shape ({self.action_dim},), got {action.shape}"
                )
            if "latency_ms" in reply:
                self._latencies.append(float(reply["latency_ms"]))

            # 3c. 4 路消融:action 维度转换
            # - 10D rot6d: 走 rotation_6d_to_axis_angle_batch → 7D axis_angle 给 env.step
            # - 7D axis_angle: 直接送 env.step (env.action_space 本来就是 7D)
            if self.abs_action and self.action_dim == 10:
                env_action = rotation_6d_to_axis_angle_batch(action[None, :])[0]
            else:
                # 7D abs 或任意 delta: 直接送
                env_action = action

            # 3d. env step
            obs, reward, done, info = self.env.step(env_action)
            total_reward += float(reward)
            max_reward = max(max_reward, float(reward))
            step += 1

        # 4. 上报 EP_END
        success = max_reward > 0.5
        send_framed(self.sock, pack_ep_end(ep=ep_id, success=success))

        self._ep_count += 1
        self._step_count += step
        if self.verbose:
            tag = "train" if is_train else "test"
            print(f"[client] ep {self._ep_count} {tag} {ep_id}: "
                  f"steps={step} max_reward={max_reward:.2f} success={success}")
        return max_reward

    # ============== 跑所有 ep ==============

    def run_all(self) -> dict:
        """跑 n_train + n_test 个 episode,返回统计 dict。"""
        if self.sock is None:
            raise RuntimeError("Not connected; call connect() first")
        if self.env is None:
            self._load_env()

        t_start = time.time()
        train_scores: Dict[int, float] = {}
        test_scores: Dict[int, float] = {}

        # train ep
        for i in range(self.n_train):
            demo_idx = self.train_start_idx + i
            init_state = self.train_init_states[i]
            score = self._run_episode(
                ep_id=i, is_train=True,
                init_state=init_state, seed=None,
            )
            train_scores[demo_idx] = score

        # test ep
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            score = self._run_episode(
                ep_id=self.n_train + i, is_train=False,
                init_state=None, seed=seed,
            )
            test_scores[seed] = score

        elapsed = time.time() - t_start
        train_mean = float(np.mean(list(train_scores.values()))) if train_scores else 0.0
        test_mean = float(np.mean(list(test_scores.values()))) if test_scores else 0.0
        avg_latency = float(np.mean(self._latencies)) if self._latencies else 0.0

        result = {
            "test/mean_score": test_mean,
            "train/mean_score": train_mean,
            "test/per_seed": test_scores,
            "train/per_seed": train_scores,
            "episodes": self._ep_count,
            "total_steps": self._step_count,
            "elapsed_sec": elapsed,
            "avg_inference_time_ms": avg_latency,
        }
        if self.verbose:
            print(f"[client] DONE: train_mean={train_mean:.3f} test_mean={test_mean:.3f} "
                  f"episodes={self._ep_count} steps={self._step_count} "
                  f"elapsed={elapsed:.1f}s avg_latency={avg_latency:.1f}ms")
        return result

    # ============================================================
    # v1.3:25 parallel envs (PADP_v3 AsyncVectorEnv + 1 TCP + batched 帧)
    # ============================================================
    def _make_one_env_fn(self, idx: int):
        """生成第 idx 个 env 工厂 (gym.vector.AsyncVectorEnv 调用)。"""
        from C_sim.robomimic.interface_robomimic_env import make_robomimic_env
        import F_envs.robomimic  # noqa: F401
        from F_envs.robomimic import make_env as _make_env_mod

        def _thunk():
            # 1 个 env (跟 single-env _load_env 同样的逻辑)
            if self.use_lerobot_env:
                factory_dataset_path = self.hdf5_for_init_states
            else:
                factory_dataset_path = self.dataset_path
            if factory_dataset_path and os.path.exists(factory_dataset_path):
                _make_env_mod._DEFAULT_DATASETS[self.task_name] = factory_dataset_path

            raw_env = make_robomimic_env(
                task_name=self.task_name,
                shape_meta=self.shape_meta,
                max_steps=self.max_steps,
                abs_action=self.abs_action,
            )
            if self.use_lerobot_env:
                from F_envs.robomimic.lerobot_robomimic_env import LerobotRobomimicEnv
                return LerobotRobomimicEnv(raw_env)
            return raw_env
        return _thunk

    def _load_env_batched(self, n_envs: int):
        """懒加载 n_envs 个 env (AsyncVectorEnv 模式), 全部用同 1 个 dataset_path。"""
        try:
            from gymnasium.vector import AsyncVectorEnv
        except ImportError:
            from gym.vector import AsyncVectorEnv  # gym 0.21
        if self.verbose:
            print(f"[client] Loading {n_envs} parallel envs for task {self.task_name!r} "
                  f"(lerobot_wrapper={self.use_lerobot_env}) ...")
        env_fns = [self._make_one_env_fn(i) for i in range(n_envs)]
        self.venv = AsyncVectorEnv(env_fns)
        if self.verbose:
            print(f"[client] AsyncVectorEnv ready: num_envs={n_envs}, observation_space={self.venv.single_observation_space}")

    def run_all_batched(self) -> dict:
        """v1.3:25 parallel envs (PADP_v3 风格, 1 TCP + batched 帧)。

        流程:
          1. AsyncVectorEnv 启 n_parallel 个 env (multiprocessing spawn)
          2. 1 次 EP_CHANGE → 等 RESET_ACK
          3. 跑 max_steps 步, 每步:
             a) 收集 B 个 env 的 obs → 1 个 BATCHED_OBS 帧
             b) 1 个 BATCHED_ACTION 帧回 (B, D) action
             c) 10D → 7D (如果是 1.1_10D / 1.2_10D)
             d) venv.step(actions) → B 个 env 并行 step
          4. 1 个 BATCHED_EP_END 帧报 results
        """
        if self.sock is None:
            raise RuntimeError("Not connected; call connect() first")
        B = self.n_parallel
        if B <= 1:
            # fallback 到 single-env
            return self.run_all()

        # 1. 启 n_parallel 个 env
        self._load_env_batched(B)

        t_start = time.time()
        # 2. 通知 server 开始 EP
        send_framed(self.sock, pack_ep_change())
        ack = recv_framed(self.sock)
        if ack is None or ack.get("type") != Msg.RESET_ACK:
            raise ConnectionError(f"EP_CHANGE no RESET_ACK: got {ack}")

        # 3. reset B 个 env (AsyncVectorEnv 自动 reset)
        obs_list = self.venv.reset()  # dict of (B, ...) arrays
        # 4. 主循环
        B_rewards = np.zeros(B, dtype=np.float32)
        B_max_rewards = np.zeros(B, dtype=np.float32)
        B_dones = np.zeros(B, dtype=bool)  # True 表示该 env 已 done
        B_steps = np.zeros(B, dtype=int)
        # AsyncVectorEnv 用 gymnasium 风格 done mask (terminated | truncated)
        # 当 done=True, 该 env 自动 reset, 但 obs 是 reset 后 (在 step() 返回)
        # 我们用 auto_reset 简化

        t_first_step = None
        for t in range(self.max_steps):
            # 3a. 收集 B 个 obs
            if isinstance(obs_list, tuple):
                # gymnasium 风格: (obs, info) — 但 AsyncVectorEnv 没 info
                obs_dict = obs_list
            else:
                obs_dict = obs_list
            # 编码 obs (跟 single-env env_obs_to_msgpack_obs 类似)
            mp_obs_dict = self._env_obs_to_msgpack_obs_batched(obs_dict)
            # 3b. 1 个 BATCHED_OBS 帧
            eps = list(range(B))
            steps = [t] * B
            t0 = time.time()
            if t_first_step is None:
                t_first_step = t0
            send_framed(self.sock, pack_batched_obs(eps=eps, steps=steps, obs_list=mp_obs_dict))
            # 3c. 1 个 BATCHED_ACTION 帧回
            reply = recv_framed(self.sock)
            if reply is None:
                raise ConnectionError("Server closed connection mid-batch")
            if reply.get("type") == Msg.ERROR:
                raise RuntimeError(f"Server ERROR: {reply.get('msg')}")
            if reply.get("type") != Msg.BATCHED_ACTION:
                raise RuntimeError(f"Expected BATCHED_ACTION, got {reply.get('type')}")
            actions = np.asarray(reply["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape != (B, self.action_dim):
                raise ValueError(f"Expected actions shape ({B}, {self.action_dim}), got {actions.shape}")
            if "latency_ms" in reply:
                self._latencies.append(float(reply["latency_ms"]))
            # 3d. 10D → 7D (abs_action + 10D 时)
            if self.abs_action and self.action_dim == 10:
                env_actions = rotation_6d_to_axis_angle_batch(actions)  # (B, 7)
            else:
                env_actions = actions
            # 3e. AsyncVectorEnv 并行 step
            # gym 0.21: (obs, reward, done, info) 4-tuple; gymnasium: (obs, reward, term, trunc, info) 5-tuple
            step_ret = self.venv.step(env_actions)
            if len(step_ret) == 5:
                obs_list, rewards, terms, truncs, infos = step_ret
                new_dones = terms | truncs
            else:
                obs_list, rewards, dones, infos = step_ret
                new_dones = dones
            # 3f. 累加 reward
            B_rewards += rewards
            B_max_rewards = np.maximum(B_max_rewards, rewards)
            B_steps += 1
            for i in range(B):
                if new_dones[i] and not B_dones[i]:
                    B_dones[i] = True
                    if self.verbose:
                        print(f"[client] ep {i} done at step={t}: max_reward={B_max_rewards[i]:.2f}")

        # 4. 报 EP_END (B 个 env)
        successes = [bool(mr > 0.5) for mr in B_max_rewards]
        send_framed(self.sock, pack_batched_ep_end(eps=list(range(B)), successes=successes))

        # 5. 统计
        elapsed = time.time() - t_start
        avg_latency = float(np.mean(self._latencies)) if self._latencies else 0.0
        n_success = int(np.sum([s for s in successes]))
        success_rate = n_success / B

        result = {
            "n_parallel": B,
            "n_episodes": B,
            "n_success": n_success,
            "success_rate": success_rate,
            "max_rewards": [float(x) for x in B_max_rewards],
            "rewards_sum": [float(x) for x in B_rewards],
            "steps_per_ep": [int(x) for x in B_steps],
            "episodes_done": [bool(x) for x in B_dones],
            "elapsed_sec": elapsed,
            "avg_inference_time_ms": avg_latency,
            "total_steps": int(np.sum(B_steps)),
        }
        if self.verbose:
            print(f"[client] DONE (batched, n_parallel={B}): "
                  f"success_rate={success_rate:.2%} ({n_success}/{B}) "
                  f"steps={int(np.sum(B_steps))} elapsed={elapsed:.1f}s "
                  f"avg_latency={avg_latency:.1f}ms")
        return result

    def _env_obs_to_msgpack_obs_batched(self, obs_dict: dict) -> list:
        """B 个 env 的 obs dict (B, ...) → msgpack-able list of dicts (跟 single-env obs 一致)."""
        keys = list(obs_dict.keys())
        B = len(obs_dict[keys[0]]) if keys else 0
        out = []
        for i in range(B):
            o = {k: np.asarray(obs_dict[k][i]) for k in keys}
            out.append(o)
        return out


# ============================================================
# CLI
# ============================================================
def _default_shape_meta() -> dict:
    """从 VLA golden config 抄一份 shape_meta(只在 CLI 兜底用)。"""
    return {
        "obs": {
            "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3]},
            "robot0_eef_quat": {"shape": [4]},
            "robot0_gripper_qpos": {"shape": [2]},
        },
        "action": {"shape": [10]},
    }


def main():
    parser = argparse.ArgumentParser(
        description="PADP rollout client: robomimic env + VLA policy server over TCP"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="VLA server host")
    parser.add_argument("--port", type=int, default=8765,
                        help="VLA server port")
    parser.add_argument("--task_name", type=str, default="square",
                        help="robomimic task name (lift/can/square/...)")
    parser.add_argument("--dataset_path", type=str,
                        default="data/robomimic/datasets/square_d0/square_d0_abs.hdf5",
                        help="hdf5 dataset path (relative to VLA/ or absolute)")
    parser.add_argument("--shape_meta", type=str, default=None,
                        help="shape_meta JSON 文件路径(可选,默认 hard-coded)")
    parser.add_argument("--n_train", type=int, default=2)
    parser.add_argument("--n_test", type=int, default=4)
    parser.add_argument("--train_start_idx", type=int, default=0)
    parser.add_argument("--test_start_seed", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--abs_action", action="store_true", default=True,
                        help="绝对动作(server 期望 10D 6D)")
    parser.add_argument("--timeout_sec", type=float, default=30.0)
    parser.add_argument("--max_retry", type=int, default=30)
    # === v1.2 lerobot 并行 ===
    parser.add_argument("--use_lerobot_env", action="store_true", default=False,
                        help="v1.2 lerobot:用 LerobotRobomimicEnv 包装 (obs_to_lerobot)")
    parser.add_argument("--hdf5_for_init_states", type=str, default=None,
                        help="v1.2 lerobot:train init states 源 hdf5 (lerobot parquet 没存 sim state)")
    # === 4 路消融:action 维度 ===
    parser.add_argument("--action_dim", type=int, default=10, choices=[7, 10],
                        help="action 维度: 10=rot6d (PADP), 7=axis_angle (1.2 7D)")
    # === v1.3:25 parallel envs (PADP_v3 AsyncVectorEnv 风格) ===
    parser.add_argument("--n_parallel", type=int, default=1,
                        help="v1.3:并行 env 数 (>1 时走 batched 25 parallel 模式, 跟 PADP_v3 一致)")
    args = parser.parse_args()

    # 解析 dataset_path(2026-06-15 主 AI 修复:不要拼 _VLA_ROOT,直接用 CWD)
    # 用户的 CWD 可能 = VLA/ 或 = /home/hy/Desktop/PADP_v3/ 之一,
    # 用 os.path.abspath 自动按 CWD 解析,避免双重 "VLA/" 拼接
    if not os.path.isabs(args.dataset_path):
        args.dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.exists(args.dataset_path):
        print(f"[error] dataset not found: {args.dataset_path}", file=sys.stderr)
        sys.exit(1)

    # 解析 shape_meta(同样的修复:用 CWD 解析)
    if args.shape_meta is not None:
        if not os.path.isabs(args.shape_meta):
            args.shape_meta = os.path.abspath(args.shape_meta)
        if os.path.exists(args.shape_meta):
            with open(args.shape_meta) as f:
                shape_meta = json.load(f)
        else:
            print(f"[warn] shape_meta not found: {args.shape_meta}, using default",
                  file=sys.stderr)
            shape_meta = _default_shape_meta()
    else:
        shape_meta = _default_shape_meta()

    # 解析 max_steps(从 env_meta 推断更安全)
    from F_envs.robomimic.env_meta import get_env_meta
    meta = get_env_meta(args.task_name)
    if args.max_steps <= 0:
        args.max_steps = meta["max_steps"]

    client = PadpRolloutClient(
        host=args.host,
        port=args.port,
        task_name=args.task_name,
        dataset_path=args.dataset_path,
        shape_meta=shape_meta,
        n_train=args.n_train,
        n_test=args.n_test,
        train_start_idx=args.train_start_idx,
        test_start_seed=args.test_start_seed,
        max_steps=args.max_steps,
        abs_action=args.abs_action,
        timeout_sec=args.timeout_sec,
        verbose=True,
        wait_server_ready=True,
        use_lerobot_env=args.use_lerobot_env,
        hdf5_for_init_states=args.hdf5_for_init_states,
        action_dim=args.action_dim,
        n_parallel=args.n_parallel,
    )
    try:
        client.connect(max_retry=args.max_retry)
        # v1.3:25 parallel envs 模式 (PADP_v3 风格)
        if args.n_parallel > 1:
            result = client.run_all_batched()
        else:
            result = client.run_all()
        print(json.dumps(result, indent=2, default=_json_default))
    finally:
        client.close()


def _json_default(obj):
    """支持 numpy / int64 序列化。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
