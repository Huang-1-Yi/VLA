# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================


"""E_cti.train.padp_for_test_server —— PADP policy 推理 server (真版,中文日志)。

> 仿源端 PADP `serve_padp.py` 的设计:
>   1) 加载 ckpt(从 rollout 临时路径) + config(从 yaml 或嵌入的 DEFAULT_CONFIG)
>   2) 还原 policy(走 `build_policy(cfg["policy"])` + `set_normalizer`)
>   3) 选 ema.averaged_model (若可用) 或 online model
>   4) 启 TCP server,接收 msgpack 帧,调 policy.predict_action,返回 action
> 日志已中文化
"""
import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_VLA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_VLA_ROOT))

import A_common
import B_model
import Gpolicy
from A_common.logger import get_logger
from A_common.registry.policy_registry import build_policy
from E_cti.train.padp_for_test_protocol import (
    send_framed, recv_framed,
    Msg, pack_action, pack_reset_ack, pack_pong, pack_error,
)

logger = get_logger("padp_for_test_server")


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


class PolicyShapeFixWrapper:
    """修复 VLA 现有 padp_policy 的两个预存 bug:

    Bug 1: `self.device` / `self.dtype` 未定义,BasePolicy 基类也没定义
    Bug 2: `predict_action` 内部直接构造 `ActionOutput(actions=reshape(B,n_action_steps,D))`,
           把 3D 张量塞进要求 2D 的 ActionOutput dataclass,assert 失败
           (我们无法在 policy 外面修,必须在 wrapper 里直接 re-implement)

    wrapper 注入 self.device / self.dtype,并用 policy._inference_buffer + policy.denoiser
    重写 predict_action 流程(同 padp_policy 内部逻辑,只是最后 squeeze 成 2D)。
    后续 TODO:在 BasePolicy 加 device/dtype,改 padp_policy 修复 ActionOutput 契约。
    """
    def __init__(self, policy, device: str = "cuda:0"):
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", torch.float32)

    def __getattr__(self, name):
        if name in ("_policy", "device", "dtype", "normalizer", "predict_action", "reset"):
            raise AttributeError(name)
        return getattr(self._policy, name)

    @property
    def normalizer(self):
        return self._policy.normalizer

    def reset(self):
        return self._policy.reset()

    def predict_action(self, obs):
        """重写 predict_action:直接调 policy.denoiser,避免 buggy 的 ActionOutput 构造路径。"""
        from A_common.types.action_output import ActionOutput
        p = self._policy
        B = next(iter(obs.values())).shape[0]
        device = self.device
        dtype = self.dtype
        H = p.horizon
        D = p.action_dim
        n_action_steps = p.n_action_steps

        # 初始化或更新 buffer + global_cond
        if p._inference_buffer is None or p._inference_buffer.shape[0] != B:
            p._initialize_inference_buffer(obs)
        else:
            nobs = p.normalizer.normalize(obs)
            this_obs = {
                k: (v[:, :p.n_obs_steps, ...].reshape(-1, *v.shape[2:]) if v.dim() >= 4 else v[:, :p.n_obs_steps, ...].reshape(B, -1))
                for k, v in nobs.items()
            }
            nobs_features = p.adapter(this_obs)
            if isinstance(nobs_features, dict):
                nobs_features = torch.cat([v for v in nobs_features.values()], dim=-1)
            p._inference_global_cond = nobs_features.reshape(B, -1)

        x0 = p._inference_buffer  # [B, H, D]
        noise = torch.randn_like(x0)
        noisy = p.sqrt_alpha_bar_h * x0 + p.sqrt_one_minus_alpha_bar_h * noise

        model_output = p.denoiser(noisy, local_cond=None, global_cond=p._inference_global_cond)

        if p.pred_type == "epsilon":
            pred_actions = noisy - model_output
        elif p.pred_type == "sample":
            pred_actions = model_output
        else:
            raise ValueError(f"Unsupported pred_type: {p.pred_type}")

        # 取第一步 + buffer 左移
        action_to_execute = pred_actions[:, 0:n_action_steps, :]
        p._inference_buffer[:, :H - 1, :] = pred_actions[:, 1:, :]
        new_noise = torch.randn(B, 1, D, device=device, dtype=dtype)
        p._inference_buffer[:, H - 1:, :] = p.sqrt_one_minus_alpha_bar_tail * new_noise

        # 反归一化
        action_out = p.normalizer["action"].unnormalize(action_to_execute)
        # 🚧 squeeze 到 2D [B*n_action_steps, D] (ActionOutput 契约)
        is_chunk = n_action_steps > 1
        if action_out.dim() == 3:
            B_, S_, D_ = action_out.shape
            action_out = action_out.reshape(B_ * S_, D_)
        return ActionOutput(actions=action_out, is_chunk=is_chunk, latency_ms=0.0)


def load_policy_from_ckpt(ckpt_path: str, config_path: str = None, device: str = "cuda:0"):
    """从 ckpt 还原 policy + normalizer(配置从 yaml 读,None 则用嵌入的 DEFAULT_CONFIG)。

    🚧 TODO: 当前 policy 是 online(非 EMA)。若想用 EMA,需要 ckpt 同时 dump
    ema.averaged_model.state_dict() 或 server 启动时再走一次 EMA warmup。
    """
    logger.info("[server] 加载 ckpt: %s", ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if config_path is not None:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        cfg = _resolve(raw, raw)
        logger.info("[server] 从 yaml 加载配置: %s", config_path)
    else:
        # 嵌入的默认配置(从 padp_for_test_train.py 共享)
        from E_cti.train.padp_for_test_train import DEFAULT_CONFIG
        import copy as _copy
        cfg = _copy.deepcopy(DEFAULT_CONFIG)
        cfg = _resolve(cfg, cfg)
        logger.info("[server] 使用嵌入的 DEFAULT_CONFIG (padp_for_test_golden)")

    policy = build_policy(cfg["policy"])
    policy.load_state_dict(payload["model_state"], strict=False)

    if payload.get("normalizer_state"):
        from A_common.types.normalizer import LinearNormalizer
        norm = LinearNormalizer()
        norm.load_state_dict(payload["normalizer_state"])
        policy.set_normalizer(norm)
    policy.to(device)
    policy.eval()
    # 🚧 修复 VLA 现有 padp_policy 的 3 个预存 bug:
    # (1) self.device / self.dtype 未定义 → 用 object.__setattr__ 注入
    # (2) predict_action 返回 3D 违反 ActionOutput 契约 → 用 wrapper 修
    # (3) ckpt 没存 normalizer_state → 修复后 server 端读 ckpt 内嵌
    object.__setattr__(policy, "device", device)
    object.__setattr__(policy, "dtype", torch.float32)
    wrapped = PolicyShapeFixWrapper(policy, device=device)
    logger.info("[server] Policy 已加载: %s, shape_info=%s",
                type(policy).__name__, policy.shape_info())
    return wrapped, cfg


def obs_to_torch(obs: dict, device: str, batched: bool = False) -> dict:
    """客户端发来的 obs(dict,值是 numpy array) → torch dict,加 batch 维。

    客户端约定 (single-env mode, batched=False):
      - rgb keys:  (C, H, W) uint8       (3D,转 torch 后是 (C, H, W))
      - lowdim:    (D,)  float32         (1D,转 torch 后是 (D,))

    客户端约定 (batched mode, batched=True):
      - rgb keys:  (B, C, H, W) uint8    (4D,转 torch 后是 (B, C, H, W))
      - lowdim:    (B, D)   float32      (2D,转 torch 后是 (B, D))

    Policy 期望 (B, S, ...) 形式:
      - rgb:  (B, 1, C, H, W)  ← single: unsqueeze(0).unsqueeze(0); batched: unsqueeze(1)
      - lowdim: (B, 1, D)      ← single: unsqueeze(0).unsqueeze(0); batched: unsqueeze(1)
    """
    out = {}
    for k, v in obs.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(f"obs[{k}] must be numpy array, got {type(v)}")
        t = torch.from_numpy(v.astype(np.float32) if v.dtype != np.uint8 else v)
        if batched:
            # (B, C, H, W) → (B, 1, C, H, W); (B, D) → (B, 1, D)
            t = t.unsqueeze(1)
        else:
            # (D,) → (1, 1, D); (C, H, W) → (1, 1, C, H, W); (B, D) → (B, 1, D)
            if t.dim() in (1, 3):
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 2:
                t = t.unsqueeze(1)
            else:
                t = t.unsqueeze(0)
        out[k] = t.to(device, non_blocking=True)
    return out


def handle_client(conn, addr, policy, device, cfg, server_reset_on_ep_change: bool = True):
    """单 client handler(简化版:1 client 跑完即返回)。"""
    logger.info("[server] 客户端已连接: %s", addr)
    total_steps = 0
    total_reset_count = 0
    try:
        with conn:
            while True:
                frame = recv_framed(conn)
                if frame is None:
                    break
                msg_type = frame.get("type", "?")

                if msg_type == Msg.PING:
                    send_framed(conn, pack_pong())
                    continue

                if msg_type == Msg.EP_CHANGE:
                    # 🚧 消融开关(见 TODO):server_reset_on_ep_change 控制是否真 reset
                    if server_reset_on_ep_change:
                        try:
                            policy.reset()
                            total_reset_count += 1
                        except Exception as e:
                            logger.warning("[server] policy.reset() failed: %s", e)
                        send_framed(conn, pack_reset_ack())
                        logger.info("[server] EP_CHANGE → reset (total_resets=%d)", total_reset_count)
                    else:
                        send_framed(conn, pack_reset_ack())
                        logger.info("[server] EP_CHANGE → no reset (server_reset_on_ep_change=False)")
                    continue

                if msg_type == Msg.EP_END:
                    logger.info("[server] EP_END ep=%s success=%s",
                                frame.get("ep"), frame.get("success"))
                    continue

                if msg_type == Msg.OBS:
                    ep = frame.get("ep", -1)
                    step = frame.get("step", -1)
                    t0 = time.time()
                    try:
                        obs_t = obs_to_torch(frame["obs"], device, batched=False)
                    except Exception as e:
                        send_framed(conn, pack_error(f"obs_to_torch 失败: {e}"))
                        continue
                    try:
                        with torch.no_grad():
                            out = policy.predict_action(obs_t)
                    except Exception as e:
                        logger.exception("[server] predict_action 调用失败")
                        send_framed(conn, pack_error(f"predict_action 失败: {e}"))
                        continue
                    # ActionOutput(actions=[H, D_a] 或 [1, D_a], is_chunk, latency_ms)
                    action = out.actions.detach().cpu().numpy()  # [H, D_a] 或 [1, D_a]
                    # 🚧 VLA padp_policy 返回 3D [B, n_action_steps, D] 但 ActionOutput 要求 2D,
                    # 这里手动 squeeze(本环境 v5 架构的设计 bug,后续 TODO 修 BasePolicy)
                    if action.ndim == 3:
                        action = action[0]    # → [n_action_steps, D]
                    if action.ndim == 2:
                        action = action[0]    # → [D]
                    latency_ms = (time.time() - t0) * 1000.0
                    send_framed(conn, pack_action(ep=ep, step=step,
                                                  action=action, latency_ms=latency_ms))
                    total_steps += 1
                    if total_steps % 50 == 0:
                        logger.info("[server] total_steps=%d", total_steps)
                    continue

                if msg_type == Msg.BATCHED_OBS:
                    # v1.3.1:25 parallel envs (PADP_v3 AsyncVectorEnv 风格)
                    # 1 次收 B 个 obs, 1 次 inference, 1 次回 B 个 action
                    eps = frame.get("eps", [])
                    steps = frame.get("steps", [])
                    obs_list = frame.get("obs_list", [])
                    B = len(eps)
                    t0 = time.time()
                    # 把 B 个 obs 拼成 batched obs_dict (每 key 的值 stack 在 axis=0)
                    if B == 0 or not obs_list:
                        send_framed(conn, pack_error("BATCHED_OBS 空 batch"))
                        continue
                    try:
                        # obs_to_torch 一次性处理, 每 key 的 numpy 都是 (B, ...)
                        merged = {}
                        for k in obs_list[0].keys():
                            merged[k] = np.stack(
                                [np.asarray(o[k]) for o in obs_list], axis=0)
                        obs_t = obs_to_torch(merged, device, batched=True)
                    except Exception as e:
                        logger.exception("[server] batched obs_to_torch 失败")
                        send_framed(conn, pack_error(f"batched obs_to_torch 失败: {e}"))
                        continue
                    try:
                        with torch.no_grad():
                            out = policy.predict_action(obs_t)
                    except Exception as e:
                        logger.exception("[server] batched predict_action 失败")
                        send_framed(conn, pack_error(f"batched predict_action 失败: {e}"))
                        continue
                    # out.actions shape: (B, n_action_steps, D) 或 (B, 1, D)
                    actions = out.actions.detach().cpu().numpy()  # (B, H, D) 或 (B, 1, D)
                    # 取第一个 action step → (B, D)
                    if actions.ndim == 3:
                        actions = actions[:, 0, :]  # (B, D)
                    elif actions.ndim == 2:
                        # 已经是 (B, D), OK
                        pass
                    else:
                        send_framed(conn, pack_error(f"batched action 维度异常: {actions.shape}"))
                        continue
                    latency_ms = (time.time() - t0) * 1000.0
                    send_framed(conn, pack_batched_action(eps=eps, steps=steps,
                                                         actions=actions, latency_ms=latency_ms))
                    total_steps += B
                    if total_steps % (50 * B) == 0:
                        logger.info("[server] batched 已处理 %d steps (latency=%.1fms)",
                                    total_steps, latency_ms)
                    continue

                # 未知类型
                send_framed(conn, pack_error(f"unknown msg_type: {msg_type}"))
    except Exception as e:
        logger.error("[server] handler error: %s", e)
    finally:
        logger.info("[server] Client %s disconnected (total_steps=%d, total_resets=%d)",
                    addr, total_steps, total_reset_count)


def serve_forever(host: str, port: int, policy, device: str, cfg: dict,
                  server_reset_on_ep_change: bool = True):
    """阻塞 serve。"""
    policy.eval()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)
        logger.info("[server] Listening on %s:%d (reset_on_ep_change=%s)",
                    host, port, server_reset_on_ep_change)
        while True:
            try:
                conn, addr = srv.accept()
            except KeyboardInterrupt:
                logger.info("[server] KeyboardInterrupt, shutting down.")
                break
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, policy, device, cfg, server_reset_on_ep_change),
                daemon=True,
            )
            t.start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default=None,
                    help="yaml 路径(可选,默认用嵌入的 DEFAULT_CONFIG)")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--server_reset_on_ep_change", type=lambda s: s.lower() == "true",
                    default=True, help="消融开关: 是否在 EP_CHANGE 时真调 policy.reset()")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("padp_for_test_server starting")
    logger.info("=" * 60)

    policy, cfg = load_policy_from_ckpt(args.ckpt, config_path=args.config, device=args.device)
    serve_forever(args.host, args.port, policy, args.device, cfg,
                  server_reset_on_ep_change=args.server_reset_on_ep_change)


if __name__ == "__main__":
    main()
