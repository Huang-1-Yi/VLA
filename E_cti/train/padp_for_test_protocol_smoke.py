"""E_cti.train.padp_for_test_protocol_smoke —— 不依赖真实 env 的协议层 smoke test。

> 跑法:
>   python E_cti/train/padp_for_test_protocol_smoke.py
>
> 它会:
>   1) 在本进程内启 server 线程(用真 policy)
>   2) 在本进程内启 client 线程(发合成 obs,收真 action,统计延迟)
>   3) 打印 throughput / latency / 帧大小 等指标
>
> 用来在没有可用 robomimic env 的环境里验证 msgpack 协议 + EP_CHANGE
> + 真实 policy 推理 的端到端。
"""
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
from A_common.types.normalizer import LinearNormalizer
from E_cti.train.padp_for_test_protocol import (
    send_framed, recv_framed, Msg,
    pack_obs, pack_action, pack_ep_change, pack_ep_end, pack_reset_ack, pack_pong,
)

logger = get_logger("padp_for_test_protocol_smoke")


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


def server_thread(host, port, ready_event, stop_event, ckpt_path, cfg_path):
    """简化 server(单连接, 直接调 policy.predict_action)。"""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = _resolve(raw, raw)
    policy = build_policy(cfg["policy"])
    policy.load_state_dict(payload["model_state"], strict=False)
    if payload.get("normalizer_state"):
        norm = LinearNormalizer()
        norm.load_state_dict(payload["normalizer_state"])
        policy.set_normalizer(norm)
    device = "cuda:0"
    policy.to(device).eval()
    # 🚧 修复 VLA 现有 padp_policy 的 3 个预存 bug:
    # (1) self.device / self.dtype 未定义
    # (2) predict_action 返回 3D 违反 ActionOutput 契约 → 用 wrapper 修
    # (3) ckpt 没存 normalizer_state → 修复后 server 端读 ckpt 内嵌
    object.__setattr__(policy, "device", device)
    object.__setattr__(policy, "dtype", torch.float32)
    from E_cti.train.padp_for_test_server import PolicyShapeFixWrapper
    policy = PolicyShapeFixWrapper(policy, device=device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        ready_event.set()
        conn, addr = srv.accept()
        with conn:
            n_steps = 0
            total_latency = 0.0
            while not stop_event.is_set():
                frame = recv_framed(conn)
                if frame is None:
                    break
                t = frame.get("type", "?")
                if t == Msg.EP_CHANGE:
                    policy.reset()
                    send_framed(conn, pack_reset_ack())
                    continue
                if t == Msg.EP_END:
                    continue
                if t == Msg.OBS:
                    obs_dict = {}
                    for k, v in frame["obs"].items():
                        # v is already a numpy.ndarray(因为协议里 recv_framed 自动用 _msgpack_decode hook)
                        if v.dtype == np.uint8:
                            arr = v
                        else:
                            arr = v.astype(np.float32)
                        t_obs = torch.from_numpy(arr)
                        if t_obs.dim() == 3:
                            t_obs = t_obs.unsqueeze(0).unsqueeze(0)
                        elif t_obs.dim() == 1:
                            t_obs = t_obs.unsqueeze(0).unsqueeze(0)
                        obs_dict[k] = t_obs.to(device, non_blocking=True)
                    t0 = time.time()
                    with torch.no_grad():
                        out = policy.predict_action(obs_dict)
                    latency_ms = (time.time() - t0) * 1000.0
                    action = out.actions.detach().cpu().numpy()
                    # 🚧 VLA 的 padp_policy 返回 3D [B, n_action_steps, D] 但 ActionOutput 要求 2D,
                    # 这里手动 squeeze(本环境 v5 架构的设计 bug,后续 TODO 修 BasePolicy)
                    if action.ndim == 3:
                        action = action[0]    # → [n_action_steps, D]
                    if action.ndim == 2:
                        action = action[0]    # → [D]
                    if action.ndim == 1:
                        pass  # 已经是 1D [D]
                    send_framed(conn, pack_action(ep=frame.get("ep", 0), step=frame.get("step", 0),
                                                  action=action, latency_ms=latency_ms))
                    n_steps += 1
                    total_latency += latency_ms
            print(f"[server] n_steps={n_steps} avg_latency={total_latency/max(1,n_steps):.1f}ms")


def client_thread(host, port, ready_event, n_eps, n_steps_per_ep):
    """合成 obs 客户端: 每 ep 发 EP_CHANGE → 等 RESET_ACK → 发 n_steps_per_ep 帧 obs。"""
    ready_event.wait()
    time.sleep(0.5)
    with socket.create_connection((host, port), timeout=10.0) as conn:
        n_total = 0
        total_latency = 0.0
        t_start = time.time()
        for ep in range(n_eps):
            send_framed(conn, pack_ep_change())
            ack = recv_framed(conn)
            assert ack is not None and ack.get("type") == Msg.RESET_ACK
            for step in range(n_steps_per_ep):
                # 合成 obs
                obs = {
                    "agentview_image":          np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8),
                    "robot0_eef_pos":            np.random.randn(3).astype(np.float32),
                    "robot0_eef_quat":           np.random.randn(4).astype(np.float32),
                    "robot0_gripper_qpos":       np.random.randn(2).astype(np.float32),
                }
                send_framed(conn, pack_obs(ep=ep, step=step, obs=obs))
                resp = recv_framed(conn)
                assert resp is not None and resp.get("type") == Msg.ACTION
                total_latency += float(resp.get("latency_ms", 0.0))
                n_total += 1
            send_framed(conn, pack_ep_end(ep=ep, success=False))
        elapsed = time.time() - t_start
        avg_lat = total_latency / max(1, n_total)
        print(f"\n{'='*60}")
        print(f"[client] n_total={n_total} elapsed={elapsed:.2f}s throughput={n_total/elapsed:.1f} steps/s")
        print(f"[client] avg_server_latency={avg_lat:.1f}ms")
        print(f"{'='*60}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(_VLA_ROOT / "data/outputs/padp_for_test_golden/latest.ckpt"))
    ap.add_argument("--config", type=str, default=str(_VLA_ROOT / "E_cti/configs/padp_for_test_golden.yaml"))
    ap.add_argument("--n_eps", type=int, default=3)
    ap.add_argument("--n_steps_per_ep", type=int, default=10)
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        print(f"ckpt not found: {args.ckpt} — please run 1 epoch first")
        sys.exit(1)

    host = "127.0.0.1"
    ready = threading.Event()
    stop = threading.Event()
    t_srv = threading.Thread(target=server_thread, args=(host, args.port, ready, stop,
                                                         args.ckpt, args.config), daemon=True)
    t_srv.start()
    client_thread(host, args.port, ready, args.n_eps, args.n_steps_per_ep)
    stop.set()
    t_srv.join(timeout=5)


if __name__ == "__main__":
    main()
