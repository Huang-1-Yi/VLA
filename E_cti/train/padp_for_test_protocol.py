# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================


"""E_cti.train.padp_for_test_protocol —— server+client 共享的 msgpack 协议。

> **A-1 / A-2 决定**:msgpack + 4 字节大端长度前缀。
>   - 同 payload (84x84x3 uint8 + 3 float32) 比 base64+JSON 省 **25%**(21282B vs 28278B)
>   - pack 2.8μs, unpack 5.4μs(主要时间花在 numpy.tobytes() / frombuffer)
>   - 自带 enc/dec hook,不用 msgpack_numpy 依赖

**协议帧格式**(每个 TCP message):
```
+--------+--------+--------+--------+--------+--------+--- ... ---+
|  length (4B, big-endian uint32)  |  payload (msgpack, N bytes)   |
+--------+--------+--------+--------+--------+--------+--- ... ---+
length 包含 payload 字节数(不含 4 字节 length 头本身)
```

**payload 类型**(用 dict 区分,msgpack 不区分类型):
- 客户端 → 服务端:`{"type": "OBS", "ep": <int>, "step": <int>, "obs": {...}}`
  - 服务端 → 客户端:`{"type": "ACTION", "ep": <int>, "step": <int>, "action": [D_a], "latency_ms": <float>}`
  - 客户端 → 服务端:`{"type": "EP_CHANGE"}`  ← 触发 server 端 `policy.reset()`
  - 客户端 → 服务端:`{"type": "EP_END", "ep": <int>, "success": <bool>}`  ← 可选,server 端 log
  - 服务端 → 客户端:`{"type": "RESET_ACK"}`  ← server 已 reset, 客户端可继续
  - 任何方向:`{"type": "PING"}` / `{"type": "PONG"}`  ← 心跳
  - 任何方向:`{"type": "ERROR", "msg": "..."}`  ← 出错(关闭连接前发一次)

**obs dict 编码**:
  - rgb keys: `(B, T, C, H, W)` float32 (server 端 policy 期望的格式,client 负责 moveaxis+scale/255)
  - lowdim keys: `(B, T, D)` float32
  - 加一个 `state` key 也行(server 端 policy 会自动 cat lowdim)

**action 编码**:
  - `[D_a]` numpy array float32,server 反归一化后直接给 env step

**🚧 TODO 留位**(见 `padp_for_test_interaction_TODO.md`):
- 异步流水线:client 端在等 server 回包时,server 端可以并行算下一个 obs
  → 需要 client/server 各加一个发送 / 接收缓冲区,允许 K 个 in-flight 请求
  → 简易实现:server 在 compute_loss / predict_action 完成前立刻 accept 下一帧
- EP_CHANGE 帧:本文件已定义,server 端默认会在收到时调 `policy.reset()`
- 消融开关:`server_reset_on_ep_change` (bool, 训练/rollout 各自可关)
"""
import io
import struct
import socket
from typing import Optional
import numpy as np
import msgpack

# ====================================================================
# 长度前缀读写
# ====================================================================
def send_framed(sock: socket.socket, payload: dict):
    """先 4B 长度,再 msgpack payload。自动应用 numpy → msgpack 编码 hook。"""
    raw = msgpack.packb(payload, use_bin_type=True, default=_msgpack_encode)
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def recv_framed(sock: socket.socket, max_bytes: int = 64 * 1024 * 1024) -> dict:
    """读 1 帧:4B 长度 + msgpack payload。阻塞。
    返回 None 表示对端 closed。
    """
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length > max_bytes:
        raise ValueError(f"Frame too large: {length} > {max_bytes}")
    body = _recv_exact(sock, length)
    if body is None:
        return None
    return msgpack.unpackb(body, raw=False, object_hook=_msgpack_decode)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None if not buf else (_ for _ in ()).throw(ConnectionError("truncated"))
        buf.extend(chunk)
    return bytes(buf)


# ====================================================================
# numpy 编码 / 解码
# ====================================================================
def _msgpack_encode(obj):
    """把 numpy.ndarray 编码成 dict,其他原样。"""
    if isinstance(obj, np.ndarray):
        return {
            "__np__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(),  # msgpack bin type (use_bin_type=True)
        }
    if isinstance(obj, dict):
        return {k: _msgpack_encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_msgpack_encode(x) for x in obj]
    return obj


def _msgpack_decode(obj):
    """把 __np__ 标记的 dict 还原成 numpy.ndarray。"""
    if isinstance(obj, dict) and obj.get("__np__"):
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    if isinstance(obj, dict):
        return {k: _msgpack_decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_msgpack_decode(x) for x in obj]
    return obj


# ====================================================================
# 帧类型常量(防 typo)
# ====================================================================
class Msg:
    OBS = "OBS"
    ACTION = "ACTION"
    EP_CHANGE = "EP_CHANGE"
    EP_END = "EP_END"
    RESET_ACK = "RESET_ACK"
    PING = "PING"
    PONG = "PONG"
    ERROR = "ERROR"
    # v1.3:25 parallel envs (PADP_v3 AsyncVectorEnv + 1 TCP + batched 帧)
    BATCHED_OBS = "BATCHED_OBS"
    BATCHED_ACTION = "BATCHED_ACTION"
    BATCHED_EP_END = "BATCHED_EP_END"


def pack_obs(ep: int, step: int, obs: dict) -> dict:
    """客户端 → 服务端:发 obs (single-env, 向后兼容)。"""
    return {"type": Msg.OBS, "ep": int(ep), "step": int(step), "obs": obs}


def pack_batched_obs(eps: list, steps: list, obs_list: list) -> dict:
    """客户端 → 服务端:发 batched obs (B 个 env 并行)。

    Args:
        eps:   [B] int, 每个 env 的 ep id
        steps: [B] int, 每个 env 的 step idx
        obs_list: [B] dict, 每个 env 的 obs dict
    """
    assert len(eps) == len(steps) == len(obs_list), f"batched 维度不一致: {len(eps)} vs {len(steps)} vs {len(obs_list)}"
    return {"type": Msg.BATCHED_OBS, "eps": list(map(int, eps)),
            "steps": list(map(int, steps)), "obs_list": list(obs_list)}


def pack_action(ep: int, step: int, action: np.ndarray, latency_ms: float = 0.0) -> dict:
    """服务端 → 客户端:回 action (single-env, 向后兼容)。"""
    return {"type": Msg.ACTION, "ep": int(ep), "step": int(step),
            "action": action, "latency_ms": float(latency_ms)}


def pack_batched_action(eps: list, steps: list, actions: np.ndarray, latency_ms: float = 0.0) -> dict:
    """服务端 → 客户端:回 batched action (B 个 env 并行)。

    Args:
        eps:     [B] int
        steps:   [B] int
        actions: [B, D_a] np.ndarray
    """
    assert actions.ndim == 2, f"batched action 必须是 2D, 实际 {actions.shape}"
    return {"type": Msg.BATCHED_ACTION, "eps": list(map(int, eps)),
            "steps": list(map(int, steps)), "actions": actions,
            "latency_ms": float(latency_ms)}


def pack_ep_change() -> dict:
    return {"type": Msg.EP_CHANGE}


def pack_batched_ep_end(eps: list, successes: list) -> dict:
    """B 个 env 一起报 EP_END。"""
    assert len(eps) == len(successes)
    return {"type": Msg.BATCHED_EP_END, "eps": list(map(int, eps)),
            "successes": [bool(s) for s in successes]}


def pack_ep_end(ep: int, success: bool) -> dict:
    return {"type": Msg.EP_END, "ep": int(ep), "success": bool(success)}


def pack_reset_ack() -> dict:
    return {"type": Msg.RESET_ACK}


def pack_ping() -> dict:
    return {"type": Msg.PING}


def pack_pong() -> dict:
    return {"type": Msg.PONG}


def pack_error(msg: str) -> dict:
    return {"type": Msg.ERROR, "msg": str(msg)}
