# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""契约 3:policy_registry —— Policy 注册表(全程序唯一)。"""
import threading
from typing import Dict, List, Type

_POLICY_REGISTRY: Dict[str, Type] = {}
_LOCK = threading.Lock()


def register_policy(name: str):
    """装饰器:把 Policy 类注册到 name 下面。"""
    def decorator(cls):
        with _LOCK:
            if name in _POLICY_REGISTRY:
                raise ValueError(f"Policy {name} already registered")
            _POLICY_REGISTRY[name] = cls
        return cls
    return decorator


def build_policy(config: dict):
    """工厂:从 config 字典实例化 policy。

    config 格式:
        {"name": "padp_unet", "horizon": 40, "shape_meta": {...}, ...}

    Policy 注册时签名:`def __init__(self, cfg: dict)`,所以我们把整个 dict
    当成一个 'cfg' 参数传过去(而不是 unpack kwargs),符合 Fat Policy 设计。
    """
    name = config["name"]
    if name not in _POLICY_REGISTRY:
        raise KeyError(
            f"Unknown policy: {name}. Available: {list_policies()}"
        )
    return _POLICY_REGISTRY[name](config)


def list_policies() -> List[str]:
    return list(_POLICY_REGISTRY.keys())
