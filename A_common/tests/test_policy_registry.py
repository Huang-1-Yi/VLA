"""测试 policy_registry。

v5-1 API:`build_policy(cfg)` 把整个 cfg dict 当一个参数传给 Policy 构造函数
(不是 unpack kwargs),因为 Fat Policy 的 __init__ 签名是 `(self, cfg: dict)`。
"""
import pytest
from A_common.registry import register_policy, build_policy, list_policies
from A_common.registry.policy_registry import _POLICY_REGISTRY


class _MockPolicy:
    """Mock Fat Policy:把 cfg 当 dict 存到 self.cfg,方便测试 assert。"""
    def __init__(self, cfg: dict):
        self.cfg = cfg


def setup_function(_):
    _POLICY_REGISTRY.clear()


def test_register_and_build():
    register_policy("mock_a")(_MockPolicy)
    p = build_policy({"name": "mock_a", "kwargs": {"d_h": 256}})
    assert isinstance(p, _MockPolicy)
    # v5: build_policy 把整个 cfg dict 传给 Policy,kwargs 仍在 cfg["kwargs"] 内
    assert p.cfg == {"name": "mock_a", "kwargs": {"d_h": 256}}


def test_list_policies():
    register_policy("x")(_MockPolicy)
    register_policy("y")(_MockPolicy)
    assert set(list_policies()) == {"x", "y"}


def test_unknown_raises():
    with pytest.raises(KeyError):
        build_policy({"name": "nope"})


def test_duplicate_raises():
    register_policy("dup")(_MockPolicy)
    with pytest.raises(ValueError):
        register_policy("dup")(_MockPolicy)