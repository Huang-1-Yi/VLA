"""测试 policy_registry。"""
import pytest
from A_common.registry import register_policy, build_policy, list_policies
from A_common.registry.policy_registry import _POLICY_REGISTRY


class _MockPolicy:
    def __init__(self, d_h=128):
        self.d_h = d_h


def setup_function(_):
    _POLICY_REGISTRY.clear()


def test_register_and_build():
    register_policy("mock_a")(_MockPolicy)
    p = build_policy({"name": "mock_a", "kwargs": {"d_h": 256}})
    assert isinstance(p, _MockPolicy)
    assert p.d_h == 256


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
