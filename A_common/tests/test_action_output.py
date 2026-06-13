"""测试 ActionOutput 契约。"""
import torch
import pytest
from A_common.types.action_output import ActionOutput


def test_action_output_chunk():
    ao = ActionOutput(actions=torch.randn(16, 10), is_chunk=True, latency_ms=10.0)
    assert ao.actions.shape == (16, 10)
    assert ao.is_chunk is True


def test_action_output_step():
    ao = ActionOutput(actions=torch.randn(1, 10), is_chunk=False, latency_ms=5.0)
    assert ao.actions.shape == (1, 10)
    assert ao.is_chunk is False


def test_action_output_mismatch_chunk():
    with pytest.raises(AssertionError):
        ActionOutput(actions=torch.randn(1, 10), is_chunk=True, latency_ms=1.0)


def test_action_output_mismatch_step():
    with pytest.raises(AssertionError):
        ActionOutput(actions=torch.randn(16, 10), is_chunk=False, latency_ms=1.0)
