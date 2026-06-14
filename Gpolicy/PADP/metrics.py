"""Gpolicy.PADP.metrics —— per-position MSE / NMSE 评估指标。"""
import torch


def per_position_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """返回 [H] shape,每位置 MSE。"""
    return ((pred - target) ** 2).mean(dim=(0, 2))


def per_position_nmse(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """归一化 MSE,除以 action 维度方差。"""
    return per_position_mse(pred, target) / (scale ** 2 + 1e-8)