# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""Gpolicy.PADP —— PADP 算法 Fat Policy。"""
from Gpolicy.PADP.padp_policy import SlidingWindowDiffusionPolicy

__all__ = ["SlidingWindowDiffusionPolicy"]
