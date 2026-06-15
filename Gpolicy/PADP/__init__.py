# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""Gpolicy.PADP —— PADP 算法 Fat Policy。"""
from Gpolicy.PADP.padp_policy import SlidingWindowDiffusionPolicy

__all__ = ["SlidingWindowDiffusionPolicy"]
