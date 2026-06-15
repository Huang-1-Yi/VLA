# -*- coding: utf-8 -*-
# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""A_common.registry —— 通用注册表(契约 3:policy_registry)。"""
from .policy_registry import register_policy, build_policy, list_policies

__all__ = ["register_policy", "build_policy", "list_policies"]
