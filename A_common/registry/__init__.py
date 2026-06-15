# -*- coding: utf-8 -*-
# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""A_common.registry —— 通用注册表(契约 3:policy_registry)。"""
from .policy_registry import register_policy, build_policy, list_policies

__all__ = ["register_policy", "build_policy", "list_policies"]
