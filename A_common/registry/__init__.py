# -*- coding: utf-8 -*-
"""A_common.registry —— 通用注册表(契约 3:policy_registry)。"""
from .policy_registry import register_policy, build_policy, list_policies

__all__ = ["register_policy", "build_policy", "list_policies"]
