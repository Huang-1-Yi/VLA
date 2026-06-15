# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""A_common.types —— 数据契约。"""
from .action_output import ActionOutput
from .normalizer import LinearNormalizer, SingleFieldLinearNormalizer

__all__ = [
    "ActionOutput",
    "LinearNormalizer",
    "SingleFieldLinearNormalizer",
]
