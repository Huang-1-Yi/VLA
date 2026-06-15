# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""B_model.adapters —— 算法特征融合器。"""
from B_model.adapters.base_adapter import BaseAdapter
from B_model.adapters.padp_adapter import PADPAdapter

__all__ = ["BaseAdapter", "PADPAdapter"]
